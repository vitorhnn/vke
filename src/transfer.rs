use ash::{
    prelude::VkResult,
    vk,
    Device,
};

use ash::vk::Semaphore;
use smallvec::{smallvec, SmallVec};
use std::rc::Rc;

use crate::allocator::{Allocation, Allocator, MemoryUsage};
use crate::device::RawDevice;
use crate::queue::Queue;
use crate::buffer::Buffer;

struct BufferSlice {
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
}

struct PendingCopy {
    src: BufferSlice,
    dst: BufferSlice,
}

struct QueueContext {
    queue: Rc<Queue>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    pre_barriers: Vec<vk::BufferMemoryBarrier>,
    post_barriers: Vec<vk::BufferMemoryBarrier>,
    pending_copies: Vec<PendingCopy>,
}

pub struct Transfer {
    device: Rc<RawDevice>,
    allocator: Rc<Allocator>,
    buffer: Buffer,
    allocation: Allocation,
    ptr: *mut u8,
    used: usize,
    graphics_queue_ctx: QueueContext,
    transfer_queue_ctx: Option<QueueContext>,
}

const KIBIBYTE: usize = 2usize.pow(10);
const STAGING_BUFFER_SIZE: usize = KIBIBYTE * 512;

impl Transfer {
    pub fn new(
        device: Rc<RawDevice>,
        allocator: Rc<Allocator>,
        graphics_queue: Rc<Queue>,
        transfer_queue: Option<Rc<Queue>>,
    ) -> VkResult<Self> {
        let staging_buffer_info = vk::BufferCreateInfo::builder()
            .size(STAGING_BUFFER_SIZE as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let (buffer, allocation) =
            allocator.create_buffer(&staging_buffer_info, MemoryUsage::HostOnly)?;

        let ptr = allocator.map(&allocation)?;

        let graphics_command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    queue_family_index: graphics_queue.family_index,
                    flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                        | vk::CommandPoolCreateFlags::TRANSIENT,
                    ..Default::default()
                },
                None,
            )?
        };

        let graphics_command_buffers = unsafe {
            device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_buffer_count: 1,
                command_pool: graphics_command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                ..Default::default()
            })?
        };

        let graphics_command_buffer = graphics_command_buffers[0];

        let graphics_queue_ctx = QueueContext {
            queue: graphics_queue,
            command_buffer: graphics_command_buffer,
            command_pool: graphics_command_pool,
            pre_barriers: Vec::new(),
            post_barriers: Vec::new(),
            pending_copies: Vec::new(),
        };

        let transfer_queue_ctx = if let Some(transfer_queue) = transfer_queue {
            let command_pool = unsafe {
                device.create_command_pool(
                    &vk::CommandPoolCreateInfo {
                        queue_family_index: transfer_queue.family_index,
                        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                            | vk::CommandPoolCreateFlags::TRANSIENT,
                        ..Default::default()
                    },
                    None,
                )?
            };

            let command_buffers = unsafe {
                device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                    command_buffer_count: 1,
                    level: vk::CommandBufferLevel::PRIMARY,
                    command_pool,
                    ..Default::default()
                })?
            };

            Some(QueueContext {
                queue: transfer_queue,
                command_buffer: command_buffers[0],
                pre_barriers: Vec::new(),
                post_barriers: Vec::new(),
                pending_copies: Vec::new(),
                command_pool,
            })
        } else {
            None
        };

        Ok(Self {
            used: 0,
            device,
            allocator,
            buffer,
            allocation,
            ptr,
            graphics_queue_ctx,
            transfer_queue_ctx,
        })
    }

    pub fn upload_buffer_callback<T, F>(&mut self, callback: F, dest: &Buffer) -> VkResult<()>
    where
        T: Copy,
        F: Fn(&mut [T]) -> usize,
    {
        if let Some(transfer_ctx) = &mut self.transfer_queue_ctx {
            unsafe {
                let slice = std::slice::from_raw_parts_mut(
                    self.ptr.add(self.used),
                    STAGING_BUFFER_SIZE - self.used,
                );

                // HACK: this is likely BEYOND broken. check here if we get texture / vertex corruption
                let (prefix, aligned, suffix) = slice.align_to_mut::<T>();

                let written = callback(aligned);

                let start_of_data = prefix.len() + self.used;
                self.used += prefix.len() + written;

                transfer_ctx.post_barriers.push(vk::BufferMemoryBarrier {
                    buffer: dest.inner,
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::empty(),
                    src_queue_family_index: transfer_ctx.queue.family_index,
                    dst_queue_family_index: self.graphics_queue_ctx.queue.family_index,
                    offset: 0,
                    size: written as u64,
                    ..Default::default()
                });

                transfer_ctx.pending_copies.push(PendingCopy {
                    src: BufferSlice {
                        buffer: self.buffer.inner,
                        offset: start_of_data as u64,
                        size: written as u64,
                    },
                    dst: BufferSlice {
                        buffer: dest.inner,
                        offset: 0, // currently assuming we should write the entire buffer
                        size: written as u64,
                    },
                });

                self.graphics_queue_ctx
                    .post_barriers
                    .push(vk::BufferMemoryBarrier {
                        buffer: dest.inner,
                        src_access_mask: vk::AccessFlags::empty(),
                        dst_access_mask: vk::AccessFlags::MEMORY_READ,
                        src_queue_family_index: transfer_ctx.queue.family_index,
                        dst_queue_family_index: self.graphics_queue_ctx.queue.family_index,
                        offset: 0,
                        size: written as u64,
                        ..Default::default()
                    });

                Ok(())
            }
        } else {
            todo!("no transfer queue path");
        }
    }

    pub fn upload_buffer<T: Copy>(&mut self, data: &[T], dest: &Buffer) -> VkResult<()> {
        self.upload_buffer_callback(
            |staging_buffer| {
                staging_buffer[..data.len()].copy_from_slice(data);

                eprintln!("size of data is {}", std::mem::size_of_val(data));
                std::mem::size_of_val(data)
            },
            dest,
        )
    }

    pub fn flush(&mut self) -> VkResult<()> {
        if let Some(transfer_ctx) = &mut self.transfer_queue_ctx {
            let semaphores = if !transfer_ctx.pending_copies.is_empty()
                || !transfer_ctx.pre_barriers.is_empty()
                || !transfer_ctx.post_barriers.is_empty()
            {
                let begin_info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

                unsafe {
                    self.device
                        .begin_command_buffer(transfer_ctx.command_buffer, &begin_info)?
                };

                if !transfer_ctx.pre_barriers.is_empty() {
                    todo!("pre barriers are unimplemented");
                }
                for _barrier in transfer_ctx.pre_barriers.drain(..) {
                    todo!(); // TODO: not needed for buffer uploads, come back later for texture uploads
                }

                for copy in transfer_ctx.pending_copies.drain(..) {
                    let regions = [vk::BufferCopy {
                        src_offset: copy.src.offset,
                        dst_offset: copy.dst.offset,
                        size: copy.src.size,
                    }];

                    unsafe {
                        self.device.cmd_copy_buffer(
                            transfer_ctx.command_buffer,
                            copy.src.buffer,
                            copy.dst.buffer,
                            &regions,
                        );
                    }
                }

                unsafe {
                    // TODO: come back here for image barriers
                    self.device.cmd_pipeline_barrier(
                        transfer_ctx.command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &transfer_ctx.post_barriers,
                        &[],
                    )
                }

                unsafe {
                    self.device
                        .end_command_buffer(transfer_ctx.command_buffer)?
                };

                let semaphore_create_info = vk::SemaphoreCreateInfo {
                    ..Default::default()
                };

                let semaphore =
                    unsafe { self.device.create_semaphore(&semaphore_create_info, None)? };

                let buffers = [transfer_ctx.command_buffer];
                let semaphores: SmallVec<[Semaphore; 1]> = smallvec![semaphore];
                let submits = [vk::SubmitInfo::builder()
                    .command_buffers(&buffers)
                    .signal_semaphores(&semaphores)
                    .build()];

                unsafe {
                    self.device.queue_submit(
                        transfer_ctx.queue.inner,
                        &submits,
                        vk::Fence::null(),
                    )?
                };

                semaphores
            } else {
                smallvec![]
            };

            let gfx_ctx = &mut self.graphics_queue_ctx;

            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            unsafe {
                self.device
                    .begin_command_buffer(gfx_ctx.command_buffer, &begin_info)?
            };

            if !gfx_ctx.pre_barriers.is_empty() {
                todo!("pre barriers are unimplemented");
            }
            for _barrier in gfx_ctx.pre_barriers.drain(..) {
                todo!(); // TODO: not needed for buffer uploads, come back later for texture uploads
            }

            for copy in gfx_ctx.pending_copies.drain(..) {
                let regions = [vk::BufferCopy {
                    src_offset: copy.src.offset,
                    dst_offset: copy.dst.offset,
                    size: copy.src.size,
                }];

                unsafe {
                    self.device.cmd_copy_buffer(
                        gfx_ctx.command_buffer,
                        copy.src.buffer,
                        copy.dst.buffer,
                        &regions,
                    );
                }
            }

            unsafe {
                // TODO: come back here for image barriers
                self.device.cmd_pipeline_barrier(
                    gfx_ctx.command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &gfx_ctx.post_barriers,
                    &[],
                )
            }

            unsafe { self.device.end_command_buffer(gfx_ctx.command_buffer)? };

            let buffers = [gfx_ctx.command_buffer];
            // TODO: write something that's capable of using more specific wait stages
            let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
            let submits = [vk::SubmitInfo::builder()
                .command_buffers(&buffers)
                .wait_dst_stage_mask(&stages)
                .wait_semaphores(&semaphores)
                .build()];

            let graphics_done_fence = unsafe {
                self.device.inner.create_fence(
                    &vk::FenceCreateInfo {
                        ..Default::default()
                    },
                    None,
                )?
            };

            unsafe {
                self.device
                    .queue_submit(gfx_ctx.queue.inner, &submits, graphics_done_fence)?
            };

            unsafe {
                self.device
                    .inner
                    .wait_for_fences(&[graphics_done_fence], true, u64::MAX)?;
                self.device.destroy_fence(graphics_done_fence, None);
                self.device.destroy_semaphore(semaphores[0], None);
            };

            self.used = 0;

            Ok(())
        } else {
            todo!("graphics queue only");
        }
    }
}

impl Drop for Transfer {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_command_pool(self.graphics_queue_ctx.command_pool, None);

            if let Some(ctx) = &self.transfer_queue_ctx {
                self.device.destroy_command_pool(ctx.command_pool, None);
            }
        }
    }
}
