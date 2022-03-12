use ash::{prelude::VkResult, vk};

use ash::vk::{Image, Semaphore};
use smallvec::{smallvec, SmallVec};
use std::default::Default;
use std::rc::Rc;

use crate::allocator::{Allocator, MemoryUsage};
use crate::buffer::Buffer;
use crate::device::Device;
use crate::queue::Queue;
use crate::texture::Texture;
use gpu_allocator::vulkan::Allocation;

struct BufferSlice {
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
}

struct PendingBufferCopy {
    src: vk::Buffer,
    dst: vk::Buffer,
    copy: vk::BufferCopy,
}

struct PendingImageCopy {
    src: BufferSlice,
    dst_image: vk::Image,
    copy_op: vk::BufferImageCopy,
}

struct Barriers {
    buffer: Vec<vk::BufferMemoryBarrier>,
    image: Vec<vk::ImageMemoryBarrier>,
}

struct QueueContext {
    queue: Rc<Queue>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    pre_barriers: Barriers,
    post_barriers: Barriers,
    pending_buffer_copies: Vec<PendingBufferCopy>,
    pending_image_copies: Vec<PendingImageCopy>,
}

pub struct Transfer {
    device: Rc<Device>,
    allocator: Rc<Allocator>,
    buffer: Buffer,
    allocation: Allocation,
    ptr: *mut u8,
    used: usize,
    graphics_queue_ctx: QueueContext,
    transfer_queue_ctx: Option<QueueContext>,
}

const MEBIBYTE: usize = 2usize.pow(20);
const STAGING_BUFFER_SIZE: usize = MEBIBYTE * 64;

impl Barriers {
    pub fn new() -> Self {
        Self {
            image: Vec::new(),
            buffer: Vec::new(),
        }
    }
}

impl Transfer {
    pub fn new(
        device: Rc<Device>,
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
            device.inner.create_command_pool(
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
            device
                .inner
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
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
            pre_barriers: Barriers::new(),
            post_barriers: Barriers::new(),
            pending_buffer_copies: Vec::new(),
            pending_image_copies: Vec::new(),
        };

        let transfer_queue_ctx = if let Some(transfer_queue) = transfer_queue {
            let command_pool = unsafe {
                device.inner.create_command_pool(
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
                device
                    .inner
                    .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                        command_buffer_count: 1,
                        level: vk::CommandBufferLevel::PRIMARY,
                        command_pool,
                        ..Default::default()
                    })?
            };

            Some(QueueContext {
                queue: transfer_queue,
                command_buffer: command_buffers[0],
                pre_barriers: Barriers::new(),
                post_barriers: Barriers::new(),
                pending_buffer_copies: Vec::new(),
                pending_image_copies: Vec::new(),
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

    fn upload_to_staging_buffer<T, F>(&mut self, callback: F) -> (usize, usize)
    where
        T: Copy,
        F: FnOnce(&mut [T]) -> usize,
    {
        unsafe {
            let slice = std::slice::from_raw_parts_mut(
                self.ptr.add(self.used),
                STAGING_BUFFER_SIZE - self.used,
            );

            // HACK: this is likely BEYOND broken. check here if we get vertex corruption
            let (prefix, aligned, _suffix) = slice.align_to_mut::<T>();

            let written = callback(aligned);

            let start_of_data = prefix.len() + self.used;
            self.used += prefix.len() + written;

            (written, start_of_data)
        }
    }

    pub fn upload_buffer_callback<T, F>(&mut self, callback: F, dest: &Buffer) -> VkResult<()>
    where
        T: Copy,
        F: FnOnce(&mut [T]) -> usize,
    {
        let (written, start_of_data) = self.upload_to_staging_buffer(callback);

        if let Some(transfer_ctx) = &mut self.transfer_queue_ctx {
            unsafe {
                transfer_ctx
                    .post_barriers
                    .buffer
                    .push(vk::BufferMemoryBarrier {
                        buffer: dest.inner,
                        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        dst_access_mask: vk::AccessFlags::empty(),
                        src_queue_family_index: transfer_ctx.queue.family_index,
                        dst_queue_family_index: self.graphics_queue_ctx.queue.family_index,
                        offset: 0,
                        size: written as u64,
                        ..Default::default()
                    });

                transfer_ctx.pending_buffer_copies.push(PendingBufferCopy {
                    src: self.buffer.inner,
                    dst: dest.inner,
                    copy: vk::BufferCopy {
                        src_offset: start_of_data as u64,
                        dst_offset: 0,
                        size: written as u64,
                    },
                });

                self.graphics_queue_ctx
                    .post_barriers
                    .buffer
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

    pub fn upload_image_callback<T, F>(&mut self, callback: F, dest: &Texture) -> VkResult<()>
    where
        T: Copy,
        F: FnOnce(&mut [T]) -> usize,
    {
        let (written, start_of_data) = self.upload_to_staging_buffer(callback);

        if let Some(transfer_ctx) = &mut self.transfer_queue_ctx {
            transfer_ctx
                .pre_barriers
                .image
                .push(vk::ImageMemoryBarrier {
                    image: dest.image,
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                });

            transfer_ctx.pending_image_copies.push(PendingImageCopy {
                src: BufferSlice {
                    buffer: self.buffer.inner,
                    offset: start_of_data as u64,
                    size: written as u64,
                },
                dst_image: dest.image,
                copy_op: vk::BufferImageCopy {
                    buffer_offset: start_of_data as u64,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_extent: dest.extent,
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_subresource: vk::ImageSubresourceLayers {
                        layer_count: 1,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_array_layer: 0,
                        mip_level: 0,
                    },
                },
            });

            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            transfer_ctx
                .post_barriers
                .image
                .push(vk::ImageMemoryBarrier {
                    image: dest.image,
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::empty(),
                    src_queue_family_index: transfer_ctx.queue.family_index,
                    dst_queue_family_index: self.graphics_queue_ctx.queue.family_index,
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    subresource_range,
                    ..Default::default()
                });

            self.graphics_queue_ctx
                .post_barriers
                .image
                .push(vk::ImageMemoryBarrier {
                    image: dest.image,
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::MEMORY_READ,
                    src_queue_family_index: transfer_ctx.queue.family_index,
                    dst_queue_family_index: self.graphics_queue_ctx.queue.family_index,
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    subresource_range,
                    ..Default::default()
                });

            Ok(())
        } else {
            todo!();
        }
    }

    pub fn flush(&mut self) -> VkResult<()> {
        if let Some(transfer_ctx) = &mut self.transfer_queue_ctx {
            let semaphores = if !transfer_ctx.pending_buffer_copies.is_empty()
                || !transfer_ctx.pre_barriers.buffer.is_empty()
                || !transfer_ctx.pre_barriers.image.is_empty()
                || !transfer_ctx.post_barriers.buffer.is_empty()
                || !transfer_ctx.post_barriers.image.is_empty()
            {
                let begin_info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

                unsafe {
                    self.device
                        .inner
                        .begin_command_buffer(transfer_ctx.command_buffer, &begin_info)?
                };

                unsafe {
                    self.device.inner.cmd_pipeline_barrier(
                        transfer_ctx.command_buffer,
                        vk::PipelineStageFlags::HOST,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &transfer_ctx.pre_barriers.buffer,
                        &transfer_ctx.pre_barriers.image,
                    )
                }

                for copy in transfer_ctx.pending_buffer_copies.drain(..) {
                    let regions = std::slice::from_ref(&copy.copy);

                    unsafe {
                        self.device.inner.cmd_copy_buffer(
                            transfer_ctx.command_buffer,
                            copy.src,
                            copy.dst,
                            &regions,
                        );
                    }
                }

                for copy in transfer_ctx.pending_image_copies.drain(..) {
                    unsafe {
                        self.device.inner.cmd_copy_buffer_to_image(
                            transfer_ctx.command_buffer,
                            copy.src.buffer,
                            copy.dst_image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            std::slice::from_ref(&copy.copy_op),
                        );
                    }
                }

                unsafe {
                    self.device.inner.cmd_pipeline_barrier(
                        transfer_ctx.command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &transfer_ctx.post_barriers.buffer,
                        &transfer_ctx.post_barriers.image,
                    )
                }

                unsafe {
                    self.device
                        .inner
                        .end_command_buffer(transfer_ctx.command_buffer)?
                };

                let semaphore = self.device.create_timeline_semaphore(0)?;

                let buffers = [transfer_ctx.command_buffer];
                let semaphores: SmallVec<[Semaphore; 1]> = smallvec![semaphore];
                let mut semaphore_submit =
                    vk::TimelineSemaphoreSubmitInfoKHR::builder().signal_semaphore_values(&[1]);
                let submits = [vk::SubmitInfo::builder()
                    .command_buffers(&buffers)
                    .signal_semaphores(&semaphores)
                    .push_next(&mut semaphore_submit)
                    .build()];

                unsafe {
                    self.device.inner.queue_submit(
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
                    .inner
                    .begin_command_buffer(gfx_ctx.command_buffer, &begin_info)?
            };

            for copy in gfx_ctx.pending_buffer_copies.drain(..) {
                let regions = std::slice::from_ref(&copy.copy);

                unsafe {
                    self.device.inner.cmd_copy_buffer(
                        gfx_ctx.command_buffer,
                        copy.src,
                        copy.dst,
                        &regions,
                    );
                }
            }

            unsafe {
                // TODO: come back here for image barriers
                self.device.inner.cmd_pipeline_barrier(
                    gfx_ctx.command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &gfx_ctx.post_barriers.buffer,
                    &gfx_ctx.post_barriers.image,
                )
            }

            unsafe {
                self.device
                    .inner
                    .end_command_buffer(gfx_ctx.command_buffer)?
            };

            let buffers = [gfx_ctx.command_buffer];
            // TODO: write something that's capable of using more specific wait stages
            let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
            let mut semaphore_submit = vk::TimelineSemaphoreSubmitInfoKHR::builder()
                .wait_semaphore_values(&[1])
                .signal_semaphore_values(&[2]);
            let submits = [vk::SubmitInfo::builder()
                .command_buffers(&buffers)
                .wait_dst_stage_mask(&stages)
                .wait_semaphores(&semaphores)
                .signal_semaphores(&semaphores)
                .push_next(&mut semaphore_submit)
                .build()];

            unsafe {
                self.device
                    .inner
                    .queue_submit(gfx_ctx.queue.inner, &submits, vk::Fence::null())?
            };

            let sem_wait_info = vk::SemaphoreWaitInfoKHR::builder()
                .values(&[2])
                .semaphores(&semaphores);

            unsafe {
                self.device
                    .timeline_semaphore
                    .wait_semaphores(&sem_wait_info, u64::MAX)?;
                self.device.inner.destroy_semaphore(semaphores[0], None);
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
                .inner
                .destroy_command_pool(self.graphics_queue_ctx.command_pool, None);

            if let Some(ctx) = &self.transfer_queue_ctx {
                self.device
                    .inner
                    .destroy_command_pool(ctx.command_pool, None);
            }
        }
    }
}
