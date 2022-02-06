use ash::{prelude::VkResult, vk};

use parking_lot::Mutex;

use std::ffi::c_void;
use std::rc::Rc;

use crate::buffer::Buffer;
use crate::device::{Device, RawDevice};
use crate::instance::Instance;

// stolen from ash
fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    usage: MemoryUsage,
) -> Option<(u32, vk::MemoryPropertyFlags)> {
    let acceptable_flags = match usage {
        MemoryUsage::HostToDevice => {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        }
        MemoryUsage::HostOnly => {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        }
        _ => vk::MemoryPropertyFlags::empty(),
    };

    let preferred_flags = acceptable_flags
        | match usage {
            MemoryUsage::DeviceOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryUsage::HostToDevice => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            _ => vk::MemoryPropertyFlags::empty(),
        };

    // Try to find an exactly matching memory flag
    let best_suitable_index = find_memorytype_index_f(memory_req, memory_prop, |property_flags| {
        property_flags.contains(preferred_flags)
    });
    if best_suitable_index.is_some() {
        return best_suitable_index.map(|r| (r, preferred_flags));
    }
    // Otherwise find a memory flag that works
    find_memorytype_index_f(memory_req, memory_prop, |property_flags| {
        property_flags.contains(acceptable_flags)
    })
    .map(|r| (r, acceptable_flags))
}

fn find_memorytype_index_f<F: Fn(vk::MemoryPropertyFlags) -> bool>(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    f: F,
) -> Option<u32> {
    let mut memory_type_bits = memory_req.memory_type_bits;
    for (index, ref memory_type) in memory_prop.memory_types.iter().enumerate() {
        if memory_type_bits & 1 == 1 && f(memory_type.property_flags) {
            return Some(index as u32);
        }
        memory_type_bits >>= 1;
    }
    None
}

fn align_up(num: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());

    (num + align - 1) & align.wrapping_neg()
}

enum MapInfo {
    Persistent(*mut c_void),
    Mapped { count: usize, ptr: *mut c_void },
    Unmapped,
}

struct Block {
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    current: vk::DeviceSize,
    map_info: MapInfo,
}

impl Block {
    pub fn new(
        device: &RawDevice,
        size: vk::DeviceSize,
        memory_type_index: u32,
        persistent: bool,
    ) -> VkResult<Self> {
        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.inner.allocate_memory(&memory_allocate_info, None) }?;

        let map_info = if persistent {
            MapInfo::Persistent(unsafe {
                device
                    .inner
                    .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?
            })
        } else {
            MapInfo::Unmapped
        };

        Ok(Self {
            memory,
            size,
            current: 0,
            map_info,
        })
    }
}

#[derive(Debug)]
pub enum MemoryUsage {
    DeviceOnly,
    HostToDevice,
    HostOnly,
}

#[derive(Debug)]
pub struct Allocation {
    pub memory_type_index: u32,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
}

/// very dumb bump allocator
/// the interface is similar to vk-mem's on purpose
pub struct Allocator {
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: Rc<RawDevice>,
    blocks: Mutex<[Option<Block>; vk::MAX_MEMORY_TYPES]>,
}

const MEBIBYTE: usize = 2usize.pow(20);

impl Allocator {
    pub fn new(device: &Device, instance: &Instance) -> Self {
        // this wouldn't work if MAX_MEMORY_TYPES wasn't exactly 32 because
        // rust STILL lacks const generics
        let blocks = Default::default();

        let memory_properties = unsafe {
            instance
                .inner
                .get_physical_device_memory_properties(device.physical_device)
        };

        Self {
            device: device.inner.clone(),
            memory_properties,
            blocks,
        }
    }

    pub fn create_buffer(
        &self,
        buffer_info: &vk::BufferCreateInfo,
        usage: MemoryUsage,
    ) -> VkResult<(Buffer, Allocation)> {
        let buffer = Buffer::new(&self.device, &buffer_info)?;

        let memory_requirements = unsafe {
            self.device
                .inner
                .get_buffer_memory_requirements(buffer.inner)
        };

        let (memory_type_index, used_flags) =
            find_memorytype_index(&memory_requirements, &self.memory_properties, usage)
                .expect("no suitable memory type found");

        let is_persistent = used_flags.contains(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let mut blocks = self.blocks.lock();

        if blocks[memory_type_index as usize].is_none() {
            let block = Block::new(
                &self.device,
                (4 * MEBIBYTE) as u64,
                memory_type_index,
                is_persistent,
            )?;

            blocks[memory_type_index as usize] = Some(block);
        }

        let mut block = &mut blocks[memory_type_index as usize].as_mut().unwrap();

        let aligned = align_up(block.current, memory_requirements.alignment);
        let padding = aligned - block.current;
        let used = padding + memory_requirements.size;

        if block.current + used > block.size {
            panic!("allocator block overflow");
        }

        block.current += used;

        unsafe {
            self.device
                .inner
                .bind_buffer_memory(buffer.inner, block.memory, aligned)
        }?;

        let allocation = Allocation {
            size: used,
            offset: aligned,
            memory_type_index,
        };

        eprintln!("{:#?}", allocation);

        Ok((buffer, allocation))
    }

    pub fn map(&self, allocation: &Allocation) -> VkResult<*mut u8> {
        let mut blocks = self.blocks.lock();
        let mut block = &mut blocks[allocation.memory_type_index as usize]
            .as_mut()
            .unwrap();

        let ptr = match block.map_info {
            MapInfo::Persistent(ptr) => ptr,
            MapInfo::Mapped { ref mut count, ptr } => {
                *count += 1;

                ptr
            }
            MapInfo::Unmapped => {
                let ptr = unsafe {
                    self.device.inner.map_memory(
                        block.memory,
                        0,
                        block.size,
                        vk::MemoryMapFlags::empty(),
                    )?
                };

                block.map_info = MapInfo::Mapped { count: 1, ptr };

                ptr
            }
        };

        unsafe { Ok(ptr.add(allocation.offset as usize) as *mut u8) }
    }

    pub fn unmap(&self, allocation: &Allocation) {
        let mut blocks = self.blocks.lock();
        let block = &mut blocks[allocation.memory_type_index as usize]
            .as_mut()
            .unwrap();

        match block.map_info {
            MapInfo::Mapped { ref mut count, .. } if *count > 1 => {
                *count -= 1;
            }
            MapInfo::Mapped { .. } => unsafe {
                self.device.inner.unmap_memory(block.memory);
            },
            _ => (),
        }
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        let blocks = self.blocks.lock();
        for block in blocks.iter() {
            if let Some(block) = block.as_ref() {
                unsafe {
                    self.device.inner.free_memory(block.memory, None);
                }
            }
        }
    }
}
