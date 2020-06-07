use ash::{
    prelude::VkResult,
    version::{DeviceV1_0, InstanceV1_0},
    vk, Device, Instance,
};
use tinyvec::ArrayVec;

use std::ffi::c_void;
use std::rc::Rc;

// stolen from ash
fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    usage: MemoryUsage,
) -> Option<u32> {
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
        property_flags == preferred_flags
    });
    if best_suitable_index.is_some() {
        return best_suitable_index;
    }
    // Otherwise find a memory flag that works
    find_memorytype_index_f(memory_req, memory_prop, |property_flags| {
        property_flags == acceptable_flags
    })
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

#[derive(Default)]
struct Block {
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    current: vk::DeviceSize,
    memory_type_index: u32,
}

impl Block {
    pub fn new(device: &Device, size: vk::DeviceSize, memory_type_index: u32) -> VkResult<Self> {
        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.allocate_memory(&memory_allocate_info, None) }?;

        Ok(Self {
            memory,
            size,
            memory_type_index,
            current: 0,
        })
    }
}

#[derive(Debug)]
pub enum MemoryUsage {
    DeviceOnly,
    HostToDevice,
    HostOnly,
}

pub struct Allocation {
    memory: vk::DeviceMemory,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

impl Allocation {
    // TODO: these need to be in allocator
    pub fn map(&self, device: &Device, flags: vk::MemoryMapFlags) -> VkResult<*mut c_void> {
        unsafe { device.map_memory(self.memory, self.offset, self.size, flags) }
    }

    pub fn unmap(&self, device: &Device) {
        unsafe {
            device.unmap_memory(self.memory);
        }
    }
}

/// very dumb bump allocator
/// the interface is similar to vk-mem's on purpose
pub struct Allocator {
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: Rc<Device>,
    blocks: ArrayVec<[Block; vk::MAX_MEMORY_TYPES]>,
}

const MEBIBYTE: usize = 2usize.pow(20);

impl Allocator {
    pub fn new(
        physical_device: vk::PhysicalDevice,
        device: Rc<Device>,
        instance: &Instance,
    ) -> Self {
        let blocks = ArrayVec::new();

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Self {
            memory_properties,
            device,
            blocks,
        }
    }

    pub fn create_buffer(
        &mut self,
        buffer_info: &vk::BufferCreateInfo,
        usage: MemoryUsage,
    ) -> VkResult<(vk::Buffer, Allocation)> {
        let buffer = unsafe { self.device.create_buffer(&buffer_info, None) }.unwrap();

        let memory_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let memory_type_index =
            find_memorytype_index(&memory_requirements, &self.memory_properties, usage)
                .expect("no suitable memory type found");

        let block = {
            // TODO: this is hella janky
            let mut ret = None;
            for block in self.blocks.iter_mut() {
                if block.memory_type_index == memory_type_index {
                    ret = Some(block);
                }
            }

            if ret.is_none() {
                self.blocks.push(Block::new(
                    &self.device,
                    (4 * MEBIBYTE) as u64,
                    memory_type_index,
                )?);
                let last = self.blocks.len() - 1;

                ret = Some(&mut self.blocks[last]);
            }

            ret.unwrap()
        };

        let aligned = align_up(block.current, memory_requirements.alignment);
        let padding = aligned - block.current;
        let used = padding + memory_requirements.size;

        if block.current + used > block.size {
            panic!("allocator block overflow");
        }

        block.current += used;

        unsafe {
            self.device
                .bind_buffer_memory(buffer, block.memory, aligned)
        }?;

        let allocation = Allocation {
            memory: block.memory,
            size: used,
            offset: aligned,
        };

        Ok((buffer, allocation))
    }

    // (very big) TODO: put all resources in nice RAII containers so the drop order isn't completely borked
    pub fn hacky_manual_drop(&mut self) {
        for block in self.blocks.iter() {
            unsafe {
                self.device.free_memory(block.memory, None);
            }
        }
    }
}
