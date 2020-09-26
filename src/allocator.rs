use ash::{
    prelude::VkResult,
    version::{DeviceV1_0, InstanceV1_0},
    vk, Device, Instance,
};

use parking_lot::Mutex;

use std::ffi::c_void;
use std::rc::Rc;

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
        property_flags == preferred_flags
    });
    if best_suitable_index.is_some() {
        return best_suitable_index.map(|r| (r, preferred_flags));
    }
    // Otherwise find a memory flag that works
    find_memorytype_index_f(memory_req, memory_prop, |property_flags| {
        property_flags == acceptable_flags
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

#[derive(Default)]
struct Block {
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    current: vk::DeviceSize,
    persistent_ptr: Option<*mut c_void>,
}

impl Block {
    pub fn new(
        device: &Device,
        size: vk::DeviceSize,
        memory_type_index: u32,
        persistent: bool,
    ) -> VkResult<Self> {
        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.allocate_memory(&memory_allocate_info, None) }?;

        let map_ptr = if persistent {
            Some(unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())? })
        } else {
            None
        };

        Ok(Self {
            memory,
            size,
            current: 0,
            persistent_ptr: map_ptr,
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
    memory_type_index: u32,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

/// very dumb bump allocator
/// the interface is similar to vk-mem's on purpose
pub struct Allocator {
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: Rc<Device>,
    blocks: Mutex<[Mutex<Option<Block>>; vk::MAX_MEMORY_TYPES]>,
}

const MEBIBYTE: usize = 2usize.pow(20);

impl Allocator {
    pub fn new(
        physical_device: vk::PhysicalDevice,
        device: Rc<Device>,
        instance: &Instance,
    ) -> Self {
        // this wouldn't work if MAX_MEMORY_TYPES wasn't exactly 32 because
        // rust STILL lacks const generics
        let blocks = Default::default();

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Self {
            memory_properties,
            device,
            blocks,
        }
    }

    pub fn create_buffer(
        &self,
        buffer_info: &vk::BufferCreateInfo,
        usage: MemoryUsage,
    ) -> VkResult<(vk::Buffer, Allocation)> {
        let buffer = unsafe { self.device.create_buffer(&buffer_info, None) }.unwrap();

        let memory_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let (memory_type_index, used_flags) =
            find_memorytype_index(&memory_requirements, &self.memory_properties, usage)
                .expect("no suitable memory type found");

        let is_persistent = used_flags.contains(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let mut blocks = self.blocks.lock();

        if blocks[memory_type_index as usize].lock().is_none() {
            let block = Block::new(
                &self.device,
                (4 * MEBIBYTE) as u64,
                memory_type_index,
                is_persistent,
            )?;

            blocks[memory_type_index as usize] = Mutex::new(Some(block));
        }

        let mut block = blocks[memory_type_index as usize].lock();

        let block = block.as_mut().unwrap();

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
            size: used,
            offset: aligned,
            memory_type_index,
        };

        eprintln!("{:#?}", allocation);

        Ok((buffer, allocation))
    }

    pub fn map(&self, allocation: &Allocation) -> VkResult<*mut u8> {
        let blocks = self.blocks.lock();
        let mut block = blocks[allocation.memory_type_index as usize].lock();
        let block = block.as_mut().unwrap();
        block.persistent_ptr.map_or_else(
            || unsafe {
                Ok(self.device.map_memory(
                    block.memory,
                    allocation.offset,
                    allocation.size,
                    vk::MemoryMapFlags::empty(),
                )? as *mut u8)
            },
            |ptr| unsafe { Ok(ptr.add(allocation.offset as usize) as *mut u8) },
        )
    }

    pub fn unmap(&self, allocation: &Allocation) {
        let blocks = self.blocks.lock();
        let mut block = blocks[allocation.memory_type_index as usize].lock();
        let block = block.as_mut().unwrap();

        unsafe {
            self.device.unmap_memory(block.memory);
        }
    }

    // (very big) TODO: put all resources in nice RAII containers so the drop order isn't completely borked
    pub fn hacky_manual_drop(&self) {
        let blocks = self.blocks.lock();
        for block_mutex in blocks.iter() {
            let block = block_mutex.lock();

            if let Some(block) = block.as_ref() {
                unsafe {
                    self.device.free_memory(block.memory, None);
                }
            }
        }
    }
}
