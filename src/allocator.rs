use ash::{prelude::VkResult, vk};

use parking_lot::Mutex;

use gpu_allocator::MemoryLocation;
use std::ffi::c_void;
use std::rc::Rc;

use crate::buffer::Buffer;
use crate::device::{Device, RawDevice};
use crate::instance::Instance;

use crate::Texture;
use gpu_allocator::vulkan::*;

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

#[derive(Debug)]
pub enum MemoryUsage {
    DeviceOnly,
    HostToDevice,
    HostOnly,
}

/// very dumb bump allocator
/// the interface is similar to vk-mem's on purpose
pub struct Allocator {
    inner: Mutex<gpu_allocator::vulkan::Allocator>,
    device: Rc<Device>,
}

const MEBIBYTE: usize = 2usize.pow(20);

fn to_gpu_allocator(usage: MemoryUsage) -> MemoryLocation {
    match usage {
        MemoryUsage::DeviceOnly => MemoryLocation::GpuOnly,
        MemoryUsage::HostToDevice => MemoryLocation::CpuToGpu,
        MemoryUsage::HostOnly => MemoryLocation::CpuToGpu,
    }
}

impl Allocator {
    pub fn new(device: Rc<Device>, instance: &Instance) -> Self {
        let inner = gpu_allocator::vulkan::Allocator::new(&AllocatorCreateDesc {
            device: device.inner.inner.clone(),
            physical_device: device.physical_device,
            instance: instance.inner.clone(),
            buffer_device_address: false,
            debug_settings: Default::default(),
        })
        .unwrap();

        Self {
            device: device,
            inner: Mutex::new(inner),
        }
    }

    pub fn create_buffer(
        &self,
        buffer_info: &vk::BufferCreateInfo,
        usage: MemoryUsage,
    ) -> VkResult<(Buffer, Allocation)> {
        let buffer = Buffer::new(self.device.clone(), &buffer_info)?;
        let mut allocator = self.inner.lock();

        let memory_requirements = unsafe {
            self.device
                .inner
                .get_buffer_memory_requirements(buffer.inner)
        };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "vke allocator created buffer",
                location: to_gpu_allocator(usage),
                linear: true,
                requirements: memory_requirements,
            })
            .unwrap();

        unsafe {
            self.device.inner.bind_buffer_memory(
                buffer.inner,
                allocation.memory(),
                allocation.offset(),
            )
        }?;

        Ok((buffer, allocation))
    }

    pub fn create_texture(
        &self,
        image_info: &vk::ImageCreateInfo,
        usage: MemoryUsage,
    ) -> VkResult<(Texture, Allocation)> {
        let texture = Texture::new(self.device.clone(), &image_info)?;
        let mut allocator = self.inner.lock();

        let memory_requirements = unsafe {
            self.device
                .inner
                .get_image_memory_requirements(texture.image)
        };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "vke allocator created image",
                location: to_gpu_allocator(usage),
                linear: false,
                requirements: memory_requirements,
            })
            .unwrap();

        unsafe {
            self.device.inner.bind_image_memory(
                texture.image,
                allocation.memory(),
                allocation.offset(),
            )
        }?;

        Ok((texture, allocation))
    }

    pub fn map(&self, allocation: &Allocation) -> VkResult<*mut u8> {
        let ptr = match allocation.mapped_ptr() {
            Some(ptr) => ptr.as_ptr(),
            None => unsafe {
                self.device.inner.map_memory(
                    allocation.memory(),
                    allocation.offset(),
                    allocation.size(),
                    vk::MemoryMapFlags::empty(),
                )?
            },
        };

        Ok(ptr as *mut u8)
    }

    pub fn unmap(&self, _allocation: &Allocation) {
        eprintln!("unmap does nothing currently");
    }
}
