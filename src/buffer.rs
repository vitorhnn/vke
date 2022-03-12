use ash::prelude::VkResult;
use std::rc::Rc;

use ash::vk;

use crate::device::Device;

pub struct Buffer {
    pub inner: vk::Buffer,
    pub device: Rc<Device>,
}

impl Buffer {
    pub fn new(device: Rc<Device>, create_info: &vk::BufferCreateInfo) -> VkResult<Self> {
        let buffer = unsafe { device.inner.create_buffer(create_info, None)? };

        Ok(Buffer {
            inner: buffer,
            device,
        })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.inner.destroy_buffer(self.inner, None);
        }
    }
}
