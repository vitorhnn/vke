use crate::Device;
use ash::prelude::VkResult;
use ash::vk;
use std::rc::Rc;

pub struct Sampler {
    device: Rc<Device>,
    pub inner: vk::Sampler,
}

impl Sampler {
    pub fn new(device: Rc<Device>, create_info: vk::SamplerCreateInfo) -> VkResult<Self> {
        let sampler = unsafe { device.inner.create_sampler(&create_info, None)? };

        Ok(Self {
            device,
            inner: sampler,
        })
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        eprintln!("drop sampler");
        unsafe { self.device.inner.destroy_sampler(self.inner, None) };
    }
}
