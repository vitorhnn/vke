use crate::device::RawDevice;
use crate::{Device, Transfer};
use ash::prelude::VkResult;
use ash::vk;
use gpu_allocator::vulkan::Allocation;
use std::rc::Rc;

#[derive(Debug)]
pub struct TextureInfo {
    pub extent: vk::Extent3D,
    pub format: vk::Format,
}

pub struct Texture {
    pub image: vk::Image,
    device: Rc<Device>,
    pub extent: vk::Extent3D,
    pub allocation: Option<Allocation>,
}

impl Texture {
    pub fn new(device: Rc<Device>, create_info: &vk::ImageCreateInfo) -> VkResult<Self> {
        let image = unsafe { device.inner.create_image(create_info, None)? };

        Ok(Texture {
            extent: create_info.extent.clone(),
            device,
            image,
            allocation: None,
        })
    }

    pub fn associate_allocation(&mut self, allocation: Allocation) {
        self.allocation = Some(allocation);
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            self.device.inner.destroy_image(self.image, None);
        }
    }
}
