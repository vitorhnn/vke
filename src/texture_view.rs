use crate::{Device, Texture};
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::ImageViewCreateInfo;
use std::rc::Rc;

pub struct TextureView {
    pub inner: vk::ImageView,
    pub texture: Texture,
    device: Rc<Device>,
}

impl TextureView {
    pub fn new(
        device: Rc<Device>,
        texture: Texture,
        create_info: &ImageViewCreateInfo,
    ) -> VkResult<Self> {
        let image_view = unsafe { device.inner.create_image_view(&create_info, None)? };

        Ok(TextureView {
            inner: image_view,
            texture,
            device,
        })
    }
}

impl Drop for TextureView {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_image_view(self.inner, None) };
    }
}
