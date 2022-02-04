use crate::instance::Instance;

use ash::{extensions::khr::Surface as SurfaceLoader, prelude::VkResult, vk, vk::Handle, Entry};

pub struct Surface {
    pub surface: vk::SurfaceKHR,
    pub loader: SurfaceLoader,
}

impl Surface {
    pub fn new(instance: &Instance, raw_handle: u64) -> VkResult<Self> {
        let loader = SurfaceLoader::new(&instance.entry, &instance.inner);
        let surface = vk::SurfaceKHR::from_raw(raw_handle);

        Ok(Surface { surface, loader })
    }
}
