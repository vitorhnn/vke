use ash::{
    extensions::khr::Surface as SurfaceLoader, prelude::VkResult, vk, vk::Handle, Entry, Instance,
};

pub(crate) struct Surface {
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) loader: SurfaceLoader,
}

impl Surface {
    pub(crate) fn new(entry: &Entry, instance: &Instance, raw_handle: u64) -> VkResult<Self> {
        let loader = SurfaceLoader::new(entry, instance);
        let surface = vk::SurfaceKHR::from_raw(raw_handle);

        Ok(Surface { surface, loader })
    }
}
