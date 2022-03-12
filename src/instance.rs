use std::error::Error;
use std::ffi::CStr;
use std::os::raw::c_char;

use ash::{prelude::VkResult, vk, vk::Handle, Entry, Instance as VkInstance};
use smallvec::{smallvec, SmallVec};

pub struct Instance {
    pub entry: Entry,
    pub inner: VkInstance,
    pub layers: Vec<*const c_char>,
}

const VALIDATION_LAYER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

const APPLICATION_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"vke\0") };

impl Instance {
    // TODO: get rid of boxed errors
    pub fn new(window: &sdl2::video::Window) -> Result<Self, Box<dyn Error>> {
        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::builder()
            .application_name(APPLICATION_NAME)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(APPLICATION_NAME)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 1, 0));

        let layers: SmallVec<[*const c_char; 1]> =
            if Instance::are_validation_layers_supported(&entry)?
                && Instance::should_use_validation_layers()
            {
                smallvec![VALIDATION_LAYER_NAME.as_ptr()]
            } else {
                smallvec![]
            };

        let enabled_extensions = Instance::get_vulkan_instance_extensions(window);

        let create_info_builder = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&enabled_extensions)
            .enabled_layer_names(&layers);

        let instance = unsafe { entry.create_instance(&create_info_builder, None)? };

        Ok(Self {
            inner: instance,
            entry,
            layers: layers.to_vec(),
        })
    }

    pub fn raw_handle(&self) -> u64 {
        self.inner.handle().as_raw()
    }

    fn are_validation_layers_supported(entry: &Entry) -> VkResult<bool> {
        let properties = entry.enumerate_instance_layer_properties()?;

        Ok(properties
            .iter()
            .any(|x| unsafe { CStr::from_ptr(x.layer_name.as_ptr()) } == VALIDATION_LAYER_NAME))
    }

    const fn should_use_validation_layers() -> bool {
        cfg!(debug_assertions)
    }

    fn get_vulkan_instance_extensions(window: &sdl2::video::Window) -> Vec<*const i8> {
        let mut count: u32 = 0;

        if unsafe {
            sdl2::sys::SDL_Vulkan_GetInstanceExtensions(
                window.raw(),
                &mut count,
                std::ptr::null_mut(),
            )
        } == sdl2::sys::SDL_bool::SDL_FALSE
        {
            panic!("sdl garbage");
        }

        let mut names = vec![std::ptr::null(); count as usize];

        if unsafe {
            sdl2::sys::SDL_Vulkan_GetInstanceExtensions(
                window.raw(),
                &mut count,
                names.as_mut_ptr(),
            )
        } == sdl2::sys::SDL_bool::SDL_FALSE
        {
            panic!("sdl garbage");
        }

        names
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        eprintln!("destroy instance");
        unsafe {
            self.inner.destroy_instance(None);
        }
    }
}
