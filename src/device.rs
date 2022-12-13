use std::ffi::CStr;
use std::ops::Deref;
use std::rc::Rc;

use ash::extensions::khr::{CreateRenderPass2, DynamicRendering, TimelineSemaphore};
use ash::vk::{
    KhrDepthStencilResolveFn, Semaphore, SemaphoreCreateInfo, SemaphoreType,
    SemaphoreTypeCreateInfo,
};
use ash::{extensions::khr::Swapchain as KhrSwapchain, prelude::VkResult, vk, Device as VkDevice};

use crate::instance::Instance;
use crate::surface::Surface;
use crate::swapchain;

use crate::queue::Queue;

// Used to break cycles, mostly
pub struct RawDevice {
    pub inner: VkDevice,
}

impl Deref for RawDevice {
    type Target = VkDevice;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Drop for RawDevice {
    fn drop(&mut self) {
        eprintln!("drop device");
        unsafe { self.inner.destroy_device(None) };
    }
}

// I'm not quite sure how good this abstraction is
// because it bundles Vulkan physical devices + logical devices + queues together
// it actually bundles even more, welp
// but we'll roll with it for now
pub struct Device {
    pub inner: Rc<RawDevice>,
    pub timeline_semaphore: TimelineSemaphore,
    pub dynamic_rendering: DynamicRendering,
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue: Rc<Queue>,
    pub transfer_queue: Rc<Queue>,
}

/// Returned by pick_vk_device. Contains the selected device and the index of the queue families
/// we're going to use for operations. We prefer dedicated queue families but fallback to the same queue family if needed
struct SelectedDeviceInfo {
    device: vk::PhysicalDevice,
    // we make the (currently) reasonable assumption that we can present from the graphics queue
    graphics_family: u32,
    transfer_family: u32,
}

impl Device {
    pub fn from_heuristics(
        instance: &Instance,
        surface: &Surface,
        desired_present_mode: vk::PresentModeKHR,
    ) -> VkResult<Self> {
        let selected_device_info = Device::pick_vk_device(instance, surface, desired_present_mode)?;

        // FIXME: this should be recoverable (probably)
        let selected_device_info = selected_device_info.expect("no suitable vulkan device");

        let physical_device = selected_device_info.device;
        let graphics_queue_family_index = selected_device_info.graphics_family;
        let transfer_queue_family_index = selected_device_info.transfer_family;

        let priorities = [1.0];

        let create_graphics_queue_info_builder = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(graphics_queue_family_index)
            .queue_priorities(&priorities);

        let create_transfer_queue_info_builder = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(transfer_queue_family_index)
            .queue_priorities(&priorities);

        let queues = if graphics_queue_family_index != transfer_queue_family_index {
            vec![
                create_graphics_queue_info_builder.build(),
                create_transfer_queue_info_builder.build(),
            ]
        } else {
            vec![create_graphics_queue_info_builder.build()]
        };

        let mut timeline_semaphores_features =
            vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR::builder().timeline_semaphore(true);
        let mut dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeaturesKHR::builder().dynamic_rendering(true);
        let device_features_builder = vk::PhysicalDeviceFeatures::builder();

        let device_extensions = [
            KhrSwapchain::name().as_ptr(),
            TimelineSemaphore::name().as_ptr(),
            DynamicRendering::name().as_ptr(),
            CreateRenderPass2::name().as_ptr(),
            KhrDepthStencilResolveFn::name().as_ptr(),
        ];

        let create_device_info_builder = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queues)
            // this is *technically* wrong, device layers != instance layers
            .enabled_layer_names(&instance.layers)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&device_features_builder)
            .push_next(&mut timeline_semaphores_features)
            .push_next(&mut dynamic_rendering_features);

        let raw_device = Rc::new(RawDevice {
            inner: unsafe {
                instance
                    .inner
                    .create_device(physical_device, &create_device_info_builder, None)?
            },
        });

        let raw_graphics_queue =
            unsafe { raw_device.get_device_queue(graphics_queue_family_index, 0) };
        let raw_transfer_queue =
            unsafe { raw_device.get_device_queue(transfer_queue_family_index, 0) };

        let graphics_queue = Rc::new(Queue::new(
            raw_device.clone(),
            graphics_queue_family_index,
            raw_graphics_queue,
        )?);

        let transfer_queue = Rc::new(Queue::new(
            raw_device.clone(),
            transfer_queue_family_index,
            raw_transfer_queue,
        )?);

        let timeline_semaphore = TimelineSemaphore::new(&instance.inner, &raw_device);
        let dynamic_rendering = DynamicRendering::new(&instance.inner, &raw_device);

        Ok(Self {
            inner: raw_device,
            physical_device,
            graphics_queue,
            timeline_semaphore,
            dynamic_rendering,
            transfer_queue,
        })
    }

    // FIXME: this code is a horrible mess, refactor it someday (hah!)
    fn pick_vk_device(
        instance: &Instance,
        surface: &Surface,
        desired_present_mode: vk::PresentModeKHR,
    ) -> VkResult<Option<SelectedDeviceInfo>> {
        let is_device_suitable = |props: &vk::PhysicalDeviceProperties,
                                  device: &vk::PhysicalDevice|
         -> VkResult<bool> {
            unsafe {
                let support_info = swapchain::SwapchainSupportInfo::get(*device, surface)?;

                let suitable_swapchain_format = support_info.formats.into_iter().any(|format| {
                    format.format == vk::Format::B8G8R8A8_SRGB
                        && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                });

                let suitable_swapchain_present_mode = support_info
                    .present_modes
                    .into_iter()
                    .any(|present_mode| present_mode == desired_present_mode);

                Ok((props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                    || props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU)
                    && suitable_swapchain_format
                    && suitable_swapchain_present_mode)
            }
        };

        fn pick_queue(
            queues: &[vk::QueueFamilyProperties],
            is_optimal: impl Fn(vk::QueueFlags) -> bool,
            is_acceptable: impl Fn(vk::QueueFlags) -> bool,
        ) -> Option<u32> {
            // try to find an optimal family
            let optimal = queues
                .iter()
                .enumerate()
                .find(|(_, queue_properties)| is_optimal(queue_properties.queue_flags))
                .map(|(index, _)| index as u32);

            if optimal.is_some() {
                return optimal;
            }

            // just find something
            queues
                .iter()
                .enumerate()
                .find(|(_, queue_properties)| is_acceptable(queue_properties.queue_flags))
                .map(|(index, _)| index as u32)
        }

        let devices = unsafe { instance.inner.enumerate_physical_devices()? };

        for device in devices {
            let device_props = unsafe { instance.inner.get_physical_device_properties(device) };
            let device_name = unsafe { CStr::from_ptr(device_props.device_name.as_ptr()) };
            println!("evaluating device {:#?}", device_name);
            if !is_device_suitable(&device_props, &device)? {
                continue;
            }

            let queues = unsafe {
                instance
                    .inner
                    .get_physical_device_queue_family_properties(device)
            };

            // not quite sure if there's any benefit from a dedicated graphics queue...
            // so just pick whatever
            let maybe_graphics_queue_index = queues
                .iter()
                .enumerate()
                .try_find(|(index, queue_properties)| -> VkResult<bool> {
                    Ok(queue_properties
                        .queue_flags
                        .contains(vk::QueueFlags::GRAPHICS)
                        && unsafe {
                            surface.loader.get_physical_device_surface_support(
                                device,
                                *index as u32,
                                surface.surface,
                            )?
                        })
                })?
                .map(|(index, _)| index as u32);

            let maybe_transfer_queue_index = pick_queue(
                &queues,
                |flags| {
                    flags.contains(vk::QueueFlags::TRANSFER)
                        && !flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                },
                |flags| flags.contains(vk::QueueFlags::TRANSFER),
            );

            let result = maybe_graphics_queue_index.and_then(|graphics_family| {
                maybe_transfer_queue_index.map(|transfer_family| SelectedDeviceInfo {
                    device,
                    graphics_family,
                    transfer_family,
                })
            });

            if result.is_some() {
                println!("device chosen");
                return Ok(result);
            }
        }

        Ok(None)
    }

    pub fn create_timeline_semaphore(&self, initial_value: u64) -> VkResult<Semaphore> {
        let mut type_create_info = SemaphoreTypeCreateInfo::builder()
            .semaphore_type(SemaphoreType::TIMELINE_KHR)
            .initial_value(initial_value);

        let semaphore_create_info = SemaphoreCreateInfo::builder().push_next(&mut type_create_info);

        unsafe { self.inner.create_semaphore(&semaphore_create_info, None) }
    }

    pub fn insert_image_barrier(&self, params: &ImageBarrierParameters) {
        unsafe {
            self.inner.cmd_pipeline_barrier(
                params.command_buffer,
                params.src_stage_mask,
                params.dst_stage_mask,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(
                    &vk::ImageMemoryBarrier::builder()
                        .old_layout(params.old_layout)
                        .new_layout(params.new_layout)
                        .src_access_mask(params.src_access_mask)
                        .dst_access_mask(params.dst_access_mask)
                        .image(params.image)
                        .subresource_range(params.subresource_range)
                        .build(),
                ),
            )
        }
    }
}

pub struct ImageBarrierParameters {
    pub command_buffer: vk::CommandBuffer,
    pub image: vk::Image,
    pub src_access_mask: vk::AccessFlags,
    pub dst_access_mask: vk::AccessFlags,
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
    pub src_stage_mask: vk::PipelineStageFlags,
    pub dst_stage_mask: vk::PipelineStageFlags,
    pub subresource_range: vk::ImageSubresourceRange,
}
