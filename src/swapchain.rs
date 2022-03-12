use ash::{extensions::khr::Swapchain as SwapchainLoader, prelude::VkResult, vk};

use crate::device::Device;
use crate::instance::Instance;
use crate::surface::Surface;

pub(crate) struct ImageResources {
    pub(crate) fence: vk::Fence,
    pub(crate) image_view: vk::ImageView,
    pub(crate) image: vk::Image,
}

pub(crate) struct SwapchainSupportInfo {
    pub(crate) capabilities: vk::SurfaceCapabilitiesKHR,
    pub(crate) formats: Vec<vk::SurfaceFormatKHR>,
    pub(crate) present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportInfo {
    pub(crate) fn get(device: vk::PhysicalDevice, surface: &Surface) -> VkResult<Self> {
        unsafe {
            Ok(Self {
                capabilities: surface
                    .loader
                    .get_physical_device_surface_capabilities(device, surface.surface)?,
                formats: surface
                    .loader
                    .get_physical_device_surface_formats(device, surface.surface)?,
                present_modes: surface
                    .loader
                    .get_physical_device_surface_present_modes(device, surface.surface)?,
            })
        }
    }
}

pub(crate) struct Swapchain {
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) loader: SwapchainLoader,
    pub(crate) extent: vk::Extent2D,
    pub(crate) image_resources: Vec<ImageResources>,
    pub(crate) support_info: SwapchainSupportInfo,
}

fn pick_desired_image_count(surface_capabilities: &vk::SurfaceCapabilitiesKHR) -> u32 {
    if surface_capabilities.max_image_count > 0
        && surface_capabilities.min_image_count + 1 > surface_capabilities.max_image_count
    {
        surface_capabilities.max_image_count
    } else {
        surface_capabilities.min_image_count + 1
    }
}

fn pick_swapchain_extent(
    surface_capabilities: &vk::SurfaceCapabilitiesKHR,
    desired: vk::Extent2D,
) -> vk::Extent2D {
    match surface_capabilities.current_extent.width {
        std::u32::MAX => desired,
        _ => surface_capabilities.current_extent,
    }
}

impl Swapchain {
    pub(crate) fn new(
        instance: &Instance,
        device: &Device,
        surface: &Surface,
        desired_extent: vk::Extent2D,
        present_mode: vk::PresentModeKHR,
    ) -> VkResult<Self> {
        let loader = SwapchainLoader::new(&instance.inner, &device.inner.inner);

        let support_info = SwapchainSupportInfo::get(device.physical_device, surface)?;

        let extent = pick_swapchain_extent(&support_info.capabilities, desired_extent);

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface.surface)
            .min_image_count(pick_desired_image_count(&support_info.capabilities))
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_format(vk::Format::B8G8R8A8_SRGB)
            .image_extent(extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(support_info.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1);

        let swapchain = unsafe { loader.create_swapchain(&swapchain_create_info, None) }?;

        let present_images = unsafe { loader.get_swapchain_images(swapchain) }?;

        let image_resources: Vec<ImageResources> = present_images
            .iter()
            .map(|&image| {
                let create_view_info = vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::B8G8R8A8_SRGB)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);

                let image_view =
                    unsafe { device.inner.create_image_view(&create_view_info, None)? };

                Ok(ImageResources {
                    image,
                    image_view,
                    fence: vk::Fence::null(),
                })
            })
            .collect::<Result<_, _>>()?;

        let swapchain = Swapchain {
            swapchain,
            loader,
            extent,
            image_resources,
            support_info,
        };

        println!("VKe swapchain created!");
        println!("images: {}", swapchain.image_resources.len());
        println!("extent: {:#?}", swapchain.extent);
        println!("present mode: {:#?}", present_mode);

        Ok(swapchain)
    }
}
