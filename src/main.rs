// rust devs pls stabilize
#![feature(const_cstr_unchecked)]
#![feature(try_find)]

use ash::{
    extensions::khr::{Surface, Swapchain},
    version::DeviceV1_0,
    version::EntryV1_0,
    version::InstanceV1_0,
    vk,
    vk::Handle,
    Device, Entry, Instance,
};

use sdl2::event::Event;

use std::error::Error;
use std::ffi::CStr;

use memoffset::offset_of;

use glam::{Mat4, Vec3};

// kinda lifted from DXVK's Presenter
struct FrameResources {
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
    uniform_buffer: vk::Buffer,
    uniform_buffer_memory: vk::DeviceMemory,
    ubo_ptr: *mut Ubo,
    descriptor_set: vk::DescriptorSet,
}

struct ImageResources {
    fence: vk::Fence,
    framebuffer: vk::Framebuffer,
    image_view: vk::ImageView,
}

struct Application {
    sdl_ctx: sdl2::Sdl,
    sdl_video_ctx: sdl2::VideoSubsystem,
    window: sdl2::video::Window,
    entry: Entry,
    instance: Instance,
    device: Device,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    surface: vk::SurfaceKHR,
    surface_loader: Surface,
    swapchain: vk::SwapchainKHR,
    swapchain_loader: Swapchain,
    swapchain_extent: vk::Extent2D,
    image_resources: Vec<ImageResources>,
    frame_resources: Vec<FrameResources>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    render_pass: vk::RenderPass,
    graphics_command_pool: vk::CommandPool,
    transfer_command_pool: vk::CommandPool,
    graphics_queue: vk::Queue,
    transfer_queue: vk::Queue,
    image_count: u32,
    current_frame: usize,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    descriptor_pool: vk::DescriptorPool,
    frame_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone)]
struct Ubo {
    model_view: Mat4,
    projection: Mat4,
}

/// Returned by pick_vk_device. Contains the selected device and the index of the queue families
/// we're going to use for operations. We prefer dedicated queue families but fallback to the same queue family if needed
struct SelectedDeviceInfo {
    device: vk::PhysicalDevice,
    // we make the (currently) reasonable assumption that we can present from the graphics queue
    graphics_family: u32,
    transfer_family: u32,
}

impl Vertex {
    fn get_binding_descriptor() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    fn get_attribute_descriptors() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Vertex, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, color) as u32,
            },
        ]
    }
}

// stolen from ash
pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    // Try to find an exactly matching memory flag
    let best_suitable_index =
        find_memorytype_index_f(memory_req, memory_prop, flags, |property_flags, flags| {
            property_flags == flags
        });
    if best_suitable_index.is_some() {
        return best_suitable_index;
    }
    // Otherwise find a memory flag that works
    find_memorytype_index_f(memory_req, memory_prop, flags, |property_flags, flags| {
        property_flags & flags == flags
    })
}

pub fn find_memorytype_index_f<F: Fn(vk::MemoryPropertyFlags, vk::MemoryPropertyFlags) -> bool>(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
    f: F,
) -> Option<u32> {
    let mut memory_type_bits = memory_req.memory_type_bits;
    for (index, ref memory_type) in memory_prop.memory_types.iter().enumerate() {
        if memory_type_bits & 1 == 1 && f(memory_type.property_flags, flags) {
            return Some(index as u32);
        }
        memory_type_bits >>= 1;
    }
    None
}

const VALIDATION_LAYER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

const APPLICATION_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"vke\0") };

const SHADER_MAIN_FN_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

const FRAMES_IN_FLIGHT: usize = 3;

impl Application {
    fn new() -> Result<Self, Box<dyn Error>> {
        let sdl_ctx = sdl2::init()?;
        let sdl_video_ctx = sdl_ctx.video()?;
        let window = sdl_video_ctx.window("vke", 1600, 900).vulkan().build()?;

        let entry = Entry::new()?;

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&APPLICATION_NAME)
            .application_version(vk::make_version(0, 1, 0))
            .api_version(vk::make_version(1, 1, 0));

        let layers = if Application::are_validation_layers_supported(&entry)?
            && Application::should_use_validation_layers()
        {
            vec![VALIDATION_LAYER_NAME.as_ptr()]
        } else {
            vec![]
        };

        // why the fuck is this converting to &str
        // like actually why
        // we just have to do extra work to convert it back to an array of cstr
        // actually no, this is cast to &'static str, so I can't even get a CStr back because I'd have to allocate a nul terminator
        // so yeah, just reimplement it
        // let enabled_extensions = window.vulkan_instance_extensions()?.into_iter().map(|str| CStr:)

        let enabled_extensions = Application::get_vulkan_instance_extensions(&window);

        let create_info_builder = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&enabled_extensions)
            .enabled_layer_names(&layers);

        let instance = unsafe { entry.create_instance(&create_info_builder, None)? };

        let surface = vk::SurfaceKHR::from_raw(
            window.vulkan_create_surface(instance.handle().as_raw() as usize)?,
        );

        let surface_loader = Surface::new(&entry, &instance);

        let physical_device = Application::pick_vk_device(&instance, &surface_loader, surface)?;

        let selected_device_info = physical_device.expect("no suitable vulkan device");

        let physical_device = selected_device_info.device;
        let graphics_queue_family_index = selected_device_info.graphics_family;
        let transfer_queue_family_index = selected_device_info.transfer_family;

        let device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let priorities = [1.0];

        let create_graphics_queue_info_builder = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(graphics_queue_family_index)
            .queue_priorities(&priorities);

        let create_transfer_queue_info_builder = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(transfer_queue_family_index)
            .queue_priorities(&priorities);

        let queues = [
            create_graphics_queue_info_builder.build(),
            create_transfer_queue_info_builder.build(),
        ];

        let device_features_builder = vk::PhysicalDeviceFeatures::builder();

        let device_extensions = [Swapchain::name().as_ptr()];

        let create_device_info_builder = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queues)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&device_features_builder);

        let device =
            unsafe { instance.create_device(physical_device, &create_device_info_builder, None)? };

        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family_index, 0) };

        let transfer_queue = unsafe { device.get_device_queue(transfer_queue_family_index, 0) };

        let surface_capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        };

        let swapchain_extent = Application::pick_vk_surface_extent(&surface_capabilities)?;

        let desired_image_count = Application::pick_vk_desired_image_count(&surface_capabilities);

        let swapchain_loader = Swapchain::new(&instance, &device);

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(desired_image_count)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_format(vk::Format::B8G8R8A8_SRGB)
            .image_extent(swapchain_extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO_RELAXED)
            .clipped(true)
            .image_array_layers(1);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        let present_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        let image_count = present_images.len() as u32;

        let command_pool_create_info = vk::CommandPoolCreateInfo {
            queue_family_index: graphics_queue_family_index,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };

        let graphics_command_pool =
            unsafe { device.create_command_pool(&command_pool_create_info, None)? };

        let command_pool_create_info = vk::CommandPoolCreateInfo {
            queue_family_index: transfer_queue_family_index,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };

        let transfer_command_pool =
            unsafe { device.create_command_pool(&command_pool_create_info, None)? };

        let mut frame_resources: Vec<FrameResources> = (0..FRAMES_IN_FLIGHT)
            .map::<Result<FrameResources, Box<dyn Error>>, _>(|_| {
                let semaphore_create_info = vk::SemaphoreCreateInfo::builder();

                let fence_create_info =
                    vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

                let image_available =
                    unsafe { device.create_semaphore(&semaphore_create_info, None)? };
                let render_finished =
                    unsafe { device.create_semaphore(&semaphore_create_info, None)? };
                let fence = unsafe { device.create_fence(&fence_create_info, None)? };

                let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::builder()
                    .command_pool(graphics_command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);

                let command_buffers =
                    unsafe { device.allocate_command_buffers(&command_buffer_alloc_info)? };

                let command_buffer = command_buffers[0];

                let (uniform_buffer, uniform_buffer_memory) = Application::create_buffer(
                    &device,
                    std::mem::size_of::<Ubo>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    device_memory_properties,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                        | vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT,
                    vk::SharingMode::EXCLUSIVE,
                )?;

                // we're deviating from vulkan-tutorial here
                // most modern gpus have some host visible, cached and device local memory
                // this is called "persistent buffer mapping" in opengl parlance
                let ubo_ptr = unsafe {
                    device.map_memory(
                        uniform_buffer_memory,
                        0,
                        std::mem::size_of::<Ubo>() as u64,
                        vk::MemoryMapFlags::empty(),
                    )?
                } as *mut Ubo;

                Ok(FrameResources {
                    image_available,
                    render_finished,
                    fence,
                    command_buffer,
                    uniform_buffer,
                    uniform_buffer_memory,
                    ubo_ptr,
                    descriptor_set: vk::DescriptorSet::null(),
                })
            })
            .collect::<Result<_, _>>()?;

        let render_pass = Application::create_render_pass(&device)?;

        let (pipeline_layout, pipeline, descriptor_set_layout) =
            Application::create_graphics_pipeline(&device, swapchain_extent, render_pass)?;

        let image_resources: Vec<ImageResources> = present_images
            .iter()
            .map::<Result<ImageResources, Box<dyn Error>>, _>(|&image| {
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

                let image_view = unsafe { device.create_image_view(&create_view_info, None)? };

                let attachments = [image_view];

                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain_extent.width)
                    .height(swapchain_extent.height)
                    .layers(1);

                let framebuffer =
                    unsafe { device.create_framebuffer(&framebuffer_create_info, None) }?;

                Ok(ImageResources {
                    framebuffer,
                    image_view,
                    fence: vk::Fence::null(),
                })
            })
            .collect::<Result<_, _>>()?;

        let vertices = [
            Vertex {
                pos: [-0.5, -0.5],
                color: [1.0, 0.0, 0.0],
            },
            Vertex {
                pos: [0.5, -0.5],
                color: [0.0, 1.0, 0.0],
            },
            Vertex {
                pos: [0.5, 0.5],
                color: [0.0, 0.0, 1.0],
            },
            Vertex {
                pos: [-0.5, 0.5],
                color: [1.0, 1.0, 1.0],
            },
        ];

        let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];

        let vertex_buffer_size = std::mem::size_of_val(&vertices) as u64;
        let index_buffer_size = std::mem::size_of_val(&indices) as u64;

        let (staging_buffer, staging_buffer_memory) = Application::create_buffer(
            &device,
            vertex_buffer_size + index_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::SharingMode::EXCLUSIVE,
        )?;

        let (vertex_buffer, vertex_buffer_memory) = Application::create_buffer(
            &device,
            vertex_buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            device_memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::SharingMode::EXCLUSIVE,
        )?;

        let (index_buffer, index_buffer_memory) = Application::create_buffer(
            &device,
            index_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            device_memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::SharingMode::EXCLUSIVE,
        )?;

        unsafe {
            let mapped = device.map_memory(
                staging_buffer_memory,
                0,
                vertex_buffer_size + index_buffer_size,
                vk::MemoryMapFlags::empty(),
            )?;

            {
                let slice = std::slice::from_raw_parts_mut(
                    mapped as *mut Vertex,
                    vertex_buffer_size as usize / std::mem::size_of::<Vertex>(),
                );

                slice.copy_from_slice(&vertices);
            }

            {
                let slice = std::slice::from_raw_parts_mut(
                    mapped.add(vertex_buffer_size as usize) as *mut u16,
                    index_buffer_size as usize / std::mem::size_of::<u16>(),
                );

                slice.copy_from_slice(&indices);
            }

            device.unmap_memory(staging_buffer_memory);
        }

        // the following code assumes that we get a different transfer and graphics queue
        // which is the case in my maxwell 2 gpu, but not in my work laptop
        // hell, my work laptop is UMA, which is even easier

        let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(graphics_command_pool)
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY);

        let graphics_buffers =
            unsafe { device.allocate_command_buffers(&command_buffer_alloc_info)? };

        let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(transfer_command_pool)
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY);

        let transfer_buffers =
            unsafe { device.allocate_command_buffers(&command_buffer_alloc_info)? };

        let graphics_queue_buffer = graphics_buffers[0];
        let transfer_queue_buffer = transfer_buffers[0];

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let memory_barriers = [
            vk::BufferMemoryBarrier {
                buffer: vertex_buffer,
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::empty(),
                src_queue_family_index: transfer_queue_family_index,
                dst_queue_family_index: graphics_queue_family_index,
                offset: 0,
                size: vk::WHOLE_SIZE,
                ..Default::default()
            },
            vk::BufferMemoryBarrier {
                buffer: index_buffer,
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::empty(),
                src_queue_family_index: transfer_queue_family_index,
                dst_queue_family_index: graphics_queue_family_index,
                offset: 0,
                size: vk::WHOLE_SIZE,
                ..Default::default()
            },
            vk::BufferMemoryBarrier {
                buffer: vertex_buffer,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
                src_queue_family_index: transfer_queue_family_index,
                dst_queue_family_index: graphics_queue_family_index,
                offset: 0,
                size: vk::WHOLE_SIZE,
                ..Default::default()
            },
            vk::BufferMemoryBarrier {
                buffer: index_buffer,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::INDEX_READ,
                src_queue_family_index: transfer_queue_family_index,
                dst_queue_family_index: graphics_queue_family_index,
                offset: 0,
                size: vk::WHOLE_SIZE,
                ..Default::default()
            },
        ];

        unsafe {
            device.begin_command_buffer(transfer_queue_buffer, &begin_info)?;

            let regions = [
                vk::BufferCopy {
                    dst_offset: 0,
                    src_offset: 0,
                    size: vertex_buffer_size,
                },
                vk::BufferCopy {
                    dst_offset: 0,
                    src_offset: vertex_buffer_size,
                    size: index_buffer_size,
                },
            ];

            device.cmd_copy_buffer(
                transfer_queue_buffer,
                staging_buffer,
                vertex_buffer,
                &regions[..1],
            );

            device.cmd_copy_buffer(
                transfer_queue_buffer,
                staging_buffer,
                index_buffer,
                &regions[1..],
            );

            device.cmd_pipeline_barrier(
                transfer_queue_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &memory_barriers[..2],
                &[],
            );

            device.end_command_buffer(transfer_queue_buffer)?;
        }

        unsafe {
            device.begin_command_buffer(graphics_queue_buffer, &begin_info)?;

            device.cmd_pipeline_barrier(
                graphics_queue_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::DependencyFlags::empty(),
                &[],
                &memory_barriers[2..],
                &[],
            );

            device.end_command_buffer(graphics_queue_buffer)?;
        }
        let semaphore_create_info = vk::SemaphoreCreateInfo::builder();

        let transfer_done_semaphore =
            unsafe { device.create_semaphore(&semaphore_create_info, None)? };

        let submitted_buffers = [transfer_queue_buffer];

        let semaphores = [transfer_done_semaphore];

        let submits = [vk::SubmitInfo::builder()
            .command_buffers(&submitted_buffers)
            .signal_semaphores(&semaphores)
            .build()];

        let fence_create_info = vk::FenceCreateInfo::builder();

        let transfer_done_fence = unsafe { device.create_fence(&fence_create_info, None)? };

        unsafe {
            device.queue_submit(transfer_queue, &submits, transfer_done_fence)?;
        }

        let submitted_buffers = [graphics_queue_buffer];
        let wait_stages = [vk::PipelineStageFlags::VERTEX_INPUT];

        let submits = [vk::SubmitInfo::builder()
            .command_buffers(&submitted_buffers)
            .wait_semaphores(&semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .build()];

        let graphics_done_fence = unsafe { device.create_fence(&fence_create_info, None)? };

        unsafe {
            device.queue_submit(graphics_queue, &submits, graphics_done_fence)?;
        }

        let wait_fences = [transfer_done_fence, graphics_done_fence];

        unsafe {
            device.wait_for_fences(&wait_fences, true, u64::MAX)?;

            device.free_command_buffers(graphics_command_pool, &graphics_buffers);
            device.free_command_buffers(transfer_command_pool, &transfer_buffers);
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);

            device.destroy_semaphore(transfer_done_semaphore, None);

            device.destroy_fence(transfer_done_fence, None);
            device.destroy_fence(graphics_done_fence, None);
        }

        let pool_sizes = [vk::DescriptorPoolSize {
            descriptor_count: FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        }];

        let pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(FRAMES_IN_FLIGHT as u32);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_create_info, None)? };

        let set_layouts = vec![descriptor_set_layout; FRAMES_IN_FLIGHT];

        let descriptor_set_alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let sets = unsafe { device.allocate_descriptor_sets(&descriptor_set_alloc_info)? };

        let writes: Vec<_> = sets
            .iter()
            .enumerate()
            .map(|(idx, &set)| {
                let frame = &mut frame_resources[idx];
                frame.descriptor_set = set; // side effects, yey!

                let buffer_infos = [vk::DescriptorBufferInfo {
                    buffer: frame.uniform_buffer,
                    offset: 0,
                    range: std::mem::size_of::<Ubo>() as u64,
                }];

                vk::WriteDescriptorSet::builder()
                    .dst_set(frame.descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_infos)
                    .build()
            })
            .collect();

        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let app = Self {
            image_count,
            current_frame: 0,
            sdl_ctx,
            sdl_video_ctx,
            window,
            entry,
            instance,
            device,
            device_memory_properties,
            surface,
            surface_loader,
            swapchain,
            swapchain_loader,
            swapchain_extent,
            image_resources,
            frame_resources,
            render_pass,
            pipeline_layout,
            pipeline,
            descriptor_set_layout,
            graphics_command_pool,
            transfer_command_pool,
            graphics_queue,
            transfer_queue,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            descriptor_pool,
            frame_count: 0,
        };

        println!(
            "
VKe: application created
format: {:#?},
buffer size: {:#?},
image count: {:#?},
graphics queue family index: {:#?},
tranfer queue family index: {:#?},
            ",
            swapchain_create_info.image_format,
            swapchain_extent,
            present_images.len(),
            graphics_queue_family_index,
            transfer_queue_family_index
        );

        Ok(app)
    }

    fn create_buffer(
        device: &Device,
        device_size: vk::DeviceSize,
        usage_flags: vk::BufferUsageFlags,
        device_props: vk::PhysicalDeviceMemoryProperties,
        properties: vk::MemoryPropertyFlags,
        sharing_mode: vk::SharingMode,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn Error>> {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(device_size)
            .usage(usage_flags)
            .sharing_mode(sharing_mode);

        let buffer = unsafe { device.create_buffer(&buffer_create_info, None)? };

        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(memory_requirements.size)
            .memory_type_index(
                find_memorytype_index(&memory_requirements, &device_props, properties)
                    .expect("no suitable memory types"),
            );

        let buffer_memory = unsafe { device.allocate_memory(&memory_allocate_info, None)? };

        unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0)? };

        Ok((buffer, buffer_memory))
    }

    fn create_shader_module(
        device: &Device,
        spv_code: &[u32],
    ) -> Result<vk::ShaderModule, Box<dyn Error>> {
        let create_shader_module_info = vk::ShaderModuleCreateInfo::builder().code(spv_code);

        Ok(unsafe { device.create_shader_module(&create_shader_module_info, None)? })
    }

    fn create_render_pass(device: &Device) -> Result<vk::RenderPass, Box<dyn Error>> {
        let renderpass_attachmnents = [vk::AttachmentDescription {
            format: vk::Format::B8G8R8A8_SRGB,
            samples: vk::SampleCountFlags::TYPE_1,
            store_op: vk::AttachmentStoreOp::STORE,
            load_op: vk::AttachmentLoadOp::CLEAR,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        }];

        let color_attachment_refs = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let subpasses = [vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .build()];

        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        }];

        let renderpass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&renderpass_attachmnents)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let renderpass = unsafe { device.create_render_pass(&renderpass_create_info, None)? };

        Ok(renderpass)
    }

    fn create_graphics_pipeline(
        device: &Device,
        swapchain_extent: vk::Extent2D,
        render_pass: vk::RenderPass,
    ) -> Result<(vk::PipelineLayout, vk::Pipeline, vk::DescriptorSetLayout), Box<dyn Error>> {
        let raw_vs = include_bytes!("../vert.spv");
        let raw_fs = include_bytes!("../frag.spv");

        let vs_code = ash::util::read_spv(&mut std::io::Cursor::new(&raw_vs[..]))?;
        let fs_code = ash::util::read_spv(&mut std::io::Cursor::new(&raw_fs[..]))?;

        let vs_module = Application::create_shader_module(device, &vs_code)?;
        let fs_module = Application::create_shader_module(device, &fs_code)?;

        let vs_stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vs_module)
            .name(SHADER_MAIN_FN_NAME)
            .build();

        let fs_stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fs_module)
            .name(SHADER_MAIN_FN_NAME)
            .build();

        let stages = [vs_stage_info, fs_stage_info];

        let binding_descriptors = [Vertex::get_binding_descriptor()];
        let attribute_descriptors = Vertex::get_attribute_descriptors();

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&binding_descriptors)
            .vertex_attribute_descriptions(&attribute_descriptors);

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_extent.width as f32,
            height: swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_extent,
        }];

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisampler_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .build()];

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&color_blend_attachment_states);

        let layout_bindings = [vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build()];

        let layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&layout_bindings);

        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_create_info, None)? };

        let set_layouts = [descriptor_set_layout];

        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        let pipeline_create_infos = [vk::GraphicsPipelineCreateInfo::builder()
            .stages(&stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer_state)
            .multisample_state(&multisampler_state)
            .color_blend_state(&color_blend_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0)
            .build()];

        let pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
                .unwrap()
        };

        let pipeline = pipelines[0];

        unsafe {
            device.destroy_shader_module(fs_module, None);
            device.destroy_shader_module(vs_module, None);
        }

        Ok((pipeline_layout, pipeline, descriptor_set_layout))
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

    fn pick_vk_desired_image_count(surface_capabilites: &vk::SurfaceCapabilitiesKHR) -> u32 {
        if surface_capabilites.max_image_count > 0
            && surface_capabilites.min_image_count + 1 > surface_capabilites.max_image_count
        {
            surface_capabilites.max_image_count
        } else {
            surface_capabilites.min_image_count + 1
        }
    }

    fn pick_vk_surface_extent(
        surface_capabilities: &vk::SurfaceCapabilitiesKHR,
    ) -> Result<vk::Extent2D, Box<dyn Error>> {
        Ok(match surface_capabilities.current_extent.width {
            std::u32::MAX => vk::Extent2D {
                width: 1600,
                height: 900,
            },
            _ => surface_capabilities.current_extent,
        })
    }

    fn pick_vk_device(
        instance: &Instance,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
    ) -> Result<Option<SelectedDeviceInfo>, Box<dyn Error>> {
        let is_device_suitable = |device: &vk::PhysicalDevice| -> Result<bool, Box<dyn Error>> {
            unsafe {
                let props = instance.get_physical_device_properties(*device);

                let surface_formats =
                    surface_loader.get_physical_device_surface_formats(*device, surface)?;
                let surface_present_modes =
                    surface_loader.get_physical_device_surface_present_modes(*device, surface)?;

                let suitable_swapchain_format = surface_formats.into_iter().any(|format| {
                    format.format == vk::Format::B8G8R8A8_SRGB
                        && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                });

                let suitable_swapchain_present_mode = surface_present_modes
                    .into_iter()
                    .any(|present_mode| present_mode == vk::PresentModeKHR::FIFO_RELAXED);

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

        let devices = unsafe { instance.enumerate_physical_devices()? };

        for device in devices {
            if is_device_suitable(&device)? {
                let queues =
                    unsafe { instance.get_physical_device_queue_family_properties(device) };

                // not quite sure if there's any benefit from a dedicated graphics queue...
                // so just pick whatever
                let maybe_graphics_queue_index = queues
                    .iter()
                    .enumerate()
                    .try_find::<_, Box<dyn Error>, _>(|(index, queue_properties)| {
                        Ok(queue_properties
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS)
                            && unsafe {
                                surface_loader.get_physical_device_surface_support(
                                    device,
                                    *index as u32,
                                    surface,
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
                    return Ok(result);
                }
            }
        }

        Ok(None)
    }

    fn are_validation_layers_supported(entry: &Entry) -> Result<bool, Box<dyn Error>> {
        let properties = entry.enumerate_instance_layer_properties()?;

        Ok(properties
            .iter()
            .any(|x| unsafe { CStr::from_ptr(x.layer_name.as_ptr()) } == VALIDATION_LAYER_NAME))
    }

    fn run(&mut self) -> Result<(), Box<dyn Error>> {
        let mut event_pump = self.sdl_ctx.event_pump()?;

        'running: loop {
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. } => {
                        break 'running;
                    }
                    _ => {}
                }
            }

            self.draw()?;
        }

        unsafe { self.device.device_wait_idle()? };

        Ok(())
    }

    fn update_ubos(&mut self) {
        let ubo = unsafe { &mut *self.frame_resources[self.current_frame].ubo_ptr };

        let model = Mat4::from_rotation_z(self.current_frame as f32);

        let view = Mat4::look_at_lh(
            Vec3::new(2.0, 2.0, 2.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );

        ubo.model_view = model * view;

        ubo.projection = Mat4::perspective_lh(
            f32::to_radians(45.0),
            self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32,
            0.1,
            10.0,
        );
    }

    fn draw(&mut self) -> Result<(), Box<dyn Error>> {
        let wait_fences = [self.frame_resources[self.current_frame].fence];

        unsafe {
            self.device.wait_for_fences(&wait_fences, true, u64::MAX)?;
        }

        let (image_index, _swapchain_suboptimal) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.frame_resources[self.current_frame].image_available,
                vk::Fence::null(),
            )?
        };

        if self.image_resources[image_index as usize].fence != vk::Fence::null() {
            let wait_fences = [self.image_resources[image_index as usize].fence];
            unsafe { self.device.wait_for_fences(&wait_fences, true, u64::MAX)? };
        }

        self.image_resources[image_index as usize].fence =
            self.frame_resources[self.current_frame].fence;

        let command_buffer = self.frame_resources[self.current_frame].command_buffer;

        unsafe {
            self.device.reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )?;
            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::builder())?;
        }

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.image_resources[image_index as usize].framebuffer)
            .render_area(vk::Rect2D {
                extent: self.swapchain_extent,
                offset: vk::Offset2D { x: 0, y: 0 },
            })
            .clear_values(&clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            let vertex_buffers = [self.vertex_buffer];
            let offsets = [0];

            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);

            self.device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer,
                0,
                vk::IndexType::UINT16,
            );

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.frame_resources[self.current_frame].descriptor_set],
                &[],
            );

            self.device.cmd_draw_indexed(command_buffer, 6, 1, 0, 0, 0);
            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer)?;
        }

        let wait_semaphores = [self.frame_resources[self.current_frame].image_available];
        let signal_semaphores = [self.frame_resources[self.current_frame].render_finished];
        let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [command_buffer];

        self.update_ubos();

        let submits = [vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores)
            .build()];

        unsafe {
            self.device.reset_fences(&wait_fences)?;
            self.device.queue_submit(
                self.graphics_queue,
                &submits,
                self.frame_resources[self.current_frame].fence,
            )?
        };

        let swapchains = [self.swapchain];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            self.swapchain_loader
                .queue_present(self.graphics_queue, &present_info)?;
        }

        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;

        Ok(())
    }

    const fn should_use_validation_layers() -> bool {
        cfg!(debug_assertions)
    }
}

impl Drop for Application {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);

            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);

            for frame_data in &self.frame_resources {
                self.device
                    .destroy_semaphore(frame_data.image_available, None);
                self.device
                    .destroy_semaphore(frame_data.render_finished, None);

                self.device.destroy_fence(frame_data.fence, None);

                self.device.destroy_buffer(frame_data.uniform_buffer, None);
                self.device
                    .free_memory(frame_data.uniform_buffer_memory, None);
            }

            let sets: Vec<_> = self
                .frame_resources
                .iter()
                .map(|frame_res| frame_res.descriptor_set)
                .collect();

            self.device
                .free_descriptor_sets(self.descriptor_pool, &sets);

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.device
                .destroy_command_pool(self.transfer_command_pool, None);
            self.device
                .destroy_command_pool(self.graphics_command_pool, None);

            for image_resources in &self.image_resources {
                self.device
                    .destroy_framebuffer(image_resources.framebuffer, None);
                self.device
                    .destroy_image_view(image_resources.image_view, None);
            }

            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let app = Application::new();

    let mut app = match app {
        Ok(app) => app,
        Err(e) => {
            use sdl2::messagebox;

            messagebox::show_simple_message_box(
                messagebox::MessageBoxFlag::ERROR,
                "vke",
                &format!("{:#?}", e),
                None,
            )
            .expect("unable to show an error, congrats");

            return Err(e);
        }
    };

    app.run()
}
