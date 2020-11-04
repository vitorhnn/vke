// rust devs pls stabilize
#![feature(const_cstr_unchecked)]
#![feature(try_find)]
#![feature(const_int_pow)]
#![feature(const_in_array_repeat_expressions)]

use ash::{
    extensions::khr::Swapchain, prelude::VkResult, version::DeviceV1_0, version::EntryV1_0,
    version::InstanceV1_0, vk, vk::Handle, Device, Entry, Instance,
};

use sdl2::event::Event;

use std::error::Error;
use std::ffi::CStr;
use std::fmt::Debug;
use std::rc::Rc;

use memoffset::offset_of;

use glam::{Mat4, Vec3};

mod allocator;
mod surface;
mod swapchain;

// kinda lifted from DXVK's Presenter
struct FrameResources {
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
    uniform_buffer_allocation: allocator::Allocation,
    uniform_buffer: vk::Buffer,
    descriptor_set: vk::DescriptorSet,
}

struct Application {
    desired_extent: vk::Extent2D,
    desired_present_mode: vk::PresentModeKHR,
    sdl_ctx: sdl2::Sdl,
    sdl_video_ctx: sdl2::VideoSubsystem,
    window: sdl2::video::Window,
    entry: Entry,
    instance: Instance,
    device: Rc<Device>,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    swapchain: swapchain::Swapchain,
    surface: surface::Surface,
    frame_resources: Vec<FrameResources>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    render_pass: vk::RenderPass,
    graphics_command_pool: vk::CommandPool,
    transfer_command_pool: vk::CommandPool,
    graphics_queue: vk::Queue,
    transfer_queue: vk::Queue,
    current_frame: usize,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    descriptor_pool: vk::DescriptorPool,
    frame_count: usize,
    physical_device: vk::PhysicalDevice,
    allocator: Rc<allocator::Allocator>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Default)]
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
    fn new(
        desired_extent: vk::Extent2D,
        desired_present_mode: vk::PresentModeKHR,
    ) -> Result<Self, Box<dyn Error>> {
        let sdl_ctx = sdl2::init()?;
        let sdl_video_ctx = sdl_ctx.video()?;
        let window = sdl_video_ctx
            .window("vke", desired_extent.width, desired_extent.height)
            .resizable()
            .vulkan()
            .build()?;

        let entry = Entry::new()?;

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&APPLICATION_NAME)
            .application_version(vk::make_version(0, 1, 0))
            .engine_name(&APPLICATION_NAME)
            .engine_version(vk::make_version(0, 1, 0))
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

        let surface = surface::Surface::new(
            &entry,
            &instance,
            window.vulkan_create_surface(instance.handle().as_raw() as usize)?,
        )?;

        let selected_device_info =
            Application::pick_vk_device(&instance, &surface, desired_present_mode)?;

        let selected_device_info = selected_device_info.expect("no suitable vulkan device");

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

        let queues = if graphics_queue_family_index != transfer_queue_family_index {
            vec![
                create_graphics_queue_info_builder.build(),
                create_transfer_queue_info_builder.build(),
            ]
        } else {
            vec![create_graphics_queue_info_builder.build()]
        };

        let device_features_builder = vk::PhysicalDeviceFeatures::builder();

        let device_extensions = [Swapchain::name().as_ptr()];

        let create_device_info_builder = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queues)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&device_features_builder);

        let device = Rc::new(unsafe {
            instance.create_device(physical_device, &create_device_info_builder, None)?
        });

        let allocator = Rc::new(allocator::Allocator::new(
            physical_device,
            device.clone(),
            &instance,
        ));

        let render_pass = Application::create_render_pass(&device)?;

        let swapchain = swapchain::Swapchain::new(
            &instance,
            &device,
            &surface,
            physical_device,
            desired_extent,
            desired_present_mode,
            render_pass,
        )?;

        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family_index, 0) };
        let transfer_queue = unsafe { device.get_device_queue(transfer_queue_family_index, 0) };

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

                let ubo_info = vk::BufferCreateInfo::builder()
                    .size(std::mem::size_of::<Ubo>() as u64)
                    .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);

                let (uniform_buffer, uniform_buffer_allocation) =
                    allocator.create_buffer(&ubo_info, allocator::MemoryUsage::HostToDevice)?;

                Ok(FrameResources {
                    image_available,
                    render_finished,
                    fence,
                    command_buffer,
                    uniform_buffer,
                    uniform_buffer_allocation,
                    descriptor_set: vk::DescriptorSet::null(),
                })
            })
            .collect::<Result<_, _>>()?;

        let descriptor_set_layout = Application::create_descriptor_set_layout(&device)?;

        let (pipeline_layout, pipeline) = Application::create_graphics_pipeline(
            &device,
            swapchain.extent,
            render_pass,
            &[descriptor_set_layout],
        )?;

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

        let staging_buffer_info = vk::BufferCreateInfo::builder()
            .size(vertex_buffer_size + index_buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let (staging_buffer, staging_buffer_allocation) =
            allocator.create_buffer(&staging_buffer_info, allocator::MemoryUsage::HostOnly)?;

        let vertex_buffer_info = vk::BufferCreateInfo::builder()
            .size(vertex_buffer_size)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let (vertex_buffer, vertex_buffer_allocation) =
            allocator.create_buffer(&vertex_buffer_info, allocator::MemoryUsage::DeviceOnly)?;

        let index_buffer_info = vk::BufferCreateInfo::builder()
            .size(index_buffer_size)
            .usage(vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let (index_buffer, index_buffer_allocation) =
            allocator.create_buffer(&index_buffer_info, allocator::MemoryUsage::DeviceOnly)?;

        unsafe {
            let mapped = allocator.map(&staging_buffer_allocation)?;

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

            allocator.unmap(&staging_buffer_allocation);
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

            device.destroy_semaphore(transfer_done_semaphore, None);

            device.destroy_fence(transfer_done_fence, None);
            device.destroy_fence(graphics_done_fence, None);
        }

        let pool_sizes = [vk::DescriptorPoolSize {
            descriptor_count: FRAMES_IN_FLIGHT as u32,
            ty: vk::DescriptorType::UNIFORM_BUFFER,
        }];

        let pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(FRAMES_IN_FLIGHT as u32);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_create_info, None)? };

        let set_layouts = vec![descriptor_set_layout; FRAMES_IN_FLIGHT];

        let descriptor_set_alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let sets = unsafe { device.allocate_descriptor_sets(&descriptor_set_alloc_info)? };

        for (i, &set) in sets.iter().enumerate() {
            let frame = &mut frame_resources[i];
            frame.descriptor_set = set; // side effects, yey!

            let buffer_infos = [vk::DescriptorBufferInfo {
                buffer: frame.uniform_buffer,
                offset: 0,
                range: std::mem::size_of::<Ubo>() as u64,
            }];

            let update = vk::WriteDescriptorSet::builder()
                .dst_set(frame.descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_infos)
                .build();

            unsafe {
                device.update_descriptor_sets(&[update], &[]);
            }
        }

        let app = Self {
            desired_extent,
            desired_present_mode,
            current_frame: 0,
            sdl_ctx,
            sdl_video_ctx,
            window,
            entry,
            instance,
            device,
            device_memory_properties,
            surface,
            swapchain,
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
            index_buffer,
            descriptor_pool,
            physical_device,
            frame_count: 0,
            allocator,
        };

        println!("VKe: application created");
        println!(
            "graphics queue family index: {:#?}",
            graphics_queue_family_index
        );
        println!(
            "tranfer queue family index: {:#?}",
            transfer_queue_family_index
        );

        Ok(app)
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

    fn create_descriptor_set_layout(device: &Device) -> VkResult<vk::DescriptorSetLayout> {
        let layout_bindings = [vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build()];

        let layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&layout_bindings);

        Ok(unsafe { device.create_descriptor_set_layout(&layout_create_info, None)? })
    }

    fn create_graphics_pipeline(
        device: &Device,
        swapchain_extent: vk::Extent2D,
        render_pass: vk::RenderPass,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
    ) -> Result<(vk::PipelineLayout, vk::Pipeline), Box<dyn Error>> {
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
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
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

        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);

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

        Ok((pipeline_layout, pipeline))
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

    fn pick_vk_device(
        instance: &Instance,
        surface: &surface::Surface,
        desired_present_mode: vk::PresentModeKHR,
    ) -> Result<Option<SelectedDeviceInfo>, Box<dyn Error>> {
        let is_device_suitable = |device: &vk::PhysicalDevice| -> Result<bool, Box<dyn Error>> {
            unsafe {
                let props = instance.get_physical_device_properties(*device);
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
                    Event::Window { win_event, .. } => {
                        use sdl2::event::WindowEvent;
                        use sdl2::video::FullscreenType;

                        if let WindowEvent::Maximized = win_event {
                            self.window.set_fullscreen(FullscreenType::True)?;
                        }
                    }
                    _ => {}
                }
            }

            self.draw()?;
        }

        Ok(())
    }

    fn update_ubos(&mut self) -> VkResult<()> {
        let mapped = self
            .allocator
            .map(&self.frame_resources[self.current_frame].uniform_buffer_allocation)?
            as *mut Ubo;

        let mut ubo = unsafe { &mut *mapped };

        let model = Mat4::from_rotation_z((self.frame_count as f32).to_radians());

        let view = Mat4::look_at_lh(
            Vec3::new(2.0, 2.0, 2.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );

        ubo.model_view = model * view;

        ubo.projection = Mat4::perspective_lh(
            f32::to_radians(45.0),
            self.swapchain.extent.width as f32 / self.swapchain.extent.height as f32,
            0.1,
            10.0,
        );

        Ok(())
    }

    fn draw(&mut self) -> VkResult<()> {
        let frame_resources = &self.frame_resources[self.current_frame];
        let wait_fences = [frame_resources.fence];

        unsafe {
            self.device.wait_for_fences(&wait_fences, true, u64::MAX)?;
        }

        let acquire_result = unsafe {
            self.swapchain.loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                self.frame_resources[self.current_frame].image_available,
                vk::Fence::null(),
            )
        };

        let (image_index, out_of_date) = match acquire_result {
            Ok((_, true)) => (0, false),
            Ok((image_index, false)) => (image_index, false),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => (0, true),
            Err(e) => return Err(e),
        };

        if out_of_date {
            self.recreate_swapchain().unwrap();
            return Ok(());
        }

        let image_resources = &mut self.swapchain.image_resources[image_index as usize];

        if image_resources.fence != vk::Fence::null() {
            let wait_fences = [image_resources.fence];
            unsafe { self.device.wait_for_fences(&wait_fences, true, u64::MAX)? };
        }

        image_resources.fence = self.frame_resources[self.current_frame].fence;

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
            .framebuffer(image_resources.framebuffer)
            .render_area(vk::Rect2D {
                extent: self.swapchain.extent,
                offset: vk::Offset2D { x: 0, y: 0 },
            })
            .clear_values(&clear_values);

        self.update_ubos()?;

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

        let swapchains = [self.swapchain.swapchain];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let present_result = unsafe {
            self.swapchain
                .loader
                .queue_present(self.graphics_queue, &present_info)
        };

        let should_recreate = match present_result {
            Ok(true) => true,
            Ok(false) => false,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Err(e) => return Err(e),
        };

        if should_recreate {
            self.recreate_swapchain().unwrap();
        }

        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;
        self.frame_count += 1;

        Ok(())
    }

    const fn should_use_validation_layers() -> bool {
        cfg!(debug_assertions)
    }

    fn recreate_swapchain(&mut self) -> Result<(), Box<dyn Error>> {
        unsafe { self.device.device_wait_idle()? };

        self.drop_swapchain();

        self.render_pass = Application::create_render_pass(&self.device)?;
        self.swapchain = swapchain::Swapchain::new(
            &self.instance,
            &self.device,
            &self.surface,
            self.physical_device,
            self.desired_extent,
            self.desired_present_mode,
            self.render_pass,
        )?;

        let pipeline_double = Application::create_graphics_pipeline(
            &self.device,
            self.swapchain.extent,
            self.render_pass,
            &[self.descriptor_set_layout],
        )?;

        self.pipeline_layout = pipeline_double.0;
        self.pipeline = pipeline_double.1;

        Ok(())
    }

    fn drop_swapchain(&mut self) {
        unsafe {
            for image_resources in &self.swapchain.image_resources {
                self.device
                    .destroy_framebuffer(image_resources.framebuffer, None);
                self.device
                    .destroy_image_view(image_resources.image_view, None);
            }

            self.swapchain
                .loader
                .destroy_swapchain(self.swapchain.swapchain, None);

            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
        }
    }
}

impl Drop for Application {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("couldn't wait for device idle");
            self.device.destroy_buffer(self.vertex_buffer, None);

            self.device.destroy_buffer(self.index_buffer, None);

            self.allocator.hacky_manual_drop();

            self.drop_swapchain();

            for frame_data in &self.frame_resources {
                self.device
                    .destroy_semaphore(frame_data.image_available, None);
                self.device
                    .destroy_semaphore(frame_data.render_finished, None);

                self.device.destroy_fence(frame_data.fence, None);

                self.device.destroy_buffer(frame_data.uniform_buffer, None);
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

            self.surface
                .loader
                .destroy_surface(self.surface.surface, None);

            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn result_msgbox<T, E: Debug>(result: Result<T, E>) -> Result<T, E> {
    match result {
        Ok(thing) => Ok(thing),
        Err(e) => {
            use sdl2::messagebox;

            messagebox::show_simple_message_box(
                messagebox::MessageBoxFlag::ERROR,
                "vke",
                &format!("{:#?}", e),
                None,
            )
            .expect("unable to show an error, congrats");

            Err(e)
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut app = result_msgbox(Application::new(
        vk::Extent2D {
            width: 1280,
            height: 720,
        },
        vk::PresentModeKHR::FIFO,
    ))?;

    match result_msgbox(app.run()) {
        Ok(_) => Ok(()),
        Err(e) => {
            unsafe {
                app.device.device_wait_idle().unwrap();
            }
            Err(e)
        }
    }
}
