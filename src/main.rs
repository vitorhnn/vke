// rust devs pls stabilize
#![feature(try_find)]

use ash::{prelude::VkResult, vk};

use sdl2::event::Event;

use std::error::Error;
use std::ffi::CStr;
use std::fmt::Debug;
use std::rc::Rc;

use memoffset::offset_of;

use glam::{Mat4, Vec3};

mod allocator;
mod buffer;
mod surface;
mod swapchain;

mod instance;
use instance::Instance;

mod device;
use device::Device;

mod queue;

mod transfer;
use transfer::Transfer;

// kinda lifted from DXVK's Presenter
struct FrameResources {
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
    uniform_buffer_allocation: allocator::Allocation,
    uniform_buffer: buffer::Buffer,
    descriptor_set: vk::DescriptorSet,
}

struct Application {
    desired_extent: vk::Extent2D,
    desired_present_mode: vk::PresentModeKHR,
    sdl_ctx: sdl2::Sdl,
    sdl_video_ctx: sdl2::VideoSubsystem,
    window: sdl2::video::Window,
    swapchain: swapchain::Swapchain,
    allocator: Rc<allocator::Allocator>,
    device: Device,
    surface: surface::Surface,
    transfer: Transfer,
    frame_resources: Vec<FrameResources>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    render_pass: vk::RenderPass,
    current_frame: usize,
    vertex_buffer: buffer::Buffer,
    index_buffer: buffer::Buffer,
    descriptor_pool: vk::DescriptorPool,
    frame_count: usize,
    instance: Instance,
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

        let instance = Instance::new(&window)?;

        let surface = surface::Surface::new(
            &instance,
            window.vulkan_create_surface(instance.raw_handle() as usize)?,
        )?;

        let device = Device::from_heuristics(&instance, &surface, desired_present_mode)?;

        let allocator = Rc::new(allocator::Allocator::new(&device, &instance));

        let mut transfer = Transfer::new(
            device.inner.clone(),
            allocator.clone(),
            device.graphics_queue.clone(),
            Some(device.transfer_queue.clone()),
        )?;

        let render_pass = Application::create_render_pass(&device)?;

        let swapchain = swapchain::Swapchain::new(
            &instance,
            &device,
            &surface,
            desired_extent,
            desired_present_mode,
            render_pass,
        )?;

        let mut frame_resources: Vec<FrameResources> = (0..FRAMES_IN_FLIGHT)
            .map::<VkResult<FrameResources>, _>(|_| {
                let semaphore_create_info = vk::SemaphoreCreateInfo::builder();

                let fence_create_info =
                    vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

                let image_available = unsafe {
                    device
                        .inner
                        .create_semaphore(&semaphore_create_info, None)?
                };
                let render_finished = unsafe {
                    device
                        .inner
                        .create_semaphore(&semaphore_create_info, None)?
                };
                let fence = unsafe { device.inner.create_fence(&fence_create_info, None)? };

                let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::builder()
                    .command_pool(device.graphics_queue.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);

                let command_buffers = unsafe {
                    device
                        .inner
                        .allocate_command_buffers(&command_buffer_alloc_info)?
                };

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

        transfer.upload_buffer(&vertices, &vertex_buffer)?;
        transfer.upload_buffer(&indices, &index_buffer)?;

        transfer.flush()?;

        let pool_sizes = [vk::DescriptorPoolSize {
            descriptor_count: FRAMES_IN_FLIGHT as u32,
            ty: vk::DescriptorType::UNIFORM_BUFFER,
        }];

        let pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(FRAMES_IN_FLIGHT as u32);

        let descriptor_pool = unsafe {
            device
                .inner
                .create_descriptor_pool(&pool_create_info, None)?
        };

        let descriptor_set_layout = Application::create_descriptor_set_layout(&device)?;

        let (pipeline_layout, pipeline) = Application::create_graphics_pipeline(
            &device,
            swapchain.extent,
            render_pass,
            &[descriptor_set_layout],
        )?;

        let set_layouts = vec![descriptor_set_layout; FRAMES_IN_FLIGHT];

        let descriptor_set_alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let sets = unsafe {
            device
                .inner
                .allocate_descriptor_sets(&descriptor_set_alloc_info)?
        };

        for (i, &set) in sets.iter().enumerate() {
            let frame = &mut frame_resources[i];
            frame.descriptor_set = set; // side effects, yey!

            let buffer_infos = [vk::DescriptorBufferInfo {
                buffer: frame.uniform_buffer.inner,
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
                device.inner.update_descriptor_sets(&[update], &[]);
            }
        }

        println!("VKe: application created");
        println!(
            "graphics queue family index: {:#?}",
            device.graphics_queue.family_index,
        );
        println!(
            "tranfer queue family index: {:#?}",
            device.transfer_queue.family_index,
        );

        let app = Self {
            desired_present_mode,
            desired_extent,
            current_frame: 0,
            sdl_ctx,
            instance,
            sdl_video_ctx,
            window,
            device,
            surface,
            swapchain,
            frame_resources,
            render_pass,
            pipeline_layout,
            pipeline,
            descriptor_set_layout,
            vertex_buffer,
            index_buffer,
            descriptor_pool,
            frame_count: 0,
            allocator,
            transfer,
        };

        Ok(app)
    }

    fn create_shader_module(
        device: &Device,
        spv_code: &[u32],
    ) -> Result<vk::ShaderModule, Box<dyn Error>> {
        let create_shader_module_info = vk::ShaderModuleCreateInfo::builder().code(spv_code);

        Ok(unsafe {
            device
                .inner
                .create_shader_module(&create_shader_module_info, None)?
        })
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

        let renderpass = unsafe {
            device
                .inner
                .create_render_pass(&renderpass_create_info, None)?
        };

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

        Ok(unsafe {
            device
                .inner
                .create_descriptor_set_layout(&layout_create_info, None)?
        })
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
            .color_write_mask(vk::ColorComponentFlags::RGBA)
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

        let pipeline_layout = unsafe {
            device
                .inner
                .create_pipeline_layout(&pipeline_layout_info, None)?
        };

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
                .inner
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
                .unwrap()
        };

        let pipeline = pipelines[0];

        unsafe {
            device.inner.destroy_shader_module(fs_module, None);
            device.inner.destroy_shader_module(vs_module, None);
        }

        Ok((pipeline_layout, pipeline))
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
            self.device
                .inner
                .wait_for_fences(&wait_fences, true, u64::MAX)?;
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
            unsafe {
                self.device
                    .inner
                    .wait_for_fences(&wait_fences, true, u64::MAX)?
            };
        }

        image_resources.fence = self.frame_resources[self.current_frame].fence;

        let command_buffer = self.frame_resources[self.current_frame].command_buffer;

        unsafe {
            self.device.inner.reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )?;
            self.device
                .inner
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
            self.device.inner.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
            self.device.inner.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            let vertex_buffers = [self.vertex_buffer.inner];
            let offsets = [0];

            self.device
                .inner
                .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);

            self.device.inner.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer.inner,
                0,
                vk::IndexType::UINT16,
            );

            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.frame_resources[self.current_frame].descriptor_set],
                &[],
            );

            self.device
                .inner
                .cmd_draw_indexed(command_buffer, 6, 1, 0, 0, 0);
            self.device.inner.cmd_end_render_pass(command_buffer);
            self.device.inner.end_command_buffer(command_buffer)?;
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
            self.device.inner.reset_fences(&wait_fences)?;
            self.device.inner.queue_submit(
                self.device.graphics_queue.inner,
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
                .queue_present(self.device.graphics_queue.inner, &present_info)
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

    fn recreate_swapchain(&mut self) -> Result<(), Box<dyn Error>> {
        unsafe { self.device.inner.device_wait_idle()? };

        self.drop_swapchain();

        self.render_pass = Application::create_render_pass(&self.device)?;
        self.swapchain = swapchain::Swapchain::new(
            &self.instance,
            &self.device,
            &self.surface,
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
                    .inner
                    .destroy_framebuffer(image_resources.framebuffer, None);
                self.device
                    .inner
                    .destroy_image_view(image_resources.image_view, None);
            }

            self.swapchain
                .loader
                .destroy_swapchain(self.swapchain.swapchain, None);

            self.device.inner.destroy_pipeline(self.pipeline, None);
            self.device
                .inner
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .inner
                .destroy_render_pass(self.render_pass, None);
        }
    }
}

impl Drop for Application {
    fn drop(&mut self) {
        unsafe {
            self.device
                .inner
                .device_wait_idle()
                .expect("couldn't wait for device idle");

            self.drop_swapchain();

            for frame_data in &self.frame_resources {
                self.device
                    .inner
                    .destroy_semaphore(frame_data.image_available, None);
                self.device
                    .inner
                    .destroy_semaphore(frame_data.render_finished, None);

                self.device.inner.destroy_fence(frame_data.fence, None);
            }

            let sets: Vec<_> = self
                .frame_resources
                .iter()
                .map(|frame_res| frame_res.descriptor_set)
                .collect();

            self.device
                .inner
                .free_descriptor_sets(self.descriptor_pool, &sets);

            self.device
                .inner
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.surface
                .loader
                .destroy_surface(self.surface.surface, None);

            self.device
                .inner
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
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
                app.device.inner.device_wait_idle().unwrap();
            }
            Err(e)
        }
    }
}
