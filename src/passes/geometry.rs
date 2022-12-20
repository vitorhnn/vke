use crate::allocator::{Allocator, MemoryUsage};
use crate::material::Technique;
use crate::per_frame::PerFrame;
use crate::texture::Texture;
use crate::{buffer, material, Device, SHADER_MAIN_FN_NAME};
use ash::vk;
use glam::Mat4;
use gpu_allocator::vulkan::Allocation;
use sdl2::sys::wchar_t;
use std::error::Error;
use std::path::Path;
use std::rc::Rc;

// kinda lifted from DXVK's Presenter
struct FrameResources {
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
    uniform_buffer_allocation: Allocation,
    uniform_buffer: buffer::Buffer,
    descriptor_set: vk::DescriptorSet,
}

pub struct GeometryPass {
    device: Rc<Device>,
    allocator: Rc<Allocator>,
    technique: Technique,
    descriptor_pools: PerFrame<vk::DescriptorPool>,
    per_frame_uniform_buffer: buffer::Buffer,
    ubo_allocation: Allocation,
    color_targets: PerFrame<Texture>,
    depth_targets: PerFrame<Texture>,
}

#[repr(C)]
#[derive(Debug, Clone, Default)]
struct PerFrameDataUbo {
    view: Mat4,
    projection: Mat4,
}

fn create_graphics_pipeline(
    device: &Device,
    technique: &Technique,
    render_resolution: vk::Extent2D,
) -> Result<(vk::PipelineLayout, vk::Pipeline), Box<dyn Error>> {
    let vs_module = unsafe {
        let info = vk::ShaderModuleCreateInfo::builder().code(&technique.vs_spv);
        device.inner.create_shader_module(&info, None)?
    };

    let vs_stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vs_module)
        .name(SHADER_MAIN_FN_NAME)
        .build();

    let fs_module = unsafe {
        let info = vk::ShaderModuleCreateInfo::builder().code(&technique.fs_spv);
        device.inner.create_shader_module(&info, None)?
    };

    let fs_stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(fs_module)
        .name(SHADER_MAIN_FN_NAME)
        .build();

    let stages = [vs_stage_info, fs_stage_info];

    let binding_descriptor = technique
        .vertex_layout_info
        .input_binding_description
        .as_vk();
    let attribute_descriptors: Vec<_> = technique
        .vertex_layout_info
        .input_attribute_descriptions
        .iter()
        .map(|d| d.as_vk())
        .collect();

    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(std::slice::from_ref(&binding_descriptor))
        .vertex_attribute_descriptions(&attribute_descriptors);

    let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: render_resolution.width as f32,
        height: render_resolution.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];

    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: render_resolution,
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

    /*let descriptor_set_layouts = technique
        .descriptor_set_layouts
        .iter()
        .filter_map(|x| x.map(|layout| layout.into_vk()))
        .collect::<Vec<_>>();
     */

    todo!();
}

impl GeometryPass {
    fn new(device: Rc<Device>, allocator: Rc<Allocator>, render_resolution: vk::Extent2D) -> Self {
        let descriptor_pools = PerFrame::new(|| {
            let pool_sizes = [vk::DescriptorPoolSize {
                descriptor_count: 32,
                ty: vk::DescriptorType::UNIFORM_BUFFER,
            }];

            let pool_create_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&pool_sizes)
                .max_sets(32);

            unsafe {
                device
                    .inner
                    .create_descriptor_pool(&pool_create_info, None)
                    .expect("pool allocation failure")
            }
        });

        let (per_frame_uniform_buffer, ubo_allocation) = {
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size(std::mem::size_of::<PerFrameDataUbo>() as u64)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            allocator
                .create_buffer(&buffer_create_info, MemoryUsage::HostToDevice)
                .expect("ubo allocation failure")
        };

        let color_targets = PerFrame::new(|| {
            let extent = vk::Extent3D {
                depth: 1,
                width: render_resolution.width,
                height: render_resolution.height,
            };

            let image_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_SRGB)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC);

            let (mut texture, allocation) = allocator
                .create_texture(&image_info, MemoryUsage::DeviceOnly)
                .expect("failed to allocate color targets");

            texture.associate_allocation(allocation);

            texture
        });

        let depth_targets = PerFrame::new(|| {
            let extent = vk::Extent3D {
                depth: 1,
                width: render_resolution.width,
                height: render_resolution.height,
            };

            let image_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT);

            let (mut texture, allocation) = allocator
                .create_texture(&image_info, MemoryUsage::DeviceOnly)
                .expect("failed to allocate color targets");

            texture.associate_allocation(allocation);

            texture
        });

        let technique = material::compile_shader(Path::new("../glsl/geometry"));
        Self {
            device,
            allocator,
            technique,
            descriptor_pools,
            per_frame_uniform_buffer,
            ubo_allocation,
            color_targets,
            depth_targets,
        }
    }
}
