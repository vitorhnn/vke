use std::rc::Rc;
use std::error::Error;

use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::{device::Device, allocator::Allocator, per_frame::PerFrame, buffer, technique::{CookedGraphicsTechnique, DescriptorSetLayout, PushConstantRange}, SHADER_MAIN_FN_NAME};


pub struct MarchingCubesMeshPass {
    device: Rc<Device>,
    allocator: Rc<Allocator>,
    descriptor_pools: PerFrame<vk::DescriptorPool>,
    params_uniform_buffer: buffer::Buffer,
    ubo_allocation: Allocation,
    triangles_input_ssbo: buffer::Buffer,
    triangles_input_allocation: Allocation,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    points: [glam::Vec4; 10],
}

#[derive(Debug)]
#[repr(C)]
struct ParamsUbo {
    surface_level: f32,
    smooth: f32,
    linear: u32,
    points_count: u32,
    world_bounds: glam::Vec3,
    points: [glam::Vec4; 10],
}

fn create_graphics_pipeline(
    device: &Device,
    technique: &CookedGraphicsTechnique,
    descriptor_set_layouts: &[Option<DescriptorSetLayout>; 4],
    push_constant_range: &Option<PushConstantRange>,
    render_resolution: vk::Extent2D,
) -> Result<
    (
        vk::PipelineLayout,
        vk::Pipeline,
        Vec<vk::DescriptorSetLayout>,
    ),
    Box<dyn Error>,
> {
    let ms = technique.ms_spv.as_ref().expect("hardcoded ms expect");
    let ms_module = unsafe {
        let info = vk::ShaderModuleCreateInfo::builder().code(&ms);
        device.inner.create_shader_module(&info, None)?
    };

    let ms_stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::MESH_EXT)
        .module(ms_module)
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

    let stages = [ms_stage_info, fs_stage_info];

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
        .cull_mode(vk::CullModeFlags::NONE)
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

    let descriptor_set_layouts = descriptor_set_layouts
        .iter()
        .filter_map(|x| x.as_ref().map(|layout| layout.as_vk(&device)))
        .collect::<Vec<_>>();

    let pipeline_layout = unsafe {
        let mut info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);

        let push_constant_range = push_constant_range.as_ref().map(|x| x.as_vk());

        if let Some(push_constant_range) = &push_constant_range {
            info = info.push_constant_ranges(std::slice::from_ref(push_constant_range));
        }

        device
            .inner
            .create_pipeline_layout(&info, None)
            .expect("failed to create pipeline layout")
    };

    let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfoKHR::builder()
        .color_attachment_formats(&[vk::Format::B8G8R8A8_SRGB])
        .depth_attachment_format(vk::Format::D32_SFLOAT)
        .stencil_attachment_format(vk::Format::UNDEFINED);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0);

    let pipeline_create_infos = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&stages)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer_state)
        .multisample_state(&multisampler_state)
        .color_blend_state(&color_blend_state)
        .layout(pipeline_layout)
        .depth_stencil_state(&depth_stencil_state)
        .render_pass(vk::RenderPass::null())
        .push_next(&mut pipeline_rendering_info);

    let pipelines = unsafe {
        device
            .inner
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_create_infos),
                None,
            )
            .unwrap()
    };

    let pipeline = pipelines[0];

    unsafe {
        device.inner.destroy_shader_module(fs_module, None);
        device.inner.destroy_shader_module(ms_module, None);
    }

    // this is kind of a mess. TODO refactor
    Ok((pipeline_layout, pipeline, descriptor_set_layouts))
}

impl MarchingCubesMeshPass {
    pub fn new(device: Rc<Device>, allocator: Rc<Allocator>) -> Self {
        todo!()
    }
}