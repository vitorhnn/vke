use crate::allocator::{Allocator, MemoryUsage};
use crate::buffer::Buffer;
use crate::device::ImageBarrierParameters;
use crate::loader::World;
use crate::per_frame::PerFrame;
use crate::technique::{
    CookedGraphicsTechnique, DescriptorSetLayout, PushConstantRange, Technique, TechniqueType,
};
use crate::texture_view::TextureView;
use crate::PerFrameDataUbo;
use crate::{buffer, technique, Device, FRAMES_IN_FLIGHT, SHADER_MAIN_FN_NAME};
use ash::prelude::VkResult;
use ash::vk;
use gpu_allocator::vulkan::Allocation;
use std::error::Error;
use std::path::Path;
use std::rc::Rc;

pub struct GeometryPass {
    device: Rc<Device>,
    allocator: Rc<Allocator>,
    technique: Technique,
    descriptor_pools: PerFrame<vk::DescriptorPool>,
    per_frame_uniform_buffer: buffer::Buffer,
    ubo_allocation: Allocation,
    pub color_target_views: PerFrame<TextureView>,
    depth_target_views: PerFrame<TextureView>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    render_resolution: vk::Extent2D,
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
    let vs = &technique.vs_meta.as_ref().expect("hardcoded vs expect");
    let vs_module = unsafe {
        let info = vk::ShaderModuleCreateInfo::builder().code(&vs.spv);
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

    let binding_descriptor = vs.vertex_layout_info.input_binding_description.as_vk();
    let attribute_descriptors: Vec<_> = vs
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
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly_info)
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
        device.inner.destroy_shader_module(vs_module, None);
    }

    // this is kind of a mess. TODO refactor
    Ok((pipeline_layout, pipeline, descriptor_set_layouts))
}

impl GeometryPass {
    pub fn new(
        device: Rc<Device>,
        allocator: Rc<Allocator>,
        render_resolution: vk::Extent2D,
    ) -> Self {
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
                .size((std::mem::size_of::<PerFrameDataUbo>() * FRAMES_IN_FLIGHT) as u64)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            allocator
                .create_buffer(&buffer_create_info, MemoryUsage::HostToDevice)
                .expect("ubo allocation failure")
        };

        let color_target_views = PerFrame::new(|| {
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
                .image(texture.image);

            TextureView::new(device.clone(), texture, &create_view_info)
                .expect("failed to create view for color targets")
        });

        let depth_target_views = PerFrame::new(|| {
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

            let create_view_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(texture.image);

            TextureView::new(device.clone(), texture, &create_view_info)
                .expect("failed to create view for depth targets")
        });

        let technique = technique::compile_shader(Path::new("./glsl/geometry_marching"));

        if let TechniqueType::Graphics(graphics_technique) = &technique.r#type {
            let (pipeline_layout, pipeline, descriptor_set_layouts) = create_graphics_pipeline(
                &device,
                &graphics_technique,
                &technique.descriptor_set_layouts,
                &technique.push_constant_range,
                render_resolution,
            )
            .expect("failed to create graphics pipeline");
            Self {
                device,
                allocator,
                technique,
                descriptor_pools,
                per_frame_uniform_buffer,
                ubo_allocation,
                color_target_views,
                depth_target_views,
                render_resolution,
                pipeline,
                pipeline_layout,
                descriptor_set_layouts,
            }
        } else {
            panic!("not a graphics technique");
        }
    }

    pub fn prepare_frame_wide_descriptor_set(
        &self,
        frame_idx: usize,
        ubo_data: PerFrameDataUbo,
    ) -> VkResult<vk::DescriptorSet> {
        let mapped = self
            .allocator
            .map(&self.ubo_allocation)
            .expect("failed to map ubo memory") as *mut PerFrameDataUbo;
        let slice = unsafe { std::slice::from_raw_parts_mut(mapped, FRAMES_IN_FLIGHT) };
        let ubo = &mut slice[frame_idx];

        ubo.view = ubo_data.view;
        ubo.projection = ubo_data.projection;

        self.allocator.unmap(&self.ubo_allocation);

        let descriptor_pool = self.descriptor_pools.get_resource_for_frame(frame_idx);

        let set = unsafe {
            let set_layout = [self.descriptor_set_layouts[0]];
            let allocate_info = vk::DescriptorSetAllocateInfo {
                descriptor_pool: *descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: set_layout.as_ptr(),
                ..Default::default()
            };
            self.device.inner.allocate_descriptor_sets(&allocate_info)?[0]
        };

        unsafe {
            let offset_multiplier = frame_idx;

            let buffer_info = vk::DescriptorBufferInfo {
                buffer: self.per_frame_uniform_buffer.inner,
                offset: (std::mem::size_of::<PerFrameDataUbo>() * offset_multiplier) as u64,
                range: std::mem::size_of::<PerFrameDataUbo>() as u64,
            };

            let write = vk::WriteDescriptorSet {
                dst_set: set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_buffer_info: &buffer_info as *const _,
                ..Default::default()
            };

            self.device
                .inner
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }

        Ok(set)
    }

    pub fn execute(
        &mut self,
        frame_idx: usize,
        command_buffer: &vk::CommandBuffer,
        world: &World,
        ubo: PerFrameDataUbo,
        output_ssbo: &Buffer,
        indirect_cmd_ssbo: &Buffer,
    ) -> VkResult<()> {
        let color_target = self.color_target_views.get_resource_for_frame(frame_idx);
        let depth_target = self.depth_target_views.get_resource_for_frame(frame_idx);

        unsafe {
            let pool = self.descriptor_pools.get_resource_for_frame(frame_idx);

            self.device
                .inner
                .reset_descriptor_pool(*pool, vk::DescriptorPoolResetFlags::empty())?;
        }

        unsafe {
            // transition color images to color attachment write
            self.device.insert_image_barrier(&ImageBarrierParameters {
                command_buffer: *command_buffer,
                src_stage_mask: vk::PipelineStageFlags::TOP_OF_PIPE,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                src_access_mask: vk::AccessFlags::MEMORY_READ,
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                image: color_target.texture.image,
                subresource_range: vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            });

            self.device.insert_image_barrier(&ImageBarrierParameters {
                command_buffer: *command_buffer,
                src_stage_mask: vk::PipelineStageFlags::TOP_OF_PIPE,
                dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                image: depth_target.texture.image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            });

            let attachment = vk::RenderingAttachmentInfoKHR {
                clear_value: vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 1.0, 0.0, 1.0],
                    },
                },
                image_view: color_target.inner,
                image_layout: vk::ImageLayout::ATTACHMENT_OPTIMAL_KHR,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                ..Default::default()
            };

            let depth_attachment = vk::RenderingAttachmentInfoKHR {
                clear_value: vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
                image_view: depth_target.inner,
                image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                ..Default::default()
            };

            let rendering_info = vk::RenderingInfoKHR {
                render_area: vk::Rect2D {
                    extent: self.render_resolution,
                    ..Default::default()
                },
                layer_count: 1,
                view_mask: 0,
                color_attachment_count: 1,
                p_color_attachments: &attachment as *const _,
                p_depth_attachment: &depth_attachment as *const _,
                ..Default::default()
            };

            self.device
                .dynamic_rendering
                .cmd_begin_rendering(*command_buffer, &rendering_info);

            self.device.inner.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            let frame_descriptor_set = self.prepare_frame_wide_descriptor_set(frame_idx, ubo)?;

            self.device.inner.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&frame_descriptor_set),
                &[],
            );

            /*
            for model in &world.models {
                // TODO: this is *really, really* bad for CPU performance. we should batch static geometry in a single vertex buffer.
                for mesh in &model.meshes {
                    let offsets = [0];

                    self.device.inner.cmd_bind_vertex_buffers(
                        *command_buffer,
                        0,
                        std::slice::from_ref(&mesh.buffer.inner),
                        &offsets,
                    );

                    self.device.inner.cmd_bind_index_buffer(
                        *command_buffer,
                        mesh.idx_buffer.inner,
                        0,
                        vk::IndexType::UINT16,
                    );

                    // TODO won't be identity in the future
                    let model = model.transform;

                    let ptr = bytemuck::cast_slice(model.as_ref());

                    self.device.inner.cmd_push_constants(
                        *command_buffer,
                        self.pipeline_layout,
                        self.technique
                            .push_constant_range
                            .as_ref()
                            .unwrap()
                            .stage_flags
                            .as_vk(),
                        0,
                        ptr,
                    );

                    self.device
                        .inner
                        .cmd_draw_indexed(*command_buffer, mesh.idx_count, 1, 0, 0, 0);

                }
            }
            */

            // marching cubes hacks
            let offsets = [0];
            self.device.inner.cmd_bind_vertex_buffers(
                *command_buffer,
                0,
                std::slice::from_ref(&output_ssbo.inner),
                &offsets,
            );

            let model = glam::Mat4::IDENTITY;

            let ptr = bytemuck::cast_slice(model.as_ref());

            self.device.inner.cmd_push_constants(
                *command_buffer,
                self.pipeline_layout,
                self.technique
                    .push_constant_range
                    .as_ref()
                    .unwrap()
                    .stage_flags
                    .as_vk(),
                0,
                ptr,
            );

            self.device
                .inner
                .cmd_draw_indirect(*command_buffer, indirect_cmd_ssbo.inner, 0, 1, 0);

            self.device
                .dynamic_rendering
                .cmd_end_rendering(*command_buffer);
        };

        Ok(())
    }
}
