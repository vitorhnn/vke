use crate::allocator::Allocator;
use crate::device::Device;
use crate::per_frame::PerFrame;
use crate::technique::{
    CookedComputeTechnique, DescriptorSetLayout, PushConstantRange, TechniqueType,
};
use crate::texture_view::TextureView;
use crate::{technique, FRAMES_IN_FLIGHT, SHADER_MAIN_FN_NAME};
use ash::prelude::VkResult;
use ash::vk;
use std::error::Error;
use std::path::Path;
use std::rc::Rc;

pub struct TonemapPass {
    device: Rc<Device>,
    descriptor_sets: PerFrame<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

fn create_compute_pipeline(
    device: &Device,
    technique: &CookedComputeTechnique,
    descriptor_set_layouts: &[Option<DescriptorSetLayout>; 4],
    push_constant_range: &Option<PushConstantRange>,
) -> Result<
    (
        vk::PipelineLayout,
        vk::Pipeline,
        Vec<vk::DescriptorSetLayout>,
    ),
    Box<dyn Error>,
> {
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

    let cs_module = unsafe {
        let info = vk::ShaderModuleCreateInfo::builder().code(&technique.cs_spv);
        device.inner.create_shader_module(&info, None)?
    };

    let cs_stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(cs_module)
        .name(SHADER_MAIN_FN_NAME)
        .build();

    let pipeline_create_infos = vk::ComputePipelineCreateInfo::builder()
        .stage(cs_stage_info)
        .layout(pipeline_layout);

    let pipelines = unsafe {
        device
            .inner
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_create_infos),
                None,
            )
            .unwrap()
    };

    let pipeline = pipelines[0];

    unsafe {
        device.inner.destroy_shader_module(cs_module, None);
    }

    Ok((pipeline_layout, pipeline, descriptor_set_layouts))
}

impl TonemapPass {
    pub fn new(device: Rc<Device>, in_textures: &PerFrame<TextureView>) -> Self {
        let technique = technique::compile_shader(Path::new("./glsl/tonemap"));
        let compute_technique = if let TechniqueType::Compute(compute_technique) = &technique.r#type
        {
            compute_technique
        } else {
            panic!("tonemap was not a compute technique");
        };

        let (pipeline_layout, pipeline, descriptor_set_layouts) = create_compute_pipeline(
            &device,
            &compute_technique,
            &technique.descriptor_set_layouts,
            &technique.push_constant_range,
        )
        .expect("failed to create compute pipeline");

        let descriptor_pool = {
            let pool_size = vk::DescriptorPoolSize {
                descriptor_count: 2 * FRAMES_IN_FLIGHT as u32,
                ty: vk::DescriptorType::STORAGE_IMAGE,
            };

            let pool_create_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(std::slice::from_ref(&pool_size))
                .max_sets(FRAMES_IN_FLIGHT as u32);

            unsafe { device.inner.create_descriptor_pool(&pool_create_info, None) }
                .expect("failed to create descriptor pool for tonemap")
        };

        let descriptor_sets = Self::prepare_descriptor_sets(
            &device,
            descriptor_pool,
            descriptor_set_layouts[0],
            in_textures,
        );

        Self {
            device,
            pipeline_layout,
            pipeline,
            descriptor_sets,
            descriptor_set_layouts,
            descriptor_pool,
        }
    }

    fn prepare_descriptor_sets(
        device: &Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        in_textures: &PerFrame<TextureView>,
    ) -> PerFrame<vk::DescriptorSet> {
        let layouts: [vk::DescriptorSetLayout; FRAMES_IN_FLIGHT] =
            [descriptor_set_layout; FRAMES_IN_FLIGHT];

        let allocate_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: FRAMES_IN_FLIGHT as u32,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };
        let descriptor_sets = unsafe { device.inner.allocate_descriptor_sets(&allocate_info) }
            .expect("tonemap descriptor set allocation failed");
        let images: [_; FRAMES_IN_FLIGHT] = core::array::from_fn(|i| vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            image_view: in_textures.get_resource_for_frame(i).inner,
            sampler: vk::Sampler::null(),
        });

        let writes: [vk::WriteDescriptorSet; FRAMES_IN_FLIGHT] =
            core::array::from_fn(|i| vk::WriteDescriptorSet {
                dst_set: descriptor_sets[i],
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                p_image_info: &images[i],
                descriptor_count: 1,
                ..Default::default()
            });

        unsafe {
            device.inner.update_descriptor_sets(&writes, &[]);
        }

        PerFrame::new_from_vec(descriptor_sets)
    }

    pub fn execute(
        &mut self,
        frame_idx: usize,
        command_buffer: vk::CommandBuffer,
        target_view: vk::ImageView,
    ) -> VkResult<()> {
        let descriptor_set = self.descriptor_sets.get_resource_for_frame(frame_idx);

        unsafe {
            let image_info = vk::DescriptorImageInfo {
                image_layout: vk::ImageLayout::GENERAL,
                image_view: target_view,
                sampler: vk::Sampler::null(),
            };

            let write = vk::WriteDescriptorSet {
                dst_set: *descriptor_set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                p_image_info: &image_info,
                descriptor_count: 1,
                ..Default::default()
            };

            self.device.inner.update_descriptor_sets(&[write], &[]);

            self.device.inner.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );

            // TODO: not hardcode the size of the images :)
            self.device
                .inner
                .cmd_dispatch(command_buffer, 1280 / 8, 720 / 8, 1);
        }

        Ok(())
    }
}
