use std::error::Error;
use std::path::Path;
use std::rc::Rc;

use crate::allocator::{Allocator, MemoryUsage};
use crate::buffer::Buffer;
use crate::device::Device;
use crate::per_frame::PerFrame;
use crate::technique::{
    CookedComputeTechnique, DescriptorSetLayout, PushConstantRange, TechniqueType,
};
use crate::{buffer, technique, FRAMES_IN_FLIGHT, SHADER_MAIN_FN_NAME};
use ash::prelude::VkResult;
use ash::vk::{self, AccessFlags};
use gpu_allocator::vulkan::Allocation;

use super::marching_cubes_constants::TRI_TABLE;

pub struct MarchingCubesComputePass {
    device: Rc<Device>,
    allocator: Rc<Allocator>,
    descriptor_pools: PerFrame<vk::DescriptorPool>,
    params_uniform_buffer: buffer::Buffer,
    ubo_allocation: Allocation,
    triangles_output_ssbo: buffer::Buffer,
    triangles_output_allocation: Allocation,
    triangles_input_ssbo: buffer::Buffer,
    triangles_input_allocation: Allocation,
    triangles_count_ssbo: buffer::Buffer,
    triangles_count_allocation: Allocation,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    points: [glam::Vec4; 10],
    points_velocities: [glam::Vec4; 10],
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

#[derive(Debug)]
#[repr(C)]
struct Triangle {
    v0: glam::Vec4,
    norm0: glam::Vec4,
    v1: glam::Vec4,
    norm1: glam::Vec4,
    v2: glam::Vec4,
    norm2: glam::Vec4,
}

const SURFACE_RES: usize = 64;
const MAX_TRIS: usize = SURFACE_RES.pow(3);

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

impl MarchingCubesComputePass {
    pub fn new(device: Rc<Device>, allocator: Rc<Allocator>) -> Self {
        let descriptor_pools = PerFrame::new(|| {
            let pool_sizes = [
                vk::DescriptorPoolSize {
                    descriptor_count: 32,
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                },
                vk::DescriptorPoolSize {
                    descriptor_count: 32,
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                },
            ];

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

        let (params_uniform_buffer, ubo_allocation) = {
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size((std::mem::size_of::<ParamsUbo>() * FRAMES_IN_FLIGHT) as u64)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            allocator
                .create_buffer(&buffer_create_info, MemoryUsage::HostToDevice)
                .expect("ubo allocation failure")
        };

        let (triangles_output_ssbo, triangles_output_allocation) = {
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size((std::mem::size_of::<Triangle>() * FRAMES_IN_FLIGHT * MAX_TRIS) as u64)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            allocator
                .create_buffer(&buffer_create_info, MemoryUsage::HostToDevice)
                .expect("ssbo allocation failure")
        };

        let (triangles_input_ssbo, triangles_input_allocation) = {
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size((std::mem::size_of::<i32>() * FRAMES_IN_FLIGHT * 4096) as u64)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            allocator
                .create_buffer(&buffer_create_info, MemoryUsage::HostToDevice)
                .expect("ssbo allocation failure")
        };

        let (triangles_count_ssbo, triangles_count_allocation) = {
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size((std::mem::size_of::<vk::DrawIndirectCommand>() * FRAMES_IN_FLIGHT) as u64)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            allocator
                .create_buffer(&buffer_create_info, MemoryUsage::HostToDevice)
                .expect("ssbo allocation failure")
        };

        let technique = technique::compile_shader(Path::new("./glsl/marching_compute"));

        let points: [glam::Vec4; 10] = {
            let mut rng = oorandom::Rand32::new(42);
            let mut points: [glam::Vec4; 10] = Default::default();

            for point in &mut points {
                *point = loop {
                    let mut gen_rand = || {
                        return rng.rand_float() * (5.0 - (-5.0)) + (-5.0);
                    };

                    let vec = glam::Vec4::new(gen_rand(), gen_rand(), gen_rand(), 0.0);

                    if vec.length() < 5.0 {
                        break vec;
                    }
                };
            }

            points
        };

        let points_velocities: [glam::Vec4; 10] = {
            let mut rng = oorandom::Rand32::new(42);
            let mut velocities: [glam::Vec4; 10] = Default::default();

            for velocity in &mut velocities {
                *velocity =
                    glam::Vec4::new(rng.rand_float(), rng.rand_float(), rng.rand_float(), 0.0);
                *velocity *= 0.01;
            }

            velocities
        };

        if let TechniqueType::Compute(compute_technique) = &technique.r#type {
            let (pipeline_layout, pipeline, descriptor_set_layouts) = create_compute_pipeline(
                &device,
                &compute_technique,
                &technique.descriptor_set_layouts,
                &technique.push_constant_range,
            )
            .expect("failed to create compute pipeline");

            Self {
                device,
                allocator,
                descriptor_pools,
                params_uniform_buffer,
                ubo_allocation,
                triangles_output_ssbo,
                triangles_output_allocation,
                triangles_input_ssbo,
                triangles_input_allocation,
                triangles_count_ssbo,
                triangles_count_allocation,
                descriptor_set_layouts,
                pipeline,
                pipeline_layout,
                points,
                points_velocities,
            }
        } else {
            panic!("not a compute technique");
        }
    }

    fn prepare_descriptor_set(&mut self, frame_idx: usize) -> VkResult<vk::DescriptorSet> {
        let mapped = self
            .allocator
            .map(&self.ubo_allocation)
            .expect("failed to map ubo memory") as *mut ParamsUbo;
        let slice = unsafe { std::slice::from_raw_parts_mut(mapped, FRAMES_IN_FLIGHT) };
        let ubo = &mut slice[frame_idx];

        ubo.surface_level = -0.75;
        ubo.smooth = 2.0;
        ubo.linear = 1;
        ubo.points_count = 2;
        ubo.world_bounds = glam::Vec3::new(10.0, 10.0, 10.0);

        for (idx, point) in self.points.iter_mut().enumerate() {
            let velocity = &mut self.points_velocities[idx];
            *point += *velocity;

            if point.x.abs() > 5.0 {
                velocity.x *= -1.0;
            }

            if point.y.abs() > 5.0 {
                velocity.y *= -1.0;
            }

            if point.z.abs() > 5.0 {
                velocity.z *= -1.0;
            }
        }

        ubo.points.copy_from_slice(&self.points);

        self.allocator.unmap(&self.ubo_allocation);

        let mapped = self
            .allocator
            .map(&self.triangles_input_allocation)
            .expect("failed to map ssbo memory") as *mut i32;
        let slice = unsafe { std::slice::from_raw_parts_mut(mapped.add(4096 * frame_idx), 4096) };
        slice.copy_from_slice(&TRI_TABLE);

        self.allocator.unmap(&self.triangles_input_allocation);

        let mapped =
            self.allocator
                .map(&self.triangles_count_allocation)
                .expect("failed to map ssbo memory") as *mut vk::DrawIndirectCommand;
        unsafe {
            let mut cmd = &mut *(mapped.add(frame_idx));
            cmd.vertex_count = 0;
            cmd.instance_count = 1;
            cmd.first_vertex = 0;
            cmd.first_instance = 0;
        };

        self.allocator.unmap(&self.triangles_count_allocation);

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

            let buffer_infos = [
                vk::DescriptorBufferInfo {
                    buffer: self.params_uniform_buffer.inner,
                    offset: (std::mem::size_of::<ParamsUbo>() * offset_multiplier) as u64,
                    range: std::mem::size_of::<ParamsUbo>() as u64,
                },
                vk::DescriptorBufferInfo {
                    buffer: self.triangles_output_ssbo.inner,
                    offset: (std::mem::size_of::<Triangle>() * MAX_TRIS * offset_multiplier) as u64,
                    range: (std::mem::size_of::<Triangle>() * MAX_TRIS) as u64,
                },
                // 4096 is currently a magic constant
                vk::DescriptorBufferInfo {
                    buffer: self.triangles_input_ssbo.inner,
                    offset: (std::mem::size_of::<i32>() * 4096 * offset_multiplier) as u64,
                    range: (std::mem::size_of::<i32>() * 4096) as u64,
                },
                vk::DescriptorBufferInfo {
                    buffer: self.triangles_count_ssbo.inner,
                    offset: (std::mem::size_of::<vk::DrawIndirectCommand>() * offset_multiplier)
                        as u64,
                    range: std::mem::size_of::<vk::DrawIndirectCommand>() as u64,
                },
            ];

            let writes = [
                vk::WriteDescriptorSet {
                    dst_set: set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    p_buffer_info: &buffer_infos[0] as *const _,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: set,
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 3,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &buffer_infos[1] as *const _,
                    ..Default::default()
                },
            ];

            self.device.inner.update_descriptor_sets(&writes, &[]);
        }

        Ok(set)
    }

    pub fn execute(
        &mut self,
        frame_idx: usize,
        command_buffer: &vk::CommandBuffer,
    ) -> VkResult<(&Buffer, &Buffer)> {
        unsafe {
            let pool = self.descriptor_pools.get_resource_for_frame(frame_idx);

            self.device
                .inner
                .reset_descriptor_pool(*pool, vk::DescriptorPoolResetFlags::empty())?;

            self.device.inner.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            let frame_descriptor_set = self.prepare_descriptor_set(frame_idx)?;

            self.device.inner.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&frame_descriptor_set),
                &[],
            );

            self.device.inner.cmd_dispatch(
                *command_buffer,
                (SURFACE_RES as u32) / 8,
                (SURFACE_RES as u32) / 8,
                (SURFACE_RES as u32) / 8,
            );
        }

        let output_ssbo = &self.triangles_output_ssbo;
        let count_ssbo = &self.triangles_count_ssbo;

        unsafe {
            let barriers = [
                vk::MemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                    .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)
                    .build(),
                vk::MemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT)
                    .dst_access_mask(vk::AccessFlags2::VERTEX_ATTRIBUTE_READ)
                    .build(),
            ];
            self.device.inner.cmd_pipeline_barrier2(
                *command_buffer,
                &vk::DependencyInfo::builder().memory_barriers(&barriers),
            );
        }

        Ok((output_ssbo, count_ssbo))
    }
}
