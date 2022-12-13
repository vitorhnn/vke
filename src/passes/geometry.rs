use std::error::Error;
use crate::allocator::{Allocator, MemoryUsage};
use crate::material::Technique;
use crate::per_frame::PerFrame;
use crate::texture::Texture;
use crate::{buffer, material, Device};
use ash::vk;
use glam::Mat4;
use gpu_allocator::vulkan::Allocation;
use sdl2::sys::wchar_t;
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

fn create_graphics_pipeline(device: &Device, technique: &Technique) -> Result<(vk::PipelineLayout, vk::Pipeline), Box<dyn Error>> {
    let vs_module = unsafe {
        let info = vk::ShaderModuleCreateInfo::builder().code(&technique.vs_spv);
        device.inner.create_shader_module(&info, None)?
    };

    let fs_module = unsafe {
        let info = vk::ShaderModuleCreateInfo::builder().code(&technique.fs_spv);
        device.inner.create_shader_module(&info, None)?
    };


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
        }
    }
}
