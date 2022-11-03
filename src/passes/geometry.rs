use crate::{buffer, Device};
use std::rc::Rc;
use ash::vk;
use gpu_allocator::vulkan::Allocation;

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
    frame_resources: FrameResources,
}

impl GeometryPass {
    fn new(device: Rc<Device>) -> Self {

        Self { device }
    }
}
