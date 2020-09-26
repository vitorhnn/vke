use ash::{version::DeviceV1_0, Device, vk};

use std::rc::Rc;

use crate::allocator::Allocation;

struct Transfer {
    device: Rc<Device>,
    staging_allocation: Allocation,
    staging_buffer: vk::Buffer,
}

impl Transfer {
    pub fn new(device: Rc<Device>, ) -> VkResult<Self> {

    }
}
