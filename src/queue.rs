use ash::{prelude::VkResult, vk};

use std::rc::Rc;

use crate::device::RawDevice;

pub struct Queue {
    pub device: Rc<RawDevice>,
    pub inner: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub family_index: u32,
}

impl Queue {
    pub fn new(device: Rc<RawDevice>, family_index: u32, queue: vk::Queue) -> VkResult<Self> {
        let command_pool_create_info = vk::CommandPoolCreateInfo {
            queue_family_index: family_index,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };

        let pool = unsafe { device.create_command_pool(&command_pool_create_info, None)? };

        Ok(Self {
            device,
            family_index,
            inner: queue,
            command_pool: pool,
        })
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        eprintln!("destroy command pool");
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}
