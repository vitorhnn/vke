use ash::vk;
/// Serde-compatible, Vulkan convertable types.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Extent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl Extent3D {
    pub fn from_vk(extent: vk::Extent3D) -> Self {
        Self {
            width: extent.width,
            height: extent.height,
            depth: extent.depth,
        }
    }

    pub fn as_vk(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Format {
    R32G32Sfloat,
    R32G32B32Sfloat,
    R32G32B32A32Sfloat,

    R8G8B8A8Unorm,
}

impl Format {
    pub fn from_vk(format: vk::Format) -> Self {
        match format {
            vk::Format::R32G32_SFLOAT => Format::R32G32Sfloat,
            vk::Format::R32G32B32_SFLOAT => Format::R32G32B32Sfloat,
            vk::Format::R32G32B32A32_SFLOAT => Format::R32G32B32A32Sfloat,

            vk::Format::R8G8B8A8_UNORM => Format::R8G8B8A8Unorm,
            _ => todo!(),
        }
    }
    pub fn as_vk(&self) -> vk::Format {
        match self {
            Format::R32G32Sfloat => vk::Format::R32G32_SFLOAT,
            Format::R32G32B32Sfloat => vk::Format::R32G32B32_SFLOAT,
            Format::R32G32B32A32Sfloat => vk::Format::R32G32B32A32_SFLOAT,

            Format::R8G8B8A8Unorm => vk::Format::R8G8B8A8_UNORM,
        }
    }
}
