use std::ptr;
use ash::vk;
use glam::{Vec2, Vec3, Vec4};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
enum VertexInputRate {
    Vertex,
    Instance
}

impl VertexInputRate {
    fn to_vk(self) -> vk::VertexInputRate {
        match self {
            VertexInputRate::Vertex => vk::VertexInputRate::VERTEX,
            VertexInputRate::Instance => vk::VertexInputRate::INSTANCE,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VertexInputBindingDescription {
    binding: u32,
    stride: u32,
    input_rate: VertexInputRate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Format {
    R32G32Sfloat,
    R32G32B32Sfloat,
    R32G32B32A32Sfloat,
}

impl Format {
    fn to_vk(self) -> vk::Format {
        match self {
            Format::R32G32Sfloat => vk::Format::R32G32_SFLOAT,
            Format::R32G32B32Sfloat => vk::Format::R32G32B32_SFLOAT,
            Format::R32G32B32A32Sfloat => vk::Format::R32G32B32A32_SFLOAT
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VertexInputAttributeDescription {
    location: u32,
    binding: u32,
    format: Format,
    offset: u32,
}

impl VertexInputAttributeDescription {
    fn to_vk(self) -> vk::VertexInputAttributeDescription {
        vk::VertexInputAttributeDescription {
            location: self.location,
            binding: self.binding,
            format: self.format.to_vk(),
            offset: self.offset,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VertexLayoutInfo {
    input_binding_description: VertexInputBindingDescription,
    input_attribute_description: Vec<VertexInputAttributeDescription>
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum DescriptorType {
    UniformBuffer,
    Sampler,
    SampledImage,
}

impl DescriptorType {
    fn to_vk(self) -> vk::DescriptorType {
        match self {
            Self::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            Self::Sampler => vk::DescriptorType::SAMPLER,
            Self::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ShaderStageFlags {
    Vertex,
    Fragment,
    All
}

impl ShaderStageFlags {
    fn to_vk(self) -> vk::ShaderStageFlags {
        match self {
            Self::Vertex => vk::ShaderStageFlags::VERTEX,
            Self::Fragment => vk::ShaderStageFlags::FRAGMENT,
            Self::All => vk::ShaderStageFlags::ALL,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DescriptorSetLayoutBinding {
    binding: u32,
    descriptor_type: DescriptorType,
    stage_flags: ShaderStageFlags,
}

impl DescriptorSetLayoutBinding {
    fn to_vk(self) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: self.binding,
            descriptor_type: self.descriptor_type.to_vk(),
            descriptor_count: 0,
            stage_flags: self.stage_flags.to_vk(),
            p_immutable_samplers: ptr::null(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DescriptorSetLayout {
    // might need to increase this in the future
    bindings: [DescriptorSetLayoutBinding; 8],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Technique<'a> {
    vertex_format: VertexLayoutInfo,
    vs_spv: &'a str,
    fs_spv: &'a str,
    descriptor_set_layouts: [DescriptorSetLayout; 4],
}