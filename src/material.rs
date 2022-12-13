use ash::vk;
use bitflags::bitflags;
use glam::{Vec2, Vec3, Vec4};
use sdl2::libc::wait;
use serde::{Deserialize, Serialize};
use shaderc;
use shaderc::{ResolvedInclude, ShaderKind};
use spirv_cross::spirv::{Decoration, Resource, ShaderResources};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::ptr;
use sdl2::sys::wchar_t;

#[derive(Debug, Clone, Serialize, Deserialize)]
enum VertexInputRate {
    Vertex,
    Instance,
}

impl VertexInputRate {
    fn into_vk(self) -> vk::VertexInputRate {
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
    fn into_vk(self) -> vk::Format {
        match self {
            Format::R32G32Sfloat => vk::Format::R32G32_SFLOAT,
            Format::R32G32B32Sfloat => vk::Format::R32G32B32_SFLOAT,
            Format::R32G32B32A32Sfloat => vk::Format::R32G32B32A32_SFLOAT,
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
    fn into_vk(self) -> vk::VertexInputAttributeDescription {
        vk::VertexInputAttributeDescription {
            location: self.location,
            binding: self.binding,
            format: self.format.into_vk(),
            offset: self.offset,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VertexLayoutInfo {
    input_binding_description: VertexInputBindingDescription,
    input_attribute_descriptions: Vec<VertexInputAttributeDescription>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
enum DescriptorType {
    UniformBuffer,
    Sampler,
    SampledImage,
    StorageBuffer,
}

impl DescriptorType {
    fn into_vk(self) -> vk::DescriptorType {
        match self {
            Self::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            Self::Sampler => vk::DescriptorType::SAMPLER,
            Self::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
            Self::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
    struct ShaderStageFlags: u32 {
        const Vertex = 0b1;
        const Fragment = 0b10;
        const Compute = 0b100;
        const Mesh = 0b1000;
        const Task = 0b10000;
        const All = Self::Vertex.bits() | Self::Fragment.bits() | Self::Compute.bits() | Self::Mesh.bits() | Self::Task.bits();
    }
}

impl ShaderStageFlags {
    fn into_vk(self) -> vk::ShaderStageFlags {
        let mut output_flags = vk::ShaderStageFlags::empty();

        if self.contains(Self::Vertex) {
            output_flags |= vk::ShaderStageFlags::VERTEX;
        }

        if self.contains(Self::Fragment) {
            output_flags |= vk::ShaderStageFlags::FRAGMENT;
        }

        if self.contains(Self::Compute) {
            output_flags |= vk::ShaderStageFlags::COMPUTE;
        }

        if self.contains(Self::Mesh) {
            output_flags |= vk::ShaderStageFlags::MESH_EXT;
        }

        if self.contains(Self::Task) {
            output_flags |= vk::ShaderStageFlags::TASK_EXT;
        }

        output_flags
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DescriptorSetLayoutBinding {
    binding: u32,
    descriptor_type: DescriptorType,
    stage_flags: ShaderStageFlags,
}

impl DescriptorSetLayoutBinding {
    fn into_vk(self) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: self.binding,
            descriptor_type: self.descriptor_type.into_vk(),
            descriptor_count: 0,
            stage_flags: self.stage_flags.into_vk(),
            p_immutable_samplers: ptr::null(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DescriptorSetLayout {
    // might need to increase this in the future
    bindings: HashMap<u32, DescriptorSetLayoutBinding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Technique {
    pub vertex_layout_info: VertexLayoutInfo,
    pub vs_spv: Vec<u32>,
    pub fs_spv: Vec<u32>,
    pub descriptor_set_layouts: [Option<DescriptorSetLayout>; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VertexShaderMetadata {
    path: String,
    vertex_layout_info: VertexLayoutInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TechniqueMetadata {
    vs: Option<VertexShaderMetadata>,
    fs_path: String,
}

type Ast = spirv_cross::spirv::Ast<spirv_cross::glsl::Target>;

fn build_descriptor_set_layouts_for_descriptor_type(
    sets: &mut [Option<DescriptorSetLayout>; 4],
    stage: ShaderStageFlags,
    ast: &Ast,
    resources: &[Resource],
    descriptor_type: DescriptorType,
) {
    for resource in resources {
        let set_index = ast
            .get_decoration(resource.id, Decoration::DescriptorSet)
            .unwrap();
        let binding_index = ast.get_decoration(resource.id, Decoration::Binding).unwrap();

        let set = &mut sets[set_index as usize].get_or_insert_with(|| DescriptorSetLayout {
            bindings: HashMap::new(),
        });

        match set.bindings.entry(binding_index) {
            Entry::Occupied(mut binding_entry) => {
                let binding = binding_entry.get_mut();

                if binding.descriptor_type != descriptor_type {
                    panic!(
                        "mismatched descriptor types at set {} binding {}",
                        set_index, binding_index
                    );
                }

                binding.stage_flags |= stage;
            }
            Entry::Vacant(mut vacant_slot) => {
                vacant_slot.insert(DescriptorSetLayoutBinding {
                    binding: binding_index,
                    stage_flags: stage,
                    descriptor_type: descriptor_type,
                });
            }
        }
    }
}

fn build_descriptor_set_layouts_for_stage(
    sets: &mut [Option<DescriptorSetLayout>; 4],
    stage: ShaderStageFlags,
    ast: &Ast,
    resources: &ShaderResources,
) {
    build_descriptor_set_layouts_for_descriptor_type(
        sets,
        stage,
        ast,
        &resources.uniform_buffers,
        DescriptorType::UniformBuffer,
    );
    build_descriptor_set_layouts_for_descriptor_type(
        sets,
        stage,
        ast,
        &resources.separate_samplers,
        DescriptorType::Sampler,
    );
    build_descriptor_set_layouts_for_descriptor_type(
        sets,
        stage,
        ast,
        &resources.separate_images,
        DescriptorType::SampledImage,
    );
}

pub fn compile_shader(dir: &Path) -> Technique {
    // collect source files
    if !dir.is_dir() {
        panic!("dir was not dir, we don't have proper error handling for this yet");
    }

    let metadata_path: PathBuf = dir.join(Path::new("metadata.json"));
    let metadata_reader = BufReader::new(File::open(&metadata_path).unwrap());
    let metadata: TechniqueMetadata = serde_json::from_reader(metadata_reader).unwrap();

    let mut sets: [Option<DescriptorSetLayout>; 4] = Default::default();
    let compiler = shaderc::Compiler::new().unwrap();
    let mut compile_options = shaderc::CompileOptions::new().unwrap();
    compile_options.set_include_callback(|file_name, _, _, _| {
        let resolved_path: PathBuf = dir.join(Path::new(&file_name));

        Ok(ResolvedInclude {
            content: fs::read_to_string(&resolved_path).unwrap(),
            resolved_name: resolved_path.to_str().unwrap().to_owned(),
        })
    });

    let compiled_vs = {
        let vs_meta = metadata.vs.as_ref().unwrap();
        let resolved_path: PathBuf = dir.join(Path::new(&vs_meta.path));
        let vs_source = fs::read_to_string(&resolved_path).unwrap();
        let compiled_vs = compiler
            .compile_into_spirv(
                &vs_source,
                ShaderKind::Vertex,
                &vs_meta.path,
                "main",
                Some(&compile_options),
            )
            .unwrap();

        let module = spirv_cross::spirv::Module::from_words(compiled_vs.as_binary());
        let ast = Ast::parse(&module).unwrap();

        let resources = ast.get_shader_resources().unwrap();

        // build descriptor set layouts
        build_descriptor_set_layouts_for_stage(
            &mut sets,
            ShaderStageFlags::Vertex,
            &ast,
            &resources,
        );

        compiled_vs
    };

    let compiled_fs = {
        let resolved_path: PathBuf = dir.join(Path::new(&metadata.fs_path));
        let fs_source = fs::read_to_string(&resolved_path).unwrap();
        let compiled_fs = compiler
            .compile_into_spirv(
                &fs_source,
                ShaderKind::Fragment,
                &metadata.fs_path,
                "main",
                Some(&compile_options),
            )
            .unwrap();

        let module = spirv_cross::spirv::Module::from_words(compiled_fs.as_binary());
        let ast = Ast::parse(&module).unwrap();

        let resources = ast.get_shader_resources().unwrap();

        // build descriptor set layouts
        build_descriptor_set_layouts_for_stage(
            &mut sets,
            ShaderStageFlags::Fragment,
            &ast,
            &resources,
        );

        compiled_fs
    };

    Technique {
        fs_spv: compiled_fs.as_binary().to_vec(),
        vs_spv: compiled_vs.as_binary().to_vec(),
        vertex_layout_info: metadata.vs.unwrap().vertex_layout_info,
        descriptor_set_layouts: sets,
    }
}
