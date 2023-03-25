use crate::device::Device;
use ash::vk;
use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use shaderc;
use shaderc::{ResolvedInclude, ShaderKind};
use spirv_cross::spirv::{Decoration, Resource, ShaderResources};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::ptr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VertexInputRate {
    Vertex,
    Instance,
}

impl VertexInputRate {
    fn as_vk(&self) -> vk::VertexInputRate {
        match self {
            VertexInputRate::Vertex => vk::VertexInputRate::VERTEX,
            VertexInputRate::Instance => vk::VertexInputRate::INSTANCE,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexInputBindingDescription {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: VertexInputRate,
}

impl VertexInputBindingDescription {
    pub(crate) fn as_vk(&self) -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: self.binding,
            stride: self.stride,
            input_rate: self.input_rate.as_vk(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Format {
    R32G32Sfloat,
    R32G32B32Sfloat,
    R32G32B32A32Sfloat,
}

impl Format {
    fn as_vk(&self) -> vk::Format {
        match self {
            Format::R32G32Sfloat => vk::Format::R32G32_SFLOAT,
            Format::R32G32B32Sfloat => vk::Format::R32G32B32_SFLOAT,
            Format::R32G32B32A32Sfloat => vk::Format::R32G32B32A32_SFLOAT,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexInputAttributeDescription {
    pub location: u32,
    pub binding: u32,
    pub format: Format,
    pub offset: u32,
}

impl VertexInputAttributeDescription {
    pub fn as_vk(&self) -> vk::VertexInputAttributeDescription {
        vk::VertexInputAttributeDescription {
            location: self.location,
            binding: self.binding,
            format: self.format.as_vk(),
            offset: self.offset,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexLayoutInfo {
    pub input_binding_description: VertexInputBindingDescription,
    pub input_attribute_descriptions: Vec<VertexInputAttributeDescription>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DescriptorType {
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
    pub struct ShaderStageFlags: u32 {
        const Vertex = 0b1;
        const Fragment = 0b10;
        const Compute = 0b100;
        const Mesh = 0b1000;
        const Task = 0b10000;
        const All = Self::Vertex.bits() | Self::Fragment.bits() | Self::Compute.bits() | Self::Mesh.bits() | Self::Task.bits();
    }
}

impl ShaderStageFlags {
    pub fn as_vk(&self) -> vk::ShaderStageFlags {
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
pub struct DescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptor_type: DescriptorType,
    pub stage_flags: ShaderStageFlags,
}

impl DescriptorSetLayoutBinding {
    fn as_vk(&self) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: self.binding,
            descriptor_type: self.descriptor_type.into_vk(),
            descriptor_count: 1,
            stage_flags: self.stage_flags.as_vk(),
            p_immutable_samplers: ptr::null(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptorSetLayout {
    // might need to increase this in the future
    pub bindings: HashMap<u32, DescriptorSetLayoutBinding>,
    is_update_after_bind: bool,
}

impl DescriptorSetLayout {
    pub fn as_vk(&self, device: &Device) -> vk::DescriptorSetLayout {
        let bindings: Vec<_> = self
            .bindings
            .iter()
            .map(|(_, binding)| binding.as_vk())
            .collect();

        let flags = if self.is_update_after_bind {
            vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL
        } else {
            vk::DescriptorSetLayoutCreateFlags::empty()
        };

        let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(flags)
            .bindings(&bindings);

        unsafe {
            device
                .inner
                .create_descriptor_set_layout(&create_info, None)
                .expect("failed to create descriptor set layouts")
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushConstantRange {
    size: u32,
    pub stage_flags: ShaderStageFlags,
}

impl PushConstantRange {
    pub fn as_vk(&self) -> vk::PushConstantRange {
        vk::PushConstantRange {
            offset: 0,
            size: self.size,
            stage_flags: self.stage_flags.as_vk(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CookedVertexShaderMetadata {
    pub spv: Vec<u32>,
    pub vertex_layout_info: VertexLayoutInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CookedGraphicsTechnique {
    pub vs_meta: Option<CookedVertexShaderMetadata>,
    pub ms_spv: Option<Vec<u32>>,
    pub fs_spv: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CookedComputeTechnique {
    pub cs_spv: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TechniqueType {
    Graphics(CookedGraphicsTechnique),
    Compute(CookedComputeTechnique),
}

impl TechniqueType {
    pub fn graphics(&self) -> Option<&CookedGraphicsTechnique> {
        match self {
            TechniqueType::Graphics(graphics_technique) => Some(&graphics_technique),
            _ => None,
        }
    }

    pub fn compute(&self) -> Option<&CookedComputeTechnique> {
        match self {
            TechniqueType::Compute(compute_technique) => Some(&compute_technique),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Technique {
    pub r#type: TechniqueType,
    pub descriptor_set_layouts: [Option<DescriptorSetLayout>; 4],
    pub push_constant_range: Option<PushConstantRange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VertexShaderMetadata {
    path: String,
    vertex_layout_info: VertexLayoutInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DescriptorSetMetadata {
    is_update_after_bind: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComputeTechnique {
    cs_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum TechniqueMetadataType {
    Graphics(GraphicsTechnique),
    Compute(ComputeTechnique),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphicsTechnique {
    vs: Option<VertexShaderMetadata>,
    ms_path: Option<String>,
    fs_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TechniqueMetadata {
    r#type: TechniqueMetadataType,
    descriptor_sets: HashMap<u32, DescriptorSetMetadata>,
}

type Ast = spirv_cross::spirv::Ast<spirv_cross::glsl::Target>;

fn build_descriptor_set_layouts_for_descriptor_type(
    sets: &mut [Option<DescriptorSetLayout>; 4],
    sets_metadata: &HashMap<u32, DescriptorSetMetadata>,
    stage: ShaderStageFlags,
    ast: &Ast,
    resources: &[Resource],
    descriptor_type: DescriptorType,
) {
    for resource in resources {
        let set_index = ast
            .get_decoration(resource.id, Decoration::DescriptorSet)
            .unwrap();
        let binding_index = ast
            .get_decoration(resource.id, Decoration::Binding)
            .unwrap();

        let set = &mut sets[set_index as usize].get_or_insert_with(|| DescriptorSetLayout {
            bindings: HashMap::new(),
            is_update_after_bind: sets_metadata
                .get(&set_index)
                .map_or(false, |metadata| metadata.is_update_after_bind),
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
            Entry::Vacant(vacant_slot) => {
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
    sets_metadata: &HashMap<u32, DescriptorSetMetadata>,
    stage: ShaderStageFlags,
    ast: &Ast,
    resources: &ShaderResources,
) {
    build_descriptor_set_layouts_for_descriptor_type(
        sets,
        sets_metadata,
        stage,
        ast,
        &resources.uniform_buffers,
        DescriptorType::UniformBuffer,
    );
    build_descriptor_set_layouts_for_descriptor_type(
        sets,
        sets_metadata,
        stage,
        ast,
        &resources.separate_samplers,
        DescriptorType::Sampler,
    );
    build_descriptor_set_layouts_for_descriptor_type(
        sets,
        sets_metadata,
        stage,
        ast,
        &resources.separate_images,
        DescriptorType::SampledImage,
    );
    build_descriptor_set_layouts_for_descriptor_type(
        sets,
        sets_metadata,
        stage,
        ast,
        &resources.storage_buffers,
        DescriptorType::StorageBuffer,
    );
}

fn parse_push_constant_for_stage(
    stage: ShaderStageFlags,
    ast: &Ast,
    resources: &ShaderResources,
) -> Option<PushConstantRange> {
    assert!(resources.push_constant_buffers.len() < 2);

    resources.push_constant_buffers.get(0).map(|res| {
        let size = ast
            .get_declared_struct_size(res.base_type_id)
            .expect("failed to get push constant size");

        PushConstantRange {
            size,
            stage_flags: stage,
        }
    })
}

pub fn compile_shader(dir: &Path) -> Technique {
    // collect source files
    if !dir.is_dir() {
        panic!("dir was not dir, we don't have proper error handling for this yet");
    }

    let metadata_path: PathBuf = dir.join(Path::new("metadata.ron"));
    let metadata_reader = BufReader::new(File::open(&metadata_path).unwrap());
    let metadata: TechniqueMetadata = ron::de::from_reader(metadata_reader).unwrap();

    let mut sets: [Option<DescriptorSetLayout>; 4] = Default::default();
    let mut push_constant_range = None;
    let compiler = shaderc::Compiler::new().unwrap();
    let mut compile_options = shaderc::CompileOptions::new().unwrap();
    compile_options.set_include_callback(|file_name, _, _, _| {
        let resolved_path: PathBuf = dir.join(Path::new(&file_name));

        Ok(ResolvedInclude {
            content: fs::read_to_string(&resolved_path).unwrap(),
            resolved_name: resolved_path.to_str().unwrap().to_owned(),
        })
    });

    match metadata.r#type {
        TechniqueMetadataType::Graphics(GraphicsTechnique {
            vs,
            ms_path,
            fs_path,
        }) => {
            let compiled_vs = match vs {
                Some(vs_meta) => {
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
                        &metadata.descriptor_sets,
                        ShaderStageFlags::Vertex,
                        &ast,
                        &resources,
                    );

                    push_constant_range =
                        parse_push_constant_for_stage(ShaderStageFlags::Vertex, &ast, &resources);

                    Some((compiled_vs, vs_meta.vertex_layout_info))
                }
                None => None,
            };

            let compiled_ms = match ms_path {
                Some(ms_path) => {
                    let resolved_path: PathBuf = dir.join(Path::new(&ms_path));
                    let ms_source = fs::read_to_string(&resolved_path).unwrap();
                    let compiled_ms = compiler
                        .compile_into_spirv(
                            &ms_source,
                            ShaderKind::Mesh,
                            &ms_path,
                            "main",
                            Some(&compile_options),
                        )
                        .unwrap();

                    let module = spirv_cross::spirv::Module::from_words(compiled_ms.as_binary());
                    let ast = Ast::parse(&module).unwrap();

                    let resources = ast.get_shader_resources().unwrap();

                    // build descriptor set layouts
                    build_descriptor_set_layouts_for_stage(
                        &mut sets,
                        &metadata.descriptor_sets,
                        ShaderStageFlags::Vertex,
                        &ast,
                        &resources,
                    );

                    push_constant_range =
                        parse_push_constant_for_stage(ShaderStageFlags::Vertex, &ast, &resources);

                    Some(compiled_ms)
                }
                None => None,
            };

            let compiled_fs = {
                let resolved_path: PathBuf = dir.join(Path::new(&fs_path));
                let fs_source = fs::read_to_string(&resolved_path).unwrap();
                let compiled_fs = compiler
                    .compile_into_spirv(
                        &fs_source,
                        ShaderKind::Fragment,
                        &fs_path,
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
                    &metadata.descriptor_sets,
                    ShaderStageFlags::Fragment,
                    &ast,
                    &resources,
                );

                let fs_push_constants =
                    parse_push_constant_for_stage(ShaderStageFlags::Fragment, &ast, &resources);

                if let Some(fs_push_constants) = fs_push_constants {
                    if let Some(push_constant_range) = push_constant_range.as_mut() {
                        if fs_push_constants.size != push_constant_range.size {
                            panic!("push constant size mismatch between stages");
                        }

                        push_constant_range.stage_flags |= ShaderStageFlags::Fragment;
                    } else {
                        push_constant_range = Some(fs_push_constants);
                    }
                }

                compiled_fs
            };

            Technique {
                push_constant_range,
                r#type: TechniqueType::Graphics(CookedGraphicsTechnique {
                    vs_meta: compiled_vs.map(|(compiled_vs, vertex_layout_info)| {
                        CookedVertexShaderMetadata {
                            spv: compiled_vs.as_binary().to_vec(),
                            vertex_layout_info,
                        }
                    }),
                    ms_spv: compiled_ms.map(|x| x.as_binary().to_vec()),
                    fs_spv: compiled_fs.as_binary().to_vec(),
                }),
                descriptor_set_layouts: sets,
            }
        }
        TechniqueMetadataType::Compute(ComputeTechnique { cs_path }) => {
            let compiled_cs = {
                let resolved_path: PathBuf = dir.join(Path::new(&cs_path));
                let cs_source = fs::read_to_string(&resolved_path).unwrap();
                let compiled_cs = compiler
                    .compile_into_spirv(
                        &cs_source,
                        ShaderKind::Compute,
                        &cs_path,
                        "main",
                        Some(&compile_options),
                    )
                    .unwrap();

                let module = spirv_cross::spirv::Module::from_words(compiled_cs.as_binary());
                let ast = Ast::parse(&module).unwrap();

                let resources = ast.get_shader_resources().unwrap();

                // build descriptor set layouts
                build_descriptor_set_layouts_for_stage(
                    &mut sets,
                    &metadata.descriptor_sets,
                    ShaderStageFlags::Compute,
                    &ast,
                    &resources,
                );

                push_constant_range =
                    parse_push_constant_for_stage(ShaderStageFlags::Compute, &ast, &resources);

                compiled_cs
            };

            Technique {
                push_constant_range,
                r#type: TechniqueType::Compute(CookedComputeTechnique {
                    cs_spv: compiled_cs.as_binary().to_vec(),
                }),
                descriptor_set_layouts: sets,
            }
        }
    }
}
