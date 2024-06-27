use std::convert::TryInto;
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use gltf::image::{Format, Source};
use gltf::mesh::util::{ReadIndices, ReadTexCoords};
use gltf::mesh::Mode;
use gltf::scene::Transform;
use gltf::Semantic;
use snafu::prelude::*;
use ash::vk;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Unable to read gltf file from filesystem: {}", source))]
    Import { source: gltf::Error },
    #[snafu(display("Gltf file has no default scene"))]
    NoDefaultScene,
    #[snafu(display("one of the primitives did not have positions"))]
    NoPositions,
    #[snafu(display("one of the primitives did not have uvs"))]
    NoUvs,
    #[snafu(display("one of the primitives did not have normals"))]
    NoNormals,
    #[snafu(display("one of the primitives did not have tangents"))]
    NoTangents,
    #[snafu(display("one of the primitives did not have indices"))]
    NoIndices,
}

#[derive(Debug)]
pub struct Texture {
    pub info: crate::texture::TextureInfo,
    pub data: Vec<u8>,
}

impl Texture {
    fn from_gltf(tex: &gltf::Texture, images: &Vec<gltf::image::Data>) -> Self {
        use gltf::image::Format;
        let image = &images[tex.source().index()];
        let info = crate::texture::TextureInfo {
            extent: vk::Extent3D {
                depth: 1,
                width: image.width,
                height: image.height,
            },
            // TODO: this is probably incorrect. check later if we have color problems
            format: match image.format {
                Format::R8 => vk::Format::R8_UNORM,
                Format::R8G8 => vk::Format::R8G8_UNORM,
                Format::R8G8B8 => vk::Format::R8G8B8_UNORM,
                Format::R8G8B8A8 => vk::Format::R8G8B8A8_UNORM,
                Format::R16 => vk::Format::R16_UNORM,
                Format::R16G16 => vk::Format::R16G16_UNORM,
                Format::R16G16B16 => vk::Format::R16G16B16_UNORM,
                Format::R16G16B16A16 => vk::Format::R16G16B16A16_UNORM,
                Format::R32G32B32FLOAT => vk::Format::R32G32B32_SFLOAT,
                Format::R32G32B32A32FLOAT => vk::Format::R32G32B32A32_SFLOAT,
                _ => todo!()
            }
        };

        Self {
            data: image.pixels.clone(),
            info
        }
    }
}

#[derive(Debug)]
pub struct Mesh {
    pub indices: Vec<u16>,
    pub vertices: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub normals: Vec<Vec3>,
    pub tangents: Vec<Vec4>,
    pub material_index: u32,
}

#[derive(Debug)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub transform: Mat4,
}

#[derive(Debug)]
pub struct Material {
    pub base_color_factor: Vec4,
    pub diffuse: Option<Texture>,
    pub normal: Option<Texture>,
    pub metallic_roughness: Option<Texture>,
}

#[derive(Debug)]
pub struct Scene {
    pub models: Vec<Model>,
    pub materials: Vec<Material>
}

impl Scene {
    pub fn from_gltf(path: &std::path::Path) -> Result<Self, Error> {
        let (document, buffers, images) = gltf::import(path).context(ImportSnafu {})?;

        let mut materials = Vec::new();

        for material in document.materials() {
            let base_color_factor: Vec4 = material.pbr_metallic_roughness().base_color_factor().into();
            let diffuse = material.pbr_metallic_roughness().base_color_texture().map(|image| Texture::from_gltf(&image.texture(), &images));
            let metallic_roughness = material.pbr_metallic_roughness().metallic_roughness_texture().map(|image| Texture::from_gltf(&image.texture(), &images));
            let normal = material.normal_texture().map(|image| Texture::from_gltf(&image.texture(), &images));
            materials.push(Material {
                base_color_factor,
                diffuse,
                metallic_roughness,
                normal,
            })
        }

        let default_scene = document.default_scene().ok_or(Error::NoDefaultScene {})?;

        let mut models = Vec::new();

        for node in default_scene.nodes() {
            let transform = match node.transform() {
                Transform::Matrix { matrix } => Mat4::from_cols_array_2d(&matrix),
                Transform::Decomposed {
                    translation,
                    rotation,
                    scale,
                } => Mat4::from_scale_rotation_translation(
                    Vec3::from(scale),
                    Quat::from_array(rotation),
                    Vec3::from(translation),
                ),
            };

            if node.children().len() > 0 {
                todo!("children node not implemented")
            }

            let mut model = Model {
                meshes: Vec::new(),
                transform,
            };

            if let Some(mesh) = node.mesh() {
                for primitive in mesh.primitives() {
                    assert_eq!(primitive.mode(), Mode::Triangles);

                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                    let vertices: Vec<_> = reader
                        .read_positions()
                        .ok_or(Error::NoPositions)?
                        .map(Vec3::from)
                        .collect();

                    let uvs: Vec<_> = if let ReadTexCoords::F32(iter) =
                        reader.read_tex_coords(0).ok_or(Error::NoUvs)?
                    {
                        iter.map(Vec2::from).collect()
                    } else {
                        unimplemented!()
                    };

                    let normals: Vec<_> = reader
                        .read_normals()
                        .ok_or(Error::NoNormals)?
                        .map(Vec3::from)
                        .collect();

                    let tangents: Vec<_> = reader
                        .read_tangents()
                        .ok_or(Error::NoTangents)?
                        .map(Vec4::from)
                        .collect();

                    let indices: Vec<u16> = if let ReadIndices::U16(iter) =
                        reader.read_indices().ok_or(Error::NoIndices)?
                    {
                        iter.collect()
                    } else {
                        unimplemented!()
                    };

                    // idx 0 is used for the default material
                    let material_index = primitive.material().index().map_or(0, |idx| idx + 1).try_into().expect("material index overflow");

                    model.meshes.push(Mesh {
                        vertices,
                        uvs,
                        normals,
                        tangents,
                        indices,
                        material_index,
                    })
                }
            }

            models.push(model);
        }

        Ok(Self { models, materials })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_sponza() {
        Scene::from_gltf(Path::new(&"./Sponza.glb"));
    }

    #[test]
    fn test_bistro() {
        Scene::from_gltf(Path::new(&"/tmp/Bistro_Godot.glb"));
    }
}
