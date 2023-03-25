use std::ops::Mul;

use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use gltf::mesh::util::{ReadIndices, ReadTexCoords};
use gltf::mesh::Mode;
use gltf::scene::Transform;
use snafu::prelude::*;

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
    info: crate::texture::TextureInfo,
    data: Vec<u8>,
}

#[derive(Debug)]
pub struct Mesh {
    pub indices: Vec<u16>,
    pub vertices: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub normals: Vec<Vec3>,
    pub tangents: Vec<Vec4>,
}

#[derive(Debug)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub transform: Mat4,
}

#[derive(Debug)]
pub struct Scene {
    pub models: Vec<Model>,
}

impl Scene {
    pub fn from_gltf(path: &std::path::Path) -> Result<Self, Error> {
        let (document, buffers, images) = gltf::import(path).context(ImportSnafu {})?;

        let default_scene = document.default_scene().ok_or(Error::NoDefaultScene {})?;

        let mut models = Vec::new();

        for node in default_scene.nodes() {
            let mut transform = match node.transform() {
                Transform::Matrix { matrix } => Mat4::from_cols_array_2d(&matrix),
                Transform::Decomposed {
                    translation,
                    rotation,
                    scale,
                } => Mat4::from_scale_rotation_translation(
                    Vec3::from(scale),
                    Quat::from_array(rotation).normalize(),
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

                    model.meshes.push(Mesh {
                        vertices,
                        uvs,
                        normals,
                        tangents,
                        indices,
                    })
                }
            }

            models.push(model);
        }

        Ok(Self { models })
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
