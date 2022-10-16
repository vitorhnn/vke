use crate::allocator::Allocator;
use crate::allocator::MemoryUsage;
use crate::asset::Scene;
use crate::buffer::Buffer;
use crate::texture::Texture;
use crate::texture_view::TextureView;
use crate::{Device, Transfer};
use ash::vk;
use glam::{Mat4, Vec2, Vec3, Vec4};
use gpu_allocator::vulkan::Allocation;
use itertools::{all, izip};
use std::io::Read;
use std::mem::size_of;
use std::rc::Rc;

pub fn load_png(
    device: Rc<Device>,
    source: &mut impl Read,
    allocator: &Allocator,
    transfer: &mut Transfer,
) -> (TextureView, Allocation) {
    let decoder = png::Decoder::new(source);
    let mut reader = decoder.read_info().unwrap();

    let (x, y) = reader.info().size();
    let extent = vk::Extent3D {
        depth: 1,
        width: x,
        height: y,
    };

    let (texture, allocation) = allocator
        .create_texture(
            &vk::ImageCreateInfo::builder()
                .flags(vk::ImageCreateFlags::empty())
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_SRGB)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            MemoryUsage::DeviceOnly,
        )
        .unwrap();

    let image = texture.image.clone();
    let view = TextureView::new(
        device,
        texture,
        &vk::ImageViewCreateInfo::builder()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .components(vk::ComponentMapping::default())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image(image),
    )
    .unwrap();

    transfer
        .upload_image_callback(
            |buf| {
                let read_info = reader.next_frame(buf).unwrap();

                read_info.buffer_size()
            },
            &view.texture,
        )
        .unwrap();

    transfer.flush().unwrap();

    (view, allocation)
}

#[repr(C)]
#[derive(Debug, Clone)]
struct PosUvNormalTangentVertex {
    pos: Vec3,
    uv: Vec2,
    normal: Vec3,
    tangent: Vec4,
}

pub struct Mesh {
    pub buffer: Buffer,
    pub idx_buffer: Buffer,
    pub allocation: Option<Allocation>,
    pub idx_allocation: Option<Allocation>,
    pub allocator: Rc<Allocator>,
    pub idx_count: u32,
}

impl Drop for Mesh {
    // this is kinda jank. should probably back buffer + allocation into a single struct
    // or hell, just suballocate a big vertex buffer and manage that ourselves
    fn drop(&mut self) {
        let allocation = self.allocation.take().unwrap();

        self.allocator.free(allocation);
    }
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub transform: Mat4,
}

// this couples rendering code and data with world management and logic.
// this is fine for the purposes of this project (graphics sandbox for my final project)
// but should be rewritten for any serious attempt at a game engine in the future
pub struct World {
    pub models: Vec<Model>,
}

pub fn load_scene(allocator: Rc<Allocator>, transfer: &mut Transfer, scene: Scene) -> World {
    let mut models = Vec::new();

    for model in scene.models {
        let mut meshes = Vec::new();
        for mesh in model.meshes {
            let buf_size = (size_of::<PosUvNormalTangentVertex>() * mesh.vertices.len()) as u64;

            let (buf, allocation) = allocator
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(buf_size)
                        .usage(
                            vk::BufferUsageFlags::TRANSFER_DST
                                | vk::BufferUsageFlags::VERTEX_BUFFER,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    MemoryUsage::DeviceOnly,
                )
                .unwrap();

            transfer
                .upload_buffer_callback(
                    |buf| {
                        for (pos, uv, normal, tangent, storage) in izip!(
                            &mesh.vertices,
                            &mesh.uvs,
                            &mesh.normals,
                            &mesh.tangents,
                            buf
                        ) {
                            let vtx = PosUvNormalTangentVertex {
                                pos: *pos,
                                uv: *uv,
                                normal: *normal,
                                tangent: *tangent,
                            };

                            *storage = vtx.clone();
                        }

                        buf_size as usize
                    },
                    &buf,
                )
                .unwrap();

            let (idx_buf, idx_allocation) = allocator
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size((size_of::<u16>() * mesh.indices.len()) as u64)
                        .usage(
                            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    MemoryUsage::DeviceOnly,
                )
                .unwrap();

            transfer.upload_buffer(&mesh.indices, &idx_buf).unwrap();

            meshes.push(Mesh {
                allocation: Some(allocation),
                idx_allocation: Some(idx_allocation),
                idx_buffer: idx_buf,
                buffer: buf,
                idx_count: mesh.indices.len() as u32,
                allocator: allocator.clone(),
            })
        }

        models.push(Model {
            transform: model.transform,
            meshes,
        })
    }

    transfer.flush();

    World { models }
}
