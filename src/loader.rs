use crate::allocator::Allocator;
use crate::allocator::MemoryUsage;
use crate::texture::Texture;
use crate::texture_view::TextureView;
use crate::{Device, Transfer};
use ash::vk;
use gpu_allocator::vulkan::Allocation;
use std::io::Read;
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
