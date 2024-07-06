use std::{collections::HashMap, rc::Rc};

use ash::vk;

use crate::{asset::Texture, device::Device, technique::TechniqueCache, GEOMETRY_TECHNIQUE_PATH, TEXTURE_HEAP_SET_INDEX};

#[derive(Debug)]
struct TextureHeapInfo {
    heap_index: u64,
    texture: Texture,
}

#[derive(Debug)]
pub struct TextureManager {
    textures: HashMap<String, TextureHeapInfo>,
    device: Rc<Device>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    current_heap_index: u32,
}

impl TextureManager {
    pub fn new(device: Rc<Device>, technique_cache: &mut TechniqueCache) -> Self {
        let descriptor_pool = {
            let pool_sizes = [vk::DescriptorPoolSize {
                descriptor_count: 512,
                ty: vk::DescriptorType::SAMPLED_IMAGE,
            }];

            let pool_create_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&pool_sizes)
                .max_sets(1);

            unsafe {
                device
                    .inner
                    .create_descriptor_pool(&pool_create_info, None)
                    .expect("texture manager pool allocation failure")
            }
        };

        let technique = technique_cache.get(GEOMETRY_TECHNIQUE_PATH);
        let set_layout = technique.descriptor_set_layouts[TEXTURE_HEAP_SET_INDEX].as_ref().unwrap().as_vk(&device);
        let descriptor_set = unsafe {
            let allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&set_layout));

            device.inner.allocate_descriptor_sets(&allocate_info).unwrap()[0]
        };

        TextureManager {
            device,
            descriptor_pool,
            descriptor_set,
            textures: HashMap::new(),
            current_heap_index: 0,
        }
    }

    pub fn add(&mut self, id: String, texture: Texture) {
        let write = vk::WriteDescriptorSet {
            dst_set: self.descriptor_set,
            dst_binding: 0,
            dst_array_element: self.current_heap_index,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
            p_image_info: std::ptr::null(),
            ..Default::default()
        };
    }
}
