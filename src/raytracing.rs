use crate::allocator::{Allocator, MemoryUsage};
use crate::buffer::Buffer;
use crate::loader::World;
use crate::vk;
use crate::Device;
use gpu_allocator::vulkan::Allocation;
use std::rc::Rc;

pub struct RaytracingSupport {
    device: Rc<Device>,
    as_bufs: Vec<Buffer>,
    as_allocations: Vec<Allocation>,
}

fn addressify(bda: vk::DeviceAddress) -> vk::DeviceOrHostAddressConstKHR {
    vk::DeviceOrHostAddressConstKHR {
        device_address: bda,
    }
}

impl RaytracingSupport {
    pub fn new(device: Rc<Device>, allocator: &Allocator, world: &World) -> Self {
        let vtx_size = std::mem::size_of::<crate::loader::PosUvNormalTangentVertex>() as u64;
        let mut as_bufs = Vec::new();
        let mut as_allocations = Vec::new();
        for model in &world.models {
            // Vulkan "please triple sign this" bs
            let mut geometries = Vec::with_capacity(model.meshes.len());
            let mut geometry_ranges = Vec::with_capacity(model.meshes.len());
            let mut counts = Vec::with_capacity(model.meshes.len());

            for mesh in &model.meshes {
                let geometry_vertex_address = unsafe {
                    device.bda.get_buffer_device_address(
                        &vk::BufferDeviceAddressInfo::builder().buffer(mesh.buffer.inner),
                    )
                };

                let geometry_index_address = unsafe {
                    device.bda.get_buffer_device_address(
                        &vk::BufferDeviceAddressInfo::builder().buffer(mesh.idx_buffer.inner),
                    )
                };

                let geometry_data = vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                        .vertex_data(addressify(geometry_vertex_address))
                        .index_data(addressify(geometry_index_address))
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_stride(vtx_size)
                        .max_vertex(
                            ((mesh.allocation.as_ref().unwrap().size() / vtx_size) - 1) as u32,
                        )
                        .index_type(vk::IndexType::UINT16)
                        .build(),
                };
                let geometry = vk::AccelerationStructureGeometryKHR::builder()
                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                    .geometry(geometry_data)
                    .flags(vk::GeometryFlagsKHR::OPAQUE)
                    .build();

                let geometry_range = vk::AccelerationStructureBuildRangeInfoKHR::builder()
                    .primitive_count(mesh.idx_count)
                    .primitive_offset(0)
                    .first_vertex(0)
                    .transform_offset(0)
                    .build();

                geometries.push(geometry);
                geometry_ranges.push(geometry_range);
                counts.push(mesh.idx_count);
            }

            let build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                // All of our geometry is static. TODO: revisit with dynamic geometry
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(&geometries);

            let sizes = unsafe {
                device
                    .acceleration_structure
                    .get_acceleration_structure_build_sizes(
                        vk::AccelerationStructureBuildTypeKHR::DEVICE,
                        &build_geometry_info,
                        &counts,
                    )
            };

            let buffer_create_info = vk::BufferCreateInfo::builder()
                .usage(
                    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .size(sizes.acceleration_structure_size)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let (buf, allocation) = allocator
                .create_buffer(&buffer_create_info, MemoryUsage::DeviceOnly)
                .expect("failed to allocate AS buffer");

            let as_create_info = vk::AccelerationStructureCreateInfoKHR::builder()
                .buffer(buf.inner)
                .size(sizes.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

            let acceleration_structure = unsafe {
                device
                    .acceleration_structure
                    .create_acceleration_structure(&as_create_info, None)
                    .expect("as creation failed")
            };

            as_bufs.push(buf);
            as_allocations.push(allocation);

            let acceleration_device_address_info =
                vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                    .acceleration_structure(acceleration_structure);

            let device_address = unsafe {
                device
                    .acceleration_structure
                    .get_acceleration_structure_device_address(&acceleration_device_address_info);
            };

            let scratch_buf_create_info = vk::BufferCreateInfo::builder()
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .size(sizes.acceleration_structure_size)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let (scratch_buf, scratch_alloc) = allocator
                .create_buffer(&scratch_buf_create_info, MemoryUsage::DeviceOnly)
                .expect("failed to allocate AS build scratch buffer");

            let scratch_buf_addr = unsafe {
                device.bda.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::builder().buffer(scratch_buf.inner),
                )
            };
        }

        Self {
            as_bufs,
            as_allocations,
            device,
        }
    }
}
