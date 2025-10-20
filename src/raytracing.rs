use crate::allocator::{Allocator, MemoryUsage};
use crate::buffer::Buffer;
use crate::loader::World;
use crate::vk;
use crate::Device;
use gpu_allocator::vulkan::Allocation;
use std::fmt::Debug;
use std::rc::Rc;

pub struct RaytracingSupport {
    device: Rc<Device>,
    as_bufs: Vec<Buffer>,
    as_allocations: Vec<Allocation>,
    blas_addrs: Vec<u64>,
    tlas_addr: u64,
    instance_bufs: Vec<Buffer>,
    pub tlas_handle: vk::AccelerationStructureKHR,
}

fn addressify_const(bda: vk::DeviceAddress) -> vk::DeviceOrHostAddressConstKHR {
    vk::DeviceOrHostAddressConstKHR {
        device_address: bda,
    }
}

fn addressify(bda: vk::DeviceAddress) -> vk::DeviceOrHostAddressKHR {
    vk::DeviceOrHostAddressKHR {
        device_address: bda,
    }
}

fn convert_to_khr(matrix: glam::Mat4) -> vk::TransformMatrixKHR {
    let transposed = matrix.transpose();

    vk::TransformMatrixKHR {
        matrix: [
            transposed.x_axis.x,
            transposed.x_axis.y,
            transposed.x_axis.z,
            transposed.x_axis.w,
            transposed.y_axis.x,
            transposed.y_axis.y,
            transposed.y_axis.z,
            transposed.y_axis.w,
            transposed.z_axis.x,
            transposed.z_axis.y,
            transposed.z_axis.z,
            transposed.z_axis.w,
        ],
    }
}

impl RaytracingSupport {
    fn build_blas(
        device: &Device,
        allocator: &Allocator,
        world: &World,
        cmd_buf: vk::CommandBuffer,
        as_bufs: &mut Vec<Buffer>,
        as_allocations: &mut Vec<Allocation>,
        blas_addrs: &mut Vec<u64>,
        matrices: &mut Vec<vk::TransformMatrixKHR>,
    ) {
        let vtx_size = std::mem::size_of::<crate::loader::PosUvNormalTangentVertex>() as u64;
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
                        .vertex_data(addressify_const(geometry_vertex_address))
                        .index_data(addressify_const(geometry_index_address))
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
                    .primitive_count(mesh.idx_count / 3)
                    .primitive_offset(0)
                    .first_vertex(0)
                    .transform_offset(0)
                    .build();

                geometries.push(geometry);
                geometry_ranges.push(geometry_range);
                counts.push(mesh.idx_count / 3);
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
            matrices.push(convert_to_khr(model.transform));

            let acceleration_device_address_info =
                vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                    .acceleration_structure(acceleration_structure);

            let blas_addr = unsafe {
                device
                    .acceleration_structure
                    .get_acceleration_structure_device_address(&acceleration_device_address_info)
            };

            blas_addrs.push(blas_addr);

            let scratch_buf_create_info = vk::BufferCreateInfo::builder()
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .size(sizes.build_scratch_size)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let (scratch_buf, _scratch_alloc) = allocator
                .create_buffer(&scratch_buf_create_info, MemoryUsage::DeviceOnly)
                .expect("failed to allocate AS build scratch buffer");

            let scratch_buf_addr = unsafe {
                device.bda.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::builder().buffer(scratch_buf.inner),
                )
            };

            let build_geometry_info = build_geometry_info
                .dst_acceleration_structure(acceleration_structure)
                .scratch_data(addressify(scratch_buf_addr));

            unsafe {
                device
                    .inner
                    .begin_command_buffer(
                        cmd_buf,
                        &vk::CommandBufferBeginInfo::builder()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .expect("failed to begin rt as build cmd buf");

                device
                    .acceleration_structure
                    .cmd_build_acceleration_structures(
                        cmd_buf,
                        &[*build_geometry_info],
                        &[&geometry_ranges],
                    );

                device
                    .inner
                    .end_command_buffer(cmd_buf)
                    .expect("failed to end rt as build cmd buf");

                let cmd_bufs = [cmd_buf];
                let submit = vk::SubmitInfo::builder().command_buffers(&cmd_bufs);

                device
                    .inner
                    .queue_submit(device.graphics_queue.inner, &[*submit], vk::Fence::null())
                    .expect("rt as build queue submit failed");
                device
                    .inner
                    .queue_wait_idle(device.graphics_queue.inner)
                    .expect("as queue wait idle failed");
            }
        }
    }

    fn build_tlas(
        device: &Device,
        allocator: &Allocator,
        cmd_buf: vk::CommandBuffer,
        blas_addrs: &Vec<u64>,
        matrices: &Vec<vk::TransformMatrixKHR>,
    ) -> (u64, vk::AccelerationStructureKHR, Vec<Buffer>) {
        let mut geometries = Vec::with_capacity(blas_addrs.len());
        let mut geometry_ranges = Vec::with_capacity(blas_addrs.len());
        let mut bufs = Vec::with_capacity(blas_addrs.len());
        assert_eq!(blas_addrs.len(), matrices.len());
        for (addr, mat) in blas_addrs.iter().zip(matrices) {
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .usage(
                    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .size(std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let (buf, allocation) = allocator
                .create_buffer(&buffer_create_info, MemoryUsage::HostToDevice)
                .expect("failed to allocate tlas instance buf");

            let mem = allocator
                .map(&allocation)
                .expect("failed to map tlas instance buf")
                as *mut vk::AccelerationStructureInstanceKHR;

            dbg!(*addr);
            unsafe {
                *mem = vk::AccelerationStructureInstanceKHR {
                    transform: *mat,
                    instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF),
                    instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                        0, 0,
                    ),
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: *addr,
                    },
                };
            }

            let instance_addr = unsafe {
                device.bda.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::builder().buffer(buf.inner),
                )
            };

            dbg!(instance_addr);

            let geometry_data = vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                    .data(addressify_const(instance_addr))
                    .array_of_pointers(false)
                    .build(),
            };

            let geometry = vk::AccelerationStructureGeometryKHR::builder()
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .geometry(geometry_data)
                .build();

            let geometry_range = vk::AccelerationStructureBuildRangeInfoKHR::builder()
                .primitive_count(1)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0)
                .build();

            geometries.push(geometry);
            geometry_ranges.push(geometry_range);
            bufs.push(buf);
        }

        let build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);

        let counts = vec![1; geometries.len()];

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

        let (buf, _allocation) = allocator
            .create_buffer(&buffer_create_info, MemoryUsage::DeviceOnly)
            .expect("failed to allocate AS buffer");

        let as_create_info = vk::AccelerationStructureCreateInfoKHR::builder()
            .buffer(buf.inner)
            .size(sizes.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        bufs.push(buf);

        let acceleration_structure = unsafe {
            device
                .acceleration_structure
                .create_acceleration_structure(&as_create_info, None)
                .expect("as creation failed")
        };

        let scratch_buf_create_info = vk::BufferCreateInfo::builder()
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .size(sizes.build_scratch_size)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let (scratch_buf, _scratch_alloc) = allocator
            .create_buffer(&scratch_buf_create_info, MemoryUsage::DeviceOnly)
            .expect("failed to allocate AS build scratch buffer");

        let scratch_buf_addr = unsafe {
            device.bda.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::builder().buffer(scratch_buf.inner),
            )
        };

        let build_geometry_info = build_geometry_info
            .dst_acceleration_structure(acceleration_structure)
            .scratch_data(addressify(scratch_buf_addr));

        unsafe {
            device
                .inner
                .begin_command_buffer(
                    cmd_buf,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::empty()),
                )
                .expect("failed to begin rt as build cmd buf");

            device
                .acceleration_structure
                .cmd_build_acceleration_structures(
                    cmd_buf,
                    &[*build_geometry_info],
                    &[&geometry_ranges],
                );

            device
                .inner
                .end_command_buffer(cmd_buf)
                .expect("failed to end rt as build cmd buf");

            let cmd_bufs = [cmd_buf];
            let submit = vk::SubmitInfo::builder().command_buffers(&cmd_bufs);

            device
                .inner
                .queue_submit(device.graphics_queue.inner, &[*submit], vk::Fence::null())
                .expect("rt as build queue submit failed");
            device
                .inner
                .queue_wait_idle(device.graphics_queue.inner)
                .expect("as queue wait idle failed");
        }

        let acceleration_device_address_info =
            vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                .acceleration_structure(acceleration_structure);

        let tlas_addr = unsafe {
            device
                .acceleration_structure
                .get_acceleration_structure_device_address(&acceleration_device_address_info)
        };


        (tlas_addr, acceleration_structure, bufs)
    }

    pub fn new(device: Rc<Device>, allocator: &Allocator, world: &World) -> Self {
        let mut as_bufs = Vec::new();
        let mut as_allocations = Vec::new();
        let mut blas_addrs = Vec::new();
        let mut matrices = Vec::new();
        let cmd_buf = unsafe {
            device
                .inner
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(device.graphics_queue.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("rt cmd buf allocate fail")[0]
        };

        Self::build_blas(
            &device,
            allocator,
            world,
            cmd_buf,
            &mut as_bufs,
            &mut as_allocations,
            &mut blas_addrs,
            &mut matrices,
        );

        unsafe {
            device
                .inner
                .reset_command_buffer(cmd_buf, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
                .unwrap();
        }

        let (tlas_addr, tlas_handle, bufs) =
            Self::build_tlas(&device, allocator, cmd_buf, &blas_addrs, &matrices);


        Self {
            as_bufs,
            as_allocations,
            device,
            blas_addrs,
            tlas_addr,
            tlas_handle,
            instance_bufs: bufs,
        }
    }
}
