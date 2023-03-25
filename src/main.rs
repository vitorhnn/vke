// rust devs pls stabilize
#![feature(try_find)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(array_zip)]

use ash::{prelude::VkResult, vk};
use passes::marching_cubes_compute::MarchingCubesComputePass;

use std::error::Error;
use std::ffi::CStr;
use std::fmt::Debug;
use std::path::Path;
use std::rc::Rc;

use glam::Mat4;

use sdl2::video::FullscreenType;

mod buffer;
mod surface;
mod swapchain;

mod instance;
use instance::Instance;

mod device;
use device::Device;

mod queue;

mod allocator;
mod asset;
mod fly_camera;
mod input;
mod loader;
mod technique;
mod passes;
mod per_frame;
mod sampler;
mod texture;
mod texture_view;
mod transfer;

use crate::device::ImageBarrierParameters;
use crate::loader::World;
use crate::passes::geometry::GeometryPass;
use crate::texture::Texture;

use transfer::Transfer;

struct FrameResources {
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
}

struct Application {
    desired_extent: vk::Extent2D,
    desired_present_mode: vk::PresentModeKHR,
    sdl_ctx: sdl2::Sdl,
    _sdl_video_ctx: sdl2::VideoSubsystem,
    window: sdl2::video::Window,
    swapchain: swapchain::Swapchain,
    allocator: Rc<allocator::Allocator>,
    device: Rc<Device>,
    surface: surface::Surface,
    geometry_pass: GeometryPass,
    marching_cubes_compute_pass: MarchingCubesComputePass,
    transfer: Transfer,
    frame_resources: Vec<FrameResources>,
    current_frame: usize,
    frame_count: usize,
    world: World,
    instance: Instance,
    input_state: input::InputState,
    fly_camera: fly_camera::FlyCamera,
    relative_mouse: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct PerFrameDataUbo {
    view: Mat4,
    projection: Mat4,
}

const SHADER_MAIN_FN_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

const FRAMES_IN_FLIGHT: usize = 3;

impl Application {
    fn new(
        desired_extent: vk::Extent2D,
        desired_present_mode: vk::PresentModeKHR,
    ) -> Result<Self, Box<dyn Error>> {
        let sdl_ctx = sdl2::init()?;
        let sdl_video_ctx = sdl_ctx.video()?;
        let window = sdl_video_ctx
            .window("vke", desired_extent.width, desired_extent.height)
            .resizable()
            .vulkan()
            .build()?;

        sdl_ctx.mouse().set_relative_mouse_mode(false);

        let instance = Instance::new(&window)?;

        let surface = surface::Surface::new(
            &instance,
            window.vulkan_create_surface(instance.raw_handle() as usize)?,
        )?;

        let device = Rc::new(Device::from_heuristics(
            &instance,
            &surface,
            desired_present_mode,
        )?);

        let allocator = Rc::new(allocator::Allocator::new(device.clone(), &instance));

        let mut transfer = Transfer::new(
            device.clone(),
            allocator.clone(),
            device.graphics_queue.clone(),
            Some(device.transfer_queue.clone()),
        )?;

        let swapchain = swapchain::Swapchain::new(
            &instance,
            &device,
            &surface,
            desired_extent,
            desired_present_mode,
        )?;

        let frame_resources: Vec<FrameResources> = (0..FRAMES_IN_FLIGHT)
            .map::<VkResult<FrameResources>, _>(|_| {
                let semaphore_create_info = vk::SemaphoreCreateInfo::builder();

                let fence_create_info =
                    vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

                let image_available = unsafe {
                    device
                        .inner
                        .create_semaphore(&semaphore_create_info, None)?
                };
                let render_finished = unsafe {
                    device
                        .inner
                        .create_semaphore(&semaphore_create_info, None)?
                };
                let fence = unsafe { device.inner.create_fence(&fence_create_info, None)? };

                let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::builder()
                    .command_pool(device.graphics_queue.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);

                let command_buffers = unsafe {
                    device
                        .inner
                        .allocate_command_buffers(&command_buffer_alloc_info)?
                };

                let command_buffer = command_buffers[0];

                Ok(FrameResources {
                    image_available,
                    render_finished,
                    fence,
                    command_buffer,
                })
            })
            .collect::<Result<_, _>>()?;

        let scene = asset::Scene::from_gltf(Path::new("./cube.glb"))?;
        //let scene = asset::Scene::from_gltf(Path::new("./suzanne/Suzanne.gltf"))?;
        let world = loader::load_scene(allocator.clone(), &mut transfer, scene);

        let geometry_pass = GeometryPass::new(device.clone(), allocator.clone(), desired_extent);
        let marching_cubes_compute_pass = MarchingCubesComputePass::new(device.clone(), allocator.clone());

        println!("VKe: application created");
        println!(
            "graphics queue family index: {:#?}",
            device.graphics_queue.family_index,
        );
        println!(
            "tranfer queue family index: {:#?}",
            device.transfer_queue.family_index,
        );

        let app = Self {
            desired_present_mode,
            desired_extent,
            current_frame: 0,
            sdl_ctx,
            instance,
            _sdl_video_ctx: sdl_video_ctx,
            window,
            device,
            surface,
            swapchain,
            frame_resources,
            frame_count: 0,
            allocator,
            transfer,
            world,
            input_state: input::InputState::new(),
            fly_camera: fly_camera::FlyCamera::new(),
            geometry_pass,
            marching_cubes_compute_pass,
            relative_mouse: false,
        };

        Ok(app)
    }

    fn run(&mut self) -> Result<(), Box<dyn Error>> {
        let mut event_pump = self.sdl_ctx.event_pump()?;

        'running: loop {
            self.input_state.update(&mut event_pump);

            if self.input_state.should_quit {
                break 'running;
            }

            if self.input_state.maximize {
                self.window.set_fullscreen(FullscreenType::True).expect("failed to doobly doo");
                self.input_state.maximize = false;
            }
            
            if self.input_state.flip_relative_mouse {
                self.relative_mouse = !self.relative_mouse;
                self.sdl_ctx.mouse().set_relative_mouse_mode(self.relative_mouse);
                self.input_state.flip_relative_mouse = false;
            }

            self.fly_camera.update(&mut self.input_state);

            self.draw()?;
        }

        Ok(())
    }

    fn update_ubo(&self) -> PerFrameDataUbo {
        let mut ubo = PerFrameDataUbo::default();

        let view = self.fly_camera.get_matrix();
        ubo.view = view;

        let aspect_ratio = self.swapchain.extent.width as f32 / self.swapchain.extent.height as f32;
        ubo.projection = Mat4::perspective_infinite_rh(f32::to_radians(45.0), aspect_ratio, 0.1);

        ubo
    }

    fn draw(&mut self) -> VkResult<()> {
        let frame_resources = &self.frame_resources[self.current_frame];
        let wait_fences = [frame_resources.fence];

        unsafe {
            self.device
                .inner
                .wait_for_fences(&wait_fences, true, u64::MAX)?;
        }

        let acquire_result = unsafe {
            self.swapchain.loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                self.frame_resources[self.current_frame].image_available,
                vk::Fence::null(),
            )
        };

        let (image_index, out_of_date) = match acquire_result {
            Ok((_, true)) => (0, false),
            Ok((image_index, false)) => (image_index, false),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => (0, true),
            Err(e) => return Err(e),
        };

        if out_of_date {
            self.recreate_swapchain().unwrap();
            return Ok(());
        }

        let mut image_resources = &mut self.swapchain.image_resources[image_index as usize];

        if image_resources.fence != vk::Fence::null() {
            let wait_fences = [image_resources.fence];
            unsafe {
                self.device
                    .inner
                    .wait_for_fences(&wait_fences, true, u64::MAX)?
            };
        }

        image_resources.fence = self.frame_resources[self.current_frame].fence;

        let image_resources = &self.swapchain.image_resources[image_index as usize];

        let command_buffer = self.frame_resources[self.current_frame].command_buffer;

        unsafe {
            self.device.inner.reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )?;
            self.device
                .inner
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::builder())?;
        }

        let frame_ubo = self.update_ubo();

        unsafe {
            let (output_ssbo, count_ssbo) = self.marching_cubes_compute_pass.execute(self.current_frame, &command_buffer)?;

            self.geometry_pass
                .execute(self.current_frame, &command_buffer, &self.world, frame_ubo, output_ssbo, count_ssbo)?;

            self.device.insert_image_barrier(&ImageBarrierParameters {
                command_buffer,
                src_stage_mask: vk::PipelineStageFlags::TOP_OF_PIPE,
                dst_stage_mask: vk::PipelineStageFlags::TRANSFER,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                image: image_resources.image,
                subresource_range: vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            });

            let geometry_color_output = self
                .geometry_pass
                .color_target_views
                .get_resource_for_frame(self.current_frame);

            // transition render images to transfer src
            self.device.insert_image_barrier(&ImageBarrierParameters {
                command_buffer,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: vk::PipelineStageFlags::TRANSFER,
                old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                image: geometry_color_output.texture.image,
                subresource_range: vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            });

            let region = vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    layer_count: 1,
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_array_layer: 0,
                    mip_level: 0,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: self.desired_extent.width as i32,
                        y: self.desired_extent.height as i32,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    layer_count: 1,
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_array_layer: 0,
                    mip_level: 0,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: self.desired_extent.width as i32,
                        y: self.desired_extent.height as i32,
                        z: 1,
                    },
                ],
            };

            self.device.inner.cmd_blit_image(
                command_buffer,
                geometry_color_output.texture.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image_resources.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
                vk::Filter::LINEAR,
            );

            self.device.insert_image_barrier(&ImageBarrierParameters {
                command_buffer,
                src_stage_mask: vk::PipelineStageFlags::TRANSFER,
                dst_stage_mask: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::empty(),
                image: image_resources.image,
                subresource_range: vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            });

            self.device.inner.end_command_buffer(command_buffer)?;
        }

        let wait_semaphores = [self.frame_resources[self.current_frame].image_available];
        let signal_semaphores = [self.frame_resources[self.current_frame].render_finished];
        let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [command_buffer];

        let submits = [vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores)
            .build()];

        unsafe {
            self.device.inner.reset_fences(&wait_fences)?;
            self.device.inner.queue_submit(
                self.device.graphics_queue.inner,
                &submits,
                self.frame_resources[self.current_frame].fence,
            )?
        };

        let swapchains = [self.swapchain.swapchain];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let present_result = unsafe {
            self.swapchain
                .loader
                .queue_present(self.device.graphics_queue.inner, &present_info)
        };

        let should_recreate = match present_result {
            Ok(true) => true,
            Ok(false) => false,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Err(e) => return Err(e),
        };

        if should_recreate {
            self.recreate_swapchain().unwrap();
        }

        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;
        self.frame_count += 1;

        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<(), Box<dyn Error>> {
        unsafe { self.device.inner.device_wait_idle()? };

        self.drop_swapchain();

        self.swapchain = swapchain::Swapchain::new(
            &self.instance,
            &self.device,
            &self.surface,
            self.desired_extent,
            self.desired_present_mode,
        )?;

        Ok(())
    }

    fn drop_swapchain(&mut self) {
        unsafe {
            for image_resources in &self.swapchain.image_resources {
                self.device
                    .inner
                    .destroy_image_view(image_resources.image_view, None);
            }

            self.swapchain
                .loader
                .destroy_swapchain(self.swapchain.swapchain, None);
        }
    }
}

impl Drop for Application {
    fn drop(&mut self) {
        unsafe {
            self.device
                .inner
                .device_wait_idle()
                .expect("couldn't wait for device idle");

            self.drop_swapchain();

            for frame_data in &self.frame_resources {
                self.device
                    .inner
                    .destroy_semaphore(frame_data.image_available, None);
                self.device
                    .inner
                    .destroy_semaphore(frame_data.render_finished, None);

                self.device.inner.destroy_fence(frame_data.fence, None);
            }

            self.surface
                .loader
                .destroy_surface(self.surface.surface, None);
        }
    }
}

fn result_msgbox<T, E: Debug>(result: Result<T, E>) -> Result<T, E> {
    match result {
        Ok(thing) => Ok(thing),
        Err(e) => {
            use sdl2::messagebox;

            messagebox::show_simple_message_box(
                messagebox::MessageBoxFlag::ERROR,
                "vke",
                &format!("{:#?}", e),
                None,
            )
            .expect("unable to show an error, congrats");

            Err(e)
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut app = result_msgbox(Application::new(
        vk::Extent2D {
            width: 1920,
            height: 1080,
        },
        vk::PresentModeKHR::FIFO,
    ))?;

    match result_msgbox(app.run()) {
        Ok(_) => Ok(()),
        Err(e) => {
            unsafe {
                app.device.inner.device_wait_idle().unwrap();
            }
            Err(e)
        }
    }
}
