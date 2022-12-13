use std::mem::MaybeUninit;
use crate::FRAMES_IN_FLIGHT;

pub struct PerFrame<T> {
    resources: [T; FRAMES_IN_FLIGHT],
}

impl<T: Sized> PerFrame<T> {
    pub fn new<F: FnMut() -> T>(mut f: F) -> Self {
        let mut resources: [MaybeUninit<T>; FRAMES_IN_FLIGHT] = MaybeUninit::uninit_array();

        for elem in &mut resources {
            elem.write(f());
        }

        let resources = unsafe { MaybeUninit::array_assume_init(resources) };
        Self {
            resources
        }
    }

    pub fn get_resource_for_frame(&mut self, frame_idx: usize) -> &mut T {
        &mut self.resources[frame_idx & FRAMES_IN_FLIGHT]
    }
}
