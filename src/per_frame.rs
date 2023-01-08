use crate::FRAMES_IN_FLIGHT;
use std::convert::TryInto;
use std::fmt::Debug;
use std::mem::MaybeUninit;

pub struct PerFrame<T> {
    resources: [T; FRAMES_IN_FLIGHT],
}

impl<T: Sized + Debug> PerFrame<T> {
    pub fn new<F: FnMut() -> T>(mut f: F) -> Self {
        let mut resources: [MaybeUninit<T>; FRAMES_IN_FLIGHT] = MaybeUninit::uninit_array();

        for elem in &mut resources {
            elem.write(f());
        }

        let resources = unsafe { MaybeUninit::array_assume_init(resources) };
        Self { resources }
    }

    pub fn new_from_vec(vec: Vec<T>) -> Self {
        Self {
            resources: vec
                .try_into()
                .expect("slice had more elements than FRAMES_IN_FLIGHT"),
        }
    }

    pub fn get_resource_for_frame(&self, frame_idx: usize) -> &T {
        &self.resources[frame_idx]
    }

    pub fn get_mut_resource_for_frame(&mut self, frame_idx: usize) -> &mut T {
        &mut self.resources[frame_idx]
    }
}
