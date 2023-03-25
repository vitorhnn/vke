use crate::input::InputState;
use glam::{Mat4, Vec3};
use std::mem::replace;

pub struct FlyCamera {
    position: Vec3,
    front: Vec3,
    up: Vec3,
    right: Vec3,
    world_up: Vec3,
    yaw: f32,
    pitch: f32,
}

const SPEED_MODIFIER: f32 = 0.1;
const MOUSE_MODIFIER: f32 = 0.15;

impl FlyCamera {
    pub fn new() -> FlyCamera {
        Self {
            position: Vec3::new(0.0, 0.0, 0.0),
            front: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::new(0.0, -1.0, 0.0),
            right: Vec3::new(-1.0, 0.0, 0.0),
            world_up: Vec3::new(0.0, 1.0, 0.0),
            yaw: -90.0,
            pitch: 0.0,
        }
    }

    pub fn get_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.front, self.up)
    }

    fn update_vectors(&mut self) {
        let front = Vec3::new(
            self.yaw.to_radians().cos() * self.pitch.to_radians().cos(),
            self.pitch.to_radians().sin(),
            self.yaw.to_radians().sin() * self.pitch.to_radians().cos(),
        );

        self.front = front.normalize();
        self.right = self.world_up.cross(self.front).normalize();
        self.up = self.right.cross(self.front).normalize();
    }

    pub fn update(&mut self, input_state: &mut InputState) {
        if input_state.forward {
            self.position += self.front * SPEED_MODIFIER;
        }

        if input_state.backward {
            self.position -= self.front * SPEED_MODIFIER;
        }

        if input_state.right {
            self.position += self.right * SPEED_MODIFIER;
        }

        if input_state.left {
            self.position -= self.right * SPEED_MODIFIER;
        }

        if input_state.up {
            self.position += self.world_up * SPEED_MODIFIER;
        }

        if input_state.down {
            self.position -= self.world_up * SPEED_MODIFIER;
        }

        self.yaw -= input_state.mouse_delta.x * MOUSE_MODIFIER;
        self.pitch -= input_state.mouse_delta.y * MOUSE_MODIFIER;

        self.pitch = self.pitch.clamp(-89.0, 89.0);

        self.update_vectors();
    }
}
