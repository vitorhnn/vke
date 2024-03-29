use glam::Vec2;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::EventPump;

pub struct InputState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub should_quit: bool,
    pub maximize: bool,
    pub mouse_delta: Vec2,
    pub flip_relative_mouse: bool,
}

impl InputState {
    pub(crate) fn new() -> Self {
        InputState {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            should_quit: false,
            maximize: false,
            mouse_delta: Vec2::new(0.0, 0.0),
            flip_relative_mouse: false,
        }
    }

    pub(crate) fn update(&mut self, event_pump: &mut EventPump) {
        self.mouse_delta = Vec2::new(0.0, 0.0);
        for event in event_pump.poll_iter() {
            match event {
                Event::KeyDown { keycode, .. } => {
                    if let Some(keycode) = keycode {
                        match keycode {
                            Keycode::W => self.forward = true,
                            Keycode::S => self.backward = true,
                            Keycode::D => self.right = true,
                            Keycode::A => self.left = true,
                            Keycode::E => self.up = true,
                            Keycode::Q => self.down = true,
                            Keycode::F11 => self.flip_relative_mouse = true,
                            _ => (),
                        }
                    }
                }
                Event::KeyUp { keycode, .. } => {
                    if let Some(keycode) = keycode {
                        match keycode {
                            Keycode::W => self.forward = false,
                            Keycode::S => self.backward = false,
                            Keycode::D => self.right = false,
                            Keycode::A => self.left = false,
                            Keycode::E => self.up = false,
                            Keycode::Q => self.down = false,
                            _ => (),
                        }
                    }
                }
                Event::MouseMotion { xrel, yrel, .. } => {
                    let delta = Vec2::new(xrel as f32, yrel as f32);

                    self.mouse_delta += delta;
                }
                Event::Window { win_event, .. } => match win_event {
                    WindowEvent::Maximized => self.maximize = true,
                    _ => (),
                },
                Event::Quit { .. } => self.should_quit = true,
                _ => (),
            }
        }
    }
}
