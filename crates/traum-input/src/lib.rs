//! Unified multi-modal input layer (Mouse/KB, Multi-Touch, Gamepad) & spring-damped 3D Camera Rig.

use gilrs::{Axis, Button, EventType, Gilrs};
use glam::{Mat4, Vec2, Vec3};
use std::collections::HashSet;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, Touch, TouchPhase, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveInputDevice {
    MouseKeyboard,
    Touch,
    Gamepad,
}

impl ActiveInputDevice {
    pub fn label(self) -> &'static str {
        match self {
            ActiveInputDevice::MouseKeyboard => "Mouse + KB",
            ActiveInputDevice::Touch => "Touch",
            ActiveInputDevice::Gamepad => "Controller",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct TouchFinger {
    id: u64,
    prev_pos: Vec2,
    pos: Vec2,
    age: f32,
}

/// Spring-smoothed orbital & pan camera rig shared across Mouse, Multi-Touch, and Gamepad.
#[derive(Debug, Clone)]
pub struct CameraRig {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub focus: Vec3,

    pub target_yaw: f32,
    pub target_pitch: f32,
    pub target_distance: f32,
    pub target_focus: Vec3,

    pub fov_y_radians: f32,
}

impl Default for CameraRig {
    fn default() -> Self {
        let focus = Vec3::new(0.0, 0.9, 0.0);
        Self {
            yaw: 0.58,
            pitch: 0.36,
            distance: 13.5,
            focus,
            target_yaw: 0.58,
            target_pitch: 0.36,
            target_distance: 13.5,
            target_focus: focus,
            fov_y_radians: 48.0_f32.to_radians(),
        }
    }
}

impl CameraRig {
    pub fn eye_position(&self) -> Vec3 {
        let cp = self.pitch.cos();
        let dir = Vec3::new(self.yaw.sin() * cp, self.pitch.sin(), self.yaw.cos() * cp);
        self.focus + dir * self.distance
    }

    pub fn forward(&self) -> Vec3 {
        (self.focus - self.eye_position()).normalize_or_zero()
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize_or_zero()
    }

    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize_or_zero()
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye_position(), self.focus, Vec3::Y)
    }

    pub fn proj_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y_radians, aspect.max(0.1), 0.1, 160.0)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj_matrix(aspect) * self.view_matrix()
    }

    /// Computes a world-space ray `(origin, unit_dir)` from pixel coordinates `screen_px`.
    pub fn screen_ray(&self, screen_px: Vec2, viewport: Vec2) -> (Vec3, Vec3) {
        let nx = (screen_px.x / viewport.x.max(1.0)) * 2.0 - 1.0;
        let ny = 1.0 - (screen_px.y / viewport.y.max(1.0)) * 2.0;
        let aspect = viewport.x.max(1.0) / viewport.y.max(1.0);
        let half_h = (self.fov_y_radians * 0.5).tan();
        let half_w = half_h * aspect;

        let dir = (self.forward() + self.right() * (nx * half_w) + self.up() * (ny * half_h))
            .normalize_or_zero();
        (self.eye_position(), dir)
    }

    /// Projects a 3D world point to screen pixel coordinates `(px_x, px_y)`.
    pub fn world_to_screen(&self, world_pos: Vec3, viewport: Vec2) -> Option<Vec2> {
        let aspect = viewport.x.max(1.0) / viewport.y.max(1.0);
        let clip = self.view_proj(aspect) * world_pos.extend(1.0);
        if clip.w <= 0.05 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(Vec2::new(
            (ndc.x * 0.5 + 0.5) * viewport.x,
            (0.5 - ndc.y * 0.5) * viewport.y,
        ))
    }

    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.target_yaw -= delta_yaw;
        self.target_pitch = (self.target_pitch + delta_pitch).clamp(-1.15, 1.42);
    }

    pub fn pan(&mut self, screen_delta: Vec2) {
        let scale = self.target_distance * 0.0016;
        let right = self.right();
        let up = self.up();
        self.target_focus += (-right * screen_delta.x + up * screen_delta.y) * scale;
    }

    pub fn zoom(&mut self, factor: f32) {
        self.target_distance = (self.target_distance * factor).clamp(1.8, 34.0);
    }

    pub fn update(&mut self, dt: f32) {
        let alpha = 1.0 - (-16.0 * dt).exp();
        self.yaw += (self.target_yaw - self.yaw) * alpha;
        self.pitch += (self.target_pitch - self.pitch) * alpha;
        self.distance += (self.target_distance - self.distance) * alpha;
        self.focus = self.focus.lerp(self.target_focus, alpha);
    }
}

/// Hardware-agnostic frame actions produced by `InputHub` for `traum-editor`.
#[derive(Debug, Clone, Default)]
pub struct FrameInput {
    pub imp_px: Vec2,
    pub viewport: Vec2,
    /// Display DPI scale factor (`1.0` = standard 96dpi, `2.0` = 200% HiDPI). `0.0` defaults to `1.0`.
    pub ui_scale: f32,
    pub primary_down: bool,
    pub primary_just_pressed: bool,
    pub primary_just_released: bool,
    /// Analog brush/trigger pressure (`0.0..1.0`, defaults to `1.0` for mouse).
    pub pressure: f32,
    pub carve_modifier: bool,
    /// True if a 1-finger touch transitioned into a 2-finger pinch/orbit within 140ms.
    pub cancel_active_stroke: bool,

    pub radial_open: bool,
    pub radial_just_released: bool,
    pub radial_stick_dir: Option<Vec2>,

    pub brush_size_delta: f32,
    pub brush_blend_delta: f32,

    pub undo_pressed: bool,
    pub redo_pressed: bool,
    pub toggle_mirror: bool,
    pub cycle_shape: bool,
    pub cycle_fleck: bool,
    pub toggle_play: bool,
    pub spawn_prop: bool,
    pub toggle_scope: bool,
    pub clone_prop: bool,
    pub make_unique: bool,
    pub spawn_child_part: bool,
    pub delete_selected: bool,
    pub toggle_record: bool,
    pub save_requested: bool,
    pub reload_requested: bool,
    pub mode_hotkey: Option<usize>,

    /// Normalized 2D character movement (`x` = strafe, `y` = forward) for Play Mode.
    pub move_axis: Vec2,
    pub jump_just_pressed: bool,
}

/// Unified input manager aggregating `winit` Mouse/Keyboard, Multi-Touch, and `gilrs` Gamepad.
pub struct InputHub {
    pub active_device: ActiveInputDevice,
    viewport: Vec2,
    scale_factor: f32,
    imp_px: Vec2,

    // Mouse & Keyboard state
    keys_down: HashSet<KeyCode>,
    keys_just_pressed: HashSet<KeyCode>,
    keys_just_released: HashSet<KeyCode>,
    mouse_left_down: bool,
    mouse_left_just_pressed: bool,
    mouse_left_just_released: bool,
    mouse_right_down: bool,
    mouse_middle_down: bool,
    last_mouse_pos: Vec2,
    scroll_delta_y: f32,

    // Multi-Touch state
    touches: Vec<TouchFinger>,
    max_simultaneous_touches: usize,
    touch_gesture_duration: f32,
    touch_was_single_sculpting: bool,
    cancel_stroke_flag: bool,
    touch_undo_flag: bool,
    touch_redo_flag: bool,
    touch_just_pressed: bool,
    touch_just_released: bool,
    play_virtual_stick_id: Option<u64>,
    play_virtual_stick_origin: Vec2,
    play_touch_jump: bool,

    // Gamepad state
    gilrs: Option<Gilrs>,
    gp_primary_down: bool,
    gp_primary_prev: bool,
    gp_radial_prev: bool,
    gp_undo: bool,
    gp_redo: bool,
    gp_cycle_shape: bool,
    gp_cycle_fleck: bool,
    gp_toggle_mirror: bool,
    gp_toggle_play: bool,
    gp_toggle_scope: bool,
    gp_clone_prop: bool,
    gp_jump: bool,
}

impl InputHub {
    pub fn new(initial_size: PhysicalSize<u32>) -> Self {
        let viewport = Vec2::new(initial_size.width as f32, initial_size.height as f32);
        let gilrs = match Gilrs::new() {
            Ok(g) => Some(g),
            Err(err) => {
                log::warn!("Gamepad subsystem unavailable ({err}); Mouse/KB & Touch remain active.");
                None
            }
        };

        Self {
            active_device: ActiveInputDevice::MouseKeyboard,
            viewport,
            scale_factor: 1.0,
            imp_px: viewport * 0.5,
            keys_down: HashSet::new(),
            keys_just_pressed: HashSet::new(),
            keys_just_released: HashSet::new(),
            mouse_left_down: false,
            mouse_left_just_pressed: false,
            mouse_left_just_released: false,
            mouse_right_down: false,
            mouse_middle_down: false,
            last_mouse_pos: viewport * 0.5,
            scroll_delta_y: 0.0,
            touches: Vec::new(),
            max_simultaneous_touches: 0,
            touch_gesture_duration: 0.0,
            touch_was_single_sculpting: false,
            cancel_stroke_flag: false,
            touch_undo_flag: false,
            touch_redo_flag: false,
            touch_just_pressed: false,
            touch_just_released: false,
            play_virtual_stick_id: None,
            play_virtual_stick_origin: Vec2::ZERO,
            play_touch_jump: false,
            gilrs,
            gp_primary_down: false,
            gp_primary_prev: false,
            gp_radial_prev: false,
            gp_undo: false,
            gp_redo: false,
            gp_cycle_shape: false,
            gp_cycle_fleck: false,
            gp_toggle_mirror: false,
            gp_toggle_play: false,
            gp_toggle_scope: false,
            gp_clone_prop: false,
            gp_jump: false,
        }
    }

    pub fn set_viewport(&mut self, size: PhysicalSize<u32>) {
        self.viewport = Vec2::new(size.width.max(1) as f32, size.height.max(1) as f32);
        self.imp_px = self.imp_px.clamp(Vec2::ZERO, self.viewport);
    }

    pub fn set_scale_factor(&mut self, factor: f32) {
        self.scale_factor = factor.clamp(0.75, 3.5);
    }

    pub fn handle_window_event(
        &mut self,
        event: &WindowEvent,
        camera: &mut CameraRig,
        is_play_mode: bool,
    ) {
        match event {
            WindowEvent::Resized(size) => {
                self.set_viewport(*size);
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.set_scale_factor(*scale_factor as f32);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.active_device = ActiveInputDevice::MouseKeyboard;
                let pos = Vec2::new(position.x as f32, position.y as f32);
                let delta = pos - self.last_mouse_pos;
                self.last_mouse_pos = pos;
                self.imp_px = pos;

                let shift = self.is_shift_down();
                if self.mouse_right_down {
                    if shift {
                        camera.pan(delta);
                    } else {
                        camera.orbit(delta.x * 0.0055, delta.y * 0.0055);
                    }
                } else if self.mouse_middle_down {
                    camera.pan(delta);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.active_device = ActiveInputDevice::MouseKeyboard;
                let pressed = *state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        if pressed && !self.mouse_left_down {
                            self.mouse_left_just_pressed = true;
                        } else if !pressed && self.mouse_left_down {
                            self.mouse_left_just_released = true;
                        }
                        self.mouse_left_down = pressed;
                    }
                    MouseButton::Right => {
                        self.mouse_right_down = pressed;
                    }
                    MouseButton::Middle => {
                        self.mouse_middle_down = pressed;
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.active_device = ActiveInputDevice::MouseKeyboard;
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => (p.y as f32) * 0.02,
                };
                self.scroll_delta_y += lines;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if self.keys_down.insert(code) {
                                self.keys_just_pressed.insert(code);
                            }
                        }
                        ElementState::Released => {
                            self.keys_down.remove(&code);
                            self.keys_just_released.insert(code);
                        }
                    }
                }
            }
            WindowEvent::Touch(touch) => {
                self.handle_touch(*touch, camera, is_play_mode);
            }
            _ => {}
        }
    }

    fn handle_touch(&mut self, touch: Touch, camera: &mut CameraRig, is_play_mode: bool) {
        self.active_device = ActiveInputDevice::Touch;
        let pos = Vec2::new(touch.location.x as f32, touch.location.y as f32);

        match touch.phase {
            TouchPhase::Started => {
                if self.touches.is_empty() {
                    self.max_simultaneous_touches = 0;
                    self.touch_gesture_duration = 0.0;
                }
                self.touches.push(TouchFinger {
                    id: touch.id,
                    prev_pos: pos,
                    pos,
                    age: 0.0,
                });
                self.max_simultaneous_touches =
                    self.max_simultaneous_touches.max(self.touches.len());

                if is_play_mode {
                    if pos.x < self.viewport.x * 0.42 && self.play_virtual_stick_id.is_none() {
                        self.play_virtual_stick_id = Some(touch.id);
                        self.play_virtual_stick_origin = pos;
                    } else if pos.x >= self.viewport.x * 0.42 && pos.y > self.viewport.y * 0.55 {
                        self.play_touch_jump = true;
                    }
                } else if self.touches.len() == 1 {
                    self.imp_px = pos;
                    self.touch_just_pressed = true;
                    self.touch_was_single_sculpting = true;
                } else if self.touches.len() == 2 && self.touch_was_single_sculpting {
                    if self.touch_gesture_duration < 0.14 {
                        self.cancel_stroke_flag = true;
                    }
                    self.touch_was_single_sculpting = false;
                    self.touch_just_released = true;
                }
            }
            TouchPhase::Moved => {
                if let Some(f) = self.touches.iter_mut().find(|t| t.id == touch.id) {
                    f.prev_pos = f.pos;
                    f.pos = pos;
                }

                if is_play_mode {
                    if Some(touch.id) != self.play_virtual_stick_id {
                        if let Some(f) = self.touches.iter().find(|t| t.id == touch.id) {
                            let d = f.pos - f.prev_pos;
                            camera.orbit(d.x * 0.006, d.y * 0.006);
                        }
                    }
                } else if self.touches.len() == 1 {
                    self.imp_px = pos;
                } else if self.touches.len() == 2 {
                    let a = self.touches[0];
                    let b = self.touches[1];
                    let prev_mid = (a.prev_pos + b.prev_pos) * 0.5;
                    let curr_mid = (a.pos + b.pos) * 0.5;
                    let mid_delta = curr_mid - prev_mid;

                    let prev_dist = a.prev_pos.distance(b.prev_pos).max(8.0);
                    let curr_dist = a.pos.distance(b.pos).max(8.0);
                    let zoom_ratio = prev_dist / curr_dist;

                    camera.orbit(mid_delta.x * 0.0032, mid_delta.y * 0.0032);
                    camera.zoom(zoom_ratio.powf(0.65));
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                if Some(touch.id) == self.play_virtual_stick_id {
                    self.play_virtual_stick_id = None;
                }
                self.touches.retain(|t| t.id != touch.id);
                if self.touches.is_empty() {
                    if self.touch_was_single_sculpting {
                        self.touch_just_released = true;
                        self.touch_was_single_sculpting = false;
                    } else if self.touch_gesture_duration < 0.24 {
                        if self.max_simultaneous_touches == 2 {
                            self.touch_undo_flag = true;
                        } else if self.max_simultaneous_touches >= 3 {
                            self.touch_redo_flag = true;
                        }
                    }
                }
            }
        }
    }

    /// Advances input state by `dt`, updates camera from keyboard/gamepad, and returns unified `FrameInput`.
    pub fn begin_frame(
        &mut self,
        dt: f32,
        camera: &mut CameraRig,
        is_play_mode: bool,
    ) -> FrameInput {
        self.touch_gesture_duration += dt;
        for t in &mut self.touches {
            t.age += dt;
        }

        let mut gp_radial_held = false;
        let mut gp_radial_dir = None;
        let mut gp_pressure = 0.0_f32;
        let mut gp_carve = false;
        let mut gp_size_delta = 0.0_f32;
        let mut gp_move = Vec2::ZERO;

        if let Some(gilrs) = &mut self.gilrs {
            while let Some(ev) = gilrs.next_event() {
                self.active_device = ActiveInputDevice::Gamepad;
                if let EventType::ButtonPressed(btn, _) = ev.event {
                    match btn {
                        Button::DPadLeft => self.gp_undo = true,
                        Button::DPadRight => self.gp_redo = true,
                        Button::North => self.gp_cycle_shape = true,
                        Button::East => self.gp_cycle_fleck = true,
                        Button::West => self.gp_toggle_mirror = true,
                        Button::RightTrigger => self.gp_toggle_scope = true, // R1 / RB = Scope In / Out!
                        Button::Select => self.gp_clone_prop = true,         // Select = Live Clone!
                        Button::Start => self.gp_toggle_play = true,
                        Button::South if is_play_mode => self.gp_jump = true,
                        _ => {}
                    }
                }
            }

            for (_id, gp) in gilrs.gamepads() {
                if !gp.is_connected() {
                    continue;
                }
                let lx = deadzone(gp.value(Axis::LeftStickX), 0.14);
                let ly = deadzone(gp.value(Axis::LeftStickY), 0.14);
                let rx = deadzone(gp.value(Axis::RightStickX), 0.14);
                let ry = deadzone(gp.value(Axis::RightStickY), 0.14);

                let r2 = gp
                    .button_data(Button::RightTrigger2)
                    .map(|b| b.value())
                    .unwrap_or(0.0);
                let l2 = gp
                    .button_data(Button::LeftTrigger2)
                    .map(|b| b.value())
                    .unwrap_or(0.0);
                let south = gp.is_pressed(Button::South);
                let l1 = gp.is_pressed(Button::LeftTrigger);

                if lx.abs() > 0.01
                    || ly.abs() > 0.01
                    || rx.abs() > 0.01
                    || ry.abs() > 0.01
                    || r2 > 0.05
                    || l2 > 0.05
                    || south
                    || l1
                {
                    self.active_device = ActiveInputDevice::Gamepad;
                }

                if is_play_mode {
                    gp_move = Vec2::new(lx, ly);
                    camera.orbit(rx * dt * 2.4, -ry * dt * 1.8);
                } else {
                    gp_radial_held = l1;
                    if l1 {
                        let stick = Vec2::new(rx, -ry);
                        if stick.length() > 0.32 {
                            gp_radial_dir = Some(stick.normalize());
                        }
                    } else {
                        let imp_speed = 680.0 * dt;
                        self.imp_px.x =
                            (self.imp_px.x + rx * imp_speed).clamp(0.0, self.viewport.x);
                        self.imp_px.y =
                            (self.imp_px.y - ry * imp_speed).clamp(0.0, self.viewport.y);
                        camera.orbit(lx * dt * 1.9, -ly * dt * 1.4);
                    }

                    gp_pressure = r2.max(if south { 1.0 } else { 0.0 });
                    gp_carve = l2 > 0.35;
                    if gp.is_pressed(Button::DPadUp) {
                        gp_size_delta += dt * 1.4;
                    }
                    if gp.is_pressed(Button::DPadDown) {
                        gp_size_delta -= dt * 1.4;
                    }
                }
                break;
            }
        }

        self.gp_primary_down = gp_pressure > 0.08;
        let gp_just_pressed = self.gp_primary_down && !self.gp_primary_prev;
        let gp_just_released = !self.gp_primary_down && self.gp_primary_prev;
        self.gp_primary_prev = self.gp_primary_down;

        let gp_radial_just_released = !gp_radial_held && self.gp_radial_prev;
        self.gp_radial_prev = gp_radial_held;

        let ctrl = self.is_ctrl_down();
        let shift = self.is_shift_down();
        let alt = self.is_alt_down();

        let mut kb_move = Vec2::ZERO;
        if !ctrl {
            if self.keys_down.contains(&KeyCode::KeyW) || self.keys_down.contains(&KeyCode::ArrowUp)
            {
                kb_move.y += 1.0;
            }
            if self.keys_down.contains(&KeyCode::KeyS)
                || self.keys_down.contains(&KeyCode::ArrowDown)
            {
                kb_move.y -= 1.0;
            }
            if self.keys_down.contains(&KeyCode::KeyD)
                || self.keys_down.contains(&KeyCode::ArrowRight)
            {
                kb_move.x += 1.0;
            }
            if self.keys_down.contains(&KeyCode::KeyA)
                || self.keys_down.contains(&KeyCode::ArrowLeft)
            {
                kb_move.x -= 1.0;
            }
        }

        if !is_play_mode && kb_move.length_squared() > 0.0 {
            let flat_fwd =
                Vec3::new(camera.forward().x, 0.0, camera.forward().z).normalize_or_zero();
            let flat_right = camera.right();
            let speed = camera.target_distance * 0.85 * dt;
            camera.target_focus += (flat_fwd * kb_move.y + flat_right * kb_move.x) * speed;
        }
        if !is_play_mode {
            if self.keys_down.contains(&KeyCode::KeyE) {
                camera.target_focus.y += camera.target_distance * 0.6 * dt;
            }
            if self.keys_down.contains(&KeyCode::KeyQ) {
                camera.target_focus.y -= camera.target_distance * 0.6 * dt;
            }
        }

        let mut brush_size_delta = gp_size_delta;
        let mut brush_blend_delta = 0.0_f32;
        if self.scroll_delta_y.abs() > 0.001 {
            if shift {
                brush_size_delta += self.scroll_delta_y * 0.12;
            } else if ctrl {
                brush_blend_delta += self.scroll_delta_y * 0.06;
            } else {
                camera.zoom((0.89_f32).powf(self.scroll_delta_y));
            }
        }
        if self.keys_down.contains(&KeyCode::BracketRight) {
            brush_size_delta += dt * 1.5;
        }
        if self.keys_down.contains(&KeyCode::BracketLeft) {
            brush_size_delta -= dt * 1.5;
        }

        camera.update(dt);

        let mut touch_move = Vec2::ZERO;
        if let Some(stick_id) = self.play_virtual_stick_id {
            if let Some(f) = self.touches.iter().find(|t| t.id == stick_id) {
                let offset = f.pos - self.play_virtual_stick_origin;
                let max_r = 68.0;
                touch_move = Vec2::new(offset.x / max_r, -offset.y / max_r).clamp_length_max(1.0);
            }
        }

        let move_axis = (kb_move + gp_move + touch_move).clamp_length_max(1.0);

        let kb_radial = !is_play_mode && self.keys_down.contains(&KeyCode::Space);
        let kb_radial_released = !is_play_mode && self.keys_just_released.contains(&KeyCode::Space);

        let touch_primary_down =
            !is_play_mode && self.touches.len() == 1 && self.touch_was_single_sculpting;

        let primary_down = self.mouse_left_down || touch_primary_down || self.gp_primary_down;
        let primary_just_pressed =
            self.mouse_left_just_pressed || self.touch_just_pressed || gp_just_pressed;
        let primary_just_released =
            self.mouse_left_just_released || self.touch_just_released || gp_just_released;

        let pressure = if self.gp_primary_down {
            gp_pressure.clamp(0.15, 1.0)
        } else {
            1.0
        };

        let undo_pressed = (ctrl && self.keys_just_pressed.contains(&KeyCode::KeyZ))
            || (!ctrl && self.keys_just_pressed.contains(&KeyCode::KeyZ))
            || self.touch_undo_flag
            || self.gp_undo;
        let redo_pressed = (ctrl && self.keys_just_pressed.contains(&KeyCode::KeyY))
            || self.touch_redo_flag
            || self.gp_redo;

        let mode_hotkey = if self.keys_just_pressed.contains(&KeyCode::Digit1) {
            Some(0)
        } else if self.keys_just_pressed.contains(&KeyCode::Digit2) {
            Some(1)
        } else if self.keys_just_pressed.contains(&KeyCode::Digit3) {
            Some(2)
        } else if self.keys_just_pressed.contains(&KeyCode::Digit4) {
            Some(3)
        } else {
            None
        };

        let out = FrameInput {
            imp_px: self.imp_px,
            viewport: self.viewport,
            ui_scale: self.scale_factor,
            primary_down,
            primary_just_pressed,
            primary_just_released,
            pressure,
            carve_modifier: alt || gp_carve,
            cancel_active_stroke: self.cancel_stroke_flag,
            radial_open: kb_radial || gp_radial_held,
            radial_just_released: kb_radial_released || gp_radial_just_released,
            radial_stick_dir: gp_radial_dir,
            brush_size_delta,
            brush_blend_delta,
            undo_pressed,
            redo_pressed,
            toggle_mirror: self.keys_just_pressed.contains(&KeyCode::KeyM) || self.gp_toggle_mirror,
            cycle_shape: self.keys_just_pressed.contains(&KeyCode::KeyF) || self.gp_cycle_shape,
            cycle_fleck: self.keys_just_pressed.contains(&KeyCode::KeyG) || self.gp_cycle_fleck,
            toggle_play: self.keys_just_pressed.contains(&KeyCode::Tab) || self.gp_toggle_play,
            spawn_prop: self.keys_just_pressed.contains(&KeyCode::KeyP),
            toggle_scope: self.keys_just_pressed.contains(&KeyCode::Enter)
                || self.keys_just_pressed.contains(&KeyCode::KeyV)
                || self.gp_toggle_scope,
            clone_prop: (!ctrl && self.keys_just_pressed.contains(&KeyCode::KeyC))
                || (ctrl && self.keys_just_pressed.contains(&KeyCode::KeyD))
                || self.gp_clone_prop,
            make_unique: self.keys_just_pressed.contains(&KeyCode::KeyU),
            spawn_child_part: self.keys_just_pressed.contains(&KeyCode::KeyL),
            delete_selected: self.keys_just_pressed.contains(&KeyCode::Delete)
                || self.keys_just_pressed.contains(&KeyCode::Backspace),
            toggle_record: self.keys_just_pressed.contains(&KeyCode::KeyR),
            save_requested: (ctrl && self.keys_just_pressed.contains(&KeyCode::KeyS))
                || self.keys_just_pressed.contains(&KeyCode::F5),
            reload_requested: self.keys_just_pressed.contains(&KeyCode::F9),
            mode_hotkey,
            move_axis,
            jump_just_pressed: (is_play_mode && self.keys_just_pressed.contains(&KeyCode::Space))
                || self.play_touch_jump
                || self.gp_jump,
        };

        self.keys_just_pressed.clear();
        self.keys_just_released.clear();
        self.mouse_left_just_pressed = false;
        self.mouse_left_just_released = false;
        self.scroll_delta_y = 0.0;
        self.cancel_stroke_flag = false;
        self.touch_undo_flag = false;
        self.touch_redo_flag = false;
        self.touch_just_pressed = false;
        self.touch_just_released = false;
        self.play_touch_jump = false;
        self.gp_undo = false;
        self.gp_redo = false;
        self.gp_cycle_shape = false;
        self.gp_cycle_fleck = false;
        self.gp_toggle_mirror = false;
        self.gp_toggle_play = false;
        self.gp_toggle_scope = false;
        self.gp_clone_prop = false;
        self.gp_jump = false;

        out
    }

    fn is_ctrl_down(&self) -> bool {
        self.keys_down.contains(&KeyCode::ControlLeft)
            || self.keys_down.contains(&KeyCode::ControlRight)
    }

    fn is_shift_down(&self) -> bool {
        self.keys_down.contains(&KeyCode::ShiftLeft)
            || self.keys_down.contains(&KeyCode::ShiftRight)
    }

    fn is_alt_down(&self) -> bool {
        self.keys_down.contains(&KeyCode::AltLeft) || self.keys_down.contains(&KeyCode::AltRight)
    }
}

fn deadzone(val: f32, dz: f32) -> f32 {
    if val.abs() < dz {
        0.0
    } else {
        (val.abs() - dz) / (1.0 - dz) * val.signum()
    }
}
