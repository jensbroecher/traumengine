//! Tactile sculpting tools, Surface-Magnet 3D Imp, Bouncy Radial UI, Action Recorder, and Play Mode puppet.

use glam::{Quat, Vec2, Vec3};
use traum_core::{
    srgb_to_linear, ActionClip, EditorMode, FleckStyle, KeyframeSample, MotionGadget, SceneProp,
    SculptEdit, SculptOp, SculptShape, TraumScene, PALETTE_SRGB,
};
use traum_input::{ActiveInputDevice, CameraRig, FrameInput};
use traum_sdf::{
    eval_world_only_distance, raycast_scene, resolve_sphere_sdf_collision, BrickDirtyTracker,
    VOLUME_MAX, VOLUME_MIN,
};

pub const DEFAULT_SCENE_FILE: &str = "island_dream.ron";

/// Brush settings controlled by tactile sliders, radial slices, and analog pressure.
#[derive(Debug, Clone)]
pub struct BrushSettings {
    pub mode: EditorMode,
    pub op: SculptOp,
    pub shape: SculptShape,
    pub fleck_style: FleckStyle,
    pub palette_index: usize,
    pub size: f32,
    pub blend: f32,
    pub looseness: f32,
    pub mirror_x: bool,
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            mode: EditorMode::Sculpt,
            op: SculptOp::Add,
            shape: SculptShape::Sphere,
            fleck_style: FleckStyle::ClayImpasto,
            palette_index: 0,
            size: 0.85,
            blend: 0.35,
            looseness: 0.28,
            mirror_x: false,
        }
    }
}

impl BrushSettings {
    pub fn linear_color(&self) -> [f32; 3] {
        srgb_to_linear(PALETTE_SRGB[self.palette_index % PALETTE_SRGB.len()])
    }

    pub fn srgb_color(&self) -> [f32; 3] {
        PALETTE_SRGB[self.palette_index % PALETTE_SRGB.len()]
    }
}

/// Surface-Magnet 3D Imp state (visualized as a charming glowing 3D cursor + ground drop-shadow).
#[derive(Debug, Clone)]
pub struct ImpCursor {
    pub world_pos: Vec3,
    pub surface_normal: Vec3,
    pub ground_shadow_pos: Vec3,
    pub on_surface: bool,
    pub hovered_prop_id: u32,
    last_hit_distance: f32,
}

impl Default for ImpCursor {
    fn default() -> Self {
        Self {
            world_pos: Vec3::new(0.0, 0.5, 0.0),
            surface_normal: Vec3::Y,
            ground_shadow_pos: Vec3::ZERO,
            on_surface: true,
            hovered_prop_id: 0,
            last_hit_distance: 12.0,
        }
    }
}

/// Playable bouncy character buddy for instant `Edit <-> Play` exploration!
#[derive(Debug, Clone)]
pub struct PlayablePuppet {
    pub position: Vec3,
    pub velocity: Vec3,
    pub facing_yaw: f32,
    pub grounded: bool,
    pub walk_cycle: f32,
    pub squash: f32,
}

impl Default for PlayablePuppet {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 2.2, 1.5),
            velocity: Vec3::ZERO,
            facing_yaw: 0.0,
            grounded: false,
            walk_cycle: 0.0,
            squash: 1.0,
        }
    }
}

/// Reversible actions stored in the non-destructive Undo / Redo stack.
#[derive(Debug, Clone)]
enum UndoEntry {
    SculptStroke { added_edits: Vec<SculptEdit> },
    SpawnProp { prop: SceneProp },
    ModifyProp { before: SceneProp, after: SceneProp },
    ResetScene { previous: Box<TraumScene> },
}

/// Interactive 2D Toy UI elements rendered by `traum-render` (zero external UI crate friction!).
#[derive(Debug, Clone)]
pub struct UiRect {
    pub min: Vec2,
    pub max: Vec2,
    pub fill_srgb: [f32; 4],
    pub border_srgb: [f32; 4],
    pub corner_radius: f32,
    pub label: String,
    pub text_srgb: [f32; 4],
    pub action: Option<UiAction>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiAction {
    SetMode(EditorMode),
    SetOp(SculptOp),
    CycleShape,
    CycleFleck,
    SelectColor(usize),
    ToggleMirror,
    Undo,
    Redo,
    SpawnProp,
    CyclePropGadget,
    ToggleRecord,
    ClearPropClip,
    SaveScene,
    ResetIsland,
    OpenRadialButton,
    DragSliderSize,
    DragSliderBlend,
    DragSliderLoose,
    DragSliderDof,
}

#[derive(Debug, Clone)]
pub struct RadialMenuVisual {
    pub center_px: Vec2,
    pub outer_radius: f32,
    pub hovered_slice: Option<usize>,
    pub labels: [&'static str; 8],
}

/// Frame-by-frame UI snapshot consumed by the GPU renderer.
#[derive(Debug, Clone, Default)]
pub struct UiFrame {
    pub rects: Vec<UiRect>,
    pub radial: Option<RadialMenuVisual>,
    pub banner_status: String,
}

/// Main stateful Traum Engine editor, tool controller, animator, and play-mode simulator.
pub struct TraumEditor {
    pub scene: TraumScene,
    pub brush: BrushSettings,
    pub imp: ImpCursor,
    pub puppet: PlayablePuppet,
    pub dirty_bricks: BrickDirtyTracker,
    pub edits_buffer_dirty: bool,
    pub time: f32,

    // Active stroke state
    stroke_active: bool,
    last_stamp_pos: Vec3,
    stroke_edits_count: usize,
    stroke_locked_depth: Option<f32>,

    // Animate / Prop state
    pub selected_prop_id: Option<u32>,
    pub record_armed: bool,
    recording_active: bool,
    recording_start_time: f32,
    recording_samples: Vec<KeyframeSample>,
    dragging_prop_before: Option<SceneProp>,
    dragging_prop_plane_dist: f32,
    dragging_prop_grab_offset: Vec3,

    // UI interaction state
    active_slider: Option<UiAction>,
    radial_button_open: bool,
    radial_anchor_px: Vec2,
    radial_hovered: Option<usize>,
    status_toast: String,
    status_timer: f32,

    // Undo / Redo stacks
    undo_stack: Vec<UndoEntry>,
    redo_stack: Vec<UndoEntry>,
}

impl TraumEditor {
    pub fn new() -> Self {
        let scene = TraumScene::starter_diorama_island();
        Self {
            scene,
            brush: BrushSettings::default(),
            imp: ImpCursor::default(),
            puppet: PlayablePuppet::default(),
            dirty_bricks: BrickDirtyTracker::new(),
            edits_buffer_dirty: true,
            time: 0.0,
            stroke_active: false,
            last_stamp_pos: Vec3::ZERO,
            stroke_edits_count: 0,
            stroke_locked_depth: None,
            selected_prop_id: Some(1),
            record_armed: false,
            recording_active: false,
            recording_start_time: 0.0,
            recording_samples: Vec::new(),
            dragging_prop_before: None,
            dragging_prop_plane_dist: 10.0,
            dragging_prop_grab_offset: Vec3::ZERO,
            active_slider: None,
            radial_button_open: false,
            radial_anchor_px: Vec2::new(400.0, 300.0),
            radial_hovered: None,
            status_toast: "Welcome to Traum Engine! Drag to sculpt clay, Right-Drag to orbit, Hold Space for Radial Menu.".into(),
            status_timer: 6.5,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }

    pub fn set_toast(&mut self, msg: impl Into<String>) {
        self.status_toast = msg.into();
        self.status_timer = 3.6;
    }

    pub fn update(
        &mut self,
        dt: f32,
        input: &FrameInput,
        device_kind: ActiveInputDevice,
        camera: &mut CameraRig,
    ) -> UiFrame {
        self.time += dt;
        if self.status_timer > 0.0 {
            self.status_timer = (self.status_timer - dt).max(0.0);
        }

        // 1. Hotkeys & analog brush adjustments
        if let Some(m) = input.mode_hotkey {
            let new_mode = EditorMode::ALL[m.min(3)];
            self.switch_mode(new_mode, camera);
        }
        if input.toggle_play {
            let next = if self.brush.mode == EditorMode::Play {
                EditorMode::Sculpt
            } else {
                EditorMode::Play
            };
            self.switch_mode(next, camera);
        }
        if input.brush_size_delta.abs() > 1e-4 {
            self.brush.size = (self.brush.size + input.brush_size_delta).clamp(0.18, 3.6);
        }
        if input.brush_blend_delta.abs() > 1e-4 {
            self.brush.blend = (self.brush.blend + input.brush_blend_delta).clamp(0.03, 1.1);
        }
        if input.toggle_mirror {
            self.brush.mirror_x = !self.brush.mirror_x;
            self.set_toast(if self.brush.mirror_x {
                "Kaleidoscope X-Mirror Symmetry ON"
            } else {
                "Mirror Symmetry OFF"
            });
        }
        if input.cycle_shape {
            self.brush.shape = self.brush.shape.next();
            self.set_toast(format!("Brush Shape: {}", self.brush.shape.label()));
        }
        if input.cycle_fleck {
            self.brush.fleck_style = self.brush.fleck_style.next();
            self.set_toast(format!("Fleck Style: {}", self.brush.fleck_style.label()));
        }
        if input.spawn_prop && self.brush.mode != EditorMode::Play {
            self.spawn_prop_at_imp();
        }
        if input.toggle_record && self.brush.mode == EditorMode::Animate {
            self.record_armed = !self.record_armed;
            self.set_toast(if self.record_armed {
                "Action Recorder ARMED — Grab & wave any Toy Prop to record live!"
            } else {
                "Action Recorder disarmed"
            });
        }
        if input.undo_pressed {
            self.undo();
        }
        if input.redo_pressed {
            self.redo();
        }
        if input.save_requested {
            self.save_scene_default();
        }
        if input.reload_requested {
            self.load_scene_default();
        }
        if input.cancel_active_stroke && self.stroke_active {
            self.cancel_in_progress_stroke();
        }

        // 2. Update 3D Surface-Magnet Imp via raycast against world + props
        self.update_imp(input, camera);

        // 3. Build interactive Toy UI and process UI clicks/sliders/radial menu first
        let mut ui = self.build_ui_frame(input.viewport, device_kind);
        let ui_consumed = self.process_ui_interactions(input, camera, &mut ui);

        // 4. Mode-specific world interaction
        if self.brush.mode == EditorMode::Play {
            self.update_play_mode(dt, input, camera);
        } else if !ui_consumed {
            match self.brush.mode {
                EditorMode::Sculpt | EditorMode::Paint => {
                    self.update_sculpt_or_paint(input, camera);
                }
                EditorMode::Animate => {
                    self.update_animate_mode(input, camera);
                }
                EditorMode::Play => {}
            }
        }

        ui
    }

    pub fn switch_mode(&mut self, next: EditorMode, camera: &mut CameraRig) {
        if self.brush.mode == next {
            return;
        }
        self.finish_stroke();
        let prev = self.brush.mode;
        self.brush.mode = next;

        if next == EditorMode::Play {
            let spawn = if self.imp.on_surface {
                self.imp.world_pos + Vec3::Y * 0.9
            } else {
                Vec3::new(0.0, 2.2, 1.5)
            };
            self.puppet = PlayablePuppet {
                position: spawn,
                velocity: Vec3::ZERO,
                facing_yaw: camera.yaw + std::f32::consts::PI,
                grounded: false,
                walk_cycle: 0.0,
                squash: 1.2,
            };
            self.set_toast("PLAY MODE! Move with WASD / Left-Touch / Stick, Space / A to Jump! Tab to Sculpt.");
        } else if prev == EditorMode::Play {
            camera.target_distance = camera.target_distance.max(10.5);
            self.set_toast(format!("Back to {} Mode!", next.label()));
        } else {
            self.set_toast(format!("Mode: {}", next.label()));
        }
    }

    fn update_imp(&mut self, input: &FrameInput, camera: &CameraRig) {
        let (ray_o, ray_d) = camera.screen_ray(input.imp_px, input.viewport);

        // If mid-stroke in open air, lock depth smoothly so dragging doesn't shoot toward infinity
        if let Some(locked_t) = self.stroke_locked_depth {
            if !self.imp.on_surface {
                let p = (ray_o + ray_d * locked_t).clamp(VOLUME_MIN + Vec3::splat(0.6), VOLUME_MAX - Vec3::splat(0.6));
                self.imp.world_pos = self.imp.world_pos.lerp(p, 0.55);
                self.imp.ground_shadow_pos = self.project_ground_shadow(self.imp.world_pos);
                return;
            }
        }

        if let Some((prop_id, hit_pos, normal, hit_t)) =
            raycast_scene(&self.scene, ray_o, ray_d, self.time, 42.0)
        {
            self.imp.world_pos = hit_pos;
            self.imp.surface_normal = normal;
            self.imp.on_surface = true;
            self.imp.hovered_prop_id = prop_id;
            self.imp.last_hit_distance = hit_t.clamp(3.0, 26.0);
        } else {
            // Smart context depth plane passing through camera focus / last valid surface depth!
            let focus_dist = (camera.focus - ray_o).dot(camera.forward()).max(4.0);
            let plane_t = self.imp.last_hit_distance.lerp_scalar(focus_dist, 0.35);
            let air_pos = (ray_o + ray_d * plane_t)
                .clamp(VOLUME_MIN + Vec3::splat(1.0), VOLUME_MAX - Vec3::splat(1.0));
            self.imp.world_pos = air_pos;
            self.imp.surface_normal = (-ray_d).normalize_or_zero();
            self.imp.on_surface = false;
            self.imp.hovered_prop_id = 0;
        }
        self.imp.ground_shadow_pos = self.project_ground_shadow(self.imp.world_pos);
    }

    fn project_ground_shadow(&self, from: Vec3) -> Vec3 {
        let mut t = 0.1_f32;
        for _ in 0..48 {
            let p = from - Vec3::Y * t;
            let d = eval_world_only_distance(&self.scene, p);
            if d < 0.04 {
                return p;
            }
            t += d.max(0.05);
            if t > 18.0 {
                break;
            }
        }
        Vec3::new(from.x, -2.0, from.z)
    }

    fn update_sculpt_or_paint(&mut self, input: &FrameInput, camera: &CameraRig) {
        if input.primary_just_pressed {
            self.stroke_active = true;
            self.stroke_edits_count = 0;
            let (ray_o, _) = camera.screen_ray(input.imp_px, input.viewport);
            self.stroke_locked_depth = Some(ray_o.distance(self.imp.world_pos));
            self.stamp_brush(input);
        } else if self.stroke_active && input.primary_down {
            let step_spacing = (self.brush.size * input.pressure * 0.32).max(0.11);
            if self.imp.world_pos.distance(self.last_stamp_pos) >= step_spacing {
                self.stamp_brush(input);
            }
        }

        if self.stroke_active && input.primary_just_released {
            self.finish_stroke();
        }
    }

    fn stamp_brush(&mut self, input: &FrameInput) {
        let effective_op = if self.brush.mode == EditorMode::Paint {
            SculptOp::Paint
        } else if input.carve_modifier {
            SculptOp::Subtract
        } else {
            self.brush.op
        };

        let radius = (self.brush.size * input.pressure).max(0.14);
        let pos = self.imp.world_pos.clamp(
            VOLUME_MIN + Vec3::splat(radius + 0.4),
            VOLUME_MAX - Vec3::splat(radius + 0.4),
        );

        // Orient non-spherical brushes smoothly to surface normal!
        let rot = if self.brush.shape == SculptShape::Sphere {
            Quat::IDENTITY
        } else {
            Quat::from_rotation_arc(Vec3::Y, self.imp.surface_normal.normalize_or_zero())
        };

        let scale = match self.brush.shape {
            SculptShape::Capsule | SculptShape::Cylinder | SculptShape::Cone => {
                Vec3::new(radius * 0.65, radius * 1.15, radius * 0.65)
            }
            SculptShape::Torus => Vec3::new(radius, radius * 0.32, radius),
            _ => Vec3::splat(radius),
        };

        let primary_edit = SculptEdit {
            position: pos,
            rotation: rot,
            scale,
            shape: self.brush.shape,
            op: effective_op,
            blend: self.brush.blend,
            color: self.brush.linear_color(),
            looseness: self.brush.looseness,
            fleck_style: self.brush.fleck_style,
        };

        self.push_world_edit(primary_edit.clone());
        self.stroke_edits_count += 1;

        if self.brush.mirror_x && pos.x.abs() > 0.08 {
            let mut mirrored = primary_edit;
            mirrored.position.x = -mirrored.position.x;
            mirrored.rotation = Quat::from_xyzw(
                mirrored.rotation.x,
                -mirrored.rotation.y,
                -mirrored.rotation.z,
                mirrored.rotation.w,
            );
            self.push_world_edit(mirrored);
            self.stroke_edits_count += 1;
        }

        self.last_stamp_pos = pos;
    }

    fn push_world_edit(&mut self, edit: SculptEdit) {
        let (aabb_min, aabb_max) = edit.world_aabb();
        self.dirty_bricks.mark_aabb_dirty(aabb_min, aabb_max);
        self.scene.world_edits.push(edit);
        self.edits_buffer_dirty = true;
        self.redo_stack.clear();
    }

    fn finish_stroke(&mut self) {
        if self.stroke_active && self.stroke_edits_count > 0 {
            let total = self.scene.world_edits.len();
            let start = total.saturating_sub(self.stroke_edits_count);
            let added = self.scene.world_edits[start..].to_vec();
            self.undo_stack
                .push(UndoEntry::SculptStroke { added_edits: added });
        }
        self.stroke_active = false;
        self.stroke_edits_count = 0;
        self.stroke_locked_depth = None;
    }

    fn cancel_in_progress_stroke(&mut self) {
        if self.stroke_active && self.stroke_edits_count > 0 {
            for _ in 0..self.stroke_edits_count {
                if let Some(removed) = self.scene.world_edits.pop() {
                    let (lo, hi) = removed.world_aabb();
                    self.dirty_bricks.mark_aabb_dirty(lo, hi);
                }
            }
            self.edits_buffer_dirty = true;
        }
        self.stroke_active = false;
        self.stroke_edits_count = 0;
        self.stroke_locked_depth = None;
    }

    // ========================================================================
    // Animate Mode: Prop Selection, Live Puppeteering & Action Recorder
    // ========================================================================

    pub fn spawn_prop_at_imp(&mut self) {
        let id = self.scene.next_prop_id;
        self.scene.next_prop_id += 1;
        let pos = self.imp.world_pos + self.imp.surface_normal * (self.brush.size * 0.75);
        let prop = SceneProp {
            id,
            name: format!("Toy #{id}"),
            base_position: pos,
            base_rotation: Quat::IDENTITY,
            scale: Vec3::splat(self.brush.size.clamp(0.25, 2.2)),
            shape: self.brush.shape,
            color: self.brush.linear_color(),
            looseness: self.brush.looseness,
            fleck_style: self.brush.fleck_style,
            gadget: MotionGadget::HoverBob,
            clip: None,
        };
        self.scene.props.push(prop.clone());
        self.selected_prop_id = Some(id);
        self.undo_stack.push(UndoEntry::SpawnProp { prop });
        self.redo_stack.clear();
        self.brush.mode = EditorMode::Animate;
        self.set_toast(format!(
            "Spawned Toy #{id}! Drag to move, or arm Record (R) to animate live!"
        ));
    }

    fn update_animate_mode(&mut self, input: &FrameInput, camera: &CameraRig) {
        let (ray_o, ray_d) = camera.screen_ray(input.imp_px, input.viewport);

        if input.primary_just_pressed {
            if self.imp.hovered_prop_id > 0 {
                let pid = self.imp.hovered_prop_id;
                self.selected_prop_id = Some(pid);
                if let Some(prop) = self.scene.props.iter().find(|p| p.id == pid).cloned() {
                    let (curr_pos, _) = prop.evaluate_pose(self.time);
                    self.dragging_prop_plane_dist = ray_o.distance(curr_pos).max(2.5);
                    self.dragging_prop_grab_offset = curr_pos - self.imp.world_pos;
                    self.dragging_prop_before = Some(prop);

                    if self.record_armed {
                        self.recording_active = true;
                        self.recording_start_time = self.time;
                        self.recording_samples.clear();
                        self.recording_samples.push(KeyframeSample {
                            time: 0.0,
                            position: curr_pos,
                            rotation: Quat::IDENTITY,
                        });
                    }
                }
            }
        } else if input.primary_down && self.dragging_prop_before.is_some() {
            let new_pos = (ray_o + ray_d * self.dragging_prop_plane_dist
                + self.dragging_prop_grab_offset)
                .clamp(VOLUME_MIN + Vec3::ONE, VOLUME_MAX - Vec3::ONE);

            if let Some(pid) = self.selected_prop_id {
                if let Some(prop) = self.scene.props.iter_mut().find(|p| p.id == pid) {
                    if self.recording_active {
                        let rel_t = self.time - self.recording_start_time;
                        if rel_t <= 8.0 {
                            let should_push = self
                                .recording_samples
                                .last()
                                .map(|s| rel_t - s.time >= 0.03)
                                .unwrap_or(true);
                            if should_push {
                                // Subtle playful tilt along velocity!
                                let prev_p = self
                                    .recording_samples
                                    .last()
                                    .map(|s| s.position)
                                    .unwrap_or(new_pos);
                                let vel = (new_pos - prev_p) * 8.0;
                                let tilt = Quat::from_rotation_z((-vel.x * 0.08).clamp(-0.4, 0.4))
                                    * Quat::from_rotation_x((vel.z * 0.08).clamp(-0.4, 0.4));
                                self.recording_samples.push(KeyframeSample {
                                    time: rel_t,
                                    position: new_pos,
                                    rotation: tilt * prop.base_rotation,
                                });
                                prop.base_position = new_pos;
                            }
                        }
                    } else {
                        prop.base_position = new_pos;
                        prop.clip = None;
                    }
                }
            }
        }

        if input.primary_just_released {
            if let Some(before) = self.dragging_prop_before.take() {
                if let Some(pid) = self.selected_prop_id {
                    if let Some(prop) = self.scene.props.iter_mut().find(|p| p.id == pid) {
                        if self.recording_active && self.recording_samples.len() >= 3 {
                            let dur = (self.time - self.recording_start_time).max(0.4);
                            prop.clip = Some(ActionClip {
                                samples: std::mem::take(&mut self.recording_samples),
                                duration: dur,
                            });
                            prop.gadget = MotionGadget::None;
                        }
                        let after = prop.clone();
                        self.undo_stack.push(UndoEntry::ModifyProp { before, after });
                        self.redo_stack.clear();
                    }
                }
                if self.recording_active {
                    self.recording_active = false;
                    self.set_toast("Recorded live looping animation! Tap Clear Path to reset.");
                }
            }
        }
    }

    // ========================================================================
    // Play Mode: Live SDF Physics & Bouncy Explorer Buddy
    // ========================================================================

    fn update_play_mode(&mut self, dt: f32, input: &FrameInput, camera: &mut CameraRig) {
        let dt = dt.min(0.05);
        let radius = 0.45_f32;

        // Camera-relative movement direction
        let cam_fwd = Vec3::new(camera.forward().x, 0.0, camera.forward().z).normalize_or_zero();
        let cam_right = Vec3::new(camera.right().x, 0.0, camera.right().z).normalize_or_zero();
        let wish_dir = (cam_fwd * input.move_axis.y + cam_right * input.move_axis.x)
            .clamp_length_max(1.0);

        let move_speed = 6.2;
        let accel = if self.puppet.grounded { 14.0 } else { 6.5 };
        self.puppet.velocity.x +=
            (wish_dir.x * move_speed - self.puppet.velocity.x) * (accel * dt).min(1.0);
        self.puppet.velocity.z +=
            (wish_dir.z * move_speed - self.puppet.velocity.z) * (accel * dt).min(1.0);

        if wish_dir.length_squared() > 0.02 {
            let target_yaw = wish_dir.x.atan2(wish_dir.z);
            let mut diff = target_yaw - self.puppet.facing_yaw;
            while diff > std::f32::consts::PI {
                diff -= std::f32::consts::TAU;
            }
            while diff < -std::f32::consts::PI {
                diff += std::f32::consts::TAU;
            }
            self.puppet.facing_yaw += diff * (12.0 * dt).min(1.0);
            if self.puppet.grounded {
                self.puppet.walk_cycle += dt * 11.0 * wish_dir.length();
            }
        }

        // Gravity & Bouncy Jump
        self.puppet.velocity.y -= 18.5 * dt;
        if input.jump_just_pressed && self.puppet.grounded {
            self.puppet.velocity.y = 8.4;
            self.puppet.grounded = false;
            self.puppet.squash = 1.38;
        }

        let was_airborne = !self.puppet.grounded && self.puppet.velocity.y < -2.0;
        let next_pos = self.puppet.position + self.puppet.velocity * dt;
        let (resolved, grounded, ground_normal) =
            resolve_sphere_sdf_collision(&self.scene, next_pos, radius, self.time);

        if grounded {
            if was_airborne {
                self.puppet.squash = 0.68; // Juicy landing squash!
            }
            if self.puppet.velocity.y < 0.0 {
                let vn = self.puppet.velocity.dot(ground_normal);
                if vn < 0.0 {
                    self.puppet.velocity -= ground_normal * vn;
                }
            }
        }

        self.puppet.position = resolved;
        self.puppet.grounded = grounded;
        self.puppet.squash += (1.0 - self.puppet.squash) * (12.0 * dt).min(1.0);

        // Respawn gently on the island if player jumps off into the clouds!
        if self.puppet.position.y < -10.5 {
            self.puppet.position = Vec3::new(0.0, 3.0, 0.0);
            self.puppet.velocity = Vec3::ZERO;
            self.puppet.squash = 1.3;
            self.set_toast("Whoosh! Floated back up onto the island.");
        }

        // Smooth 3rd-person follow camera
        camera.target_focus = self.puppet.position + Vec3::new(0.0, 0.65, 0.0);
        camera.target_distance = camera.target_distance.clamp(5.0, 14.0);
    }

    // ========================================================================
    // Undo / Redo & Scene Persistence
    // ========================================================================

    pub fn undo(&mut self) {
        self.finish_stroke();
        let Some(entry) = self.undo_stack.pop() else {
            self.set_toast("Nothing left to undo");
            return;
        };

        match &entry {
            UndoEntry::SculptStroke { added_edits } => {
                for _ in 0..added_edits.len() {
                    if let Some(removed) = self.scene.world_edits.pop() {
                        let (lo, hi) = removed.world_aabb();
                        self.dirty_bricks.mark_aabb_dirty(lo, hi);
                    }
                }
                self.edits_buffer_dirty = true;
            }
            UndoEntry::SpawnProp { prop } => {
                self.scene.props.retain(|p| p.id != prop.id);
            }
            UndoEntry::ModifyProp { before, .. } => {
                if let Some(p) = self.scene.props.iter_mut().find(|p| p.id == before.id) {
                    *p = before.clone();
                }
            }
            UndoEntry::ResetScene { previous } => {
                let current = Box::new(self.scene.clone());
                self.scene = (**previous).clone();
                self.dirty_bricks.mark_all_dirty();
                self.edits_buffer_dirty = true;
                self.redo_stack
                    .push(UndoEntry::ResetScene { previous: current });
                self.set_toast("Undid scene reset");
                return;
            }
        }

        self.redo_stack.push(entry);
        self.set_toast("Undo");
    }

    pub fn redo(&mut self) {
        let Some(entry) = self.redo_stack.pop() else {
            self.set_toast("Nothing to redo");
            return;
        };

        match &entry {
            UndoEntry::SculptStroke { added_edits } => {
                for edit in added_edits {
                    let (lo, hi) = edit.world_aabb();
                    self.dirty_bricks.mark_aabb_dirty(lo, hi);
                    self.scene.world_edits.push(edit.clone());
                }
                self.edits_buffer_dirty = true;
            }
            UndoEntry::SpawnProp { prop } => {
                self.scene.props.push(prop.clone());
            }
            UndoEntry::ModifyProp { after, .. } => {
                if let Some(p) = self.scene.props.iter_mut().find(|p| p.id == after.id) {
                    *p = after.clone();
                }
            }
            UndoEntry::ResetScene { previous } => {
                let current = Box::new(self.scene.clone());
                self.scene = (**previous).clone();
                self.dirty_bricks.mark_all_dirty();
                self.edits_buffer_dirty = true;
                self.undo_stack
                    .push(UndoEntry::ResetScene { previous: current });
                self.set_toast("Redid scene reset");
                return;
            }
        }

        self.undo_stack.push(entry);
        self.set_toast("Redo");
    }

    pub fn save_scene_default(&mut self) {
        match self.scene.save_to_file(DEFAULT_SCENE_FILE) {
            Ok(()) => self.set_toast(format!("Saved scene to {DEFAULT_SCENE_FILE}!")),
            Err(e) => self.set_toast(format!("Save error: {e}")),
        }
    }

    pub fn load_scene_default(&mut self) {
        match TraumScene::load_from_file(DEFAULT_SCENE_FILE) {
            Ok(loaded) => {
                self.scene = loaded;
                self.dirty_bricks.mark_all_dirty();
                self.edits_buffer_dirty = true;
                self.undo_stack.clear();
                self.redo_stack.clear();
                self.set_toast(format!("Loaded {DEFAULT_SCENE_FILE}!"));
            }
            Err(e) => self.set_toast(format!("Load error: {e}")),
        }
    }

    pub fn reset_starter_island(&mut self) {
        let prev = Box::new(self.scene.clone());
        self.scene = TraumScene::starter_diorama_island();
        self.dirty_bricks.mark_all_dirty();
        self.edits_buffer_dirty = true;
        self.undo_stack.push(UndoEntry::ResetScene { previous: prev });
        self.redo_stack.clear();
        self.set_toast("Fresh Diorama Island restored! (Undo available)");
    }

    // ========================================================================
    // Tactile Toy UI Builder & Interaction (Mouse, Touch & Gamepad Friendly!)
    // ========================================================================

    fn build_ui_frame(&mut self, viewport: Vec2, device: ActiveInputDevice) -> UiFrame {
        let mut rects = Vec::with_capacity(48);
        let w = viewport.x;
        let h = viewport.y;

        // Top Mode Bar (4 big tactile pills: Sculpt | Paint | Animate | Play!)
        let bar_w = 436.0_f32.min(w - 24.0);
        let start_x = (w - bar_w) * 0.5;
        let pill_w = (bar_w - 24.0) / 4.0;
        for (i, mode) in EditorMode::ALL.iter().enumerate() {
            let x0 = start_x + (i as f32) * (pill_w + 8.0);
            let active = self.brush.mode == *mode;
            let fill = if active {
                match mode {
                    EditorMode::Play => [0.26, 0.78, 0.44, 0.96],
                    EditorMode::Animate => [0.64, 0.44, 0.88, 0.96],
                    EditorMode::Paint => [0.92, 0.52, 0.36, 0.96],
                    EditorMode::Sculpt => [0.95, 0.68, 0.26, 0.96],
                }
            } else {
                [0.14, 0.15, 0.20, 0.82]
            };
            rects.push(UiRect {
                min: Vec2::new(x0, 14.0),
                max: Vec2::new(x0 + pill_w, 50.0),
                fill_srgb: fill,
                border_srgb: if active {
                    [1.0, 0.96, 0.86, 0.95]
                } else {
                    [0.40, 0.42, 0.52, 0.55]
                },
                corner_radius: 18.0,
                label: mode.label().to_string(),
                text_srgb: if active {
                    [0.10, 0.10, 0.13, 1.0]
                } else {
                    [0.92, 0.93, 0.96, 1.0]
                },
                action: Some(UiAction::SetMode(*mode)),
            });
        }

        // Top-Left Quick Action Chips (Undo, Redo, Radial Menu Button for Touch, Save, Reset)
        let quick_buttons = [
            ("Undo", UiAction::Undo),
            ("Redo", UiAction::Redo),
            ("Wheel", UiAction::OpenRadialButton),
            ("Save", UiAction::SaveScene),
            ("Island", UiAction::ResetIsland),
        ];
        for (i, (lbl, act)) in quick_buttons.iter().enumerate() {
            let x0 = 14.0 + (i as f32) * 68.0;
            if x0 + 62.0 < start_x - 8.0 || h > 640.0 {
                let y0 = if x0 + 62.0 < start_x - 8.0 { 14.0 } else { 58.0 };
                let x_adj = if y0 > 20.0 { 14.0 + (i as f32) * 68.0 } else { x0 };
                rects.push(UiRect {
                    min: Vec2::new(x_adj, y0),
                    max: Vec2::new(x_adj + 62.0, y0 + 34.0),
                    fill_srgb: [0.15, 0.16, 0.22, 0.84],
                    border_srgb: [0.48, 0.50, 0.62, 0.55],
                    corner_radius: 14.0,
                    label: (*lbl).to_string(),
                    text_srgb: [0.94, 0.94, 0.97, 1.0],
                    action: Some(*act),
                });
            }
        }

        // Top-Right Input Badge
        rects.push(UiRect {
            min: Vec2::new((w - 134.0).max(16.0), 14.0),
            max: Vec2::new((w - 14.0).max(120.0), 46.0),
            fill_srgb: [0.13, 0.15, 0.20, 0.78],
            border_srgb: [0.42, 0.66, 0.88, 0.55],
            corner_radius: 15.0,
            label: device.label().to_string(),
            text_srgb: [0.82, 0.92, 1.0, 1.0],
            action: None,
        });

        if self.brush.mode != EditorMode::Play {
            // Left Dock: Tool Context Pills
            let mut y = 104.0;
            if self.brush.mode == EditorMode::Sculpt {
                for op in [SculptOp::Add, SculptOp::Subtract] {
                    let active = self.brush.op == op;
                    rects.push(UiRect {
                        min: Vec2::new(14.0, y),
                        max: Vec2::new(138.0, y + 36.0),
                        fill_srgb: if active {
                            [0.94, 0.56, 0.34, 0.95]
                        } else {
                            [0.14, 0.16, 0.21, 0.82]
                        },
                        border_srgb: [0.85, 0.82, 0.78, if active { 0.9 } else { 0.35 }],
                        corner_radius: 16.0,
                        label: op.label().to_string(),
                        text_srgb: [0.98, 0.98, 0.99, 1.0],
                        action: Some(UiAction::SetOp(op)),
                    });
                    y += 42.0;
                }
            }

            // Shape, Fleck Style, and Mirror pills
            for (lbl, active, act) in [
                (
                    format!("Shape: {}", self.brush.shape.label()),
                    false,
                    UiAction::CycleShape,
                ),
                (
                    format!("Fleck: {}", self.brush.fleck_style.label()),
                    false,
                    UiAction::CycleFleck,
                ),
                (
                    if self.brush.mirror_x {
                        "Mirror: ON".to_string()
                    } else {
                        "Mirror: OFF".to_string()
                    },
                    self.brush.mirror_x,
                    UiAction::ToggleMirror,
                ),
            ] {
                rects.push(UiRect {
                    min: Vec2::new(14.0, y),
                    max: Vec2::new(166.0, y + 36.0),
                    fill_srgb: if active {
                        [0.32, 0.68, 0.88, 0.92]
                    } else {
                        [0.14, 0.16, 0.22, 0.84]
                    },
                    border_srgb: [0.55, 0.62, 0.76, 0.55],
                    corner_radius: 16.0,
                    label: lbl,
                    text_srgb: [0.95, 0.96, 0.99, 1.0],
                    action: Some(act),
                });
                y += 42.0;
            }

            // Animate Mode specific controls!
            if self.brush.mode == EditorMode::Animate {
                y += 8.0;
                let prop_gadget = self
                    .selected_prop_id
                    .and_then(|id| self.scene.props.iter().find(|p| p.id == id))
                    .map(|p| p.gadget.label())
                    .unwrap_or("None");

                for (lbl, highlight, act) in [
                    ("+ Spawn Toy".to_string(), false, UiAction::SpawnProp),
                    (
                        format!("Motion: {prop_gadget}"),
                        false,
                        UiAction::CyclePropGadget,
                    ),
                    (
                        if self.record_armed {
                            "REC: ARMED!".to_string()
                        } else {
                            "Record Path".to_string()
                        },
                        self.record_armed,
                        UiAction::ToggleRecord,
                    ),
                    ("Clear Path".to_string(), false, UiAction::ClearPropClip),
                ] {
                    rects.push(UiRect {
                        min: Vec2::new(14.0, y),
                        max: Vec2::new(166.0, y + 36.0),
                        fill_srgb: if highlight {
                            [0.92, 0.28, 0.34, 0.96]
                        } else {
                            [0.20, 0.17, 0.29, 0.88]
                        },
                        border_srgb: [0.76, 0.62, 0.95, 0.65],
                        corner_radius: 16.0,
                        label: lbl,
                        text_srgb: [0.98, 0.97, 1.0, 1.0],
                        action: Some(act),
                    });
                    y += 42.0;
                }
            }

            // Right-Side Tactile Sliders (Size, Blend Softness, Looseness, Bokeh DOF)
            let sx0 = (w - 188.0).max(200.0);
            let sx1 = (w - 16.0).max(360.0);
            let sliders = [
                (
                    "Size",
                    (self.brush.size - 0.18) / (3.6 - 0.18),
                    UiAction::DragSliderSize,
                    [0.95, 0.68, 0.28, 0.95],
                ),
                (
                    "Blend",
                    (self.brush.blend - 0.03) / (1.1 - 0.03),
                    UiAction::DragSliderBlend,
                    [0.44, 0.78, 0.56, 0.95],
                ),
                (
                    "Loose",
                    self.brush.looseness,
                    UiAction::DragSliderLoose,
                    [0.90, 0.48, 0.62, 0.95],
                ),
                (
                    "Bokeh",
                    self.scene.dof_amount,
                    UiAction::DragSliderDof,
                    [0.46, 0.68, 0.94, 0.95],
                ),
            ];
            let mut sy = 104.0;
            for (name, norm_val, act, bar_col) in sliders {
                let t = norm_val.clamp(0.0, 1.0);
                // Track background
                rects.push(UiRect {
                    min: Vec2::new(sx0, sy),
                    max: Vec2::new(sx1, sy + 30.0),
                    fill_srgb: [0.13, 0.14, 0.19, 0.84],
                    border_srgb: [0.42, 0.45, 0.56, 0.55],
                    corner_radius: 15.0,
                    label: String::new(),
                    text_srgb: [1.0; 4],
                    action: Some(act),
                });
                // Filled value bar
                let fill_w = 18.0 + (sx1 - sx0 - 18.0) * t;
                rects.push(UiRect {
                    min: Vec2::new(sx0, sy),
                    max: Vec2::new(sx0 + fill_w, sy + 30.0),
                    fill_srgb: bar_col,
                    border_srgb: [1.0, 1.0, 1.0, 0.0],
                    corner_radius: 15.0,
                    label: format!("{name} {:>3}%", (t * 100.0).round() as i32),
                    text_srgb: [0.10, 0.11, 0.14, 1.0],
                    action: Some(act),
                });
                sy += 38.0;
            }

            // Bottom Palette Swatch Dock (12 warm clay colors!)
            let swatch_size = 34.0_f32;
            let gap = 6.0_f32;
            let total_pw = PALETTE_SRGB.len() as f32 * (swatch_size + gap) - gap;
            let px0 = (w - total_pw) * 0.5;
            let py0 = h - swatch_size - 16.0;
            for (idx, col) in PALETTE_SRGB.iter().enumerate() {
                let x = px0 + (idx as f32) * (swatch_size + gap);
                let selected = self.brush.palette_index == idx;
                let lift = if selected { 6.0 } else { 0.0 };
                rects.push(UiRect {
                    min: Vec2::new(x, py0 - lift),
                    max: Vec2::new(x + swatch_size, py0 + swatch_size - lift),
                    fill_srgb: [col[0], col[1], col[2], 1.0],
                    border_srgb: if selected {
                        [1.0, 0.98, 0.90, 1.0]
                    } else {
                        [0.12, 0.12, 0.15, 0.7]
                    },
                    corner_radius: 17.0,
                    label: String::new(),
                    text_srgb: [1.0; 4],
                    action: Some(UiAction::SelectColor(idx)),
                });
            }
        }

        let banner = if self.status_timer > 0.0 {
            self.status_toast.clone()
        } else {
            String::new()
        };

        UiFrame {
            rects,
            radial: None,
            banner_status: banner,
        }
    }

    fn process_ui_interactions(
        &mut self,
        input: &FrameInput,
        camera: &mut CameraRig,
        ui: &mut UiFrame,
    ) -> bool {
        // 1. Handle Bouncy Radial Pie Menu (Space / Gamepad L1 / Touch "Wheel" button)
        let radial_active = input.radial_open || self.radial_button_open;
        if radial_active {
            if self.radial_anchor_px == Vec2::ZERO
                || (input.radial_open && !self.radial_button_open && self.radial_hovered.is_none())
            {
                // Anchor near screen center or Imp
                self.radial_anchor_px = input
                    .imp_px
                    .clamp(Vec2::splat(150.0), input.viewport - Vec2::splat(150.0));
            }

            let dir_vec = if let Some(stick) = input.radial_stick_dir {
                Some(stick)
            } else {
                let delta = input.imp_px - self.radial_anchor_px;
                if delta.length() > 28.0 {
                    Some(delta.normalize())
                } else {
                    None
                }
            };

            self.radial_hovered = dir_vec.map(|d| {
                let angle = d.y.atan2(d.x); // -PI..PI, 0 = Right
                let norm = (angle + std::f32::consts::PI * 2.5).rem_euclid(std::f32::consts::TAU);
                (((norm / std::f32::consts::TAU) * 8.0).round() as usize) % 8
            });

            ui.radial = Some(RadialMenuVisual {
                center_px: self.radial_anchor_px,
                outer_radius: 118.0,
                hovered_slice: self.radial_hovered,
                labels: [
                    "Add Clay",
                    "Carve",
                    "Coat Paint",
                    "Next Shape",
                    "Next Fleck",
                    "Mirror X",
                    "+ Toy Prop",
                    "Play!",
                ],
            });

            if input.radial_just_released
                || (self.radial_button_open && input.primary_just_pressed)
            {
                if let Some(slice) = self.radial_hovered {
                    self.execute_radial_slice(slice, camera);
                }
                self.radial_button_open = false;
                self.radial_hovered = None;
                self.radial_anchor_px = Vec2::ZERO;
            }
            return true;
        } else if input.radial_just_released {
            if let Some(slice) = self.radial_hovered.take() {
                self.execute_radial_slice(slice, camera);
            }
            self.radial_anchor_px = Vec2::ZERO;
            return true;
        }

        // 2. Continue active slider drag
        if let Some(slider_act) = self.active_slider {
            if input.primary_down {
                self.apply_slider_drag(slider_act, input.imp_px.x, input.viewport.x);
                return true;
            } else {
                self.active_slider = None;
            }
        }

        // 3. Hit-test UI pills/buttons on press (and block hover pass-through when pointer is over a pill)
        let mut hit_action = None;
        for rect in ui.rects.iter().rev() {
            if rect.action.is_some()
                && input.imp_px.x >= rect.min.x
                && input.imp_px.x <= rect.max.x
                && input.imp_px.y >= rect.min.y
                && input.imp_px.y <= rect.max.y
            {
                hit_action = rect.action;
                break;
            }
        }

        if let Some(act) = hit_action {
            if input.primary_just_pressed {
                match act {
                    UiAction::DragSliderSize
                    | UiAction::DragSliderBlend
                    | UiAction::DragSliderLoose
                    | UiAction::DragSliderDof => {
                        self.active_slider = Some(act);
                        self.apply_slider_drag(act, input.imp_px.x, input.viewport.x);
                    }
                    UiAction::SetMode(m) => self.switch_mode(m, camera),
                    UiAction::SetOp(op) => {
                        self.brush.op = op;
                        self.set_toast(format!("Brush Operation: {}", op.label()));
                    }
                    UiAction::CycleShape => {
                        self.brush.shape = self.brush.shape.next();
                        self.set_toast(format!("Shape: {}", self.brush.shape.label()));
                    }
                    UiAction::CycleFleck => {
                        self.brush.fleck_style = self.brush.fleck_style.next();
                        self.set_toast(format!("Fleck: {}", self.brush.fleck_style.label()));
                    }
                    UiAction::SelectColor(idx) => {
                        self.brush.palette_index = idx;
                    }
                    UiAction::ToggleMirror => {
                        self.brush.mirror_x = !self.brush.mirror_x;
                    }
                    UiAction::Undo => self.undo(),
                    UiAction::Redo => self.redo(),
                    UiAction::SpawnProp => self.spawn_prop_at_imp(),
                    UiAction::CyclePropGadget => {
                        if let Some(pid) = self.selected_prop_id {
                            if let Some(p) = self.scene.props.iter_mut().find(|p| p.id == pid) {
                                p.gadget = p.gadget.next();
                                let lbl = p.gadget.label();
                                self.set_toast(format!("Toy Motion Gadget: {lbl}"));
                            }
                        }
                    }
                    UiAction::ToggleRecord => {
                        self.record_armed = !self.record_armed;
                        self.set_toast(if self.record_armed {
                            "Action Recorder ARMED — Drag any Toy Prop to record live!"
                        } else {
                            "Action Recorder OFF"
                        });
                    }
                    UiAction::ClearPropClip => {
                        if let Some(pid) = self.selected_prop_id {
                            if let Some(p) = self.scene.props.iter_mut().find(|p| p.id == pid) {
                                p.clip = None;
                                self.set_toast("Cleared recorded path on selected Toy");
                            }
                        }
                    }
                    UiAction::SaveScene => self.save_scene_default(),
                    UiAction::ResetIsland => self.reset_starter_island(),
                    UiAction::OpenRadialButton => {
                        self.radial_button_open = !self.radial_button_open;
                        self.radial_anchor_px = input.viewport * 0.5;
                    }
                }
            }
            return true;
        }

        false
    }

    fn apply_slider_drag(&mut self, slider: UiAction, cursor_x: f32, viewport_w: f32) {
        let sx0 = (viewport_w - 188.0).max(200.0);
        let sx1 = (viewport_w - 16.0).max(360.0);
        let t = ((cursor_x - sx0) / (sx1 - sx0).max(1.0)).clamp(0.0, 1.0);
        match slider {
            UiAction::DragSliderSize => self.brush.size = 0.18 + t * (3.6 - 0.18),
            UiAction::DragSliderBlend => self.brush.blend = 0.03 + t * (1.1 - 0.03),
            UiAction::DragSliderLoose => self.brush.looseness = t,
            UiAction::DragSliderDof => self.scene.dof_amount = t,
            _ => {}
        }
    }

    fn execute_radial_slice(&mut self, slice: usize, camera: &mut CameraRig) {
        match slice {
            0 => {
                self.brush.mode = EditorMode::Sculpt;
                self.brush.op = SculptOp::Add;
                self.set_toast("Sculpt: Add Clay");
            }
            1 => {
                self.brush.mode = EditorMode::Sculpt;
                self.brush.op = SculptOp::Subtract;
                self.set_toast("Sculpt: Carve Clay");
            }
            2 => {
                self.brush.mode = EditorMode::Paint;
                self.set_toast("Paint: Coat Flecks & Color");
            }
            3 => {
                self.brush.shape = self.brush.shape.next();
                self.set_toast(format!("Shape: {}", self.brush.shape.label()));
            }
            4 => {
                self.brush.fleck_style = self.brush.fleck_style.next();
                self.set_toast(format!("Fleck: {}", self.brush.fleck_style.label()));
            }
            5 => {
                self.brush.mirror_x = !self.brush.mirror_x;
                self.set_toast(if self.brush.mirror_x {
                    "Mirror Symmetry ON"
                } else {
                    "Mirror Symmetry OFF"
                });
            }
            6 => self.spawn_prop_at_imp(),
            7 => self.switch_mode(EditorMode::Play, camera),
            _ => {}
        }
    }
}

impl Default for TraumEditor {
    fn default() -> Self {
        Self::new()
    }
}

trait FloatLerpExt {
    fn lerp_scalar(self, target: f32, t: f32) -> f32;
}
impl FloatLerpExt for f32 {
    fn lerp_scalar(self, target: f32, t: f32) -> f32 {
        self + (target - self) * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sculpt_stroke_and_undo_redo_work() {
        let mut editor = TraumEditor::new();
        let initial_len = editor.scene.world_edits.len();
        let _ = editor.dirty_bricks.take_dirty_bricks();

        let mut cam = CameraRig::default();
        let mut inp = FrameInput {
            imp_px: Vec2::new(640.0, 360.0),
            viewport: Vec2::new(1280.0, 720.0),
            primary_down: true,
            primary_just_pressed: true,
            pressure: 1.0,
            ..Default::default()
        };
        editor.update(0.016, &inp, ActiveInputDevice::MouseKeyboard, &mut cam);
        assert_eq!(editor.scene.world_edits.len(), initial_len + 1);
        assert!(editor.dirty_bricks.has_dirty());

        // Release stroke -> commits undo entry
        inp.primary_down = false;
        inp.primary_just_pressed = false;
        inp.primary_just_released = true;
        editor.update(0.016, &inp, ActiveInputDevice::MouseKeyboard, &mut cam);

        editor.undo();
        assert_eq!(editor.scene.world_edits.len(), initial_len);

        editor.redo();
        assert_eq!(editor.scene.world_edits.len(), initial_len + 1);
    }
}
