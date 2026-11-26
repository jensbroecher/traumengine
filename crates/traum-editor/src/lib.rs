//! Tactile sculpting tools, Scope-In Custom Prop Sculpting, Live Cloning, Multi-Part Connectors,
//! 3D Transform Handles, Surface-Magnet Imp, Bouncy Radial UI, Action Recorder, and Play Mode puppet.

use glam::{Quat, Vec2, Vec3};
use std::collections::HashMap;
use traum_core::{
    srgb_to_linear, ActionClip, ConnectorKind, EditorMode, FleckStyle, KeyframeSample,
    MotionGadget, PropConnector, SceneProp, SculptAsset, SculptEdit, SculptOp, SculptShape,
    TraumScene, PALETTE_SRGB,
};
use traum_input::{ActiveInputDevice, CameraRig, FrameInput};
use traum_sdf::{
    eval_world_only_distance, find_safe_play_spawn, raycast_scene, raycast_single_prop,
    resolve_sphere_sdf_collision, AssetDirtyTracker, BrickDirtyTracker, ASSET_LOCAL_MAX,
    ASSET_LOCAL_MIN, MAX_SCULPT_ASSETS, VOLUME_MAX, VOLUME_MIN,
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

/// Surface-Magnet 3D Imp state (visualized as a glowing 3D cursor + ground drop-shadow).
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
    pub coyote_timer: f32,
    pub standing_prop_id: u32,
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
            coyote_timer: 0.0,
            standing_prop_id: 0,
            walk_cycle: 0.0,
            squash: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PropDragKind {
    MoveOrRecord,
    RotateRing,
    ScaleOrb,
}

/// Reversible actions stored in the non-destructive Undo / Redo stack.
#[derive(Debug, Clone)]
enum UndoEntry {
    WorldSculptStroke {
        added_edits: Vec<SculptEdit>,
    },
    AssetSculptStroke {
        asset_id: u32,
        added_edits: Vec<SculptEdit>,
    },
    SpawnProps {
        added_asset: Option<SculptAsset>,
        props: Vec<SceneProp>,
    },
    DeleteProps {
        props: Vec<SceneProp>,
    },
    ModifyProp {
        before: SceneProp,
        after: SceneProp,
    },
    ForkUniqueAsset {
        prop_id: u32,
        old_asset_id: u32,
        created_asset: SculptAsset,
    },
    ResetScene {
        previous: Box<TraumScene>,
    },
}

/// Visual style variant for a 2D UI rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiRectStyle {
    /// Frosted translucent container panel or recessed track.
    Panel,
    /// Tactile glossy 3D button pill with hover lift.
    Button,
    /// Flat vibrant accent (slider fill bar or color swatch).
    FlatAccent,
}

/// Rich descriptive tooltip card displayed when hovering any UI control or radial slice.
#[derive(Debug, Clone)]
pub struct TooltipCard {
    pub anchor_min: Vec2,
    pub anchor_max: Vec2,
    pub title: String,
    pub shortcut: String,
    pub description: String,
}

/// Interactive 2D Toy UI elements rendered by `traum-render`.
#[derive(Debug, Clone)]
pub struct UiRect {
    pub min: Vec2,
    pub max: Vec2,
    pub fill_srgb: [f32; 4],
    pub border_srgb: [f32; 4],
    pub corner_radius: f32,
    pub border_width: f32,
    pub style: UiRectStyle,
    pub icon: Option<u32>,
    pub label: String,
    pub right_label: String,
    pub font_px: f32,
    pub left_align_text: bool,
    pub text_srgb: [f32; 4],
    pub hovered: bool,
    pub cast_shadow: bool,
    pub action: Option<UiAction>,
    pub tooltip: Option<(String, String, String)>,
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
    ToggleScope,
    CloneProp,
    MakeUnique,
    SpawnChildPart,
    CycleConnectorKind,
    DeleteProp,
    CyclePropGadget,
    ToggleRecord,
    ClearPropClip,
    SaveScene,
    ResetIsland,
    OpenRadialButton,
    ToggleHelp,
    ToggleOutliner,
    OutlinerSelectWorld,
    OutlinerSelectProp(u32),
    OutlinerMoveProp(u32),
    OutlinerEditPropClay(u32),
    OutlinerPagePrev,
    OutlinerPageNext,
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
    pub ui_scale: f32,
    pub rects: Vec<UiRect>,
    pub radial: Option<RadialMenuVisual>,
    pub tooltip: Option<TooltipCard>,
    pub help_open: bool,
    pub banner_status: String,
}

/// Main stateful Traum Engine editor, tool controller, animator, and play-mode simulator.
pub struct TraumEditor {
    pub scene: TraumScene,
    pub brush: BrushSettings,
    pub imp: ImpCursor,
    pub puppet: PlayablePuppet,
    pub dirty_bricks: BrickDirtyTracker,
    pub asset_dirty: AssetDirtyTracker,
    pub edits_buffer_dirty: bool,
    pub asset_edits_dirty: bool,
    pub time: f32,

    /// When `Some(prop_id)`, Sculpt & Paint strokes edit that prop's reusable `SculptAsset` in local space!
    pub scoped_prop_id: Option<u32>,

    // Active stroke state
    stroke_active: bool,
    last_stamp_pos: Vec3,
    stroke_edits_count: usize,
    stroke_target_asset_id: Option<u32>,
    stroke_locked_depth: Option<f32>,

    // Animate / 3D Transform / Prop state
    pub selected_prop_id: Option<u32>,
    pub record_armed: bool,
    recording_active: bool,
    recording_start_time: f32,
    recording_samples: Vec<KeyframeSample>,
    dragging_prop_before: Option<SceneProp>,
    dragging_kind: PropDragKind,
    dragging_prop_plane_dist: f32,
    dragging_prop_grab_offset: Vec3,
    dragging_start_px: Vec2,
    dragging_start_scale: f32,
    dragging_start_rot: Quat,

    // UI interaction state
    active_slider: Option<UiAction>,
    radial_button_open: bool,
    radial_anchor_px: Vec2,
    radial_hovered: Option<usize>,
    pub show_help_guide: bool,
    pub outliner_open: bool,
    pub outliner_page: usize,
    status_toast: String,
    status_timer: f32,

    // Undo / Redo stacks
    undo_stack: Vec<UndoEntry>,
    redo_stack: Vec<UndoEntry>,
}

impl TraumEditor {
    pub fn new() -> Self {
        let scene = TraumScene::starter_diorama_island();
        let mut asset_dirty = AssetDirtyTracker::new();
        asset_dirty.sync_scene_assets(&scene.assets);

        Self {
            scene,
            brush: BrushSettings::default(),
            imp: ImpCursor::default(),
            puppet: PlayablePuppet::default(),
            dirty_bricks: BrickDirtyTracker::new(),
            asset_dirty,
            edits_buffer_dirty: true,
            asset_edits_dirty: true,
            time: 0.0,
            scoped_prop_id: None,
            stroke_active: false,
            last_stamp_pos: Vec3::ZERO,
            stroke_edits_count: 0,
            stroke_target_asset_id: None,
            stroke_locked_depth: None,
            selected_prop_id: Some(4), // Start with the multi-part Bouncy Critter selected!
            record_armed: false,
            recording_active: false,
            recording_start_time: 0.0,
            recording_samples: Vec::new(),
            dragging_prop_before: None,
            dragging_kind: PropDragKind::MoveOrRecord,
            dragging_prop_plane_dist: 10.0,
            dragging_prop_grab_offset: Vec3::ZERO,
            dragging_start_px: Vec2::ZERO,
            dragging_start_scale: 1.0,
            dragging_start_rot: Quat::IDENTITY,
            active_slider: None,
            radial_button_open: false,
            radial_anchor_px: Vec2::new(400.0, 300.0),
            radial_hovered: None,
            show_help_guide: false,
            outliner_open: true,
            outliner_page: 0,
            status_toast: "Hover any button to see a tooltip! Drag to sculpt clay, Right-Drag to orbit.".into(),
            status_timer: 6.5,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }

    pub fn set_toast(&mut self, msg: impl Into<String>) {
        self.status_toast = msg.into();
        self.status_timer = 3.8;
    }

    /// Returns the atlas slot (`0..MAX_SCULPT_ASSETS`) for a given `asset_id`.
    pub fn asset_slot_of(&self, asset_id: u32) -> Option<usize> {
        self.scene
            .assets
            .iter()
            .position(|a| a.id == asset_id)
            .filter(|&s| s < MAX_SCULPT_ASSETS)
    }

    /// Computes the world-space 3D gizmo ring radius and scale orb position for a selected prop in Animate mode.
    pub fn selected_gizmo_geometry(&self, prop: &SceneProp) -> (f32, Vec3) {
        let local_r = self
            .scene
            .asset_by_id(prop.asset_id)
            .map(|a| a.local_bound_radius())
            .unwrap_or(0.65);
        let ring_r = (local_r * prop.uniform_scale + 0.42).clamp(0.65, 4.2);
        let (p_pos, p_rot) = self.scene.evaluate_prop_pose(prop.id, self.time);
        let orb_dir = (p_rot * Vec3::X).normalize_or_zero();
        let orb_pos = p_pos + orb_dir * ring_r;
        (ring_r, orb_pos)
    }

    pub fn update(
        &mut self,
        dt: f32,
        input: &FrameInput,
        device_kind: ActiveInputDevice,
        camera: &mut CameraRig,
    ) -> UiFrame {
        let prev_time = self.time;
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
            if self.brush.mode == EditorMode::Animate {
                // Shift+Scroll or [ / ] in Animate mode scales the selected prop smoothly!
                if let Some(pid) = self.selected_prop_id {
                    if let Some(p) = self.scene.props.iter_mut().find(|p| p.id == pid) {
                        p.uniform_scale =
                            (p.uniform_scale + input.brush_size_delta * 0.45).clamp(0.22, 3.4);
                    }
                }
            } else {
                self.brush.size = (self.brush.size + input.brush_size_delta).clamp(0.12, 3.6);
            }
        }
        if input.brush_blend_delta.abs() > 1e-4 {
            self.brush.blend = (self.brush.blend + input.brush_blend_delta).clamp(0.03, 1.1);
        }
        if input.toggle_mirror {
            self.brush.mirror_x = !self.brush.mirror_x;
            self.set_toast(if self.brush.mirror_x {
                if self.scoped_prop_id.is_some() {
                    "Local Toy X-Mirror Symmetry ON!"
                } else {
                    "World X-Mirror Symmetry ON"
                }
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

        // Update Imp first so hover-sensitive actions (`Scope In`, `Clone`, `+ Limb`) know which prop is under the cursor!
        self.update_imp(input, camera);

        if input.spawn_prop && self.brush.mode != EditorMode::Play {
            self.spawn_prop_at_imp(camera);
        }
        if input.toggle_scope && self.brush.mode != EditorMode::Play {
            self.toggle_scope_in_out(camera);
        }
        if input.clone_prop && self.brush.mode != EditorMode::Play {
            self.clone_selected_prop();
        }
        if input.make_unique && self.brush.mode != EditorMode::Play {
            self.make_selected_prop_unique();
        }
        if input.spawn_child_part && self.brush.mode != EditorMode::Play {
            self.spawn_connected_child_part(camera);
        }
        if input.delete_selected && self.brush.mode != EditorMode::Play {
            if self.scoped_prop_id.is_some() {
                self.scope_out();
            } else if self.brush.mode == EditorMode::Animate {
                self.delete_selected_prop();
            }
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

        // Build interactive Toy UI and process UI clicks/sliders/radial menu first
        let mut ui = self.build_ui_frame(input.viewport, input.ui_scale, device_kind);
        let ui_consumed = self.process_ui_interactions(input, camera, &mut ui);

        // Mode-specific world interaction
        if self.brush.mode == EditorMode::Play {
            self.update_play_mode(prev_time, dt, input, camera);
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

        self.asset_dirty.sync_scene_assets(&self.scene.assets);
        ui
    }

    pub fn switch_mode(&mut self, next: EditorMode, camera: &mut CameraRig) {
        if self.brush.mode == next {
            return;
        }
        self.finish_stroke();
        let prev = self.brush.mode;
        self.brush.mode = next;

        if next == EditorMode::Animate || next == EditorMode::Play {
            // Scoping out when entering Animate or Play keeps transforms crystal clear
            self.scoped_prop_id = None;
        }

        if next == EditorMode::Play {
            // Raycast straight down from sky near camera focus (or valid upward-facing Imp hit) so we NEVER spawn underground!
            let hint_xz = if self.imp.on_surface && self.imp.surface_normal.y > 0.55 {
                Vec2::new(self.imp.world_pos.x, self.imp.world_pos.z)
            } else {
                Vec2::new(camera.focus.x, camera.focus.z)
            };
            let spawn = find_safe_play_spawn(&self.scene, hint_xz, 0.54, self.time);
            self.puppet = PlayablePuppet {
                position: spawn,
                velocity: Vec3::ZERO,
                facing_yaw: camera.yaw + std::f32::consts::PI,
                grounded: false,
                coyote_timer: 0.12,
                standing_prop_id: 0,
                walk_cycle: 0.0,
                squash: 1.2,
            };
            self.set_toast(
                "PLAY MODE! Move with WASD / Arrow Keys / Stick, Space to Jump! Tab to Sculpt.",
            );
        } else if prev == EditorMode::Play {
            camera.target_distance = camera.target_distance.max(10.5);
            self.set_toast(format!("Back to {} Mode!", next.label()));
        } else {
            self.set_toast(format!("Mode: {}", next.label()));
        }
    }

    // ========================================================================
    // Scope-In / Scope-Out, Live Cloning, Unique Forking & Multi-Part Joints!
    // ========================================================================

    pub fn toggle_scope_in_out(&mut self, camera: &mut CameraRig) {
        if self.scoped_prop_id.is_some() {
            self.scope_out();
            return;
        }

        // Prefer hovered prop under the Imp, otherwise currently selected prop, otherwise spawn a new sculpt prop!
        let target_id = if self.imp.hovered_prop_id > 0 {
            Some(self.imp.hovered_prop_id)
        } else {
            self.selected_prop_id
        };

        if let Some(pid) = target_id {
            if let Some(prop) = self.scene.props.iter().find(|p| p.id == pid).cloned() {
                self.finish_stroke();
                self.scoped_prop_id = Some(pid);
                self.selected_prop_id = Some(pid);
                if self.brush.mode != EditorMode::Paint {
                    self.brush.mode = EditorMode::Sculpt;
                }
                let (p_pos, _) = self.scene.evaluate_prop_pose(pid, self.time);
                camera.target_focus = p_pos;
                camera.target_distance = (prop.uniform_scale * 5.2).clamp(3.6, 9.5);
                self.brush.size = self.brush.size.min(0.55);

                let clones = self
                    .scene
                    .props
                    .iter()
                    .filter(|p| p.asset_id == prop.asset_id)
                    .count();
                if clones > 1 {
                    self.set_toast(format!(
                        "SCOPED INTO {} ({clones} Live Clones will update together! Enter to Scope Out)",
                        prop.name
                    ));
                } else {
                    self.set_toast(format!(
                        "SCOPED INTO {} — Sculpt & Paint in local space! Press Enter to Scope Out.",
                        prop.name
                    ));
                }
                return;
            }
        }

        self.spawn_prop_at_imp(camera);
    }

    pub fn scope_out(&mut self) {
        self.finish_stroke();
        if self.scoped_prop_id.take().is_some() {
            self.set_toast("Scoped Out to World Island!");
        }
    }

    pub fn spawn_prop_at_imp(&mut self, camera: &mut CameraRig) {
        if self.scene.assets.len() >= MAX_SCULPT_ASSETS {
            self.set_toast("Asset Atlas full (24 custom sculpts) — use Live Clone (C)!");
            return;
        }
        self.finish_stroke();

        let asset_id = self.scene.next_asset_id;
        self.scene.next_asset_id += 1;

        let seed_r = self.brush.size.clamp(0.28, 1.4);
        let asset = SculptAsset {
            id: asset_id,
            name: format!("Custom Sculpt #{asset_id}"),
            edits: vec![SculptEdit {
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::splat(seed_r),
                shape: self.brush.shape,
                op: SculptOp::Add,
                blend: 0.18,
                color: self.brush.linear_color(),
                looseness: self.brush.looseness,
                fleck_style: self.brush.fleck_style,
            }],
        };
        let slot = self.scene.assets.len();
        self.scene.assets.push(asset.clone());
        self.asset_dirty.mark_slot_all_dirty(slot);
        self.asset_edits_dirty = true;

        let prop_id = self.scene.next_prop_id;
        self.scene.next_prop_id += 1;
        let pos = self.imp.world_pos + self.imp.surface_normal * (seed_r * 0.85);
        let prop = SceneProp {
            id: prop_id,
            name: format!("Toy #{prop_id}"),
            asset_id,
            base_position: pos,
            base_rotation: Quat::IDENTITY,
            uniform_scale: 1.0,
            scale: Vec3::splat(seed_r),
            shape: self.brush.shape,
            color: self.brush.linear_color(),
            looseness: self.brush.looseness,
            fleck_style: self.brush.fleck_style,
            gadget: MotionGadget::None,
            clip: None,
            connector: None,
        };
        self.scene.props.push(prop.clone());
        self.selected_prop_id = Some(prop_id);
        self.scoped_prop_id = Some(prop_id);
        self.brush.mode = EditorMode::Sculpt;
        self.brush.size = (seed_r * 0.55).clamp(0.18, 0.75);

        camera.target_focus = pos;
        camera.target_distance = camera.target_distance.min(7.5);

        self.undo_stack.push(UndoEntry::SpawnProps {
            added_asset: Some(asset),
            props: vec![prop],
        });
        self.redo_stack.clear();
        self.set_toast(format!(
            "Created & Scoped Into Toy #{prop_id}! Sculpt right on it, or press Enter to Scope Out."
        ));
    }

    /// Live-Clones the selected prop (plus any connected child limbs!), sharing the same `SculptAsset`s!
    pub fn clone_selected_prop(&mut self) {
        let target_id = self
            .scoped_prop_id
            .or(if self.imp.hovered_prop_id > 0 {
                Some(self.imp.hovered_prop_id)
            } else {
                self.selected_prop_id
            });
        let Some(root_id) = target_id else {
            self.set_toast("Hover or select a Toy Prop first to Live-Clone it!");
            return;
        };
        let Some(root_prop) = self.scene.props.iter().find(|p| p.id == root_id).cloned() else {
            return;
        };

        // Collect root + any direct/indirect child limbs connected to it so multi-part puppets clone intact!
        let mut to_clone = vec![root_prop.clone()];
        let mut idx = 0;
        while idx < to_clone.len() {
            let pid = to_clone[idx].id;
            for p in &self.scene.props {
                if p.connector.as_ref().map(|c| c.parent_id) == Some(pid)
                    && !to_clone.iter().any(|existing| existing.id == p.id)
                {
                    to_clone.push(p.clone());
                }
            }
            idx += 1;
        }

        if self.scene.props.len() + to_clone.len() > traum_sdf::MAX_GPU_PROPS {
            self.set_toast("Maximum scene props reached!");
            return;
        }

        let mut id_map = HashMap::new();
        for old in &to_clone {
            let new_id = self.scene.next_prop_id;
            self.scene.next_prop_id += 1;
            id_map.insert(old.id, new_id);
        }

        let offset = if self.imp.on_surface && self.imp.hovered_prop_id == 0 {
            self.imp.world_pos + Vec3::Y * 0.65 - root_prop.base_position
        } else {
            Vec3::new(1.35 * root_prop.uniform_scale, 0.0, 0.65)
        };

        let mut spawned = Vec::with_capacity(to_clone.len());
        for old in to_clone {
            let new_id = id_map[&old.id];
            let mut cloned = old.clone();
            cloned.id = new_id;
            cloned.name = format!("{} Clone", old.name);
            cloned.base_position = (old.base_position + offset)
                .clamp(VOLUME_MIN + Vec3::ONE, VOLUME_MAX - Vec3::ONE);
            cloned.clip = None; // Fresh clone starts ready for its own placement/animation
            if let Some(conn) = &mut cloned.connector {
                if let Some(&mapped_parent) = id_map.get(&conn.parent_id) {
                    conn.parent_id = mapped_parent;
                }
            }
            self.scene.props.push(cloned.clone());
            spawned.push(cloned);
        }

        let new_root_id = id_map[&root_id];
        self.selected_prop_id = Some(new_root_id);
        self.scoped_prop_id = None;
        self.brush.mode = EditorMode::Animate;
        self.undo_stack.push(UndoEntry::SpawnProps {
            added_asset: None,
            props: spawned,
        });
        self.redo_stack.clear();
        self.set_toast(
            "Live-Cloned Toy! Drag to place — Scope In (Enter) on any clone to edit all siblings at once!",
        );
    }

    /// Forks a Live-Cloned prop's `SculptAsset` into its own independent asset slot.
    pub fn make_selected_prop_unique(&mut self) {
        let Some(pid) = self.scoped_prop_id.or(self.selected_prop_id) else {
            return;
        };
        if self.scene.assets.len() >= MAX_SCULPT_ASSETS {
            self.set_toast("Asset Atlas full (24 unique sculpts)!");
            return;
        }
        let Some(prop) = self.scene.props.iter().find(|p| p.id == pid).cloned() else {
            return;
        };
        let shared_count = self
            .scene
            .props
            .iter()
            .filter(|p| p.asset_id == prop.asset_id)
            .count();
        if shared_count <= 1 {
            self.set_toast("This Toy already has its own unique Sculpt Asset!");
            return;
        }
        let Some(src_asset) = self.scene.asset_by_id(prop.asset_id).cloned() else {
            return;
        };

        let new_aid = self.scene.next_asset_id;
        self.scene.next_asset_id += 1;
        let forked = SculptAsset {
            id: new_aid,
            name: format!("{} Unique", prop.name),
            edits: src_asset.edits,
        };
        let slot = self.scene.assets.len();
        self.scene.assets.push(forked.clone());
        self.asset_dirty.mark_slot_all_dirty(slot);
        self.asset_edits_dirty = true;

        if let Some(p) = self.scene.props.iter_mut().find(|p| p.id == pid) {
            p.asset_id = new_aid;
        }
        self.undo_stack.push(UndoEntry::ForkUniqueAsset {
            prop_id: pid,
            old_asset_id: prop.asset_id,
            created_asset: forked,
        });
        self.redo_stack.clear();
        self.set_toast("Made Toy Unique — sculpting it will no longer change its clones!");
    }

    /// Spawns a new custom-sculptable child part attached to the selected prop via a `PropConnector`!
    pub fn spawn_connected_child_part(&mut self, camera: &mut CameraRig) {
        let parent_id = self
            .scoped_prop_id
            .or(self.selected_prop_id)
            .or((self.imp.hovered_prop_id > 0).then_some(self.imp.hovered_prop_id));
        let Some(pid) = parent_id else {
            self.set_toast("Select a parent Toy Prop first to attach a connected limb/part!");
            return;
        };
        if self.scene.assets.len() >= MAX_SCULPT_ASSETS {
            self.set_toast("Asset Atlas full (24 custom sculpts)!");
            return;
        }
        let Some(parent) = self.scene.props.iter().find(|p| p.id == pid).cloned() else {
            return;
        };

        let (p_pos, p_rot) = self.scene.evaluate_prop_pose(pid, self.time);
        let p_scale = parent.uniform_scale.max(0.1);
        let parent_r = self
            .scene
            .asset_by_id(parent.asset_id)
            .map(|a| a.local_bound_radius())
            .unwrap_or(0.5);

        // Anchor on parent surface where Imp is hovering, or neatly atop the parent!
        let parent_anchor = if self.imp.hovered_prop_id == pid {
            ((p_rot.conjugate() * (self.imp.world_pos - p_pos)) / p_scale)
                .clamp(Vec3::splat(-2.2), Vec3::splat(2.2))
        } else {
            Vec3::new(0.0, parent_r * 0.88, 0.0)
        };

        let limb_r = (self.brush.size * 0.48).clamp(0.20, 0.75);
        let asset_id = self.scene.next_asset_id;
        self.scene.next_asset_id += 1;
        let asset = SculptAsset {
            id: asset_id,
            name: format!("Limb Sculpt #{asset_id}"),
            edits: vec![SculptEdit {
                position: Vec3::new(0.0, limb_r * 0.75, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::new(limb_r * 0.65, limb_r * 1.15, limb_r * 0.65),
                shape: SculptShape::Capsule,
                op: SculptOp::Add,
                blend: 0.14,
                color: self.brush.linear_color(),
                looseness: self.brush.looseness,
                fleck_style: self.brush.fleck_style,
            }],
        };
        let slot = self.scene.assets.len();
        self.scene.assets.push(asset.clone());
        self.asset_dirty.mark_slot_all_dirty(slot);
        self.asset_edits_dirty = true;

        let child_id = self.scene.next_prop_id;
        self.scene.next_prop_id += 1;
        let pivot_world = p_pos + p_rot * (parent_anchor * p_scale);
        let child = SceneProp {
            id: child_id,
            name: format!("Limb #{child_id}"),
            asset_id,
            base_position: pivot_world,
            base_rotation: Quat::IDENTITY,
            uniform_scale: parent.uniform_scale,
            scale: Vec3::splat(limb_r),
            shape: SculptShape::Capsule,
            color: self.brush.linear_color(),
            looseness: self.brush.looseness,
            fleck_style: self.brush.fleck_style,
            gadget: MotionGadget::None,
            clip: None,
            connector: Some(PropConnector {
                parent_id: pid,
                parent_anchor,
                child_anchor: Vec3::new(0.0, -0.05, 0.0),
                kind: ConnectorKind::BallSpring,
            }),
        };

        self.scene.props.push(child.clone());
        self.selected_prop_id = Some(child_id);
        self.scoped_prop_id = None;
        self.brush.mode = EditorMode::Animate;
        camera.target_focus = pivot_world;

        self.undo_stack.push(UndoEntry::SpawnProps {
            added_asset: Some(asset),
            props: vec![child],
        });
        self.redo_stack.clear();
        self.set_toast(
            "Attached Spring-Jointed Part! Click 'Joint' to switch Spring / Spin Hinge / Swing Hinge, or Enter to Sculpt it!",
        );
    }

    pub fn delete_selected_prop(&mut self) {
        let Some(pid) = self.selected_prop_id else {
            return;
        };
        // Remove selected prop + detach/remove direct children cleanly
        let mut removed = Vec::new();
        let mut i = 0;
        while i < self.scene.props.len() {
            let remove_it = self.scene.props[i].id == pid
                || self.scene.props[i]
                    .connector
                    .as_ref()
                    .map(|c| c.parent_id == pid)
                    .unwrap_or(false);
            if remove_it {
                removed.push(self.scene.props.remove(i));
            } else {
                i += 1;
            }
        }
        if !removed.is_empty() {
            self.selected_prop_id = self.scene.props.last().map(|p| p.id);
            self.scoped_prop_id = None;
            self.undo_stack
                .push(UndoEntry::DeleteProps { props: removed });
            self.redo_stack.clear();
            self.set_toast("Deleted Toy Prop (Undo available)");
        }
    }

    // ========================================================================
    // 3D Surface-Magnet Imp & World / Local Scope-In Sculpting
    // ========================================================================

    fn update_imp(&mut self, input: &FrameInput, camera: &CameraRig) {
        let (ray_o, ray_d) = camera.screen_ray(input.imp_px, input.viewport);

        if let Some(locked_t) = self.stroke_locked_depth {
            if !self.imp.on_surface {
                let p = (ray_o + ray_d * locked_t)
                    .clamp(VOLUME_MIN + Vec3::splat(0.6), VOLUME_MAX - Vec3::splat(0.6));
                self.imp.world_pos = self.imp.world_pos.lerp(p, 0.55);
                self.imp.ground_shadow_pos = self.project_ground_shadow(self.imp.world_pos);
                return;
            }
        }

        // When Scoped Into a custom prop, magnetize directly to that prop's surface (or its local center plane)!
        if let Some(scoped_id) = self.scoped_prop_id {
            if let Some((hit_pos, normal, hit_t)) =
                raycast_single_prop(&self.scene, scoped_id, ray_o, ray_d, self.time, 36.0)
            {
                self.imp.world_pos = hit_pos;
                self.imp.surface_normal = normal;
                self.imp.on_surface = true;
                self.imp.hovered_prop_id = scoped_id;
                self.imp.last_hit_distance = hit_t.clamp(1.5, 22.0);
                self.imp.ground_shadow_pos = self.project_ground_shadow(hit_pos);
                return;
            } else if self.scene.props.iter().any(|p| p.id == scoped_id) {
                let (p_pos, _) = self.scene.evaluate_prop_pose(scoped_id, self.time);
                let plane_t = (p_pos - ray_o).dot(camera.forward()).max(1.8);
                let air_pos = ray_o + ray_d * plane_t;
                self.imp.world_pos = air_pos;
                self.imp.surface_normal = (-ray_d).normalize_or_zero();
                self.imp.on_surface = false;
                self.imp.hovered_prop_id = scoped_id;
                self.imp.ground_shadow_pos = self.project_ground_shadow(air_pos);
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
            self.imp.last_hit_distance = hit_t.clamp(2.5, 26.0);
        } else {
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
            self.stroke_target_asset_id = self.scoped_prop_id.and_then(|pid| {
                self.scene
                    .props
                    .iter()
                    .find(|p| p.id == pid)
                    .map(|p| p.asset_id)
            });
            let (ray_o, _) = camera.screen_ray(input.imp_px, input.viewport);
            self.stroke_locked_depth = Some(ray_o.distance(self.imp.world_pos));
            self.stamp_brush(input);
        } else if self.stroke_active && input.primary_down {
            let step_spacing = (self.brush.size * input.pressure * 0.30).max(0.08);
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

        let world_radius = (self.brush.size * input.pressure).max(0.10);
        let world_rot = if self.brush.shape == SculptShape::Sphere {
            Quat::IDENTITY
        } else {
            Quat::from_rotation_arc(Vec3::Y, self.imp.surface_normal.normalize_or_zero())
        };
        let world_scale = match self.brush.shape {
            SculptShape::Capsule | SculptShape::Cylinder | SculptShape::Cone => Vec3::new(
                world_radius * 0.65,
                world_radius * 1.15,
                world_radius * 0.65,
            ),
            SculptShape::Torus => Vec3::new(world_radius, world_radius * 0.32, world_radius),
            _ => Vec3::splat(world_radius),
        };

        // Branch A: Scoped Into a Custom Prop -> Stamp in Prop's Local Asset Space!
        if let Some(scoped_pid) = self.scoped_prop_id {
            if let Some(prop) = self.scene.props.iter().find(|p| p.id == scoped_pid).cloned() {
                let (p_pos, p_rot) = self.scene.evaluate_prop_pose(scoped_pid, self.time);
                let inv_scale = 1.0 / prop.uniform_scale.max(0.1);
                let local_scale = (world_scale * inv_scale).max(Vec3::splat(0.04));
                let local_blend = (self.brush.blend * inv_scale).clamp(0.02, 0.9);
                let pad = local_scale.max_element() + 0.22;
                let local_pos = ((p_rot.conjugate() * (self.imp.world_pos - p_pos)) * inv_scale)
                    .clamp(ASSET_LOCAL_MIN + Vec3::splat(pad), ASSET_LOCAL_MAX - Vec3::splat(pad));
                let local_rot = (p_rot.conjugate() * world_rot).normalize();

                let primary_edit = SculptEdit {
                    position: local_pos,
                    rotation: local_rot,
                    scale: local_scale,
                    shape: self.brush.shape,
                    op: effective_op,
                    blend: local_blend,
                    color: self.brush.linear_color(),
                    looseness: self.brush.looseness,
                    fleck_style: self.brush.fleck_style,
                };

                self.push_asset_edit(prop.asset_id, primary_edit.clone());
                self.stroke_edits_count += 1;

                if self.brush.mirror_x && local_pos.x.abs() > 0.04 {
                    let mut mirrored = primary_edit;
                    mirrored.position.x = -mirrored.position.x;
                    mirrored.rotation = Quat::from_xyzw(
                        mirrored.rotation.x,
                        -mirrored.rotation.y,
                        -mirrored.rotation.z,
                        mirrored.rotation.w,
                    );
                    self.push_asset_edit(prop.asset_id, mirrored);
                    self.stroke_edits_count += 1;
                }

                self.last_stamp_pos = self.imp.world_pos;
                return;
            }
        }

        // Branch B: World Island Sculpting
        let pos = self.imp.world_pos.clamp(
            VOLUME_MIN + Vec3::splat(world_radius + 0.4),
            VOLUME_MAX - Vec3::splat(world_radius + 0.4),
        );

        let primary_edit = SculptEdit {
            position: pos,
            rotation: world_rot,
            scale: world_scale,
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

    fn push_asset_edit(&mut self, asset_id: u32, edit: SculptEdit) {
        let (aabb_min, aabb_max) = edit.world_aabb();
        if let Some(slot) = self.asset_slot_of(asset_id) {
            self.asset_dirty
                .mark_slot_aabb_dirty(slot, aabb_min, aabb_max);
        }
        if let Some(asset) = self.scene.asset_by_id_mut(asset_id) {
            asset.edits.push(edit);
            self.asset_edits_dirty = true;
            self.redo_stack.clear();
        }
    }

    fn finish_stroke(&mut self) {
        if self.stroke_active && self.stroke_edits_count > 0 {
            if let Some(aid) = self.stroke_target_asset_id {
                if let Some(asset) = self.scene.asset_by_id(aid) {
                    let total = asset.edits.len();
                    let start = total.saturating_sub(self.stroke_edits_count);
                    let added = asset.edits[start..].to_vec();
                    self.undo_stack.push(UndoEntry::AssetSculptStroke {
                        asset_id: aid,
                        added_edits: added,
                    });
                }
            } else {
                let total = self.scene.world_edits.len();
                let start = total.saturating_sub(self.stroke_edits_count);
                let added = self.scene.world_edits[start..].to_vec();
                self.undo_stack
                    .push(UndoEntry::WorldSculptStroke { added_edits: added });
            }
        }
        self.stroke_active = false;
        self.stroke_edits_count = 0;
        self.stroke_target_asset_id = None;
        self.stroke_locked_depth = None;
    }

    fn cancel_in_progress_stroke(&mut self) {
        if self.stroke_active && self.stroke_edits_count > 0 {
            if let Some(aid) = self.stroke_target_asset_id {
                let slot = self.asset_slot_of(aid);
                if let Some(asset) = self.scene.asset_by_id_mut(aid) {
                    for _ in 0..self.stroke_edits_count {
                        if let Some(removed) = asset.edits.pop() {
                            let (lo, hi) = removed.world_aabb();
                            if let Some(s) = slot {
                                self.asset_dirty.mark_slot_aabb_dirty(s, lo, hi);
                            }
                        }
                    }
                    self.asset_edits_dirty = true;
                }
            } else {
                for _ in 0..self.stroke_edits_count {
                    if let Some(removed) = self.scene.world_edits.pop() {
                        let (lo, hi) = removed.world_aabb();
                        self.dirty_bricks.mark_aabb_dirty(lo, hi);
                    }
                }
                self.edits_buffer_dirty = true;
            }
        }
        self.stroke_active = false;
        self.stroke_edits_count = 0;
        self.stroke_target_asset_id = None;
        self.stroke_locked_depth = None;
    }

    // ========================================================================
    // Animate Mode: 3D Scale/Rotate Gizmo Handles, Prop Grab & Action Recorder
    // ========================================================================

    fn hit_test_selected_gizmo(&self, input: &FrameInput, camera: &CameraRig) -> Option<PropDragKind> {
        let pid = self.selected_prop_id?;
        let prop = self.scene.props.iter().find(|p| p.id == pid)?;
        let (ring_r, orb_world) = self.selected_gizmo_geometry(prop);

        // 1. Check Golden Scale Orb handle in screen space (generous 26px touch/gamepad/mouse target!)
        if let Some(orb_px) = camera.world_to_screen(orb_world, input.viewport) {
            if orb_px.distance(input.imp_px) <= 26.0 {
                return Some(PropDragKind::ScaleOrb);
            }
        }

        // 2. Check 3D Turquoise Rotation Ring around prop equator!
        let (p_pos, p_rot) = self.scene.evaluate_prop_pose(pid, self.time);
        let ring_normal = (p_rot * Vec3::Y).normalize_or_zero();
        let (ray_o, ray_d) = camera.screen_ray(input.imp_px, input.viewport);
        let denom = ray_d.dot(ring_normal);
        if denom.abs() > 1e-3 {
            let t = (p_pos - ray_o).dot(ring_normal) / denom;
            if t > 0.2 {
                let hit = ray_o + ray_d * t;
                let radial_err = (hit.distance(p_pos) - ring_r).abs();
                if radial_err < (0.18 * prop.uniform_scale.max(0.7)).max(0.16) {
                    return Some(PropDragKind::RotateRing);
                }
            }
        }

        None
    }

    fn update_animate_mode(&mut self, input: &FrameInput, camera: &CameraRig) {
        let (ray_o, ray_d) = camera.screen_ray(input.imp_px, input.viewport);

        if input.primary_just_pressed {
            // Check 3D Scale Orb & Yaw/Tilt Rotation Ring on currently selected prop first!
            if !self.record_armed {
                if let Some(gizmo_kind) = self.hit_test_selected_gizmo(input, camera) {
                    if let Some(pid) = self.selected_prop_id {
                        if let Some(prop) = self.scene.props.iter().find(|p| p.id == pid).cloned() {
                            self.dragging_kind = gizmo_kind;
                            self.dragging_start_px = input.imp_px;
                            self.dragging_start_scale = prop.uniform_scale;
                            self.dragging_start_rot = prop.base_rotation;
                            self.dragging_prop_before = Some(prop);
                            return;
                        }
                    }
                }
            }

            if self.imp.hovered_prop_id > 0 {
                let pid = self.imp.hovered_prop_id;
                self.selected_prop_id = Some(pid);
                if let Some(prop) = self.scene.props.iter().find(|p| p.id == pid).cloned() {
                    let (curr_pos, curr_rot) = self.scene.evaluate_prop_pose(pid, self.time);
                    self.dragging_kind = PropDragKind::MoveOrRecord;
                    self.dragging_prop_plane_dist = ray_o.distance(curr_pos).max(2.5);
                    self.dragging_prop_grab_offset = curr_pos - self.imp.world_pos;
                    self.dragging_start_px = input.imp_px;
                    self.dragging_start_rot = prop.base_rotation;
                    self.dragging_prop_before = Some(prop);

                    if self.record_armed {
                        self.recording_active = true;
                        self.recording_start_time = self.time;
                        self.recording_samples.clear();
                        self.recording_samples.push(KeyframeSample {
                            time: 0.0,
                            position: curr_pos,
                            rotation: curr_rot,
                        });
                    }
                }
            }
        } else if input.primary_down && self.dragging_prop_before.is_some() {
            let Some(pid) = self.selected_prop_id else {
                return;
            };
            let p_center_screen = {
                let (p_pos, _) = self.scene.evaluate_prop_pose(pid, self.time);
                camera.world_to_screen(p_pos, input.viewport)
            };

            match self.dragging_kind {
                PropDragKind::ScaleOrb => {
                    let center_px = p_center_screen.unwrap_or(input.viewport * 0.5);
                    let start_d = self.dragging_start_px.distance(center_px).max(24.0);
                    let curr_d = input.imp_px.distance(center_px).max(8.0);
                    let ratio = curr_d / start_d;
                    if let Some(prop) = self.scene.props.iter_mut().find(|p| p.id == pid) {
                        prop.uniform_scale = (self.dragging_start_scale * ratio).clamp(0.22, 3.4);
                    }
                }
                PropDragKind::RotateRing => {
                    let delta = input.imp_px - self.dragging_start_px;
                    let yaw_rot = Quat::from_rotation_y(delta.x * 0.014);
                    let tilt_rot = Quat::from_rotation_x(delta.y * 0.008);
                    if let Some(prop) = self.scene.props.iter_mut().find(|p| p.id == pid) {
                        prop.base_rotation =
                            (yaw_rot * tilt_rot * self.dragging_start_rot).normalize();
                    }
                }
                PropDragKind::MoveOrRecord => {
                    let new_pos = (ray_o + ray_d * self.dragging_prop_plane_dist
                        + self.dragging_prop_grab_offset)
                        .clamp(VOLUME_MIN + Vec3::ONE, VOLUME_MAX - Vec3::ONE);

                    // Check if this prop is a connected child limb: dragging a child limb poses its joint rotation!
                    let parent_pivot = self
                        .scene
                        .props
                        .iter()
                        .find(|p| p.id == pid)
                        .and_then(|p| p.connector.as_ref().map(|c| c.parent_id))
                        .and_then(|parent_id| {
                            let parent = self.scene.props.iter().find(|p| p.id == parent_id)?;
                            let conn = self
                                .scene
                                .props
                                .iter()
                                .find(|p| p.id == pid)?
                                .connector
                                .as_ref()?;
                            let (p_pos, p_rot) =
                                self.scene.evaluate_prop_pose(parent_id, self.time);
                            Some((
                                p_pos + p_rot * (conn.parent_anchor * parent.uniform_scale),
                                p_rot,
                            ))
                        });

                    if let Some(prop) = self.scene.props.iter_mut().find(|p| p.id == pid) {
                        if let Some((pivot, p_rot)) = parent_pivot {
                            let local_aim =
                                (p_rot.conjugate() * (new_pos - pivot)).normalize_or_zero();
                            if local_aim.length_squared() > 0.1 {
                                let posed_rot = Quat::from_rotation_arc(Vec3::Y, local_aim);
                                prop.base_rotation = posed_rot;
                                if self.recording_active {
                                    let rel_t = self.time - self.recording_start_time;
                                    if rel_t <= 8.0 {
                                        let should_push = self
                                            .recording_samples
                                            .last()
                                            .map(|s| rel_t - s.time >= 0.03)
                                            .unwrap_or(true);
                                        if should_push {
                                            self.recording_samples.push(KeyframeSample {
                                                time: rel_t,
                                                position: new_pos,
                                                rotation: posed_rot,
                                            });
                                        }
                                    }
                                }
                            }
                        } else if self.recording_active {
                            let rel_t = self.time - self.recording_start_time;
                            if rel_t <= 8.0 {
                                let should_push = self
                                    .recording_samples
                                    .last()
                                    .map(|s| rel_t - s.time >= 0.03)
                                    .unwrap_or(true);
                                if should_push {
                                    let prev_p = self
                                        .recording_samples
                                        .last()
                                        .map(|s| s.position)
                                        .unwrap_or(new_pos);
                                    let vel = (new_pos - prev_p) * 8.0;
                                    let tilt =
                                        Quat::from_rotation_z((-vel.x * 0.08).clamp(-0.4, 0.4))
                                            * Quat::from_rotation_x(
                                                (vel.z * 0.08).clamp(-0.4, 0.4),
                                            );
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
    // Play Mode: Live SDF Physics, Moving Platform Riding & Explorer Buddy
    // ========================================================================

    fn update_play_mode(
        &mut self,
        prev_time: f32,
        dt: f32,
        input: &FrameInput,
        camera: &mut CameraRig,
    ) {
        let dt = dt.min(0.05);
        let radius = 0.54_f32;

        // Carry the player smoothly when standing on a moving/recorded sculpted platform!
        if self.puppet.grounded && self.puppet.standing_prop_id > 0 {
            let pid = self.puppet.standing_prop_id;
            if self.scene.props.iter().any(|p| p.id == pid) {
                let (prev_p, _) = self.scene.evaluate_prop_pose(pid, prev_time);
                let (curr_p, _) = self.scene.evaluate_prop_pose(pid, self.time);
                self.puppet.position += curr_p - prev_p;
            }
        }

        let cam_fwd = Vec3::new(camera.forward().x, 0.0, camera.forward().z).normalize_or_zero();
        let cam_right = Vec3::new(camera.right().x, 0.0, camera.right().z).normalize_or_zero();
        let wish_dir =
            (cam_fwd * input.move_axis.y + cam_right * input.move_axis.x).clamp_length_max(1.0);

        let move_speed = 6.5;
        let accel = if self.puppet.grounded { 16.0 } else { 8.0 };
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
                self.puppet.walk_cycle += dt * 11.5 * wish_dir.length();
            }
        }

        if self.puppet.grounded {
            self.puppet.coyote_timer = 0.14;
        } else {
            self.puppet.coyote_timer = (self.puppet.coyote_timer - dt).max(0.0);
        }

        // Gravity (with safe terminal velocity) & Bouncy Jump with Coyote Time!
        self.puppet.velocity.y = (self.puppet.velocity.y - 19.0 * dt).max(-14.0);
        let jumped_this_frame = input.jump_just_pressed && self.puppet.coyote_timer > 0.0;
        if jumped_this_frame {
            self.puppet.velocity.y = 8.8;
            self.puppet.grounded = false;
            self.puppet.coyote_timer = 0.0;
            self.puppet.standing_prop_id = 0;
            self.puppet.squash = 1.38;
        }

        let was_airborne = !self.puppet.grounded && self.puppet.velocity.y < -2.2;

        // 3 sub-steps per frame prevent tunneling and allow buttery slope/wall sliding!
        let sub_dt = dt / 3.0;
        let mut grounded = false;
        let mut ground_normal = Vec3::Y;
        let mut standing_prop = 0u32;

        for _ in 0..3 {
            let step_start = self.puppet.position;
            let next_pos = step_start + self.puppet.velocity * sub_dt;
            let (resolved, step_grounded, step_normal, step_prop) =
                resolve_sphere_sdf_collision(&self.scene, next_pos, radius, self.time);

            // If vertical collision lifted us up and we aren't actively jumping upward, zero downward fall speed!
            if resolved.y > next_pos.y + 0.0005 && self.puppet.velocity.y < 0.0 {
                self.puppet.velocity.y = 0.0;
            }
            // Slide horizontal velocity smoothly along walls so walking into cliffs doesn't stick
            let push = resolved - next_pos;
            if push.length_squared() > 1e-6 {
                let push_dir = push.normalize();
                let into_surf = self.puppet.velocity.dot(push_dir);
                if into_surf < 0.0 {
                    self.puppet.velocity -= push_dir * into_surf;
                }
            }

            self.puppet.position = resolved;
            if step_grounded && !jumped_this_frame {
                grounded = true;
                ground_normal = step_normal;
                standing_prop = step_prop;
            }
        }

        if grounded {
            if was_airborne {
                self.puppet.squash = 0.68;
            }
            if self.puppet.velocity.y < 0.0 {
                let vn = self.puppet.velocity.dot(ground_normal);
                if vn < 0.0 {
                    self.puppet.velocity -= ground_normal * vn;
                }
            }
        }

        self.puppet.grounded = grounded;
        self.puppet.standing_prop_id = if grounded { standing_prop } else { 0 };
        self.puppet.squash += (1.0 - self.puppet.squash) * (12.0 * dt).min(1.0);

        if self.puppet.position.y < -10.5 {
            self.puppet.position =
                find_safe_play_spawn(&self.scene, Vec2::new(0.0, 1.5), radius, self.time);
            self.puppet.velocity = Vec3::ZERO;
            self.puppet.squash = 1.3;
            self.set_toast("Whoosh! Floated back up onto the island.");
        }

        camera.target_focus = self.puppet.position + Vec3::new(0.0, 0.55, 0.0);
        camera.target_distance = camera.target_distance.clamp(3.2, 28.0);
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
            UndoEntry::WorldSculptStroke { added_edits } => {
                for _ in 0..added_edits.len() {
                    if let Some(removed) = self.scene.world_edits.pop() {
                        let (lo, hi) = removed.world_aabb();
                        self.dirty_bricks.mark_aabb_dirty(lo, hi);
                    }
                }
                self.edits_buffer_dirty = true;
            }
            UndoEntry::AssetSculptStroke {
                asset_id,
                added_edits,
            } => {
                let slot = self.asset_slot_of(*asset_id);
                if let Some(asset) = self.scene.asset_by_id_mut(*asset_id) {
                    for _ in 0..added_edits.len() {
                        if let Some(removed) = asset.edits.pop() {
                            let (lo, hi) = removed.world_aabb();
                            if let Some(s) = slot {
                                self.asset_dirty.mark_slot_aabb_dirty(s, lo, hi);
                            }
                        }
                    }
                    self.asset_edits_dirty = true;
                }
            }
            UndoEntry::SpawnProps { added_asset, props } => {
                for pr in props {
                    self.scene.props.retain(|p| p.id != pr.id);
                    if self.scoped_prop_id == Some(pr.id) {
                        self.scoped_prop_id = None;
                    }
                }
                if let Some(a) = added_asset {
                    self.scene.assets.retain(|x| x.id != a.id);
                    self.asset_edits_dirty = true;
                }
            }
            UndoEntry::DeleteProps { props } => {
                for pr in props {
                    self.scene.props.push(pr.clone());
                }
            }
            UndoEntry::ModifyProp { before, .. } => {
                if let Some(p) = self.scene.props.iter_mut().find(|p| p.id == before.id) {
                    *p = before.clone();
                }
            }
            UndoEntry::ForkUniqueAsset {
                prop_id,
                old_asset_id,
                created_asset,
            } => {
                if let Some(p) = self.scene.props.iter_mut().find(|p| p.id == *prop_id) {
                    p.asset_id = *old_asset_id;
                }
                self.scene.assets.retain(|a| a.id != created_asset.id);
                self.asset_edits_dirty = true;
            }
            UndoEntry::ResetScene { previous } => {
                let current = Box::new(self.scene.clone());
                self.scene = (**previous).clone();
                self.scoped_prop_id = None;
                self.dirty_bricks.mark_all_dirty();
                self.asset_dirty
                    .mark_all_assets_dirty(self.scene.assets.len());
                self.edits_buffer_dirty = true;
                self.asset_edits_dirty = true;
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
            UndoEntry::WorldSculptStroke { added_edits } => {
                for edit in added_edits {
                    let (lo, hi) = edit.world_aabb();
                    self.dirty_bricks.mark_aabb_dirty(lo, hi);
                    self.scene.world_edits.push(edit.clone());
                }
                self.edits_buffer_dirty = true;
            }
            UndoEntry::AssetSculptStroke {
                asset_id,
                added_edits,
            } => {
                let slot = self.asset_slot_of(*asset_id);
                if let Some(asset) = self.scene.asset_by_id_mut(*asset_id) {
                    for edit in added_edits {
                        let (lo, hi) = edit.world_aabb();
                        if let Some(s) = slot {
                            self.asset_dirty.mark_slot_aabb_dirty(s, lo, hi);
                        }
                        asset.edits.push(edit.clone());
                    }
                    self.asset_edits_dirty = true;
                }
            }
            UndoEntry::SpawnProps { added_asset, props } => {
                if let Some(a) = added_asset {
                    let slot = self.scene.assets.len();
                    self.scene.assets.push(a.clone());
                    self.asset_dirty.mark_slot_all_dirty(slot);
                    self.asset_edits_dirty = true;
                }
                for pr in props {
                    self.scene.props.push(pr.clone());
                }
            }
            UndoEntry::DeleteProps { props } => {
                for pr in props {
                    self.scene.props.retain(|p| p.id != pr.id);
                }
            }
            UndoEntry::ModifyProp { after, .. } => {
                if let Some(p) = self.scene.props.iter_mut().find(|p| p.id == after.id) {
                    *p = after.clone();
                }
            }
            UndoEntry::ForkUniqueAsset {
                prop_id,
                created_asset,
                ..
            } => {
                let slot = self.scene.assets.len();
                self.scene.assets.push(created_asset.clone());
                self.asset_dirty.mark_slot_all_dirty(slot);
                self.asset_edits_dirty = true;
                if let Some(p) = self.scene.props.iter_mut().find(|p| p.id == *prop_id) {
                    p.asset_id = created_asset.id;
                }
            }
            UndoEntry::ResetScene { previous } => {
                let current = Box::new(self.scene.clone());
                self.scene = (**previous).clone();
                self.scoped_prop_id = None;
                self.dirty_bricks.mark_all_dirty();
                self.asset_dirty
                    .mark_all_assets_dirty(self.scene.assets.len());
                self.edits_buffer_dirty = true;
                self.asset_edits_dirty = true;
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
                self.scoped_prop_id = None;
                self.dirty_bricks.mark_all_dirty();
                self.asset_dirty = AssetDirtyTracker::new();
                self.asset_dirty.sync_scene_assets(&self.scene.assets);
                self.edits_buffer_dirty = true;
                self.asset_edits_dirty = true;
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
        self.scoped_prop_id = None;
        self.dirty_bricks.mark_all_dirty();
        self.asset_dirty = AssetDirtyTracker::new();
        self.asset_dirty.sync_scene_assets(&self.scene.assets);
        self.edits_buffer_dirty = true;
        self.asset_edits_dirty = true;
        self.undo_stack.push(UndoEntry::ResetScene { previous: prev });
        self.redo_stack.clear();
        self.set_toast("Fresh Diorama Island restored! (Undo available)");
    }

    // ========================================================================
    // Tactile Toy UI Builder, Rich Tooltips & Hover Interaction
    // ========================================================================

    fn compute_ui_scale(viewport: Vec2, raw_dpi_scale: f32) -> f32 {
        let dpi = if raw_dpi_scale > 0.1 {
            raw_dpi_scale
        } else {
            1.0
        };
        // Scale up for HiDPI screens, while gently adapting down if the window is very small so panels never collide.
        let fit_h = (viewport.y / 660.0).max(0.68);
        let fit_w = (viewport.x / 1060.0).max(0.68);
        dpi.min(fit_h).min(fit_w).clamp(0.72, 2.75)
    }

    fn build_ui_frame(
        &mut self,
        viewport: Vec2,
        raw_dpi_scale: f32,
        device: ActiveInputDevice,
    ) -> UiFrame {
        let s = Self::compute_ui_scale(viewport, raw_dpi_scale);
        let mut rects = Vec::with_capacity(80);
        let w = viewport.x;
        let h = viewport.y;

        let mk_panel = |min: Vec2, max: Vec2, radius: f32| UiRect {
            min,
            max,
            fill_srgb: [0.08, 0.09, 0.14, 0.86],
            border_srgb: [0.38, 0.43, 0.56, 0.55],
            corner_radius: radius * s,
            border_width: 1.4 * s,
            style: UiRectStyle::Panel,
            icon: None,
            label: String::new(),
            right_label: String::new(),
            font_px: 13.0 * s,
            left_align_text: false,
            text_srgb: [0.95; 4],
            hovered: false,
            cast_shadow: true,
            action: None,
            tooltip: None,
        };

        let mk_header = |x0: f32, x1: f32, y: f32, title: &str| UiRect {
            min: Vec2::new(x0, y),
            max: Vec2::new(x1, y + 20.0 * s),
            fill_srgb: [0.0, 0.0, 0.0, 0.0],
            border_srgb: [0.0, 0.0, 0.0, 0.0],
            corner_radius: 4.0,
            border_width: 0.0,
            style: UiRectStyle::Panel,
            icon: None,
            label: title.to_string(),
            right_label: String::new(),
            font_px: 10.2 * s,
            left_align_text: true,
            text_srgb: [0.72, 0.78, 0.92, 0.95],
            hovered: false,
            cast_shadow: false,
            action: None,
            tooltip: None,
        };

        // 1. Top-Center Mode Switcher Tray (Sculpt | Paint | Animate | Play!)
        let bar_w = (496.0 * s).min(w - 24.0 * s);
        let start_x = (w - bar_w) * 0.5;
        let top_y = 12.0 * s;
        let tab_h = 40.0 * s;
        rects.push(mk_panel(
            Vec2::new(start_x - 6.0 * s, top_y - 4.0 * s),
            Vec2::new(start_x + bar_w + 6.0 * s, top_y + tab_h + 4.0 * s),
            22.0,
        ));

        let gap = 7.0 * s;
        let pill_w = (bar_w - gap * 3.0) / 4.0;
        for (i, mode) in EditorMode::ALL.iter().enumerate() {
            let x0 = start_x + (i as f32) * (pill_w + gap);
            let active = self.brush.mode == *mode;
            let (icon_id, tip_title, tip_key, tip_desc) = match mode {
                EditorMode::Sculpt => (
                    0u32,
                    "Sculpt Mode",
                    "Key: 1",
                    "Build or carve 3D digital clay! Drag on the island or inside a Toy Object to sculpt shapes smoothly.",
                ),
                EditorMode::Paint => (
                    2u32,
                    "Paint & Coat Mode",
                    "Key: 2",
                    "Color and re-texture existing clay surfaces without changing their 3D shape.",
                ),
                EditorMode::Animate => (
                    3u32,
                    "Animate & Assemble Mode",
                    "Key: 3",
                    "Select, move, rotate, scale, attach jointed limbs, or record live movement loops on Toy Objects.",
                ),
                EditorMode::Play => (
                    4u32,
                    "Play Mode!",
                    "Key: 4 or Tab",
                    "Drop a playable character right onto your sculpted island to run, jump, and test your world instantly!",
                ),
            };
            let fill = if active {
                match mode {
                    EditorMode::Sculpt => [0.94, 0.62, 0.20, 0.98],
                    EditorMode::Paint => [0.90, 0.44, 0.34, 0.98],
                    EditorMode::Animate => [0.58, 0.40, 0.88, 0.98],
                    EditorMode::Play => [0.22, 0.74, 0.42, 0.98],
                }
            } else {
                [0.16, 0.18, 0.26, 0.92]
            };
            rects.push(UiRect {
                min: Vec2::new(x0, top_y),
                max: Vec2::new(x0 + pill_w, top_y + tab_h),
                fill_srgb: fill,
                border_srgb: if active {
                    [1.0, 0.97, 0.88, 0.98]
                } else {
                    [0.42, 0.46, 0.58, 0.65]
                },
                corner_radius: 18.0 * s,
                border_width: if active { 2.0 * s } else { 1.3 * s },
                style: UiRectStyle::Button,
                icon: Some(icon_id),
                label: mode.label().to_string(),
                right_label: String::new(),
                font_px: 12.6 * s,
                left_align_text: false,
                text_srgb: [0.99, 0.99, 1.0, 1.0],
                hovered: false,
                cast_shadow: active,
                action: Some(UiAction::SetMode(*mode)),
                tooltip: Some((tip_title.into(), tip_key.into(), tip_desc.into())),
            });
        }

        // 2. Top-Left Quick Action Bar (Undo, Redo, Wheel, Save, Reset)
        let quick_buttons: [(&str, u32, UiAction, &str, &str, &str); 5] = [
            (
                "Undo",
                10,
                UiAction::Undo,
                "Undo Last Action",
                "Ctrl+Z / 2-Finger Tap",
                "Steps backward to undo your most recent brush stroke, prop move, or edit.",
            ),
            (
                "Redo",
                11,
                UiAction::Redo,
                "Redo Action",
                "Ctrl+Y / 3-Finger Tap",
                "Re-applies an action that you previously undid.",
            ),
            (
                "Menu",
                12,
                UiAction::OpenRadialButton,
                "Radial Quick Wheel",
                "Hold Space / LB",
                "Opens a fast 8-slice pie menu around your cursor for switching tools on touch, mouse, or controller.",
            ),
            (
                "Save",
                13,
                UiAction::SaveScene,
                "Save Scene to Disk",
                "Ctrl+S or F5",
                "Saves your sculpted island, custom toys, and recorded animations to island_dream.ron (press F9 to reload).",
            ),
            (
                "Reset",
                14,
                UiAction::ResetIsland,
                "Restore Starter Island",
                "Safe (Undoable)",
                "Resets the world back to the fresh starter Floating Diorama Island. You can Undo if you change your mind!",
            ),
        ];

        let q_btn_w = 78.0 * s;
        let q_gap = 6.0 * s;
        let q_total_w = 5.0 * q_btn_w + 4.0 * q_gap;
        let q_row_y = if 14.0 * s + q_total_w < start_x - 14.0 * s {
            top_y + 2.0 * s
        } else {
            top_y + tab_h + 12.0 * s
        };
        for (i, (lbl, icon_id, act, t_title, t_key, t_desc)) in quick_buttons.iter().enumerate() {
            let x0 = 14.0 * s + (i as f32) * (q_btn_w + q_gap);
            rects.push(UiRect {
                min: Vec2::new(x0, q_row_y),
                max: Vec2::new(x0 + q_btn_w, q_row_y + 36.0 * s),
                fill_srgb: [0.15, 0.17, 0.24, 0.92],
                border_srgb: [0.46, 0.50, 0.64, 0.65],
                corner_radius: 16.0 * s,
                border_width: 1.4 * s,
                style: UiRectStyle::Button,
                icon: Some(*icon_id),
                label: (*lbl).to_string(),
                right_label: String::new(),
                font_px: 11.4 * s,
                left_align_text: false,
                text_srgb: [0.96, 0.97, 1.0, 1.0],
                hovered: false,
                cast_shadow: true,
                action: Some(*act),
                tooltip: Some(((*t_title).into(), (*t_key).into(), (*t_desc).into())),
            });
        }

        // 3. Top-Right Help Guide Button & Active Input Badge
        let dev_w = 122.0 * s;
        let help_w = 92.0 * s;
        let right_y = if start_x + bar_w + 16.0 * s < w - dev_w - help_w - 24.0 * s {
            top_y + 2.0 * s
        } else {
            q_row_y
        };
        rects.push(UiRect {
            min: Vec2::new(w - dev_w - help_w - 20.0 * s, right_y),
            max: Vec2::new(w - dev_w - 20.0 * s, right_y + 36.0 * s),
            fill_srgb: if self.show_help_guide {
                [0.24, 0.62, 0.88, 0.96]
            } else {
                [0.16, 0.19, 0.28, 0.92]
            },
            border_srgb: [0.58, 0.78, 0.98, 0.78],
            corner_radius: 16.0 * s,
            border_width: 1.5 * s,
            style: UiRectStyle::Button,
            icon: Some(18),
            label: "Guide".to_string(),
            right_label: String::new(),
            font_px: 11.4 * s,
            left_align_text: false,
            text_srgb: [0.98, 0.99, 1.0, 1.0],
            hovered: false,
            cast_shadow: true,
            action: Some(UiAction::ToggleHelp),
            tooltip: Some((
                "Controls & Workflow Guide".into(),
                "Click to Toggle".into(),
                "Shows a handy cheat-sheet for orbiting the camera, sculpting clay, editing toys, and Play Mode.".into(),
            )),
        });

        rects.push(UiRect {
            min: Vec2::new(w - dev_w - 14.0 * s, right_y),
            max: Vec2::new(w - 14.0 * s, right_y + 36.0 * s),
            fill_srgb: [0.11, 0.13, 0.19, 0.86],
            border_srgb: [0.40, 0.62, 0.86, 0.55],
            corner_radius: 16.0 * s,
            border_width: 1.3 * s,
            style: UiRectStyle::Panel,
            icon: None,
            label: device.label().to_string(),
            right_label: String::new(),
            font_px: 11.2 * s,
            left_align_text: false,
            text_srgb: [0.84, 0.93, 1.0, 1.0],
            hovered: false,
            cast_shadow: false,
            action: Some(UiAction::ToggleHelp),
            tooltip: Some((
                "Active Input Mode".into(),
                "Auto-Detected".into(),
                "Automatically switches between Mouse + Keyboard, Multi-Touch gestures, and Gamepad sticks/triggers!".into(),
            )),
        });

        // 4. Glowing Scope-In Banner when editing inside a Custom Toy Object
        let docks_top_y = q_row_y + 48.0 * s;
        if let Some(scoped_pid) = self.scoped_prop_id {
            let prop_name = self
                .scene
                .props
                .iter()
                .find(|p| p.id == scoped_pid)
                .map(|p| p.name.as_str())
                .unwrap_or("Toy");
            let clones = self
                .scene
                .props
                .iter()
                .find(|p| p.id == scoped_pid)
                .map(|pr| {
                    self.scene
                        .props
                        .iter()
                        .filter(|x| x.asset_id == pr.asset_id)
                        .count()
                })
                .unwrap_or(1);
            let badge_lbl = if clones > 1 {
                format!("Editing Inside {prop_name} ({clones} Linked Copies) — Click to Exit")
            } else {
                format!("Editing Inside {prop_name} — Click to Exit Back to World")
            };
            let bw = (460.0 * s).min(w - 32.0 * s);
            let bx0 = (w - bw) * 0.5;
            let by0 = top_y + tab_h + 10.0 * s;
            rects.push(UiRect {
                min: Vec2::new(bx0, by0),
                max: Vec2::new(bx0 + bw, by0 + 36.0 * s),
                fill_srgb: [0.16, 0.58, 0.78, 0.96],
                border_srgb: [0.88, 0.98, 1.0, 0.96],
                corner_radius: 18.0 * s,
                border_width: 1.8 * s,
                style: UiRectStyle::Button,
                icon: Some(8),
                label: badge_lbl,
                right_label: String::new(),
                font_px: 11.6 * s,
                left_align_text: false,
                text_srgb: [0.99, 1.0, 1.0, 1.0],
                hovered: false,
                cast_shadow: true,
                action: Some(UiAction::ToggleScope),
                tooltip: Some((
                    "Exit Toy Edit Mode (Scope Out)".into(),
                    "Enter or Esc".into(),
                    "Leaves solo toy editing and returns your brush to sculpting the main Floating Island.".into(),
                )),
            });
        }

        // 5. Left & Right Contextual Docks (hidden in Play Mode so viewport is wide open!)
        if self.brush.mode != EditorMode::Play {
            let lx0 = 14.0 * s;
            let btn_w = 208.0 * s;
            let lx1 = lx0 + btn_w;
            let btn_h = 36.0 * s;
            let row_step = 42.0 * s;
            let mut y = docks_top_y;

            if self.brush.mode == EditorMode::Sculpt || self.brush.mode == EditorMode::Paint {
                // Measure panel height cleanly
                let op_rows = if self.brush.mode == EditorMode::Sculpt { 2 } else { 0 };
                let brush_card_h = 28.0 * s + ((op_rows + 3) as f32) * row_step + 6.0 * s;
                rects.push(mk_panel(
                    Vec2::new(lx0 - 6.0 * s, y - 6.0 * s),
                    Vec2::new(lx1 + 6.0 * s, y + brush_card_h),
                    18.0,
                ));
                rects.push(mk_header(lx0 + 6.0 * s, lx1, y, "BRUSH SETTINGS"));
                y += 24.0 * s;

                if self.brush.mode == EditorMode::Sculpt {
                    for (op, icon_id, title, key, desc) in [
                        (
                            SculptOp::Add,
                            0u32,
                            "Add Clay",
                            "Left-Drag",
                            "Builds new clay onto surfaces. Melts smoothly into nearby clay based on Soft Blend.",
                        ),
                        (
                            SculptOp::Subtract,
                            1u32,
                            "Carve Away",
                            "Alt + Drag",
                            "Scoops and cuts caves, arches, or dents out of existing clay. Tip: hold Alt anytime to quick-carve!",
                        ),
                    ] {
                        let active = self.brush.op == op;
                        let active_col = if op == SculptOp::Add {
                            [0.92, 0.54, 0.24, 0.97]
                        } else {
                            [0.88, 0.30, 0.38, 0.97]
                        };
                        rects.push(UiRect {
                            min: Vec2::new(lx0, y),
                            max: Vec2::new(lx1, y + btn_h),
                            fill_srgb: if active {
                                active_col
                            } else {
                                [0.16, 0.18, 0.25, 0.92]
                            },
                            border_srgb: if active {
                                [1.0, 0.95, 0.86, 0.95]
                            } else {
                                [0.42, 0.46, 0.58, 0.55]
                            },
                            corner_radius: 15.0 * s,
                            border_width: if active { 1.9 * s } else { 1.3 * s },
                            style: UiRectStyle::Button,
                            icon: Some(icon_id),
                            label: title.to_string(),
                            right_label: String::new(),
                            font_px: 11.8 * s,
                            left_align_text: true,
                            text_srgb: [0.99, 0.99, 1.0, 1.0],
                            hovered: false,
                            cast_shadow: false,
                            action: Some(UiAction::SetOp(op)),
                            tooltip: Some((title.into(), key.into(), desc.into())),
                        });
                        y += row_step;
                    }
                }

                // Shape, Surface Fleck Style, and Symmetry buttons
                let tool_rows: [(String, u32, bool, UiAction, &str, &str, &str); 3] = [
                    (
                        format!("Shape: {}", self.brush.shape.label()),
                        5,
                        false,
                        UiAction::CycleShape,
                        "3D Brush Shape",
                        "Key: F",
                        "Cycles the 3D shape stamped by your brush: Sphere, Soft Box, Capsule, Cylinder, Cone, or Ring.",
                    ),
                    (
                        format!("Style: {}", self.brush.fleck_style.label()),
                        6,
                        false,
                        UiAction::CycleFleck,
                        "Painterly Surface Style",
                        "Key: G",
                        "Picks the living brushstroke finish: Impasto Clay, Wind-Swaying Leaves, 12fps Sketch Boil, Fluffy Cloud, or Glowing Ember!",
                    ),
                    (
                        if self.brush.mirror_x {
                            "Symmetry: ON".to_string()
                        } else {
                            "Symmetry: OFF".to_string()
                        },
                        7,
                        self.brush.mirror_x,
                        UiAction::ToggleMirror,
                        "Mirror X Symmetry",
                        "Key: M",
                        "Mirrors every brush stroke across the center line so left and right sides sculpt together automatically.",
                    ),
                ];

                for (lbl, icon_id, active, act, t_title, t_key, t_desc) in tool_rows {
                    rects.push(UiRect {
                        min: Vec2::new(lx0, y),
                        max: Vec2::new(lx1, y + btn_h),
                        fill_srgb: if active {
                            [0.22, 0.62, 0.86, 0.96]
                        } else {
                            [0.16, 0.18, 0.25, 0.92]
                        },
                        border_srgb: [0.48, 0.56, 0.72, if active { 0.95 } else { 0.55 }],
                        corner_radius: 15.0 * s,
                        border_width: 1.4 * s,
                        style: UiRectStyle::Button,
                        icon: Some(icon_id),
                        label: lbl,
                        right_label: String::new(),
                        font_px: 11.6 * s,
                        left_align_text: true,
                        text_srgb: [0.97, 0.98, 1.0, 1.0],
                        hovered: false,
                        cast_shadow: false,
                        action: Some(act),
                        tooltip: Some((t_title.into(), t_key.into(), t_desc.into())),
                    });
                    y += row_step;
                }

                // Second Left Card: TOY OBJECTS
                y += 14.0 * s;
                let toy_card_h = 28.0 * s + 3.0 * row_step + 6.0 * s;
                rects.push(mk_panel(
                    Vec2::new(lx0 - 6.0 * s, y - 6.0 * s),
                    Vec2::new(lx1 + 6.0 * s, y + toy_card_h),
                    18.0,
                ));
                rects.push(mk_header(lx0 + 6.0 * s, lx1, y, "TOY OBJECTS"));
                y += 24.0 * s;

                let scope_on = self.scoped_prop_id.is_some();
                let toy_rows: [(String, u32, bool, UiAction, &str, &str, &str); 3] = [
                    (
                        if scope_on {
                            "Exit Toy Edit".to_string()
                        } else {
                            "Edit Inside Toy".to_string()
                        },
                        8,
                        scope_on,
                        UiAction::ToggleScope,
                        "Edit Inside Toy (Scope In/Out)",
                        "Key: Enter",
                        "Point at any movable Toy and click this (or press Enter) to sculpt that toy by itself without touching the ground!",
                    ),
                    (
                        "New Toy Object".to_string(),
                        0,
                        false,
                        UiAction::SpawnProp,
                        "Create New Movable Toy",
                        "Key: P",
                        "Spawns a brand-new separate Toy Object at your cursor that you can sculpt, move around, clone, and animate.",
                    ),
                    (
                        "Copy Linked Toy".to_string(),
                        9,
                        false,
                        UiAction::CloneProp,
                        "Duplicate Linked Toy",
                        "Key: C or Ctrl+D",
                        "Creates a linked twin of the selected Toy. Sculpting inside one linked twin updates all of its copies live!",
                    ),
                ];

                for (lbl, icon_id, active, act, t_title, t_key, t_desc) in toy_rows {
                    rects.push(UiRect {
                        min: Vec2::new(lx0, y),
                        max: Vec2::new(lx1, y + btn_h),
                        fill_srgb: if active {
                            [0.18, 0.62, 0.78, 0.96]
                        } else {
                            [0.17, 0.19, 0.28, 0.92]
                        },
                        border_srgb: [0.50, 0.62, 0.80, 0.60],
                        corner_radius: 15.0 * s,
                        border_width: 1.4 * s,
                        style: UiRectStyle::Button,
                        icon: Some(icon_id),
                        label: lbl,
                        right_label: String::new(),
                        font_px: 11.6 * s,
                        left_align_text: true,
                        text_srgb: [0.97, 0.98, 1.0, 1.0],
                        hovered: false,
                        cast_shadow: false,
                        action: Some(act),
                        tooltip: Some((t_title.into(), t_key.into(), t_desc.into())),
                    });
                    y += row_step;
                }
            }

            // Animate Mode Left Dock: Grouped into Toy Assembly + Live Motion!
            if self.brush.mode == EditorMode::Animate {
                let sel_prop = self
                    .selected_prop_id
                    .and_then(|id| self.scene.props.iter().find(|p| p.id == id));
                let prop_gadget = sel_prop.map(|p| p.gadget.label()).unwrap_or("Still");
                let has_conn = sel_prop.and_then(|p| p.connector.as_ref()).is_some();
                let conn_lbl = sel_prop
                    .and_then(|p| p.connector.as_ref())
                    .map(|c| format!("Joint: {}", c.kind.label()))
                    .unwrap_or_default();

                let mut anim_rows: Vec<(String, u32, bool, UiAction, &str, &str, &str)> = vec![
                    (
                        "Edit Toy Clay".into(),
                        8,
                        false,
                        UiAction::ToggleScope,
                        "Sculpt Selected Toy",
                        "Key: Enter",
                        "Switches into solo sculpt mode on the selected Toy so you can reshape or paint its clay details.",
                    ),
                    (
                        "New Toy Object".into(),
                        0,
                        false,
                        UiAction::SpawnProp,
                        "Create New Toy",
                        "Key: P",
                        "Creates a fresh movable Toy Object at your cursor ready to sculpt and animate.",
                    ),
                    (
                        "Copy Linked Toy".into(),
                        9,
                        false,
                        UiAction::CloneProp,
                        "Duplicate Linked Toy",
                        "Key: C",
                        "Duplicates the selected Toy (and its attached limbs!). Edits to its clay update all linked copies.",
                    ),
                    (
                        "Unlink Copy".into(),
                        5,
                        false,
                        UiAction::MakeUnique,
                        "Make Toy Independent",
                        "Key: U",
                        "Unlinks this copy from its twins so sculpting inside it only changes this single Toy.",
                    ),
                    (
                        "Attach Limb".into(),
                        16,
                        false,
                        UiAction::SpawnChildPart,
                        "Attach Jointed Limb / Part",
                        "Key: L",
                        "Spawns a connected child part (like an arm, ear, wing, or wheel) attached to the selected Toy with a joint!",
                    ),
                ];
                if has_conn {
                    anim_rows.push((
                        conn_lbl,
                        16,
                        true,
                        UiAction::CycleConnectorKind,
                        "Change Joint Behavior",
                        "Click to Cycle",
                        "Switches how this attached limb moves on its parent: Bouncy Spring, Spinning Axle, or Swinging Hinge.",
                    ));
                }
                anim_rows.extend([
                    (
                        format!("Auto-Move: {prop_gadget}"),
                        3,
                        prop_gadget != "Still",
                        UiAction::CyclePropGadget,
                        "Automatic Toy Motion",
                        "Click to Cycle",
                        "Adds instant built-in motion without keyframes: Still, Floating Hover, Spin, Bouncy Hop, or Swing!",
                    ),
                    (
                        if self.record_armed {
                            "REC ARMED! Drag Toy".into()
                        } else {
                            "Record Live Path".into()
                        },
                        15,
                        self.record_armed,
                        UiAction::ToggleRecord,
                        "Live Action Recorder",
                        "Key: R",
                        "Click to arm recording, then grab and wave any Toy in 3D! When you let go, your motion loops smoothly forever.",
                    ),
                    (
                        "Erase Recording".into(),
                        14,
                        false,
                        UiAction::ClearPropClip,
                        "Clear Recorded Path",
                        "Click",
                        "Removes any recorded movement path from the selected Toy so it stays still where placed.",
                    ),
                    (
                        "Delete Toy".into(),
                        17,
                        false,
                        UiAction::DeleteProp,
                        "Delete Selected Toy",
                        "Key: Delete",
                        "Removes the selected Toy (and any limbs attached to it) from the scene. Undoable with Ctrl+Z.",
                    ),
                ]);

                let anim_row_h = 34.0 * s;
                let anim_step = 38.5 * s;
                let anim_card_h = 26.0 * s + (anim_rows.len() as f32) * anim_step + 6.0 * s;
                rects.push(mk_panel(
                    Vec2::new(lx0 - 6.0 * s, y - 6.0 * s),
                    Vec2::new(lx1 + 6.0 * s, y + anim_card_h),
                    18.0,
                ));
                rects.push(mk_header(lx0 + 6.0 * s, lx1, y, "ANIMATE & JOINTS"));
                y += 22.0 * s;

                for (lbl, icon_id, highlight, act, t_title, t_key, t_desc) in anim_rows {
                    let fill = if act == UiAction::ToggleRecord && highlight {
                        [0.90, 0.24, 0.32, 0.97]
                    } else if act == UiAction::DeleteProp {
                        [0.32, 0.15, 0.20, 0.90]
                    } else if highlight {
                        [0.48, 0.34, 0.80, 0.95]
                    } else {
                        [0.18, 0.17, 0.27, 0.92]
                    };
                    rects.push(UiRect {
                        min: Vec2::new(lx0, y),
                        max: Vec2::new(lx1, y + anim_row_h),
                        fill_srgb: fill,
                        border_srgb: [0.68, 0.58, 0.90, 0.62],
                        corner_radius: 14.0 * s,
                        border_width: 1.3 * s,
                        style: UiRectStyle::Button,
                        icon: Some(icon_id),
                        label: lbl,
                        right_label: String::new(),
                        font_px: 11.4 * s,
                        left_align_text: true,
                        text_srgb: [0.98, 0.98, 1.0, 1.0],
                        hovered: false,
                        cast_shadow: false,
                        action: Some(act),
                        tooltip: Some((t_title.into(), t_key.into(), t_desc.into())),
                    });
                    y += anim_step;
                }
            }

            // 6. Right-Side Tactile Sliders Card (`FEEL & CAMERA`) — High-Contrast Full-Width Text!
            let slider_w = 228.0 * s;
            let sx1 = w - 14.0 * s;
            let sx0 = (sx1 - slider_w).max(lx1 + 24.0 * s);
            let mut sy = docks_top_y;
            let s_row_h = 34.0 * s;
            let s_step = 42.0 * s;
            let slider_card_h = 28.0 * s + 4.0 * s_step + 6.0 * s;

            rects.push(mk_panel(
                Vec2::new(sx0 - 6.0 * s, sy - 6.0 * s),
                Vec2::new(sx1 + 6.0 * s, sy + slider_card_h),
                18.0,
            ));
            rects.push(mk_header(sx0 + 6.0 * s, sx1, sy, "FEEL & CAMERA"));
            sy += 24.0 * s;

            let sliders: [(&str, f32, UiAction, [f32; 4], &str, &str, &str); 4] = [
                (
                    "Brush Size",
                    (self.brush.size - 0.12) / (3.6 - 0.12),
                    UiAction::DragSliderSize,
                    [0.86, 0.54, 0.18, 0.92],
                    "Brush Size",
                    "Shift + Scroll or [ ]",
                    "Adjusts the 3D size of your sculpting/painting brush (in Animate mode, Shift+Scroll scales the selected Toy).",
                ),
                (
                    "Soft Blend",
                    (self.brush.blend - 0.03) / (1.1 - 0.03),
                    UiAction::DragSliderBlend,
                    [0.24, 0.66, 0.44, 0.92],
                    "Soft Clay Blend",
                    "Ctrl + Scroll",
                    "Controls how smoothly new strokes melt into existing clay. Low = crisp sharp edges; High = soft gooey clay!",
                ),
                (
                    "Fluffiness",
                    self.brush.looseness,
                    UiAction::DragSliderLoose,
                    [0.82, 0.36, 0.54, 0.92],
                    "Painterly Fluffiness (Looseness)",
                    "Drag Slider",
                    "Controls brushstroke looseness! Low = solid smooth clay; High = billowing painterly leaves, fur, & clouds.",
                ),
                (
                    "Focus Blur",
                    self.scene.dof_amount,
                    UiAction::DragSliderDof,
                    [0.32, 0.56, 0.86, 0.92],
                    "Miniature Camera Blur (Bokeh)",
                    "Drag Slider",
                    "Adjusts tilt-shift background blur (Depth of Field) for a warm storybook miniature diorama look.",
                ),
            ];

            for (name, norm_val, act, bar_col, t_title, t_key, t_desc) in sliders {
                let t = norm_val.clamp(0.0, 1.0);
                let pct_txt = format!("{}%", (t * 100.0).round() as i32);

                // Recessed dark track
                rects.push(UiRect {
                    min: Vec2::new(sx0, sy),
                    max: Vec2::new(sx1, sy + s_row_h),
                    fill_srgb: [0.09, 0.11, 0.16, 0.95],
                    border_srgb: [0.42, 0.46, 0.60, 0.65],
                    corner_radius: 15.0 * s,
                    border_width: 1.4 * s,
                    style: UiRectStyle::Panel,
                    icon: None,
                    label: String::new(),
                    right_label: String::new(),
                    font_px: 13.8 * s,
                    left_align_text: true,
                    text_srgb: [1.0; 4],
                    hovered: false,
                    cast_shadow: false,
                    action: Some(act),
                    tooltip: Some((t_title.into(), t_key.into(), t_desc.into())),
                });

                // Colored progress fill bar (pure visual, non-overflowing!)
                let min_fill = 14.0 * s;
                let fill_w = min_fill + (sx1 - sx0 - min_fill) * t;
                rects.push(UiRect {
                    min: Vec2::new(sx0 + 2.0 * s, sy + 2.0 * s),
                    max: Vec2::new(sx0 + fill_w - 2.0 * s, sy + s_row_h - 2.0 * s),
                    fill_srgb: bar_col,
                    border_srgb: [1.0, 1.0, 1.0, 0.25],
                    corner_radius: 13.0 * s,
                    border_width: 1.0 * s,
                    style: UiRectStyle::FlatAccent,
                    icon: None,
                    label: String::new(),
                    right_label: String::new(),
                    font_px: 13.8 * s,
                    left_align_text: true,
                    text_srgb: [1.0; 4],
                    hovered: false,
                    cast_shadow: false,
                    action: None,
                    tooltip: None,
                });

                // Full-width crisp white text overlay with dark halo (readable over BOTH dark track and colored bar!)
                rects.push(UiRect {
                    min: Vec2::new(sx0, sy),
                    max: Vec2::new(sx1, sy + s_row_h),
                    fill_srgb: [0.0, 0.0, 0.0, 0.0],
                    border_srgb: [0.0, 0.0, 0.0, 0.0],
                    corner_radius: 15.0 * s,
                    border_width: 0.0,
                    style: UiRectStyle::Panel,
                    icon: None,
                    label: name.to_string(),
                    right_label: pct_txt,
                    font_px: 11.4 * s,
                    left_align_text: true,
                    text_srgb: [0.99, 0.99, 1.0, 1.0],
                    hovered: false,
                    cast_shadow: false,
                    action: Some(act),
                    tooltip: Some((t_title.into(), t_key.into(), t_desc.into())),
                });

                sy += s_step;
            }

            // 6B. Right-Side `SCENE OBJECTS` Outliner Card (World Canvas + Hierarchical Toy Tree + Quick Actions!)
            let swatch_top_reserve = h - 74.0 * s;
            let outliner_y0 = sy + 14.0 * s;
            let avail_outliner_h = swatch_top_reserve - outliner_y0;

            if avail_outliner_h > 44.0 * s {
                let prop_count = self.scene.props.len();
                if !self.outliner_open || avail_outliner_h < 145.0 * s {
                    rects.push(UiRect {
                        min: Vec2::new(sx0, outliner_y0),
                        max: Vec2::new(sx1, outliner_y0 + 34.0 * s),
                        fill_srgb: [0.14, 0.16, 0.24, 0.92],
                        border_srgb: [0.46, 0.54, 0.72, 0.65],
                        corner_radius: 15.0 * s,
                        border_width: 1.4 * s,
                        style: UiRectStyle::Button,
                        icon: Some(5),
                        label: format!("Scene Objects ({prop_count})"),
                        right_label: "Show".into(),
                        font_px: 11.2 * s,
                        left_align_text: true,
                        text_srgb: [0.95, 0.97, 1.0, 1.0],
                        hovered: false,
                        cast_shadow: true,
                        action: Some(UiAction::ToggleOutliner),
                        tooltip: Some((
                            "Expand Scene Objects Outliner".into(),
                            "Click to Open".into(),
                            "Shows a hierarchical list of the World Island and all movable Toy Objects in your scene.".into(),
                        )),
                    });
                } else {
                    let tree = self.build_outliner_tree();
                    let row_h = 28.0 * s;
                    let row_step_o = 31.5 * s;
                    // Reserve header (24) + World row (32) + bottom action strip (36) + optional pager (28)
                    let chrome_h = 24.0 * s + row_step_o + 40.0 * s;
                    let raw_max_rows = ((avail_outliner_h - chrome_h) / row_step_o).floor().max(1.0) as usize;
                    let needs_pager = tree.len() > raw_max_rows;
                    let max_rows = if needs_pager {
                        ((avail_outliner_h - chrome_h - 28.0 * s) / row_step_o)
                            .floor()
                            .max(1.0) as usize
                    } else {
                        raw_max_rows
                    };

                    let total_pages = tree.len().div_ceil(max_rows).max(1);
                    if self.outliner_page >= total_pages {
                        self.outliner_page = total_pages - 1;
                    }
                    let start_i = self.outliner_page * max_rows;
                    let end_i = (start_i + max_rows).min(tree.len());
                    let visible_slice = &tree[start_i..end_i];

                    let has_sel_prop = self
                        .selected_prop_id
                        .and_then(|id| self.scene.props.iter().find(|p| p.id == id))
                        .is_some();
                    let pager_h = if needs_pager { 28.0 * s } else { 0.0 };
                    let footer_h = if has_sel_prop { 36.0 * s } else { 4.0 * s };
                    let card_h = 24.0 * s
                        + (1 + visible_slice.len()) as f32 * row_step_o
                        + pager_h
                        + footer_h
                        + 6.0 * s;

                    rects.push(mk_panel(
                        Vec2::new(sx0 - 6.0 * s, outliner_y0 - 6.0 * s),
                        Vec2::new(sx1 + 6.0 * s, outliner_y0 + card_h),
                        18.0,
                    ));

                    // Header + Hide toggle chip
                    rects.push(mk_header(
                        sx0 + 6.0 * s,
                        sx1 - 54.0 * s,
                        outliner_y0,
                        &format!("SCENE OBJECTS ({prop_count})"),
                    ));
                    rects.push(UiRect {
                        min: Vec2::new(sx1 - 48.0 * s, outliner_y0 - 1.0 * s),
                        max: Vec2::new(sx1, outliner_y0 + 19.0 * s),
                        fill_srgb: [0.16, 0.19, 0.28, 0.88],
                        border_srgb: [0.44, 0.50, 0.66, 0.55],
                        corner_radius: 9.0 * s,
                        border_width: 1.0 * s,
                        style: UiRectStyle::Button,
                        icon: None,
                        label: "Hide".into(),
                        right_label: String::new(),
                        font_px: 9.8 * s,
                        left_align_text: false,
                        text_srgb: [0.84, 0.89, 0.98, 1.0],
                        hovered: false,
                        cast_shadow: false,
                        action: Some(UiAction::ToggleOutliner),
                        tooltip: Some((
                            "Collapse Outliner".into(),
                            "Click".into(),
                            "Hides the Scene Objects list to free up right-side screen space.".into(),
                        )),
                    });

                    let mut oy = outliner_y0 + 23.0 * s;

                    // Fixed Row 0: World Island Canvas!
                    let world_active = self.scoped_prop_id.is_none()
                        && (self.brush.mode == EditorMode::Sculpt
                            || self.brush.mode == EditorMode::Paint);
                    rects.push(UiRect {
                        min: Vec2::new(sx0, oy),
                        max: Vec2::new(sx1, oy + row_h),
                        fill_srgb: if world_active {
                            [0.72, 0.44, 0.22, 0.95]
                        } else {
                            [0.13, 0.15, 0.22, 0.90]
                        },
                        border_srgb: if world_active {
                            [1.0, 0.92, 0.72, 0.92]
                        } else {
                            [0.36, 0.40, 0.52, 0.50]
                        },
                        corner_radius: 12.0 * s,
                        border_width: if world_active { 1.6 * s } else { 1.1 * s },
                        style: UiRectStyle::Button,
                        icon: Some(14),
                        label: "World Island".into(),
                        right_label: "Canvas".into(),
                        font_px: 10.8 * s,
                        left_align_text: true,
                        text_srgb: [0.98, 0.98, 1.0, 1.0],
                        hovered: false,
                        cast_shadow: false,
                        action: Some(UiAction::OutlinerSelectWorld),
                        tooltip: Some((
                            "World Island Canvas".into(),
                            "Base Terrain".into(),
                            "The shared floating island ground, hills, caves, trees, and clouds. Click to sculpt the main island.".into(),
                        )),
                    });
                    oy += row_step_o;

                    // Hierarchical Toy Object rows
                    for &(prop_id, depth) in visible_slice {
                        let Some(prop) = self.scene.props.iter().find(|p| p.id == prop_id) else {
                            continue;
                        };
                        let is_sel = self.selected_prop_id == Some(prop_id);
                        let is_scoped = self.scoped_prop_id == Some(prop_id);
                        let indent = (depth as f32 * 13.0 * s).min(26.0 * s);

                        let clones = self
                            .scene
                            .props
                            .iter()
                            .filter(|x| x.asset_id == prop.asset_id)
                            .count();

                        let badge = if is_scoped {
                            "Clay".to_string()
                        } else if prop.clip.is_some() {
                            "Rec".to_string()
                        } else if clones > 1 {
                            format!("{clones}x")
                        } else if prop.gadget != MotionGadget::None {
                            prop.gadget.label().to_string()
                        } else {
                            String::new()
                        };

                        let max_name_chars = if badge.is_empty() {
                            18usize.saturating_sub(depth * 2)
                        } else {
                            13usize.saturating_sub(depth * 2)
                        };
                        let short_name = truncate_label_chars(&prop.name, max_name_chars.max(6));
                        let icon_id = if is_scoped {
                            8
                        } else if depth > 0 {
                            16
                        } else {
                            5
                        };

                        let fill = if is_scoped {
                            [0.16, 0.58, 0.78, 0.97]
                        } else if is_sel {
                            [0.44, 0.34, 0.78, 0.96]
                        } else {
                            [0.15, 0.17, 0.25, 0.90]
                        };

                        let tip_sub = if is_sel {
                            "Click Again for 3D Gizmo"
                        } else {
                            "Click to Select & Frame"
                        };
                        let parent_info = prop
                            .connector
                            .as_ref()
                            .and_then(|c| {
                                self.scene
                                    .props
                                    .iter()
                                    .find(|p| p.id == c.parent_id)
                                    .map(|par| format!("Attached to {} ({}). ", par.name, c.kind.label()))
                            })
                            .unwrap_or_default();
                        let clone_info = if clones > 1 {
                            format!("Shares linked clay with {clones} copies. ")
                        } else {
                            String::new()
                        };

                        rects.push(UiRect {
                            min: Vec2::new(sx0 + indent, oy),
                            max: Vec2::new(sx1, oy + row_h),
                            fill_srgb: fill,
                            border_srgb: if is_sel || is_scoped {
                                [1.0, 0.94, 0.76, 0.96]
                            } else {
                                [0.38, 0.42, 0.56, 0.50]
                            },
                            corner_radius: 12.0 * s,
                            border_width: if is_sel || is_scoped { 1.7 * s } else { 1.1 * s },
                            style: UiRectStyle::Button,
                            icon: Some(icon_id),
                            label: short_name,
                            right_label: badge,
                            font_px: 10.6 * s,
                            left_align_text: true,
                            text_srgb: [0.98, 0.99, 1.0, 1.0],
                            hovered: false,
                            cast_shadow: false,
                            action: Some(UiAction::OutlinerSelectProp(prop_id)),
                            tooltip: Some((
                                format!("{} (Toy #{prop_id})", prop.name),
                                tip_sub.into(),
                                format!("{parent_info}{clone_info}Click to select and center the camera on this Toy. Use Move / Clay / Del below!"),
                            )),
                        });
                        oy += row_step_o;
                    }

                    // Optional Pagination Row (`< Prev` | `1/2` | `Next >`)
                    if needs_pager {
                        let pw = 56.0 * s;
                        rects.push(UiRect {
                            min: Vec2::new(sx0, oy),
                            max: Vec2::new(sx0 + pw, oy + 24.0 * s),
                            fill_srgb: [0.16, 0.18, 0.27, 0.92],
                            border_srgb: [0.45, 0.50, 0.65, 0.55],
                            corner_radius: 10.0 * s,
                            border_width: 1.1 * s,
                            style: UiRectStyle::Button,
                            icon: None,
                            label: "< Prev".into(),
                            right_label: String::new(),
                            font_px: 9.8 * s,
                            left_align_text: false,
                            text_srgb: [0.92, 0.95, 1.0, 1.0],
                            hovered: false,
                            cast_shadow: false,
                            action: Some(UiAction::OutlinerPagePrev),
                            tooltip: Some((
                                "Previous Page".into(),
                                "Click".into(),
                                "Shows earlier Toy Objects in the Scene Outliner.".into(),
                            )),
                        });
                        rects.push(UiRect {
                            min: Vec2::new(sx0 + pw, oy),
                            max: Vec2::new(sx1 - pw, oy + 24.0 * s),
                            fill_srgb: [0.0; 4],
                            border_srgb: [0.0; 4],
                            corner_radius: 4.0,
                            border_width: 0.0,
                            style: UiRectStyle::Panel,
                            icon: None,
                            label: format!("{}/{}", self.outliner_page + 1, total_pages),
                            right_label: String::new(),
                            font_px: 9.8 * s,
                            left_align_text: false,
                            text_srgb: [0.76, 0.82, 0.94, 1.0],
                            hovered: false,
                            cast_shadow: false,
                            action: None,
                            tooltip: None,
                        });
                        rects.push(UiRect {
                            min: Vec2::new(sx1 - pw, oy),
                            max: Vec2::new(sx1, oy + 24.0 * s),
                            fill_srgb: [0.16, 0.18, 0.27, 0.92],
                            border_srgb: [0.45, 0.50, 0.65, 0.55],
                            corner_radius: 10.0 * s,
                            border_width: 1.1 * s,
                            style: UiRectStyle::Button,
                            icon: None,
                            label: "Next >".into(),
                            right_label: String::new(),
                            font_px: 9.8 * s,
                            left_align_text: false,
                            text_srgb: [0.92, 0.95, 1.0, 1.0],
                            hovered: false,
                            cast_shadow: false,
                            action: Some(UiAction::OutlinerPageNext),
                            tooltip: Some((
                                "Next Page".into(),
                                "Click".into(),
                                "Shows more Toy Objects in the Scene Outliner.".into(),
                            )),
                        });
                        oy += 28.0 * s;
                    }

                    // Quick Action Strip for Selected Toy: `[Move]` | `[Clay / Exit]` | `[Del]`
                    if let Some(sel_id) = self
                        .selected_prop_id
                        .filter(|id| self.scene.props.iter().any(|p| p.id == *id))
                    {
                        oy += 4.0 * s;
                        let agap = 5.0 * s;
                        let aw = (sx1 - sx0 - agap * 2.0) / 3.0;
                        let ah = 29.0 * s;
                        let is_scoped_sel = self.scoped_prop_id == Some(sel_id);
                        let is_move_active =
                            self.brush.mode == EditorMode::Animate && !is_scoped_sel;

                        // [Move] button
                        rects.push(UiRect {
                            min: Vec2::new(sx0, oy),
                            max: Vec2::new(sx0 + aw, oy + ah),
                            fill_srgb: if is_move_active {
                                [0.56, 0.38, 0.86, 0.97]
                            } else {
                                [0.18, 0.20, 0.30, 0.94]
                            },
                            border_srgb: [0.64, 0.56, 0.90, 0.72],
                            corner_radius: 12.0 * s,
                            border_width: 1.3 * s,
                            style: UiRectStyle::Button,
                            icon: Some(3),
                            label: "Move".into(),
                            right_label: String::new(),
                            font_px: 10.4 * s,
                            left_align_text: false,
                            text_srgb: [0.98, 0.99, 1.0, 1.0],
                            hovered: false,
                            cast_shadow: false,
                            action: Some(UiAction::OutlinerMoveProp(sel_id)),
                            tooltip: Some((
                                "Move, Rotate & Scale Toy".into(),
                                "3D Gizmo".into(),
                                "Switches to Animate mode with the 3D ring & scale orb around this Toy so you can drag, spin, or resize it.".into(),
                            )),
                        });

                        // [Clay] / [Exit] Scope button
                        rects.push(UiRect {
                            min: Vec2::new(sx0 + aw + agap, oy),
                            max: Vec2::new(sx0 + aw * 2.0 + agap, oy + ah),
                            fill_srgb: if is_scoped_sel {
                                [0.16, 0.60, 0.80, 0.97]
                            } else {
                                [0.18, 0.22, 0.32, 0.94]
                            },
                            border_srgb: [0.48, 0.76, 0.94, 0.75],
                            corner_radius: 12.0 * s,
                            border_width: 1.3 * s,
                            style: UiRectStyle::Button,
                            icon: Some(8),
                            label: if is_scoped_sel {
                                "Exit".into()
                            } else {
                                "Clay".into()
                            },
                            right_label: String::new(),
                            font_px: 10.4 * s,
                            left_align_text: false,
                            text_srgb: [0.98, 0.99, 1.0, 1.0],
                            hovered: false,
                            cast_shadow: false,
                            action: Some(UiAction::OutlinerEditPropClay(sel_id)),
                            tooltip: Some((
                                "Sculpt Inside This Toy".into(),
                                "Key: Enter".into(),
                                "Enters (or exits) solo clay sculpting on this Toy so your brush reshapes or paints it without touching the island.".into(),
                            )),
                        });

                        // [Del] button
                        rects.push(UiRect {
                            min: Vec2::new(sx0 + (aw + agap) * 2.0, oy),
                            max: Vec2::new(sx1, oy + ah),
                            fill_srgb: [0.34, 0.16, 0.21, 0.93],
                            border_srgb: [0.82, 0.42, 0.48, 0.68],
                            corner_radius: 12.0 * s,
                            border_width: 1.3 * s,
                            style: UiRectStyle::Button,
                            icon: Some(17),
                            label: "Del".into(),
                            right_label: String::new(),
                            font_px: 10.4 * s,
                            left_align_text: false,
                            text_srgb: [1.0, 0.94, 0.95, 1.0],
                            hovered: false,
                            cast_shadow: false,
                            action: Some(UiAction::DeleteProp),
                            tooltip: Some((
                                "Delete Selected Toy".into(),
                                "Key: Delete".into(),
                                "Deletes this Toy (and any attached child parts) from the scene. Undoable anytime with Ctrl+Z.".into(),
                            )),
                        });
                    }
                }
            }

            // 7. Bottom Clay Color Palette Tray (12 Named Colors with Tooltips!)
            const COLOR_NAMES: [&str; 12] = [
                "Warm Terracotta Clay",
                "Sunlit Ochre Gold",
                "Meadow Moss Green",
                "Deep Pine Teal",
                "Storybook Sky Blue",
                "Twilight Lavender",
                "Coral Blossom Rose",
                "Cloud Cream White",
                "Tree Bark Umber",
                "Carved Cliff Slate",
                "Charcoal Sketch Ink",
                "Lantern Amber Glow",
            ];
            let swatch_size = 36.0 * s;
            let sw_gap = 7.0 * s;
            let total_pw = PALETTE_SRGB.len() as f32 * (swatch_size + sw_gap) - sw_gap;
            let px0 = (w - total_pw) * 0.5;
            let py0 = h - swatch_size - 16.0 * s;

            rects.push(mk_panel(
                Vec2::new(px0 - 10.0 * s, py0 - 8.0 * s),
                Vec2::new(px0 + total_pw + 10.0 * s, py0 + swatch_size + 8.0 * s),
                22.0,
            ));

            for (idx, col) in PALETTE_SRGB.iter().enumerate() {
                let x = px0 + (idx as f32) * (swatch_size + sw_gap);
                let selected = self.brush.palette_index == idx;
                let lift = if selected { 6.0 * s } else { 0.0 };
                let cname = COLOR_NAMES[idx % COLOR_NAMES.len()];
                rects.push(UiRect {
                    min: Vec2::new(x, py0 - lift),
                    max: Vec2::new(x + swatch_size, py0 + swatch_size - lift),
                    fill_srgb: [col[0], col[1], col[2], 1.0],
                    border_srgb: if selected {
                        [1.0, 0.99, 0.90, 1.0]
                    } else {
                        [0.08, 0.09, 0.12, 0.85]
                    },
                    corner_radius: 18.0 * s,
                    border_width: if selected { 2.6 * s } else { 1.4 * s },
                    style: UiRectStyle::FlatAccent,
                    icon: None,
                    label: String::new(),
                    right_label: String::new(),
                    font_px: 12.0 * s,
                    left_align_text: false,
                    text_srgb: [1.0; 4],
                    hovered: false,
                    cast_shadow: selected,
                    action: Some(UiAction::SelectColor(idx)),
                    tooltip: Some((
                        cname.into(),
                        "Click to Select".into(),
                        "Sets the clay & brushstroke color for new Sculpt and Paint strokes.".into(),
                    )),
                });
            }
        }

        let banner = if self.status_timer > 0.0 {
            self.status_toast.clone()
        } else {
            String::new()
        };

        UiFrame {
            ui_scale: s,
            rects,
            radial: None,
            tooltip: None,
            help_open: self.show_help_guide,
            banner_status: banner,
        }
    }

    fn process_ui_interactions(
        &mut self,
        input: &FrameInput,
        camera: &mut CameraRig,
        ui: &mut UiFrame,
    ) -> bool {
        let s = ui.ui_scale.max(0.72);
        let radial_active = input.radial_open || self.radial_button_open;
        if radial_active {
            let margin = 155.0 * s;
            if self.radial_anchor_px == Vec2::ZERO
                || (input.radial_open && !self.radial_button_open && self.radial_hovered.is_none())
            {
                self.radial_anchor_px = input
                    .imp_px
                    .clamp(Vec2::splat(margin), (input.viewport - Vec2::splat(margin)).max(Vec2::splat(margin)));
            }

            let dir_vec = if let Some(stick) = input.radial_stick_dir {
                Some(stick)
            } else {
                let delta = input.imp_px - self.radial_anchor_px;
                if delta.length() > 30.0 * s {
                    Some(delta.normalize())
                } else {
                    None
                }
            };

            self.radial_hovered = dir_vec.map(|d| {
                let angle = d.y.atan2(d.x);
                let norm = (angle + std::f32::consts::PI * 2.5).rem_euclid(std::f32::consts::TAU);
                (((norm / std::f32::consts::TAU) * 8.0).round() as usize) % 8
            });

            let scope_lbl = if self.scoped_prop_id.is_some() {
                "Exit Toy"
            } else {
                "Edit Toy"
            };
            let wheel_r = 128.0 * s;

            ui.radial = Some(RadialMenuVisual {
                center_px: self.radial_anchor_px,
                outer_radius: wheel_r,
                hovered_slice: self.radial_hovered,
                labels: [
                    "Add Clay",
                    "Carve",
                    "Paint",
                    scope_lbl,
                    "Copy Toy",
                    "+ Limb",
                    "Symmetry",
                    "Play!",
                ],
            });

            if let Some(slice) = self.radial_hovered {
                let (t_title, t_desc) = match slice {
                    0 => ("Add Clay", "Build smooth new clay onto surfaces."),
                    1 => ("Carve Away", "Cut tunnels, caves, or dents out of clay."),
                    2 => ("Paint & Coat", "Color & re-style clay without changing its shape."),
                    3 => ("Edit Inside Toy", "Enter or exit solo sculpting on a movable Toy Object."),
                    4 => ("Copy Linked Toy", "Duplicate a Toy so all linked copies update together."),
                    5 => ("Attach Limb", "Add a spring/hinge-connected limb to the selected Toy."),
                    6 => ("Mirror Symmetry", "Toggle left/right X-axis mirror sculpting."),
                    _ => ("Play Mode!", "Drop in with a playable character and explore your world!"),
                };
                ui.tooltip = Some(TooltipCard {
                    anchor_min: self.radial_anchor_px - Vec2::splat(wheel_r),
                    anchor_max: self.radial_anchor_px + Vec2::splat(wheel_r),
                    title: t_title.into(),
                    shortcut: "Release to Pick".into(),
                    description: t_desc.into(),
                });
            }

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

        if let Some(slider_act) = self.active_slider {
            if input.primary_down {
                self.apply_slider_drag(slider_act, input.imp_px.x, input.viewport.x, s);
                return true;
            } else {
                self.active_slider = None;
            }
        }

        let mut hit_idx = None;
        for (idx, rect) in ui.rects.iter().enumerate().rev() {
            if rect.action.is_some()
                && input.imp_px.x >= rect.min.x
                && input.imp_px.x <= rect.max.x
                && input.imp_px.y >= rect.min.y
                && input.imp_px.y <= rect.max.y
            {
                hit_idx = Some(idx);
                break;
            }
        }

        if let Some(idx) = hit_idx {
            let act = ui.rects[idx].action.unwrap();
            // Apply tactile hover highlight & populate floating TooltipCard!
            {
                let r = &mut ui.rects[idx];
                r.hovered = true;
                if r.style == UiRectStyle::Button || r.style == UiRectStyle::FlatAccent {
                    r.min.y -= 1.5 * s;
                    r.max.y -= 1.5 * s;
                    r.fill_srgb[0] = (r.fill_srgb[0] + 0.08).min(1.0);
                    r.fill_srgb[1] = (r.fill_srgb[1] + 0.08).min(1.0);
                    r.fill_srgb[2] = (r.fill_srgb[2] + 0.10).min(1.0);
                    r.border_srgb = [1.0, 0.92, 0.68, 0.98];
                    r.border_width = (r.border_width + 0.7 * s).min(3.2 * s);
                }
                if let Some((title, shortcut, description)) = r.tooltip.clone() {
                    ui.tooltip = Some(TooltipCard {
                        anchor_min: r.min,
                        anchor_max: r.max,
                        title,
                        shortcut,
                        description,
                    });
                }
            }

            if input.primary_just_pressed {
                match act {
                    UiAction::DragSliderSize
                    | UiAction::DragSliderBlend
                    | UiAction::DragSliderLoose
                    | UiAction::DragSliderDof => {
                        self.active_slider = Some(act);
                        self.apply_slider_drag(act, input.imp_px.x, input.viewport.x, s);
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
                        self.set_toast(format!("Surface Style: {}", self.brush.fleck_style.label()));
                    }
                    UiAction::SelectColor(cidx) => {
                        self.brush.palette_index = cidx;
                    }
                    UiAction::ToggleMirror => {
                        self.brush.mirror_x = !self.brush.mirror_x;
                    }
                    UiAction::Undo => self.undo(),
                    UiAction::Redo => self.redo(),
                    UiAction::SpawnProp => self.spawn_prop_at_imp(camera),
                    UiAction::ToggleScope => self.toggle_scope_in_out(camera),
                    UiAction::CloneProp => self.clone_selected_prop(),
                    UiAction::MakeUnique => self.make_selected_prop_unique(),
                    UiAction::SpawnChildPart => self.spawn_connected_child_part(camera),
                    UiAction::CycleConnectorKind => {
                        if let Some(pid) = self.selected_prop_id {
                            if let Some(p) = self.scene.props.iter_mut().find(|p| p.id == pid) {
                                if let Some(conn) = &mut p.connector {
                                    conn.kind = conn.kind.next();
                                    let lbl = conn.kind.label();
                                    self.set_toast(format!("Connector Joint: {lbl}"));
                                }
                            }
                        }
                    }
                    UiAction::DeleteProp => self.delete_selected_prop(),
                    UiAction::CyclePropGadget => {
                        if let Some(pid) = self.selected_prop_id {
                            if let Some(p) = self.scene.props.iter_mut().find(|p| p.id == pid) {
                                p.gadget = p.gadget.next();
                                let lbl = p.gadget.label();
                                self.set_toast(format!("Auto-Motion: {lbl}"));
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
                    UiAction::ToggleHelp => {
                        self.show_help_guide = !self.show_help_guide;
                    }
                    UiAction::ToggleOutliner => {
                        self.outliner_open = !self.outliner_open;
                    }
                    UiAction::OutlinerSelectWorld => {
                        self.finish_stroke();
                        self.scoped_prop_id = None;
                        if self.brush.mode == EditorMode::Animate {
                            self.brush.mode = EditorMode::Sculpt;
                        }
                        camera.target_focus = Vec3::new(0.0, 0.9, 0.0);
                        camera.target_distance = camera.target_distance.clamp(9.0, 16.0);
                        self.set_toast("Selected World Island Canvas — Drag to sculpt or paint the island!");
                    }
                    UiAction::OutlinerSelectProp(pid) => {
                        self.finish_stroke();
                        let already_selected = self.selected_prop_id == Some(pid);
                        self.selected_prop_id = Some(pid);
                        if let Some(prop) = self.scene.props.iter().find(|p| p.id == pid).cloned() {
                            let (p_pos, _) = self.scene.evaluate_prop_pose(pid, self.time);
                            camera.target_focus = p_pos;
                            camera.target_distance = (prop.uniform_scale * 6.8).clamp(4.5, 13.0);

                            if self.scoped_prop_id.is_some() && self.scoped_prop_id != Some(pid) {
                                self.scoped_prop_id = Some(pid);
                                self.set_toast(format!("Switched Clay Edit to {}", prop.name));
                            } else if already_selected && self.scoped_prop_id.is_none() {
                                self.brush.mode = EditorMode::Animate;
                                self.set_toast(format!(
                                    "{} — Drag body to move, ring to rotate, or orb to resize!",
                                    prop.name
                                ));
                            } else {
                                self.set_toast(format!(
                                    "Selected {} — Click Move to pose, or Clay to sculpt inside it!",
                                    prop.name
                                ));
                            }
                        }
                    }
                    UiAction::OutlinerMoveProp(pid) => {
                        self.finish_stroke();
                        self.selected_prop_id = Some(pid);
                        self.scoped_prop_id = None;
                        self.brush.mode = EditorMode::Animate;
                        if let Some(prop) = self.scene.props.iter().find(|p| p.id == pid) {
                            let (p_pos, _) = self.scene.evaluate_prop_pose(pid, self.time);
                            camera.target_focus = p_pos;
                            camera.target_distance = (prop.uniform_scale * 6.8).clamp(4.5, 13.0);
                            self.set_toast(format!(
                                "Moving {} — Drag body to move, turquoise ring to rotate, gold orb to scale!",
                                prop.name
                            ));
                        }
                    }
                    UiAction::OutlinerEditPropClay(pid) => {
                        self.finish_stroke();
                        if self.scoped_prop_id == Some(pid) {
                            self.scope_out();
                        } else {
                            self.selected_prop_id = Some(pid);
                            self.scoped_prop_id = None;
                            self.imp.hovered_prop_id = pid;
                            self.toggle_scope_in_out(camera);
                        }
                    }
                    UiAction::OutlinerPagePrev => {
                        self.outliner_page = self.outliner_page.saturating_sub(1);
                    }
                    UiAction::OutlinerPageNext => {
                        self.outliner_page += 1;
                    }
                }
            }
            return true;
        }

        false
    }

    fn apply_slider_drag(&mut self, slider: UiAction, cursor_x: f32, viewport_w: f32, s: f32) {
        let lx1 = 14.0 * s + 208.0 * s;
        let slider_w = 228.0 * s;
        let sx1 = viewport_w - 14.0 * s;
        let sx0 = (sx1 - slider_w).max(lx1 + 24.0 * s);
        let t = ((cursor_x - sx0) / (sx1 - sx0).max(1.0)).clamp(0.0, 1.0);
        match slider {
            UiAction::DragSliderSize => self.brush.size = 0.12 + t * (3.6 - 0.12),
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
            3 => self.toggle_scope_in_out(camera),
            4 => self.clone_selected_prop(),
            5 => self.spawn_connected_child_part(camera),
            6 => {
                self.brush.mirror_x = !self.brush.mirror_x;
                self.set_toast(if self.brush.mirror_x {
                    "Mirror Symmetry ON"
                } else {
                    "Mirror Symmetry OFF"
                });
            }
            7 => self.switch_mode(EditorMode::Play, camera),
            _ => {}
        }
    }

    /// Builds a clean parent -> child depth-first list `(prop_id, depth)` for the Scene Objects Outliner.
    pub fn build_outliner_tree(&self) -> Vec<(u32, usize)> {
        let mut out = Vec::with_capacity(self.scene.props.len());
        let mut visited = Vec::with_capacity(self.scene.props.len());

        fn visit_children(
            props: &[SceneProp],
            parent_id: u32,
            depth: usize,
            visited: &mut Vec<u32>,
            out: &mut Vec<(u32, usize)>,
        ) {
            if depth > 6 {
                return;
            }
            for p in props {
                if p.connector.as_ref().map(|c| c.parent_id) == Some(parent_id)
                    && !visited.contains(&p.id)
                {
                    visited.push(p.id);
                    out.push((p.id, depth));
                    visit_children(props, p.id, depth + 1, visited, out);
                }
            }
        }

        // Emit root props first (no connector, or parent doesn't exist), followed by their attached children!
        for p in &self.scene.props {
            let is_root = match &p.connector {
                None => true,
                Some(c) => !self.scene.props.iter().any(|x| x.id == c.parent_id),
            };
            if is_root && !visited.contains(&p.id) {
                visited.push(p.id);
                out.push((p.id, 0));
                visit_children(&self.scene.props, p.id, 1, &mut visited, &mut out);
            }
        }

        // Safety fallback for any cyclic connector edge case
        for p in &self.scene.props {
            if !visited.contains(&p.id) {
                visited.push(p.id);
                out.push((p.id, 0));
            }
        }

        out
    }
}

fn truncate_label_chars(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        text.to_string()
    } else {
        let mut s: String = text.chars().take(max_chars.saturating_sub(2)).collect();
        s.push_str("..");
        s
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
    fn world_sculpt_stroke_and_undo_redo_work() {
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

        inp.primary_down = false;
        inp.primary_just_pressed = false;
        inp.primary_just_released = true;
        editor.update(0.016, &inp, ActiveInputDevice::MouseKeyboard, &mut cam);

        editor.undo();
        assert_eq!(editor.scene.world_edits.len(), initial_len);

        editor.redo();
        assert_eq!(editor.scene.world_edits.len(), initial_len + 1);
    }

    #[test]
    fn scope_in_sculpt_live_clone_and_multi_part_child_work() {
        let mut editor = TraumEditor::new();
        let mut cam = CameraRig::default();

        // Scope into selected Critter (`id = 4`, `asset_id = 3`)
        editor.selected_prop_id = Some(4);
        editor.toggle_scope_in_out(&mut cam);
        assert_eq!(editor.scoped_prop_id, Some(4));

        let initial_asset_edits = editor.scene.asset_by_id(3).unwrap().edits.len();
        let _ = editor.asset_dirty.take_dirty_bricks();

        // Stamp a stroke while Scoped In -> edits `SculptAsset(3)`, NOT `world_edits`!
        let world_before = editor.scene.world_edits.len();
        let mut inp = FrameInput {
            imp_px: Vec2::new(640.0, 360.0),
            viewport: Vec2::new(1280.0, 720.0),
            primary_down: true,
            primary_just_pressed: true,
            pressure: 1.0,
            ..Default::default()
        };
        editor.update(0.016, &inp, ActiveInputDevice::MouseKeyboard, &mut cam);
        inp.primary_down = false;
        inp.primary_just_pressed = false;
        inp.primary_just_released = true;
        editor.update(0.016, &inp, ActiveInputDevice::MouseKeyboard, &mut cam);

        assert_eq!(editor.scene.world_edits.len(), world_before);
        assert_eq!(
            editor.scene.asset_by_id(3).unwrap().edits.len(),
            initial_asset_edits + 1
        );
        assert!(editor.asset_dirty.has_dirty());

        // Live-Cloning Critter (`id = 4`) should clone BOTH the Critter AND its connected Sprout Ears (`id = 5`)!
        let props_before = editor.scene.props.len();
        editor.clone_selected_prop();
        assert_eq!(
            editor.scene.props.len(),
            props_before + 2,
            "Cloning a multi-part puppet should clone root + connected child part!"
        );

        // Attach an extra limb to the new clone, then Undo both!
        editor.spawn_connected_child_part(&mut cam);
        assert_eq!(editor.scene.props.len(), props_before + 3);

        editor.undo();
        assert_eq!(editor.scene.props.len(), props_before + 2);
        editor.undo();
        assert_eq!(editor.scene.props.len(), props_before);
    }

    #[test]
    fn hovering_button_emits_tooltip_and_hidpi_scaling_works() {
        let mut editor = TraumEditor::new();
        let mut cam = CameraRig::default();

        // First frame on a 2880x1800 200% HiDPI screen
        let base_ui = editor.update(
            0.016,
            &FrameInput {
                imp_px: Vec2::new(1440.0, 900.0),
                viewport: Vec2::new(2880.0, 1800.0),
                ui_scale: 2.0,
                ..Default::default()
            },
            ActiveInputDevice::MouseKeyboard,
            &mut cam,
        );
        assert!((base_ui.ui_scale - 2.0).abs() < 0.05);

        // Find the first interactive button rect and hover its center
        let target_btn = base_ui
            .rects
            .iter()
            .find(|r| r.action == Some(UiAction::Undo))
            .expect("Expected Undo button in UI");
        let hover_pt = (target_btn.min + target_btn.max) * 0.5;

        let hovered_ui = editor.update(
            0.016,
            &FrameInput {
                imp_px: hover_pt,
                viewport: Vec2::new(2880.0, 1800.0),
                ui_scale: 2.0,
                ..Default::default()
            },
            ActiveInputDevice::MouseKeyboard,
            &mut cam,
        );
        let tip = hovered_ui
            .tooltip
            .expect("Hovering Undo button should populate a descriptive TooltipCard");
        assert!(tip.title.contains("Undo"));
        assert!(!tip.description.is_empty());
    }

    #[test]
    fn play_mode_spawns_cleanly_walks_jumps_and_pops_out_of_ground() {
        let mut editor = TraumEditor::new();
        let mut cam = CameraRig::default();

        // Enter Play Mode and let the character settle onto the grassy courtyard
        editor.switch_mode(EditorMode::Play, &mut cam);
        for _ in 0..45 {
            editor.update(
                0.016,
                &FrameInput {
                    viewport: Vec2::new(1280.0, 720.0),
                    ..Default::default()
                },
                ActiveInputDevice::MouseKeyboard,
                &mut cam,
            );
        }
        assert!(editor.puppet.grounded, "Expected puppet to stand grounded on island");
        let rest_y = editor.puppet.position.y;
        assert!(rest_y > 0.45, "Puppet should rest above grass, got y={rest_y}");

        // Walk across the open courtyard meadow (`+X / -Z` away from the stone cliff) for 45 frames
        let start_pos = editor.puppet.position;
        for _ in 0..45 {
            editor.update(
                0.016,
                &FrameInput {
                    viewport: Vec2::new(1280.0, 720.0),
                    move_axis: Vec2::new(1.0, -0.4),
                    ..Default::default()
                },
                ActiveInputDevice::MouseKeyboard,
                &mut cam,
            );
        }
        let walked_dist = Vec2::new(
            editor.puppet.position.x - start_pos.x,
            editor.puppet.position.z - start_pos.z,
        )
        .length();
        assert!(
            walked_dist > 2.5,
            "Expected character to walk >2.5m freely across meadow, moved only {walked_dist}m"
        );
        // Verify shader inverse-Y rotation maps actual walked direction onto local `+Z` (where eyes & visor are)!
        let travel_dir = Vec2::new(
            editor.puppet.position.x - start_pos.x,
            editor.puppet.position.z - start_pos.z,
        )
        .normalize();
        let yaw = editor.puppet.facing_yaw;
        let local_z_of_travel = travel_dir.x * yaw.sin() + travel_dir.y * yaw.cos();
        assert!(
            local_z_of_travel > 0.96,
            "Character local +Z (eyes/visor) must align with travel direction, dot={local_z_of_travel}"
        );

        // Jump! Must launch upward cleanly
        editor.update(
            0.016,
            &FrameInput {
                viewport: Vec2::new(1280.0, 720.0),
                jump_just_pressed: true,
                ..Default::default()
            },
            ActiveInputDevice::MouseKeyboard,
            &mut cam,
        );
        for _ in 0..15 {
            editor.update(
                0.016,
                &FrameInput {
                    viewport: Vec2::new(1280.0, 720.0),
                    ..Default::default()
                },
                ActiveInputDevice::MouseKeyboard,
                &mut cam,
            );
        }
        assert!(
            editor.puppet.position.y > rest_y + 0.9,
            "Expected jump to rise >0.9m above ground, got y={}",
            editor.puppet.position.y
        );

        // Force-embed character deep underground at (0, -1.2, 0) -> must auto-pop out onto surface!
        editor.puppet.position = Vec3::new(0.0, -1.2, 0.0);
        editor.update(
            0.016,
            &FrameInput {
                viewport: Vec2::new(1280.0, 720.0),
                ..Default::default()
            },
            ActiveInputDevice::MouseKeyboard,
            &mut cam,
        );
        assert!(
            editor.puppet.position.y > 0.45,
            "Embedded puppet must pop out above ground, got y={}",
            editor.puppet.position.y
        );
    }

    #[test]
    fn scene_outliner_tree_and_row_selection_work() {
        let mut editor = TraumEditor::new();
        let mut cam = CameraRig::default();

        // Starter island has Critter (id=4, depth=0) followed immediately by its child Sprout Ears (id=5, depth=1)
        let tree = editor.build_outliner_tree();
        assert_eq!(tree.len(), editor.scene.props.len());
        let critter_pos = tree.iter().position(|&(id, _)| id == 4).unwrap();
        assert_eq!(tree[critter_pos], (4, 0));
        assert_eq!(tree[critter_pos + 1], (5, 1));

        // Build UI on a standard 1360x820 viewport and click the Spinning Sky Ring (id=3) row in the Outliner!
        let ui = editor.update(
            0.016,
            &FrameInput {
                imp_px: Vec2::new(680.0, 410.0),
                viewport: Vec2::new(1360.0, 820.0),
                ui_scale: 1.0,
                ..Default::default()
            },
            ActiveInputDevice::MouseKeyboard,
            &mut cam,
        );
        let ring_row = ui
            .rects
            .iter()
            .find(|r| r.action == Some(UiAction::OutlinerSelectProp(3)))
            .expect("Expected Spinning Sky Ring row in Scene Objects Outliner");
        let click_pt = (ring_row.min + ring_row.max) * 0.5;

        editor.update(
            0.016,
            &FrameInput {
                imp_px: click_pt,
                viewport: Vec2::new(1360.0, 820.0),
                ui_scale: 1.0,
                primary_down: true,
                primary_just_pressed: true,
                ..Default::default()
            },
            ActiveInputDevice::MouseKeyboard,
            &mut cam,
        );
        assert_eq!(editor.selected_prop_id, Some(3));
        assert!(
            cam.target_focus.distance(Vec3::new(-2.4, 4.15, -1.8)) < 1.0,
            "Clicking Outliner row should focus camera on Spinning Sky Ring"
        );
    }
}
