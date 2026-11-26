//! Core types, reusable Sculpt Assets, CSG edit lists, multi-part puppet connectors,
//! performative animation clips, and scene serialization for Traum Engine.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Primitive shapes available for sculpting and toy props.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum SculptShape {
    Sphere = 0,
    RoundedBox = 1,
    Capsule = 2,
    Cylinder = 3,
    Cone = 4,
    Torus = 5,
}

impl SculptShape {
    pub const ALL: [SculptShape; 6] = [
        SculptShape::Sphere,
        SculptShape::RoundedBox,
        SculptShape::Capsule,
        SculptShape::Cylinder,
        SculptShape::Cone,
        SculptShape::Torus,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SculptShape::Sphere => "Sphere",
            SculptShape::RoundedBox => "Soft Box",
            SculptShape::Capsule => "Capsule",
            SculptShape::Cylinder => "Cylinder",
            SculptShape::Cone => "Cone",
            SculptShape::Torus => "Ring",
        }
    }

    pub fn next(self) -> Self {
        let idx = (self as usize + 1) % Self::ALL.len();
        Self::ALL[idx]
    }

    pub fn prev(self) -> Self {
        let idx = (self as usize + Self::ALL.len() - 1) % Self::ALL.len();
        Self::ALL[idx]
    }
}

/// CSG operation applied by a brush stroke.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum SculptOp {
    /// Smoothly adds clay to the sculpt.
    Add = 0,
    /// Smoothly carves/subtracts clay away.
    Subtract = 1,
    /// Paints color, looseness, and fleck style onto existing surfaces without altering shape.
    Paint = 2,
}

impl SculptOp {
    pub const ALL: [SculptOp; 3] = [SculptOp::Add, SculptOp::Subtract, SculptOp::Paint];

    pub fn label(self) -> &'static str {
        match self {
            SculptOp::Add => "Add Clay",
            SculptOp::Subtract => "Carve",
            SculptOp::Paint => "Coat",
        }
    }
}

/// Painterly fleck brush styles with built-in "Living Flecks" shader micro-motion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum FleckStyle {
    /// Tactile painterly impasto brushstrokes.
    ClayImpasto = 0,
    /// Leafy strokes that sway gently in a warm breeze.
    FoliageBreeze = 1,
    /// 12fps stop-motion / hand-drawn boiling strokes.
    SketchBoil = 2,
    /// Fluffy drifting cloud / cotton puffs.
    FluffyCloud = 3,
    /// Warm self-illuminated lantern / magic ember glow.
    GlowEmber = 4,
}

impl FleckStyle {
    pub const ALL: [FleckStyle; 5] = [
        FleckStyle::ClayImpasto,
        FleckStyle::FoliageBreeze,
        FleckStyle::SketchBoil,
        FleckStyle::FluffyCloud,
        FleckStyle::GlowEmber,
    ];

    pub fn label(self) -> &'static str {
        match self {
            FleckStyle::ClayImpasto => "Impasto",
            FleckStyle::FoliageBreeze => "Breeze Leaf",
            FleckStyle::SketchBoil => "12fps Boil",
            FleckStyle::FluffyCloud => "Cloud Puff",
            FleckStyle::GlowEmber => "Glow Ember",
        }
    }

    pub fn next(self) -> Self {
        let idx = (self as usize + 1) % Self::ALL.len();
        Self::ALL[idx]
    }
}

/// Top-level creative modes in Traum Engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EditorMode {
    /// Sculpt clay (Add / Carve) on the world or scoped into a custom Toy Prop!
    Sculpt,
    /// Re-coat existing clay with color, looseness, and living fleck styles.
    Paint,
    /// Transform, clone, connect limbs, and record live animations on Toy Props.
    Animate,
    /// Drop in and explore your sculpted world with a playable bouncy character!
    Play,
}

impl EditorMode {
    pub const ALL: [EditorMode; 4] = [
        EditorMode::Sculpt,
        EditorMode::Paint,
        EditorMode::Animate,
        EditorMode::Play,
    ];

    pub fn label(self) -> &'static str {
        match self {
            EditorMode::Sculpt => "Sculpt",
            EditorMode::Paint => "Paint",
            EditorMode::Animate => "Animate",
            EditorMode::Play => "Play!",
        }
    }
}

/// Zero-keyframe procedural motion toys attachable to any prop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionGadget {
    None,
    HoverBob,
    Spin,
    BouncyHop,
    Pendulum,
}

impl MotionGadget {
    pub const ALL: [MotionGadget; 5] = [
        MotionGadget::None,
        MotionGadget::HoverBob,
        MotionGadget::Spin,
        MotionGadget::BouncyHop,
        MotionGadget::Pendulum,
    ];

    pub fn label(self) -> &'static str {
        match self {
            MotionGadget::None => "Still",
            MotionGadget::HoverBob => "Hover",
            MotionGadget::Spin => "Spin",
            MotionGadget::BouncyHop => "Hop",
            MotionGadget::Pendulum => "Swing",
        }
    }

    pub fn next(self) -> Self {
        let idx = (self as usize + 1) % Self::ALL.len();
        Self::ALL[idx]
    }
}

/// Joint behavior connecting a child `SceneProp` to a parent `SceneProp` for multi-part puppets & contraptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectorKind {
    /// Bouncy secondary-motion ball joint that wobbles and lags organically behind parent movement.
    BallSpring,
    /// Motor/free-spinning hinge around local Y (windmills, propellers, spinning hats).
    HingeSpin,
    /// Rhythmic swinging hinge around local Z (wings, arms, hanging lanterns, tails).
    HingeSwing,
    /// Welded rigid attachment that follows parent pose 1:1.
    Rigid,
}

impl ConnectorKind {
    pub const ALL: [ConnectorKind; 4] = [
        ConnectorKind::BallSpring,
        ConnectorKind::HingeSpin,
        ConnectorKind::HingeSwing,
        ConnectorKind::Rigid,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ConnectorKind::BallSpring => "Spring Ball",
            ConnectorKind::HingeSpin => "Spin Hinge",
            ConnectorKind::HingeSwing => "Swing Hinge",
            ConnectorKind::Rigid => "Rigid Weld",
        }
    }

    pub fn next(self) -> Self {
        let idx = (self as usize + 1) % Self::ALL.len();
        Self::ALL[idx]
    }
}

/// Articulated joint linking a child `SceneProp` to a parent `SceneProp`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropConnector {
    pub parent_id: u32,
    /// Joint pivot in the parent's unscaled local space.
    pub parent_anchor: Vec3,
    /// Joint pivot in the child's unscaled local space (child rotates around this point!).
    pub child_anchor: Vec3,
    pub kind: ConnectorKind,
}

/// A single non-destructive CSG brush edit (in world space for `world_edits`, or local space inside a `SculptAsset`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SculptEdit {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub shape: SculptShape,
    pub op: SculptOp,
    pub blend: f32,
    /// Linear sRGB color `[r, g, b]`.
    pub color: [f32; 3],
    /// `0.0` = crisp clay core, `1.0` = billowing loose painterly flecks.
    pub looseness: f32,
    pub fleck_style: FleckStyle,
}

impl SculptEdit {
    /// Conservative AABB `(min, max)` touched by this edit (including smooth blend & loose shell).
    pub fn world_aabb(&self) -> (Vec3, Vec3) {
        let max_axis = self.scale.max_element();
        let radius = max_axis * 1.45 + self.blend * 1.35 + self.looseness * 0.65 + 0.35;
        let extent = Vec3::splat(radius);
        (self.position - extent, self.position + extent)
    }
}

/// A reusable, multi-stroke custom SDF sculpt shared by one or more Live-Cloned `SceneProp` instances!
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SculptAsset {
    pub id: u32,
    pub name: String,
    pub edits: Vec<SculptEdit>,
}

impl SculptAsset {
    /// Conservative local-space bounding sphere radius around `Vec3::ZERO` for fast ray & frustum culling.
    pub fn local_bound_radius(&self) -> f32 {
        let mut max_r = 0.45_f32;
        for e in &self.edits {
            if e.op == SculptOp::Add {
                let ext = e.scale.max_element() * 1.35 + e.blend * 0.8 + e.looseness * 0.35;
                max_r = max_r.max(e.position.length() + ext);
            }
        }
        max_r.min(3.3)
    }
}

/// A single recorded sample from performative mouse/touch/controller puppeteering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyframeSample {
    pub time: f32,
    pub position: Vec3,
    pub rotation: Quat,
}

/// A looping performative animation clip recorded live via the Action Recorder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionClip {
    pub samples: Vec<KeyframeSample>,
    pub duration: f32,
}

impl ActionClip {
    pub fn sample(&self, global_time: f32) -> Option<(Vec3, Quat)> {
        if self.samples.is_empty() {
            return None;
        }
        if self.samples.len() == 1 || self.duration <= 0.01 {
            let s = &self.samples[0];
            return Some((s.position, s.rotation));
        }

        let t = global_time.rem_euclid(self.duration);
        let mut idx = 0usize;
        while idx + 1 < self.samples.len() && self.samples[idx + 1].time < t {
            idx += 1;
        }

        if idx + 1 < self.samples.len() {
            let a = &self.samples[idx];
            let b = &self.samples[idx + 1];
            let span = (b.time - a.time).max(1e-4);
            let alpha = ((t - a.time) / span).clamp(0.0, 1.0);
            let smooth = alpha * alpha * (3.0 - 2.0 * alpha);
            Some((
                a.position.lerp(b.position, smooth),
                a.rotation.slerp(b.rotation, smooth),
            ))
        } else {
            let last = self.samples.last().unwrap();
            let first = &self.samples[0];
            let rem = (self.duration - last.time).max(0.05);
            let alpha = ((t - last.time) / rem).clamp(0.0, 1.0);
            Some((
                last.position.lerp(first.position, alpha),
                last.rotation.slerp(first.rotation, alpha),
            ))
        }
    }
}

fn default_uniform_scale() -> f32 {
    1.0
}

/// An independent, clonable,connectable, animatable 3D Toy Prop instance in the scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneProp {
    pub id: u32,
    pub name: String,
    /// ID of the `SculptAsset` rendered by this prop (clones share the same `asset_id`!).
    #[serde(default)]
    pub asset_id: u32,
    pub base_position: Vec3,
    pub base_rotation: Quat,
    /// Uniform instance scale (`0.2..3.5`) preserving exact Euclidean SDF distances.
    #[serde(default = "default_uniform_scale")]
    pub uniform_scale: f32,
    /// Legacy fallback fields so older `.ron` files still load cleanly!
    #[serde(default = "default_legacy_scale")]
    pub scale: Vec3,
    #[serde(default = "default_legacy_shape")]
    pub shape: SculptShape,
    #[serde(default = "default_legacy_color")]
    pub color: [f32; 3],
    #[serde(default)]
    pub looseness: f32,
    #[serde(default = "default_legacy_fleck")]
    pub fleck_style: FleckStyle,
    pub gadget: MotionGadget,
    pub clip: Option<ActionClip>,
    #[serde(default)]
    pub connector: Option<PropConnector>,
}

fn default_legacy_scale() -> Vec3 {
    Vec3::ONE
}
fn default_legacy_shape() -> SculptShape {
    SculptShape::Sphere
}
fn default_legacy_color() -> [f32; 3] {
    [0.8, 0.45, 0.32]
}
fn default_legacy_fleck() -> FleckStyle {
    FleckStyle::ClayImpasto
}

impl SceneProp {
    /// Evaluates this prop's un-parented local/root pose `(pos, rot)` at `time`.
    pub fn evaluate_self_pose(&self, time: f32) -> (Vec3, Quat) {
        let (mut pos, mut rot) = if let Some(clip) = &self.clip {
            clip.sample(time)
                .unwrap_or((self.base_position, self.base_rotation))
        } else {
            (self.base_position, self.base_rotation)
        };

        let phase = self.id as f32 * 1.37;
        match self.gadget {
            MotionGadget::None => {}
            MotionGadget::HoverBob => {
                pos.y += (time * 2.1 + phase).sin() * 0.35;
                rot = Quat::from_rotation_y((time * 0.8 + phase).sin() * 0.2) * rot;
            }
            MotionGadget::Spin => {
                rot = Quat::from_rotation_y(time * 1.6 + phase) * rot;
            }
            MotionGadget::BouncyHop => {
                let hop = (time * 3.2 + phase).sin().abs();
                pos.y += hop * 0.85;
                rot = Quat::from_rotation_z((time * 3.2 + phase).cos() * 0.14) * rot;
            }
            MotionGadget::Pendulum => {
                let angle = (time * 2.2 + phase).sin() * 0.55;
                rot = Quat::from_rotation_z(angle) * rot;
                pos.x += angle.sin() * 0.6;
            }
        }
        (pos, rot)
    }
}

/// Curated warm Dreams-inspired clay color palette (sRGB).
pub const PALETTE_SRGB: [[f32; 3]; 12] = [
    [0.88, 0.46, 0.32], // Warm Terracotta Clay
    [0.94, 0.76, 0.42], // Sunlit Ochre Gold
    [0.42, 0.72, 0.38], // Meadow Moss Green
    [0.24, 0.56, 0.46], // Deep Pine Teal
    [0.38, 0.64, 0.88], // Storybook Sky Blue
    [0.62, 0.46, 0.84], // Twilight Lavender
    [0.90, 0.44, 0.58], // Coral Rose
    [0.95, 0.92, 0.86], // Cloud Cream White
    [0.54, 0.43, 0.35], // Bark Umber Brown
    [0.58, 0.61, 0.66], // Carved Cliff Slate
    [0.22, 0.23, 0.27], // Charcoal Ink
    [1.00, 0.68, 0.22], // Lantern Amber Glow
];

pub fn srgb_to_linear(c: [f32; 3]) -> [f32; 3] {
    [c[0].powf(2.2), c[1].powf(2.2), c[2].powf(2.2)]
}

/// Complete serializable Traum Engine scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraumScene {
    pub title: String,
    pub world_edits: Vec<SculptEdit>,
    #[serde(default)]
    pub assets: Vec<SculptAsset>,
    #[serde(default = "default_next_asset_id")]
    pub next_asset_id: u32,
    pub props: Vec<SceneProp>,
    pub next_prop_id: u32,
    pub sun_azimuth: f32,
    pub sun_elevation: f32,
    pub dof_amount: f32,
}

fn default_next_asset_id() -> u32 {
    1
}

impl TraumScene {
    pub fn asset_by_id(&self, asset_id: u32) -> Option<&SculptAsset> {
        self.assets.iter().find(|a| a.id == asset_id)
    }

    pub fn asset_by_id_mut(&mut self, asset_id: u32) -> Option<&mut SculptAsset> {
        self.assets.iter_mut().find(|a| a.id == asset_id)
    }

    /// Evaluates the full hierarchical world `(position, rotation)` of `prop_id` at `time`,
    /// resolving parent connectors (`BallSpring` lag/wobble, `HingeSpin`, `HingeSwing`, `Rigid`)!
    pub fn evaluate_prop_pose(&self, prop_id: u32, time: f32) -> (Vec3, Quat) {
        self.evaluate_prop_pose_depth(prop_id, time, 0)
    }

    fn evaluate_prop_pose_depth(&self, prop_id: u32, time: f32, depth: u32) -> (Vec3, Quat) {
        let Some(prop) = self.props.iter().find(|p| p.id == prop_id) else {
            return (Vec3::ZERO, Quat::IDENTITY);
        };

        if depth >= 6 {
            return prop.evaluate_self_pose(time);
        }

        let Some(conn) = &prop.connector else {
            return prop.evaluate_self_pose(time);
        };

        let Some(parent) = self.props.iter().find(|p| p.id == conn.parent_id) else {
            return prop.evaluate_self_pose(time);
        };

        let (p_pos, p_rot) = self.evaluate_prop_pose_depth(parent.id, time, depth + 1);
        let pivot_world = p_pos + p_rot * (conn.parent_anchor * parent.uniform_scale.max(0.1));

        // Child's own local animation/pose offset around its base
        let (_, self_rot) = if let Some(clip) = &prop.clip {
            clip.sample(time)
                .unwrap_or((prop.base_position, prop.base_rotation))
        } else {
            (prop.base_position, prop.base_rotation)
        };

        let phase = prop.id as f32 * 1.91;
        let joint_rot = match conn.kind {
            ConnectorKind::Rigid => p_rot * self_rot,
            ConnectorKind::BallSpring => {
                // Sample parent velocity slightly behind in time for juicy secondary spring follow-through!
                let (p_prev, _) =
                    self.evaluate_prop_pose_depth(parent.id, (time - 0.09).max(0.0), depth + 1);
                let vel = (p_pos - p_prev) / 0.09;
                let lag_pitch = (-vel.z * 0.085).clamp(-0.55, 0.55);
                let lag_roll = (vel.x * 0.085).clamp(-0.55, 0.55);
                let bounce_wobble = (-vel.y * 0.06).clamp(-0.35, 0.35)
                    + (time * 5.2 + phase).sin() * 0.07;
                let spring_q = Quat::from_rotation_z(lag_roll + bounce_wobble)
                    * Quat::from_rotation_x(lag_pitch);
                spring_q * p_rot * self_rot
            }
            ConnectorKind::HingeSpin => {
                let spin_q = Quat::from_rotation_y(time * 2.6 + phase);
                p_rot * spin_q * self_rot
            }
            ConnectorKind::HingeSwing => {
                let swing_q = Quat::from_rotation_z((time * 3.4 + phase).sin() * 0.62);
                p_rot * swing_q * self_rot
            }
        };

        let child_pos = pivot_world - joint_rot * (conn.child_anchor * prop.uniform_scale.max(0.1));
        (child_pos, joint_rot)
    }

    /// Returns world-space `(parent_pivot, child_center)` for rendering visual joint connectors.
    pub fn evaluate_connector_segment(&self, prop: &SceneProp, time: f32) -> Option<(Vec3, Vec3)> {
        let conn = prop.connector.as_ref()?;
        let parent = self.props.iter().find(|p| p.id == conn.parent_id)?;
        let (p_pos, p_rot) = self.evaluate_prop_pose(parent.id, time);
        let pivot = p_pos + p_rot * (conn.parent_anchor * parent.uniform_scale.max(0.1));
        let (c_pos, _) = self.evaluate_prop_pose(prop.id, time);
        Some((pivot, c_pos))
    }

    /// Upgrades any legacy `.ron` scene whose `props` didn't have `SculptAsset`s yet.
    pub fn migrate_legacy_props(&mut self) {
        for prop in &mut self.props {
            if prop.uniform_scale <= 0.001 {
                prop.uniform_scale = 1.0;
            }
            if prop.asset_id == 0 || !self.assets.iter().any(|a| a.id == prop.asset_id) {
                let aid = self.next_asset_id.max(1);
                self.next_asset_id = aid + 1;
                self.assets.push(SculptAsset {
                    id: aid,
                    name: format!("{} Sculpt", prop.name),
                    edits: vec![SculptEdit {
                        position: Vec3::ZERO,
                        rotation: Quat::IDENTITY,
                        scale: prop.scale,
                        shape: prop.shape,
                        op: SculptOp::Add,
                        blend: 0.15,
                        color: prop.color,
                        looseness: prop.looseness,
                        fleck_style: prop.fleck_style,
                    }],
                });
                prop.asset_id = aid;
            }
        }
    }

    /// Builds the sunlit "Floating Diorama Island" starter scene with custom multi-stroke Sculpt Assets & Connected Puppets!
    pub fn starter_diorama_island() -> Self {
        let mut edits = Vec::new();

        let terracotta = srgb_to_linear([0.84, 0.46, 0.33]);
        let cliff_rock = srgb_to_linear([0.58, 0.50, 0.46]);
        let moss_green = srgb_to_linear([0.40, 0.71, 0.36]);
        let canopy_green = srgb_to_linear([0.30, 0.64, 0.34]);
        let blossom_pink = srgb_to_linear([0.92, 0.50, 0.60]);
        let trunk_brown = srgb_to_linear([0.48, 0.34, 0.24]);
        let cloud_cream = srgb_to_linear([0.96, 0.94, 0.90]);
        let lantern_gold = srgb_to_linear([1.0, 0.70, 0.24]);
        let sky_blue = srgb_to_linear([0.38, 0.64, 0.88]);
        let charcoal = srgb_to_linear([0.16, 0.17, 0.22]);

        // 1. Floating island rocky underside & terracotta cliff base
        edits.push(SculptEdit {
            position: Vec3::new(0.0, -1.6, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(6.8, 2.1, 6.2),
            shape: SculptShape::Sphere,
            op: SculptOp::Add,
            blend: 0.45,
            color: terracotta,
            looseness: 0.18,
            fleck_style: FleckStyle::ClayImpasto,
        });
        edits.push(SculptEdit {
            position: Vec3::new(0.0, -3.4, 0.0),
            rotation: Quat::from_rotation_z(std::f32::consts::PI),
            scale: Vec3::new(4.8, 2.8, 4.8),
            shape: SculptShape::Cone,
            op: SculptOp::Add,
            blend: 0.85,
            color: cliff_rock,
            looseness: 0.24,
            fleck_style: FleckStyle::ClayImpasto,
        });

        // 2. Flatten top courtyard + soft grassy meadow cap
        edits.push(SculptEdit {
            position: Vec3::new(0.0, 1.8, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(8.5, 1.8, 8.5),
            shape: SculptOp::Subtract.default_scale_or(Vec3::new(8.5, 1.8, 8.5)),
            op: SculptOp::Subtract,
            blend: 0.35,
            color: terracotta,
            looseness: 0.1,
            fleck_style: FleckStyle::ClayImpasto,
        });
        // Fix shape explicitly to RoundedBox as before
        edits.last_mut().unwrap().shape = SculptShape::RoundedBox;

        edits.push(SculptEdit {
            position: Vec3::new(0.0, -0.35, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(6.1, 0.42, 5.6),
            shape: SculptShape::Cylinder,
            op: SculptOp::Add,
            blend: 0.4,
            color: moss_green,
            looseness: 0.36,
            fleck_style: FleckStyle::FoliageBreeze,
        });

        // 3. Raised hill + carved stone archway on the island!
        edits.push(SculptEdit {
            position: Vec3::new(-2.4, 1.1, -1.8),
            rotation: Quat::from_rotation_y(0.35),
            scale: Vec3::new(2.5, 1.7, 2.2),
            shape: SculptShape::Sphere,
            op: SculptOp::Add,
            blend: 0.55,
            color: cliff_rock,
            looseness: 0.16,
            fleck_style: FleckStyle::ClayImpasto,
        });
        edits.push(SculptEdit {
            position: Vec3::new(-2.4, 2.45, -1.8),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(2.2, 0.45, 1.9),
            shape: SculptShape::Sphere,
            op: SculptOp::Add,
            blend: 0.45,
            color: moss_green,
            looseness: 0.45,
            fleck_style: FleckStyle::FoliageBreeze,
        });
        edits.push(SculptEdit {
            position: Vec3::new(-2.4, 0.75, -1.8),
            rotation: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)
                * Quat::from_rotation_z(0.35),
            scale: Vec3::new(1.05, 2.8, 1.05),
            shape: SculptShape::Capsule,
            op: SculptOp::Subtract,
            blend: 0.25,
            color: cliff_rock,
            looseness: 0.12,
            fleck_style: FleckStyle::ClayImpasto,
        });

        // 4. Swaying Dreams-style painterly tree on the right side
        edits.push(SculptEdit {
            position: Vec3::new(2.7, 1.35, -1.4),
            rotation: Quat::from_rotation_z(0.12),
            scale: Vec3::new(0.36, 1.55, 0.36),
            shape: SculptShape::Capsule,
            op: SculptOp::Add,
            blend: 0.25,
            color: trunk_brown,
            looseness: 0.20,
            fleck_style: FleckStyle::ClayImpasto,
        });
        for (offset, rad, col) in [
            (Vec3::new(2.45, 3.1, -1.4), 1.35, canopy_green),
            (Vec3::new(3.25, 2.75, -1.1), 1.05, moss_green),
            (Vec3::new(2.6, 3.55, -1.9), 0.95, blossom_pink),
        ] {
            edits.push(SculptEdit {
                position: offset,
                rotation: Quat::IDENTITY,
                scale: Vec3::splat(rad),
                shape: SculptShape::Sphere,
                op: SculptOp::Add,
                blend: 0.55,
                color: col,
                looseness: 0.82,
                fleck_style: FleckStyle::FoliageBreeze,
            });
        }

        // 5. Drifting fluffy clouds hugging the base of the floating island
        for (pos, scale) in [
            (Vec3::new(-5.4, -1.4, 2.4), Vec3::new(1.6, 0.85, 1.3)),
            (Vec3::new(-4.3, -1.6, 3.1), Vec3::new(1.2, 0.7, 1.1)),
            (Vec3::new(5.1, -1.8, 1.8), Vec3::new(1.7, 0.9, 1.4)),
        ] {
            edits.push(SculptEdit {
                position: pos,
                rotation: Quat::IDENTITY,
                scale,
                shape: SculptShape::Sphere,
                op: SculptOp::Add,
                blend: 0.65,
                color: cloud_cream,
                looseness: 0.92,
                fleck_style: FleckStyle::FluffyCloud,
            });
        }

        // 6. Warm glowing lantern stone by the arch entrance
        edits.push(SculptEdit {
            position: Vec3::new(-0.85, 0.45, -0.55),
            rotation: Quat::from_rotation_y(0.5),
            scale: Vec3::new(0.32, 0.48, 0.32),
            shape: SculptShape::RoundedBox,
            op: SculptOp::Add,
            blend: 0.12,
            color: lantern_gold,
            looseness: 0.35,
            fleck_style: FleckStyle::GlowEmber,
        });

        // ====================================================================
        // 7. Reusable Custom Multi-Stroke Sculpt Assets!
        //    Scope into any prop (`Enter`) to sculpt right on it — all Live Clones update at once!
        // ====================================================================
        let assets = vec![
            // Asset 1: Mossy Floating Stepping Platform (2 Live Clones in starter scene!)
            SculptAsset {
                id: 1,
                name: "Mossy Stepping Stone".into(),
                edits: vec![
                    SculptEdit {
                        position: Vec3::new(0.0, -0.10, 0.0),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::new(1.15, 0.30, 1.15),
                        shape: SculptShape::Cylinder,
                        op: SculptOp::Add,
                        blend: 0.18,
                        color: terracotta,
                        looseness: 0.18,
                        fleck_style: FleckStyle::ClayImpasto,
                    },
                    SculptEdit {
                        position: Vec3::new(0.0, 0.14, 0.0),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::new(1.06, 0.16, 1.06),
                        shape: SculptShape::Cylinder,
                        op: SculptOp::Add,
                        blend: 0.22,
                        color: moss_green,
                        looseness: 0.42,
                        fleck_style: FleckStyle::FoliageBreeze,
                    },
                ],
            },
            // Asset 2: Ornate Glowing Sky Ring
            SculptAsset {
                id: 2,
                name: "Sunlit Sky Ring".into(),
                edits: vec![
                    SculptEdit {
                        position: Vec3::ZERO,
                        rotation: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                        scale: Vec3::new(0.82, 0.20, 0.82),
                        shape: SculptShape::Torus,
                        op: SculptOp::Add,
                        blend: 0.12,
                        color: lantern_gold,
                        looseness: 0.36,
                        fleck_style: FleckStyle::GlowEmber,
                    },
                    SculptEdit {
                        position: Vec3::ZERO,
                        rotation: Quat::IDENTITY,
                        scale: Vec3::splat(0.26),
                        shape: SculptShape::Sphere,
                        op: SculptOp::Add,
                        blend: 0.15,
                        color: cloud_cream,
                        looseness: 0.45,
                        fleck_style: FleckStyle::GlowEmber,
                    },
                ],
            },
            // Asset 3: Sculpted Bouncy Critter Body (with eyes & rosy cheeks!)
            SculptAsset {
                id: 3,
                name: "Critter Body".into(),
                edits: vec![
                    SculptEdit {
                        position: Vec3::ZERO,
                        rotation: Quat::IDENTITY,
                        scale: Vec3::new(0.56, 0.50, 0.54),
                        shape: SculptShape::Sphere,
                        op: SculptOp::Add,
                        blend: 0.18,
                        color: sky_blue,
                        looseness: 0.42,
                        fleck_style: FleckStyle::SketchBoil,
                    },
                    SculptEdit {
                        position: Vec3::new(-0.18, 0.12, 0.44),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::splat(0.11),
                        shape: SculptShape::Sphere,
                        op: SculptOp::Add,
                        blend: 0.04,
                        color: charcoal,
                        looseness: 0.08,
                        fleck_style: FleckStyle::ClayImpasto,
                    },
                    SculptEdit {
                        position: Vec3::new(0.18, 0.12, 0.44),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::splat(0.11),
                        shape: SculptShape::Sphere,
                        op: SculptOp::Add,
                        blend: 0.04,
                        color: charcoal,
                        looseness: 0.08,
                        fleck_style: FleckStyle::ClayImpasto,
                    },
                ],
            },
            // Asset 4: Sculpted Bunny-Ear Sprout Hat (connected to Critter Body via BallSpring joint!)
            SculptAsset {
                id: 4,
                name: "Sprout Ears Hat".into(),
                edits: vec![
                    SculptEdit {
                        position: Vec3::new(0.0, 0.06, 0.0),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::new(0.34, 0.16, 0.34),
                        shape: SculptShape::Sphere,
                        op: SculptOp::Add,
                        blend: 0.14,
                        color: blossom_pink,
                        looseness: 0.36,
                        fleck_style: FleckStyle::SketchBoil,
                    },
                    SculptEdit {
                        position: Vec3::new(-0.20, 0.38, 0.0),
                        rotation: Quat::from_rotation_z(0.24),
                        scale: Vec3::new(0.12, 0.34, 0.10),
                        shape: SculptShape::Capsule,
                        op: SculptOp::Add,
                        blend: 0.14,
                        color: blossom_pink,
                        looseness: 0.44,
                        fleck_style: FleckStyle::SketchBoil,
                    },
                    SculptEdit {
                        position: Vec3::new(0.20, 0.38, 0.0),
                        rotation: Quat::from_rotation_z(-0.24),
                        scale: Vec3::new(0.12, 0.34, 0.10),
                        shape: SculptShape::Capsule,
                        op: SculptOp::Add,
                        blend: 0.14,
                        color: blossom_pink,
                        looseness: 0.44,
                        fleck_style: FleckStyle::SketchBoil,
                    },
                ],
            },
        ];

        let props = vec![
            SceneProp {
                id: 1,
                name: "Stepping Stone A".into(),
                asset_id: 1,
                base_position: Vec3::new(0.4, 1.35, 2.8),
                base_rotation: Quat::IDENTITY,
                uniform_scale: 1.0,
                scale: Vec3::ONE,
                shape: SculptShape::Cylinder,
                color: terracotta,
                looseness: 0.25,
                fleck_style: FleckStyle::ClayImpasto,
                gadget: MotionGadget::HoverBob,
                clip: None,
                connector: None,
            },
            // Live Clone of Asset #1 leading higher up toward the arch!
            SceneProp {
                id: 2,
                name: "Stepping Stone B (Clone)".into(),
                asset_id: 1,
                base_position: Vec3::new(-1.35, 2.25, 1.55),
                base_rotation: Quat::from_rotation_y(0.6),
                uniform_scale: 0.82,
                scale: Vec3::ONE,
                shape: SculptShape::Cylinder,
                color: terracotta,
                looseness: 0.25,
                fleck_style: FleckStyle::ClayImpasto,
                gadget: MotionGadget::HoverBob,
                clip: None,
                connector: None,
            },
            SceneProp {
                id: 3,
                name: "Spinning Sky Ring".into(),
                asset_id: 2,
                base_position: Vec3::new(-2.4, 4.15, -1.8),
                base_rotation: Quat::IDENTITY,
                uniform_scale: 0.95,
                scale: Vec3::ONE,
                shape: SculptShape::Torus,
                color: lantern_gold,
                looseness: 0.40,
                fleck_style: FleckStyle::GlowEmber,
                gadget: MotionGadget::Spin,
                clip: None,
                connector: None,
            },
            // Multi-part Puppet Root: Bouncy Sketch Critter Body
            SceneProp {
                id: 4,
                name: "Bouncy Critter".into(),
                asset_id: 3,
                base_position: Vec3::new(1.6, 0.55, 1.3),
                base_rotation: Quat::from_rotation_y(-0.45),
                uniform_scale: 1.0,
                scale: Vec3::ONE,
                shape: SculptShape::Sphere,
                color: sky_blue,
                looseness: 0.45,
                fleck_style: FleckStyle::SketchBoil,
                gadget: MotionGadget::BouncyHop,
                clip: None,
                connector: None,
            },
            // Multi-part Puppet Child: Connected via BallSpring joint so its ears bobble as the critter hops!
            SceneProp {
                id: 5,
                name: "Critter Sprout Ears".into(),
                asset_id: 4,
                base_position: Vec3::new(1.6, 1.05, 1.3),
                base_rotation: Quat::IDENTITY,
                uniform_scale: 1.0,
                scale: Vec3::ONE,
                shape: SculptShape::Sphere,
                color: blossom_pink,
                looseness: 0.45,
                fleck_style: FleckStyle::SketchBoil,
                gadget: MotionGadget::None,
                clip: None,
                connector: Some(PropConnector {
                    parent_id: 4,
                    parent_anchor: Vec3::new(0.0, 0.46, 0.0),
                    child_anchor: Vec3::new(0.0, -0.04, 0.0),
                    kind: ConnectorKind::BallSpring,
                }),
            },
        ];

        Self {
            title: "Floating Diorama Island".into(),
            world_edits: edits,
            assets,
            next_asset_id: 5,
            props,
            next_prop_id: 6,
            sun_azimuth: 0.72,
            sun_elevation: 0.68,
            dof_amount: 0.35,
        }
    }

    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let pretty = ron::ser::PrettyConfig::default();
        let text = ron::ser::to_string_pretty(self, pretty).map_err(|e| e.to_string())?;
        fs::write(path, text).map_err(|e| e.to_string())
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut scene: Self = ron::from_str(&text).map_err(|e| e.to_string())?;
        scene.migrate_legacy_props();
        Ok(scene)
    }
}

trait SculptOpExt {
    fn default_scale_or(self, v: Vec3) -> SculptShape;
}
impl SculptOpExt for SculptOp {
    fn default_scale_or(self, _v: Vec3) -> SculptShape {
        SculptShape::RoundedBox
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starter_island_roundtrips_ron_with_assets_and_connectors() {
        let scene = TraumScene::starter_diorama_island();
        let serialized = ron::to_string(&scene).expect("serialize scene");
        let restored: TraumScene = ron::from_str(&serialized).expect("deserialize scene");
        assert_eq!(scene.world_edits.len(), restored.world_edits.len());
        assert_eq!(scene.assets.len(), restored.assets.len());
        assert_eq!(scene.props.len(), restored.props.len());

        // Verify connected child prop (id=5) tracks hopping parent prop (id=4)!
        let (parent_p, _) = restored.evaluate_prop_pose(4, 0.35);
        let (child_p, _) = restored.evaluate_prop_pose(5, 0.35);
        assert!(
            child_p.y > parent_p.y + 0.3,
            "Connected sprout ears should sit atop hopping critter body"
        );
    }

    #[test]
    fn action_clip_interpolates_and_loops() {
        let clip = ActionClip {
            duration: 2.0,
            samples: vec![
                KeyframeSample {
                    time: 0.0,
                    position: Vec3::ZERO,
                    rotation: Quat::IDENTITY,
                },
                KeyframeSample {
                    time: 1.0,
                    position: Vec3::new(2.0, 0.0, 0.0),
                    rotation: Quat::IDENTITY,
                },
            ],
        };
        let (mid, _) = clip.sample(0.5).unwrap();
        assert!((mid.x - 1.0).abs() < 0.05);
        let (looped, _) = clip.sample(2.5).unwrap();
        assert!((looped.x - 1.0).abs() < 0.05);
    }
}
