//! SDF brick volume tracking (World + Reusable Sculpt Asset Atlas), GPU Pod data layouts,
//! and shared CPU/GPU CSG distance & collision math.

use bytemuck::{Pod, Zeroable};
use glam::{UVec3, Vec2, Vec3};
use traum_core::{SculptAsset, SculptEdit, SculptOp, SculptShape, TraumScene};

/// Brick dimension (`8x8x8` voxels per brick).
pub const BRICK_SIZE: u32 = 8;

/// World coarse brick grid (`24 x 16 x 24` = 9,216 bricks -> `192 x 128 x 192` voxels).
pub const BRICK_GRID: UVec3 = UVec3::new(24, 16, 24);

/// Total voxel resolution of the world SDF volume (`192 x 128 x 192`).
pub const VOLUME_DIMS: UVec3 = UVec3::new(
    BRICK_GRID.x * BRICK_SIZE,
    BRICK_GRID.y * BRICK_SIZE,
    BRICK_GRID.z * BRICK_SIZE,
);

/// World-space bounds of the sculptable diorama volume (`[-14.4..14.4] x [-9.6..9.6] x [-14.4..14.4]`, voxel = 0.15m).
pub const VOXEL_SIZE: f32 = 0.15;
pub const VOLUME_MIN: Vec3 = Vec3::new(
    -(VOLUME_DIMS.x as f32) * 0.5 * VOXEL_SIZE,
    -(VOLUME_DIMS.y as f32) * 0.5 * VOXEL_SIZE,
    -(VOLUME_DIMS.z as f32) * 0.5 * VOXEL_SIZE,
);
pub const VOLUME_MAX: Vec3 = Vec3::new(
    (VOLUME_DIMS.x as f32) * 0.5 * VOXEL_SIZE,
    (VOLUME_DIMS.y as f32) * 0.5 * VOXEL_SIZE,
    (VOLUME_DIMS.z as f32) * 0.5 * VOXEL_SIZE,
);

// ============================================================================
// Reusable Sculpt Asset 3D Brick Atlas Constants (`24` Custom Assets @ 64^3!)
// ============================================================================

/// Voxels per axis inside a single `SculptAsset` tile (`64x64x64` = `8x8x8` bricks).
pub const ASSET_TILE_DIM: u32 = 64;
pub const ASSET_BRICKS_PER_AXIS: u32 = ASSET_TILE_DIM / BRICK_SIZE; // 8
pub const ASSET_BRICKS_PER_SLOT: u32 =
    ASSET_BRICKS_PER_AXIS * ASSET_BRICKS_PER_AXIS * ASSET_BRICKS_PER_AXIS; // 512

/// Tile arrangement inside the 3D Asset Atlas (`4 x 2 x 3` = 24 custom sculpt assets).
pub const ASSET_ATLAS_TILES: UVec3 = UVec3::new(4, 2, 3);
pub const MAX_SCULPT_ASSETS: usize =
    (ASSET_ATLAS_TILES.x * ASSET_ATLAS_TILES.y * ASSET_ATLAS_TILES.z) as usize; // 24

/// Total 3D texture resolution of the Asset Atlas (`256 x 128 x 192`).
pub const ASSET_ATLAS_DIMS: UVec3 = UVec3::new(
    ASSET_ATLAS_TILES.x * ASSET_TILE_DIM,
    ASSET_ATLAS_TILES.y * ASSET_TILE_DIM,
    ASSET_ATLAS_TILES.z * ASSET_TILE_DIM,
);

/// Crisp local-space voxel size (`0.095m` -> `[-3.04m .. +3.04m]` local sculpt space per asset).
pub const ASSET_VOXEL_SIZE: f32 = 0.095;
pub const ASSET_LOCAL_HALF_EXTENT: f32 = (ASSET_TILE_DIM as f32) * 0.5 * ASSET_VOXEL_SIZE; // 3.04m
pub const ASSET_LOCAL_MIN: Vec3 = Vec3::splat(-ASSET_LOCAL_HALF_EXTENT);
pub const ASSET_LOCAL_MAX: Vec3 = Vec3::splat(ASSET_LOCAL_HALF_EXTENT);

/// Maximum cached iso-surface flecks stored in local space per `SculptAsset` slot.
pub const MAX_ASSET_LOCAL_FLECKS: u32 = 3072;

/// Maximum surface flecks emitted into the GPU painterly splat buffer per frame.
pub const MAX_SURFACE_FLECKS: u32 = 550_000;

/// Maximum CSG edits uploaded to world or asset edit buffers.
pub const MAX_GPU_EDITS: usize = 4096;
pub const MAX_GPU_ASSET_EDITS: usize = 4096;

/// Maximum animated/cloned props rendered simultaneously.
pub const MAX_GPU_PROPS: usize = 128;

/// Maximum visual joint connector springs/hinges drawn simultaneously.
pub const MAX_GPU_CONNECTORS: usize = 64;

/// 80-byte (`5 x vec4`) GPU representation of a CSG `SculptEdit`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuSculptEdit {
    /// `xyz` = position (world or asset-local), `w` = smooth blend radius `k`.
    pub pos_blend: [f32; 4],
    /// `xyzw` = inverse rotation quaternion (`rot.conjugate()`).
    pub inv_rot: [f32; 4],
    /// `xyz` = primitive scale/extents, `w` = looseness (`0.0..1.0`).
    pub scale_loose: [f32; 4],
    /// `rgb` = linear clay color, `w` = reserved.
    pub color_pad: [f32; 4],
    /// `[shape, op, fleck_style, reserved]`.
    pub meta: [u32; 4],
}

impl GpuSculptEdit {
    pub fn from_edit(edit: &SculptEdit) -> Self {
        let inv = edit.rotation.normalize().conjugate();
        Self {
            pos_blend: [
                edit.position.x,
                edit.position.y,
                edit.position.z,
                edit.blend.max(0.01),
            ],
            inv_rot: [inv.x, inv.y, inv.z, inv.w],
            scale_loose: [
                edit.scale.x.max(0.03),
                edit.scale.y.max(0.03),
                edit.scale.z.max(0.03),
                edit.looseness.clamp(0.0, 1.0),
            ],
            color_pad: [edit.color[0], edit.color[1], edit.color[2], 0.0],
            meta: [
                edit.shape as u32,
                edit.op as u32,
                edit.fleck_style as u32,
                0,
            ],
        }
    }
}

/// 32-byte (`2 x vec4<u32>`) GPU metadata per `SculptAsset` slot (`0..MAX_SCULPT_ASSETS`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuAssetHeader {
    /// `[edit_offset, edit_count, tile_x, tile_y]`.
    pub edit_and_tile_xy: [u32; 4],
    /// `[tile_z, asset_id, pad, pad]`.
    pub tile_z_and_meta: [u32; 4],
}

/// 16-byte (`vec4<u32>`) entry identifying one dirty `8x8x8` brick inside an Asset Atlas slot.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuAssetDirtyBrick {
    /// `[local_brick_idx (0..512), asset_slot (0..24), 0, 0]`.
    pub brick_and_slot: [u32; 4],
}

/// 80-byte (`5 x vec4`) GPU representation of an animated/cloned `SceneProp` instance.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuSceneProp {
    /// `xyz` = animated world position, `w` = world bounding sphere radius.
    pub pos_radius: [f32; 4],
    /// `xyzw` = inverse world rotation quaternion (`world_rot.conjugate()`).
    pub inv_rot: [f32; 4],
    /// `x` = uniform_scale, `y` = 1.0 / uniform_scale, `z` = gizmo_ring_radius, `w` = pad.
    pub scale_params: [f32; 4],
    /// `xyz` = scale_orb_world_pos (when selected in Animate mode), `w` = pad.
    pub gizmo_orb: [f32; 4],
    /// `[asset_slot, prop_id, selected_flag, scoped_flag]`.
    pub meta: [u32; 4],
}

/// 32-byte (`2 x vec4<f32>`) visual connector segment between a parent pivot and child prop.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuConnectorVisual {
    /// `xyz` = parent pivot world pos, `w` = connector kind (`0..3`).
    pub start_kind: [f32; 4],
    /// `xyz` = child world pos, `w` = selected highlight (`1.0` or `0.0`).
    pub end_sel: [f32; 4],
}

/// 48-byte (`3 x vec4`) GPU representation of a single painterly surface fleck.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuFleck {
    /// `xyz` = surface position (world or asset-local), `w` = fleck radius.
    pub pos_size: [f32; 4],
    /// `xyz` = surface normal, `w` = looseness.
    pub normal_loose: [f32; 4],
    /// `rgb` = linear color, `w` = packed `(style_index + frac_seed)`.
    pub color_style_seed: [f32; 4],
}

/// Converts an asset slot index (`0..MAX_SCULPT_ASSETS`) into its 3D tile coordinates `(tx, ty, tz)`.
#[inline]
pub fn asset_slot_tile_coords(slot: usize) -> UVec3 {
    let s = (slot as u32).min(MAX_SCULPT_ASSETS as u32 - 1);
    let tx = s % ASSET_ATLAS_TILES.x;
    let ty = (s / ASSET_ATLAS_TILES.x) % ASSET_ATLAS_TILES.y;
    let tz = s / (ASSET_ATLAS_TILES.x * ASSET_ATLAS_TILES.y);
    UVec3::new(tx, ty, tz)
}

/// Tracks which `8x8x8` bricks in the `24x16x24` world grid need GPU compute re-evaluation.
pub struct BrickDirtyTracker {
    dirty_bricks: Vec<u32>,
    is_queued: Vec<bool>,
}

impl BrickDirtyTracker {
    pub fn new() -> Self {
        let total = (BRICK_GRID.x * BRICK_GRID.y * BRICK_GRID.z) as usize;
        let mut tracker = Self {
            dirty_bricks: Vec::with_capacity(total),
            is_queued: vec![false; total],
        };
        tracker.mark_all_dirty();
        tracker
    }

    pub fn mark_all_dirty(&mut self) {
        self.dirty_bricks.clear();
        let total = BRICK_GRID.x * BRICK_GRID.y * BRICK_GRID.z;
        for idx in 0..total {
            self.dirty_bricks.push(idx);
            self.is_queued[idx as usize] = true;
        }
    }

    pub fn mark_aabb_dirty(&mut self, aabb_min: Vec3, aabb_max: Vec3) {
        let brick_world = VOXEL_SIZE * (BRICK_SIZE as f32);
        let min_b = ((aabb_min - VOLUME_MIN) / brick_world).floor().as_ivec3();
        let max_b = ((aabb_max - VOLUME_MIN) / brick_world).floor().as_ivec3();

        let x0 = min_b.x.clamp(0, BRICK_GRID.x as i32 - 1) as u32;
        let y0 = min_b.y.clamp(0, BRICK_GRID.y as i32 - 1) as u32;
        let z0 = min_b.z.clamp(0, BRICK_GRID.z as i32 - 1) as u32;

        let x1 = max_b.x.clamp(0, BRICK_GRID.x as i32 - 1) as u32;
        let y1 = max_b.y.clamp(0, BRICK_GRID.y as i32 - 1) as u32;
        let z1 = max_b.z.clamp(0, BRICK_GRID.z as i32 - 1) as u32;

        for z in z0..=z1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let flat = x + y * BRICK_GRID.x + z * BRICK_GRID.x * BRICK_GRID.y;
                    if !self.is_queued[flat as usize] {
                        self.is_queued[flat as usize] = true;
                        self.dirty_bricks.push(flat);
                    }
                }
            }
        }
    }

    pub fn take_dirty_bricks(&mut self) -> Vec<u32> {
        for &idx in &self.dirty_bricks {
            self.is_queued[idx as usize] = false;
        }
        std::mem::take(&mut self.dirty_bricks)
    }

    pub fn has_dirty(&self) -> bool {
        !self.dirty_bricks.is_empty()
    }
}

impl Default for BrickDirtyTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks dirty `8x8x8` bricks and local fleck cache invalidation per `SculptAsset` slot (`0..MAX_SCULPT_ASSETS`).
pub struct AssetDirtyTracker {
    dirty_entries: Vec<GpuAssetDirtyBrick>,
    queued: Vec<bool>,
    dirty_fleck_slots: [bool; MAX_SCULPT_ASSETS],
    known_asset_ids: [u32; MAX_SCULPT_ASSETS],
}

impl AssetDirtyTracker {
    pub fn new() -> Self {
        let total = MAX_SCULPT_ASSETS * (ASSET_BRICKS_PER_SLOT as usize);
        Self {
            dirty_entries: Vec::with_capacity(2048),
            queued: vec![false; total],
            dirty_fleck_slots: [false; MAX_SCULPT_ASSETS],
            known_asset_ids: [0; MAX_SCULPT_ASSETS],
        }
    }

    /// Marks all `512` bricks of `slot` dirty (e.g. on initial load, new asset creation, or reset).
    pub fn mark_slot_all_dirty(&mut self, slot: usize) {
        if slot >= MAX_SCULPT_ASSETS {
            return;
        }
        let base = slot * (ASSET_BRICKS_PER_SLOT as usize);
        for b in 0..ASSET_BRICKS_PER_SLOT {
            let flat = base + b as usize;
            if !self.queued[flat] {
                self.queued[flat] = true;
                self.dirty_entries.push(GpuAssetDirtyBrick {
                    brick_and_slot: [b, slot as u32, 0, 0],
                });
            }
        }
        self.dirty_fleck_slots[slot] = true;
    }

    /// Marks only the `8x8x8` local bricks overlapping `[local_min, local_max]` inside `slot` as dirty!
    pub fn mark_slot_aabb_dirty(&mut self, slot: usize, local_min: Vec3, local_max: Vec3) {
        if slot >= MAX_SCULPT_ASSETS {
            return;
        }
        let brick_world = ASSET_VOXEL_SIZE * (BRICK_SIZE as f32);
        let min_b = ((local_min - ASSET_LOCAL_MIN) / brick_world)
            .floor()
            .as_ivec3();
        let max_b = ((local_max - ASSET_LOCAL_MIN) / brick_world)
            .floor()
            .as_ivec3();

        let max_idx = ASSET_BRICKS_PER_AXIS as i32 - 1;
        let x0 = min_b.x.clamp(0, max_idx) as u32;
        let y0 = min_b.y.clamp(0, max_idx) as u32;
        let z0 = min_b.z.clamp(0, max_idx) as u32;
        let x1 = max_b.x.clamp(0, max_idx) as u32;
        let y1 = max_b.y.clamp(0, max_idx) as u32;
        let z1 = max_b.z.clamp(0, max_idx) as u32;

        let base = slot * (ASSET_BRICKS_PER_SLOT as usize);
        let g = ASSET_BRICKS_PER_AXIS;
        for z in z0..=z1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let b = x + y * g + z * g * g;
                    let flat = base + b as usize;
                    if !self.queued[flat] {
                        self.queued[flat] = true;
                        self.dirty_entries.push(GpuAssetDirtyBrick {
                            brick_and_slot: [b, slot as u32, 0, 0],
                        });
                    }
                }
            }
        }
        self.dirty_fleck_slots[slot] = true;
    }

    /// Synchronizes slot -> `asset.id` mappings and auto-dirties any newly added/reassigned slot.
    pub fn sync_scene_assets(&mut self, assets: &[SculptAsset]) {
        for slot in 0..MAX_SCULPT_ASSETS {
            let new_id = assets.get(slot).map(|a| a.id).unwrap_or(0);
            if self.known_asset_ids[slot] != new_id {
                self.known_asset_ids[slot] = new_id;
                if new_id != 0 {
                    self.mark_slot_all_dirty(slot);
                }
            }
        }
    }

    pub fn mark_all_assets_dirty(&mut self, count: usize) {
        for slot in 0..count.min(MAX_SCULPT_ASSETS) {
            self.mark_slot_all_dirty(slot);
        }
    }

    pub fn take_dirty_bricks(&mut self) -> Vec<GpuAssetDirtyBrick> {
        for entry in &self.dirty_entries {
            let b = entry.brick_and_slot[0] as usize;
            let slot = entry.brick_and_slot[1] as usize;
            let flat = slot * (ASSET_BRICKS_PER_SLOT as usize) + b;
            if flat < self.queued.len() {
                self.queued[flat] = false;
            }
        }
        std::mem::take(&mut self.dirty_entries)
    }

    pub fn take_dirty_fleck_slots(&mut self) -> Vec<u32> {
        let mut out = Vec::new();
        for (slot, flag) in self.dirty_fleck_slots.iter_mut().enumerate() {
            if *flag {
                *flag = false;
                out.push(slot as u32);
            }
        }
        out
    }

    pub fn has_dirty(&self) -> bool {
        !self.dirty_entries.is_empty()
    }
}

impl Default for AssetDirtyTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Exact CPU SDF Math (mirrors WGSL shaders for zero-lag physics & ray picking)
// ============================================================================

#[inline]
pub fn eval_primitive_sdf(local_p: Vec3, scale: Vec3, shape: SculptShape) -> f32 {
    match shape {
        SculptShape::Sphere => {
            let k0 = (local_p / scale).length();
            if k0 < 1e-5 {
                -scale.min_element()
            } else {
                let k1 = (local_p / (scale * scale)).length();
                k0 * (k0 - 1.0) / k1
            }
        }
        SculptShape::RoundedBox => {
            let round = scale.min_element() * 0.18;
            let b = (scale - Vec3::splat(round)).max(Vec3::splat(0.01));
            let q = local_p.abs() - b;
            q.max(Vec3::ZERO).length() + q.max_element().min(0.0) - round
        }
        SculptShape::Capsule => {
            let r = scale.x.max(scale.z);
            let h = (scale.y - r).max(0.0);
            let mut p = local_p;
            p.y -= p.y.clamp(-h, h);
            p.length() - r
        }
        SculptShape::Cylinder => {
            let r = scale.x.max(scale.z);
            let h = scale.y;
            let d = Vec2::new(Vec2::new(local_p.x, local_p.z).length(), local_p.y.abs())
                - Vec2::new(r, h);
            d.max_element().min(0.0) + d.max(Vec2::ZERO).length() - 0.04 * r.min(h)
        }
        SculptShape::Cone => {
            let r = scale.x.max(scale.z);
            let h = scale.y * 2.0;
            let q = Vec2::new(r, -h);
            let w = Vec2::new(Vec2::new(local_p.x, local_p.z).length(), local_p.y - scale.y);
            let a = w - q * (w.dot(q) / q.dot(q)).clamp(0.0, 1.0);
            let b = w - q * Vec2::new((w.x / q.x).clamp(0.0, 1.0), 1.0);
            let k = q.y.signum();
            let d = a.dot(a).min(b.dot(b));
            let s = (k * (w.x * q.y - w.y * q.x)).max(k * (w.y - q.y));
            d.sqrt() * s.signum()
        }
        SculptShape::Torus => {
            let major = scale.x.max(scale.z);
            let minor = scale.y.max(0.05);
            let q = Vec2::new(Vec2::new(local_p.x, local_p.z).length() - major, local_p.y);
            q.length() - minor
        }
    }
}

#[inline]
pub fn smooth_min(a: f32, b: f32, k: f32) -> (f32, f32) {
    let k = k.max(0.001);
    let h = (k - (a - b).abs()).max(0.0) / k;
    let d = a.min(b) - h * h * k * 0.25;
    let blend_weight = if a < b { 0.5 * h } else { 1.0 - 0.5 * h };
    (d, blend_weight)
}

#[inline]
pub fn smooth_subtract(base_d: f32, cut_d: f32, k: f32) -> (f32, f32) {
    let k = k.max(0.001);
    let h = (0.5 - 0.5 * (base_d + cut_d) / k).clamp(0.0, 1.0);
    let d = base_d + (-cut_d - base_d) * h + k * h * (1.0 - h);
    (d, h)
}

/// Evaluates a `SculptAsset`'s CSG edit list in its local coordinate space.
pub fn eval_asset_distance(asset: &SculptAsset, local_p: Vec3) -> f32 {
    let mut d = 1000.0_f32;
    for edit in &asset.edits {
        if edit.op == SculptOp::Paint {
            continue;
        }
        let p = edit.rotation.conjugate() * (local_p - edit.position);
        let prim_d = eval_primitive_sdf(p, edit.scale, edit.shape);
        match edit.op {
            SculptOp::Add => (d, _) = smooth_min(d, prim_d, edit.blend),
            SculptOp::Subtract => (d, _) = smooth_subtract(d, prim_d, edit.blend),
            SculptOp::Paint => {}
        }
    }
    d
}

/// Evaluates the world distance of a single `SceneProp` instance at `world_pos` and `time`.
pub fn eval_single_prop_distance(scene: &TraumScene, prop_idx: usize, world_pos: Vec3, time: f32) -> f32 {
    let prop = &scene.props[prop_idx];
    let scale = prop.uniform_scale.max(0.1);
    let (p_pos, p_rot) = scene.evaluate_prop_pose(prop.id, time);
    let delta = world_pos - p_pos;

    if let Some(asset) = scene.asset_by_id(prop.asset_id) {
        let bound_r = asset.local_bound_radius() * scale;
        let center_d = delta.length();
        if center_d > bound_r + 1.2 {
            return center_d - bound_r;
        }
        let local_p = (p_rot.conjugate() * delta) / scale;
        eval_asset_distance(asset, local_p) * scale
    } else {
        let local_p = (p_rot.conjugate() * delta) / scale;
        eval_primitive_sdf(local_p, prop.scale, prop.shape) * scale
    }
}

/// Evaluates `(signed_distance, closest_prop_id)` across `world_edits` (`0`) and all `props` (`prop.id`).
pub fn eval_scene_distance_with_id(scene: &TraumScene, world_pos: Vec3, time: f32) -> (f32, u32) {
    let mut best_d = eval_world_only_distance(scene, world_pos);
    let mut best_id = 0u32;

    for idx in 0..scene.props.len() {
        let pd = eval_single_prop_distance(scene, idx, world_pos, time);
        if pd < best_d {
            best_d = pd;
            best_id = scene.props[idx].id;
        }
    }
    (best_d, best_id)
}

pub fn eval_scene_distance(scene: &TraumScene, world_pos: Vec3, time: f32) -> f32 {
    eval_scene_distance_with_id(scene, world_pos, time).0
}

pub fn eval_world_only_distance(scene: &TraumScene, world_pos: Vec3) -> f32 {
    let mut d = 1000.0_f32;
    for edit in &scene.world_edits {
        if edit.op == SculptOp::Paint {
            continue;
        }
        let delta = world_pos - edit.position;
        let max_r = edit.scale.max_element() * 1.45 + edit.blend * 1.35 + 2.8;
        if d < 2.5 && delta.length_squared() > max_r * max_r {
            continue;
        }
        let local_p = edit.rotation.conjugate() * delta;
        let prim_d = eval_primitive_sdf(local_p, edit.scale, edit.shape);
        match edit.op {
            SculptOp::Add => (d, _) = smooth_min(d, prim_d, edit.blend),
            SculptOp::Subtract => (d, _) = smooth_subtract(d, prim_d, edit.blend),
            SculptOp::Paint => {}
        }
    }
    d
}

/// Evaluates the outward surface normal at `world_pos` via tetrahedron gradient finite differences.
pub fn eval_scene_normal(scene: &TraumScene, world_pos: Vec3, time: f32) -> Vec3 {
    let e = 0.025_f32;
    let k1 = Vec3::new(1.0, -1.0, -1.0);
    let k2 = Vec3::new(-1.0, -1.0, 1.0);
    let k3 = Vec3::new(-1.0, 1.0, -1.0);
    let k4 = Vec3::new(1.0, 1.0, 1.0);
    let grad = k1 * eval_scene_distance(scene, world_pos + k1 * e, time)
        + k2 * eval_scene_distance(scene, world_pos + k2 * e, time)
        + k3 * eval_scene_distance(scene, world_pos + k3 * e, time)
        + k4 * eval_scene_distance(scene, world_pos + k4 * e, time);
    grad.try_normalize().unwrap_or(Vec3::Y)
}

/// Raycasts against a single specific `SceneProp` (used when Scoped In so brushes magnetize to the active toy!).
pub fn raycast_single_prop(
    scene: &TraumScene,
    prop_id: u32,
    ray_origin: Vec3,
    ray_dir: Vec3,
    time: f32,
    max_dist: f32,
) -> Option<(Vec3, Vec3, f32)> {
    let prop_idx = scene.props.iter().position(|p| p.id == prop_id)?;
    let dir = ray_dir.normalize_or_zero();
    let mut t = 0.05_f32;
    for _ in 0..96 {
        let p = ray_origin + dir * t;
        let d = eval_single_prop_distance(scene, prop_idx, p, time);
        if d < 0.012 {
            let e = 0.02_f32;
            let k1 = Vec3::new(1.0, -1.0, -1.0);
            let k2 = Vec3::new(-1.0, -1.0, 1.0);
            let k3 = Vec3::new(-1.0, 1.0, -1.0);
            let k4 = Vec3::new(1.0, 1.0, 1.0);
            let grad = k1 * eval_single_prop_distance(scene, prop_idx, p + k1 * e, time)
                + k2 * eval_single_prop_distance(scene, prop_idx, p + k2 * e, time)
                + k3 * eval_single_prop_distance(scene, prop_idx, p + k3 * e, time)
                + k4 * eval_single_prop_distance(scene, prop_idx, p + k4 * e, time);
            let normal = grad.try_normalize().unwrap_or(Vec3::Y);
            return Some((p, normal, t));
        }
        t += d.max(0.018);
        if t > max_dist {
            break;
        }
    }
    None
}

/// Raycasts against all animated `SceneProp`s and the world sculpt (`prop_id = 0`).
pub fn raycast_scene(
    scene: &TraumScene,
    ray_origin: Vec3,
    ray_dir: Vec3,
    time: f32,
    max_dist: f32,
) -> Option<(u32, Vec3, Vec3, f32)> {
    let dir = ray_dir.normalize_or_zero();
    let mut t = 0.05_f32;
    for _ in 0..112 {
        let p = ray_origin + dir * t;
        let (min_d, best_prop_id) = eval_scene_distance_with_id(scene, p, time);

        if min_d < 0.015 {
            let normal = eval_scene_normal(scene, p, time);
            return Some((best_prop_id, p, normal, t));
        }
        t += min_d.max(0.02);
        if t > max_dist {
            break;
        }
    }
    None
}

/// Resolves sphere collision against the live SDF world + custom sculpted props!
/// Includes upward anti-embed pop-out (so a player can never get stuck underground) and
/// wall-safe vertical foot probing so standing beside walls/arches still detects ground reliably.
pub fn resolve_sphere_sdf_collision(
    scene: &TraumScene,
    mut center: Vec3,
    radius: f32,
    time: f32,
) -> (Vec3, bool, Vec3, u32) {
    let mut grounded = false;
    let mut ground_normal = Vec3::Y;
    let mut standing_prop_id = 0u32;

    // 1. If ever deeply embedded inside solid clay (`d < 0`), lift upward first toward open air
    //    so interior CSG pockets or downward underside normals can never trap the character!
    let initial_d = eval_scene_distance(scene, center, time);
    if initial_d < 0.02 {
        for step in 1..=32 {
            let test_p = center + Vec3::Y * (step as f32 * 0.22);
            if eval_scene_distance(scene, test_p, time) >= radius {
                center = test_p;
                grounded = true;
                break;
            }
        }
    }

    // 2. Iterative SDF sphere depenetration + step-up assist over small sculpted clay lips
    for _ in 0..6 {
        let (d, hit_prop_id) = eval_scene_distance_with_id(scene, center, time);
        let penetration = radius - d;
        if penetration > 0.0005 {
            let mut n = eval_scene_normal(scene, center, time);
            // Bias slightly upward on shallow steps/bumps so walking across rough clay never snags
            if n.y > -0.15 && n.y < 0.48 && penetration < radius * 0.35 {
                n = (n + Vec3::Y * 0.45).normalize_or_zero();
            }
            center += n * (penetration + 0.001);
            if n.y > 0.45 {
                grounded = true;
                ground_normal = n;
                standing_prop_id = hit_prop_id;
            }
        } else {
            break;
        }
    }

    // 3. Dedicated downward foot-support check (evaluates vertical clearance directly below feet
    //    so touching a vertical wall/arch never masks the floor underneath!)
    if !grounded {
        let foot_bottom = center - Vec3::Y * (radius + 0.08);
        let (foot_d, hit_prop_id) = eval_scene_distance_with_id(scene, foot_bottom, time);
        let center_d = eval_scene_distance(scene, center, time);
        // If point right below feet is inside/near clay (`foot_d < 0.12`) while body center is clear:
        if foot_d < 0.12 && center_d > radius * 0.75 {
            let n = eval_scene_normal(scene, center - Vec3::Y * (radius * 0.65), time);
            grounded = true;
            ground_normal = if n.y > 0.35 { n } else { Vec3::Y };
            standing_prop_id = hit_prop_id;
        }
    }

    (center, grounded, ground_normal, standing_prop_id)
}

/// Finds a guaranteed open-air spawn point directly above the island/prop surface near `(x, z)`.
pub fn find_safe_play_spawn(scene: &TraumScene, hint_xz: Vec2, radius: f32, time: f32) -> Vec3 {
    for candidate in [hint_xz, Vec2::new(0.0, 1.5), Vec2::ZERO] {
        let origin = Vec3::new(candidate.x, 12.0, candidate.y);
        if let Some((_, hit_pos, normal, _)) = raycast_scene(scene, origin, Vec3::NEG_Y, time, 22.0)
        {
            if normal.y > 0.35 {
                let mut spawn = hit_pos + Vec3::Y * (radius + 0.55);
                for _ in 0..16 {
                    if eval_scene_distance(scene, spawn, time) >= radius + 0.05 {
                        return spawn;
                    }
                    spawn.y += 0.25;
                }
            }
        }
    }
    Vec3::new(0.0, 4.5, 1.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_struct_alignments_are_vec4_clean() {
        assert_eq!(std::mem::size_of::<GpuSculptEdit>(), 80);
        assert_eq!(std::mem::size_of::<GpuAssetHeader>(), 32);
        assert_eq!(std::mem::size_of::<GpuAssetDirtyBrick>(), 16);
        assert_eq!(std::mem::size_of::<GpuSceneProp>(), 80);
        assert_eq!(std::mem::size_of::<GpuConnectorVisual>(), 32);
        assert_eq!(std::mem::size_of::<GpuFleck>(), 48);
    }

    #[test]
    fn starter_island_raycast_and_custom_prop_picking_work() {
        let scene = TraumScene::starter_diorama_island();
        let hit = raycast_scene(&scene, Vec3::new(0.0, 8.0, 0.0), Vec3::NEG_Y, 0.0, 20.0);
        assert!(hit.is_some(), "Expected raycast to hit starter island");
        let (prop_id, pos, normal, _) = hit.unwrap();
        assert_eq!(prop_id, 0);
        assert!(pos.y.abs() < 1.0);
        assert!(normal.y > 0.7);

        // Ray straight down onto Stepping Stone A (`id = 1` at `(0.4, 1.35, 2.8)`) should pick `prop_id == 1`!
        let prop_hit = raycast_scene(&scene, Vec3::new(0.4, 5.0, 2.8), Vec3::NEG_Y, 0.0, 12.0);
        assert!(prop_hit.is_some(), "Expected raycast to hit custom sculpted Stepping Stone");
        assert_eq!(prop_hit.unwrap().0, 1);

        let arch_d = eval_world_only_distance(&scene, Vec3::new(-2.4, 0.75, -1.8));
        assert!(arch_d > 0.2, "Carved tunnel arch should be hollow air, got d={arch_d}");
    }

    #[test]
    fn asset_dirty_tracker_marks_local_bricks_only() {
        let mut tracker = AssetDirtyTracker::new();
        tracker.mark_slot_aabb_dirty(2, Vec3::splat(-0.2), Vec3::splat(0.2));
        let dirty = tracker.take_dirty_bricks();
        assert!(!dirty.is_empty());
        assert!(dirty.len() <= 8, "Small local stroke should only dirty <= 8 asset bricks, got {}", dirty.len());
        assert!(dirty.iter().all(|b| b.brick_and_slot[1] == 2));
        let fleck_slots = tracker.take_dirty_fleck_slots();
        assert_eq!(fleck_slots, vec![2]);
    }
}
