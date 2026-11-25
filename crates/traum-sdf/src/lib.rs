//! SDF brick volume tracking, GPU Pod data layouts, and shared CPU/GPU CSG distance math.

use bytemuck::{Pod, Zeroable};
use glam::{UVec3, Vec2, Vec3};
use traum_core::{SculptEdit, SculptOp, SculptShape, TraumScene};

/// Brick dimension (`8x8x8` voxels per brick).
pub const BRICK_SIZE: u32 = 8;

/// Coarse brick grid dimensions (`24 x 16 x 24` = 9,216 bricks -> `192 x 128 x 192` voxels).
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

/// Maximum surface flecks emitted into the GPU painterly splat buffer.
pub const MAX_SURFACE_FLECKS: u32 = 450_000;

/// Maximum CSG edits uploaded to the GPU storage buffer.
pub const MAX_GPU_EDITS: usize = 4096;

///Maximum animated props rendered simultaneously.
pub const MAX_GPU_PROPS: usize = 64;

/// 80-byte (`5 x vec4`) GPU representation of a CSG `SculptEdit`.
/// Uses exclusively `vec4` fields to guarantee identical std430/std140 layout in Rust and WGSL.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuSculptEdit {
    /// `xyz` = world position, `w` = smooth blend radius `k`.
    pub pos_blend: [f32; 4],
    /// `xyzw` = inverse rotation quaternion (`rot.conjugate()`).
    pub inv_rot: [f32; 4],
    /// `xyz` = primitive scale/extents, `w` = looseness (`0.0..1.0`).
    pub scale_loose: [f32; 4],
    /// `rgb` = linear clay color, `w` = reserved material param.
    pub color_pad: [f32; 4],
    /// `[shape, op, fleck_style, prop_id]`.
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
                edit.scale.x.max(0.04),
                edit.scale.y.max(0.04),
                edit.scale.z.max(0.04),
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

/// 64-byte (`4 x vec4`) GPU representation of an animated `SceneProp`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuSceneProp {
    /// `xyz` = animated world position, `w` = bounding radius.
    pub pos_radius: [f32; 4],
    /// `xyzw` = inverse rotation quaternion.
    pub inv_rot: [f32; 4],
    /// `xyz` = scale, `w` = looseness.
    pub scale_loose: [f32; 4],
    /// `rgb` = linear color, `w` = packed `(shape | (fleck_style << 8) | (prop_id << 16))` as bits? No, separate `meta`!
    pub color_style: [f32; 4],
    /// `[shape, fleck_style, prop_id, selected_flag]`.
    pub meta: [u32; 4],
}

/// 48-byte (`3 x vec4`) GPU representation of a single painterly surface fleck.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuFleck {
    /// `xyz` = world surface position, `w` = fleck radius.
    pub pos_size: [f32; 4],
    /// `xyz` = surface normal, `w` = looseness.
    pub normal_loose: [f32; 4],
    /// `rgb` = linear color, `w` = packed `(style_index + frac_seed)` (`floor(w)` = style, `fract(w)` = random seed `0..0.999`).
    pub color_style_seed: [f32; 4],
}

/// Tracks which `8x8x8` bricks in the `24x16x24` grid need GPU compute re-evaluation.
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

    /// Marks only the `8x8x8` bricks overlapping `[aabb_min, aabb_max]` as dirty.
    pub fn mark_aabb_dirty(&mut self, aabb_min: Vec3, aabb_max: Vec3) {
        let brick_world = VOXEL_SIZE * (BRICK_SIZE as f32);
        let min_b = ((aabb_min - VOLUME_MIN) / brick_world).floor().as_ivec3();
        let max_b = ((aabb_max - VOLUME_MIN) / brick_world).ceil().as_ivec3();

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

// ============================================================================
// Exact CPU SDF Math (mirrors WGSL shaders for zero-lag physics & ray picking)
// ============================================================================

#[inline]
pub fn eval_primitive_sdf(local_p: Vec3, scale: Vec3, shape: SculptShape) -> f32 {
    match shape {
        SculptShape::Sphere => {
            // Ellipsoid approximation that stays exact for uniform spheres
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

/// Evaluates the exact signed distance at `world_pos` across both `world_edits` and animated `props` (for Play Mode collisions & fallback picking).
pub fn eval_scene_distance(scene: &TraumScene, world_pos: Vec3, time: f32) -> f32 {
    let mut d = 1000.0_f32;
    for edit in &scene.world_edits {
        if edit.op == SculptOp::Paint {
            continue;
        }
        let local_p = edit.rotation.conjugate() * (world_pos - edit.position);
        let prim_d = eval_primitive_sdf(local_p, edit.scale, edit.shape);
        match edit.op {
            SculptOp::Add => {
                (d, _) = smooth_min(d, prim_d, edit.blend);
            }
            SculptOp::Subtract => {
                (d, _) = smooth_subtract(d, prim_d, edit.blend);
            }
            SculptOp::Paint => {}
        }
    }

    for prop in &scene.props {
        let (p_pos, p_rot) = prop.evaluate_pose(time);
        let local_p = p_rot.conjugate() * (world_pos - p_pos);
        let prop_d = eval_primitive_sdf(local_p, prop.scale, prop.shape);
        d = d.min(prop_d);
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

/// Raycasts against animated `SceneProp`s first (returns `(prop_id, hit_pos, normal, t)`), or the world sculpt (`prop_id = 0`).
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
        // Check props individually so we know exact prop_id if hit!
        let mut best_prop_id = 0u32;
        let mut min_d = eval_world_only_distance(scene, p);

        for prop in &scene.props {
            let (p_pos, p_rot) = prop.evaluate_pose(time);
            let local_p = p_rot.conjugate() * (p - p_pos);
            let pd = eval_primitive_sdf(local_p, prop.scale, prop.shape);
            if pd < min_d {
                min_d = pd;
                best_prop_id = prop.id;
            }
        }

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

pub fn eval_world_only_distance(scene: &TraumScene, world_pos: Vec3) -> f32 {
    let mut d = 1000.0_f32;
    for edit in &scene.world_edits {
        if edit.op == SculptOp::Paint {
            continue;
        }
        let local_p = edit.rotation.conjugate() * (world_pos - edit.position);
        let prim_d = eval_primitive_sdf(local_p, edit.scale, edit.shape);
        match edit.op {
            SculptOp::Add => (d, _) = smooth_min(d, prim_d, edit.blend),
            SculptOp::Subtract => (d, _) = smooth_subtract(d, prim_d, edit.blend),
            SculptOp::Paint => {}
        }
    }
    d
}

/// Resolves sphere collision against the live SDF world + props for Play Mode character movement!
/// Returns `(resolved_center, grounded, ground_normal)`.
pub fn resolve_sphere_sdf_collision(
    scene: &TraumScene,
    mut center: Vec3,
    radius: f32,
    time: f32,
) -> (Vec3, bool, Vec3) {
    let mut grounded = false;
    let mut ground_normal = Vec3::Y;

    for _ in 0..4 {
        let d = eval_scene_distance(scene, center, time);
        let penetration = radius - d;
        if penetration > 0.0 {
            let n = eval_scene_normal(scene, center, time);
            center += n * penetration;
            if n.y > 0.52 {
                grounded = true;
                ground_normal = n;
            }
        } else {
            break;
        }
    }

    // Also check slight ground probe right beneath feet for stable slope walking
    if !grounded {
        let foot_d = eval_scene_distance(scene, center - Vec3::Y * 0.06, time);
        if foot_d <= radius + 0.02 {
            let n = eval_scene_normal(scene, center - Vec3::Y * 0.06, time);
            if n.y > 0.52 {
                grounded = true;
                ground_normal = n;
            }
        }
    }

    (center, grounded, ground_normal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_struct_alignments_are_vec4_clean() {
        assert_eq!(std::mem::size_of::<GpuSculptEdit>(), 80);
        assert_eq!(std::mem::size_of::<GpuSceneProp>(), 80);
        assert_eq!(std::mem::size_of::<GpuFleck>(), 48);
    }

    #[test]
    fn starter_island_raycast_and_arch_carve_work() {
        let scene = TraumScene::starter_diorama_island();
        // Ray straight down onto the center meadow should hit the island top near y ~ 0.0
        let hit = raycast_scene(&scene, Vec3::new(0.0, 8.0, 0.0), Vec3::NEG_Y, 0.0, 20.0);
        assert!(hit.is_some(), "Expected raycast to hit starter island");
        let (prop_id, pos, normal, _) = hit.unwrap();
        assert_eq!(prop_id, 0);
        assert!(pos.y.abs() < 1.0, "Island meadow surface near y=0, got {}", pos.y);
        assert!(normal.y > 0.7, "Expected upward normal, got {:?}", normal);

        // Inside the carved archway tunnel at (-2.4, 0.75, -1.8), SDF should be positive (open air!)
        let arch_d = eval_world_only_distance(&scene, Vec3::new(-2.4, 0.75, -1.8));
        assert!(arch_d > 0.2, "Carved tunnel arch should be hollow air, got d={}", arch_d);
    }

    #[test]
    fn brick_dirty_tracker_marks_local_region_only() {
        let mut tracker = BrickDirtyTracker::new();
        let all = tracker.take_dirty_bricks();
        assert_eq!(all.len(), (BRICK_GRID.x * BRICK_GRID.y * BRICK_GRID.z) as usize);
        assert!(!tracker.has_dirty());

        tracker.mark_aabb_dirty(Vec3::splat(-0.5), Vec3::splat(0.5));
        let local = tracker.take_dirty_bricks();
        assert!(!local.is_empty());
        assert!(local.len() <= 27, "Small brush should only dirty <= 27 bricks, got {}", local.len());
    }
}
