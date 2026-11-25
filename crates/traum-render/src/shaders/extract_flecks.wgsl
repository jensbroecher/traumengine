// Compute Pass 2: Iso-Surface Painterly Fleck Extractor (World Bricks + Animated Toy Props!)

struct ComputeUniforms {
    volume_min_voxel: vec4<f32>,
    volume_dims_edits: vec4<u32>,
    brick_grid_dirty: vec4<u32>,
    counts: vec4<u32>,
};

struct GpuSceneProp {
    pos_radius: vec4<f32>,
    inv_rot: vec4<f32>,
    scale_loose: vec4<f32>,
    color_style: vec4<f32>,
    info: vec4<u32>,
};

struct GpuFleck {
    pos_size: vec4<f32>,
    normal_loose: vec4<f32>,
    color_style_seed: vec4<f32>,
};

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

struct FleckCounter {
    count: atomic<u32>,
};

@group(0) @binding(0) var<uniform> u_comp: ComputeUniforms;
@group(0) @binding(1) var sdf_tex: texture_3d<f32>;
@group(0) @binding(2) var color_tex: texture_3d<f32>;
@group(0) @binding(3) var vol_sampler: sampler;
@group(0) @binding(4) var<storage, read> props: array<GpuSceneProp>;
@group(0) @binding(5) var<storage, read_write> flecks: array<GpuFleck>;
@group(0) @binding(6) var<storage, read_write> fleck_counter: FleckCounter;
@group(0) @binding(7) var<storage, read_write> indirect_args: DrawIndirectArgs;

fn hash31(p: vec3<f32>) -> f32 {
    let p3 = fract(p * 0.1031);
    let d = dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z + d);
}

fn hash33(p: vec3<f32>) -> vec3<f32> {
    var p3 = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

fn quat_conj_mul(q_inv: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    // Rotate from local prop space back to world space using conjugate of inv_rot!
    let q = vec4<f32>(-q_inv.xyz, q_inv.w);
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn world_to_uvw(world_p: vec3<f32>) -> vec3<f32> {
    let vol_min = u_comp.volume_min_voxel.xyz;
    let voxel_size = u_comp.volume_min_voxel.w;
    let vol_extent = vec3<f32>(u_comp.volume_dims_edits.xyz) * voxel_size;
    return (world_p - vol_min) / vol_extent;
}

fn sample_sdf_world(world_p: vec3<f32>) -> vec4<f32> {
    let uvw = clamp(world_to_uvw(world_p), vec3<f32>(0.002), vec3<f32>(0.998));
    return textureSampleLevel(sdf_tex, vol_sampler, uvw, 0.0);
}

@compute @workgroup_size(1, 1, 1)
fn reset_fleck_counter() {
    atomicStore(&fleck_counter.count, 0u);
    indirect_args.vertex_count = 6u;
    indirect_args.instance_count = 0u;
    indirect_args.first_vertex = 0u;
    indirect_args.first_instance = 0u;
}

@compute @workgroup_size(4, 4, 4)
fn extract_surface_flecks(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid = u_comp.volume_dims_edits.xyz / 2u; // 96 x 64 x 96
    if (gid.x >= grid.x || gid.y >= grid.y || gid.z >= grid.z) {
        return;
    }

    let cell_size = u_comp.volume_min_voxel.w * 2.0; // 0.30m
    let vol_min = u_comp.volume_min_voxel.xyz;
    let cell_f = vec3<f32>(gid);
    let jitter = (hash33(cell_f + vec3<f32>(17.0, 59.0, 113.0)) - vec3<f32>(0.5)) * 0.85;
    let sample_p = vol_min + (cell_f + vec3<f32>(0.5) + jitter) * cell_size;

    let sdf_val = sample_sdf_world(sample_p);
    let d = sdf_val.x;
    if (abs(d) > cell_size * 0.92) {
        return;
    }

    let eps = u_comp.volume_min_voxel.w * 0.9;
    let nx = sample_sdf_world(sample_p + vec3<f32>(eps, 0.0, 0.0)).x - sample_sdf_world(sample_p - vec3<f32>(eps, 0.0, 0.0)).x;
    let ny = sample_sdf_world(sample_p + vec3<f32>(0.0, eps, 0.0)).x - sample_sdf_world(sample_p - vec3<f32>(0.0, eps, 0.0)).x;
    let nz = sample_sdf_world(sample_p + vec3<f32>(0.0, 0.0, eps)).x - sample_sdf_world(sample_p - vec3<f32>(0.0, 0.0, eps)).x;
    let grad_len = length(vec3<f32>(nx, ny, nz));
    if (grad_len < 1e-4) {
        return;
    }
    let normal = vec3<f32>(nx, ny, nz) / grad_len;

    let surf_p = sample_p - normal * d;
    let surf_sdf = sample_sdf_world(surf_p);
    let loose = clamp(surf_sdf.y, 0.0, 1.0);
    let style_idx = floor(surf_sdf.z + 0.5);
    let uvw = clamp(world_to_uvw(surf_p), vec3<f32>(0.002), vec3<f32>(0.998));
    let surf_col = textureSampleLevel(color_tex, vol_sampler, uvw, 0.0).xyz;

    let seed = clamp(hash31(cell_f * 1.73 + vec3<f32>(3.1, 7.7, 11.3)), 0.001, 0.998);
    let max_flecks = u_comp.counts.y;

    let idx = atomicAdd(&fleck_counter.count, 1u);
    if (idx < max_flecks) {
        let lift = 0.015 + loose * (0.05 + seed * 0.22);
        let radius = cell_size * (0.58 + loose * 0.95) * (0.8 + seed * 0.45);
        flecks[idx] = GpuFleck(
            vec4<f32>(surf_p + normal * lift, radius),
            vec4<f32>(normal, loose),
            vec4<f32>(surf_col, style_idx + seed),
        );
    }

    if (loose > 0.42) {
        let idx2 = atomicAdd(&fleck_counter.count, 1u);
        if (idx2 < max_flecks) {
            let jitter2 = hash33(cell_f * 3.17) - vec3<f32>(0.5);
            let seed2 = clamp(hash31(cell_f * 5.91), 0.001, 0.998);
            let puff_pos = surf_p + normal * (0.10 + loose * 0.36 * seed2) + jitter2 * (loose * 0.28);
            let radius2 = cell_size * (0.65 + loose * 1.15);
            flecks[idx2] = GpuFleck(
                vec4<f32>(puff_pos, radius2),
                vec4<f32>(normal, loose),
                vec4<f32>(surf_col * (0.94 + 0.14 * seed2), style_idx + seed2),
            );
        }
    }
}

// Emits 64 living surface flecks around each animated Toy Prop so moving toys keep painterly coats!
@compute @workgroup_size(64, 1, 1)
fn emit_prop_flecks(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let prop_i = wg_id.x;
    if (prop_i >= u_comp.counts.x) {
        return;
    }

    let pr = props[prop_i];
    let loose = pr.scale_loose.w;
    if (loose < 0.08) {
        return;
    }

    let fi = f32(lid.x);
    // Golden-sphere Fibonacci distribution on unit sphere
    let y = 1.0 - (fi / 63.0) * 2.0;
    let r = sqrt(max(1.0 - y * y, 0.0));
    let phi = fi * 2.3999632;
    let local_dir = vec3<f32>(cos(phi) * r, y, sin(phi) * r);

    let local_surf = local_dir * pr.scale_loose.xyz * (1.02 + loose * 0.18);
    let world_pos = pr.pos_radius.xyz + quat_conj_mul(pr.inv_rot, local_surf);
    let world_norm = normalize(quat_conj_mul(pr.inv_rot, local_dir));

    let seed = clamp(fract(sin(fi * 12.9898 + f32(prop_i) * 41.3) * 43758.5453), 0.001, 0.998);
    let style_idx = f32(pr.info.y);
    let max_f = u_comp.counts.y;

    let idx = atomicAdd(&fleck_counter.count, 1u);
    if (idx < max_f) {
        let f_rad = max(pr.scale_loose.x, pr.scale_loose.y) * (0.20 + loose * 0.28);
        flecks[idx] = GpuFleck(
            vec4<f32>(world_pos, f_rad),
            vec4<f32>(world_norm, loose),
            vec4<f32>(pr.color_style.xyz, style_idx + seed),
        );
    }
}

@compute @workgroup_size(1, 1, 1)
fn finalize_fleck_indirect() {
    let total = atomicLoad(&fleck_counter.count);
    indirect_args.instance_count = min(total, u_comp.counts.y);
}
