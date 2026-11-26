// Compute Pass 2: Iso-Surface Painterly Fleck Extractor
// - Extracts cached local iso-surface flecks per custom `SculptAsset` slot (`0..24`)
// - Extracts world island flecks + instances custom prop flecks across all Live Clones & Multi-Part Limbs!

struct ComputeUniforms {
    volume_min_voxel: vec4<f32>,
    volume_dims_edits: vec4<u32>,
    // xyz = asset_atlas_dims (256, 128, 192), w = max_asset_local_flecks (3072)
    asset_atlas_info: vec4<u32>,
    // x = prop_count, y = max_surface_flecks, z = frame_index, w = dirty_asset_slot_mask (bits 0..23)
    counts: vec4<u32>,
    // xyz = asset_local_min (-3.04), w = asset_voxel_size (0.095)
    asset_min_voxel: vec4<f32>,
};

struct GpuAssetHeader {
    edit_and_tile_xy: vec4<u32>,
    tile_z_and_meta: vec4<u32>,
};

struct GpuSceneProp {
    pos_radius: vec4<f32>,
    inv_rot: vec4<f32>,
    // x = uniform_scale, y = inv_scale, z = gizmo_ring_r, w = pad
    scale_params: vec4<f32>,
    gizmo_orb: vec4<f32>,
    // [asset_slot, prop_id, selected_flag, scoped_flag]
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
@group(0) @binding(8) var asset_sdf_tex: texture_3d<f32>;
@group(0) @binding(9) var asset_color_tex: texture_3d<f32>;
@group(0) @binding(10) var<storage, read> asset_headers: array<GpuAssetHeader>;
@group(0) @binding(11) var<storage, read_write> asset_local_flecks: array<GpuFleck>;
@group(0) @binding(12) var<storage, read_write> asset_fleck_counts: array<atomic<u32>>;

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

fn asset_local_to_atlas_uvw(slot: u32, local_p: vec3<f32>) -> vec3<f32> {
    let h = asset_headers[slot];
    let tile_origin = vec3<f32>(
        f32(h.edit_and_tile_xy.z),
        f32(h.edit_and_tile_xy.w),
        f32(h.tile_z_and_meta.x),
    ) * 64.0;
    let lmin = u_comp.asset_min_voxel.xyz;
    let lvox = u_comp.asset_min_voxel.w;
    // Clamp strictly inside [1.25 .. 62.75] texels so trilinear sampling never crosses tile borders!
    let local_texel = clamp((local_p - lmin) / lvox, vec3<f32>(1.25), vec3<f32>(62.75));
    let atlas_dims = vec3<f32>(u_comp.asset_atlas_info.xyz);
    return (tile_origin + local_texel) / atlas_dims;
}

fn sample_asset_sdf(slot: u32, local_p: vec3<f32>) -> vec4<f32> {
    let uvw = asset_local_to_atlas_uvw(slot, local_p);
    return textureSampleLevel(asset_sdf_tex, vol_sampler, uvw, 0.0);
}

@compute @workgroup_size(24, 1, 1)
fn reset_dirty_asset_flecks(@builtin(local_invocation_id) lid: vec3<u32>) {
    let slot = lid.x;
    if (slot < 24u && (u_comp.counts.w & (1u << slot)) != 0u) {
        atomicStore(&asset_fleck_counts[slot], 0u);
    }
}

@compute @workgroup_size(4, 4, 4)
fn extract_asset_local_flecks(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 32u || gid.y >= 32u || gid.z >= 32u) {
        return;
    }
    let mask = u_comp.counts.w;
    if (mask == 0u) {
        return;
    }

    let cell_size = u_comp.asset_min_voxel.w * 2.0; // 0.19m local cell
    let lmin = u_comp.asset_min_voxel.xyz;
    let cell_f = vec3<f32>(gid);
    let jitter = (hash33(cell_f + vec3<f32>(13.0, 41.0, 97.0)) - vec3<f32>(0.5)) * 0.80;
    let sample_p = lmin + (cell_f + vec3<f32>(0.5) + jitter) * cell_size;
    let eps = u_comp.asset_min_voxel.w * 0.95;
    let max_local = u_comp.asset_atlas_info.w;

    for (var slot = 0u; slot < 24u; slot = slot + 1u) {
        if ((mask & (1u << slot)) == 0u) {
            continue;
        }
        let sv = sample_asset_sdf(slot, sample_p);
        let d = sv.x;
        if (abs(d) > cell_size * 0.90) {
            continue;
        }

        let nx = sample_asset_sdf(slot, sample_p + vec3<f32>(eps, 0.0, 0.0)).x
               - sample_asset_sdf(slot, sample_p - vec3<f32>(eps, 0.0, 0.0)).x;
        let ny = sample_asset_sdf(slot, sample_p + vec3<f32>(0.0, eps, 0.0)).x
               - sample_asset_sdf(slot, sample_p - vec3<f32>(0.0, eps, 0.0)).x;
        let nz = sample_asset_sdf(slot, sample_p + vec3<f32>(0.0, 0.0, eps)).x
               - sample_asset_sdf(slot, sample_p - vec3<f32>(0.0, 0.0, eps)).x;
        let glen = length(vec3<f32>(nx, ny, nz));
        if (glen < 1e-4) {
            continue;
        }
        let normal = vec3<f32>(nx, ny, nz) / glen;
        let surf_p = sample_p - normal * d;
        let surf_sv = sample_asset_sdf(slot, surf_p);
        let loose = clamp(surf_sv.y, 0.0, 1.0);
        let style_idx = floor(surf_sv.z + 0.5);
        let uvw = asset_local_to_atlas_uvw(slot, surf_p);
        let surf_col = textureSampleLevel(asset_color_tex, vol_sampler, uvw, 0.0).xyz;

        let seed = clamp(hash31(cell_f * 2.13 + vec3<f32>(f32(slot) * 5.7)), 0.001, 0.998);
        let base_offset = slot * max_local;

        let idx = atomicAdd(&asset_fleck_counts[slot], 1u);
        if (idx < max_local) {
            let lift = 0.01 + loose * (0.04 + seed * 0.14);
            let rad = cell_size * (0.58 + loose * 0.85) * (0.82 + seed * 0.38);
            asset_local_flecks[base_offset + idx] = GpuFleck(
                vec4<f32>(surf_p + normal * lift, rad),
                vec4<f32>(normal, loose),
                vec4<f32>(surf_col, style_idx + seed),
            );
        }

        if (loose > 0.44) {
            let idx2 = atomicAdd(&asset_fleck_counts[slot], 1u);
            if (idx2 < max_local) {
                let jitter2 = hash33(cell_f * 3.71) - vec3<f32>(0.5);
                let seed2 = clamp(hash31(cell_f * 6.31), 0.001, 0.998);
                let puff_p = surf_p + normal * (0.06 + loose * 0.24 * seed2) + jitter2 * (loose * 0.18);
                let rad2 = cell_size * (0.65 + loose * 1.05);
                asset_local_flecks[base_offset + idx2] = GpuFleck(
                    vec4<f32>(puff_p, rad2),
                    vec4<f32>(normal, loose),
                    vec4<f32>(surf_col * (0.94 + 0.14 * seed2), style_idx + seed2),
                );
            }
        }
    }
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

// Instances the cached local surface flecks of each prop's `SculptAsset` into animated world space!
@compute @workgroup_size(64, 1, 1)
fn emit_prop_flecks(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let prop_i = wg_id.y;
    if (prop_i >= u_comp.counts.x) {
        return;
    }

    let pr = props[prop_i];
    let slot = pr.info.x;
    if (slot >= 24u) {
        return;
    }

    let max_local = u_comp.asset_atlas_info.w;
    let asset_count = min(atomicLoad(&asset_fleck_counts[slot]), max_local);
    let local_idx = wg_id.x * 64u + lid.x;
    if (local_idx >= asset_count) {
        return;
    }

    let lf = asset_local_flecks[slot * max_local + local_idx];
    let scale = pr.scale_params.x;
    let world_pos = pr.pos_radius.xyz + quat_conj_mul(pr.inv_rot, lf.pos_size.xyz * scale);
    let world_norm = normalize(quat_conj_mul(pr.inv_rot, lf.normal_loose.xyz));
    let world_rad = lf.pos_size.w * scale;

    // Offset seed per prop_id so Live Clones don't boil/sway in robotic lockstep!
    let base_style = floor(lf.color_style_seed.w);
    let base_seed = fract(lf.color_style_seed.w);
    let inst_seed = clamp(fract(base_seed + f32(pr.info.y) * 0.1731), 0.001, 0.998);

    let out_idx = atomicAdd(&fleck_counter.count, 1u);
    if (out_idx < u_comp.counts.y) {
        flecks[out_idx] = GpuFleck(
            vec4<f32>(world_pos, world_rad),
            vec4<f32>(world_norm, lf.normal_loose.w),
            vec4<f32>(lf.color_style_seed.xyz, base_style + inst_seed),
        );
    }
}

@compute @workgroup_size(1, 1, 1)
fn finalize_fleck_indirect() {
    let total = atomicLoad(&fleck_counter.count);
    indirect_args.instance_count = min(total, u_comp.counts.y);
}
