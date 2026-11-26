// Compute Pass 1B: Dirty 8x8x8 Brick Evaluator for the Reusable Sculpt Asset 3D Atlas (`256x128x192`)

struct AssetComputeUniforms {
    // xyz = asset_local_min (-3.04), w = asset_voxel_size (0.095)
    local_min_voxel: vec4<f32>,
    // x = dirty_brick_count, y = bricks_per_axis (8), z = tile_dim (64), w = pad
    counts: vec4<u32>,
};

struct GpuSculptEdit {
    pos_blend: vec4<f32>,
    inv_rot: vec4<f32>,
    scale_loose: vec4<f32>,
    color_pad: vec4<f32>,
    info: vec4<u32>,
};

struct GpuAssetHeader {
    // x = edit_offset, y = edit_count, z = tile_x, w = tile_y
    edit_and_tile_xy: vec4<u32>,
    // x = tile_z, y = asset_id, z = pad, w = pad
    tile_z_and_meta: vec4<u32>,
};

struct GpuAssetDirtyBrick {
    // x = local_brick_idx (0..512), y = asset_slot (0..24), zw = pad
    brick_and_slot: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u_acomp: AssetComputeUniforms;
@group(0) @binding(1) var<storage, read> asset_edits: array<GpuSculptEdit>;
@group(0) @binding(2) var<storage, read> asset_headers: array<GpuAssetHeader>;
@group(0) @binding(3) var<storage, read> dirty_bricks: array<GpuAssetDirtyBrick>;
@group(0) @binding(4) var asset_sdf_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var asset_color_out: texture_storage_3d<rgba16float, write>;

fn quat_mul_vec3(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn eval_primitive(local_p: vec3<f32>, scale: vec3<f32>, shape: u32) -> f32 {
    switch shape {
        case 0u: {
            let k0 = length(local_p / scale);
            if (k0 < 1e-5) {
                return -min(scale.x, min(scale.y, scale.z));
            }
            let k1 = length(local_p / (scale * scale));
            return k0 * (k0 - 1.0) / k1;
        }
        case 1u: {
            let rnd = min(scale.x, min(scale.y, scale.z)) * 0.18;
            let b = max(scale - vec3<f32>(rnd), vec3<f32>(0.01));
            let q = abs(local_p) - b;
            return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - rnd;
        }
        case 2u: {
            let r = max(scale.x, scale.z);
            let h = max(scale.y - r, 0.0);
            var p = local_p;
            p.y -= clamp(p.y, -h, h);
            return length(p) - r;
        }
        case 3u: {
            let r = max(scale.x, scale.z);
            let h = scale.y;
            let d = vec2<f32>(length(local_p.xz), abs(local_p.y)) - vec2<f32>(r, h);
            return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0))) - 0.04 * min(r, h);
        }
        case 4u: {
            let r = max(scale.x, scale.z);
            let h = scale.y * 2.0;
            let q = vec2<f32>(r, -h);
            let w = vec2<f32>(length(local_p.xz), local_p.y - scale.y);
            let a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
            let b = w - q * vec2<f32>(clamp(w.x / q.x, 0.0, 1.0), 1.0);
            let k = sign(q.y);
            let d = min(dot(a, a), dot(b, b));
            let s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
            return sqrt(d) * sign(s);
        }
        default: {
            let major = max(scale.x, scale.z);
            let minor = max(scale.y, 0.05);
            let q = vec2<f32>(length(local_p.xz) - major, local_p.y);
            return length(q) - minor;
        }
    }
}

@compute @workgroup_size(4, 4, 4)
fn update_dirty_asset_bricks(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let dirty_idx = wg_id.x;
    if (dirty_idx >= u_acomp.counts.x) {
        return;
    }

    let entry = dirty_bricks[dirty_idx];
    let local_brick = entry.brick_and_slot.x;
    let slot = entry.brick_and_slot.y;
    let header = asset_headers[slot];

    let g = u_acomp.counts.y; // 8 bricks per axis
    let bx = local_brick % g;
    let by = (local_brick / g) % g;
    let bz = local_brick / (g * g);

    let tile_dim = u_acomp.counts.z; // 64
    let atlas_tile_origin = vec3<u32>(
        header.edit_and_tile_xy.z,
        header.edit_and_tile_xy.w,
        header.tile_z_and_meta.x,
    ) * tile_dim;

    let base_local_vox = vec3<u32>(bx, by, bz) * 8u + local_id * 2u;
    let edit_start = header.edit_and_tile_xy.x;
    let edit_end = edit_start + header.edit_and_tile_xy.y;

    let local_min = u_acomp.local_min_voxel.xyz;
    let voxel_size = u_acomp.local_min_voxel.w;

    for (var dz = 0u; dz < 2u; dz = dz + 1u) {
        for (var dy = 0u; dy < 2u; dy = dy + 1u) {
            for (var dx = 0u; dx < 2u; dx = dx + 1u) {
                let lv = base_local_vox + vec3<u32>(dx, dy, dz);
                let pos_p = local_min + (vec3<f32>(lv) + vec3<f32>(0.5)) * voxel_size;

                var d = 12.0;
                var col = vec3<f32>(0.84, 0.46, 0.33);
                var loose = 0.25;
                var style = 0.0;

                for (var i = edit_start; i < edit_end; i = i + 1u) {
                    let e = asset_edits[i];
                    let e_pos = e.pos_blend.xyz;
                    let k = max(e.pos_blend.w, 0.01);
                    let scale = e.scale_loose.xyz;

                    let delta = pos_p - e_pos;
                    let local_p = quat_mul_vec3(e.inv_rot, delta);
                    let prim_d = eval_primitive(local_p, scale, e.info.x);
                    let op = e.info.y;
                    let e_col = e.color_pad.xyz;
                    let e_loose = e.scale_loose.w;
                    let e_style = f32(e.info.z);

                    if (op == 0u) {
                        let h = max(k - abs(d - prim_d), 0.0) / k;
                        let w = select(1.0 - 0.5 * h, 0.5 * h, d < prim_d);
                        let new_d = min(d, prim_d) - h * h * k * 0.25;
                        col = mix(col, e_col, w);
                        loose = mix(loose, e_loose, w);
                        if (w > 0.48) {
                            style = e_style;
                        }
                        d = new_d;
                    } else if (op == 1u) {
                        let h = clamp(0.5 - 0.5 * (d + prim_d) / k, 0.0, 1.0);
                        let new_d = mix(d, -prim_d, h) + k * h * (1.0 - h);
                        col = mix(col, e_col, h * 0.65);
                        loose = mix(loose, e_loose, h * 0.5);
                        d = new_d;
                    } else {
                        let mask = 1.0 - smoothstep(-k * 0.4, k + 0.14, prim_d);
                        col = mix(col, e_col, mask);
                        loose = mix(loose, e_loose, mask);
                        if (mask > 0.45) {
                            style = e_style;
                        }
                    }
                }

                // Keep a 1-voxel positive air shell at tile borders so trilinear sampling is 100% watertight & isolated
                if (lv.x == 0u || lv.y == 0u || lv.z == 0u || lv.x >= 63u || lv.y >= 63u || lv.z >= 63u) {
                    d = max(d, 0.12);
                }

                let atlas_coord = vec3<i32>(atlas_tile_origin + lv);
                textureStore(asset_sdf_out, atlas_coord, vec4<f32>(d, loose, style, 1.0));
                textureStore(asset_color_out, atlas_coord, vec4<f32>(col, 1.0));
            }
        }
    }
}
