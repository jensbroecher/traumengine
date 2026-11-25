// Compute Pass 1: Sparse Dirty 8x8x8 SDF Brick Evaluator

struct ComputeUniforms {
    volume_min_voxel: vec4<f32>,
    volume_dims_edits: vec4<u32>,
    brick_grid_dirty: vec4<u32>,
    counts: vec4<u32>,
};

struct GpuSculptEdit {
    pos_blend: vec4<f32>,
    inv_rot: vec4<f32>,
    scale_loose: vec4<f32>,
    color_pad: vec4<f32>,
    info: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u_comp: ComputeUniforms;
@group(0) @binding(1) var<storage, read> edits: array<GpuSculptEdit>;
@group(0) @binding(2) var<storage, read> dirty_bricks: array<u32>;
@group(0) @binding(3) var sdf_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var color_out: texture_storage_3d<rgba16float, write>;

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
fn update_dirty_bricks(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let dirty_idx = wg_id.x;
    if (dirty_idx >= u_comp.brick_grid_dirty.w) {
        return;
    }

    let flat_brick = dirty_bricks[dirty_idx];
    let gx = u_comp.brick_grid_dirty.x;
    let gy = u_comp.brick_grid_dirty.y;
    let bx = flat_brick % gx;
    let by = (flat_brick / gx) % gy;
    let bz = flat_brick / (gx * gy);

    let base_voxel = vec3<u32>(bx, by, bz) * 8u + local_id * 2u;
    let edit_count = u_comp.volume_dims_edits.w;
    let vol_min = u_comp.volume_min_voxel.xyz;
    let voxel_size = u_comp.volume_min_voxel.w;

    for (var dz = 0u; dz < 2u; dz = dz + 1u) {
        for (var dy = 0u; dy < 2u; dy = dy + 1u) {
            for (var dx = 0u; dx < 2u; dx = dx + 1u) {
                let v_coord = base_voxel + vec3<u32>(dx, dy, dz);
                let world_p = vol_min + (vec3<f32>(v_coord) + vec3<f32>(0.5)) * voxel_size;

                var d = 20.0;
                var col = vec3<f32>(0.80, 0.45, 0.32);
                var loose = 0.2;
                var style = 0.0;

                for (var i = 0u; i < edit_count; i = i + 1u) {
                    let e = edits[i];
                    let pos = e.pos_blend.xyz;
                    let k = max(e.pos_blend.w, 0.01);
                    let scale = e.scale_loose.xyz;

                    let max_s = max(scale.x, max(scale.y, scale.z)) * 1.45 + k * 1.35;
                    let delta = world_p - pos;
                    let dist_sq = dot(delta, delta);
                    if (dist_sq > (max_s + 2.8) * (max_s + 2.8) && d < 2.5) {
                        continue;
                    }

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
                        let mask = 1.0 - smoothstep(-k * 0.4, k + 0.18, prim_d);
                        col = mix(col, e_col, mask);
                        loose = mix(loose, e_loose, mask);
                        if (mask > 0.45) {
                            style = e_style;
                        }
                    }
                }

                let coord_i = vec3<i32>(v_coord);
                textureStore(sdf_out, coord_i, vec4<f32>(d, loose, style, 1.0));
                textureStore(color_out, coord_i, vec4<f32>(col, 1.0));
            }
        }
    }
}
