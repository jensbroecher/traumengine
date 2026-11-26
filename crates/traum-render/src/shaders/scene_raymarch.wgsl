// Render Pass 1: Watertight Painterly Clay Core + Custom Sculpted Props & Clones +
// Multi-Part Joint Connectors + 3D Rotate/Scale Gizmo + Playable Buddy + 3D Imp Guides!

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos_time: vec4<f32>,
    cam_fwd_focus: vec4<f32>,
    sun_dof: vec4<f32>,
    // xy = viewport_px, z = prop_count_f32, w = mirror_x_flag
    viewport_meta: vec4<f32>,
    imp_pos_size: vec4<f32>,
    // xyz = imp_normal, w = mode (0=Sculpt, 1=Paint, 2=Animate, 3=Play)
    imp_norm_mode: vec4<f32>,
    imp_shadow_hit: vec4<f32>,
    brush_color_op: vec4<f32>,
    puppet_pos_yaw: vec4<f32>,
    // x = puppet_squash, y = puppet_walk_cycle, z = connector_count_f32, w = selected_prop_id
    puppet_extra: vec4<f32>,
    vol_min_voxel: vec4<f32>,
    // xyz = volume_max, w = scoped_prop_id (0.0 if world scope)
    vol_max_scoped: vec4<f32>,
    // xyz = asset_atlas_dims (256, 128, 192), w = asset_voxel_size (0.095)
    asset_atlas_info: vec4<f32>,
    // xyz = asset_local_min (-3.04), w = scoped_asset_slot (-1.0 if none)
    asset_min_slot: vec4<f32>,
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

struct GpuConnectorVisual {
    start_kind: vec4<f32>,
    end_sel: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_frame: FrameUniforms;
@group(0) @binding(1) var sdf_tex: texture_3d<f32>;
@group(0) @binding(2) var color_tex: texture_3d<f32>;
@group(0) @binding(3) var vol_sampler: sampler;
@group(0) @binding(4) var<storage, read> props: array<GpuSceneProp>;
@group(0) @binding(5) var asset_sdf_tex: texture_3d<f32>;
@group(0) @binding(6) var asset_color_tex: texture_3d<f32>;
@group(0) @binding(7) var<storage, read> connectors: array<GpuConnectorVisual>;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var vout: VsOut;
    vout.uv = vec2<f32>(x, y);
    vout.clip_pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return vout;
}

fn quat_mul_vec3(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn quat_conj_mul(q_inv: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let q = vec4<f32>(-q_inv.xyz, q_inv.w);
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn eval_capsule_y(local_p: vec3<f32>, r: f32, half_h: f32) -> f32 {
    var p = local_p;
    p.y -= clamp(p.y, -half_h, half_h);
    return length(p) - r;
}

fn world_to_uvw(p: vec3<f32>) -> vec3<f32> {
    let vmin = u_frame.vol_min_voxel.xyz;
    let vmax = u_frame.vol_max_scoped.xyz;
    return (p - vmin) / (vmax - vmin);
}

fn sample_world_brick_sdf(p: vec3<f32>) -> vec4<f32> {
    let vmin = u_frame.vol_min_voxel.xyz;
    let vmax = u_frame.vol_max_scoped.xyz;
    let box_d = length(max(max(vmin - p, p - vmax), vec3<f32>(0.0)));
    if (box_d > 0.15) {
        return vec4<f32>(box_d + 0.25, 0.0, 0.0, 0.0);
    }
    let uvw = clamp((p - vmin) / (vmax - vmin), vec3<f32>(0.002), vec3<f32>(0.998));
    return textureSampleLevel(sdf_tex, vol_sampler, uvw, 0.0);
}

// Maps an asset slot (0..23 in a 4x2x3 grid of 64^3 tiles) + local pos to isolated atlas UVW!
fn asset_slot_uvw(slot: u32, local_p: vec3<f32>) -> vec3<f32> {
    let tx = slot % 4u;
    let ty = (slot / 4u) % 2u;
    let tz = slot / 8u;
    let tile_origin = vec3<f32>(f32(tx), f32(ty), f32(tz)) * 64.0;
    let lmin = u_frame.asset_min_slot.xyz;
    let lvox = u_frame.asset_atlas_info.w;
    let local_texel = clamp((local_p - lmin) / lvox, vec3<f32>(1.25), vec3<f32>(62.75));
    return (tile_origin + local_texel) / u_frame.asset_atlas_info.xyz;
}

fn sample_prop_asset_sdf(slot: u32, local_p: vec3<f32>) -> vec4<f32> {
    let lmax = -u_frame.asset_min_slot.xyz - vec3<f32>(0.12);
    let lmin = u_frame.asset_min_slot.xyz + vec3<f32>(0.12);
    let box_d = length(max(max(lmin - local_p, local_p - lmax), vec3<f32>(0.0)));
    if (box_d > 0.08) {
        return vec4<f32>(box_d + 0.14, 0.0, 0.0, 0.0);
    }
    let uvw = asset_slot_uvw(slot, local_p);
    return textureSampleLevel(asset_sdf_tex, vol_sampler, uvw, 0.0);
}

struct HitMaterial {
    d: f32,
    color: vec3<f32>,
    style: f32,
    selected: f32,
    in_active_scope: f32,
};

fn eval_puppet(p: vec3<f32>) -> HitMaterial {
    var m = HitMaterial(1e5, vec3<f32>(0.0), 0.0, 0.0, 1.0);
    if (u_frame.imp_norm_mode.w < 2.5) {
        return m;
    }

    let base = u_frame.puppet_pos_yaw.xyz;
    let dist_bound = length(p - base);
    if (dist_bound > 1.8) {
        m.d = dist_bound - 1.3;
        return m;
    }

    // Inverse Y-rotation (`R_y(yaw)^T`) mapping world forward `(sin(yaw), 0, cos(yaw))` -> local `+Z` `(0, 0, 1)`!
    let yaw = u_frame.puppet_pos_yaw.w;
    let cy = cos(yaw);
    let sy = sin(yaw);
    let rel = p - base;
    let lp = vec3<f32>(rel.x * cy - rel.z * sy, rel.y, rel.x * sy + rel.z * cy);

    let sq = u_frame.puppet_extra.x;
    let inv_xz = sqrt(sq);
    // Lift character visual slightly inside the 0.54m physics sphere so boots rest cleanly atop grass flecks!
    let lifted_lp = lp - vec3<f32>(0.0, 0.14, 0.0);
    let body_p = vec3<f32>(lifted_lp.x * inv_xz, lifted_lp.y / sq, lifted_lp.z * inv_xz);

    var d_body = eval_capsule_y(body_p - vec3<f32>(0.0, 0.10, 0.0), 0.32, 0.11);
    var col = vec3<f32>(0.94, 0.38, 0.26);
    var style = 0.0;

    // Bright cream front visor patch on `+Z` + teal explorer backpack on `-Z` so facing direction is unmistakable!
    let d_visor = length((body_p - vec3<f32>(0.0, 0.21, 0.22)) / vec3<f32>(0.23, 0.14, 0.13)) * 0.14 - 0.14;
    if (d_visor < d_body) {
        d_body = d_visor;
        col = vec3<f32>(0.98, 0.94, 0.86);
    }
    let d_pack = eval_capsule_y(body_p - vec3<f32>(0.0, 0.08, -0.26), 0.16, 0.07);
    if (d_pack < d_body) {
        d_body = d_pack;
        col = vec3<f32>(0.20, 0.62, 0.72);
    }

    let bob = sin(u_frame.cam_pos_time.w * 7.0) * 0.05;
    let d_pom = length(body_p - vec3<f32>(bob, 0.68, 0.06)) - 0.11;
    if (d_pom < d_body) {
        d_body = d_pom;
        col = vec3<f32>(1.0, 0.72, 0.18);
        style = 4.0;
    }

    let eye_x = abs(body_p.x) - 0.11;
    let d_eye = length(vec3<f32>(eye_x, body_p.y - 0.22, body_p.z - 0.33)) - 0.058;
    if (d_eye < d_body) {
        d_body = d_eye;
        col = vec3<f32>(0.03, 0.03, 0.05);
        style = 0.0;
    }

    let cycle = u_frame.puppet_extra.y;
    let foot_l = vec3<f32>(-0.16, -0.32 + max(0.0, sin(cycle)) * 0.15, sin(cycle) * 0.16);
    let foot_r = vec3<f32>(0.16, -0.32 + max(0.0, -sin(cycle)) * 0.15, -sin(cycle) * 0.16);
    let d_feet = min(length(lifted_lp - foot_l) - 0.11, length(lifted_lp - foot_r) - 0.11);
    if (d_feet < d_body) {
        d_body = d_feet;
        col = vec3<f32>(0.22, 0.18, 0.26);
    }

    m.d = d_body * min(sq, 1.0 / inv_xz);
    m.color = col;
    m.style = style;
    return m;
}

fn eval_full_scene_dist(p: vec3<f32>) -> f32 {
    var d = sample_world_brick_sdf(p).x;

    let prop_count = u32(u_frame.viewport_meta.z);
    for (var i = 0u; i < prop_count; i = i + 1u) {
        let pr = props[i];
        let delta = p - pr.pos_radius.xyz;
        let b_rad = pr.pos_radius.w;
        let center_d = length(delta);
        if (center_d - b_rad < d) {
            let local_p = quat_mul_vec3(pr.inv_rot, delta) * pr.scale_params.y;
            let pd = sample_prop_asset_sdf(pr.info.x, local_p).x * pr.scale_params.x;
            d = min(d, pd);
        } else {
            d = min(d, center_d - b_rad + 0.04);
        }
    }

    let pup = eval_puppet(p);
    d = min(d, pup.d);
    return d;
}

fn eval_full_material(p: vec3<f32>) -> HitMaterial {
    let ws = sample_world_brick_sdf(p);
    let uvw = clamp(world_to_uvw(p), vec3<f32>(0.002), vec3<f32>(0.998));
    let wcol = textureSampleLevel(color_tex, vol_sampler, uvw, 0.0).xyz;
    let is_world_scope = u_frame.vol_max_scoped.w < 0.5;
    var best = HitMaterial(ws.x, wcol, ws.z, 0.0, select(0.0, 1.0, is_world_scope));

    let prop_count = u32(u_frame.viewport_meta.z);
    let scoped_slot = u_frame.asset_min_slot.w;
    for (var i = 0u; i < prop_count; i = i + 1u) {
        let pr = props[i];
        let delta = p - pr.pos_radius.xyz;
        if (length(delta) - pr.pos_radius.w > best.d + 0.05) {
            continue;
        }
        let local_p = quat_mul_vec3(pr.inv_rot, delta) * pr.scale_params.y;
        let asdf = sample_prop_asset_sdf(pr.info.x, local_p);
        let pd = asdf.x * pr.scale_params.x;
        if (pd < best.d) {
            let auvw = asset_slot_uvw(pr.info.x, local_p);
            let acol = textureSampleLevel(asset_color_tex, vol_sampler, auvw, 0.0).xyz;
            best.d = pd;
            best.color = acol;
            best.style = asdf.z;
            best.selected = select(0.0, 1.0, pr.info.z == 1u && u_frame.imp_norm_mode.w < 2.5 && is_world_scope);
            best.in_active_scope = select(
                select(0.0, 1.0, abs(f32(pr.info.x) - scoped_slot) < 0.25),
                1.0,
                is_world_scope,
            );
        }
    }

    let pup = eval_puppet(p);
    if (pup.d < best.d) {
        best = pup;
    }
    return best;
}

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.026;
    let k1 = vec3<f32>(1.0, -1.0, -1.0);
    let k2 = vec3<f32>(-1.0, -1.0, 1.0);
    let k3 = vec3<f32>(-1.0, 1.0, -1.0);
    let k4 = vec3<f32>(1.0, 1.0, 1.0);
    return normalize(
        k1 * eval_full_scene_dist(p + k1 * e) +
        k2 * eval_full_scene_dist(p + k2 * e) +
        k3 * eval_full_scene_dist(p + k3 * e) +
        k4 * eval_full_scene_dist(p + k4 * e)
    );
}

fn impasto_normal_perturb(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let s = p * 9.5;
    let wave = vec3<f32>(
        sin(s.y * 1.7 + cos(s.z * 2.1)),
        sin(s.z * 1.7 + cos(s.x * 2.1)),
        sin(s.x * 1.7 + cos(s.y * 2.1))
    );
    return normalize(n + wave * 0.045);
}

fn calc_soft_shadow(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    var res = 1.0;
    var t = 0.14;
    for (var i = 0; i < 24; i = i + 1) {
        let h = eval_full_scene_dist(ro + rd * t);
        if (h < 0.008) {
            return 0.08;
        }
        res = min(res, 7.5 * h / t);
        t += clamp(h, 0.08, 0.75);
        if (t > 18.0) {
            break;
        }
    }
    return clamp(res, 0.08, 1.0);
}

fn calc_ao(pos: vec3<f32>, nor: vec3<f32>) -> f32 {
    var occ = 0.0;
    var sca = 1.0;
    for (var i = 0; i < 5; i = i + 1) {
        let h = 0.06 + 0.18 * f32(i);
        let d = eval_full_scene_dist(pos + h * nor);
        occ += (h - d) * sca;
        sca *= 0.72;
    }
    return clamp(1.0 - 1.55 * occ, 0.15, 1.0);
}

fn eval_dream_sky(rd: vec3<f32>, sun_dir: vec3<f32>, time: f32) -> vec3<f32> {
    let up = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    let zenith = vec3<f32>(0.16, 0.34, 0.68);
    let mid_sky = vec3<f32>(0.52, 0.70, 0.92);
    let horizon_peach = vec3<f32>(0.98, 0.74, 0.56);
    let lower_mist = vec3<f32>(0.42, 0.36, 0.54);

    var sky = mix(horizon_peach, mid_sky, smoothstep(0.45, 0.72, up));
    sky = mix(sky, zenith, smoothstep(0.68, 1.0, up));
    sky = mix(lower_mist, sky, smoothstep(0.18, 0.50, up));

    let sun_dot = max(dot(rd, sun_dir), 0.0);
    sky += vec3<f32>(1.15, 0.78, 0.42) * pow(sun_dot, 7.0) * 0.48;
    sky += vec3<f32>(1.4, 1.15, 0.85) * pow(sun_dot, 96.0) * 0.85;

    let cloud_uv = rd.xz / (abs(rd.y) + 0.28) + vec2<f32>(time * 0.015, -time * 0.008);
    let c1 = sin(cloud_uv.x * 1.4 + sin(cloud_uv.y * 1.9)) * 0.5 + 0.5;
    let c2 = sin(cloud_uv.y * 2.7 - cos(cloud_uv.x * 2.2)) * 0.5 + 0.5;
    let cloud_mask = smoothstep(0.54, 0.88, c1 * 0.65 + c2 * 0.35) * smoothstep(0.02, 0.35, abs(rd.y));
    sky = mix(sky, vec3<f32>(1.05, 0.94, 0.86), cloud_mask * 0.36);

    return sky;
}

fn ray_segment_glow(ro: vec3<f32>, rd: vec3<f32>, a: vec3<f32>, b: vec3<f32>, max_t: f32) -> vec2<f32> {
    let ab = b - a;
    let ab_len2 = dot(ab, ab);
    if (ab_len2 < 1e-4) {
        return vec2<f32>(1e5, 0.0);
    }
    let ao = ro - a;
    let b_dot = dot(rd, ab);
    let d_dot = dot(rd, ao);
    let e_dot = dot(ab, ao);
    let det = ab_len2 - b_dot * b_dot;
    if (abs(det) < 1e-5) {
        return vec2<f32>(1e5, 0.0);
    }
    let s = clamp((e_dot - b_dot * d_dot) / det, 0.0, 1.0);
    let p_seg = a + ab * s;
    let t_ray = dot(p_seg - ro, rd);
    if (t_ray < 0.1 || t_ray > max_t + 0.15) {
        return vec2<f32>(1e5, s);
    }
    return vec2<f32>(length((ro + rd * t_ray) - p_seg), s);
}

// Analytic 3D Imp Ring, Drop-Shadow, Symmetry Plane, 3D Transform Gizmo (Yaw Ring + Scale Orb) & Joint Connectors!
fn eval_editor_overlays(ro: vec3<f32>, rd: vec3<f32>, max_t: f32) -> vec4<f32> {
    let mode = u_frame.imp_norm_mode.w;
    if (mode > 2.5) {
        // Play Mode X-Ray Silhouette Ring when solid trees/tunnels pass between camera and player!
        let pup_center = u_frame.puppet_pos_yaw.xyz + vec3<f32>(0.0, 0.22, 0.0);
        let to_pup = pup_center - ro;
        let t_pup = dot(to_pup, rd);
        if (t_pup > 0.5 && max_t < t_pup - 0.65) {
            let perp = length((ro + rd * t_pup) - pup_center);
            let sil_ring = 1.0 - smoothstep(0.03, 0.09, abs(perp - 0.38));
            let sil_fill = (1.0 - smoothstep(0.0, 0.38, perp)) * 0.26;
            let sil = max(sil_ring * 0.85, sil_fill);
            return vec4<f32>(vec3<f32>(1.0, 0.68, 0.28) * sil, sil * 0.65);
        }
        return vec4<f32>(0.0);
    }

    var glow = vec3<f32>(0.0);
    var alpha = 0.0;

    let imp_p = u_frame.imp_pos_size.xyz;
    let brush_r = u_frame.imp_pos_size.w;
    let imp_n = normalize(u_frame.imp_norm_mode.xyz);
    let brush_col = mix(u_frame.brush_color_op.xyz, vec3<f32>(1.0), 0.42);
    let is_carve = u_frame.brush_color_op.w > 0.5 && u_frame.brush_color_op.w < 1.5;
    let ring_col = select(brush_col, vec3<f32>(1.1, 0.28, 0.36), is_carve);

    // 1. Brush ring in Sculpt / Paint modes, or subtle pointer dot in Animate mode
    if (mode < 1.5) {
        let denom = dot(rd, imp_n);
        if (abs(denom) > 1e-3) {
            let t_plane = dot(imp_p - ro, imp_n) / denom;
            if (t_plane > 0.1 && t_plane < max_t + 0.18) {
                let hit = ro + rd * t_plane;
                let r_dist = length(hit - imp_p);
                let ring_band = 1.0 - smoothstep(0.012, 0.048, abs(r_dist - brush_r));
                let center_dot = 1.0 - smoothstep(0.018, 0.055, r_dist);
                let ring_a = max(ring_band * 0.85, center_dot * 0.95);
                glow += ring_col * ring_a * 1.45;
                alpha = max(alpha, ring_a * 0.75);
            }
        }
    }

    // 2. Vertical drop-shadow guide line
    let seg_res = ray_segment_glow(ro, rd, u_frame.imp_shadow_hit.xyz, imp_p, max_t);
    if (seg_res.x < 0.05) {
        let dash = step(0.45, fract(seg_res.y * 8.0 - u_frame.cam_pos_time.w * 2.0));
        let line_glow = (1.0 - smoothstep(0.008, 0.042, seg_res.x)) * dash * 0.60;
        glow += vec3<f32>(1.0, 0.92, 0.72) * line_glow;
        alpha = max(alpha, line_glow * 0.55);
    }

    // 3. Mirror X=0 plane shimmer (World X=0, or Scoped Prop's Local X=0!)
    if (u_frame.viewport_meta.w > 0.5 && mode < 1.5) {
        var plane_o = vec3<f32>(0.0);
        var plane_n = vec3<f32>(1.0, 0.0, 0.0);
        let scoped_pid = u32(u_frame.vol_max_scoped.w);
        if (scoped_pid > 0u) {
            let prop_count = u32(u_frame.viewport_meta.z);
            for (var i = 0u; i < prop_count; i = i + 1u) {
                if (props[i].info.y == scoped_pid) {
                    plane_o = props[i].pos_radius.xyz;
                    plane_n = normalize(quat_conj_mul(props[i].inv_rot, vec3<f32>(1.0, 0.0, 0.0)));
                    break;
                }
            }
        }
        let m_denom = dot(rd, plane_n);
        if (abs(m_denom) > 1e-3) {
            let t_mir = dot(plane_o - ro, plane_n) / m_denom;
            if (t_mir > 0.1 && t_mir < max_t + 0.1) {
                let mp = ro + rd * t_mir;
                let imp_proj = imp_p - plane_n * dot(imp_p - plane_o, plane_n);
                let dist_to_imp = length(mp - imp_proj);
                let sym_ring = (1.0 - smoothstep(0.02, 0.075, abs(dist_to_imp - brush_r))) * 0.60;
                let sym_grid = exp(-dist_to_imp * 0.45) * 0.11;
                let g = sym_ring + sym_grid;
                glow += vec3<f32>(0.42, 0.85, 1.25) * g;
                alpha = max(alpha, g * 0.52);
            }
        }
    }

    // 4. Animate Mode: 3D Rotation Ring + Golden Scale Orb on Selected Prop + Joint Connector Springs!
    if (mode > 1.5 && mode < 2.5) {
        let sel_id = u32(u_frame.puppet_extra.w);
        let prop_count = u32(u_frame.viewport_meta.z);
        for (var i = 0u; i < prop_count; i = i + 1u) {
            let pr = props[i];
            if (pr.info.y == sel_id) {
                let ring_n = normalize(quat_conj_mul(pr.inv_rot, vec3<f32>(0.0, 1.0, 0.0)));
                let ring_r = pr.scale_params.z;
                let r_denom = dot(rd, ring_n);
                if (abs(r_denom) > 1e-3) {
                    let t_r = dot(pr.pos_radius.xyz - ro, ring_n) / r_denom;
                    if (t_r > 0.1 && t_r < max_t + 0.25) {
                        let rhit = ro + rd * t_r;
                        let rerr = abs(length(rhit - pr.pos_radius.xyz) - ring_r);
                        let band = 1.0 - smoothstep(0.014, 0.052, rerr);
                        glow += vec3<f32>(0.30, 0.95, 0.92) * band * 1.35;
                        alpha = max(alpha, band * 0.78);
                    }
                }

                // Golden Scale Orb Handle on the ring's +X point!
                let orb_c = pr.gizmo_orb.xyz;
                let to_orb = orb_c - ro;
                let t_orb = dot(to_orb, rd);
                if (t_orb > 0.1 && t_orb < max_t + 0.35) {
                    let closest = length((ro + rd * t_orb) - orb_c);
                    let orb_r = clamp(ring_r * 0.095, 0.08, 0.20);
                    let orb_mask = 1.0 - smoothstep(orb_r * 0.65, orb_r * 1.25, closest);
                    glow += vec3<f32>(1.25, 0.85, 0.22) * orb_mask * 1.6;
                    alpha = max(alpha, orb_mask * 0.90);
                }
                break;
            }
        }

        // Render Multi-Part Puppet Joint Connectors!
        let conn_count = u32(u_frame.puppet_extra.z);
        for (var ci = 0u; ci < conn_count; ci = ci + 1u) {
            let cv = connectors[ci];
            let a = cv.start_kind.xyz;
            let b = cv.end_sel.xyz;
            let cs = ray_segment_glow(ro, rd, a, b, max_t + 0.5);
            if (cs.x < 0.075) {
                let coil = 0.55 + 0.45 * sin(cs.y * 28.0 - u_frame.cam_pos_time.w * 4.0);
                let c_glow = (1.0 - smoothstep(0.012, 0.055, cs.x)) * coil;
                let col = select(vec3<f32>(0.95, 0.58, 0.92), vec3<f32>(1.15, 0.88, 0.32), cv.end_sel.w > 0.5);
                glow += col * c_glow * 1.3;
                alpha = max(alpha, c_glow * 0.75);
            }
            // Pivot joint bead at `a`
            let t_piv = dot(a - ro, rd);
            if (t_piv > 0.1 && t_piv < max_t + 0.4) {
                let pd = length((ro + rd * t_piv) - a);
                let bead = 1.0 - smoothstep(0.04, 0.09, pd);
                glow += vec3<f32>(1.1, 0.72, 0.98) * bead * 1.2;
                alpha = max(alpha, bead * 0.8);
            }
        }
    }

    return vec4<f32>(glow, alpha);
}

struct FragOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_raymarch(vin: VsOut) -> FragOut {
    let ndc = vec2<f32>(vin.uv.x * 2.0 - 1.0, 1.0 - vin.uv.y * 2.0);
    let near_h = u_frame.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_h = u_frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let near_p = near_h.xyz / near_h.w;
    let far_p = far_h.xyz / far_h.w;

    let ro = u_frame.cam_pos_time.xyz;
    let rd = normalize(far_p - near_p);
    let sun_dir = normalize(u_frame.sun_dof.xyz);
    let time = u_frame.cam_pos_time.w;

    var t = 0.12;
    var hit = false;
    let max_dist = 62.0;

    for (var step_i = 0; step_i < 112; step_i = step_i + 1) {
        let p = ro + rd * t;
        let d = eval_full_scene_dist(p);
        if (d < 0.011) {
            hit = true;
            break;
        }
        t += max(d * 0.90, 0.016);
        if (t > max_dist) {
            break;
        }
    }

    var final_rgb = eval_dream_sky(rd, sun_dir, time);
    var out_depth = 0.99999;
    var linear_t = max_dist;

    if (hit) {
        let hit_p = ro + rd * t;
        linear_t = t;
        let clip = u_frame.view_proj * vec4<f32>(hit_p, 1.0);
        out_depth = clamp(clip.z / clip.w, 0.0001, 0.9999);

        let mat = eval_full_material(hit_p);
        let raw_n = calc_normal(hit_p);
        let n = impasto_normal_perturb(hit_p, raw_n);

        let shadow = calc_soft_shadow(hit_p + raw_n * 0.05, sun_dir);
        let ao = calc_ao(hit_p, raw_n);

        let ndotl = dot(n, sun_dir);
        let wrap_diff = clamp((ndotl + 0.35) / 1.35, 0.0, 1.0);
        let sun_light = vec3<f32>(1.38, 1.16, 0.92) * wrap_diff * shadow;

        let sky_hemi = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
        let bounce_hemi = clamp(-n.y * 0.5 + 0.5, 0.0, 1.0);
        let ambient = (vec3<f32>(0.24, 0.36, 0.58) * sky_hemi + vec3<f32>(0.34, 0.20, 0.14) * bounce_hemi) * ao;

        let view_dir = -rd;
        let half_v = normalize(sun_dir + view_dir);
        let spec = pow(max(dot(n, half_v), 0.0), 22.0) * 0.18 * shadow;
        let rim = pow(1.0 - max(dot(raw_n, view_dir), 0.0), 3.5) * 0.22 * ao;

        var lit = mat.color * (sun_light + ambient) + vec3<f32>(1.0, 0.92, 0.80) * (spec + rim);

        if (abs(mat.style - 4.0) < 0.4) {
            lit += mat.color * 2.2;
        }
        if (mat.selected > 0.5) {
            let sel_rim = pow(1.0 - max(dot(raw_n, view_dir), 0.0), 2.2);
            lit += vec3<f32>(0.45, 0.85, 1.35) * sel_rim * 0.85;
        }

        // Dreams-style Scope-In visual focus: gently cool/dim out-of-scope background clay so your active Toy pops!
        if (mat.in_active_scope < 0.5) {
            let luma = dot(lit, vec3<f32>(0.299, 0.587, 0.114));
            let muted = mix(lit, vec3<f32>(luma) * vec3<f32>(0.75, 0.82, 0.98), 0.45) * 0.68;
            lit = muted;
        }

        if (u_frame.imp_norm_mode.w < 2.5) {
            let sh_dist = length(hit_p - u_frame.imp_shadow_hit.xyz);
            let sh_ring = 1.0 - smoothstep(0.04, 0.14, abs(sh_dist - 0.22));
            lit = mix(lit, lit * 0.55 + vec3<f32>(0.35, 0.30, 0.22), sh_ring * 0.55);
        }

        let fog = 1.0 - exp(-max(t - 10.0, 0.0) * 0.022);
        final_rgb = mix(lit, eval_dream_sky(rd, sun_dir, time), clamp(fog, 0.0, 0.72));
    }

    let imp_viz = eval_editor_overlays(ro, rd, linear_t);
    final_rgb = final_rgb * (1.0 - imp_viz.a * 0.35) + imp_viz.rgb;

    var fout: FragOut;
    fout.color = vec4<f32>(max(final_rgb, vec3<f32>(0.0)), linear_t);
    fout.depth = out_depth;
    return fout;
}
