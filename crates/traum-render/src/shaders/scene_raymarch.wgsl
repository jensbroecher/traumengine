// Render Pass 1: Watertight Painterly Clay Core + Animated Toy Props + Playable Buddy + 3D Imp Guides
// Writes HDR Linear Scene Color + Hardware Depth (for Fleck Splatting & Bokeh DOF!)

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // xyz = camera eye, w = time
    cam_pos_time: vec4<f32>,
    // xyz = camera forward, w = focus_distance
    cam_fwd_focus: vec4<f32>,
    // xyz = sun_dir (unit), w = dof_amount
    sun_dof: vec4<f32>,
    // xy = viewport_px, z = prop_count_f32, w = mirror_x_flag
    viewport_meta: vec4<f32>,
    // xyz = imp_world_pos, w = brush_size
    imp_pos_size: vec4<f32>,
    // xyz = imp_normal, w = mode (0=Sculpt, 1=Paint, 2=Animate, 3=Play)
    imp_norm_mode: vec4<f32>,
    // xyz = imp_ground_shadow, w = imp_on_surface (1.0 or 0.0)
    imp_shadow_hit: vec4<f32>,
    // xyz = brush_linear_color, w = op (0=Add, 1=Subtract, 2=Paint)
    brush_color_op: vec4<f32>,
    // xyz = puppet_pos, w = puppet_facing_yaw
    puppet_pos_yaw: vec4<f32>,
    // x = puppet_squash, y = puppet_walk_cycle, z = output_manual_srgb, w = selected_prop_id
    puppet_extra: vec4<f32>,
    // xyz = volume_min, w = voxel_size
    vol_min_voxel: vec4<f32>,
    // xyz = volume_max, w = pad
    vol_max_pad: vec4<f32>,
};

struct GpuSceneProp {
    pos_radius: vec4<f32>,
    inv_rot: vec4<f32>,
    scale_loose: vec4<f32>,
    color_style: vec4<f32>,
    info: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u_frame: FrameUniforms;
@group(0) @binding(1) var sdf_tex: texture_3d<f32>;
@group(0) @binding(2) var color_tex: texture_3d<f32>;
@group(0) @binding(3) var vol_sampler: sampler;
@group(0) @binding(4) var<storage, read> props: array<GpuSceneProp>;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    // Oversized fullscreen triangle
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

fn eval_primitive(local_p: vec3<f32>, scale: vec3<f32>, shape: u32) -> f32 {
    switch shape {
        case 0u: {
            let k0 = length(local_p / scale);
            if (k0 < 1e-5) { return -min(scale.x, min(scale.y, scale.z)); }
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

fn world_to_uvw(p: vec3<f32>) -> vec3<f32> {
    let vmin = u_frame.vol_min_voxel.xyz;
    let vmax = u_frame.vol_max_pad.xyz;
    return (p - vmin) / (vmax - vmin);
}

fn sample_world_brick_sdf(p: vec3<f32>) -> vec4<f32> {
    let vmin = u_frame.vol_min_voxel.xyz;
    let vmax = u_frame.vol_max_pad.xyz;
    // Smooth box distance outside volume so rays march cleanly into the volume
    let box_d = length(max(max(vmin - p, p - vmax), vec3<f32>(0.0)));
    if (box_d > 0.15) {
        return vec4<f32>(box_d + 0.25, 0.0, 0.0, 0.0);
    }
    let uvw = clamp((p - vmin) / (vmax - vmin), vec3<f32>(0.002), vec3<f32>(0.998));
    return textureSampleLevel(sdf_tex, vol_sampler, uvw, 0.0);
}

struct HitMaterial {
    d: f32,
    color: vec3<f32>,
    style: f32,
    selected: f32,
};

// Evaluates Playable Explorer Buddy (bouncy Dreams-like character in Play Mode!)
fn eval_puppet(p: vec3<f32>) -> HitMaterial {
    var m = HitMaterial(1e5, vec3<f32>(0.0), 0.0, 0.0);
    if (u_frame.imp_norm_mode.w < 2.5) {
        return m;
    }

    let base = u_frame.puppet_pos_yaw.xyz;
    let dist_bound = length(p - base);
    if (dist_bound > 1.8) {
        m.d = dist_bound - 1.3;
        return m;
    }

    let yaw = u_frame.puppet_pos_yaw.w;
    let cy = cos(-yaw);
    let sy = sin(-yaw);
    let rel = p - base;
    var lp = vec3<f32>(rel.x * cy - rel.z * sy, rel.y, rel.x * sy + rel.z * cy);

    // Squash & stretch!
    let sq = u_frame.puppet_extra.x;
    let inv_xz = sqrt(sq);
    let body_p = vec3<f32>(lp.x * inv_xz, lp.y / sq, lp.z * inv_xz);

    // Chubby coral-gold body capsule
    var d_body = eval_primitive(body_p - vec3<f32>(0.0, 0.08, 0.0), vec3<f32>(0.34, 0.44, 0.34), 2u);
    var col = vec3<f32>(0.94, 0.38, 0.26);
    var style = 0.0;

    // Little bobbing antenna pom-pom (glowing!)
    let bob = sin(u_frame.cam_pos_time.w * 7.0) * 0.05;
    let d_pom = length(body_p - vec3<f32>(bob, 0.68, -0.04)) - 0.11;
    if (d_pom < d_body) {
        d_body = d_pom;
        col = vec3<f32>(1.0, 0.72, 0.18);
        style = 4.0;
    }

    // Expressive eyes facing +Z
    let eye_x = abs(body_p.x) - 0.13;
    let d_eye = length(vec3<f32>(eye_x, body_p.y - 0.20, body_p.z - 0.30)) - 0.065;
    if (d_eye < d_body) {
        d_body = d_eye;
        col = vec3<f32>(0.04, 0.04, 0.06);
        style = 0.0;
    }

    // Bouncy walking feet!
    let cycle = u_frame.puppet_extra.y;
    let foot_l = vec3<f32>(-0.16, -0.34 + max(0.0, sin(cycle)) * 0.14, sin(cycle) * 0.16);
    let foot_r = vec3<f32>(0.16, -0.34 + max(0.0, -sin(cycle)) * 0.14, -sin(cycle) * 0.16);
    let d_feet = min(length(lp - foot_l) - 0.12, length(lp - foot_r) - 0.12);
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
            let lp = quat_mul_vec3(pr.inv_rot, delta);
            let pd = eval_primitive(lp, pr.scale_loose.xyz, pr.info.x);
            d = min(d, pd);
        } else {
            d = min(d, center_d - b_rad);
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
    var best = HitMaterial(ws.x, wcol, ws.z, 0.0);

    let prop_count = u32(u_frame.viewport_meta.z);
    let sel_id = u32(u_frame.puppet_extra.w);
    for (var i = 0u; i < prop_count; i = i + 1u) {
        let pr = props[i];
        let lp = quat_mul_vec3(pr.inv_rot, p - pr.pos_radius.xyz);
        let pd = eval_primitive(lp, pr.scale_loose.xyz, pr.info.x);
        if (pd < best.d) {
            best.d = pd;
            best.color = pr.color_style.xyz;
            best.style = f32(pr.info.y);
            best.selected = select(0.0, 1.0, pr.info.z == sel_id && u_frame.imp_norm_mode.w > 1.5 && u_frame.imp_norm_mode.w < 2.5);
        }
    }

    let pup = eval_puppet(p);
    if (pup.d < best.d) {
        best = pup;
    }
    return best;
}

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.03;
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

// Procedural impasto brush-stroke normal micro-relief so even tight clay looks hand-painted!
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

// Dreamy storybook atmospheric sky with soft sunlit clouds & warm horizon bounce!
fn eval_dream_sky(rd: vec3<f32>, sun_dir: vec3<f32>, time: f32) -> vec3<f32> {
    let up = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    let zenith = vec3<f32>(0.16, 0.34, 0.68);
    let mid_sky = vec3<f32>(0.52, 0.70, 0.92);
    let horizon_peach = vec3<f32>(0.98, 0.74, 0.56);
    let lower_mist = vec3<f32>(0.42, 0.36, 0.54);

    var sky = mix(horizon_peach, mid_sky, smoothstep(0.45, 0.72, up));
    sky = mix(sky, zenith, smoothstep(0.68, 1.0, up));
    sky = mix(lower_mist, sky, smoothstep(0.18, 0.50, up));

    // Warm golden sun halo
    let sun_dot = max(dot(rd, sun_dir), 0.0);
    sky += vec3<f32>(1.15, 0.78, 0.42) * pow(sun_dot, 7.0) * 0.48;
    sky += vec3<f32>(1.4, 1.15, 0.85) * pow(sun_dot, 96.0) * 0.85;

    // Soft distant painted cloud bands
    let cloud_uv = rd.xz / (abs(rd.y) + 0.28) + vec2<f32>(time * 0.015, -time * 0.008);
    let c1 = sin(cloud_uv.x * 1.4 + sin(cloud_uv.y * 1.9)) * 0.5 + 0.5;
    let c2 = sin(cloud_uv.y * 2.7 - cos(cloud_uv.x * 2.2)) * 0.5 + 0.5;
    let cloud_mask = smoothstep(0.54, 0.88, c1 * 0.65 + c2 * 0.35) * smoothstep(0.02, 0.35, abs(rd.y));
    sky = mix(sky, vec3<f32>(1.05, 0.94, 0.86), cloud_mask * 0.36);

    return sky;
}

// Analytic glowing 3D Imp Ring + Ground Drop-Shadow Line (so mid-air depth is effortless!)
fn eval_imp_overlay(ro: vec3<f32>, rd: vec3<f32>, max_t: f32) -> vec4<f32> {
    if (u_frame.imp_norm_mode.w > 2.5) {
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

    // 1. Ray-plane intersection with brush ring tangent plane!
    let denom = dot(rd, imp_n);
    if (abs(denom) > 1e-3) {
        let t_plane = dot(imp_p - ro, imp_n) / denom;
        if (t_plane > 0.1 && t_plane < max_t + 0.18) {
            let hit = ro + rd * t_plane;
            let r_dist = length(hit - imp_p);
            let ring_band = 1.0 - smoothstep(0.015, 0.055, abs(r_dist - brush_r));
            let center_dot = 1.0 - smoothstep(0.02, 0.065, r_dist);
            let ring_a = max(ring_band * 0.85, center_dot * 0.95);
            glow += ring_col * ring_a * 1.45;
            alpha = max(alpha, ring_a * 0.75);
        }
    }

    // 2. Vertical drop-shadow line from Imp down to island ground when sculpting in air!
    let shadow_p = u_frame.imp_shadow_hit.xyz;
    let seg_a = shadow_p;
    let seg_b = imp_p;
    let ab = seg_b - seg_a;
    let ab_len2 = dot(ab, ab);
    if (ab_len2 > 0.08) {
        // Closest point between camera ray and vertical drop segment
        let ao = ro - seg_a;
        let a_dot = dot(rd, rd);
        let b_dot = dot(rd, ab);
        let c_dot = ab_len2;
        let d_dot = dot(rd, ao);
        let e_dot = dot(ab, ao);
        let det = a_dot * c_dot - b_dot * b_dot;
        if (abs(det) > 1e-4) {
            let s_seg = clamp((a_dot * e_dot - b_dot * d_dot) / det, 0.0, 1.0);
            let p_on_seg = seg_a + ab * s_seg;
            let t_ray = dot(p_on_seg - ro, rd);
            if (t_ray > 0.1 && t_ray < max_t) {
                let line_dist = length((ro + rd * t_ray) - p_on_seg);
                let dash = step(0.45, fract(p_on_seg.y * 3.5 - u_frame.cam_pos_time.w * 2.0));
                let line_glow = (1.0 - smoothstep(0.01, 0.045, line_dist)) * dash * 0.65;
                glow += vec3<f32>(1.0, 0.92, 0.72) * line_glow;
                alpha = max(alpha, line_glow * 0.6);
            }
        }
    }

    // 3. Mirror X=0 plane subtle guide shimmer when symmetry is ON!
    if (u_frame.viewport_meta.w > 0.5 && abs(rd.x) > 1e-3) {
        let t_mir = -ro.x / rd.x;
        if (t_mir > 0.1 && t_mir < max_t) {
            let mp = ro + rd * t_mir;
            let dist_to_imp = length(mp.yz - imp_p.yz);
            let sym_ring = (1.0 - smoothstep(0.02, 0.08, abs(dist_to_imp - brush_r))) * 0.55;
            let sym_grid = exp(-dist_to_imp * 0.35) * 0.10;
            let g = sym_ring + sym_grid;
            glow += vec3<f32>(0.45, 0.82, 1.2) * g;
            alpha = max(alpha, g * 0.5);
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

    for (var step_i = 0; step_i < 108; step_i = step_i + 1) {
        let p = ro + rd * t;
        let d = eval_full_scene_dist(p);
        if (d < 0.012) {
            hit = true;
            break;
        }
        t += max(d * 0.92, 0.018);
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

        // Warm clay subsurface wrap diffuse + hemisphere sky/terracotta bounce!
        let ndotl = dot(n, sun_dir);
        let wrap_diff = clamp((ndotl + 0.35) / 1.35, 0.0, 1.0);
        let sun_light = vec3<f32>(1.38, 1.16, 0.92) * wrap_diff * shadow;

        let sky_hemi = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
        let bounce_hemi = clamp(-n.y * 0.5 + 0.5, 0.0, 1.0);
        let ambient = (vec3<f32>(0.24, 0.36, 0.58) * sky_hemi + vec3<f32>(0.34, 0.20, 0.14) * bounce_hemi) * ao;

        // Soft clay rim & velvety specular sheen
        let view_dir = -rd;
        let half_v = normalize(sun_dir + view_dir);
        let spec = pow(max(dot(n, half_v), 0.0), 22.0) * 0.18 * shadow;
        let rim = pow(1.0 - max(dot(raw_n, view_dir), 0.0), 3.5) * 0.22 * ao;

        var lit = mat.color * (sun_light + ambient) + vec3<f32>(1.0, 0.92, 0.80) * (spec + rim);

        // GlowEmber style (4.0) radiates warm lantern light!
        if (abs(mat.style - 4.0) < 0.4) {
            lit += mat.color * 2.2;
        }
        // Selected Toy Prop highlight rim in Animate mode
        if (mat.selected > 0.5) {
            let sel_rim = pow(1.0 - max(dot(raw_n, view_dir), 0.0), 2.2);
            lit += vec3<f32>(0.85, 0.55, 1.35) * sel_rim * 0.9;
        }

        // Ground drop-shadow ring directly on clay surface below the 3D Imp!
        if (u_frame.imp_norm_mode.w < 2.5) {
            let sh_dist = length(hit_p - u_frame.imp_shadow_hit.xyz);
            let sh_ring = 1.0 - smoothstep(0.04, 0.14, abs(sh_dist - 0.22));
            lit = mix(lit, lit * 0.55 + vec3<f32>(0.35, 0.30, 0.22), sh_ring * 0.55);
        }

        // Dreamy aerial perspective haze toward distant sky
        let fog = 1.0 - exp(-max(t - 10.0, 0.0) * 0.022);
        final_rgb = mix(lit, eval_dream_sky(rd, sun_dir, time), clamp(fog, 0.0, 0.72));
    }

    // Composite glowing 3D Imp brush ring & depth guide!
    let imp_viz = eval_imp_overlay(ro, rd, linear_t);
    final_rgb = final_rgb * (1.0 - imp_viz.a * 0.35) + imp_viz.rgb;

    var fout: FragOut;
    // Store linear view distance in alpha channel for Render Pass 3's Tilt-Shift Bokeh DOF!
    fout.color = vec4<f32>(max(final_rgb, vec3<f32>(0.0)), linear_t);
    fout.depth = out_depth;
    return fout;
}
