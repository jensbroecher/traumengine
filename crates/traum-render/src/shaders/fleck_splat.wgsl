// Render Pass 2: Dreams-Style Instanced Painterly Fleck Splatting + "Living Flecks" Motion!
// Uses stochastic stipple discard + hardware depth test (zero CPU sorting required!)

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos_time: vec4<f32>,
    cam_fwd_focus: vec4<f32>,
    sun_dof: vec4<f32>,
    viewport_meta: vec4<f32>,
    imp_pos_size: vec4<f32>,
    imp_norm_mode: vec4<f32>,
    imp_shadow_hit: vec4<f32>,
    brush_color_op: vec4<f32>,
    puppet_pos_yaw: vec4<f32>,
    puppet_extra: vec4<f32>,
    vol_min_voxel: vec4<f32>,
    vol_max_pad: vec4<f32>,
};

struct GpuFleck {
    pos_size: vec4<f32>,
    normal_loose: vec4<f32>,
    color_style_seed: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_frame: FrameUniforms;
@group(0) @binding(1) var sdf_tex: texture_3d<f32>;
@group(0) @binding(2) var vol_sampler: sampler;
@group(0) @binding(3) var<storage, read> flecks: array<GpuFleck>;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_uv: vec2<f32>,
    @location(1) lit_color: vec3<f32>,
    @location(2) style_seed_loose: vec3<f32>,
    @location(3) linear_dist: f32,
};

fn sample_sdf(p: vec3<f32>) -> f32 {
    let vmin = u_frame.vol_min_voxel.xyz;
    let vmax = u_frame.vol_max_pad.xyz;
    let uvw = clamp((p - vmin) / (vmax - vmin), vec3<f32>(0.002), vec3<f32>(0.998));
    return textureSampleLevel(sdf_tex, vol_sampler, uvw, 0.0).x;
}

@vertex
fn vs_fleck(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VsOut {
    let f = flecks[iid];
    var center = f.pos_size.xyz;
    var radius = f.pos_size.w;
    let n = normalize(f.normal_loose.xyz);
    let loose = f.normal_loose.w;
    let packed = f.color_style_seed.w;
    let style = floor(packed);
    let seed = fract(packed);
    let time = u_frame.cam_pos_time.w;

    // "Living Flecks" built-in world micro-motion!
    var rot_angle = seed * 6.2831853;
    if (abs(style - 1.0) < 0.3) {
        // FoliageBreeze: Gentle warm wind rustle!
        let wind = sin(time * 2.3 + center.x * 0.85 + center.z * 0.65 + seed * 4.0);
        center += vec3<f32>(wind * 0.06, abs(wind) * 0.025, cos(time * 1.9 + center.z) * 0.04) * (0.35 + loose);
        rot_angle += wind * 0.18;
    } else if (abs(style - 2.0) < 0.3) {
        // SketchBoil: 12fps claymation / Spider-Verse hand-drawn boil!
        let boil_frame = floor(time * 12.0);
        let j1 = fract(sin(boil_frame * 12.9898 + seed * 78.233) * 43758.5453) - 0.5;
        let j2 = fract(sin(boil_frame * 39.346 + seed * 19.19) * 24634.6345) - 0.5;
        center += vec3<f32>(j1, j2, -j1) * 0.035;
        rot_angle += j1 * 0.45;
    } else if (abs(style - 3.0) < 0.3) {
        // FluffyCloud: Slow dreamy drift & breathing puff!
        let drift = sin(time * 0.9 + seed * 6.28 + center.x * 0.4);
        center += vec3<f32>(drift * 0.08, cos(time * 0.7 + seed * 5.0) * 0.05, 0.0) * loose;
        radius *= 1.0 + 0.08 * sin(time * 1.3 + seed * 9.0);
    }

    // Build tangent basis on surface, blended toward camera billboard at high looseness!
    let cam_pos = u_frame.cam_pos_time.xyz;
    let to_cam = normalize(cam_pos - center);
    let orient_n = normalize(mix(n, to_cam, clamp(loose * 0.68, 0.12, 0.75)));
    let ref_up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(orient_n.y) > 0.92);
    let t1 = normalize(cross(ref_up, orient_n));
    let t2 = cross(orient_n, t1);

    let ca = cos(rot_angle);
    let sa = sin(rot_angle);
    let u_axis = t1 * ca + t2 * sa;
    let v_axis = -t1 * sa + t2 * ca;

    // 2-triangle quad corners
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    let corner = corners[vid];
    // Painterly elongated brushstroke aspect ratio (1.35 : 0.78)
    let aspect = select(vec2<f32>(1.28, 0.82), vec2<f32>(1.05, 1.05), abs(style - 3.0) < 0.3);
    let world_p = center + (u_axis * (corner.x * aspect.x) + v_axis * (corner.y * aspect.y)) * radius;

    // Fast 2-tap SDF shadow + AO right in the vertex shader so flecks match clay lighting!
    let sun_dir = normalize(u_frame.sun_dof.xyz);
    let sh1 = clamp(sample_sdf(center + sun_dir * 0.45) * 2.2, 0.15, 1.0);
    let sh2 = clamp(sample_sdf(center + sun_dir * 1.35) * 0.9, 0.15, 1.0);
    let shadow = min(sh1, sh2);
    let ao = clamp(0.45 + sample_sdf(center + n * 0.35) * 1.8, 0.25, 1.05);

    let ndotl = clamp((dot(n, sun_dir) + 0.45) / 1.45, 0.0, 1.0);
    let sun_light = vec3<f32>(1.36, 1.15, 0.90) * ndotl * shadow;
    let sky_amb = mix(vec3<f32>(0.30, 0.19, 0.14), vec3<f32>(0.26, 0.38, 0.60), clamp(n.y * 0.5 + 0.5, 0.0, 1.0)) * ao;

    var lit = f.color_style_seed.xyz * (sun_light + sky_amb);
    if (abs(style - 4.0) < 0.3) {
        // GlowEmber self-illumination!
        lit = f.color_style_seed.xyz * 2.6;
    }

    let dist_cam = length(world_p - cam_pos);
    // Constant 1.8cm world-space offset toward camera prevents z-fighting on clay WITHOUT pulling flecks meters forward when zoomed out!
    let biased_world = world_p + to_cam * 0.018;
    var vout: VsOut;
    vout.clip_pos = u_frame.view_proj * vec4<f32>(biased_world, 1.0);
    vout.local_uv = corner;
    vout.lit_color = lit;
    vout.style_seed_loose = vec3<f32>(style, seed, loose);
    vout.linear_dist = dist_cam;
    return vout;
}

@fragment
fn fs_fleck(vin: VsOut) -> @location(0) vec4<f32> {
    let uv = vin.local_uv;
    let style = vin.style_seed_loose.x;
    let seed = vin.style_seed_loose.y;

    // Organic painterly brushstroke silhouettes per FleckStyle!
    var mask = 0.0;
    var bristle_shade = 1.0;

    if (abs(style - 1.0) < 0.3) {
        // Tapered pointed leaf stroke
        let pinch = 1.0 - (uv.x * 0.5 + 0.5) * 0.55;
        let d = length(vec2<f32>(uv.x, uv.y / max(pinch, 0.2)));
        let vein = sin(uv.y * 14.0 + seed * 6.0) * 0.04;
        mask = 1.0 - smoothstep(0.62, 0.98, d + vein);
        bristle_shade = 0.92 + 0.16 * smoothstep(-0.4, 0.6, uv.x);
    } else if (abs(style - 3.0) < 0.3) {
        // Multi-lobed fluffy cloud puff
        let d1 = length(uv - vec2<f32>(-0.22, 0.05));
        let d2 = length(uv - vec2<f32>(0.24, -0.04));
        let d3 = length(uv - vec2<f32>(0.0, 0.20));
        let d = min(d1, min(d2, d3));
        mask = 1.0 - smoothstep(0.45, 0.82, d);
        bristle_shade = 0.96 + 0.12 * (1.0 - d);
    } else {
        // Painterly oil/impasto bristle dab!
        let grooves = sin(uv.y * 19.0 + seed * 31.0) * 0.08 + sin(uv.y * 37.0) * 0.04;
        let d = length(vec2<f32>(uv.x * 0.92, uv.y * 1.06)) + grooves * smoothstep(0.1, 0.9, abs(uv.x));
        mask = 1.0 - smoothstep(0.64, 0.98, d);
        // Subtle impasto ridge lighting across bristles!
        bristle_shade = 0.90 + 0.20 * (0.5 + 0.5 * sin(uv.y * 19.0 + seed * 31.0));
    }

    // Dreams-style stochastic screen stipple discard (crisp depth-tested painterly overlap, zero alpha sorting!)
    let dither = fract(52.9829189 * fract(dot(vin.clip_pos.xy + vec2<f32>(seed * 17.0), vec2<f32>(0.06711056, 0.00583715))));
    if (mask < 0.18 + dither * 0.72) {
        discard;
    }

    return vec4<f32>(vin.lit_color * bristle_shade, vin.linear_dist);
}
