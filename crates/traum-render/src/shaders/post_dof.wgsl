// Render Pass 3A: Dreams Tilt-Shift Bokeh Depth-of-Field + Warm ACES Filmic Tone Mapping

struct PostUniforms {
    // x = focus_dist, y = dof_amount, z = manual_srgb (1.0 if non-srgb swapchain), w = time
    focus_dof_srgb: vec4<f32>,
    // xy = inv_viewport, zw = viewport
    viewport_info: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_post: PostUniforms;
@group(0) @binding(1) var hdr_scene: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_post(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var vout: VsOut;
    vout.uv = vec2<f32>(x, y);
    vout.clip_pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return vout;
}

fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_post(vin: VsOut) -> @location(0) vec4<f32> {
    let center_sample = textureSampleLevel(hdr_scene, linear_sampler, vin.uv, 0.0);
    let center_dist = center_sample.w;
    let focus_d = u_post.focus_dof_srgb.x;
    let dof_amt = u_post.focus_dof_srgb.y;

    // Miniature diorama Circle-of-Confusion radius in pixels
    let rel_blur = clamp(abs(center_dist - focus_d) / max(focus_d * 0.85, 2.0) - 0.18, 0.0, 1.0);
    let max_radius_px = rel_blur * dof_amt * 9.5;

    var accum_rgb = center_sample.rgb;
    var accum_w = 1.0;

    if (max_radius_px > 0.45) {
        let px = u_post.viewport_info.xy;
        // 12-tap Golden-Angle Vogel Disk Bokeh kernel!
        for (var i = 1; i <= 12; i = i + 1) {
            let fi = f32(i);
            let r = sqrt(fi / 12.5) * max_radius_px;
            let theta = fi * 2.3999632;
            let offset = vec2<f32>(cos(theta), sin(theta)) * r * px;
            let tap = textureSampleLevel(hdr_scene, linear_sampler, clamp(vin.uv + offset, vec2<f32>(0.001), vec2<f32>(0.999)), 0.0);
            let tap_blur = clamp(abs(tap.w - focus_d) / max(focus_d * 0.85, 2.0) - 0.12, 0.05, 1.0);
            // Highlight bokeh boost so glowing embers form soft circular discs!
            let luma = dot(tap.rgb, vec3<f32>(0.299, 0.587, 0.114));
            let w = tap_blur * (1.0 + max(luma - 0.85, 0.0) * 1.4);
            accum_rgb += tap.rgb * w;
            accum_w += w;
        }
    }

    var hdr = accum_rgb / accum_w;

    // Subtle warm storybook vignette
    let centered = vin.uv - vec2<f32>(0.5);
    let vig = 1.0 - dot(centered, centered) * 0.32;
    hdr *= vig;

    var mapped = aces_filmic(hdr * 1.04);

    if (u_post.focus_dof_srgb.z > 0.5) {
        mapped = pow(mapped, vec3<f32>(1.0 / 2.2));
    }

    return vec4<f32>(mapped, 1.0);
}
