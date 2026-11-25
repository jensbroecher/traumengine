// Render Pass 3B: Tactile Toy UI Renderer (Rounded Pills, 8-Slice Radial Pie Wheel, & Crisp Font Quads)

struct UiUniforms {
    // xy = viewport_px, z = manual_srgb, w = radial_open (1.0 or 0.0)
    viewport_srgb_radial: vec4<f32>,
    // xy = radial_center_px, z = radial_outer_radius, w = hovered_slice (-1.0 if none)
    radial_params: vec4<f32>,
};

struct GpuUiQuad {
    // xy = min_px, zw = max_px
    rect_px: vec4<f32>,
    // xy = uv_min, zw = uv_max (if uv_max.x < 0 -> vector SDF rounded rect; else font glyph quad!)
    uv_rect: vec4<f32>,
    fill_color: vec4<f32>,
    border_color: vec4<f32>,
    // x = corner_radius, y = border_width, z = kind (0=pill, 1=glyph, 2=radial_canvas), w = pad
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_ui: UiUniforms;
@group(0) @binding(1) var<storage, read> quads: array<GpuUiQuad>;
@group(0) @binding(2) var font_tex: texture_2d<f32>;
@group(0) @binding(3) var ui_sampler: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) screen_px: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) quad_idx: u32,
};

@vertex
fn vs_ui(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VsOut {
    let q = quads[iid];
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );
    let c = corners[vid];
    let px = mix(q.rect_px.xy, q.rect_px.zw, c);
    let vp = max(u_ui.viewport_srgb_radial.xy, vec2<f32>(1.0));
    let ndc = vec2<f32>((px.x / vp.x) * 2.0 - 1.0, 1.0 - (px.y / vp.y) * 2.0);

    var vout: VsOut;
    vout.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    vout.screen_px = px;
    vout.uv = mix(q.uv_rect.xy, q.uv_rect.zw, c);
    vout.quad_idx = iid;
    return vout;
}

fn srgb_to_lin(c: vec4<f32>) -> vec4<f32> {
    if (u_ui.viewport_srgb_radial.z > 0.5) {
        return c; // Swapchain expects sRGB values directly
    }
    return vec4<f32>(pow(c.rgb, vec3<f32>(2.2)), c.a);
}

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec2<f32>(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
}

@fragment
fn fs_ui(vin: VsOut) -> @location(0) vec4<f32> {
    let q = quads[vin.quad_idx];
    let kind = q.params.z;

    if (abs(kind - 1.0) < 0.3) {
        // Crisp pixel-art / smooth embedded bitmap font glyph!
        let cov = textureSampleLevel(font_tex, ui_sampler, vin.uv, 0.0).r;
        if (cov < 0.05) {
            discard;
        }
        var col = srgb_to_lin(q.fill_color);
        col.a *= cov;
        return col;
    }

    if (abs(kind - 2.0) < 0.3) {
        // Bouncy 8-Slice Radial Pie Menu!
        let center = u_ui.radial_params.xy;
        let outer_r = u_ui.radial_params.z;
        let inner_r = 34.0;
        let rel = vin.screen_px - center;
        let dist = length(rel);

        let angle = atan2(rel.y, rel.x);
        let norm_ang = (angle + 3.14159265 * 2.5) % 6.2831853;
        let slice_f = (norm_ang / 6.2831853) * 8.0;
        let slice_idx = floor(slice_f + 0.5) % 8.0;
        let hovered = abs(slice_idx - u_ui.radial_params.w) < 0.25 && u_ui.radial_params.w >= 0.0;

        let slice_max_r = select(outer_r, outer_r + 10.0, hovered);
        let ring_d = max(inner_r - dist, dist - slice_max_r);

        // Angular gap between pie slices
        let ang_frac = abs(fract(slice_f + 0.5) - 0.5);
        let gap_px = ang_frac * (6.2831853 / 8.0) * dist;
        let wedge_d = max(ring_d, 2.2 - gap_px);

        let alpha_mask = 1.0 - smoothstep(-0.8, 1.2, wedge_d);
        if (alpha_mask < 0.02) {
            discard;
        }

        let fill_srgb = select(
            vec4<f32>(0.13, 0.15, 0.22, 0.90),
            vec4<f32>(0.96, 0.66, 0.24, 0.97),
            hovered
        );
        let rim = 1.0 - smoothstep(0.0, 2.2, abs(wedge_d + 1.1));
        var col = srgb_to_lin(mix(fill_srgb, vec4<f32>(1.0, 0.96, 0.86, 0.98), rim * 0.7));
        col.a *= alpha_mask;
        return col;
    }

    // Standard Rounded Toy Pill / Swatch / Slider
    let center = (q.rect_px.xy + q.rect_px.zw) * 0.5;
    let half_ext = (q.rect_px.zw - q.rect_px.xy) * 0.5;
    let r = min(q.params.x, min(half_ext.x, half_ext.y));
    let d = sd_rounded_box(vin.screen_px - center, half_ext, r);

    let aa = 1.0 - smoothstep(-0.7, 0.9, d);
    if (aa < 0.01) {
        discard;
    }

    let bw = q.params.y;
    let border_t = smoothstep(-bw - 0.8, -bw + 0.6, d) * q.border_color.a;
    let mixed_srgb = mix(q.fill_color, vec4<f32>(q.border_color.rgb, 1.0), border_t);
    var out_col = srgb_to_lin(mixed_srgb);
    out_col.a *= aa;
    return out_col;
}
