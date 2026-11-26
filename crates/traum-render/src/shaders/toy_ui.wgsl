// Render Pass 3B: Tactile Toy UI Renderer
// - Kind 0: Rounded Pills, Frosted Panels, Glossy Buttons, & Soft Drop Shadows
// - Kind 1: Smooth Anti-Aliased SDF Typography with Built-In Contrast Halo
// - Kind 2: Bouncy 8-Slice Radial Pie Wheel
// - Kind 3: Crisp Procedural Vector Icons

struct UiUniforms {
    // xy = viewport_px, z = manual_srgb, w = ui_scale
    viewport_srgb_scale: vec4<f32>,
    // xy = radial_center_px, z = radial_outer_radius, w = hovered_slice (-1.0 if none)
    radial_params: vec4<f32>,
};

struct GpuUiQuad {
    // xy = min_px, zw = max_px
    rect_px: vec4<f32>,
    // xy = uv_min, zw = uv_max
    uv_rect: vec4<f32>,
    fill_color: vec4<f32>,
    border_color: vec4<f32>,
    // x = corner_radius (or icon_id for kind=3), y = border_width, z = kind, w = style_variant
    // style_variant for kind=0: 0=panel/flat, 1=glossy button bevel, 2=soft drop shadow
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
    let vp = max(u_ui.viewport_srgb_scale.xy, vec2<f32>(1.0));
    let ndc = vec2<f32>((px.x / vp.x) * 2.0 - 1.0, 1.0 - (px.y / vp.y) * 2.0);

    var vout: VsOut;
    vout.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    vout.screen_px = px;
    vout.uv = mix(q.uv_rect.xy, q.uv_rect.zw, c);
    vout.quad_idx = iid;
    return vout;
}

fn srgb_to_lin(c: vec4<f32>) -> vec4<f32> {
    if (u_ui.viewport_srgb_scale.z > 0.5) {
        return c; // Non-sRGB swapchain expects sRGB values directly
    }
    return vec4<f32>(pow(max(c.rgb, vec3<f32>(0.0)), vec3<f32>(2.2)), c.a);
}

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec2<f32>(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
}

fn sd_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-5), 0.0, 1.0);
    return length(pa - ba * h);
}

// Evaluates crisp procedural vector icons in local `p` in `[-1..1]^2` (returns signed distance, < 0 is inside icon)
fn eval_ui_icon(p: vec2<f32>, icon_id: i32) -> f32 {
    switch icon_id {
        case 0: { // Plus (Add Clay)
            let h = sd_segment(p, vec2<f32>(-0.56, 0.0), vec2<f32>(0.56, 0.0)) - 0.16;
            let v = sd_segment(p, vec2<f32>(0.0, -0.56), vec2<f32>(0.0, 0.56)) - 0.16;
            return min(h, v);
        }
        case 1: { // Minus / Carve Scoop
            let bar = sd_segment(p, vec2<f32>(-0.56, 0.0), vec2<f32>(0.56, 0.0)) - 0.16;
            let ring = abs(length(p) - 0.72) - 0.06;
            return min(bar, ring);
        }
        case 2: { // Paint Brush / Droplet
            let tip_p = vec2<f32>(p.x * 1.15, p.y - 0.10);
            let stretch = clamp(0.5 - tip_p.y * 0.65, 0.15, 1.0);
            return length(vec2<f32>(tip_p.x / stretch, tip_p.y)) - 0.56;
        }
        case 3: { // Animate Orbit Wave
            let orbit = abs(length(p * vec2<f32>(0.85, 1.25)) - 0.56) - 0.10;
            let dot_d = length(p - vec2<f32>(0.46, -0.22)) - 0.22;
            return min(orbit, dot_d);
        }
        case 4: { // Play Triangle (pointing right)
            let q = vec2<f32>(p.x + 0.10, abs(p.y));
            let d1 = (q.x * 0.52 + q.y * 0.85) - 0.36;
            let d2 = -q.x - 0.36;
            return max(d1, d2) - 0.08;
        }
        case 5: { // Brush Shape (Circle + Soft Square)
            let c = length(p - vec2<f32>(-0.20, -0.16)) - 0.42;
            let b = sd_rounded_box(p - vec2<f32>(0.24, 0.22), vec2<f32>(0.36, 0.36), 0.10);
            return min(c, abs(b) - 0.09);
        }
        case 6: { // Fleck / Painterly 4-Point Star
            let ap = abs(p) + vec2<f32>(0.06);
            let star = (ap.x + ap.y + length(ap) * 0.55) * 0.52 - 0.48;
            return star;
        }
        case 7: { // Mirror Symmetry (| axis + 2 dots/wings)
            let axis = sd_segment(p, vec2<f32>(0.0, -0.66), vec2<f32>(0.0, 0.66)) - 0.08;
            let wing = length(vec2<f32>(abs(p.x) - 0.44, p.y)) - 0.23;
            return min(axis, wing);
        }
        case 8: { // Scope In / Focus Brackets
            let ap = abs(p);
            let corner_h = sd_segment(ap, vec2<f32>(0.24, 0.62), vec2<f32>(0.62, 0.62)) - 0.10;
            let corner_v = sd_segment(ap, vec2<f32>(0.62, 0.24), vec2<f32>(0.62, 0.62)) - 0.10;
            let core = length(p) - 0.22;
            return min(min(corner_h, corner_v), core);
        }
        case 9: { // Duplicate / Clone Squares
            let back = abs(sd_rounded_box(p - vec2<f32>(-0.18, -0.18), vec2<f32>(0.38, 0.38), 0.10)) - 0.08;
            let front = sd_rounded_box(p - vec2<f32>(0.18, 0.18), vec2<f32>(0.38, 0.38), 0.10);
            return min(back, front);
        }
        case 10: { // Undo (Left Arrow)
            let shaft = sd_segment(p, vec2<f32>(-0.48, 0.0), vec2<f32>(0.52, 0.0)) - 0.13;
            let head1 = sd_segment(p, vec2<f32>(-0.52, 0.0), vec2<f32>(-0.08, -0.42)) - 0.13;
            let head2 = sd_segment(p, vec2<f32>(-0.52, 0.0), vec2<f32>(-0.08, 0.42)) - 0.13;
            return min(shaft, min(head1, head2));
        }
        case 11: { // Redo (Right Arrow)
            let shaft = sd_segment(p, vec2<f32>(-0.52, 0.0), vec2<f32>(0.48, 0.0)) - 0.13;
            let head1 = sd_segment(p, vec2<f32>(0.52, 0.0), vec2<f32>(0.08, -0.42)) - 0.13;
            let head2 = sd_segment(p, vec2<f32>(0.52, 0.0), vec2<f32>(0.08, 0.42)) - 0.13;
            return min(shaft, min(head1, head2));
        }
        case 12: { // Radial Wheel
            let outer = abs(length(p) - 0.56) - 0.12;
            let hub = length(p) - 0.16;
            return min(outer, hub);
        }
        case 13: { // Save (Arrow into Tray)
            let arrow_v = sd_segment(p, vec2<f32>(0.0, -0.58), vec2<f32>(0.0, 0.16)) - 0.12;
            let head_l = sd_segment(p, vec2<f32>(0.0, 0.20), vec2<f32>(-0.32, -0.10)) - 0.11;
            let head_r = sd_segment(p, vec2<f32>(0.0, 0.20), vec2<f32>(0.32, -0.10)) - 0.11;
            let tray = sd_segment(p, vec2<f32>(-0.54, 0.52), vec2<f32>(0.54, 0.52)) - 0.12;
            return min(min(arrow_v, tray), min(head_l, head_r));
        }
        case 14: { // Reset / Island Sun-Hill
            let hill = length(p - vec2<f32>(0.0, 0.45)) - 0.62;
            let clip_bottom = max(hill, p.y - 0.42);
            let sun = length(p - vec2<f32>(0.34, -0.36)) - 0.18;
            return min(clip_bottom, sun);
        }
        case 15: { // Record Dot
            let core = length(p) - 0.34;
            let ring = abs(length(p) - 0.60) - 0.08;
            return min(core, ring);
        }
        case 16: { // Limb / Joint Link (o-o)
            let bar = sd_segment(p, vec2<f32>(-0.42, 0.36), vec2<f32>(0.42, -0.36)) - 0.10;
            let n1 = length(p - vec2<f32>(-0.42, 0.36)) - 0.23;
            let n2 = length(p - vec2<f32>(0.42, -0.36)) - 0.23;
            return min(bar, min(n1, n2));
        }
        case 17: { // Delete / X Cross
            let d1 = sd_segment(p, vec2<f32>(-0.44, -0.44), vec2<f32>(0.44, 0.44)) - 0.13;
            let d2 = sd_segment(p, vec2<f32>(-0.44, 0.44), vec2<f32>(0.44, -0.44)) - 0.13;
            return min(d1, d2);
        }
        default: { // 18: Info / Help Circle
            let ring = abs(length(p) - 0.62) - 0.10;
            let stem = sd_segment(p, vec2<f32>(0.0, -0.05), vec2<f32>(0.0, 0.34)) - 0.11;
            let dot_i = length(p - vec2<f32>(0.0, -0.32)) - 0.11;
            return min(ring, min(stem, dot_i));
        }
    }
}

@fragment
fn fs_ui(vin: VsOut) -> @location(0) vec4<f32> {
    let q = quads[vin.quad_idx];
    let kind = q.params.z;

    // Kind 1: Smooth Anti-Aliased SDF Text Glyph + Built-In Dark Contrast Halo!
    if (abs(kind - 1.0) < 0.25) {
        let dist = textureSampleLevel(font_tex, ui_sampler, vin.uv, 0.0).r;
        let glyph_h = max(q.rect_px.w - q.rect_px.y, 6.0);
        // Analytical screen-space anti-aliasing width (zero fwidth uniformity issues!)
        let aa = clamp((32.0 / glyph_h) / 11.0 * 0.62, 0.022, 0.13);

        let body_a = smoothstep(0.49 - aa, 0.49 + aa, dist);
        let halo_a = smoothstep(0.26 - aa, 0.47, dist) * q.border_color.a;
        let combined_a = max(halo_a, body_a * q.fill_color.a);

        if (combined_a < 0.02) {
            discard;
        }
        let srgb_rgb = mix(q.border_color.rgb, q.fill_color.rgb, body_a);
        var col = srgb_to_lin(vec4<f32>(srgb_rgb, combined_a));
        return col;
    }

    // Kind 3: Crisp Procedural Vector UI Icon
    if (abs(kind - 3.0) < 0.25) {
        let center = (q.rect_px.xy + q.rect_px.zw) * 0.5;
        let half_ext = max((q.rect_px.zw - q.rect_px.xy) * 0.5, vec2<f32>(1.0));
        let local_p = (vin.screen_px - center) / half_ext;
        let px_unit = 1.0 / min(half_ext.x, half_ext.y);
        let d = eval_ui_icon(local_p, i32(round(q.params.x)));

        let body_a = 1.0 - smoothstep(-px_unit * 0.85, px_unit * 0.95, d);
        let shadow_d = eval_ui_icon(local_p - vec2<f32>(0.0, px_unit * 1.4), i32(round(q.params.x)));
        let shadow_a = (1.0 - smoothstep(-px_unit, px_unit * 1.6, shadow_d)) * q.border_color.a;
        let out_a = max(shadow_a, body_a * q.fill_color.a);
        if (out_a < 0.02) {
            discard;
        }
        let icon_rgb = mix(q.border_color.rgb, q.fill_color.rgb, body_a);
        return srgb_to_lin(vec4<f32>(icon_rgb, out_a));
    }

    // Kind 2: Bouncy 8-Slice Radial Pie Wheel
    if (abs(kind - 2.0) < 0.25) {
        let ui_s = max(u_ui.viewport_srgb_scale.w, 1.0);
        let center = u_ui.radial_params.xy;
        let outer_r = u_ui.radial_params.z;
        let inner_r = 38.0 * ui_s;
        let rel = vin.screen_px - center;
        let dist = length(rel);

        let angle = atan2(rel.y, rel.x);
        let norm_ang = (angle + 3.14159265 * 2.5) % 6.2831853;
        let slice_f = (norm_ang / 6.2831853) * 8.0;
        let slice_idx = floor(slice_f + 0.5) % 8.0;
        let hovered = abs(slice_idx - u_ui.radial_params.w) < 0.25 && u_ui.radial_params.w >= 0.0;

        let slice_max_r = select(outer_r, outer_r + 12.0 * ui_s, hovered);
        let ring_d = max(inner_r - dist, dist - slice_max_r);

        let ang_frac = abs(fract(slice_f + 0.5) - 0.5);
        let gap_px = ang_frac * (6.2831853 / 8.0) * dist;
        let wedge_d = max(ring_d, 2.5 * ui_s - gap_px);

        let alpha_mask = 1.0 - smoothstep(-0.9, 1.3, wedge_d);
        if (alpha_mask < 0.02) {
            discard;
        }

        let fill_srgb = select(
            vec4<f32>(0.11, 0.13, 0.20, 0.94),
            vec4<f32>(0.98, 0.68, 0.24, 0.98),
            hovered
        );
        let rim = 1.0 - smoothstep(0.0, 2.4 * ui_s, abs(wedge_d + 1.2 * ui_s));
        var col = srgb_to_lin(mix(fill_srgb, vec4<f32>(1.0, 0.96, 0.86, 0.99), rim * 0.72));
        col.a *= alpha_mask;
        return col;
    }

    // Kind 0: Rounded Pill / Panel / Glossy Button / Soft Drop Shadow
    let center = (q.rect_px.xy + q.rect_px.zw) * 0.5;
    let half_ext = (q.rect_px.zw - q.rect_px.xy) * 0.5;
    let r = min(q.params.x, min(half_ext.x, half_ext.y));
    let d = sd_rounded_box(vin.screen_px - center, half_ext, r);
    let variant = q.params.w;

    // Soft Drop Shadow quad (`variant = 2.0`)
    if (abs(variant - 2.0) < 0.25) {
        let blur = max(q.params.y, 4.0);
        let sh_a = (1.0 - smoothstep(-blur * 0.35, blur, d)) * q.fill_color.a;
        if (sh_a < 0.01) {
            discard;
        }
        return srgb_to_lin(vec4<f32>(q.fill_color.rgb, sh_a));
    }

    let aa = 1.0 - smoothstep(-0.75, 0.95, d);
    if (aa < 0.01) {
        discard;
    }

    var base_fill = q.fill_color;
    // Tactile glossy top-to-bottom button bevel (`variant = 1.0`)
    if (abs(variant - 1.0) < 0.25) {
        let v_grad = clamp((vin.screen_px.y - q.rect_px.y) / max(q.rect_px.w - q.rect_px.y, 1.0), 0.0, 1.0);
        let bevel = mix(1.11, 0.89, v_grad);
        base_fill = vec4<f32>(clamp(base_fill.rgb * bevel, vec3<f32>(0.0), vec3<f32>(1.0)), base_fill.a);
    }

    let bw = q.params.y;
    let border_t = smoothstep(-bw - 0.85, -bw + 0.55, d) * q.border_color.a;
    let mixed_srgb = mix(base_fill, vec4<f32>(q.border_color.rgb, 1.0), border_t);
    var out_col = srgb_to_lin(mixed_srgb);
    out_col.a *= aa;
    return out_col;
}
