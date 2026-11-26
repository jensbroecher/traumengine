//! Dreams-inspired `wgpu` rendering pipeline for Traum Engine:
//! 1. Compute Pass 1A & 1B (`sdf_compute.wgsl` + `sdf_asset_compute.wgsl`): Evaluates dirty `8x8x8` SDF bricks
//!    for both the World Island (`192x128x192`) and the 24-slot Reusable Sculpt Asset Atlas (`256x128x192`).
//! 2. Compute Pass 2 (`extract_flecks.wgsl`): Caches local iso-surface flecks per custom `SculptAsset` and emits
//!    world + instanced prop/clone/limb flecks + indirect draw args.
//! 3. Render Pass 1 (`scene_raymarch.wgsl`): Watertight impasto clay core, custom sculpted props, 3D gizmos,
//!    joint connectors, playable buddy, & Imp guides.
//! 4. Render Pass 2 (`fleck_splat.wgsl`): Instanced painterly brushstroke splats with Living Flecks motion.
//! 5. Render Pass 3 (`post_dof.wgsl` + `toy_ui.wgsl`): Tilt-shift Bokeh DOF + ACES + Tactile Toy UI & Radial Wheel.

mod font;

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use std::sync::Arc;
use traum_core::EditorMode;
use traum_editor::{TooltipCard, TraumEditor, UiFrame, UiRectStyle};
use traum_input::{ActiveInputDevice, CameraRig, FrameInput};
use traum_sdf::{
    asset_slot_tile_coords, GpuAssetDirtyBrick, GpuAssetHeader, GpuConnectorVisual, GpuFleck,
    GpuSceneProp, GpuSculptEdit, ASSET_ATLAS_DIMS, ASSET_BRICKS_PER_AXIS, ASSET_BRICKS_PER_SLOT,
    ASSET_LOCAL_MIN, ASSET_TILE_DIM, ASSET_VOXEL_SIZE, BRICK_GRID, MAX_ASSET_LOCAL_FLECKS,
    MAX_GPU_ASSET_EDITS, MAX_GPU_CONNECTORS, MAX_GPU_EDITS, MAX_GPU_PROPS, MAX_SCULPT_ASSETS,
    MAX_SURFACE_FLECKS, VOLUME_DIMS, VOLUME_MAX, VOLUME_MIN, VOXEL_SIZE,
};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::window::Window;

const MAX_UI_QUADS: usize = 8192;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WorldBrickUniforms {
    volume_min_voxel: [f32; 4],
    volume_dims_edits: [u32; 4],
    brick_grid_dirty: [u32; 4],
    counts: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AssetBrickUniforms {
    local_min_voxel: [f32; 4],
    counts: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FleckExtractUniforms {
    volume_min_voxel: [f32; 4],
    volume_dims_edits: [u32; 4],
    asset_atlas_info: [u32; 4],
    counts: [u32; 4],
    asset_min_voxel: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniforms {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    cam_pos_time: [f32; 4],
    cam_fwd_focus: [f32; 4],
    sun_dof: [f32; 4],
    viewport_meta: [f32; 4],
    imp_pos_size: [f32; 4],
    imp_norm_mode: [f32; 4],
    imp_shadow_hit: [f32; 4],
    brush_color_op: [f32; 4],
    puppet_pos_yaw: [f32; 4],
    puppet_extra: [f32; 4],
    vol_min_voxel: [f32; 4],
    vol_max_scoped: [f32; 4],
    asset_atlas_info: [f32; 4],
    asset_min_slot: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostUniforms {
    focus_dof_srgb: [f32; 4],
    viewport_info: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UiUniforms {
    viewport_srgb_scale: [f32; 4],
    radial_params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuUiQuad {
    rect_px: [f32; 4],
    uv_rect: [f32; 4],
    fill_color: [f32; 4],
    border_color: [f32; 4],
    params: [f32; 4],
}

pub struct TraumRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: Option<wgpu::Surface<'static>>,
    surface_config: Option<wgpu::SurfaceConfiguration>,
    output_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    frame_index: u32,

    // World & Reusable Asset SDF Buffers
    world_brick_uniform_buf: wgpu::Buffer,
    asset_brick_uniform_buf: wgpu::Buffer,
    extract_uniform_buf: wgpu::Buffer,

    edits_buf: wgpu::Buffer,
    dirty_bricks_buf: wgpu::Buffer,

    asset_edits_buf: wgpu::Buffer,
    asset_headers_buf: wgpu::Buffer,
    asset_dirty_bricks_buf: wgpu::Buffer,
    #[allow(dead_code)]
    asset_local_flecks_buf: wgpu::Buffer,
    #[allow(dead_code)]
    asset_fleck_counts_buf: wgpu::Buffer,

    props_buf: wgpu::Buffer,
    connectors_buf: wgpu::Buffer,
    #[allow(dead_code)]
    flecks_buf: wgpu::Buffer,
    #[allow(dead_code)]
    fleck_counter_buf: wgpu::Buffer,
    indirect_buf: wgpu::Buffer,

    frame_uniform_buf: wgpu::Buffer,
    post_uniform_buf: wgpu::Buffer,
    ui_uniform_buf: wgpu::Buffer,
    ui_quads_buf: wgpu::Buffer,

    // Screen-size dependent offscreen HDR & Depth targets
    hdr_tex: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    depth_tex: wgpu::Texture,
    depth_view: wgpu::TextureView,
    linear_sampler: wgpu::Sampler,

    // Pipelines & BindGroups
    pipe_update_bricks: wgpu::ComputePipeline,
    bg_update_bricks: wgpu::BindGroup,

    pipe_update_asset_bricks: wgpu::ComputePipeline,
    bg_update_asset_bricks: wgpu::BindGroup,

    pipe_reset_asset_flecks: wgpu::ComputePipeline,
    pipe_extract_asset_flecks: wgpu::ComputePipeline,
    pipe_reset_flecks: wgpu::ComputePipeline,
    pipe_extract_flecks: wgpu::ComputePipeline,
    pipe_prop_flecks: wgpu::ComputePipeline,
    pipe_finalize_flecks: wgpu::ComputePipeline,
    bg_extract_flecks: wgpu::BindGroup,

    pipe_raymarch: wgpu::RenderPipeline,
    bg_raymarch: wgpu::BindGroup,

    pipe_fleck_splat: wgpu::RenderPipeline,
    bg_fleck_splat: wgpu::BindGroup,

    pipe_post_dof: wgpu::RenderPipeline,
    bgl_post_dof: wgpu::BindGroupLayout,
    bg_post_dof: wgpu::BindGroup,

    pipe_toy_ui: wgpu::RenderPipeline,
    bg_toy_ui: wgpu::BindGroup,

    // Cached scratch vectors
    gpu_edits_scratch: Vec<GpuSculptEdit>,
    gpu_asset_edits_scratch: Vec<GpuSculptEdit>,
    gpu_asset_headers_scratch: Vec<GpuAssetHeader>,
    gpu_props_scratch: Vec<GpuSceneProp>,
    gpu_connectors_scratch: Vec<GpuConnectorVisual>,
    ui_quads_scratch: Vec<GpuUiQuad>,
    needs_world_fleck_refresh: bool,
}

impl TraumRenderer {
    pub fn new_windowed(window: Arc<Window>) -> Result<Self, String> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(window)
            .map_err(|e| format!("create_surface failed: {e}"))?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| "No compatible GPU adapter found".to_string())?;

        log::info!("Traum Engine GPU Adapter: {:?}", adapter.get_info());

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("traum_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("request_device failed: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        let output_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let width = size.width.max(1);
        let height = size.height.max(1);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: output_format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Ok(Self::build_core(
            device,
            queue,
            Some(surface),
            Some(config),
            output_format,
            width,
            height,
        ))
    }

    fn new_headless(width: u32, height: u32) -> Result<(Self, wgpu::AdapterInfo), String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .or_else(|| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
            }))
        })
        .ok_or_else(|| "No GPU or fallback adapter available for headless test".to_string())?;

        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("traum_headless_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("headless request_device failed: {e}"))?;

        let renderer = Self::build_core(
            device,
            queue,
            None,
            None,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            width,
            height,
        );
        Ok((renderer, info))
    }

    fn build_core(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: Option<wgpu::Surface<'static>>,
        surface_config: Option<wgpu::SurfaceConfiguration>,
        output_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let make_3d_tex = |label: &str, dims: glam::UVec3| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: dims.x,
                    height: dims.y,
                    depth_or_array_layers: dims.z,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };

        // 1. World SDF & Color 3D Volumes (`192 x 128 x 192`)
        let sdf_vol_tex = make_3d_tex("sdf_brick_volume", VOLUME_DIMS);
        let sdf_vol_view = sdf_vol_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let color_vol_tex = make_3d_tex("color_brick_volume", VOLUME_DIMS);
        let color_vol_view = color_vol_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // 2. Reusable Sculpt Asset 3D Atlas (`256 x 128 x 192` = 24 slots of `64^3`!)
        let asset_sdf_tex = make_3d_tex("asset_sdf_atlas", ASSET_ATLAS_DIMS);
        let asset_sdf_view = asset_sdf_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let asset_color_tex = make_3d_tex("asset_color_atlas", ASSET_ATLAS_DIMS);
        let asset_color_view = asset_color_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let vol_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vol_trilinear_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hdr_linear_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let ui_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ui_sdf_linear_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let (fw, fh, fpixels) = font::build_font_atlas();
        let font_tex = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("embedded_font_atlas"),
                size: wgpu::Extent3d {
                    width: fw,
                    height: fh,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &fpixels,
        );
        let font_view = font_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Uniform & Storage Buffers
        let world_brick_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("world_brick_uniforms"),
            size: std::mem::size_of::<WorldBrickUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let asset_brick_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("asset_brick_uniforms"),
            size: std::mem::size_of::<AssetBrickUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let extract_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("extract_uniforms"),
            size: std::mem::size_of::<FleckExtractUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let edits_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sculpt_edits_buf"),
            size: (MAX_GPU_EDITS * std::mem::size_of::<GpuSculptEdit>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let total_bricks = (BRICK_GRID.x * BRICK_GRID.y * BRICK_GRID.z) as usize;
        let dirty_bricks_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dirty_bricks_buf"),
            size: (total_bricks * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let asset_edits_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("asset_edits_buf"),
            size: (MAX_GPU_ASSET_EDITS * std::mem::size_of::<GpuSculptEdit>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let asset_headers_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("asset_headers_buf"),
            size: (MAX_SCULPT_ASSETS * std::mem::size_of::<GpuAssetHeader>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let total_asset_bricks = MAX_SCULPT_ASSETS * (ASSET_BRICKS_PER_SLOT as usize);
        let asset_dirty_bricks_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("asset_dirty_bricks_buf"),
            size: (total_asset_bricks * std::mem::size_of::<GpuAssetDirtyBrick>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let asset_local_flecks_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("asset_local_flecks_buf"),
            size: (MAX_SCULPT_ASSETS
                * (MAX_ASSET_LOCAL_FLECKS as usize)
                * std::mem::size_of::<GpuFleck>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let asset_fleck_counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("asset_fleck_counts_buf"),
            size: 128, // 24 x u32 = 96 bytes, padded cleanly to 128 bytes
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let props_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_props_buf"),
            size: (MAX_GPU_PROPS * std::mem::size_of::<GpuSceneProp>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let connectors_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("connectors_buf"),
            size: (MAX_GPU_CONNECTORS * std::mem::size_of::<GpuConnectorVisual>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let flecks_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_flecks_buf"),
            size: (MAX_SURFACE_FLECKS as usize * std::mem::size_of::<GpuFleck>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let fleck_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fleck_counter_buf"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let indirect_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fleck_indirect_buf"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let frame_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_uniforms"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let post_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post_uniforms"),
            size: std::mem::size_of::<PostUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ui_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui_uniforms"),
            size: std::mem::size_of::<UiUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ui_quads_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui_quads_buf"),
            size: (MAX_UI_QUADS * std::mem::size_of::<GpuUiQuad>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (hdr_tex, hdr_view, depth_tex, depth_view) =
            Self::create_frame_attachments(&device, width, height);

        // ====================================================================
        // Shader Modules
        // ====================================================================
        let mod_bricks = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf_compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sdf_compute.wgsl").into()),
        });
        let mod_asset_bricks = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf_asset_compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/sdf_asset_compute.wgsl").into(),
            ),
        });
        let mod_extract = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("extract_flecks.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/extract_flecks.wgsl").into()),
        });
        let mod_raymarch = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene_raymarch.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scene_raymarch.wgsl").into()),
        });
        let mod_fleck = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fleck_splat.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fleck_splat.wgsl").into()),
        });
        let mod_post = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post_dof.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/post_dof.wgsl").into()),
        });
        let mod_ui = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("toy_ui.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/toy_ui.wgsl").into()),
        });

        // 1A. Compute Pass 1A (`update_dirty_bricks`)
        let bgl_bricks = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_update_bricks"),
            entries: &[
                bgl_uniform(0, wgpu::ShaderStages::COMPUTE),
                bgl_storage(1, wgpu::ShaderStages::COMPUTE, true),
                bgl_storage(2, wgpu::ShaderStages::COMPUTE, true),
                bgl_storage_tex_3d(3, wgpu::TextureFormat::Rgba16Float),
                bgl_storage_tex_3d(4, wgpu::TextureFormat::Rgba16Float),
            ],
        });
        let pl_bricks = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_update_bricks"),
            bind_group_layouts: &[&bgl_bricks],
            push_constant_ranges: &[],
        });
        let pipe_update_bricks = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipe_update_bricks"),
            layout: Some(&pl_bricks),
            module: &mod_bricks,
            entry_point: Some("update_dirty_bricks"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bg_update_bricks = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_update_bricks"),
            layout: &bgl_bricks,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: world_brick_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: edits_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dirty_bricks_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&sdf_vol_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&color_vol_view),
                },
            ],
        });

        // 1B. Compute Pass 1B (`update_dirty_asset_bricks`)
        let bgl_asset_bricks = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_update_asset_bricks"),
            entries: &[
                bgl_uniform(0, wgpu::ShaderStages::COMPUTE),
                bgl_storage(1, wgpu::ShaderStages::COMPUTE, true),
                bgl_storage(2, wgpu::ShaderStages::COMPUTE, true),
                bgl_storage(3, wgpu::ShaderStages::COMPUTE, true),
                bgl_storage_tex_3d(4, wgpu::TextureFormat::Rgba16Float),
                bgl_storage_tex_3d(5, wgpu::TextureFormat::Rgba16Float),
            ],
        });
        let pl_asset_bricks = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_update_asset_bricks"),
            bind_group_layouts: &[&bgl_asset_bricks],
            push_constant_ranges: &[],
        });
        let pipe_update_asset_bricks =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pipe_update_asset_bricks"),
                layout: Some(&pl_asset_bricks),
                module: &mod_asset_bricks,
                entry_point: Some("update_dirty_asset_bricks"),
                compilation_options: Default::default(),
                cache: None,
            });
        let bg_update_asset_bricks = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_update_asset_bricks"),
            layout: &bgl_asset_bricks,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: asset_brick_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: asset_edits_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: asset_headers_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: asset_dirty_bricks_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&asset_sdf_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&asset_color_view),
                },
            ],
        });

        // 2. Compute Pass 2 (`extract_flecks.wgsl`)
        let bgl_extract = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_extract_flecks"),
            entries: &[
                bgl_uniform(0, wgpu::ShaderStages::COMPUTE),
                bgl_tex_3d(1, wgpu::ShaderStages::COMPUTE),
                bgl_tex_3d(2, wgpu::ShaderStages::COMPUTE),
                bgl_sampler(3, wgpu::ShaderStages::COMPUTE),
                bgl_storage(4, wgpu::ShaderStages::COMPUTE, true),
                bgl_storage(5, wgpu::ShaderStages::COMPUTE, false),
                bgl_storage(6, wgpu::ShaderStages::COMPUTE, false),
                bgl_storage(7, wgpu::ShaderStages::COMPUTE, false),
                bgl_tex_3d(8, wgpu::ShaderStages::COMPUTE),
                bgl_tex_3d(9, wgpu::ShaderStages::COMPUTE),
                bgl_storage(10, wgpu::ShaderStages::COMPUTE, true),
                bgl_storage(11, wgpu::ShaderStages::COMPUTE, false),
                bgl_storage(12, wgpu::ShaderStages::COMPUTE, false),
            ],
        });
        let pl_extract = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_extract_flecks"),
            bind_group_layouts: &[&bgl_extract],
            push_constant_ranges: &[],
        });
        let make_extract_pipe = |ep: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: Some(&pl_extract),
                module: &mod_extract,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipe_reset_asset_flecks = make_extract_pipe("reset_dirty_asset_flecks");
        let pipe_extract_asset_flecks = make_extract_pipe("extract_asset_local_flecks");
        let pipe_reset_flecks = make_extract_pipe("reset_fleck_counter");
        let pipe_extract_flecks = make_extract_pipe("extract_surface_flecks");
        let pipe_prop_flecks = make_extract_pipe("emit_prop_flecks");
        let pipe_finalize_flecks = make_extract_pipe("finalize_fleck_indirect");

        let bg_extract_flecks = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_extract_flecks"),
            layout: &bgl_extract,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: extract_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&sdf_vol_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&color_vol_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&vol_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: props_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: flecks_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: fleck_counter_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: indirect_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&asset_sdf_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&asset_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: asset_headers_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: asset_local_flecks_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: asset_fleck_counts_buf.as_entire_binding(),
                },
            ],
        });

        // 3. Render Pass 1: Clay Raymarch (`scene_raymarch.wgsl`)
        let bgl_raymarch = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_raymarch"),
            entries: &[
                bgl_uniform(0, wgpu::ShaderStages::FRAGMENT),
                bgl_tex_3d(1, wgpu::ShaderStages::FRAGMENT),
                bgl_tex_3d(2, wgpu::ShaderStages::FRAGMENT),
                bgl_sampler(3, wgpu::ShaderStages::FRAGMENT),
                bgl_storage(4, wgpu::ShaderStages::FRAGMENT, true),
                bgl_tex_3d(5, wgpu::ShaderStages::FRAGMENT),
                bgl_tex_3d(6, wgpu::ShaderStages::FRAGMENT),
                bgl_storage(7, wgpu::ShaderStages::FRAGMENT, true),
            ],
        });
        let pl_raymarch = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_raymarch"),
            bind_group_layouts: &[&bgl_raymarch],
            push_constant_ranges: &[],
        });
        let pipe_raymarch = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_raymarch"),
            layout: Some(&pl_raymarch),
            vertex: wgpu::VertexState {
                module: &mod_raymarch,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &mod_raymarch,
                entry_point: Some("fs_raymarch"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let bg_raymarch = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_raymarch"),
            layout: &bgl_raymarch,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&sdf_vol_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&color_vol_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&vol_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: props_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&asset_sdf_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&asset_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: connectors_buf.as_entire_binding(),
                },
            ],
        });

        // 4. Render Pass 2: Painterly Fleck Splatting
        let bgl_fleck = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_fleck_splat"),
            entries: &[
                bgl_uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                bgl_tex_3d(1, wgpu::ShaderStages::VERTEX),
                bgl_sampler(2, wgpu::ShaderStages::VERTEX),
                bgl_storage(3, wgpu::ShaderStages::VERTEX, true),
            ],
        });
        let pl_fleck = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_fleck_splat"),
            bind_group_layouts: &[&bgl_fleck],
            push_constant_ranges: &[],
        });
        let pipe_fleck_splat = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_fleck_splat"),
            layout: Some(&pl_fleck),
            vertex: wgpu::VertexState {
                module: &mod_fleck,
                entry_point: Some("vs_fleck"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &mod_fleck,
                entry_point: Some("fs_fleck"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let bg_fleck_splat = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_fleck_splat"),
            layout: &bgl_fleck,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&sdf_vol_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&vol_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: flecks_buf.as_entire_binding(),
                },
            ],
        });

        // 5. Render Pass 3A: Bokeh DOF + ACES Post
        let bgl_post_dof = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_post_dof"),
            entries: &[
                bgl_uniform(0, wgpu::ShaderStages::FRAGMENT),
                bgl_tex_2d(1, wgpu::ShaderStages::FRAGMENT),
                bgl_sampler(2, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let pl_post = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_post_dof"),
            bind_group_layouts: &[&bgl_post_dof],
            push_constant_ranges: &[],
        });
        let pipe_post_dof = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_post_dof"),
            layout: Some(&pl_post),
            vertex: wgpu::VertexState {
                module: &mod_post,
                entry_point: Some("vs_post"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &mod_post,
                entry_point: Some("fs_post"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let bg_post_dof = Self::create_post_bind_group(
            &device,
            &bgl_post_dof,
            &post_uniform_buf,
            &hdr_view,
            &linear_sampler,
        );

        // 6. Render Pass 3B: Tactile Toy UI + Radial Wheel
        let bgl_ui = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_toy_ui"),
            entries: &[
                bgl_uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                bgl_storage(1, wgpu::ShaderStages::VERTEX_FRAGMENT, true),
                bgl_tex_2d(2, wgpu::ShaderStages::FRAGMENT),
                bgl_sampler(3, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let pl_ui = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_toy_ui"),
            bind_group_layouts: &[&bgl_ui],
            push_constant_ranges: &[],
        });
        let pipe_toy_ui = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipe_toy_ui"),
            layout: Some(&pl_ui),
            vertex: wgpu::VertexState {
                module: &mod_ui,
                entry_point: Some("vs_ui"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &mod_ui,
                entry_point: Some("fs_ui"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let bg_toy_ui = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_toy_ui"),
            layout: &bgl_ui,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ui_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ui_quads_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&font_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&ui_sampler),
                },
            ],
        });

        Self {
            device,
            queue,
            surface,
            surface_config,
            output_format,
            width,
            height,
            frame_index: 0,
            world_brick_uniform_buf,
            asset_brick_uniform_buf,
            extract_uniform_buf,
            edits_buf,
            dirty_bricks_buf,
            asset_edits_buf,
            asset_headers_buf,
            asset_dirty_bricks_buf,
            asset_local_flecks_buf,
            asset_fleck_counts_buf,
            props_buf,
            connectors_buf,
            flecks_buf,
            fleck_counter_buf,
            indirect_buf,
            frame_uniform_buf,
            post_uniform_buf,
            ui_uniform_buf,
            ui_quads_buf,
            hdr_tex,
            hdr_view,
            depth_tex,
            depth_view,
            linear_sampler,
            pipe_update_bricks,
            bg_update_bricks,
            pipe_update_asset_bricks,
            bg_update_asset_bricks,
            pipe_reset_asset_flecks,
            pipe_extract_asset_flecks,
            pipe_reset_flecks,
            pipe_extract_flecks,
            pipe_prop_flecks,
            pipe_finalize_flecks,
            bg_extract_flecks,
            pipe_raymarch,
            bg_raymarch,
            pipe_fleck_splat,
            bg_fleck_splat,
            pipe_post_dof,
            bgl_post_dof,
            bg_post_dof,
            pipe_toy_ui,
            bg_toy_ui,
            gpu_edits_scratch: Vec::with_capacity(512),
            gpu_asset_edits_scratch: Vec::with_capacity(512),
            gpu_asset_headers_scratch: Vec::with_capacity(MAX_SCULPT_ASSETS),
            gpu_props_scratch: Vec::with_capacity(64),
            gpu_connectors_scratch: Vec::with_capacity(32),
            ui_quads_scratch: Vec::with_capacity(1024),
            needs_world_fleck_refresh: true,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.width = new_size.width;
        self.height = new_size.height;

        if let (Some(surface), Some(cfg)) = (&self.surface, &mut self.surface_config) {
            cfg.width = self.width;
            cfg.height = self.height;
            surface.configure(&self.device, cfg);
        }

        let (hdr_tex, hdr_view, depth_tex, depth_view) =
            Self::create_frame_attachments(&self.device, self.width, self.height);
        self.hdr_tex = hdr_tex;
        self.hdr_view = hdr_view;
        self.depth_tex = depth_tex;
        self.depth_view = depth_view;
        self.bg_post_dof = Self::create_post_bind_group(
            &self.device,
            &self.bgl_post_dof,
            &self.post_uniform_buf,
            &self.hdr_view,
            &self.linear_sampler,
        );
    }

    pub fn render_window_frame(
        &mut self,
        editor: &mut TraumEditor,
        camera: &CameraRig,
        ui: &UiFrame,
    ) -> Result<(), String> {
        let surface = self.surface.as_ref().ok_or("No window surface")?;
        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                if let Some(cfg) = &self.surface_config {
                    surface.configure(&self.device, cfg);
                }
                return Ok(());
            }
            Err(wgpu::SurfaceError::Timeout) => return Ok(()),
            Err(e) => return Err(format!("Surface error: {e}")),
        };

        let target_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.encode_and_submit_frame(editor, camera, ui, &target_view);
        frame.present();
        Ok(())
    }

    fn encode_and_submit_frame(
        &mut self,
        editor: &mut TraumEditor,
        camera: &CameraRig,
        ui: &UiFrame,
        target_view: &wgpu::TextureView,
    ) {
        self.frame_index = self.frame_index.wrapping_add(1);
        editor.asset_dirty.sync_scene_assets(&editor.scene.assets);

        // 1A. Sync World Sculpt Edits if changed
        let edit_count = editor.scene.world_edits.len().min(MAX_GPU_EDITS);
        if editor.edits_buffer_dirty {
            self.gpu_edits_scratch.clear();
            for e in editor.scene.world_edits.iter().take(edit_count) {
                self.gpu_edits_scratch.push(GpuSculptEdit::from_edit(e));
            }
            if !self.gpu_edits_scratch.is_empty() {
                self.queue.write_buffer(
                    &self.edits_buf,
                    0,
                    bytemuck::cast_slice(&self.gpu_edits_scratch),
                );
            }
            editor.edits_buffer_dirty = false;
        }

        // 1B. Sync Reusable `SculptAsset` Headers & Local Edits if changed
        if editor.asset_edits_dirty {
            self.gpu_asset_edits_scratch.clear();
            self.gpu_asset_headers_scratch.clear();

            for slot in 0..MAX_SCULPT_ASSETS {
                let tile = asset_slot_tile_coords(slot);
                if let Some(asset) = editor.scene.assets.get(slot) {
                    let offset = self.gpu_asset_edits_scratch.len() as u32;
                    let mut count = 0u32;
                    for e in &asset.edits {
                        if self.gpu_asset_edits_scratch.len() >= MAX_GPU_ASSET_EDITS {
                            break;
                        }
                        self.gpu_asset_edits_scratch
                            .push(GpuSculptEdit::from_edit(e));
                        count += 1;
                    }
                    self.gpu_asset_headers_scratch.push(GpuAssetHeader {
                        edit_and_tile_xy: [offset, count, tile.x, tile.y],
                        tile_z_and_meta: [tile.z, asset.id, 0, 0],
                    });
                } else {
                    self.gpu_asset_headers_scratch.push(GpuAssetHeader {
                        edit_and_tile_xy: [0, 0, tile.x, tile.y],
                        tile_z_and_meta: [tile.z, 0, 0, 0],
                    });
                }
            }

            if !self.gpu_asset_edits_scratch.is_empty() {
                self.queue.write_buffer(
                    &self.asset_edits_buf,
                    0,
                    bytemuck::cast_slice(&self.gpu_asset_edits_scratch),
                );
            }
            self.queue.write_buffer(
                &self.asset_headers_buf,
                0,
                bytemuck::cast_slice(&self.gpu_asset_headers_scratch),
            );
            editor.asset_edits_dirty = false;
        }

        // 2. Upload hierarchical animated Toy Props & Visual Joint Connectors
        let prop_count = editor.scene.props.len().min(MAX_GPU_PROPS);
        self.gpu_props_scratch.clear();
        self.gpu_connectors_scratch.clear();

        let scoped_asset_slot = editor
            .scoped_prop_id
            .and_then(|pid| editor.scene.props.iter().find(|p| p.id == pid))
            .and_then(|p| editor.asset_slot_of(p.asset_id));

        for prop in editor.scene.props.iter().take(prop_count) {
            let Some(slot) = editor.asset_slot_of(prop.asset_id) else {
                continue;
            };
            let (pos, rot) = editor.scene.evaluate_prop_pose(prop.id, editor.time);
            let inv = rot.normalize().conjugate();
            let scale = prop.uniform_scale.max(0.15);
            let local_r = editor
                .scene
                .asset_by_id(prop.asset_id)
                .map(|a| a.local_bound_radius())
                .unwrap_or(0.6);
            let world_b_rad = local_r * scale + 0.28;

            let sel = if editor.selected_prop_id == Some(prop.id) {
                1u32
            } else {
                0u32
            };
            let scoped_flag = if scoped_asset_slot == Some(slot) {
                1u32
            } else {
                0u32
            };
            let (ring_r, orb_pos) = editor.selected_gizmo_geometry(prop);

            self.gpu_props_scratch.push(GpuSceneProp {
                pos_radius: [pos.x, pos.y, pos.z, world_b_rad],
                inv_rot: [inv.x, inv.y, inv.z, inv.w],
                scale_params: [scale, 1.0 / scale, ring_r, 0.0],
                gizmo_orb: [orb_pos.x, orb_pos.y, orb_pos.z, 0.0],
                meta: [slot as u32, prop.id, sel, scoped_flag],
            });

            if self.gpu_connectors_scratch.len() < MAX_GPU_CONNECTORS {
                if let Some((pivot, child_p)) =
                    editor.scene.evaluate_connector_segment(prop, editor.time)
                {
                    let kind_f = prop
                        .connector
                        .as_ref()
                        .map(|c| c.kind as u32 as f32)
                        .unwrap_or(0.0);
                    let is_sel = if editor.selected_prop_id == Some(prop.id)
                        || prop.connector.as_ref().map(|c| Some(c.parent_id))
                            == Some(editor.selected_prop_id)
                    {
                        1.0
                    } else {
                        0.0
                    };
                    self.gpu_connectors_scratch.push(GpuConnectorVisual {
                        start_kind: [pivot.x, pivot.y, pivot.z, kind_f],
                        end_sel: [child_p.x, child_p.y, child_p.z, is_sel],
                    });
                }
            }
        }

        let gpu_prop_count = self.gpu_props_scratch.len();
        if !self.gpu_props_scratch.is_empty() {
            self.queue.write_buffer(
                &self.props_buf,
                0,
                bytemuck::cast_slice(&self.gpu_props_scratch),
            );
        }
        let connector_count = self.gpu_connectors_scratch.len();
        if !self.gpu_connectors_scratch.is_empty() {
            self.queue.write_buffer(
                &self.connectors_buf,
                0,
                bytemuck::cast_slice(&self.gpu_connectors_scratch),
            );
        }

        // 3. Check dirty World Bricks & dirty Asset Atlas Bricks
        let dirty_world = if editor.dirty_bricks.has_dirty() {
            self.needs_world_fleck_refresh = true;
            editor.dirty_bricks.take_dirty_bricks()
        } else {
            Vec::new()
        };
        if !dirty_world.is_empty() {
            self.queue.write_buffer(
                &self.dirty_bricks_buf,
                0,
                bytemuck::cast_slice(&dirty_world),
            );
        }

        let dirty_assets = if editor.asset_dirty.has_dirty() {
            self.needs_world_fleck_refresh = true;
            editor.asset_dirty.take_dirty_bricks()
        } else {
            Vec::new()
        };
        if !dirty_assets.is_empty() {
            self.queue.write_buffer(
                &self.asset_dirty_bricks_buf,
                0,
                bytemuck::cast_slice(&dirty_assets),
            );
        }

        let dirty_fleck_slots = editor.asset_dirty.take_dirty_fleck_slots();
        let mut dirty_slot_mask = 0u32;
        for s in &dirty_fleck_slots {
            if *s < 24 {
                dirty_slot_mask |= 1u32 << *s;
            }
        }

        // Upload Compute Uniforms
        let wb_u = WorldBrickUniforms {
            volume_min_voxel: [VOLUME_MIN.x, VOLUME_MIN.y, VOLUME_MIN.z, VOXEL_SIZE],
            volume_dims_edits: [
                VOLUME_DIMS.x,
                VOLUME_DIMS.y,
                VOLUME_DIMS.z,
                edit_count as u32,
            ],
            brick_grid_dirty: [
                BRICK_GRID.x,
                BRICK_GRID.y,
                BRICK_GRID.z,
                dirty_world.len() as u32,
            ],
            counts: [0; 4],
        };
        self.queue
            .write_buffer(&self.world_brick_uniform_buf, 0, bytemuck::bytes_of(&wb_u));

        let ab_u = AssetBrickUniforms {
            local_min_voxel: [
                ASSET_LOCAL_MIN.x,
                ASSET_LOCAL_MIN.y,
                ASSET_LOCAL_MIN.z,
                ASSET_VOXEL_SIZE,
            ],
            counts: [
                dirty_assets.len() as u32,
                ASSET_BRICKS_PER_AXIS,
                ASSET_TILE_DIM,
                0,
            ],
        };
        self.queue
            .write_buffer(&self.asset_brick_uniform_buf, 0, bytemuck::bytes_of(&ab_u));

        let ext_u = FleckExtractUniforms {
            volume_min_voxel: [VOLUME_MIN.x, VOLUME_MIN.y, VOLUME_MIN.z, VOXEL_SIZE],
            volume_dims_edits: [
                VOLUME_DIMS.x,
                VOLUME_DIMS.y,
                VOLUME_DIMS.z,
                edit_count as u32,
            ],
            asset_atlas_info: [
                ASSET_ATLAS_DIMS.x,
                ASSET_ATLAS_DIMS.y,
                ASSET_ATLAS_DIMS.z,
                MAX_ASSET_LOCAL_FLECKS,
            ],
            counts: [
                gpu_prop_count as u32,
                MAX_SURFACE_FLECKS,
                self.frame_index,
                dirty_slot_mask,
            ],
            asset_min_voxel: [
                ASSET_LOCAL_MIN.x,
                ASSET_LOCAL_MIN.y,
                ASSET_LOCAL_MIN.z,
                ASSET_VOXEL_SIZE,
            ],
        };
        self.queue
            .write_buffer(&self.extract_uniform_buf, 0, bytemuck::bytes_of(&ext_u));

        // 4. Upload Frame / Post / UI Uniforms
        let aspect = (self.width as f32) / (self.height.max(1) as f32);
        let vp_mat = camera.view_proj(aspect);
        let inv_vp_mat = vp_mat.inverse();
        let eye = camera.eye_position();
        let fwd = camera.forward();

        let az = editor.scene.sun_azimuth;
        let el = editor.scene.sun_elevation;
        let sun_dir = Vec3::new(az.cos() * el.cos(), el.sin(), az.sin() * el.cos()).normalize();

        let mode_f = match editor.brush.mode {
            EditorMode::Sculpt => 0.0,
            EditorMode::Paint => 1.0,
            EditorMode::Animate => 2.0,
            EditorMode::Play => 3.0,
        };
        let manual_srgb = if self.output_format.is_srgb() {
            0.0
        } else {
            1.0
        };
        let brush_col = editor.brush.linear_color();

        let frame_u = FrameUniforms {
            view_proj: vp_mat.to_cols_array_2d(),
            inv_view_proj: inv_vp_mat.to_cols_array_2d(),
            cam_pos_time: [eye.x, eye.y, eye.z, editor.time],
            cam_fwd_focus: [fwd.x, fwd.y, fwd.z, camera.distance],
            sun_dof: [sun_dir.x, sun_dir.y, sun_dir.z, editor.scene.dof_amount],
            viewport_meta: [
                self.width as f32,
                self.height as f32,
                gpu_prop_count as f32,
                if editor.brush.mirror_x { 1.0 } else { 0.0 },
            ],
            imp_pos_size: [
                editor.imp.world_pos.x,
                editor.imp.world_pos.y,
                editor.imp.world_pos.z,
                editor.brush.size,
            ],
            imp_norm_mode: [
                editor.imp.surface_normal.x,
                editor.imp.surface_normal.y,
                editor.imp.surface_normal.z,
                mode_f,
            ],
            imp_shadow_hit: [
                editor.imp.ground_shadow_pos.x,
                editor.imp.ground_shadow_pos.y,
                editor.imp.ground_shadow_pos.z,
                if editor.imp.on_surface { 1.0 } else { 0.0 },
            ],
            brush_color_op: [
                brush_col[0],
                brush_col[1],
                brush_col[2],
                editor.brush.op as u32 as f32,
            ],
            puppet_pos_yaw: [
                editor.puppet.position.x,
                editor.puppet.position.y,
                editor.puppet.position.z,
                editor.puppet.facing_yaw,
            ],
            puppet_extra: [
                editor.puppet.squash,
                editor.puppet.walk_cycle,
                connector_count as f32,
                editor.selected_prop_id.unwrap_or(0) as f32,
            ],
            vol_min_voxel: [VOLUME_MIN.x, VOLUME_MIN.y, VOLUME_MIN.z, VOXEL_SIZE],
            vol_max_scoped: [
                VOLUME_MAX.x,
                VOLUME_MAX.y,
                VOLUME_MAX.z,
                editor.scoped_prop_id.unwrap_or(0) as f32,
            ],
            asset_atlas_info: [
                ASSET_ATLAS_DIMS.x as f32,
                ASSET_ATLAS_DIMS.y as f32,
                ASSET_ATLAS_DIMS.z as f32,
                ASSET_VOXEL_SIZE,
            ],
            asset_min_slot: [
                ASSET_LOCAL_MIN.x,
                ASSET_LOCAL_MIN.y,
                ASSET_LOCAL_MIN.z,
                scoped_asset_slot.map(|s| s as f32).unwrap_or(-1.0),
            ],
        };
        self.queue
            .write_buffer(&self.frame_uniform_buf, 0, bytemuck::bytes_of(&frame_u));

        let post_u = PostUniforms {
            focus_dof_srgb: [
                camera.distance,
                editor.scene.dof_amount,
                manual_srgb,
                editor.time,
            ],
            viewport_info: [
                1.0 / (self.width as f32),
                1.0 / (self.height as f32),
                self.width as f32,
                self.height as f32,
            ],
        };
        self.queue
            .write_buffer(&self.post_uniform_buf, 0, bytemuck::bytes_of(&post_u));

        let ui_quad_count = self.build_gpu_ui_quads(ui, manual_srgb);

        // ====================================================================
        // Record Command Encoder across Compute & Render Passes
        // ====================================================================
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("traum_frame_encoder"),
            });

        // Compute Pass 1A: Update Dirty World SDF Bricks
        if !dirty_world.is_empty() {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpass_world_bricks"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipe_update_bricks);
            cpass.set_bind_group(0, &self.bg_update_bricks, &[]);
            cpass.dispatch_workgroups(dirty_world.len() as u32, 1, 1);
        }

        // Compute Pass 1B: Update Dirty Reusable Sculpt Asset Bricks!
        if !dirty_assets.is_empty() {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpass_asset_bricks"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipe_update_asset_bricks);
            cpass.set_bind_group(0, &self.bg_update_asset_bricks, &[]);
            cpass.dispatch_workgroups(dirty_assets.len() as u32, 1, 1);
        }

        // Compute Pass 2: Refresh Dirty Asset Local Fleck Templates + Emit World & Animated Prop Flecks!
        if dirty_slot_mask != 0 || self.needs_world_fleck_refresh || gpu_prop_count > 0 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpass_extract_flecks"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &self.bg_extract_flecks, &[]);

            if dirty_slot_mask != 0 {
                cpass.set_pipeline(&self.pipe_reset_asset_flecks);
                cpass.dispatch_workgroups(1, 1, 1);

                cpass.set_pipeline(&self.pipe_extract_asset_flecks);
                cpass.dispatch_workgroups(8, 8, 8);
            }

            cpass.set_pipeline(&self.pipe_reset_flecks);
            cpass.dispatch_workgroups(1, 1, 1);

            cpass.set_pipeline(&self.pipe_extract_flecks);
            let gx = (VOLUME_DIMS.x / 2).div_ceil(4);
            let gy = (VOLUME_DIMS.y / 2).div_ceil(4);
            let gz = (VOLUME_DIMS.z / 2).div_ceil(4);
            cpass.dispatch_workgroups(gx, gy, gz);

            if gpu_prop_count > 0 {
                cpass.set_pipeline(&self.pipe_prop_flecks);
                let local_wg = MAX_ASSET_LOCAL_FLECKS.div_ceil(64);
                cpass.dispatch_workgroups(local_wg, gpu_prop_count as u32, 1);
            }

            cpass.set_pipeline(&self.pipe_finalize_flecks);
            cpass.dispatch_workgroups(1, 1, 1);

            self.needs_world_fleck_refresh = false;
        }

        // Render Pass 1: Watertight Painterly Clay Core + Custom Sculpted Props + Gizmos + Connectors
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rpass_clay_raymarch"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.pipe_raymarch);
            rpass.set_bind_group(0, &self.bg_raymarch, &[]);
            rpass.draw(0..3, 0..1);
        }

        // Render Pass 2: Instanced Painterly Fleck Splatting (`draw_indirect`)
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rpass_fleck_splat"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.pipe_fleck_splat);
            rpass.set_bind_group(0, &self.bg_fleck_splat, &[]);
            rpass.draw_indirect(&self.indirect_buf, 0);
        }

        // Render Pass 3: Tilt-Shift Bokeh DOF + ACES + Tactile Toy UI Overlay
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rpass_post_and_ui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.pipe_post_dof);
            rpass.set_bind_group(0, &self.bg_post_dof, &[]);
            rpass.draw(0..3, 0..1);

            if ui_quad_count > 0 {
                rpass.set_pipeline(&self.pipe_toy_ui);
                rpass.set_bind_group(0, &self.bg_toy_ui, &[]);
                rpass.draw(0..6, 0..ui_quad_count);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn build_gpu_ui_quads(&mut self, ui: &UiFrame, manual_srgb: f32) -> u32 {
        self.ui_quads_scratch.clear();
        let s = ui.ui_scale.max(0.72);
        let vp = Vec2::new(self.width as f32, self.height as f32);

        // 1. Render Panels, Glossy Buttons, Icons, & Crisp SDF Labels
        for r in &ui.rects {
            if self.ui_quads_scratch.len() + 8 >= MAX_UI_QUADS {
                break;
            }

            if r.cast_shadow && r.fill_srgb[3] > 0.05 {
                let sh_pad = 5.0 * s;
                let sh_drop = 3.5 * s;
                self.ui_quads_scratch.push(GpuUiQuad {
                    rect_px: [
                        r.min.x - sh_pad,
                        r.min.y - sh_pad * 0.5 + sh_drop,
                        r.max.x + sh_pad,
                        r.max.y + sh_pad + sh_drop,
                    ],
                    uv_rect: [-1.0; 4],
                    fill_color: [0.02, 0.03, 0.06, 0.52],
                    border_color: [0.0; 4],
                    params: [r.corner_radius + sh_pad, 7.0 * s, 0.0, 2.0],
                });
            }

            if r.fill_srgb[3] > 0.01 || r.border_srgb[3] > 0.01 {
                let variant = match r.style {
                    UiRectStyle::Button => 1.0,
                    UiRectStyle::Panel | UiRectStyle::FlatAccent => 0.0,
                };
                self.ui_quads_scratch.push(GpuUiQuad {
                    rect_px: [r.min.x, r.min.y, r.max.x, r.max.y],
                    uv_rect: [-1.0; 4],
                    fill_color: r.fill_srgb,
                    border_color: r.border_srgb,
                    params: [r.corner_radius, r.border_width, 0.0, variant],
                });
            }

            let rect_h = r.max.y - r.min.y;
            let mut text_left_pad = 11.0 * s;

            if let Some(icon_id) = r.icon {
                let icon_sz = (rect_h * 0.42).clamp(11.0, 22.0 * s);
                let ix0 = if r.left_align_text {
                    r.min.x + 11.0 * s
                } else {
                    let tw = font::measure_text_width(&r.label, r.font_px);
                    let total_w = icon_sz + 6.5 * s + tw;
                    ((r.min.x + r.max.x - total_w) * 0.5).max(r.min.x + 8.0 * s)
                };
                let iy0 = (r.min.y + r.max.y - icon_sz) * 0.5;
                self.ui_quads_scratch.push(GpuUiQuad {
                    rect_px: [ix0, iy0, ix0 + icon_sz, iy0 + icon_sz],
                    uv_rect: [0.0; 4],
                    fill_color: r.text_srgb,
                    border_color: [0.03, 0.04, 0.08, 0.72],
                    params: [icon_id as f32, 0.0, 3.0, 0.0],
                });
                text_left_pad = (ix0 - r.min.x) + icon_sz + 6.5 * s;
            }

            if !r.label.is_empty() {
                let font_px = r.font_px.max(9.0);
                let cy = (r.min.y + r.max.y - font_px) * 0.5;
                let cx = if r.left_align_text || r.icon.is_some() {
                    r.min.x + text_left_pad
                } else {
                    let tw = font::measure_text_width(&r.label, font_px);
                    (r.min.x + r.max.x - tw) * 0.5
                };
                push_text_quads(
                    &mut self.ui_quads_scratch,
                    &r.label,
                    Vec2::new(cx, cy),
                    font_px,
                    r.text_srgb,
                );
            }

            if !r.right_label.is_empty() {
                let font_px = r.font_px.max(10.0);
                let rw = font::measure_text_width(&r.right_label, font_px);
                let cy = (r.min.y + r.max.y - font_px) * 0.5;
                let rx = (r.max.x - rw - 11.0 * s).max(r.min.x + 8.0 * s);
                push_text_quads(
                    &mut self.ui_quads_scratch,
                    &r.right_label,
                    Vec2::new(rx, cy),
                    font_px,
                    r.text_srgb,
                );
            }
        }

        // 2. Optional Quick Controls & Workflow Guide Card (`Guide` button)
        if ui.help_open {
            self.push_help_guide_card(vp, s);
        }

        // 3. Status Toast Banner above bottom color palette
        if !ui.banner_status.is_empty() && self.ui_quads_scratch.len() + 16 < MAX_UI_QUADS {
            let font_px = 12.0 * s;
            let text_w = font::measure_text_width(&ui.banner_status, font_px);
            let pad_x = 20.0 * s;
            let box_h = 36.0 * s;
            let box_w = (text_w + pad_x * 2.0).min(vp.x - 32.0 * s);
            let x0 = (vp.x - box_w) * 0.5;
            let y0 = (vp.y - 108.0 * s).max(70.0 * s);

            self.ui_quads_scratch.push(GpuUiQuad {
                rect_px: [x0 - 4.0 * s, y0 + 2.0 * s, x0 + box_w + 4.0 * s, y0 + box_h + 7.0 * s],
                uv_rect: [-1.0; 4],
                fill_color: [0.02, 0.03, 0.06, 0.55],
                border_color: [0.0; 4],
                params: [20.0 * s, 7.0 * s, 0.0, 2.0],
            });
            self.ui_quads_scratch.push(GpuUiQuad {
                rect_px: [x0, y0, x0 + box_w, y0 + box_h],
                uv_rect: [-1.0; 4],
                fill_color: [0.09, 0.11, 0.17, 0.95],
                border_color: border_srgb_or_gold(),
                params: [18.0 * s, 1.6 * s, 0.0, 0.0],
            });

            let clipped = clip_text_to_width(&ui.banner_status, font_px, box_w - 22.0 * s);
            let cw = font::measure_text_width(&clipped, font_px);
            push_text_quads(
                &mut self.ui_quads_scratch,
                &clipped,
                Vec2::new((vp.x - cw) * 0.5, y0 + (box_h - font_px) * 0.5),
                font_px,
                [0.99, 0.96, 0.88, 1.0],
            );
        }

        // 4. Bouncy 8-Slice Radial Pie Wheel
        let mut radial_params = [0.0_f32; 4];
        if let Some(rad) = &ui.radial {
            let hov = rad.hovered_slice.map(|sl| sl as f32).unwrap_or(-1.0);
            radial_params = [rad.center_px.x, rad.center_px.y, rad.outer_radius, hov];

            let ext = rad.outer_radius + 18.0 * s;
            if self.ui_quads_scratch.len() < MAX_UI_QUADS {
                self.ui_quads_scratch.push(GpuUiQuad {
                    rect_px: [
                        rad.center_px.x - ext,
                        rad.center_px.y - ext,
                        rad.center_px.x + ext,
                        rad.center_px.y + ext,
                    ],
                    uv_rect: [-1.0; 4],
                    fill_color: [1.0; 4],
                    border_color: [1.0; 4],
                    params: [0.0, 0.0, 2.0, 0.0],
                });
            }

            let inner_r = 38.0 * s;
            let font_px = 11.4 * s;
            for (idx, &lbl) in rad.labels.iter().enumerate() {
                let angle = -std::f32::consts::FRAC_PI_2
                    + (idx as f32) * (std::f32::consts::TAU / 8.0);
                let label_r = (inner_r + rad.outer_radius) * 0.53;
                let pos = rad.center_px + Vec2::new(angle.cos(), angle.sin()) * label_r;
                let tw = font::measure_text_width(lbl, font_px);
                let is_hov = rad.hovered_slice == Some(idx);
                push_text_quads(
                    &mut self.ui_quads_scratch,
                    lbl,
                    Vec2::new(pos.x - tw * 0.5, pos.y - font_px * 0.5),
                    font_px,
                    if is_hov {
                        [1.0, 1.0, 1.0, 1.0]
                    } else {
                        [0.95, 0.96, 1.0, 1.0]
                    },
                );
            }
        }

        // 5. Rich Floating Tooltip Card (always drawn on top of everything!)
        if let Some(tip) = &ui.tooltip {
            self.push_tooltip_card(tip, vp, s);
        }

        let ui_u = UiUniforms {
            viewport_srgb_scale: [vp.x, vp.y, manual_srgb, s],
            radial_params,
        };
        self.queue
            .write_buffer(&self.ui_uniform_buf, 0, bytemuck::bytes_of(&ui_u));

        if !self.ui_quads_scratch.is_empty() {
            self.queue.write_buffer(
                &self.ui_quads_buf,
                0,
                bytemuck::cast_slice(&self.ui_quads_scratch),
            );
        }

        self.ui_quads_scratch.len() as u32
    }

    fn push_tooltip_card(&mut self, tip: &TooltipCard, vp: Vec2, s: f32) {
        let title_px = 12.6 * s;
        let key_px = 10.4 * s;
        let body_px = 11.4 * s;
        let pad = 12.0 * s;
        let max_body_w = (300.0 * s).min(vp.x - 40.0 * s).max(160.0);

        let lines = wrap_text_lines(&tip.description, body_px, max_body_w);
        let key_badge_w = if tip.shortcut.is_empty() {
            0.0
        } else {
            font::measure_text_width(&tip.shortcut, key_px) + 18.0 * s
        };
        let header_w = font::measure_text_width(&tip.title, title_px) + key_badge_w + 14.0 * s;
        let body_max_line = lines
            .iter()
            .map(|l| font::measure_text_width(l, body_px))
            .fold(0.0_f32, f32::max);
        let card_w = header_w.max(body_max_line) + pad * 2.0;
        let line_step = body_px * 1.36;
        let card_h = pad * 2.0 + title_px + 10.0 * s + (lines.len() as f32) * line_step;

        // Smart placement beside hovered button so we never cover the control or clip offscreen!
        let anchor_cx = (tip.anchor_min.x + tip.anchor_max.x) * 0.5;
        let anchor_cy = (tip.anchor_min.y + tip.anchor_max.y) * 0.5;
        let gap = 12.0 * s;

        let (mut x0, mut y0) = if tip.anchor_min.y > vp.y - 120.0 * s {
            // Bottom palette -> place above swatch
            (anchor_cx - card_w * 0.5, tip.anchor_min.y - card_h - gap)
        } else if tip.anchor_max.y < 105.0 * s {
            // Top bar -> place below button
            (anchor_cx - card_w * 0.5, tip.anchor_max.y + gap)
        } else if anchor_cx < vp.x * 0.36 {
            // Left dock -> place to the right of button
            (tip.anchor_max.x + gap, anchor_cy - card_h * 0.35)
        } else if anchor_cx > vp.x * 0.64 {
            // Right sliders -> place to the left of slider
            (tip.anchor_min.x - card_w - gap, anchor_cy - card_h * 0.35)
        } else {
            // Center (Radial Wheel) -> place cleanly below
            (anchor_cx - card_w * 0.5, tip.anchor_max.y + gap)
        };

        x0 = x0.clamp(10.0 * s, (vp.x - card_w - 10.0 * s).max(10.0 * s));
        y0 = y0.clamp(10.0 * s, (vp.y - card_h - 10.0 * s).max(10.0 * s));

        // Soft drop shadow
        self.ui_quads_scratch.push(GpuUiQuad {
            rect_px: [
                x0 - 6.0 * s,
                y0 - 2.0 * s,
                x0 + card_w + 6.0 * s,
                y0 + card_h + 10.0 * s,
            ],
            uv_rect: [-1.0; 4],
            fill_color: [0.01, 0.02, 0.05, 0.68],
            border_color: [0.0; 4],
            params: [18.0 * s, 9.0 * s, 0.0, 2.0],
        });

        // Deep high-contrast card background with warm gold rim
        self.ui_quads_scratch.push(GpuUiQuad {
            rect_px: [x0, y0, x0 + card_w, y0 + card_h],
            uv_rect: [-1.0; 4],
            fill_color: [0.07, 0.09, 0.14, 0.97],
            border_color: [0.96, 0.74, 0.32, 0.95],
            params: [14.0 * s, 1.7 * s, 0.0, 0.0],
        });

        // Title in warm gold
        push_text_quads(
            &mut self.ui_quads_scratch,
            &tip.title,
            Vec2::new(x0 + pad, y0 + pad),
            title_px,
            [1.0, 0.84, 0.42, 1.0],
        );

        // Shortcut pill badge on top-right
        if !tip.shortcut.is_empty() {
            let bx1 = x0 + card_w - pad;
            let bx0 = bx1 - key_badge_w;
            let by0 = y0 + pad - 2.0 * s;
            let bh = title_px + 4.0 * s;
            self.ui_quads_scratch.push(GpuUiQuad {
                rect_px: [bx0, by0, bx1, by0 + bh],
                uv_rect: [-1.0; 4],
                fill_color: [0.18, 0.22, 0.32, 0.96],
                border_color: [0.52, 0.60, 0.78, 0.70],
                params: [8.0 * s, 1.1 * s, 0.0, 0.0],
            });
            let kw = font::measure_text_width(&tip.shortcut, key_px);
            push_text_quads(
                &mut self.ui_quads_scratch,
                &tip.shortcut,
                Vec2::new((bx0 + bx1 - kw) * 0.5, by0 + (bh - key_px) * 0.5),
                key_px,
                [0.86, 0.93, 1.0, 1.0],
            );
        }

        // Subtle divider line
        let div_y = y0 + pad + title_px + 5.0 * s;
        self.ui_quads_scratch.push(GpuUiQuad {
            rect_px: [x0 + pad, div_y, x0 + card_w - pad, div_y + 1.2 * s],
            uv_rect: [-1.0; 4],
            fill_color: [0.35, 0.40, 0.54, 0.55],
            border_color: [0.0; 4],
            params: [1.0, 0.0, 0.0, 0.0],
        });

        // Wrapped plain-English description lines
        let mut ly = div_y + 6.0 * s;
        for line in &lines {
            push_text_quads(
                &mut self.ui_quads_scratch,
                line,
                Vec2::new(x0 + pad, ly),
                body_px,
                [0.94, 0.96, 0.99, 1.0],
            );
            ly += line_step;
        }
    }

    fn push_help_guide_card(&mut self, vp: Vec2, s: f32) {
        let rows = [
            ("Camera Orbit / Pan:", "Right-Drag / Middle-Drag / 2-Finger"),
            ("Zoom In & Out:", "Scroll Wheel / 2-Finger Pinch"),
            ("Sculpt / Quick-Carve:", "Left-Drag / Alt + Left-Drag"),
            ("Radial Quick Wheel:", "Hold Space (or Gamepad LB)"),
            ("Edit Inside a Toy:", "Hover Toy + Enter (or Edit Inside Toy)"),
            ("Animate Live Path:", "Animate Tab -> Record -> Drag Toy!"),
            ("Explore in Play Mode:", "Tab -> WASD + Space to Jump!"),
        ];
        let header = "Traum Engine — Quick Controls Guide (Click Guide to Close)";
        let title_px = 12.8 * s;
        let row_px = 10.6 * s;
        let pad = 16.0 * s;
        let row_step = row_px * 1.85;

        // Compute exact required card width so left & right columns NEVER overlap!
        let max_row_w = rows
            .iter()
            .map(|(k, v)| {
                font::measure_text_width(k, row_px)
                    + 28.0 * s
                    + font::measure_text_width(v, row_px)
            })
            .fold(0.0_f32, f32::max);
        let header_w = font::measure_text_width(header, title_px);
        let card_w = (max_row_w.max(header_w) + pad * 2.0)
            .max(460.0 * s)
            .min(vp.x - 32.0 * s);

        let card_h = pad * 2.0 + title_px + 18.0 * s + (rows.len() as f32) * row_step;
        let x0 = (vp.x - card_w) * 0.5;
        let y0 = ((vp.y - card_h) * 0.46).max(74.0 * s);

        self.ui_quads_scratch.push(GpuUiQuad {
            rect_px: [x0 - 8.0 * s, y0 - 2.0 * s, x0 + card_w + 8.0 * s, y0 + card_h + 12.0 * s],
            uv_rect: [-1.0; 4],
            fill_color: [0.01, 0.02, 0.05, 0.72],
            border_color: [0.0; 4],
            params: [20.0 * s, 10.0 * s, 0.0, 2.0],
        });
        self.ui_quads_scratch.push(GpuUiQuad {
            rect_px: [x0, y0, x0 + card_w, y0 + card_h],
            uv_rect: [-1.0; 4],
            fill_color: [0.08, 0.10, 0.16, 0.96],
            border_color: [0.48, 0.76, 0.98, 0.92],
            params: [18.0 * s, 1.8 * s, 0.0, 0.0],
        });

        let clipped_header = clip_text_to_width(header, title_px, card_w - pad * 2.0);
        let hw = font::measure_text_width(&clipped_header, title_px);
        push_text_quads(
            &mut self.ui_quads_scratch,
            &clipped_header,
            Vec2::new((vp.x - hw) * 0.5, y0 + pad),
            title_px,
            [0.58, 0.86, 1.0, 1.0],
        );

        let div_y = y0 + pad + title_px + 7.0 * s;
        self.ui_quads_scratch.push(GpuUiQuad {
            rect_px: [x0 + pad, div_y, x0 + card_w - pad, div_y + 1.2 * s],
            uv_rect: [-1.0; 4],
            fill_color: [0.35, 0.42, 0.58, 0.55],
            border_color: [0.0; 4],
            params: [1.0, 0.0, 0.0, 0.0],
        });

        let mut ry = div_y + 9.0 * s;
        for (k, v) in rows {
            push_text_quads(
                &mut self.ui_quads_scratch,
                k,
                Vec2::new(x0 + pad, ry),
                row_px,
                [1.0, 0.84, 0.44, 1.0],
            );
            let vw = font::measure_text_width(v, row_px);
            push_text_quads(
                &mut self.ui_quads_scratch,
                v,
                Vec2::new(x0 + card_w - pad - vw, ry),
                row_px,
                [0.95, 0.97, 1.0, 1.0],
            );
            ry += row_step;
        }
    }

    /// End-to-end headless GPU verification: compiles all WGSL shaders, evaluates World + Asset Atlas
    /// 3D SDF bricks, tests Scope-In custom prop sculpting + Live Cloning + Multi-Part Connectors, and verifies pixels!
    pub fn run_headless_self_test() -> Result<String, String> {
        let w = 640u32;
        let h = 360u32;
        let (mut renderer, adapter_info) = Self::new_headless(w, h)?;

        let out_tex = renderer.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("self_test_target"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: renderer.output_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut editor = TraumEditor::new();
        let mut camera = CameraRig::default();

        // Frame 1: Initial Diorama Island + 4 Custom Multi-Stroke Sculpt Assets + Connected Critter
        let ui1 = editor.update(
            0.016,
            &FrameInput {
                imp_px: Vec2::new(320.0, 180.0),
                viewport: Vec2::new(w as f32, h as f32),
                ..Default::default()
            },
            ActiveInputDevice::MouseKeyboard,
            &mut camera,
        );
        renderer.encode_and_submit_frame(&mut editor, &camera, &ui1, &out_view);

        // Frame 2: Scope Into Critter, stamp a local sculpt stroke, Live-Clone it, & render!
        editor.toggle_scope_in_out(&mut camera);
        let ui2 = editor.update(
            0.016,
            &FrameInput {
                imp_px: Vec2::new(320.0, 180.0),
                viewport: Vec2::new(w as f32, h as f32),
                primary_down: true,
                primary_just_pressed: true,
                pressure: 1.0,
                ..Default::default()
            },
            ActiveInputDevice::MouseKeyboard,
            &mut camera,
        );
        renderer.encode_and_submit_frame(&mut editor, &camera, &ui2, &out_view);

        editor.clone_selected_prop();
        editor.show_help_guide = true;
        let ui3 = editor.update(
            0.016,
            &FrameInput {
                // Hover over top-left Undo button to also exercise TooltipCard + Help Guide GPU rendering!
                imp_px: Vec2::new(36.0, 24.0),
                viewport: Vec2::new(w as f32, h as f32),
                ui_scale: 1.0,
                primary_just_released: true,
                ..Default::default()
            },
            ActiveInputDevice::MouseKeyboard,
            &mut camera,
        );
        renderer.encode_and_submit_frame(&mut editor, &camera, &ui3, &out_view);

        let read_indirect = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_indirect"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bytes_per_row = (w * 4).div_ceil(256) * 256;
        let read_pixels = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_pixels"),
            size: (bytes_per_row * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut copy_enc =
            renderer
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("readback_encoder"),
                });
        copy_enc.copy_buffer_to_buffer(&renderer.indirect_buf, 0, &read_indirect, 0, 16);
        copy_enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &out_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &read_pixels,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        renderer.queue.submit(std::iter::once(copy_enc.finish()));

        let ind_slice = read_indirect.slice(..);
        let pix_slice = read_pixels.slice(..);
        ind_slice.map_async(wgpu::MapMode::Read, |_| {});
        pix_slice.map_async(wgpu::MapMode::Read, |_| {});
        renderer.device.poll(wgpu::Maintain::Wait);

        let ind_bytes = ind_slice.get_mapped_range();
        let ind_u32: &[u32] = bytemuck::cast_slice(&ind_bytes);
        let fleck_instances = ind_u32[1];

        let pix_bytes = pix_slice.get_mapped_range();
        let center_offset = ((h / 2) * bytes_per_row + (w / 2) * 4) as usize;
        let center_rgba = [
            pix_bytes[center_offset],
            pix_bytes[center_offset + 1],
            pix_bytes[center_offset + 2],
            pix_bytes[center_offset + 3],
        ];

        if fleck_instances < 1000 {
            return Err(format!(
                "Expected >= 1000 extracted world + custom asset flecks, got {fleck_instances}"
            ));
        }
        if center_rgba[0] == 0 && center_rgba[1] == 0 && center_rgba[2] == 0 {
            return Err("Rendered center pixel was completely black!".into());
        }

        Ok(format!(
            "SELF-TEST PASSED!\n  GPU Adapter  : {} ({:?})\n  World Volume : {}x{}x{} voxels ({}x{}x{} bricks)\n  Asset Atlas  : {}x{}x{} voxels ({} custom 64^3 sculpt slots)\n  Scene Props  : {} instances ({} custom assets)\n  Live Flecks  : {} painterly brushstroke instances\n  Center RGBA  : {:?}",
            adapter_info.name,
            adapter_info.backend,
            VOLUME_DIMS.x,
            VOLUME_DIMS.y,
            VOLUME_DIMS.z,
            BRICK_GRID.x,
            BRICK_GRID.y,
            BRICK_GRID.z,
            ASSET_ATLAS_DIMS.x,
            ASSET_ATLAS_DIMS.y,
            ASSET_ATLAS_DIMS.z,
            MAX_SCULPT_ASSETS,
            editor.scene.props.len(),
            editor.scene.assets.len(),
            fleck_instances,
            center_rgba,
        ))
    }

    fn create_frame_attachments(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
    ) {
        let extent = wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        };
        let hdr_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr_scene_tex"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let hdr_view = hdr_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene_depth_tex"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        (hdr_tex, hdr_view, depth_tex, depth_view)
    }

    fn create_post_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform_buf: &wgpu::Buffer,
        hdr_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_post_dof"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }
}

fn border_srgb_or_gold() -> [f32; 4] {
    [0.95, 0.72, 0.34, 0.75]
}

fn push_text_quads(
    out: &mut Vec<GpuUiQuad>,
    text: &str,
    origin: Vec2,
    font_px: f32,
    color_srgb: [f32; 4],
) {
    let mut pen_x = origin.x;
    // Exact 32x32 cell half-extent so 24-texel (8-row) glyph body equals `font_px` tall 1-to-1!
    let half = font_px * font::QUAD_HALF_FACTOR;
    let luma = color_srgb[0] * 0.299 + color_srgb[1] * 0.587 + color_srgb[2] * 0.114;
    let halo = if luma > 0.35 {
        [0.03, 0.04, 0.08, 0.88]
    } else {
        [1.0, 1.0, 1.0, 0.35]
    };

    for ch in text.chars() {
        if out.len() >= MAX_UI_QUADS {
            break;
        }
        if ch != ' ' {
            let (uv0, uv1) = font::glyph_uv(ch);
            let ink_w = font::glyph_ink_width_factor(ch) * font_px;
            let cx = pen_x + ink_w * 0.5;
            let cy = origin.y + font_px * 0.5;
            out.push(GpuUiQuad {
                rect_px: [cx - half, cy - half, cx + half, cy + half],
                uv_rect: [uv0[0], uv0[1], uv1[0], uv1[1]],
                fill_color: color_srgb,
                border_color: halo,
                params: [0.0, 0.0, 1.0, 0.0],
            });
        }
        pen_x += font::glyph_advance_factor(ch) * font_px;
    }
}

fn wrap_text_lines(text: &str, font_px: f32, max_width: f32) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let candidate = if current.is_empty() {
            word.to_string()
        } else {
            format!("{current} {word}")
        };
        if font::measure_text_width(&candidate, font_px) <= max_width || current.is_empty() {
            current = candidate;
        } else {
            lines.push(current);
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

fn clip_text_to_width(text: &str, font_px: f32, max_width: f32) -> String {
    let mut out = String::new();
    let mut w = 0.0_f32;
    for ch in text.chars() {
        let adv = font::glyph_advance_factor(ch) * font_px;
        if w + adv > max_width {
            break;
        }
        out.push(ch);
        w += adv;
    }
    out
}

// ============================================================================
// Compact BindGroupLayout Entry Helpers
// ============================================================================

fn bgl_uniform(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage(
    binding: u32,
    visibility: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_tex_2d(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn bgl_tex_3d(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    }
}

fn bgl_storage_tex_3d(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D3,
        },
        count: None,
    }
}

fn bgl_sampler(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}
