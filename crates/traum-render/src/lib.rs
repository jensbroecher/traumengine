//! Dreams-inspired `wgpu` rendering pipeline for Traum Engine:
//! 1. Compute Pass 1 (`sdf_compute.wgsl`): Evaluates dirty `8x8x8` SDF bricks from CSG Edit Lists.
//! 2. Compute Pass 2 (`extract_flecks.wgsl`): Extracts iso-surface Painterly Flecks + indirect draw args.
//! 3. Render Pass 1 (`scene_raymarch.wgsl`): Watertight impasto clay core, animated props, buddy, & Imp guides.
//! 4. Render Pass 2 (`fleck_splat.wgsl`): Instanced painterly brushstroke splats with Living Flecks motion.
//! 5. Render Pass 3 (`post_dof.wgsl` + `toy_ui.wgsl`): Tilt-shift Bokeh DOF + ACES + Tactile Toy UI & Radial Wheel.

mod font;

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use std::sync::Arc;
use traum_core::EditorMode;
use traum_editor::{TraumEditor, UiFrame};
use traum_input::{ActiveInputDevice, CameraRig, FrameInput};
use traum_sdf::{
    GpuFleck, GpuSceneProp, GpuSculptEdit, BRICK_GRID, MAX_GPU_EDITS, MAX_GPU_PROPS,
    MAX_SURFACE_FLECKS, VOLUME_DIMS, VOLUME_MAX, VOLUME_MIN, VOXEL_SIZE,
};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::window::Window;

const MAX_UI_QUADS: usize = 2048;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ComputeUniforms {
    volume_min_voxel: [f32; 4],
    volume_dims_edits: [u32; 4],
    brick_grid_dirty: [u32; 4],
    counts: [u32; 4],
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
    vol_max_pad: [f32; 4],
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
    viewport_srgb_radial: [f32; 4],
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

    // 3D SDF Volume & Buffers
    comp_uniform_buf: wgpu::Buffer,
    edits_buf: wgpu::Buffer,
    dirty_bricks_buf: wgpu::Buffer,
    props_buf: wgpu::Buffer,
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
    gpu_props_scratch: Vec<GpuSceneProp>,
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
        let vol_extent = wgpu::Extent3d {
            width: VOLUME_DIMS.x,
            height: VOLUME_DIMS.y,
            depth_or_array_layers: VOLUME_DIMS.z,
        };

        let make_vol_tex = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: vol_extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };

        let sdf_vol_tex = make_vol_tex("sdf_brick_volume");
        let sdf_vol_view = sdf_vol_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let color_vol_tex = make_vol_tex("color_brick_volume");
        let color_vol_view = color_vol_tex.create_view(&wgpu::TextureViewDescriptor::default());

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
            label: Some("ui_nearest_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Font atlas texture (`R8Unorm`)
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

        // GPU Buffers
        let comp_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("comp_uniforms"),
            size: std::mem::size_of::<ComputeUniforms>() as u64,
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

        let props_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_props_buf"),
            size: (MAX_GPU_PROPS * std::mem::size_of::<GpuSceneProp>()) as u64,
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

        // 1. Compute Pass 1 Layout & Pipeline (`update_dirty_bricks`)
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
        let pipe_update_bricks =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
                    resource: comp_uniform_buf.as_entire_binding(),
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

        // 2. Compute Pass 2 Layout & Pipelines (`extract_flecks.wgsl`)
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
                    resource: comp_uniform_buf.as_entire_binding(),
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
            ],
        });

        // 3. Render Pass 1: Clay Raymarch
        let bgl_raymarch = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_raymarch"),
            entries: &[
                bgl_uniform(0, wgpu::ShaderStages::FRAGMENT),
                bgl_tex_3d(1, wgpu::ShaderStages::FRAGMENT),
                bgl_tex_3d(2, wgpu::ShaderStages::FRAGMENT),
                bgl_sampler(3, wgpu::ShaderStages::FRAGMENT),
                bgl_storage(4, wgpu::ShaderStages::FRAGMENT, true),
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
        let bg_post_dof =
            Self::create_post_bind_group(&device, &bgl_post_dof, &post_uniform_buf, &hdr_view, &linear_sampler);

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
            comp_uniform_buf,
            edits_buf,
            dirty_bricks_buf,
            props_buf,
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
            gpu_props_scratch: Vec::with_capacity(32),
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

        // 1. Sync Sculpt Edits if changed
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

        // 2. Upload animated Toy Props for current frame time
        let prop_count = editor.scene.props.len().min(MAX_GPU_PROPS);
        self.gpu_props_scratch.clear();
        for prop in editor.scene.props.iter().take(prop_count) {
            let (pos, rot) = prop.evaluate_pose(editor.time);
            let inv = rot.normalize().conjugate();
            let b_rad = prop.scale.max_element() * 1.5 + 0.25;
            let sel = if editor.selected_prop_id == Some(prop.id) {
                1u32
            } else {
                0u32
            };
            self.gpu_props_scratch.push(GpuSceneProp {
                pos_radius: [pos.x, pos.y, pos.z, b_rad],
                inv_rot: [inv.x, inv.y, inv.z, inv.w],
                scale_loose: [
                    prop.scale.x.max(0.05),
                    prop.scale.y.max(0.05),
                    prop.scale.z.max(0.05),
                    prop.looseness,
                ],
                color_style: [prop.color[0], prop.color[1], prop.color[2], 0.0],
                meta: [prop.shape as u32, prop.fleck_style as u32, prop.id, sel],
            });
        }
        if !self.gpu_props_scratch.is_empty() {
            self.queue.write_buffer(
                &self.props_buf,
                0,
                bytemuck::cast_slice(&self.gpu_props_scratch),
            );
        }

        // 3. Check dirty SDF bricks
        let dirty_list = if editor.dirty_bricks.has_dirty() {
            self.needs_world_fleck_refresh = true;
            editor.dirty_bricks.take_dirty_bricks()
        } else {
            Vec::new()
        };
        if !dirty_list.is_empty() {
            self.queue
                .write_buffer(&self.dirty_bricks_buf, 0, bytemuck::cast_slice(&dirty_list));
        }

        let comp_u = ComputeUniforms {
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
                dirty_list.len() as u32,
            ],
            counts: [
                prop_count as u32,
                MAX_SURFACE_FLECKS,
                self.frame_index,
                0,
            ],
        };
        self.queue
            .write_buffer(&self.comp_uniform_buf, 0, bytemuck::bytes_of(&comp_u));

        // 4. Upload Frame / Post / UI uniforms
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
                prop_count as f32,
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
                manual_srgb,
                editor.selected_prop_id.unwrap_or(0) as f32,
            ],
            vol_min_voxel: [VOLUME_MIN.x, VOLUME_MIN.y, VOLUME_MIN.z, VOXEL_SIZE],
            vol_max_pad: [VOLUME_MAX.x, VOLUME_MAX.y, VOLUME_MAX.z, 0.0],
        };
        self.queue
            .write_buffer(&self.frame_uniform_buf, 0, bytemuck::bytes_of(&frame_u));

        let post_u = PostUniforms {
            focus_dof_srgb: [camera.distance, editor.scene.dof_amount, manual_srgb, editor.time],
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

        // Compute Pass 1: Update Dirty 8x8x8 SDF Bricks (only when strokes/edits occurred!)
        if !dirty_list.is_empty() {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpass_dirty_bricks"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipe_update_bricks);
            cpass.set_bind_group(0, &self.bg_update_bricks, &[]);
            cpass.dispatch_workgroups(dirty_list.len() as u32, 1, 1);
        }

        // Compute Pass 2: Extract Surface Flecks + Moving Prop Flecks!
        if self.needs_world_fleck_refresh || prop_count > 0 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpass_extract_flecks"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &self.bg_extract_flecks, &[]);

            cpass.set_pipeline(&self.pipe_reset_flecks);
            cpass.dispatch_workgroups(1, 1, 1);

            cpass.set_pipeline(&self.pipe_extract_flecks);
            let gx = (VOLUME_DIMS.x / 2).div_ceil(4);
            let gy = (VOLUME_DIMS.y / 2).div_ceil(4);
            let gz = (VOLUME_DIMS.z / 2).div_ceil(4);
            cpass.dispatch_workgroups(gx, gy, gz);

            if prop_count > 0 {
                cpass.set_pipeline(&self.pipe_prop_flecks);
                cpass.dispatch_workgroups(prop_count as u32, 1, 1);
            }

            cpass.set_pipeline(&self.pipe_finalize_flecks);
            cpass.dispatch_workgroups(1, 1, 1);

            self.needs_world_fleck_refresh = false;
        }

        // Render Pass 1: Watertight Painterly Clay Core + Props + Playable Buddy + Imp Ring
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

        // 1. Pills, Sliders & Palette Swatches + Centered Text Labels
        for r in &ui.rects {
            if self.ui_quads_scratch.len() >= MAX_UI_QUADS {
                break;
            }
            self.ui_quads_scratch.push(GpuUiQuad {
                rect_px: [r.min.x, r.min.y, r.max.x, r.max.y],
                uv_rect: [-1.0; 4],
                fill_color: r.fill_srgb,
                border_color: r.border_srgb,
                params: [r.corner_radius, 1.6, 0.0, 0.0],
            });

            if !r.label.is_empty() {
                let scale = 1.55_f32;
                let char_w = 8.0 * scale;
                let char_h = 8.0 * scale;
                let text_w = (r.label.chars().count() as f32) * char_w;
                let cx = (r.min.x + r.max.x - text_w) * 0.5;
                let cy = (r.min.y + r.max.y - char_h) * 0.5;
                push_text_quads(
                    &mut self.ui_quads_scratch,
                    &r.label,
                    Vec2::new(cx, cy),
                    scale,
                    r.text_srgb,
                );
            }
        }

        // 2. Status Toast Banner above bottom color palette
        if !ui.banner_status.is_empty() && self.ui_quads_scratch.len() + 4 < MAX_UI_QUADS {
            let scale = 1.5_f32;
            let char_w = 8.0 * scale;
            let text_w = (ui.banner_status.chars().count() as f32) * char_w;
            let pad_x = 18.0;
            let box_w = (text_w + pad_x * 2.0).min(self.width as f32 - 28.0);
            let x0 = (self.width as f32 - box_w) * 0.5;
            let y0 = (self.height as f32 - 96.0).max(60.0);
            self.ui_quads_scratch.push(GpuUiQuad {
                rect_px: [x0, y0, x0 + box_w, y0 + 32.0],
                uv_rect: [-1.0; 4],
                fill_color: [0.10, 0.11, 0.16, 0.90],
                border_color: border_srgb_or_gold(),
                params: [16.0, 1.5, 0.0, 0.0],
            });
            let tx = (self.width as f32 - text_w).max(x0 + 8.0) * 0.5 + x0 * 0.0;
            let centered_tx = (self.width as f32 - text_w.min(box_w - 16.0)) * 0.5;
            let _ = tx;
            let max_chars = ((box_w - 16.0) / char_w).floor().max(1.0) as usize;
            let clipped: String = ui.banner_status.chars().take(max_chars).collect();
            push_text_quads(
                &mut self.ui_quads_scratch,
                &clipped,
                Vec2::new(centered_tx, y0 + 10.0),
                scale,
                [0.98, 0.95, 0.86, 1.0],
            );
        }

        // 3. Bouncy 8-Slice Radial Pie Wheel + Slice Labels!
        let mut radial_open_f = 0.0;
        let mut radial_params = [0.0_f32; 4];
        if let Some(rad) = &ui.radial {
            radial_open_f = 1.0;
            let hov = rad.hovered_slice.map(|s| s as f32).unwrap_or(-1.0);
            radial_params = [rad.center_px.x, rad.center_px.y, rad.outer_radius, hov];

            let ext = rad.outer_radius + 16.0;
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

            // Place crisp labels at each of the 8 slice centers (slice 0 = Top = -PI/2)
            for (idx, &lbl) in rad.labels.iter().enumerate() {
                let angle = -std::f32::consts::FRAC_PI_2
                    + (idx as f32) * (std::f32::consts::TAU / 8.0);
                let label_r = (34.0 + rad.outer_radius) * 0.54;
                let pos = rad.center_px + Vec2::new(angle.cos(), angle.sin()) * label_r;
                let scale = 1.35_f32;
                let tw = (lbl.chars().count() as f32) * 8.0 * scale;
                let is_hov = rad.hovered_slice == Some(idx);
                push_text_quads(
                    &mut self.ui_quads_scratch,
                    lbl,
                    Vec2::new(pos.x - tw * 0.5, pos.y - 5.0),
                    scale,
                    if is_hov {
                        [0.08, 0.08, 0.11, 1.0]
                    } else {
                        [0.96, 0.96, 0.99, 1.0]
                    },
                );
            }
        }

        let ui_u = UiUniforms {
            viewport_srgb_radial: [
                self.width as f32,
                self.height as f32,
                manual_srgb,
                radial_open_f,
            ],
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

    /// End-to-end headless GPU verification: compiles all WGSL shaders, evaluates 3D SDF bricks,
    /// extracts live surface flecks, renders clay + flecks + DOF + Toy UI, and verifies GPU output!
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

        // Frame 1: Initial Diorama Island full brick evaluation + fleck extraction + render
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

        // Frame 2: Simulate an interactive clay sculpt stroke + Play Mode frame!
        let ui2 = editor.update(
            0.016,
            &FrameInput {
                imp_px: Vec2::new(320.0, 200.0),
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

        // Read back DrawIndirectArgs (`[vertex_count, instance_count, 0, 0]`) + 1 row of pixels
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

        if fleck_instances < 500 {
            return Err(format!(
                "Expected >= 500 extracted surface flecks on starter island, got {fleck_instances}"
            ));
        }
        if center_rgba[0] == 0 && center_rgba[1] == 0 && center_rgba[2] == 0 {
            return Err("Rendered center pixel was completely black!".into());
        }

        Ok(format!(
            "SELF-TEST PASSED!\n  GPU Adapter : {} ({:?})\n  SDF Volume  : {}x{}x{} voxels ({}x{}x{} bricks)\n  Live Flecks : {} painterly brushstroke instances\n  Center RGBA : {:?}",
            adapter_info.name,
            adapter_info.backend,
            VOLUME_DIMS.x,
            VOLUME_DIMS.y,
            VOLUME_DIMS.z,
            BRICK_GRID.x,
            BRICK_GRID.y,
            BRICK_GRID.z,
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
    scale: f32,
    color_srgb: [f32; 4],
) {
    let cw = 8.0 * scale;
    let ch_h = 8.0 * scale;
    for (i, ch) in text.chars().enumerate() {
        if out.len() >= MAX_UI_QUADS {
            break;
        }
        if ch == ' ' {
            continue;
        }
        let (uv0, uv1) = font::glyph_uv(ch);
        let x0 = origin.x + (i as f32) * cw;
        let y0 = origin.y;
        out.push(GpuUiQuad {
            rect_px: [x0, y0, x0 + cw, y0 + ch_h],
            uv_rect: [uv0[0], uv0[1], uv1[0], uv1[1]],
            fill_color: color_srgb,
            border_color: [0.0; 4],
            params: [0.0, 0.0, 1.0, 0.0],
        });
    }
}

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

fn bgl_sampler(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}
