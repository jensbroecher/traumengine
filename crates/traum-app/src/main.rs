//! Traum Engine — Application Entrypoint (`winit 0.30` event loop + `--self-test` GPU verification).

use std::sync::Arc;
use std::time::Instant;
use traum_core::EditorMode;
use traum_editor::TraumEditor;
use traum_input::{CameraRig, InputHub};
use traum_render::TraumRenderer;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

struct TraumApp {
    window: Option<Arc<Window>>,
    renderer: Option<TraumRenderer>,
    input: Option<InputHub>,
    camera: CameraRig,
    editor: TraumEditor,
    last_frame_instant: Instant,
}

impl TraumApp {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            input: None,
            camera: CameraRig::default(),
            editor: TraumEditor::new(),
            last_frame_instant: Instant::now(),
        }
    }
}

impl ApplicationHandler for TraumApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("Traum Engine — Sculpt • Animate • Play")
            .with_inner_size(LogicalSize::new(1360.0, 820.0))
            .with_min_inner_size(LogicalSize::new(800.0, 540.0));

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create Traum Engine window"),
        );

        let size = window.inner_size();
        let scale_factor = window.scale_factor() as f32;
        let renderer = TraumRenderer::new_windowed(window.clone())
            .expect("Failed to initialize wgpu renderer for Traum Engine");

        let mut hub = InputHub::new(size);
        hub.set_scale_factor(scale_factor);
        self.input = Some(hub);
        self.renderer = Some(renderer);
        self.window = Some(window.clone());
        self.last_frame_instant = Instant::now();
        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let WindowEvent::CloseRequested = &event {
            event_loop.exit();
            return;
        }
        if let WindowEvent::KeyboardInput { event: kev, .. } = &event {
            if let PhysicalKey::Code(KeyCode::Escape) = kev.physical_key {
                if kev.state.is_pressed() {
                    if self.editor.brush.mode == EditorMode::Play {
                        self.editor.switch_mode(EditorMode::Sculpt, &mut self.camera);
                    } else if self.editor.scoped_prop_id.is_some() {
                        self.editor.scope_out();
                    } else {
                        event_loop.exit();
                    }
                    return;
                }
            }
        }

        if let WindowEvent::Resized(new_size) = &event {
            if let Some(renderer) = &mut self.renderer {
                renderer.resize(*new_size);
            }
        }

        let is_play = self.editor.brush.mode == EditorMode::Play;
        if let Some(input) = &mut self.input {
            input.handle_window_event(&event, &mut self.camera, is_play);
        }

        if let WindowEvent::RedrawRequested = event {
            let now = Instant::now();
            let dt = (now - self.last_frame_instant)
                .as_secs_f32()
                .clamp(1.0 / 240.0, 0.05);
            self.last_frame_instant = now;

            if let (Some(input), Some(renderer)) = (&mut self.input, &mut self.renderer) {
                let frame_in = input.begin_frame(dt, &mut self.camera, is_play);
                let device_kind = input.active_device;
                let ui = self
                    .editor
                    .update(dt, &frame_in, device_kind, &mut self.camera);

                if let Err(err) = renderer.render_window_frame(&mut self.editor, &self.camera, &ui)
                {
                    log::error!("Frame render error: {err}");
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--self-test") {
        log::info!("Running Traum Engine GPU headless self-test...");
        match TraumRenderer::run_headless_self_test() {
            Ok(summary) => {
                println!("{summary}");
                return;
            }
            Err(err) => {
                eprintln!("Traum Engine GPU self-test failed: {err}");
                std::process::exit(1);
            }
        }
    }

    let event_loop = EventLoop::new().expect("Failed to create winit event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = TraumApp::new();
    event_loop
        .run_app(&mut app)
        .expect("Traum Engine event loop exited with error");
}
