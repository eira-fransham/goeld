#![feature(
    osstring_ascii,
    const_generics,
    min_type_alias_impl_trait,
    async_closure
)]

use bsp::Bsp;

use fnv::FnvHashSet as HashSet;
use std::{iter, time};
use winit::{
    event::{self, DeviceEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::unix::EventLoopExtUnix,
    window::Window,
};
use winit_async::{EventAsync as Event, EventLoopAsync};

#[global_allocator]
#[cfg(feature = "jemallocator")]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

mod assets;
mod cache;
mod gui;
mod kawase;
mod loader;
mod render;

use assets::{BspAsset, MdlAsset, SkyboxAsset};
use loader::{Load, LoadAsset, Loader};
use render::{Camera, Renderer};

const DEFAULT_SIZE: (u32, u32) = (800, 600);

fn to_normal_event<E>(event: Event<E>) -> Option<winit::event::Event<'static, E>> {
    match event {
        Event::WindowEvent { window_id, event } => {
            Some(winit::event::Event::WindowEvent { window_id, event })
        }
        Event::DeviceEvent { device_id, event } => {
            Some(winit::event::Event::DeviceEvent { device_id, event })
        }
        Event::UserEvent(val) => Some(winit::event::Event::UserEvent(val)),
        Event::Suspended | Event::Resumed => None,
    }
}

async fn run(loader: Loader, bsp: Bsp, event_loop: EventLoop<()>, window: Window) {
    let (width, height) = DEFAULT_SIZE;
    window.set_inner_size(winit::dpi::LogicalSize { width, height });
    let size = window.inner_size();
    let scale_factor = window.scale_factor();

    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
            compatible_surface: Some(&surface),
        })
        .await
        .unwrap();

    let out_path = if cfg!(debug_assertions) {
        use std::convert::TryFrom;

        Some(
            std::path::PathBuf::try_from(env!("CARGO_MANIFEST_DIR"))
                .unwrap()
                .join("calls.dbg"),
        )
    } else {
        None
    };
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::default(),
                limits: wgpu::Limits::default(),
                shader_validation: cfg!(debug_assertions),
            },
            out_path.as_ref().map(|p| &**p),
        )
        .await
        .unwrap();

    let mut renderer = Renderer::init(
        &device,
        (
            (size.width as f64 / scale_factor) as _,
            (size.height as f64 / scale_factor) as _,
        ),
        size.into(),
        1.2,
        1.0,
    );

    let mut sky = None;
    let mut camera = None;
    let mut imgui = imgui::Context::create();
    let mut imgui_platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
    imgui_platform.attach_window(
        imgui.io_mut(),
        &window,
        imgui_winit_support::HiDpiMode::Default,
    );
    imgui.set_ini_filename(None);

    let hidpi_factor = window.scale_factor();
    let font_size = (13.0 * hidpi_factor) as f32;
    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

    imgui
        .fonts()
        .add_font(&[imgui::FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                oversample_h: 1,
                pixel_snap_h: true,
                size_pixels: font_size,
                ..Default::default()
            }),
        }]);

    'find_special_entities: for entity in bsp.entities.iter() {
        let mut is_player_start = false;
        let mut is_worldspawn = false;
        let mut player_start_pos: Option<cgmath::Vector3<f32>> = None;
        let mut player_start_angle: Option<f32> = None;

        for (key, val) in entity.properties() {
            match (key, val) {
                ("classname", "info_player_start") => is_player_start = true,
                ("classname", "worldspawn") => is_worldspawn = true,
                ("classname", _) => continue 'find_special_entities,
                ("sky", val) => {
                    if is_worldspawn {
                        sky = Some(SkyboxAsset(val.into()))
                    }
                }
                ("origin", origin) => {
                    let pos = origin.find(' ').unwrap();
                    let x = origin[..pos].parse::<f32>().unwrap();
                    let origin = &origin[pos + 1..];

                    let pos = origin.find(' ').unwrap();
                    let y = origin[..pos].parse::<f32>().unwrap();
                    let origin = &origin[pos + 1..];

                    let z = origin.parse::<f32>().unwrap();

                    player_start_pos = Some([x, y, z].into());
                }
                ("angle", angle) => {
                    player_start_angle = Some(angle[..].parse::<f32>().unwrap());
                }
                _ => {}
            }
        }

        if is_player_start {
            let mut new_camera = Camera::new(cgmath::Deg(70.), size.width, size.height);
            new_camera.position = player_start_pos.unwrap() + cgmath::Vector3::from([0., 0., 64.]);
            new_camera.yaw = cgmath::Deg(player_start_angle.unwrap_or_default());

            camera = Some(new_camera);
        }
    }

    let sky = if bsp
        .textures
        .iter()
        .any(|t| t.flags.contains(bsp::SurfaceFlags::SKY))
    {
        sky.and_then(|sky| {
            sky.load(&loader, renderer.cache_mut())
                .map_err(|e| dbg!(e))
                .ok()
        })
        .or_else(|| {
            SkyboxAsset("unit1_".into())
                .load(&loader, renderer.cache_mut())
                .map_err(|e| dbg!(e))
                .ok()
        })
    } else {
        None
    };

    let mut camera = camera.unwrap_or(Camera::new(cgmath::Deg(70.), size.width, size.height));

    let mut bsp = BspAsset(bsp).load(&loader, renderer.cache_mut()).unwrap();

    let mut model = MdlAsset {
        main: goldsrc_mdl::Mdl::read(std::fs::File::open("data/models/bullsquid.mdl").unwrap())
            .unwrap(),
        textures: Some(
            goldsrc_mdl::Mdl::read(std::fs::File::open("data/models/bullsquidT.mdl").unwrap())
                .unwrap(),
        ),
    }
    .load(&loader, renderer.cache_mut())
    .unwrap();

    model.set_animation(0);

    let mut sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: render::pipelines::SWAPCHAIN_FORMAT,
        width: (size.width as f64 / scale_factor) as _,
        height: (size.height as f64 / scale_factor) as _,
        present_mode: wgpu::PresentMode::Mailbox,
    };

    let mut swap_chain = device.create_swap_chain(&surface, &sc_desc);

    let start = time::Instant::now();
    let mut last_update_inst = time::Instant::now();
    let mut last_render_inst = time::Instant::now();

    const FPS: f64 = 60.;
    const DT: f64 = 1. / FPS;
    const ANIM_DT: f64 = 1. / 5.;
    const MOVEMENT_VEL: cgmath::Vector3<f32> = cgmath::Vector3::new(400., 0., 0.);

    let mut locked_mouse = false;
    let mut last_cursor = None;
    let mut debug_gui = false;
    let mut keys_down = HashSet::default();

    let update_dt = time::Duration::from_secs_f64(DT);
    let render_dt = time::Duration::from_secs_f64(DT);
    let anim_dt = time::Duration::from_secs_f64(ANIM_DT);

    let mut consecutive_timeouts = 0usize;

    // IMGUI SETUP
    let renderer_config = imgui_wgpu::RendererConfig {
        texture_format: sc_desc.format,
        ..Default::default()
    };
    let mut imgui_renderer =
        imgui_wgpu::Renderer::new(&mut imgui, &device, &queue, renderer_config);
    // END IMGUI SETUP (TODO: Extract to function)

    let mut frametimes = arraydeque::ArrayDeque::<[f64; 64], arraydeque::Wrapping>::new();

    event_loop.run_async(async move |mut runner| 'main: loop {
        runner.wait().await;

        let now = time::Instant::now();

        let mut elapsed = now - last_update_inst;

        frametimes.push_back(elapsed.as_secs_f64());

        if elapsed >= update_dt {
            while elapsed >= update_dt {
                for keycode in &keys_down {
                    match keycode {
                        event::VirtualKeyCode::W => {
                            camera.position += cgmath::Matrix3::from_angle_z(camera.yaw)
                                * cgmath::Matrix3::from_angle_y(camera.pitch)
                                * MOVEMENT_VEL
                                * DT as f32;
                        }
                        event::VirtualKeyCode::S => {
                            camera.position -= cgmath::Matrix3::from_angle_z(camera.yaw)
                                * cgmath::Matrix3::from_angle_y(camera.pitch)
                                * MOVEMENT_VEL
                                * DT as f32;
                        }
                        event::VirtualKeyCode::A => {
                            camera.position +=
                                cgmath::Matrix3::from_angle_z(camera.yaw + cgmath::Deg(90.))
                                    * MOVEMENT_VEL
                                    * DT as f32;
                        }
                        event::VirtualKeyCode::D => {
                            camera.position +=
                                cgmath::Matrix3::from_angle_z(camera.yaw - cgmath::Deg(90.))
                                    * MOVEMENT_VEL
                                    * DT as f32;
                        }
                        event::VirtualKeyCode::Up => {
                            model.update_position(cgmath::Vector3::unit_x() * 100. * DT as f32);
                        }
                        event::VirtualKeyCode::Down => {
                            model.update_position(-cgmath::Vector3::unit_x() * 100. * DT as f32);
                        }
                        event::VirtualKeyCode::Left => {
                            model.update_position(cgmath::Vector3::unit_y() * 100. * DT as f32);
                        }
                        event::VirtualKeyCode::Right => {
                            model.update_position(-cgmath::Vector3::unit_y() * 100. * DT as f32);
                        }
                        _ => {}
                    }
                }

                elapsed -= update_dt;
            }

            last_update_inst = now - elapsed;
        }

        let mut events = runner.recv_events().await;
        while let Some(event) = events.next().await {
            let captured = match &event {
                &Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => {
                    sc_desc.width = size.width;
                    sc_desc.height = size.height;
                    swap_chain = device.create_swap_chain(&surface, &sc_desc);

                    camera.set_aspect_ratio(sc_desc.width, sc_desc.height);

                    renderer.set_size((
                        (size.width as f64 / scale_factor) as _,
                        (size.height as f64 / scale_factor) as _,
                    ));
                    renderer.set_framebuffer_size(size.into());

                    false
                }

                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => break 'main,
                    WindowEvent::KeyboardInput {
                        input:
                            event::KeyboardInput {
                                virtual_keycode: Some(event::VirtualKeyCode::Escape),
                                state: event::ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        locked_mouse = false;

                        true
                    }
                    &WindowEvent::MouseInput {
                        state: event::ElementState::Pressed,
                        button: event::MouseButton::Left,
                        ..
                    } if !debug_gui => {
                        locked_mouse = true;

                        true
                    }
                    &WindowEvent::KeyboardInput {
                        input:
                            event::KeyboardInput {
                                virtual_keycode: Some(keycode),
                                state: event::ElementState::Pressed,
                                ..
                            },
                        ..
                    } => match keycode {
                        event::VirtualKeyCode::Grave => {
                            debug_gui = !debug_gui;
                            locked_mouse &= !debug_gui;

                            true
                        }
                        event::VirtualKeyCode::Space if !debug_gui => {
                            locked_mouse = !locked_mouse;

                            true
                        }
                        keycode if !debug_gui => {
                            keys_down.insert(keycode);

                            true
                        }
                        _ => false,
                    },
                    WindowEvent::KeyboardInput {
                        input:
                            event::KeyboardInput {
                                virtual_keycode: Some(keycode),
                                state: event::ElementState::Released,
                                ..
                            },
                        ..
                    } => {
                        keys_down.remove(&keycode);
                        true
                    }
                    _ => false,
                },
                &Event::DeviceEvent {
                    event: DeviceEvent::MouseMotion { delta: (dx, dy) },
                    ..
                } if locked_mouse => {
                    let (dx, dy) = (dx as f32, dy as f32);

                    camera.update_pitch(|p| p + cgmath::Deg::from(cgmath::Rad(dy / 100.0)));
                    camera.update_yaw(|y| y - cgmath::Deg::from(cgmath::Rad(dx / 100.0)));

                    true
                }
                _ => false,
            };

            window.set_cursor_visible(!locked_mouse);
            window.set_cursor_grab(locked_mouse).unwrap();

            if !captured {
                if let Some(event) = to_normal_event(event) {
                    imgui_platform.handle_event(imgui.io_mut(), &window, &event);
                }
            }
        }

        let mut redraw_requests = events.redraw_requests().await;
        while let Some(_window_id) = redraw_requests.next().await {
            // TODO: Doesn't work for multiple windows
            let elapsed = now - last_render_inst;
            last_render_inst = now;

            match swap_chain.get_current_frame() {
                Ok(frame) => {
                    consecutive_timeouts = 0;

                    model.update(renderer.cache_mut());
                    model.step(elapsed.as_secs_f64());
                    renderer.set_time((now - start).as_secs_f32());

                    let fut = renderer.prepare();
                    device.poll(wgpu::Maintain::Poll);
                    fut.await.expect("Prepare buffers failed");

                    renderer.transfer_data(&queue, std::iter::once(&model));

                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("render"),
                        });

                    renderer.render(
                        &device,
                        &mut encoder,
                        &camera,
                        &frame.output.view,
                        &queue,
                        |mut ctx| {
                            if let Some(sky) = &sky {
                                ctx.render(sky);
                            }

                            ctx.render(&mut bsp);

                            let mut ctx = ctx.with_world(&bsp);

                            ctx.render(&model);
                        },
                    );

                    {
                        let mut imgui_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                                    attachment: &frame.output.view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: true,
                                    },
                                }],
                                depth_stencil_attachment: None,
                            });

                        imgui.io_mut().update_delta_time(elapsed);

                        imgui_platform
                            .prepare_frame(imgui.io_mut(), &window)
                            .expect("Failed to prepare frame");
                        let ui = imgui.frame();

                        gui::fps(
                            &ui,
                            frametimes.iter().copied().sum::<f64>() / frametimes.len() as f64,
                        );

                        if debug_gui {
                            renderer.update_config(|config| gui::config(&ui, config));
                        }

                        if last_cursor != Some(ui.mouse_cursor()) {
                            last_cursor = Some(ui.mouse_cursor());
                            imgui_platform.prepare_render(&ui, &window);
                        }

                        imgui_renderer
                            .render(ui.render(), &queue, &device, &mut imgui_pass)
                            .expect("Rendering failed");
                    }

                    queue.submit(iter::once(encoder.finish()));
                }
                Err(_) => {
                    consecutive_timeouts += 1;

                    if consecutive_timeouts > FPS as usize * 3 {
                        panic!(
                            "Timeout aquiring swap chain texture ({} consecutive timeouts",
                            consecutive_timeouts
                        );
                    }
                }
            }
        }

        let _ = runner
            .wait_until((last_update_inst + update_dt).min(last_render_inst + render_dt))
            .await;
        window.request_redraw();
    });
}

fn main() {
    let loader = loader::Loader::new();

    let (file, path) = loader
        .maps()
        .load(std::path::PathBuf::from(std::env::args().nth(1).unwrap()).into())
        .unwrap();
    let bsp = Bsp::read(file).unwrap();

    let events = EventLoop::new_any_thread();
    let mut window = Window::new(&events).unwrap();

    window.set_title(&format!("Göld - {}", path.display()));

    futures::executor::block_on(run(loader, bsp, events, window));
}
