#![feature(osstring_ascii, const_generics, type_alias_impl_trait)]

extern crate open_asset_importer as assimp;

use bsp::Bsp;

use fnv::FnvHashSet as HashSet;
use std::{iter, time};
use winit::{
    event::{self, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

#[global_allocator]
#[cfg(feature = "jemallocator")]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

mod assets;
mod cache;
mod loader;
mod render;

use assets::{BspAsset, MdlAsset, SkyboxAsset};
use loader::{Load, LoadAsset, Loader};
use render::{Camera, DoRender, Renderer};

async fn run(loader: Loader, bsp: Bsp, event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();
    let instance = wgpu::Instance::new();
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: Some(&surface),
            },
            wgpu::BackendBit::PRIMARY,
        )
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                extensions: wgpu::Extensions {
                    anisotropic_filtering: false,
                },
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .unwrap();

    let mut renderer = Renderer::init(&device, size.into(), 1.6, 1.4);
    let mut sky = None;
    let mut camera = None;

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

    let mut model_importer = assimp::Importer::new();

    model_importer.join_identical_vertices(true);
    model_importer.optimize_meshes(true);
    model_importer.optimize_graph(|opt| {
        opt.enable = true;
    });
    model_importer.debone(|opt| {
        opt.enable = true;
    });
    model_importer.triangulate(true);

    let mut camera = camera.unwrap_or(Camera::new(cgmath::Deg(70.), size.width, size.height));

    let mut bsp = BspAsset(bsp).load(&loader, renderer.cache_mut()).unwrap();

    let mut sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };

    let mut swap_chain = device.create_swap_chain(&surface, &sc_desc);

    let mut last_update_inst = time::Instant::now();
    let mut last_render_inst = time::Instant::now();

    const FPS: f64 = 60.;
    const DT: f64 = 1. / FPS;
    const DEG_PER_SEC: cgmath::Deg<f32> = cgmath::Deg(30.);
    const MOVEMENT_VEL: cgmath::Vector3<f32> = cgmath::Vector3::new(400., 0., 0.);

    let mut lock_mouse = false;
    let mut keys_down = HashSet::default();

    let update_dt = time::Duration::from_secs_f64(DT);
    let render_dt = time::Duration::from_secs_f64(DT);

    let mut consecutive_timeouts = 0usize;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::WaitUntil(
            (last_update_inst + update_dt).min(last_render_inst + render_dt),
        );

        let mut elapsed = last_update_inst.elapsed();

        if elapsed >= update_dt {
            while elapsed >= update_dt {
                for keycode in &keys_down {
                    match keycode {
                        event::VirtualKeyCode::Up => {
                            camera.pitch -= DEG_PER_SEC * DT as f32;
                        }
                        event::VirtualKeyCode::Down => {
                            camera.pitch += DEG_PER_SEC * DT as f32;
                        }
                        event::VirtualKeyCode::Right => {
                            camera.yaw -= DEG_PER_SEC * DT as f32;
                        }
                        event::VirtualKeyCode::Left => {
                            camera.yaw += DEG_PER_SEC * DT as f32;
                        }
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
                        _ => {}
                    }
                }

                elapsed -= update_dt;
            }

            last_update_inst = time::Instant::now() - elapsed;
        }

        match event {
            Event::MainEventsCleared => {
                if last_render_inst.elapsed() > render_dt {
                    window.request_redraw();
                    last_render_inst = time::Instant::now();
                }
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                sc_desc.width = size.width;
                sc_desc.height = size.height;
                swap_chain = device.create_swap_chain(&surface, &sc_desc);

                camera.set_aspect_ratio(sc_desc.width, sc_desc.height);

                renderer.set_size(size.into());
            }

            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }

                WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(event::VirtualKeyCode::Escape),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    lock_mouse = false;
                    window.set_cursor_visible(true);
                }
                WindowEvent::MouseInput {
                    state: event::ElementState::Pressed,
                    button: event::MouseButton::Left,
                    ..
                } => {
                    lock_mouse = !lock_mouse;
                    window.set_cursor_visible(!lock_mouse);

                    if lock_mouse {
                        window
                            .set_cursor_position(winit::dpi::LogicalPosition::new(
                                size.width as f32 / 2.,
                                size.height as f32 / 2.,
                            ))
                            .unwrap();
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if !lock_mouse {
                        return;
                    }

                    let position: (f32, f32) = position.into();
                    let (hw, hh) = (size.width as f32 / 2., size.height as f32 / 2.);
                    let dx = hw - position.0;
                    let dy = hh - position.1;

                    window
                        .set_cursor_position(winit::dpi::LogicalPosition::new(hw, hh))
                        .unwrap();

                    camera.update_pitch(|p| p - cgmath::Deg::from(cgmath::Rad(dy / 100.0)));
                    camera.update_yaw(|y| y + cgmath::Deg::from(cgmath::Rad(dx / 100.0)));
                }
                WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                } => match keycode {
                    event::VirtualKeyCode::O => {
                        renderer.set_msaa_factor((renderer.msaa_factor() / 2).max(1).min(4));
                    }
                    event::VirtualKeyCode::P => {
                        renderer.set_msaa_factor((renderer.msaa_factor() * 2).max(1).min(4));
                    }
                    event::VirtualKeyCode::K => {
                        renderer.set_gamma((renderer.gamma() - 0.05).max(0.5).min(4.0));
                    }
                    event::VirtualKeyCode::L => {
                        renderer.set_gamma((renderer.gamma() + 0.05).max(0.5).min(4.0));
                    }
                    keycode => {
                        keys_down.insert(keycode);
                    }
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
                }
                _ => {}
            },
            Event::RedrawRequested(_) => match swap_chain.get_next_texture() {
                Ok(frame) => {
                    consecutive_timeouts = 0;

                    for commands in
                        renderer.render(&device, &camera, &frame.view, &queue, |mut ctx| {
                            if let Some(sky) = &sky {
                                ctx.render(sky);
                            }
                            ctx.render(&mut bsp);
                        })
                    {
                        queue.submit(iter::once(commands));
                    }
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
            },
            _ => {}
        }
    });
}

fn main() {
    let loader = loader::Loader::new();

    let bsp = Bsp::read(
        loader
            .maps()
            .load(std::path::PathBuf::from(std::env::args().nth(1).unwrap()).into())
            .unwrap()
            .0,
    )
    .unwrap();

    let events = EventLoop::new();
    let window = Window::new(&events).unwrap();

    futures::executor::block_on(run(loader, bsp, events, window));
}
