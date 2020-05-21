#![feature(osstring_ascii, const_generics)]

use bsp::Bsp;

use std::time;
use wgpu;
use winit::{
    event::{self, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

mod loader;
mod render;

use loader::{Load, Loader};
use render::Renderer;

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

    let mut renderer = Renderer::init(loader, &device).unwrap();

    queue.submit(std::iter::once(renderer.set_map(&device, &bsp)));

    let mut sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };

    let mut swap_chain = device.create_swap_chain(&surface, &sc_desc);

    let mut last_update_inst = time::Instant::now();

    const FPS: f64 = 60.;

    'find_player_start: for entity in bsp.entities.iter() {
        let mut is_player_start = false;
        let mut player_start_pos: Option<cgmath::Vector3<f32>> = None;

        for (key, val) in entity.properties() {
            match (key, val) {
                ("classname", "info_player_start") => is_player_start = true,
                ("classname", _) => continue 'find_player_start,
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
                _ => {}
            }
        }

        if is_player_start {
            renderer.camera_mut().position =
                player_start_pos.unwrap() + cgmath::Vector3::from([0., 0., 64.]);
        }
    }

    event_loop.run(move |event, _, control_flow| {
        let _ = (&instance, &adapter); // force ownership by the closure
        *control_flow =
            ControlFlow::WaitUntil(last_update_inst + time::Duration::from_secs_f64(1. / FPS));
        match event {
            Event::MainEventsCleared => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if last_update_inst.elapsed() > time::Duration::from_millis(20) {
                        window.request_redraw();
                        last_update_inst = time::Instant::now();
                    }
                }

                #[cfg(target_arch = "wasm32")]
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                sc_desc.width = size.width;
                sc_desc.height = size.height;
                swap_chain = device.create_swap_chain(&surface, &sc_desc);
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(event::VirtualKeyCode::Escape),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                }
                | WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
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
                    event::VirtualKeyCode::Up => {
                        renderer.camera_mut().pitch += 3.;
                    }
                    event::VirtualKeyCode::Down => {
                        renderer.camera_mut().pitch -= 3.;
                    }
                    event::VirtualKeyCode::Right => {
                        renderer.camera_mut().yaw += 3.;
                    }
                    event::VirtualKeyCode::Left => {
                        renderer.camera_mut().yaw -= 3.;
                    }
                    _ => {}
                },
                _ => {}
            },
            Event::RedrawRequested(_) => {
                let frame = swap_chain
                    .get_next_texture()
                    .expect("Timeout when acquiring next swap chain texture");

                let pos: [f32; 3] = renderer.camera().position.into();
                if let Some(cluster) = bsp.cluster_at::<bsp::XEastYSouthZUp, _>(pos) {
                    let command_buf = renderer.render(
                        &device,
                        &sc_desc,
                        &frame,
                        &queue,
                        bsp.visible_clusters(cluster).map(|c| c as usize),
                    );
                    queue.submit(command_buf);
                }
            }
            _ => {}
        }
    });
}

fn main() {
    use goldsrc_mdl::Mdl;

    let loader = loader::Loader::new();

    let mut mdl = Mdl::read(
        loader
            .models()
            .load(std::path::PathBuf::from(std::env::args().nth(1).unwrap()).into())
            .unwrap()
            .0,
    )
    .unwrap();

    let mut bodyparts = mdl.bodyparts();

    while let Some((name, mut models)) = bodyparts.next().unwrap() {
        dbg!(name);

        while let Some(mut model) = models.next().unwrap() {
            dbg!(&*model);

            let mut meshes = model.meshes().unwrap();

            while let Some(mut mesh) = meshes.next().unwrap() {
                dbg!(&*mesh);

                dbg!(mesh
                    .triverts()
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap());
            }
        }
    }

    panic!();

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
