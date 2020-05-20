use crate::loader::{Load, Loader};
use bsp::Bsp;
use bytemuck::{Pod, Zeroable};
use fnv::FnvHashMap as HashMap;
use itertools::Itertools;
use std::{iter, mem, ops::Range};

struct World {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    cluster_ranges: Vec<Range<u32>>,
}

pub struct Camera {
    pub vertical_fov: f32,
    pub position: cgmath::Vector3<f32>,
    pub pitch: f32,
    pub yaw: f32,
}

impl Camera {
    fn new(vertical_fov: f32) -> Self {
        Camera {
            vertical_fov,
            position: cgmath::vec3(0., 0., 0.),
            pitch: 0.,
            yaw: 0.,
        }
    }

    fn matrix(&self, width: u32, height: u32) -> cgmath::Matrix4<f32> {
        let projection =
            cgmath::perspective(cgmath::Deg(70f32), width as f32 / height as f32, 1., 4096.);
        let view = cgmath::Matrix4::from_nonuniform_scale(1.0, -1.0, 1.0)
            * cgmath::Matrix4::from_angle_x(cgmath::Deg(self.pitch) + cgmath::Deg(90.))
            * cgmath::Matrix4::from_angle_z(cgmath::Deg(-self.yaw) + cgmath::Deg(180.))
            * cgmath::Matrix4::from_translation(-self.position);

        OPENGL_TO_WGPU_MATRIX * projection * view
    }
}

pub struct Renderer {
    world: Option<World>,
    diffuse_atlas: wgpu::Texture,
    lightmap_atlas: wgpu::Texture,
    camera: Camera,
    loader: Loader,

    bind_group: wgpu::BindGroup,
    matrix: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
}

// TODO: Do this better somehow
const ATLAS_EXTENT: wgpu::Extent3d = wgpu::Extent3d {
    width: 1024,
    height: 1024,
    depth: 1,
};

const VERTEX_SHADER: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/basic.vert.spv"
));
const FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/basic.frag.spv"
));

#[derive(Debug, Copy, Clone)]
pub struct InitRendererError;

#[cfg_attr(rustfmt, rustfmt_skip)]
#[allow(unused)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

/// The vertex type for the device. We use `_` to suppress warnings about
/// unused fields - we never read them since we just cast to a byte array.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Vertex {
    pos: [f32; 4],
    tex: [f32; 4],
    tex_coord: [f32; 2],
    lightmap_coord: [f32; 2],
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

struct LightmapAtlas {
    inner: image::RgbaImage,
    alloc: rect_packer::Packer,
}

impl LightmapAtlas {
    fn new(width: u32, height: u32) -> Self {
        Self {
            alloc: rect_packer::Packer::new(rect_packer::Config {
                width: width as i32,
                height: height as i32,
                border_padding: 1,
                rectangle_padding: 1,
            }),
            inner: image::RgbaImage::from_raw(
                width,
                height,
                vec![0; (width * height * 4) as usize],
            )
            .unwrap(),
        }
    }

    fn append(&mut self, width: u32, height: u32, data: &[u8]) -> rect_packer::Rect {
        if width == 0 || height == 0 {
            return rect_packer::Rect {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            };
        }

        assert_eq!((width * height) as usize * 3, data.len());

        let rect = self.alloc.pack(width as i32, height as i32, false).unwrap();

        for y in -1..height as i32 + 1 {
            for x in -1..width as i32 + 1 {
                use std::convert::TryInto;

                let src_x = x.min(width as i32 - 1).max(0) as u32;
                let src_y = y.min(height as i32 - 1).max(0) as u32;

                let [r, g, b]: [u8; 3] = data[(src_y * width + src_x) as usize * 3..][..3]
                    .try_into()
                    .unwrap();

                self.inner.put_pixel(
                    (rect.x + x) as u32,
                    (rect.y + y) as u32,
                    image::Rgba([r, g, b, 255]),
                );
            }
        }

        rect
    }

    fn into_inner(self) -> image::RgbaImage {
        self.inner
    }
}

/// Get the vertices per-cluster. First element is the vertices for cluster 0, then for cluster 1, and so
/// forth.
// TODO: Maybe generate the indices-per-cluster lazily to reduce GPU memory pressure? We can recalculate
//       _only_ the indices for the cluster we're entering when we change clusters.
fn leaf_meshes<'a>(
    bsp: &'a Bsp,
    face_start_indices: &'a mut Vec<u32>,
    texture_map: &'a HashMap<&str, rect_packer::Rect>,
    lightmap_atlas: &mut LightmapAtlas,
) -> (
    Vec<Vertex>,
    impl ExactSizeIterator<
            Item = (
                bsp::Handle<'a, bsp::Leaf>,
                impl Iterator<Item = u32> + Clone + 'a,
            ),
        > + Clone
        + 'a,
) {
    // We'll probably need to reallocate a few times since vertices are reused,
    // but this is a reasonable lower bound
    let mut vertices = Vec::with_capacity(bsp.vertices.len());

    for face in bsp.faces() {
        face_start_indices.push(vertices.len() as u32);

        let texture = if let Some(texture) = face.texture() {
            texture
        } else {
            continue;
        };

        let tex_rect = if let Some(tex_rect) = texture_map.get(&texture.name[..]) {
            tex_rect
        } else {
            continue;
        };

        let lightmap = if let Some(lightmap) = face.lightmap() {
            Some((
                lightmap.mins,
                lightmap_atlas.append(lightmap.width(), lightmap.height(), lightmap.data),
            ))
        } else {
            None
        };

        vertices.extend(face.vertices().map(|vert| {
            let (u, v) = (
                vert.dot(&texture.axis_u) + texture.offset_u,
                vert.dot(&texture.axis_v) + texture.offset_v,
            );

            Vertex {
                pos: [vert.x(), vert.y(), vert.z(), 1.],
                tex: [
                    tex_rect.x as f32 / ATLAS_EXTENT.width as f32,
                    tex_rect.y as f32 / ATLAS_EXTENT.height as f32,
                    tex_rect.width as f32 / ATLAS_EXTENT.width as f32,
                    tex_rect.height as f32 / ATLAS_EXTENT.height as f32,
                ],
                tex_coord: [
                    u / ATLAS_EXTENT.width as f32,
                    v / ATLAS_EXTENT.height as f32,
                ],
                lightmap_coord: lightmap
                    .map(|((minu, minv), lightmap_rect)| {
                        [
                            (lightmap_rect.x as f32 + (u / 16.).floor() - minu)
                                / ATLAS_EXTENT.width as f32,
                            (lightmap_rect.y as f32 + (v / 16.).floor() - minv)
                                / ATLAS_EXTENT.height as f32,
                        ]
                    })
                    .unwrap_or([0., 0.]),
            }
        }));
    }

    let face_start_indices: &'a [u32] = &*face_start_indices;

    (
        vertices,
        bsp.leaves().map(move |leaf| {
            (
                leaf,
                leaf.leaf_faces().flat_map(move |leaf_face| {
                    let start = face_start_indices[leaf_face.face as usize] as u32;
                    let face = leaf_face.face();

                    // We don't need to care about handling winding order here since that's
                    // dealt with in `Face::vertices`
                    match face.vertices().len() {
                        0 | 1 | 2 => &[][..],
                        3 => &[0, 1, 2],
                        4 => &[0, 1, 2, 2, 3, 0],
                        _ => todo!(),
                    }
                    .iter()
                    .map(move |i| i + start)
                }),
            )
        }),
    )
}

// TODO: Make async.
// TODO: We never write to the same area twice, so using unsafe code we can make a type that allows
//       parallel writes.
fn build_diffuse_atlas<'a, L: Load, I>(
    loader: L,
    textures: I,
) -> Result<(image::RgbaImage, HashMap<&'a str, rect_packer::Rect>), Box<dyn std::error::Error>>
where
    I: IntoIterator<Item = &'a str> + 'a,
{
    use image::GenericImage;
    use rect_packer::DensePacker;
    use std::{collections::hash_map::Entry, path::Path};

    let textures = textures.into_iter();
    let mut map = HashMap::<&str, rect_packer::Rect>::with_capacity_and_hasher(
        textures.size_hint().0,
        Default::default(),
    );

    let mut allocator = DensePacker::new(ATLAS_EXTENT.width as i32, ATLAS_EXTENT.height as i32);
    let mut out = image::RgbaImage::new(ATLAS_EXTENT.width, ATLAS_EXTENT.height);

    let loader = &loader;

    for tex_name in textures {
        if let Entry::Vacant(entry) = map.entry(tex_name) {
            let load = move |path: &str| -> Result<image::RgbaImage, Box<dyn std::error::Error>> {
                let (file, path) = loader.load(Path::new(path).into())?;

                Ok(image::load(
                    std::io::BufReader::new(file),
                    image::ImageFormat::from_path(&path)?,
                )?
                .into_rgba())
            };

            let image = if let Ok(image) = load(tex_name) {
                image
            } else {
                entry.insert(rect_packer::Rect {
                    x: 0,
                    y: 0,
                    width: 0,
                    height: 0,
                });
                continue;
            };

            let rect = allocator
                .pack(image.width() as i32, image.height() as i32, false)
                .ok_or_else(|| format!("Ran out of space in texture atlas"))?;

            out.copy_from(&image, rect.x as u32, rect.y as u32)?;
            entry.insert(rect);
        }
    }

    let _ = out.save("atlas.png");

    Ok((out, map))
}

impl Renderer {
    pub fn init(loader: Loader, device: &wgpu::Device) -> Result<Self, InitRendererError> {
        use memoffset::offset_of;

        let vs_module = device
            .create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(VERTEX_SHADER)).unwrap());

        let fs_module = device.create_shader_module(
            &wgpu::read_spirv(std::io::Cursor::new(FRAGMENT_SHADER)).unwrap(),
        );

        // Create pipeline layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            bindings: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Cw,
                cull_mode: wgpu::CullMode::Front,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint32,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float4,
                            offset: offset_of!(Vertex, pos) as u64,
                            shader_location: 0,
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float4,
                            offset: offset_of!(Vertex, tex) as u64,
                            shader_location: 1,
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float2,
                            offset: offset_of!(Vertex, tex_coord) as u64,
                            shader_location: 2,
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float2,
                            offset: offset_of!(Vertex, lightmap_coord) as u64,
                            shader_location: 3,
                        },
                    ],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let camera = Camera::new(70.);
        let combined = camera.matrix(1, 1);
        let combined: &[f32; 16] = combined.as_ref();

        let matrix = device.create_buffer_with_data(
            bytemuck::cast_slice(combined),
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        );

        let diffuse_atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: ATLAS_EXTENT,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });
        let lightmap_atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: ATLAS_EXTENT,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });

        let diffuse_atlas_view = diffuse_atlas.create_default_view();

        let lightmap_atlas_view = lightmap_atlas.create_default_view();

        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: wgpu::CompareFunction::Undefined,
        });

        let lightmap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: wgpu::CompareFunction::Undefined,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(matrix.slice(..)),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&diffuse_atlas_view),
                },
                wgpu::Binding {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&lightmap_atlas_view),
                },
                wgpu::Binding {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                wgpu::Binding {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&lightmap_sampler),
                },
            ],
            label: None,
        });

        Ok(Self {
            world: None,
            loader,
            bind_group,
            matrix,
            camera,
            pipeline,
            diffuse_atlas,
            lightmap_atlas,
        })
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    // TODO: Allow multiple maps to be loaded simultaneously so we can do a streaming open world
    pub fn set_map(&mut self, device: &wgpu::Device, bsp: &Bsp) -> wgpu::CommandBuffer {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut buf = Vec::new();
        let (diffuse_atlas, texture_map) = build_diffuse_atlas(
            self.loader.textures(),
            bsp.textures.iter().map(|tex| &tex.name[..]),
        )
        .unwrap();

        let _ = diffuse_atlas.save("atlas.png");

        let (w, h) = (diffuse_atlas.width(), diffuse_atlas.height());
        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &device.create_buffer_with_data(
                    diffuse_atlas.into_raw().as_slice(),
                    wgpu::BufferUsage::COPY_SRC,
                ),
                offset: 0,
                bytes_per_row: 4 * w,
                rows_per_image: h,
            },
            wgpu::TextureCopyView {
                texture: &self.diffuse_atlas,
                mip_level: 0,
                array_layer: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            ATLAS_EXTENT,
        );

        let mut lightmap_atlas = LightmapAtlas::new(ATLAS_EXTENT.width, ATLAS_EXTENT.height);

        let (vertices, leaf_indices) =
            leaf_meshes(&bsp, &mut buf, &texture_map, &mut lightmap_atlas);
        let lightmap_atlas = lightmap_atlas.into_inner();

        let (w, h) = (lightmap_atlas.width(), lightmap_atlas.height());
        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &device.create_buffer_with_data(
                    lightmap_atlas.into_raw().as_slice(),
                    wgpu::BufferUsage::COPY_SRC,
                ),
                offset: 0,
                bytes_per_row: 4 * w,
                rows_per_image: h,
            },
            wgpu::TextureCopyView {
                texture: &self.lightmap_atlas,
                mip_level: 0,
                array_layer: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            ATLAS_EXTENT,
        );

        let vbuffer = device
            .create_buffer_with_data(bytemuck::cast_slice(&vertices), wgpu::BufferUsage::VERTEX);

        let mut all_indices = Vec::new();

        let mut cluster_ranges = Vec::with_capacity(leaf_indices.size_hint().0);

        let clusters = leaf_indices.group_by(|(leaf, _)| leaf.cluster);

        for (cluster, indices) in clusters.into_iter() {
            if cluster != cluster_ranges.len() as i16 {
                continue;
            }

            let start = all_indices.len();
            all_indices.extend(indices.flat_map(|(_, i)| i));
            let end = all_indices.len();

            cluster_ranges.push(start as u32..end as u32);
        }

        let ibuffer = device.create_buffer_with_data(
            bytemuck::cast_slice(&all_indices[..]),
            wgpu::BufferUsage::INDEX,
        );

        self.world = Some(World {
            vertices: vbuffer,
            indices: ibuffer,
            cluster_ranges,
        });

        encoder.finish()
    }

    pub fn render<I>(
        &mut self,
        device: &wgpu::Device,
        swapchain_desc: &wgpu::SwapChainDescriptor,
        frame: &wgpu::SwapChainOutput,
        queue: &wgpu::Queue,
        clusters: I,
    ) -> Option<wgpu::CommandBuffer>
    where
        I: IntoIterator<Item = usize>,
    {
        let world = self.world.as_ref()?;
        let mut clusters = clusters.into_iter();
        let first_cluster = clusters.next()?;

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let matrix = self
                .camera
                .matrix(swapchain_desc.width, swapchain_desc.height);
            let matrix: &[f32; 16] = matrix.as_ref();
            queue.write_buffer(bytemuck::cast_slice(matrix), &self.matrix, 0);

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    },
                }],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(0, world.vertices.slice(..));
            rpass.set_index_buffer(world.indices.slice(..));

            for range in iter::once(first_cluster)
                .chain(clusters)
                .map(|cluster| &world.cluster_ranges[cluster])
            {
                if range.clone().count() > 0 {
                    rpass.draw_indexed(range.clone(), 0, 0..1);
                }
            }
        }

        Some(encoder.finish())
    }
}
