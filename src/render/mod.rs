use crate::cache::{Atlas, BufferCache, CacheCommon};
use bytemuck::{Pod, Zeroable};
use std::ops::Range;

mod pipelines;

use pipelines::Pipeline;

#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    pub vertical_fov: cgmath::Deg<f32>,
    pub position: cgmath::Vector3<f32>,
    pub pitch: cgmath::Deg<f32>,
    pub yaw: cgmath::Deg<f32>,
    pub aspect_ratio: f32,
}

impl Camera {
    pub fn new(vertical_fov: impl Into<cgmath::Deg<f32>>, width: u32, height: u32) -> Self {
        Camera {
            vertical_fov: vertical_fov.into(),
            position: cgmath::vec3(0., 0., 0.),
            pitch: cgmath::Deg(0.),
            yaw: cgmath::Deg(0.),
            aspect_ratio: width as f32 / height as f32,
        }
    }

    pub fn update_pitch<F>(&mut self, func: F) -> cgmath::Deg<f32>
    where
        F: FnOnce(cgmath::Deg<f32>) -> cgmath::Deg<f32>,
    {
        self.pitch = cgmath::Deg(func(self.pitch).0.min(90.).max(-90.));
        self.pitch
    }

    pub fn update_yaw<F>(&mut self, func: F) -> cgmath::Deg<f32>
    where
        F: FnOnce(cgmath::Deg<f32>) -> cgmath::Deg<f32>,
    {
        self.yaw = cgmath::Deg(func(self.yaw).0 % 360.);
        self.yaw
    }

    pub fn set_aspect_ratio(&mut self, width: u32, height: u32) {
        self.aspect_ratio = width as f32 / height as f32;
    }

    pub fn matrix(&self) -> cgmath::Matrix4<f32> {
        use cgmath::{Matrix3, Matrix4};

        #[cfg_attr(rustfmt, rustfmt_skip)]
        const QUAKE_TO_OPENGL_TRANSFORMATION_MATRIX: Matrix4<f32> = Matrix4::new(
            0.0,0.0,  -1.0, 0.0,
            -1.0, 0.0, 0.0, 0.0,
            0.0, 1.0,  0.0, 0.0,
            0.0, 0.0,  0.0, 1.0,
        );

        let quake_to_opengl_matrix = Matrix3::from_cols(
            QUAKE_TO_OPENGL_TRANSFORMATION_MATRIX.x.truncate(),
            QUAKE_TO_OPENGL_TRANSFORMATION_MATRIX.y.truncate(),
            QUAKE_TO_OPENGL_TRANSFORMATION_MATRIX.z.truncate(),
        );

        let projection = cgmath::perspective(self.vertical_fov, self.aspect_ratio, 1., 4096.);
        let view = Matrix4::from_angle_x(self.pitch)
            * Matrix4::from_angle_y(-self.yaw)
            * Matrix4::from_translation(-quake_to_opengl_matrix * self.position)
            * QUAKE_TO_OPENGL_TRANSFORMATION_MATRIX;

        OPENGL_TO_WGPU_MATRIX * projection * view
    }
}

pub struct RenderCache {
    pub diffuse: Atlas,
    pub lightmap: Atlas,

    pub vertices: BufferCache<Vertex>,
    pub indices: BufferCache<u32>,
}

impl RenderCache {
    fn new(device: &wgpu::Device) -> Self {
        let diffuse = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tex_diffuseatlas"),
            size: DIFFUSE_ATLAS_EXTENT,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });
        let lightmap = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tex_lightmapatlas"),
            size: LIGHTMAP_ATLAS_EXTENT,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });

        Self {
            diffuse: Atlas::new(
                diffuse,
                DIFFUSE_ATLAS_EXTENT.width,
                DIFFUSE_ATLAS_EXTENT.height,
                1,
            ),
            lightmap: Atlas::new(
                lightmap,
                LIGHTMAP_ATLAS_EXTENT.width,
                LIGHTMAP_ATLAS_EXTENT.height,
                1,
            ),

            vertices: BufferCache::new(wgpu::BufferUsage::VERTEX),
            indices: BufferCache::new(wgpu::BufferUsage::INDEX),
        }
    }

    fn update(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.diffuse.update(device, encoder);
        self.lightmap.update(device, encoder);
        self.vertices.update(device, encoder);
        self.indices.update(device, encoder);
    }
}

pub trait Render {
    type Indices: Iterator<Item = Range<u32>>;

    fn indices(self, ctx: &RenderContext<'_>) -> (u64, Self::Indices);
}

pub trait DoRender {
    fn render<T>(&mut self, to_render: T)
    where
        T: Render;
}

pub struct Renderer {
    cache: RenderCache,

    matrix: wgpu::Buffer,
    atlas_sizes: wgpu::Buffer,
    pipeline: Pipeline,
}

// TODO: Do this better somehow
const DIFFUSE_ATLAS_EXTENT: wgpu::Extent3d = wgpu::Extent3d {
    width: 2048,
    height: 2048,
    depth: 1,
};
const LIGHTMAP_ATLAS_EXTENT: wgpu::Extent3d = wgpu::Extent3d {
    width: 2048,
    height: 2048,
    depth: 1,
};

#[derive(Debug, Copy, Clone)]
pub struct InitRendererError;

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub pos: [f32; 4],
    pub tex: [f32; 4],
    pub tex_coord: [f32; 2],
    pub value: f32,
    // TODO: Can we split these into a separate vertex to save space?
    //       Not all vertices need it (unlike `tex_coord`).
    //       `tex` is also unnecessary for skybox faces and faces whose textures
    //       do not repeat (maybe true for models?).
    pub lightmap_coord: [f32; 2],
    pub lightmap_width: f32,
    pub lightmap_count: u32,
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

/// Get the vertices per-cluster. First element is the vertices for cluster 0, then for cluster 1, and so
/// forth.
// TODO: Maybe generate the indices-per-cluster lazily to reduce GPU memory pressure? We can recalculate
//       _only_ the indices for the cluster we're entering when we change clusters.
pub struct RenderContext<'a> {
    pub renderer: &'a Renderer,
    pub camera: &'a Camera,
    rpass: wgpu::RenderPass<'a>,
}

impl DoRender for RenderContext<'_> {
    fn render<T>(&mut self, to_render: T)
    where
        T: Render,
    {
        use std::convert::TryInto;

        // Yeah I know this is weird but it's basically free and it makes the output way
        // easier to debug in renderdoc
        struct MergeRanges<I> {
            ranges: I,
        }

        impl<I> Iterator for MergeRanges<std::iter::Peekable<I>>
        where
            I: Iterator<Item = Range<u32>>,
        {
            type Item = Range<u32>;

            fn next(&mut self) -> Option<Self::Item> {
                let mut cur = self.ranges.next()?;

                loop {
                    let next: Option<Range<u32>> = self.ranges.peek().cloned();

                    match next {
                        Some(next) if next.start == cur.end => {
                            self.ranges.next().unwrap();
                            cur = cur.start..next.end;
                        }
                        _ => return Some(cur),
                    }
                }
            }
        }

        let (offset, ranges) = to_render.indices(&self);
        let ranges = MergeRanges {
            ranges: ranges.peekable(),
        };

        for range in ranges {
            self.rpass
                .draw_indexed(range, offset.try_into().unwrap(), 0..1);
        }
    }
}

impl Renderer {
    pub fn init(device: &wgpu::Device, gamma: f32, intensity: f32) -> Self {
        let cache = RenderCache::new(device);

        let diffuse_atlas_view = cache.diffuse.texture_view();
        let lightmap_atlas_view = cache.lightmap.texture_view();

        let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler_diffuse"),
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
            label: Some("sampler_lightmap"),
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

        let matrix = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mat4_viewmatrix"),
            size: std::mem::size_of::<[f32; 16]>() as u64,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let atlas_sizes = device.create_buffer_with_data(
            bytemuck::cast_slice(&[
                cache.diffuse.width() as f32,
                cache.diffuse.height() as f32,
                cache.lightmap.width() as f32,
                cache.lightmap.height() as f32,
                1. / gamma,
                intensity,
            ]),
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        );

        // Create pipeline layout
        let pipeline = self::pipelines::world::build(
            device,
            &diffuse_atlas_view,
            &lightmap_atlas_view,
            &diffuse_sampler,
            &lightmap_sampler,
            &matrix,
            &atlas_sizes,
        );

        Self {
            matrix,
            atlas_sizes,
            pipeline,
            cache,
        }
    }

    pub fn cache_mut(&mut self) -> &mut RenderCache {
        &mut self.cache
    }

    pub fn cache(&self) -> &RenderCache {
        &self.cache
    }

    pub fn render<F>(
        &mut self,
        device: &wgpu::Device,
        camera: &Camera,
        (image, depth): (&wgpu::TextureView, &wgpu::TextureView),
        queue: &wgpu::Queue,
        render: F,
        label: Option<&str>,
    ) -> Option<wgpu::CommandBuffer>
    where
        F: FnOnce(RenderContext<'_>),
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label });

        self.cache.update(device, &mut encoder);

        if let (Some(vertices), Some(indices)) = (&*self.cache.vertices, &*self.cache.indices) {
            let matrix = camera.matrix();
            let matrix: &[f32; 16] = matrix.as_ref();
            queue.write_buffer(&self.matrix, 0, bytemuck::cast_slice(matrix));

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: image,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Load,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: depth,
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_depth: 1.0,
                    clear_stencil: 0,
                }),
            });

            rpass.set_pipeline(&self.pipeline.pipeline);
            rpass.set_bind_group(0, &self.pipeline.bind_group, &[]);
            rpass.set_vertex_buffer(0, vertices.slice(..));
            rpass.set_index_buffer(indices.slice(..));

            render(RenderContext {
                renderer: self,
                camera,
                rpass,
            })
        }

        Some(encoder.finish())
    }
}
