use crate::{
    cache::{AlignedBufferCache, Atlas, BufferCache, BufferCacheMut, CacheCommon},
    gui, kawase,
};
use bytemuck::{Pod, Zeroable};
use std::{convert::TryFrom, iter, mem, num::NonZeroU8, ops::Range};
use wgpu::util::DeviceExt;

pub mod pipelines;

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

    pub fn translation(&self) -> cgmath::Matrix4<f32> {
        use cgmath::Matrix4;
        Matrix4::from_translation(-self.position)
    }

    pub fn view(&self) -> cgmath::Matrix3<f32> {
        use cgmath::Matrix3;

        Matrix3::from_angle_y(-self.pitch) * Matrix3::from_angle_z(-self.yaw)
    }

    pub fn projection(&self) -> cgmath::Matrix4<f32> {
        use cgmath::Matrix4;

        #[cfg_attr(rustfmt, rustfmt_skip)]
        const QUAKE_TO_OPENGL_TRANSFORMATION_MATRIX: Matrix4<f32> = Matrix4::new(
            0.0,0.0,  -1.0, 0.0,
            -1.0, 0.0, 0.0, 0.0,
            0.0, 1.0,  0.0, 0.0,
            0.0, 0.0,  0.0, 1.0,
        );

        let view = QUAKE_TO_OPENGL_TRANSFORMATION_MATRIX * Matrix4::from(self.view());
        let projection = cgmath::perspective(self.vertical_fov, self.aspect_ratio, 1., 4096.);
        OPENGL_TO_WGPU_MATRIX * projection * view
    }

    pub fn matrix(&self) -> cgmath::Matrix4<f32> {
        self.projection() * self.translation()
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct ModelData {
    pub translation: cgmath::Matrix4<f32>,
    pub origin: cgmath::Vector3<f32>,
}

unsafe impl Pod for ModelData {}
unsafe impl Zeroable for ModelData {}

type BoneMatrix = [[f32; 4]; 4];

pub struct RenderCache {
    pub diffuse: Atlas,
    pub lightmap: Atlas,

    pub model_vertices: BufferCache<ModelVertex>,
    pub textured_vertices: BufferCache<TexturedVertex>,
    pub world_vertices: BufferCache<WorldVertex>,

    pub indices: BufferCache<u32>,

    pub model_data: AlignedBufferCache<ModelData>,
    pub bone_matrices: BufferCacheMut<BoneMatrix>,
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

            model_vertices: BufferCache::new(wgpu::BufferUsage::VERTEX),
            textured_vertices: BufferCache::new(wgpu::BufferUsage::VERTEX),
            world_vertices: BufferCache::new(wgpu::BufferUsage::VERTEX),

            indices: BufferCache::new(wgpu::BufferUsage::INDEX),

            model_data: AlignedBufferCache::new(
                wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
                MINIMUM_ALIGNMENT as u16,
            ),
            bone_matrices: BufferCacheMut::new(
                // TODO: We need `MAP_WRITE` but it's not possible to also have that be a uniform
                wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            ),
        }
    }

    async fn update_buffers(&mut self) -> Result<(), wgpu::BufferAsyncError> {
        let (
            diffuse_res,
            lightmap_res,
            textured_vertices_res,
            world_vertices_res,
            model_vertices_res,
            indices_res,
            model_data_res,
            bone_matrices_res,
        ) = futures::join!(
            self.diffuse.update_buffers(),
            self.lightmap.update_buffers(),
            self.textured_vertices.update_buffers(),
            self.world_vertices.update_buffers(),
            self.model_vertices.update_buffers(),
            self.indices.update_buffers(),
            self.model_data.update_buffers(),
            self.bone_matrices.update_buffers(),
        );

        diffuse_res?;
        lightmap_res?;
        textured_vertices_res?;
        world_vertices_res?;
        model_vertices_res?;
        indices_res?;
        model_data_res?;
        bone_matrices_res?;

        Ok(())
    }

    fn update(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.diffuse.update(device, encoder);
        self.lightmap.update(device, encoder);

        self.textured_vertices.update(device, encoder);
        self.world_vertices.update(device, encoder);
        self.model_vertices.update(device, encoder);

        self.indices.update(device, encoder);

        self.model_data.update(device, encoder);
        self.bone_matrices.update(device, encoder);
    }
}

pub struct VertexOffset<T> {
    pub id: u64,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Clone for VertexOffset<T> {
    fn clone(&self) -> Self {
        self.id.into()
    }
}

impl<T> Copy for VertexOffset<T> {}

impl<T> Default for VertexOffset<T> {
    fn default() -> Self {
        Self {
            id: Default::default(),
            _marker: Default::default(),
        }
    }
}

impl<T> From<u64> for VertexOffset<T> {
    fn from(other: u64) -> Self {
        Self {
            id: other,
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(Clone, Default)]
pub struct RenderOffsets(
    pub Option<VertexOffset<TexturedVertex>>,
    pub Option<VertexOffset<WorldVertex>>,
    pub Option<VertexOffset<ModelVertex>>,
);

impl From<VertexOffset<TexturedVertex>> for RenderOffsets {
    fn from(other: VertexOffset<TexturedVertex>) -> Self {
        Self(Some(other), None, None)
    }
}

impl From<VertexOffset<WorldVertex>> for RenderOffsets {
    fn from(other: VertexOffset<WorldVertex>) -> Self {
        Self(None, Some(other), None)
    }
}

impl From<VertexOffset<ModelVertex>> for RenderOffsets {
    fn from(other: VertexOffset<ModelVertex>) -> Self {
        Self(None, None, Some(other))
    }
}

impl From<(VertexOffset<TexturedVertex>, VertexOffset<WorldVertex>)> for RenderOffsets {
    fn from(other: (VertexOffset<TexturedVertex>, VertexOffset<WorldVertex>)) -> Self {
        Self(Some(other.0), Some(other.1), None)
    }
}

impl From<(VertexOffset<TexturedVertex>, VertexOffset<ModelVertex>)> for RenderOffsets {
    fn from(other: (VertexOffset<TexturedVertex>, VertexOffset<ModelVertex>)) -> Self {
        Self(Some(other.0), None, Some(other.1))
    }
}

#[derive(Clone)]
pub struct RenderMesh<O, I> {
    pub offsets: O,
    pub indices: I,
    pub pipeline: PipelineDesc,
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum PipelineDesc {
    Skybox,
    World,
    Models {
        origin: cgmath::Vector3<f32>,
        data_offset: wgpu::BufferAddress,
        bone_offset: wgpu::BufferAddress,
    },
}

pub trait Render {
    type Indices: Iterator<Item = Range<u32>> + Clone;
    type Offsets: Into<RenderOffsets>;

    fn is_visible<T: World>(&self, _world: &T) -> bool {
        true
    }

    fn indices<T: Context>(self, ctx: &T) -> RenderMesh<Self::Offsets, Self::Indices>;
}

pub trait TransferData {
    fn transfer_data(self) -> Option<(wgpu::BufferAddress, ModelData)>;
}

pub trait World {
    fn is_visible(&self, _camera: &Camera, _position: cgmath::Vector3<f32>) -> bool {
        true
    }
}

impl World for () {}

struct MsaaTexture {
    factor: NonZeroU8,
    diffuse_buffer: wgpu::TextureView,
}

pub struct Uniforms<T> {
    dirty: bool,
    buffer: wgpu::Buffer,
    value: T,
}

impl<T> Uniforms<T>
where
    T: bytemuck::Pod + std::cmp::PartialEq + Copy,
{
    pub fn new(value: T, device: &wgpu::Device) -> Self {
        Self {
            value,
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of::<T>() as u64,
                usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }),
            dirty: true,
        }
    }

    pub fn set(&mut self, value: T) {
        if self.value != value {
            self.value = value;
            self.dirty = true;
        }
    }

    pub fn get(&self) -> &T {
        &self.value
    }

    pub fn update(&mut self, f: impl FnOnce(&mut T)) {
        let mut value = self.value;
        f(&mut value);

        self.set(value);
    }

    pub fn write_if_dirty(&mut self, queue: &wgpu::Queue) {
        if self.dirty {
            self.dirty = false;
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[self.value]));
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

pub struct Renderer {
    cache: RenderCache,
    matrices_buffer: wgpu::Buffer,
    fragment_uniforms: Uniforms<FragmentUniforms>,
    post_uniforms: Uniforms<PostUniforms>,
    hipass_uniforms: Uniforms<HipassUniforms>,
    out_size: (u32, u32),
    framebuffer_size: (u32, u32),
    nearest_sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,
    kawase_blur: Option<kawase::Blur>,
    world_pipeline: Option<pipelines::world::Pipeline>,
    skybox_pipeline: Option<pipelines::sky::Pipeline>,
    model_pipeline: Option<pipelines::models::Pipeline>,
    post_pipeline: Option<pipelines::postprocess::Pipeline>,
    fxaa_pipeline: Option<pipelines::fxaa::Pipeline>,
    hipass_pipeline: Option<pipelines::hipass::Pipeline>,
    /// In order to upscale when rendering at less than swapchain resolution
    passthrough_pipeline: Option<pipelines::passthrough::Pipeline>,
    msaa_factor: NonZeroU8,
    bloom_enabled: bool,
    bloom_downscale: u8,
    bloom_factor: f32,
    bloom_iterations: usize,
    bloom_radius: f32,
    bloom_texture: Option<wgpu::TextureView>,
    post_texture: Option<MsaaTexture>,
    fxaa_texture: Option<wgpu::TextureView>,
    post_verts: wgpu::Buffer,
    depth_buffer: Option<wgpu::TextureView>,
    final_texture: Option<wgpu::TextureView>,
}

// TODO: Do this better somehow
const DIFFUSE_ATLAS_EXTENT: wgpu::Extent3d = wgpu::Extent3d {
    width: 2048,
    height: 2048,
    depth: 1,
};
// TODO: Use multiple atlases in a texture array
const LIGHTMAP_ATLAS_EXTENT: wgpu::Extent3d = wgpu::Extent3d {
    width: 4096,
    height: 4096,
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

const BONES_PER_VERTEX: usize = 1;

#[derive(Debug, Clone, Copy)]
pub struct ModelVertex {
    pub normal: [f32; 3],
    pub bone_id: u32,
}

unsafe impl Pod for ModelVertex {}
unsafe impl Zeroable for ModelVertex {}

#[derive(Debug, Clone, Copy)]
pub struct TexturedVertex {
    pub pos: [f32; 3],
    /// For world vertices, this starts at zero and must be added to `WorldVertex::atlas_texture.xy`,
    /// for vertices of textures which don't have wrapping implemented in a shader (models and
    /// skyboxes) this is the absolute coord. We might change this to be consistent in the future,
    /// especially if we want to support wrapping everywhere.
    pub tex_coord: [f32; 2],
    /// The coordinates and size of the texture in the atlas, so we can do wrapping
    /// in the shader while still using our fake megatexture thing.
    pub atlas_texture: [u32; 4],
}

unsafe impl Pod for TexturedVertex {}
unsafe impl Zeroable for TexturedVertex {}

#[derive(Debug, Clone, Copy)]
pub struct PostVertex {
    pub pos: [f32; 2],
}

unsafe impl Pod for PostVertex {}
unsafe impl Zeroable for PostVertex {}

#[derive(Debug, Clone, Copy)]
pub struct WorldVertex {
    /// For animated textures (TODO: We can split this out even further since 99% of faces have
    /// non-animated textures)
    pub count: i32,
    pub texture_stride: u32,
    pub lightmap_coord: [f32; 2],
    pub lightmap_stride: f32,
    pub lightmap_count: u32,
    pub value: f32,
}

unsafe impl Pod for WorldVertex {}
unsafe impl Zeroable for WorldVertex {}

/// Get the vertices per-cluster. First element is the vertices for cluster 0, then for cluster 1, and so
/// forth.
// TODO: Maybe generate the indices-per-cluster lazily to reduce GPU memory pressure? We can recalculate
//       _only_ the indices for the cluster we're entering when we change clusters.
pub struct RenderContext<'pass, W = ()> {
    pub renderer: &'pass Renderer,
    pub camera: &'pass Camera,

    world: W,
    cur_pipeline: Option<PipelineDesc>,
    rpass: wgpu::RenderPass<'pass>,
}

pub trait Context {
    fn camera(&self) -> &Camera;
}

impl<W> Context for RenderContext<'_, W> {
    fn camera(&self) -> &Camera {
        self.camera
    }
}

impl<'pass, W> RenderContext<'pass, W>
where
    W: World + Copy,
{
    pub fn with_world<NewWorld: World>(self, world: NewWorld) -> RenderContext<'pass, NewWorld> {
        RenderContext {
            renderer: self.renderer,
            camera: self.camera,
            world,
            cur_pipeline: self.cur_pipeline,
            rpass: self.rpass,
        }
    }

    pub fn render<T>(&mut self, to_render: T)
    where
        T: Render,
    {
        // Yeah I know this is weird but it's basically free and it makes the output way
        // easier to debug in renderdoc
        #[derive(Clone)]
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

        let RenderMesh {
            offsets,
            indices,
            pipeline,
        } = to_render.indices(&*self);
        let ranges = MergeRanges {
            ranges: indices.peekable(),
        };

        let RenderOffsets(tex_o, world_o, norm_o) = offsets.into();

        if self.cur_pipeline != Some(pipeline) {
            self.cur_pipeline = Some(pipeline);

            let pipeline = match pipeline {
                PipelineDesc::Skybox => {
                    let pipeline = self.renderer.skybox_pipeline.as_ref().unwrap();

                    self.rpass.set_bind_group(0, &pipeline.bind_group, &[]);

                    &pipeline.pipeline
                }
                PipelineDesc::World => {
                    let pipeline = self.renderer.world_pipeline.as_ref().unwrap();

                    self.rpass.set_bind_group(0, &pipeline.bind_group, &[]);

                    &pipeline.pipeline
                }
                PipelineDesc::Models {
                    data_offset,
                    bone_offset,
                    ..
                } => {
                    let pipeline = self.renderer.model_pipeline.as_ref().unwrap();

                    self.rpass.set_bind_group(
                        0,
                        &pipeline.bind_group,
                        &[
                            u32::try_from(data_offset).unwrap(),
                            u32::try_from(bone_offset).unwrap()
                                * mem::size_of::<BoneMatrix>() as u32,
                        ],
                    );

                    &pipeline.pipeline
                }
            };

            self.rpass.set_pipeline(&pipeline);
        }

        if let Some(verts) = &*self.renderer.cache().textured_vertices {
            self.rpass.set_vertex_buffer(
                0,
                verts.slice(tex_o.unwrap().id * mem::size_of::<TexturedVertex>() as u64..),
            );
        } else {
            return;
        }

        if let Some(world_o) = world_o {
            if let Some(verts) = &*self.renderer.cache().world_vertices {
                self.rpass.set_vertex_buffer(
                    1,
                    verts.slice(world_o.id * mem::size_of::<WorldVertex>() as u64..),
                );
            } else {
                return;
            }
        } else if let Some(norm_o) = norm_o {
            if let Some(verts) = &*self.renderer.cache().model_vertices {
                self.rpass.set_vertex_buffer(
                    1,
                    verts.slice(norm_o.id * mem::size_of::<ModelVertex>() as u64..),
                );
            } else {
                return;
            }
        }

        for range in ranges {
            self.rpass.draw_indexed(range, 0, 0..1);
        }
    }
}

/// The view and projection matrices_buffer, used by the vertex shader.
/// Since we only split these out to make implementing the sky
/// shader simpler, the "projection matrix" actually includes
/// rotation - we just don't do translation.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Matrices {
    /// Translation, in the Quake coordinate system
    translation: cgmath::Matrix4<f32>,
    /// The rest of the view/projection matrix
    projection: cgmath::Matrix4<f32>,
}

bitflags::bitflags! {
    #[derive(Default)]
    pub struct PostFlags: u32 {
        const TONEMAPPING = 0b0000_0001;
        const XYY_ACES = 0b0000_0010;
        const CROSSTALK = 0b0000_0100;
        const XYY_CROSSTALK = 0b0000_1000;
        const BLOOM = 0b0001_0000;
        const FXAA_ENABLED = 0b0010_0000;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct PostUniforms {
    flags: PostFlags,
    inv_crosstalk_amt: f32,
    saturation: f32,
    crosstalk_saturation: f32,
    bloom_influence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct HipassUniforms {
    cutoff: f32,
    intensity: f32,
}

/// The uniforms used by the
#[derive(Debug, Clone, PartialEq, Copy)]
#[repr(C)]
struct FragmentUniforms {
    /// The amount to exponentiate the output colour by
    inv_gamma: f32,
    /// The amount to multiply the output colour by
    intensity: f32,
    /// To get the x coord of the current texture, do `texture.x + (animation frame % count) * texture.width`
    animation_frame: f32,
    /// Level of ambient light, for model shading
    ambient_light: f32,
}

unsafe impl Pod for Matrices {}
unsafe impl Zeroable for Matrices {}

unsafe impl Pod for PostUniforms {}
unsafe impl Zeroable for PostUniforms {}

unsafe impl Pod for FragmentUniforms {}
unsafe impl Zeroable for FragmentUniforms {}

unsafe impl Pod for HipassUniforms {}
unsafe impl Zeroable for HipassUniforms {}

pub const MAX_LIGHTS: usize = 128;
const MINIMUM_ALIGNMENT: usize = 256;

const DEFAULT_BLOOM_CUTOFF: f32 = 0.5;
const DEFAULT_BLOOM_RADIUS: f32 = 1.0;
const DEFAULT_BLOOM_INFLUENCE: f32 = 0.3;
const DEFAULT_BLOOM_DOWNSCALE_FACTOR: f32 = 1.7;
const DEFAULT_BLOOM_ITERATIONS: usize = 2;
const DEFAULT_BLOOM_DOWNSCALE: u8 = 1;

impl Renderer {
    pub fn init(
        device: &wgpu::Device,
        out_size: (u32, u32),
        framebuffer_size: (u32, u32),
        gamma: f32,
        intensity: f32,
    ) -> Self {
        let cache = RenderCache::new(device);

        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler_diffuse"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: None,
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler_diffuse_blurred"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: None,
        });

        let matrices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mat4_viewmatrix"),
            size: std::mem::size_of::<Matrices>() as u64,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        let fragment_uniforms = Uniforms::new(
            FragmentUniforms {
                inv_gamma: gamma.recip(),
                intensity,
                animation_frame: 0.,
                ambient_light: 0.,
            },
            device,
        );

        let post_uniforms = Uniforms::new(
            PostUniforms {
                flags: PostFlags::TONEMAPPING
                    | PostFlags::CROSSTALK
                    | PostFlags::BLOOM
                    | PostFlags::XYY_ACES
                    | PostFlags::XYY_CROSSTALK
                    | PostFlags::FXAA_ENABLED,
                inv_crosstalk_amt: 1.0,
                saturation: 1.1,
                crosstalk_saturation: 2.,
                bloom_influence: DEFAULT_BLOOM_INFLUENCE,
            },
            device,
        );

        let hipass_uniforms = Uniforms::new(
            HipassUniforms {
                cutoff: DEFAULT_BLOOM_CUTOFF,
                intensity,
            },
            device,
        );

        let post_verts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[
                PostVertex { pos: [-1., -1.] },
                PostVertex { pos: [1., -1.] },
                PostVertex { pos: [1., 1.] },
                PostVertex { pos: [1., 1.] },
                PostVertex { pos: [-1., 1.] },
                PostVertex { pos: [-1., -1.] },
            ]),
            usage: wgpu::BufferUsage::VERTEX,
        });

        Self {
            out_size,
            framebuffer_size,
            matrices_buffer,
            fragment_uniforms,
            hipass_uniforms,
            post_uniforms,
            kawase_blur: None,
            hipass_pipeline: None,
            world_pipeline: None,
            skybox_pipeline: None,
            model_pipeline: None,
            post_pipeline: None,
            fxaa_pipeline: None,
            passthrough_pipeline: None,
            nearest_sampler,
            linear_sampler,
            cache,
            msaa_factor: NonZeroU8::new(1).unwrap(),
            bloom_enabled: true,
            bloom_downscale: DEFAULT_BLOOM_DOWNSCALE,
            bloom_iterations: DEFAULT_BLOOM_ITERATIONS,
            bloom_factor: DEFAULT_BLOOM_DOWNSCALE_FACTOR,
            bloom_radius: DEFAULT_BLOOM_RADIUS,
            bloom_texture: None,
            post_texture: None,
            fxaa_texture: None,
            final_texture: None,
            post_verts,
            depth_buffer: None,
        }
    }

    pub fn set_size(&mut self, size: (u32, u32)) {
        if size != self.out_size {
            self.out_size = size;
            self.post_texture = None;
            self.depth_buffer = None;
            self.final_texture = None;
        }
    }

    pub fn set_framebuffer_size(&mut self, size: (u32, u32)) {
        if size != self.framebuffer_size {
            self.final_texture = None;
            self.framebuffer_size = size;
        }
    }

    pub fn set_msaa_factor(&mut self, factor: u8) {
        self.msaa_factor = NonZeroU8::new(factor).unwrap();
        if self
            .post_texture
            .as_ref()
            .map(|buf| buf.factor)
            .unwrap_or(NonZeroU8::new(1).unwrap())
            != self.msaa_factor
        {
            self.post_texture = None;
            self.depth_buffer = None;
        }
    }

    pub fn set_flag_bit(&mut self, mask: PostFlags, enabled: bool) {
        self.post_uniforms.update(|uniforms| {
            uniforms.flags &= !mask;

            if enabled {
                uniforms.flags |= mask;
            }
        });
    }

    pub fn update_config(&mut self, f: impl FnOnce(&mut gui::Config)) {
        let mut config = gui::Config {
            gamma: self.gamma(),
            intensity: self.intensity(),
            fxaa: self.fxaa_enabled(),
            tonemapping: gui::Tonemapping {
                enabled: self
                    .post_uniforms
                    .get()
                    .flags
                    .contains(PostFlags::TONEMAPPING),
                xyy_aces: self.post_uniforms.get().flags.contains(PostFlags::XYY_ACES),
                crosstalk: self
                    .post_uniforms
                    .get()
                    .flags
                    .contains(PostFlags::CROSSTALK),
                xyy_crosstalk: self
                    .post_uniforms
                    .get()
                    .flags
                    .contains(PostFlags::XYY_CROSSTALK),
                crosstalk_amt: self.crosstalk_amount(),
                saturation: self.post_uniforms.get().saturation,
                crosstalk_saturation: self.post_uniforms.get().crosstalk_saturation,
            },
            bloom: gui::Bloom {
                enabled: self.bloom_enabled,
                radius: self.bloom_radius,
                cutoff: self.hipass_uniforms.get().cutoff,
                downscale: self.bloom_downscale as i32,
                factor: self.bloom_factor,
                iterations: self.bloom_iterations as i32,
                influence: self.post_uniforms.get().bloom_influence,
            },
        };

        f(&mut config);

        self.set_gamma(config.gamma);
        self.set_intensity(config.intensity);

        self.bloom_enabled = config.bloom.enabled;
        self.set_flag_bit(PostFlags::BLOOM, config.bloom.enabled);

        self.bloom_radius = config.bloom.radius;
        if let Some(blur) = self.kawase_blur.as_mut() {
            blur.set_radius(self.bloom_radius);
        }

        self.hipass_uniforms
            .update(|uniforms| uniforms.cutoff = config.bloom.cutoff);
        self.post_uniforms.update(|uniforms| {
            uniforms.bloom_influence = config.bloom.influence;
        });

        self.set_crosstalk_amount(config.tonemapping.crosstalk_amt);
        self.set_flag_bit(PostFlags::TONEMAPPING, config.tonemapping.enabled);
        self.set_flag_bit(PostFlags::XYY_ACES, config.tonemapping.xyy_aces);
        self.set_flag_bit(PostFlags::CROSSTALK, config.tonemapping.crosstalk);
        self.set_flag_bit(PostFlags::XYY_CROSSTALK, config.tonemapping.xyy_crosstalk);
        self.set_flag_bit(PostFlags::FXAA_ENABLED, config.fxaa);

        self.post_uniforms.update(|uniforms| {
            uniforms.saturation = config.tonemapping.saturation;
            uniforms.crosstalk_saturation = config.tonemapping.crosstalk_saturation;
        });

        if config.bloom.factor != self.bloom_factor
            || config.bloom.iterations as usize != self.bloom_iterations
            || config.bloom.downscale as u8 != self.bloom_downscale
        {
            self.kawase_blur = None;
            self.bloom_factor = config.bloom.factor;
            self.bloom_iterations = config.bloom.iterations as usize;
            self.bloom_downscale = config.bloom.downscale as u8;
        }
    }

    pub fn msaa_factor(&self) -> u8 {
        self.msaa_factor.get()
    }

    pub fn set_gamma(&mut self, gamma: f32) {
        self.fragment_uniforms
            .update(|uniforms| uniforms.inv_gamma = gamma.recip());
    }

    pub fn gamma(&self) -> f32 {
        self.fragment_uniforms.get().inv_gamma.recip()
    }

    pub fn set_crosstalk_amount(&mut self, crosstalk_amt: f32) {
        self.post_uniforms
            .update(|uniforms| uniforms.inv_crosstalk_amt = crosstalk_amt.recip());
    }

    pub fn crosstalk_amount(&mut self) -> f32 {
        self.post_uniforms.get().inv_crosstalk_amt.recip()
    }

    pub fn set_intensity(&mut self, intensity: f32) {
        self.fragment_uniforms
            .update(|uniforms| uniforms.intensity = intensity);
        self.hipass_uniforms
            .update(|uniforms| uniforms.intensity = intensity);
    }

    pub fn intensity(&self) -> f32 {
        self.fragment_uniforms.get().intensity
    }

    pub fn fxaa_enabled(&self) -> bool {
        self.post_uniforms
            .get()
            .flags
            .contains(PostFlags::FXAA_ENABLED)
    }

    pub fn set_fxaa_enabled(&mut self, enabled: bool) {
        self.set_flag_bit(PostFlags::FXAA_ENABLED, enabled);
    }

    fn make_depth_tex(&self, device: &wgpu::Device) -> wgpu::TextureView {
        let depth_texture_desc = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: self.out_size.0,
                height: self.out_size.1,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: self.msaa_factor() as u32,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            label: Some("tex_depth"),
        };

        let depth_texture = device.create_texture(&depth_texture_desc);

        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn make_pipelines(&mut self, device: &wgpu::Device) {
        let diffuse_atlas_view = self.cache.diffuse.texture_view();
        let lightmap_atlas_view = self.cache.lightmap.texture_view();

        if self.world_pipeline.is_none() {
            self.world_pipeline = Some(pipelines::world::build(
                device,
                &diffuse_atlas_view,
                &lightmap_atlas_view,
                &self.nearest_sampler,
                &self.linear_sampler,
                &self.matrices_buffer,
                &self.fragment_uniforms.buffer(),
                self.msaa_factor.get() as u32,
            ));
        }

        if self.skybox_pipeline.is_none() {
            self.skybox_pipeline = Some(pipelines::sky::build(
                device,
                &diffuse_atlas_view,
                &self.linear_sampler,
                &self.matrices_buffer,
                &self.fragment_uniforms.buffer(),
                self.msaa_factor.get() as u32,
            ));
        }

        if self.model_pipeline.is_none() {
            self.model_pipeline = Some(pipelines::models::build(
                device,
                &diffuse_atlas_view,
                &self.linear_sampler,
                &self.matrices_buffer,
                self.cache.model_data.as_ref().unwrap(),
                self.cache.bone_matrices.as_ref().unwrap(),
                self.msaa_factor.get() as u32,
            ));
        }

        if self.fxaa_pipeline.is_none() {
            self.fxaa_pipeline = Some(pipelines::fxaa::build(
                device,
                self.fxaa_texture.as_ref().unwrap(),
                &self.linear_sampler,
            ));
        }

        if self.final_texture.is_none() && self.out_size != self.framebuffer_size {
            self.final_texture = Some(
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("final_texture"),
                        size: wgpu::Extent3d {
                            width: self.out_size.0,
                            height: self.out_size.1,
                            depth: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: pipelines::SWAPCHAIN_FORMAT,
                        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::OUTPUT_ATTACHMENT,
                    })
                    .create_view(&Default::default()),
            );

            self.passthrough_pipeline = Some(pipelines::passthrough::build(
                device,
                self.final_texture.as_ref().unwrap(),
                &self.nearest_sampler,
            ));
        }

        if self.kawase_blur.is_none() {
            let blur = kawase::Blur::new(
                device,
                self.bloom_texture.as_ref().unwrap(),
                &self.linear_sampler,
                self.bloom_factor as f64,
                self.bloom_iterations,
                self.bloom_downscale,
                self.bloom_radius,
                self.out_size,
            );

            let MsaaTexture { diffuse_buffer, .. } = self.post_texture.as_ref().unwrap();

            self.post_pipeline = Some(pipelines::postprocess::build(
                device,
                diffuse_buffer,
                blur.output(),
                &self.linear_sampler,
                &self.post_uniforms.buffer(),
                &self.fragment_uniforms.buffer(),
            ));

            self.kawase_blur = Some(blur);
        }
    }

    fn update_msaa_buffer(&mut self, device: &wgpu::Device) {
        if self.post_texture.as_ref().map(|buf| buf.factor) != Some(self.msaa_factor) {
            let (width, height) = self.out_size;

            let buffer = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("msaa_diffuse_buffer"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count: self.msaa_factor.get() as u32,
                dimension: wgpu::TextureDimension::D2,
                format: pipelines::postprocess::DIFFUSE_BUFFER_FORMAT,
                usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            });

            self.post_texture = Some(MsaaTexture {
                factor: self.msaa_factor,
                diffuse_buffer: buffer.create_view(&wgpu::TextureViewDescriptor::default()),
            });

            self.skybox_pipeline = None;
            self.world_pipeline = None;
            self.model_pipeline = None;

            let MsaaTexture { diffuse_buffer, .. } = self.post_texture.as_ref().unwrap();

            self.bloom_texture = Some(
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("bloom_buffer"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: pipelines::postprocess::DIFFUSE_BUFFER_FORMAT,
                        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::OUTPUT_ATTACHMENT,
                    })
                    .create_view(&Default::default()),
            );

            self.hipass_pipeline = Some(pipelines::hipass::build(
                device,
                diffuse_buffer,
                &self.linear_sampler,
                &self.hipass_uniforms.buffer(),
            ));

            self.kawase_blur = None;

            let fxaa_buffer = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fxaa_input_buffer"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: pipelines::SWAPCHAIN_FORMAT,
                usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            });

            self.fxaa_texture = Some(fxaa_buffer.create_view(&Default::default()));
            self.fxaa_pipeline = None;
        }
    }

    pub fn cache_mut(&mut self) -> &mut RenderCache {
        &mut self.cache
    }

    pub fn cache(&self) -> &RenderCache {
        &self.cache
    }

    pub fn set_time(&mut self, time: f32) {
        self.fragment_uniforms
            .update(|uniforms| uniforms.animation_frame = time);
        // HACK!!
        // self.model_pipeline = None;
    }

    pub fn transfer_data<I>(&mut self, queue: &wgpu::Queue, items: I)
    where
        I: IntoIterator,
        I::Item: TransferData,
    {
        if let Some(model_data) = self.cache.model_data.as_ref() {
            for i in items {
                if let Some((offset, data)) = i.transfer_data() {
                    // TODO: We can batch this
                    // ALSO TODO: Maybe this is batched by default already?
                    queue.write_buffer(&model_data, offset, bytemuck::bytes_of(&data));
                }
            }
        }
    }

    pub async fn prepare(&mut self) -> Result<(), wgpu::BufferAsyncError> {
        self.cache.update_buffers().await
    }

    pub fn render<'a, F>(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        camera: &Camera,
        screen_tex: &wgpu::TextureView,
        queue: &wgpu::Queue,
        render: F,
    ) where
        F: FnOnce(RenderContext<'_>),
    {
        self.cache.update(device, encoder);

        self.update_msaa_buffer(device);
        self.make_pipelines(device);

        let indices = if let Some(i) = self.cache.indices.as_ref() {
            i
        } else {
            return;
        };

        let translation = camera.translation();
        let projection = camera.projection();
        queue.write_buffer(
            &self.matrices_buffer,
            0,
            bytemuck::cast_slice(&[Matrices {
                translation,
                projection,
            }]),
        );

        self.fragment_uniforms.write_if_dirty(queue);
        self.hipass_uniforms.write_if_dirty(queue);
        self.post_uniforms.write_if_dirty(queue);

        if let Some(blur) = self.kawase_blur.as_mut() {
            blur.write_if_dirty(queue);
        }

        let depth = if let Some(depth) = &self.depth_buffer {
            depth
        } else {
            self.depth_buffer = Some(self.make_depth_tex(device));
            self.depth_buffer.as_ref().unwrap()
        };

        {
            let MsaaTexture { diffuse_buffer, .. } = self.post_texture.as_ref().unwrap();

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: diffuse_buffer,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: true,
                    }),
                }),
            });

            rpass.set_index_buffer(indices.slice(..));

            render(RenderContext {
                renderer: self,
                camera,
                rpass,
                world: (),
                cur_pipeline: None,
            });
        }

        if self.bloom_enabled {
            {
                let mut hipass_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: self.bloom_texture.as_ref().unwrap(),
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        },
                    }],
                    depth_stencil_attachment: None,
                });

                hipass_pass.set_pipeline(&self.hipass_pipeline.as_ref().unwrap().pipeline);
                hipass_pass.set_bind_group(
                    0,
                    &self.hipass_pipeline.as_ref().unwrap().bind_group,
                    &[],
                );
                hipass_pass.set_vertex_buffer(0, self.post_verts.slice(..));
                hipass_pass.draw(0..6, 0..1);
            }

            self.kawase_blur.as_ref().unwrap().blur(encoder);
        }

        let output_tex = if let Some(output) = &self.final_texture {
            output
        } else {
            screen_tex
        };

        {
            let mut post_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: if self.fxaa_enabled() {
                        self.fxaa_texture.as_ref().unwrap()
                    } else {
                        output_tex
                    },
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            post_pass.set_pipeline(&self.post_pipeline.as_ref().unwrap().pipeline);
            post_pass.set_bind_group(0, &self.post_pipeline.as_ref().unwrap().bind_group, &[]);
            post_pass.set_vertex_buffer(0, self.post_verts.slice(..));
            post_pass.draw(0..6, 0..1);
        }

        if self.fxaa_enabled() {
            let mut fxaa_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: output_tex,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            fxaa_pass.set_pipeline(&self.fxaa_pipeline.as_ref().unwrap().pipeline);
            fxaa_pass.set_bind_group(0, &self.fxaa_pipeline.as_ref().unwrap().bind_group, &[]);
            fxaa_pass.set_vertex_buffer(0, self.post_verts.slice(..));
            fxaa_pass.draw(0..6, 0..1);
        }

        if self.final_texture.is_some() {
            let mut passthrough_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: screen_tex,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            passthrough_pass.set_pipeline(&self.passthrough_pipeline.as_ref().unwrap().pipeline);
            passthrough_pass.set_bind_group(
                0,
                &self.passthrough_pipeline.as_ref().unwrap().bind_group,
                &[],
            );
            passthrough_pass.set_vertex_buffer(0, self.post_verts.slice(..));
            passthrough_pass.draw(0..6, 0..1);
        }
    }
}
