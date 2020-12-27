pub trait ShaderModuleSourceExt<'a> {
    fn as_ref<'b: 'a>(&'b self) -> wgpu::ShaderModuleSource<'b>;
}

impl<'a> ShaderModuleSourceExt<'a> for wgpu::ShaderModuleSource<'a> {
    fn as_ref<'b: 'a>(&'b self) -> wgpu::ShaderModuleSource<'b> {
        match self {
            Self::SpirV(data) => Self::SpirV(std::borrow::Cow::Borrowed(&**data)),
            Self::Wgsl(data) => Self::Wgsl(std::borrow::Cow::Borrowed(&**data)),
        }
    }
}

pub struct Pipeline<BindGroup = wgpu::BindGroup> {
    pub bind_group: BindGroup,
    pub pipeline: wgpu::RenderPipeline,
}

#[derive(Default, Clone)]
struct BindId {
    cur: u16,
}

impl BindId {
    fn next<T>(&mut self) -> T
    where
        T: From<u16>,
    {
        let out = self.cur;
        self.cur += 1;
        out.into()
    }
}

pub const WINDING_MODE: wgpu::FrontFace = wgpu::FrontFace::Ccw;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The world pipeline (unlike the skybox pipeline) does proper depth testing, although doesn't
/// handle transparency right now. It takes a lightmap atlas and a diffuse texture atlas, and
/// handles animated textures with the `animation_frame` property of the `FragmentUniforms`.
///
/// It can handle multiple lightmaps, as long as they are contiguous along the X axis of the
/// lightmap atlas. It's possible that in the future I will use separate lightmap atlases, but
/// right now that would be horribly inefficient as I have no way to expand the size of atlases,
/// so I'd end up with most of these lightmap atlases being empty (as most faces will only
/// have a single lightmap style). The number of light _styles_ (i.e. sets of lights that can
/// independently change intensity) affecting each face in the map has a fixed maximum, so it
/// would be relatively easy to implement this in hardware.
///
/// It renders directly to the MSAA/post-processing buffer.
pub mod world {
    use super::ShaderModuleSourceExt;
    use lazy_static::lazy_static;

    use super::BindId;
    pub use super::Pipeline;

    use crate::render::{TexturedVertex, WorldVertex};
    use memoffset::offset_of;
    use std::mem;

    lazy_static! {
        static ref VERTEX_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/world.vert.spv"));
        static ref FRAGMENT_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/world.frag.spv"));
    }

    pub fn build(
        device: &wgpu::Device,
        diffuse_atlas_view: &wgpu::TextureView,
        lightmap_atlas_view: &wgpu::TextureView,
        diffuse_sampler: &wgpu::Sampler,
        lightmap_sampler: &wgpu::Sampler,
        matrices: &wgpu::Buffer,
        fragment_uniforms: &wgpu::Buffer,
        sample_count: u32,
    ) -> Pipeline {
        let vs_module = device.create_shader_module(VERTEX_SHADER.as_ref());
        let fs_module = device.create_shader_module(FRAGMENT_SHADER.as_ref());

        let mut ids = BindId::default();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bindgrouplayout_world"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout_world"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let mut ids = BindId::default();
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: super::WINDING_MODE,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: super::postprocess::DIFFUSE_BUFFER_FORMAT,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: super::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilStateDescriptor {
                    front: wgpu::StencilStateFaceDescriptor::IGNORE,
                    back: wgpu::StencilStateFaceDescriptor::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
            }),
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint32,
                vertex_buffers: &[
                    {
                        wgpu::VertexBufferDescriptor {
                            stride: mem::size_of::<TexturedVertex>() as wgpu::BufferAddress,
                            step_mode: wgpu::InputStepMode::Vertex,
                            attributes: &[
                                wgpu::VertexAttributeDescriptor {
                                    format: wgpu::VertexFormat::Float3,
                                    offset: offset_of!(TexturedVertex, pos) as u64,
                                    shader_location: ids.next(),
                                },
                                wgpu::VertexAttributeDescriptor {
                                    format: wgpu::VertexFormat::Float2,
                                    offset: offset_of!(TexturedVertex, tex_coord) as u64,
                                    shader_location: ids.next(),
                                },
                                wgpu::VertexAttributeDescriptor {
                                    format: wgpu::VertexFormat::Float4,
                                    offset: offset_of!(TexturedVertex, atlas_texture) as u64,
                                    shader_location: ids.next(),
                                },
                            ],
                        }
                    },
                    wgpu::VertexBufferDescriptor {
                        stride: mem::size_of::<WorldVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Int,
                                offset: offset_of!(WorldVertex, texture_stride) as u64,
                                shader_location: ids.next(),
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float4,
                                offset: offset_of!(WorldVertex, count) as u64,
                                shader_location: ids.next(),
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float2,
                                offset: offset_of!(WorldVertex, lightmap_coord) as u64,
                                shader_location: ids.next(),
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float,
                                offset: offset_of!(WorldVertex, lightmap_stride) as u64,
                                shader_location: ids.next(),
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Uint,
                                offset: offset_of!(WorldVertex, lightmap_count) as u64,
                                shader_location: ids.next(),
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float,
                                offset: offset_of!(WorldVertex, value) as u64,
                                shader_location: ids.next(),
                            },
                        ],
                    },
                ],
            },
            sample_count,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let mut ids = BindId::default();
        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bindgroup_world"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::Buffer(matrices.slice(..)),
                },
                wgpu::BindGroupEntry {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::TextureView(diffuse_atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::TextureView(lightmap_atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::Sampler(&lightmap_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::Buffer(fragment_uniforms.slice(..)),
                },
            ],
        });

        Pipeline {
            bind_group,
            pipeline,
        }
    }
}

/// The skybox pipeline ignores the view matrix (or rather, it ignores the camera position),
/// does not write to the depth buffer, and ignores the current value in the depth buffer
/// when writing. We don't just set `depth_stencil_state` to `None` because that would make the
/// pipeline incompatible with our render pass.
///
/// The output for this is directly to the MSAA/post-processing buffer.
pub mod sky {
    use super::ShaderModuleSourceExt;
    use lazy_static::lazy_static;

    pub use super::Pipeline;
    use crate::render::TexturedVertex;
    use memoffset::offset_of;
    use std::mem;

    lazy_static! {
        static ref VERTEX_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/skybox.vert.spv"));
        static ref FRAGMENT_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/skybox.frag.spv"));
    }

    pub fn build(
        device: &wgpu::Device,
        diffuse_atlas_view: &wgpu::TextureView,
        diffuse_sampler: &wgpu::Sampler,
        matrices: &wgpu::Buffer,
        fragment_uniforms: &wgpu::Buffer,
        sample_count: u32,
    ) -> Pipeline {
        let vs_module = device.create_shader_module(VERTEX_SHADER.as_ref());
        let fs_module = device.create_shader_module(FRAGMENT_SHADER.as_ref());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bindgrouplayout_sky"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: super::WINDING_MODE,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: super::postprocess::DIFFUSE_BUFFER_FORMAT,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: super::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilStateDescriptor {
                    front: wgpu::StencilStateFaceDescriptor::IGNORE,
                    back: wgpu::StencilStateFaceDescriptor::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
            }),
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint32,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: mem::size_of::<TexturedVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float3,
                            offset: offset_of!(TexturedVertex, pos) as u64,
                            shader_location: 0,
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float2,
                            offset: offset_of!(TexturedVertex, tex_coord) as u64,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            sample_count,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bindgroup_sky"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(matrices.slice(..)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(diffuse_atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(fragment_uniforms.slice(..)),
                },
            ],
        });

        Pipeline {
            bind_group,
            pipeline,
        }
    }
}

pub mod models {
    use super::{BindId, ShaderModuleSourceExt};
    use lazy_static::lazy_static;

    pub use super::Pipeline;
    use crate::render::{ModelVertex, TexturedVertex};
    use memoffset::offset_of;
    use std::mem;

    lazy_static! {
        static ref VERTEX_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/models.vert.spv"));
        static ref FRAGMENT_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/models.frag.spv"));
    }

    // TODO: We should use a separate bindgroup for lights, since that way we could just set the bind
    //       group every time we render a model to be a slice of the full light buffer instead of
    //       transferring the lights over every frame. This would make it easier to implement a way of
    //       lighting where for every model we find the cluster it's in and check all the lights in
    //       the visible set of _that_ cluster, instead of the visible set of the camera cluster. Plus
    //       it means massively reducing communication with the GPU.
    //
    //       The only issue is that since we can't index, we'd have to duplicate lights multiple times
    //       in order to, for every cluster, store the lights of it and its visible set contiguously.
    //
    //       Maybe a job for deferred shading?
    pub fn build(
        device: &wgpu::Device,
        diffuse_atlas_view: &wgpu::TextureView,
        diffuse_sampler: &wgpu::Sampler,
        matrices: &wgpu::Buffer,
        model_data: &wgpu::Buffer,
        bones: &wgpu::Buffer,
        sample_count: u32,
    ) -> Pipeline {
        let vs_module = device.create_shader_module(VERTEX_SHADER.as_ref());
        let fs_module = device.create_shader_module(FRAGMENT_SHADER.as_ref());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bindgrouplayout_models"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let mut ids = BindId::default();
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: super::WINDING_MODE,
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: super::postprocess::DIFFUSE_BUFFER_FORMAT,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: super::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilStateDescriptor {
                    front: wgpu::StencilStateFaceDescriptor::IGNORE,
                    back: wgpu::StencilStateFaceDescriptor::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
            }),
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint32,
                vertex_buffers: &[
                    wgpu::VertexBufferDescriptor {
                        stride: mem::size_of::<TexturedVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float3,
                                offset: offset_of!(TexturedVertex, pos) as u64,
                                shader_location: ids.next(),
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float2,
                                offset: offset_of!(TexturedVertex, tex_coord) as u64,
                                shader_location: ids.next(),
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float4,
                                offset: offset_of!(TexturedVertex, atlas_texture) as u64,
                                shader_location: ids.next(),
                            },
                        ],
                    },
                    wgpu::VertexBufferDescriptor {
                        stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float3,
                                offset: offset_of!(ModelVertex, normal) as u64,
                                shader_location: ids.next(),
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Uint,
                                offset: offset_of!(ModelVertex, bone_id) as u64,
                                shader_location: ids.next(),
                            },
                        ],
                    },
                ],
            },
            sample_count,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bindgroup_models"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(matrices.slice(..)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(diffuse_atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(model_data.slice(..)),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(bones.slice(..)),
                },
            ],
        });

        Pipeline {
            bind_group,
            pipeline,
        }
    }
}

/// The postprocess pipeline handles gamma/intensity - the exponent and multiplication factor
/// for each pixel, respectively. Additionally, it implicitly handles MSAA, since the buffer
/// that the rest of the pipelines write to can be larger or have more samples than the output
/// buffer.
pub mod postprocess {
    use super::{BindId, ShaderModuleSourceExt as _};
    use lazy_static::lazy_static;

    pub use super::Pipeline;
    use crate::render::PostVertex;
    use memoffset::offset_of;
    use std::mem;

    lazy_static! {
        static ref VERTEX_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/post.vert.spv"));
        static ref FRAGMENT_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/post.frag.spv"));
    }

    pub const DIFFUSE_BUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg11b10Float;

    pub fn build(
        device: &wgpu::Device,
        diffuse_tex: &wgpu::TextureView,
        bloom_tex: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        post_uniforms: &wgpu::Buffer,
        fragment_uniforms: &wgpu::Buffer,
    ) -> Pipeline {
        let vs_module = device.create_shader_module(VERTEX_SHADER.as_ref());
        let fs_module = device.create_shader_module(FRAGMENT_SHADER.as_ref());

        let mut ids = BindId::default();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bindgrouplayout_post"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: true,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: super::WINDING_MODE,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
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
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: mem::size_of::<PostVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[wgpu::VertexAttributeDescriptor {
                        format: wgpu::VertexFormat::Float2,
                        offset: offset_of!(PostVertex, pos) as u64,
                        shader_location: 0,
                    }],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let mut ids = BindId::default();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bindgroup_post"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::TextureView(diffuse_tex),
                },
                wgpu::BindGroupEntry {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::TextureView(bloom_tex),
                },
                wgpu::BindGroupEntry {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::Buffer(fragment_uniforms.slice(..)),
                },
                wgpu::BindGroupEntry {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::Buffer(post_uniforms.slice(..)),
                },
            ],
        });

        Pipeline {
            bind_group,
            pipeline,
        }
    }
}

/// Hi-pass filter (at the level of luminosity)
pub mod hipass {
    use super::ShaderModuleSourceExt;
    use lazy_static::lazy_static;

    pub use super::Pipeline;
    use crate::render::PostVertex;
    use memoffset::offset_of;
    use std::mem;

    lazy_static! {
        static ref VERTEX_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/post.vert.spv"));
        static ref FRAGMENT_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/hipass.frag.spv"));
    }

    pub const DIFFUSE_BUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg11b10Float;

    pub fn build(
        device: &wgpu::Device,
        diffuse_tex: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        post_uniforms: &wgpu::Buffer,
    ) -> Pipeline {
        let vs_module = device.create_shader_module(VERTEX_SHADER.as_ref());
        let fs_module = device.create_shader_module(FRAGMENT_SHADER.as_ref());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bindgrouplayout_post"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: true,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: super::WINDING_MODE,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: super::postprocess::DIFFUSE_BUFFER_FORMAT,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: mem::size_of::<PostVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[wgpu::VertexAttributeDescriptor {
                        format: wgpu::VertexFormat::Float2,
                        offset: offset_of!(PostVertex, pos) as u64,
                        shader_location: 0,
                    }],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bindgroup_post"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(diffuse_tex),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(post_uniforms.slice(..)),
                },
            ],
        });

        Pipeline {
            bind_group,
            pipeline,
        }
    }
}
