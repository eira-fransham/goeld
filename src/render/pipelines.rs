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

const WINDING_MODE: wgpu::FrontFace = wgpu::FrontFace::Ccw;

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
    use super::BindId;
    pub use super::Pipeline;

    use crate::render::{TexturedVertex, WorldVertex};
    use memoffset::offset_of;
    use std::mem;

    const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/world.vert.spv"));
    const FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/world.frag.spv"));

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
        let vs_module = device
            .create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(VERTEX_SHADER)).unwrap());

        let fs_module = device.create_shader_module(
            &wgpu::read_spirv(std::io::Cursor::new(FRAGMENT_SHADER)).unwrap(),
        );

        let mut ids = BindId::default();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bindgrouplayout_world"),
            bindings: &[
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: ids.next(),
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        let mut ids = BindId::default();
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
                front_face: super::WINDING_MODE,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[
                wgpu::ColorStateDescriptor {
                    format: super::postprocess::DIFFUSE_BUFFER_FORMAT,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
                wgpu::ColorStateDescriptor {
                    format: super::postprocess::LIGHT_BUFFER_FORMAT,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
            ],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: super::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
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
                                    format: wgpu::VertexFormat::Float4,
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
                                offset: offset_of!(WorldVertex, lightmap_width) as u64,
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
            bindings: &[
                wgpu::Binding {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::Buffer(matrices.slice(..)),
                },
                wgpu::Binding {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::TextureView(diffuse_atlas_view),
                },
                wgpu::Binding {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::TextureView(lightmap_atlas_view),
                },
                wgpu::Binding {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                },
                wgpu::Binding {
                    binding: ids.next(),
                    resource: wgpu::BindingResource::Sampler(&lightmap_sampler),
                },
                wgpu::Binding {
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
    pub use super::Pipeline;
    use crate::render::TexturedVertex;
    use memoffset::offset_of;
    use std::mem;

    const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/skybox.vert.spv"));
    const FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/skybox.frag.spv"));

    pub fn build(
        device: &wgpu::Device,
        diffuse_atlas_view: &wgpu::TextureView,
        diffuse_sampler: &wgpu::Sampler,
        matrices: &wgpu::Buffer,
        fragment_uniforms: &wgpu::Buffer,
        sample_count: u32,
    ) -> Pipeline {
        let vs_module = device
            .create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(VERTEX_SHADER)).unwrap());

        let fs_module = device.create_shader_module(
            &wgpu::read_spirv(std::io::Cursor::new(FRAGMENT_SHADER)).unwrap(),
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bindgrouplayout_sky"),
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
                    ty: wgpu::BindingType::Sampler { comparison: false },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
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
                front_face: super::WINDING_MODE,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[
                wgpu::ColorStateDescriptor {
                    format: super::postprocess::DIFFUSE_BUFFER_FORMAT,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
                wgpu::ColorStateDescriptor {
                    format: super::postprocess::LIGHT_BUFFER_FORMAT,
                    color_blend: wgpu::BlendDescriptor {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha_blend: wgpu::BlendDescriptor {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    write_mask: wgpu::ColorWrite::empty(),
                },
            ],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: super::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint32,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: mem::size_of::<TexturedVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float4,
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
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(matrices.slice(..)),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(diffuse_atlas_view),
                },
                wgpu::Binding {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                },
                wgpu::Binding {
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

pub mod rtlights {
    pub use super::Pipeline;
    use crate::render::{Light, NormalVertex, TexturedVertex};
    use memoffset::offset_of;
    use std::mem;

    const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rtlights.vert.spv"));
    const FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rtlights.frag.spv"));

    pub fn build(
        device: &wgpu::Device,
        matrices: &wgpu::Buffer,
        fragment_uniforms: &wgpu::Buffer,
        model_data: &wgpu::Buffer,
        sample_count: u32,
    ) -> Pipeline {
        let vs_module = device
            .create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(VERTEX_SHADER)).unwrap());

        let fs_module = device.create_shader_module(
            &wgpu::read_spirv(std::io::Cursor::new(FRAGMENT_SHADER)).unwrap(),
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bindgrouplayout_rtlights"),
            bindings: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: true },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
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
                front_face: super::WINDING_MODE,
                cull_mode: wgpu::CullMode::None,
                depth_bias: -1,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[
                wgpu::ColorStateDescriptor {
                    format: super::postprocess::DIFFUSE_BUFFER_FORMAT,
                    color_blend: wgpu::BlendDescriptor {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha_blend: wgpu::BlendDescriptor {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    write_mask: wgpu::ColorWrite::empty(),
                },
                wgpu::ColorStateDescriptor {
                    format: super::postprocess::LIGHT_BUFFER_FORMAT,
                    color_blend: wgpu::BlendDescriptor {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
            ],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: super::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint32,
                vertex_buffers: &[
                    wgpu::VertexBufferDescriptor {
                        stride: mem::size_of::<TexturedVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float4,
                            offset: offset_of!(TexturedVertex, pos) as u64,
                            shader_location: 0,
                        }],
                    },
                    wgpu::VertexBufferDescriptor {
                        stride: mem::size_of::<NormalVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float4,
                            offset: offset_of!(NormalVertex, normal) as u64,
                            shader_location: 3,
                        }],
                    },
                    wgpu::VertexBufferDescriptor {
                        stride: mem::size_of::<Light>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float4,
                                offset: offset_of!(Light, position) as u64,
                                shader_location: 4,
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float4,
                                offset: offset_of!(Light, color) as u64,
                                shader_location: 5,
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
            label: Some("bindgroup_rtlights"),
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(matrices.slice(..)),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(model_data.slice(..)),
                },
                wgpu::Binding {
                    binding: 2,
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
    pub use super::Pipeline;
    use crate::render::{NormalVertex, TexturedVertex};
    use memoffset::offset_of;
    use std::mem;

    const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/models.vert.spv"));
    const FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/models.frag.spv"));

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
        fragment_uniforms: &wgpu::Buffer,
        model_data: &wgpu::Buffer,
        sample_count: u32,
    ) -> Pipeline {
        let vs_module = device
            .create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(VERTEX_SHADER)).unwrap());

        let fs_module = device.create_shader_module(
            &wgpu::read_spirv(std::io::Cursor::new(FRAGMENT_SHADER)).unwrap(),
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bindgrouplayout_models"),
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
                    ty: wgpu::BindingType::Sampler { comparison: false },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: true },
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
                front_face: super::WINDING_MODE,
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[
                wgpu::ColorStateDescriptor {
                    format: super::postprocess::DIFFUSE_BUFFER_FORMAT,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
                wgpu::ColorStateDescriptor {
                    format: super::postprocess::LIGHT_BUFFER_FORMAT,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
            ],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: super::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint32,
                vertex_buffers: &[
                    wgpu::VertexBufferDescriptor {
                        stride: mem::size_of::<TexturedVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float4,
                                offset: offset_of!(TexturedVertex, pos) as u64,
                                shader_location: 0,
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float2,
                                offset: offset_of!(TexturedVertex, tex_coord) as u64,
                                shader_location: 1,
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float4,
                                offset: offset_of!(TexturedVertex, atlas_texture) as u64,
                                shader_location: 2,
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
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(matrices.slice(..)),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(diffuse_atlas_view),
                },
                wgpu::Binding {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                },
                wgpu::Binding {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(model_data.slice(..)),
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
    pub use super::Pipeline;
    use crate::render::TexturedVertex;
    use memoffset::offset_of;
    use std::mem;

    const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/post.vert.spv"));
    const FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/post.frag.spv"));

    pub const LIGHT_BUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    pub const DIFFUSE_BUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg11b10Float;

    pub fn build(
        device: &wgpu::Device,
        diffuse_tex: &wgpu::TextureView,
        lights_tex: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        fragment_uniforms: &wgpu::Buffer,
    ) -> Pipeline {
        let vs_module = device
            .create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(VERTEX_SHADER)).unwrap());

        let fs_module = device.create_shader_module(
            &wgpu::read_spirv(std::io::Cursor::new(FRAGMENT_SHADER)).unwrap(),
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bindgrouplayout_post"),
            bindings: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: true,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: true,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
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
                front_face: super::WINDING_MODE,
                cull_mode: wgpu::CullMode::Back,
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
                    stride: mem::size_of::<TexturedVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float4,
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
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bindgroup_post"),
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(diffuse_tex),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(lights_tex),
                },
                wgpu::Binding {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::Binding {
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
