trait ShaderModuleSourceExt<'a> {
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

// TODO: Use really low-res environment maps for lighting - this would probably also be useful to
//       emulate HL1's fake "chrome" effect (although that might just be screen-space since HL1
//       didn't have cubemaps).
//
// ------------------------------------------------------------------------------------------------
// Process:
//   Generate a bunch of probes
//     (to start with we can just grab each leaf's midpoint but further down the line we'll need
//     a better heuristic, we should absolutely have denser probes where there are more lights.
//     Maybe keep generating new probes if the "initial probe" has a certain overall intensity
//     or a certain differential in intensity between its faces)
//   For each probe, allocate 6 rectangles in an atlas at really low res for each face of the
//   cubemap (maybe 8x8px)
//   Render skybox by doing a single pass where the whole cubemap has intensity calculated as
//   its normal dot env light direction.
//   Render world, so that any parts of the envmap where the sky isn't visible get correctly
//   occluded.
//     Here we can choose to either "properly" render the world (and get lovely bounce lighting)
//     or ignore that and just render everything as black. The former might look anachronistic,
//     but the way that lighting was implemented in these old engines is actually not so different,
//     since it's based on the lightmap level at the character's feet.
//   With additive blending, render all lights (using instancing) by calculating the dot of the
//   cubemap's normal at each pixel and using that to calculate the correct light amount. For
//   emissive faces, use area light-style calculation instead of pointlight calculation.
//   When rendering models, blend between N cubemaps weighted by distance or distance squared
//     (maybe best to do this with deferred shading?)
//   Save envmap atlas and probe positions/clusters to external file so we don't need to regenerate
//   it every time. Maybe include the map name and a hash of the bsp file itself so we can regen
//   when the bsp changes.
// ------------------------------------------------------------------------------------------------
//
// We can absolutely do this lazily, we don't even necessarily have to generate the probes upfront.
// The only upfront work we absolutely need to do is cache the point lights (and spotlights for
// Goldsrc). This might make serialising it to a file more complicated though.
//
// Since we're not really relying on the GLSL 3D vertex transformation pipeline, we could probably
// do an entire cubemap in a single render. Without a geometry shader we couldn't do multiple
// cubemaps in a single render, but it's unlikely that a geometry shader would be worth it
// performance-wise. In fact, to remind myself later, there's a direct quote from someone on a
// thread about geom shaders on Reddit that specifically says:
// > All other cases I have personally tested, like using gs to render to multiple cubemap faces,
// > we're slower than other techniques.
//
// From my approximations, we could store 1000 cubemaps at 8x8 per face (approx. 10 degree
// granularity) in a 620x620 atlas, so probably a 1024x1024 atlas is more than enough. Since our
// diffuse atlas and our lightmap atlases are both so small this is easily cheap enough to use.
// There's nothing really stopping us from combining all our atlases into a single, large texture,
// which would improve efficiency since less space would be wasted.
pub mod rtlights {
    use super::ShaderModuleSourceExt;
    use lazy_static::lazy_static;

    pub use super::Pipeline;
    use crate::render::{Light, NormalVertex, TexturedVertex};
    use memoffset::offset_of;
    use std::mem;

    lazy_static! {
        static ref VERTEX_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/rtlights.vert.spv"));
        static ref FRAGMENT_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/rtlights.frag.spv"));
    }
    pub fn build(
        device: &wgpu::Device,
        matrices: &wgpu::Buffer,
        fragment_uniforms: &wgpu::Buffer,
        model_data: &wgpu::Buffer,
        sample_count: u32,
    ) -> Pipeline {
        let vs_module = device.create_shader_module(VERTEX_SHADER.as_ref());
        let fs_module = device.create_shader_module(FRAGMENT_SHADER.as_ref());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bindgrouplayout_rtlights"),
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
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: true,
                        min_binding_size: None,
                    },
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
                cull_mode: wgpu::CullMode::None,
                depth_bias: -1,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
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
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(matrices.slice(..)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(model_data.slice(..)),
                },
                wgpu::BindGroupEntry {
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
    use super::ShaderModuleSourceExt;
    use lazy_static::lazy_static;

    pub use super::Pipeline;
    use crate::render::{NormalVertex, TexturedVertex};
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
        fragment_uniforms: &wgpu::Buffer,
        model_data: &wgpu::Buffer,
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
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
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
                }],
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
    use super::ShaderModuleSourceExt;
    use lazy_static::lazy_static;

    pub use super::Pipeline;
    use crate::render::TexturedVertex;
    use memoffset::offset_of;
    use std::mem;

    lazy_static! {
        static ref VERTEX_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/post.vert.spv"));
        static ref FRAGMENT_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/post.frag.spv"));
    }

    pub const LIGHT_BUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    pub const DIFFUSE_BUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg11b10Float;

    pub fn build(
        device: &wgpu::Device,
        diffuse_tex: &wgpu::TextureView,
        lights_tex: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        fragment_uniforms: &wgpu::Buffer,
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
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: true,
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
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(diffuse_tex),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(lights_tex),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
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
