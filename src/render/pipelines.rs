pub struct Pipeline {
    pub bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
}

pub mod world {
    use super::Pipeline;
    use crate::render::Vertex;
    use memoffset::offset_of;
    use std::mem;

    const VERTEX_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/basic.vert.spv"));
    const FRAGMENT_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/basic.frag.spv"));

    pub fn build(
        device: &wgpu::Device,
        diffuse_atlas_view: &wgpu::TextureView,
        lightmap_atlas_view: &wgpu::TextureView,
        diffuse_sampler: &wgpu::Sampler,
        lightmap_sampler: &wgpu::Sampler,
        matrix: &wgpu::Buffer,
        atlas_sizes: &wgpu::Buffer,
    ) -> Pipeline {
        let vs_module = device
            .create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(VERTEX_SHADER)).unwrap());

        let fs_module = device.create_shader_module(
            &wgpu::read_spirv(std::io::Cursor::new(FRAGMENT_SHADER)).unwrap(),
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bindgrouplayout_world"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 2,
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
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
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
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float,
                            offset: offset_of!(Vertex, lightmap_width) as u64,
                            shader_location: 4,
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Uint,
                            offset: offset_of!(Vertex, lightmap_count) as u64,
                            shader_location: 5,
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float,
                            offset: offset_of!(Vertex, value) as u64,
                            shader_location: 6,
                        },
                    ],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bindgroup_world"),
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(matrix.slice(..)),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(diffuse_atlas_view),
                },
                wgpu::Binding {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(lightmap_atlas_view),
                },
                wgpu::Binding {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                },
                wgpu::Binding {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&lightmap_sampler),
                },
                wgpu::Binding {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(atlas_sizes.slice(..)),
                },
            ],
        });

        Pipeline {
            bind_group,
            pipeline,
        }
    }
}

pub mod sky {
    use super::Pipeline;

    pub fn build(_device: &wgpu::Device) -> Pipeline {
        todo!()
    }
}
