use crate::render::{PostVertex, Uniforms};
use wgpu::util::DeviceExt as _;

mod pipelines {
    use crate::render::pipelines::{self, ShaderModuleSourceExt as _};
    use lazy_static::lazy_static;

    use crate::render::PostVertex;
    use memoffset::offset_of;
    use std::mem;

    lazy_static! {
        static ref VERTEX_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/post.vert.spv"));
        static ref DOWN_FRAGMENT_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/dual_kawase_down.frag.spv"));
        static ref UP_FRAGMENT_SHADER: wgpu::ShaderModuleSource<'static> =
            wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/dual_kawase_up.frag.spv"));
    }

    pub struct Pipeline {
        pub down_pipeline: wgpu::RenderPipeline,
        pub up_pipeline: wgpu::RenderPipeline,
        pub bind_group: wgpu::BindGroup,
        pub texture_bind_group_layout: wgpu::BindGroupLayout,
    }

    pub fn build(
        device: &wgpu::Device,
        sampler: &wgpu::Sampler,
        uniforms: &wgpu::Buffer,
    ) -> Pipeline {
        let vs_module = device.create_shader_module(VERTEX_SHADER.as_ref());
        let down_fs_module = device.create_shader_module(DOWN_FRAGMENT_SHADER.as_ref());
        let up_fs_module = device.create_shader_module(UP_FRAGMENT_SHADER.as_ref());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bindgrouplayout_kawase_uniforms"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bindgrouplayout_kawase_uniforms"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::SampledTexture {
                            multisampled: false,
                            component_type: wgpu::TextureComponentType::Float,
                            dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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
            bind_group_layouts: &[&bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let down_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &down_fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: pipelines::WINDING_MODE,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: pipelines::postprocess::DIFFUSE_BUFFER_FORMAT,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint32,
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

        let up_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &up_fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: pipelines::WINDING_MODE,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: pipelines::postprocess::DIFFUSE_BUFFER_FORMAT,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint32,
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
            label: Some("bindgroup_kawase_uniforms"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(uniforms.slice(..)),
                },
            ],
        });

        Pipeline {
            down_pipeline,
            up_pipeline,
            texture_bind_group_layout,
            bind_group,
        }
    }
}

struct Target {
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

pub struct Blur {
    targets: Vec<Target>,
    vertices: wgpu::Buffer,
    uniforms: Uniforms<[f32; 2]>,
    framebuffer_size: (u32, u32),
    pipeline: pipelines::Pipeline,
    output: wgpu::TextureView,
}

impl Blur {
    pub fn new(
        device: &wgpu::Device,
        input: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        iterations: usize,
        initial_downsample: u8,
        radius: f32,
        framebuffer_size: (u32, u32),
    ) -> Self {
        let (radius_x, radius_y) = (
            radius / framebuffer_size.0 as f32,
            radius / framebuffer_size.1 as f32,
        );

        let uniforms = Uniforms::new([radius_x, radius_y], device);

        let pipeline = pipelines::build(device, sampler, &uniforms.buffer());

        let mut buffers = Vec::with_capacity(iterations);

        struct InterimTarget {
            texture: wgpu::Texture,
            uniforms: wgpu::Buffer,
        }

        for i in 1..iterations {
            let downsample = initial_downsample as usize + i;

            let resolution = (
                framebuffer_size.0 >> downsample,
                framebuffer_size.1 >> downsample,
            );

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d {
                    width: resolution.0,
                    height: resolution.1,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: crate::render::pipelines::postprocess::DIFFUSE_BUFFER_FORMAT,
                usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
                label: Some("kawase_intermediate_buffer"),
            });

            let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[
                    (resolution.0 as f32).recip(),
                    (resolution.1 as f32).recip(),
                ]),
                usage: wgpu::BufferUsage::UNIFORM,
            });

            buffers.push(InterimTarget { texture, uniforms });
        }

        let mut prev_owned;
        let mut prev = input;

        let mut targets = Vec::with_capacity(iterations * 2);

        for target in &buffers {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bindgroup_kawase_tex"),
                layout: &pipeline.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&prev),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(target.uniforms.slice(..)),
                    },
                ],
            });

            prev_owned = target.texture.create_view(&Default::default());
            prev = &prev_owned;

            targets.push(Target {
                view: target.texture.create_view(&Default::default()),
                bind_group,
            });
        }

        let mut targets = Vec::with_capacity(iterations);

        let output_size = (
            framebuffer_size.0 >> initial_downsample,
            framebuffer_size.1 >> initial_downsample,
        );
        let output = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: output_size.0,
                height: output_size.1,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::render::pipelines::postprocess::DIFFUSE_BUFFER_FORMAT,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
            label: Some("kawase_intermediate_buffer"),
        });

        if let Some(mut prev) = buffers
            .last()
            .map(|target| target.texture.create_view(&Default::default()))
        {
            for target in buffers.iter().rev().skip(1) {
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bindgroup_kawase_tex"),
                    layout: &pipeline.texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&prev),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(target.uniforms.slice(..)),
                        },
                    ],
                });

                prev = target.texture.create_view(&Default::default());

                targets.push(Target {
                    view: target.texture.create_view(&Default::default()),
                    bind_group,
                });
            }

            let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[
                    (output_size.0 as f32).recip(),
                    (output_size.1 as f32).recip(),
                ]),
                usage: wgpu::BufferUsage::UNIFORM,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bindgroup_kawase_tex"),
                layout: &pipeline.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&prev),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(uniforms.slice(..)),
                    },
                ],
            });

            targets.push(Target {
                view: output.create_view(&Default::default()),
                bind_group,
            });
        }

        // TODO: Share this with `Renderer`
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
            targets,
            vertices,
            uniforms,
            framebuffer_size,
            pipeline,
            output: output.create_view(&Default::default()),
        }
    }

    pub fn set_radius(&mut self, radius: f32) {
        let (radius_x, radius_y) = (
            radius / self.framebuffer_size.0 as f32,
            radius / self.framebuffer_size.1 as f32,
        );

        self.uniforms.set([radius_x, radius_y]);
    }

    pub fn output(&self) -> &wgpu::TextureView {
        &self.output
    }

    pub fn blur(&self, encoder: &mut wgpu::CommandEncoder) {
        for cur in &self.targets {
            let mut post_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &cur.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            post_pass.set_pipeline(&self.pipeline.up_pipeline);
            post_pass.set_bind_group(0, &self.pipeline.bind_group, &[]);
            post_pass.set_vertex_buffer(0, self.vertices.slice(..));

            post_pass.set_bind_group(1, &cur.bind_group, &[]);
            post_pass.draw(0..6, 0..1);
        }
    }
}
