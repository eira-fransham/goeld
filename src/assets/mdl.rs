use crate::{
    cache::Cache,
    loader::{LoadAsset, Loader},
    render::{
        ModelData, NormalVertex, PipelineDesc, Render, RenderCache, RenderMesh, TexturedVertex,
        TransferData, VertexOffset,
    },
};
use cgmath::Transform;
use fnv::FnvHashMap as HashMap;
use image::ImageBuffer;
use std::{borrow::Cow, collections::hash_map::Entry, ops::Range};

pub struct MdlAsset<'a>(pub assimp::Scene<'a>);

#[derive(Clone)]
pub struct Model {
    vert_offset: u64,
    norm_offset: u64,
    index_ranges: Vec<Range<u32>>,
    model_data_offset: u64,

    position: cgmath::Vector3<f32>,
    position_dirty: bool,
}

impl Model {
    pub fn update_position(&mut self, by: cgmath::Vector3<f32>) {
        self.position += by;
        self.position_dirty = true;
    }
}

impl LoadAsset for MdlAsset<'_> {
    type Asset = Model;

    #[inline]
    fn load(self, loader: &Loader, cache: &mut RenderCache) -> anyhow::Result<Self::Asset> {
        use crate::loader::Load;

        let mut textures_by_name =
            HashMap::with_capacity_and_hasher(self.0.num_textures() as usize, Default::default());

        for texture in self.0.textures() {
            if let Some(data) = texture.data() {
                // Assimp lies about the texture format for HL1 models, and it also incorrectly tags
                // some as needing to be inverted when this isn't true. I don't know why.
                let tex = match texture.format_hint() {
                    Some("rgba8888") | Some("bgra8888") | None => cache.diffuse.append(
                        ImageBuffer::<image::Bgra<u8>, _>::from_raw(
                            texture.width(),
                            texture.height(),
                            data.bytes(),
                        )
                        .unwrap(),
                    ),
                    _ => unimplemented!(),
                };

                textures_by_name.insert(Cow::Borrowed(texture.filename()), tex);
            }
        }

        let loader = loader.textures();

        let materials = self
            .0
            .materials()
            .map(|mat| {
                let mut tex = mat.diffuse().unwrap();

                if let Some(first) = tex.textures.next() {
                    if true || tex.textures.len() == 0 {
                        // TODO: Handle the "invert" flag
                        if first.blend_op == assimp::BlendOp::Replace {
                            match textures_by_name.entry(first.path.to_string().into()) {
                                Entry::Occupied(entry) => (first.channel, entry.get().clone()),
                                Entry::Vacant(entry) => {
                                    use std::path::Path;

                                    let (file, path) = loader
                                        .load(Path::new(&first.path[..]).into())
                                        .unwrap_or_else(|e| {
                                            panic!("Could not find {:?}: {:?}", &first.path, e)
                                        });

                                    let img = image::load(
                                        std::io::BufReader::new(file),
                                        image::ImageFormat::from_path(&path).unwrap(),
                                    )
                                    .unwrap();

                                    let appended = cache.diffuse.append(img);

                                    let out = entry.insert(appended).clone();

                                    (first.channel, out)
                                }
                            }
                        } else {
                            unimplemented!()
                        }
                    } else {
                        unimplemented!(
                            "{:?}",
                            std::iter::once(first)
                                .chain(tex.textures)
                                .collect::<Vec<_>>()
                        )
                    }
                } else {
                    unimplemented!()
                }
            })
            .collect::<Vec<_>>();

        fn push_nodes(
            ranges: &mut Vec<Range<u32>>,
            offset: u64,
            materials: &[(u32, rect_packer::Rect)],
            cache: &mut RenderCache,
            scene: &assimp::Scene,

            transform: cgmath::Matrix4<f32>,
            node: &assimp::Node,
        ) {
            let transform = cgmath::Matrix4::<f32>::from(node.transform()) * transform;

            for id in node.meshes() {
                let mesh = scene.mesh(*id).unwrap();

                let (channel, tex_rect) = materials[mesh.material_id() as usize];

                let vert_range = cache.textured_vertices.append(
                    mesh.positions().zip(mesh.texture_coords(channel)).map(
                        move |(position, uvs)| {
                            let position: cgmath::Vector3<f32> = position.into();
                            let position = transform
                                * cgmath::Vector4::from([position.x, position.y, position.z, 1.]);

                            TexturedVertex {
                                pos: position.into(),
                                tex_coord: [
                                    uvs.x * (tex_rect.width as f32),
                                    uvs.y * (tex_rect.height as f32),
                                ],
                                atlas_texture: [
                                    tex_rect.x as u32,
                                    tex_rect.y as u32,
                                    tex_rect.width as u32,
                                    tex_rect.height as u32,
                                ],
                            }
                        },
                    ),
                );

                let normals = mesh.normals();

                assert!(normals.len() > 0);

                cache.normal_vertices.append(normals.map(move |norm| {
                    let norm: cgmath::Vector3<f32> = norm.into();
                    let norm = transform.transform_vector(norm);
                    NormalVertex {
                        normal: norm.into(),
                    }
                }));

                let to_add = (vert_range.start as u64 - offset) as u32;

                let index_range = cache.indices.append(
                    mesh.faces()
                        .flat_map(|face| {
                            assert_eq!(face.primitive_type(), assimp::PrimitiveType::Triangle);
                            face.indices().iter().copied()
                        })
                        .map(|i| i + to_add),
                );

                ranges.push(index_range.start as u32..index_range.end as u32);
            }

            for node in node.children() {
                push_nodes(ranges, offset, materials, cache, scene, transform, node);
            }
        }

        let vert_offset = cache.textured_vertices.len();
        let norm_offset = cache.normal_vertices.len();

        let mut index_ranges = Vec::new();

        if let Some(node) = self.0.root_node() {
            push_nodes(
                &mut index_ranges,
                vert_offset,
                &materials,
                cache,
                &self.0,
                cgmath::Matrix4::<f32>::from_angle_z(cgmath::Deg(90.))
                    * cgmath::Matrix4::<f32>::from_angle_y(cgmath::Deg(-90.)),
                node,
            );
        }

        let model_data_offset = cache
            .model_data
            .append(std::iter::once(ModelData {
                translation: cgmath::SquareMatrix::identity(),
                origin: cgmath::Vector3::from([0., 0., 0.]),
            }))
            .start;

        Ok(Model {
            vert_offset,
            norm_offset,
            index_ranges,
            model_data_offset,
            position: [0., 0., 0.].into(),
            position_dirty: false,
        })
    }
}

impl TransferData for &'_ Model {
    fn transfer_data(self) -> Option<(wgpu::BufferAddress, ModelData)> {
        if self.position_dirty {
            Some((
                self.model_data_offset,
                ModelData {
                    translation: cgmath::Matrix4::from_translation(self.position),
                    origin: self.position,
                },
            ))
        } else {
            None
        }
    }
}

impl<'a> Render for &'a Model {
    type Indices = std::iter::Cloned<std::slice::Iter<'a, Range<u32>>>;
    type Offsets = (VertexOffset<TexturedVertex>, VertexOffset<NormalVertex>);

    fn indices<T: crate::render::Context>(
        self,
        _ctx: &T,
    ) -> RenderMesh<Self::Offsets, Self::Indices> {
        RenderMesh {
            offsets: (self.vert_offset.into(), self.norm_offset.into()),
            indices: self.index_ranges.iter().cloned(),
            pipeline: PipelineDesc::Models {
                origin: self.position,
                data_offset: self.model_data_offset,
            },
        }
    }
}
