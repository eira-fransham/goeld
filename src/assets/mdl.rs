use crate::{
    cache::Cache,
    loader::{LoadAsset, Loader},
    render::{
        ModelData, ModelVertex, PipelineDesc, Render, RenderCache, RenderMesh, TexturedVertex,
        TransferData, VertexOffset,
    },
};
use cgmath::Transform;
use fnv::FnvHashMap as HashMap;
use image::ImageBuffer;
use std::{
    borrow::Cow,
    collections::hash_map::Entry,
    mem,
    ops::{Add, Mul, Range},
};

// TODO: Is this the root for all file types or just HL1 .mdl files?
const ROOT_NAME: &str = "<MDL_root>";

pub struct MdlAsset<'a>(pub assimp::Scene<'a>);

// TODO: Stop using String
#[derive(Debug, Clone, PartialEq)]
struct Node {
    default_transform: cgmath::Matrix4<f32>,
    bone_offset: cgmath::Matrix4<f32>,
    children: Vec<String>,
    bone: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
struct Keyframe<T> {
    time: f64,
    value: T,
}

impl<T> Keyframe<T>
where
    T: Copy + Mul<f32, Output = T> + Add<Output = T>,
{
    fn lerp(&self, other: &Self, factor: f32) -> Self {
        Self {
            time: self.time * (1. - factor as f64) + other.time * factor as f64,
            value: self.value * (1. - factor) + other.value * factor,
        }
    }
}

fn get_keyframe<T>(frames: &[Keyframe<T>], time: f64) -> Keyframe<T>
where
    T: Copy + Mul<f32, Output = T> + Add<Output = T>,
{
    match frames.binary_search_by(|f| {
        f.time
            .partial_cmp(&time)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Ok(pos) => frames[pos].clone(),
        Err(next) => {
            let prev = (next + frames.len() - 1) % frames.len();
            let next = next % frames.len();

            if next == prev {
                frames[prev].clone()
            } else {
                let prev = &frames[prev];
                let next = &frames[next];

                let frame_time = next.time - prev.time;

                prev.lerp(next, ((time - prev.time) / frame_time) as f32)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Keyframes {
    translation: Vec<Keyframe<cgmath::Vector3<f32>>>,
    rotation: Vec<Keyframe<cgmath::Quaternion<f32>>>,
    scaling: Vec<Keyframe<cgmath::Vector3<f32>>>,
}

// TODO: Allocate way less
#[derive(Debug, Clone, PartialEq)]
struct Animation {
    keyframes_per_node: HashMap<String, Keyframes>,
    fps: f64,
    duration: f64,
}

impl Animation {
    fn keyframe_at(&self, name: &str, time: f64) -> Option<cgmath::Matrix4<f32>> {
        let time = (time * self.fps) % 150.; // self.duration;

        let frames = self.keyframes_per_node.get(name)?;

        let rotation = if frames.rotation.is_empty() {
            cgmath::SquareMatrix::identity()
        } else {
            let rotation = get_keyframe(&frames.rotation, time).value;
            cgmath::Matrix4::from(rotation)
        };

        let translation = if frames.translation.is_empty() {
            cgmath::SquareMatrix::identity()
        } else {
            let translation = get_keyframe(&frames.translation, time).value;
            cgmath::Matrix4::from_translation(translation)
        };

        let scaling = if frames.scaling.is_empty() {
            cgmath::SquareMatrix::identity()
        } else {
            let scaling = get_keyframe(&frames.scaling, time).value;
            cgmath::Matrix4::from_nonuniform_scale(scaling.x, scaling.y, scaling.z)
        };

        Some(translation * rotation * scaling)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct CurrentAnimation {
    timestamp: f64,
    index: u32,
}

#[derive(Clone)]
pub struct Model {
    vert_offset: u64,
    norm_offset: u64,
    index_ranges: Vec<Range<u32>>,
    model_data_offset: u64,

    nodes: HashMap<String, Node>,
    animations: Vec<Animation>,
    current_animation: Option<CurrentAnimation>,
    transferred_data: Option<CurrentAnimation>,
    bone_range: Range<u64>,

    position: cgmath::Vector3<f32>,
    position_dirty: bool,
}

impl Model {
    pub fn update_position(&mut self, by: cgmath::Vector3<f32>) {
        self.position += by;
        self.position_dirty = true;
    }

    pub fn step(&mut self, dt: f64) {
        if let Some(CurrentAnimation { timestamp, index }) = self.current_animation.as_mut() {
            *timestamp += dt;
        }
    }

    pub fn set_animation(&mut self, index: u32) {
        self.current_animation = Some(CurrentAnimation {
            timestamp: 0.,
            index,
        });
    }

    pub fn update(&mut self, cache: &mut RenderCache) {
        use cgmath::SquareMatrix as _;

        if self.transferred_data == self.current_animation {
            return;
        }

        let bones = cache.bone_matrices.range_mut(self.bone_range.clone());

        let node = if let Some(node) = self.nodes.get(ROOT_NAME) {
            node
        } else {
            return;
        };

        let transform = node.default_transform;

        let anim =
            self.current_animation
                .as_ref()
                .and_then(|CurrentAnimation { index, timestamp }| {
                    self.animations
                        .get(*index as usize)
                        .map(|anim| (*timestamp, anim))
                });
        let keyframe = anim.and_then(|(timestamp, anim)| anim.keyframe_at(ROOT_NAME, timestamp));

        fn update_node(
            bones: &mut [[[f32; 4]; 4]],
            nodes: &HashMap<String, Node>,
            animation: Option<(f64, &Animation)>,
            parent_transform: cgmath::Matrix4<f32>,
            cur_node: &Node,
            keyframe: Option<cgmath::Matrix4<f32>>,
        ) {
            let keyframe_transform = keyframe.clone().unwrap_or(cur_node.default_transform);

            let transform = parent_transform * keyframe_transform;

            if let Some(bone_id) = &cur_node.bone {
                bones[*bone_id as usize] =
                    (cur_node.default_transform * transform * cur_node.bone_offset).into();
            }

            for child in &cur_node.children {
                let node = if let Some(node) = nodes.get(child) {
                    node
                } else {
                    continue;
                };

                let keyframe = animation
                    .as_ref()
                    .and_then(|(timestamp, anim)| anim.keyframe_at(child, *timestamp));

                update_node(bones, nodes, animation, transform, node, keyframe);
            }
        }

        update_node(bones, &self.nodes, anim, transform, node, keyframe);
    }
}

// We don't support scaling, and the order is always: rotate around bone origin, translate
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Bone {
    pub parent: Option<u8>,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
}

impl LoadAsset for MdlAsset<'_> {
    type Asset = Model;

    #[inline]
    fn load(self, loader: &Loader, cache: &mut RenderCache) -> anyhow::Result<Self::Asset> {
        use crate::loader::Load;
        use cgmath::SquareMatrix as _;

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

        const ASSUME_SIMPLE_TEXTURES: bool = true;

        let loader = loader.textures();

        let materials = self
            .0
            .materials()
            .map(|mat| {
                let mut tex = mat.diffuse().unwrap();

                if let Some(first) = tex.textures.next() {
                    if ASSUME_SIMPLE_TEXTURES || tex.textures.len() == 0 {
                        if ASSUME_SIMPLE_TEXTURES || first.blend_op == assimp::BlendOp::Replace {
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
            bones: &mut Vec<cgmath::Matrix4<f32>>,
            nodes: &mut HashMap<String, Node>,
            offset: u64,
            materials: &[(u32, rect_packer::Rect)],
            cache: &mut RenderCache,
            scene: &assimp::Scene,

            original_transform: cgmath::Matrix4<f32>,
            node: &assimp::Node,
        ) {
            use std::convert::TryFrom;

            let cur_transform = cgmath::Matrix4::<f32>::from(node.transform());
            let transform = cur_transform * original_transform;

            let mut node_metadata = Node {
                default_transform: cur_transform,
                bone_offset: cgmath::SquareMatrix::identity(),
                children: Vec::with_capacity(node.num_children() as usize),
                bone: None,
            };

            dbg!(node.name());

            for id in node.meshes() {
                let mesh = scene.mesh(*id).unwrap();

                dbg!(mesh.name());

                let mut vertex_bones = vec![[None, None]; mesh.num_vertices() as usize];

                for bone in mesh.bones() {
                    let node = if let Some(node) = nodes.get_mut(bone.name()) {
                        node
                    } else {
                        continue;
                    };

                    if node.bone.is_none() {
                        let bone_id = bones.len() as u32;
                        bones.push(transform);

                        node.bone = Some(bone_id);
                        node.bone_offset = bone.offset_matrix().into();
                    }

                    let bone_id = node.bone.unwrap();

                    for v_weight in bone.weights() {
                        let id = v_weight.mVertexId;
                        let weight = v_weight.mWeight;

                        let v_weight = &mut vertex_bones[id as usize];

                        match v_weight {
                            [Some(_), other @ None] | [other @ None, None] => {
                                *other = Some((
                                    u32::try_from(bone_id).expect("Too many bones in model"),
                                    weight,
                                ))
                            }
                            _ => panic!(),
                        }
                    }
                }

                let (channel, tex_rect) = materials[mesh.material_id() as usize];

                let vert_range = cache.textured_vertices.append(
                    mesh.positions().zip(mesh.texture_coords(channel)).map(
                        move |(position, uvs)| {
                            let position: cgmath::Vector3<f32> = position.into();
                            let position =
                                cgmath::Vector4::from([position.x, position.y, position.z, 1.]);

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

                cache.model_vertices.append(normals.zip(&vertex_bones).map(
                    move |(norm, [bone_a, bone_b])| {
                        let norm: cgmath::Vector3<f32> = norm.into();
                        let norm = transform.transform_vector(norm);
                        let bones = [bone_a.unwrap_or_default(), bone_b.unwrap_or_default()];
                        ModelVertex {
                            normal: norm.into(),
                            bone_ids: [bones[0].0, bones[1].0],
                            bone_weights: [bones[0].1, bones[1].1],
                        }
                    },
                ));

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
                push_nodes(
                    ranges, bones, nodes, offset, materials, cache, scene, transform, node,
                );

                node_metadata.children.push(node.name().to_string());
            }

            nodes.insert(node.name().to_string(), node_metadata);
        }

        let vert_offset = cache.textured_vertices.len();
        let norm_offset = cache.model_vertices.len();

        let mut index_ranges = Vec::new();
        let mut bones = Vec::new();
        let mut nodes = HashMap::default();

        if let Some(node) = self.0.root_node() {
            push_nodes(
                &mut index_ranges,
                &mut bones,
                &mut nodes,
                vert_offset,
                &materials,
                cache,
                &self.0,
                node.transform().into(),
                node,
            );
        }

        let model_data_offset = cache
            .model_data
            .append(std::iter::once(ModelData {
                translation: cgmath::Matrix4::identity(),
                origin: cgmath::Vector3::from([0., 0., 0.]),
            }))
            .start;

        let bone_range = cache
            .bone_matrices
            .append(bones.into_iter().map(Into::into));

        let animations = self
            .0
            .animations()
            .map(|anim| Animation {
                fps: anim.fps(),
                duration: anim.duration(),

                keyframes_per_node: anim
                    .node_anims()
                    .map(|node_anim| {
                        // TODO: Split transform/rotate/scale keys like assimp does
                        //       so we don't need to jankily translate it back to
                        //       a monolithic keyframe.
                        (
                            node_anim.node_name().to_string(),
                            Keyframes {
                                translation: node_anim
                                    .position_keys()
                                    .map(|key| Keyframe {
                                        time: key.time(),
                                        value: cgmath::Vector3::<f32>::from(key.value()),
                                    })
                                    .filter(|Keyframe { value, .. }| {
                                        !value.x.is_nan() && !value.y.is_nan() && !value.z.is_nan()
                                    })
                                    .collect(),
                                rotation: node_anim
                                    .rotation_keys()
                                    .map(|key| Keyframe {
                                        time: key.time(),
                                        value: cgmath::Quaternion::<f32>::from(key.value()),
                                    })
                                    .filter(|Keyframe { value, .. }| {
                                        !value.v.x.is_nan()
                                            && !value.v.y.is_nan()
                                            && !value.v.z.is_nan()
                                            && !value.s.is_nan()
                                    })
                                    .collect(),
                                scaling: node_anim
                                    .scaling_keys()
                                    .map(|key| Keyframe {
                                        time: key.time(),
                                        value: cgmath::Vector3::<f32>::from(key.value()),
                                    })
                                    .filter(|Keyframe { value, .. }| {
                                        !value.x.is_nan() && !value.y.is_nan() && !value.z.is_nan()
                                    })
                                    .collect(),
                            },
                        )
                    })
                    .collect(),
            })
            .collect();

        Ok(Model {
            vert_offset,
            norm_offset,
            index_ranges,
            model_data_offset,

            // TODO
            animations,
            current_animation: None,
            transferred_data: None,
            bone_range,
            nodes,

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
    type Offsets = (VertexOffset<TexturedVertex>, VertexOffset<ModelVertex>);

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
                bone_offset: self.bone_range.start * mem::size_of::<[[f32; 4]; 4]>() as u64,
            },
        }
    }
}
