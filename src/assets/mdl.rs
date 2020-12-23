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
use goldsrc_mdl as mdl;
use image::ImageBuffer;
use std::{
    borrow::Cow,
    collections::hash_map::Entry,
    io, mem,
    ops::{Add, Mul, Range},
};

pub struct MdlAsset<R> {
    pub main: mdl::Mdl<R>,
    pub textures: Option<mdl::Mdl<R>>,
}

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

#[derive(Debug, Clone)]
pub struct Model {
    vert_offset: u64,
    model_vert_offset: u64,
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

    pub fn update(&mut self, cache: &mut RenderCache) {}
}

// We don't support scaling, and the order is always: rotate around bone origin, translate
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Bone {
    pub parent: Option<u8>,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
}

impl<R> LoadAsset for MdlAsset<R>
where
    R: io::Read + io::Seek,
{
    type Asset = Model;

    #[inline]
    fn load(mut self, loader: &Loader, cache: &mut RenderCache) -> anyhow::Result<Self::Asset> {
        use crate::loader::Load;
        use cgmath::SquareMatrix as _;
        use std::convert::TryFrom;

        let mut textures = Vec::with_capacity(
            self.main.textures.len()
                + self
                    .textures
                    .as_ref()
                    .map(|t| t.textures.len())
                    .unwrap_or(0),
        );

        let mut texture_iter = self.main.textures();
        while let Some(mut tex) = texture_iter.next()? {
            textures.push(cache.diffuse.append(tex.data()?));
        }

        if let Some(mut tex) = self.textures {
            let mut texture_iter = tex.textures();
            while let Some(mut tex) = texture_iter.next()? {
                textures.push(cache.diffuse.append(tex.data()?));
            }
        }

        let mut bodyparts = self.main.bodyparts();

        let vert_offset = cache.textured_vertices.len();
        let mut vertices = Vec::new();
        let mut model_vertices = Vec::new();
        let mut indices = Vec::new();

        while let Some((_name, mut models)) = bodyparts.next()? {
            let mut model = if let Some(main_model) = models.next()? {
                main_model
            } else {
                continue;
            };

            let start = vertices.len();
            vertices.extend(model.vertices()?.map(|qv| TexturedVertex {
                pos: qv.unwrap().into(),
                tex_coord: [0.; 2],
                // TODO: Textures
                atlas_texture: [0; 4],
            }));
            let end = vertices.len();

            model_vertices.extend(model.vertex_bones()?.map(|bone| ModelVertex {
                normal: Default::default(),
                bone_id: bone.unwrap() as _,
            }));

            let mut meshes = model.meshes()?;
            while let Some(mut mesh) = meshes.next()? {
                let texture = textures.get(mesh.skin_ref as usize).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "Texture ref out of range: {} >= {}",
                            mesh.skin_ref,
                            textures.len()
                        ),
                    )
                })?;

                for trivert in mesh.triverts()? {
                    let trivert = trivert?;

                    let cur_index = u32::try_from(vertices.len())?;

                    let vert = vertices[start..end]
                        .get_mut(trivert.position_index as usize)
                        .ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "Vertex out of range: {} >= {}",
                                    trivert.position_index as usize, cur_index
                                ),
                            )
                        })?;
                    let trivert_uv = [trivert.u as f32, trivert.v as f32];
                    let trivert_atlas = [
                        texture.x as u32,
                        texture.y as u32,
                        texture.width as u32,
                        texture.height as u32,
                    ];

                    let index = if (vert.tex_coord, vert.atlas_texture) == Default::default()
                        || (vert.tex_coord, vert.atlas_texture) == (trivert_uv, trivert_atlas)
                    {
                        vert.tex_coord = trivert_uv;
                        vert.atlas_texture = trivert_atlas;

                        u32::try_from(start)? + trivert.position_index as u32
                    } else {
                        let vert = TexturedVertex {
                            tex_coord: trivert_uv,
                            atlas_texture: trivert_atlas,
                            ..vert.clone()
                        };

                        vertices.push(vert);
                        model_vertices.push(model_vertices[trivert.position_index as usize]);

                        cur_index
                    };

                    indices.push(index);
                }
            }
        }

        debug_assert_eq!(model_vertices.len(), vertices.len());

        let vert_offset = cache.textured_vertices.append(vertices).start;
        let model_vert_offset = cache.model_vertices.append(model_vertices).start;
        let index_range = cache.indices.append(indices);
        let index_ranges = vec![index_range.start as u32..index_range.end as u32];

        let mut nodes = HashMap::default();

        let model_data_offset = cache
            .model_data
            .append(std::iter::once(ModelData {
                translation: cgmath::Matrix4::identity(),
                origin: cgmath::Vector3::from([0., 0., 0.]),
            }))
            .start;

        let bones = &self.main.bones;
        let bone_range = cache.bone_matrices.append(bones.iter().map(|mut bone| {
            use std::convert::TryFrom;

            let mut out = cgmath::Matrix4::from(bone.value.clone());

            while let Some(parent) = usize::try_from(bone.parent).ok().and_then(|p| bones.get(p)) {
                bone = parent;
                out = cgmath::Matrix4::from(bone.value.clone()) * out;
            }

            out.into()
        }));

        Ok(Model {
            vert_offset,
            model_vert_offset,
            index_ranges,
            model_data_offset,

            // TODO
            animations: Default::default(),
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
            offsets: (self.vert_offset.into(), self.model_vert_offset.into()),
            indices: self.index_ranges.iter().cloned(),
            pipeline: PipelineDesc::Models {
                origin: self.position,
                data_offset: self.model_data_offset,
                bone_offset: self.bone_range.start,
            },
        }
    }
}
