use crate::{
    cache::{Atlas, Cache},
    loader::{Load, LoadAsset, Loader},
    render::{
        Light, PipelineDesc, Render, RenderCache, RenderMesh, TexturedVertex, VertexOffset,
        World as RenderWorld, WorldVertex,
    },
};
use cgmath::{InnerSpace, Point3};
use collision::{Aabb3, Frustum, Relation};
use fnv::FnvHashMap as HashMap;
use std::ops::Range;

pub struct BspAsset(pub bsp::Bsp);

#[derive(Debug, Clone, PartialEq)]
struct ClusterMeta {
    aabb: Aabb3<f32>,
    index_range: Range<u32>,
}

pub struct World {
    vis: bsp::Vis,
    last_cluster: Option<u16>,
    tex_vert_offset: u64,
    world_vert_offset: u64,
    // Key is `(model, cluster)`
    cluster_meta: Vec<ClusterMeta>,
    model_ranges: Vec<Range<u32>>,

    cluster_lights: Vec<Range<u32>>,
}

#[inline]
fn leaf_meshes<'a, F>(
    bsp: &'a bsp::Bsp,
    face_start_indices: &'a mut HashMap<u16, u32>,
    mut get_texture: F,
    lightmap_cache: &mut Atlas,
) -> (
    Vec<TexturedVertex>,
    Vec<WorldVertex>,
    impl ExactSizeIterator<
            Item = (
                &'a bsp::Q2Model,
                impl Iterator<Item = (&'a bsp::Q2Leaf, impl Iterator<Item = u32> + Clone + 'a)>
                    + Clone
                    + 'a,
            ),
        > + Clone
        + 'a,
)
where
    F: FnMut(&bsp::Q2Texture) -> Option<rect_packer::Rect>,
{
    // We'll probably need to reallocate a few times since vertices are reused,
    // but this is a reasonable lower bound
    let mut tex_vertices = Vec::with_capacity(bsp.vertices.len());
    let mut world_vertices = Vec::with_capacity(bsp.vertices.len());

    for (i, face) in bsp.faces().enumerate() {
        let texture = if let Some(texture) = face.texture() {
            texture
        } else {
            continue;
        };

        let tex_rect = if let Some(tex_rect) = get_texture(&texture) {
            tex_rect
        } else {
            continue;
        };

        let (mins, _, w, h) = face.lightmap_dimensions().unwrap_or_default();

        let (count, lightmap) = if let Some(lightmaps) = face.lightmaps() {
            let count = lightmaps.len() as u32;

            (
                count,
                Some((
                    mins,
                    lightmap_cache.append_many(w, h, lightmaps.map(|l| l.as_image())),
                )),
            )
        } else {
            (0, None)
        };

        debug_assert_eq!(tex_vertices.len(), world_vertices.len());
        face_start_indices.insert(i as u16, tex_vertices.len() as u32);

        for (tex_vert, world_vert) in face.vertices().map(|vert| {
            let (u, v) = (
                vert.dot(&texture.offsets.axis_u) + texture.offsets.offset_u,
                vert.dot(&texture.offsets.axis_v) + texture.offsets.offset_v,
            );
            (
                TexturedVertex {
                    pos: [vert.x(), vert.y(), vert.z(), 1.],
                    tex_coord: [u, v],
                    atlas_texture: [
                        tex_rect.x as u32,
                        tex_rect.y as u32,
                        tex_rect.width as u32,
                        tex_rect.height as u32,
                    ],
                },
                WorldVertex {
                    count: texture.frames().count() as u32,
                    value: texture.value as f32 / 255.,
                    lightmap_coord: lightmap
                        .map(|((minu, minv), lightmap_result)| {
                            [
                                (lightmap_result.first.x as f32 + (u / 16.).floor() - minu),
                                (lightmap_result.first.y as f32 + (v / 16.).floor() - minv),
                            ]
                        })
                        .unwrap_or_default(),
                    lightmap_width: lightmap
                        .map(|(_, lightmap_result)| lightmap_result.stride_x as f32)
                        .unwrap_or_default(),
                    lightmap_count: count,
                },
            )
        }) {
            tex_vertices.push(tex_vert);
            world_vertices.push(world_vert);
        }
    }

    let face_start_indices: &'a HashMap<_, _> = &*face_start_indices;

    (
        tex_vertices,
        world_vertices,
        bsp.models().map(move |model| {
            (
                model.data,
                model
                    .leaves()
                    .into_iter()
                    .flatten()
                    .map(move |leaf| {
                        (
                            leaf,
                            leaf.leaf_faces()
                                .filter_map(move |leaf_face| {
                                    Some((leaf_face, face_start_indices.get(&leaf_face.face)?))
                                })
                                .flat_map(move |(leaf_face, start)| {
                                    let face = leaf_face.face();

                                    (1..face.vertices().len().saturating_sub(1))
                                        .flat_map(|face_number| {
                                            use std::iter::once;

                                            once(0)
                                                .chain(once(face_number + 1))
                                                .chain(once(face_number))
                                        })
                                        .map(move |i| i as u32 + start)
                                }),
                        )
                    })
                    .map(|(leaf_handle, indices)| (leaf_handle.data, indices)),
            )
        }),
    )
}

impl LoadAsset for BspAsset {
    type Asset = World;

    #[inline]
    fn load(self, loader: &Loader, cache: &mut RenderCache) -> anyhow::Result<Self::Asset> {
        use image::GenericImageView;
        use std::{collections::hash_map::Entry, path::Path};

        let mut buf = Default::default();
        let Self(bsp) = self;

        let RenderCache {
            diffuse,
            lightmap,
            textured_vertices,
            world_vertices,
            indices,
            lights,
            ..
        } = cache;

        let bsp_lights = bsp
            .entities
            .iter()
            .filter(|ent| ent.properties().any(|kv| kv == ("classname", "light")))
            .filter_map(|ent| {
                ent.properties()
                    .find_map(|(key, val)| if key == "origin" { Some(val) } else { None })
                    .map(|origin| {
                        (
                            origin,
                            ent.properties()
                                .find_map(|(key, val)| {
                                    if key.starts_with("light") {
                                        val.parse::<f32>().ok()
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(300.0),
                        )
                    })
            })
            .map(|(origin, light)| {
                let pos = origin.find(' ').unwrap();
                let x = origin[..pos].parse::<f32>().unwrap();
                let origin = &origin[pos + 1..];

                let pos = origin.find(' ').unwrap();
                let y = origin[..pos].parse::<f32>().unwrap();
                let origin = &origin[pos + 1..];

                let z = origin.parse::<f32>().unwrap();

                ([x, y, z], light)
            })
            .collect::<Vec<_>>();

        let mut cluster_lights = HashMap::<u16, Vec<Light>>::with_hasher(Default::default());

        for (pos, intensity) in bsp_lights {
            let cluster = bsp
                .vis
                .cluster_at::<bsp::XEastYSouthZUp, _>(bsp.vis.root_node().unwrap(), pos);

            if let Some(cluster) = cluster {
                cluster_lights.entry(cluster).or_default().push(Light {
                    position: [pos[0], pos[1], pos[2], 1.],
                    color: [1., 1., 1., intensity].into(),
                });
            }
        }

        let missing = diffuse.append(
            image::load(
                std::io::Cursor::new(
                    &include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/missing.png"))[..],
                ),
                image::ImageFormat::Png,
            )
            .unwrap(),
        );

        let loader = loader.textures();
        let mut texture_map: HashMap<_, rect_packer::Rect> =
            HashMap::with_capacity_and_hasher(bsp.textures.len(), Default::default());

        let bsp_ref = &bsp;

        let mut get_texture = move |texture: &bsp::Q2Texture| {
            if texture.flags.contains(bsp::SurfaceFlags::NODRAW)
                || texture.flags.contains(bsp::SurfaceFlags::SKY)
            {
                None
            } else {
                let rect = (|| match texture_map.entry(
                    bsp::Handle::new(bsp_ref, texture)
                        .frames()
                        .map(|t| t.name.clone())
                        .collect::<Box<[_]>>(),
                ) {
                    Entry::Occupied(val) => Some(val.get().clone()),
                    Entry::Vacant(entry) => {
                        let (file, path) = loader.load(Path::new(&texture.name[..]).into()).ok()?;

                        let first = image::load(
                            std::io::BufReader::new(file),
                            image::ImageFormat::from_path(&path).ok()?,
                        )
                        .ok()?;

                        let width = first.width();
                        let height = first.height();
                        let frames =
                            std::iter::once(Ok(first))
                                .chain(bsp::Handle::new(bsp_ref, texture).frames().skip(1).map(
                                    |t| {
                                        let (file, path) =
                                            loader.load(Path::new(&t.name[..]).into())?;

                                        image::load(
                                            std::io::BufReader::new(file),
                                            image::ImageFormat::from_path(&path)?,
                                        )
                                    },
                                ))
                                .collect::<Result<Vec<_>, _>>()
                                .ok()?;

                        let appended = diffuse.append_many(width, height, frames.into_iter());

                        Some(entry.insert(appended.first).clone())
                    }
                })();

                Some(rect.unwrap_or(missing.clone()))
            }
        };

        let (tex_vert_offset, world_vert_offset, model_ranges, cluster_meta) = {
            use std::convert::TryInto;

            let (leaf_tex_vertices, leaf_world_vertices, mut model_indices) =
                leaf_meshes(&bsp, &mut buf, &mut get_texture, lightmap);
            let mut clusters = vec![
                (vec![], Point3::from([0f32; 3]), Point3::from([0f32; 3]));
                bsp.clusters().count()
            ];

            if let Some((model, leaf_indices)) = model_indices.next() {
                let (model_mins, model_maxs) =
                    (Point3::from(model.mins.0), Point3::from(model.maxs.0));

                for (leaf, iterator) in leaf_indices {
                    if let Ok(c) = leaf.cluster.try_into() {
                        let (iterators, mins, maxs) = clusters.get_mut::<usize>(c).unwrap();
                        iterators.push(iterator);

                        mins.x = mins.x.min(model_mins.x);
                        mins.y = mins.y.min(model_mins.y);
                        mins.z = mins.z.min(model_mins.z);
                        maxs.x = maxs.x.max(model_maxs.x);
                        maxs.y = maxs.y.max(model_maxs.y);
                        maxs.z = maxs.z.max(model_maxs.z);
                    }
                }
            }

            let leaf_tex_vertices = textured_vertices.append(leaf_tex_vertices);
            let leaf_world_vertices = world_vertices.append(leaf_world_vertices);

            let (base_model_range, cluster_ranges): (_, Result<Vec<_>, _>) = indices.append_many(
                clusters
                    .iter_mut()
                    .map(|(iterators, _, _)| iterators.drain(..).flatten()),
            );

            let cluster_meta = cluster_ranges?
                .into_iter()
                .zip(
                    clusters
                        .into_iter()
                        .map(|(_, mins, maxs)| Aabb3::new(mins, maxs)),
                )
                .map(|(index_range, aabb)| ClusterMeta { index_range, aabb })
                .collect::<Vec<_>>();

            let mut model_ranges = Vec::with_capacity(bsp.vis.models.len());

            model_ranges.push(base_model_range.start as u32..base_model_range.end as u32);

            struct ThrowAway;

            impl<T> std::iter::FromIterator<T> for ThrowAway {
                #[inline]
                fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
                    for _ in iter {}

                    ThrowAway
                }
            }

            for (_, leaf_indices) in model_indices {
                let (range, _): (_, ThrowAway) =
                    indices.append_many::<u64, _, _, _>(leaf_indices.map(|(_, iter)| iter));

                model_ranges.push(range.start as u32..range.end as u32);
            }

            (
                leaf_tex_vertices.start,
                leaf_world_vertices.start,
                model_ranges,
                cluster_meta,
            )
        };

        let cluster_lights = bsp
            .clusters()
            .map(|cluster| {
                let mut bsp_lights = bsp
                    .vis
                    .visible_clusters(cluster, ..)
                    .flat_map(|cluster| cluster_lights.get(&cluster))
                    .flat_map(|lights| lights.iter().cloned())
                    .collect::<Vec<_>>();
                let aabb = cluster_meta[cluster as usize].aabb.clone();
                let center = (aabb.min.to_homogeneous() + aabb.max.to_homogeneous()) / 2.;

                bsp_lights.sort_unstable_by(|a, b| {
                    let a_score = (cgmath::Vector4::from(a.position) - center).magnitude2()
                        * cgmath::Vector4::from(a.color).magnitude2();
                    let b_score = (cgmath::Vector4::from(b.position) - center).magnitude2()
                        * cgmath::Vector4::from(b.color).magnitude2();

                    a_score
                        .partial_cmp(&b_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                bsp_lights.dedup();
                bsp_lights.truncate(crate::render::MAX_LIGHTS);

                let range = lights.append(bsp_lights);

                range.start as u32..range.end as u32
            })
            .collect();

        Ok(World {
            vis: bsp.vis,
            tex_vert_offset,
            world_vert_offset,
            cluster_meta,
            model_ranges,
            cluster_lights,
            last_cluster: None,
        })
    }
}

pub struct WorldIndexIter<'a> {
    clusters: hack::ImplTraitHack<'a>,
    cluster_meta: &'a [ClusterMeta],
    model_ranges: std::slice::Iter<'a, Range<u32>>,
    models: std::slice::Iter<'a, bsp::Q2Model>,
    clipper: Frustum<f32>,
}

impl Iterator for WorldIndexIter<'_> {
    type Item = Range<u32>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            clusters,
            cluster_meta,
            model_ranges,
            models,
            clipper,
        } = self;
        let clipper = &*clipper;

        clusters
            .filter_map(move |cluster| {
                let meta = cluster_meta[cluster as usize].clone();

                if clipper.contains(&meta.aabb) != Relation::Out {
                    Some(meta.index_range)
                } else {
                    None
                }
            })
            .next()
            .or_else(move || loop {
                let model = models.next()?;
                let range = model_ranges.next()?;

                if clipper.contains(&Aabb3::new(model.mins.0.into(), model.maxs.0.into()))
                    != Relation::Out
                {
                    break Some(range.clone());
                }
            })
    }
}

mod hack {
    pub type ImplTraitHack<'a> = impl Iterator<Item = u16> + Clone + 'a;

    #[inline]
    pub fn impl_trait_hack(vis: &bsp::Vis, cluster: Option<u16>) -> ImplTraitHack<'_> {
        cluster
            .into_iter()
            .flat_map(move |cluster| vis.visible_clusters(cluster, ..))
    }
}

impl<'a> Render for &'a mut World {
    type Indices = WorldIndexIter<'a>;
    type Offsets = (VertexOffset<TexturedVertex>, VertexOffset<WorldVertex>);

    #[inline]
    fn indices<T: crate::render::Context>(
        self,
        ctx: &T,
    ) -> RenderMesh<Self::Offsets, Self::Indices> {
        let pos: [f32; 3] = ctx.camera().position.into();
        let cluster_meta = &self.cluster_meta;
        let vis = &self.vis;
        let clipper = Frustum::<f32>::from_matrix4(ctx.camera().matrix()).unwrap();

        let cluster = vis
            .model(0)
            .unwrap()
            .cluster_at::<bsp::XEastYSouthZUp, _>(pos)
            .or(self.last_cluster);

        self.last_cluster = cluster;

        // TODO: We should separate these somewhat so we have a way to move models around
        let model_start_index = 1;

        let clusters = hack::impl_trait_hack(vis, cluster);

        RenderMesh {
            offsets: (self.tex_vert_offset.into(), self.world_vert_offset.into()),
            indices: WorldIndexIter {
                clusters,
                cluster_meta,
                models: self.vis.models[model_start_index..].iter(),
                model_ranges: self.model_ranges[model_start_index..].iter(),
                clipper,
            },
            pipeline: PipelineDesc::World,
        }
    }
}

impl RenderWorld for World {
    #[inline]
    fn lights(&self, position: cgmath::Vector3<f32>) -> Option<Range<u32>> {
        let pos: [f32; 3] = position.into();

        self.cluster_lights
            .get(
                self.vis
                    .model(0)
                    .unwrap()
                    .cluster_at::<bsp::XEastYSouthZUp, _>(pos)? as usize,
            )
            .cloned()
    }
}
