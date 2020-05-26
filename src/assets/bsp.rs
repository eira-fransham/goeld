use crate::{
    cache::{Atlas, Cache},
    loader::{Load, LoadAsset, Loader},
    render::{Render, RenderCache, RenderContext, Vertex},
};
use cgmath::{InnerSpace, Matrix};
use fnv::FnvHashMap as HashMap;
use std::ops::Range;

pub struct BspAsset(pub bsp::Bsp);

pub struct World {
    vis: bsp::Vis,
    vert_offset: u64,
    // Key is `(model, cluster)`
    cluster_ranges: Vec<Range<u32>>,
    model_ranges: Vec<Range<u32>>,
}

#[inline]
fn leaf_meshes<'a, F>(
    bsp: &'a bsp::Bsp,
    face_start_indices: &'a mut Vec<u32>,
    mut get_texture: F,
    lightmap_cache: &mut Atlas,
) -> (
    Vec<Vertex>,
    impl ExactSizeIterator<
            Item = (
                &'a bsp::Model,
                impl Iterator<Item = (&'a bsp::Leaf, impl Iterator<Item = u32> + Clone + 'a)>
                    + Clone
                    + 'a,
            ),
        > + Clone
        + 'a,
)
where
    F: FnMut(&bsp::Texture) -> Option<rect_packer::Rect>,
{
    // We'll probably need to reallocate a few times since vertices are reused,
    // but this is a reasonable lower bound
    let mut vertices = Vec::with_capacity(bsp.vertices.len());

    for face in bsp.faces() {
        face_start_indices.push(vertices.len() as u32);

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

        vertices.extend(face.vertices().map(|vert| {
            let (u, v) = (
                vert.dot(&texture.axis_u) + texture.offset_u,
                vert.dot(&texture.axis_v) + texture.offset_v,
            );

            Vertex {
                pos: [vert.x(), vert.y(), vert.z(), 1.],
                tex: [
                    tex_rect.x as f32,
                    tex_rect.y as f32,
                    tex_rect.width as f32,
                    tex_rect.height as f32,
                ],
                tex_coord: [u, v],
                value: texture.value as f32 / u8::max_value() as f32,
                lightmap_coord: lightmap
                    .map(|((minu, minv), lightmap_result)| {
                        [
                            (lightmap_result.first.x as f32 + (u / 16.).floor() - minu),
                            (lightmap_result.first.y as f32 + (v / 16.).floor() - minv),
                        ]
                    })
                    .unwrap_or([0., 0.]),
                lightmap_width: lightmap
                    .map(|(_, lightmap_result)| lightmap_result.stride_x as f32)
                    .unwrap_or_default(),
                lightmap_count: count,
            }
        }));
    }

    let face_start_indices: &'a [u32] = &*face_start_indices;

    (
        vertices,
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
                            leaf.leaf_faces().flat_map(move |leaf_face| {
                                let start = face_start_indices[leaf_face.face as usize] as u32;
                                let face = leaf_face.face();

                                (1..face.vertices().len().saturating_sub(1))
                                    .flat_map(|face_number| {
                                        use std::iter::once;

                                        once(0)
                                            .chain(once(face_number))
                                            .chain(once(face_number + 1))
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
        use std::{collections::hash_map::Entry, path::Path};

        let mut buf = Vec::new();
        let Self(bsp) = self;

        let missing = cache.diffuse.append(
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

        let RenderCache {
            diffuse,
            lightmap,
            vertices,
            indices,
        } = cache;

        let mut get_texture = move |texture: &bsp::Texture| {
            if texture.flags.contains(bsp::SurfaceFlags::NODRAW)
                || texture.flags.contains(bsp::SurfaceFlags::SKY)
            {
                None
            } else {
                let rect = (|| match texture_map.entry(texture.name.clone()) {
                    Entry::Occupied(val) => Some(val.get().clone()),
                    Entry::Vacant(entry) => {
                        let (file, path) = loader.load(Path::new(&texture.name[..]).into()).ok()?;

                        let rect = diffuse.append(
                            image::load(
                                std::io::BufReader::new(file),
                                image::ImageFormat::from_path(&path).ok()?,
                            )
                            .ok()?,
                        );

                        Some(entry.insert(rect).clone())
                    }
                })();

                Some(rect.unwrap_or(missing.clone()))
            }
        };

        let (vert_offset, model_ranges, cluster_ranges) = {
            use std::convert::TryInto;

            let (leaf_vertices, mut model_indices) =
                leaf_meshes(&bsp, &mut buf, &mut get_texture, lightmap);
            let mut clusters = vec![vec![]; bsp.clusters().count()];

            if let Some((_, leaf_indices)) = model_indices.next() {
                for (leaf, iterator) in leaf_indices {
                    if let Ok(c) = leaf.cluster.try_into() {
                        clusters.get_mut::<usize>(c).unwrap().push(iterator);
                    }
                }
            }

            let leaf_vertices = vertices.append(leaf_vertices);

            let (base_model_range, cluster_ranges): (_, Result<Vec<_>, _>) = indices.append_many(
                clusters
                    .into_iter()
                    .map(|iterators| iterators.into_iter().flat_map(|i| i)),
            );

            let cluster_ranges = cluster_ranges?;

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

            (leaf_vertices.start, model_ranges, cluster_ranges)
        };

        Ok(World {
            vis: bsp.vis,
            vert_offset,
            cluster_ranges,
            model_ranges,
        })
    }
}

pub struct WorldIndexIter<'a> {
    clusters: hack::ImplTraitHack<'a>,
    cluster_ranges: &'a [Range<u32>],
    model_ranges: std::slice::Iter<'a, Range<u32>>,
    models: std::slice::Iter<'a, bsp::Model>,
    clipper: Clipper,
}

impl Iterator for WorldIndexIter<'_> {
    type Item = Range<u32>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            clusters,
            cluster_ranges,

            model_ranges,
            models,
            clipper,
        } = self;
        let cluster_ranges = &*cluster_ranges;
        clusters
            .next()
            .map(move |cluster| cluster_ranges[cluster as usize].clone())
            .or_else(move || loop {
                let model = models.next()?;
                let range = model_ranges.next()?;
                let bsp::BoundingSphere {
                    center,
                    radius_squared,
                } = model.bounding_sphere();

                if clipper.check_sphere(center.0.into(), radius_squared.sqrt()) {
                    break Some(range.clone());
                }
            })
    }
}

struct Clipper {
    planes: [cgmath::Vector4<f32>; 6],
}

impl Clipper {
    #[inline]
    pub fn new(matrix: cgmath::Matrix4<f32>) -> Self {
        let planes = [
            (matrix.row(3) + matrix.row(0)),
            (matrix.row(3) - matrix.row(0)),
            (matrix.row(3) - matrix.row(1)),
            (matrix.row(3) + matrix.row(1)),
            (matrix.row(3) + matrix.row(2)),
            (matrix.row(3) - matrix.row(2)),
        ];

        Self {
            planes: [
                planes[0] * (1.0 / planes[0].truncate().magnitude()),
                planes[1] * (1.0 / planes[1].truncate().magnitude()),
                planes[2] * (1.0 / planes[2].truncate().magnitude()),
                planes[3] * (1.0 / planes[3].truncate().magnitude()),
                planes[4] * (1.0 / planes[4].truncate().magnitude()),
                planes[5] * (1.0 / planes[5].truncate().magnitude()),
            ],
        }
    }

    /// Check if the given sphere is within the Frustum
    #[inline]
    pub fn check_sphere(&self, center: cgmath::Vector3<f32>, radius: f32) -> bool {
        for plane in &self.planes {
            if plane.truncate().dot(center) + plane.w <= -radius {
                return false;
            }
        }

        true
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

impl<'a> Render for &'a World {
    type Indices = WorldIndexIter<'a>;

    #[inline]
    fn indices(self, ctx: &RenderContext) -> (u64, Self::Indices) {
        let pos: [f32; 3] = ctx.camera.position.into();
        let cluster_ranges: &'a [Range<u32>] = &self.cluster_ranges;
        let vis = &self.vis;
        let clipper = Clipper::new(ctx.camera.matrix());

        let cluster = vis
            .model(0)
            .unwrap()
            .cluster_at::<bsp::XEastYSouthZUp, _>(pos);

        let model_start_index = if cluster.is_some() { 1 } else { 0 };

        let clusters = hack::impl_trait_hack(vis, cluster);

        (
            self.vert_offset,
            WorldIndexIter {
                clusters,
                cluster_ranges,
                models: self.vis.models[model_start_index..].iter(),
                model_ranges: self.model_ranges[model_start_index..].iter(),
                clipper,
            },
        )
    }
}
