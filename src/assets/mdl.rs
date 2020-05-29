use crate::{
    cache::{Atlas, Cache},
    loader::{Load, LoadAsset, Loader},
    render::{Render, RenderCache, RenderContext},
};

use itertools::Itertools;
use std::ops::Range;

pub struct MdlAsset<'a>(pub assimp::Scene<'a>);

pub struct Model {
    vert_offset: u64,
    index_ranges: Range<u32>,
}
/*
impl LoadAsset for MdlAsset<'_> {
    type Asset = Model;
    #[inline]
    fn load(self, loader: &Loader, cache: &mut RenderCache) -> anyhow::Result<Self::Asset> {
        for mesh in self.0.mesh_iter() {
            debug_assert_eq!(
                mesh.primitive_types(),
                assimp::import::structs::PrimitiveTypes::TRIANGLE
            );

            cache.vertices.extend(
                mesh.vertex_iter()
                    .zip_longest(mesh.texture_coords_iter())
                    .map(|(position, uvs)| {
                        let position: cgmath::Vector3 = position.into();

                        Vertex {
                            pos: [vert.x(), vert.y(), vert.z(), 1.],
                            tex: [
                                tex_rect.x as f32,
                                tex_rect.y as f32,
                                tex_rect.width as f32,
                                tex_rect.height as f32,
                            ],
                            tex_coord: [u, v],
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
                        }
                    }),
            );
        }

        unimplemented!()
    }
}
*/
