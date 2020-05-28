use crate::{
    cache::{Atlas, Cache},
    loader::{Load, LoadAsset, Loader},
    render::{Render, RenderCache, RenderContext, Vertex},
};

use std::ops::Range;

pub struct MdlAsset;
// pub struct MdlAsset<'a>(pub assimp::Scene<'a>);

pub struct Model {
    vert_offset: u64,
    index_ranges: Range<u32>,
}

// impl LoadAsset for MdlAsset<'_> {
//     type Asset = Model;

//     #[inline]
//     fn load(self, loader: &Loader, cache: &mut RenderCache) -> anyhow::Result<Self::Asset> {
//         unimplemented!()
//     }
// }
