use std::ops::Range;

pub struct MdlAsset<R>(pub goldsrc_mdl::Mdl<R>);

pub struct Model {
    vert_offset: u64,
    index_ranges: Range<u32>,
}
