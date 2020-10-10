#![cfg_attr(feature = "bench", feature(test))]

// TODO: Support other similar BSP versions (especially GoldSrc BSP)

#[cfg(feature = "bench")]
extern crate test;

use arrayvec::ArrayString;
use bitflags::bitflags;
use bv::BitVec;
pub use goldsrc_format_common::{
    parseable, CoordSystem, ElementSize, QVec, SimpleParse, XEastYDownZSouth, XEastYNorthZUp,
    XEastYSouthZUp, V3,
};
use std::{
    convert::{TryFrom, TryInto},
    fmt,
    io::{self, ErrorKind, Read, Seek, SeekFrom},
    iter::{self, FromIterator},
    ops::Deref,
};

#[cfg(not(debug_assertions))]
#[inline]
fn error(msg: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

#[cfg(debug_assertions)]
#[inline]
fn error(msg: impl ToString) -> io::Error {
    panic!("{}", msg.to_string())
}

macro_rules! magic {
    (struct $name:ident($magic:expr);) => {
        #[derive(PartialEq, Default, Copy, Clone)]
        pub struct $name;

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                $magic.fmt(f)
            }
        }

        impl $name {
            pub const fn into_inner(self) -> [u8; 4] {
                $magic
            }
        }

        impl Deref for $name {
            type Target = [u8; 4];

            fn deref(&self) -> &Self::Target {
                &$magic
            }
        }

        impl ElementSize for $name {
            const SIZE: usize = <[u8; 4]>::SIZE;
        }

        impl SimpleParse for $name {
            fn parse<R: io::Read>(r: &mut R) -> io::Result<Self> {
                let val = <[u8; 4]>::parse(r)?;

                if val == $magic {
                    Ok($name)
                } else {
                    Err(error(format!(
                        "Invalid magic number: expected {:?}, got {:?}",
                        $magic, val
                    )))
                }
            }
        }
    };
}

#[derive(Debug, Default, PartialEq)]
pub struct GenericDirectories {
    area_portals: DirEntry,
    areas: DirEntry,
    brushes: DirEntry,
    brush_sides: DirEntry,

    // Goldsrc-specific
    clipnodes: DirEntry,

    edges: DirEntry,
    entities: DirEntry,
    faces: DirEntry,
    leaf_brushes: DirEntry,
    leaf_faces: DirEntry,
    leaves: DirEntry,
    lightmaps: DirEntry,

    // Goldsrc-specific
    miptexes: DirEntry,

    models: DirEntry,
    nodes: DirEntry,
    planes: DirEntry,
    surf_edges: DirEntry,
    textures: DirEntry,
    vertices: DirEntry,
    visdata: DirEntry,
}

pub trait BspFormat {
    const VERSION: u32;

    type Magic: SimpleParse;
    type Directories: ElementSize + SimpleParse + Into<GenericDirectories>;
    type Texture: BspTexture<Self> + SimpleParse + ElementSize + Clone;
    type Miptexes: ReadEntry + Clone + Default;
    type Model: BspModel + SimpleParse + ElementSize + Clone;
    type Leaf: BspLeaf + SimpleParse + ElementSize + Clone;
    type VisData: ReadEntry + Clone + Default;
}

pub trait ReadEntry: Sized {
    fn read<R: io::Read + io::Seek>(reader: R, entry: &DirEntry) -> io::Result<Self>;
}

impl ReadEntry for () {
    fn read<R: io::Read + io::Seek>(reader: R, entry: &DirEntry) -> io::Result<Self> {
        Ok(())
    }
}

pub trait BspTexture<B: BspFormat + ?Sized>: Sized {
    type Format: image::Pixel;

    fn name<'a>(&'a self, bsp: &'a Bsp<B>) -> &'a str;
    fn data(&self, bsp: &Bsp<B>) -> Option<image::ImageBuffer<Self::Format, &[u8]>>;
    fn offsets(&self) -> &TextureOffsets;
}

pub trait BspLeaf {
    fn cluster(&self) -> i16;
}

pub trait BspModel {
    type Hulls: Iterator<Item = u32>;

    fn hulls(&self) -> Self::Hulls;
}

pub enum Quake2 {}

const QUAKE2_MAGIC: [u8; 4] = [b'I', b'B', b'S', b'P'];

#[derive(Debug, Clone, Copy)]
pub enum NotApplicable {}

impl SimpleParse for NotApplicable {
    fn parse<R: io::Read>(_r: &mut R) -> io::Result<Self> {
        unreachable!()
    }
}

impl ElementSize for NotApplicable {
    const SIZE: usize = usize::MAX;
}

magic! {
    struct Q2Magic(QUAKE2_MAGIC);
}

impl BspFormat for Quake2 {
    type Magic = Q2Magic;

    const VERSION: u32 = 0x26;

    type Directories = Quake2Directories;
    type Texture = Q2Texture;
    type Miptexes = ();
    type Model = Q2Model;
    type Leaf = Q2Leaf;
    type VisData = Q2VisData;
}

pub enum Goldsrc {}

impl BspFormat for Goldsrc {
    const VERSION: u32 = 0x1e;

    type Magic = ();
    type Directories = GoldsrcDirectories;
    type Texture = GoldsrcTexture;
    type Miptexes = Box<[GoldsrcMiptex]>;
    type Model = GoldsrcModel;
    type Leaf = GoldsrcLeaf;
    type VisData = GoldsrcVisData;
}

parseable! {
    #[derive(Debug, Clone, Default, PartialEq)]
    pub struct GoldsrcMiptex {
        pub name: ArrayString<[u8; 16]>,
        pub extents: [u32; 2],
        pub offsets: [u32; 4], // TODO: We don't support loading these yet
    }
}

impl ReadEntry for Box<[GoldsrcMiptex]> {
    fn read<R: io::Read + io::Seek>(mut reader: R, entry: &DirEntry) -> io::Result<Self> {
        reader.seek(io::SeekFrom::Start(entry.offset as u64))?;

        let count: u32 = SimpleParse::parse(&mut reader)?;

        let offsets: Box<[i32]> = i32::parse_many(&mut reader, count as usize)?;

        let mut miptexes = Vec::with_capacity(count as usize);

        for o in &*offsets {
            reader.seek(io::SeekFrom::Start(
                entry.offset as u64 + u64::try_from(*o).map_err(|e| error(e))?,
            ))?;

            miptexes.push(GoldsrcMiptex::parse(&mut reader)?);
        }

        Ok(miptexes.into_boxed_slice())
    }
}

parseable! {
    #[derive(Debug, Default, PartialEq)]
    pub struct GoldsrcDirectories {
        entities: DirEntry,
        planes: DirEntry,
        miptexes: DirEntry,
        vertices: DirEntry,
        visdata: DirEntry,
        nodes: DirEntry,
        textures: DirEntry,
        faces: DirEntry,
        lightmaps: DirEntry,
        clipnodes: DirEntry,
        leaves: DirEntry,
        leaf_faces: DirEntry,
        edges: DirEntry,
        surf_edges: DirEntry,
        models: DirEntry,
    }
}

impl From<GoldsrcDirectories> for GenericDirectories {
    fn from(other: GoldsrcDirectories) -> Self {
        Self {
            entities: other.entities,
            planes: other.planes,
            miptexes: other.miptexes,
            vertices: other.vertices,
            visdata: other.visdata,
            nodes: other.nodes,
            textures: other.textures,
            faces: other.faces,
            lightmaps: other.lightmaps,
            clipnodes: other.clipnodes,
            leaves: other.leaves,
            leaf_faces: other.leaf_faces,
            edges: other.edges,
            surf_edges: other.surf_edges,
            models: other.models,
            ..Default::default()
        }
    }
}

parseable! {
    #[derive(Debug, Default, PartialEq)]
    pub struct Quake2Directories {
        entities: DirEntry,
        planes: DirEntry,
        vertices: DirEntry,
        visdata: DirEntry,
        nodes: DirEntry,
        textures: DirEntry,
        faces: DirEntry,
        lightmaps: DirEntry,
        leaves: DirEntry,
        leaf_faces: DirEntry,
        leaf_brushes: DirEntry,
        edges: DirEntry,
        surf_edges: DirEntry,
        models: DirEntry,
        brushes: DirEntry,
        brush_sides: DirEntry,
        // Appears to be unused, even in Quake 2 itself
        pop: DirEntry,
        areas: DirEntry,
        area_portals: DirEntry,
    }
}

impl From<Quake2Directories> for GenericDirectories {
    fn from(other: Quake2Directories) -> Self {
        Self {
            entities: other.entities,
            planes: other.planes,
            vertices: other.vertices,
            visdata: other.visdata,
            nodes: other.nodes,
            textures: other.textures,
            faces: other.faces,
            lightmaps: other.lightmaps,
            leaves: other.leaves,
            leaf_faces: other.leaf_faces,
            leaf_brushes: other.leaf_brushes,
            edges: other.edges,
            surf_edges: other.surf_edges,
            models: other.models,
            brushes: other.brushes,
            brush_sides: other.brush_sides,
            areas: other.areas,
            area_portals: other.area_portals,
            ..Default::default()
        }
    }
}

parseable! {
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct DirEntry {
        offset: u32,
        length: u32,
    }
}

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct LeafFace {
        pub face: u16,
    }
}

#[derive(Default, Clone, PartialEq)]
pub struct Entities {
    entities: String,
}

impl fmt::Debug for Entities {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        #[derive(Default, Debug, PartialEq)]
        struct Entities<'a> {
            entities: Vec<Entity<'a>>,
        }

        Entities {
            entities: self.iter().collect(),
        }
        .fmt(f)
    }
}

impl Entities {
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = Entity<'_>> {
        struct Iter<'a> {
            buf: &'a str,
        }

        impl<'a> Iterator for Iter<'a> {
            type Item = Entity<'a>;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                let start = self.buf.find('{')? + 1;
                let end = start + self.buf[start..].find('}')?;

                let out = &self.buf[start..end];

                self.buf = &self.buf[end + 1..];

                Some(Entity { buf: out })
            }
        }

        Iter {
            buf: &self.entities,
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct Entity<'a> {
    buf: &'a str,
}

impl fmt::Debug for Entity<'_> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::collections::HashMap;

        self.properties().collect::<HashMap<_, _>>().fmt(f)
    }
}

impl<'a> Entity<'a> {
    #[inline]
    pub fn properties(&self) -> impl Iterator<Item = (&'a str, &'a str)> {
        struct Iter<'a> {
            buf: &'a str,
        }

        impl<'a> Iterator for Iter<'a> {
            type Item = (&'a str, &'a str);

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                let start = self.buf.find('"')? + 1;
                let end = start + self.buf[start..].find('"')?;

                let key = &self.buf[start..end];

                let rest = &self.buf[end + 1..];

                let start = rest.find('"')? + 1;
                let end = start + rest[start..].find('"')?;

                let value = &rest[start..end];

                self.buf = &rest[end + 1..];

                Some((key, value))
            }
        }

        Iter { buf: &self.buf }
    }
}

bitflags! {
    /// "Surface flags", as defined in Quake 2. It's possible to reuse any
    /// Quake-specific flags for your own purposes if you prefer, but the
    /// names and descriptions here correspond to how Quake 2 sees these flags.
    ///
    /// Specifically, `SKY`, `NODRAW`, `NOLIGHTMAP`, `POINTLIGHT` and `NODLIGHT`
    /// are special and should be implemented in a renderer, and `ALPHASHADOW`
    /// and `LIGHTFILTER` should be implemented in anything that wants to generate
    /// lightmaps like `q3map -light`.
    #[derive(Default)]
    pub struct SurfaceFlags: u32 {
        /// Never give falling damage
        const NODAMAGE    = 0b0000_0000_0000_0000_0001;
        /// Affects game physics
        const SLICK       = 0b0000_0000_0000_0000_0010;
        /// Lighting from environment map
        const SKY         = 0b0000_0000_0000_0000_0100;
        /// Climbable ladder
        const WARP        = 0b0000_0000_0000_0000_1000;
        /// Don't make missile explosions
        const NOIMPACT    = 0b0000_0000_0000_0001_0000;
        /// Don't leave missile marks
        const NOMARKS     = 0b0000_0000_0000_0010_0000;
        /// Make flesh sounds and effects
        const FLESH       = 0b0000_0000_0000_0100_0000;
        /// Don't generate a drawsurface at all
        const NODRAW      = 0b0000_0000_0000_1000_0000;
        /// Make a primary bsp splitter
        const HINT        = 0b0000_0000_0001_0000_0000;
        /// Completely ignore, allowing non-closed brushes
        const SKIP        = 0b0000_0000_0010_0000_0000;
        /// Surface doesn't need a lightmap
        const NOLIGHTMAP  = 0b0000_0000_0100_0000_0000;
        /// Generate lighting info at vertices
        const POINTLIGHT  = 0b0000_0000_1000_0000_0000;
        /// Clanking footsteps
        const METALSTEPS  = 0b0000_0001_0000_0000_0000;
        /// No footstep sounds
        const NOSTEPS     = 0b0000_0010_0000_0000_0000;
        /// Don't collide against curves with this set
        const NONSOLID    = 0b0000_0100_0000_0000_0000;
        /// Act as a light filter during q3map -light
        const LIGHTFILTER = 0b0000_1000_0000_0000_0000;
        /// Do per-pixel light shadow casting in q3map
        const ALPHASHADOW = 0b0001_0000_0000_0000_0000;
        /// Never add dynamic lights
        const NODLIGHT    = 0b0010_0000_0000_0000_0000;
    }
}

impl SurfaceFlags {
    #[inline]
    pub fn should_draw(&self) -> bool {
        !self.intersects(Self::HINT | Self::SKIP | Self::NODRAW | Self::LIGHTFILTER)
    }
}

bitflags! {
    #[derive(Default)]
    pub struct ContentFlags: u32 {
        // An eye is never valid in a solid
        const SOLID          = 0b0000_0000_0000_0000_0000_0000_0000_0001;
        const WINDOW         = 0b0000_0000_0000_0000_0000_0000_0000_0010;
        const AUX            = 0b0000_0000_0000_0000_0000_0000_0000_0100;
        const LAVA           = 0b0000_0000_0000_0000_0000_0000_0000_1000;
        const SLIME          = 0b0000_0000_0000_0000_0000_0000_0001_0000;
        const WATER          = 0b0000_0000_0000_0000_0000_0000_0010_0000;
        const MIST           = 0b0000_0000_0000_0000_0000_0000_0100_0000;

        const AREAPORTAL     = 0b0000_0000_0000_0000_1000_0000_0000_0000;

        const PLAYERCLIP     = 0b0000_0000_0000_0001_0000_0000_0000_0000;
        const MONSTERCLIP    = 0b0000_0000_0000_0010_0000_0000_0000_0000;

        // Bot-specific contents types
        const CURRENT_0      = 0b0000_0000_0000_0100_0000_0000_0000_0000;
        const CURRENT_90     = 0b0000_0000_0000_1000_0000_0000_0000_0000;
        const CURRENT_180    = 0b0000_0000_0001_0000_0000_0000_0000_0000;
        const CURRENT_270    = 0b0000_0000_0010_0000_0000_0000_0000_0000;
        const CURRENT_UP     = 0b0000_0000_0100_0000_0000_0000_0000_0000;
        const CURRENT_DOWN   = 0b0000_0000_1000_0000_0000_0000_0000_0000;

        // Removed before bsping an entity
        const ORIGIN         = 0b0000_0001_0000_0000_0000_0000_0000_0000;

        // Should never be on a brush, only in game
        const MONSTER        = 0b0000_0010_0000_0000_0000_0000_0000_0000;
        const DEADMONSTER    = 0b0000_0100_0000_0000_0000_0000_0000_0000;
        // Brushes not used for the bsp
        const DETAIL         = 0b0000_1000_0000_0000_0000_0000_0000_0000;
        // Don't consume surface fragments inside
        const TRANSLUCENT    = 0b0001_0000_0000_0000_0000_0000_0000_0000;
        const LADDER        = 0b0010_0000_0000_0000_0000_0000_0000_0000;
    }
}

impl ElementSize for ContentFlags {
    const SIZE: usize = u32::SIZE;
}

impl SimpleParse for ContentFlags {
    #[inline]
    fn parse<R: io::Read>(r: &mut R) -> io::Result<Self> {
        u32::parse(r).and_then(|v| {
            Ok(ContentFlags::from_bits(v).unwrap_or_default()) // ContentFlags::from_bits(v).ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))
        })
    }
}

impl ElementSize for SurfaceFlags {
    const SIZE: usize = u32::SIZE;
}

impl SimpleParse for SurfaceFlags {
    #[inline]
    fn parse<R: io::Read>(r: &mut R) -> io::Result<Self> {
        u32::parse(r).and_then(|v| {
            Ok(SurfaceFlags::from_bits(v).unwrap_or_default()) //SurfaceFlags::from_bits(v).ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))
        })
    }
}

const TEXTURE_NAME_SIZE: usize = 32;

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct TextureOffsets {
        pub axis_u: QVec,
        pub offset_u: f32,

        pub axis_v: QVec,
        pub offset_v: f32,
    }
}

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct Q2Texture {
        pub offsets: TextureOffsets,

        pub flags: SurfaceFlags,
        pub value: u32,

        pub name: ArrayString<[u8; TEXTURE_NAME_SIZE]>,
        pub next: i32,
    }
}

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct GoldsrcTexture {
        pub offsets: TextureOffsets,

        pub texture_index: u32,
        pub flags: SurfaceFlags,
    }
}

#[derive(Copy, Clone)]
pub enum NoFormat {}

impl image::Pixel for NoFormat {
    type Subpixel = u8;

    const CHANNEL_COUNT: u8 = 0;
    const COLOR_MODEL: &'static str = "";
    const COLOR_TYPE: image::ColorType = image::ColorType::L8;

    fn channels(&self) -> &[Self::Subpixel] {
        unreachable!()
    }

    fn channels_mut(&mut self) -> &mut [Self::Subpixel] {
        unreachable!()
    }

    fn channels4(
        &self,
    ) -> (
        Self::Subpixel,
        Self::Subpixel,
        Self::Subpixel,
        Self::Subpixel,
    ) {
        unreachable!()
    }

    fn from_channels(
        _a: Self::Subpixel,
        _b: Self::Subpixel,
        _c: Self::Subpixel,
        _d: Self::Subpixel,
    ) -> Self {
        unreachable!()
    }

    fn from_slice(_slice: &[Self::Subpixel]) -> &Self {
        unreachable!()
    }

    fn from_slice_mut(_slice: &mut [Self::Subpixel]) -> &mut Self {
        unreachable!()
    }

    fn to_rgb(&self) -> image::Rgb<Self::Subpixel> {
        unreachable!()
    }

    fn to_rgba(&self) -> image::Rgba<Self::Subpixel> {
        unreachable!()
    }

    fn to_luma(&self) -> image::Luma<Self::Subpixel> {
        unreachable!()
    }

    fn to_luma_alpha(&self) -> image::LumaA<Self::Subpixel> {
        unreachable!()
    }

    fn to_bgr(&self) -> image::Bgr<Self::Subpixel> {
        unreachable!()
    }

    fn to_bgra(&self) -> image::Bgra<Self::Subpixel> {
        unreachable!()
    }

    fn map<F>(&self, _f: F) -> Self
    where
        F: FnMut(Self::Subpixel) -> Self::Subpixel,
    {
        unreachable!()
    }

    fn apply<F>(&mut self, _f: F)
    where
        F: FnMut(Self::Subpixel) -> Self::Subpixel,
    {
        unreachable!()
    }

    fn map_with_alpha<F, G>(&self, _f: F, _g: G) -> Self
    where
        F: FnMut(Self::Subpixel) -> Self::Subpixel,
        G: FnMut(Self::Subpixel) -> Self::Subpixel,
    {
        unreachable!()
    }

    fn apply_with_alpha<F, G>(&mut self, _f: F, _g: G)
    where
        F: FnMut(Self::Subpixel) -> Self::Subpixel,
        G: FnMut(Self::Subpixel) -> Self::Subpixel,
    {
        unreachable!()
    }

    fn map2<F>(&self, _other: &Self, _f: F) -> Self
    where
        F: FnMut(Self::Subpixel, Self::Subpixel) -> Self::Subpixel,
    {
        unreachable!()
    }

    fn apply2<F>(&mut self, _other: &Self, _f: F)
    where
        F: FnMut(Self::Subpixel, Self::Subpixel) -> Self::Subpixel,
    {
        unreachable!()
    }

    fn invert(&mut self) {
        unreachable!()
    }

    fn blend(&mut self, _other: &Self) {
        unreachable!()
    }
}

impl BspTexture<Quake2> for Q2Texture {
    type Format = NoFormat;

    fn name<'a>(&'a self, bsp: &'a Bsp<Quake2>) -> &'a str {
        &self.name
    }

    fn data(&self, _bsp: &Bsp<Quake2>) -> Option<image::ImageBuffer<Self::Format, &[u8]>> {
        None
    }

    fn offsets(&self) -> &TextureOffsets {
        &self.offsets
    }
}

impl BspTexture<Goldsrc> for GoldsrcTexture {
    type Format = NoFormat;

    fn name<'a>(&'a self, bsp: &'a Bsp<Goldsrc>) -> &'a str {
        &bsp.miptexes[self.texture_index as usize].name
    }

    fn data(&self, _bsp: &Bsp<Goldsrc>) -> Option<image::ImageBuffer<Self::Format, &[u8]>> {
        unimplemented!()
    }

    fn offsets(&self) -> &TextureOffsets {
        &self.offsets
    }
}
parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct Plane {
        pub normal: QVec,
        pub dist: f32,
        pub type_: u32,
    }
}

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct Node {
        pub plane: u32,
        pub children: [i32; 2],
        pub mins: [i16; 3],
        pub maxs: [i16; 3],
        pub face: u16,
        pub num_faces: u16,
    }
}

pub type Cluster = u16;

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct Q2Leaf {
        pub contents: ContentFlags,
        pub cluster: i16,
        pub area: u16,
        pub mins: [i16; 3],
        pub maxs: [i16; 3],
        pub leaf_face: u16,
        pub num_leaf_faces: u16,
        pub leaf_brush: u16,
        pub num_leaf_brushes: u16,
    }
}

impl BspLeaf for Q2Leaf {
    fn cluster(&self) -> i16 {
        self.cluster
    }
}

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct GoldsrcLeaf {
        pub contents: ContentFlags,
        // TODO: How to handle this?
        pub visdata_offset: i32,
        pub mins: [i16; 3],
        pub maxs: [i16; 3],
        pub leaf_face: u16,
        pub num_leaf_faces: u16,
        pub ambient_sound_levels: [u8; 4],
    }
}

impl BspLeaf for GoldsrcLeaf {
    fn cluster(&self) -> i16 {
        unimplemented!()
    }
}

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct LeafBrush {
        pub brush: u16,
    }
}

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct Q2Model {
        pub mins: QVec,
        pub maxs: QVec,
        pub origin: QVec,
        pub headnode: u32,
        pub face: u32,
        pub num_faces: u32,
    }
}

impl BspModel for Q2Model {
    type Hulls = std::iter::Once<u32>;

    fn hulls(&self) -> Self::Hulls {
        std::iter::once(self.headnode)
    }
}

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct GoldsrcModel {
        pub mins: QVec,
        pub maxs: QVec,
        pub origin: QVec,
        pub headnodes: [u32; 4],
        pub visleaf: i32,
        pub face: u32,
        pub num_faces: u32,
    }
}

impl BspModel for GoldsrcModel {
    type Hulls = arrayvec::IntoIter<[u32; 4]>;

    fn hulls(&self) -> Self::Hulls {
        arrayvec::ArrayVec::from(self.headnodes).into_iter()
    }
}

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct Brush {
        pub brush_side: u32,
        pub num_brush_sides: u32,
        pub contents: ContentFlags,
    }
}

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct BrushSide {
        pub plane: u16,
        pub texture: i16,
    }
}

pub const MAX_LIGHTMAPS_PER_FACE: usize = 4;

parseable! {
    #[derive(Debug, Clone, PartialEq)]
    pub struct Face {
        pub plane: u16,
        pub side: i16,
        pub surf_edge: u32,
        pub num_surf_edges: u16,
        pub texture: i16,
        pub styles: [i8; MAX_LIGHTMAPS_PER_FACE],
        pub lightmap: i32,
    }
}

pub struct LightmapRef<'a> {
    pub style: u8,
    pub mins: (f32, f32),
    pub maxs: (f32, f32),
    pub width: u32,
    pub data: &'a [u8],
}

impl<'a> LightmapRef<'a> {
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.data.len() as u32 / 3 / self.width
    }

    #[inline]
    pub fn size(&self) -> (u32, u32) {
        (self.width(), self.height())
    }

    #[inline]
    pub fn as_image(&self) -> image::ImageBuffer<image::Rgb<u8>, &'a [u8]> {
        image::ImageBuffer::from_raw(self.width(), self.height(), self.data).unwrap()
    }
}

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct Lightvol {
        ambient: [u8; 3],
        directional: [u8; 3],
        dir: [u8; 2],
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct VisDataOffsets {
    pub pvs: u32,
    pub phs: u32,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Q2VisData {
    pub cluster_offsets: Vec<VisDataOffsets>,
    pub vecs: BitVec<u8>,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct GoldsrcVisData {
    pub vecs: BitVec<u8>,
}

impl ReadEntry for Q2VisData {
    fn read<R: io::Read + io::Seek>(mut reader: R, entry: &DirEntry) -> io::Result<Self> {
        if (entry.length as usize) < std::mem::size_of::<u32>() * 2 {
            return Ok(Q2VisData::default());
        }

        reader.seek(SeekFrom::Start(entry.offset as u64))?;

        let num_clusters = u32::parse(&mut reader)?;
        let mut clusters = Vec::with_capacity(num_clusters.try_into().map_err(|e| error(e))?);

        let mut vecs = Vec::with_capacity((num_clusters as usize * 2) / 8);

        for _ in 0..num_clusters {
            let pvs = u32::parse(&mut reader)?;
            let phs = u32::parse(&mut reader)?;

            let current_pos = reader.seek(SeekFrom::Current(0))?;

            reader.seek(SeekFrom::Start(entry.offset as u64 + pvs as u64))?;
            let pvs_start = vecs.len() as _;

            let mut cluster_bytes = 1 + (num_clusters as usize - 1) / 8;
            let mut bytes = reader.by_ref().bytes();
            while cluster_bytes > 0 {
                let byte = bytes.next().ok_or_else(|| {
                    error(format!(
                        "Can't get visdata byte (remaining: {})",
                        cluster_bytes
                    ))
                })??;

                if byte == 0 {
                    let count = bytes
                        .next()
                        .ok_or_else(|| error("RLE visdata has 0 with no count"))??
                        as usize;

                    cluster_bytes = cluster_bytes.saturating_sub(count);

                    vecs.extend(iter::repeat(0).take(count));
                } else {
                    cluster_bytes -= 1;

                    vecs.push(byte);
                }
            }

            reader.seek(SeekFrom::Start(entry.offset as u64 + phs as u64))?;
            let phs_start = vecs.len() as _;

            let mut cluster_bytes = 1 + (num_clusters as usize - 1) / 8;
            let mut bytes = reader.by_ref().bytes();
            while cluster_bytes > 0 {
                let byte = bytes.next().ok_or_else(|| {
                    error(format!(
                        "Can't get visdata byte (remaining: {})",
                        cluster_bytes
                    ))
                })??;

                if byte == 0 {
                    let count = bytes
                        .next()
                        .ok_or_else(|| error("RLE visdata has 0 with no count"))??
                        as usize;

                    cluster_bytes = cluster_bytes.saturating_sub(count);

                    vecs.extend(iter::repeat(0).take(count));
                } else {
                    cluster_bytes -= 1;

                    vecs.push(byte);
                }
            }

            clusters.push(VisDataOffsets {
                pvs: pvs_start,
                phs: phs_start,
            });

            reader.seek(SeekFrom::Start(current_pos))?;
        }

        let vecs = BitVec::from_bits(vecs);

        Ok(Q2VisData {
            cluster_offsets: clusters,
            vecs,
        })
    }
}

impl ReadEntry for GoldsrcVisData {
    fn read<R: io::Read + io::Seek>(mut reader: R, entry: &DirEntry) -> io::Result<Self> {
        reader.seek(SeekFrom::Start(entry.offset as u64))?;

        let mut vecs = Vec::new();
        let mut bytes = reader.by_ref().bytes();

        while let Some(byte) = bytes.next().map(Result::ok).flatten() {
            if byte == 0 {
                let count = bytes
                    .next()
                    .ok_or_else(|| error("RLE visdata has 0 with no count"))??
                    as usize;

                vecs.extend(iter::repeat(0).take(count));
            } else {
                vecs.push(byte);
            }
        }

        let vecs = BitVec::from_bits(vecs);

        Ok(GoldsrcVisData { vecs })
    }
}

struct BspReader<R> {
    inner: R,
}

impl<R: Read + Seek> BspReader<R> {
    #[inline]
    fn read_entities(&mut self, dir_entry: &DirEntry) -> io::Result<Entities> {
        let mut entities = Vec::with_capacity(dir_entry.length as usize);
        self.inner.seek(SeekFrom::Start(dir_entry.offset as u64))?;
        self.inner
            .by_ref()
            .take(dir_entry.length as u64)
            .read_to_end(&mut entities)?;
        let entities = String::from_utf8(entities).map_err(|err| error(err))?;
        Ok(Entities { entities })
    }

    #[inline]
    fn read_entry<T, O>(&mut self, dir_entry: &DirEntry) -> io::Result<O>
    where
        T: SimpleParse + ElementSize,
        O: FromIterator<T>,
    {
        if dir_entry.length % T::SIZE as u32 != 0 {
            return Err(error(format!(
                "Directory entry length isn't a multiple of element size \
                (length: {}, element size: {})",
                dir_entry.length,
                T::SIZE
            )));
        }

        let num_entries = dir_entry.length as usize / T::SIZE;
        self.inner.seek(SeekFrom::Start(dir_entry.offset as u64))?;

        T::parse_many(&mut self.inner, num_entries)
    }

    #[inline]
    fn read_lightmaps(&mut self, dir_entry: &DirEntry) -> io::Result<Box<[u8]>> {
        self.inner.seek(SeekFrom::Start(dir_entry.offset as u64))?;

        let mut vec = Vec::with_capacity(dir_entry.length as usize);
        self.inner
            .by_ref()
            .take(dir_entry.length as u64)
            .read_to_end(&mut vec)?;

        if vec.len() != dir_entry.length as usize {
            return Err(ErrorKind::UnexpectedEof.into());
        }

        debug_assert_eq!(vec.len() % 3, 0);

        Ok(vec.into_boxed_slice())
    }
}

#[derive(Debug, PartialEq)]
pub struct Handle<'a, B, T> {
    pub bsp: &'a B,
    pub data: &'a T,
}

impl<'a, B, T> Handle<'a, B, T> {
    #[inline]
    pub fn new(bsp: &'a B, data: &'a T) -> Self {
        Handle { bsp, data }
    }
}

impl<B, T> Clone for Handle<'_, B, T> {
    #[inline]
    fn clone(&self) -> Self {
        Handle { ..*self }
    }
}

impl<B, T> Copy for Handle<'_, B, T> {}

impl<'a, B, T> Handle<'a, B, T> {
    #[inline]
    pub fn as_ref(&self) -> &'a T {
        self.data
    }
}

impl<B, T> Deref for Handle<'_, B, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

parseable! {
    #[derive(Default, Debug, Clone, Copy, PartialEq)]
    pub struct Edge {
        pub first: u16,
        pub second: u16,
    }
}

impl Edge {
    #[inline]
    pub fn rev(self) -> Edge {
        Edge {
            first: self.second,
            second: self.first,
        }
    }
}

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct SurfEdge {
        // Use `abs(edge_index)` for actual index, and `signum(edge_index)` for winding order
        pub edge: i32,
    }
}

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct Area {
        pub num_area_portals: u32,
        pub area_portal: u32,
    }
}

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct AreaPortal {
        pub num_portals: u32,
        pub other_area: u32,
    }
}

// TODO: Store all the allocated objects inline to improve cache usage
pub struct Bsp<Format: BspFormat + ?Sized = Quake2> {
    pub entities: Entities,
    pub miptexes: Format::Miptexes,
    pub textures: Box<[Format::Texture]>,
    pub leaf_faces: Box<[LeafFace]>,
    pub leaf_brushes: Box<[LeafBrush]>,
    pub edges: Box<[Edge]>,
    pub surf_edges: Box<[SurfEdge]>,
    pub brushes: Box<[Brush]>,
    pub brush_sides: Box<[BrushSide]>,
    pub vertices: Box<[QVec]>,
    pub faces: Box<[Face]>,
    pub lightmaps: Box<[u8]>,
    pub areas: Box<[Area]>,
    pub area_portals: Box<[AreaPortal]>,
    pub vis: Vis<Format>,
}

impl<B: BspFormat + ?Sized> Default for Bsp<B> {
    fn default() -> Self {
        Self {
            entities: Default::default(),
            miptexes: Default::default(),
            textures: Default::default(),
            leaf_faces: Default::default(),
            leaf_brushes: Default::default(),
            edges: Default::default(),
            surf_edges: Default::default(),
            brushes: Default::default(),
            brush_sides: Default::default(),
            vertices: Default::default(),
            faces: Default::default(),
            lightmaps: Default::default(),
            areas: Default::default(),
            area_portals: Default::default(),
            vis: Default::default(),
        }
    }
}

impl<B: BspFormat + ?Sized> Clone for Bsp<B> {
    fn clone(&self) -> Self {
        Self {
            entities: self.entities.clone(),
            miptexes: self.miptexes.clone(),
            textures: self.textures.clone(),
            leaf_faces: self.leaf_faces.clone(),
            leaf_brushes: self.leaf_brushes.clone(),
            edges: self.edges.clone(),
            surf_edges: self.surf_edges.clone(),
            brushes: self.brushes.clone(),
            brush_sides: self.brush_sides.clone(),
            vertices: self.vertices.clone(),
            faces: self.faces.clone(),
            lightmaps: self.lightmaps.clone(),
            areas: self.areas.clone(),
            area_portals: self.area_portals.clone(),
            vis: self.vis.clone(),
        }
    }
}

pub struct Vis<B: BspFormat + ?Sized = Quake2> {
    pub leaves: Box<[B::Leaf]>,
    pub nodes: Box<[Node]>,
    pub planes: Box<[Plane]>,
    pub models: Box<[B::Model]>,
    pub visdata: B::VisData,
}

impl<B: BspFormat + ?Sized> Clone for Vis<B> {
    fn clone(&self) -> Self {
        Self {
            leaves: self.leaves.clone(),
            nodes: self.nodes.clone(),
            planes: self.planes.clone(),
            models: self.models.clone(),
            visdata: self.visdata.clone(),
        }
    }
}

impl<B: BspFormat + ?Sized> Default for Vis<B> {
    fn default() -> Self {
        Self {
            leaves: Default::default(),
            nodes: Default::default(),
            planes: Default::default(),
            models: Default::default(),
            visdata: Default::default(),
        }
    }
}

impl<F: BspFormat> AsRef<[Node]> for Vis<F> {
    #[inline]
    fn as_ref(&self) -> &[Node] {
        &self.nodes
    }
}

impl<F: BspFormat> AsRef<[Plane]> for Vis<F> {
    #[inline]
    fn as_ref(&self) -> &[Plane] {
        &self.planes
    }
}

impl<F: BspFormat> Vis<F> {
    #[inline]
    pub fn node(&self, n: usize) -> Option<Handle<'_, Self, Node>> {
        self.nodes.get(n).map(|node| Handle {
            bsp: self,
            data: node,
        })
    }

    #[inline]
    pub fn root_node(&self) -> Option<Handle<'_, Self, Node>> {
        self.node(self.models.get(0)?.hulls().next().unwrap() as usize)
    }

    #[inline]
    pub fn leaf(&self, n: usize) -> Option<Handle<'_, Self, F::Leaf>> {
        self.leaves.get(n).map(|leaf| Handle {
            bsp: self,
            data: leaf,
        })
    }

    #[inline]
    pub fn plane(&self, n: usize) -> Option<Handle<'_, Self, Plane>> {
        self.planes.get(n).map(|plane| Handle {
            bsp: self,
            data: plane,
        })
    }

    #[inline]
    pub fn model(&self, n: usize) -> Option<Handle<'_, Self, F::Model>> {
        self.models.get(n).map(|model| Handle {
            bsp: self,
            data: model,
        })
    }

    #[inline]
    pub fn models(&self) -> impl ExactSizeIterator<Item = Handle<'_, Self, F::Model>> + Clone {
        self.models.iter().map(move |m| Handle::new(self, m))
    }

    #[inline]
    pub fn leaf_at<C, I: Into<V3<C>>>(
        &self,
        root: Handle<'_, Self, Node>,
        point: I,
    ) -> Option<Handle<'_, Self, F::Leaf>>
    where
        C: CoordSystem,
    {
        self.leaf_index_at(root, point).and_then(|i| self.leaf(i))
    }

    fn cast_ray_between(
        &self,
        root: Handle<'_, Self, Node>,
        start: QVec,
        end: QVec,
    ) -> Option<QVec> {
        let plane = root.plane()?;
        let start_dot = if plane.type_ < 3 {
            start.0[plane.type_ as usize] - plane.dist
        } else {
            start.dot(&plane.normal) - plane.dist
        };
        let end_dot = if plane.type_ < 3 {
            end.0[plane.type_ as usize] - plane.dist
        } else {
            end.dot(&plane.normal) - plane.dist
        };

        let [front, back] = root.children;

        if start_dot > 0. && end_dot > 0. {
            let child: i32 = todo!();
            let child = if front < 0 {
                todo!()
            } else {
                self.node(child as usize)?
            };

            self.cast_ray_between(child, start, end)
        } else if start_dot < 0. && end_dot < 0. {
            let child: i32 = todo!();
            let child = if back < 0 {
                todo!()
            } else {
                self.node(child as usize)?
            };

            self.cast_ray_between(child, start, end)
        } else {
            todo!()
        }
    }

    pub fn cast_ray<C, P: Into<V3<C>>, N: Into<V3<C>>>(
        &self,
        root: Handle<'_, Self, Node>,
        point: P,
        norm: N,
    ) -> Option<QVec>
    where
        C: CoordSystem,
    {
        let point = C::into_qvec(point.into());
        let norm = C::into_qvec(norm.into());

        let plane = root.plane()?;
        let dot = if plane.type_ < 3 {
            point.0[plane.type_ as usize] - plane.dist
        } else {
            point.dot(&plane.normal) - plane.dist
        };
        let norm_dot = if plane.type_ < 3 {
            norm.0[plane.type_ as usize] - plane.dist
        } else {
            norm.dot(&plane.normal) - plane.dist
        };

        let [front, back] = root.children;

        if dot < 0. && norm_dot < 0. || dot > 0. && norm_dot > 0. {
            let child = if norm_dot > 0. { front } else { back };
            let child = if child < 0 {
                todo!()
            } else {
                self.node(child as usize)?
            };

            self.cast_ray(child, point, norm)
        } else {
            let child = if norm_dot < 0. { front } else { back };
            let child = if child < 0 {
                todo!()
            } else {
                self.node(child as usize)?
            };

            self.cast_ray_between(child, point, norm * dot)
        }
    }

    #[inline]
    pub fn leaf_index_at<C, I: Into<V3<C>>>(
        &self,
        root: Handle<'_, Self, Node>,
        point: I,
    ) -> Option<usize>
    where
        C: CoordSystem,
    {
        let point = C::into_qvec(point.into());
        let mut current = root;

        loop {
            let plane = current.plane()?;
            let norm: &QVec = &plane.normal;

            let dot = if plane.type_ < 3 {
                point.0[plane.type_ as usize] - plane.dist
            } else {
                point.dot(norm) - plane.dist
            };

            let [front, back] = current.children;

            let next = if dot < 0. { back } else { front };

            if next < 0 {
                return Some(-(next + 1) as usize);
            } else {
                current = self.node(next as usize)?;
            }
        }
    }

    #[inline]
    pub fn leaves(&self) -> impl ExactSizeIterator<Item = Handle<'_, Self, F::Leaf>> + Clone {
        self.leaves.iter().map(move |leaf| Handle {
            bsp: self,
            data: leaf,
        })
    }

    #[inline]
    pub fn cluster_at<C, I: Into<V3<C>>>(
        &self,
        root: Handle<'_, Self, Node>,
        point: I,
    ) -> Option<Cluster>
    where
        C: CoordSystem,
    {
        self.leaf_at(root, point)
            .and_then(|leaf| leaf.cluster().try_into().ok())
    }

    #[inline]
    pub fn leaves_in_cluster(
        &self,
        cluster: impl TryInto<i16>,
    ) -> impl Iterator<Item = Handle<'_, Self, F::Leaf>> + Clone + '_ {
        // We do this eagerly, so that the returned iterator can be trivially cloned
        cluster
            .try_into()
            .ok()
            .and_then(|cluster| {
                let any_leaf = self
                    .leaves
                    .binary_search_by_key(&cluster, |leaf| leaf.cluster())
                    .ok()?;

                let mut first_leaf = any_leaf;
                while first_leaf
                    .checked_sub(1)
                    .and_then(|i| self.leaves.get(i).map(|leaf| leaf.cluster() == cluster))
                    .unwrap_or(false)
                {
                    first_leaf -= 1;
                }

                Some((cluster, first_leaf))
            })
            .into_iter()
            // And then this is done lazily, to avoid allocation
            .flat_map(move |(cluster, first_leaf)| {
                self.leaves[first_leaf..]
                    .iter()
                    .take_while(move |leaf| leaf.cluster() == cluster)
                    .map(move |leaf| Handle {
                        bsp: self,
                        data: leaf,
                    })
            })
    }
}

impl Vis<Quake2> {
    #[inline]
    fn potential_set(
        &self,
        leaf_id: usize,
        offset_fn: impl FnOnce(&VisDataOffsets) -> u32,
    ) -> impl Iterator<Item = usize> + '_ {
        use itertools::Either;

        self.leaf(leaf_id)
            .and_then(|leaf| u32::try_from(leaf.cluster()).ok())
            .map(move |cluster| {
                let offset = offset_fn(&self.visdata.cluster_offsets[cluster as usize]);
                Either::Left(
                    self.leaves
                        .iter()
                        .enumerate()
                        .filter(move |(_, leaf)| {
                            if let Ok(other_cluster) = u32::try_from(leaf.cluster()) {
                                cluster == other_cluster
                                    || self.visdata.vecs[offset as u64 * 8 + other_cluster as u64]
                            } else {
                                false
                            }
                        })
                        .map(|(i, _)| i),
                )
            })
            .unwrap_or(Either::Right(0..self.leaves.len()))
    }

    #[inline]
    pub fn potentially_visible_set(&self, leaf_id: usize) -> impl Iterator<Item = usize> + '_ {
        self.potential_set(leaf_id, |o| o.pvs)
    }

    #[inline]
    pub fn potentially_hearable_set(&self, leaf_id: usize) -> impl Iterator<Item = usize> + '_ {
        self.potential_set(leaf_id, |o| o.phs)
    }

    #[inline]
    pub fn clusters(&self) -> impl ExactSizeIterator<Item = u16> + Clone {
        0..self.visdata.cluster_offsets.len() as u16
    }

    #[inline]
    pub fn visible_from(&self, a: u16, b: u16) -> bool {
        a == b
            || self.visdata.vecs[self.visdata.cluster_offsets[a as usize].pvs as u64 * 8 + b as u64]
    }

    /// We use `impl TryInto` so that `-1` is transparently converted to "no visible clusters",
    /// but if you know your cluster ID is valid then you can skip that check.
    #[inline]
    pub fn visible_clusters<'a>(
        &'a self,
        from: u16,
        range: impl std::ops::RangeBounds<u16> + Clone + 'a,
    ) -> impl Iterator<Item = u16> + Clone + 'a {
        let cluster_vis_start = self.visdata.cluster_offsets[usize::from(from)].pvs;

        self.clusters().filter(move |&other| {
            if !range.contains(&other) {
                false
            } else if other == from {
                true
            } else {
                self.visdata.vecs[cluster_vis_start as u64 * 8 + other as u64]
            }
        })
    }
}

impl<F: BspFormat> AsRef<[LeafFace]> for Bsp<F> {
    #[inline]
    fn as_ref(&self) -> &[LeafFace] {
        &self.leaf_faces
    }
}

impl<F: BspFormat> AsRef<[LeafBrush]> for Bsp<F> {
    #[inline]
    fn as_ref(&self) -> &[LeafBrush] {
        &self.leaf_brushes
    }
}

impl<F: BspFormat> AsRef<[Edge]> for Bsp<F> {
    #[inline]
    fn as_ref(&self) -> &[Edge] {
        &self.edges
    }
}

impl<F: BspFormat> AsRef<[SurfEdge]> for Bsp<F> {
    #[inline]
    fn as_ref(&self) -> &[SurfEdge] {
        &self.surf_edges
    }
}

impl<F: BspFormat> AsRef<[Brush]> for Bsp<F> {
    #[inline]
    fn as_ref(&self) -> &[Brush] {
        &self.brushes
    }
}

impl<F: BspFormat> AsRef<[BrushSide]> for Bsp<F> {
    #[inline]
    fn as_ref(&self) -> &[BrushSide] {
        &self.brush_sides
    }
}

impl<F: BspFormat> AsRef<[QVec]> for Bsp<F> {
    #[inline]
    fn as_ref(&self) -> &[QVec] {
        &self.vertices
    }
}

impl<F: BspFormat> AsRef<[Face]> for Bsp<F> {
    #[inline]
    fn as_ref(&self) -> &[Face] {
        &self.faces
    }
}

impl<F: BspFormat> AsRef<[Area]> for Bsp<F> {
    #[inline]
    fn as_ref(&self) -> &[Area] {
        &self.areas
    }
}

impl<F: BspFormat> AsRef<[AreaPortal]> for Bsp<F> {
    #[inline]
    fn as_ref(&self) -> &[AreaPortal] {
        &self.area_portals
    }
}

impl<F: BspFormat> AsRef<[Node]> for Bsp<F> {
    #[inline]
    fn as_ref(&self) -> &[Node] {
        self.vis.as_ref()
    }
}

impl<F: BspFormat> AsRef<[Plane]> for Bsp<F> {
    #[inline]
    fn as_ref(&self) -> &[Plane] {
        self.vis.as_ref()
    }
}

impl<F: BspFormat> Bsp<F> {
    #[inline]
    pub fn read<R: Read + Seek>(mut reader: R) -> io::Result<Self> {
        let _magic = F::Magic::parse(&mut reader)?;
        let version = u32::parse(&mut reader)?;

        if version != F::VERSION {
            return Err(error(format!(
                "Invalid version (expected {:?}, got {:?})",
                F::VERSION,
                version
            )));
        }

        let dir_entries: GenericDirectories = F::Directories::parse(&mut reader)?.into();

        let mut reader = BspReader { inner: reader };

        let entities = reader.read_entities(&dir_entries.entities)?;
        let planes = reader.read_entry(&dir_entries.planes)?;
        let vertices = reader.read_entry(&dir_entries.vertices)?;
        let miptexes = F::Miptexes::read(&mut reader.inner, &dir_entries.miptexes)?;
        let visdata = F::VisData::read(&mut reader.inner, &dir_entries.visdata)?;
        let nodes = reader.read_entry(&dir_entries.nodes)?;
        let textures = reader.read_entry(&dir_entries.textures)?;
        let faces = reader.read_entry(&dir_entries.faces)?;
        let lightmaps = reader.read_lightmaps(&dir_entries.lightmaps)?;
        let leaves = reader.read_entry(&dir_entries.leaves)?;

        let leaf_faces = reader.read_entry(&dir_entries.leaf_faces)?;
        let leaf_brushes = reader.read_entry(&dir_entries.leaf_brushes)?;
        let edges = reader.read_entry(&dir_entries.edges)?;
        let surf_edges = reader.read_entry(&dir_entries.surf_edges)?;
        let models = reader.read_entry(&dir_entries.models)?;
        let brushes = reader.read_entry(&dir_entries.brushes)?;
        let brush_sides = reader.read_entry(&dir_entries.brush_sides)?;
        let areas = reader.read_entry(&dir_entries.areas)?;
        let area_portals = reader.read_entry(&dir_entries.area_portals)?;

        Ok({
            Bsp {
                entities,
                miptexes,
                textures,
                leaf_faces,
                leaf_brushes,
                edges,
                surf_edges,
                brushes,
                brush_sides,
                vertices,
                faces,
                lightmaps,
                areas,
                area_portals,
                vis: Vis {
                    models,
                    planes,
                    nodes,
                    leaves,
                    visdata,
                },
            }
        })
    }

    #[inline]
    pub fn node(&self, n: usize) -> Option<Handle<'_, Self, Node>> {
        self.vis.nodes.get(n).map(|node| Handle {
            bsp: self,
            data: node,
        })
    }

    #[inline]
    pub fn root_node(&self) -> Option<Handle<'_, Self, Node>> {
        self.node(self.model(0)?.hulls().next().unwrap() as usize)
    }

    #[inline]
    pub fn leaf(&self, n: usize) -> Option<Handle<'_, Self, F::Leaf>> {
        self.vis.leaves.get(n).map(|leaf| Handle {
            bsp: self,
            data: leaf,
        })
    }

    #[inline]
    pub fn plane(&self, n: usize) -> Option<Handle<'_, Self, Plane>> {
        self.vis.planes.get(n).map(|plane| Handle {
            bsp: self,
            data: plane,
        })
    }

    #[inline]
    pub fn face(&self, n: usize) -> Option<Handle<'_, Self, Face>> {
        self.faces.get(n).map(|face| Handle {
            bsp: self,
            data: face,
        })
    }

    #[inline]
    pub fn faces(&self) -> impl Iterator<Item = Handle<'_, Self, Face>> + '_ {
        self.faces.iter().map(move |face| Handle {
            bsp: self,
            data: face,
        })
    }

    #[inline]
    pub fn texture(&self, n: usize) -> Option<Handle<'_, Self, F::Texture>> {
        self.textures.get(n).map(move |texture| Handle {
            bsp: self,
            data: texture,
        })
    }

    #[inline]
    pub fn textures(&self) -> impl ExactSizeIterator<Item = Handle<'_, Self, F::Texture>> + Clone {
        self.textures.iter().map(move |m| Handle::new(self, m))
    }

    #[inline]
    pub fn model(&self, i: usize) -> Option<Handle<'_, Self, F::Model>> {
        self.vis.models.get(i).map(|model| Handle {
            bsp: self,
            data: model,
        })
    }

    #[inline]
    pub fn models(&self) -> impl ExactSizeIterator<Item = Handle<'_, Self, F::Model>> + Clone {
        self.vis.models().map(move |m| Handle::new(self, m.data))
    }

    #[inline]
    pub fn leaves(&self) -> impl ExactSizeIterator<Item = Handle<'_, Self, F::Leaf>> + Clone {
        self.vis.leaves.iter().map(move |leaf| Handle {
            bsp: self,
            data: leaf,
        })
    }
}

impl Bsp<Quake2> {
    #[inline]
    pub fn clusters(&self) -> impl ExactSizeIterator<Item = u16> + Clone {
        0..self.vis.visdata.cluster_offsets.len() as u16
    }
}

impl<'a> Handle<'a, Bsp, LeafFace> {
    #[inline]
    pub fn face(self) -> Handle<'a, Bsp, Face> {
        self.bsp.face(self.face as usize).unwrap()
    }
}

pub struct BoundingSphere {
    pub center: QVec,
    pub radius_squared: f32,
}

impl Q2Model {
    #[inline]
    pub fn bounding_sphere(&self) -> BoundingSphere {
        let center = [
            (self.mins.0[0] + self.maxs.0[0]) / 2.,
            (self.mins.0[1] + self.maxs.0[1]) / 2.,
            (self.mins.0[2] + self.maxs.0[2]) / 2.,
        ];

        let (diffx, diffy, diffz) = (
            self.maxs.0[2] - self.mins.0[0],
            self.maxs.0[0] - self.mins.0[1],
            self.maxs.0[1] - self.mins.0[2],
        );
        let radius_squared = diffx * diffx + diffy * diffy + diffz * diffz;

        BoundingSphere {
            center: center.into(),
            radius_squared,
        }
    }
}

impl<'a> Handle<'a, Bsp, Q2Model> {
    // #[inline]
    // pub fn leaf_indices(self) -> Option<impl Iterator<Item = u32> + Clone + 'a> {
    //     use itertools::Either;

    //     let mut stack = vec![Either::Left((self.headnode, self.bsp.node(self.headnode as usize)?))];

    //     Some(iter::from_fn(move || loop {
    //         let next = stack.pop()?;
    //         let node = match next {
    //             Either::Left(node) => node,
    //             Either::Right(leaf) => break Some(leaf),
    //         };
    //         let [left, right] = node.children;
    //         let left = if left < 0 {
    //             Either::Right(self.bsp.leaf(-(left + 1) as usize)?)
    //         } else {
    //             Either::Left(self.bsp.node(left as usize)?)
    //         };
    //         let right = if right < 0 {
    //             Either::Right(self.bsp.leaf(-(right + 1) as usize)?)
    //         } else {
    //             Either::Left(self.bsp.node(right as usize)?)
    //         };

    //         stack.push(left);
    //         stack.push(right);
    //     }))
    // }

    #[inline]
    pub fn leaves(self) -> Option<impl Iterator<Item = Handle<'a, Bsp, Q2Leaf>> + Clone + 'a> {
        use itertools::Either;

        let mut stack = vec![Either::Left(self.bsp.node(self.headnode as usize)?)];

        Some(iter::from_fn(move || loop {
            let next = stack.pop()?;
            let node = match next {
                Either::Left(node) => node,
                Either::Right(leaf) => break Some(leaf),
            };
            let [left, right] = node.children;
            let left = if left < 0 {
                Either::Right(self.bsp.leaf(-(left + 1) as usize)?)
            } else {
                Either::Left(self.bsp.node(left as usize)?)
            };
            let right = if right < 0 {
                Either::Right(self.bsp.leaf(-(right + 1) as usize)?)
            } else {
                Either::Left(self.bsp.node(right as usize)?)
            };

            stack.push(left);
            stack.push(right);
        }))
    }

    #[inline]
    pub fn faces(self) -> impl Iterator<Item = Handle<'a, Bsp, Face>> {
        let start = self.face as usize;
        let end = start + self.num_faces as usize;

        self.bsp.faces[start..end]
            .iter()
            .map(move |face| Handle::new(self.bsp, face))
    }
}

impl<'a> Handle<'a, Vis, Q2Model> {
    #[inline]
    pub fn cluster_at<C, I: Into<V3<C>>>(self, point: I) -> Option<u16>
    where
        C: CoordSystem,
    {
        let point = C::into_qvec(point.into());

        if point < self.mins || point > self.maxs {
            None
        } else {
            self.bsp
                .cluster_at(self.bsp.node(self.headnode as usize)?, point)
        }
    }
}

impl<'a> Handle<'a, Bsp, Q2Texture> {
    #[inline]
    pub fn next_frame(self) -> Option<Self> {
        u32::try_from(self.next)
            .ok()
            .and_then(|next| self.bsp.texture(next as usize))
    }

    #[inline]
    pub fn frames(self) -> impl Iterator<Item = Handle<'a, Bsp, Q2Texture>> {
        let mut texture = Some(self);
        let this = self;

        iter::from_fn(move || {
            let out = texture?;

            texture = match out.next_frame() {
                None => None,
                Some(tex) if tex.data == this.data => None,
                Some(other) => Some(other),
            };

            Some(out)
        })
    }
}

impl<'a> Handle<'a, Bsp, Face> {
    #[inline]
    pub fn texture(self) -> Option<Handle<'a, Bsp, Q2Texture>> {
        self.bsp.texture(self.texture as _)
    }

    #[inline]
    pub fn textures(self) -> impl Iterator<Item = Handle<'a, Bsp, Q2Texture>> {
        self.texture().into_iter().flat_map(|tex| tex.frames())
    }

    #[inline]
    pub fn texture_uvs(self) -> Option<impl Iterator<Item = (f32, f32)> + 'a> {
        let texture = self.texture()?;

        Some(self.vertices().map(move |vert| {
            (
                vert.dot(&texture.offsets.axis_u) + texture.offsets.offset_u,
                vert.dot(&texture.offsets.axis_v) + texture.offsets.offset_v,
            )
        }))
    }

    #[inline]
    pub fn lightmap_dimensions(self) -> Option<((f32, f32), (f32, f32), u32, u32)> {
        use std::f32;

        let mut min_u: Option<f32> = None;
        let mut min_v: Option<f32> = None;
        let mut max_u: Option<f32> = None;
        let mut max_v: Option<f32> = None;

        for (current_u, current_v) in self.texture_uvs()? {
            min_u = Some(min_u.map(|v| v.min(current_u)).unwrap_or(current_u));
            min_v = Some(min_v.map(|v| v.min(current_v)).unwrap_or(current_v));
            max_u = Some(max_u.map(|v| v.max(current_u)).unwrap_or(current_u));
            max_v = Some(max_v.map(|v| v.max(current_v)).unwrap_or(current_v));
        }

        let light_min_u = (min_u? / 16.).floor();
        let light_min_v = (min_v? / 16.).floor();
        let light_max_u = (max_u? / 16.).ceil();
        let light_max_v = (max_v? / 16.).ceil();

        let width = (light_max_u - light_min_u) + 1.;
        let height = (light_max_v - light_min_v) + 1.;

        Some((
            (light_min_u, light_min_v),
            (light_max_u, light_max_v),
            width as u32,
            height as u32,
        ))
    }

    #[inline]
    pub fn lightmaps(self) -> Option<impl ExactSizeIterator<Item = LightmapRef<'a>>> {
        if self.texture()?.flags.intersects(
            SurfaceFlags::WARP
                | SurfaceFlags::NOLIGHTMAP
                | SurfaceFlags::SKY
                | SurfaceFlags::NODRAW,
        ) {
            return None;
        }

        let start = usize::try_from(self.lightmap).ok()?;
        let (mins, maxs, w, h) = self.lightmap_dimensions()?;
        let num_styles = self.styles.iter().take_while(|&&s| s >= 0).count();
        let lightmap_bytes = (w * h) as usize * 3;

        Some((0..num_styles).map(move |i| LightmapRef {
            style: self.styles[i as usize] as u8,
            mins,
            maxs,
            width: w,
            data: &self.bsp.lightmaps[start + lightmap_bytes * i..start + lightmap_bytes * (i + 1)],
        }))
    }

    #[inline]
    pub fn edges(self) -> impl ExactSizeIterator<Item = Edge> + Clone + 'a {
        let start = self.surf_edge as usize;
        let end = (self.surf_edge + self.num_surf_edges as u32) as usize;

        self.bsp.surf_edges[start..end]
            .iter()
            .map(move |&SurfEdge { edge }| {
                if edge < 0 {
                    self.bsp.edges[(-edge) as usize]
                } else {
                    self.bsp.edges[edge as usize].rev()
                }
            })
    }

    #[inline]
    fn vert_indices(self) -> impl ExactSizeIterator<Item = u16> + 'a {
        let start = self.surf_edge as usize;
        let end = (self.surf_edge + self.num_surf_edges as u32) as usize;

        self.bsp.surf_edges[start..end]
            .iter()
            .map(move |&SurfEdge { edge }| {
                if edge < 0 {
                    self.bsp.edges[(-edge) as usize].first
                } else {
                    self.bsp.edges[edge as usize].second
                }
            })
    }

    #[inline]
    pub fn vertices(self) -> impl ExactSizeIterator<Item = &'a QVec> + 'a {
        self.vert_indices()
            .map(move |i| &self.bsp.vertices[i as usize])
    }

    #[inline]
    pub fn center(self) -> QVec {
        let tot: QVec = self.vertices().cloned().sum();
        tot / (self.vert_indices().len() as f32)
    }
}

impl<'a, T> Handle<'a, T, Node>
where
    T: AsRef<[Plane]>,
{
    #[inline]
    pub fn plane(self) -> Option<Handle<'a, T, Plane>> {
        self.bsp
            .as_ref()
            .get(self.plane as usize)
            .map(|p| Handle::new(self.bsp, p))
    }
}

impl<'a, T> Handle<'a, T, Node>
where
    T: AsRef<[Face]>,
{
    #[inline]
    pub fn face(self) -> Option<Handle<'a, T, Face>> {
        self.bsp
            .as_ref()
            .get(self.face as usize)
            .map(|f| Handle::new(self.bsp, f))
    }
}

impl<'a> Handle<'a, Bsp, Q2Leaf> {
    #[inline]
    pub fn leaf_faces(self) -> impl ExactSizeIterator<Item = Handle<'a, Bsp, LeafFace>> + Clone {
        let start = self.leaf_face as usize;
        let end = start + self.num_leaf_faces as usize;

        self.bsp.leaf_faces[start..end]
            .iter()
            .map(move |leaf_face| Handle {
                bsp: self.bsp,
                data: leaf_face,
            })
    }

    #[inline]
    pub fn faces(self) -> impl Iterator<Item = Handle<'a, Bsp, Face>> {
        self.leaf_faces()
            .filter_map(move |leaf_face| self.bsp.face(leaf_face.face as usize))
    }
}
