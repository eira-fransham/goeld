#![cfg_attr(feature = "bench", feature(test))]

// TODO: Support other similar BSP versions (especially GoldSrc BSP)

#[cfg(feature = "bench")]
extern crate test;

use arrayvec::ArrayString;
use bitflags::bitflags;
use bv::BitVec;
pub use goldsrc_format_common::{
    parseable, CoordSystem, ElementSize, Magic, QVec, SimpleParse, XEastYDownZSouth,
    XEastYNorthZUp, XEastYSouthZUp, V3,
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

pub trait BspFormat {
    const VERSION: u32;

    type Magic: SimpleParse;
    type Directories: ElementSize + SimpleParse + Into<Quake2Directories>;
}

pub struct Quake2;

const QUAKE2_MAGIC: [u8; 4] = [b'I', b'B', b'S', b'P'];

impl BspFormat for Quake2 {
    type Magic = Magic<QUAKE2_MAGIC>;

    const VERSION: u32 = 0x26;

    type Directories = Quake2Directories;
}

pub struct Goldsrc;

impl BspFormat for Goldsrc {
    const VERSION: u32 = 0x1e;

    type Magic = ();
    type Directories = GoldsrcDirectories;
}

parseable! {
    #[derive(Debug, Default, PartialEq)]
    pub struct GoldsrcDirectories {
        entities: DirEntry,
        planes: DirEntry,
        // TODO: In Goldsrc, `textures` is for _inline textures_, and `texinfo` is more
        //       similar to the `textures` lump in Quake 2.
        textures: DirEntry,
        vertices: DirEntry,
        visdata: DirEntry,
        nodes: DirEntry,
        faces: DirEntry,
        lightmaps: DirEntry,
        clipnodes: DirEntry,
        leaves: DirEntry,
        leaf_faces: DirEntry,
        leaf_brushes: DirEntry,
        edges: DirEntry,
        surf_edges: DirEntry,
        models: DirEntry,
        brushes: DirEntry,
        brush_sides: DirEntry,
        // Appears to be unused, even in Quake 2 itself
        areas: DirEntry,
        area_portals: DirEntry,
    }
}

impl From<GoldsrcDirectories> for Quake2Directories {
    #[inline]
    fn from(_other: GoldsrcDirectories) -> Self {
        todo!()
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

parseable! {
    #[derive(Clone, Debug, Default, PartialEq)]
    struct DirEntry {
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
        const LADDER      = 0b0000_0000_0000_0000_1000;
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
            ContentFlags::from_bits(v).ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))
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
            SurfaceFlags::from_bits(v).ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))
        })
    }
}

const TEXTURE_NAME_SIZE: usize = 32;

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct Texture {
        pub axis_u: QVec,
        pub offset_u: f32,

        pub axis_v: QVec,
        pub offset_v: f32,

        pub flags: SurfaceFlags,
        pub value: u32,

        pub name: ArrayString<[u8; TEXTURE_NAME_SIZE]>,
        pub next: i32,
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
    pub struct Leaf {
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

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct LeafBrush {
        pub brush: u16,
    }
}

parseable! {
    #[derive(Default, Debug, Clone, PartialEq)]
    pub struct Model {
        pub mins: QVec,
        pub maxs: QVec,
        pub origin: QVec,
        pub headnode: u32,
        pub face: u32,
        pub num_faces: u32,
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
pub struct VisData {
    pub cluster_offsets: Vec<VisDataOffsets>,
    pub vecs: BitVec<u8>,
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
    fn read_visdata(&mut self, entry: &DirEntry) -> io::Result<VisData> {
        if (entry.length as usize) < std::mem::size_of::<u32>() * 2 {
            return Ok(VisData::default());
        }

        self.inner.seek(SeekFrom::Start(entry.offset as u64))?;

        let num_clusters = u32::parse(&mut self.inner)?;
        let mut clusters = Vec::with_capacity(num_clusters.try_into().map_err(|e| error(e))?);

        let mut vecs = Vec::with_capacity((num_clusters as usize * 2) / 8);

        for _ in 0..num_clusters {
            let pvs = u32::parse(&mut self.inner)?;
            let phs = u32::parse(&mut self.inner)?;

            let current_pos = self.inner.seek(SeekFrom::Current(0))?;

            self.inner
                .seek(SeekFrom::Start(entry.offset as u64 + pvs as u64))?;
            let pvs_start = vecs.len() as _;

            let mut cluster_bytes = 1 + (num_clusters as usize - 1) / 8;
            let mut bytes = self.inner.by_ref().bytes();
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

            self.inner
                .seek(SeekFrom::Start(entry.offset as u64 + phs as u64))?;
            let phs_start = vecs.len() as _;

            let mut cluster_bytes = 1 + (num_clusters as usize - 1) / 8;
            let mut bytes = self.inner.by_ref().bytes();
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

            self.inner.seek(SeekFrom::Start(current_pos))?;
        }

        let vecs = BitVec::from_bits(vecs);

        Ok(VisData {
            cluster_offsets: clusters,
            vecs,
        })
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
#[derive(Default, Debug, Clone, PartialEq)]
pub struct Bsp {
    pub entities: Entities,
    pub textures: Box<[Texture]>,
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
    pub vis: Vis,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Vis {
    pub leaves: Box<[Leaf]>,
    pub nodes: Box<[Node]>,
    pub planes: Box<[Plane]>,
    pub models: Box<[Model]>,
    pub visdata: VisData,
}

impl AsRef<[Leaf]> for Vis {
    #[inline]
    fn as_ref(&self) -> &[Leaf] {
        &self.leaves
    }
}

impl AsRef<[Node]> for Vis {
    #[inline]
    fn as_ref(&self) -> &[Node] {
        &self.nodes
    }
}

impl AsRef<[Plane]> for Vis {
    #[inline]
    fn as_ref(&self) -> &[Plane] {
        &self.planes
    }
}

impl Vis {
    #[inline]
    pub fn node(&self, n: usize) -> Option<Handle<'_, Self, Node>> {
        self.nodes.get(n).map(|node| Handle {
            bsp: self,
            data: node,
        })
    }

    #[inline]
    pub fn root_node(&self) -> Option<Handle<'_, Self, Node>> {
        self.node(self.models.get(0)?.headnode as usize)
    }

    #[inline]
    pub fn leaf(&self, n: usize) -> Option<Handle<'_, Self, Leaf>> {
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
    pub fn model(&self, n: usize) -> Option<Handle<'_, Self, Model>> {
        self.models.get(n).map(|model| Handle {
            bsp: self,
            data: model,
        })
    }

    #[inline]
    pub fn models(&self) -> impl ExactSizeIterator<Item = Handle<'_, Self, Model>> + Clone {
        self.models.iter().map(move |m| Handle::new(self, m))
    }

    #[inline]
    fn potential_set(
        &self,
        leaf_id: usize,
        offset_fn: impl FnOnce(&VisDataOffsets) -> u32,
    ) -> impl Iterator<Item = usize> + '_ {
        use itertools::Either;

        self.leaf(leaf_id)
            .and_then(|leaf| u32::try_from(leaf.cluster).ok())
            .map(move |cluster| {
                let offset = offset_fn(&self.visdata.cluster_offsets[cluster as usize]);
                Either::Left(
                    self.leaves
                        .iter()
                        .enumerate()
                        .filter(move |(_, leaf)| {
                            if let Ok(other_cluster) = u32::try_from(leaf.cluster) {
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
    pub fn leaf_at<C, I: Into<V3<C>>>(
        &self,
        root: Handle<'_, Self, Node>,
        point: I,
    ) -> Option<Handle<'_, Self, Leaf>>
    where
        C: CoordSystem,
    {
        self.leaf_index_at(root, point).and_then(|i| self.leaf(i))
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

            let dot: f32 = if plane.type_ < 3 {
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
    pub fn leaves(&self) -> impl ExactSizeIterator<Item = Handle<'_, Self, Leaf>> + Clone {
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
            .and_then(|leaf| leaf.cluster.try_into().ok())
    }

    #[inline]
    pub fn leaves_in_cluster(
        &self,
        cluster: impl TryInto<i16>,
    ) -> impl Iterator<Item = Handle<'_, Self, Leaf>> + Clone + '_ {
        // We do this eagerly, so that the returned iterator can be trivially cloned
        cluster
            .try_into()
            .ok()
            .and_then(|cluster| {
                let any_leaf = self
                    .leaves
                    .binary_search_by_key(&cluster, |leaf| leaf.cluster)
                    .ok()?;

                let mut first_leaf = any_leaf;
                while first_leaf
                    .checked_sub(1)
                    .and_then(|i| self.leaves.get(i).map(|leaf| leaf.cluster == cluster))
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
                    .take_while(move |leaf| leaf.cluster == cluster)
                    .map(move |leaf| Handle {
                        bsp: self,
                        data: leaf,
                    })
            })
    }

    #[inline]
    pub fn clusters(&self) -> impl ExactSizeIterator<Item = u16> + Clone {
        0..self.visdata.cluster_offsets.len() as u16
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

impl AsRef<[Texture]> for Bsp {
    #[inline]
    fn as_ref(&self) -> &[Texture] {
        &self.textures
    }
}

impl AsRef<[LeafFace]> for Bsp {
    #[inline]
    fn as_ref(&self) -> &[LeafFace] {
        &self.leaf_faces
    }
}

impl AsRef<[LeafBrush]> for Bsp {
    #[inline]
    fn as_ref(&self) -> &[LeafBrush] {
        &self.leaf_brushes
    }
}

impl AsRef<[Edge]> for Bsp {
    #[inline]
    fn as_ref(&self) -> &[Edge] {
        &self.edges
    }
}

impl AsRef<[SurfEdge]> for Bsp {
    #[inline]
    fn as_ref(&self) -> &[SurfEdge] {
        &self.surf_edges
    }
}

impl AsRef<[Brush]> for Bsp {
    #[inline]
    fn as_ref(&self) -> &[Brush] {
        &self.brushes
    }
}

impl AsRef<[BrushSide]> for Bsp {
    #[inline]
    fn as_ref(&self) -> &[BrushSide] {
        &self.brush_sides
    }
}

impl AsRef<[QVec]> for Bsp {
    #[inline]
    fn as_ref(&self) -> &[QVec] {
        &self.vertices
    }
}

impl AsRef<[Face]> for Bsp {
    #[inline]
    fn as_ref(&self) -> &[Face] {
        &self.faces
    }
}

impl AsRef<[Area]> for Bsp {
    #[inline]
    fn as_ref(&self) -> &[Area] {
        &self.areas
    }
}

impl AsRef<[AreaPortal]> for Bsp {
    #[inline]
    fn as_ref(&self) -> &[AreaPortal] {
        &self.area_portals
    }
}

impl AsRef<[Model]> for Bsp {
    #[inline]
    fn as_ref(&self) -> &[Model] {
        &self.vis.models
    }
}

impl AsRef<[Leaf]> for Bsp {
    #[inline]
    fn as_ref(&self) -> &[Leaf] {
        self.vis.as_ref()
    }
}

impl AsRef<[Node]> for Bsp {
    #[inline]
    fn as_ref(&self) -> &[Node] {
        self.vis.as_ref()
    }
}

impl AsRef<[Plane]> for Bsp {
    #[inline]
    fn as_ref(&self) -> &[Plane] {
        self.vis.as_ref()
    }
}

impl Bsp {
    #[inline]
    pub fn read<R: Read + Seek>(reader: R) -> io::Result<Self> {
        Self::read_with_format::<R, Quake2>(reader)
    }

    #[inline]
    pub fn read_with_format<R: Read + Seek, T: BspFormat>(mut reader: R) -> io::Result<Self> {
        let _magic = T::Magic::parse(&mut reader)?;
        let version = u32::parse(&mut reader)?;

        if version != T::VERSION {
            return Err(error(format!(
                "Invalid version (expected {:?}, got {:?})",
                T::VERSION,
                version
            )));
        }

        let dir_entries: Quake2Directories = T::Directories::parse(&mut reader)?.into();

        let mut reader = BspReader { inner: reader };

        let entities = reader.read_entities(&dir_entries.entities)?;
        let planes = reader.read_entry(&dir_entries.planes)?;
        let vertices = reader.read_entry(&dir_entries.vertices)?;
        let visdata = reader.read_visdata(&dir_entries.visdata)?;
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
        self.node(self.model(0)?.headnode as usize)
    }

    #[inline]
    pub fn leaf(&self, n: usize) -> Option<Handle<'_, Self, Leaf>> {
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
    pub fn texture(&self, n: usize) -> Option<Handle<'_, Self, Texture>> {
        self.textures.get(n).map(move |texture| Handle {
            bsp: self,
            data: texture,
        })
    }

    #[inline]
    pub fn textures(&self) -> impl ExactSizeIterator<Item = Handle<'_, Self, Texture>> + Clone {
        self.textures.iter().map(move |m| Handle::new(self, m))
    }

    #[inline]
    pub fn model(&self, i: usize) -> Option<Handle<'_, Self, Model>> {
        self.vis.models.get(i).map(|model| Handle {
            bsp: self,
            data: model,
        })
    }

    #[inline]
    pub fn models(&self) -> impl ExactSizeIterator<Item = Handle<'_, Self, Model>> + Clone {
        self.vis.models().map(move |m| Handle::new(self, m.data))
    }

    #[inline]
    pub fn leaves(&self) -> impl ExactSizeIterator<Item = Handle<'_, Self, Leaf>> + Clone {
        self.vis.leaves.iter().map(move |leaf| Handle {
            bsp: self,
            data: leaf,
        })
    }

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

impl Model {
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

impl<'a> Handle<'a, Bsp, Model> {
    #[inline]
    pub fn leaves(self) -> Option<impl Iterator<Item = Handle<'a, Bsp, Leaf>> + Clone + 'a> {
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

impl<'a> Handle<'a, Vis, Model> {
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

impl<'a> Handle<'a, Bsp, Texture> {
    #[inline]
    pub fn next_frame(self) -> Option<Self> {
        u32::try_from(self.next)
            .ok()
            .and_then(|next| self.bsp.texture(next as usize))
    }

    #[inline]
    pub fn frames(self) -> impl Iterator<Item = Handle<'a, Bsp, Texture>> {
        let mut texture = Some(self);
        let this = self;

        iter::from_fn(move || {
            let out = texture?;

            texture = out.next_frame();

            Some(out)
        })
        .take_while(move |&t| t != this)
    }
}

impl<'a> Handle<'a, Bsp, Face> {
    #[inline]
    pub fn texture(self) -> Option<Handle<'a, Bsp, Texture>> {
        self.bsp.texture(self.texture as _)
    }

    #[inline]
    pub fn textures(self) -> impl Iterator<Item = Handle<'a, Bsp, Texture>> {
        self.texture().into_iter().flat_map(|tex| tex.frames())
    }

    #[inline]
    pub fn texture_uvs(self) -> Option<impl Iterator<Item = (f32, f32)> + 'a> {
        let texture = self.texture()?;

        Some(self.vertices().map(move |vert| {
            (
                vert.dot(&texture.axis_u) + texture.offset_u,
                vert.dot(&texture.axis_v) + texture.offset_v,
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
        if self
            .texture()?
            .flags
            .intersects(SurfaceFlags::NOLIGHTMAP | SurfaceFlags::SKY | SurfaceFlags::NODRAW)
        {
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

impl<'a> Handle<'a, Bsp, Leaf> {
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
