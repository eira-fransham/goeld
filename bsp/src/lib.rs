#![cfg_attr(feature = "bench", feature(test))]

// TODO: Support other similar BSP versions (especially GoldSrc BSP)

#[cfg(feature = "bench")]
extern crate test;

use arrayvec::ArrayString;
use bitflags::bitflags;
use bv::BitVec;
pub use goldsrc_format_common::{
    parseable, CoordSystem, ElementSize, Magic, QVec, SimpleParse, XEastYDownZSouth,
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
fn error(msg: impl ToString) -> io::Error {
    Error::new(ErrorKind::InvalidData, msg.to_string())
}

#[cfg(debug_assertions)]
fn error(msg: impl ToString) -> io::Error {
    panic!("{}", msg.to_string())
}

parseable! {
    #[derive(Debug, Default)]
    struct Directories {
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
    #[derive(Clone, Debug, Default)]
    struct DirEntry {
        offset: u32,
        length: u32,
    }
}

parseable! {
    #[derive(Default, Debug, Clone)]
    pub struct LeafFace {
        pub face: u16,
    }
}

#[derive(Default, Clone)]
pub struct Entities {
    entities: String,
}

impl fmt::Debug for Entities {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        #[derive(Default, Debug)]
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
    pub fn iter(&self) -> impl Iterator<Item = Entity<'_>> {
        struct Iter<'a> {
            buf: &'a str,
        }

        impl<'a> Iterator for Iter<'a> {
            type Item = Entity<'a>;

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

#[derive(Clone)]
pub struct Entity<'a> {
    buf: &'a str,
}

impl fmt::Debug for Entity<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::collections::HashMap;

        self.properties().collect::<HashMap<_, _>>().fmt(f)
    }
}

impl<'a> Entity<'a> {
    pub fn properties(&self) -> impl Iterator<Item = (&'a str, &'a str)> {
        struct Iter<'a> {
            buf: &'a str,
        }

        impl<'a> Iterator for Iter<'a> {
            type Item = (&'a str, &'a str);

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
    #[derive(Default)]
    pub struct SurfaceFlags: u32 {
        const NODAMAGE    = 0b0000_0000_0000_0000_0001; // Never give falling damage
        const SLICK       = 0b0000_0000_0000_0000_0010; // Affects game physics
        const SKY         = 0b0000_0000_0000_0000_0100; // Lighting from environment map
        const LADDER      = 0b0000_0000_0000_0000_1000; // Climbable ladder
        const NOIMPACT    = 0b0000_0000_0000_0001_0000; // Don't make missile explosions
        const NOMARKS     = 0b0000_0000_0000_0010_0000; // Don't leave missile marks
        const FLESH       = 0b0000_0000_0000_0100_0000; // Make flesh sounds and effects
        const NODRAW      = 0b0000_0000_0000_1000_0000; // Don't generate a drawsurface at all
        const HINT        = 0b0000_0000_0001_0000_0000; // Make a primary bsp splitter
        const SKIP        = 0b0000_0000_0010_0000_0000; // Completely ignore, allowing non-closed brushes
        const NOLIGHTMAP  = 0b0000_0000_0100_0000_0000; // Surface doesn't need a lightmap
        const POINTLIGHT  = 0b0000_0000_1000_0000_0000; // Generate lighting info at vertices
        const METALSTEPS  = 0b0000_0001_0000_0000_0000; // Clanking footsteps
        const NOSTEPS     = 0b0000_0010_0000_0000_0000; // No footstep sounds
        const NONSOLID    = 0b0000_0100_0000_0000_0000; // Don't collide against curves with this set
        const LIGHTFILTER = 0b0000_1000_0000_0000_0000; // Act as a light filter during q3map -light
        const ALPHASHADOW = 0b0001_0000_0000_0000_0000; // Do per-pixel light shadow casting in q3map
        const NODLIGHT    = 0b0010_0000_0000_0000_0000; // Never add dynamic lights
    }
}

impl SurfaceFlags {
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
    fn parse<R: io::Read>(r: &mut R) -> io::Result<Self> {
        u32::parse(r).and_then(|v| {
            SurfaceFlags::from_bits(v).ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))
        })
    }
}

const TEXTURE_NAME_SIZE: usize = 32;

parseable! {
    #[derive(Default, Debug, Clone)]
    pub struct Texture {
        pub axis_u: QVec,
        pub offset_u: f32,

        pub axis_v: QVec,
        pub offset_v: f32,

        pub flags: SurfaceFlags,
        pub value: i32,

        pub name: ArrayString<[u8; TEXTURE_NAME_SIZE]>,
        pub next: i32,
    }
}

parseable! {
    #[derive(Default, Debug, Clone)]
    pub struct Plane {
        pub normal: QVec,
        pub dist: f32,
        pub type_: u32,
    }
}

parseable! {
    #[derive(Default, Debug, Clone)]
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
    #[derive(Default, Debug, Clone)]
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
    #[derive(Default, Debug, Clone)]
    pub struct LeafBrush {
        pub brush: u32,
    }
}

parseable! {
    #[derive(Default, Debug, Clone)]
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
    #[derive(Default, Debug, Clone)]
    pub struct Brush {
        pub brush_side: u32,
        pub num_brush_sides: u32,
        pub contents: ContentFlags,
    }
}

parseable! {
    #[derive(Default, Debug, Clone)]
    pub struct BrushSide {
        pub plane: u32,
        pub texture: u32,
    }
}

const MAX_LIGHTMAPS_PER_FACE: usize = 4;

parseable! {
    #[derive(Debug, Clone)]
    pub struct Face {
        pub plane: u16,
        pub side: i16,
        pub surf_edge: u32,
        pub num_surf_edges: u16,
        pub texture: i16,
        pub styles: [u8; MAX_LIGHTMAPS_PER_FACE],
        pub lightmap: i32,
    }
}

pub struct LightmapRef<'a> {
    pub mins: (f32, f32),
    pub maxs: (f32, f32),
    pub width: u32,
    pub data: &'a [u8],
}

impl<'a> LightmapRef<'a> {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.data.len() as u32 / 3 / self.width
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width(), self.height())
    }
}

parseable! {
    #[derive(Default, Debug, Clone)]
    pub struct Lightvol {
        ambient: [u8; 3],
        directional: [u8; 3],
        dir: [u8; 2],
    }
}

#[derive(Default, Debug, Clone)]
pub struct VisDataOffsets {
    pub pvs: u32,
    pub phs: u32,
}

#[derive(Default, Debug, Clone)]
pub struct VisData {
    pub cluster_offsets: Vec<VisDataOffsets>,
    pub vecs: BitVec<u8>,
}

struct BspReader<R> {
    inner: R,
}

impl<R: Read + Seek> BspReader<R> {
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

    fn read_visdata(&mut self, entry: &DirEntry) -> io::Result<VisData> {
        if (entry.length as usize) < std::mem::size_of::<u32>() * 2 {
            return Ok(VisData::default());
        }

        self.inner.seek(SeekFrom::Start(entry.offset as u64))?;

        let num_clusters = u32::parse(&mut self.inner)?;
        let mut clusters = Vec::with_capacity(num_clusters.try_into().map_err(|e| error(e))?);

        let mut max_offset = 0;
        let visdata_start = std::mem::size_of::<u32>() as u32 * (1 + 2 * num_clusters);

        for _ in 0..num_clusters {
            let pvs = u32::parse(&mut self.inner)?;
            let phs = u32::parse(&mut self.inner)?;
            max_offset = max_offset.max(pvs).max(phs);
            clusters.push(VisDataOffsets {
                pvs: pvs - visdata_start,
                phs: phs - visdata_start,
            });
        }

        let cluster_bytes = 1 + ((num_clusters - 1) / 8);
        let visdata_size = max_offset + cluster_bytes;

        let mut vecs = Vec::with_capacity((num_clusters as usize * 2) / 8);

        self.inner
            .by_ref()
            .take(visdata_size as u64)
            .read_to_end(&mut vecs)?;

        if (vecs.len() as u32) < visdata_size {
            return Err(error("Unexpected EOF while reading VisData"));
        }

        if (vecs.len() as u32) > visdata_size {
            return Err(error("Extra data at end of file"));
        }

        let vecs = BitVec::from_bits(vecs);

        Ok(VisData {
            cluster_offsets: clusters,
            vecs,
        })
    }

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

        assert_eq!(vec.len() % 3, 0);

        Ok(vec.into_boxed_slice())
    }
}

#[derive(Debug)]
pub struct Handle<'a, T> {
    bsp: &'a Bsp,
    data: &'a T,
}

impl<T> Clone for Handle<'_, T> {
    fn clone(&self) -> Self {
        Handle { ..*self }
    }
}

impl<T> Copy for Handle<'_, T> {}

impl<'a, T> Handle<'a, T> {
    pub fn as_ref(&self) -> &'a T {
        self.data
    }
}

impl<T> Deref for Handle<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

parseable! {
    #[derive(Default, Debug, Clone, Copy)]
    pub struct Edge {
        pub first: u16,
        pub second: u16,
    }
}

impl Edge {
    pub fn rev(self) -> Edge {
        Edge {
            first: self.second,
            second: self.first,
        }
    }
}

parseable! {
    #[derive(Default, Debug, Clone)]
    pub struct SurfEdge {
        // Use `abs(edge_index)` for actual index, and `signum(edge_index)` for winding order
        pub edge: i32,
    }
}

parseable! {
    #[derive(Default, Debug, Clone)]
    pub struct Area {
        pub num_area_portals: u32,
        pub area_portal: u32,
    }
}

parseable! {
    #[derive(Default, Debug, Clone)]
    pub struct AreaPortal {
        pub num_portals: u32,
        pub other_area: u32,
    }
}

const MAGIC: [u8; 4] = [b'I', b'B', b'S', b'P'];

// TODO: Store all the allocated objects inline to improve cache usage
#[derive(Default, Debug, Clone)]
pub struct Bsp {
    pub header: Magic<MAGIC>,
    pub entities: Entities,
    pub textures: Box<[Texture]>,
    pub planes: Box<[Plane]>,
    pub nodes: Box<[Node]>,
    pub leaves: Box<[Leaf]>,
    pub leaf_faces: Box<[LeafFace]>,
    pub leaf_brushes: Box<[LeafBrush]>,
    pub edges: Box<[Edge]>,
    pub surf_edges: Box<[SurfEdge]>,
    pub models: Box<[Model]>,
    pub brushes: Box<[Brush]>,
    pub brush_sides: Box<[BrushSide]>,
    pub vertices: Box<[QVec]>,
    pub faces: Box<[Face]>,
    pub lightmaps: Box<[u8]>,
    pub areas: Box<[Area]>,
    pub area_portals: Box<[AreaPortal]>,
    pub visdata: VisData,
}

impl Bsp {
    pub fn read<R: Read + Seek>(mut reader: R) -> io::Result<Self> {
        // TODO: Use this to decide on the version to parse it as
        const EXPECTED_VERSION: u32 = 0x26;

        let header = SimpleParse::parse(&mut reader)?;
        let version = u32::parse(&mut reader)?;

        if version != EXPECTED_VERSION {
            return Err(error(format!(
                "Invalid version (expected {:?}, got {:?})",
                EXPECTED_VERSION, version
            )));
        }

        let dir_entries = Directories::parse(&mut reader)?;

        let mut reader = BspReader { inner: reader };

        let entities = reader.read_entities(&dir_entries.entities)?;
        let planes = reader.read_entry(&dir_entries.planes)?;
        let vertices = reader.read_entry(&dir_entries.vertices)?;
        let visdata = reader.read_visdata(&dir_entries.visdata)?;
        let nodes = reader.read_entry(&dir_entries.nodes)?;
        let textures = reader.read_entry(&dir_entries.textures)?;
        let faces = reader.read_entry(&dir_entries.faces)?;
        let lightmaps = reader.read_lightmaps(&dir_entries.lightmaps)?;
        let mut leaves: Box<[Leaf]> = reader.read_entry(&dir_entries.leaves)?;
        leaves.sort_unstable_by_key(|leaf| leaf.cluster);

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
                header,
                entities,
                textures,
                planes,
                nodes,
                leaves,
                leaf_faces,
                leaf_brushes,
                edges,
                surf_edges,
                models,
                brushes,
                brush_sides,
                vertices,
                faces,
                lightmaps,
                areas,
                area_portals,
                visdata,
            }
        })
    }

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

    pub fn potentially_visible_set(&self, leaf_id: usize) -> impl Iterator<Item = usize> + '_ {
        self.potential_set(leaf_id, |o| o.pvs)
    }

    pub fn potentially_hearable_set(&self, leaf_id: usize) -> impl Iterator<Item = usize> + '_ {
        self.potential_set(leaf_id, |o| o.phs)
    }

    pub fn leaf(&self, n: usize) -> Option<Handle<'_, Leaf>> {
        self.leaves.get(n).map(|leaf| Handle {
            bsp: self,
            data: leaf,
        })
    }

    pub fn plane(&self, n: usize) -> Option<Handle<'_, Plane>> {
        self.planes.get(n).map(|plane| Handle {
            bsp: self,
            data: plane,
        })
    }

    pub fn face(&self, n: usize) -> Option<Handle<'_, Face>> {
        self.faces.get(n).map(|face| Handle {
            bsp: self,
            data: face,
        })
    }

    pub fn faces(&self) -> impl Iterator<Item = Handle<'_, Face>> + '_ {
        self.faces.iter().map(move |face| Handle {
            bsp: self,
            data: face,
        })
    }

    pub fn texture(&self, n: usize) -> Option<&Texture> {
        self.textures.get(n)
    }

    pub fn node(&self, n: usize) -> Option<Handle<'_, Node>> {
        self.nodes.get(n).map(|node| Handle {
            bsp: self,
            data: node,
        })
    }

    pub fn root_node(&self) -> Option<Handle<'_, Node>> {
        self.node(self.models.get(0)?.headnode as usize)
    }

    pub fn model(&self, i: usize) -> Handle<'_, Model> {
        Handle {
            data: &self.models[i],
            bsp: self,
        }
    }

    pub fn models(&self) -> impl ExactSizeIterator<Item = Handle<'_, Model>> + Clone {
        self.models.iter().map(move |m| Handle::new(self, m))
    }

    pub fn leaf_at<C, I: Into<V3<C>>>(&self, point: I) -> Option<Handle<'_, Leaf>>
    where
        C: CoordSystem,
    {
        self.leaf_index_at(point).and_then(|i| self.leaf(i))
    }

    pub fn leaf_index_at<C, I: Into<V3<C>>>(&self, point: I) -> Option<usize>
    where
        C: CoordSystem,
    {
        let point = C::into_qvec(point.into());
        let mut current = self.root_node()?;

        loop {
            let plane = current.plane()?;
            let norm: &QVec = &plane.normal;
            let dot: f32 = -point.dot(norm);

            let [front, back] = current.children;

            let next = if dot < plane.dist { front } else { back };

            if next < 0 {
                return Some(-(next + 1) as usize);
            } else {
                current = self.node(next as usize)?;
            }
        }
    }

    pub fn leaves(&self) -> impl ExactSizeIterator<Item = Handle<'_, Leaf>> + Clone {
        self.leaves.iter().map(move |leaf| Handle {
            bsp: self,
            data: leaf,
        })
    }

    pub fn cluster_at<C, I: Into<V3<C>>>(&self, point: I) -> Option<Cluster>
    where
        C: CoordSystem,
    {
        self.leaf_at(point)
            .and_then(|leaf| leaf.cluster.try_into().ok())
    }

    pub fn leaves_in_cluster(
        &self,
        cluster: impl TryInto<i16>,
    ) -> impl Iterator<Item = Handle<'_, Leaf>> + Clone + '_ {
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

    pub fn clusters(&self) -> impl ExactSizeIterator<Item = u16> + Clone {
        0..self.visdata.cluster_offsets.len() as u16
    }

    /// We use `impl TryInto` so that `-1` is transparently converted to "no visible clusters",
    /// but if you know your cluster ID is valid then you can skip that check.
    pub fn visible_clusters(
        &self,
        from: impl TryInto<u16>,
    ) -> impl Iterator<Item = u16> + Clone + '_ {
        from.try_into().ok().into_iter().flat_map(move |from| {
            let cluster_vis_start = self.visdata.cluster_offsets[usize::from(from)].pvs;

            self.clusters().filter(move |&other| {
                if other == from {
                    true
                } else {
                    self.visdata.vecs[cluster_vis_start as u64 * 8 + other as u64]
                }
            })
        })
    }
}

impl<'a, T> Handle<'a, T> {
    pub fn new(bsp: &'a Bsp, data: &'a T) -> Self {
        Handle { bsp, data }
    }
}

impl<'a> Handle<'a, LeafFace> {
    pub fn face(self) -> Handle<'a, Face> {
        self.bsp.face(self.face as usize).unwrap()
    }
}

impl<'a> Handle<'a, Model> {
    pub fn faces(self) -> impl Iterator<Item = Handle<'a, Face>> {
        let start = self.face as usize;
        let end = start + self.num_faces as usize;

        self.bsp.faces[start..end]
            .iter()
            .map(move |face| Handle::new(self.bsp, face))
    }
}

impl<'a> Handle<'a, Face> {
    pub fn texture(self) -> Option<&'a Texture> {
        self.bsp.texture(self.texture as _)
    }

    pub fn texture_uvs(self) -> Option<impl Iterator<Item = (f32, f32)> + 'a> {
        let texture = self.texture()?;

        Some(self.vertices().map(move |vert| {
            (
                vert.dot(&texture.axis_u) + texture.offset_u,
                vert.dot(&texture.axis_v) + texture.offset_v,
            )
        }))
    }

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

    pub fn lightmap(self) -> Option<LightmapRef<'a>> {
        let start = usize::try_from(self.lightmap).ok()?;
        let (mins, maxs, w, h) = self.lightmap_dimensions()?;

        Some(LightmapRef {
            mins,
            maxs,
            width: w,
            data: &self.bsp.lightmaps[start..start + (w * h) as usize * 3],
        })
    }

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

    pub fn vertices(self) -> impl ExactSizeIterator<Item = &'a QVec> + 'a {
        self.vert_indices()
            .map(move |i| &self.bsp.vertices[i as usize])
    }
}

impl<'a> Handle<'a, Node> {
    pub fn plane(self) -> Option<Handle<'a, Plane>> {
        self.bsp.plane(self.plane as _)
    }

    pub fn face(self) -> Option<Handle<'a, Face>> {
        self.bsp.face(self.face as _)
    }
}

impl<'a> Handle<'a, Leaf> {
    pub fn leaf_faces(self) -> impl ExactSizeIterator<Item = Handle<'a, LeafFace>> + Clone {
        let start = self.leaf_face as usize;
        let end = start + self.num_leaf_faces as usize;

        self.bsp.leaf_faces[start..end]
            .iter()
            .map(move |leaf_face| Handle {
                bsp: self.bsp,
                data: leaf_face,
            })
    }

    pub fn faces(self) -> impl Iterator<Item = Handle<'a, Face>> {
        self.leaf_faces()
            .filter_map(move |leaf_face| self.bsp.face(leaf_face.face as usize))
    }
}
