mod types;

use arrayvec::ArrayVec;
use goldsrc_format_common::{QVec, SimpleParse};
use itertools::Itertools;
use std::{
    fmt, io, iter,
    num::{NonZeroI16, NonZeroI32, NonZeroU16, NonZeroU32},
    ops,
};
use types::{
    BodyPart, Bone, BoneController, BoneFlags, Bounds, Coordinates, DirEntry, DirEntryBones,
    Directories, Header, Hitbox, HitboxFlags, Mesh, Model, MotionFlags, TexInfo, Texture, TriVert,
    MODEL_NAME_SIZE,
};

#[cfg(not(debug_assertions))]
fn error(msg: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

#[cfg(debug_assertions)]
fn error(msg: impl ToString) -> io::Error {
    panic!("{}", msg.to_string())
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Todo;

pub struct Mdl<R> {
    pub header: Header,
    pub bones: Box<[Bone]>,
    pub bone_controllers: Box<[BoneController]>,
    pub hitboxes: Box<[Hitbox]>,
    pub sequences: Box<[Todo]>,
    pub sequence_groups: Box<[Todo]>,
    pub textures: Box<[Texture]>,
    pub skins: Box<[Box<[Todo]>]>,
    pub bodyparts: Box<[BodyPart]>,
    pub attachments: Box<[Todo]>,
    pub transitions: Box<[Todo]>,
    pub reader: R,
}

impl<R> fmt::Debug for Mdl<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        #[derive(Debug)]
        struct Mdl<'a> {
            header: &'a Header,
            bones: &'a [Bone],
            bone_controllers: &'a [BoneController],
            hitboxes: &'a [Hitbox],
            sequences: &'a [Todo],
            sequence_groups: &'a [Todo],
            textures: &'a [Texture],
            skins: &'a [Box<[Todo]>],
            bodyparts: &'a [BodyPart],
            attachments: &'a [Todo],
            transitions: &'a [Todo],
        }

        Mdl {
            header: &self.header,
            bones: &self.bones,
            bone_controllers: &self.bone_controllers,
            hitboxes: &self.hitboxes,
            sequences: &self.sequences,
            sequence_groups: &self.sequence_groups,
            textures: &self.textures,
            skins: &self.skins,
            bodyparts: &self.bodyparts,
            attachments: &self.attachments,
            transitions: &self.transitions,
        }
        .fmt(f)
    }
}

fn parse_entry<R: io::Read + io::Seek, T: SimpleParse, O: iter::FromIterator<T>>(
    reader: &mut R,
    directory: DirEntry,
) -> io::Result<O> {
    reader.seek(io::SeekFrom::Start(directory.offset as u64))?;

    T::parse_many(reader, directory.count as usize)
}

pub struct BodyParts<'a, R> {
    reader: &'a mut R,
    bodyparts: std::slice::Iter<'a, BodyPart>,
}

pub struct Models<'a, R> {
    // This assumes that at the first call to `Models::next`, `mdl.reader` has already
    // been `seek`'d to the start of the model chunk.
    reader: &'a mut R,
    count: Option<NonZeroU32>,
    offset: Option<u64>,
}

pub struct Meshes<'a, R> {
    reader: &'a mut R,
    count: Option<NonZeroU32>,
    offset: Option<u64>,
}

pub struct Handle<'a, T, R> {
    data: T,
    reader: &'a mut R,
}

type ModelHandle<'a, R> = Handle<'a, Model, R>;
type MeshHandle<'a, R> = Handle<'a, Mesh, R>;

impl<'a, T, R> ops::Deref for Handle<'a, T, R> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a, R> BodyParts<'a, R>
where
    R: io::Seek,
{
    pub fn next(&mut self) -> io::Result<Option<(&str, Models<'_, R>)>> {
        let cur = if let Some(cur) = self.bodyparts.next() {
            cur
        } else {
            return Ok(None);
        };

        self.reader
            .seek(io::SeekFrom::Start(cur.model_offset as u64))?;

        Ok(Some((
            &cur.name,
            Models {
                reader: &mut *self.reader,
                count: NonZeroU32::new(cur.model_count),
                offset: None,
            },
        )))
    }
}

impl<'a, R> Models<'a, R>
where
    R: io::Seek + io::Read,
{
    pub fn next(&mut self) -> io::Result<Option<ModelHandle<'_, R>>> {
        self.count = match self.count {
            Some(count) => NonZeroU32::new(count.get() - 1),
            None => return Ok(None),
        };

        if let Some(offset) = self.offset {
            self.reader.seek(io::SeekFrom::Start(offset))?;
        }

        let model = Model::parse(self.reader)?;
        self.offset = Some(self.reader.seek(io::SeekFrom::Current(0))?);
        Ok(Some(ModelHandle {
            data: model,
            reader: &mut *self.reader,
        }))
    }
}

impl<'a, R> Meshes<'a, R>
where
    R: io::Seek + io::Read,
{
    pub fn next(&mut self) -> io::Result<Option<MeshHandle<'_, R>>> {
        self.count = match self.count {
            Some(count) => NonZeroU32::new(count.get() - 1),
            None => return Ok(None),
        };

        if let Some(offset) = self.offset {
            self.reader.seek(io::SeekFrom::Start(offset))?;
        }

        let mesh = Mesh::parse(self.reader)?;
        self.offset = Some(self.reader.seek(io::SeekFrom::Current(0))?);
        Ok(Some(MeshHandle {
            data: mesh,
            reader: &mut *self.reader,
        }))
    }
}

impl<'a, R> ModelHandle<'a, R>
where
    R: io::Seek + io::Read,
{
    pub fn vertices(&mut self) -> io::Result<impl Iterator<Item = io::Result<QVec>> + '_> {
        let dir_entry = self.vertices.data();

        self.reader
            .seek(io::SeekFrom::Start(dir_entry.offset as u64))?;

        let reader = &mut *self.reader;

        Ok(iter::repeat_with(move || SimpleParse::parse(reader)).take(dir_entry.count as usize))
    }

    pub fn normals(&mut self) -> io::Result<impl Iterator<Item = io::Result<QVec>> + '_> {
        let dir_entry = self.normals.data();

        self.reader
            .seek(io::SeekFrom::Start(dir_entry.offset as u64))?;

        let reader = &mut *self.reader;

        Ok(iter::repeat_with(move || SimpleParse::parse(reader)).take(dir_entry.count as usize))
    }

    pub fn meshes(&mut self) -> io::Result<Meshes<'_, R>> {
        let dir_entry = self.meshes;

        self.reader
            .seek(io::SeekFrom::Start(dir_entry.offset as u64))?;

        let reader = &mut *self.reader;

        Ok(Meshes {
            reader,
            count: NonZeroU32::new(dir_entry.count),
            offset: None,
        })
    }
}

#[derive(Debug, Copy, Clone)]
enum VertType {
    Fan { center: TriVert, last: TriVert },
    Strip { last: (TriVert, TriVert) },
}

type TriVertCount = NonZeroU16;
type TriVertCountAndFlag = i16;

struct TriVertIter<R> {
    reader: R,
    to_emit: <ArrayVec<[TriVert; 3]> as IntoIterator>::IntoIter,
    state: Option<(TriVertCount, VertType)>,
    total: Option<NonZeroU32>,
}

impl<R> TriVertIter<R> {
    fn new(reader: R, total: u32) -> Self {
        TriVertIter {
            reader,
            to_emit: ArrayVec::default().into_iter(),
            total: NonZeroU32::new(total),
            state: None,
        }
    }
}

impl<R> Iterator for TriVertIter<R>
where
    R: io::Read,
{
    type Item = io::Result<TriVert>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(to_emit) = self.to_emit.next() {
            self.state = self.state.and_then(|(cur_count, type_)| {
                TriVertCount::new(cur_count.get() - 1).map(|c| (c, type_))
            });
            return Some(Ok(to_emit));
        }

        // The sequences _are not_ null-terminated, and it appears to be possible for a sequence
        // to be 0-sized
        let total = self.total?.get();

        let out = (|| -> io::Result<Option<_>> {
            let (cur_count, mut type_) = if let Some(state) = self.state {
                state
            } else {
                match TriVertCountAndFlag::parse(&mut self.reader)? {
                    count if count.abs() < 3 => {
                        for _ in 0..count.abs() {
                            TriVert::parse(&mut self.reader)?;
                        }

                        return Ok(None);
                    }
                    count if count < 0 => {
                        let (center, last) = SimpleParse::parse(&mut self.reader)?;

                        (
                            if let Some(c) = TriVertCount::new(-count as u16 - 2) {
                                c
                            } else {
                                return Ok(None);
                            },
                            VertType::Fan { center, last },
                        )
                    }
                    count => {
                        let last = SimpleParse::parse(&mut self.reader)?;
                        (
                            if let Some(c) = TriVertCount::new(count as u16 - 1) {
                                c
                            } else {
                                return Ok(None);
                            },
                            VertType::Strip { last },
                        )
                    }
                }
            };

            match &mut type_ {
                VertType::Fan { center, last } => {
                    let cur = SimpleParse::parse(&mut self.reader)?;
                    self.to_emit = ArrayVec::from([*last, cur, *center]).into_iter();
                    *last = cur;
                }
                VertType::Strip { last: (a, b) } => {
                    let cur = SimpleParse::parse(&mut self.reader)?;
                    self.to_emit = ArrayVec::from([*a, *b, cur]).into_iter();
                    *a = *b;
                    *b = cur;
                }
            }

            self.state = TriVertCount::new(cur_count.get() - 1).map(|c| (c, type_));

            Ok(self.to_emit.next())
        })();

        // Every time we set `to_emit`, that's a new triangle. So, here we decrease the triangle count.
        self.total = NonZeroU32::new(total - 1);

        match out {
            Ok(Some(v)) => Some(Ok(v)),
            Ok(None) => self.next(),
            Err(e) => Some(Err(e)),
        }
    }
}

impl<'a, R> MeshHandle<'a, R>
where
    R: io::Seek + io::Read,
{
    // TODO: We should use `flat_map` here to just return the indices in the format that they'll need
    //       to be consumed in, so the consumer doesn't have to worry about the fan vs strip debacle.
    pub fn triverts(&mut self) -> io::Result<impl Iterator<Item = io::Result<TriVert>> + '_> {
        self.reader
            .seek(io::SeekFrom::Start(self.triverts.offset as u64))?;

        // The wiki page states that this is the total number of triverts, while if you look at the source
        // you'll see that it's actually the number of _triangles_. This is important, as you will definitely
        // get the wrong information if you mix the two up.
        let count = self.triverts.count;

        let reader = &mut *self.reader;

        // TODO: Remove unwrap
        Ok(TriVertIter::new(reader, count))
    }
}

impl<R> Mdl<R>
where
    R: io::Read + io::Seek,
{
    pub fn read(mut reader: R) -> io::Result<Self> {
        const EXPECTED_VERSION: u32 = 0xa;

        let header = Header::parse(&mut reader)?;

        if header.version != EXPECTED_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Invalid version (expected {:?}, got {:?})",
                    EXPECTED_VERSION, header.version
                ),
            ));
        }

        let directories = Directories::parse(&mut reader)?;

        let bones = parse_entry(&mut reader, directories.bones)?;
        let bone_controllers = parse_entry(&mut reader, directories.bone_controllers)?;
        let hitboxes = parse_entry(&mut reader, directories.hitboxes)?;
        let textures = parse_entry(&mut reader, directories.textures)?;
        let bodyparts = parse_entry(&mut reader, directories.bodyparts)?;

        Ok(Mdl {
            header,
            bones,
            bone_controllers,
            hitboxes,
            textures,
            bodyparts,
            skins: Box::new([]),
            sequences: Box::new([]),
            sequence_groups: Box::new([]),
            attachments: Box::new([]),
            transitions: Box::new([]),
            reader,
        })
    }

    pub fn bodyparts(&mut self) -> BodyParts<'_, R> {
        BodyParts {
            reader: &mut self.reader,
            bodyparts: self.bodyparts.iter(),
        }
    }
}
