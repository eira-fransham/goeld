use super::error;
use arrayvec::ArrayString;
use bitflags::bitflags;
use goldsrc_format_common::{parseable, ElementSize, QVec, SimpleParse};

macro_rules! magic {
    (struct $name:ident($magic:expr);) => {
        #[derive(PartialEq, Default, Copy, Clone)]
        pub struct $name;

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                $magic.fmt(f)
            }
        }

        impl $name {
            pub const fn into_inner(self) -> [u8; 4] {
                $magic
            }
        }

        impl std::ops::Deref for $name {
            type Target = [u8; 4];

            fn deref(&self) -> &Self::Target {
                &$magic
            }
        }

        impl ElementSize for $name {
            const SIZE: usize = <[u8; 4]>::SIZE;
        }

        impl SimpleParse for $name {
            fn parse<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
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
pub const MODEL_NAME_SIZE: usize = 64;

parseable! {
    #[derive(Clone, Debug, Default, Copy)]
    pub struct DirEntry {
        pub count: u32,
        pub offset: u32,
    }
}

parseable! {
    #[derive(Debug, Clone, Copy)]
    pub struct Bounds {
        pub min: QVec,
        pub max: QVec,
    }
}

parseable! {
    #[derive(Debug, Clone)]
    pub struct Directories {
        pub bones: DirEntry,
        pub bone_controllers: DirEntry,
        pub hitboxes: DirEntry,
        pub sequences: DirEntry,
        pub sequence_groups: DirEntry,
        pub textures: DirEntry,
        pub texture_data_offset: u32,
        pub skin_ref_count: u32,
        pub skin_family_count: u32,
        pub skin_offset: u32,
        pub bodyparts: DirEntry,
        pub attachments: DirEntry,
        pub soundtable: u32,
        pub soundindex: u32,
        pub soundgroups: DirEntry,
        pub transitions: DirEntry,
    }
}

const MAGIC_NUMBER: [u8; 4] = [b'I', b'D', b'S', b'T'];

magic! {
    struct MdlMagic(MAGIC_NUMBER);
}

parseable! {
    #[derive(Debug, Clone)]
    pub struct Header {
        pub magic: MdlMagic,
        pub version: u32,
        pub name: ArrayString<[u8; MODEL_NAME_SIZE]>,
        pub filesize: u32,
        pub eye_position: QVec,
        pub size: Bounds,
        pub bounding_box: Bounds,
        pub flags: u32, // Unknown usage for now
    }
}

parseable! {
    #[derive(Debug, Clone)]
    pub struct BodyPart {
        pub name: ArrayString<[u8; MODEL_NAME_SIZE]>,
        pub model_count: u32,
        pub base: u32,
        pub model_offset: u32,
    }
}

parseable! {
    #[derive(Debug, Clone)]
    pub struct DirEntryBones {
        pub count: u32,
        pub bones_offset: u32,
        pub data_offset: u32,
    }
}

impl DirEntryBones {
    pub fn bones(&self) -> DirEntry {
        DirEntry {
            count: self.count,
            offset: self.bones_offset,
        }
    }

    pub fn data(&self) -> DirEntry {
        DirEntry {
            count: self.count,
            offset: self.data_offset,
        }
    }
}

parseable! {
    #[derive(Debug, Clone)]
    pub struct Model {
        pub name: ArrayString<[u8; MODEL_NAME_SIZE]>,
        pub type_: u32, // Unused
        pub bounding_radius: f32, // Unused
        pub meshes: DirEntry,
        pub vertices: DirEntryBones,
        pub normals: DirEntryBones,
        pub groups: DirEntry, // Unused
    }
}

parseable! {
    #[derive(Debug, Clone)]
    pub struct Mesh {
        pub triverts: DirEntry,
        pub skin_ref: u32,
        pub normals: DirEntry,
    }
}

parseable! {
    #[derive(Debug, Clone, Copy)]
    pub struct TriVert {
        pub position: u16,
        pub normal: u16,
        pub u: u16,
        pub v: u16,
    }
}

parseable! {
    #[derive(Debug, Clone)]
    pub struct TexInfo {
        pub name: ArrayString<[u8; MODEL_NAME_SIZE]>,
        pub flags: u32,
        pub width: u32,
        pub height: u32,
        pub texture_data_offset: u32,
    }
}

#[derive(Debug, Clone)]
pub struct Coordinates<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub roll: T,
    pub pitch: T,
    pub yaw: T,
}

impl<T> ElementSize for Coordinates<T>
where
    T: ElementSize,
{
    const SIZE: usize = T::SIZE * 6;
}

impl<T> SimpleParse for Coordinates<T>
where
    T: SimpleParse,
{
    fn parse<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        let [x, y, z, roll, pitch, yaw] = <[T; 6]>::parse(r)?;
        Ok(Coordinates {
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
        })
    }
}

parseable! {
    #[derive(Debug, Clone)]
    pub struct Texture {
        pub name: ArrayString<[u8; MODEL_NAME_SIZE]>,
        pub flags: u32,
        pub width: u32,
        pub height: u32,
        pub texture_data_offset: u32,
    }
}

bitflags! {
    #[derive(Default)]
    pub struct MotionFlags: u32 {
        const X       = 0b0000_0000_0000_0000_0001;
        const Y       = 0b0000_0000_0000_0000_0010;
        const Z       = 0b0000_0000_0000_0000_0100;
        const ROLL    = 0b0000_0000_0000_0000_1000;
        const PITCH   = 0b0000_0000_0000_0001_0000;
        const YAW     = 0b0000_0000_0000_0010_0000;
        const L_X     = 0b0000_0000_0000_0100_0000;
        const L_Y     = 0b0000_0000_0000_1000_0000;
        const L_Z     = 0b0000_0000_0001_0000_0000;
        const A_X     = 0b0000_0000_0010_0000_0000;
        const A_Y     = 0b0000_0000_0100_0000_0000;
        const A_Z     = 0b0000_0000_1000_0000_0000;
        const A_ROLL  = 0b0000_0001_0000_0000_0000;
        const A_PITCH = 0b0000_0010_0000_0000_0000;
        const A_YAW   = 0b0000_0100_0000_0000_0000;
        const RLOOP   = 0b0000_1000_0000_0000_0000;
    }
}

impl ElementSize for MotionFlags {
    const SIZE: usize = u32::SIZE;
}

impl SimpleParse for MotionFlags {
    fn parse<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        u32::parse(r).and_then(|v| {
            MotionFlags::from_bits(v).ok_or_else(|| error(format!("Invalid motion flags: {:b}", v)))
        })
    }
}

bitflags! {
    #[derive(Default)]
    pub struct BoneFlags: u32 {
        const HAS_NORMALS  = 0b0000_0000_0000_0000_0001;
        const HAS_VERTICES = 0b0000_0000_0000_0000_0010;
        const HAS_BBOX     = 0b0000_0000_0000_0000_0100;
        const HAS_CHROME   = 0b0000_0000_0000_0000_1000;
    }
}

impl ElementSize for BoneFlags {
    const SIZE: usize = u32::SIZE;
}

impl SimpleParse for BoneFlags {
    fn parse<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        u32::parse(r).and_then(|v| {
            BoneFlags::from_bits(v).ok_or_else(|| error(format!("Invalid bone flags: {:b}", v)))
        })
    }
}

parseable! {
    #[derive(Debug, Clone)]
    pub struct Bone {
        pub name: ArrayString<[u8; 32]>,
        pub parent: i32,
        pub flags: u32,
        pub bone_controller: Coordinates<i32>,
        pub value: Coordinates<f32>,
        pub scale: Coordinates<f32>,
    }
}

parseable! {
    #[derive(Debug, Clone)]
    pub struct BoneController {
        pub bone: u32,
        pub motion_type: MotionFlags,
        pub start: f32,
        pub end: f32,
        pub rest: i32,
        pub index: u32,
    }
}

bitflags! {
    #[derive(Default)]
    pub struct HitboxFlags: u32 {
        const GENERIC   = 0b0000_0000_0000_0000_0000;
        const HEAD      = 0b0000_0000_0000_0000_0001;
        const CHEST     = 0b0000_0000_0000_0000_0010;
        const STOMACH   = 0b0000_0000_0000_0000_0100;
        const LEFT_ARM  = 0b0000_0000_0000_0000_1000;
        const RIGHT_ARM = 0b0000_0000_0000_0001_0000;
        const LEFT_LEG  = 0b0000_0000_0000_0010_0000;
        const RIGHT_LEG = 0b0000_0000_0000_0100_0000;
    }
}

impl ElementSize for HitboxFlags {
    const SIZE: usize = u32::SIZE;
}

impl SimpleParse for HitboxFlags {
    fn parse<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        u32::parse(r).and_then(|v| {
            HitboxFlags::from_bits(v).ok_or_else(|| error(format!("Invalid hitbox flags: {:b}", v)))
        })
    }
}

parseable! {
    #[derive(Debug, Clone)]
    pub struct Hitbox {
        pub bone: u32,
        pub group: HitboxFlags,
        pub bounding_box: Bounds,
    }
}
