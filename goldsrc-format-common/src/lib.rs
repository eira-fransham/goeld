use arrayvec::{Array, ArrayString, ArrayVec};
use byteorder::{LittleEndian, ReadBytesExt};
use std::{io, iter, num};

#[cfg(not(debug_assertions))]
fn error(msg: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

#[cfg(debug_assertions)]
fn error(msg: impl ToString) -> io::Error {
    panic!("{}", msg.to_string())
}

pub trait ElementSize {
    const SIZE: usize;
}

pub trait SimpleParse: Sized {
    fn parse<R: io::Read>(r: &mut R) -> io::Result<Self>;

    fn parse_many<O: iter::FromIterator<Self>, R: io::Read>(
        r: &mut R,
        count: usize,
    ) -> io::Result<O> {
        iter::repeat_with(move || Self::parse(r))
            .take(count)
            .collect()
    }
}

macro_rules! impl_simpleparse_tup {
    () => {
        impl ElementSize for () {
            const SIZE: usize = 0;
        }

        impl SimpleParse for () {
            fn parse<R: io::Read>(_: &mut R) -> io::Result<Self> {
                Ok(())
            }
        }
    };
    ($a:ident $(, $rest:ident)*) => {
        impl<$a, $($rest,)*> ElementSize for ($a, $($rest,)*)
        where
            $a: ElementSize,
            $( $rest: ElementSize, )*
        {
            const SIZE: usize = (
                <$a>::SIZE $(+ <$rest>::SIZE)*
            );
        }

        impl<$a, $($rest,)*> SimpleParse for ($a, $($rest,)*)
        where
            $a: SimpleParse,
            $( $rest: SimpleParse, )*
        {
            fn parse<R: io::Read>(r: &mut R) -> io::Result<Self> {
                Ok((
                    <$a>::parse(r)?,
                    $(
                        <$rest>::parse(r)?,
                    )*
                ))
            }
        }

        impl_simpleparse_tup!($($rest),*);
    }
}

macro_rules! impl_simpleparse_nonzeroint {
    ($(($nonzero:ty, $maybezero:ty),)*) => {
        $(
            impl ElementSize for $nonzero {
                const SIZE: usize = <$maybezero as ElementSize>::SIZE;
            }

            impl SimpleParse for $nonzero {
                fn parse<R: io::Read>(r: &mut R) -> io::Result<Self> {
                    <$maybezero as SimpleParse>::parse(r)
                        .and_then(|val| {
                            Self::new(val)
                                .ok_or_else(|| {
                                    io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "Number should be nonzero",
                                    )
                                })
                        })
                }
            }
        )*
    }
}

macro_rules! impl_simpleparse_int {
    ($(($t:ty, $f:ident, $size:expr),)*) => {
        $(
            impl ElementSize for $t {
                const SIZE: usize = $size;
            }

            impl SimpleParse for $t {
                fn parse<R: io::Read>(r: &mut R) -> io::Result<Self> {
                    r.$f::<LittleEndian>()
                }
            }
        )*
    }
}

macro_rules! impl_simpleparse_arr {
    ($($count:expr),*) => {
        $(
            impl<T> ElementSize for [T; $count]
            where
                T: ElementSize
            {
                const SIZE: usize = T::SIZE * $count;
            }

            impl<T> SimpleParse for [T; $count]
            where
                T: SimpleParse,
            {
                fn parse<R: io::Read>(r: &mut R) -> io::Result<Self> {
                    let mut out = ArrayVec::<[T; $count]>::new();
                    for _ in 0..$count {
                        out.push(T::parse(r)?);
                    }

                    out.into_inner().map_err(|_| io::Error::from(io::ErrorKind::UnexpectedEof))
                }
            }
        )*
    }
}

impl<C> ElementSize for V3<C> {
    const SIZE: usize = <[f32; 3]>::SIZE;
}

impl<C> SimpleParse for V3<C>
where
    C: Default,
{
    fn parse<R: io::Read>(r: &mut R) -> io::Result<Self> {
        <[f32; 3]>::parse(r).map(|f| V3(f, C::default()))
    }
}

impl_simpleparse_arr!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128);

impl ElementSize for u8 {
    const SIZE: usize = 1;
}

impl ElementSize for i8 {
    const SIZE: usize = 1;
}

impl SimpleParse for u8 {
    fn parse<R: io::Read>(r: &mut R) -> io::Result<Self> {
        r.read_u8()
    }
}

impl SimpleParse for i8 {
    fn parse<R: io::Read>(r: &mut R) -> io::Result<Self> {
        r.read_i8()
    }
}

impl_simpleparse_int! {
    (u16, read_u16, 2),
    (i16, read_i16, 2),
    (u32, read_u32, 4),
    (i32, read_i32, 4),
    (u64, read_u64, 8),
    (i64, read_i64, 8),
    (f32, read_f32, 4),
    (f64, read_f64, 8),
}

impl_simpleparse_nonzeroint! {
    (num::NonZeroU8, u8),
    (num::NonZeroI8, i8),
    (num::NonZeroU16, u16),
    (num::NonZeroI16, i16),
    (num::NonZeroU32, u32),
    (num::NonZeroI32, i32),
    (num::NonZeroU64, u64),
    (num::NonZeroI64, i64),
}

impl_simpleparse_tup!(A, B, C, D, E, F, G);

impl<T> ElementSize for ArrayString<T>
where
    T: Array<Item = u8> + Copy,
{
    const SIZE: usize = T::CAPACITY;
}

impl<T> SimpleParse for ArrayString<T>
where
    T: Array<Item = u8> + Copy,
{
    fn parse<R: io::Read>(r: &mut R) -> io::Result<Self> {
        use std::{mem::MaybeUninit, str};

        let mut name_buf = unsafe { MaybeUninit::<T>::zeroed().assume_init() };
        r.read_exact(name_buf.as_mut_slice())?;
        let zero_pos = name_buf
            .as_slice()
            .iter()
            .position(|c| *c == 0)
            .ok_or_else(|| error("Name isn't null-terminated"))?;
        let name = &name_buf.as_slice()[..zero_pos];
        Ok(ArrayString::from(
            str::from_utf8(name).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?,
        )
        .expect("Programmer error: it should be impossible for the string to exceed the capacity"))
    }
}

pub trait CoordSystem: Sized {
    fn into_qvec(vec: V3<Self>) -> QVec;
    fn from_qvec(vec: QVec) -> V3<Self>;
}

#[derive(PartialEq, PartialOrd, Eq, Ord, Debug, Default, Copy, Clone)]
pub struct XEastYSouthZUp;
#[derive(PartialEq, PartialOrd, Eq, Ord, Debug, Default, Copy, Clone)]
pub struct XEastYNorthZUp;
#[derive(PartialEq, PartialOrd, Eq, Ord, Debug, Default, Copy, Clone)]
pub struct XEastYDownZSouth;

impl CoordSystem for XEastYSouthZUp {
    fn into_qvec(vec: V3<Self>) -> QVec {
        vec
    }

    fn from_qvec(vec: QVec) -> V3<Self> {
        vec
    }
}

impl CoordSystem for XEastYDownZSouth {
    fn into_qvec(vec: V3<Self>) -> QVec {
        V3::new([vec.x(), -vec.z(), vec.y()])
    }
    fn from_qvec(vec: QVec) -> V3<Self> {
        V3::new([vec.x(), vec.z(), -vec.y()])
    }
}

impl CoordSystem for XEastYNorthZUp {
    fn into_qvec(vec: V3<Self>) -> QVec {
        V3::new([vec.x(), -vec.y(), vec.y()])
    }
    fn from_qvec(vec: QVec) -> V3<Self> {
        V3::new([vec.x(), -vec.y(), vec.z()])
    }
}

#[derive(PartialEq, PartialOrd, Debug, Default, Clone, Copy)]
// So that we can ensure that `size_of` correctly reports the size of this type.
// `C` should be a ZST but if it isn't then this should still act correctly.
#[repr(C)]
pub struct V3<C>(pub [f32; 3], pub C);

impl<C> std::ops::Neg for V3<C> {
    type Output = Self;

    fn neg(self) -> Self {
        V3([-self.0[0], -self.0[1], -self.0[2]], self.1)
    }
}

impl<C> std::ops::Add<Self> for V3<C> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        V3(
            [
                self.0[0] + other.0[0],
                self.0[1] + other.0[1],
                self.0[2] + other.0[2],
            ],
            self.1,
        )
    }
}

impl<C> std::ops::Div<f32> for V3<C> {
    type Output = Self;

    fn div(self, other: f32) -> Self {
        V3(
            [self.0[0] / other, self.0[1] / other, self.0[2] / other],
            self.1,
        )
    }
}

impl<C> std::ops::Mul<f32> for V3<C> {
    type Output = Self;

    fn mul(self, other: f32) -> Self {
        V3(
            [self.0[0] * other, self.0[1] * other, self.0[2] * other],
            self.1,
        )
    }
}

impl<C> std::iter::Sum<Self> for V3<C>
where
    C: Default,
{
    fn sum<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let mut out = iter.next().unwrap_or(V3([0., 0., 0.], C::default()));

        for v in iter {
            out = out + v;
        }

        out
    }
}

impl<C> V3<C> {
    pub fn x(&self) -> f32 {
        self.0[0]
    }
    pub fn y(&self) -> f32 {
        self.0[1]
    }
    pub fn z(&self) -> f32 {
        self.0[2]
    }
}

impl<C: Default> V3<C> {
    pub fn new(xyz: [f32; 3]) -> Self {
        V3(xyz, Default::default())
    }
}

impl<C> V3<C> {
    pub fn dot(&self, other: &Self) -> f32 {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }
}

impl<C: Default> From<[f32; 3]> for V3<C> {
    fn from(xyz: [f32; 3]) -> Self {
        V3::new(xyz)
    }
}

pub type QVec = V3<XEastYSouthZUp>;

#[macro_export]
macro_rules! parseable {
    ($(#[$($any:tt)*])* $v:vis struct $name:ident { $($v0:vis $fname:ident : $fty:ty,)* }) => {
        impl $crate::ElementSize for $name {
            const SIZE: usize = {
                use std::mem;

                let mut a = 0;

                $(
                    a += <$fty as $crate::ElementSize>::SIZE;
                )*

                a
            };
        }

        impl $crate::SimpleParse for $name {
            fn parse<R: ::std::io::Read>(r: &mut R) -> ::std::io::Result<Self> {
                $( let $fname = <$fty as $crate::SimpleParse>::parse(&mut *r)?; )*

                Ok(Self { $($fname),* })
            }
        }

        $(#[$($any)*])* $v struct $name {
            $($v0 $fname : $fty,)*
        }
    }
}
