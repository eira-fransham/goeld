use std::borrow::Cow;
use std::io::{self, Read, Seek};
use std::path::Path;

const DATA_FOLDER: &str = "data";
const TEXTURE_FOLDER: &str = "textures";
const MAP_FOLDER: &str = "maps";

pub struct Loader {
    _noconstruct: (),
}

pub struct WithExtensions<L> {
    inner: L,
    extensions: &'static [&'static str],
}

pub struct WithPath<L, const PATH: &'static str> {
    inner: L,
}

pub struct CaseInsensitive<L> {
    inner: L,
}

const MAP_EXTENSIONS: &[&str] = &["bsp"];
const TEXTURE_EXTENSIONS: &[&str] = &["bmp", "png", "tga", "gif", "tiff", "jpg", "jpeg"];

impl<L> Load for WithExtensions<L>
where
    L: Load,
{
    type Reader = L::Reader;

    fn load<'a>(&self, filename: Cow<'a, Path>) -> io::Result<(Self::Reader, Cow<'a, Path>)> {
        let mut filename = filename.into_owned();
        let mut err = None;

        for e in self.extensions {
            filename.set_extension(e);
            match self.inner.load(filename.clone().into()) {
                Ok(file) => return Ok(file),
                Err(e) => err = Some(e),
            }
        }

        Err(err.unwrap())
    }

    fn exists<'a>(&self, filename: Cow<'a, Path>) -> io::Result<(bool, Cow<'a, Path>)> {
        let mut filename = filename.into_owned();
        let mut out = Err(None);

        for e in self.extensions {
            filename.set_extension(e);
            match self.inner.exists(filename.clone().into()) {
                Ok((true, name)) => return Ok((true, name)),
                other => out = out.or(other.map_err(Some)),
            }
        }

        out.map_err(|e| e.unwrap())
    }
}

impl<L, const PATH: &'static str> Load for WithPath<L, PATH>
where
    L: Load,
{
    type Reader = L::Reader;

    fn load<'a>(&self, filename: Cow<'a, Path>) -> io::Result<(Self::Reader, Cow<'a, Path>)> {
        self.inner.load(Path::new(PATH).join(filename).into())
    }

    fn exists<'a>(&self, filename: Cow<'a, Path>) -> io::Result<(bool, Cow<'a, Path>)> {
        self.inner.exists(Path::new(PATH).join(filename).into())
    }
}

impl<L> Load for CaseInsensitive<L>
where
    L: Load,
{
    type Reader = L::Reader;

    fn load<'a>(&self, filename: Cow<'a, Path>) -> io::Result<(Self::Reader, Cow<'a, Path>)> {
        if let Ok(out) = self.inner.load(filename.clone()) {
            return Ok(out);
        }

        let mut filename = filename.into_owned();
        filename.set_file_name(filename.file_name().unwrap().to_ascii_uppercase());
        if let Some(e) = filename.extension() {
            filename.set_extension(e.to_ascii_lowercase());
        }

        self.inner.load(filename.into())
    }

    fn exists<'a>(&self, filename: Cow<'a, Path>) -> io::Result<(bool, Cow<'a, Path>)> {
        if let Ok(out) = self.inner.exists(filename.clone()) {
            return Ok(out);
        }

        let mut filename = filename.into_owned();
        filename.set_file_name(filename.file_name().unwrap().to_ascii_uppercase());
        if let Some(e) = filename.extension() {
            filename.set_extension(e.to_ascii_lowercase());
        }

        self.inner.exists(filename.into())
    }
}

pub trait Load {
    type Reader: Read + Seek;

    fn load<'a>(&self, filename: Cow<'a, Path>) -> io::Result<(Self::Reader, Cow<'a, Path>)>;
    fn exists<'a>(&self, filename: Cow<'a, Path>) -> io::Result<(bool, Cow<'a, Path>)>;
}

impl<T> Load for &'_ T
where
    T: Load,
{
    type Reader = T::Reader;

    fn load<'a>(&self, filename: Cow<'a, Path>) -> io::Result<(Self::Reader, Cow<'a, Path>)> {
        T::load(*self, filename)
    }
    fn exists<'a>(&self, filename: Cow<'a, Path>) -> io::Result<(bool, Cow<'a, Path>)> {
        T::exists(*self, filename)
    }
}

impl Load for Loader {
    type Reader = std::fs::File;

    fn load<'a>(&self, filename: Cow<'a, Path>) -> io::Result<(Self::Reader, Cow<'a, Path>)> {
        Ok((std::fs::File::open(filename.as_ref())?, filename))
    }
    fn exists<'a>(&self, filename: Cow<'a, Path>) -> io::Result<(bool, Cow<'a, Path>)> {
        Ok((filename.exists(), filename))
    }
}

impl Loader {
    pub const fn new() -> Self {
        Loader { _noconstruct: () }
    }

    fn data(&self) -> impl Load + '_ {
        WithPath::<_, DATA_FOLDER> { inner: self }
    }

    pub fn textures(&self) -> impl Load + '_ {
        WithExtensions {
            inner: CaseInsensitive {
                inner: WithPath::<_, TEXTURE_FOLDER> { inner: self.data() },
            },
            extensions: TEXTURE_EXTENSIONS,
        }
    }

    pub fn maps(&self) -> impl Load + '_ {
        WithExtensions {
            inner: CaseInsensitive {
                inner: WithPath::<_, MAP_FOLDER> { inner: self.data() },
            },
            extensions: MAP_EXTENSIONS,
        }
    }
}
