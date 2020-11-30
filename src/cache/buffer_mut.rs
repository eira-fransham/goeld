use crate::cache::{Cache, CacheCommon};
use std::{
    convert::TryFrom,
    iter::{self, FromIterator},
    marker::PhantomData,
    mem,
    ops::{Deref, Range},
};
use wgpu::util::DeviceExt;

pub struct BufferCacheMut<T> {
    values: Vec<T>,
    is_dirty: bool,
    buffer: Option<wgpu::Buffer>,
    buffer_usage: wgpu::BufferUsage,
    buffer_len: usize,
}

impl<T> BufferCacheMut<T> {
    pub fn new(buffer_usage: wgpu::BufferUsage) -> Self {
        Self {
            values: Vec::new(),
            is_dirty: false,
            buffer: None,
            buffer_usage: buffer_usage | wgpu::BufferUsage::COPY_SRC,
            buffer_len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn as_ref(&self) -> &[T] {
        self.values.as_ref()
    }

    pub fn as_mut(&mut self) -> &mut [T] {
        self.is_dirty = true;
        self.values.as_mut()
    }

    pub fn append_many<
        Int: TryFrom<u64>,
        O: FromIterator<Result<Range<Int>, <Int as TryFrom<u64>>::Error>>,
        Outer: IntoIterator<Item = Inner>,
        Inner: IntoIterator<Item = T>,
    >(
        &mut self,
        vals: Outer,
    ) -> (Range<u64>, O) {
        let start = self.len() as u64;
        let out = O::from_iter(vals.into_iter().map(|inner| {
            let start = self.len() as u64;
            self.values.extend(inner);
            let end = self.len() as u64;

            Ok(Int::try_from(start)?..Int::try_from(end)?)
        }));
        let end = self.len() as u64;

        (start..end, out)
    }
}

impl<T> Deref for BufferCacheMut<T> {
    type Target = Option<wgpu::Buffer>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

#[async_trait::async_trait]
impl<T> CacheCommon for BufferCacheMut<T>
where
    T: bytemuck::Pod + std::marker::Send,
{
    type Key = Range<u64>;

    async fn update_buffers(&mut self) -> Result<(), wgpu::BufferAsyncError> {
        if self.is_dirty && self.buffer_len >= self.values.len() {
            if let Some(buf) = &mut self.buffer {
                let slice = buf.slice(0..self.values.len() as u64);
                slice.map_async(wgpu::MapMode::Write).await?;
                slice
                    .get_mapped_range_mut()
                    .copy_from_slice(bytemuck::cast_slice(&self.values));
                buf.unmap();
            }
        }

        Ok(())
    }

    // TODO: Reuse the same buffer for transferring the data to the GPU, we don't need a new one every time.
    //       Really, this should be something like a `Vec` where it automatically resizes to a power of two.
    // Sadly it's not possible to do any buffer reuse because this function can't be async. Even if we made
    // it an inherent function instead of a trait fn we'd hit problems at the top-level event loop.
    fn update(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if self.is_dirty {
            self.buffer = Some(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&self.values),
                    usage: self.buffer_usage,
                }),
            );
            self.buffer_len = self.values.len();
            self.values.clear();
        }
    }

    fn clear(&mut self) {
        self.values.clear();
        self.buffer_len = 0;
    }
}

impl<T, I> Cache<I> for BufferCacheMut<T>
where
    I: IntoIterator<Item = T>,
    T: bytemuck::Pod + std::marker::Send,
{
    fn append(&mut self, vals: I) -> Self::Key {
        let start = self.values.len() as u64 + self.buffer_len as u64;
        self.values.extend(vals);
        let end = self.values.len() as u64 + self.buffer_len as u64;

        if start == end {
            0..0
        } else {
            start..end
        }
    }
}
