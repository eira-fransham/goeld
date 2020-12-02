use crate::cache::{Cache, CacheCommon};
use std::{
    convert::TryFrom,
    iter::FromIterator,
    ops::{Deref, Range},
};
use wgpu::util::DeviceExt;

pub struct BufferCacheMut<T> {
    values: Vec<T>,
    dirty_range: Option<Range<u64>>,
    buffer: Option<wgpu::Buffer>,
    buffer_usage: wgpu::BufferUsage,
    buffer_len: usize,
}

impl<T> BufferCacheMut<T> {
    pub fn new(buffer_usage: wgpu::BufferUsage) -> Self {
        Self {
            values: Vec::new(),
            dirty_range: None,
            buffer: None,
            buffer_usage: buffer_usage | wgpu::BufferUsage::COPY_SRC,
            buffer_len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn range_mut(&mut self, range: Range<u64>) -> &mut [T] {
        let dirty_range = if let Some(cur_range) = &self.dirty_range {
            cur_range.start.min(range.start)..cur_range.end.max(range.end)
        } else {
            range
        };

        self.dirty_range = Some(dirty_range);

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

        let dirty_range = if let Some(cur_range) = &self.dirty_range {
            cur_range.start.min(start)..cur_range.end.max(end)
        } else {
            start..end
        };

        self.dirty_range = Some(dirty_range);

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
        return Ok(());

        if let Some(range) = &self.dirty_range {
            if self.buffer_len >= self.values.len() {
                if let Some(buf) = &mut self.buffer {
                    let slice = buf.slice(range.clone());
                    slice.map_async(wgpu::MapMode::Write).await?;
                    slice
                        .get_mapped_range_mut()
                        .copy_from_slice(bytemuck::cast_slice(
                            &self.values[range.start as usize..range.end as usize],
                        ));
                    buf.unmap();

                    self.dirty_range = None;
                }
            }
        }

        Ok(())
    }

    fn update(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if self.dirty_range.is_some() {
            self.buffer = Some(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&self.values),
                    usage: self.buffer_usage,
                }),
            );
            self.buffer_len = self.values.len();
            self.dirty_range = None;
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

        let dirty_range = if let Some(cur_range) = &self.dirty_range {
            cur_range.start.min(start)..cur_range.end.max(end)
        } else {
            start..end
        };

        self.dirty_range = Some(dirty_range);

        if start == end {
            0..0
        } else {
            start..end
        }
    }
}
