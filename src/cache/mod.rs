use async_trait::async_trait;

mod buffer;
mod buffer_mut;
mod texture;

pub use buffer::{AlignedBufferCache, BufferCache};
pub use buffer_mut::BufferCacheMut;
pub use texture::{AppendManyResult, Atlas};

#[async_trait]
pub trait CacheCommon {
    type Key;

    async fn update_buffers(&mut self) -> Result<(), wgpu::BufferAsyncError> {
        Ok(())
    }
    fn update(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder);
    fn clear(&mut self);
}

pub trait Cache<T>: CacheCommon {
    fn append(&mut self, val: T) -> Self::Key;
}
