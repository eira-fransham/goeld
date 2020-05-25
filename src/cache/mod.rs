mod buffer;
mod texture;

pub use buffer::BufferCache;
pub use texture::Atlas;

pub trait CacheCommon {
    type Key;

    fn update(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder);
    fn clear(&mut self);
}

pub trait Cache<T>: CacheCommon {
    fn append(&mut self, val: T) -> Self::Key;
}
