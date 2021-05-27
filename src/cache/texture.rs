use image::{GenericImageView, Pixel};
use rect_packer::{Config, Packer, Rect};
use wgpu::util::DeviceExt;

use crate::cache::{Cache, CacheCommon};

type CachedImage = image::RgbaImage;
type CachedSubpixel = <<CachedImage as GenericImageView>::Pixel as Pixel>::Subpixel;

pub struct Atlas {
    cache: Vec<CachedSubpixel>,
    unwritten: Vec<Rect>,
    alloc: Packer,
    texture: wgpu::Texture,
    padding: u32,
}

#[derive(Debug, Copy, Clone)]
pub struct AppendManyResult {
    pub first: Rect,
    pub all: Rect,
    pub stride_x: i32,
}

impl Default for AppendManyResult {
    fn default() -> Self {
        Self {
            first: Rect {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            },
            all: Rect {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            },
            stride_x: 0,
        }
    }
}

impl Atlas {
    pub fn new(texture: wgpu::Texture, width: u32, height: u32, padding: u32) -> Self {
        Self {
            alloc: Packer::new(Config {
                width: width as i32,
                height: height as i32,
                border_padding: padding as i32,
                rectangle_padding: 2 * padding as i32,
            }),
            cache: vec![],
            unwritten: vec![],
            texture,
            padding,
        }
    }

    pub fn texture_view(&self) -> wgpu::TextureView {
        self.texture.create_view(&Default::default())
    }

    pub fn append_many<I, Img>(
        &mut self,
        each_width: u32,
        each_height: u32,
        iter: I,
    ) -> AppendManyResult
    where
        I: ExactSizeIterator<Item = Img>,
        Img: GenericImageView,
        Img::Pixel: Pixel<Subpixel = <<CachedImage as GenericImageView>::Pixel as Pixel>::Subpixel>,
    {
        if each_width == 0 || each_height == 0 {
            return Default::default();
        }

        let aligned_width = match (each_width + self.padding * 2) % 64 {
            0 => each_width,
            other => each_width + (64 - other),
        };
        let allocated_width =
            iter.len() as u32 * aligned_width + (iter.len() as u32 - 1) * self.padding * 2;

        let rect = self
            .alloc
            .pack(allocated_width as i32, each_height as i32, false)
            .unwrap();

        let pad = self.padding as i32;
        let prev_size = self.cache.len();

        let total_space =
            (4 * (allocated_width + self.padding * 2) * (each_height + self.padding * 2)) as usize;
        self.cache.reserve(total_space);

        let mut current_x = rect.x - pad;

        for image in iter {
            let (width, height) = image.dimensions();

            debug_assert_eq!(width, each_width);
            debug_assert_eq!(height, each_height);

            for y in -pad..each_height as i32 + pad {
                for x in -pad..aligned_width as i32 + pad {
                    let src_x = x.min(width as i32 - 1).max(0) as u32;
                    let src_y = y.min(height as i32 - 1).max(0) as u32;

                    self.cache
                        .extend(&image.get_pixel(src_x, src_y).to_rgba().0);
                }
            }

            let width = aligned_width as i32 + 2 * pad;
            self.unwritten.push(Rect {
                x: current_x,
                y: rect.y - pad,
                width,
                height: height as i32 + 2 * pad,
            });

            current_x += width;
        }

        debug_assert_eq!(self.cache.len() - prev_size, total_space);

        AppendManyResult {
            first: Rect {
                width: each_width as i32,
                height: each_height as i32,
                ..rect
            },
            all: rect,
            stride_x: aligned_width as i32 + 2 * pad,
        }
    }
}

impl CacheCommon for Atlas {
    type Key = Rect;

    // TODO: Reuse the same buffer for transferring the data to the GPU, we don't need a new one every time.
    fn update(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if !self.unwritten.is_empty() {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &*self.cache,
                usage: wgpu::BufferUsage::COPY_SRC,
            });
            let mut offset = 0;

            for rect in self.unwritten.drain(..) {
                encoder.copy_buffer_to_texture(
                    wgpu::BufferCopyView {
                        buffer: &buffer,
                        layout: wgpu::TextureDataLayout {
                            offset: offset,
                            bytes_per_row: 4 * rect.width as u32,
                            rows_per_image: rect.height as u32,
                        },
                    },
                    wgpu::TextureCopyView {
                        texture: &self.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: rect.x as u32,
                            y: rect.y as u32,
                            z: 0,
                        },
                    },
                    wgpu::Extent3d {
                        width: rect.width as u32,
                        height: rect.height as u32,
                        depth: 1,
                    },
                );

                offset += 4 * rect.width as u64 * rect.height as u64;
            }

            self.cache.clear();
        }
    }

    fn clear(&mut self) {
        self.unwritten.clear();
        self.cache.clear();
        self.alloc = Packer::new(self.alloc.config());
    }
}

impl<I> Cache<I> for Atlas
where
    I: GenericImageView,
    I::Pixel: Pixel<Subpixel = <<CachedImage as GenericImageView>::Pixel as Pixel>::Subpixel>,
{
    fn append(&mut self, image: I) -> Rect {
        let (width, height) = image.dimensions();

        self.append_many(width, height, std::iter::once(image))
            .first
    }
}
