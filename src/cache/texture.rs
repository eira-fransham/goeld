use image::{GenericImageView, Pixel, RgbaImage};
use rect_packer::{Config, Packer, Rect};

use crate::cache::{Cache, CacheCommon};

type CachedImage = image::RgbaImage;
type CachedSubpixel = <<CachedImage as GenericImageView>::Pixel as Pixel>::Subpixel;

pub struct Atlas {
    cache: Vec<CachedSubpixel>,
    unwritten: Vec<Rect>,
    alloc: Packer,
    texture: wgpu::Texture,
    padding: u32,
    width: u32,
    height: u32,
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
            width,
            height,
        }
    }

    pub fn texture_view(&self) -> wgpu::TextureView {
        self.texture.create_default_view()
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn append_many<I, Img>(&mut self, each_width: u32, each_height: u32, iter: I) -> Rect
    where
        I: ExactSizeIterator<Item = Img>,
        Img: GenericImageView,
        Img::Pixel: Pixel<Subpixel = <<CachedImage as GenericImageView>::Pixel as Pixel>::Subpixel>,
    {
        if each_width == 0 || each_height == 0 {
            return Rect {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            };
        }

        let allocated_width =
            iter.len() as u32 * each_width + (iter.len() as u32 - 1) * self.padding * 2;

        let rect = self
            .alloc
            .pack(allocated_width as i32, each_height as i32, false)
            .unwrap();

        let pad = self.padding as i32;
        let prev_size = self.cache.len();

        for image in iter {
            let (width, height) = image.dimensions();

            debug_assert_eq!(width, each_width);
            debug_assert_eq!(height, each_height);

            for y in -pad..each_height as i32 + pad {
                for x in -pad..width as i32 + pad {
                    let src_x = x.min(width as i32 - 1).max(0) as u32;
                    let src_y = y.min(height as i32 - 1).max(0) as u32;

                    self.cache
                        .extend(&image.get_pixel(src_x, src_y).to_rgba().0);
                }
            }
        }

        assert_eq!(
            self.cache.len() - prev_size,
            (4 * (allocated_width + self.padding * 2) * (each_height + self.padding * 2)) as usize
        );

        self.unwritten.push(Rect {
            x: rect.x - pad,
            y: rect.y - pad,
            width: allocated_width as i32 + 2 * pad,
            height: rect.height + 2 * pad,
        });

        rect
    }
}

impl CacheCommon for Atlas {
    type Key = Rect;

    // TODO: Reuse the same buffer for transferring the data to the GPU, we don't need a new one every time.
    fn update(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if !self.unwritten.is_empty() {
            let buffer = device.create_buffer_with_data(&*self.cache, wgpu::BufferUsage::COPY_SRC);
            let mut offset = 0;

            for rect in self.unwritten.drain(..) {
                encoder.copy_buffer_to_texture(
                    wgpu::BufferCopyView {
                        buffer: &buffer,
                        offset: offset,
                        bytes_per_row: 4 * rect.width as u32,
                        rows_per_image: rect.height as u32,
                    },
                    wgpu::TextureCopyView {
                        texture: &self.texture,
                        mip_level: 0,
                        array_layer: 0,
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
    }
}
