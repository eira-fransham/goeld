use crate::{
    cache::Cache,
    loader::{Load, LoadAsset, Loader},
    render::{Render, RenderCache, RenderContext, Vertex},
};
use rect_packer::Rect;
use std::{borrow::Cow, iter, ops::Range, path::Path};

#[derive(Debug)]
pub struct SkyboxAsset<'a>(pub Cow<'a, str>);

#[derive(Debug)]
pub struct Skybox {
    vert_offset: u64,
    index_range: Range<u32>,
}

impl LoadAsset for &'_ SkyboxAsset<'_> {
    type Asset = Skybox;

    fn load(self, loader: &Loader, cache: &mut RenderCache) -> anyhow::Result<Self::Asset> {
        let mut up: Option<Rect> = None;
        let mut bk: Option<Rect> = None;
        let mut rt: Option<Rect> = None;
        let mut ft: Option<Rect> = None;
        let mut lf: Option<Rect> = None;
        let mut dn: Option<Rect> = None;

        let names = &["up", "bk", "rt", "ft", "lf", "dn"];
        let name_len = self.0.len();
        let mut name = self.0.as_ref().to_owned();

        let loader = loader.env();

        for (rect_ptr, ext) in [&mut up, &mut bk, &mut rt, &mut ft, &mut lf, &mut dn]
            .iter_mut()
            .zip(names)
        {
            name.push_str(ext);

            if let Ok((file, path)) = loader.load(Path::new(&*name).into()) {
                let rect = cache.diffuse.append(image::load(
                    std::io::BufReader::new(file),
                    image::ImageFormat::from_path(&path)?,
                )?);

                **rect_ptr = Some(rect);
            }

            name.truncate(name_len);
        }

        let face_verts = [[-1., -1.], [-1., 1.], [1., 1.], [1., -1.]];

        let no_rect = Rect {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        };

        let up_verts = face_verts.iter().rev().map(|&[x, y]| ([x, y, 1.], [-x, y]));
        let bk_verts = face_verts.iter().rev().map(|&[y, z]| ([1., y, z], [-y, z]));
        let lf_verts = face_verts.iter().map(|&[x, z]| ([x, 1., z], [x, z]));
        let ft_verts = face_verts.iter().map(|&[y, z]| ([-1., y, z], [y, z]));
        let rt_verts = face_verts
            .iter()
            .rev()
            .map(|&[x, z]| ([x, -1., z], [-x, z]));
        let dn_verts = face_verts.iter().map(|&[x, y]| ([x, y, -1.], [-x, -y]));

        let cube_verts = up_verts
            .zip(iter::repeat(up.as_ref().unwrap_or(&no_rect)))
            .chain(bk_verts.zip(iter::repeat(bk.as_ref().unwrap_or(&no_rect))))
            .chain(rt_verts.zip(iter::repeat(rt.as_ref().unwrap_or(&no_rect))))
            .chain(ft_verts.zip(iter::repeat(ft.as_ref().unwrap_or(&no_rect))))
            .chain(lf_verts.zip(iter::repeat(lf.as_ref().unwrap_or(&no_rect))))
            .chain(dn_verts.zip(iter::repeat(dn.as_ref().unwrap_or(&no_rect))));

        let cube_verts = cube_verts.map(move |((pos, coord), rect)| {
            let tex_w = rect.width as f32;
            let tex_h = rect.height as f32;

            Vertex {
                pos: [pos[0] * 2., pos[1] * 2., pos[2] * 2., 1.],
                tex: [rect.x as f32, rect.y as f32, tex_w, tex_h],
                tex_coord: [
                    // We do `(x + 1) / 2` to convert from `-1..1` to `0..1`
                    (((coord[0] + 1.) / 2.) * tex_w).round(),
                    ((1. - (coord[1] + 1.) / 2.) * tex_h).round(),
                ],
                value: 1.0,
                lightmap_coord: [0., 0.],
                lightmap_width: 0.,
                lightmap_count: 0,
            }
        });

        let vert_offset = cache.vertices.append(cube_verts).start;
        let indices = (0..6)
            .flat_map(|face| {
                [0, 2, 1, 3, 2, 0]
                    .iter()
                    .map(move |&i| i + face * face_verts.len())
            })
            .map(|i| i as u32);
        let index_range = cache.indices.append(indices);
        let index_range = index_range.start as u32..index_range.end as u32;

        Ok(Skybox {
            vert_offset,
            index_range,
        })
    }
}

impl<'a> Render for &'a Skybox {
    type Indices = iter::Once<Range<u32>>;

    fn indices(self, _ctx: &RenderContext<'_>) -> (u64, Self::Indices) {
        (self.vert_offset, iter::once(self.index_range.clone()))
    }
}
