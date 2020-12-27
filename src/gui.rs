use imgui::{im_str, Condition, Ui, Window};
use std::time::Duration;

pub struct Tonemapping {
    pub enabled: bool,
    pub xyy_aces: bool,
    pub crosstalk: bool,
    pub xyy_crosstalk: bool,
    pub crosstalk_amt: f32,
    pub saturation: f32,
    pub crosstalk_saturation: f32,
}

pub struct Bloom {
    pub enabled: bool,
    pub radius: f32,
    pub cutoff: f32,
    pub influence: f32,
}

pub struct Config {
    pub gamma: f32,
    pub intensity: f32,
    pub tonemapping: Tonemapping,
    pub bloom: Bloom,
}

#[derive(Copy, Clone, Default)]
pub struct ConfigDirty {
    pub gamma: bool,
    pub intensity: bool,
    pub tonemapping: bool,
    pub bloom: bool,
}

#[must_use]
pub fn draw(ui: &Ui<'_>, config: &mut Config, avg: f64) -> ConfigDirty {
    let mut out = ConfigDirty::default();

    let window = Window::new(im_str!("FPS Display"));
    window
        .title_bar(false)
        .scrollable(false)
        .resizable(false)
        .movable(false)
        .collapsible(false)
        .size([70., 10.], Condition::FirstUseEver)
        .position([0., 0.], Condition::FirstUseEver)
        .build(&ui, || {
            ui.text(im_str!("FPS: {:.0}", 1. / avg));
        });

    let window = Window::new(im_str!("Config"));
    window
        .size([300.0, 400.0], Condition::FirstUseEver)
        .position([20.0, 100.0], Condition::FirstUseEver)
        .build(&ui, || {
            out.gamma = ui.input_float(im_str!("Gamma"), &mut config.gamma).build();
            out.intensity = ui
                .input_float(im_str!("Intensity"), &mut config.intensity)
                .build();

            if imgui::CollapsingHeader::new(im_str!("ACES Tonemapping"))
                .default_open(true)
                .build(&ui)
            {
                out.tonemapping |= ui.checkbox(
                    im_str!("Tonemapping enabled"),
                    &mut config.tonemapping.enabled,
                );
                out.tonemapping |= ui.checkbox(
                    im_str!("Tonemap in XYY colorspace"),
                    &mut config.tonemapping.xyy_aces,
                );

                if imgui::CollapsingHeader::new(im_str!("Crosstalk"))
                    .default_open(true)
                    .build(&ui)
                {
                    out.tonemapping |= ui.checkbox(
                        im_str!("Crosstalk enabled"),
                        &mut config.tonemapping.crosstalk,
                    );
                    out.tonemapping |= ui.checkbox(
                        im_str!("Crosstalk using luminance"),
                        &mut config.tonemapping.xyy_crosstalk,
                    );

                    out.tonemapping |= ui
                        .input_float(
                            im_str!("Crosstalk amount"),
                            &mut config.tonemapping.crosstalk_amt,
                        )
                        .build();
                    out.tonemapping |= ui
                        .input_float(im_str!("Saturation"), &mut config.tonemapping.saturation)
                        .build();
                    out.tonemapping |= ui
                        .input_float(
                            im_str!("Crosstalk saturation"),
                            &mut config.tonemapping.crosstalk_saturation,
                        )
                        .build();
                }

                if imgui::CollapsingHeader::new(im_str!("Bloom"))
                    .default_open(true)
                    .build(&ui)
                {
                    out.bloom |= ui.checkbox(im_str!("Bloom enabled"), &mut config.bloom.enabled);
                    out.bloom |= ui
                        .input_float(im_str!("Bloom radius"), &mut config.bloom.radius)
                        .build();
                    out.bloom |= ui
                        .input_float(im_str!("Bloom cutoff"), &mut config.bloom.cutoff)
                        .build();
                    out.bloom |= ui
                        .input_float(im_str!("Bloom influence"), &mut config.bloom.influence)
                        .build();
                }
            }
        });

    out
}
