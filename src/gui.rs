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
    pub iterations: i32,
    pub downscale: i32,
    pub factor: f32,
}

pub struct Config {
    pub gamma: f32,
    pub intensity: f32,
    pub tonemapping: Tonemapping,
    pub bloom: Bloom,
    pub fxaa: bool,
}

pub fn fps(ui: &Ui<'_>, fps: f64) {
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
            ui.text(im_str!("FPS: {:.0}", 1. / fps));
        });
}

pub fn config(ui: &Ui<'_>, config: &mut Config) {
    let window = Window::new(im_str!("Config"));
    window
        .size([300.0, 400.0], Condition::FirstUseEver)
        .position([20.0, 100.0], Condition::FirstUseEver)
        .build(&ui, || {
            ui.input_float(im_str!("Gamma"), &mut config.gamma).build();
            ui.input_float(im_str!("Intensity"), &mut config.intensity)
                .build();
            ui.checkbox(im_str!("FXAA"), &mut config.fxaa);

            if imgui::CollapsingHeader::new(im_str!("ACES Tonemapping"))
                .default_open(true)
                .build(&ui)
            {
                ui.checkbox(
                    im_str!("Tonemapping enabled"),
                    &mut config.tonemapping.enabled,
                );
                ui.checkbox(
                    im_str!("Tonemap in XYY colorspace"),
                    &mut config.tonemapping.xyy_aces,
                );

                if imgui::CollapsingHeader::new(im_str!("Crosstalk"))
                    .default_open(true)
                    .build(&ui)
                {
                    ui.checkbox(
                        im_str!("Crosstalk enabled"),
                        &mut config.tonemapping.crosstalk,
                    );
                    ui.checkbox(
                        im_str!("Crosstalk using luminance"),
                        &mut config.tonemapping.xyy_crosstalk,
                    );

                    ui.input_float(
                        im_str!("Crosstalk amount"),
                        &mut config.tonemapping.crosstalk_amt,
                    )
                    .build();
                    ui.input_float(im_str!("Saturation"), &mut config.tonemapping.saturation)
                        .build();
                    ui.input_float(
                        im_str!("Crosstalk saturation"),
                        &mut config.tonemapping.crosstalk_saturation,
                    )
                    .build();
                }

                if imgui::CollapsingHeader::new(im_str!("Bloom"))
                    .default_open(true)
                    .build(&ui)
                {
                    ui.checkbox(im_str!("Bloom enabled"), &mut config.bloom.enabled);
                    ui.input_float(im_str!("Bloom radius"), &mut config.bloom.radius)
                        .build();
                    ui.input_int(im_str!("Bloom iterations"), &mut config.bloom.iterations)
                        .build();
                    ui.input_int(
                        im_str!("Bloom initial downscale level"),
                        &mut config.bloom.downscale,
                    )
                    .build();
                    ui.input_float(im_str!("Bloom downscale factor"), &mut config.bloom.factor)
                        .build();
                    ui.input_float(im_str!("Bloom cutoff"), &mut config.bloom.cutoff)
                        .build();
                    ui.input_float(im_str!("Bloom influence"), &mut config.bloom.influence)
                        .build();
                }
            }
        });
}
