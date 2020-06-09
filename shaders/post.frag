#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec2 v_TexCoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D t_Diffuse;
layout(set = 0, binding = 1) uniform sampler s_Color;
layout(set = 0, binding = 2) uniform Locals {
    float invGamma;
    float intensity;
};

float luminance(vec3 color) {
    const float RAmount = 0.2126;
    const float GAmount = 0.7152;
    const float BAmount = 0.0722;

    return RAmount * color.r + GAmount * color.g + BAmount * color.b;
}

vec3 applyLuminance(vec3 color, float lum) {
    float originalLuminance = luminance(color);
    float scale = lum / originalLuminance;

    return color * scale;
}

vec3 exposure(vec3 color) {
    color = 1 - exp(-color * intensity);

    return pow(
        color,
        vec3(invGamma)
    );
}

vec4 exposure(vec4 color) {
    return vec4(exposure(vec3(color)), color.a);
}

vec3 aces(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    color *= intensity;

    float lum = luminance(color);

    // color = clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
    // return pow(color, vec3(invGamma));

    lum = clamp((lum * (a * lum + b)) / (lum * (c * lum + d) + e), 0.0, 1.0);
    lum = pow(lum, invGamma);

    return applyLuminance(color, lum);
}

vec4 aces(vec4 color) {
    return vec4(aces(vec3(color)), color.a);
}

vec3 reinhard(vec3 color) {
    const float WhitePoint = 5.0;

    float lum = luminance(color) * intensity;
    lum = lum * (1 + (lum / (WhitePoint * WhitePoint))) / (1 + lum);

    return pow(
        applyLuminance(color, lum),
        vec3(invGamma)
    );
}

vec4 reinhard(vec4 color) {
    return vec4(reinhard(vec3(color)), color.a);
}

void main() {
    outColor = aces(texture(sampler2D(t_Diffuse, s_Color), v_TexCoord));
}
