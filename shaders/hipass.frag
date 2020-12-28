#version 450
#pragma optimize(on)
#pragma shader_stage(fragment)

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D t_Diffuse;
layout(set = 0, binding = 1) uniform sampler s_Color;
layout(set = 0, binding = 2) uniform Locals {
    vec2 a_InvResolution;
    float a_Cutoff;
    float a_Intensity;
};

const mat3 ToXYZMatrix = mat3(
    0.4124564,  0.3575761,  0.1804375,
    0.2126729,  0.7151522,  0.0721750,
    0.0193339,  0.1191920,  0.9503041
);

float luminance(vec3 color) {
    return dot(color, ToXYZMatrix[1]);
}

vec3 applyLuminance(vec3 color, float lum) {
    float originalLuminance = luminance(color);
    float scale = lum / originalLuminance;

    return color * scale;
}

void main() {
    vec2 uv = gl_FragCoord.xy * a_InvResolution;
    vec3 color = texture(sampler2D(t_Diffuse, s_Color), uv).rgb * a_Intensity;

    outColor = vec4(
        ToXYZMatrix * clamp(color - applyLuminance(color, a_Cutoff), vec3(0), vec3(100)),
        1
    );
}
