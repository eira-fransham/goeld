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

float luminance(vec3 color) {
    const float RAmount = 0.2126;
    const float GAmount = 0.7152;
    const float BAmount = 0.0722;

    return dot(color, vec3(RAmount, GAmount, BAmount));
}

vec3 applyLuminance(vec3 color, float lum) {
    float originalLuminance = luminance(color);
    float scale = lum / originalLuminance;

    return color * scale;
}

vec3 ifLt(float a, float b, vec3 ifTrue, vec3 ifFalse) {
    float lt = step(b, a);

    return ifFalse * lt + ifTrue * (1 - lt);
}

void main() {
    vec2 uv = gl_FragCoord.xy * a_InvResolution;
    vec3 color = texture(sampler2D(t_Diffuse, s_Color), uv).rgb * a_Intensity;

    outColor = vec4(clamp(color - applyLuminance(color, a_Cutoff), vec3(0), vec3(100)), 1);
}
