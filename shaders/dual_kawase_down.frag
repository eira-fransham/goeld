#version 450
#pragma optimize(on)
#pragma shader_stage(fragment)

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler s_Color;
layout(set = 0, binding = 1) uniform Locals {
    vec2 a_Offset;
};
layout(set = 1, binding = 0) uniform texture2D t_Diffuse;
layout(set = 1, binding = 1) uniform TexLocals {
    vec2 a_InvResolution;
};

void main() {
    vec2 uv = gl_FragCoord.xy * a_InvResolution;

    vec4 sum = texture(sampler2D(t_Diffuse, s_Color), uv) * 4.0;
    sum += texture(sampler2D(t_Diffuse, s_Color), uv - vec2(1.0) * a_Offset);
    sum += texture(sampler2D(t_Diffuse, s_Color), uv + vec2(1.0) * a_Offset);
    sum += texture(sampler2D(t_Diffuse, s_Color), uv + vec2(1.0, -1.0) * a_Offset);
    sum += texture(sampler2D(t_Diffuse, s_Color), uv - vec2(1.0, -1.0) * a_Offset);

    outColor = sum / 8.0;
}
