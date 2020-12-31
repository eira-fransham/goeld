#version 450
#pragma optimize(on)
#pragma shader_stage(fragment)

layout(location = 0) in vec2 i_UV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler s_Color;
layout(set = 0, binding = 1) uniform Locals {
    vec2 a_Offset;
};
layout(set = 1, binding = 0) uniform texture2D t_Diffuse;

void main() {
    vec4 sum = texture(sampler2D(t_Diffuse, s_Color), i_UV) * 4.0;
    sum += texture(sampler2D(t_Diffuse, s_Color), i_UV - vec2(1.0) * a_Offset);
    sum += texture(sampler2D(t_Diffuse, s_Color), i_UV + vec2(1.0) * a_Offset);
    sum += texture(sampler2D(t_Diffuse, s_Color), i_UV + vec2(1.0, -1.0) * a_Offset);
    sum += texture(sampler2D(t_Diffuse, s_Color), i_UV - vec2(1.0, -1.0) * a_Offset);

    outColor = sum / 8;
}
