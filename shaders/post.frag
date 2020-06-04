#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec2 v_TexCoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D t_Diffuse;
layout(set = 0, binding = 1) uniform sampler s_Color;
layout(set = 0, binding = 2) uniform Locals {
    float inv_gamma;
    float intensity;
};

void main() {
    vec4 calculated = texture(sampler2D(t_Diffuse, s_Color), v_TexCoord);
    outColor = vec4(pow(calculated.rgb * 1.5, vec3(inv_gamma)) * intensity, 1);
}
