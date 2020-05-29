#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec2 v_TexCoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform texture2D t_Diffuse;
layout(set = 0, binding = 2) uniform sampler s_Color;
layout(set = 0, binding = 3) uniform Locals {
    vec4 v_AtlasSizes;
};

void main() {
    outColor = texture(sampler2D(t_Diffuse, s_Color), v_TexCoord / v_AtlasSizes.xy);
}
