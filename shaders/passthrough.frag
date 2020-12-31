#version 450
#pragma optimize(on)
#pragma shader_stage(fragment)

layout(location = 0) in vec2 i_UV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D t_Diffuse;
layout(set = 0, binding = 1) uniform sampler s_Color;

void main() {
    outColor = texture(sampler2D(t_Diffuse, s_Color), i_UV);
}

