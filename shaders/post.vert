#version 450
#pragma shader_stage(vertex)
#pragma optimize(on)

layout(location = 0) in vec4 a_Pos;
layout(location = 1) in vec2 a_TexCoord;

layout(location = 0) out vec2 v_TexCoord;
layout(location = 1) out vec2 v_RgbNW;
layout(location = 2) out vec2 v_RgbNE;
layout(location = 3) out vec2 v_RgbSW;
layout(location = 4) out vec2 v_RgbSE;

layout(binding = 4) uniform Locals {
    vec2 u_InvResolution;
    bool u_FxaaEnabled;
};

void main() {
    v_TexCoord = a_TexCoord;
    gl_Position = a_Pos;

    if (u_FxaaEnabled) {
        v_RgbNW = v_TexCoord + vec2(-1.0, -1.0) * u_InvResolution;
        v_RgbNE = v_TexCoord + vec2(1.0, -1.0) * u_InvResolution;
        v_RgbSW = v_TexCoord + vec2(-1.0, 1.0) * u_InvResolution;
        v_RgbSE = v_TexCoord + vec2(1.0, 1.0) * u_InvResolution;
    }
}
