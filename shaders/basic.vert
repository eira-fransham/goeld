#version 450

layout(location = 0) in vec4 a_Pos;
layout(location = 1) in vec4 a_Tex;
layout(location = 2) in vec2 a_TexCoord;
layout(location = 3) in vec2 a_LightmapCoord;
layout(location = 4) in float a_LightmapWidth;
layout(location = 5) in uint a_LightmapCount;
layout(location = 6) in float a_Value;
layout(location = 0) out vec4 v_Tex;
layout(location = 1) out vec2 v_TexCoord;
layout(location = 2) out vec2 v_LightmapCoord;
layout(location = 3) out float v_LightmapWidth;
layout(location = 4) out flat uint v_LightmapCount;
layout(location = 5) out float v_Value;

layout(set = 0, binding = 0) uniform Locals {
    mat4 u_Transform;
};

void main() {
    v_TexCoord = a_TexCoord;
    v_LightmapCoord = a_LightmapCoord;
    v_LightmapWidth = a_LightmapWidth;
    v_LightmapCount = a_LightmapCount;
    v_Tex = a_Tex;
    v_Value = a_Value;
    gl_Position = u_Transform * a_Pos;
}
