#version 450

layout(location = 0) in vec4 a_Pos;
layout(location = 1) in vec4 a_Tex;
layout(location = 2) in vec2 a_TexCoord;
layout(location = 3) in vec2 a_LightmapCoord;
layout(location = 0) out vec4 v_Tex;
layout(location = 1) out vec2 v_TexCoord;
layout(location = 2) out vec2 v_LightmapCoord;

layout(set = 0, binding = 0) uniform Locals {
    mat4 u_Transform;
};

void main() {
    v_TexCoord = a_TexCoord;
    v_LightmapCoord = a_LightmapCoord;
    v_Tex = a_Tex;
    gl_Position = u_Transform * a_Pos;
}
