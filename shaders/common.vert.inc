#pragma optimize(on)

layout(location = 0) in vec4 a_Pos;
layout(location = 1) in vec2 a_TexCoord;

layout(location = 0) out vec2 v_TexCoord;

layout(binding = 0) uniform Locals {
    mat4 u_View;
    mat4 u_Proj;
};

void transformTexturedVertex(mat4 view, mat4 proj) {
    v_TexCoord = a_TexCoord;
    gl_Position = proj * view * a_Pos;
}