#pragma optimize(on)

layout(location = 0) in vec4 a_Pos;
layout(location = 1) in vec2 a_TexCoord;
layout(location = 2) in uvec4 a_Tex;

layout(location = 0) out vec2 v_TexCoord;
layout(location = 1) out uvec4 v_Tex;

layout(binding = 0) uniform Locals {
    mat4 u_View;
    mat4 u_Proj;
};

void transformTexturedVertex(mat4 view, mat4 proj) {
    v_TexCoord = a_TexCoord;
    v_Tex = a_Tex;
    gl_Position = proj * view * a_Pos;
}
