#version 450
#pragma shader_stage(vertex)

#include <common.vert.inc>

layout(location = 3) in vec3 a_Normal;

layout(location = 2) out vec3 v_Normal;
layout(location = 3) out vec4 v_Position;

void main() {
    transformTexturedVertex(u_View, u_Proj);

    v_Normal = a_Normal;
    v_Position = a_Pos;
}
