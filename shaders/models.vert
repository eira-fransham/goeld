#version 450
#pragma shader_stage(vertex)

#include <common.vert.inc>

layout(location = 3) in vec3 a_Normal;

layout(location = 2) out vec3 v_Pos;
layout(location = 3) out vec3 v_Normal;

layout(set = 0, binding = 4) uniform ModelData {
    mat4 translation;
};

void main() {
    vec4 pos = translation * a_Pos;

    v_Pos = pos.xyz;
    v_Normal = a_Normal;

    transformTexturedVertex(u_View, u_Proj, pos);
}
