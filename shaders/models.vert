#version 450
#pragma shader_stage(vertex)

#include <common.vert.inc>

layout(set = 0, binding = 3) uniform ModelData {
    mat4 translation;
};

void main() {
    vec4 pos = translation * a_Pos;

    transformTexturedVertex(u_View, u_Proj, pos);
}
