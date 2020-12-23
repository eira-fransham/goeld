#version 450
#pragma shader_stage(vertex)

#include <common.vert.inc>

void main() {
    transformTexturedVertex(mat4(1), u_Proj, vec4(a_Pos, 0.5));
}
