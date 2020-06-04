#version 450
#pragma shader_stage(vertex)

#include <common.vert.inc>

void main() {
    transformTexturedVertex(u_View, u_Proj);
}
