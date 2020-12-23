#version 450
#pragma shader_stage(vertex)

#include <common.vert.inc>

layout(location = 3) in vec3 a_Normal;
layout(location = 4) in uint a_BoneId;

#define MAX_BONES 128

layout(set = 0, binding = 3) uniform ModelData {
    mat4 translation;
};
layout(set = 0, binding = 4) uniform Bones {
    mat4 bones[MAX_BONES];
};

void main() {
    mat4 boneTrans = bones[a_BoneId];
    vec4 pos = translation * boneTrans * vec4(a_Pos, 1.0);

    transformTexturedVertex(u_View, u_Proj, pos);
}
