#version 450
#pragma shader_stage(vertex)

#include <common.vert.inc>

layout(location = 3) in vec3 a_Normal;
layout(location = 4) in uvec2 a_BoneIds;
layout(location = 5) in vec2 a_BoneWeights;

#define MAX_BONES 128

layout(set = 0, binding = 3) uniform ModelData {
    mat4 translation;
};
layout(set = 0, binding = 4) uniform Bones {
    mat4 bones[MAX_BONES];
};

void main() {
    mat4 boneTrans = bones[a_BoneIds[0]] * a_BoneWeights[0] + bones[a_BoneIds[1]] * a_BoneWeights[1];
    vec4 pos = translation * boneTrans * a_Pos;

    transformTexturedVertex(u_View, u_Proj, pos);
}
