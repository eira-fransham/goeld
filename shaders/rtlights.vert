#version 450
#pragma shader_stage(vertex)

#include <common.vert.inc>

layout(location = 3) in vec3 a_Normal;
layout(location = 4) in vec4 a_LightPos;
layout(location = 5) in vec4 a_LightColor;

layout(location = 2) out vec3 v_Pos;
layout(location = 3) out vec3 v_Normal;

layout(location = 4) out vec4 v_LightPos;
layout(location = 5) out vec4 v_LightColor;

layout(set = 0, binding = 1) uniform ModelData {
    mat4 translation;
};

void main() {
    vec4 pos = translation * a_Pos;

    v_Pos = pos.xyz;
    v_Normal = a_Normal;

    v_LightPos = a_LightPos;
    v_LightColor = a_LightColor;

    transformTexturedVertex(u_View, u_Proj, pos);
}
