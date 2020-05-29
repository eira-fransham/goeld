#version 450
#pragma shader_stage(vertex)

#include <common.vert.inc>

layout(location = 2) in vec4 a_Tex;
layout(location = 3) in vec2 a_LightmapCoord;
layout(location = 4) in float a_LightmapWidth;
layout(location = 5) in uint a_LightmapCount;
layout(location = 6) in float a_Value;

layout(location = 1) out vec4 v_Tex;
layout(location = 2) out vec2 v_LightmapCoord;
layout(location = 3) out float v_LightmapWidth;
layout(location = 4) out flat uint v_LightmapCount;
layout(location = 5) out float v_Value;

void main() {
    transformTexturedVertex(u_View, u_Proj);

    v_LightmapCoord = a_LightmapCoord;
    v_LightmapWidth = a_LightmapWidth;
    v_LightmapCount = a_LightmapCount;
    v_Tex = a_Tex;
    v_Value = a_Value;
}
