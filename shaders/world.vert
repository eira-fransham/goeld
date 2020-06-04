#version 450
#pragma shader_stage(vertex)

#include <common.vert.inc>

layout(location = 3) in uint a_TexCount;
layout(location = 4) in vec2 a_LightmapCoord;
layout(location = 5) in float a_LightmapWidth;
layout(location = 6) in uint a_LightmapCount;
layout(location = 7) in float a_Value;

layout(location = 2) out uint v_TexCount;
layout(location = 3) out vec2 v_LightmapCoord;
layout(location = 4) out float v_LightmapWidth;
layout(location = 5) out flat uint v_LightmapCount;
layout(location = 6) out float v_Value;

void main() {
    transformTexturedVertex(u_View, u_Proj);

    v_LightmapCoord = a_LightmapCoord;
    v_LightmapWidth = a_LightmapWidth;
    v_LightmapCount = a_LightmapCount;
    v_TexCount = a_TexCount;
    v_Value = a_Value;
}
