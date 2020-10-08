#version 450
#pragma shader_stage(vertex)

#include <common.vert.inc>

layout(location = 3) in uint a_TexStride;
layout(location = 4) in uint a_TexCount;
layout(location = 5) in vec2 a_LightmapCoord;
layout(location = 6) in float a_LightmapWidth;
layout(location = 7) in uint a_LightmapCount;
layout(location = 8) in float a_Value;

layout(location = 2) out uint v_TexStride;
layout(location = 3) out uint v_TexCount;
layout(location = 4) out vec2 v_LightmapCoord;
layout(location = 5) out float v_LightmapWidth;
layout(location = 6) out flat uint v_LightmapCount;
layout(location = 7) out float v_Value;

void main() {
    transformTexturedVertex(u_View, u_Proj);

    v_LightmapCoord = a_LightmapCoord;
    v_LightmapWidth = a_LightmapWidth;
    v_LightmapCount = a_LightmapCount;
    v_TexStride = a_TexStride;
    v_TexCount = a_TexCount;
    v_Value = a_Value;
}
