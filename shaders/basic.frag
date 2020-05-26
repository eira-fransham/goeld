#version 450

#define INTENSITY 2.0

layout(location = 0) out vec4 outColor;
layout(location = 0) in vec4 v_Tex;
layout(location = 1) in vec2 v_TexCoord;
layout(location = 2) in vec2 v_LightmapCoord;
layout(location = 3) in float v_LightmapWidth;
layout(location = 4) in flat uint v_LightmapCount;
layout(location = 5) in float v_Value;

layout(set = 0, binding = 1) uniform texture2D t_Diffuse;
layout(set = 0, binding = 2) uniform texture2D t_Lightmap;
layout(set = 0, binding = 3) uniform sampler s_Color;
layout(set = 0, binding = 4) uniform sampler s_Lightmap;
layout(set = 0, binding = 5) uniform Locals {
    vec4 v_AtlasSizes;
    float inv_gamma;
    float intensity;
};

void main() {
    vec2 offset = vec2(
        v_TexCoord.x < 0 
            ? mod(v_Tex.z - mod(-v_TexCoord.x, v_Tex.z), v_Tex.z)
            : mod(v_TexCoord.x, v_Tex.z),
        v_TexCoord.y < 0 
            ? mod(v_Tex.w - mod(-v_TexCoord.y, v_Tex.w), v_Tex.w)
            : mod(v_TexCoord.y, v_Tex.w)
    );

    vec4 tmpLight = vec4(vec3(v_Value), 1.);

    if (v_LightmapCount >= 1) {
        tmpLight += texture(sampler2D(t_Lightmap, s_Lightmap), v_LightmapCoord / v_AtlasSizes.zw) 
            / v_LightmapCount;
    }

    if (v_LightmapCount >= 2) {
        tmpLight += texture(
            sampler2D(t_Lightmap, s_Lightmap),
            (v_LightmapCoord + vec2(v_LightmapWidth, 0)) / v_AtlasSizes.zw
        ) / v_LightmapCount;
    }

    if (v_LightmapCount >= 3) {
        tmpLight += texture(
            sampler2D(t_Lightmap, s_Lightmap),
            (v_LightmapCoord + vec2(v_LightmapWidth, 0) * 2) / v_AtlasSizes.zw
        ) / v_LightmapCount;
    }

    if (v_LightmapCount >= 4) {
        tmpLight += texture(
            sampler2D(t_Lightmap, s_Lightmap),
            (v_LightmapCoord + vec2(v_LightmapWidth, 0) * 3) / v_AtlasSizes.zw
        ) / v_LightmapCount;
    }

    vec4 light = vec4(
        min(tmpLight.r, v_Value + 1),
        min(tmpLight.g, v_Value + 1),
        min(tmpLight.b, v_Value + 1), 
        1.0
    );

    vec4 calculated = texture(sampler2D(t_Diffuse, s_Color), (offset + v_Tex.xy) / v_AtlasSizes.xy) * light;
    outColor = vec4(pow(calculated.rgb * 1.5, vec3(inv_gamma)) * intensity, calculated.a);
}
