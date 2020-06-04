#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec2 v_TexCoord;
layout(location = 1) in flat uvec4 v_Tex;
layout(location = 2) in flat uint v_TexCount;
layout(location = 3) in vec2 v_LightmapCoord;
layout(location = 4) in float v_LightmapWidth;
layout(location = 5) in flat uint v_LightmapCount;
layout(location = 6) in float v_Value;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform texture2D t_Diffuse;
layout(binding = 2) uniform texture2D t_Lightmap;
layout(binding = 3) uniform sampler s_Color;
layout(binding = 4) uniform sampler s_Lightmap;
layout(binding = 5) uniform Locals {
    vec2 _unused;
    uint animationFrame;
    uint atlasPadding;
};

vec4 antialiasGet(texture2D src, sampler smplr, vec2 uv) {
    vec2 size = textureSize(sampler2D(src, smplr), 0);
    vec2 puv = uv*size;

    vec2 hfw = 0.5*fwidth(puv);
    vec2 fl = floor(puv - 0.5) + 0.5;

    vec2 nnn = (fl + smoothstep(0.5 - hfw, 0.5 + hfw, puv - fl))/size;
    return texture(sampler2D(src, smplr), nnn);
}

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

    ivec2 lightmapSize = textureSize(sampler2D(t_Lightmap, s_Lightmap), 0);

    if (v_LightmapCount >= 1) {
        tmpLight += texture(sampler2D(t_Lightmap, s_Lightmap), v_LightmapCoord / lightmapSize);
    }

    if (v_LightmapCount >= 2) {
        tmpLight += texture(
            sampler2D(t_Lightmap, s_Lightmap),
            (v_LightmapCoord + vec2(v_LightmapWidth, 0)) / lightmapSize
        );
    }

    if (v_LightmapCount >= 3) {
        tmpLight += texture(
            sampler2D(t_Lightmap, s_Lightmap),
            (v_LightmapCoord + vec2(v_LightmapWidth, 0) * 2) / lightmapSize
        );
    }

    if (v_LightmapCount >= 4) {
        tmpLight += texture(
            sampler2D(t_Lightmap, s_Lightmap),
            (v_LightmapCoord + vec2(v_LightmapWidth, 0) * 3) / lightmapSize
        );
    }

    vec4 light = vec4(
        min(tmpLight.r, v_Value + 1),
        min(tmpLight.g, v_Value + 1),
        min(tmpLight.b, v_Value + 1), 
        1.0
    );

    outColor = antialiasGet(
        t_Diffuse, 
        s_Color,
        (offset + vec2(v_Tex.xy) + vec2((v_Tex.z + atlasPadding * 2) * (animationFrame % v_TexCount), 0)) /
            textureSize(sampler2D(t_Diffuse, s_Color), 0)
    ) * light;
}
