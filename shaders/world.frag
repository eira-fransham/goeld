#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec2 v_TexCoord;
layout(location = 1) in flat uvec4 v_Tex;
layout(location = 2) in flat uint v_TexStride;
layout(location = 3) in flat uint v_TexCount;
layout(location = 4) in vec2 v_LightmapCoord;
layout(location = 5) in float v_LightmapWidth;
layout(location = 6) in flat uint v_LightmapCount;
layout(location = 7) in float v_Value;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform texture2D t_Diffuse;
layout(binding = 2) uniform texture2D t_Lightmap;
layout(binding = 3) uniform sampler s_Color;
layout(binding = 4) uniform sampler s_Lightmap;
layout(binding = 5) uniform Locals {
    vec2 _unused;
    uint animationFrame;
};

float ifLt(float a, float b, float ifTrue, float ifFalse) {
    // `step` is equivalent to <=, so this is equivalent to `a >= b`
    // We want `a < b`, so we swap `ifFalse` and `ifTrue`.
    float ge = step(b, a);

    return ifFalse * ge + ifTrue * (1 - ge);
}

void main() {
    vec2 offset = vec2(
        ifLt(
            v_TexCoord.x, 
            0,
            mod(v_Tex.z - mod(-v_TexCoord.x, v_Tex.z), v_Tex.z),
            mod(v_TexCoord.x, v_Tex.z)
        ),
        ifLt(
            v_TexCoord.y, 
            0,
            mod(v_Tex.w - mod(-v_TexCoord.y, v_Tex.w), v_Tex.w),
            mod(v_TexCoord.y, v_Tex.w)
        )
    );

    vec4 light = vec4(vec3(v_Value), 1.);

    ivec2 lightmapSize = textureSize(sampler2D(t_Lightmap, s_Lightmap), 0);

    light += step(1, v_LightmapCount) * texture(
        sampler2D(t_Lightmap, s_Lightmap),
        v_LightmapCoord / lightmapSize
    );

    light += step(2, v_LightmapCount) * texture(
        sampler2D(t_Lightmap, s_Lightmap),
        (v_LightmapCoord + vec2(v_LightmapWidth, 0)) / lightmapSize
    );

    light += step(3, v_LightmapCount) * texture(
        sampler2D(t_Lightmap, s_Lightmap),
        (v_LightmapCoord + vec2(v_LightmapWidth, 0) * 2) / lightmapSize
    );

    light += step(4, v_LightmapCount) * texture(
        sampler2D(t_Lightmap, s_Lightmap),
        (v_LightmapCoord + vec2(v_LightmapWidth, 0) * 3) / lightmapSize
    );

    outColor = texture(
        sampler2D(t_Diffuse,  s_Color),
        (offset + vec2(v_Tex.xy) + vec2(v_TexStride * (animationFrame % v_TexCount), 0)) /
            textureSize(sampler2D(t_Diffuse, s_Color), 0)
    ) * light * 4;
}
