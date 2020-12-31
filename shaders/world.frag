#version 450
#pragma shader_stage(fragment)

#define WARP_SIZE 0.001
#define WARP_AMOUNT 2.6
#define WARP_SPEED 0.6

layout(location = 0) in vec2 v_TexCoord;
layout(location = 1) in flat uvec4 v_Tex;
layout(location = 2) in flat uint v_TexStride;
layout(location = 3) in flat int v_TexCount;
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
    float animationFrame;
    float _unused2;
};

float ifLt(float a, float b, float ifTrue, float ifFalse) {
    // `step` is equivalent to <=, so this is equivalent to `a >= b`
    // We want `a < b`, so we swap `ifFalse` and `ifTrue`.
    float ge = step(b, a);

    return ifFalse * ge + ifTrue * (1 - ge);
}

const mat3 FromXYZMatrix = mat3(
    3.2404542, -1.5371385, -0.4985314,
    -0.9692660, 1.8760108, 0.0415560,
    0.0556434, -0.2040259, 1.0572252
);

const mat3 ToXYZMatrix = mat3(
    0.4124564,  0.3575761,  0.1804375,
    0.2126729,  0.7151522,  0.0721750,
    0.0193339,  0.1191920,  0.9503041
);

void main() {
    uint texCount;
    vec2 texCoord;

    if (v_TexCount < 0) {
        texCount = -v_TexCount;

        vec2 sinoffsetvec = animationFrame * WARP_SPEED + 
            WARP_SIZE * vec2(v_TexCoord.y * 0.9, v_TexCoord.x * 1.4) * v_TexStride / (gl_FragCoord.z + 1);

        texCoord = v_TexCoord + WARP_AMOUNT * sin(sinoffsetvec);
    } else {
        texCount = v_TexCount;
        texCoord = v_TexCoord;
    }

    vec2 offset = vec2(
        ifLt(
            texCoord.x, 
            0,
            mod(v_Tex.z - mod(-texCoord.x, v_Tex.z), v_Tex.z),
            mod(texCoord.x, v_Tex.z)
        ),
        ifLt(
            texCoord.y, 
            0,
            mod(v_Tex.w - mod(-texCoord.y, v_Tex.w), v_Tex.w),
            mod(texCoord.y, v_Tex.w)
        )
    );

    vec3 light;

    light = vec3(log2(v_Value + 1));

    ivec2 lightmapSize = textureSize(sampler2D(t_Lightmap, s_Lightmap), 0);

    switch (v_LightmapCount) {
    case 4:
        light += step(4, v_LightmapCount) * texture(
            sampler2D(t_Lightmap, s_Lightmap),
            (v_LightmapCoord + vec2(v_LightmapWidth, 0) * 3) / lightmapSize
        ).rgb;
    case 3:
        light += step(3, v_LightmapCount) * texture(
            sampler2D(t_Lightmap, s_Lightmap),
            (v_LightmapCoord + vec2(v_LightmapWidth, 0) * 2) / lightmapSize
        ).rgb;
    case 2:
        light += step(2, v_LightmapCount) * texture(
            sampler2D(t_Lightmap, s_Lightmap),
            (v_LightmapCoord + vec2(v_LightmapWidth, 0)) / lightmapSize
        ).rgb;
    case 1:
        light += step(1, v_LightmapCount) * texture(
            sampler2D(t_Lightmap, s_Lightmap),
            v_LightmapCoord / lightmapSize
        ).rgb;
    }

    uint frame = uint(animationFrame);

    outColor = 32 * texture(
        sampler2D(t_Diffuse,  s_Color),
        (offset + vec2(v_Tex.xy) + vec2(v_TexStride * (frame % texCount), 0)) /
            textureSize(sampler2D(t_Diffuse, s_Color), 0)
    ) * vec4(light, 1);
}
