#version 450
#pragma shader_stage(fragment)

#define MAX_LIGHTS 16
#define LIGHT_MULTIPLIER 1
#define LIGHT_FALLOFF 1
#define LIGHT_FALLOFF_SCALE 10

layout(location = 0) in vec2 v_TexCoord;
layout(location = 1) in flat uvec4 v_Tex;
layout(location = 2) in vec3 v_Pos;
layout(location = 3) in vec3 v_Normal;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform texture2D t_Diffuse;
layout(set = 0, binding = 2) uniform sampler s_Color;

struct Light {
    vec4 pos;
    vec4 color;
};

layout(set = 0, binding = 3) uniform Locals {
    vec4 _unused;
    float ambientLight;
};

layout(set = 0, binding = 5) uniform Lights {
    Light lights[MAX_LIGHTS];
};
layout(set = 0, binding = 6) uniform NumLights {
    uint numLights;
};

float ifLt(float a, float b, float ifTrue, float ifFalse) {
    float lt = step(b, a);

    return ifFalse * lt + ifTrue * (1 - lt);
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

    vec3 shadedAmount = vec3(ambientLight);

    for (uint i = 0; i < MAX_LIGHTS; i++) {
        vec3 lightVec = lights[i].pos.xyz - v_Pos;

        float dist = length(lightVec) * LIGHT_FALLOFF_SCALE;
        float amt = LIGHT_MULTIPLIER * lights[i].color.a / (1.0 + pow(dist, LIGHT_FALLOFF));
        float normAmt = max(dot(v_Normal, normalize(lightVec)), 0);

        amt = ifLt(amt, 0.001, 0, amt);

        shadedAmount += step(i + 1, numLights) * 
            amt *
            pow(normAmt, 2) *
            lights[i].color.rgb;
    }

    outColor = vec4(shadedAmount, 1) * texture(
        sampler2D(t_Diffuse, s_Color), (offset + v_Tex.xy) /
            textureSize(sampler2D(t_Diffuse, s_Color), 0)
    );
}
