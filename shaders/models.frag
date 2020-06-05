#version 450
#pragma shader_stage(fragment)

#define MAX_LIGHTS 4

layout(location = 0) in vec2 v_TexCoord;
layout(location = 1) in flat uvec4 v_Tex;
layout(location = 2) in vec3 v_Normal;
layout(location = 3) in vec4 v_Position;

layout(location = 0) out vec4 outColor;

struct Light {
    vec4 pos;
    vec4 color;
};

layout(set = 0, binding = 1) uniform texture2D t_Diffuse;
layout(set = 0, binding = 2) uniform sampler s_Color;
layout(set = 0, binding = 3) uniform Locals {
    vec4 _unused;
    uint numLights;
};
layout(set = 0, binding = 4) uniform Lights {
    Light lights[MAX_LIGHTS];
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

    vec3 shadedAmount = vec3(0.001);

    for (uint i = 0; i < MAX_LIGHTS; i++) {
        float dist = length(vec3(lights[i].pos - v_Position));
        float amt = 1.0 / (1.0 + (0.01 * dist * dist));

        shadedAmount += step(i + 1, numLights) * 
            amt *
            max(dot(v_Normal, vec3(lights[i].pos - v_Position)), 0) *
            vec3(lights[i].color);
    }

    shadedAmount.x = min(shadedAmount.x, 1);
    shadedAmount.y = min(shadedAmount.y, 1);
    shadedAmount.z = min(shadedAmount.z, 1);

    outColor = vec4(shadedAmount, 1) * texture(
        sampler2D(t_Diffuse, s_Color), (offset + v_Tex.xy) /
            textureSize(sampler2D(t_Diffuse, s_Color), 0)
    );
}
