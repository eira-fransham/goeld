#version 450
#pragma shader_stage(fragment)

#define LIGHT_MULTIPLIER 1
#define LIGHT_FALLOFF 1

layout(location = 2) in vec3 v_Pos;
layout(location = 3) in vec3 v_Normal;

layout(location = 4) in vec4 v_LightPos;
layout(location = 5) in vec4 v_LightColor;

layout(location = 1) out vec4 outLight;

layout(set = 0, binding = 2) uniform Locals {
    vec4 _unused;
    float ambientLight;
};

float ifLt(float a, float b, float ifTrue, float ifFalse) {
    float lt = step(b, a);

    return ifFalse * lt + ifTrue * (1 - lt);
}

void main() {
    vec3 shadedAmount = vec3(ambientLight);

    vec3 lightVec = v_LightPos.xyz - v_Pos;

    float dist = length(lightVec) * LIGHT_FALLOFF;
    float amt = LIGHT_MULTIPLIER * v_LightColor.a / (1.0 + dist * dist);

    outLight = vec4(
        v_LightColor.rgb * amt * max(dot(v_Normal, normalize(lightVec)), 0),
        1
    );
}
