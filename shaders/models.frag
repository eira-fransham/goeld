#version 450
#pragma shader_stage(fragment)
#pragma optimize(on)

layout(location = 0) in vec4 v_TexCoord;
layout(location = 1) in flat uvec4 v_Tex;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform texture2D t_Diffuse;
layout(set = 0, binding = 2) uniform sampler s_Color;

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

    outColor = vec4(
        texture(
            sampler2D(t_Diffuse, s_Color),
            (offset + v_Tex.xy) /
                textureSize(sampler2D(t_Diffuse, s_Color), 0)
        ).rgb,
        1
    );
}
