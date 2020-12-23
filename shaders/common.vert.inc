#pragma optimize(on)

layout(location = 0) in vec3 a_Pos;
layout(location = 1) in vec2 a_TexCoord;
layout(location = 2) in uvec4 a_Tex;

layout(location = 0) out vec2 v_TexCoord;
layout(location = 1) out uvec4 v_Tex;

layout(binding = 0) uniform Locals {
    mat4 u_View;
    mat4 u_Proj;
};

void transformTexturedVertex(mat4 view, mat4 proj, vec4 pos) {
    v_TexCoord = a_TexCoord;
    v_Tex = a_Tex;
    gl_Position = proj * view * pos;
}

void transformTexturedVertex(mat4 view, mat4 proj) {
    transformTexturedVertex(view, proj, vec4(a_Pos, 1.0));
}

void texCoords(
    vec2 fragCoord,
    vec2 resolution,
    out vec2 v_rgbNW,
    out vec2 v_rgbNE,
    out vec2 v_rgbSW,
    out vec2 v_rgbSE,
    out vec2 v_rgbM
) {
    vec2 inverseVP = 1.0 / resolution.xy;

    v_rgbNW = (fragCoord + vec2(-1.0, -1.0)) * inverseVP;
    v_rgbNE = (fragCoord + vec2(1.0, -1.0)) * inverseVP;
    v_rgbSW = (fragCoord + vec2(-1.0, 1.0)) * inverseVP;
    v_rgbSE = (fragCoord + vec2(1.0, 1.0)) * inverseVP;
    v_rgbM = vec2(fragCoord * inverseVP);
}
