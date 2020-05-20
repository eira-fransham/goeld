#version 450

layout(location = 0) out vec4 outColor;
layout(location = 0) in vec4 v_Tex;
layout(location = 1) in vec2 v_TexCoord;
layout(location = 2) in vec2 v_LightmapCoord;

layout(set = 0, binding = 1) uniform texture2D t_Diffuse;
layout(set = 0, binding = 2) uniform texture2D t_Lightmap;
layout(set = 0, binding = 3) uniform sampler s_Color;
layout(set = 0, binding = 4) uniform sampler s_Lightmap;

void main() {
    vec2 offset = vec2(
        v_TexCoord.x < 0 
            ? mod(v_Tex.z - mod(-v_TexCoord.x, v_Tex.z), v_Tex.z)
            : mod(v_TexCoord.x, v_Tex.z),
        v_TexCoord.y < 0 
            ? mod(v_Tex.w - mod(-v_TexCoord.y, v_Tex.w), v_Tex.w)
            : mod(v_TexCoord.y, v_Tex.w)
    );

    outColor = 
        texture(sampler2D(t_Diffuse, s_Color), offset + v_Tex.xy) *
        texture(sampler2D(t_Lightmap, s_Lightmap), v_LightmapCoord);
}
