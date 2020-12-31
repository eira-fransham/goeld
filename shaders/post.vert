#version 450
#pragma shader_stage(vertex)
#pragma optimize(on)

layout(location = 0) in vec2 a_Pos;
layout(location = 0) out vec2 o_UV;

void main() {
    gl_Position = vec4(a_Pos, 0, 1);
    o_UV = vec2(a_Pos.x + 1, 1 - a_Pos.y) / 2;
}
