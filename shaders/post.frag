#version 450
#pragma optimize(on)
#pragma shader_stage(fragment)

/**
Basic FXAA implementation based on the code on geeks3d.com with the
modification that the texture2DLod stuff was removed since it's
unsupported by WebGL.

--

From:
https://github.com/mitsuhiko/webgl-meincraft

Copyright (c) 2011 by Armin Ronacher.

Some rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * The names of the contributors may not be used to endorse or
      promote products derived from this software without specific
      prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

layout(location = 0) in vec2 v_TexCoord;
layout(location = 1) in vec2 v_RgbNW;
layout(location = 2) in vec2 v_RgbNE;
layout(location = 3) in vec2 v_RgbSW;
layout(location = 4) in vec2 v_RgbSE;
layout(location = 5) in vec2 v_RgbM;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D t_Diffuse;
layout(set = 0, binding = 1) uniform sampler s_Color;
layout(set = 0, binding = 2) uniform Locals {
    float invGamma;
    float intensity;
};
layout(set = 0, binding = 3) uniform PostLocals {
    vec2 a_InvResolution;
    bool a_FxaaEnabled;
};

float luminance(vec3 color) {
    const float RAmount = 0.2126;
    const float GAmount = 0.7152;
    const float BAmount = 0.0722;

    return dot(color, vec3(RAmount, GAmount, BAmount));
}

vec3 applyLuminance(vec3 color, float lum) {
    float originalLuminance = luminance(color);
    float scale = lum / originalLuminance;

    return color * scale;
}

float aces(float lum) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    lum = (lum * (a * lum + b)) / (lum * (c * lum + d) + e);
    lum = pow(lum, invGamma);

    return clamp(lum, 0, 1);
}

vec3 aces(vec3 color) {
    float lum = aces(luminance(color));

    return applyLuminance(color, lum);
}

vec4 aces(vec4 color) {
    return vec4(aces(vec3(color)), color.a);
}

#ifndef FXAA_REDUCE_MIN
    #define FXAA_REDUCE_MIN   (1.0 / 256.0)
#endif
#ifndef FXAA_REDUCE_MUL
    #define FXAA_REDUCE_MUL   (1.0 / 16.0)
#endif
#ifndef FXAA_SPAN_MAX
    #define FXAA_SPAN_MAX     4.0
#endif

vec3 ifBetween(float val, float min, float max, vec3 ifTrue, vec3 ifFalse) {
    float valGeMin = step(min, val);
    float valLeMax = step(val, max);

    float cond = valGeMin * valLeMax;

    return ifTrue * cond + ifFalse * (1 - cond);
}

vec3 fxaa(
    texture2D diffuse,
    sampler smp,
    vec2 fragCoord,
    vec2 invResolution,
    vec2 inRgbNW,
    vec2 inRgbNE,
    vec2 inRgbSW,
    vec2 inRgbSE
) {
    vec3 rgbNW = texture(sampler2D(diffuse, smp), inRgbNW).xyz;
    vec3 rgbNE = texture(sampler2D(diffuse, smp), inRgbNE).xyz;
    vec3 rgbSW = texture(sampler2D(diffuse, smp), inRgbSW).xyz;
    vec3 rgbSE = texture(sampler2D(diffuse, smp), inRgbSE).xyz;
    vec3 rgbM  = texture(sampler2D(diffuse, smp), fragCoord).xyz;

    float lumaNW = clamp(luminance(rgbNW), 0, 1);
    float lumaNE = clamp(luminance(rgbNE), 0, 1);
    float lumaSW = clamp(luminance(rgbSW), 0, 1);
    float lumaSE = clamp(luminance(rgbSE), 0, 1);
    float lumaM  = clamp(luminance(rgbM), 0, 1);
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    vec2 dir = vec2(
        -((lumaNW + lumaNE) - (lumaSW + lumaSE)),
         ((lumaNW + lumaSW) - (lumaNE + lumaSE))
    );

    float dirReduce = max(
        (lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * FXAA_REDUCE_MUL,
        FXAA_REDUCE_MIN
    );

    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    vec2 spanMax = vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX);
    dir = clamp(
        dir * rcpDirMin,
        -spanMax,
        spanMax
    ) * invResolution;

    vec3 rgbA = 0.5 * (
        texture(sampler2D(diffuse, smp), fragCoord + dir * (1.0 / 3.0 - 0.5)).xyz +
        texture(sampler2D(diffuse, smp), fragCoord + dir * (2.0 / 3.0 - 0.5)).xyz);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture(sampler2D(diffuse, smp), fragCoord + dir * -0.5).xyz +
        texture(sampler2D(diffuse, smp), fragCoord + dir * 0.5).xyz);

    float lumaB = clamp(luminance(rgbB), 0, 1);
    // Equivalent to:
    // ```
    // if ((lumaB >= lumaMin) && (lumaB <= lumaMax))
    //     return vec4(rgbA, texColor.a);
    // else
    //     return vec4(rgbB, texColor.a);
    // ```
    return ifBetween(
        lumaB, lumaMin, lumaMax,
        rgbB,
        rgbA
    );
}
vec3 exposure(vec3 color) {
    color = 1 - exp(-color * intensity);

    return pow(
        color,
        vec3(invGamma)
    );
}

vec4 exposure(vec4 color) {
    return vec4(exposure(vec3(color)), color.a);
}

vec3 reinhard(vec3 color) {
    const float WhitePoint = 5.0;

    float lum = luminance(color) * intensity;
    lum = lum * (1 + (lum / (WhitePoint * WhitePoint))) / (1 + lum);

    return pow(
        applyLuminance(color, lum),
        vec3(invGamma)
    );
}

vec4 reinhard(vec4 color) {
    return vec4(reinhard(vec3(color)), color.a);
}

void main() {
    vec3 diffuse;

    if (a_FxaaEnabled) {
        diffuse = fxaa(
            t_Diffuse,
            s_Color,
            v_TexCoord,
            a_InvResolution,
            v_RgbNW,
            v_RgbNE,
            v_RgbSW,
            v_RgbSE
        );
    } else {
        diffuse = texture(sampler2D(t_Diffuse, s_Color), v_TexCoord).rgb;
    }

    outColor = vec4(aces(diffuse * intensity), 1);
}
