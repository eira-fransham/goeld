#version 450
#pragma optimize(on)
#pragma shader_stage(fragment)

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D t_Diffuse;
layout(set = 0, binding = 1) uniform texture2D t_Bloom;
layout(set = 0, binding = 2) uniform sampler s_Color;
layout(set = 0, binding = 3) uniform Locals {
    float invGamma;
    float intensity;
};
layout(set = 0, binding = 4) uniform PostLocals {
    vec2 a_InvResolution;
    uint a_TonemappingBitmap;
    float a_InvCrosstalkAmt;
    float a_Saturation;
    float a_CrosstalkSaturation;
    float a_BloomInfluence;
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

float acesLum(float lum) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    lum = (lum * (a * lum + b)) / (lum * (c * lum + d) + e);

    return lum;
}

vec3 acesLum(vec3 color) {
    float lum = acesLum(luminance(color));

    return applyLuminance(color, lum);
}

vec4 acesLum(vec4 color) {
    return vec4(acesLum(vec3(color)), color.a);
}

vec3 aces(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    return (color * (a * color + b)) / (color * (c * color + d) + e);
}

vec3 ifBetween(float val, float min, float max, vec3 ifTrue, vec3 ifFalse) {
    float valGeMin = step(min, val);
    float valLeMax = step(val, max);

    float cond = valGeMin * valLeMax;

    return ifTrue * cond + ifFalse * (1 - cond);
}

vec3 exposure(vec3 color) {
    color = 1 - exp(-color * intensity);

    return pow(
        color,
        vec3(invGamma)
    );
}

float exposure(float val) {
    return 1 - exp(-val);
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

vec3 crosstalk(vec3 tonemapped) {
    float tonemappedMax = max(tonemapped.r, max(tonemapped.g, tonemapped.b));
    vec3 ratio = tonemapped / tonemappedMax;
    tonemappedMax = min(tonemappedMax, 1.0);

    ratio = pow(ratio, vec3(a_Saturation / a_CrosstalkSaturation));
    ratio = mix(ratio, vec3(1.0), pow(tonemappedMax, a_InvCrosstalkAmt));
    ratio = pow(ratio, vec3(a_CrosstalkSaturation));

    return ratio * tonemappedMax;
}

vec3 crosstalkLum(vec3 tonemapped) {
    float tonemappedMax = luminance(tonemapped);
    vec3 ratio = tonemapped / tonemappedMax;
    tonemappedMax = min(tonemappedMax, 1.0);

    ratio = pow(ratio, vec3(a_Saturation / a_CrosstalkSaturation));
    ratio = mix(ratio, vec3(1.0), pow(tonemappedMax, a_InvCrosstalkAmt));
    ratio = pow(ratio, vec3(a_CrosstalkSaturation));

    return ratio * tonemappedMax;
}

void main() {
    vec3 diffuse = texture(sampler2D(t_Diffuse, s_Color), gl_FragCoord.xy * a_InvResolution).rgb * intensity;
    vec3 bloom = texture(sampler2D(t_Bloom, s_Color), gl_FragCoord.xy * a_InvResolution).rgb;

    diffuse = bloom * a_BloomInfluence + diffuse;

    bool tonemapping = (a_TonemappingBitmap & 0x1) != 0;
    bool xyySpaceAces = (a_TonemappingBitmap & 0x2) != 0;
    bool crosstalkEnabled = (a_TonemappingBitmap & 0x4) != 0;
    bool xyySpaceCrosstalk = (a_TonemappingBitmap & 0x8) != 0;

    vec3 final;
    if (tonemapping) {
        // We want to allow RGB>1 but we also want to fix degenerate/non-finite values
        vec3 tonemapped = clamp(
            xyySpaceAces ? acesLum(diffuse) : aces(diffuse),
            vec3(0.0),
            vec3(100.0)
        );

        if (crosstalkEnabled) {
            final = xyySpaceCrosstalk ? crosstalkLum(tonemapped) : crosstalk(tonemapped);
        } else {
            final = tonemapped;
        }
    } else {
        final = diffuse;
    }

    outColor = vec4(pow(final, vec3(invGamma)), 1.0);
}
