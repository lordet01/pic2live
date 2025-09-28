from __future__ import annotations

INK_FRAGMENT_GLSL = """
#version 330 core
uniform sampler2D u_base;
uniform vec2 u_resolution;
uniform vec2 u_dir;
uniform float u_strength;
uniform float u_noise_amt;
uniform vec2 u_pos;   // hand position in UV with origin bottom-left
uniform int u_mode;   // 1..9 effect mode selector
in vec2 v_uv;
out vec4 fragColor;

float hash(vec2 p){
    return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453123);
}

vec3 effect_directional_smear(vec2 uv, vec3 src, vec2 dir, vec2 px, float strength, float noise){
    vec3 acc = src * 0.05;
    for(int i=1;i<=12;i++){
        float t = float(i)/8.0;
        vec2 offset = dir * t * 36.0 * px * strength;
        float n = hash(uv*vec2(123.4, 345.6) + float(i));
        vec2 nuv = uv + offset + (n-0.5) * px * noise;
        acc += texture(u_base, nuv).rgb * (0.18 + 0.12*t);
    }
    return acc;
}

vec3 effect_perp_smear(vec2 uv, vec3 src, vec2 dir, vec2 px, float strength, float noise){
    vec2 pdir = vec2(-dir.y, dir.x);
    vec3 acc = src * 0.05;
    for(int i=1;i<=12;i++){
        float t = float(i)/8.0;
        vec2 offset = pdir * t * 28.0 * px * strength;
        float n = hash(uv*vec2(331.1, 97.3) + float(i));
        vec2 nuv = uv + offset + (n-0.5) * px * noise;
        acc += texture(u_base, nuv).rgb * (0.16 + 0.10*t);
    }
    return acc;
}

vec3 effect_radial(vec2 uv, vec3 src, vec2 center, vec2 px, float strength, float noise){
    vec2 d = uv - center;
    float r = length(d) + 1e-5;
    vec2 dir = normalize(d);
    vec3 acc = src * 0.05;
    for(int i=1;i<=12;i++){
        float t = float(i)/12.0;
        vec2 offset = dir * t * r * 0.8 * strength;
        float n = hash(uv*vec2(17.1, 61.7) + float(i));
        vec2 nuv = uv + offset + (n-0.5) * px * noise;
        acc += texture(u_base, nuv).rgb * (0.14 + 0.08*t);
    }
    return acc;
}

vec3 effect_zoom(vec2 uv, vec3 src, vec2 center, float strength){
    vec3 acc = src * 0.05;
    for(int i=1;i<=12;i++){
        float t = float(i)/12.0;
        vec2 nuv = mix(uv, center, t * strength);
        acc += texture(u_base, nuv).rgb * (0.14 + 0.08*t);
    }
    return acc;
}

vec3 effect_chromatic(vec2 uv, vec2 dir, vec2 px, float strength){
    float s = strength * 24.0;
    vec2 o = dir * s * px;
    float r = texture(u_base, uv + o).r;
    float g = texture(u_base, uv).g;
    float b = texture(u_base, uv - o).b;
    return vec3(r,g,b);
}

vec3 effect_threshold_smear(vec2 uv, vec2 dir, vec2 px, float strength, float noise){
    vec3 c = texture(u_base, uv).rgb;
    float l = dot(c, vec3(0.299,0.587,0.114));
    float th = 0.5 - 0.3*strength;
    float ink = step(th, l);
    vec3 acc = c * (0.2 + 0.6*ink);
    for(int i=1;i<=8;i++){
        float t = float(i)/8.0;
        vec2 nuv = uv + dir * t * 24.0 * px * strength + (hash(uv*vec2(71.1,19.7)+float(i))-0.5)*px*noise;
        acc += texture(u_base, nuv).rgb * (0.08 + 0.10*t) * ink;
    }
    return acc;
}

void main(){
    vec2 uv = v_uv;
    vec4 baseCol = texture(u_base, uv);
    vec3 src = baseCol.rgb;

    vec2 dir = normalize(u_dir + vec2(1e-5));
    float strength = clamp(u_strength, 0.0, 1.0);
    vec2 px = 1.0 / max(u_resolution, vec2(1.0));
    vec2 center = u_pos;

    vec3 eff;
    if(u_mode == 1){
        eff = effect_directional_smear(uv, src, dir, px, strength, u_noise_amt);
    } else if(u_mode == 2){
        eff = effect_perp_smear(uv, src, dir, px, strength, u_noise_amt);
    } else if(u_mode == 3){
        eff = effect_radial(uv, src, center, px, strength, u_noise_amt);
    } else if(u_mode == 4){
        eff = effect_zoom(uv, src, center, strength);
    } else if(u_mode == 5){
        eff = effect_chromatic(uv, dir, px, strength);
    } else if(u_mode == 6){
        eff = mix(src, effect_directional_smear(uv, src, dir, px, strength*0.6, u_noise_amt*0.5), 0.8);
    } else if(u_mode == 7){
        eff = effect_threshold_smear(uv, dir, px, strength, u_noise_amt);
    } else if(u_mode == 8){
        eff = mix(effect_perp_smear(uv, src, dir, px, strength, u_noise_amt), effect_radial(uv, src, center, px, strength*0.8, u_noise_amt*0.5), 0.5);
    } else if(u_mode == 9){
        vec2 n = vec2(hash(uv*vec2(11.1, 7.7)), hash(uv*vec2(5.3, 17.9))) - 0.5;
        eff = texture(u_base, uv + n * px * (8.0 + 40.0*strength)).rgb;
    } else {
        eff = src;
    }

    // final mix
    vec3 outCol = mix(src, eff, clamp(strength * 1.2, 0.0, 1.0));
    fragColor = vec4(outCol, 1.0);
}
"""

INK_VERTEX_GLSL = """
#version 330 core
layout (location = 0) in vec2 in_pos;
layout (location = 1) in vec2 in_uv;
out vec2 v_uv;
void main(){
    // Flip Y because OpenCV image origin is top-left, OpenGL UV origin is bottom-left
    v_uv = vec2(in_uv.x, 1.0 - in_uv.y);
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


