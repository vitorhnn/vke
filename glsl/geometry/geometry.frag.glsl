#version 450
#include "descriptor_sets.inc.glsl"

layout(location = 0) in vec2 uv;
layout(location = 1) in vec3 worldSpacePos;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangentLightPos;
layout(location = 4) in vec3 tangentCamPos;
layout(location = 5) in vec3 tangentPos;

layout(location = 0) out vec4 outColor;

const vec3 color = vec3(1.0, 0.0, 0.0);

void main() {
    vec3 normal = constants.normalIndex != 999 ?
        texture(sampler2D(textureHeap[constants.normalIndex], stdSampler), uv).rgb : vec3(0.0);
    normal = normalize(normal * 2.0 - 1.0);
    vec3 lightDir = normalize(tangentLightPos - tangentPos);

    float dist = length(lightPos - worldSpacePos);
    float attenuation = 1.0 / (dist * dist);

    float diffPower = max(dot(normal, lightDir), 0.0);
    vec3 albedo = texture(sampler2D(textureHeap[constants.albedoIndex], stdSampler), uv).rgb;
    outColor = vec4(diffPower * albedo * attenuation, 1.0);
}
