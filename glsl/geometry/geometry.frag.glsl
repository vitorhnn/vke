#version 450
#include "descriptor_sets.inc.glsl"

layout(location = 0) in vec2 uv;
layout(location = 1) in vec3 worldSpacePos;
layout(location = 2) in vec3 normal;

layout(location = 0) out vec4 outColor;

const vec3 lightPos = vec3(1.2, 1.0, 2.0);
const vec3 color = vec3(1.0, 0.0, 0.0);

void main() {
    vec3 normal = normalize(normal);
    vec3 lightDir = normalize(lightPos - worldSpacePos);
    float diffPower = max(dot(normal, lightDir), 0.0);
    vec4 albedo = texture(sampler2D(textureHeap[constants.albedoIndex], stdSampler), uv);
    outColor = diffPower * albedo;
}
