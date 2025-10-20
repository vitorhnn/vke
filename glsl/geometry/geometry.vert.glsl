#version 450
#include "descriptor_sets.inc.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUv;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec4 inTangent;

layout(location = 0) out vec2 uv;
layout(location = 1) out vec3 worldSpacePos;
layout(location = 2) out vec3 tangentLightDir;
layout(location = 3) out vec3 tangentViewDir;
layout(location = 4) out float lightDistance;


void main() {
    worldSpacePos = vec3(constants.model * vec4(inPosition, 1.0));
    gl_Position = ubo.projection * ubo.view * vec4(worldSpacePos, 1.0);

    mat3 invModelT = mat3(constants.invModelT);
    vec3 T = normalize(mat3(constants.model) * vec3(inTangent));
    vec3 N = normalize(invModelT * inNormal);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T) * inTangent.w;

    mat3 TBN = transpose(mat3(T, B, N));

    vec3 lightDir = lightPos - worldSpacePos;
    lightDistance = length(lightDir);
    lightDir = normalize(lightDir);
    vec3 viewDir = normalize(ubo.camPos - worldSpacePos);

    tangentLightDir = TBN * lightDir;
    tangentViewDir = TBN * viewDir;

    lightDistance = length(lightPos - worldSpacePos);

    uv = inUv;
}
