#version 450
#include "descriptor_sets.inc.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUv;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec4 inTangent;

layout(location = 0) out vec2 uv;
layout(location = 1) out vec3 worldSpacePos;
layout(location = 2) out vec3 normal;
layout(location = 3) out vec3 tangentLightPos;
layout(location = 4) out vec3 tangentCamPos;
layout(location = 5) out vec3 tangentPos;

void main() {
    gl_Position = ubo.projection * ubo.view * constants.model * vec4(inPosition, 1.0);
    worldSpacePos = vec3(constants.model * vec4(inPosition, 1.0));

    mat3 invModel = mat3(constants.invModel);
    vec3 T = normalize(invModel * vec3(inTangent));
    vec3 N = normalize(invModel * inNormal);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);

    mat3 TBN = transpose(mat3(T, B, N));

    tangentLightPos = TBN * lightPos;
    tangentCamPos = TBN * ubo.camPos;
    tangentPos = TBN * worldSpacePos;

    uv = inUv;
    normal = inNormal;
}
