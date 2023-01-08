#version 450

layout(push_constant) uniform PushConstants {
    mat4 model;
} constants;

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 projection;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUv;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec4 inTangent;

layout(location = 0) out vec2 uv;
layout(location = 1) out vec3 worldSpacePos;
layout(location = 2) out vec3 normal;


void main() {
    gl_Position = ubo.projection * ubo.view * constants.model * vec4(inPosition, 1.0);
    worldSpacePos = vec3(constants.model * vec4(inPosition, 1.0));

    uv = inUv;
    normal = inNormal;
}
