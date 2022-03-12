#version 450

layout(set = 0, binding = 0) uniform UBO {
    mat4 modelView;
    mat4 projection;
} ubo;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUv;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 uv;

void main() {
    gl_Position = ubo.projection * ubo.modelView * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
    uv = inUv;
}
