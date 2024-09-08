layout(push_constant, std430) uniform PushConstants {
    mat4 model;
    mat4 invModel;
    vec4 baseColor;
    uint albedoIndex;
    uint metallicIndex;
    uint normalIndex;
    uint padding;
} constants;

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 projection;
    vec3 camPos;
} ubo;

const vec3 lightPos = vec3(1.2, 1.0, 2.0);

layout(set = 1, binding = 0) uniform sampler stdSampler;
// This is unsized, we just specify a size here to keep glslc happy
// I should switch to slang or something. glslc is crusty.
layout(set = 1, binding = 1) uniform texture2D textureHeap[4096];
