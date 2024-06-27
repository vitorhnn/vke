layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 baseColor;
    uint albedoIndex;
    uint metalicIndex;
    uint normalIndex;
} constants;

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 projection;
} ubo;

layout(set = 1, binding = 0) uniform sampler stdSampler;
// This is unsized, we just specify a size here to keep glslc happy
// I should switch to slang or something. glslc is crusty.
layout(set = 1, binding = 1) uniform texture2D textureHeap[4096];
