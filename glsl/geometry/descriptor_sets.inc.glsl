#extension GL_EXT_ray_query : enable
layout(push_constant, std430) uniform PushConstants {
    mat4 model;
    mat4 invModelT;
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

const vec3 lightPos = vec3(0.0, 12.0, 0.0);

layout(set = 1, binding = 0) uniform sampler stdSampler;
layout(set = 1, binding = 1) uniform accelerationStructureEXT tlas;
// This is unsized, we just specify a size here to keep glslc happy
// I should switch to slang or something. glslc is crusty.
layout(set = 1, binding = 2) uniform texture2D textureHeap[4096];
