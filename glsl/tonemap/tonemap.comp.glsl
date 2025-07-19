#version 450

layout (local_size_x = 8, local_size_y = 8) in;

layout (set = 0, binding = 0, rgba16f) readonly uniform image2D inImage;
layout (set = 0, binding = 1) writeonly uniform image2D outImage;

void main() {
    ivec2 inputSize = imageSize(inImage);
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);

    if (gl_GlobalInvocationID.x >= inputSize.x || gl_GlobalInvocationID.y >= inputSize.y) {
        return;
    }

    vec3 color = imageLoad(inImage, coords).xyz;
    vec3 mapped = color / (color + vec3(1.0));

    imageStore(outImage, coords, vec4(mapped, 1.0));
}

