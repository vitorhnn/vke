float remap(float value, float fromLow, float fromHigh, float toLow, float toHigh) {
    return toLow + (value - fromLow) * (toHigh - toLow) / (fromHigh - fromLow);
}

ivec3 getId() {
    return ivec3(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y, gl_GlobalInvocationID.z);
}

uint currPosFromCord(uint x, uint y, uint z) {
    return x + (gl_NumWorkGroups.y * gl_WorkGroupSize.y * y) + (gl_NumWorkGroups.y * gl_WorkGroupSize.y * gl_NumWorkGroups.z * gl_WorkGroupSize.z * z);
}

float smoothMin(float a, float b, float k) {
    float h = max(k-abs(a-b), 0) / k;
    return min(a, b) - h*h*h*k*1/6.0;
}