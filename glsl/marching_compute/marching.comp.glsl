#version 450
#extension GL_GOOGLE_include_directive : enable
#include "utils.glsl"
#include "tables.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

struct Triangle {
    vec4 v0;
    vec4 norm0;
    vec4 v1;
    vec4 norm1;
    vec4 v2;
    vec4 norm2;
};

layout(set = 0, binding = 0) uniform Params {
    float u_surfaceLevel;
    float u_smooth;
    uint u_linear;
    uint u_pointsCount;
    vec3 u_worldBounds;
    vec4 u_points[10];
};

layout(std430, set = 0, binding = 1) buffer Triangles { Triangle Data[]; };
layout(std430, set = 0, binding = 2) readonly buffer TriTable  { int triTable[]; };
layout(std430, set = 0, binding = 3) buffer IndirectDrawBuffer {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

vec4 getNormalVector(Triangle t) {
    vec3 a = vec3(t.v0.x, t.v0.y, t.v0.z);
    vec3 b = vec3(t.v1.x, t.v1.y, t.v1.z);
    vec3 c = vec3(t.v2.x, t.v2.y, t.v2.z);

    return vec4(normalize(cross(b - a, c - a)), 1.0);
}

vec4 interpolate(vec4 a, vec4 b) {
    if (u_linear != 0) {
        vec3 aPos = vec3(a.x, a.y, a.z);
        vec3 bPos = vec3(b.x, b.y, b.z);

        float t = (u_surfaceLevel - a.w) / (b.w - a.w);
        vec3 i = aPos + t * (bPos - aPos); 
        return vec4(i.x, i.y, i.z, 1.0f);
    } else {
        return (a + b) / 2.0f;
    }
}

int getTriTableIndex(int _x, int _y) {
    return _y + (_x * 16);
}

vec4 getPoint(uint _x, uint _y, uint _z){
    float x = remap(_x, 0, gl_NumWorkGroups.x * gl_WorkGroupSize.x - 1, -u_worldBounds.x/2.0f, u_worldBounds.x/2.0f);
    float y = remap(_y, 0, gl_NumWorkGroups.y * gl_WorkGroupSize.y - 1, -u_worldBounds.y/2.0f, u_worldBounds.y/2.0f);
    float z = remap(_z, 0, gl_NumWorkGroups.z * gl_WorkGroupSize.z - 1, -u_worldBounds.z/2.0f, u_worldBounds.z/2.0f);

    float value = 2.5;
    for (int i = 0; i < u_pointsCount; i++) {
        float distanceToPoint = distance(vec3(x, y, z), vec3(u_points[i].x, u_points[i].y, u_points[i].z));
        value = smoothMin(remap(abs(distanceToPoint), 0, 5.5, -1, 1), value, u_smooth);
    }

    return vec4(x, y, z, value);
}


void main() {
    ivec3 id = getId();

    if (
        id.x >= gl_NumWorkGroups.x * gl_WorkGroupSize.x - 1 || 
        id.y >= gl_NumWorkGroups.y * gl_WorkGroupSize.y - 1 || 
        id.z >= gl_NumWorkGroups.z * gl_WorkGroupSize.z - 1
    ) return;

    uint currPos = currPosFromCord(id.x, id.y, id.z);

    vec4 corners[8] = {
        getPoint(id.x    , id.y    , id.z    ),
        getPoint(id.x + 1, id.y    , id.z    ),
        getPoint(id.x + 1, id.y    , id.z + 1),
        getPoint(id.x    , id.y    , id.z + 1),
        getPoint(id.x    , id.y + 1, id.z    ),
        getPoint(id.x + 1, id.y + 1, id.z    ),
        getPoint(id.x + 1, id.y + 1, id.z + 1),
        getPoint(id.x    , id.y + 1, id.z + 1)
    };

    int index = 0;
    if (corners[0].w < u_surfaceLevel) index |=   1;
    if (corners[1].w < u_surfaceLevel) index |=   2;
    if (corners[2].w < u_surfaceLevel) index |=   4;
    if (corners[3].w < u_surfaceLevel) index |=   8;
    if (corners[4].w < u_surfaceLevel) index |=  16;
    if (corners[5].w < u_surfaceLevel) index |=  32;
    if (corners[6].w < u_surfaceLevel) index |=  64;
    if (corners[7].w < u_surfaceLevel) index |= 128;

    if (index == 0 || index == 256) return;

    for (int i = 0; triTable[getTriTableIndex(index, i)] != -1; i+=3) {
        int idx = triTable[getTriTableIndex(index, i+0)];
        int idy = triTable[getTriTableIndex(index, i+1)];
        int idz = triTable[getTriTableIndex(index, i+2)];

        Triangle t;
        t.v0 = interpolate(corners[cornerListA[idx]], corners[cornerListB[idx]]);
        t.v1 = interpolate(corners[cornerListA[idy]], corners[cornerListB[idy]]);
        t.v2 = interpolate(corners[cornerListA[idz]], corners[cornerListB[idz]]);
        vec4 normal = getNormalVector(t);
        t.norm0 = normal;
        t.norm1 = normal;
        t.norm2 = normal;

        uint atomicCounter = atomicAdd(vertexCount, 3);
        Data[atomicCounter / 3] = t;
    }
}