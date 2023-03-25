#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_mesh_shader : require

layout(local_size_x=8) in;

// TODO: revise for amd
layout(triangles, max_vertices=64, max_primitives=126) out;

#include "tables.glsl"

layout(location = 0) out PerVertexData 
{ 
    vec4 color;
    vec4 normal;
} v_out[];

layout(set = 0, binding = 0) uniform Params {
    mat4 view;
    mat4 projection;
    float u_surfaceLevel;
    float u_smooth;
    uint u_linear;
    uint u_pointsCount;
    vec3 u_worldBounds;
    vec4 u_points[10];
};

layout(push_constant) uniform PushConstants {
    mat4 model;
} constants;

layout(std430, set = 0, binding = 1) buffer Triangles { Triangle Data[]; };
layout(std430, set = 0, binding = 2) readonly buffer TriTable  { int triTable[]; };
layout(std430, set = 0, binding = 3) buffer IndirectDrawBuffer {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

float remap(float value, float fromLow, float fromHigh, float toLow, float toHigh) {
    return toLow + (value - fromLow) * (toHigh - toLow) / (fromHigh - fromLow);
}

ivec3 getId() {
    return ivec3(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y, gl_GlobalInvocationID.z);
}

float smoothMin(float a, float b, float k) {
    float h = max(k-abs(a-b), 0) / k;
    return min(a, b) - h*h*h*k*1/6.0;
}

/** Obtém o vetor normal do triangulo calculando o produto cruzado de seus vértices **/
vec4 getNormalVector(vec4 v0, vec4 v1, vec4 v2) 
{
    vec3 a = vec3(v0.x, v0.y, v0.z);
    vec3 b = vec3(v1.x, v1.y, v1.z);
    vec3 c = vec3(v2.x, v2.y, v2.z);

    return vec4(normalize(cross(b - a, c - a)), 1.0);
}

/*
* Obtém o ponto intermediário entre dois pontos.
* Caso o uniform u_linear seja != 0, o ponto intermediário é calculado de acordo com o valor dos pontos.
* Caso contrário o ponto intermediário é apenas o ponto médio entre os dois pontos.
*/
vec4 interpolate(vec4 a, vec4 b) 
{
    if (u_linear != 0) 
    {
        vec3 aPos = vec3(a.x, a.y, a.z);
        vec3 bPos = vec3(b.x, b.y, b.z);

        float t = (u_surfaceLevel - a.w) / (b.w - a.w);
        vec3 i = aPos + t * (bPos - aPos); 
        return vec4(i.x, i.y, i.z, 1.0f);
    } else {
        return (a + b) / 2.0f;
    }
}

/*
* Obtém a posição no vetor de triangularização utilizando cordenadas bidimencionais
*/
int getTriTableIndex(int _x, int _y) 
{
    return _y + (_x * 16);
}

/*
* Obtém o valor de cada ponto no espaço tridimencional
*/
vec4 getPoint(uint _x, uint _y, uint _z)
{
    float x = remap(_x, 0, u_pointsCount.x - 1, -u_worldBounds.x/2.0, u_worldBounds.x/2.0);
    float y = remap(_y, 0, u_pointsCount.y - 1, -u_worldBounds.y/2.0, u_worldBounds.y/2.0);
    float z = remap(_z, 0, u_pointsCount.z - 1, -u_worldBounds.z/2.0, u_worldBounds.z/2.0);

    float value = 2.5;
    for (int i = 0; i < u_spheresCount; i++) 
    {
        float distanteToPoint = distance(vec3(x, y, z), vec3(u_spheres[i].x, u_spheres[i].y, u_spheres[i].z));
        value = smoothMin(remap(abs(distanteToPoint), 0, 5.5, -1, 1), value, u_smooth);
    }

    return vec4(x, y, z, value);
}

uvec3 _getId() 
{
    uint id = gl_GlobalInvocationID.x;
    return uvec3(id / (u_pointsCount.x * u_pointsCount.x), (id / u_pointsCount.y) % u_pointsCount.y, id % u_pointsCount.z);
}

void main()
{
    uvec3 id = _getId();
    uint localId = gl_LocalInvocationID.x;
    uint globalId = gl_WorkGroupID.x;

    // Somente thread 0 inicializa o contador de primitivos 
    if (localId == 0) 
    {
        gl_PrimitiveCountNV = 0;
        triCountBuff[globalId] = 0;
    }

    if (
        id.x >= u_pointsCount.x - 1 || 
        id.y >= u_pointsCount.y - 1 || 
        id.z >= u_pointsCount.z - 1
    ) return;

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

    if (index > 0 && index < 256) 
    {
        for (int i = 0; triTable[getTriTableIndex(index, i)] != -1; i+=3) 
        {
            uint offset = atomicAdd(triCountBuff[globalId], 1) * 3;

            int idx = triTable[getTriTableIndex(index, i+0)];
            int idy = triTable[getTriTableIndex(index, i+1)];
            int idz = triTable[getTriTableIndex(index, i+2)];

            vec4 v0 = interpolate(corners[cornerListA[idx]], corners[cornerListB[idx]]);
            vec4 v1 = interpolate(corners[cornerListA[idy]], corners[cornerListB[idy]]);
            vec4 v2 = interpolate(corners[cornerListA[idz]], corners[cornerListB[idz]]);
            vec4 norm = getNormalVector(v0, v1, v2);

            gl_MeshVerticesNV[offset + 0].gl_Position = MVP * v0;
            gl_MeshVerticesNV[offset + 1].gl_Position = MVP * v1;
            gl_MeshVerticesNV[offset + 2].gl_Position = MVP * v2;

            v_out[offset + 0].normal = norm;
            v_out[offset + 1].normal = norm;
            v_out[offset + 2].normal = norm;

            v_out[offset + 0].color = vec4(1.0, 1.0, 1.0, 1.0);
            v_out[offset + 1].color = vec4(1.0, 1.0, 1.0, 1.0);
            v_out[offset + 2].color = vec4(1.0, 1.0, 1.0, 1.0);

            gl_PrimitiveIndicesNV[offset + 0] = offset + 0;
            gl_PrimitiveIndicesNV[offset + 1] = offset + 1;
            gl_PrimitiveIndicesNV[offset + 2] = offset + 2;
        }
    }
    

    barrier();
    if (localId == 0) {
        gl_PrimitiveCountNV = triCountBuff[globalId];
        atomicCounterAdd(trianglesCount, triCountBuff[globalId]);
    }
}
