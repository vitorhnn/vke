#version 460
#include "descriptor_sets.inc.glsl"

layout(location = 0) in vec2 uv;
layout(location = 1) in vec3 worldSpacePos;
layout(location = 2) in vec3 tangentLightDir;
layout(location = 3) in vec3 tangentViewDir;
layout(location = 4) in float lightDistance;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

float maxDot(vec3 a, vec3 b) {
    return max(dot(a, b), 0.0);
}

float distributionGGX(vec3 normal, vec3 halfway, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float ndoth = maxDot(normal, halfway);
    float ndoth2 = ndoth * ndoth;
    float x = ndoth2 * (a2 - 1.0) + 1.0;
    x *= x;

    return a2 / PI * x;
}

float geometrySchlickGGX(vec3 normal, vec3 viewDir, float roughness) {
    float r = roughness + 1.0;
    float k = r * r / 8.0;
    float ndotv = maxDot(normal, viewDir);

    return ndotv / (ndotv * (1.0 - k) + k);
}

float geometrySmith(vec3 normal, vec3 viewDir, vec3 lightDir, float roughness) {
    float ggx1 = geometrySchlickGGX(normal, viewDir, roughness);
    float ggx2 = geometrySchlickGGX(normal, lightDir, roughness);
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

const vec3 lightColor = vec3(1000.0);

bool trace_shadow_rays(vec3 lightOrigin, vec3 pos) {
    const vec3 dir = lightOrigin - pos;
    const float tmin = 0.01, tmax = length(dir);
    rayQueryEXT query;

    rayQueryInitializeEXT(query, tlas, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, pos, tmin, dir, tmax);
    rayQueryProceedEXT(query);

    if (rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionNoneEXT)
        return true;
    return false;
}

void main() {
    vec3 normal = constants.normalIndex != 999 ?
        texture(sampler2D(textureHeap[constants.normalIndex], stdSampler), uv).rgb : vec3(0.5, 0.5, 1.0);
    vec3 metallicRoughness = constants.metallicIndex != 999 ?
            texture(sampler2D(textureHeap[constants.metallicIndex], stdSampler), uv).rgb : vec3(0.0);
    vec3 albedo = texture(sampler2D(textureHeap[constants.albedoIndex], stdSampler), uv).rgb;

    float metallic = metallicRoughness.b;
    float roughness = metallicRoughness.g;

    normal = normalize(normal * 2.0 - 1.0);
    vec3 lightDir = normalize(tangentLightDir);
    vec3 viewDir = normalize(tangentViewDir);
    vec3 halfway = normalize(viewDir + lightDir);
    float dist = lightDistance;
    float attenuation = 1.0 / (dist * dist);
    vec3 radiance = lightColor * attenuation;

    float ndf = distributionGGX(normal, halfway, roughness);
    float g = geometrySmith(normal, viewDir, lightDir, roughness);
    vec3 f = fresnelSchlick(maxDot(halfway, viewDir), mix(vec3(0.04), albedo, metallic));

    vec3 kD = vec3(1.0) - f;
    kD *= 1.0 - metallic;
    vec3 specular = (ndf * g * f) / (4.0 * maxDot(normal, viewDir) * maxDot(normal, lightDir) + 0.0001);
    vec3 pbr = (kD * albedo / PI + specular) * radiance * maxDot(normal, lightDir);
    vec3 ambient = vec3(0.05) * albedo;

    bool res = trace_shadow_rays(lightPos, worldSpacePos);

    float mod = res ? 0.1 : 1.0;

    pbr *= mod;

    outColor = vec4(ambient + pbr, 1.0);
}
