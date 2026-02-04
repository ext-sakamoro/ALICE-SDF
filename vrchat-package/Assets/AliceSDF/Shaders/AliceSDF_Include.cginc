// =============================================================================
// AliceSDF_Include.cginc - SDF Primitive & Operation Library for VRChat
// =============================================================================
// All SDF functions ported from ALICE-SDF Rust engine.
// "Don't send meshes. Send the law of shape."
//
// Author: Moroya Sakamoto
// License: ALICE Community License
// =============================================================================

#ifndef ALICE_SDF_INCLUDE
#define ALICE_SDF_INCLUDE

// =============================================================================
// Primitives
// =============================================================================

// Sphere: distance to sphere surface at origin
float sdSphere(float3 p, float radius)
{
    return length(p) - radius;
}

// Box: axis-aligned box at origin
float sdBox(float3 p, float3 halfExtents)
{
    float3 q = abs(p) - halfExtents;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Cylinder: vertical capped cylinder at origin
float sdCylinder(float3 p, float radius, float halfHeight)
{
    float2 d = float2(length(p.xz) - radius, abs(p.y) - halfHeight);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// Torus: torus in XZ plane at origin
float sdTorus(float3 p, float majorRadius, float minorRadius)
{
    float2 q = float2(length(p.xz) - majorRadius, p.y);
    return length(q) - minorRadius;
}

// Plane: infinite plane
float sdPlane(float3 p, float3 normal, float dist)
{
    return dot(p, normal) + dist;
}

// Capsule: line segment with radius
float sdCapsule(float3 p, float3 a, float3 b, float radius)
{
    float3 pa = p - a;
    float3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - radius;
}

// Cone: capped cone along Y-axis (base radius at y=0, tip at y=height)
float sdCone(float3 p, float radius, float height)
{
    float2 q = height * float2(radius / height, -1.0);
    float2 w = float2(length(p.xz), p.y);
    float2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    float2 b = w - q * float2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    float k = sign(q.y);
    float d = min(dot(a, a), dot(b, b));
    float s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
    return sqrt(d) * sign(s);
}

// Ellipsoid: approximate SDF (exact at surface, slight overestimate far away)
float sdEllipsoid(float3 p, float3 radii)
{
    float k0 = length(p / radii);
    float k1 = length(p / (radii * radii));
    return k0 * (k0 - 1.0) / k1;
}

// Hexagonal Prism along Z-axis
float sdHexPrism(float3 p, float halfHeight, float radius)
{
    float3 k = float3(-0.8660254038, 0.5, 0.57735026919);
    p = abs(p);
    p.xy -= 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
    float2 d = float2(
        length(p.xy - float2(clamp(p.x, -k.z * radius, k.z * radius), radius)) * sign(p.y - radius),
        p.z - halfHeight
    );
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// helper for sdTriangle
float _dot2(float3 v) { return dot(v, v); }

// Triangle: exact distance to a 3D triangle (vertices a, b, c)
float sdTriangle(float3 p, float3 a, float3 b, float3 c)
{
    float3 ba = b - a; float3 pa = p - a;
    float3 cb = c - b; float3 pb = p - b;
    float3 ac = a - c; float3 pc = p - c;
    float3 nor = cross(ba, ac);
    return sqrt(
        (sign(dot(cross(ba, nor), pa)) +
         sign(dot(cross(cb, nor), pb)) +
         sign(dot(cross(ac, nor), pc)) < 2.0)
        ?
        min(min(
            _dot2(ba * clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0) - pa),
            _dot2(cb * clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0) - pb)),
            _dot2(ac * clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0) - pc))
        :
        dot(nor, pa) * dot(nor, pa) / dot(nor, nor));
}

// Quadratic Bezier curve with radius (analytical cubic solver)
float sdBezier(float3 pos, float3 A, float3 B, float3 C, float rad)
{
    float3 a = B - A;
    float3 b = A - 2.0 * B + C;
    float3 c = a * 2.0;
    float3 d = A - pos;
    float kk = 1.0 / dot(b, b);
    float kx = kk * dot(a, b);
    float ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
    float kz = kk * dot(d, a);
    float p2 = ky - kx * kx;
    float p3 = p2 * p2 * p2;
    float q2 = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    float h = q2 * q2 + 4.0 * p3;
    float res;
    if (h >= 0.0)
    {
        h = sqrt(h);
        float2 x = (float2(h, -h) - q2) / 2.0;
        float2 uv = sign(x) * pow(abs(x), float2(1.0/3.0, 1.0/3.0));
        float t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        res = length(d + (c + b * t) * t);
    }
    else
    {
        float z = sqrt(-p2);
        float v = acos(q2 / (p2 * z * 2.0)) / 3.0;
        float m = cos(v);
        float n = sin(v) * 1.732050808;
        float3 t = clamp(float3(m + m, -n - m, n - m) * z - kx, 0.0, 1.0);
        res = min(
            length(d + (c + b * t.x) * t.x),
            length(d + (c + b * t.y) * t.y)
        );
    }
    return res - rad;
}

// =============================================================================
// Boolean Operations
// =============================================================================

float opUnion(float d1, float d2)
{
    return min(d1, d2);
}

float opIntersection(float d1, float d2)
{
    return max(d1, d2);
}

float opSubtraction(float d1, float d2)
{
    return max(d1, -d2);
}

// Smooth Union (Deep Fried: division exorcism, pre-computed inv_k)
float opSmoothUnion(float d1, float d2, float k)
{
    float inv_k = 1.0 / k;
    float h = max(k - abs(d1 - d2), 0.0) * inv_k;
    return min(d1, d2) - h * h * k * 0.25;
}

// Smooth Intersection
float opSmoothIntersection(float d1, float d2, float k)
{
    float inv_k = 1.0 / k;
    float h = max(k - abs(d1 - d2), 0.0) * inv_k;
    return max(d1, d2) + h * h * k * 0.25;
}

// Smooth Subtraction
float opSmoothSubtraction(float d1, float d2, float k)
{
    float inv_k = 1.0 / k;
    float h = max(k - abs(d1 + d2), 0.0) * inv_k;
    return max(d1, -d2) + h * h * k * 0.25;
}

// =============================================================================
// Transforms
// =============================================================================

// Infinite repetition (space folding)
float3 opRepeatInfinite(float3 p, float3 spacing)
{
    return fmod(p + spacing * 0.5, spacing) - spacing * 0.5;
}

// Finite repetition
float3 opRepeatFinite(float3 p, float3 spacing, float3 count)
{
    float3 r = clamp(round(p / spacing), -count, count);
    return p - spacing * r;
}

// =============================================================================
// Modifiers
// =============================================================================

// Twist around Y-axis
float3 opTwist(float3 p, float strength)
{
    float angle = strength * p.y;
    float c = cos(angle);
    float s = sin(angle);
    return float3(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}

// Bend along X-axis
float3 opBend(float3 p, float curvature)
{
    float angle = curvature * p.x;
    float c = cos(angle);
    float s = sin(angle);
    return float3(c * p.x + s * p.y, c * p.y - s * p.x, p.z);
}

// Round: soften edges
float opRound(float d, float radius)
{
    return d - radius;
}

// Onion: create hollow shell
float opOnion(float d, float thickness)
{
    return abs(d) - thickness;
}

// Taper: shrink cross-section along Y-axis (factor > 0 = narrower at top)
float3 opTaper(float3 p, float factor)
{
    float s = 1.0 / (1.0 - p.y * factor);
    return float3(p.x * s, p.y, p.z * s);
}

// Displacement: add sin-based noise to distance field
float opDisplacement(float d, float3 p, float strength)
{
    return d + sin(p.x * 5.0) * sin(p.y * 5.0) * sin(p.z * 5.0) * strength;
}

// Symmetry: mirror across axes (mask: 1.0 = mirror, 0.0 = no mirror)
float3 opSymmetry(float3 p, float3 mask)
{
    return lerp(p, abs(p), mask);
}

// Polar repetition around Y-axis
float3 opPolarRepeat(float3 p, float count)
{
    float angle = 6.283185307 / count;
    float a = atan2(p.z, p.x) + angle * 0.5;
    float r = length(p.xz);
    a = fmod(a + 100.0 * angle, angle) - angle * 0.5;
    return float3(r * cos(a), p.y, r * sin(a));
}

// Elongate: stretch SDF along axes (apply to point before SDF evaluation)
float3 opElongate(float3 p, float3 h)
{
    return p - clamp(p, -h, h);
}

// =============================================================================
// Material-aware Boolean Operations (float2: x=distance, y=materialID)
// =============================================================================

float2 opUnionMat(float2 a, float2 b)
{
    return (a.x < b.x) ? a : b;
}

float2 opIntersectionMat(float2 a, float2 b)
{
    return (a.x > b.x) ? a : b;
}

float2 opSubtractionMat(float2 a, float2 b)
{
    return float2(max(a.x, -b.x), a.y);
}

float2 opSmoothUnionMat(float2 a, float2 b, float k)
{
    float inv_k = 1.0 / k;
    float h = max(k - abs(a.x - b.x), 0.0) * inv_k;
    float d = min(a.x, b.x) - h * h * k * 0.25;
    float m = lerp(b.y, a.y, saturate((b.x - a.x) * inv_k * 0.5 + 0.5));
    return float2(d, m);
}

float2 opSmoothIntersectionMat(float2 a, float2 b, float k)
{
    float inv_k = 1.0 / k;
    float h = max(k - abs(a.x - b.x), 0.0) * inv_k;
    float d = max(a.x, b.x) + h * h * k * 0.25;
    float m = lerp(a.y, b.y, saturate((a.x - b.x) * inv_k * 0.5 + 0.5));
    return float2(d, m);
}

float2 opSmoothSubtractionMat(float2 a, float2 b, float k)
{
    float inv_k = 1.0 / k;
    float h = max(k - abs(a.x + b.x), 0.0) * inv_k;
    float d = max(a.x, -b.x) + h * h * k * 0.25;
    return float2(d, a.y);
}

// =============================================================================
// Utility
// =============================================================================

// Normal calculation via central differences
float3 calcNormal(float3 p, float eps)
{
    // Requires: float map(float3 p) to be defined in the including shader
    float3 n;
    n.x = map(p + float3(eps, 0, 0)) - map(p - float3(eps, 0, 0));
    n.y = map(p + float3(0, eps, 0)) - map(p - float3(0, eps, 0));
    n.z = map(p + float3(0, 0, eps)) - map(p - float3(0, 0, eps));
    return normalize(n);
}

// Quaternion rotation
float3 quatRotate(float3 v, float4 q)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// =============================================================================
// Soft Shadow (penumbra estimation via SDF)
// =============================================================================

// Requires: float map(float3 p) to be defined
float aliceSoftShadow(float3 ro, float3 rd, float mint, float maxt, float k)
{
    float res = 1.0;
    float t = mint;
    float ph = 1e20;
    for (int i = 0; i < 64; i++)
    {
        float h = map(ro + rd * t);
        if (h < 0.0001)
            return 0.0;
        float y = h * h / (2.0 * ph);
        float d = sqrt(h * h - y * y);
        res = min(res, k * d / max(0.0, t - y));
        ph = h;
        t += h;
        if (t > maxt) break;
    }
    return saturate(res);
}

// =============================================================================
// PBR Lighting Helpers
// =============================================================================

// GGX Normal Distribution Function
float distributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom + 0.0001);
}

// Schlick-GGX Geometry Function
float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

// Fresnel-Schlick
float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

// Fresnel-Schlick with roughness (for environment/reflection)
float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
    float3 oneMinusR = float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness);
    return F0 + (max(oneMinusR, F0) - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

#endif // ALICE_SDF_INCLUDE
