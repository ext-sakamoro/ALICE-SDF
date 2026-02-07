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

// Rounded Cone: smooth-capped cone along Y-axis
float sdRoundedCone(float3 p, float r1, float r2, float h)
{
    float qx = length(p.xz);
    float qy = p.y + h;
    float ht = h * 2.0;
    float b = (r1 - r2) / ht;
    float a = sqrt(1.0 - b * b);
    float k = qx * (-b) + qy * a;
    if (k < 0.0) return length(float2(qx, qy)) - r1;
    if (k > a * ht) return length(float2(qx, qy - ht)) - r2;
    return qx * a + qy * b - r1;
}

// Pyramid: 4-sided pyramid along Y-axis (unit base)
float sdPyramid(float3 p, float h)
{
    float ht = h * 2.0;
    float m2 = ht * ht + 0.25;
    float py = p.y + h;
    float px = abs(p.x);
    float pz = abs(p.z);
    if (pz > px) { float tmp = px; px = pz; pz = tmp; }
    px -= 0.5;
    pz -= 0.5;
    float qx = pz;
    float qy = ht * py - 0.5 * px;
    float qz = ht * px + 0.5 * py;
    float s = max(-qx, 0.0);
    float t = clamp((qy - 0.5 * pz) / (m2 + 0.25), 0.0, 1.0);
    float aa = m2 * (qx + s) * (qx + s) + qy * qy;
    float bb = m2 * (qx + 0.5 * t) * (qx + 0.5 * t) + (qy - m2 * t) * (qy - m2 * t);
    float d2 = (min(-qx * m2 - qy * 0.5, qy) > 0.0) ? 0.0 : min(aa, bb);
    return sqrt((d2 + qz * qz) / m2) * sign(max(qz, -py));
}

// Octahedron: regular octahedron centered at origin
float sdOctahedron(float3 p, float s)
{
    float3 ap = abs(p);
    float m = ap.x + ap.y + ap.z - s;
    float3 q;
    if (3.0 * ap.x < m) q = ap;
    else if (3.0 * ap.y < m) q = float3(ap.y, ap.z, ap.x);
    else if (3.0 * ap.z < m) q = float3(ap.z, ap.x, ap.y);
    else return m * 0.57735027;
    float kk = clamp(0.5 * (q.z - q.y + s), 0.0, s);
    return length(float3(q.x, q.y - s + kk, q.z - kk));
}

// Link: chain link shape (torus stretched along Y)
float sdLink(float3 p, float le, float r1, float r2)
{
    float qx = p.x;
    float qy = max(abs(p.y) - le, 0.0);
    float qz = p.z;
    float xy_len = sqrt(qx * qx + qy * qy) - r1;
    return sqrt(xy_len * xy_len + qz * qz) - r2;
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
// Guarded: k <= 0 falls back to hard union (no division by zero)
float opSmoothUnion(float d1, float d2, float k)
{
    if (k < 0.0001) return min(d1, d2);
    float inv_k = 1.0 / k;
    float h = max(k - abs(d1 - d2), 0.0) * inv_k;
    return min(d1, d2) - h * h * k * 0.25;
}

// Smooth Intersection
// Guarded: k <= 0 falls back to hard intersection
float opSmoothIntersection(float d1, float d2, float k)
{
    if (k < 0.0001) return max(d1, d2);
    float inv_k = 1.0 / k;
    float h = max(k - abs(d1 - d2), 0.0) * inv_k;
    return max(d1, d2) + h * h * k * 0.25;
}

// Smooth Subtraction
// Guarded: k <= 0 falls back to hard subtraction
float opSmoothSubtraction(float d1, float d2, float k)
{
    if (k < 0.0001) return max(d1, -d2);
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

#endif // ALICE_SDF_INCLUDE
