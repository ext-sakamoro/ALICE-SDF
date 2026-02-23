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

// Cone: capped cone along Y-axis (base at -halfHeight, tip at +halfHeight)
float sdCone(float3 p, float radius, float halfHeight)
{
    float qx = length(p.xz);
    float qy = p.y;
    float h = halfHeight;
    float k2x = -radius;
    float k2y = 2.0 * h;
    float ca_r = (qy < 0.0) ? radius : 0.0;
    float ca_x = qx - min(qx, ca_r);
    float ca_y = abs(qy) - h;
    float diff_x = -qx;
    float diff_y = h - qy;
    float t = clamp((diff_x * k2x + diff_y * k2y) / (k2x * k2x + k2y * k2y), 0.0, 1.0);
    float cb_x = qx + k2x * t;
    float cb_y = qy - h + k2y * t;
    float s = (cb_x < 0.0 && ca_y < 0.0) ? -1.0 : 1.0;
    float d2 = min(ca_x * ca_x + ca_y * ca_y, cb_x * cb_x + cb_y * cb_y);
    return s * sqrt(d2);
}

// Ellipsoid: approximate SDF (Inigo Quilez formula)
float sdEllipsoid(float3 p, float3 radii)
{
    float3 sr = max(radii, 1e-10);
    float k0 = length(p / sr);
    float k1 = length(p / (sr * sr));
    if (k1 < 1e-10) return -min(sr.x, min(sr.y, sr.z));
    return k0 * (k0 - 1.0) / k1;
}

// Gyroid: triply-periodic minimal surface (TPMS)
float sdGyroid(float3 p, float scale, float thickness)
{
    float3 sp = p * scale;
    float d = sin(sp.x) * cos(sp.y) + sin(sp.y) * cos(sp.z) + sin(sp.z) * cos(sp.x);
    return abs(d) / scale - thickness;
}

// Schwarz P: triply-periodic minimal surface (TPMS)
float sdSchwarzP(float3 p, float scale, float thickness)
{
    float3 sp = p * scale;
    float d = cos(sp.x) + cos(sp.y) + cos(sp.z);
    return abs(d) / scale - thickness;
}

// BoxFrame: wireframe box (only edges)
float sdBoxFrame(float3 p, float3 halfExtents, float edge)
{
    float3 pp = abs(p) - halfExtents;
    float3 q = abs(pp + edge) - edge;
    float d1 = length(max(float3(pp.x, q.y, q.z), 0.0))
             + min(max(pp.x, max(q.y, q.z)), 0.0);
    float d2 = length(max(float3(q.x, pp.y, q.z), 0.0))
             + min(max(q.x, max(pp.y, q.z)), 0.0);
    float d3 = length(max(float3(q.x, q.y, pp.z), 0.0))
             + min(max(q.x, max(q.y, pp.z)), 0.0);
    return min(d1, min(d2, d3));
}

// CappedTorus: torus arc in XZ plane
float sdCappedTorus(float3 p, float majorRadius, float minorRadius, float capAngle)
{
    float sc_s = sin(capAngle);
    float sc_c = cos(capAngle);
    float px = abs(p.x);
    float k = (sc_c * px > sc_s * p.y)
        ? (px * sc_s + p.y * sc_c)
        : sqrt(px * px + p.y * p.y);
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z
         + majorRadius * majorRadius - 2.0 * majorRadius * k)
         - minorRadius;
}

// Heart: 3D heart shape (revolution of 2D contour)
float sdHeart(float3 p, float size)
{
    float3 sp = p / size;
    float2 q = float2(length(sp.xz), sp.y);
    q.y -= 0.5;
    float qx = abs(q.x);
    if (qx + q.y > 1.0)
    {
        float dx = qx - 0.25;
        float dy = q.y - 0.75;
        return (sqrt(dx * dx + dy * dy) - 1.41421356 * 0.25) * size;
    }
    float d1 = qx * qx + (q.y - 1.0) * (q.y - 1.0);
    float t = max(qx + q.y, 0.0) * 0.5;
    float d2 = (qx - t) * (qx - t) + (q.y - t) * (q.y - t);
    float sg = (qx > q.y) ? 1.0 : -1.0;
    return sqrt(min(d1, d2)) * sg * size;
}

// HexPrism: hexagonal prism centered at origin
float sdHexPrism(float3 p, float hexRadius, float halfHeight)
{
    float kx = -0.8660254;
    float ky = 0.5;
    float kz = 0.57735027;
    float px = abs(p.x);
    float py = abs(p.y);
    float pz = abs(p.z);
    float dot_kxy = kx * px + ky * py;
    float reflect = 2.0 * min(dot_kxy, 0.0);
    px -= reflect * kx;
    py -= reflect * ky;
    float clamped_x = clamp(px, -kz * hexRadius, kz * hexRadius);
    float dx = px - clamped_x;
    float dy = py - hexRadius;
    float d_xy = sqrt(dx * dx + dy * dy) * sign(dy);
    float d_z = pz - halfHeight;
    return min(max(d_xy, d_z), 0.0) + length(max(float2(d_xy, d_z), 0.0));
}

// RoundedBox: box with rounded edges
float sdRoundedBox(float3 p, float3 halfExtents, float roundRadius)
{
    float3 q = abs(p) - halfExtents;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - roundRadius;
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

// XOR (Symmetric Difference): inside exactly one shape
float opXor(float d1, float d2)
{
    return max(min(d1, d2), -max(d1, d2));
}

// Morph: linear interpolation between two SDFs
// t=0 gives d1, t=1 gives d2
float opMorph(float d1, float d2, float t)
{
    return d1 * (1.0 - t) + d2 * t;
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

// Quaternion rotation
float3 quatRotate(float3 v, float4 q)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

#endif // ALICE_SDF_INCLUDE
