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

// InfiniteCylinder: infinite along Y-axis
float sdInfiniteCylinder(float3 p, float radius)
{
    return length(p.xz) - radius;
}

// RoundedCylinder: cylinder with rounded edges
float sdRoundedCylinder(float3 p, float radius, float roundRadius, float halfHeight)
{
    float2 d = float2(length(p.xz) - 2.0 * radius + roundRadius, abs(p.y) - halfHeight);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - roundRadius;
}

// TriangularPrism: equilateral triangle along Z
float sdTriangularPrism(float3 p, float width, float halfDepth)
{
    float3 q = abs(p);
    return max(q.z - halfDepth, max(q.x * 0.866025 + p.y * 0.5, -p.y) - width * 0.5);
}

// CutSphere: sphere with planar cut
float sdCutSphere(float3 p, float radius, float cutHeight)
{
    float w = sqrt(max(radius * radius - cutHeight * cutHeight, 0.0));
    float2 q = float2(length(p.xz), p.y);
    float s1 = (cutHeight - radius) * q.x * q.x + w * w * (cutHeight + radius - 2.0 * q.y);
    float s2 = cutHeight * q.x - w * q.y;
    float s = max(s1, s2);
    if (s < 0.0) return length(q) - radius;
    if (q.x < w) return cutHeight - q.y;
    return length(q - float2(w, cutHeight));
}

// CutHollowSphere: hollow sphere with planar cut
float sdCutHollowSphere(float3 p, float radius, float cutHeight, float thickness)
{
    float w = sqrt(max(radius * radius - cutHeight * cutHeight, 0.0));
    float2 q = float2(length(p.xz), p.y);
    if (cutHeight * q.x < w * q.y)
        return length(q - float2(w, cutHeight)) - thickness;
    return abs(length(q) - radius) - thickness;
}

// DeathStar: large sphere with spherical indentation
float sdDeathStar(float3 p, float ra, float rb, float d)
{
    float a = (ra * ra - rb * rb + d * d) / (2.0 * d);
    float b = sqrt(max(ra * ra - a * a, 0.0));
    float2 p2 = float2(p.x, length(p.yz));
    if (p2.x * b - p2.y * a > d * max(b - p2.y, 0.0))
        return length(p2 - float2(a, b));
    return max(length(p2) - ra, -(length(p2 - float2(d, 0.0)) - rb));
}

// SolidAngle: 3D cone sector
float sdSolidAngle(float3 p, float angle, float radius)
{
    float2 c = float2(sin(angle), cos(angle));
    float2 q = float2(length(p.xz), p.y);
    float l = length(q) - radius;
    float m = length(q - c * clamp(dot(q, c), 0.0, radius));
    float s = (c.y * q.x - c.x * q.y < 0.0) ? -1.0 : 1.0;
    return max(l, m * s);
}

// Rhombus: 3D diamond shape with rounding
float sdRhombus(float3 p, float la, float lb, float halfHeight, float roundRadius)
{
    float3 pp = abs(p);
    float2 b = float2(la, lb);
    float f = clamp((b.x * b.x - b.y * b.y + 2.0 * pp.x * b.x - 2.0 * pp.z * b.y)
              / (b.x * b.x + b.y * b.y), -1.0, 1.0);
    // ndot(b, b - 2*pxz) / dot(b,b) simplified above
    float2 corner = 0.5 * b * float2(1.0 - f, 1.0 + f);
    float qx = length(float2(pp.x, pp.z) - corner)
             * sign(pp.x * b.y + pp.z * b.x - b.x * b.y) - roundRadius;
    float qy = pp.y - halfHeight;
    return min(max(qx, qy), 0.0) + length(max(float2(qx, qy), 0.0));
}

// Vesica: vesica piscis (lens shape) revolved around Y
float sdVesica(float3 p, float radius, float halfDist)
{
    float2 q = float2(length(p.xz), abs(p.y));
    float b = sqrt(max(radius * radius - halfDist * halfDist, 0.0));
    if ((q.y - b) * halfDist > q.x * b)
        return length(q - float2(0.0, b));
    return length(q - float2(-halfDist, 0.0)) - radius;
}

// InfiniteCone: infinite cone along Y-axis
float sdInfiniteCone(float3 p, float angle)
{
    float2 c = float2(sin(angle), cos(angle));
    float2 q = float2(length(p.xz), -p.y);
    float d = length(q - c * max(dot(q, c), 0.0));
    return d * ((q.x * c.y - q.y * c.x < 0.0) ? -1.0 : 1.0);
}

// Tube: hollow cylinder (pipe)
float sdTube(float3 p, float outerRadius, float thickness, float halfHeight)
{
    float r = length(p.xz);
    float dRing = abs(r - outerRadius) - thickness;
    float dY = abs(p.y) - halfHeight;
    return min(max(dRing, dY), 0.0) + length(max(float2(dRing, dY), 0.0));
}

// Barrel: cylinder with parabolic bulge
float sdBarrel(float3 p, float radius, float halfHeight, float bulge)
{
    float r = length(p.xz);
    float yNorm = clamp(p.y / halfHeight, -1.0, 1.0);
    float effectiveR = radius + bulge * (1.0 - yNorm * yNorm);
    float dR = r - effectiveR;
    float dY = abs(p.y) - halfHeight;
    return min(max(dR, dY), 0.0) + length(max(float2(dR, dY), 0.0));
}

// Diamond: bipyramid (double-cone)
float sdDiamond(float3 p, float radius, float halfHeight)
{
    float2 q = float2(length(p.xz), abs(p.y));
    float2 ba = float2(-radius, halfHeight);
    float2 qa = q - float2(radius, 0.0);
    float t = clamp(dot(qa, ba) / dot(ba, ba), 0.0, 1.0);
    float2 closest = float2(radius, 0.0) + ba * t;
    float dist = length(q - closest);
    if (q.x * halfHeight + q.y * radius < radius * halfHeight) return -dist;
    return dist;
}

// ChamferedCube: box with octahedral chamfer
float sdChamferedCube(float3 p, float3 halfExtents, float chamfer)
{
    float3 ap = abs(p);
    float3 q = ap - halfExtents;
    float dBox = length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
    float s = halfExtents.x + halfExtents.y + halfExtents.z;
    float dChamfer = (ap.x + ap.y + ap.z - s + chamfer) * 0.57735027;
    return max(dBox, dChamfer);
}

// RoundedX: X-shaped cross in XZ, extruded along Y
float sdRoundedX(float3 p, float width, float roundRadius, float halfHeight)
{
    float2 q = float2(abs(p.x), abs(p.z));
    float s = min(q.x + q.y, width) * 0.5;
    float d2d = length(q - float2(s, s)) - roundRadius;
    float dY = abs(p.y) - halfHeight;
    return min(max(d2d, dY), 0.0) + length(max(float2(d2d, dY), 0.0));
}

// Pie: sector/pie shape in XZ, extruded Y
float sdPie(float3 p, float angle, float radius, float halfHeight)
{
    float qx = abs(p.x);
    float qz = p.z;
    float2 sc = float2(sin(angle), cos(angle));
    float l = length(float2(qx, qz)) - radius;
    float dotQC = clamp(dot(float2(qx, qz), sc), 0.0, radius);
    float m = length(float2(qx, qz) - sc * dotQC);
    float crossVal = sc.y * qx - sc.x * qz;
    float s = (crossVal > 0.0) ? 1.0 : ((crossVal < 0.0) ? -1.0 : 0.0);
    float d2d = max(l, m * s);
    float dY = abs(p.y) - halfHeight;
    return min(max(d2d, dY), 0.0) + length(max(float2(d2d, dY), 0.0));
}

// Tunnel: D-shaped tunnel (rectangle + semicircle dome), extruded Z
float sdTunnel(float3 p, float width, float height2d, float halfDepth)
{
    float px = abs(p.x);
    float py = p.y;
    float dx = px - width;
    float dyRect = abs(py) - height2d;
    float dRect = length(max(float2(dx, dyRect), 0.0)) + min(max(dx, dyRect), 0.0);
    float dCircle = length(float2(px, py - height2d)) - width;
    float d2d = (py > height2d) ? min(dRect, dCircle) : dRect;
    float dZ = abs(p.z) - halfDepth;
    return min(max(d2d, dZ), 0.0) + length(max(float2(d2d, dZ), 0.0));
}

// UnevenCapsule: two circles of different radii, extruded Z
float sdUnevenCapsule(float3 p, float r1, float r2, float capHeight, float halfDepth)
{
    float px = abs(p.x);
    float h = capHeight * 2.0;
    float b = (r1 - r2) / h;
    float a = sqrt(max(1.0 - b * b, 0.0));
    float k = dot(float2(-b, a), float2(px, p.y));
    float d2d;
    if (k < 0.0) d2d = length(float2(px, p.y)) - r1;
    else if (k > a * h) d2d = length(float2(px, p.y - h)) - r2;
    else d2d = dot(float2(px, p.y), float2(a, b)) - r1;
    float dZ = abs(p.z) - halfDepth;
    return min(max(d2d, dZ), 0.0) + length(max(float2(d2d, dZ), 0.0));
}

// Egg: egg shape revolved around Y
float sdEgg(float3 p, float ra, float rb)
{
    float px = length(p.xz);
    float py = p.y;
    float r = ra - rb;
    if (py < 0.0) return length(float2(px, py)) - r;
    if (px * ra < py * rb) return length(float2(px, py - ra));
    return length(float2(px + rb, py)) - ra;
}

// ArcShape: thick ring sector in XZ, extruded Y
float sdArcShape(float3 p, float aperture, float radius, float thickness, float halfHeight)
{
    float qx = abs(p.x);
    float qz = p.z;
    float2 sc = float2(sin(aperture), cos(aperture));
    float d2d;
    if (sc.y * qx > sc.x * qz)
        d2d = length(float2(qx, qz) - sc * radius) - thickness;
    else
        d2d = abs(length(float2(qx, qz)) - radius) - thickness;
    float dY = abs(p.y) - halfHeight;
    return min(max(d2d, dY), 0.0) + length(max(float2(d2d, dY), 0.0));
}

// Moon: crescent moon (two circles subtracted) in XZ, extruded Y
float sdMoon(float3 p, float d, float ra, float rb, float halfHeight)
{
    float qx = abs(p.x);
    float qz = p.z;
    float dOuter = length(float2(qx, qz)) - ra;
    float dInner = length(float2(qx - d, qz)) - rb;
    float d2d = max(dOuter, -dInner);
    float dY = abs(p.y) - halfHeight;
    return min(max(d2d, dY), 0.0) + length(max(float2(d2d, dY), 0.0));
}

// CrossShape: plus/cross shape in XZ, extruded Y
float sdCrossShape(float3 p, float len, float thickness, float roundRadius, float halfHeight)
{
    float qx = abs(p.x);
    float qz = abs(p.z);
    float2 dH = float2(qx - len, qz - thickness);
    float2 dV = float2(qx - thickness, qz - len);
    float dHsdf = length(max(dH, 0.0)) + min(max(dH.x, dH.y), 0.0);
    float dVsdf = length(max(dV, 0.0)) + min(max(dV.x, dV.y), 0.0);
    float d2d = min(dHsdf, dVsdf) - roundRadius;
    float dY = abs(p.y) - halfHeight;
    return min(max(d2d, dY), 0.0) + length(max(float2(d2d, dY), 0.0));
}

// Triangle: exact unsigned distance to 3D triangle
float sdTriangle(float3 p, float3 a, float3 b, float3 c)
{
    float3 ba = b - a; float3 pa = p - a;
    float3 cb = c - b; float3 pb = p - b;
    float3 ac = a - c; float3 pc = p - c;
    float3 nor = cross(ba, ac);
    float signA = sign(dot(cross(ba, nor), pa));
    float signB = sign(dot(cross(cb, nor), pb));
    float signC = sign(dot(cross(ac, nor), pc));
    float d2;
    if (signA + signB + signC < 2.0)
    {
        float tAB = clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0);
        float dAB = dot(ba * tAB - pa, ba * tAB - pa);
        float tBC = clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0);
        float dBC = dot(cb * tBC - pb, cb * tBC - pb);
        float tCA = clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0);
        float dCA = dot(ac * tCA - pc, ac * tCA - pc);
        d2 = min(dAB, min(dBC, dCA));
    }
    else
    {
        float dn = dot(nor, pa);
        d2 = dn * dn / dot(nor, nor);
    }
    return sqrt(d2);
}

// Bezier: quadratic Bezier curve with tube radius
float sdBezier(float3 pos, float3 a, float3 b, float3 c, float radius)
{
    float3 ab = b - a;
    float3 ba2c = a - 2.0 * b + c;
    float3 cv = ab * 2.0;
    float3 dv = a - pos;
    float ba2cDot = dot(ba2c, ba2c);
    if (ba2cDot < 1e-10)
    {
        float3 ac = c - a;
        float acDot = dot(ac, ac);
        if (acDot < 1e-10) return length(pos - a) - radius;
        float t = clamp(dot(pos - a, ac) / acDot, 0.0, 1.0);
        return length(pos - a - ac * t) - radius;
    }
    float kk = 1.0 / ba2cDot;
    float kx = kk * dot(ab, ba2c);
    float ky = kk * (2.0 * dot(ab, ab) + dot(dv, ba2c)) / 3.0;
    float kz = kk * dot(dv, ab);
    float p2 = ky - kx * kx;
    float p3 = p2 * p2 * p2;
    float q2 = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    float h = q2 * q2 + 4.0 * p3;
    float res;
    if (h >= 0.0)
    {
        float hSqrt = sqrt(h);
        float x0 = (hSqrt - q2) * 0.5;
        float x1 = (-hSqrt - q2) * 0.5;
        float uvX = sign(x0) * pow(abs(x0), 0.333333);
        float uvY = sign(x1) * pow(abs(x1), 0.333333);
        float t = clamp(uvX + uvY - kx, 0.0, 1.0);
        res = length(dv + (cv + ba2c * t) * t);
    }
    else
    {
        float z = sqrt(-p2);
        float v = acos(q2 / (p2 * z * 2.0)) / 3.0;
        float m = cos(v);
        float n = sin(v) * 1.7320508;
        float t0 = clamp((m + m) * z - kx, 0.0, 1.0);
        float t1 = clamp((-n - m) * z - kx, 0.0, 1.0);
        float d0 = length(dv + (cv + ba2c * t0) * t0);
        float d1 = length(dv + (cv + ba2c * t1) * t1);
        res = min(d0, d1);
    }
    return res - radius;
}

// Horseshoe: U-shaped horseshoe
float sdHorseshoe(float3 p, float angle, float radius, float halfLength, float width, float thickness)
{
    float2 cc = float2(cos(angle), sin(angle));
    float px = abs(p.x);
    float l = length(float2(px, p.y));
    float qx = -cc.x * px + cc.y * p.y;
    float qy = cc.y * px + cc.x * p.y;
    if (!(qy > 0.0 || qx > 0.0)) qx = l * sign(-cc.x);
    if (qx <= 0.0) qy = l;
    qx = abs(qx);
    qy -= radius;
    float rx = max(qx - halfLength, 0.0);
    float innerLen = length(float2(rx, qy)) + min(max(qx - halfLength, qy), 0.0);
    float dx = max(innerLen - width, 0.0);
    float dy = max(abs(p.z) - thickness, 0.0);
    return -min(width, thickness) + length(float2(dx, dy)) + max(innerLen - width, abs(p.z) - thickness) * step(max(innerLen - width, abs(p.z) - thickness), 0.0);
}

// Superellipsoid: generalized ellipsoid with power exponents
float sdSuperellipsoid(float3 p, float3 halfExtents, float e1, float e2)
{
    e1 = max(e1, 0.02);
    e2 = max(e2, 0.02);
    float3 q = float3(
        max(abs(p.x / halfExtents.x), 1e-10),
        max(abs(p.y / halfExtents.y), 1e-10),
        max(abs(p.z / halfExtents.z), 1e-10)
    );
    float m1 = 2.0 / e2;
    float m2 = 2.0 / e1;
    float w = pow(q.x, m1) + pow(q.z, m1);
    float v = pow(w, e2 / e1) + pow(q.y, m2);
    float f = pow(v, e1 * 0.5);
    float minE = min(halfExtents.x, min(halfExtents.y, halfExtents.z));
    return (f - 1.0) * minE * 0.5;
}

// Trapezoid: trapezoid in XY, extruded Z
float sdTrapezoid(float3 p, float r1, float r2, float trapHeight, float halfDepth)
{
    float px = abs(p.x);
    float py = p.y;
    float he = trapHeight;
    // Distance to three edges
    float dBot = length(float2(max(px - r1, 0.0), max(-py - he, 0.0))) + min(max(px - r1, -py - he), 0.0);
    float2 slantDir = float2(2.0 * he, r1 - r2);
    float slantLen = length(slantDir);
    float2 slantN = slantDir / max(slantLen, 1e-10);
    float dSlant = dot(float2(px - r1, py + he), slantN);
    float dTop = length(float2(max(px - r2, 0.0), max(py - he, 0.0))) + min(max(px - r2, py - he), 0.0);
    float dUnsigned = min(min(abs(dBot), abs(dSlant)), abs(dTop));
    // Simplified inside check
    bool inside = py >= -he && py <= he && dSlant <= 0.0;
    float d2d = inside ? -dUnsigned : dUnsigned;
    float dZ = abs(p.z) - halfDepth;
    return min(max(d2d, dZ), 0.0) + length(max(float2(d2d, dZ), 0.0));
}

// Parallelogram: parallelogram in XY (with skew), extruded Z
float sdParallelogram(float3 p, float width, float paraHeight, float skew, float halfDepth)
{
    float he = paraHeight;
    float wi = width;
    float sk = skew;
    // Four corners: (wi-sk,-he), (wi+sk,he), (-wi+sk,he), (-wi-sk,-he)
    float ex = wi + abs(sk);
    float ey = he;
    // Distance to skewed box (simplified: transform to axis-aligned)
    float qx = p.x - p.y * sk / he;
    float d2d_x = abs(qx) - wi;
    float d2d_y = abs(p.y) - he;
    float d2d = length(max(float2(d2d_x, d2d_y), 0.0)) + min(max(d2d_x, d2d_y), 0.0);
    float dZ = abs(p.z) - halfDepth;
    return min(max(d2d, dZ), 0.0) + length(max(float2(d2d, dZ), 0.0));
}

// BlobbyCross: organic blobby cross in XZ, extruded Y
float sdBlobbyCross(float3 p, float size, float halfHeight)
{
    float qx = abs(p.x) / size;
    float qz = abs(p.z) / size;
    float n = qx + qz;
    float d2d;
    if (n < 1.0)
    {
        float t = 1.0 - n;
        float bb = qx * qz;
        d2d = (-sqrt(max(t * t - 2.0 * bb, 0.0)) + n - 1.0) * size * 0.70710678;
    }
    else
    {
        float dx = max(qx - 1.0, 0.0);
        float dz = max(qz - 1.0, 0.0);
        float dxLen = length(float2(qx - 1.0, qz));
        float dzLen = length(float2(qx, qz - 1.0));
        d2d = min(dxLen, min(dzLen, sqrt(dx * dx + dz * dz))) * size;
    }
    float dY = abs(p.y) - halfHeight;
    return min(max(d2d, dY), 0.0) + length(max(float2(d2d, dY), 0.0));
}

// ParabolaSegment: parabolic arch in XY, extruded Z (Newton's method)
float sdParabolaSegment(float3 p, float width, float paraHeight, float halfDepth)
{
    float px = abs(p.x);
    float py = p.y;
    float w = width;
    float h = paraHeight;
    float ww = w * w;
    // Closest point on parabola via Newton's method
    float t = clamp(px, 0.0, w);
    for (int i = 0; i < 8; i++)
    {
        float ft = h * (1.0 - t * t / ww);
        float dft = -2.0 * h * t / ww;
        float ex = px - t;
        float ey = py - ft;
        float f = -ex + ey * dft;
        float df = 1.0 + dft * dft + ey * (-2.0 * h / ww);
        if (abs(df) > 1e-10) t = clamp(t - f / df, 0.0, w);
    }
    float closestY = h * (1.0 - t * t / ww);
    float dPara = length(float2(px - t, py - closestY));
    float dBase = (px <= w) ? abs(py) : length(float2(px - w, py));
    float dUnsigned = min(dPara, dBase);
    float yArch = (px <= w) ? h * (1.0 - (px / w) * (px / w)) : 0.0;
    bool inside = px <= w && py >= 0.0 && py <= yArch;
    float d2d = inside ? -dUnsigned : dUnsigned;
    float dZ = abs(p.z) - halfDepth;
    return min(max(d2d, dZ), 0.0) + length(max(float2(d2d, dZ), 0.0));
}

// RegularPolygon: N-sided polygon in XZ, extruded Y
float sdRegularPolygon(float3 p, float radius, float nSides, float halfHeight)
{
    float qx = abs(p.x);
    float qz = p.z;
    float n = max(nSides, 3.0);
    float an = 3.14159265 / n;
    float he = radius * cos(an);
    float angle = atan2(qz, qx);
    float bn = an * floor((angle + an) / (2.0 * an));
    float cosB = cos(bn);
    float sinB = sin(bn);
    float rx = cosB * qx + sinB * qz;
    float d2d = rx - he;
    float dY = abs(p.y) - halfHeight;
    return min(max(d2d, dY), 0.0) + length(max(float2(d2d, dY), 0.0));
}

// StarPolygon: star polygon in XZ, extruded Y
float sdStarPolygon(float3 p, float radius, float nPoints, float m, float halfHeight)
{
    float qx = abs(p.x);
    float qz = p.z;
    float n = max(nPoints, 3.0);
    float an = 3.14159265 / n;
    float r = length(float2(qx, qz));
    float angle = atan2(qz, qx);
    angle = fmod(fmod(angle, 2.0 * an) + 2.0 * an, 2.0 * an);
    if (angle > an) angle = 2.0 * an - angle;
    float2 pt = float2(r * cos(angle), r * sin(angle));
    float2 aa = float2(radius, 0.0);
    float2 bb = float2(m * cos(an), m * sin(an));
    float2 ab = bb - aa;
    float2 ap = pt - aa;
    float t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
    float2 closest = aa + ab * t;
    float dist = length(pt - closest);
    float crossV = ab.x * ap.y - ab.y * ap.x;
    float d2d = (crossV > 0.0) ? -dist : dist;
    float dY = abs(p.y) - halfHeight;
    return min(max(d2d, dY), 0.0) + length(max(float2(d2d, dY), 0.0));
}

// Stairs: staircase in XY, extruded Z
float _sdStairsStepBox(float lx, float ly, float s, float sw, float sh)
{
    float cx = s * sw + sw * 0.5;
    float hy = (s + 1.0) * sh * 0.5;
    float dx = abs(lx - cx) - sw * 0.5;
    float dy = abs(ly - hy) - hy;
    return length(max(float2(dx, dy), 0.0)) + min(max(dx, dy), 0.0);
}

float sdStairs(float3 p, float stepWidth, float stepHeight, float nSteps, float halfDepth)
{
    float sw = stepWidth;
    float sh = stepHeight;
    float n = max(nSteps, 1.0);
    float tw = n * sw;
    float th = n * sh;
    float lx = p.x + tw * 0.5;
    float ly = p.y + th * 0.5;
    float si = clamp(floor(lx / sw), 0.0, n - 1.0);
    float d2d = _sdStairsStepBox(lx, ly, si, sw, sh);
    if (si > 0.0) d2d = min(d2d, _sdStairsStepBox(lx, ly, si - 1.0, sw, sh));
    if (si < n - 1.0) d2d = min(d2d, _sdStairsStepBox(lx, ly, si + 1.0, sw, sh));
    float sj = clamp(ceil(ly / sh) - 1.0, 0.0, n - 1.0);
    if (sj != si && sj != si - 1.0 && sj != si + 1.0)
        d2d = min(d2d, _sdStairsStepBox(lx, ly, sj, sw, sh));
    float dZ = abs(p.z) - halfDepth;
    return min(max(d2d, dZ), 0.0) + length(max(float2(d2d, dZ), 0.0));
}

// Helix: spiral tube along Y-axis
float sdHelix(float3 p, float majorR, float minorR, float pitch, float halfHeight)
{
    float rXZ = length(p.xz);
    float theta = atan2(p.z, p.x);
    float py = p.y;
    float tau = 6.28318530;
    float dRadial = rXZ - majorR;
    float yAtTheta = theta * pitch / tau;
    float k = round((py - yAtTheta) / pitch);
    float dTube = 1e20;
    for (int dk = -1; dk <= 1; dk++)
    {
        float yHelix = yAtTheta + (k + float(dk)) * pitch;
        float dy = py - yHelix;
        float d = length(float2(dRadial, dy)) - minorR;
        dTube = min(dTube, d);
    }
    float dCap = abs(py) - halfHeight;
    return max(dTube, dCap);
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

// Polar Repeat: repeat child around Y-axis
float3 opPolarRepeat(float3 p, float count)
{
    float sector = 6.28318530 / count;
    float a = atan2(p.z, p.x) + sector * 0.5;
    float r = length(p.xz);
    a = fmod(a, sector) - sector * 0.5;
    return float3(r * cos(a), p.y, r * sin(a));
}

// Taper: taper XZ cross-section along Y
float3 opTaper(float3 p, float factor)
{
    float s = 1.0 / (1.0 - p.y * factor);
    return float3(p.x * s, p.y, p.z * s);
}

// Displacement: sin-based oscillating displacement
float opDisplacement(float d, float3 p, float strength)
{
    return d + sin(p.x * 5.0) * sin(p.y * 5.0) * sin(p.z * 5.0) * strength;
}

// Symmetry: mirror point along specified axes (1.0 = mirror, 0.0 = keep)
float3 opSymmetry(float3 p, float3 axes)
{
    return float3(
        axes.x > 0.5 ? abs(p.x) : p.x,
        axes.y > 0.5 ? abs(p.y) : p.y,
        axes.z > 0.5 ? abs(p.z) : p.z
    );
}

// Elongate: box elongation (pre-processing transform)
float3 opElongate(float3 p, float3 amount)
{
    return p - clamp(p, -amount, amount);
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
