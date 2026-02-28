// =============================================================================
// AliceSDF_Primitives.cs - Pure C# SDF Functions for UdonSharp
// =============================================================================
// 53 primitives + 17 operations + transforms ported from ALICE-SDF Rust engine.
// No DllImport - works in VRChat's UdonSharp sandbox.
//
// These functions mirror the HLSL versions in AliceSDF_Include.cginc exactly,
// ensuring Shader rendering and Udon collision agree on the same surface.
//
// Author: Moroya Sakamoto
// =============================================================================

using UnityEngine;

namespace AliceSDF
{
    /// <summary>
    /// Pure C# SDF evaluation functions.
    /// Every function here has a 1:1 HLSL counterpart in AliceSDF_Include.cginc.
    /// </summary>
    public static class Sdf
    {
        // =====================================================================
        // Primitives
        // =====================================================================

        /// <summary>Sphere at origin with given radius.</summary>
        public static float Sphere(Vector3 p, float radius)
        {
            return p.magnitude - radius;
        }

        /// <summary>Axis-aligned box at origin with given half-extents.</summary>
        public static float Box(Vector3 p, Vector3 halfExtents)
        {
            Vector3 q = SdfMath.Abs(p) - halfExtents;
            return SdfMath.Max(q, 0f).magnitude
                 + Mathf.Min(SdfMath.MaxComponent(q), 0f);
        }

        /// <summary>Vertical capped cylinder at origin.</summary>
        public static float Cylinder(Vector3 p, float radius, float halfHeight)
        {
            float dx = SdfMath.LengthXZ(p) - radius;
            float dy = Mathf.Abs(p.y) - halfHeight;
            return Mathf.Min(Mathf.Max(dx, dy), 0f)
                 + new Vector2(Mathf.Max(dx, 0f), Mathf.Max(dy, 0f)).magnitude;
        }

        /// <summary>Torus in XZ plane at origin.</summary>
        public static float Torus(Vector3 p, float majorRadius, float minorRadius)
        {
            float qx = SdfMath.LengthXZ(p) - majorRadius;
            Vector2 q = new Vector2(qx, p.y);
            return q.magnitude - minorRadius;
        }

        /// <summary>Infinite plane with normal and distance.</summary>
        public static float Plane(Vector3 p, Vector3 normal, float distance)
        {
            return Vector3.Dot(p, normal) + distance;
        }

        /// <summary>Capsule between two points with radius.</summary>
        public static float Capsule(Vector3 p, Vector3 a, Vector3 b, float radius)
        {
            Vector3 pa = p - a;
            Vector3 ba = b - a;
            float baDot = Vector3.Dot(ba, ba);
            // Guard: degenerate capsule (a == b) → sphere
            float h = baDot > 0.0001f ? Mathf.Clamp01(Vector3.Dot(pa, ba) / baDot) : 0f;
            return (pa - ba * h).magnitude - radius;
        }

        /// <summary>Rounded cone along Y-axis (bottom r1, top r2).</summary>
        public static float RoundedCone(Vector3 p, float r1, float r2, float halfHeight)
        {
            float h = halfHeight * 2f;
            float qx = Mathf.Sqrt(p.x * p.x + p.z * p.z);
            float qy = p.y + halfHeight;
            float b = (r1 - r2) / h;
            float a = Mathf.Sqrt(1f - b * b);
            float k = qx * (-b) + qy * a;
            if (k < 0f) return new Vector2(qx, qy).magnitude - r1;
            if (k > a * h) return new Vector2(qx, qy - h).magnitude - r2;
            return qx * a + qy * b - r1;
        }

        /// <summary>4-sided pyramid along Y-axis (unit base).</summary>
        public static float Pyramid(Vector3 p, float halfHeight)
        {
            float h = halfHeight * 2f;
            float m2 = h * h + 0.25f;
            float py = p.y + halfHeight;
            float px = Mathf.Abs(p.x);
            float pz = Mathf.Abs(p.z);
            if (pz > px) { float tmp = px; px = pz; pz = tmp; }
            px -= 0.5f;
            pz -= 0.5f;
            float qx = pz;
            float qy = h * py - 0.5f * px;
            float qz = h * px + 0.5f * py;
            float s = Mathf.Max(-qx, 0f);
            float t = Mathf.Clamp((qy - 0.5f * pz) / (m2 + 0.25f), 0f, 1f);
            float aa = m2 * (qx + s) * (qx + s) + qy * qy;
            float bb = m2 * (qx + 0.5f * t) * (qx + 0.5f * t) + (qy - m2 * t) * (qy - m2 * t);
            float d2 = (Mathf.Min(-qx * m2 - qy * 0.5f, qy) > 0f) ? 0f : Mathf.Min(aa, bb);
            return Mathf.Sqrt((d2 + qz * qz) / m2) * Mathf.Sign(Mathf.Max(qz, -py));
        }

        /// <summary>Regular octahedron centered at origin.</summary>
        public static float Octahedron(Vector3 p, float s)
        {
            Vector3 ap = SdfMath.Abs(p);
            float m = ap.x + ap.y + ap.z - s;
            Vector3 q;
            if (3f * ap.x < m) q = ap;
            else if (3f * ap.y < m) q = new Vector3(ap.y, ap.z, ap.x);
            else if (3f * ap.z < m) q = new Vector3(ap.z, ap.x, ap.y);
            else return m * 0.57735027f;
            float k = Mathf.Clamp(0.5f * (q.z - q.y + s), 0f, s);
            return new Vector3(q.x, q.y - s + k, q.z - k).magnitude;
        }

        /// <summary>Chain link shape (torus stretched along Y).</summary>
        public static float Link(Vector3 p, float halfLength, float r1, float r2)
        {
            float qx = p.x;
            float qy = Mathf.Max(Mathf.Abs(p.y) - halfLength, 0f);
            float qz = p.z;
            float xyLen = Mathf.Sqrt(qx * qx + qy * qy) - r1;
            return Mathf.Sqrt(xyLen * xyLen + qz * qz) - r2;
        }

        /// <summary>Cone along Y-axis.</summary>
        public static float Cone(Vector3 p, float radius, float halfHeight)
        {
            float h = halfHeight * 2f;
            float qx = SdfMath.LengthXZ(p);
            float qy = p.y + halfHeight;
            float s = Mathf.Max(0f, qx * h - qy * radius);
            float d = new Vector2(
                qx - radius * Mathf.Clamp(s > 0f ? (qx * h - qy * radius) / (radius * radius + h * h) : qx / radius, 0f, 1f),
                qy - h * (s > 0f ? (qx * h - qy * radius) / (radius * radius + h * h) : 0f)
            ).magnitude;
            return d * (qy > h ? 1f : (qx > radius ? 1f : -1f));
        }

        /// <summary>Ellipsoid at origin.</summary>
        public static float Ellipsoid(Vector3 p, Vector3 radii)
        {
            float k0 = new Vector3(p.x / radii.x, p.y / radii.y, p.z / radii.z).magnitude;
            float k1 = new Vector3(p.x / (radii.x * radii.x), p.y / (radii.y * radii.y), p.z / (radii.z * radii.z)).magnitude;
            return k0 > 0.0001f ? k0 * (k0 - 1f) / k1 : 0f;
        }

        /// <summary>Hex prism along Y-axis.</summary>
        public static float HexPrism(Vector3 p, float hexRadius, float halfHeight)
        {
            Vector3 ap = SdfMath.Abs(p);
            float dx = ap.x * 0.866025f + ap.z * 0.5f;
            float dy = ap.z;
            float d2d = Mathf.Max(dx, dy) - hexRadius;
            float dh = ap.y - halfHeight;
            return Mathf.Min(Mathf.Max(d2d, dh), 0f) +
                   new Vector2(Mathf.Max(d2d, 0f), Mathf.Max(dh, 0f)).magnitude;
        }

        /// <summary>Rounded box at origin.</summary>
        public static float RoundedBox(Vector3 p, Vector3 halfExtents, float roundRadius)
        {
            Vector3 q = SdfMath.Abs(p) - halfExtents;
            return SdfMath.Max(q, 0f).magnitude + Mathf.Min(SdfMath.MaxComponent(q), 0f) - roundRadius;
        }

        /// <summary>Heart shape at origin.</summary>
        public static float Heart(Vector3 p, float size)
        {
            float px = Mathf.Abs(p.x);
            float py = p.y;
            float pz = p.z;
            float s = size;
            if (px + py > s) return Mathf.Sqrt((px - 0.25f * s) * (px - 0.25f * s) + (py - 0.75f * s) * (py - 0.75f * s) + pz * pz) - s * 0.3536f;
            return Mathf.Sqrt(Mathf.Min(px * px + (py - s) * (py - s), (px + py - s) * 0.5f * (px + py - s) * 0.5f) + pz * pz) * Mathf.Sign(px + py - s);
        }

        /// <summary>Capped torus in XZ plane.</summary>
        public static float CappedTorus(Vector3 p, float majorRadius, float minorRadius, float capAngle)
        {
            float sc = Mathf.Sin(capAngle);
            float cc = Mathf.Cos(capAngle);
            float px = Mathf.Abs(p.x);
            float lenXZ = Mathf.Sqrt(px * px + p.z * p.z);
            float kx = (cc * px > sc * lenXZ) ? px * sc - p.z * cc : lenXZ;
            return Mathf.Sqrt(kx * kx + p.y * p.y + majorRadius * majorRadius + minorRadius * 0f
                - 2f * majorRadius * (cc * px > sc * lenXZ ? px * cc + p.z * sc : lenXZ)) - minorRadius;
        }

        /// <summary>Box frame (hollow wireframe box).</summary>
        public static float BoxFrame(Vector3 p, Vector3 halfExtents, float edge)
        {
            Vector3 ap = SdfMath.Abs(p) - halfExtents;
            Vector3 q = SdfMath.Abs(ap + new Vector3(edge, edge, edge)) - new Vector3(edge, edge, edge);
            float d1 = SdfMath.Max(new Vector3(ap.x, q.y, q.z), 0f).magnitude + Mathf.Min(Mathf.Max(ap.x, Mathf.Max(q.y, q.z)), 0f);
            float d2 = SdfMath.Max(new Vector3(q.x, ap.y, q.z), 0f).magnitude + Mathf.Min(Mathf.Max(q.x, Mathf.Max(ap.y, q.z)), 0f);
            float d3 = SdfMath.Max(new Vector3(q.x, q.y, ap.z), 0f).magnitude + Mathf.Min(Mathf.Max(q.x, Mathf.Max(q.y, ap.z)), 0f);
            return Mathf.Min(d1, Mathf.Min(d2, d3));
        }

        /// <summary>Gyroid TPMS surface.</summary>
        public static float Gyroid(Vector3 p, float scale, float thickness)
        {
            float t = Mathf.PI * 2f / scale;
            float qx = p.x * t, qy = p.y * t, qz = p.z * t;
            float d = Mathf.Sin(qx) * Mathf.Cos(qy) + Mathf.Sin(qy) * Mathf.Cos(qz) + Mathf.Sin(qz) * Mathf.Cos(qx);
            return Mathf.Abs(d) - thickness;
        }

        /// <summary>Schwarz P TPMS surface.</summary>
        public static float SchwarzP(Vector3 p, float scale, float thickness)
        {
            float t = Mathf.PI * 2f / scale;
            float d = Mathf.Cos(p.x * t) + Mathf.Cos(p.y * t) + Mathf.Cos(p.z * t);
            return Mathf.Abs(d) - thickness;
        }

        /// <summary>Infinite cylinder along Y-axis.</summary>
        public static float InfiniteCylinder(Vector3 p, float radius)
        {
            return SdfMath.LengthXZ(p) - radius;
        }

        /// <summary>Cylinder with rounded edges.</summary>
        public static float RoundedCylinder(Vector3 p, float radius, float roundRadius, float halfHeight)
        {
            float dx = SdfMath.LengthXZ(p) - 2f * radius + roundRadius;
            float dy = Mathf.Abs(p.y) - halfHeight;
            return Mathf.Min(Mathf.Max(dx, dy), 0f)
                 + new Vector2(Mathf.Max(dx, 0f), Mathf.Max(dy, 0f)).magnitude - roundRadius;
        }

        /// <summary>Equilateral triangular prism along Z.</summary>
        public static float TriangularPrism(Vector3 p, float width, float halfDepth)
        {
            Vector3 q = SdfMath.Abs(p);
            return Mathf.Max(q.z - halfDepth, Mathf.Max(q.x * 0.866025f + p.y * 0.5f, -p.y) - width * 0.5f);
        }

        /// <summary>Sphere with planar cut.</summary>
        public static float CutSphere(Vector3 p, float radius, float cutHeight)
        {
            float w = Mathf.Sqrt(Mathf.Max(radius * radius - cutHeight * cutHeight, 0f));
            Vector2 q = new Vector2(SdfMath.LengthXZ(p), p.y);
            float s1 = (cutHeight - radius) * q.x * q.x + w * w * (cutHeight + radius - 2f * q.y);
            float s2 = cutHeight * q.x - w * q.y;
            float s = Mathf.Max(s1, s2);
            if (s < 0f) return q.magnitude - radius;
            if (q.x < w) return cutHeight - q.y;
            return (q - new Vector2(w, cutHeight)).magnitude;
        }

        /// <summary>Hollow sphere with planar cut.</summary>
        public static float CutHollowSphere(Vector3 p, float radius, float cutHeight, float thickness)
        {
            float w = Mathf.Sqrt(Mathf.Max(radius * radius - cutHeight * cutHeight, 0f));
            Vector2 q = new Vector2(SdfMath.LengthXZ(p), p.y);
            if (cutHeight * q.x < w * q.y)
                return (q - new Vector2(w, cutHeight)).magnitude - thickness;
            return Mathf.Abs(q.magnitude - radius) - thickness;
        }

        /// <summary>Death Star: large sphere with spherical indentation.</summary>
        public static float DeathStar(Vector3 p, float ra, float rb, float d)
        {
            float a = (ra * ra - rb * rb + d * d) / (2f * d);
            float b = Mathf.Sqrt(Mathf.Max(ra * ra - a * a, 0f));
            Vector2 p2 = new Vector2(p.x, Mathf.Sqrt(p.y * p.y + p.z * p.z));
            if (p2.x * b - p2.y * a > d * Mathf.Max(b - p2.y, 0f))
                return (p2 - new Vector2(a, b)).magnitude;
            return Mathf.Max(p2.magnitude - ra, -(new Vector2(p2.x - d, p2.y).magnitude - rb));
        }

        /// <summary>Solid angle (3D cone sector).</summary>
        public static float SolidAngle(Vector3 p, float angle, float radius)
        {
            Vector2 c = new Vector2(Mathf.Sin(angle), Mathf.Cos(angle));
            Vector2 q = new Vector2(SdfMath.LengthXZ(p), p.y);
            float l = q.magnitude - radius;
            float dotQC = Mathf.Clamp(Vector2.Dot(q, c), 0f, radius);
            float m = (q - c * dotQC).magnitude;
            float s = (c.y * q.x - c.x * q.y < 0f) ? -1f : 1f;
            return Mathf.Max(l, m * s);
        }

        /// <summary>Rhombus (3D diamond shape with rounding).</summary>
        public static float Rhombus(Vector3 p, float la, float lb, float halfHeight, float roundRadius)
        {
            Vector3 pp = SdfMath.Abs(p);
            Vector2 b = new Vector2(la, lb);
            float f = Mathf.Clamp((b.x * b.x - b.y * b.y + 2f * pp.x * b.x - 2f * pp.z * b.y)
                      / (b.x * b.x + b.y * b.y), -1f, 1f);
            Vector2 corner = 0.5f * new Vector2(b.x * (1f - f), b.y * (1f + f));
            float qx = (new Vector2(pp.x, pp.z) - corner).magnitude
                     * Mathf.Sign(pp.x * b.y + pp.z * b.x - b.x * b.y) - roundRadius;
            float qy = pp.y - halfHeight;
            return Mathf.Min(Mathf.Max(qx, qy), 0f) + new Vector2(Mathf.Max(qx, 0f), Mathf.Max(qy, 0f)).magnitude;
        }

        /// <summary>Vesica piscis (lens shape) revolved around Y.</summary>
        public static float Vesica(Vector3 p, float radius, float halfDist)
        {
            Vector2 q = new Vector2(SdfMath.LengthXZ(p), Mathf.Abs(p.y));
            float b = Mathf.Sqrt(Mathf.Max(radius * radius - halfDist * halfDist, 0f));
            if ((q.y - b) * halfDist > q.x * b)
                return (q - new Vector2(0f, b)).magnitude;
            return (q - new Vector2(-halfDist, 0f)).magnitude - radius;
        }

        /// <summary>Infinite cone along Y-axis.</summary>
        public static float InfiniteCone(Vector3 p, float angle)
        {
            Vector2 c = new Vector2(Mathf.Sin(angle), Mathf.Cos(angle));
            Vector2 q = new Vector2(SdfMath.LengthXZ(p), -p.y);
            float dotQC = Mathf.Max(Vector2.Dot(q, c), 0f);
            float d = (q - c * dotQC).magnitude;
            return d * ((q.x * c.y - q.y * c.x < 0f) ? -1f : 1f);
        }

        /// <summary>Tube (hollow cylinder/pipe).</summary>
        public static float Tube(Vector3 p, float outerRadius, float thickness, float halfHeight)
        {
            float r = SdfMath.LengthXZ(p);
            float dRing = Mathf.Abs(r - outerRadius) - thickness;
            float dY = Mathf.Abs(p.y) - halfHeight;
            return Mathf.Min(Mathf.Max(dRing, dY), 0f)
                 + new Vector2(Mathf.Max(dRing, 0f), Mathf.Max(dY, 0f)).magnitude;
        }

        /// <summary>Barrel (cylinder with parabolic bulge).</summary>
        public static float Barrel(Vector3 p, float radius, float halfHeight, float bulge)
        {
            float r = SdfMath.LengthXZ(p);
            float yNorm = Mathf.Clamp(p.y / halfHeight, -1f, 1f);
            float effectiveR = radius + bulge * (1f - yNorm * yNorm);
            float dR = r - effectiveR;
            float dY = Mathf.Abs(p.y) - halfHeight;
            return Mathf.Min(Mathf.Max(dR, dY), 0f)
                 + new Vector2(Mathf.Max(dR, 0f), Mathf.Max(dY, 0f)).magnitude;
        }

        /// <summary>Diamond (bipyramid/double-cone).</summary>
        public static float Diamond(Vector3 p, float radius, float halfHeight)
        {
            Vector2 q = new Vector2(SdfMath.LengthXZ(p), Mathf.Abs(p.y));
            Vector2 ba = new Vector2(-radius, halfHeight);
            Vector2 qa = q - new Vector2(radius, 0f);
            float t = Mathf.Clamp(Vector2.Dot(qa, ba) / Vector2.Dot(ba, ba), 0f, 1f);
            Vector2 closest = new Vector2(radius, 0f) + ba * t;
            float dist = (q - closest).magnitude;
            if (q.x * halfHeight + q.y * radius < radius * halfHeight) return -dist;
            return dist;
        }

        /// <summary>Chamfered cube (box with octahedral chamfer).</summary>
        public static float ChamferedCube(Vector3 p, Vector3 halfExtents, float chamfer)
        {
            Vector3 ap = SdfMath.Abs(p);
            Vector3 q = ap - halfExtents;
            float dBox = SdfMath.Max(q, 0f).magnitude + Mathf.Min(SdfMath.MaxComponent(q), 0f);
            float s = halfExtents.x + halfExtents.y + halfExtents.z;
            float dChamfer = (ap.x + ap.y + ap.z - s + chamfer) * 0.57735027f;
            return Mathf.Max(dBox, dChamfer);
        }

        /// <summary>X-shaped cross in XZ, extruded along Y.</summary>
        public static float RoundedX(Vector3 p, float width, float roundRadius, float halfHeight)
        {
            Vector2 q = new Vector2(Mathf.Abs(p.x), Mathf.Abs(p.z));
            float ss = Mathf.Min(q.x + q.y, width) * 0.5f;
            float d2d = (q - new Vector2(ss, ss)).magnitude - roundRadius;
            float dY = Mathf.Abs(p.y) - halfHeight;
            return Mathf.Min(Mathf.Max(d2d, dY), 0f)
                 + new Vector2(Mathf.Max(d2d, 0f), Mathf.Max(dY, 0f)).magnitude;
        }

        /// <summary>Pie/sector shape in XZ, extruded Y.</summary>
        public static float Pie(Vector3 p, float angle, float radius, float halfHeight)
        {
            float qx = Mathf.Abs(p.x);
            float qz = p.z;
            Vector2 sc = new Vector2(Mathf.Sin(angle), Mathf.Cos(angle));
            float l = new Vector2(qx, qz).magnitude - radius;
            float dotQC = Mathf.Clamp(Vector2.Dot(new Vector2(qx, qz), sc), 0f, radius);
            float m = (new Vector2(qx, qz) - sc * dotQC).magnitude;
            float crossVal = sc.y * qx - sc.x * qz;
            float s = (crossVal > 0f) ? 1f : ((crossVal < 0f) ? -1f : 0f);
            float d2d = Mathf.Max(l, m * s);
            float dY = Mathf.Abs(p.y) - halfHeight;
            return Mathf.Min(Mathf.Max(d2d, dY), 0f)
                 + new Vector2(Mathf.Max(d2d, 0f), Mathf.Max(dY, 0f)).magnitude;
        }

        /// <summary>D-shaped tunnel (rectangle + semicircle dome), extruded Z.</summary>
        public static float Tunnel(Vector3 p, float width, float height2d, float halfDepth)
        {
            float px = Mathf.Abs(p.x);
            float py = p.y;
            float dx = px - width;
            float dyRect = Mathf.Abs(py) - height2d;
            float dRect = new Vector2(Mathf.Max(dx, 0f), Mathf.Max(dyRect, 0f)).magnitude
                        + Mathf.Min(Mathf.Max(dx, dyRect), 0f);
            float dCircle = new Vector2(px, py - height2d).magnitude - width;
            float d2d = (py > height2d) ? Mathf.Min(dRect, dCircle) : dRect;
            float dZ = Mathf.Abs(p.z) - halfDepth;
            return Mathf.Min(Mathf.Max(d2d, dZ), 0f)
                 + new Vector2(Mathf.Max(d2d, 0f), Mathf.Max(dZ, 0f)).magnitude;
        }

        /// <summary>Uneven capsule (two circles of different radii), extruded Z.</summary>
        public static float UnevenCapsule(Vector3 p, float r1, float r2, float capHeight, float halfDepth)
        {
            float px = Mathf.Abs(p.x);
            float h = capHeight * 2f;
            float b = (r1 - r2) / h;
            float a = Mathf.Sqrt(Mathf.Max(1f - b * b, 0f));
            float k = Vector2.Dot(new Vector2(-b, a), new Vector2(px, p.y));
            float d2d;
            if (k < 0f) d2d = new Vector2(px, p.y).magnitude - r1;
            else if (k > a * h) d2d = new Vector2(px, p.y - h).magnitude - r2;
            else d2d = Vector2.Dot(new Vector2(px, p.y), new Vector2(a, b)) - r1;
            float dZ = Mathf.Abs(p.z) - halfDepth;
            return Mathf.Min(Mathf.Max(d2d, dZ), 0f)
                 + new Vector2(Mathf.Max(d2d, 0f), Mathf.Max(dZ, 0f)).magnitude;
        }

        /// <summary>Egg shape revolved around Y.</summary>
        public static float Egg(Vector3 p, float ra, float rb)
        {
            float px = SdfMath.LengthXZ(p);
            float py = p.y;
            float r = ra - rb;
            if (py < 0f) return new Vector2(px, py).magnitude - r;
            if (px * ra < py * rb) return new Vector2(px, py - ra).magnitude;
            return new Vector2(px + rb, py).magnitude - ra;
        }

        /// <summary>Thick ring sector in XZ, extruded Y.</summary>
        public static float ArcShape(Vector3 p, float aperture, float radius, float thickness, float halfHeight)
        {
            float qx = Mathf.Abs(p.x);
            float qz = p.z;
            Vector2 sc = new Vector2(Mathf.Sin(aperture), Mathf.Cos(aperture));
            float d2d;
            if (sc.y * qx > sc.x * qz)
                d2d = (new Vector2(qx, qz) - sc * radius).magnitude - thickness;
            else
                d2d = Mathf.Abs(new Vector2(qx, qz).magnitude - radius) - thickness;
            float dY = Mathf.Abs(p.y) - halfHeight;
            return Mathf.Min(Mathf.Max(d2d, dY), 0f)
                 + new Vector2(Mathf.Max(d2d, 0f), Mathf.Max(dY, 0f)).magnitude;
        }

        /// <summary>Crescent moon (two circles subtracted) in XZ, extruded Y.</summary>
        public static float Moon(Vector3 p, float d, float ra, float rb, float halfHeight)
        {
            float qx = Mathf.Abs(p.x);
            float qz = p.z;
            float dOuter = new Vector2(qx, qz).magnitude - ra;
            float dInner = new Vector2(qx - d, qz).magnitude - rb;
            float d2d = Mathf.Max(dOuter, -dInner);
            float dY = Mathf.Abs(p.y) - halfHeight;
            return Mathf.Min(Mathf.Max(d2d, dY), 0f)
                 + new Vector2(Mathf.Max(d2d, 0f), Mathf.Max(dY, 0f)).magnitude;
        }

        /// <summary>Plus/cross shape in XZ, extruded Y.</summary>
        public static float CrossShape(Vector3 p, float len, float thickness, float roundRadius, float halfHeight)
        {
            float qx = Mathf.Abs(p.x);
            float qz = Mathf.Abs(p.z);
            Vector2 dH = new Vector2(qx - len, qz - thickness);
            Vector2 dV = new Vector2(qx - thickness, qz - len);
            float dHsdf = new Vector2(Mathf.Max(dH.x, 0f), Mathf.Max(dH.y, 0f)).magnitude
                        + Mathf.Min(Mathf.Max(dH.x, dH.y), 0f);
            float dVsdf = new Vector2(Mathf.Max(dV.x, 0f), Mathf.Max(dV.y, 0f)).magnitude
                        + Mathf.Min(Mathf.Max(dV.x, dV.y), 0f);
            float d2d = Mathf.Min(dHsdf, dVsdf) - roundRadius;
            float dY = Mathf.Abs(p.y) - halfHeight;
            return Mathf.Min(Mathf.Max(d2d, dY), 0f)
                 + new Vector2(Mathf.Max(d2d, 0f), Mathf.Max(dY, 0f)).magnitude;
        }

        /// <summary>Exact unsigned distance to 3D triangle.</summary>
        public static float Triangle(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
        {
            Vector3 ba = b - a, pa = p - a;
            Vector3 cb = c - b, pb = p - b;
            Vector3 ac = a - c, pc = p - c;
            Vector3 nor = Vector3.Cross(ba, ac);
            float signA = Mathf.Sign(Vector3.Dot(Vector3.Cross(ba, nor), pa));
            float signB = Mathf.Sign(Vector3.Dot(Vector3.Cross(cb, nor), pb));
            float signC = Mathf.Sign(Vector3.Dot(Vector3.Cross(ac, nor), pc));
            float d2;
            if (signA + signB + signC < 2f)
            {
                float tAB = Mathf.Clamp01(Vector3.Dot(ba, pa) / Vector3.Dot(ba, ba));
                float dAB = (ba * tAB - pa).sqrMagnitude;
                float tBC = Mathf.Clamp01(Vector3.Dot(cb, pb) / Vector3.Dot(cb, cb));
                float dBC = (cb * tBC - pb).sqrMagnitude;
                float tCA = Mathf.Clamp01(Vector3.Dot(ac, pc) / Vector3.Dot(ac, ac));
                float dCA = (ac * tCA - pc).sqrMagnitude;
                d2 = Mathf.Min(dAB, Mathf.Min(dBC, dCA));
            }
            else
            {
                float dn = Vector3.Dot(nor, pa);
                d2 = dn * dn / Vector3.Dot(nor, nor);
            }
            return Mathf.Sqrt(d2);
        }

        /// <summary>Quadratic Bezier curve with tube radius.</summary>
        public static float Bezier(Vector3 pos, Vector3 a, Vector3 b, Vector3 c, float radius)
        {
            Vector3 ab = b - a;
            Vector3 ba2c = a - 2f * b + c;
            Vector3 cv = ab * 2f;
            Vector3 dv = a - pos;
            float ba2cDot = Vector3.Dot(ba2c, ba2c);
            if (ba2cDot < 1e-10f)
            {
                Vector3 ac = c - a;
                float acDot = Vector3.Dot(ac, ac);
                if (acDot < 1e-10f) return (pos - a).magnitude - radius;
                float t = Mathf.Clamp01(Vector3.Dot(pos - a, ac) / acDot);
                return (pos - a - ac * t).magnitude - radius;
            }
            float kk = 1f / ba2cDot;
            float kx = kk * Vector3.Dot(ab, ba2c);
            float ky = kk * (2f * Vector3.Dot(ab, ab) + Vector3.Dot(dv, ba2c)) / 3f;
            float kz = kk * Vector3.Dot(dv, ab);
            float p2 = ky - kx * kx;
            float p3 = p2 * p2 * p2;
            float q2 = kx * (2f * kx * kx - 3f * ky) + kz;
            float h = q2 * q2 + 4f * p3;
            float res;
            if (h >= 0f)
            {
                float hSqrt = Mathf.Sqrt(h);
                float x0 = (hSqrt - q2) * 0.5f;
                float x1 = (-hSqrt - q2) * 0.5f;
                float uvX = Mathf.Sign(x0) * Mathf.Pow(Mathf.Abs(x0), 0.333333f);
                float uvY = Mathf.Sign(x1) * Mathf.Pow(Mathf.Abs(x1), 0.333333f);
                float t = Mathf.Clamp01(uvX + uvY - kx);
                res = (dv + (cv + ba2c * t) * t).magnitude;
            }
            else
            {
                float z = Mathf.Sqrt(-p2);
                float v = Mathf.Acos(q2 / (p2 * z * 2f)) / 3f;
                float m = Mathf.Cos(v);
                float n = Mathf.Sin(v) * 1.7320508f;
                float t0 = Mathf.Clamp01((m + m) * z - kx);
                float t1 = Mathf.Clamp01((-n - m) * z - kx);
                float d0 = (dv + (cv + ba2c * t0) * t0).magnitude;
                float d1 = (dv + (cv + ba2c * t1) * t1).magnitude;
                res = Mathf.Min(d0, d1);
            }
            return res - radius;
        }

        /// <summary>U-shaped horseshoe.</summary>
        public static float Horseshoe(Vector3 p, float angle, float radius, float halfLength, float width, float thickness)
        {
            Vector2 cc = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
            float px = Mathf.Abs(p.x);
            float l = Mathf.Sqrt(px * px + p.y * p.y);
            float qx = -cc.x * px + cc.y * p.y;
            float qy = cc.y * px + cc.x * p.y;
            if (!(qy > 0f || qx > 0f)) qx = l * Mathf.Sign(-cc.x);
            if (qx <= 0f) qy = l;
            qx = Mathf.Abs(qx);
            qy -= radius;
            float rx = Mathf.Max(qx - halfLength, 0f);
            float innerLen = new Vector2(rx, qy).magnitude + Mathf.Min(Mathf.Max(qx - halfLength, qy), 0f);
            float dxW = Mathf.Max(innerLen - width, 0f);
            float dyT = Mathf.Max(Mathf.Abs(p.z) - thickness, 0f);
            return -Mathf.Min(width, thickness) + new Vector2(dxW, dyT).magnitude
                 + Mathf.Min(Mathf.Max(innerLen - width, Mathf.Abs(p.z) - thickness), 0f);
        }

        /// <summary>Superellipsoid (generalized ellipsoid).</summary>
        public static float Superellipsoid(Vector3 p, Vector3 halfExtents, float e1, float e2)
        {
            e1 = Mathf.Max(e1, 0.02f);
            e2 = Mathf.Max(e2, 0.02f);
            float qx = Mathf.Max(Mathf.Abs(p.x / halfExtents.x), 1e-10f);
            float qy = Mathf.Max(Mathf.Abs(p.y / halfExtents.y), 1e-10f);
            float qz = Mathf.Max(Mathf.Abs(p.z / halfExtents.z), 1e-10f);
            float m1 = 2f / e2;
            float m2 = 2f / e1;
            float w = Mathf.Pow(qx, m1) + Mathf.Pow(qz, m1);
            float v = Mathf.Pow(w, e2 / e1) + Mathf.Pow(qy, m2);
            float f = Mathf.Pow(v, e1 * 0.5f);
            float minE = Mathf.Min(halfExtents.x, Mathf.Min(halfExtents.y, halfExtents.z));
            return (f - 1f) * minE * 0.5f;
        }

        /// <summary>Trapezoid in XY, extruded Z.</summary>
        public static float Trapezoid(Vector3 p, float r1, float r2, float trapHeight, float halfDepth)
        {
            float px = Mathf.Abs(p.x);
            float py = p.y;
            float he = trapHeight;
            Vector2 slantDir = new Vector2(2f * he, r1 - r2);
            float slantLen = slantDir.magnitude;
            Vector2 slantN = slantLen > 1e-10f ? slantDir / slantLen : Vector2.up;
            float dSlant = Vector2.Dot(new Vector2(px - r1, py + he), slantN);
            float dBot = new Vector2(Mathf.Max(px - r1, 0f), Mathf.Max(-py - he, 0f)).magnitude
                       + Mathf.Min(Mathf.Max(px - r1, -py - he), 0f);
            float dTop = new Vector2(Mathf.Max(px - r2, 0f), Mathf.Max(py - he, 0f)).magnitude
                       + Mathf.Min(Mathf.Max(px - r2, py - he), 0f);
            float dUnsigned = Mathf.Min(Mathf.Min(Mathf.Abs(dBot), Mathf.Abs(dSlant)), Mathf.Abs(dTop));
            bool inside = py >= -he && py <= he && dSlant <= 0f;
            float d2d = inside ? -dUnsigned : dUnsigned;
            float dZ = Mathf.Abs(p.z) - halfDepth;
            return Mathf.Min(Mathf.Max(d2d, dZ), 0f)
                 + new Vector2(Mathf.Max(d2d, 0f), Mathf.Max(dZ, 0f)).magnitude;
        }

        /// <summary>Parallelogram in XY (with skew), extruded Z.</summary>
        public static float Parallelogram(Vector3 p, float width, float paraHeight, float skew, float halfDepth)
        {
            float he = paraHeight;
            float qx = p.x - p.y * skew / he;
            float d2dX = Mathf.Abs(qx) - width;
            float d2dY = Mathf.Abs(p.y) - he;
            float d2d = new Vector2(Mathf.Max(d2dX, 0f), Mathf.Max(d2dY, 0f)).magnitude
                      + Mathf.Min(Mathf.Max(d2dX, d2dY), 0f);
            float dZ = Mathf.Abs(p.z) - halfDepth;
            return Mathf.Min(Mathf.Max(d2d, dZ), 0f)
                 + new Vector2(Mathf.Max(d2d, 0f), Mathf.Max(dZ, 0f)).magnitude;
        }

        /// <summary>Organic blobby cross in XZ, extruded Y.</summary>
        public static float BlobbyCross(Vector3 p, float size, float halfHeight)
        {
            float qx = Mathf.Abs(p.x) / size;
            float qz = Mathf.Abs(p.z) / size;
            float n = qx + qz;
            float d2d;
            if (n < 1f)
            {
                float t = 1f - n;
                float bb = qx * qz;
                d2d = (-Mathf.Sqrt(Mathf.Max(t * t - 2f * bb, 0f)) + n - 1f) * size * 0.70710678f;
            }
            else
            {
                float dx = Mathf.Max(qx - 1f, 0f);
                float dz = Mathf.Max(qz - 1f, 0f);
                float dxLen = new Vector2(qx - 1f, qz).magnitude;
                float dzLen = new Vector2(qx, qz - 1f).magnitude;
                d2d = Mathf.Min(dxLen, Mathf.Min(dzLen, Mathf.Sqrt(dx * dx + dz * dz))) * size;
            }
            float dY = Mathf.Abs(p.y) - halfHeight;
            return Mathf.Min(Mathf.Max(d2d, dY), 0f)
                 + new Vector2(Mathf.Max(d2d, 0f), Mathf.Max(dY, 0f)).magnitude;
        }

        /// <summary>Parabolic arch in XY, extruded Z (Newton's method).</summary>
        public static float ParabolaSegment(Vector3 p, float width, float paraHeight, float halfDepth)
        {
            float px = Mathf.Abs(p.x);
            float py = p.y;
            float w = width;
            float h = paraHeight;
            float ww = w * w;
            float t = Mathf.Clamp(px, 0f, w);
            for (int i = 0; i < 8; i++)
            {
                float ft = h * (1f - t * t / ww);
                float dft = -2f * h * t / ww;
                float ex = px - t;
                float ey = py - ft;
                float f = -ex + ey * dft;
                float df = 1f + dft * dft + ey * (-2f * h / ww);
                if (Mathf.Abs(df) > 1e-10f) t = Mathf.Clamp(t - f / df, 0f, w);
            }
            float closestY = h * (1f - t * t / ww);
            float dPara = new Vector2(px - t, py - closestY).magnitude;
            float dBase = (px <= w) ? Mathf.Abs(py) : new Vector2(px - w, py).magnitude;
            float dUnsigned = Mathf.Min(dPara, dBase);
            float yArch = (px <= w) ? h * (1f - (px / w) * (px / w)) : 0f;
            bool inside = px <= w && py >= 0f && py <= yArch;
            float d2d = inside ? -dUnsigned : dUnsigned;
            float dZ = Mathf.Abs(p.z) - halfDepth;
            return Mathf.Min(Mathf.Max(d2d, dZ), 0f)
                 + new Vector2(Mathf.Max(d2d, 0f), Mathf.Max(dZ, 0f)).magnitude;
        }

        /// <summary>Regular N-sided polygon in XZ, extruded Y.</summary>
        public static float RegularPolygon(Vector3 p, float radius, float nSides, float halfHeight)
        {
            float qx = Mathf.Abs(p.x);
            float qz = p.z;
            float n = Mathf.Max(nSides, 3f);
            float an = Mathf.PI / n;
            float he = radius * Mathf.Cos(an);
            float angle = Mathf.Atan2(qz, qx);
            float bn = an * Mathf.Floor((angle + an) / (2f * an));
            float cosB = Mathf.Cos(bn);
            float sinB = Mathf.Sin(bn);
            float rx = cosB * qx + sinB * qz;
            float d2d = rx - he;
            float dY = Mathf.Abs(p.y) - halfHeight;
            return Mathf.Min(Mathf.Max(d2d, dY), 0f)
                 + new Vector2(Mathf.Max(d2d, 0f), Mathf.Max(dY, 0f)).magnitude;
        }

        /// <summary>Star polygon in XZ, extruded Y.</summary>
        public static float StarPolygon(Vector3 p, float radius, float nPoints, float m, float halfHeight)
        {
            float qx = Mathf.Abs(p.x);
            float qz = p.z;
            float n = Mathf.Max(nPoints, 3f);
            float an = Mathf.PI / n;
            float r = new Vector2(qx, qz).magnitude;
            float angle = Mathf.Atan2(qz, qx);
            angle = ((angle % (2f * an)) + 2f * an) % (2f * an);
            if (angle > an) angle = 2f * an - angle;
            Vector2 pt = new Vector2(r * Mathf.Cos(angle), r * Mathf.Sin(angle));
            Vector2 aa = new Vector2(radius, 0f);
            Vector2 bb = new Vector2(m * Mathf.Cos(an), m * Mathf.Sin(an));
            Vector2 ab = bb - aa;
            Vector2 ap = pt - aa;
            float t = Mathf.Clamp01(Vector2.Dot(ap, ab) / Vector2.Dot(ab, ab));
            Vector2 closest = aa + ab * t;
            float dist = (pt - closest).magnitude;
            float crossV = ab.x * ap.y - ab.y * ap.x;
            float d2d = (crossV > 0f) ? -dist : dist;
            float dY = Mathf.Abs(p.y) - halfHeight;
            return Mathf.Min(Mathf.Max(d2d, dY), 0f)
                 + new Vector2(Mathf.Max(d2d, 0f), Mathf.Max(dY, 0f)).magnitude;
        }

        /// <summary>Staircase in XY, extruded Z.</summary>
        public static float Stairs(Vector3 p, float stepWidth, float stepHeight, float nSteps, float halfDepth)
        {
            float sw = stepWidth;
            float sh = stepHeight;
            float n = Mathf.Max(nSteps, 1f);
            float tw = n * sw;
            float th = n * sh;
            float lx = p.x + tw * 0.5f;
            float ly = p.y + th * 0.5f;
            float si = Mathf.Clamp(Mathf.Floor(lx / sw), 0f, n - 1f);
            float d2d = StairsStepBox(lx, ly, si, sw, sh);
            if (si > 0f) d2d = Mathf.Min(d2d, StairsStepBox(lx, ly, si - 1f, sw, sh));
            if (si < n - 1f) d2d = Mathf.Min(d2d, StairsStepBox(lx, ly, si + 1f, sw, sh));
            float sj = Mathf.Clamp(Mathf.Ceil(ly / sh) - 1f, 0f, n - 1f);
            if (sj != si && sj != si - 1f && sj != si + 1f)
                d2d = Mathf.Min(d2d, StairsStepBox(lx, ly, sj, sw, sh));
            float dZ = Mathf.Abs(p.z) - halfDepth;
            return Mathf.Min(Mathf.Max(d2d, dZ), 0f)
                 + new Vector2(Mathf.Max(d2d, 0f), Mathf.Max(dZ, 0f)).magnitude;
        }

        private static float StairsStepBox(float lx, float ly, float s, float sw, float sh)
        {
            float cx = s * sw + sw * 0.5f;
            float hy = (s + 1f) * sh * 0.5f;
            float dx = Mathf.Abs(lx - cx) - sw * 0.5f;
            float dy = Mathf.Abs(ly - hy) - hy;
            return new Vector2(Mathf.Max(dx, 0f), Mathf.Max(dy, 0f)).magnitude
                 + Mathf.Min(Mathf.Max(dx, dy), 0f);
        }

        /// <summary>Helix (spiral tube) along Y-axis.</summary>
        public static float Helix(Vector3 p, float majorR, float minorR, float pitch, float halfHeight)
        {
            float rXZ = SdfMath.LengthXZ(p);
            float theta = Mathf.Atan2(p.z, p.x);
            float py = p.y;
            float tau = 2f * Mathf.PI;
            float dRadial = rXZ - majorR;
            float yAtTheta = theta * pitch / tau;
            float k = Mathf.Round((py - yAtTheta) / pitch);
            float dTube = float.MaxValue;
            for (int dk = -1; dk <= 1; dk++)
            {
                float yHelix = yAtTheta + (k + dk) * pitch;
                float dy = py - yHelix;
                float d = new Vector2(dRadial, dy).magnitude - minorR;
                dTube = Mathf.Min(dTube, d);
            }
            float dCap = Mathf.Abs(py) - halfHeight;
            return Mathf.Max(dTube, dCap);
        }

        /// <summary>XOR (symmetric difference) of two distances.</summary>
        public static float Xor(float d1, float d2)
        {
            return Mathf.Max(Mathf.Min(d1, d2), -Mathf.Max(d1, d2));
        }

        /// <summary>Morph (linear interpolation between two shapes).</summary>
        public static float Morph(float d1, float d2, float t)
        {
            return d1 * (1f - t) + d2 * t;
        }

        // =====================================================================
        // Boolean Operations
        // =====================================================================

        /// <summary>Union (min).</summary>
        public static float Union(float d1, float d2)
        {
            return Mathf.Min(d1, d2);
        }

        /// <summary>Intersection (max).</summary>
        public static float Intersection(float d1, float d2)
        {
            return Mathf.Max(d1, d2);
        }

        /// <summary>Subtraction: d1 minus d2.</summary>
        public static float Subtraction(float d1, float d2)
        {
            return Mathf.Max(d1, -d2);
        }

        /// <summary>Smooth union with blending factor k.</summary>
        public static float SmoothUnion(float d1, float d2, float k)
        {
            if (k < 0.0001f) return Mathf.Min(d1, d2);
            float invK = 1f / k;
            float h = Mathf.Max(k - Mathf.Abs(d1 - d2), 0f) * invK;
            return Mathf.Min(d1, d2) - h * h * k * 0.25f;
        }

        /// <summary>Smooth intersection with blending factor k.</summary>
        public static float SmoothIntersection(float d1, float d2, float k)
        {
            if (k < 0.0001f) return Mathf.Max(d1, d2);
            float invK = 1f / k;
            float h = Mathf.Max(k - Mathf.Abs(d1 - d2), 0f) * invK;
            return Mathf.Max(d1, d2) + h * h * k * 0.25f;
        }

        /// <summary>Smooth subtraction with blending factor k.</summary>
        public static float SmoothSubtraction(float d1, float d2, float k)
        {
            if (k < 0.0001f) return Mathf.Max(d1, -d2);
            float invK = 1f / k;
            float h = Mathf.Max(k - Mathf.Abs(d1 + d2), 0f) * invK;
            return Mathf.Max(d1, -d2) + h * h * k * 0.25f;
        }

        // =====================================================================
        // Space Transforms
        // =====================================================================

        /// <summary>Infinite repetition (space folding).</summary>
        public static Vector3 RepeatInfinite(Vector3 p, Vector3 spacing)
        {
            return SdfMath.Fmod(p + spacing * 0.5f, spacing) - spacing * 0.5f;
        }

        /// <summary>Finite repetition with count.</summary>
        public static Vector3 RepeatFinite(Vector3 p, Vector3 spacing, Vector3 count)
        {
            Vector3 r = SdfMath.Clamp(
                SdfMath.Round(SdfMath.Divide(p, spacing)),
                -count, count
            );
            return p - SdfMath.Multiply(spacing, r);
        }

        // =====================================================================
        // Modifiers
        // =====================================================================

        /// <summary>Twist around Y-axis. Returns transformed point.</summary>
        public static Vector3 Twist(Vector3 p, float strength)
        {
            float angle = strength * p.y;
            float c = Mathf.Cos(angle);
            float s = Mathf.Sin(angle);
            return new Vector3(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
        }

        /// <summary>Bend along X-axis. Returns transformed point.</summary>
        public static Vector3 Bend(Vector3 p, float curvature)
        {
            float angle = curvature * p.x;
            float c = Mathf.Cos(angle);
            float s = Mathf.Sin(angle);
            return new Vector3(c * p.x + s * p.y, c * p.y - s * p.x, p.z);
        }

        /// <summary>Round: soften edges by subtracting radius from distance.</summary>
        public static float Round(float d, float radius)
        {
            return d - radius;
        }

        /// <summary>Onion: hollow shell with thickness.</summary>
        public static float Onion(float d, float thickness)
        {
            return Mathf.Abs(d) - thickness;
        }

        /// <summary>Polar repeat around Y-axis.</summary>
        public static Vector3 PolarRepeat(Vector3 p, float count)
        {
            float sector = 2f * Mathf.PI / count;
            float a = Mathf.Atan2(p.z, p.x) + sector * 0.5f;
            float r = SdfMath.LengthXZ(p);
            a = a % sector - sector * 0.5f;
            return new Vector3(r * Mathf.Cos(a), p.y, r * Mathf.Sin(a));
        }

        /// <summary>Taper XZ cross-section along Y.</summary>
        public static Vector3 Taper(Vector3 p, float factor)
        {
            float s = 1f / (1f - p.y * factor);
            return new Vector3(p.x * s, p.y, p.z * s);
        }

        /// <summary>Sin-based oscillating displacement.</summary>
        public static float Displacement(float d, Vector3 p, float strength)
        {
            return d + Mathf.Sin(p.x * 5f) * Mathf.Sin(p.y * 5f) * Mathf.Sin(p.z * 5f) * strength;
        }

        /// <summary>Mirror point along specified axes.</summary>
        public static Vector3 Symmetry(Vector3 p, Vector3 axes)
        {
            return new Vector3(
                axes.x > 0.5f ? Mathf.Abs(p.x) : p.x,
                axes.y > 0.5f ? Mathf.Abs(p.y) : p.y,
                axes.z > 0.5f ? Mathf.Abs(p.z) : p.z
            );
        }

        /// <summary>Box elongation (pre-processing transform).</summary>
        public static Vector3 Elongate(Vector3 p, Vector3 amount)
        {
            return new Vector3(
                p.x - Mathf.Clamp(p.x, -amount.x, amount.x),
                p.y - Mathf.Clamp(p.y, -amount.y, amount.y),
                p.z - Mathf.Clamp(p.z, -amount.z, amount.z)
            );
        }

        // =====================================================================
        // Gradient (Normal) Estimation
        // =====================================================================

        /// <summary>
        /// Estimate SDF gradient at point p using central differences.
        /// The evaluate function should return the SDF distance for a given point.
        /// </summary>
        public static Vector3 Gradient(Vector3 p, System.Func<Vector3, float> evaluate, float eps = 0.01f)
        {
            float dx = evaluate(p + Vector3.right * eps) - evaluate(p - Vector3.right * eps);
            float dy = evaluate(p + Vector3.up * eps) - evaluate(p - Vector3.up * eps);
            float dz = evaluate(p + Vector3.forward * eps) - evaluate(p - Vector3.forward * eps);
            Vector3 grad = new Vector3(dx, dy, dz);
            // NaN guard: zero gradient → safe up vector
            float lenSq = grad.sqrMagnitude;
            return lenSq > 0.0001f ? grad / Mathf.Sqrt(lenSq) : Vector3.up;
        }
    }
}
