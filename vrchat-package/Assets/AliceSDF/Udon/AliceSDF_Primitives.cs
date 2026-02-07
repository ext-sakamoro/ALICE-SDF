// =============================================================================
// AliceSDF_Primitives.cs - Pure C# SDF Functions for UdonSharp
// =============================================================================
// All 6 primitives + operations + transforms ported from ALICE-SDF Rust engine.
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
            float h = Mathf.Clamp01(Vector3.Dot(pa, ba) / Vector3.Dot(ba, ba));
            return (pa - ba * h).magnitude - radius;
        }

        /// <summary>Capped cone along Y-axis (base radius at y=0, tip at y=height).</summary>
        public static float Cone(Vector3 p, float radius, float height)
        {
            Vector2 q = new Vector2(radius, -height);
            Vector2 w = new Vector2(SdfMath.LengthXZ(p), p.y);
            float qq = Vector2.Dot(q, q);
            Vector2 a = w - q * Mathf.Clamp(Vector2.Dot(w, q) / qq, 0f, 1f);
            Vector2 b = w - q * new Vector2(Mathf.Clamp(w.x / q.x, 0f, 1f), 1f);
            float k = Mathf.Sign(q.y);
            float d = Mathf.Min(Vector2.Dot(a, a), Vector2.Dot(b, b));
            float s = Mathf.Max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
            return Mathf.Sqrt(d) * Mathf.Sign(s);
        }

        /// <summary>Ellipsoid: approximate SDF (exact at surface).</summary>
        public static float Ellipsoid(Vector3 p, Vector3 radii)
        {
            float k0 = SdfMath.Divide(p, radii).magnitude;
            float k1 = SdfMath.Divide(p, SdfMath.Multiply(radii, radii)).magnitude;
            return k0 * (k0 - 1f) / k1;
        }

        /// <summary>Hexagonal prism along Z-axis.</summary>
        public static float HexPrism(Vector3 p, float halfHeight, float radius)
        {
            Vector3 k = new Vector3(-0.8660254038f, 0.5f, 0.57735026919f);
            p = SdfMath.Abs(p);
            float dot_kxy = k.x * p.x + k.y * p.y;
            float m = Mathf.Min(dot_kxy, 0f);
            p.x -= 2f * m * k.x;
            p.y -= 2f * m * k.y;
            float dx = new Vector2(
                p.x - Mathf.Clamp(p.x, -k.z * radius, k.z * radius),
                p.y - radius
            ).magnitude * Mathf.Sign(p.y - radius);
            float dy = p.z - halfHeight;
            return Mathf.Min(Mathf.Max(dx, dy), 0f)
                 + new Vector2(Mathf.Max(dx, 0f), Mathf.Max(dy, 0f)).magnitude;
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

        /// <summary>Exact distance to a 3D triangle (vertices a, b, c).</summary>
        public static float Triangle(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
        {
            Vector3 ba = b - a; Vector3 pa = p - a;
            Vector3 cb = c - b; Vector3 pb = p - b;
            Vector3 ac = a - c; Vector3 pc = p - c;
            Vector3 nor = Vector3.Cross(ba, ac);
            float sba = Mathf.Sign(Vector3.Dot(Vector3.Cross(ba, nor), pa));
            float scb = Mathf.Sign(Vector3.Dot(Vector3.Cross(cb, nor), pb));
            float sac = Mathf.Sign(Vector3.Dot(Vector3.Cross(ac, nor), pc));
            if (sba + scb + sac < 2f)
            {
                float d1 = (ba * Mathf.Clamp(Vector3.Dot(ba, pa) / Vector3.Dot(ba, ba), 0f, 1f) - pa).sqrMagnitude;
                float d2 = (cb * Mathf.Clamp(Vector3.Dot(cb, pb) / Vector3.Dot(cb, cb), 0f, 1f) - pb).sqrMagnitude;
                float d3 = (ac * Mathf.Clamp(Vector3.Dot(ac, pc) / Vector3.Dot(ac, ac), 0f, 1f) - pc).sqrMagnitude;
                return Mathf.Sqrt(Mathf.Min(d1, Mathf.Min(d2, d3)));
            }
            float dn = Vector3.Dot(nor, pa);
            return Mathf.Sqrt(dn * dn / Vector3.Dot(nor, nor));
        }

        /// <summary>Quadratic Bezier curve with radius (analytical cubic solver).</summary>
        public static float Bezier(Vector3 pos, Vector3 A, Vector3 B, Vector3 C, float rad)
        {
            Vector3 a = B - A;
            Vector3 b = A - 2f * B + C;
            Vector3 c2 = a * 2f;
            Vector3 d = A - pos;
            float kk = 1f / Vector3.Dot(b, b);
            float kx = kk * Vector3.Dot(a, b);
            float ky = kk * (2f * Vector3.Dot(a, a) + Vector3.Dot(d, b)) / 3f;
            float kz = kk * Vector3.Dot(d, a);
            float p2 = ky - kx * kx;
            float p3 = p2 * p2 * p2;
            float q2 = kx * (2f * kx * kx - 3f * ky) + kz;
            float h = q2 * q2 + 4f * p3;
            float res;
            if (h >= 0f)
            {
                h = Mathf.Sqrt(h);
                float x0 = (h - q2) * 0.5f;
                float x1 = (-h - q2) * 0.5f;
                float uv0 = Mathf.Sign(x0) * Mathf.Pow(Mathf.Abs(x0), 1f / 3f);
                float uv1 = Mathf.Sign(x1) * Mathf.Pow(Mathf.Abs(x1), 1f / 3f);
                float t = Mathf.Clamp01(uv0 + uv1 - kx);
                res = (d + (c2 + b * t) * t).magnitude;
            }
            else
            {
                float z = Mathf.Sqrt(-p2);
                float v = Mathf.Acos(q2 / (p2 * z * 2f)) / 3f;
                float m = Mathf.Cos(v);
                float n = Mathf.Sin(v) * 1.732050808f;
                float t0 = Mathf.Clamp01((m + m) * z - kx);
                float t1 = Mathf.Clamp01((-n - m) * z - kx);
                float r0 = (d + (c2 + b * t0) * t0).magnitude;
                float r1 = (d + (c2 + b * t1) * t1).magnitude;
                res = Mathf.Min(r0, r1);
            }
            return res - rad;
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
            float invK = 1f / k;
            float h = Mathf.Max(k - Mathf.Abs(d1 - d2), 0f) * invK;
            return Mathf.Min(d1, d2) - h * h * k * 0.25f;
        }

        /// <summary>Smooth intersection with blending factor k.</summary>
        public static float SmoothIntersection(float d1, float d2, float k)
        {
            float invK = 1f / k;
            float h = Mathf.Max(k - Mathf.Abs(d1 - d2), 0f) * invK;
            return Mathf.Max(d1, d2) + h * h * k * 0.25f;
        }

        /// <summary>Smooth subtraction with blending factor k.</summary>
        public static float SmoothSubtraction(float d1, float d2, float k)
        {
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

        /// <summary>Taper: shrink cross-section along Y-axis.</summary>
        public static Vector3 Taper(Vector3 p, float factor)
        {
            float s = 1f / (1f - p.y * factor);
            return new Vector3(p.x * s, p.y, p.z * s);
        }

        /// <summary>Displacement: add sin-based noise to distance field.</summary>
        public static float Displacement(float d, Vector3 p, float strength)
        {
            return d + Mathf.Sin(p.x * 5f) * Mathf.Sin(p.y * 5f) * Mathf.Sin(p.z * 5f) * strength;
        }

        /// <summary>Symmetry: mirror across axes (mask: 1=mirror, 0=no mirror).</summary>
        public static Vector3 Symmetry(Vector3 p, Vector3 mask)
        {
            return new Vector3(
                mask.x > 0.5f ? Mathf.Abs(p.x) : p.x,
                mask.y > 0.5f ? Mathf.Abs(p.y) : p.y,
                mask.z > 0.5f ? Mathf.Abs(p.z) : p.z
            );
        }

        /// <summary>Polar repetition around Y-axis.</summary>
        public static Vector3 PolarRepeat(Vector3 p, float count)
        {
            float angle = 6.283185307f / count;
            float a = Mathf.Atan2(p.z, p.x) + angle * 0.5f;
            float r = SdfMath.LengthXZ(p);
            a = ((a % angle) + angle) % angle - angle * 0.5f;
            return new Vector3(r * Mathf.Cos(a), p.y, r * Mathf.Sin(a));
        }

        /// <summary>Elongate: stretch SDF along axes (apply to point before evaluation).</summary>
        public static Vector3 Elongate(Vector3 p, Vector3 h)
        {
            return new Vector3(
                p.x - Mathf.Clamp(p.x, -h.x, h.x),
                p.y - Mathf.Clamp(p.y, -h.y, h.y),
                p.z - Mathf.Clamp(p.z, -h.z, h.z)
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
            return new Vector3(dx, dy, dz).normalized;
        }
    }
}
