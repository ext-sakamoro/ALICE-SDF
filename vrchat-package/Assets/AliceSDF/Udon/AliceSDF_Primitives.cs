// =============================================================================
// AliceSDF_Primitives.cs - Pure C# SDF Functions for UdonSharp
// =============================================================================
// 22 primitives + operations + transforms ported from ALICE-SDF Rust engine.
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
