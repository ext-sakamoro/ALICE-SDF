// =============================================================================
// AliceSDF_Math.cs - Vector Math Helpers for UdonSharp
// =============================================================================
// Lightweight math utilities that work within UdonSharp constraints.
// No DllImport, no native plugins - pure C# only.
//
// Author: Moroya Sakamoto
// =============================================================================

using UnityEngine;

namespace AliceSDF
{
    /// <summary>
    /// Static math helpers for SDF evaluation in UdonSharp.
    /// All methods are pure C# with no external dependencies.
    /// </summary>
    public static class SdfMath
    {
        /// <summary>
        /// Component-wise absolute value of a Vector3.
        /// </summary>
        public static Vector3 Abs(Vector3 v)
        {
            return new Vector3(
                Mathf.Abs(v.x),
                Mathf.Abs(v.y),
                Mathf.Abs(v.z)
            );
        }

        /// <summary>
        /// Component-wise max of Vector3 and scalar.
        /// </summary>
        public static Vector3 Max(Vector3 v, float s)
        {
            return new Vector3(
                Mathf.Max(v.x, s),
                Mathf.Max(v.y, s),
                Mathf.Max(v.z, s)
            );
        }

        /// <summary>
        /// Component-wise max of two Vector3s.
        /// </summary>
        public static Vector3 Max(Vector3 a, Vector3 b)
        {
            return new Vector3(
                Mathf.Max(a.x, b.x),
                Mathf.Max(a.y, b.y),
                Mathf.Max(a.z, b.z)
            );
        }

        /// <summary>
        /// Component-wise clamp of Vector3.
        /// </summary>
        public static Vector3 Clamp(Vector3 v, Vector3 min, Vector3 max)
        {
            return new Vector3(
                Mathf.Clamp(v.x, min.x, max.x),
                Mathf.Clamp(v.y, min.y, max.y),
                Mathf.Clamp(v.z, min.z, max.z)
            );
        }

        /// <summary>
        /// Component-wise round of Vector3.
        /// </summary>
        public static Vector3 Round(Vector3 v)
        {
            return new Vector3(
                Mathf.Round(v.x),
                Mathf.Round(v.y),
                Mathf.Round(v.z)
            );
        }

        /// <summary>
        /// Component-wise fmod (positive modulo).
        /// </summary>
        public static Vector3 Fmod(Vector3 a, Vector3 b)
        {
            return new Vector3(
                a.x - b.x * Mathf.Floor(a.x / b.x),
                a.y - b.y * Mathf.Floor(a.y / b.y),
                a.z - b.z * Mathf.Floor(a.z / b.z)
            );
        }

        /// <summary>
        /// Component-wise divide.
        /// </summary>
        public static Vector3 Divide(Vector3 a, Vector3 b)
        {
            return new Vector3(a.x / b.x, a.y / b.y, a.z / b.z);
        }

        /// <summary>
        /// Component-wise multiply.
        /// </summary>
        public static Vector3 Multiply(Vector3 a, Vector3 b)
        {
            return new Vector3(a.x * b.x, a.y * b.y, a.z * b.z);
        }

        /// <summary>
        /// Length of XZ components (horizontal distance).
        /// </summary>
        public static float LengthXZ(Vector3 v)
        {
            return Mathf.Sqrt(v.x * v.x + v.z * v.z);
        }

        /// <summary>
        /// Max component of Vector3.
        /// </summary>
        public static float MaxComponent(Vector3 v)
        {
            return Mathf.Max(v.x, Mathf.Max(v.y, v.z));
        }

        /// <summary>
        /// Component-wise min of Vector3 and scalar.
        /// </summary>
        public static Vector3 Min(Vector3 v, float s)
        {
            return new Vector3(
                Mathf.Min(v.x, s),
                Mathf.Min(v.y, s),
                Mathf.Min(v.z, s)
            );
        }

        /// <summary>
        /// Component-wise min of two Vector3s.
        /// </summary>
        public static Vector3 Min(Vector3 a, Vector3 b)
        {
            return new Vector3(
                Mathf.Min(a.x, b.x),
                Mathf.Min(a.y, b.y),
                Mathf.Min(a.z, b.z)
            );
        }

        /// <summary>
        /// Min component of Vector3.
        /// </summary>
        public static float MinComponent(Vector3 v)
        {
            return Mathf.Min(v.x, Mathf.Min(v.y, v.z));
        }

        /// <summary>
        /// Component-wise sign of Vector3.
        /// </summary>
        public static Vector3 Sign(Vector3 v)
        {
            return new Vector3(
                Mathf.Sign(v.x),
                Mathf.Sign(v.y),
                Mathf.Sign(v.z)
            );
        }
    }
}
