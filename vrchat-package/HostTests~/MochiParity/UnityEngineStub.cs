// Minimal UnityEngine surface for host-side compile + parity of the Mochi collider.
// Only what SampleMochi_Collider touches on the non-UDONSHARP path; extend it
// when a new sample needs more, never add Unity behaviour it does not have.
using System;
namespace UnityEngine
{
    public struct Vector3
    {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public static readonly Vector3 zero = new Vector3(0, 0, 0);
        public static readonly Vector3 up = new Vector3(0, 1, 0);
        public static readonly Vector3 down = new Vector3(0, -1, 0);
        public float magnitude => (float)Math.Sqrt(x * x + y * y + z * z);
        public float sqrMagnitude => x * x + y * y + z * z;
        public static float Dot(Vector3 a, Vector3 b) => a.x * b.x + a.y * b.y + a.z * b.z;
        public static Vector3 operator +(Vector3 a, Vector3 b) => new Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
        public static Vector3 operator -(Vector3 a, Vector3 b) => new Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
        public static Vector3 operator *(Vector3 a, float s) => new Vector3(a.x * s, a.y * s, a.z * s);
        public static Vector3 operator /(Vector3 a, float s) => new Vector3(a.x / s, a.y / s, a.z / s);
        // Unity: approximately equal (sqr distance < 1e-10)
        public static bool operator ==(Vector3 a, Vector3 b) => (a - b).sqrMagnitude < 9.99999944E-11f;
        public static bool operator !=(Vector3 a, Vector3 b) => !(a == b);
        public override bool Equals(object o) => o is Vector3 v && this == v;
        public override int GetHashCode() => x.GetHashCode() ^ y.GetHashCode() << 2 ^ z.GetHashCode() >> 2;
    }
    public struct Vector4
    {
        public float x, y, z, w;
        public Vector4(float x, float y, float z, float w) { this.x = x; this.y = y; this.z = z; this.w = w; }
        public static readonly Vector4 zero = new Vector4(0, 0, 0, 0);
    }
    public static class Mathf
    {
        public static float Min(float a, float b) => a < b ? a : b;
        public static float Max(float a, float b) => a > b ? a : b;
        public static float Abs(float a) => Math.Abs(a);
        public static float Pow(float a, float b) => (float)Math.Pow(a, b);
        public static float Exp(float a) => (float)Math.Exp(a);
        public static float Lerp(float a, float b, float t) => a + (b - a) * (t < 0 ? 0 : t > 1 ? 1 : t);
    }
    public static class Time { public static float deltaTime = 1f / 90f; }
    public static class Debug { public static void LogWarning(object m) { Console.Error.WriteLine("[warn] " + m); } }
    public class Material
    {
        public void SetVectorArray(string n, Vector4[] v) { }
        public void SetFloat(string n, float v) { }
        public void SetVector(string n, Vector4 v) { }
    }
    public class Component { public T GetComponent<T>() where T : class => null; }
    public class MeshRenderer : Component { public Material material = new Material(); }
    public class MonoBehaviour : Component { }
    [AttributeUsage(AttributeTargets.Field)] public class HeaderAttribute : Attribute { public HeaderAttribute(string h) { } }
    [AttributeUsage(AttributeTargets.Field)] public class TooltipAttribute : Attribute { public TooltipAttribute(string t) { } }
    [AttributeUsage(AttributeTargets.Field)] public class RangeAttribute : Attribute { public RangeAttribute(float a, float b) { } }
}
