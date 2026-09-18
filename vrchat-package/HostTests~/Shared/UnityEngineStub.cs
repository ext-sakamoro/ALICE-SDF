// Minimal UnityEngine surface for host-side compile + parity of the sample
// colliders (Mochi, TerrainSculpt). Only what they touch on the non-UDONSHARP
// path; extend it when a new sample needs more, never add Unity behaviour it
// does not have.
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
        public static readonly Vector3 forward = new Vector3(0, 0, 1);
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
    public struct Vector2
    {
        public float x, y;
        public Vector2(float x, float y) { this.x = x; this.y = y; }
        public float magnitude => (float)Math.Sqrt(x * x + y * y);
    }
    public struct Color
    {
        public float r, g, b, a;
        public Color(float r, float g, float b, float a) { this.r = r; this.g = g; this.b = b; this.a = a; }
    }
    public struct Vector4
    {
        public float x, y, z, w;
        public Vector4(float x, float y, float z, float w) { this.x = x; this.y = y; this.z = z; this.w = w; }
        public static readonly Vector4 zero = new Vector4(0, 0, 0, 0);
    }
    public static class Mathf
    {
        public const float PI = 3.14159274f;
        public static float Min(float a, float b) => a < b ? a : b;
        public static float Max(float a, float b) => a > b ? a : b;
        public static float Abs(float a) => Math.Abs(a);
        public static float Pow(float a, float b) => (float)Math.Pow(a, b);
        public static float Exp(float a) => (float)Math.Exp(a);
        public static float Lerp(float a, float b, float t) => a + (b - a) * (t < 0 ? 0 : t > 1 ? 1 : t);
        public static float Sqrt(float a) => (float)Math.Sqrt(a);
        public static float Cos(float a) => (float)Math.Cos(a);
        public static float Sin(float a) => (float)Math.Sin(a);
        public static float Floor(float a) => (float)Math.Floor(a);
    }
    // time is settable so a host scenario can step the clock past a cooldown
    public static class Time { public static float deltaTime = 1f / 90f; public static float time = 0f; }
    public struct Quaternion
    {
        public float x, y, z, w;
        public Quaternion(float x, float y, float z, float w) { this.x = x; this.y = y; this.z = z; this.w = w; }
        public static readonly Quaternion identity = new Quaternion(0, 0, 0, 1);
        // Rotation taking a onto b (both unit), as Unity does; a = -b gives a half turn about any perpendicular
        public static Quaternion FromToRotation(Vector3 a, Vector3 b)
        {
            float cx = a.y * b.z - a.z * b.y, cy = a.z * b.x - a.x * b.z, cz = a.x * b.y - a.y * b.x;
            float d = Vector3.Dot(a, b);
            float w = 1f + d;
            if (w < 1e-6f) return new Quaternion(1, 0, 0, 0);
            float len = (float)Math.Sqrt(cx * cx + cy * cy + cz * cz + w * w);
            return new Quaternion(cx / len, cy / len, cz / len, w / len);
        }
    }
    public class Transform
    {
        public Vector3 position;
        public Quaternion rotation = Quaternion.identity;
        public Transform Find(string name) => null;
    }
    public class GameObject
    {
        public Transform transform = new Transform();
        public static GameObject Find(string name) => null;
    }
    public static class Debug
    {
        public static void LogWarning(object m) { Console.Error.WriteLine("[warn] " + m); }
        public static void Log(object m) { Console.WriteLine(m); }
    }
    public class Texture2D { }
    public class Material
    {
        public void SetTexture(string n, Texture2D t) { }
        public void SetVectorArray(string n, Vector4[] v) { }
        public void SetFloat(string n, float v) { }
        public void SetVector(string n, Vector4 v) { }
        public void SetColor(string n, Color c) { }
    }
    public class Component
    {
        public T GetComponent<T>() where T : class => null;
        public Transform transform = new Transform();
    }
    public class MeshRenderer : Component { public Material material = new Material(); }
    public class MonoBehaviour : Component { }
    [AttributeUsage(AttributeTargets.Field)] public class HeaderAttribute : Attribute { public HeaderAttribute(string h) { } }
    [AttributeUsage(AttributeTargets.Field)] public class TooltipAttribute : Attribute { public TooltipAttribute(string t) { } }
    [AttributeUsage(AttributeTargets.Field)] public class RangeAttribute : Attribute { public RangeAttribute(float a, float b) { } }
}
