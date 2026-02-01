/**
 * ALICE-SDF C# Bindings for Unity
 *
 * Usage:
 * 1. Copy this file to your Unity project's Assets folder
 * 2. Copy libalice_sdf.dylib (macOS), alice_sdf.dll (Windows), or libalice_sdf.so (Linux)
 *    to Assets/Plugins/
 * 3. Use AliceSdf class to create and evaluate SDFs
 *
 * Example:
 *   var sphere = AliceSdf.Sphere(1.0f);
 *   var box = AliceSdf.Box(0.5f, 0.5f, 0.5f);
 *   var shape = AliceSdf.SmoothUnion(sphere, box, 0.2f);
 *   float distance = AliceSdf.Eval(shape, new Vector3(0.5f, 0, 0));
 *   string glsl = AliceSdf.ToGlsl(shape);
 *   AliceSdf.Free(shape);
 *   AliceSdf.Free(box);
 *   AliceSdf.Free(sphere);
 *
 * Author: Moroya Sakamoto
 */

using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace AliceSdfUnity
{
    /// <summary>
    /// Result codes from FFI operations
    /// </summary>
    public enum SdfResult : int
    {
        Ok = 0,
        InvalidHandle = 1,
        NullPointer = 2,
        InvalidParameter = 3,
        OutOfMemory = 4,
        IoError = 5,
        Unknown = 99
    }

    /// <summary>
    /// Version information
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct VersionInfo
    {
        public ushort Major;
        public ushort Minor;
        public ushort Patch;

        public override string ToString() => $"{Major}.{Minor}.{Patch}";
    }

    /// <summary>
    /// Batch evaluation result
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct BatchResult
    {
        public uint Count;
        public SdfResult Result;
    }

    /// <summary>
    /// String result from shader generation
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct StringResult
    {
        public IntPtr Data;
        public uint Len;
        public SdfResult Result;
    }

    /// <summary>
    /// ALICE-SDF Native Library Bindings
    /// </summary>
    public static class AliceSdf
    {
        #if UNITY_IOS && !UNITY_EDITOR
        private const string LibraryName = "__Internal";
        #elif UNITY_ANDROID && !UNITY_EDITOR
        private const string LibraryName = "alice_sdf";
        #elif UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
        private const string LibraryName = "libalice_sdf";
        #elif UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        private const string LibraryName = "alice_sdf";
        #else
        private const string LibraryName = "alice_sdf";
        #endif

        // ============================================================================
        // Library Info
        // ============================================================================

        [DllImport(LibraryName)]
        private static extern VersionInfo alice_sdf_version();

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_version_string();

        /// <summary>
        /// Get library version
        /// </summary>
        public static VersionInfo Version => alice_sdf_version();

        /// <summary>
        /// Get version string
        /// </summary>
        public static string VersionString
        {
            get
            {
                IntPtr ptr = alice_sdf_version_string();
                if (ptr == IntPtr.Zero) return "unknown";
                string result = Marshal.PtrToStringAnsi(ptr);
                alice_sdf_free_string(ptr);
                return result;
            }
        }

        // ============================================================================
        // Primitives
        // ============================================================================

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_sphere(float radius);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_box(float hx, float hy, float hz);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_cylinder(float radius, float half_height);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_torus(float major_radius, float minor_radius);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_capsule(float ax, float ay, float az,
                                                        float bx, float by, float bz,
                                                        float radius);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_plane(float nx, float ny, float nz, float distance);

        /// <summary>Create a sphere SDF</summary>
        public static IntPtr Sphere(float radius) => alice_sdf_sphere(radius);

        /// <summary>Create a box SDF</summary>
        public static IntPtr Box(float hx, float hy, float hz) => alice_sdf_box(hx, hy, hz);

        /// <summary>Create a box SDF</summary>
        public static IntPtr Box(Vector3 halfExtents) => alice_sdf_box(halfExtents.x, halfExtents.y, halfExtents.z);

        /// <summary>Create a cylinder SDF</summary>
        public static IntPtr Cylinder(float radius, float halfHeight) => alice_sdf_cylinder(radius, halfHeight);

        /// <summary>Create a torus SDF</summary>
        public static IntPtr Torus(float majorRadius, float minorRadius) => alice_sdf_torus(majorRadius, minorRadius);

        /// <summary>Create a capsule SDF</summary>
        public static IntPtr Capsule(Vector3 a, Vector3 b, float radius) =>
            alice_sdf_capsule(a.x, a.y, a.z, b.x, b.y, b.z, radius);

        /// <summary>Create a plane SDF</summary>
        public static IntPtr Plane(Vector3 normal, float distance) =>
            alice_sdf_plane(normal.x, normal.y, normal.z, distance);

        // ============================================================================
        // Boolean Operations
        // ============================================================================

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_union(IntPtr a, IntPtr b);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_intersection(IntPtr a, IntPtr b);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_subtract(IntPtr a, IntPtr b);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_smooth_union(IntPtr a, IntPtr b, float k);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_smooth_intersection(IntPtr a, IntPtr b, float k);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_smooth_subtract(IntPtr a, IntPtr b, float k);

        /// <summary>Union of two SDFs (A ∪ B)</summary>
        public static IntPtr Union(IntPtr a, IntPtr b) => alice_sdf_union(a, b);

        /// <summary>Intersection of two SDFs (A ∩ B)</summary>
        public static IntPtr Intersection(IntPtr a, IntPtr b) => alice_sdf_intersection(a, b);

        /// <summary>Subtraction of SDFs (A - B)</summary>
        public static IntPtr Subtract(IntPtr a, IntPtr b) => alice_sdf_subtract(a, b);

        /// <summary>Smooth union of two SDFs</summary>
        public static IntPtr SmoothUnion(IntPtr a, IntPtr b, float k) => alice_sdf_smooth_union(a, b, k);

        /// <summary>Smooth intersection of two SDFs</summary>
        public static IntPtr SmoothIntersection(IntPtr a, IntPtr b, float k) => alice_sdf_smooth_intersection(a, b, k);

        /// <summary>Smooth subtraction of SDFs</summary>
        public static IntPtr SmoothSubtract(IntPtr a, IntPtr b, float k) => alice_sdf_smooth_subtract(a, b, k);

        // ============================================================================
        // Transforms
        // ============================================================================

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_translate(IntPtr node, float x, float y, float z);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_rotate(IntPtr node, float qx, float qy, float qz, float qw);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_rotate_euler(IntPtr node, float x, float y, float z);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_scale(IntPtr node, float factor);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_scale_xyz(IntPtr node, float x, float y, float z);

        /// <summary>Translate an SDF</summary>
        public static IntPtr Translate(IntPtr node, Vector3 offset) =>
            alice_sdf_translate(node, offset.x, offset.y, offset.z);

        /// <summary>Rotate an SDF using quaternion</summary>
        public static IntPtr Rotate(IntPtr node, Quaternion rotation) =>
            alice_sdf_rotate(node, rotation.x, rotation.y, rotation.z, rotation.w);

        /// <summary>Rotate an SDF using Euler angles (radians)</summary>
        public static IntPtr RotateEuler(IntPtr node, Vector3 euler) =>
            alice_sdf_rotate_euler(node, euler.x, euler.y, euler.z);

        /// <summary>Uniform scale an SDF</summary>
        public static IntPtr Scale(IntPtr node, float factor) => alice_sdf_scale(node, factor);

        /// <summary>Non-uniform scale an SDF</summary>
        public static IntPtr Scale(IntPtr node, Vector3 factors) =>
            alice_sdf_scale_xyz(node, factors.x, factors.y, factors.z);

        // ============================================================================
        // Modifiers
        // ============================================================================

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_round(IntPtr node, float radius);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_onion(IntPtr node, float thickness);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_twist(IntPtr node, float strength);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_bend(IntPtr node, float curvature);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_repeat(IntPtr node, float sx, float sy, float sz);

        /// <summary>Apply rounding</summary>
        public static IntPtr Round(IntPtr node, float radius) => alice_sdf_round(node, radius);

        /// <summary>Apply onion (shell) modifier</summary>
        public static IntPtr Onion(IntPtr node, float thickness) => alice_sdf_onion(node, thickness);

        /// <summary>Apply twist modifier</summary>
        public static IntPtr Twist(IntPtr node, float strength) => alice_sdf_twist(node, strength);

        /// <summary>Apply bend modifier</summary>
        public static IntPtr Bend(IntPtr node, float curvature) => alice_sdf_bend(node, curvature);

        /// <summary>Apply infinite repetition</summary>
        public static IntPtr Repeat(IntPtr node, Vector3 spacing) =>
            alice_sdf_repeat(node, spacing.x, spacing.y, spacing.z);

        // ============================================================================
        // Evaluation
        // ============================================================================

        [DllImport(LibraryName)]
        private static extern float alice_sdf_eval(IntPtr node, float x, float y, float z);

        [DllImport(LibraryName)]
        private static extern BatchResult alice_sdf_eval_batch(IntPtr node,
                                                                [In] float[] points,
                                                                [Out] float[] distances,
                                                                uint count);

        /// <summary>Evaluate SDF at a single point</summary>
        public static float Eval(IntPtr node, Vector3 point) =>
            alice_sdf_eval(node, point.x, point.y, point.z);

        /// <summary>Evaluate SDF at multiple points (parallel)</summary>
        public static float[] EvalBatch(IntPtr node, Vector3[] points)
        {
            int count = points.Length;
            float[] pointsFlat = new float[count * 3];
            for (int i = 0; i < count; i++)
            {
                pointsFlat[i * 3 + 0] = points[i].x;
                pointsFlat[i * 3 + 1] = points[i].y;
                pointsFlat[i * 3 + 2] = points[i].z;
            }

            float[] distances = new float[count];
            var result = alice_sdf_eval_batch(node, pointsFlat, distances, (uint)count);

            if (result.Result != SdfResult.Ok)
            {
                Debug.LogError($"EvalBatch failed: {result.Result}");
                return null;
            }

            return distances;
        }

        // ============================================================================
        // Shader Generation
        // ============================================================================

        [DllImport(LibraryName)]
        private static extern StringResult alice_sdf_to_wgsl(IntPtr node);

        [DllImport(LibraryName)]
        private static extern StringResult alice_sdf_to_hlsl(IntPtr node);

        [DllImport(LibraryName)]
        private static extern StringResult alice_sdf_to_glsl(IntPtr node);

        /// <summary>Generate WGSL shader code</summary>
        public static string ToWgsl(IntPtr node)
        {
            var result = alice_sdf_to_wgsl(node);
            if (result.Result != SdfResult.Ok || result.Data == IntPtr.Zero) return null;
            string shader = Marshal.PtrToStringAnsi(result.Data);
            alice_sdf_free_string(result.Data);
            return shader;
        }

        /// <summary>Generate HLSL shader code (for UE5/DirectX)</summary>
        public static string ToHlsl(IntPtr node)
        {
            var result = alice_sdf_to_hlsl(node);
            if (result.Result != SdfResult.Ok || result.Data == IntPtr.Zero) return null;
            string shader = Marshal.PtrToStringAnsi(result.Data);
            alice_sdf_free_string(result.Data);
            return shader;
        }

        /// <summary>Generate GLSL shader code (for Unity/OpenGL)</summary>
        public static string ToGlsl(IntPtr node)
        {
            var result = alice_sdf_to_glsl(node);
            if (result.Result != SdfResult.Ok || result.Data == IntPtr.Zero) return null;
            string shader = Marshal.PtrToStringAnsi(result.Data);
            alice_sdf_free_string(result.Data);
            return shader;
        }

        // ============================================================================
        // File I/O
        // ============================================================================

        [DllImport(LibraryName)]
        private static extern SdfResult alice_sdf_save(IntPtr node, string path);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_load(string path);

        /// <summary>Save SDF to .asdf file</summary>
        public static bool Save(IntPtr node, string path) =>
            alice_sdf_save(node, path) == SdfResult.Ok;

        /// <summary>Load SDF from .asdf file</summary>
        public static IntPtr Load(string path) => alice_sdf_load(path);

        // ============================================================================
        // Memory Management
        // ============================================================================

        [DllImport(LibraryName)]
        private static extern void alice_sdf_free(IntPtr node);

        [DllImport(LibraryName)]
        private static extern void alice_sdf_free_string(IntPtr str);

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_clone(IntPtr node);

        /// <summary>Free an SDF handle</summary>
        public static void Free(IntPtr node) => alice_sdf_free(node);

        /// <summary>Clone an SDF handle</summary>
        public static IntPtr Clone(IntPtr node) => alice_sdf_clone(node);

        // ============================================================================
        // Utilities
        // ============================================================================

        [DllImport(LibraryName)]
        private static extern uint alice_sdf_node_count(IntPtr node);

        [DllImport(LibraryName)]
        private static extern bool alice_sdf_is_valid(IntPtr node);

        /// <summary>Get node count in an SDF tree</summary>
        public static uint NodeCount(IntPtr node) => alice_sdf_node_count(node);

        /// <summary>Check if a handle is valid</summary>
        public static bool IsValid(IntPtr node) => alice_sdf_is_valid(node);
    }
}
