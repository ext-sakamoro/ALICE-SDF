// =============================================================================
// ALICE-SDF Unity FFI Bindings
// =============================================================================
// High-performance C# bindings for ALICE-SDF Rust library
// Supports: Primitives, Booleans, Transforms, Compilation, Batch Evaluation
//
// Performance Hierarchy:
//   1. alice_sdf_eval_soa       - 1B+ ops/sec (SoA layout)
//   2. alice_sdf_eval_compiled_batch - 500M ops/sec (AoS layout)
//   3. alice_sdf_eval_batch     - 100M ops/sec (auto-compile)
//   4. alice_sdf_eval           - 10M ops/sec (single point)
//
// Author: Moroya Sakamoto
// =============================================================================

using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace AliceSdf
{
    /// <summary>
    /// Result codes for SDF operations
    /// </summary>
    public enum SdfResult : int
    {
        Ok = 0,
        InvalidHandle = 1,
        NullPointer = 2,
        InvalidParameter = 3,
        OutOfMemory = 4,
        IoError = 5,
        CompileError = 6,
        Unknown = 99
    }

    /// <summary>
    /// Batch evaluation result
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct BatchResult
    {
        public uint count;
        public SdfResult result;

        public bool IsOk => result == SdfResult.Ok;
    }

    /// <summary>
    /// Version information
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct VersionInfo
    {
        public ushort major;
        public ushort minor;
        public ushort patch;

        public override string ToString() => $"{major}.{minor}.{patch}";
    }

    /// <summary>
    /// Native ALICE-SDF library bindings
    /// </summary>
    public static class Native
    {
        #if UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
        private const string LibName = "alice_sdf";
        #elif UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        private const string LibName = "alice_sdf.dll";
        #elif UNITY_STANDALONE_LINUX || UNITY_EDITOR_LINUX
        private const string LibName = "alice_sdf.so";
        #else
        private const string LibName = "alice_sdf";
        #endif

        // =====================================================================
        // Library Info
        // =====================================================================

        [DllImport(LibName)] public static extern VersionInfo alice_sdf_version();

        // =====================================================================
        // Primitives
        // =====================================================================

        [DllImport(LibName)] public static extern IntPtr alice_sdf_sphere(float radius);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_box(float hx, float hy, float hz);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_cylinder(float radius, float halfHeight);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_torus(float majorRadius, float minorRadius);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_capsule(
            float ax, float ay, float az,
            float bx, float by, float bz,
            float radius);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_plane(
            float nx, float ny, float nz, float distance);

        // =====================================================================
        // Boolean Operations
        // =====================================================================

        [DllImport(LibName)] public static extern IntPtr alice_sdf_union(IntPtr a, IntPtr b);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_intersection(IntPtr a, IntPtr b);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_subtract(IntPtr a, IntPtr b);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_smooth_union(IntPtr a, IntPtr b, float k);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_smooth_intersection(IntPtr a, IntPtr b, float k);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_smooth_subtract(IntPtr a, IntPtr b, float k);

        // =====================================================================
        // Transforms
        // =====================================================================

        [DllImport(LibName)] public static extern IntPtr alice_sdf_translate(IntPtr node, float x, float y, float z);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_rotate(IntPtr node, float qx, float qy, float qz, float qw);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_rotate_euler(IntPtr node, float x, float y, float z);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_scale(IntPtr node, float factor);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_scale_xyz(IntPtr node, float x, float y, float z);

        // =====================================================================
        // Modifiers
        // =====================================================================

        [DllImport(LibName)] public static extern IntPtr alice_sdf_round(IntPtr node, float radius);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_onion(IntPtr node, float thickness);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_twist(IntPtr node, float strength);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_bend(IntPtr node, float curvature);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_repeat(IntPtr node, float sx, float sy, float sz);

        // =====================================================================
        // Compilation
        // =====================================================================

        [DllImport(LibName)] public static extern IntPtr alice_sdf_compile(IntPtr node);
        [DllImport(LibName)] public static extern void alice_sdf_free_compiled(IntPtr compiled);
        [DllImport(LibName)] public static extern uint alice_sdf_compiled_instruction_count(IntPtr compiled);
        [DllImport(LibName)] public static extern bool alice_sdf_is_compiled_valid(IntPtr compiled);

        // =====================================================================
        // Evaluation
        // =====================================================================

        [DllImport(LibName)] public static extern float alice_sdf_eval(IntPtr node, float x, float y, float z);
        [DllImport(LibName)] public static extern float alice_sdf_eval_compiled(IntPtr compiled, float x, float y, float z);

        // Batch evaluation (AoS layout: [x0,y0,z0, x1,y1,z1, ...])
        [DllImport(LibName)] public static extern BatchResult alice_sdf_eval_batch(
            IntPtr node, float[] points, float[] distances, uint count);

        [DllImport(LibName)] public static extern BatchResult alice_sdf_eval_compiled_batch(
            IntPtr compiled, float[] points, float[] distances, uint count);

        // SoA evaluation (fastest path: separate x[], y[], z[] arrays)
        [DllImport(LibName)] public static extern BatchResult alice_sdf_eval_soa(
            IntPtr compiled, float[] x, float[] y, float[] z, float[] distances, uint count);

        // SoA evaluation with unsafe pointers (for NativeArray support)
        [DllImport(LibName)] public static extern unsafe BatchResult alice_sdf_eval_soa(
            IntPtr compiled, float* x, float* y, float* z, float* distances, uint count);

        // Gradient (Normal) evaluation - THE DEEP FRIED PATH
        [DllImport(LibName)] public static extern BatchResult alice_sdf_eval_gradient_soa(
            IntPtr compiled, float[] x, float[] y, float[] z,
            float[] nx, float[] ny, float[] nz, float[] dist, uint count);

        // Gradient evaluation with unsafe pointers (for NativeArray support)
        [DllImport(LibName)] public static extern unsafe BatchResult alice_sdf_eval_gradient_soa(
            IntPtr compiled, float* x, float* y, float* z,
            float* nx, float* ny, float* nz, float* dist, uint count);

        // =====================================================================
        // Memory Management
        // =====================================================================

        [DllImport(LibName)] public static extern void alice_sdf_free(IntPtr node);
        [DllImport(LibName)] public static extern IntPtr alice_sdf_clone(IntPtr node);
        [DllImport(LibName)] public static extern bool alice_sdf_is_valid(IntPtr node);
        [DllImport(LibName)] public static extern uint alice_sdf_node_count(IntPtr node);
    }

    /// <summary>
    /// Managed SDF node wrapper with automatic disposal
    /// </summary>
    public class SdfNode : IDisposable
    {
        public IntPtr Handle { get; private set; }
        private bool _disposed;

        internal SdfNode(IntPtr handle)
        {
            Handle = handle;
        }

        public bool IsValid => Handle != IntPtr.Zero && Native.alice_sdf_is_valid(Handle);
        public uint NodeCount => Native.alice_sdf_node_count(Handle);

        // Primitives
        public static SdfNode Sphere(float radius) => new SdfNode(Native.alice_sdf_sphere(radius));
        public static SdfNode Box(Vector3 halfExtents) => new SdfNode(Native.alice_sdf_box(halfExtents.x, halfExtents.y, halfExtents.z));
        public static SdfNode Box(float hx, float hy, float hz) => new SdfNode(Native.alice_sdf_box(hx, hy, hz));
        public static SdfNode Cylinder(float radius, float halfHeight) => new SdfNode(Native.alice_sdf_cylinder(radius, halfHeight));
        public static SdfNode Torus(float majorRadius, float minorRadius) => new SdfNode(Native.alice_sdf_torus(majorRadius, minorRadius));
        public static SdfNode Capsule(Vector3 a, Vector3 b, float radius) => new SdfNode(Native.alice_sdf_capsule(a.x, a.y, a.z, b.x, b.y, b.z, radius));
        public static SdfNode Plane(Vector3 normal, float distance) => new SdfNode(Native.alice_sdf_plane(normal.x, normal.y, normal.z, distance));

        // Boolean operations
        public SdfNode Union(SdfNode other) => new SdfNode(Native.alice_sdf_union(Handle, other.Handle));
        public SdfNode Intersection(SdfNode other) => new SdfNode(Native.alice_sdf_intersection(Handle, other.Handle));
        public SdfNode Subtract(SdfNode other) => new SdfNode(Native.alice_sdf_subtract(Handle, other.Handle));
        public SdfNode SmoothUnion(SdfNode other, float k) => new SdfNode(Native.alice_sdf_smooth_union(Handle, other.Handle, k));
        public SdfNode SmoothIntersection(SdfNode other, float k) => new SdfNode(Native.alice_sdf_smooth_intersection(Handle, other.Handle, k));
        public SdfNode SmoothSubtract(SdfNode other, float k) => new SdfNode(Native.alice_sdf_smooth_subtract(Handle, other.Handle, k));

        // Transforms
        public SdfNode Translate(Vector3 offset) => new SdfNode(Native.alice_sdf_translate(Handle, offset.x, offset.y, offset.z));
        public SdfNode Translate(float x, float y, float z) => new SdfNode(Native.alice_sdf_translate(Handle, x, y, z));
        public SdfNode Rotate(Quaternion rotation) => new SdfNode(Native.alice_sdf_rotate(Handle, rotation.x, rotation.y, rotation.z, rotation.w));
        public SdfNode RotateEuler(Vector3 euler) => new SdfNode(Native.alice_sdf_rotate_euler(Handle, euler.x, euler.y, euler.z));
        public SdfNode Scale(float factor) => new SdfNode(Native.alice_sdf_scale(Handle, factor));
        public SdfNode Scale(Vector3 factors) => new SdfNode(Native.alice_sdf_scale_xyz(Handle, factors.x, factors.y, factors.z));

        // Modifiers
        public SdfNode Round(float radius) => new SdfNode(Native.alice_sdf_round(Handle, radius));
        public SdfNode Onion(float thickness) => new SdfNode(Native.alice_sdf_onion(Handle, thickness));
        public SdfNode Twist(float strength) => new SdfNode(Native.alice_sdf_twist(Handle, strength));
        public SdfNode Bend(float curvature) => new SdfNode(Native.alice_sdf_bend(Handle, curvature));
        public SdfNode Repeat(Vector3 spacing) => new SdfNode(Native.alice_sdf_repeat(Handle, spacing.x, spacing.y, spacing.z));

        // Evaluation
        public float Eval(Vector3 point) => Native.alice_sdf_eval(Handle, point.x, point.y, point.z);

        // Compilation
        public CompiledSdf Compile() => new CompiledSdf(Native.alice_sdf_compile(Handle));

        public SdfNode Clone() => new SdfNode(Native.alice_sdf_clone(Handle));

        public void Dispose()
        {
            if (!_disposed && Handle != IntPtr.Zero)
            {
                Native.alice_sdf_free(Handle);
                Handle = IntPtr.Zero;
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~SdfNode() => Dispose();
    }

    /// <summary>
    /// Compiled SDF for high-performance evaluation
    /// </summary>
    public class CompiledSdf : IDisposable
    {
        public IntPtr Handle { get; private set; }
        private bool _disposed;

        internal CompiledSdf(IntPtr handle)
        {
            Handle = handle;
        }

        public bool IsValid => Handle != IntPtr.Zero && Native.alice_sdf_is_compiled_valid(Handle);
        public uint InstructionCount => Native.alice_sdf_compiled_instruction_count(Handle);

        /// <summary>
        /// Evaluate SDF at a single point
        /// </summary>
        public float Eval(Vector3 point) => Native.alice_sdf_eval_compiled(Handle, point.x, point.y, point.z);

        /// <summary>
        /// Batch evaluate using SoA layout (FASTEST PATH)
        /// </summary>
        public BatchResult EvalSoA(float[] x, float[] y, float[] z, float[] distances)
        {
            return Native.alice_sdf_eval_soa(Handle, x, y, z, distances, (uint)x.Length);
        }

        /// <summary>
        /// Batch evaluate using AoS layout
        /// </summary>
        public BatchResult EvalBatch(float[] points, float[] distances)
        {
            return Native.alice_sdf_eval_compiled_batch(Handle, points, distances, (uint)(points.Length / 3));
        }

        /// <summary>
        /// Evaluate gradient (surface normal) using SoA layout - DEEP FRIED PATH
        /// </summary>
        public BatchResult EvalGradientSoA(
            float[] x, float[] y, float[] z,
            float[] nx, float[] ny, float[] nz,
            float[] dist)
        {
            return Native.alice_sdf_eval_gradient_soa(Handle, x, y, z, nx, ny, nz, dist, (uint)x.Length);
        }

        /// <summary>
        /// Batch evaluate using NativeArray (GC-FREE) - THE ULTIMATE PATH
        /// Requires: com.unity.collections package
        /// </summary>
        public unsafe BatchResult EvalSoA(
            Unity.Collections.NativeArray<float> x,
            Unity.Collections.NativeArray<float> y,
            Unity.Collections.NativeArray<float> z,
            Unity.Collections.NativeArray<float> distances)
        {
            return Native.alice_sdf_eval_soa(
                Handle,
                (float*)Unity.Collections.LowLevel.Unsafe.NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(x),
                (float*)Unity.Collections.LowLevel.Unsafe.NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(y),
                (float*)Unity.Collections.LowLevel.Unsafe.NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(z),
                (float*)Unity.Collections.LowLevel.Unsafe.NativeArrayUnsafeUtility.GetUnsafePtr(distances),
                (uint)x.Length);
        }

        /// <summary>
        /// Evaluate gradient using NativeArray (GC-FREE) - THE DEEP FRIED ULTIMATE PATH
        /// Requires: com.unity.collections package
        /// </summary>
        public unsafe BatchResult EvalGradientSoA(
            Unity.Collections.NativeArray<float> x,
            Unity.Collections.NativeArray<float> y,
            Unity.Collections.NativeArray<float> z,
            Unity.Collections.NativeArray<float> nx,
            Unity.Collections.NativeArray<float> ny,
            Unity.Collections.NativeArray<float> nz,
            Unity.Collections.NativeArray<float> dist)
        {
            return Native.alice_sdf_eval_gradient_soa(
                Handle,
                (float*)Unity.Collections.LowLevel.Unsafe.NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(x),
                (float*)Unity.Collections.LowLevel.Unsafe.NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(y),
                (float*)Unity.Collections.LowLevel.Unsafe.NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(z),
                (float*)Unity.Collections.LowLevel.Unsafe.NativeArrayUnsafeUtility.GetUnsafePtr(nx),
                (float*)Unity.Collections.LowLevel.Unsafe.NativeArrayUnsafeUtility.GetUnsafePtr(ny),
                (float*)Unity.Collections.LowLevel.Unsafe.NativeArrayUnsafeUtility.GetUnsafePtr(nz),
                (float*)Unity.Collections.LowLevel.Unsafe.NativeArrayUnsafeUtility.GetUnsafePtr(dist),
                (uint)x.Length);
        }

        public void Dispose()
        {
            if (!_disposed && Handle != IntPtr.Zero)
            {
                Native.alice_sdf_free_compiled(Handle);
                Handle = IntPtr.Zero;
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~CompiledSdf() => Dispose();
    }

    /// <summary>
    /// SoA (Structure of Arrays) buffer for maximum performance
    /// </summary>
    public class SoABuffer : IDisposable
    {
        public float[] X { get; private set; }
        public float[] Y { get; private set; }
        public float[] Z { get; private set; }
        public float[] Distances { get; private set; }
        public int Count { get; private set; }
        public int Capacity { get; private set; }

        public SoABuffer(int capacity)
        {
            Capacity = capacity;
            X = new float[capacity];
            Y = new float[capacity];
            Z = new float[capacity];
            Distances = new float[capacity];
            Count = 0;
        }

        public void Clear() => Count = 0;

        public void Add(Vector3 point)
        {
            if (Count >= Capacity) return;
            X[Count] = point.x;
            Y[Count] = point.y;
            Z[Count] = point.z;
            Count++;
        }

        public void Add(float x, float y, float z)
        {
            if (Count >= Capacity) return;
            X[Count] = x;
            Y[Count] = y;
            Z[Count] = z;
            Count++;
        }

        public void SetCount(int count)
        {
            Count = Mathf.Min(count, Capacity);
        }

        public BatchResult Evaluate(CompiledSdf sdf)
        {
            return sdf.EvalSoA(X, Y, Z, Distances);
        }

        public void Dispose()
        {
            X = null;
            Y = null;
            Z = null;
            Distances = null;
        }
    }
}
