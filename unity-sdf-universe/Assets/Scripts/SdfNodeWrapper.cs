// ALICE-SDF Managed Wrapper Layer for Unity
// Provides SdfNode (DSL), CompiledSdf (bytecode VM), and Native (raw FFI)
//
// Author: Moroya Sakamoto

using System;
using System.Runtime.InteropServices;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Ffi = AliceSdfUnity.AliceSdf;
using AliceSdfUnity;

namespace AliceSdf
{
    /// <summary>
    /// Managed wrapper around an SDF node handle.
    /// Provides a fluent DSL for building SDF trees.
    /// </summary>
    public sealed class SdfNode : IDisposable
    {
        private IntPtr _handle;

        internal SdfNode(IntPtr handle) { _handle = handle; }

        public IntPtr Handle => _handle;
        public bool IsValid => _handle != IntPtr.Zero && Ffi.IsValid(_handle);
        public uint NodeCount => Ffi.NodeCount(_handle);

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                Ffi.Free(_handle);
                _handle = IntPtr.Zero;
            }
        }

        ~SdfNode() { Dispose(); }

        private static SdfNode Wrap(IntPtr h) => h == IntPtr.Zero ? null : new SdfNode(h);

        public SdfNode Clone() => Wrap(Ffi.Clone(_handle));

        // ================================================================
        // Primitives
        // ================================================================

        public static SdfNode Sphere(float radius) => Wrap(Ffi.Sphere(radius));
        public static SdfNode Box(float hx, float hy, float hz) => Wrap(Ffi.Box(hx, hy, hz));
        public static SdfNode Box(Vector3 halfExtents) => Wrap(Ffi.Box(halfExtents));
        public static SdfNode Cylinder(float radius, float halfHeight) => Wrap(Ffi.Cylinder(radius, halfHeight));
        public static SdfNode Torus(float majorRadius, float minorRadius) => Wrap(Ffi.Torus(majorRadius, minorRadius));
        public static SdfNode Capsule(Vector3 a, Vector3 b, float radius) => Wrap(Ffi.Capsule(a, b, radius));
        public static SdfNode Plane(Vector3 normal, float distance) => Wrap(Ffi.Plane(normal, distance));
        public static SdfNode Cone(float radius, float halfHeight) => Wrap(Ffi.Cone(radius, halfHeight));
        public static SdfNode Ellipsoid(Vector3 radii) => Wrap(Ffi.Ellipsoid(radii));
        public static SdfNode RoundedCone(float r1, float r2, float halfHeight) => Wrap(Ffi.RoundedCone(r1, r2, halfHeight));
        public static SdfNode Pyramid(float halfHeight) => Wrap(Ffi.Pyramid(halfHeight));
        public static SdfNode Octahedron(float size) => Wrap(Ffi.Octahedron(size));
        public static SdfNode HexPrism(float hexRadius, float halfHeight) => Wrap(Ffi.HexPrism(hexRadius, halfHeight));
        public static SdfNode Link(float halfLength, float r1, float r2) => Wrap(Ffi.Link(halfLength, r1, r2));
        public static SdfNode RoundedBox(Vector3 halfExtents, float roundRadius) => Wrap(Ffi.RoundedBox(halfExtents, roundRadius));
        public static SdfNode CappedCone(float halfHeight, float r1, float r2) => Wrap(Ffi.CappedCone(halfHeight, r1, r2));
        public static SdfNode CappedTorus(float majorRadius, float minorRadius, float capAngle) => Wrap(Ffi.CappedTorus(majorRadius, minorRadius, capAngle));
        public static SdfNode RoundedCylinder(float radius, float roundRadius, float halfHeight) => Wrap(Ffi.RoundedCylinder(radius, roundRadius, halfHeight));
        public static SdfNode TriangularPrism(float width, float halfDepth) => Wrap(Ffi.TriangularPrism(width, halfDepth));
        public static SdfNode CutSphere(float radius, float cutHeight) => Wrap(Ffi.CutSphere(radius, cutHeight));
        public static SdfNode CutHollowSphere(float radius, float cutHeight, float thickness) => Wrap(Ffi.CutHollowSphere(radius, cutHeight, thickness));
        public static SdfNode DeathStar(float ra, float rb, float d) => Wrap(Ffi.DeathStar(ra, rb, d));
        public static SdfNode SolidAngle(float angle, float radius) => Wrap(Ffi.SolidAngle(angle, radius));
        public static SdfNode Heart(float size) => Wrap(Ffi.Heart(size));
        public static SdfNode Barrel(float radius, float halfHeight, float bulge) => Wrap(Ffi.Barrel(radius, halfHeight, bulge));
        public static SdfNode Diamond(float radius, float halfHeight) => Wrap(Ffi.Diamond(radius, halfHeight));
        public static SdfNode Egg(float ra, float rb) => Wrap(Ffi.Egg(ra, rb));
        public static SdfNode Triangle(Vector3 a, Vector3 b, Vector3 c) => Wrap(Ffi.Triangle(a, b, c));
        public static SdfNode Bezier(Vector3 a, Vector3 b, Vector3 c, float radius) => Wrap(Ffi.Bezier(a, b, c, radius));
        public static SdfNode Rhombus(float la, float lb, float halfHeight, float roundRadius) => Wrap(Ffi.Rhombus(la, lb, halfHeight, roundRadius));
        public static SdfNode Horseshoe(float angle, float radius, float halfLength, float width, float thickness) => Wrap(Ffi.Horseshoe(angle, radius, halfLength, width, thickness));
        public static SdfNode Vesica(float radius, float halfDist) => Wrap(Ffi.Vesica(radius, halfDist));
        public static SdfNode InfiniteCylinder(float radius) => Wrap(Ffi.InfiniteCylinder(radius));
        public static SdfNode InfiniteCone(float angle) => Wrap(Ffi.InfiniteCone(angle));
        public static SdfNode Superellipsoid(Vector3 halfExtents, float e1, float e2) => Wrap(Ffi.Superellipsoid(halfExtents, e1, e2));
        public static SdfNode RoundedX(float width, float roundRadius, float halfHeight) => Wrap(Ffi.RoundedX(width, roundRadius, halfHeight));
        public static SdfNode Pie(float angle, float radius, float halfHeight) => Wrap(Ffi.Pie(angle, radius, halfHeight));
        public static SdfNode Trapezoid(float r1, float r2, float trapHeight, float halfDepth) => Wrap(Ffi.Trapezoid(r1, r2, trapHeight, halfDepth));
        public static SdfNode Parallelogram(float width, float paraHeight, float skew, float halfDepth) => Wrap(Ffi.Parallelogram(width, paraHeight, skew, halfDepth));
        public static SdfNode Tunnel(float width, float height2D, float halfDepth) => Wrap(Ffi.Tunnel(width, height2D, halfDepth));
        public static SdfNode UnevenCapsule(float r1, float r2, float capHeight, float halfDepth) => Wrap(Ffi.UnevenCapsule(r1, r2, capHeight, halfDepth));
        public static SdfNode ArcShape(float aperture, float radius, float thickness, float halfHeight) => Wrap(Ffi.ArcShape(aperture, radius, thickness, halfHeight));
        public static SdfNode Moon(float d, float ra, float rb, float halfHeight) => Wrap(Ffi.Moon(d, ra, rb, halfHeight));
        public static SdfNode CrossShape(float length, float thickness, float roundRadius, float halfHeight) => Wrap(Ffi.CrossShape(length, thickness, roundRadius, halfHeight));
        public static SdfNode BlobbyCross(float size, float halfHeight) => Wrap(Ffi.BlobbyCross(size, halfHeight));
        public static SdfNode ParabolaSegment(float width, float paraHeight, float halfDepth) => Wrap(Ffi.ParabolaSegment(width, paraHeight, halfDepth));
        public static SdfNode RegularPolygon(float radius, float nSides, float halfHeight) => Wrap(Ffi.RegularPolygon(radius, nSides, halfHeight));
        public static SdfNode StarPolygon(float radius, float nPoints, float m, float halfHeight) => Wrap(Ffi.StarPolygon(radius, nPoints, m, halfHeight));

        // 2D/Extruded
        public static SdfNode Circle2D(float radius, float halfHeight) => Wrap(Ffi.Circle2D(radius, halfHeight));
        public static SdfNode Rect2D(float halfW, float halfH, float halfHeight) => Wrap(Ffi.Rect2D(halfW, halfH, halfHeight));
        public static SdfNode Segment2D(Vector2 a, Vector2 b, float thickness, float halfHeight) => Wrap(Ffi.Segment2D(a, b, thickness, halfHeight));
        public static SdfNode RoundedRect2D(float halfW, float halfH, float roundRadius, float halfHeight) => Wrap(Ffi.RoundedRect2D(halfW, halfH, roundRadius, halfHeight));
        public static SdfNode Annular2D(float outerRadius, float thickness, float halfHeight) => Wrap(Ffi.Annular2D(outerRadius, thickness, halfHeight));

        // Platonic & Archimedean
        public static SdfNode Tetrahedron(float size) => Wrap(Ffi.Tetrahedron(size));
        public static SdfNode Dodecahedron(float radius) => Wrap(Ffi.Dodecahedron(radius));
        public static SdfNode Icosahedron(float radius) => Wrap(Ffi.Icosahedron(radius));
        public static SdfNode TruncatedOctahedron(float radius) => Wrap(Ffi.TruncatedOctahedron(radius));
        public static SdfNode TruncatedIcosahedron(float radius) => Wrap(Ffi.TruncatedIcosahedron(radius));

        // TPMS
        public static SdfNode Gyroid(float scale, float thickness) => Wrap(Ffi.Gyroid(scale, thickness));
        public static SdfNode SchwarzP(float scale, float thickness) => Wrap(Ffi.SchwarzP(scale, thickness));
        public static SdfNode DiamondSurface(float scale, float thickness) => Wrap(Ffi.DiamondSurface(scale, thickness));
        public static SdfNode Neovius(float scale, float thickness) => Wrap(Ffi.Neovius(scale, thickness));
        public static SdfNode Lidinoid(float scale, float thickness) => Wrap(Ffi.Lidinoid(scale, thickness));
        public static SdfNode IWP(float scale, float thickness) => Wrap(Ffi.IWP(scale, thickness));
        public static SdfNode FRD(float scale, float thickness) => Wrap(Ffi.FRD(scale, thickness));
        public static SdfNode FischerKochS(float scale, float thickness) => Wrap(Ffi.FischerKochS(scale, thickness));
        public static SdfNode PMY(float scale, float thickness) => Wrap(Ffi.PMY(scale, thickness));

        // Structural
        public static SdfNode BoxFrame(Vector3 halfExtents, float edge) => Wrap(Ffi.BoxFrame(halfExtents, edge));
        public static SdfNode Tube(float outerRadius, float thickness, float halfHeight) => Wrap(Ffi.Tube(outerRadius, thickness, halfHeight));
        public static SdfNode ChamferedCube(Vector3 halfExtents, float chamfer) => Wrap(Ffi.ChamferedCube(halfExtents, chamfer));
        public static SdfNode Stairs(float stepWidth, float stepHeight, float numSteps, float halfDepth) => Wrap(Ffi.Stairs(stepWidth, stepHeight, numSteps, halfDepth));
        public static SdfNode Helix(float majorRadius, float minorRadius, float pitch, float halfHeight) => Wrap(Ffi.Helix(majorRadius, minorRadius, pitch, halfHeight));

        // ================================================================
        // Boolean Operations (consume both operands, return new node)
        // ================================================================

        public SdfNode Union(SdfNode other) => Wrap(Ffi.Union(_handle, other._handle));
        public SdfNode Intersection(SdfNode other) => Wrap(Ffi.Intersection(_handle, other._handle));
        public SdfNode Subtract(SdfNode other) => Wrap(Ffi.Subtract(_handle, other._handle));
        public SdfNode SmoothUnion(SdfNode other, float k) => Wrap(Ffi.SmoothUnion(_handle, other._handle, k));
        public SdfNode SmoothIntersection(SdfNode other, float k) => Wrap(Ffi.SmoothIntersection(_handle, other._handle, k));
        public SdfNode SmoothSubtract(SdfNode other, float k) => Wrap(Ffi.SmoothSubtract(_handle, other._handle, k));
        public SdfNode ChamferUnion(SdfNode other, float radius) => Wrap(Ffi.ChamferUnion(_handle, other._handle, radius));
        public SdfNode ChamferIntersection(SdfNode other, float radius) => Wrap(Ffi.ChamferIntersection(_handle, other._handle, radius));
        public SdfNode ChamferSubtract(SdfNode other, float radius) => Wrap(Ffi.ChamferSubtract(_handle, other._handle, radius));
        public SdfNode StairsUnion(SdfNode other, float radius, float steps) => Wrap(Ffi.StairsUnion(_handle, other._handle, radius, steps));
        public SdfNode StairsIntersection(SdfNode other, float radius, float steps) => Wrap(Ffi.StairsIntersection(_handle, other._handle, radius, steps));
        public SdfNode StairsSubtract(SdfNode other, float radius, float steps) => Wrap(Ffi.StairsSubtract(_handle, other._handle, radius, steps));
        public SdfNode ColumnsUnion(SdfNode other, float radius, float count) => Wrap(Ffi.ColumnsUnion(_handle, other._handle, radius, count));
        public SdfNode ColumnsIntersection(SdfNode other, float radius, float count) => Wrap(Ffi.ColumnsIntersection(_handle, other._handle, radius, count));
        public SdfNode ColumnsSubtract(SdfNode other, float radius, float count) => Wrap(Ffi.ColumnsSubtract(_handle, other._handle, radius, count));
        public SdfNode Xor(SdfNode other) => Wrap(Ffi.Xor(_handle, other._handle));
        public SdfNode Morph(SdfNode other, float t) => Wrap(Ffi.Morph(_handle, other._handle, t));
        public SdfNode Pipe(SdfNode other, float radius) => Wrap(Ffi.Pipe(_handle, other._handle, radius));
        public SdfNode Engrave(SdfNode other, float depth) => Wrap(Ffi.Engrave(_handle, other._handle, depth));
        public SdfNode Groove(SdfNode other, float ra, float rb) => Wrap(Ffi.Groove(_handle, other._handle, ra, rb));
        public SdfNode Tongue(SdfNode other, float ra, float rb) => Wrap(Ffi.Tongue(_handle, other._handle, ra, rb));
        public SdfNode ExpSmoothUnion(SdfNode other, float k) => Wrap(Ffi.ExpSmoothUnion(_handle, other._handle, k));
        public SdfNode ExpSmoothIntersection(SdfNode other, float k) => Wrap(Ffi.ExpSmoothIntersection(_handle, other._handle, k));
        public SdfNode ExpSmoothSubtract(SdfNode other, float k) => Wrap(Ffi.ExpSmoothSubtract(_handle, other._handle, k));

        // ================================================================
        // Transforms (return new node, original remains valid)
        // ================================================================

        public SdfNode Translate(Vector3 offset) => Wrap(Ffi.Translate(_handle, offset));
        public SdfNode Rotate(Quaternion rotation) => Wrap(Ffi.Rotate(_handle, rotation));
        public SdfNode RotateEuler(Vector3 euler) => Wrap(Ffi.RotateEuler(_handle, euler));
        public SdfNode Scale(float factor) => Wrap(Ffi.Scale(_handle, factor));
        public SdfNode Scale(Vector3 factors) => Wrap(Ffi.Scale(_handle, factors));
        public SdfNode ScaleNonUniform(float x, float y, float z) => Wrap(Ffi.ScaleNonUniform(_handle, x, y, z));

        // ================================================================
        // Modifiers
        // ================================================================

        public SdfNode Round(float radius) => Wrap(Ffi.Round(_handle, radius));
        public SdfNode Onion(float thickness) => Wrap(Ffi.Onion(_handle, thickness));
        public SdfNode Twist(float strength) => Wrap(Ffi.Twist(_handle, strength));
        public SdfNode Bend(float curvature) => Wrap(Ffi.Bend(_handle, curvature));
        public SdfNode Repeat(Vector3 spacing) => Wrap(Ffi.Repeat(_handle, spacing));
        public SdfNode RepeatFinite(Vector3Int count, Vector3 spacing) => Wrap(Ffi.RepeatFinite(_handle, count, spacing));
        public SdfNode Mirror(bool x, bool y, bool z) => Wrap(Ffi.Mirror(_handle, x, y, z));
        public SdfNode Elongate(Vector3 amount) => Wrap(Ffi.Elongate(_handle, amount));
        public SdfNode Revolution(float offset) => Wrap(Ffi.Revolution(_handle, offset));
        public SdfNode Extrude(float halfHeight) => Wrap(Ffi.Extrude(_handle, halfHeight));
        public SdfNode Noise(float amplitude, float frequency, uint seed) => Wrap(Ffi.Noise(_handle, amplitude, frequency, seed));
        public SdfNode Taper(float factor) => Wrap(Ffi.Taper(_handle, factor));
        public SdfNode Displacement(float strength) => Wrap(Ffi.Displacement(_handle, strength));
        public SdfNode PolarRepeat(uint count) => Wrap(Ffi.PolarRepeat(_handle, count));
        public SdfNode OctantMirror() => Wrap(Ffi.OctantMirror(_handle));
        public SdfNode SweepBezier(Vector2 p0, Vector2 p1, Vector2 p2) => Wrap(Ffi.SweepBezier(_handle, p0, p1, p2));
        public SdfNode Shear(float xy, float xz, float yz) => Wrap(Ffi.Shear(_handle, xy, xz, yz));
        public SdfNode Animated(float speed, float amplitude) => Wrap(Ffi.Animated(_handle, speed, amplitude));
        public SdfNode WithMaterial(uint materialId) => Wrap(Ffi.WithMaterial(_handle, materialId));

        // ================================================================
        // Evaluation
        // ================================================================

        public float Eval(Vector3 point) => Ffi.Eval(_handle, point);
        public float[] EvalBatch(Vector3[] points) => Ffi.EvalBatch(_handle, points);

        // ================================================================
        // Compilation
        // ================================================================

        public CompiledSdf Compile()
        {
            IntPtr compiled = Ffi.Compile(_handle);
            return compiled == IntPtr.Zero ? null : new CompiledSdf(compiled);
        }

        // ================================================================
        // Shader Generation
        // ================================================================

        public string ToGlsl() => Ffi.ToGlsl(_handle);
        public string ToHlsl() => Ffi.ToHlsl(_handle);
        public string ToWgsl() => Ffi.ToWgsl(_handle);

        // ================================================================
        // File I/O
        // ================================================================

        public bool Save(string path) => Ffi.Save(_handle, path);
        public static SdfNode Load(string path) => Wrap(Ffi.Load(path));
    }

    /// <summary>
    /// Compiled SDF bytecode for fast evaluation.
    /// </summary>
    public sealed class CompiledSdf : IDisposable
    {
        private IntPtr _handle;

        internal CompiledSdf(IntPtr handle) { _handle = handle; }

        public IntPtr Handle => _handle;
        public bool IsValid => _handle != IntPtr.Zero;
        public uint InstructionCount => Native.alice_sdf_compiled_instruction_count(_handle);

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                Ffi.FreeCompiled(_handle);
                _handle = IntPtr.Zero;
            }
        }

        ~CompiledSdf() { Dispose(); }

        public float Eval(Vector3 point) => Ffi.EvalCompiled(_handle, point);

        public float[] EvalBatch(Vector3[] points) => Ffi.EvalCompiledBatch(_handle, points);

        public unsafe BatchResult EvalGradientSoA(
            NativeArray<float> posX, NativeArray<float> posY, NativeArray<float> posZ,
            NativeArray<float> normX, NativeArray<float> normY, NativeArray<float> normZ,
            NativeArray<float> distances)
        {
            return Native.alice_sdf_eval_gradient_soa(
                _handle,
                (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(posX),
                (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(posY),
                (float*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(posZ),
                (float*)NativeArrayUnsafeUtility.GetUnsafePtr(normX),
                (float*)NativeArrayUnsafeUtility.GetUnsafePtr(normY),
                (float*)NativeArrayUnsafeUtility.GetUnsafePtr(normZ),
                (float*)NativeArrayUnsafeUtility.GetUnsafePtr(distances),
                (uint)posX.Length);
        }
    }

    /// <summary>
    /// Raw FFI bindings for unsafe/Burst code paths.
    /// </summary>
    public static class Native
    {
#if UNITY_IOS && !UNITY_EDITOR
        private const string Lib = "__Internal";
#elif UNITY_ANDROID && !UNITY_EDITOR
        private const string Lib = "alice_sdf";
#elif UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
        private const string Lib = "libalice_sdf";
#elif UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        private const string Lib = "alice_sdf";
#else
        private const string Lib = "alice_sdf";
#endif

        [DllImport(Lib)]
        public static extern unsafe BatchResult alice_sdf_eval_gradient_soa(
            IntPtr compiled,
            float* x, float* y, float* z,
            float* nx, float* ny, float* nz,
            float* dist,
            uint count);

        [DllImport(Lib)]
        public static extern uint alice_sdf_compiled_instruction_count(IntPtr compiled);
    }
}
