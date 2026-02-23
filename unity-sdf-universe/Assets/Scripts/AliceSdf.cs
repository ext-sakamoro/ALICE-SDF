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
        // Category Counts
        // ============================================================================

        [DllImport(LibraryName)]
        private static extern uint alice_sdf_primitive_count();

        [DllImport(LibraryName)]
        private static extern uint alice_sdf_operation_count();

        [DllImport(LibraryName)]
        private static extern uint alice_sdf_transform_count();

        [DllImport(LibraryName)]
        private static extern uint alice_sdf_modifier_count();

        [DllImport(LibraryName)]
        private static extern uint alice_sdf_total_count();

        /// <summary>Number of primitive SDF variants</summary>
        public static uint PrimitiveCount => alice_sdf_primitive_count();
        /// <summary>Number of operation SDF variants</summary>
        public static uint OperationCount => alice_sdf_operation_count();
        /// <summary>Number of transform SDF variants</summary>
        public static uint TransformCount => alice_sdf_transform_count();
        /// <summary>Number of modifier SDF variants</summary>
        public static uint ModifierCount => alice_sdf_modifier_count();
        /// <summary>Total number of all SDF variants</summary>
        public static uint TotalCount => alice_sdf_total_count();

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

        // --- Basic (continued) ---

        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_cone(float radius, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_ellipsoid(float rx, float ry, float rz);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_rounded_cone(float r1, float r2, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_pyramid(float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_octahedron(float size);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_hex_prism(float hex_radius, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_link(float half_length, float r1, float r2);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_rounded_box(float hx, float hy, float hz, float round_radius);

        public static IntPtr Cone(float radius, float halfHeight) => alice_sdf_cone(radius, halfHeight);
        public static IntPtr Ellipsoid(Vector3 radii) => alice_sdf_ellipsoid(radii.x, radii.y, radii.z);
        public static IntPtr RoundedCone(float r1, float r2, float halfHeight) => alice_sdf_rounded_cone(r1, r2, halfHeight);
        public static IntPtr Pyramid(float halfHeight) => alice_sdf_pyramid(halfHeight);
        public static IntPtr Octahedron(float size) => alice_sdf_octahedron(size);
        public static IntPtr HexPrism(float hexRadius, float halfHeight) => alice_sdf_hex_prism(hexRadius, halfHeight);
        public static IntPtr Link(float halfLength, float r1, float r2) => alice_sdf_link(halfLength, r1, r2);
        public static IntPtr RoundedBox(Vector3 halfExtents, float roundRadius) => alice_sdf_rounded_box(halfExtents.x, halfExtents.y, halfExtents.z, roundRadius);

        // --- Advanced ---

        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_capped_cone(float half_height, float r1, float r2);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_capped_torus(float major_radius, float minor_radius, float cap_angle);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_rounded_cylinder(float radius, float round_radius, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_triangular_prism(float width, float half_depth);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_cut_sphere(float radius, float cut_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_cut_hollow_sphere(float radius, float cut_height, float thickness);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_death_star(float ra, float rb, float d);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_solid_angle(float angle, float radius);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_heart(float size);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_barrel(float radius, float half_height, float bulge);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_diamond(float radius, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_egg(float ra, float rb);

        public static IntPtr CappedCone(float halfHeight, float r1, float r2) => alice_sdf_capped_cone(halfHeight, r1, r2);
        public static IntPtr CappedTorus(float majorRadius, float minorRadius, float capAngle) => alice_sdf_capped_torus(majorRadius, minorRadius, capAngle);
        public static IntPtr RoundedCylinder(float radius, float roundRadius, float halfHeight) => alice_sdf_rounded_cylinder(radius, roundRadius, halfHeight);
        public static IntPtr TriangularPrism(float width, float halfDepth) => alice_sdf_triangular_prism(width, halfDepth);
        public static IntPtr CutSphere(float radius, float cutHeight) => alice_sdf_cut_sphere(radius, cutHeight);
        public static IntPtr CutHollowSphere(float radius, float cutHeight, float thickness) => alice_sdf_cut_hollow_sphere(radius, cutHeight, thickness);
        public static IntPtr DeathStar(float ra, float rb, float d) => alice_sdf_death_star(ra, rb, d);
        public static IntPtr SolidAngle(float angle, float radius) => alice_sdf_solid_angle(angle, radius);
        public static IntPtr Heart(float size) => alice_sdf_heart(size);
        public static IntPtr Barrel(float radius, float halfHeight, float bulge) => alice_sdf_barrel(radius, halfHeight, bulge);
        public static IntPtr Diamond(float radius, float halfHeight) => alice_sdf_diamond(radius, halfHeight);
        public static IntPtr Egg(float ra, float rb) => alice_sdf_egg(ra, rb);

        // --- 2D/Extruded ---

        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_triangle(float ax, float ay, float az, float bx, float by, float bz, float cx, float cy, float cz);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_bezier(float ax, float ay, float az, float bx, float by, float bz, float cx, float cy, float cz, float radius);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_rhombus(float la, float lb, float half_height, float round_radius);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_horseshoe(float angle, float radius, float half_length, float width, float thickness);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_vesica(float radius, float half_dist);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_infinite_cylinder(float radius);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_infinite_cone(float angle);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_superellipsoid(float hx, float hy, float hz, float e1, float e2);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_rounded_x(float width, float round_radius, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_pie(float angle, float radius, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_trapezoid(float r1, float r2, float trap_height, float half_depth);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_parallelogram(float width, float para_height, float skew, float half_depth);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_tunnel(float width, float height_2d, float half_depth);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_uneven_capsule(float r1, float r2, float cap_height, float half_depth);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_arc_shape(float aperture, float radius, float thickness, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_moon(float d, float ra, float rb, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_cross_shape(float length, float thickness, float round_radius, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_blobby_cross(float size, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_parabola_segment(float width, float para_height, float half_depth);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_regular_polygon(float radius, float n_sides, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_star_polygon(float radius, float n_points, float m, float half_height);

        public static IntPtr Triangle(Vector3 a, Vector3 b, Vector3 c) => alice_sdf_triangle(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
        public static IntPtr Bezier(Vector3 a, Vector3 b, Vector3 c, float radius) => alice_sdf_bezier(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, radius);
        public static IntPtr Rhombus(float la, float lb, float halfHeight, float roundRadius) => alice_sdf_rhombus(la, lb, halfHeight, roundRadius);
        public static IntPtr Horseshoe(float angle, float radius, float halfLength, float width, float thickness) => alice_sdf_horseshoe(angle, radius, halfLength, width, thickness);
        public static IntPtr Vesica(float radius, float halfDist) => alice_sdf_vesica(radius, halfDist);
        public static IntPtr InfiniteCylinder(float radius) => alice_sdf_infinite_cylinder(radius);
        public static IntPtr InfiniteCone(float angle) => alice_sdf_infinite_cone(angle);
        public static IntPtr Superellipsoid(Vector3 halfExtents, float e1, float e2) => alice_sdf_superellipsoid(halfExtents.x, halfExtents.y, halfExtents.z, e1, e2);
        public static IntPtr RoundedX(float width, float roundRadius, float halfHeight) => alice_sdf_rounded_x(width, roundRadius, halfHeight);
        public static IntPtr Pie(float angle, float radius, float halfHeight) => alice_sdf_pie(angle, radius, halfHeight);
        public static IntPtr Trapezoid(float r1, float r2, float trapHeight, float halfDepth) => alice_sdf_trapezoid(r1, r2, trapHeight, halfDepth);
        public static IntPtr Parallelogram(float width, float paraHeight, float skew, float halfDepth) => alice_sdf_parallelogram(width, paraHeight, skew, halfDepth);
        public static IntPtr Tunnel(float width, float height2D, float halfDepth) => alice_sdf_tunnel(width, height2D, halfDepth);
        public static IntPtr UnevenCapsule(float r1, float r2, float capHeight, float halfDepth) => alice_sdf_uneven_capsule(r1, r2, capHeight, halfDepth);
        public static IntPtr ArcShape(float aperture, float radius, float thickness, float halfHeight) => alice_sdf_arc_shape(aperture, radius, thickness, halfHeight);
        public static IntPtr Moon(float d, float ra, float rb, float halfHeight) => alice_sdf_moon(d, ra, rb, halfHeight);
        public static IntPtr CrossShape(float length, float thickness, float roundRadius, float halfHeight) => alice_sdf_cross_shape(length, thickness, roundRadius, halfHeight);
        public static IntPtr BlobbyCross(float size, float halfHeight) => alice_sdf_blobby_cross(size, halfHeight);
        public static IntPtr ParabolaSegment(float width, float paraHeight, float halfDepth) => alice_sdf_parabola_segment(width, paraHeight, halfDepth);
        public static IntPtr RegularPolygon(float radius, float nSides, float halfHeight) => alice_sdf_regular_polygon(radius, nSides, halfHeight);
        public static IntPtr StarPolygon(float radius, float nPoints, float m, float halfHeight) => alice_sdf_star_polygon(radius, nPoints, m, halfHeight);

        // --- 2D/Extruded Primitives ---

        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_circle_2d(float radius, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_rect_2d(float half_w, float half_h, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_segment_2d(float ax, float ay, float bx, float by, float thickness, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_rounded_rect_2d(float half_w, float half_h, float round_radius, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_annular_2d(float outer_radius, float thickness, float half_height);

        public static IntPtr Circle2D(float radius, float halfHeight) => alice_sdf_circle_2d(radius, halfHeight);
        public static IntPtr Rect2D(float halfW, float halfH, float halfHeight) => alice_sdf_rect_2d(halfW, halfH, halfHeight);
        public static IntPtr Segment2D(Vector2 a, Vector2 b, float thickness, float halfHeight) => alice_sdf_segment_2d(a.x, a.y, b.x, b.y, thickness, halfHeight);
        public static IntPtr RoundedRect2D(float halfW, float halfH, float roundRadius, float halfHeight) => alice_sdf_rounded_rect_2d(halfW, halfH, roundRadius, halfHeight);
        public static IntPtr Annular2D(float outerRadius, float thickness, float halfHeight) => alice_sdf_annular_2d(outerRadius, thickness, halfHeight);

        // --- Platonic & Archimedean ---

        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_tetrahedron(float size);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_dodecahedron(float radius);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_icosahedron(float radius);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_truncated_octahedron(float radius);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_truncated_icosahedron(float radius);

        public static IntPtr Tetrahedron(float size) => alice_sdf_tetrahedron(size);
        public static IntPtr Dodecahedron(float radius) => alice_sdf_dodecahedron(radius);
        public static IntPtr Icosahedron(float radius) => alice_sdf_icosahedron(radius);
        public static IntPtr TruncatedOctahedron(float radius) => alice_sdf_truncated_octahedron(radius);
        public static IntPtr TruncatedIcosahedron(float radius) => alice_sdf_truncated_icosahedron(radius);

        // --- TPMS ---

        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_gyroid(float scale, float thickness);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_schwarz_p(float scale, float thickness);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_diamond_surface(float scale, float thickness);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_neovius(float scale, float thickness);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_lidinoid(float scale, float thickness);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_iwp(float scale, float thickness);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_frd(float scale, float thickness);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_fischer_koch_s(float scale, float thickness);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_pmy(float scale, float thickness);

        public static IntPtr Gyroid(float scale, float thickness) => alice_sdf_gyroid(scale, thickness);
        public static IntPtr SchwarzP(float scale, float thickness) => alice_sdf_schwarz_p(scale, thickness);
        public static IntPtr DiamondSurface(float scale, float thickness) => alice_sdf_diamond_surface(scale, thickness);
        public static IntPtr Neovius(float scale, float thickness) => alice_sdf_neovius(scale, thickness);
        public static IntPtr Lidinoid(float scale, float thickness) => alice_sdf_lidinoid(scale, thickness);
        public static IntPtr IWP(float scale, float thickness) => alice_sdf_iwp(scale, thickness);
        public static IntPtr FRD(float scale, float thickness) => alice_sdf_frd(scale, thickness);
        public static IntPtr FischerKochS(float scale, float thickness) => alice_sdf_fischer_koch_s(scale, thickness);
        public static IntPtr PMY(float scale, float thickness) => alice_sdf_pmy(scale, thickness);

        // --- Structural ---

        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_box_frame(float hx, float hy, float hz, float edge);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_tube(float outer_radius, float thickness, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_chamfered_cube(float hx, float hy, float hz, float chamfer);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_stairs(float step_width, float step_height, float num_steps, float half_depth);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_helix(float major_radius, float minor_radius, float pitch, float half_height);

        public static IntPtr BoxFrame(Vector3 halfExtents, float edge) => alice_sdf_box_frame(halfExtents.x, halfExtents.y, halfExtents.z, edge);
        public static IntPtr Tube(float outerRadius, float thickness, float halfHeight) => alice_sdf_tube(outerRadius, thickness, halfHeight);
        public static IntPtr ChamferedCube(Vector3 halfExtents, float chamfer) => alice_sdf_chamfered_cube(halfExtents.x, halfExtents.y, halfExtents.z, chamfer);
        public static IntPtr Stairs(float stepWidth, float stepHeight, float numSteps, float halfDepth) => alice_sdf_stairs(stepWidth, stepHeight, numSteps, halfDepth);
        public static IntPtr Helix(float majorRadius, float minorRadius, float pitch, float halfHeight) => alice_sdf_helix(majorRadius, minorRadius, pitch, halfHeight);

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

        // --- Chamfer ---
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_chamfer_union(IntPtr a, IntPtr b, float radius);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_chamfer_intersection(IntPtr a, IntPtr b, float radius);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_chamfer_subtract(IntPtr a, IntPtr b, float radius);

        public static IntPtr ChamferUnion(IntPtr a, IntPtr b, float radius) => alice_sdf_chamfer_union(a, b, radius);
        public static IntPtr ChamferIntersection(IntPtr a, IntPtr b, float radius) => alice_sdf_chamfer_intersection(a, b, radius);
        public static IntPtr ChamferSubtract(IntPtr a, IntPtr b, float radius) => alice_sdf_chamfer_subtract(a, b, radius);

        // --- Stairs ---
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_stairs_union(IntPtr a, IntPtr b, float radius, float steps);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_stairs_intersection(IntPtr a, IntPtr b, float radius, float steps);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_stairs_subtract(IntPtr a, IntPtr b, float radius, float steps);

        public static IntPtr StairsUnion(IntPtr a, IntPtr b, float radius, float steps) => alice_sdf_stairs_union(a, b, radius, steps);
        public static IntPtr StairsIntersection(IntPtr a, IntPtr b, float radius, float steps) => alice_sdf_stairs_intersection(a, b, radius, steps);
        public static IntPtr StairsSubtract(IntPtr a, IntPtr b, float radius, float steps) => alice_sdf_stairs_subtract(a, b, radius, steps);

        // --- Columns ---
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_columns_union(IntPtr a, IntPtr b, float radius, float count);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_columns_intersection(IntPtr a, IntPtr b, float radius, float count);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_columns_subtract(IntPtr a, IntPtr b, float radius, float count);

        public static IntPtr ColumnsUnion(IntPtr a, IntPtr b, float radius, float count) => alice_sdf_columns_union(a, b, radius, count);
        public static IntPtr ColumnsIntersection(IntPtr a, IntPtr b, float radius, float count) => alice_sdf_columns_intersection(a, b, radius, count);
        public static IntPtr ColumnsSubtract(IntPtr a, IntPtr b, float radius, float count) => alice_sdf_columns_subtract(a, b, radius, count);

        // --- Advanced ---
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_xor(IntPtr a, IntPtr b);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_morph(IntPtr a, IntPtr b, float t);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_pipe(IntPtr a, IntPtr b, float radius);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_engrave(IntPtr a, IntPtr b, float depth);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_groove(IntPtr a, IntPtr b, float ra, float rb);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_tongue(IntPtr a, IntPtr b, float ra, float rb);

        public static IntPtr Xor(IntPtr a, IntPtr b) => alice_sdf_xor(a, b);
        public static IntPtr Morph(IntPtr a, IntPtr b, float t) => alice_sdf_morph(a, b, t);
        public static IntPtr Pipe(IntPtr a, IntPtr b, float radius) => alice_sdf_pipe(a, b, radius);
        public static IntPtr Engrave(IntPtr a, IntPtr b, float depth) => alice_sdf_engrave(a, b, depth);
        public static IntPtr Groove(IntPtr a, IntPtr b, float ra, float rb) => alice_sdf_groove(a, b, ra, rb);
        public static IntPtr Tongue(IntPtr a, IntPtr b, float ra, float rb) => alice_sdf_tongue(a, b, ra, rb);

        // --- Exponential Smooth ---
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_exp_smooth_union(IntPtr a, IntPtr b, float k);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_exp_smooth_intersection(IntPtr a, IntPtr b, float k);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_exp_smooth_subtract(IntPtr a, IntPtr b, float k);

        public static IntPtr ExpSmoothUnion(IntPtr a, IntPtr b, float k) => alice_sdf_exp_smooth_union(a, b, k);
        public static IntPtr ExpSmoothIntersection(IntPtr a, IntPtr b, float k) => alice_sdf_exp_smooth_intersection(a, b, k);
        public static IntPtr ExpSmoothSubtract(IntPtr a, IntPtr b, float k) => alice_sdf_exp_smooth_subtract(a, b, k);

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

        [DllImport(LibraryName)]
        private static extern IntPtr alice_sdf_scale_non_uniform(IntPtr node, float x, float y, float z);

        /// <summary>Non-uniform scale (explicit)</summary>
        public static IntPtr ScaleNonUniform(IntPtr node, float x, float y, float z) =>
            alice_sdf_scale_non_uniform(node, x, y, z);

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

        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_repeat_finite(IntPtr node, uint cx, uint cy, uint cz, float sx, float sy, float sz);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_mirror(IntPtr node, byte mx, byte my, byte mz);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_elongate(IntPtr node, float ex, float ey, float ez);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_revolution(IntPtr node, float offset);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_extrude(IntPtr node, float half_height);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_noise(IntPtr node, float amplitude, float frequency, uint seed);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_taper(IntPtr node, float factor);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_displacement(IntPtr node, float strength);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_polar_repeat(IntPtr node, uint count);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_octant_mirror(IntPtr node);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_sweep_bezier(IntPtr node, float p0x, float p0y, float p1x, float p1y, float p2x, float p2y);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_shear(IntPtr node, float xy, float xz, float yz);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_animated(IntPtr node, float speed, float amplitude);
        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_with_material(IntPtr node, uint material_id);

        public static IntPtr Shear(IntPtr node, float xy, float xz, float yz) => alice_sdf_shear(node, xy, xz, yz);
        public static IntPtr Animated(IntPtr node, float speed, float amplitude) => alice_sdf_animated(node, speed, amplitude);

        public static IntPtr RepeatFinite(IntPtr node, Vector3Int count, Vector3 spacing) =>
            alice_sdf_repeat_finite(node, (uint)count.x, (uint)count.y, (uint)count.z, spacing.x, spacing.y, spacing.z);
        public static IntPtr Mirror(IntPtr node, bool x, bool y, bool z) =>
            alice_sdf_mirror(node, (byte)(x ? 1 : 0), (byte)(y ? 1 : 0), (byte)(z ? 1 : 0));
        public static IntPtr Elongate(IntPtr node, Vector3 amount) =>
            alice_sdf_elongate(node, amount.x, amount.y, amount.z);
        public static IntPtr Revolution(IntPtr node, float offset) => alice_sdf_revolution(node, offset);
        public static IntPtr Extrude(IntPtr node, float halfHeight) => alice_sdf_extrude(node, halfHeight);
        public static IntPtr Noise(IntPtr node, float amplitude, float frequency, uint seed) => alice_sdf_noise(node, amplitude, frequency, seed);
        public static IntPtr Taper(IntPtr node, float factor) => alice_sdf_taper(node, factor);
        public static IntPtr Displacement(IntPtr node, float strength) => alice_sdf_displacement(node, strength);
        public static IntPtr PolarRepeat(IntPtr node, uint count) => alice_sdf_polar_repeat(node, count);
        public static IntPtr OctantMirror(IntPtr node) => alice_sdf_octant_mirror(node);
        public static IntPtr SweepBezier(IntPtr node, Vector2 p0, Vector2 p1, Vector2 p2) =>
            alice_sdf_sweep_bezier(node, p0.x, p0.y, p1.x, p1.y, p2.x, p2.y);
        public static IntPtr WithMaterial(IntPtr node, uint materialId) => alice_sdf_with_material(node, materialId);

        // ============================================================================
        // Compilation
        // ============================================================================

        [DllImport(LibraryName)] private static extern IntPtr alice_sdf_compile(IntPtr node);
        [DllImport(LibraryName)] private static extern void alice_sdf_free_compiled(IntPtr compiled);
        [DllImport(LibraryName)] private static extern float alice_sdf_eval_compiled(IntPtr compiled, float x, float y, float z);
        [DllImport(LibraryName)] private static extern BatchResult alice_sdf_eval_compiled_batch(IntPtr compiled, [In] float[] points, [Out] float[] distances, uint count);

        /// <summary>Compile SDF to bytecode for fast evaluation</summary>
        public static IntPtr Compile(IntPtr node) => alice_sdf_compile(node);

        /// <summary>Free a compiled SDF handle</summary>
        public static void FreeCompiled(IntPtr compiled) => alice_sdf_free_compiled(compiled);

        /// <summary>Evaluate compiled SDF at a single point</summary>
        public static float EvalCompiled(IntPtr compiled, Vector3 point) =>
            alice_sdf_eval_compiled(compiled, point.x, point.y, point.z);

        /// <summary>Evaluate compiled SDF at multiple points (fastest AoS path)</summary>
        public static float[] EvalCompiledBatch(IntPtr compiled, Vector3[] points)
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
            var result = alice_sdf_eval_compiled_batch(compiled, pointsFlat, distances, (uint)count);
            if (result.Result != SdfResult.Ok)
            {
                Debug.LogError($"EvalCompiledBatch failed: {result.Result}");
                return null;
            }
            return distances;
        }

        // ============================================================================
        // Mesh Export
        // ============================================================================

        [DllImport(LibraryName)] private static extern SdfResult alice_sdf_export_obj(IntPtr mesh, IntPtr node, string path, uint resolution, float bounds);
        [DllImport(LibraryName)] private static extern SdfResult alice_sdf_export_glb(IntPtr mesh, IntPtr node, string path, uint resolution, float bounds);
        [DllImport(LibraryName)] private static extern SdfResult alice_sdf_export_usda(IntPtr mesh, IntPtr node, string path, uint resolution, float bounds);
        [DllImport(LibraryName)] private static extern SdfResult alice_sdf_export_fbx(IntPtr mesh, IntPtr node, string path, uint resolution, float bounds);

        public static bool ExportObj(IntPtr node, string path, uint resolution = 128, float bounds = 2f) =>
            alice_sdf_export_obj(IntPtr.Zero, node, path, resolution, bounds) == SdfResult.Ok;
        public static bool ExportGlb(IntPtr node, string path, uint resolution = 128, float bounds = 2f) =>
            alice_sdf_export_glb(IntPtr.Zero, node, path, resolution, bounds) == SdfResult.Ok;
        public static bool ExportUsda(IntPtr node, string path, uint resolution = 128, float bounds = 2f) =>
            alice_sdf_export_usda(IntPtr.Zero, node, path, resolution, bounds) == SdfResult.Ok;
        public static bool ExportFbx(IntPtr node, string path, uint resolution = 128, float bounds = 2f) =>
            alice_sdf_export_fbx(IntPtr.Zero, node, path, resolution, bounds) == SdfResult.Ok;

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
