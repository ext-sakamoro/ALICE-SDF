/**
 * ALICE-SDF C++ Bindings for Unreal Engine 5
 *
 * Usage:
 * 1. Copy this file to your UE5 project's Source/ThirdParty/AliceSdf/include/
 * 2. Copy libalice_sdf.dylib (macOS), alice_sdf.dll (Windows), or libalice_sdf.so (Linux)
 *    to Source/ThirdParty/AliceSdf/lib/
 * 3. Add library path to your Build.cs
 * 4. Include this header and use the AliceSdf namespace
 *
 * Example:
 *   auto Sphere = AliceSdf::Sphere(1.0f);
 *   auto Box = AliceSdf::Box(0.5f, 0.5f, 0.5f);
 *   auto Shape = AliceSdf::SmoothUnion(Sphere, Box, 0.2f);
 *   float Distance = AliceSdf::Eval(Shape, 0.5f, 0.0f, 0.0f);
 *   AliceSdf::Free(Shape);
 *   AliceSdf::Free(Box);
 *   AliceSdf::Free(Sphere);
 *
 * Author: Moroya Sakamoto
 */

#pragma once

#include <cstdint>

// ============================================================================
// Types
// ============================================================================

/// Opaque handle to an SDF node
typedef void* SdfHandle;

/// Opaque handle to a pre-compiled SDF (bytecode)
typedef void* CompiledHandle;

/// Opaque handle to a generated mesh
typedef void* MeshHandle;

/// Null handle constants
#define SDF_HANDLE_NULL    nullptr
#define COMPILED_HANDLE_NULL nullptr
#define MESH_HANDLE_NULL   nullptr

/// Result code for FFI operations
enum class ESdfResult : int32_t
{
    Ok              = 0,
    InvalidHandle   = 1,
    NullPointer     = 2,
    InvalidParameter = 3,
    OutOfMemory     = 4,
    IoError         = 5,
    CompileError    = 6,
    Unknown         = 99,
};

/// Shader type enum
enum class EShaderType : int32_t
{
    Wgsl = 0,
    Hlsl = 1,
    Glsl = 2,
};

/// 3D vector for FFI
struct FVec3Ffi
{
    float X;
    float Y;
    float Z;
};

/// Quaternion for FFI
struct FQuatFfi
{
    float X;
    float Y;
    float Z;
    float W;
};

/// Batch evaluation result
struct FBatchResult
{
    uint32_t Count;
    ESdfResult Result;
};

/// String result for shader generation
struct FStringResult
{
    char* Data;
    uint32_t Len;
    ESdfResult Result;
};

/// Version information
struct FVersionInfo
{
    uint16_t Major;
    uint16_t Minor;
    uint16_t Patch;
};

/// SoA batch evaluation configuration
struct FSoaBatchConfig
{
    const float* X;
    const float* Y;
    const float* Z;
    float* Distances;
    uint32_t Count;
    uint32_t Padding;
};

/// Evaluation statistics
struct FEvalStats
{
    uint64_t PointsEvaluated;
    uint64_t TimeNs;
    double Throughput;
};

// ============================================================================
// C API Declarations
// ============================================================================

extern "C"
{

// --- Info ---
FVersionInfo alice_sdf_version();
char* alice_sdf_version_string();
uint32_t alice_sdf_primitive_count();
uint32_t alice_sdf_operation_count();
uint32_t alice_sdf_transform_count();
uint32_t alice_sdf_modifier_count();
uint32_t alice_sdf_total_count();

// --- Memory ---
void alice_sdf_free(SdfHandle Node);
void alice_sdf_free_string(char* Str);
SdfHandle alice_sdf_clone(SdfHandle Node);
uint32_t alice_sdf_node_count(SdfHandle Node);
bool alice_sdf_is_valid(SdfHandle Node);
bool alice_sdf_is_compiled_valid(CompiledHandle Compiled);

// --- Primitives ---
SdfHandle alice_sdf_sphere(float Radius);
SdfHandle alice_sdf_box(float Hx, float Hy, float Hz);
SdfHandle alice_sdf_cylinder(float Radius, float HalfHeight);
SdfHandle alice_sdf_torus(float MajorRadius, float MinorRadius);
SdfHandle alice_sdf_capsule(float Ax, float Ay, float Az, float Bx, float By, float Bz, float Radius);
SdfHandle alice_sdf_plane(float Nx, float Ny, float Nz, float Distance);
SdfHandle alice_sdf_cone(float Radius, float HalfHeight);
SdfHandle alice_sdf_ellipsoid(float Rx, float Ry, float Rz);
SdfHandle alice_sdf_rounded_cone(float R1, float R2, float HalfHeight);
SdfHandle alice_sdf_pyramid(float HalfHeight);
SdfHandle alice_sdf_octahedron(float Size);
SdfHandle alice_sdf_hex_prism(float HexRadius, float HalfHeight);
SdfHandle alice_sdf_link(float HalfLength, float R1, float R2);
SdfHandle alice_sdf_triangle(float Ax, float Ay, float Az, float Bx, float By, float Bz, float Cx, float Cy, float Cz);
SdfHandle alice_sdf_bezier(float P0x, float P0y, float P0z, float P1x, float P1y, float P1z, float P2x, float P2y, float P2z, float Radius);
SdfHandle alice_sdf_rounded_box(float Hx, float Hy, float Hz, float RoundRadius);
SdfHandle alice_sdf_capped_cone(float HalfHeight, float R1, float R2);
SdfHandle alice_sdf_capped_torus(float Angle, float MajorRadius, float MinorRadius);
SdfHandle alice_sdf_rounded_cylinder(float Radius, float RoundRadius, float HalfHeight);
SdfHandle alice_sdf_triangular_prism(float Width, float HalfDepth);
SdfHandle alice_sdf_cut_sphere(float Radius, float CutHeight);
SdfHandle alice_sdf_cut_hollow_sphere(float Radius, float CutHeight, float Thickness);
SdfHandle alice_sdf_death_star(float Ra, float Rb, float D);
SdfHandle alice_sdf_solid_angle(float Angle, float Radius);
SdfHandle alice_sdf_rhombus(float La, float Lb, float H, float Ra);
SdfHandle alice_sdf_horseshoe(float Angle, float R, float Le, float W, float H);
SdfHandle alice_sdf_vesica(float Radius, float HalfDist);
SdfHandle alice_sdf_infinite_cylinder(float Radius);
SdfHandle alice_sdf_infinite_cone(float Angle);
SdfHandle alice_sdf_gyroid(float Scale, float Thickness);
SdfHandle alice_sdf_heart(float Size);
SdfHandle alice_sdf_tube(float OuterRadius, float Thickness, float HalfHeight);
SdfHandle alice_sdf_barrel(float Radius, float HalfHeight, float Bulge);
SdfHandle alice_sdf_diamond(float Radius, float HalfHeight);
SdfHandle alice_sdf_chamfered_cube(float Hx, float Hy, float Hz, float Chamfer);
SdfHandle alice_sdf_schwarz_p(float Scale, float Thickness);
SdfHandle alice_sdf_superellipsoid(float Rx, float Ry, float Rz, float E1, float E2);
SdfHandle alice_sdf_rounded_x(float W, float R, float HalfHeight);
SdfHandle alice_sdf_pie(float Angle, float Radius, float HalfHeight);
SdfHandle alice_sdf_trapezoid(float R1, float R2, float HalfHeight, float HalfDepth);
SdfHandle alice_sdf_parallelogram(float Width, float Height, float Skew, float HalfDepth);
SdfHandle alice_sdf_tunnel(float Width, float Height2d, float HalfDepth);
SdfHandle alice_sdf_uneven_capsule(float R1, float R2, float HalfHeight);
SdfHandle alice_sdf_egg(float Ra, float Rb);
SdfHandle alice_sdf_arc_shape(float Angle, float Ra, float Rb, float HalfHeight);
SdfHandle alice_sdf_moon(float D, float Ra, float Rb, float HalfHeight);
SdfHandle alice_sdf_cross_shape(float Arm, float ArmThickness, float HalfHeight);
SdfHandle alice_sdf_blobby_cross(float Size, float HalfHeight);
SdfHandle alice_sdf_parabola_segment(float Width, float Height, float HalfDepth);
SdfHandle alice_sdf_regular_polygon(uint32_t Sides, float Radius, float HalfHeight);
SdfHandle alice_sdf_star_polygon(uint32_t Points, float OuterRadius, float InnerRadius, float HalfHeight);
SdfHandle alice_sdf_stairs(float StepWidth, float StepHeight, uint32_t NumSteps, float HalfDepth);
SdfHandle alice_sdf_helix(float Radius, float Thickness, float Pitch, float HalfHeight);
SdfHandle alice_sdf_tetrahedron(float Size);
SdfHandle alice_sdf_dodecahedron(float Radius);
SdfHandle alice_sdf_icosahedron(float Radius);
SdfHandle alice_sdf_truncated_octahedron(float Radius);
SdfHandle alice_sdf_truncated_icosahedron(float Radius);
SdfHandle alice_sdf_box_frame(float Hx, float Hy, float Hz, float Edge);
SdfHandle alice_sdf_diamond_surface(float Scale, float Thickness);
SdfHandle alice_sdf_neovius(float Scale, float Thickness);
SdfHandle alice_sdf_lidinoid(float Scale, float Thickness);
SdfHandle alice_sdf_iwp(float Scale, float Thickness);
SdfHandle alice_sdf_frd(float Scale, float Thickness);
SdfHandle alice_sdf_fischer_koch_s(float Scale, float Thickness);
SdfHandle alice_sdf_pmy(float Scale, float Thickness);
SdfHandle alice_sdf_circle_2d(float Radius, float HalfHeight);
SdfHandle alice_sdf_rect_2d(float HalfW, float HalfH, float HalfHeight);
SdfHandle alice_sdf_segment_2d(float Ax, float Ay, float Bx, float By, float HalfHeight);
SdfHandle alice_sdf_rounded_rect_2d(float HalfW, float HalfH, float Radius, float HalfHeight);
SdfHandle alice_sdf_annular_2d(float InnerRadius, float OuterRadius, float HalfHeight);

// --- Operations ---
SdfHandle alice_sdf_union(SdfHandle A, SdfHandle B);
SdfHandle alice_sdf_intersection(SdfHandle A, SdfHandle B);
SdfHandle alice_sdf_subtract(SdfHandle A, SdfHandle B);
SdfHandle alice_sdf_smooth_union(SdfHandle A, SdfHandle B, float K);
SdfHandle alice_sdf_smooth_intersection(SdfHandle A, SdfHandle B, float K);
SdfHandle alice_sdf_smooth_subtract(SdfHandle A, SdfHandle B, float K);
SdfHandle alice_sdf_chamfer_union(SdfHandle A, SdfHandle B, float R);
SdfHandle alice_sdf_chamfer_intersection(SdfHandle A, SdfHandle B, float R);
SdfHandle alice_sdf_chamfer_subtract(SdfHandle A, SdfHandle B, float R);
SdfHandle alice_sdf_stairs_union(SdfHandle A, SdfHandle B, float R, float N);
SdfHandle alice_sdf_stairs_intersection(SdfHandle A, SdfHandle B, float R, float N);
SdfHandle alice_sdf_stairs_subtract(SdfHandle A, SdfHandle B, float R, float N);
SdfHandle alice_sdf_xor(SdfHandle A, SdfHandle B);
SdfHandle alice_sdf_morph(SdfHandle A, SdfHandle B, float T);
SdfHandle alice_sdf_columns_union(SdfHandle A, SdfHandle B, float R, float N);
SdfHandle alice_sdf_columns_intersection(SdfHandle A, SdfHandle B, float R, float N);
SdfHandle alice_sdf_columns_subtract(SdfHandle A, SdfHandle B, float R, float N);
SdfHandle alice_sdf_pipe(SdfHandle A, SdfHandle B, float R);
SdfHandle alice_sdf_engrave(SdfHandle A, SdfHandle B, float R);
SdfHandle alice_sdf_groove(SdfHandle A, SdfHandle B, float Ra, float Rb);
SdfHandle alice_sdf_tongue(SdfHandle A, SdfHandle B, float Ra, float Rb);
SdfHandle alice_sdf_exp_smooth_union(SdfHandle A, SdfHandle B, float K);
SdfHandle alice_sdf_exp_smooth_intersection(SdfHandle A, SdfHandle B, float K);
SdfHandle alice_sdf_exp_smooth_subtract(SdfHandle A, SdfHandle B, float K);

// --- Modifiers ---
SdfHandle alice_sdf_round(SdfHandle Node, float Radius);
SdfHandle alice_sdf_onion(SdfHandle Node, float Thickness);
SdfHandle alice_sdf_twist(SdfHandle Node, float Strength);
SdfHandle alice_sdf_bend(SdfHandle Node, float Curvature);
SdfHandle alice_sdf_repeat(SdfHandle Node, float Sx, float Sy, float Sz);
SdfHandle alice_sdf_mirror(SdfHandle Node, uint8_t Mx, uint8_t My, uint8_t Mz);
SdfHandle alice_sdf_elongate(SdfHandle Node, float X, float Y, float Z);
SdfHandle alice_sdf_revolution(SdfHandle Node, float Offset);
SdfHandle alice_sdf_extrude(SdfHandle Node, float HalfHeight);
SdfHandle alice_sdf_noise(SdfHandle Node, float Amplitude, float Frequency, uint32_t Octaves);
SdfHandle alice_sdf_repeat_finite(SdfHandle Node, float Sx, float Sy, float Sz, uint32_t Cx, uint32_t Cy, uint32_t Cz);
SdfHandle alice_sdf_taper(SdfHandle Node, float Factor);
SdfHandle alice_sdf_displacement(SdfHandle Node, float Strength);
SdfHandle alice_sdf_polar_repeat(SdfHandle Node, uint32_t Count);
SdfHandle alice_sdf_octant_mirror(SdfHandle Node);
SdfHandle alice_sdf_shear(SdfHandle Node, float Xy, float Xz, float Yz);
SdfHandle alice_sdf_animated(SdfHandle Node, float Speed, float Amplitude);
SdfHandle alice_sdf_with_material(SdfHandle Node, uint32_t MaterialId);
SdfHandle alice_sdf_sweep_bezier(SdfHandle Node, float P0x, float P0y, float P0z, float P1x, float P1y, float P1z, float P2x, float P2y, float P2z);
SdfHandle alice_sdf_scale_non_uniform(SdfHandle Node, float Sx, float Sy, float Sz);
SdfHandle alice_sdf_icosahedral_symmetry(SdfHandle Node);
SdfHandle alice_sdf_ifs(SdfHandle Node, const float* Matrices, uint32_t Count, uint32_t Iterations);
SdfHandle alice_sdf_heightmap_displacement(SdfHandle Node, const float* Data, uint32_t Width, uint32_t Height, float Scale, float Amplitude);
SdfHandle alice_sdf_surface_roughness(SdfHandle Node, float Amplitude, float Frequency, uint32_t Octaves);

// --- Transforms ---
SdfHandle alice_sdf_translate(SdfHandle Node, float X, float Y, float Z);
SdfHandle alice_sdf_rotate(SdfHandle Node, float Qx, float Qy, float Qz, float Qw);
SdfHandle alice_sdf_rotate_euler(SdfHandle Node, float X, float Y, float Z);
SdfHandle alice_sdf_scale(SdfHandle Node, float Factor);
SdfHandle alice_sdf_scale_xyz(SdfHandle Node, float X, float Y, float Z);
SdfHandle alice_sdf_projective_transform(SdfHandle Node, const float* Matrix4x4);
SdfHandle alice_sdf_lattice_deform(SdfHandle Node, const float* ControlPoints, uint32_t Nx, uint32_t Ny, uint32_t Nz);
SdfHandle alice_sdf_skinning(SdfHandle Node, const float* BoneTransforms, const float* BoneWeights, uint32_t BoneCount, uint32_t VertexCount);

// --- Compilation ---
CompiledHandle alice_sdf_compile(SdfHandle Node);
void alice_sdf_free_compiled(CompiledHandle Compiled);
uint32_t alice_sdf_compiled_instruction_count(CompiledHandle Compiled);

// --- Evaluation ---
float alice_sdf_eval(SdfHandle Node, float X, float Y, float Z);
float alice_sdf_eval_compiled(CompiledHandle Compiled, float X, float Y, float Z);

// --- Mesh ---
MeshHandle alice_sdf_generate_mesh(SdfHandle Node, float MinX, float MinY, float MinZ, float MaxX, float MaxY, float MaxZ, float Resolution);
uint32_t alice_sdf_mesh_vertex_count(MeshHandle Mesh);
uint32_t alice_sdf_mesh_triangle_count(MeshHandle Mesh);
void alice_sdf_free_mesh(MeshHandle Mesh);

// --- I/O ---
ESdfResult alice_sdf_save(SdfHandle Node, const char* Path);
SdfHandle alice_sdf_load(const char* Path);

// --- Shader ---
FStringResult alice_sdf_to_wgsl(SdfHandle Node);
FStringResult alice_sdf_to_hlsl(SdfHandle Node);
FStringResult alice_sdf_to_glsl(SdfHandle Node);

} // extern "C"

// ============================================================================
// RAII Wrappers (C++ convenience)
// ============================================================================

namespace AliceSdf
{

/// RAII wrapper for SdfHandle - automatically frees on destruction
class FSdfNode
{
public:
    FSdfNode() : Handle(SDF_HANDLE_NULL) {}
    explicit FSdfNode(SdfHandle InHandle) : Handle(InHandle) {}
    ~FSdfNode() { if (Handle) alice_sdf_free(Handle); }

    // Move only
    FSdfNode(FSdfNode&& Other) noexcept : Handle(Other.Handle) { Other.Handle = SDF_HANDLE_NULL; }
    FSdfNode& operator=(FSdfNode&& Other) noexcept
    {
        if (this != &Other)
        {
            if (Handle) alice_sdf_free(Handle);
            Handle = Other.Handle;
            Other.Handle = SDF_HANDLE_NULL;
        }
        return *this;
    }
    FSdfNode(const FSdfNode&) = delete;
    FSdfNode& operator=(const FSdfNode&) = delete;

    SdfHandle Get() const { return Handle; }
    SdfHandle Release() { SdfHandle H = Handle; Handle = SDF_HANDLE_NULL; return H; }
    bool IsValid() const { return Handle && alice_sdf_is_valid(Handle); }
    explicit operator bool() const { return IsValid(); }

private:
    SdfHandle Handle;
};

/// RAII wrapper for CompiledHandle
class FCompiledSdf
{
public:
    FCompiledSdf() : Handle(COMPILED_HANDLE_NULL) {}
    explicit FCompiledSdf(CompiledHandle InHandle) : Handle(InHandle) {}
    ~FCompiledSdf() { if (Handle) alice_sdf_free_compiled(Handle); }

    FCompiledSdf(FCompiledSdf&& Other) noexcept : Handle(Other.Handle) { Other.Handle = COMPILED_HANDLE_NULL; }
    FCompiledSdf& operator=(FCompiledSdf&& Other) noexcept
    {
        if (this != &Other)
        {
            if (Handle) alice_sdf_free_compiled(Handle);
            Handle = Other.Handle;
            Other.Handle = COMPILED_HANDLE_NULL;
        }
        return *this;
    }
    FCompiledSdf(const FCompiledSdf&) = delete;
    FCompiledSdf& operator=(const FCompiledSdf&) = delete;

    CompiledHandle Get() const { return Handle; }
    bool IsValid() const { return Handle && alice_sdf_is_compiled_valid(Handle); }
    float Eval(float X, float Y, float Z) const { return alice_sdf_eval_compiled(Handle, X, Y, Z); }
    uint32_t InstructionCount() const { return alice_sdf_compiled_instruction_count(Handle); }

private:
    CompiledHandle Handle;
};

/// RAII wrapper for MeshHandle
class FMeshData
{
public:
    FMeshData() : Handle(MESH_HANDLE_NULL) {}
    explicit FMeshData(MeshHandle InHandle) : Handle(InHandle) {}
    ~FMeshData() { if (Handle) alice_sdf_free_mesh(Handle); }

    FMeshData(FMeshData&& Other) noexcept : Handle(Other.Handle) { Other.Handle = MESH_HANDLE_NULL; }
    FMeshData& operator=(FMeshData&& Other) noexcept
    {
        if (this != &Other)
        {
            if (Handle) alice_sdf_free_mesh(Handle);
            Handle = Other.Handle;
            Other.Handle = MESH_HANDLE_NULL;
        }
        return *this;
    }
    FMeshData(const FMeshData&) = delete;
    FMeshData& operator=(const FMeshData&) = delete;

    MeshHandle Get() const { return Handle; }
    uint32_t VertexCount() const { return alice_sdf_mesh_vertex_count(Handle); }
    uint32_t TriangleCount() const { return alice_sdf_mesh_triangle_count(Handle); }

private:
    MeshHandle Handle;
};

} // namespace AliceSdf
