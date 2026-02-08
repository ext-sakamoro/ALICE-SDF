/**
 * @file alice_sdf.h
 * @brief ALICE-SDF C/C++ API Header (Deep Fried Edition)
 * @author Moroya Sakamoto
 * @version 0.1.0
 *
 * ALICE-SDF: Adaptive Lightweight Implicit Compression Engine
 * Store laws, not polygons.
 *
 * This header provides C-compatible bindings for the ALICE-SDF library.
 * Compatible with C, C++, C# (P/Invoke), and other FFI systems.
 *
 * ## Performance Hierarchy (fastest to slowest)
 *
 *  1. alice_sdf_eval_soa          - SoA layout + compiled (1B+ ops/sec)
 *  2. alice_sdf_eval_compiled_batch - AoS layout + compiled
 *  3. alice_sdf_eval_batch        - AoS layout + auto-compile
 *  4. alice_sdf_eval              - Single point (debugging only)
 *
 * ## Usage (C/C++)
 *
 * ```c
 * #include "alice_sdf.h"
 *
 * int main() {
 *     SdfHandle sphere = alice_sdf_sphere(1.0f);
 *     SdfHandle box = alice_sdf_box(0.5f, 0.5f, 0.5f);
 *     SdfHandle shape = alice_sdf_smooth_union(sphere, box, 0.2f);
 *
 *     // Compile for fast eval
 *     CompiledHandle compiled = alice_sdf_compile(shape);
 *     float distance = alice_sdf_eval_compiled(compiled, 0.5f, 0.0f, 0.0f);
 *     printf("Distance: %f\n", distance);
 *
 *     // Generate HLSL
 *     StringResult hlsl = alice_sdf_to_hlsl(shape);
 *     if (hlsl.result == SdfResult_Ok) {
 *         printf("HLSL:\n%s\n", hlsl.data);
 *         alice_sdf_free_string(hlsl.data);
 *     }
 *
 *     alice_sdf_free_compiled(compiled);
 *     alice_sdf_free(shape);
 *     alice_sdf_free(box);
 *     alice_sdf_free(sphere);
 *     return 0;
 * }
 * ```
 */

#ifndef ALICE_SDF_H
#define ALICE_SDF_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Types
 * ============================================================================ */

/**
 * @brief Opaque handle to an SDF node
 *
 * All SDF operations return handles that must be freed with alice_sdf_free().
 */
typedef void* SdfHandle;

/**
 * @brief Opaque handle to a pre-compiled SDF (bytecode)
 *
 * Compiled handles evaluate ~10x faster than raw SdfHandle.
 * Create once with alice_sdf_compile(), reuse every frame.
 */
typedef void* CompiledHandle;

/**
 * @brief Null handle constant
 */
#define SDF_HANDLE_NULL ((SdfHandle)0)

/**
 * @brief Null compiled handle constant
 */
#define COMPILED_HANDLE_NULL ((CompiledHandle)0)

/**
 * @brief Result codes for FFI operations
 */
typedef enum {
    SdfResult_Ok = 0,
    SdfResult_InvalidHandle = 1,
    SdfResult_NullPointer = 2,
    SdfResult_InvalidParameter = 3,
    SdfResult_OutOfMemory = 4,
    SdfResult_IoError = 5,
    SdfResult_CompileError = 6,
    SdfResult_Unknown = 99
} SdfResult;

/**
 * @brief Shader output types
 */
typedef enum {
    ShaderType_Wgsl = 0,  /**< WebGPU Shading Language */
    ShaderType_Hlsl = 1,  /**< High-Level Shading Language (DirectX/UE5) */
    ShaderType_Glsl = 2   /**< OpenGL Shading Language (Unity/OpenGL) */
} ShaderType;

/**
 * @brief Version information
 */
typedef struct {
    uint16_t major;
    uint16_t minor;
    uint16_t patch;
} VersionInfo;

/**
 * @brief Batch evaluation result
 */
typedef struct {
    uint32_t count;
    SdfResult result;
} BatchResult;

/**
 * @brief String result (for shader generation)
 *
 * The data pointer must be freed with alice_sdf_free_string().
 */
typedef struct {
    char* data;       /**< Null-terminated UTF-8 string (or NULL on error) */
    uint32_t len;     /**< Length in bytes (not including null terminator) */
    SdfResult result; /**< Result code */
} StringResult;

/* ============================================================================
 * Library Info
 * ============================================================================ */

/**
 * @brief Get library version
 * @return Version information struct
 */
VersionInfo alice_sdf_version(void);

/**
 * @brief Get version string
 * @return Null-terminated version string (caller must free with alice_sdf_free_string)
 */
char* alice_sdf_version_string(void);

/* ============================================================================
 * Primitives
 * ============================================================================ */

/**
 * @brief Create a sphere SDF
 * @param radius Sphere radius
 * @return Handle to the sphere SDF
 */
SdfHandle alice_sdf_sphere(float radius);

/**
 * @brief Create a box SDF
 * @param hx Half-extent in X
 * @param hy Half-extent in Y
 * @param hz Half-extent in Z
 * @return Handle to the box SDF
 */
SdfHandle alice_sdf_box(float hx, float hy, float hz);

/**
 * @brief Create a cylinder SDF
 * @param radius Cylinder radius
 * @param half_height Half-height of the cylinder
 * @return Handle to the cylinder SDF
 */
SdfHandle alice_sdf_cylinder(float radius, float half_height);

/**
 * @brief Create a torus SDF
 * @param major_radius Distance from center to tube center
 * @param minor_radius Tube radius
 * @return Handle to the torus SDF
 */
SdfHandle alice_sdf_torus(float major_radius, float minor_radius);

/**
 * @brief Create a capsule SDF
 * @param ax,ay,az First endpoint
 * @param bx,by,bz Second endpoint
 * @param radius Capsule radius
 * @return Handle to the capsule SDF
 */
SdfHandle alice_sdf_capsule(float ax, float ay, float az,
                            float bx, float by, float bz,
                            float radius);

/**
 * @brief Create a plane SDF
 * @param nx,ny,nz Plane normal (will be normalized)
 * @param distance Distance from origin
 * @return Handle to the plane SDF
 */
SdfHandle alice_sdf_plane(float nx, float ny, float nz, float distance);

SdfHandle alice_sdf_cone(float radius, float half_height);
SdfHandle alice_sdf_ellipsoid(float rx, float ry, float rz);
SdfHandle alice_sdf_rounded_cone(float r1, float r2, float half_height);
SdfHandle alice_sdf_pyramid(float half_height);
SdfHandle alice_sdf_octahedron(float size);
SdfHandle alice_sdf_hex_prism(float hex_radius, float half_height);
SdfHandle alice_sdf_link(float half_length, float r1, float r2);
SdfHandle alice_sdf_rounded_box(float hx, float hy, float hz, float round_radius);

/* Advanced */
SdfHandle alice_sdf_capped_cone(float half_height, float r1, float r2);
SdfHandle alice_sdf_capped_torus(float major_radius, float minor_radius, float cap_angle);
SdfHandle alice_sdf_rounded_cylinder(float radius, float round_radius, float half_height);
SdfHandle alice_sdf_triangular_prism(float width, float half_depth);
SdfHandle alice_sdf_cut_sphere(float radius, float cut_height);
SdfHandle alice_sdf_cut_hollow_sphere(float radius, float cut_height, float thickness);
SdfHandle alice_sdf_death_star(float ra, float rb, float d);
SdfHandle alice_sdf_solid_angle(float angle, float radius);
SdfHandle alice_sdf_heart(float size);
SdfHandle alice_sdf_barrel(float radius, float half_height, float bulge);
SdfHandle alice_sdf_diamond(float radius, float half_height);
SdfHandle alice_sdf_egg(float ra, float rb);

/* 2D/Extruded */
SdfHandle alice_sdf_triangle(float ax, float ay, float az, float bx, float by, float bz, float cx, float cy, float cz);
SdfHandle alice_sdf_bezier(float ax, float ay, float az, float bx, float by, float bz, float cx, float cy, float cz, float radius);
SdfHandle alice_sdf_rhombus(float la, float lb, float half_height, float round_radius);
SdfHandle alice_sdf_horseshoe(float angle, float radius, float half_length, float width, float thickness);
SdfHandle alice_sdf_vesica(float radius, float half_dist);
SdfHandle alice_sdf_superellipsoid(float hx, float hy, float hz, float e1, float e2);
SdfHandle alice_sdf_rounded_x(float width, float round_radius, float half_height);
SdfHandle alice_sdf_pie(float angle, float radius, float half_height);
SdfHandle alice_sdf_trapezoid(float r1, float r2, float trap_height, float half_depth);
SdfHandle alice_sdf_parallelogram(float width, float para_height, float skew, float half_depth);
SdfHandle alice_sdf_tunnel(float width, float height_2d, float half_depth);
SdfHandle alice_sdf_uneven_capsule(float r1, float r2, float cap_height, float half_depth);
SdfHandle alice_sdf_arc_shape(float aperture, float radius, float thickness, float half_height);
SdfHandle alice_sdf_moon(float d, float ra, float rb, float half_height);
SdfHandle alice_sdf_cross_shape(float length, float thickness, float round_radius, float half_height);
SdfHandle alice_sdf_blobby_cross(float size, float half_height);
SdfHandle alice_sdf_parabola_segment(float width, float para_height, float half_depth);
SdfHandle alice_sdf_regular_polygon(float radius, float n_sides, float half_height);
SdfHandle alice_sdf_star_polygon(float radius, float n_points, float m, float half_height);
SdfHandle alice_sdf_infinite_cylinder(float radius);
SdfHandle alice_sdf_infinite_cone(float angle);

/* Platonic & Archimedean */
SdfHandle alice_sdf_tetrahedron(float size);
SdfHandle alice_sdf_dodecahedron(float radius);
SdfHandle alice_sdf_icosahedron(float radius);
SdfHandle alice_sdf_truncated_octahedron(float radius);
SdfHandle alice_sdf_truncated_icosahedron(float radius);

/* TPMS */
SdfHandle alice_sdf_gyroid(float scale, float thickness);
SdfHandle alice_sdf_schwarz_p(float scale, float thickness);
SdfHandle alice_sdf_diamond_surface(float scale, float thickness);
SdfHandle alice_sdf_neovius(float scale, float thickness);
SdfHandle alice_sdf_lidinoid(float scale, float thickness);
SdfHandle alice_sdf_iwp(float scale, float thickness);
SdfHandle alice_sdf_frd(float scale, float thickness);
SdfHandle alice_sdf_fischer_koch_s(float scale, float thickness);
SdfHandle alice_sdf_pmy(float scale, float thickness);

/* Structural */
SdfHandle alice_sdf_box_frame(float hx, float hy, float hz, float edge);
SdfHandle alice_sdf_tube(float outer_radius, float thickness, float half_height);
SdfHandle alice_sdf_chamfered_cube(float hx, float hy, float hz, float chamfer);
SdfHandle alice_sdf_stairs(float step_width, float step_height, float num_steps, float half_depth);
SdfHandle alice_sdf_helix(float major_radius, float minor_radius, float pitch, float half_height);

/* ============================================================================
 * Boolean Operations
 * ============================================================================ */

/** @brief Union of two SDFs (A | B) */
SdfHandle alice_sdf_union(SdfHandle a, SdfHandle b);

/** @brief Intersection of two SDFs (A & B) */
SdfHandle alice_sdf_intersection(SdfHandle a, SdfHandle b);

/** @brief Subtraction of SDFs (A - B) */
SdfHandle alice_sdf_subtract(SdfHandle a, SdfHandle b);

/**
 * @brief Smooth union of two SDFs
 * @param k Smoothing factor (0 = sharp, higher = smoother)
 */
SdfHandle alice_sdf_smooth_union(SdfHandle a, SdfHandle b, float k);

/**
 * @brief Smooth intersection of two SDFs
 * @param k Smoothing factor
 */
SdfHandle alice_sdf_smooth_intersection(SdfHandle a, SdfHandle b, float k);

/**
 * @brief Smooth subtraction of SDFs
 * @param k Smoothing factor
 */
SdfHandle alice_sdf_smooth_subtract(SdfHandle a, SdfHandle b, float k);

/** Chamfer operations */
SdfHandle alice_sdf_chamfer_union(SdfHandle a, SdfHandle b, float radius);
SdfHandle alice_sdf_chamfer_intersection(SdfHandle a, SdfHandle b, float radius);
SdfHandle alice_sdf_chamfer_subtract(SdfHandle a, SdfHandle b, float radius);

/** Stairs (terraced) operations */
SdfHandle alice_sdf_stairs_union(SdfHandle a, SdfHandle b, float radius, float steps);
SdfHandle alice_sdf_stairs_intersection(SdfHandle a, SdfHandle b, float radius, float steps);
SdfHandle alice_sdf_stairs_subtract(SdfHandle a, SdfHandle b, float radius, float steps);

/** Columns operations */
SdfHandle alice_sdf_columns_union(SdfHandle a, SdfHandle b, float radius, float count);
SdfHandle alice_sdf_columns_intersection(SdfHandle a, SdfHandle b, float radius, float count);
SdfHandle alice_sdf_columns_subtract(SdfHandle a, SdfHandle b, float radius, float count);

/** Advanced operations */
SdfHandle alice_sdf_xor(SdfHandle a, SdfHandle b);
SdfHandle alice_sdf_morph(SdfHandle a, SdfHandle b, float t);
SdfHandle alice_sdf_pipe(SdfHandle a, SdfHandle b, float radius);
SdfHandle alice_sdf_engrave(SdfHandle a, SdfHandle b, float depth);
SdfHandle alice_sdf_groove(SdfHandle a, SdfHandle b, float ra, float rb);
SdfHandle alice_sdf_tongue(SdfHandle a, SdfHandle b, float ra, float rb);

/* ============================================================================
 * Transforms
 * ============================================================================ */

/** @brief Translate an SDF */
SdfHandle alice_sdf_translate(SdfHandle node, float x, float y, float z);

/** @brief Rotate an SDF using quaternion (x, y, z, w) */
SdfHandle alice_sdf_rotate(SdfHandle node, float qx, float qy, float qz, float qw);

/** @brief Rotate an SDF using Euler angles (radians) */
SdfHandle alice_sdf_rotate_euler(SdfHandle node, float x, float y, float z);

/** @brief Uniform scale an SDF */
SdfHandle alice_sdf_scale(SdfHandle node, float factor);

/** @brief Non-uniform scale an SDF */
SdfHandle alice_sdf_scale_xyz(SdfHandle node, float x, float y, float z);

/** @brief Non-uniform scale (alias) */
SdfHandle alice_sdf_scale_non_uniform(SdfHandle node, float x, float y, float z);

/* ============================================================================
 * Modifiers
 * ============================================================================ */

/**
 * @brief Apply rounding to an SDF
 * @param radius Rounding radius
 */
SdfHandle alice_sdf_round(SdfHandle node, float radius);

/**
 * @brief Apply onion (shell) modifier
 * @param thickness Shell thickness
 */
SdfHandle alice_sdf_onion(SdfHandle node, float thickness);

/**
 * @brief Apply twist modifier
 * @param strength Twist strength (radians per unit)
 */
SdfHandle alice_sdf_twist(SdfHandle node, float strength);

/**
 * @brief Apply bend modifier
 * @param curvature Bend curvature
 */
SdfHandle alice_sdf_bend(SdfHandle node, float curvature);

/**
 * @brief Apply infinite repetition
 * @param sx,sy,sz Spacing in each axis
 */
SdfHandle alice_sdf_repeat(SdfHandle node, float sx, float sy, float sz);
SdfHandle alice_sdf_repeat_finite(SdfHandle node, int32_t cx, int32_t cy, int32_t cz, float sx, float sy, float sz);
SdfHandle alice_sdf_mirror(SdfHandle node, float mx, float my, float mz);
SdfHandle alice_sdf_elongate(SdfHandle node, float ex, float ey, float ez);
SdfHandle alice_sdf_revolution(SdfHandle node, float offset);
SdfHandle alice_sdf_extrude(SdfHandle node, float half_height);
SdfHandle alice_sdf_noise(SdfHandle node, float amplitude, float frequency, uint32_t seed);
SdfHandle alice_sdf_taper(SdfHandle node, float factor);
SdfHandle alice_sdf_displacement(SdfHandle node, float strength);
SdfHandle alice_sdf_polar_repeat(SdfHandle node, uint32_t count);
SdfHandle alice_sdf_octant_mirror(SdfHandle node);
SdfHandle alice_sdf_sweep_bezier(SdfHandle node, float p0x, float p0y, float p1x, float p1y, float p2x, float p2y);
SdfHandle alice_sdf_with_material(SdfHandle node, uint32_t material_id);

/* ============================================================================
 * Mesh Generation & Export
 * ============================================================================ */

typedef void* MeshHandle;
#define MESH_HANDLE_NULL ((MeshHandle)0)

MeshHandle alice_sdf_generate_mesh(SdfHandle node, uint32_t resolution, float bounds);
uint32_t alice_sdf_mesh_vertex_count(MeshHandle mesh);
uint32_t alice_sdf_mesh_triangle_count(MeshHandle mesh);
void alice_sdf_free_mesh(MeshHandle mesh);
SdfResult alice_sdf_export_obj(MeshHandle mesh, SdfHandle node, const char* path, uint32_t resolution, float bounds);
SdfResult alice_sdf_export_glb(MeshHandle mesh, SdfHandle node, const char* path, uint32_t resolution, float bounds);
SdfResult alice_sdf_export_usda(MeshHandle mesh, SdfHandle node, const char* path, uint32_t resolution, float bounds);
SdfResult alice_sdf_export_fbx(MeshHandle mesh, SdfHandle node, const char* path, uint32_t resolution, float bounds);
SdfResult alice_sdf_export_alembic(MeshHandle mesh, SdfHandle node, const char* path, uint32_t resolution, float bounds);

/* ============================================================================
 * Compilation (Deep Fried)
 * ============================================================================ */

/**
 * @brief Compile an SDF to bytecode for fast evaluation
 *
 * Compilation is expensive (~0.1ms), but the resulting CompiledHandle
 * evaluates ~10x faster. Compile once at setup time, reuse every frame.
 *
 * @param node SDF handle to compile
 * @return Compiled handle (or COMPILED_HANDLE_NULL on failure)
 */
CompiledHandle alice_sdf_compile(SdfHandle node);

/**
 * @brief Free a compiled SDF handle
 */
void alice_sdf_free_compiled(CompiledHandle compiled);

/**
 * @brief Get instruction count of a compiled SDF (for profiling)
 */
uint32_t alice_sdf_compiled_instruction_count(CompiledHandle compiled);

/**
 * @brief Check if a compiled handle is valid
 */
bool alice_sdf_is_compiled_valid(CompiledHandle compiled);

/* ============================================================================
 * Evaluation
 * ============================================================================ */

/**
 * @brief Evaluate SDF at a single point
 * @return Signed distance (negative = inside, positive = outside)
 */
float alice_sdf_eval(SdfHandle node, float x, float y, float z);

/**
 * @brief Evaluate compiled SDF at a single point
 */
float alice_sdf_eval_compiled(CompiledHandle compiled, float x, float y, float z);

/**
 * @brief Evaluate SDF at multiple points (parallel, auto-compiles internally)
 * @param node SDF handle
 * @param points Array of floats [x0, y0, z0, x1, y1, z1, ...]
 * @param distances Output array (must be pre-allocated with count elements)
 * @param count Number of points
 */
BatchResult alice_sdf_eval_batch(SdfHandle node,
                                  const float* points,
                                  float* distances,
                                  uint32_t count);

/**
 * @brief Evaluate compiled SDF at multiple points (fastest AoS path)
 * @param compiled Pre-compiled SDF handle
 * @param points Array of floats [x0, y0, z0, x1, y1, z1, ...]
 * @param distances Output array (must be pre-allocated with count elements)
 * @param count Number of points
 */
BatchResult alice_sdf_eval_compiled_batch(CompiledHandle compiled,
                                           const float* points,
                                           float* distances,
                                           uint32_t count);

/**
 * @brief Evaluate using SoA (Structure of Arrays) layout - THE FASTEST PATH
 *
 * SoA layout enables SIMD vectorization. X, Y, Z coordinates are stored
 * in separate contiguous arrays. Use for physics, particles, raymarching.
 *
 * @param compiled Pre-compiled SDF handle
 * @param x Pointer to X coordinates array
 * @param y Pointer to Y coordinates array
 * @param z Pointer to Z coordinates array
 * @param distances Output array (caller-allocated)
 * @param count Number of points
 */
BatchResult alice_sdf_eval_soa(CompiledHandle compiled,
                                const float* x,
                                const float* y,
                                const float* z,
                                float* distances,
                                uint32_t count);

/**
 * @brief Evaluate distance AND gradient (normal) using SoA layout
 *
 * Computes both distance and surface normal direction for all points.
 * ~4x slower than distance-only but avoids separate gradient pass.
 *
 * @param compiled Compiled SDF handle
 * @param x,y,z Input position arrays (SoA layout)
 * @param nx,ny,nz Output gradient/normal arrays (SoA layout, normalized)
 * @param dist Output distance array (optional, can be NULL)
 * @param count Number of points
 */
BatchResult alice_sdf_eval_gradient_soa(CompiledHandle compiled,
                                         const float* x,
                                         const float* y,
                                         const float* z,
                                         float* nx,
                                         float* ny,
                                         float* nz,
                                         float* dist,
                                         uint32_t count);

/* ============================================================================
 * Animated Compiled Evaluation (Zero-Copy)
 * ============================================================================ */

/**
 * @brief Animation parameters for zero-allocation per-frame evaluation (36 bytes)
 *
 * Instead of rebuilding the SDF tree each frame, extract transform parameters
 * and apply them during compiled evaluation. Stack-allocated, no heap alloc.
 */
typedef struct {
    float translate_x;
    float translate_y;
    float translate_z;
    float rotate_x;       /**< Euler rotation X (radians) */
    float rotate_y;       /**< Euler rotation Y (radians) */
    float rotate_z;       /**< Euler rotation Z (radians) */
    float scale;          /**< Uniform scale (1.0 = no scale) */
    float twist;          /**< Twist strength (radians per unit) */
    float bend;           /**< Bend curvature */
} AnimationParams;

/**
 * @brief Evaluate compiled SDF with animation transform (zero-alloc per frame)
 *
 * @param compiled Pre-compiled base shape
 * @param params Pointer to AnimationParams struct
 * @param x,y,z Query point coordinates
 * @return Signed distance at the query point
 */
float alice_sdf_eval_animated_compiled(CompiledHandle compiled,
                                        const AnimationParams* params,
                                        float x, float y, float z);

/**
 * @brief Batch evaluate animated SDF using SoA layout
 *
 * Combines zero-copy animation with SIMD SoA evaluation.
 * Use for animated particle systems at maximum throughput.
 *
 * @param compiled Pre-compiled base shape
 * @param params Animation parameters (shared for all points)
 * @param x,y,z Input position arrays (SoA layout)
 * @param distances Output array (caller-allocated)
 * @param count Number of points
 */
BatchResult alice_sdf_eval_animated_batch_soa(CompiledHandle compiled,
                                               const AnimationParams* params,
                                               const float* x,
                                               const float* y,
                                               const float* z,
                                               float* distances,
                                               uint32_t count);

/* ============================================================================
 * Shader Generation
 * ============================================================================ */

/**
 * @brief Generate WGSL shader code
 * @note Requires 'gpu' feature
 */
StringResult alice_sdf_to_wgsl(SdfHandle node);

/**
 * @brief Generate HLSL shader code (for UE5/DirectX)
 * @note Requires 'hlsl' feature
 */
StringResult alice_sdf_to_hlsl(SdfHandle node);

/**
 * @brief Generate GLSL shader code (for Unity/OpenGL)
 * @note Requires 'glsl' feature
 */
StringResult alice_sdf_to_glsl(SdfHandle node);

/* ============================================================================
 * File I/O
 * ============================================================================ */

/**
 * @brief Save SDF to .asdf file
 * @param path File path (null-terminated UTF-8 string)
 */
SdfResult alice_sdf_save(SdfHandle node, const char* path);

/**
 * @brief Load SDF from .asdf file
 * @param path File path (null-terminated UTF-8 string)
 * @return Handle to loaded SDF (or SDF_HANDLE_NULL on error)
 */
SdfHandle alice_sdf_load(const char* path);

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/** @brief Free an SDF handle */
void alice_sdf_free(SdfHandle node);

/** @brief Free a string returned by shader generation */
void alice_sdf_free_string(char* str);

/** @brief Clone an SDF handle */
SdfHandle alice_sdf_clone(SdfHandle node);

/* ============================================================================
 * Utilities
 * ============================================================================ */

/** @brief Get node count in an SDF tree */
uint32_t alice_sdf_node_count(SdfHandle node);

/** @brief Check if a handle is valid */
bool alice_sdf_is_valid(SdfHandle node);

#ifdef __cplusplus
}
#endif

#endif /* ALICE_SDF_H */
