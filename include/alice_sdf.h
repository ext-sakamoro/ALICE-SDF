/**
 * @file alice_sdf.h
 * @brief ALICE-SDF C/C++ API Header
 * @author Moroya Sakamoto
 * @version 0.1.0
 *
 * ALICE-SDF: Adaptive Lightweight Implicit Compression Engine
 * Store laws, not polygons.
 *
 * This header provides C-compatible bindings for the ALICE-SDF library.
 * Compatible with C, C++, C# (P/Invoke), and other FFI systems.
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
 *     float distance = alice_sdf_eval(shape, 0.5f, 0.0f, 0.0f);
 *     printf("Distance: %f\n", distance);
 *
 *     StringResult hlsl = alice_sdf_to_hlsl(shape);
 *     if (hlsl.result == SdfResult_Ok) {
 *         printf("HLSL:\n%s\n", hlsl.data);
 *         alice_sdf_free_string(hlsl.data);
 *     }
 *
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
 * @brief Null handle constant
 */
#define SDF_HANDLE_NULL ((SdfHandle)0)

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

/* ============================================================================
 * Boolean Operations
 * ============================================================================ */

/**
 * @brief Union of two SDFs (A ∪ B)
 */
SdfHandle alice_sdf_union(SdfHandle a, SdfHandle b);

/**
 * @brief Intersection of two SDFs (A ∩ B)
 */
SdfHandle alice_sdf_intersection(SdfHandle a, SdfHandle b);

/**
 * @brief Subtraction of SDFs (A - B)
 */
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

/* ============================================================================
 * Transforms
 * ============================================================================ */

/**
 * @brief Translate an SDF
 */
SdfHandle alice_sdf_translate(SdfHandle node, float x, float y, float z);

/**
 * @brief Rotate an SDF using quaternion (x, y, z, w)
 */
SdfHandle alice_sdf_rotate(SdfHandle node, float qx, float qy, float qz, float qw);

/**
 * @brief Rotate an SDF using Euler angles (radians)
 */
SdfHandle alice_sdf_rotate_euler(SdfHandle node, float x, float y, float z);

/**
 * @brief Uniform scale an SDF
 */
SdfHandle alice_sdf_scale(SdfHandle node, float factor);

/**
 * @brief Non-uniform scale an SDF
 */
SdfHandle alice_sdf_scale_xyz(SdfHandle node, float x, float y, float z);

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

/* ============================================================================
 * Evaluation
 * ============================================================================ */

/**
 * @brief Evaluate SDF at a single point
 * @return Signed distance (negative = inside, positive = outside)
 */
float alice_sdf_eval(SdfHandle node, float x, float y, float z);

/**
 * @brief Evaluate SDF at multiple points (parallel)
 * @param points Array of floats [x0, y0, z0, x1, y1, z1, ...]
 * @param distances Output array (must be pre-allocated)
 * @param count Number of points
 */
BatchResult alice_sdf_eval_batch(SdfHandle node,
                                  const float* points,
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

/**
 * @brief Free an SDF handle
 */
void alice_sdf_free(SdfHandle node);

/**
 * @brief Free a string returned by shader generation
 */
void alice_sdf_free_string(char* str);

/**
 * @brief Clone an SDF handle
 */
SdfHandle alice_sdf_clone(SdfHandle node);

/* ============================================================================
 * Utilities
 * ============================================================================ */

/**
 * @brief Get node count in an SDF tree
 */
uint32_t alice_sdf_node_count(SdfHandle node);

/**
 * @brief Check if a handle is valid
 */
bool alice_sdf_is_valid(SdfHandle node);

#ifdef __cplusplus
}
#endif

#endif /* ALICE_SDF_H */
