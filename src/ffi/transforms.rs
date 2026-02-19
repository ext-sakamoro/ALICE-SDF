//! FFI functions for spatial transforms (translate, rotate, scale, projective, lattice, skinning).
//!
//! Author: Moroya Sakamoto

use super::registry::{get_node, register_node};
use super::types::*;
use std::slice;

// ============================================================================
// Transforms
// ============================================================================

/// Translate an SDF
#[no_mangle]
pub extern "C" fn alice_sdf_translate(node: SdfHandle, x: f32, y: f32, z: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().translate(x, y, z);
    register_node(new_node)
}

/// Rotate an SDF using quaternion
#[no_mangle]
pub extern "C" fn alice_sdf_rotate(
    node: SdfHandle,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let quat = glam::Quat::from_xyzw(qx, qy, qz, qw).normalize();
    let new_node = (*sdf_node).clone().rotate(quat);
    register_node(new_node)
}

/// Rotate an SDF using Euler angles (radians)
#[no_mangle]
pub extern "C" fn alice_sdf_rotate_euler(node: SdfHandle, x: f32, y: f32, z: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().rotate_euler(x, y, z);
    register_node(new_node)
}

/// Uniform scale an SDF
#[no_mangle]
pub extern "C" fn alice_sdf_scale(node: SdfHandle, factor: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().scale(factor);
    register_node(new_node)
}

/// Non-uniform scale an SDF
#[no_mangle]
pub extern "C" fn alice_sdf_scale_xyz(node: SdfHandle, x: f32, y: f32, z: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().scale_xyz(x, y, z);
    register_node(new_node)
}

// ============================================================================
// Advanced Transforms
// ============================================================================

/// Apply projective transformation (4x4 matrix)
///
/// # Parameters
/// - `node`: SDF handle
/// - `inv_matrix`: Pointer to 16 floats (4x4 matrix in column-major order)
/// - `lipschitz_bound`: Lipschitz constant bound for distance field correction
///
/// # Safety
/// `inv_matrix` must point to valid array of 16 floats
#[no_mangle]
pub extern "C" fn alice_sdf_projective_transform(
    node: SdfHandle,
    inv_matrix: *const f32,
    lipschitz_bound: f32,
) -> SdfHandle {
    if inv_matrix.is_null() {
        return SDF_HANDLE_NULL;
    }
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    // SAFETY: Caller guarantees inv_matrix points to a valid, aligned array of at least 16 f32
    // values that remains valid for the lifetime of the returned slice. This is an FFI contract.
    let matrix = unsafe { slice::from_raw_parts(inv_matrix, 16) };
    let matrix_array: [f32; 16] = match matrix.try_into() {
        Ok(m) => m,
        Err(_) => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node)
        .clone()
        .projective_transform(matrix_array, lipschitz_bound);
    register_node(new_node)
}

/// Apply lattice-based Free-Form Deformation (FFD)
///
/// # Parameters
/// - `node`: SDF handle
/// - `control_points`: Pointer to control point array (3 floats per point: x, y, z)
/// - `cp_count`: Number of control points (must equal nx * ny * nz)
/// - `nx`, `ny`, `nz`: Lattice dimensions
/// - `bbox_min`: Pointer to 3 floats (min corner of bounding box)
/// - `bbox_max`: Pointer to 3 floats (max corner of bounding box)
///
/// # Safety
/// All pointers must be valid and point to correctly sized arrays
#[no_mangle]
pub extern "C" fn alice_sdf_lattice_deform(
    node: SdfHandle,
    control_points: *const f32,
    cp_count: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    bbox_min: *const f32,
    bbox_max: *const f32,
) -> SdfHandle {
    if control_points.is_null() || bbox_min.is_null() || bbox_max.is_null() {
        return SDF_HANDLE_NULL;
    }
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };

    let expected_count = (nx * ny * nz) as usize * 3;
    if cp_count as usize * 3 != expected_count {
        return SDF_HANDLE_NULL;
    }

    // SAFETY: Caller guarantees control_points points to a valid array of at least cp_count * 3
    // f32 values (3 floats per control point: x, y, z). Null check performed above. FFI contract.
    let cp_slice = unsafe { slice::from_raw_parts(control_points, cp_count as usize * 3) };
    // SAFETY: Caller guarantees bbox_min points to a valid array of at least 3 f32 values.
    // Null check performed above. Both pointers remain valid for the duration of this call.
    let min_slice = unsafe { slice::from_raw_parts(bbox_min, 3) };
    // SAFETY: Caller guarantees bbox_max points to a valid array of at least 3 f32 values.
    // Null check performed above. Both pointers remain valid for the duration of this call.
    let max_slice = unsafe { slice::from_raw_parts(bbox_max, 3) };

    let control_pts: Vec<glam::Vec3> = cp_slice
        .chunks_exact(3)
        .map(|c| glam::Vec3::new(c[0], c[1], c[2]))
        .collect();

    let bbox_min_vec = glam::Vec3::new(min_slice[0], min_slice[1], min_slice[2]);
    let bbox_max_vec = glam::Vec3::new(max_slice[0], max_slice[1], max_slice[2]);

    let new_node =
        (*sdf_node)
            .clone()
            .lattice_deform(control_pts, nx, ny, nz, bbox_min_vec, bbox_max_vec);
    register_node(new_node)
}

/// Apply skeletal skinning deformation with bone weights
///
/// # Parameters
/// - `node`: SDF handle
/// - `bones`: Pointer to bone data (each bone: 16 floats inv_bind_pose + 16 floats current_pose + 1 float weight = 33 floats)
/// - `bone_count`: Number of bones
///
/// # Safety
/// `bones` must point to valid array of bone_count * 33 floats
#[no_mangle]
pub extern "C" fn alice_sdf_skinning(
    node: SdfHandle,
    bones: *const f32,
    bone_count: u32,
) -> SdfHandle {
    if bones.is_null() {
        return SDF_HANDLE_NULL;
    }
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };

    use crate::transforms::skinning::BoneTransform;

    // SAFETY: Caller guarantees bones points to a valid array of at least bone_count * 33 f32
    // values (16 inv_bind_pose + 16 current_pose + 1 weight per bone). Null check performed
    // above. The pointer remains valid for the duration of this call. This is an FFI contract.
    let bone_slice = unsafe { slice::from_raw_parts(bones, bone_count as usize * 33) };
    let mut bone_transforms = Vec::new();

    for chunk in bone_slice.chunks_exact(33) {
        let inv_bind: [f32; 16] = chunk[0..16].try_into().unwrap();
        let current: [f32; 16] = chunk[16..32].try_into().unwrap();
        let weight = chunk[32];

        bone_transforms.push(BoneTransform {
            inv_bind_pose: inv_bind,
            current_pose: current,
            weight,
        });
    }

    let new_node = (*sdf_node).clone().sdf_skinning(bone_transforms);
    register_node(new_node)
}
