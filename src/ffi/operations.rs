//! FFI functions for boolean/CSG operations on SDFs.
//!
//! Author: Moroya Sakamoto

use super::registry::{get_node, register_node};
use super::types::*;

// ============================================================================
// Boolean Operations
// ============================================================================

/// Union of two SDFs (A ∪ B)
#[no_mangle]
pub extern "C" fn alice_sdf_union(a: SdfHandle, b: SdfHandle) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().union((*node_b).clone());
    register_node(node)
}

/// Intersection of two SDFs (A ∩ B)
#[no_mangle]
pub extern "C" fn alice_sdf_intersection(a: SdfHandle, b: SdfHandle) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().intersection((*node_b).clone());
    register_node(node)
}

/// Subtraction of SDFs (A - B)
#[no_mangle]
pub extern "C" fn alice_sdf_subtract(a: SdfHandle, b: SdfHandle) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().subtract((*node_b).clone());
    register_node(node)
}

/// Smooth union of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_smooth_union(a: SdfHandle, b: SdfHandle, k: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().smooth_union((*node_b).clone(), k);
    register_node(node)
}

/// Smooth intersection of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_smooth_intersection(a: SdfHandle, b: SdfHandle, k: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().smooth_intersection((*node_b).clone(), k);
    register_node(node)
}

/// Smooth subtraction of SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_smooth_subtract(a: SdfHandle, b: SdfHandle, k: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().smooth_subtract((*node_b).clone(), k);
    register_node(node)
}

/// Chamfer union of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_chamfer_union(a: SdfHandle, b: SdfHandle, r: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().chamfer_union((*node_b).clone(), r);
    register_node(node)
}

/// Chamfer intersection of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_chamfer_intersection(a: SdfHandle, b: SdfHandle, r: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().chamfer_intersection((*node_b).clone(), r);
    register_node(node)
}

/// Chamfer subtraction of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_chamfer_subtract(a: SdfHandle, b: SdfHandle, r: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().chamfer_subtract((*node_b).clone(), r);
    register_node(node)
}

/// Stairs union of two SDFs (stepped/terraced blend)
#[no_mangle]
pub extern "C" fn alice_sdf_stairs_union(a: SdfHandle, b: SdfHandle, r: f32, n: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().stairs_union((*node_b).clone(), r, n);
    register_node(node)
}

/// Stairs intersection of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_stairs_intersection(
    a: SdfHandle,
    b: SdfHandle,
    r: f32,
    n: f32,
) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a)
        .clone()
        .stairs_intersection((*node_b).clone(), r, n);
    register_node(node)
}

/// Stairs subtraction of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_stairs_subtract(
    a: SdfHandle,
    b: SdfHandle,
    r: f32,
    n: f32,
) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().stairs_subtract((*node_b).clone(), r, n);
    register_node(node)
}

/// XOR (symmetric difference) of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_xor(a: SdfHandle, b: SdfHandle) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().xor((*node_b).clone());
    register_node(node)
}

/// Morph (linear interpolation) between two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_morph(a: SdfHandle, b: SdfHandle, t: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().morph((*node_b).clone(), t);
    register_node(node)
}

/// Columns union of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_columns_union(a: SdfHandle, b: SdfHandle, r: f32, n: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().columns_union((*node_b).clone(), r, n);
    register_node(node)
}

/// Columns intersection of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_columns_intersection(
    a: SdfHandle,
    b: SdfHandle,
    r: f32,
    n: f32,
) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a)
        .clone()
        .columns_intersection((*node_b).clone(), r, n);
    register_node(node)
}

/// Columns subtraction of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_columns_subtract(
    a: SdfHandle,
    b: SdfHandle,
    r: f32,
    n: f32,
) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().columns_subtract((*node_b).clone(), r, n);
    register_node(node)
}

/// Pipe operation: cylindrical surface at intersection of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_pipe(a: SdfHandle, b: SdfHandle, r: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().pipe((*node_b).clone(), r);
    register_node(node)
}

/// Engrave shape b into shape a
#[no_mangle]
pub extern "C" fn alice_sdf_engrave(a: SdfHandle, b: SdfHandle, r: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().engrave((*node_b).clone(), r);
    register_node(node)
}

/// Groove: cut a groove of shape b into shape a
#[no_mangle]
pub extern "C" fn alice_sdf_groove(a: SdfHandle, b: SdfHandle, ra: f32, rb: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().groove((*node_b).clone(), ra, rb);
    register_node(node)
}

/// Tongue: add a tongue protrusion of shape b to shape a
#[no_mangle]
pub extern "C" fn alice_sdf_tongue(a: SdfHandle, b: SdfHandle, ra: f32, rb: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().tongue((*node_b).clone(), ra, rb);
    register_node(node)
}

/// Exponential smooth union of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_exp_smooth_union(a: SdfHandle, b: SdfHandle, k: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().exp_smooth_union((*node_b).clone(), k);
    register_node(node)
}

/// Exponential smooth intersection of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_exp_smooth_intersection(
    a: SdfHandle,
    b: SdfHandle,
    k: f32,
) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a)
        .clone()
        .exp_smooth_intersection((*node_b).clone(), k);
    register_node(node)
}

/// Exponential smooth subtraction of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_exp_smooth_subtract(a: SdfHandle, b: SdfHandle, k: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().exp_smooth_subtract((*node_b).clone(), k);
    register_node(node)
}
