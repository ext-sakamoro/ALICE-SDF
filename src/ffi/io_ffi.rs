//! FFI functions for file I/O: ASDF save/load, ABM binary mesh, Unity/UE5 export, LOD chains.
//!
//! Author: Moroya Sakamoto

use super::mesh::{path_from_c, resolve_mesh};
use super::registry::{get_mesh, get_node, register_mesh, register_node};
use super::types::*;
use crate::prelude::*;
use std::ffi::c_char;
use std::slice;

// ============================================================================
// File I/O
// ============================================================================

/// Save SDF to .asdf file
#[no_mangle]
pub extern "C" fn alice_sdf_save(node: SdfHandle, path: *const c_char) -> SdfResult {
    if path.is_null() {
        return SdfResult::NullPointer;
    }

    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SdfResult::InvalidHandle,
    };

    // SAFETY: Caller guarantees path is a valid, null-terminated C string pointer
    // that remains valid for the duration of this call. Null check performed above.
    let path_str = unsafe {
        match std::ffi::CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return SdfResult::InvalidParameter,
        }
    };

    let tree = SdfTree::new((*sdf_node).clone());
    match crate::save(&tree, path_str) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

/// Load SDF from .asdf file
#[no_mangle]
pub extern "C" fn alice_sdf_load(path: *const c_char) -> SdfHandle {
    if path.is_null() {
        return SDF_HANDLE_NULL;
    }

    // SAFETY: Caller guarantees path is a valid, null-terminated C string pointer
    // that remains valid for the duration of this call. Null check performed above.
    let path_str = unsafe {
        match std::ffi::CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return SDF_HANDLE_NULL,
        }
    };

    match crate::load(path_str) {
        Ok(tree) => register_node(tree.root),
        Err(_) => SDF_HANDLE_NULL,
    }
}

// ============================================================================
// ABM Binary Mesh I/O
// ============================================================================

/// Save a mesh to ABM (ALICE Binary Mesh) format
///
/// Pass a pre-generated MeshHandle (from `alice_sdf_generate_mesh`) for best performance.
/// Alternatively, pass a null MeshHandle + valid SdfHandle to generate on the fly.
///
/// # Safety
///
/// - `path` must be a valid, null-terminated C string pointer that remains valid for the call.
/// - `mesh_handle` and `sdf_handle` must be valid handles (or null for the unused path).
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_save_abm(
    mesh_handle: MeshHandle,
    sdf_handle: SdfHandle,
    path: *const c_char,
    resolution: u32,
    bounds: f32,
) -> SdfResult {
    let path_str = match path_from_c(path) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let mesh = match resolve_mesh(mesh_handle, sdf_handle, resolution, bounds) {
        Ok(m) => m,
        Err(e) => return e,
    };
    match crate::io::abm::save_abm(&mesh, path_str) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

/// Load a mesh from ABM (ALICE Binary Mesh) format and return a MeshHandle
///
/// Returns `MESH_HANDLE_NULL` on failure (invalid path, corrupt file, etc.).
/// The caller must free the returned handle with `alice_sdf_free_mesh()`.
///
/// # Safety
///
/// - `path` must be a valid, null-terminated C string pointer that remains valid for the call.
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_load_abm(path: *const c_char) -> MeshHandle {
    let path_str = match path_from_c(path) {
        Ok(s) => s,
        Err(_) => return MESH_HANDLE_NULL,
    };
    match crate::io::abm::load_abm(path_str) {
        Ok(mesh) => register_mesh(mesh),
        Err(_) => MESH_HANDLE_NULL,
    }
}

// ============================================================================
// Unity Mesh Export
// ============================================================================

/// Export mesh to Unity JSON format (.unity_mesh)
///
/// # Safety
///
/// - `path` must be a valid, null-terminated C string pointer that remains valid for the call.
/// - `mesh_handle` and `sdf_handle` must be valid handles (or null for the unused path).
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_export_unity(
    mesh_handle: MeshHandle,
    sdf_handle: SdfHandle,
    path: *const c_char,
    resolution: u32,
    bounds: f32,
    flip_z: bool,
    flip_winding: bool,
    scale: f32,
) -> SdfResult {
    let path_str = match path_from_c(path) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let mesh = match resolve_mesh(mesh_handle, sdf_handle, resolution, bounds) {
        Ok(m) => m,
        Err(e) => return e,
    };
    let config = crate::io::unity_mesh::UnityMeshConfig {
        flip_z,
        flip_winding,
        scale,
        name: "alice_mesh".to_string(),
    };
    match crate::io::unity_mesh::export_unity_mesh(&mesh, path_str, &config) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

/// Export mesh to Unity binary format (.unity_mesh_bin)
///
/// # Safety
///
/// - `path` must be a valid, null-terminated C string pointer that remains valid for the call.
/// - `mesh_handle` and `sdf_handle` must be valid handles (or null for the unused path).
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_export_unity_binary(
    mesh_handle: MeshHandle,
    sdf_handle: SdfHandle,
    path: *const c_char,
    resolution: u32,
    bounds: f32,
    flip_z: bool,
    flip_winding: bool,
    scale: f32,
) -> SdfResult {
    let path_str = match path_from_c(path) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let mesh = match resolve_mesh(mesh_handle, sdf_handle, resolution, bounds) {
        Ok(m) => m,
        Err(e) => return e,
    };
    let config = crate::io::unity_mesh::UnityMeshConfig {
        flip_z,
        flip_winding,
        scale,
        name: "alice_mesh".to_string(),
    };
    match crate::io::unity_mesh::export_unity_mesh_binary(&mesh, path_str, &config) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

// ============================================================================
// UE5 Mesh Export
// ============================================================================

/// Export mesh to UE5 JSON format (.ue5_mesh)
///
/// # Safety
///
/// - `path` must be a valid, null-terminated C string pointer that remains valid for the call.
/// - `mesh_handle` and `sdf_handle` must be valid handles (or null for the unused path).
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_export_ue5(
    mesh_handle: MeshHandle,
    sdf_handle: SdfHandle,
    path: *const c_char,
    resolution: u32,
    bounds: f32,
    scale: f32,
) -> SdfResult {
    let path_str = match path_from_c(path) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let mesh = match resolve_mesh(mesh_handle, sdf_handle, resolution, bounds) {
        Ok(m) => m,
        Err(e) => return e,
    };
    let config = crate::io::ue5_asset::Ue5MeshConfig {
        scale,
        name: "alice_mesh".to_string(),
        lod_index: 0,
    };
    match crate::io::ue5_asset::export_ue5_mesh(&mesh, path_str, &config) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

/// Export mesh to UE5 binary format (.ue5_mesh_bin)
///
/// # Safety
///
/// - `path` must be a valid, null-terminated C string pointer that remains valid for the call.
/// - `mesh_handle` and `sdf_handle` must be valid handles (or null for the unused path).
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_export_ue5_binary(
    mesh_handle: MeshHandle,
    sdf_handle: SdfHandle,
    path: *const c_char,
    resolution: u32,
    bounds: f32,
    scale: f32,
) -> SdfResult {
    let path_str = match path_from_c(path) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let mesh = match resolve_mesh(mesh_handle, sdf_handle, resolution, bounds) {
        Ok(m) => m,
        Err(e) => return e,
    };
    let config = crate::io::ue5_asset::Ue5MeshConfig {
        scale,
        name: "alice_mesh".to_string(),
        lod_index: 0,
    };
    match crate::io::ue5_asset::export_ue5_mesh_binary(&mesh, path_str, &config) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

// ============================================================================
// LOD Chain Persistence
// ============================================================================

/// Save a LOD chain (multi-resolution meshes) to ABM + JSON sidecar
///
/// # Safety
///
/// - `mesh_handles` must point to a valid array of at least `lod_count` MeshHandle values that
///   were previously returned by this library. The pointer must be non-null (checked internally).
/// - `transition_distances` must point to a valid array of at least `lod_count` f32 values.
///   The pointer must be non-null (checked internally).
/// - `path` must be a valid, null-terminated C string pointer that remains valid for the call.
/// - All three pointers must remain valid for the entire duration of the call.
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_save_lod_chain(
    mesh_handles: *const MeshHandle,
    transition_distances: *const f32,
    lod_count: u32,
    path: *const c_char,
) -> SdfResult {
    if mesh_handles.is_null() || transition_distances.is_null() {
        return SdfResult::NullPointer;
    }
    let path_str = match path_from_c(path) {
        Ok(s) => s,
        Err(e) => return e,
    };

    let count = lod_count as usize;
    // SAFETY: Caller guarantees mesh_handles is a non-null pointer to at least lod_count valid
    // MeshHandle values. Null check performed above. Pointer remains valid for this call.
    let handles = slice::from_raw_parts(mesh_handles, count);
    // SAFETY: Caller guarantees transition_distances is a non-null pointer to at least lod_count
    // f32 values. Null check performed above. Pointer remains valid for this call.
    let distances = slice::from_raw_parts(transition_distances, count);

    let mut meshes = Vec::with_capacity(count);
    let mut dist_vec = Vec::with_capacity(count);
    for i in 0..count {
        let mesh_arc = match get_mesh(handles[i]) {
            Some(m) => m,
            None => return SdfResult::InvalidHandle,
        };
        meshes.push((*mesh_arc).clone());
        dist_vec.push(distances[i]);
    }

    let chain = crate::mesh::lod_persist::LodChainPersist::new(
        meshes,
        dist_vec,
        0,
        crate::mesh::lod_persist::LodChainConfig::default(),
    );

    match crate::mesh::lod_persist::save_lod_chain(&chain, path_str) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

/// Load a LOD chain from ABM + JSON sidecar and return mesh handles
///
/// On success, writes up to `max_lod_count` MeshHandle values into `out_mesh_handles`
/// and up to `max_lod_count` f32 values into `out_transition_distances`.
/// Returns the actual number of LOD levels loaded. Returns 0 on failure.
///
/// # Safety
///
/// - `path` must be a valid, null-terminated C string pointer that remains valid for the call.
/// - `out_mesh_handles` must point to a writable array of at least `max_lod_count` MeshHandle
///   entries. The pointer must be non-null (checked internally).
/// - `out_transition_distances` must point to a writable array of at least `max_lod_count` f32
///   entries. The pointer must be non-null (checked internally).
/// - All three pointers must remain valid for the entire duration of the call.
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_load_lod_chain(
    path: *const c_char,
    out_mesh_handles: *mut MeshHandle,
    out_transition_distances: *mut f32,
    max_lod_count: u32,
) -> u32 {
    if out_mesh_handles.is_null() || out_transition_distances.is_null() {
        return 0;
    }
    let path_str = match path_from_c(path) {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let chain = match crate::mesh::lod_persist::load_lod_chain(path_str) {
        Ok(c) => c,
        Err(_) => return 0,
    };

    let count = chain.meshes.len().min(max_lod_count as usize);
    // SAFETY: Caller guarantees out_mesh_handles is a non-null, writable pointer to at least
    // max_lod_count MeshHandle entries. count <= max_lod_count so we stay within bounds. Null
    // check performed above. Pointer remains valid and non-aliasing for this call.
    let out_handles = slice::from_raw_parts_mut(out_mesh_handles, count);
    // SAFETY: Caller guarantees out_transition_distances is a non-null, writable pointer to at
    // least max_lod_count f32 entries. count <= max_lod_count so we stay within bounds. Null
    // check performed above. Does not alias out_handles.
    let out_dists = slice::from_raw_parts_mut(out_transition_distances, count);

    for i in 0..count {
        let level = &chain.meshes[i];
        out_handles[i] = register_mesh(level.mesh.clone());
        out_dists[i] = level.transition_distance;
    }

    count as u32
}
