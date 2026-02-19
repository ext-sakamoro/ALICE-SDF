//! FFI functions for mesh generation and export (OBJ, GLB, USDA, Alembic, FBX).
//!
//! Author: Moroya Sakamoto

use super::registry::{get_mesh, get_node, register_mesh, remove_mesh};
use super::types::*;
use crate::prelude::*;
use std::ffi::c_char;

// ============================================================================
// Mesh Generation & Export
// ============================================================================

/// Generate mesh from SDF via Marching Cubes (returns reusable MeshHandle)
///
/// Generate once, export to multiple formats without re-computing.
/// Call `alice_sdf_free_mesh` when done.
#[no_mangle]
pub extern "C" fn alice_sdf_generate_mesh(
    node: SdfHandle,
    resolution: u32,
    bounds: f32,
) -> MeshHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return MESH_HANDLE_NULL,
    };

    let config = MarchingCubesConfig {
        resolution: resolution as usize,
        iso_level: 0.0,
        compute_normals: true,
        ..Default::default()
    };
    let min = glam::Vec3::splat(-bounds);
    let max = glam::Vec3::splat(bounds);

    let mesh = sdf_to_mesh(&sdf_node, min, max, &config);
    register_mesh(mesh)
}

/// Get vertex count of a mesh
#[no_mangle]
pub extern "C" fn alice_sdf_mesh_vertex_count(mesh: MeshHandle) -> u32 {
    match get_mesh(mesh) {
        Some(m) => m.vertex_count() as u32,
        None => 0,
    }
}

/// Get triangle count of a mesh
#[no_mangle]
pub extern "C" fn alice_sdf_mesh_triangle_count(mesh: MeshHandle) -> u32 {
    match get_mesh(mesh) {
        Some(m) => m.triangle_count() as u32,
        None => 0,
    }
}

/// Free a mesh handle
#[no_mangle]
pub extern "C" fn alice_sdf_free_mesh(mesh: MeshHandle) {
    if !mesh.is_null() {
        remove_mesh(mesh);
    }
}

/// Helper: extract path string from C pointer
///
/// # Safety
///
/// - `path` must be a valid, null-terminated C string pointer that remains valid for the
///   duration of the call. The pointer may be null (handled internally with `NullPointer` error).
pub(super) unsafe fn path_from_c(path: *const c_char) -> Result<&'static str, SdfResult> {
    if path.is_null() {
        return Err(SdfResult::NullPointer);
    }
    // SAFETY: Caller guarantees path is a valid, null-terminated C string pointer that remains
    // valid for the duration of this call. Null check performed above.
    std::ffi::CStr::from_ptr(path)
        .to_str()
        .map_err(|_| SdfResult::InvalidParameter)
}

/// Helper: resolve mesh from either MeshHandle or SdfHandle
///
/// # Safety
///
/// - If `mesh_handle` is non-null it must be a valid handle previously returned by this library.
/// - If `sdf_handle` is used as fallback it must be a valid handle previously returned by this
///   library.
/// - This function is safe to call with null/invalid handles; those cases return `Err`.
pub(super) unsafe fn resolve_mesh(
    mesh_handle: MeshHandle,
    sdf_handle: SdfHandle,
    resolution: u32,
    bounds: f32,
) -> Result<std::sync::Arc<crate::mesh::Mesh>, SdfResult> {
    // Prefer pre-generated MeshHandle
    if !mesh_handle.is_null() {
        return get_mesh(mesh_handle).ok_or(SdfResult::InvalidHandle);
    }

    // Fallback: generate from SDF on the fly
    let sdf_node = get_node(sdf_handle).ok_or(SdfResult::InvalidHandle)?;
    let config = MarchingCubesConfig {
        resolution: resolution as usize,
        iso_level: 0.0,
        compute_normals: true,
        ..Default::default()
    };
    let min = glam::Vec3::splat(-bounds);
    let max = glam::Vec3::splat(bounds);
    Ok(std::sync::Arc::new(sdf_to_mesh(
        &sdf_node, min, max, &config,
    )))
}

/// Export mesh to OBJ file
///
/// Pass a pre-generated MeshHandle (from `alice_sdf_generate_mesh`) for best performance.
/// Alternatively, pass a null MeshHandle + valid SdfHandle to generate on the fly.
///
/// # Safety
///
/// - `path` must be a valid, null-terminated C string pointer that remains valid for the call.
/// - `mesh_handle` and `sdf_handle` must be valid handles (or null for the unused path).
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_export_obj(
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
    match crate::io::export_obj(&mesh, path_str, &ObjConfig::default(), None) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

/// Export mesh to GLB (binary glTF) file
///
/// # Safety
///
/// - `path` must be a valid, null-terminated C string pointer that remains valid for the call.
/// - `mesh_handle` and `sdf_handle` must be valid handles (or null for the unused path).
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_export_glb(
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
    match crate::io::export_glb(&mesh, path_str, &GltfConfig::default(), None) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

/// Export mesh to USDA (Universal Scene Description) file
///
/// # Safety
///
/// - `path` must be a valid, null-terminated C string pointer that remains valid for the call.
/// - `mesh_handle` and `sdf_handle` must be valid handles (or null for the unused path).
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_export_usda(
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
    match crate::io::export_usda(&mesh, path_str, &UsdConfig::default(), None) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

/// Export mesh to Alembic (.abc) file
///
/// # Safety
///
/// - `path` must be a valid, null-terminated C string pointer that remains valid for the call.
/// - `mesh_handle` and `sdf_handle` must be valid handles (or null for the unused path).
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_export_alembic(
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
    match crate::io::export_alembic(&mesh, path_str, &AlembicConfig::default()) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

/// Export mesh to FBX file
///
/// # Safety
///
/// - `path` must be a valid, null-terminated C string pointer that remains valid for the call.
/// - `mesh_handle` and `sdf_handle` must be valid handles (or null for the unused path).
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_export_fbx(
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
    match crate::io::export_fbx(&mesh, path_str, &FbxConfig::default(), None) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}
