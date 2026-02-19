//! FFI functions for library version info and category counts.
//!
//! Author: Moroya Sakamoto

use super::types::*;
use crate::prelude::*;
use std::ffi::{c_char, CString};
use std::ptr;

// ============================================================================
// Library Info
// ============================================================================

/// Get the library version
#[no_mangle]
pub extern "C" fn alice_sdf_version() -> VersionInfo {
    VersionInfo::current()
}

/// Get version string (caller must free with alice_sdf_free_string)
#[no_mangle]
pub extern "C" fn alice_sdf_version_string() -> *mut c_char {
    let version = format!(
        "ALICE-SDF v{}.{}.{} (Deep Fried)",
        VersionInfo::current().major,
        VersionInfo::current().minor,
        VersionInfo::current().patch
    );
    match CString::new(version) {
        Ok(s) => s.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

// ============================================================================
// Category Counts
// ============================================================================

/// Get number of primitive SDF variants
#[no_mangle]
pub extern "C" fn alice_sdf_primitive_count() -> u32 {
    SdfCategory::Primitive.count()
}

/// Get number of operation SDF variants
#[no_mangle]
pub extern "C" fn alice_sdf_operation_count() -> u32 {
    SdfCategory::Operation.count()
}

/// Get number of transform SDF variants
#[no_mangle]
pub extern "C" fn alice_sdf_transform_count() -> u32 {
    SdfCategory::Transform.count()
}

/// Get number of modifier SDF variants
#[no_mangle]
pub extern "C" fn alice_sdf_modifier_count() -> u32 {
    SdfCategory::Modifier.count()
}

/// Get total number of all SDF variants
#[no_mangle]
pub extern "C" fn alice_sdf_total_count() -> u32 {
    SdfCategory::total()
}
