//! FFI functions for library version info and category counts.
//!
//! Author: Moroya Sakamoto

use super::guard::{clear_last_error, ffi_guard, take_last_error};
use super::types::*;
use crate::prelude::*;
use std::ffi::{c_char, CString};
use std::ptr;

// ============================================================================
// Library Info
// ============================================================================

/// Get the library version
#[no_mangle]
pub const extern "C" fn alice_sdf_version() -> VersionInfo {
    VersionInfo::current()
}

/// Get version string (caller must free with alice_sdf_free_string)
#[no_mangle]
pub extern "C" fn alice_sdf_version_string() -> *mut c_char {
    ffi_guard(std::ptr::null_mut(), || {
        let version = format!(
            "ALICE-SDF v{}.{}.{} (Deep Fried)",
            VersionInfo::current().major,
            VersionInfo::current().minor,
            VersionInfo::current().patch
        );
        CString::new(version).map_or_else(|_| ptr::null_mut(), CString::into_raw)
    })
}

// ============================================================================
// Error reporting
// ============================================================================

/// Message of the most recent panic / error raised by an FFI call on this
/// thread, or null if there is none. Every exported function catches panics
/// internally (returning its sentinel: null handle, `f32::MAX`, `0`, `false`
/// or `SdfResult_Unknown`) and records the message here; the message is
/// consumed by this call. Caller must free with `alice_sdf_free_string`.
#[no_mangle]
pub extern "C" fn alice_sdf_last_error() -> *mut c_char {
    ffi_guard(ptr::null_mut(), || {
        take_last_error()
            .and_then(|m| CString::new(m).ok())
            .map_or_else(ptr::null_mut, CString::into_raw)
    })
}

/// Discard the most recent error message on this thread.
#[no_mangle]
pub extern "C" fn alice_sdf_clear_last_error() {
    // cannot panic in practice; guarded so the "every extern fn is guarded"
    // invariant holds without an allowlist
    ffi_guard((), clear_last_error);
}

// ============================================================================
// Category Counts
// ============================================================================

/// Get number of primitive SDF variants
#[no_mangle]
pub const extern "C" fn alice_sdf_primitive_count() -> u32 {
    SdfCategory::Primitive.count()
}

/// Get number of operation SDF variants
#[no_mangle]
pub const extern "C" fn alice_sdf_operation_count() -> u32 {
    SdfCategory::Operation.count()
}

/// Get number of transform SDF variants
#[no_mangle]
pub const extern "C" fn alice_sdf_transform_count() -> u32 {
    SdfCategory::Transform.count()
}

/// Get number of modifier SDF variants
#[no_mangle]
pub const extern "C" fn alice_sdf_modifier_count() -> u32 {
    SdfCategory::Modifier.count()
}

/// Get total number of all SDF variants
#[no_mangle]
pub const extern "C" fn alice_sdf_total_count() -> u32 {
    SdfCategory::total()
}
