//! FFI functions for memory management, cloning, and validity checks.
//!
//! Author: Moroya Sakamoto

use super::registry::{get_compiled, get_node, register_node, remove_node};
use super::types::*;
use std::ffi::{c_char, CString};

// ============================================================================
// Memory Management
// ============================================================================

/// Free an SDF handle
#[no_mangle]
pub extern "C" fn alice_sdf_free(node: SdfHandle) {
    if !node.is_null() {
        remove_node(node);
    }
}

/// Free a string returned by shader generation functions
#[no_mangle]
pub extern "C" fn alice_sdf_free_string(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: The string was previously allocated by CString::into_raw() in this module
        // (via alice_sdf_version_string or shader generation functions). This reclaims ownership
        // and frees the memory. Must be called exactly once per allocation.
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

/// Clone an SDF handle (creates a new reference)
#[no_mangle]
pub extern "C" fn alice_sdf_clone(node: SdfHandle) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    register_node((*sdf_node).clone())
}

// ============================================================================
// Utilities
// ============================================================================

/// Get the node count in an SDF tree
#[no_mangle]
pub extern "C" fn alice_sdf_node_count(node: SdfHandle) -> u32 {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return 0,
    };
    sdf_node.node_count()
}

/// Check if a handle is valid
#[no_mangle]
pub extern "C" fn alice_sdf_is_valid(node: SdfHandle) -> bool {
    get_node(node).is_some()
}

/// Check if a compiled handle is valid
#[no_mangle]
pub extern "C" fn alice_sdf_is_compiled_valid(compiled: CompiledHandle) -> bool {
    get_compiled(compiled).is_some()
}
