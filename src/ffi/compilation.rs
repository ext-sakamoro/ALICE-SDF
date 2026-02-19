//! FFI functions for compiling SDFs to bytecode.
//!
//! Author: Moroya Sakamoto

use super::registry::{get_compiled, get_node, register_compiled, remove_compiled};
use super::types::*;
use crate::compiled::CompiledSdf;

// ============================================================================
// Compilation (Deep Fried)
// ============================================================================

/// Compile an SDF to bytecode for fast evaluation
///
/// This is expensive (~0.1ms), but the resulting CompiledHandle evaluates
/// ~10x faster. Compile once at setup time, reuse every frame.
///
/// # Returns
/// A CompiledHandle that can be used with `alice_sdf_eval_compiled*` functions.
/// Returns NULL on failure.
#[no_mangle]
pub extern "C" fn alice_sdf_compile(node: SdfHandle) -> CompiledHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return COMPILED_HANDLE_NULL,
    };

    let compiled = CompiledSdf::compile(&sdf_node);
    register_compiled(compiled)
}

/// Free a compiled SDF handle
#[no_mangle]
pub extern "C" fn alice_sdf_free_compiled(compiled: CompiledHandle) {
    if !compiled.is_null() {
        remove_compiled(compiled);
    }
}

/// Get instruction count of a compiled SDF (for profiling)
#[no_mangle]
pub extern "C" fn alice_sdf_compiled_instruction_count(compiled: CompiledHandle) -> u32 {
    match get_compiled(compiled) {
        Some(c) => c.instruction_count() as u32,
        None => 0,
    }
}
