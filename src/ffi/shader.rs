//! FFI functions for GPU shader code generation (WGSL, HLSL, GLSL).
//!
//! Author: Moroya Sakamoto

#[cfg(any(feature = "gpu", feature = "hlsl", feature = "glsl"))]
use super::registry::get_node;
#[cfg(any(feature = "gpu", feature = "hlsl", feature = "glsl"))]
use super::types::*;
#[cfg(any(feature = "gpu", feature = "hlsl", feature = "glsl"))]
use std::ffi::CString;
#[cfg(any(feature = "gpu", feature = "hlsl", feature = "glsl"))]
use std::ptr;

// ============================================================================
// Shader Generation
// ============================================================================

/// Generate WGSL shader code
#[no_mangle]
#[cfg(feature = "gpu")]
pub extern "C" fn alice_sdf_to_wgsl(node: SdfHandle) -> StringResult {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => {
            return StringResult {
                data: ptr::null_mut(),
                len: 0,
                result: SdfResult::InvalidHandle,
            }
        }
    };

    let shader = crate::compiled::WgslShader::transpile(
        &sdf_node,
        crate::compiled::TranspileMode::Hardcoded,
    );
    match CString::new(shader.source.clone()) {
        Ok(s) => {
            let len = shader.source.len() as u32;
            StringResult {
                data: s.into_raw(),
                len,
                result: SdfResult::Ok,
            }
        }
        Err(_) => StringResult {
            data: ptr::null_mut(),
            len: 0,
            result: SdfResult::Unknown,
        },
    }
}

/// Generate HLSL shader code
#[no_mangle]
#[cfg(feature = "hlsl")]
pub extern "C" fn alice_sdf_to_hlsl(node: SdfHandle) -> StringResult {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => {
            return StringResult {
                data: ptr::null_mut(),
                len: 0,
                result: SdfResult::InvalidHandle,
            }
        }
    };

    let shader = crate::compiled::HlslShader::transpile(
        &sdf_node,
        crate::compiled::HlslTranspileMode::Hardcoded,
    );
    match CString::new(shader.source.clone()) {
        Ok(s) => {
            let len = shader.source.len() as u32;
            StringResult {
                data: s.into_raw(),
                len,
                result: SdfResult::Ok,
            }
        }
        Err(_) => StringResult {
            data: ptr::null_mut(),
            len: 0,
            result: SdfResult::Unknown,
        },
    }
}

/// Generate GLSL shader code
#[no_mangle]
#[cfg(feature = "glsl")]
pub extern "C" fn alice_sdf_to_glsl(node: SdfHandle) -> StringResult {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => {
            return StringResult {
                data: ptr::null_mut(),
                len: 0,
                result: SdfResult::InvalidHandle,
            }
        }
    };

    let shader = crate::compiled::GlslShader::transpile(
        &sdf_node,
        crate::compiled::GlslTranspileMode::Hardcoded,
    );
    match CString::new(shader.source.clone()) {
        Ok(s) => {
            let len = shader.source.len() as u32;
            StringResult {
                data: s.into_raw(),
                len,
                result: SdfResult::Ok,
            }
        }
        Err(_) => StringResult {
            data: ptr::null_mut(),
            len: 0,
            result: SdfResult::Unknown,
        },
    }
}
