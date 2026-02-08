//! FFI-safe type definitions for ALICE-SDF (Deep Fried Edition)
//!
//! These types are designed to be safely passed across FFI boundaries
//! to C, C++, C#, and other languages.
//!
//! # Deep Fried Features
//! - Zero-copy batch evaluation
//! - Pre-compiled SDF handles for maximum throughput
//! - SoA (Structure of Arrays) data layout support
//!
//! Author: Moroya Sakamoto

use std::ffi::c_char;

/// Opaque handle to an SDF node
///
/// This is a pointer-sized handle that uniquely identifies an SDF node.
/// The actual SdfNode is stored in a global registry and accessed via this handle.
pub type SdfHandle = *mut std::ffi::c_void;

/// Opaque handle to a pre-compiled SDF (bytecode)
///
/// Compiled handles evaluate ~10x faster than raw SdfHandle.
/// Create once with `alice_sdf_compile`, reuse many times.
pub type CompiledHandle = *mut std::ffi::c_void;

/// Opaque handle to a generated mesh
///
/// Generate once with `alice_sdf_generate_mesh`, export to multiple formats.
pub type MeshHandle = *mut std::ffi::c_void;

/// Null handle constant
pub const SDF_HANDLE_NULL: SdfHandle = std::ptr::null_mut();

/// Null compiled handle constant
pub const COMPILED_HANDLE_NULL: CompiledHandle = std::ptr::null_mut();

/// Null mesh handle constant
pub const MESH_HANDLE_NULL: MeshHandle = std::ptr::null_mut();

/// 3D vector for FFI (C-compatible layout)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Vec3Ffi {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3Ffi {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

impl From<glam::Vec3> for Vec3Ffi {
    fn from(v: glam::Vec3) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl From<Vec3Ffi> for glam::Vec3 {
    fn from(v: Vec3Ffi) -> Self {
        glam::Vec3::new(v.x, v.y, v.z)
    }
}

/// Quaternion for FFI (C-compatible layout)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QuatFfi {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl QuatFfi {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    pub fn identity() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
}

impl Default for QuatFfi {
    fn default() -> Self {
        Self::identity()
    }
}

impl From<glam::Quat> for QuatFfi {
    fn from(q: glam::Quat) -> Self {
        Self {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        }
    }
}

impl From<QuatFfi> for glam::Quat {
    fn from(q: QuatFfi) -> Self {
        glam::Quat::from_xyzw(q.x, q.y, q.z, q.w)
    }
}

/// Result code for FFI operations
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdfResult {
    /// Operation succeeded
    Ok = 0,
    /// Invalid handle provided
    InvalidHandle = 1,
    /// Null pointer provided
    NullPointer = 2,
    /// Invalid parameter value
    InvalidParameter = 3,
    /// Out of memory
    OutOfMemory = 4,
    /// IO error (file operations)
    IoError = 5,
    /// Compilation failed
    CompileError = 6,
    /// Unknown error
    Unknown = 99,
}

/// Shader type enum for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderType {
    /// WebGPU Shading Language
    Wgsl = 0,
    /// High-Level Shading Language (DirectX/UE5)
    Hlsl = 1,
    /// OpenGL Shading Language (Unity/OpenGL)
    Glsl = 2,
}

/// Batch evaluation result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BatchResult {
    /// Number of points evaluated
    pub count: u32,
    /// Result code
    pub result: SdfResult,
}

/// String result for shader generation
#[repr(C)]
pub struct StringResult {
    /// Pointer to null-terminated UTF-8 string (caller must free with alice_sdf_free_string)
    pub data: *mut c_char,
    /// Length of string (not including null terminator)
    pub len: u32,
    /// Result code
    pub result: SdfResult,
}

impl Default for StringResult {
    fn default() -> Self {
        Self {
            data: std::ptr::null_mut(),
            len: 0,
            result: SdfResult::Ok,
        }
    }
}

/// Version information
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VersionInfo {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl VersionInfo {
    pub fn current() -> Self {
        Self {
            major: 0,
            minor: 1,
            patch: 0,
        }
    }
}

/// SoA (Structure of Arrays) batch evaluation configuration
///
/// For maximum throughput, use SoA layout where X, Y, Z coordinates
/// are stored in separate contiguous arrays. This enables SIMD
/// vectorization and better cache utilization.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SoaBatchConfig {
    /// Pointer to X coordinates array
    pub x: *const f32,
    /// Pointer to Y coordinates array
    pub y: *const f32,
    /// Pointer to Z coordinates array
    pub z: *const f32,
    /// Output distances array (caller-allocated)
    pub distances: *mut f32,
    /// Number of points (arrays must all have this length)
    pub count: u32,
    /// Padding for alignment
    pub _padding: u32,
}

/// Evaluation statistics (for profiling)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EvalStats {
    /// Number of points evaluated
    pub points_evaluated: u64,
    /// Time in nanoseconds (if timing enabled)
    pub time_ns: u64,
    /// Points per second throughput
    pub throughput: f64,
}
