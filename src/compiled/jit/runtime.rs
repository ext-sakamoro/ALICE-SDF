//! JIT Runtime: Compiled function execution
//!
//! This module provides the runtime infrastructure for executing JIT-compiled
//! SDF evaluation functions.
//!
//! Author: Moroya Sakamoto

use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::{JITBuilder, JITModule};
use glam::Vec3;
use std::mem;
use thiserror::Error;

use crate::types::SdfNode;
use super::codegen::JitCompiler;

/// Error type for JIT compilation
#[derive(Error, Debug)]
pub enum JitError {
    /// Cranelift module error
    #[error("Module error: {0}")]
    ModuleError(String),

    /// Compilation error
    #[error("Compilation error: {0}")]
    CompilationError(String),

    /// Unsupported SDF node type
    #[error("Unsupported node type: {0}")]
    UnsupportedNode(String),
}

/// Function signature for JIT-compiled SDF evaluation
/// Takes (x, y, z) and returns distance
type SdfEvalFn = unsafe extern "C" fn(f32, f32, f32) -> f32;

/// JIT-compiled SDF evaluation
///
/// This struct holds the JIT-compiled machine code for evaluating an SDF.
/// The compilation happens once, and then the function can be called many times
/// with minimal overhead.
pub struct JitCompiledSdf {
    /// The JIT module (keeps compiled code alive)
    _module: JITModule,

    /// Function pointer to the compiled evaluation function
    eval_fn: SdfEvalFn,
}

// SAFETY: The JIT module and function pointer are thread-safe once compiled
unsafe impl Send for JitCompiledSdf {}
unsafe impl Sync for JitCompiledSdf {}

impl JitCompiledSdf {
    /// Compile an SDF node tree to native machine code
    ///
    /// # Arguments
    ///
    /// * `node` - The root SDF node to compile
    ///
    /// # Returns
    ///
    /// A `JitCompiledSdf` that can evaluate the SDF at any point
    ///
    /// # Errors
    ///
    /// Returns `JitError` if compilation fails
    pub fn compile(node: &SdfNode) -> Result<Self, JitError> {
        // Create JIT builder with settings configured for the current platform
        let mut flag_builder = settings::builder();

        // Use colocated libcalls to avoid PLT issues on ARM64
        flag_builder.set("use_colocated_libcalls", "true")
            .map_err(|e| JitError::ModuleError(e.to_string()))?;

        // Optimize for speed
        flag_builder.set("opt_level", "speed")
            .map_err(|e| JitError::ModuleError(e.to_string()))?;

        let isa_builder = cranelift_native::builder()
            .map_err(|e| JitError::ModuleError(e.to_string()))?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| JitError::ModuleError(e.to_string()))?;

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Create JIT module
        let mut module = JITModule::new(builder);

        // Create compiler and compile the SDF
        let mut compiler = JitCompiler::new(&mut module);
        let func_id = compiler.compile_sdf(node)?;

        // Finalize the module
        module.finalize_definitions()
            .map_err(|e| JitError::ModuleError(e.to_string()))?;

        // Get the function pointer
        let code_ptr = module.get_finalized_function(func_id);

        // SAFETY: We just compiled this function with the correct signature
        let eval_fn: SdfEvalFn = unsafe { mem::transmute(code_ptr) };

        Ok(JitCompiledSdf {
            _module: module,
            eval_fn,
        })
    }

    /// Evaluate the SDF at a point
    ///
    /// # Arguments
    ///
    /// * `point` - The 3D point to evaluate
    ///
    /// # Returns
    ///
    /// The signed distance from the point to the surface
    #[inline]
    pub fn eval(&self, point: Vec3) -> f32 {
        // SAFETY: The function pointer is valid and has the correct signature
        unsafe { (self.eval_fn)(point.x, point.y, point.z) }
    }

    /// Evaluate the SDF at multiple points
    ///
    /// # Arguments
    ///
    /// * `points` - Slice of 3D points to evaluate
    ///
    /// # Returns
    ///
    /// Vector of signed distances, one per input point
    pub fn eval_batch(&self, points: &[Vec3]) -> Vec<f32> {
        points.iter()
            .map(|p| self.eval(*p))
            .collect()
    }

    /// Evaluate the SDF at multiple points in parallel
    ///
    /// # Arguments
    ///
    /// * `points` - Slice of 3D points to evaluate
    ///
    /// # Returns
    ///
    /// Vector of signed distances, one per input point
    pub fn eval_batch_parallel(&self, points: &[Vec3]) -> Vec<f32> {
        use rayon::prelude::*;

        points.par_iter()
            .map(|p| self.eval(*p))
            .collect()
    }
}

impl std::fmt::Debug for JitCompiledSdf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCompiledSdf")
            .field("eval_fn", &format!("{:p}", self.eval_fn as *const ()))
            .finish()
    }
}
