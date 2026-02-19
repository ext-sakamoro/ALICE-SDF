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

use super::codegen::{extract_jit_params, JitCompiler};
use crate::types::SdfNode;

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

// SAFETY: JitCompiledSdf holds a read-only function pointer to JIT-compiled machine code.
// After compilation completes, the code is immutable and safe to call from any thread.
// The JITModule is held alive by the struct, preventing use-after-free.
unsafe impl Send for JitCompiledSdf {}
// SAFETY: See Send impl above. The function pointer is read-only and thread-safe.
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
        flag_builder
            .set("use_colocated_libcalls", "true")
            .map_err(|e| JitError::ModuleError(e.to_string()))?;

        // Optimize for speed
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| JitError::ModuleError(e.to_string()))?;

        let isa_builder =
            cranelift_native::builder().map_err(|e| JitError::ModuleError(e.to_string()))?;

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
        module
            .finalize_definitions()
            .map_err(|e| JitError::ModuleError(e.to_string()))?;

        // Get the function pointer
        let code_ptr = module.get_finalized_function(func_id);

        // SAFETY: code_ptr was compiled by Cranelift with verified extern "C" ABI signature
        // matching (f32, f32, f32) -> f32. The module has been finalized, so the code is
        // ready to execute. The JITModule is moved into the struct, keeping the code alive.
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
        // SAFETY: eval_fn was transmuted from a Cranelift-compiled function with the
        // extern "C" fn(f32, f32, f32) -> f32 signature. The JITModule (_module) is held
        // alive by the struct, so the function pointer remains valid for the struct's lifetime.
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
        points.iter().map(|p| self.eval(*p)).collect()
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

        points.par_iter().map(|p| self.eval(*p)).collect()
    }
}

impl std::fmt::Debug for JitCompiledSdf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCompiledSdf")
            .field("eval_fn", &format!("{:p}", self.eval_fn as *const ()))
            .finish()
    }
}

// ============ Dynamic Parameter Support ============

/// Function signature for dynamic JIT: fn(x, y, z, params_ptr) -> distance
type SdfEvalDynamicFn = unsafe extern "C" fn(f32, f32, f32, *const f32) -> f32;

/// JIT-compiled SDF with dynamic parameter support (Deep Fried v2)
///
/// Unlike `JitCompiledSdf`, this variant loads shape parameters from a
/// runtime buffer instead of baking them as constants. This enables
/// zero-latency parameter updates (e.g., 60fps animation) without
/// recompilation.
///
/// # Usage
///
/// ```rust,ignore
/// let shape = SdfNode::sphere(1.0).translate(2.0, 0.0, 0.0);
/// let mut jit = JitCompiledSdfDynamic::compile(&shape).unwrap();
///
/// // Evaluate (reads params from internal buffer)
/// let d = jit.eval(Vec3::new(0.5, 0.0, 0.0));
///
/// // Update params without recompilation
/// let shape2 = SdfNode::sphere(1.5).translate(3.0, 0.0, 0.0);
/// jit.update_params(&shape2);
///
/// // Evaluate with new params instantly
/// let d2 = jit.eval(Vec3::new(0.5, 0.0, 0.0));
/// ```
pub struct JitCompiledSdfDynamic {
    _module: JITModule,
    eval_fn: SdfEvalDynamicFn,
    params: Vec<f32>,
}

// SAFETY: JitCompiledSdfDynamic holds a read-only function pointer to JIT-compiled machine code
// and an owned params Vec. After compilation, the code is immutable. The params Vec is only
// mutated through &mut self methods. The JITModule is held alive by the struct.
unsafe impl Send for JitCompiledSdfDynamic {}
// SAFETY: See Send impl above. The function pointer is read-only and thread-safe.
// Shared references only read params, which is safe across threads.
unsafe impl Sync for JitCompiledSdfDynamic {}

impl JitCompiledSdfDynamic {
    /// Compile an SDF node tree with dynamic parameter support
    pub fn compile(node: &SdfNode) -> Result<Self, JitError> {
        let mut flag_builder = settings::builder();

        flag_builder
            .set("use_colocated_libcalls", "true")
            .map_err(|e| JitError::ModuleError(e.to_string()))?;
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| JitError::ModuleError(e.to_string()))?;

        let isa_builder =
            cranelift_native::builder().map_err(|e| JitError::ModuleError(e.to_string()))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| JitError::ModuleError(e.to_string()))?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let mut compiler = JitCompiler::new(&mut module);
        let (func_id, params) = compiler.compile_sdf_dynamic(node)?;

        module
            .finalize_definitions()
            .map_err(|e| JitError::ModuleError(e.to_string()))?;

        let code_ptr = module.get_finalized_function(func_id);
        // SAFETY: code_ptr was compiled by Cranelift with verified extern "C" ABI signature
        // matching (f32, f32, f32, *const f32) -> f32. The module has been finalized, so the
        // code is ready to execute. The JITModule is moved into the struct, keeping the code alive.
        let eval_fn: SdfEvalDynamicFn = unsafe { mem::transmute(code_ptr) };

        Ok(JitCompiledSdfDynamic {
            _module: module,
            eval_fn,
            params,
        })
    }

    /// Update parameters from a modified SDF tree (zero-latency, no recompilation)
    ///
    /// The tree structure must be identical to the one used for compilation.
    /// Only shape parameter values (radius, position, etc.) may change.
    pub fn update_params(&mut self, node: &SdfNode) {
        self.params = extract_jit_params(node);
    }

    /// Get current parameter values
    pub fn params(&self) -> &[f32] {
        &self.params
    }

    /// Evaluate the SDF at a point
    #[inline]
    pub fn eval(&self, point: Vec3) -> f32 {
        // SAFETY: eval_fn was transmuted from a Cranelift-compiled function with the
        // extern "C" fn(f32, f32, f32, *const f32) -> f32 signature. The JITModule (_module)
        // is held alive by the struct. self.params.as_ptr() is valid because params is an
        // owned Vec kept alive by the struct.
        unsafe { (self.eval_fn)(point.x, point.y, point.z, self.params.as_ptr()) }
    }

    /// Evaluate the SDF at multiple points
    pub fn eval_batch(&self, points: &[Vec3]) -> Vec<f32> {
        points.iter().map(|p| self.eval(*p)).collect()
    }

    /// Evaluate the SDF at multiple points in parallel
    pub fn eval_batch_parallel(&self, points: &[Vec3]) -> Vec<f32> {
        use rayon::prelude::*;
        points.par_iter().map(|p| self.eval(*p)).collect()
    }
}

impl std::fmt::Debug for JitCompiledSdfDynamic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCompiledSdfDynamic")
            .field("eval_fn", &format!("{:p}", self.eval_fn as *const ()))
            .field("params_count", &self.params.len())
            .finish()
    }
}
