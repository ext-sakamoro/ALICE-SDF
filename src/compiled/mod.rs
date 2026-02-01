//! Compiled SDF: Cache-efficient linearized evaluation
//!
//! This module provides a "compiled" representation of SDF trees that
//! trades flexibility for performance. The recursive Arc-based tree
//! is converted to a flat instruction array, eliminating:
//!
//! - Pointer chasing (Arc indirection → cache misses)
//! - Branch misprediction (deep recursion → hard to predict)
//! - Function call overhead (recursive eval → stack manipulation)
//!
//! # Performance Characteristics
//!
//! | Aspect | Interpreted | Compiled | Compiled+SIMD |
//! |--------|-------------|----------|---------------|
//! | Memory layout | Scattered (heap) | Contiguous (Vec) | Contiguous |
//! | Cache efficiency | Poor | Excellent | Excellent |
//! | Parallelism | 1 point | 1 point | 8 points |
//! | Best for | Editing | Single eval | Batch eval |
//!
//! # Usage
//!
//! ```rust
//! use alice_sdf::prelude::*;
//! use alice_sdf::compiled::{CompiledSdf, eval_compiled};
//!
//! // Create an SDF tree
//! let node = SdfNode::sphere(1.0)
//!     .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);
//!
//! // Compile it once
//! let compiled = CompiledSdf::compile(&node);
//!
//! // Evaluate many times (fast!)
//! for _ in 0..1000000 {
//!     let d = eval_compiled(&compiled, glam::Vec3::new(0.5, 0.5, 0.5));
//! }
//! ```
//!
//! # SIMD Batch Evaluation (Phase 2)
//!
//! For batch evaluation, use the SIMD-accelerated functions:
//!
//! ```rust
//! use alice_sdf::prelude::*;
//! use alice_sdf::compiled::{CompiledSdf, eval_compiled_batch_simd_parallel};
//!
//! let node = SdfNode::sphere(1.0);
//! let compiled = CompiledSdf::compile(&node);
//!
//! // Generate many points
//! let points: Vec<Vec3> = (0..100000)
//!     .map(|i| Vec3::new(i as f32 * 0.01, 0.0, 0.0))
//!     .collect();
//!
//! // Evaluate all points using SIMD + multi-threading
//! let distances = eval_compiled_batch_simd_parallel(&compiled, &points);
//! ```
//!
//! # Architecture
//!
//! The compiled representation uses a stack-based virtual machine:
//!
//! - **Value Stack**: Holds intermediate SDF distance values
//! - **Coordinate Stack**: Holds saved coordinate frames for transforms
//! - **Instruction Array**: Sequential opcodes with inline parameters
//!
//! ## Instruction Format
//!
//! Each instruction is 32 bytes (half a cache line):
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │ params[6]: f32[6]  (24 bytes)                   │
//! │ opcode: u8         (1 byte)                     │
//! │ flags: u8          (1 byte)                     │
//! │ child_count: u16   (2 bytes)                    │
//! │ skip_offset: u32   (4 bytes)                    │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! ## Evaluation Order
//!
//! - **Primitives**: Push distance to value stack
//! - **Binary ops**: Pop 2 values, push result
//! - **Transforms**: Push coord frame, modify point
//! - **PopTransform**: Restore coord frame, apply post-processing
//!
//! ## SIMD Evaluation (8-wide)
//!
//! The SIMD evaluator processes 8 points simultaneously using AVX2/NEON:
//!
//! - `Vec3x8`: 8 3D vectors in SoA layout
//! - `f32x8`: 8 scalar values for distances
//! - `Quatx8`: 8 quaternions for rotations
//!
//! ## BVH Acceleration (Phase 3)
//!
//! The BVH evaluator attaches AABBs to each instruction for spatial pruning:
//!
//! - `AabbPacked`: 32-byte aligned AABB for cache efficiency
//! - `CompiledSdfBvh`: Compiled SDF with BVH data
//! - `get_scene_aabb()`: Get the bounding box of the entire scene
//!
//! BVH is most effective for:
//! - Sparse scenes with distant objects
//! - Raymarching (each step can skip distant geometry)
//! - Broad-phase collision detection
//!
//! ## JIT Compilation (Phase 4) - Feature: `jit`
//!
//! The JIT compiler converts SDF trees directly to native machine code:
//!
//! ```rust,ignore
//! use alice_sdf::prelude::*;
//! use alice_sdf::compiled::jit::JitCompiledSdf;
//!
//! let shape = SdfNode::sphere(1.0)
//!     .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);
//!
//! // JIT compile once
//! let jit = JitCompiledSdf::compile(&shape).unwrap();
//!
//! // Evaluate with zero interpreter overhead
//! let d = jit.eval(Vec3::new(0.5, 0.0, 0.0));
//! ```
//!
//! Enable with: `cargo build --features jit`
//!
//! ## SoA Memory Layout (Phase 5)
//!
//! For maximum SIMD throughput, use Structure of Arrays layout:
//!
//! ```rust,ignore
//! use alice_sdf::prelude::*;
//! use alice_sdf::soa::SoAPoints;
//! use alice_sdf::compiled::{CompiledSdf, eval_compiled_batch_soa_parallel};
//!
//! let shape = SdfNode::sphere(1.0);
//! let compiled = CompiledSdf::compile(&shape);
//!
//! // Convert to SoA layout (AoS → SoA)
//! let points: SoAPoints = (0..100000)
//!     .map(|i| Vec3::new(i as f32 * 0.01, 0.0, 0.0))
//!     .collect();
//!
//! // Direct SIMD loads without shuffle
//! let distances = eval_compiled_batch_soa_parallel(&compiled, &points);
//! ```
//!
//! SoA benefits:
//! - No gather/shuffle overhead for SIMD loads
//! - Better cache line utilization
//! - CPU prefetch-friendly sequential access
//!
//! Author: Moroya Sakamoto

mod opcode;
mod instruction;
mod compiler;
mod eval;
mod simd;
mod eval_simd;
mod aabb;
mod eval_bvh;
mod eval_soa;

#[cfg(feature = "jit")]
pub mod jit;

#[cfg(feature = "jit")]
pub mod jit_simd;

#[cfg(feature = "gpu")]
pub mod wgsl;

#[cfg(feature = "hlsl")]
pub mod hlsl;

#[cfg(feature = "glsl")]
pub mod glsl;

#[cfg(feature = "gpu")]
pub use wgsl::{
    WgslShader,
    GpuEvaluator,
    GpuError,
    GpuEvalFuture,
};

#[cfg(feature = "hlsl")]
pub use hlsl::HlslShader;

#[cfg(feature = "glsl")]
pub use glsl::GlslShader;

pub use opcode::OpCode;
pub use instruction::Instruction;
pub use compiler::CompiledSdf;
pub use eval::{
    eval_compiled,
    eval_compiled_normal,
    eval_compiled_batch,
    eval_compiled_batch_parallel,
};
pub use simd::{Vec3x8, Quatx8};
pub use eval_simd::{
    eval_compiled_simd,
    eval_compiled_batch_simd,
    eval_compiled_batch_simd_parallel,
    eval_gradient_simd,
    eval_distance_and_gradient_simd,
};
pub use aabb::AabbPacked;
pub use eval_bvh::{
    CompiledSdfBvh,
    eval_compiled_bvh,
    get_scene_aabb,
};
pub use eval_soa::{
    eval_compiled_batch_soa,
    eval_compiled_batch_soa_parallel,
    eval_compiled_batch_soa_into,
    eval_compiled_batch_soa_raw,
};
