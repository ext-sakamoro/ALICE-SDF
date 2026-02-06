//! WGSL Transpiler and GPU Evaluation (Deep Fried Edition)
//!
//! This module provides GPU-accelerated SDF evaluation using WebGPU.
//! It transpiles SDF node trees to WGSL (WebGPU Shading Language) and
//! executes them on the GPU for massively parallel evaluation.
//!
//! # Deep Fried Optimizations
//!
//! - **Division Exorcism**: Smooth ops use pre-computed reciprocals
//! - **Async Execution**: Non-blocking GPU dispatch for CPU/GPU overlap
//! - **Inline Operations**: No helper function call overhead
//!
//! # Performance Characteristics
//!
//! | Batch Size | CPU (SIMD) | GPU | GPU (Deep Fried) |
//! |------------|------------|-----|------------------|
//! | 100 | ~10µs | ~500µs | ~500µs (overhead) |
//! | 10,000 | ~1ms | ~600µs | ~400µs |
//! | 1,000,000 | ~100ms | ~5ms | ~3ms |
//!
//! GPU evaluation is most efficient for large batches (10,000+ points).
//!
//! # Usage
//!
//! ```rust,ignore
//! use alice_sdf::prelude::*;
//! use alice_sdf::compiled::wgsl::{WgslShader, GpuEvaluator};
//!
//! let shape = SdfNode::sphere(1.0)
//!     .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);
//!
//! // Method 1: Direct GPU evaluation (sync)
//! let gpu = GpuEvaluator::new(&shape).unwrap();
//! let points: Vec<Vec3> = generate_points();
//! let distances = gpu.eval_batch(&points).unwrap();
//!
//! // Method 2: Async GPU evaluation (non-blocking)
//! let gpu = GpuEvaluator::new_async(&shape).await.unwrap();
//! let distances = gpu.eval_batch_async(&points).await.unwrap();
//!
//! // Method 3: Submit and do other work
//! let future = gpu.eval_batch_submit(points);
//! // ... do CPU work while GPU computes ...
//! let distances = future.wait().unwrap();
//!
//! // Method 4: Get WGSL for custom use
//! let shader = WgslShader::transpile(&shape);
//! println!("{}", shader.source);  // Print optimized WGSL code
//! ```
//!
//! # Architecture
//!
//! The WGSL transpiler (Deep Fried Edition):
//! 1. Traverses the SDF tree
//! 2. Generates WGSL code with unique variable names
//! 3. **Inlines smooth operations with pre-computed reciprocals**
//! 4. **Applies constant folding for identity transforms**
//! 5. Produces a complete compute shader
//!
//! The GPU evaluator:
//! 1. Creates WebGPU device and queue
//! 2. Compiles the optimized WGSL shader
//! 3. Manages input/output buffers
//! 4. Dispatches compute workgroups
//! 5. **Supports async read-back for CPU/GPU overlap**
//!
//! # Supported Operations
//!
//! All SdfNode variants are supported except:
//! - `Noise`: Would require texture sampling or complex noise functions
//!
//! Author: Moroya Sakamoto

mod transpiler;
mod gpu_eval;

pub use transpiler::{WgslShader, TranspileMode};
pub use gpu_eval::{GpuEvaluator, GpuError, GpuEvalFuture, GpuBufferPool};
