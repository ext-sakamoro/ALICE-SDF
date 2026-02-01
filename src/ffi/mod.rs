//! Foreign Function Interface (FFI) for ALICE-SDF (Deep Fried Edition)
//!
//! This module provides high-performance C-compatible bindings for:
//! - C/C++ (Native, UE5)
//! - C# (Unity)
//! - Python (ctypes/cffi)
//!
//! # Deep Fried Features
//!
//! - **Pre-compilation**: Compile SDF once, evaluate millions of times
//! - **Zero-copy output**: Write directly to caller-allocated buffers
//! - **SoA support**: Structure-of-Arrays for SIMD vectorization
//! - **Parallel evaluation**: Automatic Rayon parallelization for large batches
//!
//! # Performance Hierarchy
//!
//! | Function                        | Speed      | Use Case                    |
//! |---------------------------------|------------|-----------------------------|
//! | `alice_sdf_eval_soa`            | ★★★★★     | Physics, particles, tracing |
//! | `alice_sdf_eval_compiled_batch` | ★★★★☆     | General batch evaluation    |
//! | `alice_sdf_eval_batch`          | ★★★☆☆     | Convenience (auto-compile)  |
//! | `alice_sdf_eval`                | ★☆☆☆☆     | Debugging only              |
//!
//! # Usage from C/C++
//!
//! ```c
//! #include "alice_sdf.h"
//!
//! int main() {
//!     // Create SDF
//!     SdfHandle sphere = alice_sdf_sphere(1.0f);
//!     SdfHandle box = alice_sdf_box(0.5f, 0.5f, 0.5f);
//!     SdfHandle shape = alice_sdf_smooth_union(sphere, box, 0.2f);
//!
//!     // Compile for fast evaluation (do once at startup)
//!     CompiledHandle compiled = alice_sdf_compile(shape);
//!
//!     // Batch evaluate with SoA layout (fastest path)
//!     float x[1024], y[1024], z[1024], dist[1024];
//!     // ... fill x, y, z arrays ...
//!     alice_sdf_eval_soa(compiled, x, y, z, dist, 1024);
//!
//!     // Clean up
//!     alice_sdf_free_compiled(compiled);
//!     alice_sdf_free(shape);
//!     alice_sdf_free(box);
//!     alice_sdf_free(sphere);
//!     return 0;
//! }
//! ```
//!
//! # Usage from C# (Unity)
//!
//! ```csharp
//! using System;
//! using System.Runtime.InteropServices;
//!
//! public static class AliceSdf {
//!     const string LibName = "alice_sdf";
//!
//!     [DllImport(LibName)] public static extern IntPtr alice_sdf_sphere(float radius);
//!     [DllImport(LibName)] public static extern IntPtr alice_sdf_compile(IntPtr node);
//!     [DllImport(LibName)] public static extern BatchResult alice_sdf_eval_soa(
//!         IntPtr compiled, float[] x, float[] y, float[] z, float[] dist, uint count);
//!     [DllImport(LibName)] public static extern void alice_sdf_free(IntPtr node);
//!     [DllImport(LibName)] public static extern void alice_sdf_free_compiled(IntPtr compiled);
//! }
//!
//! // High-performance usage pattern
//! public class SdfEvaluator : IDisposable {
//!     private IntPtr _node;
//!     private IntPtr _compiled;
//!
//!     public SdfEvaluator(IntPtr node) {
//!         _node = node;
//!         _compiled = AliceSdf.alice_sdf_compile(node);
//!     }
//!
//!     public void Evaluate(float[] x, float[] y, float[] z, float[] dist) {
//!         AliceSdf.alice_sdf_eval_soa(_compiled, x, y, z, dist, (uint)x.Length);
//!     }
//!
//!     public void Dispose() {
//!         if (_compiled != IntPtr.Zero) AliceSdf.alice_sdf_free_compiled(_compiled);
//!         // Note: caller owns _node
//!     }
//! }
//! ```
//!
//! # Thread Safety
//!
//! All FFI functions are thread-safe:
//! - Handles can be created and used from multiple threads
//! - Compiled handles can be shared across threads
//! - Batch evaluation automatically parallelizes across cores
//!
//! # Memory Management
//!
//! - Call `alice_sdf_free()` for every SDF handle
//! - Call `alice_sdf_free_compiled()` for every compiled handle
//! - Call `alice_sdf_free_string()` for every string from shader functions
//!
//! Author: Moroya Sakamoto

mod types;
mod registry;
mod api;

pub use types::*;
pub use api::*;
