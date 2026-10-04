//! Rust source emit: an [`SdfNode`](crate::types::SdfNode) as a dependency-free
//! `fn sdf(x, y, z) -> f32` plus `fn normal(x, y, z) -> (f32, f32, f32)`.
//!
//! The intended caller is a downstream `build.rs`: the same `SdfNode` that
//! produces the GLSL / WGSL / HLSL shader also produces a CPU evaluator, and
//! the runtime crate only depends on `alice-det-math` (for the
//! transcendentals), not on `alice-sdf`. That keeps a WASM build from pulling
//! in the evaluator, the compiler and every law it does not use.
//!
//! # Determinism
//!
//! The emitted code is the straight-line unrolling of the bytecode that
//! [`eval_compiled`](crate::compiled::eval_compiled) interprets: the same
//! instruction stream, the same precomputed constants (`1/k`, `k/4`,
//! `1/scale`, …), the same operation order and the same `alice_det_math`
//! kernels. Rust does not contract `a * b + c` into an FMA, so the result is
//! bit-identical to `eval_compiled` for every supported opcode, and `normal`
//! is bit-identical to
//! [`eval_compiled_normal`](crate::compiled::eval_compiled_normal) with the
//! epsilon fixed at transpile time (`tests/test_rust_transpiler_oracle.rs`
//! compiles the output with `rustc` and compares `to_bits()` on a grid).
//!
//! # Coverage
//!
//! Opcodes without an emitter (aux-buffer driven ones such as lattices,
//! skinning, IFS and heightmaps, noise, and the primitives not listed in
//! [`is_supported`](crate::compiled::rust::is_supported)) are rejected with
//! [`RustTranspileError::UnsupportedOpcode`](crate::compiled::rust::RustTranspileError::UnsupportedOpcode) — there is no fallback value.
//!
//! ```rust,ignore
//! // build.rs of the consuming crate
//! use alice_sdf::compiled::rust::RustSource;
//! use alice_sdf::prelude::*;
//!
//! let shape = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);
//! let src = RustSource::transpile(&shape).expect("supported scene");
//! let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("shape.rs");
//! std::fs::write(out, src.source()).unwrap();
//! // in the runtime crate: include!(concat!(env!("OUT_DIR"), "/shape.rs"));
//! ```
//!
//! Author: Moroya Sakamoto

mod transpiler;

pub use transpiler::{
    is_supported, RustSource, RustTranspileError, RustTranspileOptions, DEFAULT_NORMAL_EPSILON,
};
