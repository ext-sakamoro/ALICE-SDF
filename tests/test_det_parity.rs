//! Bit-exact parity of every CPU evaluator (3.1.0, alice-det-math).
//!
//! Since 3.1.0 every transcendental in the laws and evaluators goes through
//! `alice-det-math` and `a * b + c` is never fused, so the tree evaluator,
//! the compiled scalar evaluator, the `f32x8` SIMD evaluator, the BVH
//! evaluator and the Cranelift JITs must agree **to the bit** — not within a
//! tolerance — on every corpus node and every sample point. This is the
//! oracle for the cross-platform guarantee: the same bits here, on x86_64,
//! aarch64 and wasm32 (`tests/test_det_golden.rs` pins them).
//!
//! JIT: the SIMD JIT (`JitSimdSdf`) is bit-exact too; it has no codegen arm
//! for some opcodes and must say so with its "no codegen arm" error — any
//! *other* compile error is a failure (before 3.1.0 twist / bend failed the
//! Cranelift verifier and the tolerance test silently skipped them). The
//! scalar tree JIT (`JitCompiledSdf`) is held to a 1e-5 tolerance only: its
//! lowering has not been aligned to the law's operation order yet.
//!
//! Author: Moroya Sakamoto

mod common;

use alice_sdf::compiled::{
    eval_compiled, eval_compiled_batch_simd, eval_compiled_bvh, CompileError, CompiledSdf,
    CompiledSdfBvh,
};
use alice_sdf::prelude::*;
use common::corpus::corpus;

/// Fixed points (incl. cell-boundary ties) plus a deterministic LCG spray.
fn sample_points() -> Vec<Vec3> {
    let mut pts = vec![
        Vec3::ZERO,
        Vec3::new(0.25, 0.0, 0.0),
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.3, -0.7, 1.1),
        Vec3::new(1.5, 1.5, 1.5),
        Vec3::new(-2.0, 0.4, -0.9),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(3.0, -1.0, 1.0),
        Vec3::new(0.0, -1.5, 0.0),
        Vec3::new(-3.0, 0.0, 0.0),
        Vec3::new(-3.0, -3.0, 3.0),
        Vec3::new(0.0, -2.25, 0.0),
        // exact sector / axis ties for atan2-based laws
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(-0.5, 0.3, 0.0),
        Vec3::new(0.0, 0.3, 0.7),
    ];
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32) * 6.0 - 3.0
    };
    for _ in 0..248 {
        pts.push(Vec3::new(next(), next(), next()));
    }
    pts
}

fn bits(v: f32) -> String {
    format!("{v:.7e} ({:#010x})", v.to_bits())
}

#[test]
fn every_cpu_evaluator_is_bit_identical() {
    let pts = sample_points();
    let mut failures = Vec::new();
    #[allow(unused_mut)]
    let mut jit_unsupported: Vec<String> = Vec::new();
    let mut bytecode_unsupported: Vec<String> = Vec::new();
    for (name, node) in corpus() {
        // Nodes with no bytecode law (Terrain, Triangle, Bezier) have only
        // the tree evaluator to compare against itself — not a parity
        // question. `test_evaluator_opcode_parity.rs` pins the rejection.
        let compiled = match CompiledSdf::try_compile(&node) {
            Ok(c) => c,
            Err(CompileError::UnsupportedPrimitive(p)) => {
                bytecode_unsupported.push(format!("{name}: {p}"));
                continue;
            }
            Err(e) => panic!("{name}: CompiledSdf::try_compile failed: {e}"),
        };
        let bvh = CompiledSdfBvh::try_compile(&node)
            .unwrap_or_else(|e| panic!("{name}: CompiledSdfBvh::try_compile failed: {e}"));
        let simd_all = eval_compiled_batch_simd(&compiled, &pts);
        #[cfg(feature = "jit")]
        let jit_simd = match alice_sdf::compiled::jit::JitSimdSdf::compile(&compiled) {
            Ok(j) => Some(j),
            Err(e) if e.contains("no codegen arm") || e.contains("not supported") => {
                jit_unsupported.push(format!("{name}: {e}"));
                None
            }
            Err(e) => {
                failures.push(format!("{name}: JitSimdSdf::compile failed: {e}"));
                None
            }
        };
        #[cfg(feature = "jit")]
        let jit_scalar = match alice_sdf::compiled::jit::JitCompiledSdf::compile(&node) {
            Ok(j) => Some(j),
            Err(alice_sdf::compiled::jit::JitError::UnsupportedNode(e)) => {
                jit_unsupported.push(format!("{name} (scalar JIT): {e}"));
                None
            }
            Err(e) => {
                failures.push(format!("{name}: JitCompiledSdf::compile failed: {e}"));
                None
            }
        };
        #[cfg(feature = "jit")]
        let jit_out = jit_simd.as_ref().map(|j| {
            let xs: Vec<f32> = pts.iter().map(|p| p.x).collect();
            let ys: Vec<f32> = pts.iter().map(|p| p.y).collect();
            let zs: Vec<f32> = pts.iter().map(|p| p.z).collect();
            j.eval_batch(&xs, &ys, &zs)
        });
        for (i, &p) in pts.iter().enumerate() {
            let tree = eval(&node, p);
            #[allow(unused_mut)]
            let mut paths = vec![
                ("scalar", eval_compiled(&compiled, p)),
                ("simd", simd_all[i]),
                ("bvh", eval_compiled_bvh(&bvh, p)),
            ];
            #[cfg(feature = "jit")]
            {
                if let Some(out) = &jit_out {
                    paths.push(("jit_simd", out[i]));
                }
                if let Some(j) = &jit_scalar {
                    // The scalar tree JIT (`JitCompiledSdf`, codegen.rs) is an
                    // older independent lowering whose operation order has
                    // not been aligned yet (3.2.0 work item); it is held to a
                    // tolerance here, not to the bit.
                    let v = j.eval(p);
                    if (v - tree).abs() > 1e-5 * tree.abs().max(1.0) {
                        failures.push(format!(
                            "{name} @ {p:?}: tree={} jit(scalar, tol 1e-5)={}",
                            bits(tree),
                            bits(v)
                        ));
                    }
                }
            }
            for (path, v) in paths {
                // NaN payloads are canonical everywhere; compare bits
                if v.to_bits() != tree.to_bits() && !(v.is_nan() && tree.is_nan()) {
                    failures.push(format!(
                        "{name} @ {p:?}: tree={} {path}={}",
                        bits(tree),
                        bits(v)
                    ));
                }
            }
        }
    }
    eprintln!(
        "JIT: {} corpus nodes have no codegen arm (skipped by design)",
        jit_unsupported.len()
    );
    eprintln!(
        "bytecode: {} corpus nodes have no bytecode law (skipped by design): {}",
        bytecode_unsupported.len(),
        bytecode_unsupported.join(", ")
    );
    if !failures.is_empty() {
        // `DET_PARITY_DUMP=<file>` writes every mismatch line for triage
        if let Some(path) = std::env::var_os("DET_PARITY_DUMP") {
            std::fs::write(path, failures.join("\n")).expect("write dump");
        }
        let n = failures.len();
        let mut by_node = std::collections::BTreeMap::<String, usize>::new();
        for f in &failures {
            *by_node
                .entry(f.split(" @ ").next().unwrap_or("").to_string())
                .or_default() += 1;
        }
        let summary: Vec<String> = by_node.iter().map(|(k, v)| format!("{k}: {v}")).collect();
        panic!(
            "{n} bit mismatches across {} nodes:\n{}\n\nfirst 20:\n{}",
            by_node.len(),
            summary.join("\n"),
            failures
                .iter()
                .take(20)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}
