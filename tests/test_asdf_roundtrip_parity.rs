//! `.asdf` round trip must not change what any evaluator answers.
//!
//! `test_det_parity.rs` proves the CPU evaluators agree to the bit on an
//! in-memory tree, and `test_det_golden.rs` pins the tree evaluator's bits.
//! Neither loads a tree back from disk — so a codec that reconstructs a node
//! with a different *shape* (the same distances from the tree evaluator, a
//! different bytecode from the compiler) was invisible. The UE5 plugin hits
//! exactly that path (`alice_sdf_load` → `alice_sdf_compile` →
//! `alice_sdf_eval_compiled`), and `AliceSDF.Unreal.FfiCorpusParity` found
//! 1-ulp drifts on the smooth / scaled nodes in 3.1.0.
//!
//! Author: Moroya Sakamoto

mod common;

use alice_sdf::compiled::{eval_compiled, CompiledSdf};
use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use common::corpus::corpus;

/// Deterministic LCG spray plus the axis ties, as in the other parity tests.
fn points(n: usize) -> Vec<Vec3> {
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32) * 6.0 - 3.0
    };
    let mut pts: Vec<Vec3> = (0..n).map(|_| Vec3::new(next(), next(), next())).collect();
    for i in -6..=6 {
        let v = i as f32 * 0.5;
        pts.push(Vec3::new(v, 0.0, 0.0));
        pts.push(Vec3::new(0.0, v, 0.0));
        pts.push(Vec3::new(0.0, 0.0, v));
        pts.push(Vec3::new(v, v, -v));
        pts.push(Vec3::new(-v, 0.25, 0.0));
    }
    pts
}

#[test]
fn asdf_round_trip_keeps_every_evaluator_bit_identical() {
    let dir = std::env::temp_dir().join(format!("alice_sdf_asdf_parity_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let pts = points(512);
    let mut failures = Vec::new();

    for (name, node) in corpus() {
        let path = dir.join(format!("{name}.asdf"));
        alice_sdf::io::save(&SdfTree::new(node.clone()), &path)
            .unwrap_or_else(|e| panic!("{name}: save: {e}"));
        let loaded = alice_sdf::io::load(&path)
            .unwrap_or_else(|e| panic!("{name}: load: {e}"))
            .root;

        let Ok(before) = CompiledSdf::try_compile(&node) else {
            continue; // not compilable; test_det_parity covers that list
        };
        let after = CompiledSdf::try_compile(&loaded)
            .unwrap_or_else(|e| panic!("{name}: compile after round trip: {e}"));

        for &p in &pts {
            let (t0, t1) = (eval(&node, p), eval(&loaded, p));
            let (c0, c1) = (eval_compiled(&before, p), eval_compiled(&after, p));
            let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
            if !same(t0, t1) {
                failures.push(format!(
                    "{name} @ {p:?}: tree {:#010x} -> {:#010x} after .asdf round trip",
                    t0.to_bits(),
                    t1.to_bits()
                ));
                break;
            }
            if !same(c0, c1) {
                failures.push(format!(
                    "{name} @ {p:?}: compiled {:#010x} -> {:#010x} after .asdf round trip \
                     (tree agrees: {:#010x})",
                    c0.to_bits(),
                    c1.to_bits(),
                    t0.to_bits()
                ));
                break;
            }
            if !same(t0, c0) {
                failures.push(format!(
                    "{name} @ {p:?}: tree {:#010x} != compiled {:#010x} (before the round trip)",
                    t0.to_bits(),
                    c0.to_bits()
                ));
                break;
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "{} node(s) changed across the .asdf round trip:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
