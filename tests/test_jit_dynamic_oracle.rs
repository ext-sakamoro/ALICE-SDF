//! Oracles for the parameter-buffer JITs (`JitCompiledSdfDynamic`,
//! `JitSimdSdfDynamic`) and the batch entry points of the scalar JIT.
//!
//! - The parameter order written by the dynamic code generators equals the
//!   order of the independent extractors (`extract_jit_params`,
//!   `extract_simd_params`), node kind by node kind over the shared corpus.
//! - A dynamic JIT evaluates like the constant-baking JIT compiled from the
//!   same tree (same Cranelift lowering, parameters loaded instead of baked),
//!   and like the bytecode interpreter within the JIT parity tolerance.
//! - `update_params` with a re-parameterised tree of the same structure gives
//!   what compiling that tree from scratch gives; a tree with a different
//!   parameter count is rejected instead of being read out of bounds.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "jit")]

mod common;

use alice_sdf::compiled::jit::{
    extract_jit_params, extract_simd_params, JitCompiledSdf, JitCompiledSdfDynamic, JitSimdSdf,
    JitSimdSdfDynamic,
};
use alice_sdf::compiled::{eval_compiled, CompiledSdf};
use alice_sdf::prelude::*;
use alice_sdf::soa::SoAPoints;

/// Families whose varying parameters the dynamic SIMD generator bakes.
const SIMD_BAKED: [&str; 2] = ["box_rotate_scale", "smooth_union"];

/// The JIT parity tolerance of `tests/test_evaluator_opcode_parity.rs`.
const TOL: f32 = 1e-4;

fn points() -> Vec<Vec3> {
    common::test_grid_points(7)
        .into_iter()
        .map(|p| p * 1.6 + Vec3::new(0.01, -0.02, 0.03))
        .collect()
}

/// A scene builder: the same structure for every `s`.
type Family = fn(f32) -> SdfNode;

/// Parametric scenes: the same structure for every `s`, every parameter
/// moving with `s`.
fn families() -> Vec<(&'static str, Family)> {
    vec![
        ("sphere_translate", |s| {
            SdfNode::sphere(0.5 * s).translate(0.2 * s, -0.1, 0.3 * s)
        }),
        ("box_rotate_scale", |s| {
            SdfNode::box3d(0.6 * s, 0.4, 0.3 + 0.1 * s)
                .rotate_euler(0.3 * s, 0.2, -0.4 * s)
                .scale(0.8 + 0.2 * s)
        }),
        ("smooth_union", |s| {
            SdfNode::sphere(0.4 * s).smooth_union(
                SdfNode::torus(0.5, 0.1 * s).translate(0.3 * s, 0.0, 0.0),
                0.1 * s,
            )
        }),
        ("subtract_round", |s| {
            SdfNode::box3d(0.8, 0.8 * s, 0.8)
                .subtract(SdfNode::cylinder(0.2 * s, 1.0))
                .round(0.05 * s)
        }),
        ("intersect_twist", |s| {
            SdfNode::box3d(0.5, 1.0, 0.5)
                .intersection(SdfNode::sphere(0.7 * s))
                .twist(0.5 * s)
        }),
        ("capsule_onion_repeat", |s| {
            SdfNode::capsule(
                Vec3::new(-0.3 * s, 0.0, 0.0),
                Vec3::new(0.3, 0.2 * s, 0.0),
                0.1 * s,
            )
            .onion(0.02 * s)
            .repeat_finite([2, 1, 1], Vec3::new(0.9 * s, 1.0, 1.0))
        }),
    ]
}

#[test]
fn dynamic_scalar_params_follow_the_extractor_over_the_corpus() {
    let pts = points();
    let mut compiled_kinds = 0usize;
    let mut compared = 0usize;
    for (name, node) in common::corpus::corpus() {
        let Ok(dynamic) = JitCompiledSdfDynamic::compile(&node) else {
            continue;
        };
        let baked = JitCompiledSdf::compile(&node).expect("baked JIT accepts what dynamic does");
        compiled_kinds += 1;
        let extracted = extract_jit_params(&node);
        assert_eq!(
            dynamic
                .params()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            extracted.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{name}: code generator and extractor disagree on the parameter order"
        );
        assert_eq!(
            dynamic.lipschitz().to_bits(),
            alice_sdf::interval::eval_lipschitz(&node).to_bits(),
            "{name}"
        );
        let batch = dynamic.eval_batch(&pts);
        let par = dynamic.eval_batch_parallel(&pts);
        let baked_par = baked.eval_batch_parallel(&pts);
        for (k, p) in pts.iter().enumerate() {
            let d = dynamic.eval(*p);
            assert_eq!(batch[k].to_bits(), d.to_bits(), "{name} batch @ {p}");
            assert_eq!(par[k].to_bits(), d.to_bits(), "{name} parallel @ {p}");
            let b = baked.eval(*p);
            assert_eq!(
                baked_par[k].to_bits(),
                b.to_bits(),
                "{name} baked parallel @ {p}"
            );
            let tree = eval(&node, *p);
            if tree.is_finite() && tree.abs() < 1e6 {
                assert!(
                    (d - b).abs() <= TOL * b.abs().max(1.0),
                    "{name} @ {p}: dyn {d} baked {b}"
                );
                assert!(
                    (d - tree).abs() <= TOL * tree.abs().max(1.0),
                    "{name} @ {p}: dyn {d} tree {tree}"
                );
                compared += 1;
            }
        }
    }
    assert!(
        compiled_kinds >= 30,
        "dynamic JIT accepted {compiled_kinds} corpus trees"
    );
    assert!(compared > 5_000, "compared {compared}");
}

#[test]
fn dynamic_scalar_update_params_equals_a_fresh_compile() {
    let pts = points();
    let mut compared = 0usize;
    for (name, family) in families() {
        let mut dynamic = JitCompiledSdfDynamic::compile(&family(1.0))
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        for s in [1.35f32, 0.7, 1.0] {
            let tree = family(s);
            dynamic.update_params(&tree);
            let fresh = JitCompiledSdfDynamic::compile(&tree).expect("compile");
            assert_eq!(dynamic.params(), fresh.params(), "{name} s={s}");
            let baked = JitCompiledSdf::compile(&tree).expect("compile");
            for p in &pts {
                let d = dynamic.eval(*p);
                assert_eq!(d.to_bits(), fresh.eval(*p).to_bits(), "{name} s={s} @ {p}");
                let b = baked.eval(*p);
                assert!(
                    (d - b).abs() <= TOL * b.abs().max(1.0),
                    "{name} s={s} @ {p}: {d} vs {b}"
                );
                compared += 1;
            }
        }
    }
    assert!(compared > 5_000, "compared {compared}");
}

#[test]
#[should_panic(expected = "structure must match")]
fn dynamic_scalar_rejects_a_tree_with_fewer_parameters() {
    let mut dynamic = JitCompiledSdfDynamic::compile(
        &SdfNode::sphere(1.0).smooth_union(SdfNode::sphere(0.5), 0.2),
    )
    .expect("compile");
    dynamic.update_params(&SdfNode::sphere(1.0));
}

#[test]
fn dynamic_simd_params_follow_the_extractor_over_the_corpus() {
    let pts = points();
    let mut compiled_kinds = 0usize;
    let mut compared = 0usize;
    for (name, node) in common::corpus::corpus() {
        let Ok(compiled) = CompiledSdf::try_compile(&node) else {
            continue;
        };
        let Ok(dynamic) = JitSimdSdfDynamic::compile(&compiled) else {
            continue;
        };
        let baked =
            JitSimdSdf::compile(&compiled).expect("baked SIMD JIT accepts what dynamic does");
        compiled_kinds += 1;
        assert_eq!(
            dynamic
                .params()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            extract_simd_params(&compiled)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "{name}: code generator and extractor disagree on the parameter order"
        );
        let x: Vec<f32> = pts.iter().map(|p| p.x).collect();
        let y: Vec<f32> = pts.iter().map(|p| p.y).collect();
        let z: Vec<f32> = pts.iter().map(|p| p.z).collect();
        let d = dynamic.eval_batch(&x, &y, &z);
        let b = baked.eval_batch(&x, &y, &z);
        assert_eq!(d.len(), pts.len(), "{name}");
        let soa = dynamic.eval_soa(&SoAPoints::from_vec3_slice(&pts));
        assert!(soa.len() >= pts.len(), "{name}");
        for (k, p) in pts.iter().enumerate() {
            assert_eq!(soa[k].to_bits(), d[k].to_bits(), "{name} soa @ {p}");
            let interp = eval_compiled(&compiled, *p);
            if interp.is_finite() && interp.abs() < 1e6 {
                assert!(
                    (d[k] - b[k]).abs() <= TOL * b[k].abs().max(1.0),
                    "{name} @ {p}"
                );
                assert!(
                    (d[k] - interp).abs() <= TOL * interp.abs().max(1.0),
                    "{name} @ {p}: dyn {} interp {interp}",
                    d[k]
                );
                compared += 1;
            }
        }
        // eval_8 is one chunk of eval_batch.
        let (cx, cy, cz): ([f32; 8], [f32; 8], [f32; 8]) = (
            std::array::from_fn(|i| x[i]),
            std::array::from_fn(|i| y[i]),
            std::array::from_fn(|i| z[i]),
        );
        // SAFETY: the evaluator is alive and the arrays hold 8 lanes each.
        let lanes = unsafe { dynamic.eval_8(&cx, &cy, &cz) };
        for i in 0..8 {
            assert_eq!(lanes[i].to_bits(), d[i].to_bits(), "{name} lane {i}");
        }
    }
    assert!(
        compiled_kinds >= 20,
        "dynamic SIMD JIT accepted {compiled_kinds} corpus trees"
    );
    assert!(compared > 3_000, "compared {compared}");
}

#[test]
fn dynamic_simd_update_params_equals_a_fresh_compile() {
    let pts = points();
    let x: Vec<f32> = pts.iter().map(|p| p.x).collect();
    let y: Vec<f32> = pts.iter().map(|p| p.y).collect();
    let z: Vec<f32> = pts.iter().map(|p| p.z).collect();
    let mut compared = 0usize;
    let mut families_run = 0;
    for (name, family) in families() {
        // The dynamic SIMD generator bakes the rotation quaternion and the
        // smooth-blend radius as constants (documented on
        // `JitSimdSdfDynamic::update_params`), so a family that moves them
        // cannot follow a re-parameterisation; those two are left out here.
        if SIMD_BAKED.contains(&name) {
            continue;
        }
        let Ok(mut dynamic) = JitSimdSdfDynamic::compile(&CompiledSdf::compile(&family(1.0)))
        else {
            continue;
        };
        families_run += 1;
        for s in [1.35f32, 0.7, 1.0] {
            let compiled = CompiledSdf::compile(&family(s));
            dynamic.update_params(&compiled);
            let fresh = JitSimdSdfDynamic::compile(&compiled).expect("compile");
            assert_eq!(dynamic.params(), fresh.params(), "{name} s={s}");
            let d = dynamic.eval_batch(&x, &y, &z);
            let f = fresh.eval_batch(&x, &y, &z);
            for k in 0..pts.len() {
                assert_eq!(d[k].to_bits(), f[k].to_bits(), "{name} s={s} @ {}", pts[k]);
                let interp = eval_compiled(&compiled, pts[k]);
                assert!(
                    (d[k] - interp).abs() <= TOL * interp.abs().max(1.0),
                    "{name} s={s}"
                );
                compared += 1;
            }
        }
    }
    assert!(
        families_run >= 4,
        "dynamic SIMD JIT accepted {families_run} families"
    );
    assert!(compared > 3_000, "compared {compared}");
}

#[test]
#[should_panic(expected = "structure must match")]
fn dynamic_simd_rejects_bytecode_with_fewer_parameters() {
    let two = CompiledSdf::compile(&SdfNode::sphere(1.0).union(SdfNode::sphere(0.5)));
    let mut dynamic = JitSimdSdfDynamic::compile(&two).expect("compile");
    dynamic.update_params(&CompiledSdf::compile(&SdfNode::sphere(1.0)));
}
