//! Bit parity of every CPU evaluator, the SIMD JITs included, where the
//! `sin_cos` argument of twist / bend sits on or next to a zero of sine or
//! cosine.
//!
//! `tests/test_det_parity.rs` samples an LCG spray in `[-3, 3]^3`, which almost
//! never lands within a few ulp of `k·π/2` and never on `-0`. Those are exactly
//! the inputs where `alice-det-math` 0.5 changed `sin` / `cos` / `sin_cos`
//! (4-piece Cody–Waite reduction with an exact `k·PIO2_3`, and `±0 → ±0`), so a
//! code generator that still emits the 3-piece reduction agrees with the
//! scalar law everywhere the spray looks and disagrees here.
//!
//! The shapes put the angle directly on a coordinate (strength / curvature
//! 1.0, so the argument is the coordinate itself, exactly) and read the
//! rotated coordinate back through a plane, so a 1-ulp difference in `sin` or
//! `cos` reaches the distance unabsorbed. An offset box is included as a
//! shape where the difference may be absorbed. Points with `x = -0` keep the
//! sign of a zero `sin` visible in the output.
//!
//! Every path is compared with the tree evaluator bit for bit (NaN payloads
//! are canonical; any NaN equals any NaN). The test fails if it compared
//! nothing.
//!
//! Note: `modifier_twist_x` / `modifier_twist_z` / `modifier_bend_x` are free
//! functions with no `SdfNode` variant, so no evaluator path reaches them and
//! they are not covered here.
//!
//! Domain: this is finite, in-range angles. A non-finite or astronomically
//! large angle (strength/curvature times a huge coordinate) is out of domain
//! here: the JIT's `fmax`/`fmin` propagate `NaN` where `f32::max`/`min` do not,
//! so twist/bend over a box can differ there (JIT `NaN` vs a value). That is a
//! pre-existing JIT edge tracked separately, not a det-math reduction effect.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "jit")]

use alice_sdf::compiled::jit::{JitSimdSdf, JitSimdSdfDynamic};
use alice_sdf::compiled::{
    eval_compiled, eval_compiled_batch_simd, eval_compiled_bvh, CompiledSdf, CompiledSdfBvh,
};
use alice_sdf::prelude::*;

/// The next representable `f32` towards `+inf` (`-0` and `+0` both step to the
/// smallest positive subnormal).
fn next_up(v: f32) -> f32 {
    if v == 0.0 {
        return f32::from_bits(1);
    }
    let b = v.to_bits();
    f32::from_bits(if v > 0.0 { b + 1 } else { b - 1 })
}

/// The next representable `f32` towards `-inf`.
fn next_down(v: f32) -> f32 {
    -next_up(-v)
}

/// `a` moved by `n` ulp (`n < 0` moves down).
fn ulp_step(a: f32, n: i32) -> f32 {
    let mut v = a;
    for _ in 0..n.unsigned_abs() {
        v = if n > 0 { next_up(v) } else { next_down(v) };
    }
    v
}

/// Zeros and tiny magnitudes, plus `k·π/2 ± ulps` for `0 < |k| ≤ k_max`.
fn angles(k_max: i32, ulps: i32) -> Vec<f32> {
    let mut a = vec![
        0.0f32, -0.0, 1e-45, -1e-45, 1e-40, -1e-40, 1e-30, -1e-30, 1e-7, -1e-7,
    ];
    for k in -k_max..=k_max {
        if k == 0 {
            continue;
        }
        let base = (f64::from(k) * std::f64::consts::FRAC_PI_2) as f32;
        for d in -ulps..=ulps {
            a.push(ulp_step(base, d));
        }
    }
    a
}

#[derive(Clone, Copy)]
enum Modifier {
    /// `sin_cos(p.y * strength)`
    Twist,
    /// `sin_cos(curvature * p.x)`
    Bend,
}

fn shapes() -> Vec<(&'static str, Modifier, SdfNode)> {
    let base = || {
        vec![
            ("planeX", SdfNode::plane(Vec3::X, 0.0)),
            ("planeY", SdfNode::plane(Vec3::Y, 0.0)),
            ("planeZ", SdfNode::plane(Vec3::Z, 0.0)),
            (
                "box",
                SdfNode::box3d(0.5, 0.5, 0.5).translate(0.3, 0.2, 0.2),
            ),
        ]
    };
    let mut out = Vec::new();
    for (name, node) in base() {
        let n: &'static str = Box::leak(format!("twist/{name}").into_boxed_str());
        out.push((n, Modifier::Twist, node.twist(1.0)));
    }
    for (name, node) in base() {
        let n: &'static str = Box::leak(format!("bend/{name}").into_boxed_str());
        out.push((n, Modifier::Bend, node.bend(1.0)));
    }
    out
}

/// Points whose `sin_cos` argument is exactly `a`.
const fn points(m: Modifier, a: f32) -> [Vec3; 4] {
    match m {
        Modifier::Twist => [
            Vec3::new(1.0, a, 0.0),
            Vec3::new(-0.0, a, -1.0),
            Vec3::new(0.0, a, 1.0),
            Vec3::new(0.7, a, 0.3),
        ],
        Modifier::Bend => [
            Vec3::new(a, 1.0, 0.0),
            Vec3::new(a, -1.0, 0.0),
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(a, -0.0, 0.0),
        ],
    }
}

fn bits(v: f32) -> String {
    format!("{v:.9e} ({:#010x})", v.to_bits())
}

fn check(k_max: i32, ulps: i32) {
    let angles = angles(k_max, ulps);
    let mut comparisons = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for (name, m, node) in shapes() {
        let pts: Vec<Vec3> = angles.iter().flat_map(|&a| points(m, a)).collect();
        let compiled = CompiledSdf::try_compile(&node)
            .unwrap_or_else(|e| panic!("{name}: CompiledSdf::try_compile failed: {e}"));
        let bvh = CompiledSdfBvh::try_compile(&node)
            .unwrap_or_else(|e| panic!("{name}: CompiledSdfBvh::try_compile failed: {e}"));
        let simd = eval_compiled_batch_simd(&compiled, &pts);

        let xs: Vec<f32> = pts.iter().map(|p| p.x).collect();
        let ys: Vec<f32> = pts.iter().map(|p| p.y).collect();
        let zs: Vec<f32> = pts.iter().map(|p| p.z).collect();
        // twist and bend have codegen arms: a compile error is a failure here
        let jit = JitSimdSdf::compile(&compiled)
            .unwrap_or_else(|e| panic!("{name}: JitSimdSdf::compile failed: {e}"))
            .eval_batch(&xs, &ys, &zs);
        let jit_dyn = JitSimdSdfDynamic::compile(&compiled)
            .unwrap_or_else(|e| panic!("{name}: JitSimdSdfDynamic::compile failed: {e}"))
            .eval_batch(&xs, &ys, &zs);
        assert_eq!(jit.len(), pts.len(), "{name}: jit_simd output length");
        assert_eq!(jit_dyn.len(), pts.len(), "{name}: jit_simd_dyn output length");

        for (i, &p) in pts.iter().enumerate() {
            let tree = eval(&node, p);
            let paths = [
                ("scalar", eval_compiled(&compiled, p)),
                ("simd", simd[i]),
                ("bvh", eval_compiled_bvh(&bvh, p)),
                ("jit_simd", jit[i]),
                ("jit_simd_dyn", jit_dyn[i]),
            ];
            for (path, v) in paths {
                comparisons += 1;
                if v.to_bits() != tree.to_bits() && !(v.is_nan() && tree.is_nan()) {
                    failures.push(format!(
                        "{name} {path} @ ({}, {}, {}): tree={} {path}={}",
                        bits(p.x),
                        bits(p.y),
                        bits(p.z),
                        bits(tree),
                        bits(v)
                    ));
                }
            }
        }
    }

    assert!(comparisons > 0, "compared nothing");
    eprintln!(
        "k_max={k_max} ulps=±{ulps}: {comparisons} comparisons, {} mismatches",
        failures.len()
    );
    if !failures.is_empty() {
        let mut by_path = std::collections::BTreeMap::<String, usize>::new();
        let mut signed_zero_only = 0usize;
        for f in &failures {
            let mut it = f.split(' ');
            let key = format!("{} {}", it.next().unwrap_or(""), it.next().unwrap_or(""));
            *by_path.entry(key).or_default() += 1;
        }
        for f in &failures {
            // "tree=-0.000000000e0 (0x80000000) path=0.000000000e0 (0x00000000)"
            if f.contains("tree=0.000000000e0") || f.contains("tree=-0.000000000e0") {
                signed_zero_only += 1;
            }
        }
        let summary: Vec<String> = by_path.iter().map(|(k, v)| format!("  {k}: {v}")).collect();
        panic!(
            "{} bit mismatches in {comparisons} comparisons ({signed_zero_only} with a zero tree value):\n{}\n\nfirst 20:\n{}",
            failures.len(),
            summary.join("\n"),
            failures.iter().take(20).cloned().collect::<Vec<_>>().join("\n")
        );
    }
}

/// `|k| ≤ 8`, ±8 ulp around each `k·π/2`, plus the zeros and tiny magnitudes.
#[test]
fn twist_bend_near_trig_zeros_are_bit_identical() {
    check(8, 8);
}

/// `|k| ≤ 63` (`|angle| ≤ 99`, the documented `|x| ≤ 100` domain of
/// `alice-det-math` `sin` / `cos`), ±8 ulp around each `k·π/2`.
#[test]
fn twist_bend_near_trig_zeros_wide_are_bit_identical() {
    check(63, 8);
}
