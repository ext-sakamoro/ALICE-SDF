//! The SIMD JITs emit `alice_det_math::sin_cos` to the bit, on the angles
//! where its range reduction matters.
//!
//! oracle: `alice_det_math::sin_cos` itself (the scalar law), read through a
//! twist / bend of an axis plane so the distance *is* `cos` or `-sin` of the
//! angle; the JIT output is compared with the tree evaluator and with that
//! value. `tests/test_det_parity.rs` samples `|angle| ≤ 4.5`, where the
//! three-piece reduction of `alice-det-math` 0.4 and the four-piece one of
//! 0.5 round the same, so it did not see the change; the angles here are the
//! ones that tell them apart: next to the zeros of `sin` / `cos` out to
//! `k = 5000`, at the `x·2/π = n + ½` quadrant ties, signed zero, huge and
//! non-finite arguments.
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "jit")]

use alice_sdf::compiled::jit::{
    JitCompiledSdf, JitCompiledSdfDynamic, JitSimdSdf, JitSimdSdfDynamic,
};
use alice_sdf::compiled::CompiledSdf;
use alice_sdf::prelude::*;

/// Angles (radians) with their f32 neighbours: the multiples of π/2 out to
/// k = 5000, the quadrant ties, ±0, tiny / subnormal, huge and non-finite, and
/// every f32 in `[2^13, 2^13 + 1)` with its negative (where the far-lane
/// reduction starts: a far boundary moved even by one unit leaves angles there
/// on the single-precision reduction, which differs from the law)
fn angles() -> Vec<f32> {
    let mut out = vec![
        0.0,
        -0.0,
        1e-30,
        -1e-30,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        8192.0,
        -8192.0,
        1.0e4,
        1_048_576.0,
        16_777_216.0,
        3.0e38,
        -3.0e38,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let around = |out: &mut Vec<f32>, x: f64| {
        let c = x as f32;
        let b = c.to_bits();
        for d in 0..=3u32 {
            out.push(f32::from_bits(b.wrapping_add(d)));
            out.push(f32::from_bits(b.wrapping_sub(d)));
        }
    };
    for bits in 8192.0f32.to_bits()..8193.0f32.to_bits() {
        out.push(f32::from_bits(bits));
        out.push(-f32::from_bits(bits));
    }
    for k in (1..=5000i32).filter(|k| *k <= 64 || k % 7 == 0) {
        let x = f64::from(k) * std::f64::consts::FRAC_PI_2;
        around(&mut out, x);
        around(&mut out, -x);
        // x·2/π = k + ½: the quadrant tie
        let t = (f64::from(k) + 0.5) * std::f64::consts::FRAC_PI_2;
        around(&mut out, t);
        around(&mut out, -t);
    }
    out
}

const fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// JIT (baked and parameter-buffer) vs the tree evaluator at `pts`; returns
/// the tree values.
fn compare(name: &str, node: &SdfNode, pts: &[Vec3], failures: &mut Vec<String>) -> Vec<f32> {
    let compiled = CompiledSdf::compile(node);
    let baked = JitSimdSdf::compile(&compiled).unwrap_or_else(|e| panic!("{name}: {e}"));
    let dynamic = JitSimdSdfDynamic::compile(&compiled).unwrap_or_else(|e| panic!("{name}: {e}"));
    let xs: Vec<f32> = pts.iter().map(|p| p.x).collect();
    let ys: Vec<f32> = pts.iter().map(|p| p.y).collect();
    let zs: Vec<f32> = pts.iter().map(|p| p.z).collect();
    let jb = baked.eval_batch(&xs, &ys, &zs);
    let jd = dynamic.eval_batch(&xs, &ys, &zs);
    let mut tree = Vec::with_capacity(pts.len());
    for (i, p) in pts.iter().enumerate() {
        let t = eval(node, *p);
        for (path, v) in [("jit_simd", jb[i]), ("jit_simd_dynamic", jd[i])] {
            if !same(v, t) {
                failures.push(format!(
                    "{name} @ {p:?}: tree={t:e} ({:#010x}) {path}={v:e} ({:#010x})",
                    t.to_bits(),
                    v.to_bits()
                ));
            }
        }
        tree.push(t);
    }
    tree
}

#[test]
fn the_simd_jits_emit_det_math_sin_cos_bit_for_bit() {
    let a = angles();
    let mut failures = Vec::new();
    let mut checked = 0usize;

    // twist(1): angle = y; at (1, y, 0) the plane distance is cos(y), at
    // (0, y, 1) it is −sin(y)
    let twist = SdfNode::plane(Vec3::X, 0.0).twist(1.0);
    let cos_pts: Vec<Vec3> = a.iter().map(|&y| Vec3::new(1.0, y, 0.0)).collect();
    let sin_pts: Vec<Vec3> = a.iter().map(|&y| Vec3::new(0.0, y, 1.0)).collect();
    let tc = compare("twist cos", &twist, &cos_pts, &mut failures);
    let ts = compare("twist sin", &twist, &sin_pts, &mut failures);
    for (i, &y) in a.iter().enumerate() {
        let (s, c) = alice_det_math::sin_cos(y);
        // `+ 0·…` of the plane's dot product turns a −0 into +0; otherwise
        // the distance is exactly the law's value. A huge argument gives one
        // infinite and one NaN component (meaningless by the law's contract),
        // and `0·NaN` in the rotation spreads the NaN to both distances, so
        // those are compared JIT-vs-tree only (above)
        if !(s.is_finite() && c.is_finite()) {
            continue;
        }
        let (want_c, want_s) = (c + 0.0, -s + 0.0);
        if !same(tc[i] + 0.0, want_c) || !same(ts[i] + 0.0, want_s) {
            failures.push(format!(
                "twist @ y={y:e}: tree ({:e}, {:e}) is not det_math (cos {c:e}, −sin {:e})",
                tc[i], ts[i], -s
            ));
        }
        checked += 1;
    }

    // bend(1): angle = x, at points (x, 0, 0) and (x, 1, 0) of a Y plane
    let bend = SdfNode::plane(Vec3::Y, 0.0).bend(1.0);
    let bend_pts: Vec<Vec3> = a
        .iter()
        .flat_map(|&x| [Vec3::new(x, 0.0, 0.0), Vec3::new(x, 1.0, 0.0)])
        .collect();
    compare("bend", &bend, &bend_pts, &mut failures);
    checked += bend_pts.len();

    assert!(checked > 10_000, "compared only {checked} points");
    assert!(
        failures.is_empty(),
        "{} mismatches (first 40):\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
}

/// The scalar JITs (`JitCompiledSdf` baked and with a parameter buffer) emit
/// the law too: twist / bend through the SIMD JIT's emitter of
/// `alice_det_math::sin_cos` (its far lanes call the scalar law), to the bit,
/// on the same near and far angles. Before, the scalar JIT emitted a Taylor
/// approximation that was never the law's bits.
#[test]
fn the_scalar_jits_emit_det_math_sin_cos_bit_for_bit() {
    let a = angles();
    let mut failures = Vec::new();
    let mut checked = 0usize;
    let mut far = 0usize;
    let twist = SdfNode::plane(Vec3::X, 0.0).twist(1.0);
    let bend = SdfNode::plane(Vec3::Y, 0.0).bend(1.0);
    for (name, node, pts) in [
        (
            "twist",
            &twist,
            a.iter()
                .flat_map(|&y| [Vec3::new(1.0, y, 0.0), Vec3::new(0.0, y, 1.0)])
                .collect::<Vec<_>>(),
        ),
        (
            "bend",
            &bend,
            a.iter()
                .flat_map(|&x| [Vec3::new(x, 0.0, 0.0), Vec3::new(x, 1.0, 0.0)])
                .collect::<Vec<_>>(),
        ),
    ] {
        let baked = JitCompiledSdf::compile(node).unwrap_or_else(|e| panic!("{name}: {e}"));
        let dynamic =
            JitCompiledSdfDynamic::compile(node).unwrap_or_else(|e| panic!("{name}: {e}"));
        for p in &pts {
            let t = eval(node, *p);
            for (path, v) in [("jit", baked.eval(*p)), ("jit_dynamic", dynamic.eval(*p))] {
                if !same(v, t) {
                    failures.push(format!(
                        "{name} {path} @ {p:?}: tree={t:e} ({:#010x}) jit={v:e} ({:#010x})",
                        t.to_bits(),
                        v.to_bits()
                    ));
                }
            }
            let angle = if name == "twist" { p.y } else { p.x };
            if angle.is_finite() && angle.abs() >= 8192.0 {
                far += 1;
            }
            checked += 1;
        }
    }
    assert!(checked > 10_000, "compared only {checked} points");
    assert!(far > 0, "no angle at or above 2^13 was compared");
    assert!(
        failures.is_empty(),
        "{} mismatches (first 40):\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
}

/// The sweep reaches the part of the reduction 0.5 changed: on these angles
/// `alice_det_math::sin_cos` is not the value of a three-piece reduction with
/// `roundf` (the 0.4 law), so a JIT still emitting that would differ here.
#[test]
fn the_sweep_contains_angles_where_the_reductions_differ() {
    use alice_det_math::consts::{FRAC_2_PI, PIO2_1, PIO2_2, PIO2_3};
    let old_r = |x: f32| {
        let kf = (x * FRAC_2_PI).round();
        ((x - kf * PIO2_1) - kf * PIO2_2) - kf * PIO2_3
    };
    let differ = angles()
        .into_iter()
        .filter(|x| x.is_finite() && x.abs() < 8192.0)
        .filter(|&x| {
            let (s, c) = alice_det_math::sin_cos(x);
            let r = old_r(x);
            // the old law's (sin, cos) up to the quadrant: compare magnitudes
            let (os, oc) = alice_det_math::sin_cos(r);
            let (m1, m2) = (s.abs().to_bits(), c.abs().to_bits());
            let alt = [os.abs().to_bits(), oc.abs().to_bits()];
            !((m1 == alt[0] && m2 == alt[1]) || (m1 == alt[1] && m2 == alt[0]))
        })
        .count();
    assert!(
        differ > 100,
        "only {differ} angles tell the 0.4 and 0.5 reductions apart"
    );
}
