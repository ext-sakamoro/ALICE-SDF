//! `SdfNode::SmoothBlend` (circular / quartic / sigmoid kernels) against
//! closed forms derived from geometry, independent of the implementation.
//!
//! Each kernel is a smooth minimum `smin(a, b)` of two distances, written as
//! `(a + b)/2 − H(|a − b|)`; `H(x) = x/2` outside the blend recovers `min`.
//!
//! * **Quartic** (C2, finite support `|a − b| < k`): `H(x) = x/2 + k·g(x/k)`
//!   with `g(s) = (1 − s)³(3 + s)/16`, the quartic with `g(1) = g'(1) = g''(1) = 0`
//!   (joins `min` with two continuous derivatives) and `g'(0) = −1/2`,
//!   `g'''(0) = 0` (cancels the kink of `min` at `a = b` to second order).
//!   Deficit at `a = b`: `3k/16`.
//! * **Circular** (C1, finite support): every level set is a circular arc
//!   tangent to the two lines of `min`'s level set, `smin = (a+b)/2 − k +
//!   √(k²/2 − (a − b)²/4)` for `|a − b| < k`. Deficit: `k(1 − 1/√2)`.
//! * **Sigmoid** (C∞, unbounded support): `smin = m − x / (eˣᐟᵏ − 1)` with
//!   `m = min(a, b)`, `x = |a − b|`; the limit at `x = 0` is `m − k`. Deficit: `k`.
//!
//! Modes: intersection is `−smin(−a, −b)`, subtraction of `b` from `a` is
//! `−smin(−a, b)` (the conventions of the existing smooth operators).
//!
//! Oracles: f64 closed forms on a grid; the deficit at `a = b`; exact `min`
//! outside the blend (finite-support kernels, bit-exact); points constructed
//! on the circular kernel's arc give a constant value; a sampled Lipschitz
//! bound on a blended scene (0 pairs compared = fail); finite output for
//! finite input; and the three kernels differ (a swapped kernel table fails).
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm serves as the independent reference"
)]
#![allow(clippy::float_cmp)]

use alice_sdf::operations::{smooth_blend, BlendMode, SmoothKernel};
use alice_sdf::prelude::*;

const KERNELS: [SmoothKernel; 3] = [
    SmoothKernel::Circular,
    SmoothKernel::Quartic,
    SmoothKernel::Sigmoid,
];
const MODES: [BlendMode; 3] = [
    BlendMode::Union,
    BlendMode::Intersection,
    BlendMode::Subtraction,
];

/// The closed-form smooth minimum in f64.
fn smin_ref(kernel: SmoothKernel, a: f64, b: f64, k: f64) -> f64 {
    let m = a.min(b);
    let x = (a - b).abs();
    match kernel {
        SmoothKernel::Quartic => {
            if x >= k {
                m
            } else {
                let s = x / k;
                (a + b) / 2.0 - (x / 2.0 + k * (1.0 - s).powi(3) * (3.0 + s) / 16.0)
            }
        }
        SmoothKernel::Circular => {
            if x >= k {
                m
            } else {
                (a + b) / 2.0 - k + (k * k / 2.0 - x * x / 4.0).sqrt()
            }
        }
        SmoothKernel::Sigmoid => {
            if x == 0.0 {
                m - k
            } else {
                m - x / (x / k).exp_m1()
            }
        }
    }
}

fn blend_ref(kernel: SmoothKernel, mode: BlendMode, a: f64, b: f64, k: f64) -> f64 {
    match mode {
        BlendMode::Union => smin_ref(kernel, a, b, k),
        BlendMode::Intersection => -smin_ref(kernel, -a, -b, k),
        BlendMode::Subtraction => -smin_ref(kernel, -a, b, k),
    }
}

fn deficit(kernel: SmoothKernel) -> f64 {
    match kernel {
        SmoothKernel::Quartic => 3.0 / 16.0,
        SmoothKernel::Circular => 1.0 - std::f64::consts::FRAC_1_SQRT_2,
        SmoothKernel::Sigmoid => 1.0,
    }
}

fn grid() -> Vec<f32> {
    (-40..=40).map(|i| i as f32 * 0.0375).collect()
}

#[test]
fn every_kernel_and_mode_matches_its_closed_form() {
    let mut compared = 0usize;
    for kernel in KERNELS {
        for mode in MODES {
            for &k in &[0.05f32, 0.3, 1.0] {
                let mut worst = 0.0f64;
                for &a in &grid() {
                    for &b in &grid() {
                        let got = f64::from(smooth_blend(a, b, k, kernel, mode));
                        let want =
                            blend_ref(kernel, mode, f64::from(a), f64::from(b), f64::from(k));
                        let err = (got - want).abs() / (1.0 + want.abs());
                        worst = worst.max(err);
                        compared += 1;
                    }
                }
                assert!(
                    worst <= 2e-6,
                    "{kernel:?} {mode:?} k={k}: worst relative error {worst:e}"
                );
            }
        }
    }
    assert_eq!(compared, 3 * 3 * 3 * 81 * 81);
}

#[test]
fn deficit_at_equal_distances() {
    for kernel in KERNELS {
        for &k in &[0.1f32, 0.5, 2.0] {
            for &a in &[-1.0f32, 0.0, 0.7] {
                let got = f64::from(smooth_blend(a, a, k, kernel, BlendMode::Union));
                let want = f64::from(a) - deficit(kernel) * f64::from(k);
                assert!(
                    (got - want).abs() <= 2e-6 * (1.0 + want.abs()),
                    "{kernel:?} k={k} a={a}: got {got}, want {want}"
                );
            }
        }
    }
}

/// Outside the blend the finite-support kernels are `min` to the bit.
#[test]
fn finite_support_kernels_are_min_outside_the_blend() {
    let mut compared = 0;
    for kernel in [SmoothKernel::Circular, SmoothKernel::Quartic] {
        for &k in &[0.05f32, 0.3, 1.0] {
            for &a in &grid() {
                for &b in &grid() {
                    if (a - b).abs() >= k {
                        assert_eq!(
                            smooth_blend(a, b, k, kernel, BlendMode::Union),
                            a.min(b),
                            "{kernel:?} k={k} a={a} b={b}"
                        );
                        compared += 1;
                    }
                }
            }
        }
    }
    assert!(compared > 10_000, "compared {compared}");
}

/// The sigmoid kernel has unbounded support, but its deficit decays like
/// `x·e^(−x/k)`: at `|a − b| = 20k` it is below 1e-6 of the distance.
#[test]
fn sigmoid_tail_reaches_min() {
    for &k in &[0.05f32, 0.3] {
        let a = 0.25f32;
        let b = a + 20.0 * k;
        let got = smooth_blend(a, b, k, SmoothKernel::Sigmoid, BlendMode::Union);
        assert!(got <= a, "sigmoid must not exceed min");
        assert!(a - got < 1e-6, "k={k}: deficit {}", a - got);
    }
}

/// Construct points on one level set of the circular kernel from geometry:
/// in `u = (a + b)/2`, `v = (a − b)/2`, the level set `smin = c` is the arc
/// of radius `ρ = k/√2` centred at `(c + √2ρ, 0)`, for `|v| ≤ ρ/√2`.
#[test]
fn circular_kernel_level_sets_are_arcs() {
    let k = 0.8f64;
    let rho = k / std::f64::consts::SQRT_2;
    for &c in &[-0.4f64, 0.0, 0.3] {
        for i in 0..=20 {
            // |v| from 0 to ρ/√2, both signs
            let v = (i as f64 / 20.0 - 0.5) * 2.0 * rho / std::f64::consts::SQRT_2 * 0.999;
            let u = c + std::f64::consts::SQRT_2 * rho - (rho * rho - v * v).sqrt();
            let (a, b) = (u + v, u - v);
            let got = smooth_blend(
                a as f32,
                b as f32,
                k as f32,
                SmoothKernel::Circular,
                BlendMode::Union,
            );
            assert!((f64::from(got) - c).abs() < 2e-6, "c={c} v={v}: got {got}");
        }
    }
}

/// Every kernel is a convex combination of the two gradients (weights in
/// [0, 1] summing to 1), so the blend of two 1-Lipschitz fields is
/// 1-Lipschitz. Sampled on a scene of two spheres and a box.
#[test]
fn blended_scene_is_one_lipschitz() {
    let a = SdfNode::sphere(0.6).translate(-0.35, 0.0, 0.0);
    let b = SdfNode::box3d(0.4, 0.5, 0.3).translate(0.4, 0.1, 0.0);
    for kernel in KERNELS {
        for mode in MODES {
            let node = SdfNode::smooth_blend(a.clone(), b.clone(), 0.35, kernel, mode);
            let mut pairs = 0;
            let mut worst = 0.0f32;
            let mut s = 0x9e37_79b9u32;
            let mut rnd = || {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 3.0 - 1.5
            };
            for _ in 0..4000 {
                let p = Vec3::new(rnd(), rnd(), rnd());
                let d = Vec3::new(rnd(), rnd(), rnd()) * 0.02;
                let q = p + d;
                let len = d.length();
                if len < 1e-4 {
                    continue;
                }
                let ratio = (eval(&node, p) - eval(&node, q)).abs() / len;
                worst = worst.max(ratio);
                pairs += 1;
            }
            assert!(pairs > 3000, "{kernel:?} {mode:?}: only {pairs} pairs");
            assert!(
                worst <= 1.0 + 2e-3,
                "{kernel:?} {mode:?}: difference quotient {worst}"
            );
        }
    }
}

#[test]
fn finite_input_gives_finite_output() {
    let vals = [
        -1e30f32,
        -1e6,
        -1.0,
        -1e-30,
        0.0,
        1e-30,
        1.0,
        1e6,
        1e30,
        f32::MAX,
        -f32::MAX,
    ];
    for kernel in KERNELS {
        for mode in MODES {
            for &k in &[1e-6f32, 0.5, 1e6] {
                for &a in &vals {
                    for &b in &vals {
                        let r = smooth_blend(a, b, k, kernel, mode);
                        assert!(r.is_finite(), "{kernel:?} {mode:?} k={k} a={a} b={b}: {r}");
                    }
                }
            }
        }
    }
}

/// A non-positive width is `min` (no blend), for every kernel.
#[test]
fn non_positive_width_is_min() {
    for kernel in KERNELS {
        for &k in &[0.0f32, -0.5] {
            assert_eq!(smooth_blend(0.2, 0.3, k, kernel, BlendMode::Union), 0.2);
            assert_eq!(smooth_blend(0.3, 0.3, k, kernel, BlendMode::Union), 0.3);
        }
    }
}

/// The three kernels give three different values inside the blend, so a
/// table that maps one kernel to another's law fails here.
#[test]
fn kernels_are_distinct() {
    let v: Vec<f32> = KERNELS
        .iter()
        .map(|&kn| smooth_blend(0.1, 0.15, 0.4, kn, BlendMode::Union))
        .collect();
    assert!(v[0] != v[1] && v[1] != v[2] && v[0] != v[2], "{v:?}");
}

/// The tree evaluator, the compiled bytecode and the operator agree to the bit.
#[test]
fn tree_and_compiled_agree() {
    let a = SdfNode::sphere(0.6).translate(-0.35, 0.0, 0.0);
    let b = SdfNode::box3d(0.4, 0.5, 0.3).translate(0.4, 0.1, 0.0);
    for kernel in KERNELS {
        for mode in MODES {
            let node = SdfNode::smooth_blend(a.clone(), b.clone(), 0.35, kernel, mode);
            let compiled = CompiledSdf::compile(&node);
            for i in 0..200 {
                let t = i as f32 / 200.0;
                let p = Vec3::new(t * 2.0 - 1.0, (t * 7.0).sin() * 0.8, (t * 3.0).cos() * 0.5);
                let tree = eval(&node, p);
                let comp = eval_compiled(&compiled, p);
                assert_eq!(
                    tree.to_bits(),
                    comp.to_bits(),
                    "{kernel:?} {mode:?} at {p:?}"
                );
            }
        }
    }
}

/// The analytic gradient (`eval_gradient`, from `smooth_blend_weights`)
/// against central differences of the tree evaluator, away from the
/// operands' own creases.
#[test]
fn analytic_gradient_matches_central_differences() {
    use alice_sdf::eval::eval_gradient;
    let a = SdfNode::sphere(0.6).translate(-0.35, 0.0, 0.0);
    let b = SdfNode::sphere(0.5).translate(0.4, 0.1, 0.0);
    let h = 1e-3f32;
    for kernel in KERNELS {
        for mode in MODES {
            let node = SdfNode::smooth_blend(a.clone(), b.clone(), 0.35, kernel, mode);
            let mut compared = 0;
            let mut worst = 0.0f32;
            for i in 0..400 {
                let t = i as f32 / 400.0;
                let p = Vec3::new(t * 2.4 - 1.2, (t * 13.0).sin() * 0.9, (t * 5.0).cos() * 0.7);
                // skip the spheres' centres, where their own gradient is undefined
                if (p - Vec3::new(-0.35, 0.0, 0.0)).length() < 0.05
                    || (p - Vec3::new(0.4, 0.1, 0.0)).length() < 0.05
                {
                    continue;
                }
                let fd = Vec3::new(
                    eval(&node, p + Vec3::X * h) - eval(&node, p - Vec3::X * h),
                    eval(&node, p + Vec3::Y * h) - eval(&node, p - Vec3::Y * h),
                    eval(&node, p + Vec3::Z * h) - eval(&node, p - Vec3::Z * h),
                ) / (2.0 * h);
                let g = eval_gradient(&node, p);
                worst = worst.max((g - fd).length());
                compared += 1;
            }
            assert!(compared > 350, "{kernel:?} {mode:?}: {compared}");
            assert!(
                worst < 5e-3,
                "{kernel:?} {mode:?}: |grad − fd| up to {worst}"
            );
        }
    }
}

/// `.asdf.json` round trip keeps the kernel and mode (bit-identical
/// distances), and a reader refuses a kernel or mode name it does not know.
#[test]
fn asdf_json_round_trip_and_unknown_names() {
    use alice_sdf::io::{from_json_string, to_json_string};
    let a = SdfNode::sphere(0.6).translate(-0.35, 0.0, 0.0);
    let b = SdfNode::box3d(0.4, 0.5, 0.3).translate(0.4, 0.1, 0.0);
    for kernel in KERNELS {
        for mode in MODES {
            let node = SdfNode::smooth_blend(a.clone(), b.clone(), 0.35, kernel, mode);
            let json = to_json_string(&SdfTree::new(node.clone())).unwrap();
            let back = from_json_string(&json).unwrap().root;
            for i in 0..50 {
                let p = Vec3::new(i as f32 * 0.04 - 1.0, 0.1, -0.2);
                assert_eq!(eval(&node, p).to_bits(), eval(&back, p).to_bits());
            }
            let kname = format!("\"{kernel:?}\"");
            let mname = format!("\"{mode:?}\"");
            assert!(json.contains(&kname) && json.contains(&mname), "{json}");
            let bad_kernel = json.replace(&kname, "\"Hyperbolic\"");
            assert!(
                from_json_string(&bad_kernel).is_err(),
                "unknown kernel accepted"
            );
            let bad_mode = json.replace(&mname, "\"Xor\"");
            assert!(
                from_json_string(&bad_mode).is_err(),
                "unknown mode accepted"
            );
        }
    }
}
