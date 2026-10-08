//! Smooth CSG operators vs their published closed forms.
//!
//! `operations/smooth.rs` had nine unit tests, all of the form "the result is
//! `<= min`" / "symmetric" / "`k = 0` gives `min`". None of them checks the
//! *value*, so a wrong coefficient (`k/4` -> `k/3`), a swapped argument, or a
//! drifted convention (`d/k` vs `k*d`) passes all of them.
//!
//! The references here are independent of the implementation:
//!
//! * **closed forms in `f64`** (iq's polynomial / cubic / root smooth min, the
//!   log-sum-exp in both the *width* (`d/k`) and the *rate* (`k*d`) convention),
//! * **structural laws** every smooth minimum must obey (symmetry, a bounded
//!   deficit below `min`, shift invariance, positive homogeneity, monotonicity,
//!   and "the two blend weights sum to 1"),
//! * **geometry**: two perpendicular half-spaces blended at their inner corner
//!   put the surface at a distance `u` from each plane where `smin(u, u) = 0`,
//!   i.e. `u = k/4` (polynomial), `k/6` (cubic), `k ln 2` (exp), `k/2` (root),
//! * **bit identity** between the scalar, reciprocal (`rk`) and generic
//!   (`Real`) forms, which the module documents as "one law",
//! * **robustness**: finite input must give finite output.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
#![allow(clippy::float_cmp)]

use alice_sdf::operations::{
    sdf_exp_smooth_intersection, sdf_exp_smooth_intersection_r, sdf_exp_smooth_subtraction,
    sdf_exp_smooth_subtraction_r, sdf_exp_smooth_union, sdf_exp_smooth_union_r,
    sdf_smooth_intersection, sdf_smooth_intersection_rk, sdf_smooth_intersection_rk_r,
    sdf_smooth_subtraction, sdf_smooth_subtraction_rk, sdf_smooth_subtraction_rk_r,
    sdf_smooth_union, sdf_smooth_union_rk, sdf_smooth_union_rk_r, smooth_max, smooth_max_rk,
    smooth_max_rk_r, smooth_min, smooth_min_cubic, smooth_min_exp, smooth_min_rk, smooth_min_rk_r,
    smooth_min_root,
};
use alice_sdf::{eval, SdfNode};
use glam::Vec3;
use std::sync::Arc;
use wide::f32x8;

// ───────────────────────── closed forms (f64) ─────────────────────────

fn poly_min(a: f64, b: f64, k: f64) -> f64 {
    let h = (1.0 - (a - b).abs() / k).max(0.0);
    a.min(b) - h * h * k / 4.0
}

fn poly_max(a: f64, b: f64, k: f64) -> f64 {
    let h = (1.0 - (a - b).abs() / k).max(0.0);
    a.max(b) + h * h * k / 4.0
}

fn cubic_min(a: f64, b: f64, k: f64) -> f64 {
    let h = (1.0 - (a - b).abs() / k).max(0.0);
    a.min(b) - h * h * h * k / 6.0
}

fn root_min(a: f64, b: f64, k: f64) -> f64 {
    0.5 * (a + b - ((b - a).powi(2) + k * k).sqrt())
}

/// `-k ln(e^{-a/k} + e^{-b/k})`, the width convention, in the overflow-free form
fn exp_width_min(a: f64, b: f64, k: f64) -> f64 {
    a.min(b) - k * (1.0 + (-(a - b).abs() / k).exp()).ln()
}

fn exp_width_max(a: f64, b: f64, k: f64) -> f64 {
    a.max(b) + k * (1.0 + (-(a - b).abs() / k).exp()).ln()
}

/// `-ln(e^{-ka} + e^{-kb}) / k`, the rate convention, in the overflow-free form
fn exp_rate_min(a: f64, b: f64, k: f64) -> f64 {
    a.min(b) - (1.0 + (-k * (a - b).abs()).exp()).ln() / k
}

fn close(got: f32, want: f64, rel: f64) -> bool {
    (f64::from(got) - want).abs() <= rel * (1.0 + want.abs())
}

/// a / b sample values, deliberately including equal pairs and pairs on both sides of `k`
fn samples() -> Vec<f32> {
    (-6..=6).map(|i| i as f32 * 0.5).collect()
}

// ───────────────────────── A. closed-form values ─────────────────────────

#[test]
fn polynomial_smooth_min_and_max_match_the_closed_form() {
    for &k in &[0.05_f32, 0.3, 1.0, 2.5] {
        for &a in &samples() {
            for &b in &samples() {
                let (fa, fb, fk) = (f64::from(a), f64::from(b), f64::from(k));
                assert!(
                    close(smooth_min(a, b, k), poly_min(fa, fb, fk), 1e-5),
                    "smooth_min({a}, {b}, {k})"
                );
                assert!(
                    close(smooth_max(a, b, k), poly_max(fa, fb, fk), 1e-5),
                    "smooth_max({a}, {b}, {k})"
                );
                assert!(close(sdf_smooth_union(a, b, k), poly_min(fa, fb, fk), 1e-5));
                assert!(close(
                    sdf_smooth_intersection(a, b, k),
                    poly_max(fa, fb, fk),
                    1e-5
                ));
                // subtraction of B from A = intersection with the complement of B (-d2)
                assert!(
                    close(sdf_smooth_subtraction(a, b, k), poly_max(fa, -fb, fk), 1e-5),
                    "sdf_smooth_subtraction({a}, {b}, {k})"
                );
            }
        }
    }
}

#[test]
fn smooth_max_is_the_dual_of_smooth_min() {
    // max(a, b) = -min(-a, -b) holds for the smooth versions too
    for &k in &[0.1_f32, 0.7, 2.0] {
        for &a in &samples() {
            for &b in &samples() {
                let dual = -smooth_min(-a, -b, k);
                assert!(
                    (smooth_max(a, b, k) - dual).abs() < 1e-6,
                    "dual of ({a}, {b}, {k})"
                );
            }
        }
    }
}

#[test]
fn exp_smooth_ops_match_the_closed_form_in_the_width_convention() {
    for &k in &[0.05_f32, 0.3, 1.0, 2.5] {
        for &a in &samples() {
            for &b in &samples() {
                let (fa, fb, fk) = (f64::from(a), f64::from(b), f64::from(k));
                assert!(
                    close(
                        sdf_exp_smooth_union(a, b, k),
                        exp_width_min(fa, fb, fk),
                        5e-5
                    ),
                    "exp union ({a}, {b}, {k})"
                );
                assert!(
                    close(
                        sdf_exp_smooth_intersection(a, b, k),
                        exp_width_max(fa, fb, fk),
                        5e-5
                    ),
                    "exp intersection ({a}, {b}, {k})"
                );
                assert!(
                    close(
                        sdf_exp_smooth_subtraction(a, b, k),
                        exp_width_max(fa, -fb, fk),
                        5e-5
                    ),
                    "exp subtraction ({a}, {b}, {k})"
                );
            }
        }
    }
}

#[test]
fn smooth_min_exp_matches_the_rate_convention_for_moderate_arguments() {
    // `smooth_min_exp` takes `k` as a *rate* (`exp(-k d)`), unlike the node law (`d/k`)
    for &k in &[0.5_f32, 2.0, 8.0] {
        for &a in &samples() {
            for &b in &samples() {
                let got = smooth_min_exp(a, b, k);
                let want = exp_rate_min(f64::from(a), f64::from(b), f64::from(k));
                assert!(
                    close(got, want, 5e-5),
                    "smooth_min_exp({a}, {b}, {k}) = {got}, want {want}"
                );
            }
        }
    }
}

#[test]
fn cubic_and_root_smooth_min_match_the_closed_form() {
    for &k in &[0.05_f32, 0.3, 1.0, 2.5] {
        for &a in &samples() {
            for &b in &samples() {
                let (fa, fb, fk) = (f64::from(a), f64::from(b), f64::from(k));
                assert!(
                    close(smooth_min_cubic(a, b, k), cubic_min(fa, fb, fk), 1e-5),
                    "cubic ({a}, {b}, {k})"
                );
                assert!(
                    close(smooth_min_root(a, b, k), root_min(fa, fb, fk), 5e-5),
                    "root ({a}, {b}, {k})"
                );
            }
        }
    }
}

// ───────────────────────── B. structural laws ─────────────────────────

struct Variant {
    name: &'static str,
    f: fn(f32, f32, f32) -> f32,
    /// deficit below `min(a, b)` when `a == b` (the largest the blend can pull down)
    deficit: fn(f32) -> f32,
    /// how `k` must scale so that `f(s a, s b, k') = s f(a, b, k)`
    scale_k: fn(f32, f32) -> f32,
    ks: &'static [f32],
}

fn variants() -> Vec<Variant> {
    vec![
        Variant {
            name: "polynomial",
            f: smooth_min,
            deficit: |k| k / 4.0,
            scale_k: |k, s| k * s,
            ks: &[0.1, 0.6, 2.0],
        },
        Variant {
            name: "cubic",
            f: smooth_min_cubic,
            deficit: |k| k / 6.0,
            scale_k: |k, s| k * s,
            ks: &[0.1, 0.6, 2.0],
        },
        Variant {
            name: "exp (width)",
            f: sdf_exp_smooth_union,
            deficit: |k| k * std::f32::consts::LN_2,
            scale_k: |k, s| k * s,
            ks: &[0.1, 0.6, 2.0],
        },
        Variant {
            name: "exp (rate)",
            f: smooth_min_exp,
            deficit: |k| std::f32::consts::LN_2 / k,
            scale_k: |k, s| k / s,
            ks: &[0.5, 2.0, 8.0],
        },
        Variant {
            name: "root",
            f: smooth_min_root,
            deficit: |k| k / 2.0,
            scale_k: |k, s| k * s,
            ks: &[0.1, 0.6, 2.0],
        },
    ]
}

#[test]
fn every_smooth_min_is_symmetric_and_bounded_below_by_min_minus_the_blend_deficit() {
    for v in variants() {
        for &k in v.ks {
            for &a in &samples() {
                for &b in &samples() {
                    let s = (v.f)(a, b, k);
                    assert!(
                        ((v.f)(b, a, k) - s).abs() <= 1e-6,
                        "{}: not symmetric at ({a}, {b}, {k})",
                        v.name
                    );
                    let m = a.min(b);
                    assert!(
                        s <= m + 1e-6,
                        "{}: {s} > min {m} at ({a}, {b}, {k})",
                        v.name
                    );
                    // the blend never pulls the field below min - deficit(k) (reached at a == b)
                    assert!(
                        s >= m - (v.deficit)(k) - 1e-5,
                        "{}: {s} < min - deficit at ({a}, {b}, {k})",
                        v.name
                    );
                }
            }
            // at a == b the pull is exactly the documented deficit
            let a = 1.25_f32;
            let s = (v.f)(a, a, k);
            assert!(
                (a - s - (v.deficit)(k)).abs() < 2e-5,
                "{}: deficit at a == b is {} (want {})",
                v.name,
                a - s,
                (v.deficit)(k)
            );
        }
    }
}

#[test]
fn every_smooth_min_is_shift_invariant_and_positively_homogeneous() {
    for v in variants() {
        for &k in v.ks {
            for &a in &samples() {
                for &b in &samples() {
                    let base = (v.f)(a, b, k);
                    // translating both distances translates the result
                    let c = 3.5_f32;
                    assert!(
                        ((v.f)(a + c, b + c, k) - (base + c)).abs() < 2e-5,
                        "{}: shift at ({a}, {b}, {k})",
                        v.name
                    );
                    // scaling both distances and the blend width scales the result
                    let s = 2.0_f32;
                    let scaled = (v.f)(a * s, b * s, (v.scale_k)(k, s));
                    assert!(
                        (scaled - base * s).abs() < 3e-5,
                        "{}: homogeneity at ({a}, {b}, {k}): {scaled} vs {}",
                        v.name,
                        base * s
                    );
                }
            }
        }
    }
}

#[test]
fn every_smooth_min_is_monotone_and_its_blend_weights_sum_to_one() {
    // d/da + d/db = 1 (shift invariance) and each partial derivative lies in [0, 1]
    let h = 2e-3_f32;
    for v in variants() {
        for &k in v.ks {
            for &a in &samples() {
                for &b in &samples() {
                    let f = &v.f;
                    let wa = (f(a + h, b, k) - f(a - h, b, k)) / (2.0 * h);
                    let wb = (f(a, b + h, k) - f(a, b - h, k)) / (2.0 * h);
                    assert!(
                        (-5e-3..=1.0 + 5e-3).contains(&wa),
                        "{}: d/da = {wa} at ({a}, {b}, {k})",
                        v.name
                    );
                    assert!(
                        (-5e-3..=1.0 + 5e-3).contains(&wb),
                        "{}: d/db = {wb} at ({a}, {b}, {k})",
                        v.name
                    );
                    assert!(
                        (wa + wb - 1.0).abs() < 8e-3,
                        "{}: weights sum to {} at ({a}, {b}, {k})",
                        v.name,
                        wa + wb
                    );
                }
            }
        }
    }
}

#[test]
fn polynomial_and_cubic_blends_have_compact_support() {
    // beyond |a - b| >= k the blend is exactly the plain min (no tail)
    for &k in &[0.1_f32, 0.7, 2.0] {
        for &a in &samples() {
            let b = a + k * 1.0001 + 0.001;
            assert_eq!(
                smooth_min(a, b, k),
                a.min(b),
                "polynomial tail at ({a}, {b}, {k})"
            );
            assert_eq!(
                smooth_min_cubic(a, b, k),
                a.min(b),
                "cubic tail at ({a}, {b}, {k})"
            );
            assert_eq!(
                smooth_max(a, b, k),
                a.max(b),
                "polynomial max tail at ({a}, {b}, {k})"
            );
        }
    }
}

#[test]
fn a_vanishing_or_negative_blend_width_degenerates_to_min() {
    for &(a, b) in &[(2.0_f32, 5.0), (-1.0, 0.5), (3.0, 3.0), (0.0, -0.0)] {
        for &k in &[0.0_f32, 1e-12, -1.0] {
            assert!(
                (smooth_min(a, b, k) - a.min(b)).abs() < 1e-8,
                "polynomial k={k} at ({a}, {b})"
            );
            assert!(
                (smooth_max(a, b, k) - a.max(b)).abs() < 1e-8,
                "polynomial max k={k} at ({a}, {b})"
            );
            assert!(
                (smooth_min_cubic(a, b, k) - a.min(b)).abs() < 1e-8,
                "cubic k={k} at ({a}, {b})"
            );
        }
        assert!(
            (smooth_min_root(a, b, 0.0) - a.min(b)).abs() < 1e-6,
            "root k=0 at ({a}, {b})"
        );
    }
}

// ───────────────────────── C. geometry of the blended corner ─────────────────────────

const fn plane(nx: f32, ny: f32) -> SdfNode {
    SdfNode::Plane {
        normal: Vec3::new(nx, ny, 0.0),
        distance: 0.0,
    }
}

/// distance from the inner corner to the blended surface along the diagonal of the free quadrant
fn corner_depth(node: &SdfNode) -> f32 {
    // free space: x > 0 and y > 0 (the two half-spaces y <= 0 and x <= 0 are solid)
    let dir = Vec3::new(1.0, 1.0, 0.0).normalize();
    let (mut lo, mut hi) = (0.0_f32, 6.0_f32); // lo is inside material (corner), hi is empty
    assert!(
        eval(node, dir * lo) <= 0.0 && eval(node, dir * hi) > 0.0,
        "diagonal does not cross the surface"
    );
    for _ in 0..60 {
        let mid = f32::midpoint(lo, hi);
        if eval(node, dir * mid) <= 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f32::midpoint(lo, hi)
}

#[test]
fn blended_corner_depth_is_the_closed_form_for_the_polynomial_and_exp_nodes() {
    // both planes are at distance u = t / sqrt 2 from a diagonal point at distance t from the corner,
    // so the surface is where smin(u, u) = 0  =>  u = deficit(k)  =>  t = sqrt 2 * deficit(k)
    for &k in &[0.3_f32, 1.0, 2.0] {
        let poly = SdfNode::SmoothUnion {
            a: Arc::new(plane(1.0, 0.0)),
            b: Arc::new(plane(0.0, 1.0)),
            k,
        };
        let want = std::f32::consts::SQRT_2 * k / 4.0;
        let got = corner_depth(&poly);
        assert!(
            (got - want).abs() < 2e-4,
            "SmoothUnion k={k}: corner depth {got} (closed form sqrt2 k/4 = {want})"
        );

        let exp = SdfNode::ExpSmoothUnion {
            a: Arc::new(plane(1.0, 0.0)),
            b: Arc::new(plane(0.0, 1.0)),
            k,
        };
        let want = std::f32::consts::SQRT_2 * k * std::f32::consts::LN_2;
        let got = corner_depth(&exp);
        assert!(
            (got - want).abs() < 5e-4,
            "ExpSmoothUnion k={k}: corner depth {got} (closed form sqrt2 k ln2 = {want})"
        );
    }
}

// ───────────────────────── D. one law, many evaluators ─────────────────────────

#[test]
fn the_reciprocal_forms_are_bit_identical_to_the_scalar_forms() {
    // documented: "`smooth_min` computes the same reciprocal and calls here, so the tree and the
    // bytecode round identically"
    let mut state: u32 = 0x1234_5678;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    for _ in 0..5000 {
        let (a, b) = (next() * 8.0 - 4.0, next() * 8.0 - 4.0);
        let k = next() * 2.0 + 0.01;
        let rk = 1.0 / k;
        assert_eq!(
            smooth_min_rk(a, b, k, rk).to_bits(),
            smooth_min(a, b, k).to_bits(),
            "min ({a}, {b}, {k})"
        );
        assert_eq!(
            smooth_max_rk(a, b, k, rk).to_bits(),
            smooth_max(a, b, k).to_bits(),
            "max ({a}, {b}, {k})"
        );
        assert_eq!(
            sdf_smooth_union_rk(a, b, k, rk).to_bits(),
            sdf_smooth_union(a, b, k).to_bits()
        );
        assert_eq!(
            sdf_smooth_intersection_rk(a, b, k, rk).to_bits(),
            sdf_smooth_intersection(a, b, k).to_bits()
        );
        assert_eq!(
            sdf_smooth_subtraction_rk(a, b, k, rk).to_bits(),
            sdf_smooth_subtraction(a, b, k).to_bits()
        );
    }
}

#[test]
fn the_generic_forms_are_the_scalar_law_on_f32_and_on_every_simd_lane() {
    // "one law for scalar and SIMD evaluators": the polynomial family has no transcendental, so
    // every lane must equal the scalar result bit for bit; the exp family goes through each
    // backend's `exp` / `ln`, so it is compared to the closed form instead
    let mut state: u32 = 0x9E37_79B9;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    for _ in 0..500 {
        let a: [f32; 8] = std::array::from_fn(|_| next() * 6.0 - 3.0);
        let b: [f32; 8] = std::array::from_fn(|_| next() * 6.0 - 3.0);
        let k = next() * 1.5 + 0.05;
        let rk = 1.0 / k;
        let (va, vb) = (f32x8::from(a), f32x8::from(b));
        let lanes = |v: f32x8| v.to_array();
        let (u, i, d) = (
            lanes(sdf_smooth_union_rk_r(va, vb, k, rk)),
            lanes(sdf_smooth_intersection_rk_r(va, vb, k, rk)),
            lanes(sdf_smooth_subtraction_rk_r(va, vb, k, rk)),
        );
        let (mn, mx) = (
            lanes(smooth_min_rk_r(va, vb, k, rk)),
            lanes(smooth_max_rk_r(va, vb, k, rk)),
        );
        let (eu, ei, es) = (
            lanes(sdf_exp_smooth_union_r(va, vb, k)),
            lanes(sdf_exp_smooth_intersection_r(va, vb, k)),
            lanes(sdf_exp_smooth_subtraction_r(va, vb, k)),
        );
        for l in 0..8 {
            let (x, y) = (a[l], b[l]);
            // the f32 instantiation is the scalar form
            assert_eq!(
                sdf_smooth_union_rk_r::<f32>(x, y, k, rk).to_bits(),
                sdf_smooth_union(x, y, k).to_bits()
            );
            assert_eq!(
                sdf_smooth_subtraction_rk_r::<f32>(x, y, k, rk).to_bits(),
                sdf_smooth_subtraction(x, y, k).to_bits(),
                "f32 subtraction ({x}, {y}, {k})"
            );
            // each SIMD lane equals the scalar result
            assert_eq!(
                u[l].to_bits(),
                sdf_smooth_union(x, y, k).to_bits(),
                "lane {l} union ({x}, {y}, {k})"
            );
            assert_eq!(
                i[l].to_bits(),
                sdf_smooth_intersection(x, y, k).to_bits(),
                "lane {l} intersection"
            );
            assert_eq!(
                d[l].to_bits(),
                sdf_smooth_subtraction(x, y, k).to_bits(),
                "lane {l} subtraction ({x}, {y}, {k})"
            );
            assert_eq!(
                mn[l].to_bits(),
                smooth_min(x, y, k).to_bits(),
                "lane {l} min"
            );
            assert_eq!(
                mx[l].to_bits(),
                smooth_max(x, y, k).to_bits(),
                "lane {l} max"
            );
            let (fx, fy, fk) = (f64::from(x), f64::from(y), f64::from(k));
            assert!(
                close(eu[l], exp_width_min(fx, fy, fk), 2e-4),
                "lane {l} exp union ({x}, {y}, {k}): {}",
                eu[l]
            );
            assert!(
                close(ei[l], exp_width_max(fx, fy, fk), 2e-4),
                "lane {l} exp intersection"
            );
            assert!(
                close(es[l], exp_width_max(fx, -fy, fk), 2e-4),
                "lane {l} exp subtraction"
            );
        }
    }
}

fn sphere_at(x: f32, r: f32) -> SdfNode {
    SdfNode::Translate {
        child: Arc::new(SdfNode::Sphere { radius: r }),
        offset: Vec3::new(x, 0.0, 0.0),
    }
}

/// (name, node, closed form of the node law, relative tolerance)
type NodeCase = (&'static str, SdfNode, fn(f64, f64, f64) -> f64, f64);

#[test]
fn smooth_nodes_evaluate_to_the_closed_form_of_their_children() {
    // children are exact sphere distances: d1 = |p| - 1, d2 = |p - (1.2, 0, 0)| - 0.8
    let (a, b) = (Arc::new(sphere_at(0.0, 1.0)), Arc::new(sphere_at(1.2, 0.8)));
    for &k in &[0.2_f32, 0.8] {
        let nodes: [NodeCase; 6] = [
            (
                "SmoothUnion",
                SdfNode::SmoothUnion {
                    a: a.clone(),
                    b: b.clone(),
                    k,
                },
                poly_min,
                1e-5,
            ),
            (
                "SmoothIntersection",
                SdfNode::SmoothIntersection {
                    a: a.clone(),
                    b: b.clone(),
                    k,
                },
                poly_max,
                1e-5,
            ),
            (
                "SmoothSubtraction",
                SdfNode::SmoothSubtraction {
                    a: a.clone(),
                    b: b.clone(),
                    k,
                },
                |d1, d2, k| poly_max(d1, -d2, k),
                1e-5,
            ),
            (
                "ExpSmoothUnion",
                SdfNode::ExpSmoothUnion {
                    a: a.clone(),
                    b: b.clone(),
                    k,
                },
                exp_width_min,
                5e-5,
            ),
            (
                "ExpSmoothIntersection",
                SdfNode::ExpSmoothIntersection {
                    a: a.clone(),
                    b: b.clone(),
                    k,
                },
                exp_width_max,
                5e-5,
            ),
            (
                "ExpSmoothSubtraction",
                SdfNode::ExpSmoothSubtraction {
                    a: a.clone(),
                    b: b.clone(),
                    k,
                },
                |d1, d2, k| exp_width_max(d1, -d2, k),
                5e-5,
            ),
        ];
        for ix in -8..=8 {
            for iy in -6..=6 {
                let p = Vec3::new(ix as f32 * 0.3, iy as f32 * 0.3, 0.2);
                let d1 = f64::from((p).length()) - 1.0;
                let d2 = f64::from((p - Vec3::new(1.2, 0.0, 0.0)).length()) - 0.8;
                for (name, node, f, tol) in &nodes {
                    let want = f(d1, d2, f64::from(k));
                    let got = eval(node, p);
                    assert!(
                        close(got, want, *tol),
                        "{name} k={k} at {p:?}: {got} vs {want}"
                    );
                }
            }
        }
    }
}

// ───────────────────────── E. robustness ─────────────────────────

#[test]
fn smooth_min_exp_is_finite_for_finite_arguments_far_from_the_surface() {
    // exp(-k a) underflows to 0 for k a >~ 88 (f32) and overflows to +inf for k a <~ -88; the
    // textbook form then returns +inf / -inf for a perfectly finite distance (the node law
    // `sdf_exp_smooth_union` and the generic form already factor the minimum out)
    let cases = [
        (100.0_f32, 100.0, 10.0),
        (-100.0, -100.0, 10.0),
        (100.0, -100.0, 10.0),
        (50.0, 51.0, 8.0),
        (1.0e4, 1.0e4, 1.0),
    ];
    let mut failures = Vec::new();
    for (a, b, k) in cases {
        let got = smooth_min_exp(a, b, k);
        let want = exp_rate_min(f64::from(a), f64::from(b), f64::from(k));
        if !got.is_finite() || !close(got, want, 1e-4) {
            failures.push(format!(
                "smooth_min_exp({a}, {b}, {k}) = {got} (closed form {want})"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn exp_width_forms_stay_finite_far_from_the_surface() {
    for &(a, b, k) in &[
        (1.0e4_f32, 1.0e4, 0.01),
        (-1.0e4, -1.0e4, 0.01),
        (1.0e4, -1.0e4, 0.01),
        (0.0, 0.0, 1e-6),
    ] {
        for (name, v) in [
            ("union", sdf_exp_smooth_union(a, b, k)),
            ("intersection", sdf_exp_smooth_intersection(a, b, k)),
            ("subtraction", sdf_exp_smooth_subtraction(a, b, k)),
        ] {
            assert!(v.is_finite(), "{name}({a}, {b}, {k}) = {v}");
        }
    }
}

#[test]
fn smooth_ops_do_not_panic_on_degenerate_inputs() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let specials = [
        0.0_f32,
        -0.0,
        1.0,
        -1.0,
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut acc = 0.0_f32;
        for &a in &specials {
            for &b in &specials {
                for &k in &specials {
                    acc += smooth_min(a, b, k).max(0.0) + smooth_max(a, b, k).min(0.0);
                    acc += smooth_min_cubic(a, b, k).max(0.0) + smooth_min_root(a, b, k).max(0.0);
                    acc += smooth_min_exp(a, b, k).max(0.0);
                    acc += sdf_exp_smooth_union(a, b, k).max(0.0)
                        + sdf_exp_smooth_subtraction(a, b, k).max(0.0);
                }
            }
        }
        acc
    }));
    assert!(
        r.is_ok(),
        "a smooth operator panicked on a degenerate input"
    );
}
