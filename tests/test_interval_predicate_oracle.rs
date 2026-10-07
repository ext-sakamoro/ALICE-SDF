//! `Interval` set predicates against their set-theoretic definitions, and the
//! sphere's interval extension against its exact range.
//!
//! oracle:
//! - `[a, b]` is the set {v : a ≤ v ≤ b}; `contains`, `overlaps`,
//!   `intersect`, `is_positive`, `is_negative` are checked against that set by
//!   membership of every value of a fixed lattice (exact in f32)
//! - for a sphere of radius r over an axis-aligned box, the exact range of
//!   |p| − r is [dist(0, box) − r, max corner |p| − r]; a sound interval must
//!   contain it, and the sphere's interval extension is tight up to rounding
//!
//! Author: Moroya Sakamoto

use alice_sdf::interval::{eval_interval, Interval, Vec3Interval};
use alice_sdf::types::SdfNode;
use glam::Vec3;

/// Endpoints and probe values on a 0.25 lattice (exact in f32).
fn lattice() -> Vec<f32> {
    (-12..=12).map(|i| i as f32 * 0.25).collect()
}

const fn raw(lo: f32, hi: f32) -> Interval {
    // the exact interval [lo, hi]; `Interval::new` rounds outward
    Interval { lo, hi }
}

#[test]
fn set_predicates_match_membership() {
    let vals = lattice();
    let mut compared = 0usize;
    for &a in &vals {
        for &b in vals.iter().filter(|&&b| b >= a) {
            let i = raw(a, b);
            let member = |v: f32| a <= v && v <= b;

            assert_eq!(i.is_positive(), a > 0.0);
            assert_eq!(i.is_negative(), b < 0.0);
            for &v in &vals {
                assert_eq!(i.contains(v), member(v), "[{a},{b}] ∋ {v}");
            }
            // contains(0) ⇔ neither entirely positive nor entirely negative
            assert_eq!(i.contains(0.0), !i.is_positive() && !i.is_negative());

            for &c in vals.iter().step_by(3) {
                for &d in vals.iter().step_by(3).filter(|&&d| d >= c) {
                    let j = raw(c, d);
                    let common: Vec<f32> = vals
                        .iter()
                        .copied()
                        .filter(|&v| member(v) && c <= v && v <= d)
                        .collect();
                    // on this lattice every endpoint is a lattice point, so two
                    // intervals overlap exactly when they share a lattice value
                    assert_eq!(i.overlaps(j), !common.is_empty(), "[{a},{b}] ∩ [{c},{d}]");
                    if let (Some(&lo), Some(&hi)) = (common.first(), common.last()) {
                        let k = i.intersect(j);
                        assert_eq!((k.lo, k.hi), (lo, hi));
                    }
                    compared += 1;
                }
            }
        }
    }
    assert!(compared > 10_000, "compared {compared}");
}

#[test]
fn sphere_interval_contains_the_exact_range_and_is_tight() {
    let r = 1.0_f32;
    let sphere = SdfNode::sphere(r);
    let boxes = [
        (Vec3::new(0.5, -0.25, 0.0), Vec3::new(1.5, 0.25, 0.75)),
        (Vec3::new(-2.0, -2.0, -2.0), Vec3::new(-1.0, -1.5, -1.25)),
        (Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5)),
        (Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
    ];
    let mut compared = 0;
    for (lo, hi) in boxes {
        let iv = eval_interval(&sphere, Vec3Interval::from_bounds(lo, hi));
        let nearest = Vec3::ZERO.clamp(lo, hi);
        let far = Vec3::new(
            lo.x.abs().max(hi.x.abs()),
            lo.y.abs().max(hi.y.abs()),
            lo.z.abs().max(hi.z.abs()),
        );
        let (dmin, dmax) = (
            f64::from(nearest.length()) - f64::from(r),
            f64::from(far.length()) - f64::from(r),
        );
        assert!(
            f64::from(iv.lo) <= dmin && dmax <= f64::from(iv.hi),
            "{iv:?} vs [{dmin},{dmax}]"
        );
        // tight up to a few ulps of the endpoints
        let slack = 1e-5 * (1.0 + dmax.abs());
        assert!(
            dmin - f64::from(iv.lo) <= slack && f64::from(iv.hi) - dmax <= slack,
            "{iv:?}"
        );
        assert_eq!(iv.is_positive(), dmin > 0.0);
        assert_eq!(iv.is_negative(), dmax < 0.0);
        compared += 1;
    }
    assert_eq!(compared, boxes.len());
}
