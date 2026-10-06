//! N-ary union / intersection vs the definition `min` / `max` over the set.
//!
//! `sdf_union_multi` and `sdf_intersection_multi` fold a slice of distances.
//! The references are the set minimum / maximum written as plain loops in
//! `f64`, the empty-set conventions (`f32::MAX` for the union: nothing is
//! inside; `f32::MIN` for the intersection: everything is inside), and the
//! tree evaluator: a left fold of `SdfNode::union` / `intersection` over
//! spheres must give the same distance as the slice fold of the sphere
//! distances, bit for bit (`min` / `max` are exact).
//!
//! The smooth blends (`smooth_min_cubic` / `_exp` / `_root` and the
//! `sdf_smooth_*_rk` forms) are checked against their closed forms in
//! `tests/test_smooth_ops_oracle.rs`.
//!
//! Author: Moroya Sakamoto
#![allow(clippy::float_cmp)]

use alice_sdf::operations::{sdf_intersection_multi, sdf_union_multi};
use alice_sdf::{eval, SdfNode};
use glam::Vec3;

fn slices() -> Vec<Vec<f32>> {
    let mut out = vec![
        vec![0.5],
        vec![-0.25, 0.75],
        vec![3.0, -1.5, 2.25, -1.25, 0.0],
    ];
    // a deterministic pseudo-random family
    let mut s = 0x9E37_79B9_u32;
    for len in 1..12 {
        let v = (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s % 20_001) as f32 / 1000.0 - 10.0
            })
            .collect();
        out.push(v);
    }
    out
}

#[test]
fn union_multi_is_the_set_minimum_and_intersection_multi_the_set_maximum() {
    let mut n = 0;
    for v in slices() {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &d in &v {
            lo = lo.min(d as f64);
            hi = hi.max(d as f64);
        }
        assert_eq!(sdf_union_multi(&v) as f64, lo, "union of {v:?}");
        assert_eq!(
            sdf_intersection_multi(&v) as f64,
            hi,
            "intersection of {v:?}"
        );
        n += 2;
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn empty_set_conventions() {
    assert_eq!(sdf_union_multi(&[]), f32::MAX);
    assert_eq!(sdf_intersection_multi(&[]), f32::MIN);
}

#[test]
fn slice_folds_agree_with_the_tree_evaluator() {
    let centers = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.9, 0.1, -0.2),
        Vec3::new(-0.4, 0.7, 0.3),
        Vec3::new(0.2, -0.6, 0.8),
    ];
    let radii = [0.8_f32, 0.6, 0.5, 0.7];
    let spheres: Vec<SdfNode> = centers
        .iter()
        .zip(radii)
        .map(|(c, r)| SdfNode::sphere(r).translate(c.x, c.y, c.z))
        .collect();
    let union = spheres[1..]
        .iter()
        .fold(spheres[0].clone(), |acc, s| acc.union(s.clone()));
    let inter = spheres[1..]
        .iter()
        .fold(spheres[0].clone(), |acc, s| acc.intersection(s.clone()));
    let mut n = 0;
    for ix in -5..=5 {
        for iy in -5..=5 {
            for iz in -5..=5 {
                let p = Vec3::new(ix as f32 * 0.31, iy as f32 * 0.27, iz as f32 * 0.33);
                let ds: Vec<f32> = spheres.iter().map(|s| eval(s, p)).collect();
                assert_eq!(sdf_union_multi(&ds), eval(&union, p), "union at {p:?}");
                assert_eq!(
                    sdf_intersection_multi(&ds),
                    eval(&inter, p),
                    "intersection at {p:?}"
                );
                n += 2;
            }
        }
    }
    assert!(n > 0, "no comparisons made");
}
