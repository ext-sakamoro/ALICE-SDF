//! Sparse voxel octree oracle: the stored distance must track the analytic
//! field. `query_point` returns the deepest node's distance, so near the
//! surface (leaves at `max_depth`) the error is bounded by the finest half
//! diagonal; far away the leaf is at most `|d| / distance_threshold` wide.
//! Refining the tree must reduce the error, ray queries must agree with a
//! scan oracle, and linearisation must not change any query.
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "svo")]

use alice_sdf::prelude::*;
use alice_sdf::svo::linearize::validate_linearized;
use alice_sdf::svo::{SparseVoxelOctree, SvoBuildConfig};

const B: f32 = 2.0;

fn scene() -> SdfNode {
    SdfNode::sphere(0.9)
        .smooth_union(SdfNode::box3d(1.2, 0.5, 0.8).translate(0.4, 0.3, 0.0), 0.15)
        .subtract(SdfNode::cylinder(0.25, 3.0))
}

fn cfg(depth: u32) -> SvoBuildConfig {
    SvoBuildConfig {
        max_depth: depth,
        bounds_min: Vec3::splat(-B),
        bounds_max: Vec3::splat(B),
        ..Default::default()
    }
}

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn points(n: usize, seed: u64) -> Vec<Vec3> {
    let mut r = lcg(seed);
    (0..n)
        .map(|_| {
            Vec3::new(
                r().mul_add(2.0 * B, -B),
                r().mul_add(2.0 * B, -B),
                r().mul_add(2.0 * B, -B),
            )
        })
        .collect()
}

/// Error statistics of `query_point` against `eval` on `pts`.
fn errors(node: &SdfNode, svo: &SparseVoxelOctree, pts: &[Vec3]) -> (f32, f32) {
    let mut worst = 0.0f32;
    let mut sum2 = 0.0f32;
    for &p in pts {
        let e = (svo.query_point(p) - eval(node, p)).abs();
        worst = worst.max(e);
        sum2 = e.mul_add(e, sum2);
    }
    (worst, (sum2 / pts.len() as f32).sqrt())
}

#[test]
fn svo_distance_is_bounded_by_leaf_size() {
    let node = scene();
    let depth = 7;
    let svo = SparseVoxelOctree::build(&node, &cfg(depth));
    let finest_half = B / (1u32 << depth) as f32; // half size of a max-depth leaf
    let finest_half_diag = finest_half * 3f32.sqrt();
    let threshold = cfg(depth).distance_threshold;
    let mut worst_near = 0.0f32;
    let mut violations = Vec::new();
    for p in points(20_000, 1) {
        let d = eval(&node, p);
        let q = svo.query_point(p);
        let err = (q - d).abs();
        // subdivision stops when |d_centre| > threshold · node_size (= 2h),
        // and |d_centre| ≤ |d| + h√3 (1-Lipschitz), so the leaf containing p
        // has half size h < |d| / (2·threshold − √3) unless it is a max-depth
        // leaf; either way the centre sample is within h√3 of the truth
        let leaf_half = (d.abs() / 2.0f32.mul_add(threshold, -3f32.sqrt())).max(finest_half);
        let bound = leaf_half * 3f32.sqrt() + 1e-4;
        if err > bound {
            violations.push(format!(
                "p={p:?} d={d:.4} q={q:.4} err={err:.4} bound={bound:.4}"
            ));
        }
        if d.abs() < 2.0 * finest_half {
            worst_near = worst_near.max(err);
        }
    }
    assert!(
        violations.is_empty(),
        "{} of 20000 queries exceed the leaf-size bound:\n{}",
        violations.len(),
        violations
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        worst_near <= finest_half_diag + 1e-4,
        "near-surface error {worst_near} > finest half diagonal {finest_half_diag}"
    );
}

#[test]
fn svo_error_decreases_with_depth() {
    let node = scene();
    let pts: Vec<Vec3> = points(40_000, 2)
        .into_iter()
        .filter(|&p| eval(&node, p).abs() < 0.3)
        .collect();
    assert!(pts.len() > 1000);
    let mut prev = f32::MAX;
    for depth in [4u32, 5, 6, 7] {
        let svo = SparseVoxelOctree::build(&node, &cfg(depth));
        let (worst, rmse) = errors(&node, &svo, &pts);
        eprintln!(
            "depth {depth}: worst {worst:.4} rmse {rmse:.5} nodes {}",
            svo.node_count()
        );
        assert!(
            rmse < prev,
            "depth {depth}: rmse {rmse} did not improve on {prev}"
        );
        prev = rmse;
    }
}

#[test]
fn svo_ray_query_matches_scan_oracle() {
    let node = scene();
    let depth = 7;
    let svo = SparseVoxelOctree::build(&node, &cfg(depth));
    let finest = 2.0 * B / (1u32 << depth) as f32;
    let step = 2e-3;
    let (mut agree, mut miss, mut false_hit, mut off) = (0, 0, 0, 0);
    let n = 16;
    for iy in 0..n {
        for ix in 0..n {
            let f = |i: usize| -1.5 + 3.0 * (i as f32) / ((n - 1) as f32);
            let o = Vec3::new(f(ix), f(iy), -1.99);
            let d = Vec3::Z;
            if eval(&node, o) <= 0.0 {
                continue;
            }
            // scan oracle
            let mut oracle = None;
            for i in 0..(3.98 / step) as usize {
                let t = i as f32 * step;
                if eval(&node, o + d * (t + step)) <= 0.0 {
                    oracle = Some(0.5f32.mul_add(step, t));
                    break;
                }
            }
            match (oracle, svo.ray_query(o, d, 3.98)) {
                (Some(tw), Some(h)) => {
                    if (h.distance - tw).abs() <= 2.0 * finest {
                        agree += 1;
                    } else {
                        off += 1;
                    }
                }
                (None, None) => agree += 1,
                (Some(_), None) => miss += 1,
                // a grazing ray can "hit" the voxelised surface while the
                // analytic one misses by less than a leaf — resolution, not a bug
                (None, Some(h)) => {
                    if eval(&node, h.position).abs() <= 2.0 * finest {
                        agree += 1;
                    } else {
                        false_hit += 1;
                    }
                }
            }
        }
    }
    eprintln!("svo ray query: agree {agree}, miss {miss}, false hit {false_hit}, off {off}");
    assert_eq!(miss + false_hit, 0, "rays missed or falsely hit");
    assert!(
        off * 50 <= agree,
        "{off} hits farther than two finest cells from the oracle"
    );
}

#[test]
fn svo_linearize_preserves_queries() {
    let node = scene();
    let svo = SparseVoxelOctree::build(&node, &cfg(6));
    let lin = svo.linearize();
    validate_linearized(&lin).expect("linearized SVO is valid");
    assert_eq!(lin.nodes.len(), svo.node_count());
    // same nodes, same order → identical queries through the original tree
    for (a, b) in lin.nodes.iter().zip(&svo.nodes) {
        assert_eq!(a.distance.to_bits(), b.distance.to_bits());
        assert_eq!(a.is_leaf, b.is_leaf);
    }
}
