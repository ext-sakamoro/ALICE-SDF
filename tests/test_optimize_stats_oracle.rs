//! `optimize` preserves the field, `optimization_stats` counts what it removed,
//! and the medium tight-AABB preset bounds shapes of its size class.
//!
//! oracle:
//! - an optimised tree returns the same distance as the original. Every
//!   folded node here is an exact identity (offset 0, factor 1, k = 0,
//!   radius 0, identity rotation) and the merged translations / scales are
//!   dyadic, so the merged constant and every grid point are exact in f32 —
//!   the comparison is bit equality, not a tolerance. The one exception is
//!   SmoothUnion(k = 0) → Union: the evaluator floors k at 1e-10 to avoid a
//!   division by zero, and the polynomial smooth minimum lies at most k/4
//!   below min(a, b), so that case is compared to within 1e-10 / 4
//! - node counts are counted by hand from the tree literal
//! - a sphere of radius r has the exact box [−r, r]³ and a box of half
//!   extents h has [−h, h] (the tight AABB must contain it and be at most one
//!   bisection step wider)
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::optimize::{optimization_stats, optimize, OptimizationStats};
use alice_sdf::tight_aabb::{compute_tight_aabb_with_config, TightAabbConfig};
use alice_sdf::types::SdfNode;
use glam::{Quat, Vec3};

fn grid() -> Vec<Vec3> {
    let mut v = Vec::new();
    for z in -4..=4 {
        for y in -4..=4 {
            for x in -4..=4 {
                v.push(Vec3::new(x as f32, y as f32, z as f32) * 0.375);
            }
        }
    }
    v
}

/// (tree, hand-counted nodes before, hand-counted nodes after)
fn cases() -> Vec<(SdfNode, u32, u32)> {
    let s = || SdfNode::sphere(0.75);
    let b = || SdfNode::box3d(1.0, 0.5, 1.5);
    vec![
        // Translate(0) + Scale(1) around a sphere: 3 → 1
        (s().translate(0.0, 0.0, 0.0).scale(1.0), 3, 1),
        // identity rotation and Round(0): 3 → 1
        (b().rotate(Quat::IDENTITY).round(0.0), 3, 1),
        // nested dyadic translations merge: 3 → 2
        (
            s().translate(0.25, 0.0, 0.0).translate(0.0, -0.5, 0.125),
            3,
            2,
        ),
        // nested power-of-two scales merge into Scale(1), which is folded: 3 → 1
        (b().scale(2.0).scale(0.5), 3, 1),
        // nested translations that cancel: 3 → 1
        (
            s().translate(0.5, -0.25, 0.0).translate(-0.5, 0.25, 0.0),
            3,
            1,
        ),
        // SmoothUnion(k = 0) demotes to Union (count unchanged 3 → 3)
        (s().smooth_union(b(), 0.0), 3, 3),
        // already minimal: 4 → 4
        (s().union(b().translate(0.5, 0.0, 0.0)), 4, 4),
    ]
}

#[test]
fn optimized_tree_returns_the_same_distance() {
    let pts = grid();
    let mut compared = 0;
    for (tree, _, _) in cases() {
        let opt = optimize(&tree);
        let smooth = matches!(tree, SdfNode::SmoothUnion { .. });
        for &p in &pts {
            let (o, t) = (eval(&opt, p), eval(&tree, p));
            if smooth {
                assert!(
                    o >= t && o - t <= 1e-10 / 4.0 * 1.0001,
                    "{tree:?} at {p:?}: {o} vs {t}"
                );
            } else {
                assert_eq!(o.to_bits(), t.to_bits(), "{tree:?} at {p:?}");
            }
            compared += 1;
        }
    }
    assert_eq!(compared, 7 * 729);
}

#[test]
fn optimization_stats_counts_the_removed_nodes() {
    let mut compared = 0;
    for (tree, before, after) in cases() {
        assert_eq!(tree.node_count(), before, "{tree:?}");
        let opt = optimize(&tree);
        let stats: OptimizationStats = optimization_stats(&tree, &opt);
        assert_eq!(stats.nodes_before, before);
        assert_eq!(stats.nodes_after, after, "{tree:?} → {opt:?}");
        assert_eq!(stats.nodes_removed, before - after);
        compared += 1;
    }
    assert_eq!(compared, 7);

    // stats of a tree that grew saturate at 0 removed
    let small = SdfNode::sphere(1.0);
    let big = small.clone().translate(1.0, 0.0, 0.0);
    let s = optimization_stats(&small, &big);
    assert_eq!((s.nodes_before, s.nodes_after, s.nodes_removed), (1, 2, 0));

    let s = optimization_stats(
        &SdfNode::sphere(1.0).translate(0.0, 0.0, 0.0).scale(1.0),
        &small,
    );
    assert_eq!(
        s.to_string(),
        "Optimization: 3 → 1 nodes (2 removed, 66.7% reduction)"
    );
}

#[test]
fn medium_preset_bounds_shapes_of_its_size_class() {
    let cfg = TightAabbConfig::preset_medium();
    assert_eq!(cfg.initial_half_size(), 100.0);
    assert_eq!(cfg.bisection_iterations(), 22);
    assert_eq!(cfg.coarse_subdivisions(), 12);
    // one bisection step of the initial half size
    let step = 2.0 * cfg.initial_half_size() / 2f32.powi(cfg.bisection_iterations() as i32);
    let tol = step.max(1e-3) * 4.0;

    let mut compared = 0;
    for r in [15.0_f32, 40.0, 90.0] {
        let bb = compute_tight_aabb_with_config(&SdfNode::sphere(r), &cfg);
        assert!(
            bb.min.cmple(Vec3::splat(-r)).all() && bb.max.cmpge(Vec3::splat(r)).all(),
            "r={r} {bb:?}"
        );
        assert!(
            (bb.min + Vec3::splat(r)).abs().max_element() <= tol,
            "r={r} {bb:?}"
        );
        assert!(
            (bb.max - Vec3::splat(r)).abs().max_element() <= tol,
            "r={r} {bb:?}"
        );
        compared += 1;
    }
    let half = Vec3::new(30.0, 12.5, 60.0);
    let bb = compute_tight_aabb_with_config(&SdfNode::box3d(60.0, 25.0, 120.0), &cfg);
    assert!(
        bb.min.cmple(-half).all() && bb.max.cmpge(half).all(),
        "{bb:?}"
    );
    assert!(
        (bb.max - half).abs().max_element() <= tol && (bb.min + half).abs().max_element() <= tol
    );
    compared += 1;
    assert_eq!(compared, 4);
}
