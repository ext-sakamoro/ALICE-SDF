//! Sampling a field on a grid, bounding it, and simplifying its tree.
//!
//! - `eval_batch` / `eval_grid` / `eval_grid_with_normals` sample a scene;
//!   `grid_index` / `grid_coords` convert between flat and 3D indices
//! - `eval::gradient` estimates the (unnormalised) gradient
//! - `Interval` predicates classify the interval value of a box
//! - `optimize` + `optimization_stats` remove identity nodes
//! - `TightAabbConfig::preset_medium` bounds a part of 20-200 units
//!
//! Each step checks its numbers against the closed form of the shape.
//!
//! Run: `cargo run --example grid_bounds_optimize`
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::parallel::{grid_coords, grid_index};
use alice_sdf::eval::{eval, eval_batch, eval_grid, eval_grid_with_normals, gradient};
use alice_sdf::interval::{eval_interval, Interval, Vec3Interval};
use alice_sdf::optimize::{optimization_stats, optimize};
use alice_sdf::tight_aabb::{compute_tight_aabb_with_config, TightAabbConfig};
use alice_sdf::types::SdfNode;
use glam::Vec3;

fn main() {
    let r = 1.0_f32;
    let ball = SdfNode::sphere(r);

    // ── grid sampling ──
    let res = 9;
    let (min, max) = (Vec3::splat(-2.0), Vec3::splat(2.0));
    let grid = eval_grid(&ball, min, max, res);
    let (dists, normals) = eval_grid_with_normals(&ball, min, max, res, 1e-3);
    let inside = grid.iter().filter(|d| **d < 0.0).count();
    println!("grid {res}³: {} samples, {inside} inside", grid.len());
    // step 0.5: inside points are lattice points with |p| < 1 → 0 and the 6 axis neighbours at 0.5 + 12 at 0.707 + 8 at 0.866
    assert_eq!(inside, 1 + 6 + 12 + 8);
    assert_eq!(dists, grid);
    let i = grid_index(8, 4, 4, res); // (2, 0, 0)
    assert_eq!(grid_coords(i, res), (8, 4, 4));
    println!("sample (2,0,0): d = {}, n = {:?}", grid[i], normals[i]);
    assert!((grid[i] - 1.0).abs() < 1e-6);
    assert!((normals[i] - Vec3::X).length() < 1e-3);

    let pts = [Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.5, 0.0, 0.0)];
    let batch = eval_batch(&ball, &pts);
    println!("batch: {batch:?}");
    assert_eq!(batch, vec![2.0, -0.5]);

    let g = gradient(&ball, Vec3::new(0.0, 0.0, 1.5), 1e-3);
    println!("gradient at (0,0,1.5): {g:?}");
    assert!((g - Vec3::Z).length() < 1e-3);

    // ── interval classification of boxes ──
    let classify = |lo: Vec3, hi: Vec3| {
        let iv = eval_interval(&ball, Vec3Interval::from_bounds(lo, hi));
        let kind = if iv.is_positive() {
            "outside"
        } else if iv.is_negative() {
            "inside"
        } else {
            assert!(iv.contains(0.0));
            "crosses the surface"
        };
        println!("box {lo:?}..{hi:?}: [{:.4}, {:.4}] {kind}", iv.lo, iv.hi);
        (iv, kind)
    };
    let (far, k1) = classify(Vec3::splat(1.0), Vec3::splat(1.5));
    let (core, k2) = classify(Vec3::splat(-0.25), Vec3::splat(0.25));
    let (edge, k3) = classify(Vec3::new(0.5, -0.25, -0.25), Vec3::new(1.5, 0.25, 0.25));
    assert_eq!((k1, k2, k3), ("outside", "inside", "crosses the surface"));
    assert!(!far.overlaps(core) && !edge.overlaps(far) && edge.overlaps(Interval::ZERO));
    let common = edge.intersect(Interval::point(0.0).hull(Interval::point(0.25)));
    println!("edge ∩ [0, 0.25] = [{}, {}]", common.lo, common.hi);
    assert_eq!((common.lo, common.hi), (0.0, 0.25));

    // ── tree optimisation ──
    let messy = SdfNode::sphere(r)
        .translate(0.0, 0.0, 0.0)
        .scale(2.0)
        .scale(0.5)
        .round(0.0)
        .union(
            SdfNode::box3d(1.0, 1.0, 1.0)
                .translate(1.0, 0.0, 0.0)
                .translate(0.5, 0.0, 0.0),
        );
    let clean = optimize(&messy);
    let stats = optimization_stats(&messy, &clean);
    println!("{stats}");
    assert_eq!(
        (stats.nodes_before, stats.nodes_after, stats.nodes_removed),
        (9, 4, 5)
    );
    for p in [
        Vec3::ZERO,
        Vec3::new(1.5, 0.5, -0.25),
        Vec3::new(-2.0, 1.0, 0.5),
    ] {
        assert_eq!(eval(&clean, p).to_bits(), eval(&messy, p).to_bits());
    }

    // ── tight bounds for a mid-size part ──
    let part = SdfNode::box3d(60.0, 25.0, 120.0);
    let cfg = TightAabbConfig::preset_medium();
    let bb = compute_tight_aabb_with_config(&part, &cfg);
    println!("medium preset AABB: {:?} .. {:?}", bb.min, bb.max);
    let half = Vec3::new(30.0, 12.5, 60.0);
    assert!(bb.min.cmple(-half).all() && bb.max.cmpge(half).all());
    assert!((bb.max - half).max_element() < 0.01 && (-half - bb.min).max_element() < 0.01);
}
