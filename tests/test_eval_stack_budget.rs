//! Stack budget of the tree evaluator `eval`
//!
//! `eval` recurses once per tree level. A generator that folds its parts one at a time into a
//! smooth union (not associative, so the tree cannot be balanced) builds a chain two levels per
//! part; a downstream generator caps such chains at 24 parts so the tree fits the 2 MiB thread a
//! test runs on in an unoptimised build. In an unoptimised build every arm of a `match` gets its
//! own stack slots, so a recursive `match` over all node kinds grows its frame with each new kind
//! (it reached about 43 KiB per level, and that chain no longer fitted 2 MiB).
//!
//! - `the_folded_smooth_union_chain_fits_a_test_thread`: that tree, on a 2 MiB thread, gives the
//!   bytecode evaluator's value (the compiled evaluator walks a flat instruction list, so it is an
//!   independent path; the two are bit-identical by `tests/test_det_parity.rs`)
//! - `each_level_stays_within_the_frame_budget`: a chain `CANARY_DEPTH` deep on a thread of
//!   `CANARY_BASE_BYTES + CANARY_DEPTH × EVAL_FRAME_BUDGET_BYTES`; a level over the budget runs past
//!   the end of that stack (the process stops with a stack overflow, the test run fails)
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::{eval_compiled, CompiledSdf};
use alice_sdf::{eval, SdfNode};
use glam::{Quat, Vec3};
use std::sync::Arc;

/// Bytes of stack one tree level of `eval` may use in an unoptimised build
///
/// Measured from the frame the compiler gives `eval` (its prologue's stack adjustment plus the
/// saved return address and frame pointer); the commit that set this value lists the measurement
/// per target. The budget is that measurement with headroom for other targets and toolchains, far
/// below the 43 KiB per level a recursive `match` over every node kind reached.
const EVAL_FRAME_BUDGET_BYTES: usize = 512;

/// Depth of the canary chain: deep enough that the levels, not the thread start-up and the leaf
/// evaluation at the bottom, decide the stack used
const CANARY_DEPTH: usize = 2048;

/// Stack for everything that is not a tree level: the thread start-up, the test closure, and the
/// non-recursive calls at the bottom of the tree (the leaf evaluator and its primitive)
const CANARY_BASE_BYTES: usize = 128 * 1024;

/// The stack `cargo test` gives each test thread by default
const TEST_THREAD_BYTES: usize = 2 * 1024 * 1024;

/// Parts of the folded chain (two smooth unions per part)
const FOLDED_PARTS: u32 = 24;

/// A rack of `parts` hooks on a backplate, each hook folded into the result as two smooth unions,
/// two holes subtracted, turned upright: depth `2 · parts + 5` (rotation, two subtractions, the
/// chain, the translate and the leaf of the first hook)
fn folded_rack(parts: u32) -> SdfNode {
    let (hook_width, height, hook_depth, wall, opening) = (30.0_f32, 150.0_f32, 25.0, 5.0, 15.0);
    let pitch = hook_width + 10.0;
    let outer_hx = (parts as f32 * pitch + 10.0) * 0.5;
    let outer_hy = height * 0.5;
    let outer_hz = wall * 0.5;
    let (arm_hy, arm_hz, tip_hy) = (4.0, hook_depth * 0.5, opening * 0.5);
    let hook_y = height * 0.3 - outer_hy;
    let x_start = -(parts as f32 - 1.0) * pitch * 0.5;

    let mut result = SdfNode::rounded_box(outer_hx, outer_hy, outer_hz, 3.0);
    for i in 0..parts {
        let x = i as f32 * pitch + x_start;
        let arm = SdfNode::rounded_box(hook_width * 0.5, arm_hy, arm_hz, 2.0).translate(
            x,
            hook_y,
            outer_hz + arm_hz,
        );
        let tip = SdfNode::rounded_box(hook_width * 0.5, tip_hy, arm_hy, 2.0).translate(
            x,
            hook_y - tip_hy - arm_hy,
            outer_hz + hook_depth,
        );
        result = result.smooth_union(arm, 2.0).smooth_union(tip, 2.0);
    }
    for side in [-1.0_f32, 1.0] {
        let hole = SdfNode::cylinder(2.25, 2.0 * (wall + 1.0)).translate(
            side * outer_hx * 0.85,
            outer_hy * 0.7,
            0.0,
        );
        result = result.subtract(hole);
    }
    result.rotate(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
}

/// A chain of `depth` smooth unions of spheres along x, folded one at a time
fn sphere_chain(depth: usize) -> SdfNode {
    let mut result = SdfNode::sphere(1.0);
    for i in 1..=depth {
        result = result.smooth_union(
            SdfNode::sphere(1.0).translate(i as f32 * 1.5, 0.0, 0.0),
            0.5,
        );
    }
    result
}

/// Depth of a tree (a leaf is 1)
fn depth(node: &SdfNode) -> usize {
    match node {
        SdfNode::SmoothUnion { a, b, .. } | SdfNode::Subtraction { a, b } => {
            1 + depth(a).max(depth(b))
        }
        SdfNode::Translate { child, .. } | SdfNode::Rotate { child, .. } => 1 + depth(child),
        _ => 1,
    }
}

/// Stack for computing the reference values: the bytecode compiler recurses once per level with a
/// larger frame than `eval`, and its budget is not what these tests hold
const REFERENCE_THREAD_BYTES: usize = 256 * 1024 * 1024;

fn on_thread<T: Send + 'static>(stack: usize, f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(f)
        .expect("spawn a thread")
        .join()
        .expect("the thread finished")
}

/// `eval` at each point on a thread with `stack` bytes, beside the bytecode evaluator's value
/// (compiled and evaluated on a thread of its own)
fn values_on_thread(tree: SdfNode, points: Vec<Vec3>, stack: usize) -> Vec<(f32, f32)> {
    let tree = Arc::new(tree);
    let (t, p) = (Arc::clone(&tree), points.clone());
    let reference: Vec<f32> = on_thread(REFERENCE_THREAD_BYTES, move || {
        let compiled = CompiledSdf::compile(&t);
        p.iter().map(|&q| eval_compiled(&compiled, q)).collect()
    });
    let (t, p) = (Arc::clone(&tree), points);
    let values: Vec<f32> = on_thread(stack, move || p.iter().map(|&q| eval(&t, q)).collect());
    values.into_iter().zip(reference).collect()
}

/// Points on and around the rack: in the backplate, at hooks, far outside
fn probe_points(extent: f32) -> Vec<Vec3> {
    let mut pts = Vec::new();
    for i in 0..=16 {
        let x = -extent + 2.0 * extent * i as f32 / 16.0;
        for (y, z) in [
            (0.0, 0.0),
            (1.0, 20.0),
            (-30.0, 10.0),
            (0.5, -0.5),
            (60.0, 90.0),
        ] {
            pts.push(Vec3::new(x, y, z));
        }
    }
    pts
}

fn assert_same_and_not_vacuous(values: &[(f32, f32)]) {
    let mut inside = 0;
    let mut outside = 0;
    for &(tree, compiled) in values {
        assert!(tree.is_finite(), "eval gave {tree}");
        assert_eq!(
            tree.to_bits(),
            compiled.to_bits(),
            "eval {tree} vs bytecode {compiled}"
        );
        if tree < 0.0 {
            inside += 1;
        } else {
            outside += 1;
        }
    }
    assert!(
        inside > 0 && outside > 0,
        "{inside} inside / {outside} outside"
    );
}

#[test]
fn the_folded_smooth_union_chain_fits_a_test_thread() {
    let tree = folded_rack(FOLDED_PARTS);
    assert_eq!(depth(&tree), 2 * FOLDED_PARTS as usize + 5);
    let extent = (FOLDED_PARTS as f32 * 40.0 + 10.0) * 0.5;
    let values = values_on_thread(tree, probe_points(extent), TEST_THREAD_BYTES);
    assert_same_and_not_vacuous(&values);
}

#[test]
fn each_level_stays_within_the_frame_budget() {
    let tree = sphere_chain(CANARY_DEPTH);
    assert_eq!(depth(&tree), CANARY_DEPTH + 2);
    let stack = CANARY_BASE_BYTES + CANARY_DEPTH * EVAL_FRAME_BUDGET_BYTES;
    // the chain's first sphere is the deepest leaf: probe it and along the chain
    let extent = CANARY_DEPTH as f32 * 1.5;
    let points: Vec<Vec3> = (0..=8)
        .map(|i| Vec3::new(extent * i as f32 / 8.0, 0.0, 0.0))
        .chain([Vec3::new(0.0, 5.0, 0.0), Vec3::new(-10.0, 0.0, 0.0)])
        .collect();
    let values = values_on_thread(tree, points, stack);
    assert_same_and_not_vacuous(&values);
}
