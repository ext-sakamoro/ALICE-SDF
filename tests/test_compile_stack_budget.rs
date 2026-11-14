//! Stack budget of the bytecode compiler `CompiledSdf::compile`
//!
//! The compiler walks the tree recursively (and so do the checks it runs first: the validation,
//! the evaluation stack depths and the Lipschitz bound), one call per tree level. In an
//! unoptimised build every arm of a `match` gets its own stack slots, so a recursive `match` over
//! all node kinds grows its frame with each new kind (the compiler's reached about 17.5 KiB per
//! level, so a chain of about 110 levels no longer fitted the 2 MiB thread a test runs on).
//!
//! - `the_folded_smooth_union_chain_compiles_on_a_test_thread`: a rack of 24 parts folded into a
//!   smooth union one at a time (the chain a downstream generator builds) compiles on a 2 MiB
//!   thread to the same bytecode as on a large thread, and that bytecode gives the tree
//!   evaluator's value at every probe
//! - `each_level_stays_within_the_frame_budget`: a chain `CANARY_DEPTH` deep compiles on a thread
//!   of `CANARY_BASE_BYTES + CANARY_DEPTH × COMPILE_FRAME_BUDGET_BYTES`; a level over the budget
//!   runs past the end of that stack (the process stops with a stack overflow, the test run fails)
//! - `the_leftmost_refusal_is_reported`: with two refused nodes the error names the left one, also
//!   below that deep chain on the same thread
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::{eval_compiled, CompileError, CompiledSdf};
use alice_sdf::{eval, SdfNode};
use glam::{Quat, Vec3};
use std::sync::Arc;

/// Bytes of stack one tree level of the compile path may use in an unoptimised build
///
/// Measured from the frames the compiler gives the recursive functions of `CompiledSdf::compile`
/// (their prologue's stack adjustment plus the saved return address and frame pointer; they run
/// one after another, so the largest decides); the commit that set this value lists the
/// measurement per target. The budget is that measurement with headroom for other targets and
/// toolchains, far below the 17.5 KiB per level the compiler's recursive `match` reached.
const COMPILE_FRAME_BUDGET_BYTES: usize = 512;

/// Depth of the canary chain: deep enough that the levels, not the thread start-up and the
/// non-recursive work, decide the stack used
#[cfg(not(target_arch = "wasm32"))]
const CANARY_DEPTH: usize = 2048;

/// The stack a wasm module runs on: the rust-lld default (1 MiB of linear memory), which the
/// browser background thread that evaluates scenes uses; the wasm test build sets it explicitly
/// (`-C link-arg=-zstack-size=1048576`, `scripts/stack_budget.sh`) so a change of the default
/// does not move the test
#[cfg(target_arch = "wasm32")]
const WASM_STACK_BYTES: usize = 1024 * 1024;

/// On wasm32 there are no threads: the canary runs on the module's stack, as deep as the budget
/// allows in it (the shadow stack is at the start of linear memory, so running past it traps)
#[cfg(target_arch = "wasm32")]
const CANARY_DEPTH: usize = (WASM_STACK_BYTES - CANARY_BASE_BYTES) / COMPILE_FRAME_BUDGET_BYTES;

/// Stack for everything that is not a tree level: the thread start-up, the test closure, and the
/// non-recursive calls (allocation, emitting an instruction)
const CANARY_BASE_BYTES: usize = 128 * 1024;

/// The stack `cargo test` gives each test thread by default
const TEST_THREAD_BYTES: usize = 2 * 1024 * 1024;

/// Stack for the reference compilation and the tree evaluator's values
const REFERENCE_THREAD_BYTES: usize = 256 * 1024 * 1024;

/// Parts of the folded chain (two smooth unions per part)
const FOLDED_PARTS: u32 = 24;

/// A rack of `parts` hooks on a backplate, each hook folded into the result as two smooth unions,
/// two holes subtracted, turned upright: depth `2 · parts + 5`
fn folded_rack(parts: u32) -> SdfNode {
    let (hook_width, height, hook_depth, wall, opening) = (30.0_f32, 150.0_f32, 25.0, 5.0, 15.0);
    let pitch = hook_width + 10.0;
    let outer_hx = f32::midpoint(parts as f32 * pitch, 10.0);
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

/// Run `f` on a thread with `stack` bytes
#[cfg(not(target_arch = "wasm32"))]
fn on_thread<T: Send + 'static>(stack: usize, f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(f)
        .expect("spawn a thread")
        .join()
        .expect("the thread finished")
}

/// wasm32-wasip1 has no threads: run `f` on the module's stack (`WASM_STACK_BYTES`, the size the
/// test build links with; the canary depth is chosen for it)
#[cfg(target_arch = "wasm32")]
fn on_thread<T>(_stack: usize, f: impl FnOnce() -> T) -> T {
    f()
}

/// The bytecode as text: every instruction field and every aux float (Debug prints an `f32` so
/// that it reads back to the same bits)
fn bytecode(compiled: &CompiledSdf) -> (Vec<String>, Vec<u32>) {
    (
        compiled
            .instructions()
            .iter()
            .map(|i| format!("{i:?}"))
            .collect(),
        compiled.aux_data().iter().map(|f| f.to_bits()).collect(),
    )
}

/// Compile `tree` on a thread with `stack` bytes and on a large thread; check both give the same
/// bytecode and that it evaluates to the tree evaluator's value at every point (at least one
/// inside and one outside)
fn assert_compiles_within(tree: SdfNode, points: Vec<Vec3>, stack: usize) {
    let tree = Arc::new(tree);
    let t = Arc::clone(&tree);
    let small = on_thread(stack, move || bytecode(&CompiledSdf::compile(&t)));
    let (t, p) = (Arc::clone(&tree), points);
    let (reference, values) = on_thread(REFERENCE_THREAD_BYTES, move || {
        let compiled = CompiledSdf::compile(&t);
        let values: Vec<(f32, f32)> = p
            .iter()
            .map(|&q| (eval_compiled(&compiled, q), eval(&t, q)))
            .collect();
        (bytecode(&compiled), values)
    });
    assert!(!small.0.is_empty(), "no instructions");
    assert_eq!(
        small, reference,
        "bytecode on {stack} bytes vs a large thread"
    );

    let (mut inside, mut outside) = (0, 0);
    for (compiled, tree) in values {
        assert!(compiled.is_finite(), "bytecode gave {compiled}");
        assert_eq!(
            compiled.to_bits(),
            tree.to_bits(),
            "bytecode {compiled} vs eval {tree}"
        );
        if compiled < 0.0 {
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

#[test]
fn the_folded_smooth_union_chain_compiles_on_a_test_thread() {
    let tree = folded_rack(FOLDED_PARTS);
    assert_eq!(depth(&tree), 2 * FOLDED_PARTS as usize + 5);
    let extent = f32::midpoint(FOLDED_PARTS as f32 * 40.0, 10.0);
    assert_compiles_within(tree, probe_points(extent), TEST_THREAD_BYTES);
}

#[test]
fn each_level_stays_within_the_frame_budget() {
    let tree = sphere_chain(CANARY_DEPTH);
    assert_eq!(depth(&tree), CANARY_DEPTH + 2);
    let stack = CANARY_BASE_BYTES + CANARY_DEPTH * COMPILE_FRAME_BUDGET_BYTES;
    // the chain's first sphere is the deepest leaf: probe it and along the chain
    let extent = CANARY_DEPTH as f32 * 1.5;
    let points: Vec<Vec3> = (0..=8)
        .map(|i| Vec3::new(extent * i as f32 / 8.0, 0.0, 0.0))
        .chain([Vec3::new(0.0, 5.0, 0.0), Vec3::new(-10.0, 0.0, 0.0)])
        .collect();
    assert_compiles_within(tree, points, stack);
}

/// A node the bytecode refuses: its variant name in the error
fn refused(name: &str) -> SdfNode {
    match name {
        "Triangle" => SdfNode::triangle(
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.6, 0.2),
        ),
        _ => SdfNode::bezier(
            Vec3::new(-0.6, 0.0, 0.0),
            Vec3::new(0.0, 0.8, 0.0),
            Vec3::new(0.6, 0.0, 0.0),
            0.1,
        ),
    }
}

/// The validation reports the first refused node depth first, left before right (as it did
/// before its recursion was split), also below a chain `CANARY_DEPTH` deep on the budget thread
#[test]
fn the_leftmost_refusal_is_reported() {
    let stack = CANARY_BASE_BYTES + CANARY_DEPTH * COMPILE_FRAME_BUDGET_BYTES;
    for (left, right) in [("Triangle", "Bezier"), ("Bezier", "Triangle")] {
        let shallow = refused(left).union(refused(right));
        // the left refusal at the bottom of a deep left spine, the right one at the top
        let mut deep = refused(left);
        for i in 1..=CANARY_DEPTH {
            deep = deep.smooth_union(
                SdfNode::sphere(1.0).translate(i as f32 * 1.5, 0.0, 0.0),
                0.5,
            );
        }
        let deep = deep.union(refused(right));
        for tree in [shallow, deep] {
            let got = on_thread(stack, move || match CompiledSdf::try_compile(&tree) {
                Err(CompileError::UnsupportedPrimitive(name)) => name,
                other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
            });
            assert_eq!(got, left, "refusal reported for {left} ∪ {right}");
        }
    }
}
