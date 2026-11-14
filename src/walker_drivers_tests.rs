//! Each tree walker has two drivers that apply the same per-node code: native recursion for the
//! first levels and a work-stack loop past them. These tests run every walker with the loop alone
//! (`native_levels = 0`), recursion alone (`u32::MAX`) and the shipped limit, over the shared
//! corpus, the variants it leaves out, random trees and chains deeper than the limit, and require
//! the results to be identical (bounds and distances bitwise, bytecode field by field). Nothing
//! compared fails.
//!
//! Author: Moroya Sakamoto

// ALLOW-DEAD: the shared test corpus file, of which this test calls only corpus()
#[allow(dead_code, unused_imports, clippy::all, clippy::pedantic)]
#[path = "../tests/common/corpus.rs"]
mod corpus;

use crate::compiled::{
    compile_with_native_levels, depths_rec, COMPILE_NATIVE_LEVELS, DEPTHS_NATIVE_LEVELS,
};
use crate::eval::{eval_rec, EVAL_NATIVE_LEVELS};
use crate::interval::{lipschitz_rec, LIPSCHITZ_NATIVE_LEVELS};
use crate::SdfNode;
use glam::{Quat, Vec3};

/// A deterministic random tree: binary operations, transforms and modifiers over leaves, `depth`
/// levels at most (the generator leans left or right per tree, so both spines get deep)
fn random_tree(seed: &mut u64, depth: u32) -> SdfNode {
    let mut next = |n: u64| {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) % n
    };
    fn build(next: &mut dyn FnMut(u64) -> u64, depth: u32) -> SdfNode {
        let f = |v: u64| v as f32 * 0.1 + 0.2;
        if depth == 0 {
            return match next(3) {
                0 => SdfNode::sphere(f(next(10))),
                1 => SdfNode::box3d(f(next(10)), f(next(10)), f(next(10))),
                _ => SdfNode::rounded_box(f(next(10)), f(next(10)), f(next(10)), 0.05),
            };
        }
        // the main child carries the depth; the side child is a shallow subtree
        let side = (depth / 8).min(6);
        match next(9) {
            0 => build(next, depth - 1).union(build(next, side)),
            1 => build(next, side).smooth_union(build(next, depth - 1), 0.3),
            2 => build(next, depth - 1).subtract(build(next, side)),
            3 => build(next, depth - 1).chamfer_union(build(next, side), 0.1),
            4 => build(next, side).morph(build(next, depth - 1), 0.3),
            5 => build(next, depth - 1).translate(f(next(10)), -f(next(10)), 0.5),
            6 => build(next, depth - 1).rotate(Quat::from_rotation_y(f(next(10)))),
            7 => build(next, depth - 1).scale(1.0 + f(next(5))),
            _ => build(next, depth - 1).round(0.05),
        }
    }
    build(&mut next, depth)
}

/// Depth of a tree built from the generator's node kinds (a leaf is 1; other kinds count 1)
fn tree_depth(node: &SdfNode) -> u32 {
    match node {
        SdfNode::Union { a, b }
        | SdfNode::Subtraction { a, b }
        | SdfNode::SmoothUnion { a, b, .. }
        | SdfNode::ChamferUnion { a, b, .. }
        | SdfNode::Morph { a, b, .. } => 1 + tree_depth(a).max(tree_depth(b)),
        SdfNode::Translate { child, .. }
        | SdfNode::Rotate { child, .. }
        | SdfNode::Scale { child, .. }
        | SdfNode::Round { child, .. } => 1 + tree_depth(child),
        _ => 1,
    }
}

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

fn trees() -> Vec<(String, SdfNode)> {
    let mut v: Vec<(String, SdfNode)> = corpus::corpus()
        .into_iter()
        .map(|(n, t)| (n.to_owned(), t))
        .collect();
    v.push((
        "triangle".into(),
        SdfNode::triangle(
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.6, 0.2),
        ),
    ));
    v.push((
        "twist_of_box".into(),
        SdfNode::box3d(0.4, 0.8, 0.3).twist(1.5),
    ));
    let mut seed = 0x2961_u64;
    for i in 0..40 {
        v.push((format!("random_{i}"), random_tree(&mut seed, 40 + 8 * i)));
    }
    v.push(("chain_300".into(), sphere_chain(300)));
    v
}

/// Large enough for recursion alone on every tree here
const THREAD_BYTES: usize = 512 * 1024 * 1024;

#[test]
fn every_walker_agrees_between_its_drivers() {
    let (compared, deep) = std::thread::Builder::new()
        .stack_size(THREAD_BYTES)
        .spawn(|| {
            let points = [
                Vec3::ZERO,
                Vec3::new(0.3, -0.2, 0.1),
                Vec3::new(2.0, 1.0, -1.5),
                Vec3::new(-5.0, 4.0, 3.0),
            ];
            let (mut compared, mut deep) = (0usize, 0usize);
            for (name, t) in trees() {
                let limits = [0, u32::MAX];
                let l: Vec<u32> = limits
                    .iter()
                    .map(|&k| lipschitz_rec(&t, 0, k).to_bits())
                    .collect();
                assert_eq!(l[0], l[1], "{name}: lipschitz, loop vs recursion");
                assert_eq!(
                    lipschitz_rec(&t, 0, LIPSCHITZ_NATIVE_LEVELS).to_bits(),
                    l[1],
                    "{name}: lipschitz, shipped limit"
                );
                let d0 = depths_rec(&t, 0, 0);
                assert_eq!(d0, depths_rec(&t, 0, u32::MAX), "{name}: stack depths");
                assert_eq!(
                    d0,
                    depths_rec(&t, 0, DEPTHS_NATIVE_LEVELS),
                    "{name}: stack depths, shipped limit"
                );
                let c: Vec<String> = [0, u32::MAX, COMPILE_NATIVE_LEVELS]
                    .iter()
                    .map(|&k| {
                        let (inst, aux, n, refused) = compile_with_native_levels(&t, k);
                        let aux: Vec<u32> = aux.iter().map(|f| f.to_bits()).collect();
                        format!("{inst:?}|{aux:?}|{n}|{refused:?}")
                    })
                    .collect();
                assert_eq!(c[0], c[1], "{name}: bytecode, loop vs recursion");
                assert_eq!(c[2], c[1], "{name}: bytecode, shipped limit");
                for &p in &points {
                    let e0 = eval_rec(&t, p, 0, 0).to_bits();
                    assert_eq!(
                        e0,
                        eval_rec(&t, p, 0, u32::MAX).to_bits(),
                        "{name} at {p}: eval"
                    );
                    assert_eq!(
                        e0,
                        eval_rec(&t, p, 0, EVAL_NATIVE_LEVELS).to_bits(),
                        "{name} at {p}: eval, shipped limit"
                    );
                }
                if tree_depth(&t) > EVAL_NATIVE_LEVELS.max(COMPILE_NATIVE_LEVELS) {
                    deep += 1;
                }
                compared += 1;
            }
            (compared, deep)
        })
        .expect("spawn")
        .join()
        .expect("the thread finished");
    assert!(compared > 200, "{compared} trees compared");
    // the handoff from recursion to the loop is exercised: trees deeper than the shipped limit
    assert!(deep > 10, "{deep} trees deeper than the native levels");
}

/// Past its native levels each walker hands the rest of the tree to its loop: a chain far deeper
/// than the limit runs on a thread with room for only the native levels (512 B each, the frame
/// budget) on top of a base, so a driver that kept recursing would run past the end of that stack
/// (the process stops with a stack overflow, the test run fails)
#[test]
fn past_the_native_levels_each_walker_hands_off_to_its_loop() {
    const BASE_BYTES: usize = 128 * 1024;
    const FRAME_BUDGET_BYTES: usize = 512;
    const DEPTH: usize = 2048;
    let tight = |levels: u32| BASE_BYTES + levels as usize * FRAME_BUDGET_BYTES;
    let run = |levels: u32, f: fn(&SdfNode) -> u64| -> u64 {
        let tree = sphere_chain(DEPTH);
        std::thread::Builder::new()
            .stack_size(tight(levels))
            .spawn(move || {
                let r = f(&tree);
                // dropping a deep tree recurses: keep it out of the measured stack
                std::mem::forget(tree);
                r
            })
            .expect("spawn")
            .join()
            .expect("the walker finished within the native levels' stack")
    };
    let e = run(EVAL_NATIVE_LEVELS, |t| {
        u64::from(eval_rec(t, Vec3::ZERO, 0, EVAL_NATIVE_LEVELS).to_bits())
    });
    let l = run(LIPSCHITZ_NATIVE_LEVELS, |t| {
        u64::from(lipschitz_rec(t, 0, LIPSCHITZ_NATIVE_LEVELS).to_bits())
    });
    let d = run(DEPTHS_NATIVE_LEVELS, |t| {
        depths_rec(t, 0, DEPTHS_NATIVE_LEVELS).0 as u64
    });
    let c = run(COMPILE_NATIVE_LEVELS, |t| {
        compile_with_native_levels(t, COMPILE_NATIVE_LEVELS).0.len() as u64
    });
    // the walks finished with real results (not vacuous)
    assert!(
        f32::from_bits(e as u32) < 0.0,
        "eval at the deepest sphere's centre: {}",
        f32::from_bits(e as u32)
    );
    assert!(
        f32::from_bits(l as u32) >= 1.0,
        "lipschitz bound {}",
        f32::from_bits(l as u32)
    );
    assert!(d >= 2, "value depth {d}");
    assert!(c as usize > 3 * DEPTH, "{c} instructions");
}
