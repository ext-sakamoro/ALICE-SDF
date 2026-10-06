//! Tree diff / patch oracle.
//!
//! The diff is checked by what it promises, not by re-running it:
//! - round trip: `apply(diff(a, b), a)` is structurally `b` (compared through
//!   `Debug`, which the diff code does not produce) and evaluates to `b`'s
//!   closed-form distance
//! - undo: `apply(invert(a, p), apply(p, a))` is `a`
//! - merge: `apply(merge(a, p1, p2), a)` is `apply(p2, apply(p1, a))`
//! - a single parameter change is one `Replace` at that node's path
//! - a patch applied to a tree it was not made for is rejected
//!
//! Author: Moroya Sakamoto

use alice_sdf::diff::{DiffOp, TreePath};
use alice_sdf::prelude::*;
use std::sync::Arc;

fn dbg(n: &SdfNode) -> String {
    format!("{n:?}")
}

/// Scenes with a closed-form distance at a few points.
fn scenes() -> Vec<SdfNode> {
    vec![
        SdfNode::sphere(1.0),
        SdfNode::sphere(2.0),
        SdfNode::sphere(1.0).translate(1.0, 0.0, 0.0),
        SdfNode::sphere(1.0).union(SdfNode::sphere(0.5).translate(3.0, 0.0, 0.0)),
        SdfNode::sphere(1.5).union(SdfNode::sphere(0.5).translate(3.0, 0.0, 0.0)),
        SdfNode::sphere(1.0).union(SdfNode::sphere(0.5).translate(0.0, 3.0, 0.0)),
        SdfNode::sphere(1.0).subtract(SdfNode::sphere(0.5)),
        SdfNode::box3d(1.0, 2.0, 3.0).translate(0.0, 0.0, -1.0),
    ]
}

#[test]
fn round_trip_reproduces_the_target_tree() {
    let s = scenes();
    let mut compared = 0;
    for a in &s {
        for b in &s {
            let patch = tree_diff(a, b);
            let out = apply_patch(a, &patch).expect("patch applies to its source");
            assert_eq!(dbg(&out), dbg(b), "diff({a:?} → {b:?})");
            if dbg(a) == dbg(b) {
                assert!(patch.is_empty() && patch.op_count() == 0);
            } else {
                assert!(!patch.is_empty() && patch.op_count() > 0);
            }
            compared += 1;
        }
    }
    assert_eq!(compared, s.len() * s.len());
}

#[test]
fn round_trip_evaluates_to_the_closed_form() {
    // a: unit sphere at origin; b: radius-2 sphere at (1, 0, 0)
    let a = SdfNode::sphere(1.0);
    let b = SdfNode::sphere(2.0).translate(1.0, 0.0, 0.0);
    let out = apply_patch(&a, &tree_diff(&a, &b)).unwrap();
    let mut compared = 0;
    for &p in &[
        Vec3::ZERO,
        Vec3::new(4.0, 0.0, 0.0),
        Vec3::new(0.0, 3.0, 0.0),
        Vec3::new(-2.0, 1.0, 1.0),
    ] {
        let expected = (p - Vec3::X).length() - 2.0;
        assert!((eval(&out, p) - expected).abs() < 1e-5);
        compared += 1;
    }
    assert!(compared > 0);
}

#[test]
fn one_parameter_change_is_one_replace_at_its_path() {
    let a = SdfNode::sphere(1.0).union(SdfNode::sphere(0.5).translate(3.0, 0.0, 0.0));
    // Change only the radius of the translated sphere: root → b (1) → child (0).
    let b = SdfNode::sphere(1.0).union(SdfNode::sphere(0.75).translate(3.0, 0.0, 0.0));
    let patch = tree_diff(&a, &b);
    assert_eq!(patch.op_count(), 1);
    match &patch.ops[0] {
        DiffOp::Replace { path, new_node, .. } => {
            let expected: TreePath = vec![1, 0];
            assert_eq!(path, &expected);
            assert_eq!(dbg(new_node), dbg(&SdfNode::sphere(0.75)));
        }
        other => panic!("expected Replace, got {other:?}"),
    }
}

#[test]
fn invert_undoes_and_merge_composes() {
    let s = scenes();
    let mut compared = 0;
    for a in &s {
        for b in &s {
            let p = tree_diff(a, b);
            let inv = invert_patch(a, &p).unwrap();
            let b_tree = apply_patch(a, &p).unwrap();
            assert_eq!(dbg(&apply_patch(&b_tree, &inv).unwrap()), dbg(a));
            for c in &s {
                let p2 = tree_diff(&b_tree, c);
                let merged = merge_patches(a, &p, &p2).unwrap();
                assert_eq!(dbg(&apply_patch(a, &merged).unwrap()), dbg(c));
                compared += 1;
            }
        }
    }
    assert!(compared > 0);
}

#[test]
fn tree_hash_separates_structure_and_parameters() {
    let s = scenes();
    let mut compared = 0;
    for (i, a) in s.iter().enumerate() {
        assert_eq!(tree_hash(a), tree_hash(&a.clone()));
        for b in &s[i + 1..] {
            assert_ne!(tree_hash(a), tree_hash(b), "{a:?} vs {b:?}");
            compared += 1;
        }
    }
    assert!(compared > 0);
}

#[test]
fn patch_for_another_tree_is_rejected() {
    let a = SdfNode::sphere(1.0).union(SdfNode::sphere(0.5));
    let b = SdfNode::sphere(1.0).union(SdfNode::sphere(0.25));
    let other = SdfNode::sphere(1.0).union(SdfNode::sphere(0.3));
    let patch = tree_diff(&a, &b);
    match apply_patch(&other, &patch) {
        Err(DiffError::HashMismatch { path, .. }) => {
            // The mismatch is at the node the op targets: root → b (1).
            assert_eq!(path, vec![1]);
        }
        r => panic!("expected HashMismatch, got {r:?}"),
    }

    // A path below a leaf does not exist.
    let bad = TreePatch {
        ops: vec![DiffOp::Replace {
            path: vec![0, 0],
            old_hash: 0,
            new_node: Arc::new(SdfNode::sphere(2.0)),
        }],
        content_hash: 0,
    };
    assert!(matches!(
        apply_patch(&a, &bad),
        Err(DiffError::InvalidPath(_))
    ));
}
