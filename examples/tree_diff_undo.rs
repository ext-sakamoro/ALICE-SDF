//! Undo / redo of SDF edits with structural diff and patch.
//!
//! An edit is recorded as the patch between the scene before and after it.
//! Undo applies the inverted patch, and two edits can be squashed into one
//! patch with `merge_patches`. Every step is checked against the scene it
//! should produce and against the closed-form distance of that scene.
//!
//! # Running
//! ```bash
//! cargo run --example tree_diff_undo
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::diff::{DiffOp, TreePath};
use alice_sdf::prelude::*;

fn same(a: &SdfNode, b: &SdfNode) -> bool {
    format!("{a:?}") == format!("{b:?}")
}

fn describe(patch: &TreePatch) {
    for op in &patch.ops {
        match op {
            DiffOp::Replace { path, .. } => println!("    Replace at {path:?}"),
            DiffOp::Insert { path, .. } => println!("    Insert at {path:?}"),
            DiffOp::Delete { path, .. } => println!("    Delete at {path:?}"),
        }
    }
}

fn main() {
    println!("ALICE-SDF — tree diff / undo");
    println!("============================");

    let v0 = SdfNode::sphere(1.0).union(SdfNode::sphere(0.5).translate(3.0, 0.0, 0.0));
    // Edit 1: grow the small sphere.
    let v1 = SdfNode::sphere(1.0).union(SdfNode::sphere(0.75).translate(3.0, 0.0, 0.0));
    // Edit 2: move it up.
    let v2 = SdfNode::sphere(1.0).union(SdfNode::sphere(0.75).translate(3.0, 1.0, 0.0));

    let e1 = tree_diff(&v0, &v1);
    let e2 = tree_diff(&v1, &v2);
    println!(
        "edit 1: {} op(s), hash {:#x}",
        e1.op_count(),
        e1.content_hash
    );
    describe(&e1);
    println!("edit 2: {} op(s)", e2.op_count());
    describe(&e2);

    // Edit 1 touches only the small sphere: root → b → child.
    let expected_path: TreePath = vec![1, 0];
    assert!(matches!(&e1.ops[..], [DiffOp::Replace { path, .. }] if *path == expected_path));

    // Redo / undo.
    let redo = apply_patch(&v0, &e1).expect("edit applies");
    assert!(same(&redo, &v1));
    let undo = invert_patch(&v0, &e1).expect("invert");
    assert!(same(&apply_patch(&redo, &undo).unwrap(), &v0));

    // Squash both edits.
    let squashed = merge_patches(&v0, &e1, &e2).expect("merge");
    let end = apply_patch(&v0, &squashed).unwrap();
    assert!(same(&end, &v2));
    assert_eq!(tree_hash(&end), tree_hash(&v2));

    // The final scene is min(|p| − 1, |p − (3,1,0)| − 0.75).
    let p = Vec3::new(3.0, 2.0, 0.0);
    let expected = (p.length() - 1.0).min((p - Vec3::new(3.0, 1.0, 0.0)).length() - 0.75);
    let d = eval(&end, p);
    println!("after squash: d({p}) = {d} (closed form {expected})");
    assert!((d - expected).abs() < 1e-5);

    // No-op diff, and a patch refused by a tree it was not made for.
    assert!(tree_diff(&v2, &v2).is_empty());
    match apply_patch(&v2, &e1) {
        Err(e @ DiffError::HashMismatch { .. }) => println!("stale patch refused: {e}"),
        other => panic!("expected a hash mismatch, got {other:?}"),
    }

    println!("all checks passed");
}
