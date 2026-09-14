//! `SdfNode` の drop が tree の深さに依らず native stack を消費しないことの検証
//!
//! LOL stdlib の product は `subtract` を 2,400 段 nest する (正当な形) 以前は
//! `Arc<SdfNode>` の再帰 drop がこの深さで 2 MB thread を使い切っていた
//! ここでは意図的に小さい stack (256 KB) の thread で 100,000 段を構築 →
//! drop し、overflow しないことを確認する
//!
//! Author: Moroya Sakamoto

use alice_sdf::SdfNode;
use std::sync::Arc;

const DEPTH: usize = 100_000;
const SMALL_STACK: usize = 256 * 1024;

/// `union` と `translate` を交互に積んだ深さ `depth` の chain
///
/// 構築自体は loop なので stack を使わない
fn deep_chain(depth: usize) -> SdfNode {
    let mut node = SdfNode::sphere(1.0);
    for i in 0..depth {
        node = if i % 2 == 0 {
            SdfNode::sphere(0.5).union(node)
        } else {
            node.translate(0.0, 0.1, 0.0)
        };
    }
    node
}

/// `subtract` を左に nest した chain (LOL stdlib product と同じ形)
fn deep_subtract(depth: usize) -> SdfNode {
    let mut node = SdfNode::sphere(10.0);
    for _ in 0..depth {
        node = node.subtract(SdfNode::sphere(0.1));
    }
    node
}

fn run_in_small_stack<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::Builder::new()
        .stack_size(SMALL_STACK)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("thread panicked (stack overflow aborts the whole process instead)");
}

#[test]
fn deep_union_translate_chain_drops_in_small_stack() {
    // `node_count()` は再帰実装のままなので、深い tree ではここで呼ばない
    // (形の検証は `shared_subtree_survives_partial_drop` の depth 1,000 で行う)
    run_in_small_stack(|| {
        let node = deep_chain(DEPTH);
        drop(node);
    });
}

#[test]
fn deep_subtract_chain_drops_in_small_stack() {
    run_in_small_stack(|| {
        let node = deep_subtract(DEPTH);
        drop(node);
    });
}

#[test]
fn deep_chain_inside_arc_drops_in_small_stack() {
    // 呼び出し側が `Arc<SdfNode>` で持っていても (SdfTree / CompiledSdf 等) 同じ
    run_in_small_stack(|| {
        let node = Arc::new(deep_chain(DEPTH));
        drop(node);
    });
}

#[test]
fn shared_subtree_survives_partial_drop() {
    // 共有 subtree: 2 つの親から `Arc::clone` で参照される
    let shared = Arc::new(deep_chain(1_000));
    let parent_a = SdfNode::Union {
        a: Arc::clone(&shared),
        b: Arc::new(SdfNode::sphere(1.0)),
    };
    let parent_b = SdfNode::Translate {
        child: Arc::clone(&shared),
        offset: glam::Vec3::ZERO,
    };
    assert_eq!(Arc::strong_count(&shared), 3);

    drop(parent_a);
    assert_eq!(
        Arc::strong_count(&shared),
        2,
        "shared subtree must not be freed"
    );
    // 共有部が生きている = 中身に触れる
    assert_eq!(shared.node_count(), 1_001 + 500);

    drop(parent_b);
    assert_eq!(Arc::strong_count(&shared), 1);
    assert_eq!(shared.node_count(), 1_001 + 500);
}

#[test]
fn clone_then_independent_drop() {
    let original = deep_chain(2_000);
    let cloned = original.clone();
    let expected = original.node_count();
    drop(original);
    assert_eq!(cloned.node_count(), expected);
    run_in_small_stack(move || drop(cloned));
}

#[test]
fn take_children_covers_every_family() {
    // 子を持つ variant の代表 1 つずつを深く nest しても drop できる =
    // `take_children_into` の arm から漏れていない (漏れると再帰 drop に戻り
    // 256 KB stack で overflow する)
    run_in_small_stack(|| {
        let mut node = SdfNode::sphere(1.0);
        for i in 0..20_000 {
            node = match i % 8 {
                0 => SdfNode::sphere(0.5).union(node),
                1 => node.subtract(SdfNode::sphere(0.1)),
                2 => SdfNode::sphere(0.5).smooth_union(node, 0.1),
                3 => node.translate(0.0, 0.1, 0.0),
                4 => node.scale(1.0),
                5 => node.round(0.01),
                6 => node.twist(0.1),
                _ => node.onion(0.05),
            };
        }
        drop(node);
    });
}
