//! `BvhNode::build_over_boxes` shares `MeshBvh::build`'s builder: over the triangle boxes of a mesh it
//! returns node for node the triangle BVH (same split, same order, bit-identical boxes, same leaf entries).
//! Oracle: the triangle path as it stands (`MeshBvh::build`), on meshes with many triangles per level.
#![allow(clippy::unwrap_used)]

use alice_sdf::mesh::bvh::{BvhNode, MeshBvh};
use alice_sdf::types::Aabb;
use glam::Vec3;

fn grid_mesh(n: usize, warp: f32) -> (Vec<Vec3>, Vec<u32>) {
    let mut v = Vec::new();
    for i in 0..=n {
        for j in 0..=n {
            let (x, y) = (i as f32 / n as f32 * 10.0, j as f32 / n as f32 * 10.0);
            // a non-planar height field (polynomial: the crate forbids f32::sin / cos even in tests)
            v.push(Vec3::new(
                x,
                y,
                0.02 * warp * x * (x - 5.0) * (x - 9.0) + 0.05 * y * (y - 6.0),
            ));
        }
    }
    let mut idx = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let (a, b, c, d) = (
                i * (n + 1) + j,
                i * (n + 1) + j + 1,
                (i + 1) * (n + 1) + j,
                (i + 1) * (n + 1) + j + 1,
            );
            idx.extend_from_slice(&[a as u32, b as u32, d as u32, a as u32, d as u32, c as u32]);
        }
    }
    (v, idx)
}

const fn bits(a: &Aabb) -> [u32; 6] {
    [
        a.min.x.to_bits(),
        a.min.y.to_bits(),
        a.min.z.to_bits(),
        a.max.x.to_bits(),
        a.max.y.to_bits(),
        a.max.z.to_bits(),
    ]
}

fn same(a: &BvhNode, b: &BvhNode, nodes: &mut usize) {
    *nodes += 1;
    assert_eq!(bits(a.aabb()), bits(b.aabb()), "node box");
    match (a, b) {
        (BvhNode::Leaf { triangles: ta, .. }, BvhNode::Leaf { triangles: tb, .. }) => {
            assert_eq!(ta, tb, "leaf entries");
        }
        (
            BvhNode::Internal {
                left: la,
                right: ra,
                ..
            },
            BvhNode::Internal {
                left: lb,
                right: rb,
                ..
            },
        ) => {
            same(la, lb, nodes);
            same(ra, rb, nodes);
        }
        _ => panic!("node kinds differ"),
    }
}

#[test]
fn a_box_bvh_over_triangle_boxes_is_the_triangle_bvh() {
    for (n, warp, leaf) in [(12usize, 0.9f32, 4usize), (31, 1.7, 8), (7, 0.0, 1)] {
        let (v, idx) = grid_mesh(n, warp);
        let tri = MeshBvh::build(&v, &idx, leaf);
        let boxes: Vec<Aabb> = tri.triangles.iter().map(|t| t.aabb).collect();
        let bx = BvhNode::build_over_boxes(&boxes, leaf).unwrap();
        let mut nodes = 0;
        same(tri.root.as_ref().unwrap(), &bx, &mut nodes);
        assert!(
            nodes > 2 * (2 * n * n) / leaf.max(1) / 2,
            "compared {nodes} nodes"
        );
    }
}

/// No boxes: no tree (not a panic, not an empty leaf). A leaf size of 0 is clamped to 1 (one box a leaf).
#[test]
fn degenerate_box_inputs() {
    assert!(BvhNode::build_over_boxes(&[], 4).is_none());
    let boxes = vec![
        Aabb::new(Vec3::ZERO, Vec3::ONE),
        Aabb::new(Vec3::splat(2.0), Vec3::splat(3.0)),
    ];
    let t = BvhNode::build_over_boxes(&boxes, 0).unwrap();
    fn leaves(n: &BvhNode, out: &mut Vec<usize>) {
        match n {
            BvhNode::Leaf { triangles, .. } => {
                assert_eq!(triangles.len(), 1);
                out.extend(triangles);
            }
            BvhNode::Internal { left, right, .. } => {
                leaves(left, out);
                leaves(right, out);
            }
        }
    }
    let mut got = Vec::new();
    leaves(&t, &mut got);
    got.sort_unstable();
    assert_eq!(got, vec![0, 1]);
}
