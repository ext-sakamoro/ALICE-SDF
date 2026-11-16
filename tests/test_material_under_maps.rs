//! `eval_material` under point maps and `MetricBlend`
//!
//! The scene is two translated balls with materials 3 and 7 under a union. The
//! material of a union is the material of its nearer operand, so at a point `q`
//! the expected id is the id of the ball whose closed form `|q − c| − r` (f64)
//! is smaller.
//!
//! * `ProjectiveTransform` / `LatticeDeform` / `SdfSkinning` move the point
//!   before their child reads it (`eval` reads the child at the mapped point),
//!   so the material is the scene's material at the mapped point. The mapped
//!   point comes from the public map functions of `alice_sdf::transforms`, the
//!   same laws `eval` applies.
//! * `MetricBlend` interpolates its two operands; the material follows the rule
//!   the crate already uses for its interpolating node `Morph` and the unions:
//!   the nearer operand.
//!
//! Points closer than 1e-3 to a tie are skipped.
//!
//! Author: Moroya Sakamoto

use alice_sdf::prelude::*;
use alice_sdf::transforms::{lattice, projective, skinning};
use glam::Mat4;

#[derive(Clone, Copy)]
struct Ball {
    r: f32,
    c: [f32; 3],
    id: u32,
}

impl Ball {
    fn node(self) -> SdfNode {
        SdfNode::sphere(self.r)
            .translate(self.c[0], self.c[1], self.c[2])
            .with_material(self.id)
    }
    fn dist(self, p: Vec3) -> f64 {
        let d = [
            f64::from(p.x) - f64::from(self.c[0]),
            f64::from(p.y) - f64::from(self.c[1]),
            f64::from(p.z) - f64::from(self.c[2]),
        ];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() - f64::from(self.r)
    }
}

const LEFT: Ball = Ball {
    r: 0.5,
    c: [-0.75, 0.0, 0.125],
    id: 3,
};
const RIGHT: Ball = Ball {
    r: 0.75,
    c: [0.75, 0.25, 0.0],
    id: 7,
};

fn scene() -> SdfNode {
    LEFT.node().union(RIGHT.node())
}

/// The id of the nearer ball at `q`, or `None` within 1e-3 of a tie
fn nearer(q: Vec3) -> Option<u32> {
    let (l, r) = (LEFT.dist(q), RIGHT.dist(q));
    if (l - r).abs() <= 1e-3 {
        None
    } else if l < r {
        Some(LEFT.id)
    } else {
        Some(RIGHT.id)
    }
}

/// A 9³ grid over [-1.8, 1.8]³ (inside the lattice box below)
fn grid() -> Vec<Vec3> {
    let mut out = Vec::new();
    for i in 0..9 {
        for j in 0..9 {
            for k in 0..9 {
                let f = |n: i32| -1.8 + 0.45 * n as f32;
                out.push(Vec3::new(f(i), f(j), f(k)));
            }
        }
    }
    out
}

/// Checks `node`'s material against the nearer ball at `map(p)`; both ids must be seen
fn check(name: &str, node: &SdfNode, map: impl Fn(Vec3) -> Vec3) {
    let mut seen = [0usize; 2];
    for p in grid() {
        let Some(want) = nearer(map(p)) else { continue };
        assert_eq!(eval_material(node, p), want, "{name}: material at {p:?}");
        seen[usize::from(want == RIGHT.id)] += 1;
    }
    assert!(seen[0] > 20 && seen[1] > 20, "{name}: ids seen {seen:?}");
}

#[test]
fn projective_transform_reads_the_material_at_the_mapped_point() {
    // column-major: scale x by 0.8, shift by (0.3, -0.2, 0.1), a small perspective term
    let inv = [
        0.8, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.05, //
        0.3, -0.2, 0.1, 1.0,
    ];
    let node = scene().projective_transform(inv, 2.0);
    check("projective", &node, |p| {
        projective::projective_transform(p, &inv).0
    });
}

#[test]
fn lattice_deform_reads_the_material_at_the_mapped_point() {
    let (lo, hi) = (Vec3::splat(-2.0), Vec3::splat(2.0));
    let mut cps = Vec::new();
    for k in 0..2 {
        for j in 0..2 {
            for i in 0..2 {
                let c = Vec3::new(
                    if i == 0 { -2.0 } else { 2.0 },
                    if j == 0 { -2.0 } else { 2.0 },
                    if k == 0 { -2.0 } else { 2.0 },
                );
                // pull the +x face in so the map is not the identity
                cps.push(if i == 1 {
                    c - Vec3::new(0.6, 0.0, 0.0)
                } else {
                    c
                });
            }
        }
    }
    let node = scene().lattice_deform(cps.clone(), 2, 2, 2, lo, hi);
    check("lattice", &node, |p| {
        lattice::lattice_deform(p, &cps, 2, 2, 2, lo, hi).0
    });
}

#[test]
fn sdf_skinning_reads_the_material_at_the_mapped_point() {
    let ident = Mat4::IDENTITY.to_cols_array();
    let bones = vec![skinning::BoneTransform {
        inv_bind_pose: ident,
        current_pose: Mat4::from_translation(Vec3::new(0.6, -0.3, 0.2)).to_cols_array(),
        weight: 1.0,
    }];
    let node = scene().sdf_skinning(bones.clone());
    check("skinning", &node, |p| skinning::sdf_skinning(p, &bones).0);
}

#[test]
fn metric_blend_takes_the_material_of_the_nearer_operand() {
    // the bubble covers part of the grid, so inside, across and outside the skin are all read
    let node = SdfNode::metric_blend(
        LEFT.node(),
        RIGHT.node(),
        Vec3::new(0.2, 0.0, 0.0),
        0.8,
        0.6,
    );
    check("metric_blend", &node, |p| p);
}

#[test]
fn material_of_a_deep_chain_in_a_small_stack() {
    // 100,000 unions deep: the material walk is a loop (no native recursion) and evaluates each
    // node once (the old lookup evaluated both operands at every level, O(n²))
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let mut node = SdfNode::sphere(0.5).with_material(3);
            for i in 0..100_000 {
                node = node.union(SdfNode::sphere(0.1).translate(10.0 + i as f32, 0.0, 0.0));
            }
            node = node.union(
                SdfNode::sphere(0.2)
                    .translate(0.0, 5.0, 0.0)
                    .with_material(7),
            );
            // nearest: the material-3 ball at the origin, the material-7 ball at y = 5,
            // an untagged ball (material 0) at x = 10
            assert_eq!(eval_material(&node, Vec3::new(0.0, 0.6, 0.0)), 3);
            assert_eq!(eval_material(&node, Vec3::new(0.0, 4.9, 0.0)), 7);
            assert_eq!(eval_material(&node, Vec3::new(10.0, 0.0, 0.05)), 0);
            // the chain is dropped with it, flat (`take_children_into`)
        })
        .expect("spawn")
        .join()
        .expect("thread panicked (a stack overflow aborts the whole process instead)");
}
