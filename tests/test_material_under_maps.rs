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

mod common;

use alice_sdf::eval::material::{material_rule, MaterialRule};
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

/// The pick rules against the two balls' closed form, independently of the rule table (the GPU
/// parity test cannot see a wrong rule: both sides read the same table)
#[test]
fn union_intersection_and_subtraction_pick_their_operand_as_documented() {
    /// a node of LEFT and CUT, and whether LEFT gives its material at (d_LEFT, d_CUT)
    type Case = (&'static str, SdfNode, fn(f64, f64) -> bool);
    // a ball overlapping LEFT, so the subtraction has points where it cuts
    const CUT: Ball = Ball {
        r: 0.6,
        c: [-0.2, 0.1, 0.0],
        id: 7,
    };
    let (l, r) = (LEFT.node(), CUT.node());
    let cases: [Case; 3] = [
        // the nearer operand
        ("union", l.clone().union(r.clone()), |dl, dr| dl <= dr),
        // the farther operand (the one that bounds the intersection there)
        (
            "intersection",
            l.clone().intersection(r.clone()),
            |dl, dr| dl >= dr,
        ),
        // `a` unless `b` cuts it: `b` where `dl < -dr`
        ("subtraction", l.subtract(r), |dl, dr| dl >= -dr),
    ];
    for (name, node, takes_left) in cases {
        let mut seen = [0usize; 2];
        // 21³ points over the two balls' overlap (step 0.1)
        let near = (0..21 * 21 * 21).map(|i| {
            let f = |n: usize| -1.5 + 0.1 * n as f32;
            Vec3::new(f(i % 21), f(i / 21 % 21) - 0.5, f(i / 441) - 0.5)
        });
        for p in near {
            let (dl, dr) = (LEFT.dist(p), CUT.dist(p));
            // skip ties of the rule's comparison
            if (dl - dr).abs() <= 1e-3 || (dl + dr).abs() <= 1e-3 {
                continue;
            }
            let want = if takes_left(dl, dr) { LEFT.id } else { CUT.id };
            assert_eq!(eval_material(&node, p), want, "{name} at {p:?}");
            seen[usize::from(want == CUT.id)] += 1;
        }
        assert!(seen[0] > 50 && seen[1] > 50, "{name}: ids seen {seen:?}");
    }
}

/// The operands of a two-operand node and the node rebuilt with other operands, through the
/// stored form (`a` / `b`, or `inner` / `outer`)
fn operands(node: &SdfNode) -> Option<(SdfNode, SdfNode, impl Fn(&SdfNode, &SdfNode) -> SdfNode)> {
    let v = serde_json::to_value(node).ok()?;
    let (tag, body) = v.as_object()?.iter().next()?;
    let body = body.as_object()?;
    let keys = if body.contains_key("a") {
        ("a", "b")
    } else {
        ("inner", "outer")
    };
    let a = serde_json::from_value(body.get(keys.0)?.clone()).ok()?;
    let b = serde_json::from_value(body.get(keys.1)?.clone()).ok()?;
    let (tag, body) = (tag.clone(), body.clone());
    let rebuild = move |a: &SdfNode, b: &SdfNode| {
        let mut body = body.clone();
        body.insert(keys.0.into(), serde_json::to_value(a).expect("serialise"));
        body.insert(keys.1.into(), serde_json::to_value(b).expect("serialise"));
        let mut outer = serde_json::Map::new();
        outer.insert(tag.clone(), serde_json::Value::Object(body));
        serde_json::from_value(serde_json::Value::Object(outer)).expect("deserialise")
    };
    Some((a, b, rebuild))
}

/// Where a two-operand node's value is one operand's value (and clearly not the other's), its
/// material is that operand's: the rule table read against the distance law of every two-operand
/// node of the corpus, so a family given the wrong pick fails here although the GPU and the CPU
/// still agree. Nodes whose value follows neither operand (an interpolation, `XOR`, `Pipe`) take
/// the nearer operand by convention and are left out.
#[test]
fn the_material_comes_from_the_operand_the_distance_follows() {
    const CONVENTION: [&str; 4] = ["Morph", "MetricBlend", "XOR", "Pipe"];
    const SEAM: f32 = 0.3;
    const NEAR: f32 = 0.05;
    const FAR: f32 = 0.2;
    let (mut points, mut nodes) = (0, 0);
    for (name, tree) in common::corpus::corpus() {
        if !matches!(material_rule(&tree), MaterialRule::Pick(_)) {
            continue;
        }
        let kind = format!("{tree:?}");
        if CONVENTION.iter().any(|c| kind.starts_with(c)) {
            continue;
        }
        let Some((a, b, rebuild)) = operands(&tree) else {
            panic!("{name}: no operands read from the stored form")
        };
        let node = rebuild(&a.clone().with_material(3), &b.clone().with_material(7));
        let mut here = 0;
        // 21³ points over [-2, 2]³ (step 0.2)
        let dense = (0..21 * 21 * 21).map(|i| {
            let f = |n: usize| -2.0 + 0.2 * n as f32;
            Vec3::new(f(i % 21), f(i / 21 % 21), f(i / 441))
        });
        for p in dense {
            let (da, db, d) = (eval(&a, p), eval(&b, p), eval(&node, p));
            // away from every seam (union / intersection at da = db, subtraction at da = -db) by
            // more than the corpus's blend widths, the value is one operand's (b's negated
            // under a subtraction)
            if (da - db).abs() <= SEAM || (da + db).abs() <= SEAM {
                continue;
            }
            // near one operand's value and clearly off the other's (the exponential blends
            // approach an operand without reaching it)
            let to_b = (d - db).abs().min((d + db).abs());
            let to_a = (d - da).abs();
            let want = if to_a <= NEAR && to_b > FAR {
                3
            } else if to_b <= NEAR && to_a > FAR {
                7
            } else {
                // a chamfer or a stair plane, which follows neither
                continue;
            };
            assert_eq!(
                eval_material(&node, p),
                want,
                "{name} at {p:?}: d {d} a {da} b {db}"
            );
            here += 1;
        }
        assert!(
            here >= 20,
            "{name}: only {here} points where the value follows one operand"
        );
        points += here;
        nodes += 1;
    }
    assert!(nodes >= 20, "{nodes} two-operand trees checked");
    eprintln!("{nodes} two-operand trees, {points} points");
}
