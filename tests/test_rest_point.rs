//! Rest point (`eval_rest_point`, `WgslShader::transpile_with_rest_point`)
//!
//! The rest point is the point in the frame of the tree's root at which a colour or texture law
//! defined on the unposed model reads its value: the pose nodes (the grid warp, the rigid pose)
//! are undone, the modelling transforms (translate / rotate / scale, which place a part built in
//! its own frame) are not: the rest point below one is mapped back through it.
//!
//! oracle: closed forms independent of the implementation
//! - a translated, rotated and scaled sphere (modelling only): the rest point of p is p
//! - a rigid pose (R, t): the rest point of p is R⁻¹ (p − t), and its distance is the distance of
//!   the same child rotated by R and translated by t
//! - a grid warp whose every node holds the rotation R maps p (inside the box) to R p, also when
//!   the part under it is placed by a modelling transform (the elbow tree: warp → transform → part)
//! - a union picks the nearer child's rest point (a warped part beside a rigid pose copy)
//! - a node that moves points another way (twist) has no rest point: None on the CPU, Err for the shader
//! - the shader's `sdf_eval_rest` gives the CPU rest point (GPU, when an adapter is present;
//!   `ALICE_SDF_REQUIRE_GPU=1` makes a missing adapter a failure)

#[cfg(feature = "gpu")]
#[path = "common/gpu_batches.rs"]
mod gpu_batches;

#[cfg(feature = "gpu")]
use alice_sdf::compiled::{TranspileMode, WgslShader};
use alice_sdf::eval::{eval, eval_rest_point};
use alice_sdf::prelude::*;
use alice_sdf::transforms::{GridWarp, GridWarpData};
use glam::Quat;

fn rot_grid(a: f32) -> GridWarp {
    let (s, c) = alice_det_math::sin_cos(a);
    let m = [c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0];
    GridWarp::new(GridWarpData {
        origin: [-0.1; 3],
        cell: 0.05,
        dims: [5; 3],
        maps: m.repeat(125),
        reach: 0.05,
        floor: Vec::new(),
    })
    .expect("valid grid")
}

fn lattice(n: usize, lo: f32, hi: f32) -> Vec<Vec3> {
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let t = Vec3::new(i as f32 + 0.31, j as f32 + 0.57, k as f32 + 0.13) / n as f32;
                out.push(Vec3::splat(lo) + t * (hi - lo));
            }
        }
    }
    out
}

#[test]
fn modelling_transforms_are_not_undone() {
    let q = Quat::from_rotation_y(0.6) * Quat::from_rotation_x(-0.3);
    let t = Vec3::new(0.1, -0.2, 0.05);
    let node = SdfNode::sphere(0.3)
        .scale(1.5)
        .rotate(q)
        .translate(t.x, t.y, t.z);
    let mut n = 0;
    for p in lattice(6, -0.5, 0.5) {
        let got = eval_rest_point(&node, p).expect("followed");
        assert!((got - p).length() < 2e-6, "p {p}: {got}");
        n += 1;
    }
    assert_eq!(n, 216);
}

#[test]
fn a_rigid_pose_is_undone_and_measures_as_the_moved_child() {
    let q = Quat::from_rotation_y(0.6) * Quat::from_rotation_x(-0.3);
    let t = Vec3::new(0.1, -0.2, 0.05);
    let child = SdfNode::box3d(0.2, 0.1, 0.3);
    let posed = child.clone().rigid_pose(q, t);
    let moved = child.rotate(q).translate(t.x, t.y, t.z);
    let mut n = 0;
    for p in lattice(6, -0.5, 0.5) {
        let want = q.inverse() * (p - t);
        let got = eval_rest_point(&posed, p).expect("followed");
        assert!((got - want).length() < 1e-6, "p {p}: {got} vs {want}");
        assert_eq!(
            eval(&posed, p).to_bits(),
            eval(&moved, p).to_bits(),
            "p {p}"
        );
        n += 1;
    }
    assert_eq!(n, 216);
}

#[test]
fn a_part_placed_under_a_warp_reads_the_warped_point() {
    // the elbow tree: grid warp -> modelling transform -> a part built in its own frame
    let a = 0.7f32;
    let place = Quat::from_rotation_z(1.1);
    let node = SdfNode::capsule(Vec3::new(0.0, -0.04, 0.0), Vec3::new(0.0, 0.04, 0.0), 0.02)
        .rotate(place)
        .translate(0.01, -0.02, 0.0)
        .grid_warp(rot_grid(a));
    let (s, c) = alice_det_math::sin_cos(a);
    let mut n = 0;
    for p in lattice(6, -0.09, 0.09) {
        let want = Vec3::new(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
        let got = eval_rest_point(&node, p).expect("followed");
        assert!((got - want).length() < 2e-6, "p {p}: {got} vs {want}");
        n += 1;
    }
    assert_eq!(n, 216);
}

#[test]
fn a_rotation_grid_maps_to_the_rotated_point() {
    let a = 0.7f32;
    let node = SdfNode::sphere(0.05).grid_warp(rot_grid(a));
    let (s, c) = alice_det_math::sin_cos(a);
    let mut n = 0;
    for p in lattice(6, -0.09, 0.09) {
        let want = Vec3::new(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
        let got = eval_rest_point(&node, p).expect("followed");
        assert!((got - want).length() < 2e-6, "p {p}: {got} vs {want}");
        n += 1;
    }
    assert_eq!(n, 216);
}

#[test]
fn a_union_takes_the_nearer_childs_rest_point() {
    // a warped part beside a rigid pose copy of a part (the knee's fill)
    let q = Quat::from_rotation_x(0.35);
    let t = Vec3::new(0.3, 0.0, 0.0);
    let warped = SdfNode::sphere(0.05).grid_warp(rot_grid(0.7));
    let copy = SdfNode::sphere(0.05)
        .translate(0.0, 0.02, 0.0)
        .rigid_pose(q, t);
    let node = warped.union(copy);
    let (s, c) = alice_det_math::sin_cos(0.7);
    let p = Vec3::new(0.02, 0.01, -0.03);
    let got = eval_rest_point(&node, p).expect("followed");
    let want = Vec3::new(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
    assert!((got - want).length() < 2e-6, "warped side {got} vs {want}");
    let p = Vec3::new(0.31, 0.03, 0.01);
    let got = eval_rest_point(&node, p).expect("followed");
    let want = q.inverse() * (p - t);
    assert!((got - want).length() < 1e-6, "copy side {got} vs {want}");
}

#[test]
fn a_pose_over_another_shape_has_no_rest_point() {
    // only the rigid pose form (a translate over a rotate over the part) is a pose; any other
    // shape under a Pose is refused rather than answered in another frame
    let q = Quat::from_rotation_y(0.6);
    let t = Vec3::new(0.1, -0.2, 0.05);
    let part = || SdfNode::box3d(0.2, 0.1, 0.3);
    let p = Vec3::new(0.25, 0.05, -0.1);
    for (name, node) in [
        ("translate only", part().translate_vec(t).pose()),
        ("rotate only", part().rotate(q).pose()),
        (
            "translate under a rotate",
            part().translate_vec(t).rotate(q).pose(),
        ),
        (
            "material between",
            part().rotate(q).with_material(3).translate_vec(t).pose(),
        ),
        ("a bare part", part().pose()),
    ] {
        assert!(eval_rest_point(&node, p).is_none(), "{name}");
        #[cfg(feature = "gpu")]
        assert!(
            WgslShader::transpile_with_rest_point(&node, TranspileMode::Hardcoded).is_err(),
            "{name}"
        );
    }
    // the part under the rotate keeps its own placement (mapped back)
    let placed = part().rotate(Quat::from_rotation_x(0.3)).rigid_pose(q, t);
    let got = eval_rest_point(&placed, p).expect("rigid pose form");
    let want = q.inverse() * (p - t);
    assert!((got - want).length() < 1e-6, "{got} vs {want}");
}

#[test]
fn an_intersection_takes_the_farther_and_a_subtraction_the_first_operand() {
    let a = SdfNode::sphere(0.3).rigid_pose(Quat::from_rotation_z(0.5), Vec3::new(-0.3, 0.0, 0.0));
    let b = SdfNode::box3d(0.25, 0.25, 0.25)
        .rigid_pose(Quat::from_rotation_x(-0.4), Vec3::new(0.3, 0.1, 0.0));
    let ra = |p: Vec3| Quat::from_rotation_z(0.5).inverse() * (p - Vec3::new(-0.3, 0.0, 0.0));
    let rb = |p: Vec3| Quat::from_rotation_x(-0.4).inverse() * (p - Vec3::new(0.3, 0.1, 0.0));
    let inter = a.clone().intersection(b.clone());
    let sub = a.clone().subtract(b.clone());
    let (mut na, mut nb) = (0, 0);
    for p in lattice(8, -0.6, 0.6) {
        let (da, db) = (eval(&a, p), eval(&b, p));
        let want = if da >= db { ra(p) } else { rb(p) };
        let got = eval_rest_point(&inter, p).expect("followed");
        assert!((got - want).length() < 1e-6, "intersection p {p}");
        if da >= db {
            na += 1;
        } else {
            nb += 1;
        }
        let got = eval_rest_point(&sub, p).expect("followed");
        assert!((got - ra(p)).length() < 1e-6, "subtraction p {p}");
    }
    assert!(na > 20 && nb > 20, "both operands chosen: {na} / {nb}");
}

/// The two posed operands of the family tests and their rest points
fn two_posed() -> (
    SdfNode,
    SdfNode,
    impl Fn(Vec3) -> Vec3,
    impl Fn(Vec3) -> Vec3,
) {
    let (qa, ta) = (Quat::from_rotation_z(0.5), Vec3::new(-0.3, 0.0, 0.0));
    let (qb, tb) = (Quat::from_rotation_x(-0.4), Vec3::new(0.3, 0.1, 0.0));
    (
        SdfNode::sphere(0.3).rigid_pose(qa, ta),
        SdfNode::box3d(0.25, 0.25, 0.25).rigid_pose(qb, tb),
        move |p: Vec3| qa.inverse() * (p - ta),
        move |p: Vec3| qb.inverse() * (p - tb),
    )
}

#[test]
fn every_union_intersection_and_subtraction_variant_follows_its_rule() {
    let (a, b, ra, rb) = two_posed();
    type Op = fn(SdfNode, SdfNode) -> SdfNode;
    // (name, operation, rule): 0 = the nearer operand, 1 = the farther, 2 = the first
    let ops: [(&str, Op, u8); 18] = [
        ("union", |a, b| a.union(b), 0),
        ("smooth union", |a, b| a.smooth_union(b, 0.05), 0),
        ("chamfer union", |a, b| a.chamfer_union(b, 0.05), 0),
        ("stairs union", |a, b| a.stairs_union(b, 0.05, 3.0), 0),
        ("columns union", |a, b| a.columns_union(b, 0.05, 3.0), 0),
        ("exp smooth union", |a, b| a.exp_smooth_union(b, 0.05), 0),
        ("intersection", |a, b| a.intersection(b), 1),
        (
            "smooth intersection",
            |a, b| a.smooth_intersection(b, 0.05),
            1,
        ),
        (
            "chamfer intersection",
            |a, b| a.chamfer_intersection(b, 0.05),
            1,
        ),
        (
            "stairs intersection",
            |a, b| a.stairs_intersection(b, 0.05, 3.0),
            1,
        ),
        (
            "columns intersection",
            |a, b| a.columns_intersection(b, 0.05, 3.0),
            1,
        ),
        (
            "exp smooth intersection",
            |a, b| a.exp_smooth_intersection(b, 0.05),
            1,
        ),
        ("subtraction", |a, b| a.subtract(b), 2),
        ("smooth subtraction", |a, b| a.smooth_subtract(b, 0.05), 2),
        ("chamfer subtraction", |a, b| a.chamfer_subtract(b, 0.05), 2),
        (
            "stairs subtraction",
            |a, b| a.stairs_subtract(b, 0.05, 3.0),
            2,
        ),
        (
            "columns subtraction",
            |a, b| a.columns_subtract(b, 0.05, 3.0),
            2,
        ),
        (
            "exp smooth subtraction",
            |a, b| a.exp_smooth_subtract(b, 0.05),
            2,
        ),
    ];
    let pts = lattice(8, -0.6, 0.6);
    let mut compared = 0;
    for (name, op, rule) in ops {
        let node = op(a.clone(), b.clone());
        let (mut sa, mut sb) = (0, 0);
        #[cfg(feature = "gpu")]
        let gpu = gpu_rest(&node, &pts);
        for (i, p) in pts.iter().enumerate() {
            let (da, db) = (eval(&a, *p), eval(&b, *p));
            let pick_a = match rule {
                0 => da <= db,
                1 => da >= db,
                _ => true,
            };
            if pick_a {
                sa += 1;
            } else {
                sb += 1;
            }
            let want = if pick_a { ra(*p) } else { rb(*p) };
            let got = eval_rest_point(&node, *p).expect("followed");
            assert!(
                (got - want).length() < 1e-6,
                "{name} p {p}: {got} vs {want}"
            );
            #[cfg(feature = "gpu")]
            if let Some(g) = &gpu {
                assert!(
                    (g[i] - got).length() < 2e-6,
                    "{name} gpu p {p}: {} vs {got}",
                    g[i]
                );
            }
            let _ = i;
            compared += 1;
        }
        if rule != 2 {
            assert!(
                sa > 20 && sb > 20,
                "{name}: both operands chosen ({sa} / {sb})"
            );
        }
    }
    assert_eq!(compared, 18 * 512);
}

/// The shader's rest point at `pts` (None without an adapter unless `ALICE_SDF_REQUIRE_GPU` is set)
#[cfg(feature = "gpu")]
fn gpu_rest(node: &SdfNode, pts: &[Vec3]) -> Option<Vec<Vec3>> {
    use alice_sdf::compiled::GpuEvaluator;
    let sh =
        WgslShader::transpile_with_rest_point(node, TranspileMode::Hardcoded).expect("followed");
    let mut out = vec![Vec3::ZERO; pts.len()];
    for axis in 0..3 {
        let mut s = sh.clone();
        s.source = s.source.replace("fn sdf_eval(", "fn sdf_eval_d(");
        s.source.push_str(&format!(
            "\nfn sdf_eval(p: vec3<f32>) -> f32 {{ return sdf_eval_rest(p)[{axis}]; }}\n"
        ));
        let g = match GpuEvaluator::from_shader(&s) {
            Ok(g) => g,
            Err(e) => {
                assert!(
                    std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                    "no GPU: {e}"
                );
                return None;
            }
        };
        for (o, v) in out
            .iter_mut()
            .zip(gpu_batches::eval_in_batches(&g, pts).expect("gpu eval"))
        {
            o[axis] = v;
        }
    }
    Some(out)
}

#[test]
fn a_twist_has_no_rest_point() {
    let node = SdfNode::box3d(0.2, 0.4, 0.2).twist(2.0);
    assert!(eval_rest_point(&node, Vec3::new(0.05, 0.1, 0.0)).is_none());
    #[cfg(feature = "gpu")]
    assert!(WgslShader::transpile_with_rest_point(&node, TranspileMode::Hardcoded).is_err());
}

#[cfg(feature = "gpu")]
#[test]
fn the_shader_rest_point_is_the_cpu_rest_point() {
    use alice_sdf::compiled::GpuEvaluator;
    // a union of a rotated grid-warped sphere and a translated, rotated sphere: every case above
    let node = SdfNode::sphere(0.05)
        .rotate(Quat::from_rotation_z(0.9))
        .translate(0.01, 0.0, 0.0)
        .grid_warp(rot_grid(0.7))
        .union(
            SdfNode::sphere(0.08)
                .rotate(Quat::from_rotation_z(0.4))
                .rigid_pose(Quat::from_rotation_x(0.3), Vec3::new(0.2, 0.0, 0.0)),
        )
        .union(SdfNode::sphere(0.04).scale(1.2).translate(0.0, 0.25, 0.0))
        .union(
            SdfNode::sphere(0.06)
                .rigid_pose(Quat::from_rotation_y(0.7), Vec3::new(-0.05, 0.1, 0.0))
                .intersection(SdfNode::box3d(0.05, 0.05, 0.05).translate(-0.04, 0.1, 0.0)),
        )
        .union(
            SdfNode::box3d(0.06, 0.03, 0.06)
                .rigid_pose(Quat::from_rotation_x(0.2), Vec3::new(0.1, 0.15, 0.0))
                .subtract(SdfNode::sphere(0.03).translate(0.1, 0.15, 0.0)),
        );
    let sh =
        WgslShader::transpile_with_rest_point(&node, TranspileMode::Hardcoded).expect("followed");
    assert_eq!(sh.source.matches("fn sdf_eval_rest(").count(), 1);
    let pts = lattice(12, -0.12, 0.3);
    let mut worst = 0.0f32;
    let mut compared = 0;
    for axis in 0..3 {
        // read one component per run through the evaluator's f32 output
        let mut s = sh.clone();
        s.source = s.source.replace("fn sdf_eval(", "fn sdf_eval_d(");
        s.source.push_str(&format!(
            "\nfn sdf_eval(p: vec3<f32>) -> f32 {{ return sdf_eval_rest(p)[{axis}]; }}\n"
        ));
        let g = match GpuEvaluator::from_shader(&s) {
            Ok(g) => g,
            Err(e) => {
                assert!(
                    std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                    "no GPU: {e}"
                );
                eprintln!("skipped: no GPU ({e})");
                return;
            }
        };
        let got = gpu_batches::eval_in_batches(&g, &pts).expect("gpu eval");
        for (p, v) in pts.iter().zip(&got) {
            let want = eval_rest_point(&node, *p).expect("followed")[axis];
            worst = worst.max((want - v).abs());
            compared += 1;
        }
    }
    assert_eq!(compared, 3 * 1728);
    assert!(worst < 2e-6, "max |gpu - cpu| {worst}");
}
