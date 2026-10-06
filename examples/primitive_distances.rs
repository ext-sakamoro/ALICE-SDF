//! Primitive distances — the specialised primitive functions and node constructors
//!
//! Evaluates the axis-aligned and point-defined primitive functions
//! (`sdf_capsule_vertical`, `sdf_cylinder_capped`, `sdf_plane_from_points`,
//! `sdf_torus_capped`, ...) at a few points, prints the distances, and checks
//! each one against its textbook closed form. Also builds nodes with
//! `SdfNode::box3d_half_extents`, `metric_ball` and `metric_blend` and prints
//! their category. The full closed-form oracles live in
//! `tests/test_primitive_closed_form_oracle.rs` and
//! `tests/test_metric_field_oracle.rs`.
//!
//! # Running
//! ```bash
//! cargo run --example primitive_distances
//! ```
//!
//! Author: Moroya Sakamoto

use alice_det_math::metric::MetricWeights;
use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use alice_sdf::primitives::{
    sdf_capsule_horizontal, sdf_capsule_vertical, sdf_cylinder_capped, sdf_cylinder_infinite,
    sdf_plane_from_points, sdf_plane_xy, sdf_plane_xz, sdf_plane_yz, sdf_torus_capped,
};
use glam::Vec3;

fn check(name: &str, p: Vec3, got: f32, want: f32) {
    println!("  {name:<24} at {p:>22}: {got:>9.5}  (closed form {want:>9.5})");
    assert!(
        (got - want).abs() <= 1e-5 * (1.0 + want.abs()),
        "{name} at {p}: {got} vs {want}"
    );
}

fn main() {
    println!("ALICE-SDF — primitive distances vs closed forms");
    println!("===============================================");

    let p = Vec3::new(0.3, 2.0, 0.4);
    // vertical capsule, half height 1, radius 0.25: nearest axis point is (0, 1, 0)
    check(
        "sdf_capsule_vertical",
        p,
        sdf_capsule_vertical(p, 1.0, 0.25),
        (p - Vec3::Y).length() - 0.25,
    );
    // horizontal capsule along X: |p.x| < half length, so the distance is radial
    check(
        "sdf_capsule_horizontal",
        p,
        sdf_capsule_horizontal(p, 1.0, 0.25),
        Vec3::new(0.0, p.y, p.z).length() - 0.25,
    );
    // capped cylinder from (0,-1,0) to (0,1,0), radius 0.5: above the top cap
    // and inside the radius, so the distance is axial
    check(
        "sdf_cylinder_capped",
        p,
        sdf_cylinder_capped(p, -Vec3::Y, Vec3::Y, 0.5),
        p.y - 1.0,
    );
    check(
        "sdf_cylinder_infinite",
        p,
        sdf_cylinder_infinite(p, 0.5),
        (p.x * p.x + p.z * p.z).sqrt() - 0.5,
    );
    check("sdf_plane_xy", p, sdf_plane_xy(p), p.z);
    check("sdf_plane_xz", p, sdf_plane_xz(p), p.y);
    check("sdf_plane_yz", p, sdf_plane_yz(p), p.x);
    // the plane through (0,0,1), (1,0,1), (0,1,1) is z = 1 with normal +Z
    check(
        "sdf_plane_from_points",
        p,
        sdf_plane_from_points(
            p,
            Vec3::Z,
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
        ),
        p.z - 1.0,
    );
    // capped torus (major 1, minor 0.2, half angle 60 deg around +Z):
    // (0, 0, 1) is on the tube centre line, (0, 0, -1) is beyond the cut,
    // nearest to the arc end (sin 60, 0, cos 60) after mirroring x
    let a = std::f32::consts::FRAC_PI_3;
    let on_arc = Vec3::Z;
    check(
        "sdf_torus_capped",
        on_arc,
        sdf_torus_capped(on_arc, 1.0, 0.2, a),
        -0.2,
    );
    let behind = -Vec3::Z;
    check(
        "sdf_torus_capped",
        behind,
        sdf_torus_capped(behind, 1.0, 0.2, a),
        (behind - Vec3::new(a.sin(), 0.0, a.cos())).length() - 0.2,
    );

    println!();
    println!("Nodes");
    let boxed = SdfNode::box3d_half_extents(0.5, 1.0, 1.5);
    check("box3d_half_extents", p, eval(&boxed, p), {
        let q = p.abs() - Vec3::new(0.5, 1.0, 1.5);
        q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
    });
    // the L-infinity metric ball is the cube of half side `radius`
    let cube = SdfNode::metric_ball(1.0, MetricWeights::LINF);
    let q = Vec3::new(2.0, 0.5, 0.0);
    check("metric_ball (L-inf)", q, eval(&cube, q), 1.0);
    // far outside the blend bubble the result is the outer field bit for bit
    let outer = SdfNode::sphere(1.0);
    let blend = SdfNode::metric_blend(cube.clone(), outer.clone(), Vec3::ZERO, 0.5, 0.25);
    let far = Vec3::new(3.0, 0.0, 0.0);
    let (got, want) = (eval(&blend, far), eval(&outer, far));
    println!("  metric_blend outside the bubble: {got} == outer {want}");
    assert_eq!(got, want);

    for node in [&boxed, &cube, &blend] {
        println!("  category of {:?}: {:?}", node_name(node), node.category());
    }
    assert_eq!(boxed.category(), SdfCategory::Primitive);
    println!();
    println!("all checks passed");
}

fn node_name(node: &SdfNode) -> String {
    let s = format!("{node:?}");
    s.split([' ', '{', '(']).next().unwrap_or("").to_string()
}
