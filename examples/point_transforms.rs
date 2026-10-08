//! Point transforms — the transform functions and the container types
//!
//! Shows how the SDF-side transforms (`transform_translate`, `transform_scale`,
//! `transform_rotate`, `transform_rotate_euler`) map a world point into the
//! child's frame and how the `*_inverse` forms map it back, then uses the
//! container helpers (`Aabb`, `Ray::at`, `SdfTree::with_metadata`) and the
//! node builders `translate_vec` / `sine_displacement_aniso`. Every value is
//! checked against its closed form; the full oracle is
//! `tests/test_point_transform_oracle.rs`.
//!
//! # Running
//! ```bash
//! cargo run --example point_transforms
//! ```
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "example scene and animation values; the crate output these examples show is computed by the library"
)]

use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use alice_sdf::transforms::{
    transform_rotate, transform_rotate_euler, transform_rotate_inverse, transform_scale,
    transform_scale_inverse, transform_translate, transform_translate_inverse,
};
use glam::{Quat, Vec3};

fn check(name: &str, got: Vec3, want: Vec3) {
    println!("  {name:<28} {got:>26}  (closed form {want})");
    assert!(
        (got - want).abs().max_element() <= 1e-5 * (1.0 + want.abs().max_element()),
        "{name}: {got} vs {want}"
    );
}

fn main() {
    println!("ALICE-SDF — point transforms");
    println!("============================");
    let p = Vec3::new(1.0, 2.0, 3.0);
    println!("p = {p}");

    let o = Vec3::new(0.5, -1.0, 2.0);
    let local = transform_translate(p, o);
    check("transform_translate", local, p - o);
    check(
        "transform_translate_inverse",
        transform_translate_inverse(local, o),
        p,
    );

    let (scaled, mult) = transform_scale(p, 2.0);
    check("transform_scale", scaled, p * 0.5);
    println!("  distance multiplier          {mult}");
    assert_eq!(mult, 2.0);
    check(
        "transform_scale_inverse",
        transform_scale_inverse(scaled, 2.0),
        p,
    );

    // a quarter turn about +Z maps +X to +Y; the SDF transform applies the inverse
    let q = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    check(
        "transform_rotate",
        transform_rotate(p, q),
        Vec3::new(2.0, -1.0, 3.0),
    );
    check(
        "transform_rotate_inverse",
        transform_rotate_inverse(p, q),
        Vec3::new(-2.0, 1.0, 3.0),
    );
    // Euler (0, 0, 90 deg) is the same rotation
    check(
        "transform_rotate_euler",
        transform_rotate_euler(p, 0.0, 0.0, std::f32::consts::FRAC_PI_2),
        Vec3::new(2.0, -1.0, 3.0),
    );

    println!();
    let bounds = Aabb::from_center_extents(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
    check("Aabb::center", bounds.center(), Vec3::ZERO);
    check("Aabb::size", bounds.size(), Vec3::new(2.0, 4.0, 6.0));
    check(
        "Aabb::half_extents",
        bounds.half_extents(),
        Vec3::new(1.0, 2.0, 3.0),
    );
    let grown = bounds.union(&Aabb::new(Vec3::splat(-0.5), Vec3::new(4.0, 0.0, 0.0)));
    check("Aabb::union max", grown.max, Vec3::new(4.0, 2.0, 3.0));
    println!("  contains {p}: {}", grown.contains(p));
    assert!(grown.contains(p) && !bounds.contains(Vec3::new(1.5, 0.0, 0.0)));

    let ray = Ray::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 2.0));
    check("Ray::at(1.5)", ray.at(1.5), Vec3::new(0.0, 0.0, 1.5));

    println!();
    let node = SdfNode::sphere(1.0)
        .translate_vec(o)
        .sine_displacement_aniso(0.05, Vec3::new(4.0, 8.0, 4.0));
    let d = eval(&node, p);
    let lp = p - o;
    let want = lp.length() - 1.0 + 0.05 * (4.0 * p.x).sin() * (8.0 * p.y).sin() * (4.0 * p.z).sin();
    println!("  displaced sphere at p        {d:.6}  (closed form {want:.6})");
    assert!((d - want).abs() < 1e-5);

    // the isotropic builder is the anisotropic one with the same frequency on every axis
    let iso = SdfNode::sphere(1.0).sine_displacement(0.05, 6.0);
    let aniso = SdfNode::sphere(1.0).sine_displacement_aniso(0.05, Vec3::splat(6.0));
    println!("  sine_displacement(0.05, 6) at p {:.6}", eval(&iso, p));
    assert_eq!(eval(&iso, p), eval(&aniso, p));

    let meta = SdfMetadata {
        name: Some("displaced sphere".into()),
        ..SdfMetadata::default()
    };
    let tree = SdfTree::with_metadata(node, meta);
    println!(
        "  tree {:?}: {} nodes, version {}",
        tree.metadata.as_ref().and_then(|m| m.name.as_deref()),
        tree.node_count(),
        tree.version
    );
    assert_eq!(tree.node_count(), 3);
    println!();
    println!("all checks passed");
}
