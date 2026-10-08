//! Domain modifiers — single-axis point maps, IFS folding, fBm and sweeps
//!
//! Applies the single-axis forms of mirror / twist / bend / repeat, polar
//! repetition, an IFS fold, fBm noise, the simplex-noise displacement and the
//! Bezier sweep to a sample point. Each result is printed and checked against
//! the closed-form point map. The full oracle is
//! `tests/test_domain_modifier_oracle.rs`.
//!
//! # Running
//! ```bash
//! cargo run --example domain_modifiers
//! ```
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "example scene and animation values; the crate output these examples show is computed by the library"
)]

use alice_sdf::modifiers::{
    fbm_noise_3d, ifs_fold, modifier_bend_cheap, modifier_bend_x, modifier_bend_z,
    modifier_mirror_x, modifier_mirror_y, modifier_mirror_z, modifier_noise_simplex,
    modifier_repeat_polar, modifier_repeat_x, modifier_repeat_y, modifier_repeat_z,
    modifier_twist_x, modifier_twist_z, perlin_noise_3d, sweep_bezier_dist_y,
};
use glam::Vec3;

fn check(name: &str, got: Vec3, want: Vec3) {
    println!("  {name:<22} {got:>28}  (closed form {want})");
    assert!(
        (got - want).abs().max_element() <= 1e-5 * (1.0 + want.abs().max_element()),
        "{name}: {got} vs {want}"
    );
}

fn rot(u: f32, v: f32, a: f32) -> (f32, f32) {
    let (s, c) = (a.sin(), a.cos());
    (c * u - s * v, s * u + c * v)
}

fn main() {
    println!("ALICE-SDF — domain modifiers");
    println!("============================");
    let p = Vec3::new(-0.7, 0.45, -1.3);
    println!("p = {p}");

    check(
        "modifier_mirror_x",
        modifier_mirror_x(p),
        Vec3::new(0.7, 0.45, -1.3),
    );
    check("modifier_mirror_y", modifier_mirror_y(p), p);
    check(
        "modifier_mirror_z",
        modifier_mirror_z(p),
        Vec3::new(-0.7, 0.45, 1.3),
    );

    let k = 0.8;
    let (y, z) = rot(p.y, p.z, p.x * k);
    check(
        "modifier_twist_x",
        modifier_twist_x(p, k),
        Vec3::new(p.x, y, z),
    );
    let (x, y) = rot(p.x, p.y, p.z * k);
    check(
        "modifier_twist_z",
        modifier_twist_z(p, k),
        Vec3::new(x, y, p.z),
    );
    let (x, z) = rot(p.x, p.z, p.y * k);
    check(
        "modifier_bend_x",
        modifier_bend_x(p, k),
        Vec3::new(x, p.y, z),
    );
    let (y, z) = rot(p.y, p.z, p.y * k);
    check(
        "modifier_bend_z",
        modifier_bend_z(p, k),
        Vec3::new(p.x, y, z),
    );
    check(
        "modifier_bend_cheap",
        modifier_bend_cheap(p, k),
        Vec3::new(p.x + k * p.y * p.y, p.y, p.z),
    );

    // spacing 1: -0.7 -> 0.3, 0.45 stays, -1.3 -> -0.3
    check(
        "modifier_repeat_x",
        modifier_repeat_x(p, 1.0),
        Vec3::new(0.3, 0.45, -1.3),
    );
    check("modifier_repeat_y", modifier_repeat_y(p, 1.0), p);
    check(
        "modifier_repeat_z",
        modifier_repeat_z(p, 1.0),
        Vec3::new(-0.7, 0.45, -0.3),
    );

    // polar repetition into 4 sectors: the angle is folded into [-45, 45) deg around +X
    let r = (p.x * p.x + p.z * p.z).sqrt();
    let sector = std::f32::consts::FRAC_PI_2;
    let ang = p.z.atan2(p.x);
    let folded = ang - sector * (ang / sector + 0.5).floor();
    check(
        "modifier_repeat_polar",
        modifier_repeat_polar(p, 4),
        Vec3::new(r * folded.cos(), p.y, r * folded.sin()),
    );

    // IFS: one map, "translate by +1 along X"; the greedy fold applies it while
    // it brings the point closer to the origin (-0.7 -> 0.3, then stops)
    let shift: [f32; 16] = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1.,
    ];
    check(
        "ifs_fold",
        ifs_fold(p, &[shift], 3),
        Vec3::new(0.3, 0.45, -1.3),
    );

    println!();
    // fBm: (n(p) + 0.5 n(2p)) / 1.5, and zero on the integer lattice
    let fbm = fbm_noise_3d(p.x, p.y, p.z, 1, 2, 2.0, 0.5);
    let want = (perlin_noise_3d(p.x, p.y, p.z, 1)
        + 0.5 * perlin_noise_3d(2.0 * p.x, 2.0 * p.y, 2.0 * p.z, 2))
        / 1.5;
    println!("  fbm_noise_3d (2 octaves) {fbm:.6}  (definition {want:.6})");
    assert!((fbm - want).abs() < 1e-6);
    assert_eq!(fbm_noise_3d(2.0, -1.0, 3.0, 1, 4, 2.0, 0.5), 0.0);
    // simplex displacement vanishes on the lattice: the distance is unchanged
    let d = modifier_noise_simplex(0.25, Vec3::new(1.0, 2.0, -3.0), 0.1, 1.0, 9);
    println!("  modifier_noise_simplex on the lattice: {d}");
    assert_eq!(d, 0.25);

    // sweep along the straight "Bezier" (0,0) (1,0) (2,0) in XZ: the distance is |z|
    let (dist, y) = sweep_bezier_dist_y(1.2, 0.4, -0.6, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0);
    println!("  sweep_bezier_dist_y: ({dist}, {y})  (closed form (0.6, 0.4))");
    assert!((dist - 0.6).abs() < 1e-6 && y == 0.4);
    println!();
    println!("all checks passed");
}
