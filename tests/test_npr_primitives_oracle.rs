//! NPR scalar primitives vs closed forms (`npr::NprInput`, `npr::outline`,
//! `npr::noise` frequency and lattice).
//!
//! - `NprInput::n_dot_l` / `n_dot_v` are the unclamped dot products of the
//!   stored unit vectors, compared with a f64 sum of products.
//! - `distance_field_outline(sdf, w)` is the indicator of `|sdf| < w`; on a
//!   sphere of radius `r` the outline is the shell `| |p| - r | < w`, whose
//!   radii are known in closed form. `depth_step_outline(g, t)` is the
//!   indicator of `|g| > max(t, 0)` (strict).
//! - Noise: `with_frequency(k)` scales the domain, so the field at `p` with
//!   frequency `k` is the unit-frequency field at `k p` (bit for bit, the
//!   product is the same `f32` operation). Perlin noise is 0 at every lattice
//!   point (all gradient contributions vanish or are weighted by fade(0) = 0)
//!   and simplex noise is 0 at every simplex vertex (the neighbouring
//!   vertices are farther than the kernel radius √0.6); both remap 0 to 0.5.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]

use alice_sdf::eval::eval;
use alice_sdf::npr::noise::{NoiseField, PerlinNoise, SimplexNoise, WorleyNoise};
use alice_sdf::npr::outline::{depth_step_outline, distance_field_outline};
use alice_sdf::npr::NprInput;
use alice_sdf::types::SdfNode;
use glam::Vec3;

#[test]
fn npr_input_dots_are_the_unclamped_dot_products() {
    let mut compared = 0;
    for k in 0..32 {
        let a = k as f32 * 0.37;
        let n = Vec3::new(a.cos(), a.sin(), 0.0);
        let v = Vec3::new(0.0, (a * 1.3).cos(), (a * 1.3).sin());
        let l = Vec3::new((a * 0.7).sin(), 0.0, (a * 0.7).cos());
        let p = Vec3::new(k as f32, -1.0, 2.0);
        let input = NprInput::new(0.25 - k as f32 * 0.01, n, v, l, p);
        assert_eq!(input.sdf, 0.25 - k as f32 * 0.01);
        assert_eq!(
            (input.normal, input.view, input.light, input.position),
            (n, v, l, p)
        );
        let dot = |x: Vec3, y: Vec3| {
            f64::from(x.x) * f64::from(y.x)
                + f64::from(x.y) * f64::from(y.y)
                + f64::from(x.z) * f64::from(y.z)
        };
        assert!(
            (f64::from(input.n_dot_l()) - dot(n, l)).abs() < 1e-6,
            "k={k}"
        );
        assert!(
            (f64::from(input.n_dot_v()) - dot(n, v)).abs() < 1e-6,
            "k={k}"
        );
        compared += 1;
    }
    // Back-facing light: the product is negative, not clamped to 0.
    let back = NprInput::new(0.0, Vec3::Z, Vec3::Z, -Vec3::Z, Vec3::ZERO);
    assert_eq!(back.n_dot_l(), -1.0);
    assert_eq!(compared, 32);
}

#[test]
fn distance_outline_on_a_sphere_is_the_closed_form_shell() {
    let r = 1.25f32;
    let w = 0.1f32;
    let sphere = SdfNode::sphere(r);
    let mut inside = 0;
    let mut outside = 0;
    for k in 0..400 {
        // Radii from 0 to 2.5 that avoid the shell boundary by >= 1e-3.
        let rho = k as f32 * 2.5 / 400.0 + 0.003;
        let dist = (rho - r).abs();
        if (dist - w).abs() < 1e-3 {
            continue;
        }
        let dir = Vec3::new(0.48, -0.6, 0.64); // unit
        let p = dir * rho;
        let want = if dist < w { 1.0 } else { 0.0 };
        assert_eq!(
            distance_field_outline(eval(&sphere, p), w),
            want,
            "rho = {rho}"
        );
        if want == 1.0 {
            inside += 1;
        } else {
            outside += 1;
        }
    }
    // The shell (r - w, r + w) is 0.2 wide out of 2.5: 32 of 400 samples.
    assert!(
        (30..=33).contains(&inside),
        "{inside} samples on the outline"
    );
    assert!(outside > 300);
    // Negative width is clamped to 0: nothing is on the outline.
    assert_eq!(distance_field_outline(0.0, -1.0), 0.0);
}

#[test]
fn depth_step_outline_is_a_strict_threshold_on_the_magnitude() {
    for (g, t, want) in [
        (0.3, 0.2, 1.0),
        (-0.3, 0.2, 1.0),
        (0.2, 0.2, 0.0),
        (0.1, 0.2, 0.0),
        (0.0, 0.0, 0.0),
        (1e-9, 0.0, 1.0),
        (1e-9, -5.0, 1.0),
        (0.0, -5.0, 0.0),
    ] {
        assert_eq!(depth_step_outline(g, t), want, "g={g} t={t}");
    }
}

fn sample_points() -> Vec<Vec3> {
    (0..64)
        .map(|k| {
            let a = k as f32 * 0.731;
            Vec3::new(a.sin() * 3.1, (a * 1.7).cos() * 2.3, a * 0.41 - 5.0)
        })
        .collect()
}

#[test]
fn with_frequency_scales_the_domain() {
    let mut compared = 0;
    let mut differs = 0;
    for k in [0.5f32, 2.0, 3.75] {
        let perlin = PerlinNoise::new(7).with_frequency(k);
        let simplex = SimplexNoise::new(7).with_frequency(k);
        let worley = WorleyNoise::new(7).with_frequency(k);
        assert_eq!(perlin.frequency, k);
        assert_eq!(simplex.frequency, k);
        assert_eq!(worley.frequency, k);
        for p in sample_points() {
            let q = p * k;
            for (got, want, unit) in [
                (
                    perlin.sample_scalar(p),
                    PerlinNoise::new(7).sample_scalar(q),
                    PerlinNoise::new(7).sample_scalar(p),
                ),
                (
                    simplex.sample_scalar(p),
                    SimplexNoise::new(7).sample_scalar(q),
                    SimplexNoise::new(7).sample_scalar(p),
                ),
                (
                    worley.sample_scalar(p),
                    WorleyNoise::new(7).sample_scalar(q),
                    WorleyNoise::new(7).sample_scalar(p),
                ),
            ] {
                assert_eq!(got.to_bits(), want.to_bits(), "k={k} p={p}");
                assert!((0.0..=1.0).contains(&got));
                compared += 1;
                if got != unit {
                    differs += 1;
                }
            }
        }
    }
    assert_eq!(compared, 3 * 64 * 3);
    assert!(differs > compared / 2, "frequency changes the field");
}

#[test]
fn gradient_noise_vanishes_on_its_lattice() {
    let mut compared = 0;
    for seed in [0u32, 1, 99] {
        for x in -3i32..=3 {
            for y in -3..=3 {
                for z in -3..=3 {
                    let p = Vec3::new(x as f32, y as f32, z as f32);
                    assert_eq!(
                        PerlinNoise::new(seed).sample_scalar(p),
                        0.5,
                        "Perlin at lattice point {p}"
                    );
                    compared += 1;
                    // Integer points with x + y + z divisible by 3 are simplex
                    // vertices: skew s = (x+y+z)/3 is an integer.
                    if (x + y + z).rem_euclid(3) == 0 {
                        let s = SimplexNoise::new(seed).sample_scalar(p);
                        assert!((s - 0.5).abs() < 1e-6, "simplex at vertex {p}: {s}");
                        compared += 1;
                    }
                }
            }
        }
    }
    assert!(compared > 3 * 343);
    // Off the lattice the fields are not constant.
    let off = Vec3::new(0.37, 1.21, -0.58);
    assert_ne!(SimplexNoise::new(0).sample_scalar(off), 0.5);
    assert_ne!(PerlinNoise::new(0).sample_scalar(off), 0.5);
}
