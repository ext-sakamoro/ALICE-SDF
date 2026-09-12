//! NPR primitives demonstration
//!
//! Exercises toon, outline, and sky primitives at a small grid of inputs
//! and prints the results.
//!
//! # Running
//! ```bash
//! cargo run --example npr_toon_demo
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::npr::outline::{composite_outline, distance_field_outline_soft};
use alice_sdf::npr::sky::{
    distance_color_quantize, light_shaft_beam, puffy_cloud_layer, sky_gradient_bands, sun_disc,
};
use alice_sdf::npr::toon::{posterize_color, soft_toon_ramp, toon_ramp, two_tone};
use glam::Vec3;

fn main() {
    println!("ALICE-SDF — NPR primitives demo");
    println!("================================");

    println!("\n[toon] N-band ramp vs soft ramp (3 bands, smoothness = 0.05)");
    for i in 0..=10 {
        let nl = i as f32 / 10.0;
        let hard = toon_ramp(nl, 3);
        let soft = soft_toon_ramp(nl, 3, 0.05);
        println!("  n.l = {nl:.2}  hard = {hard:.3}  soft = {soft:.3}");
    }

    println!("\n[toon] Two-tone shading");
    let shadow = Vec3::new(0.1, 0.15, 0.35);
    let light = Vec3::new(0.95, 0.9, 0.8);
    for &nl in &[0.1_f32, 0.4, 0.5, 0.6, 0.9] {
        let c = two_tone(nl, shadow, light, 0.5);
        println!(
            "  n.l = {nl:.2}  color = ({:.2}, {:.2}, {:.2})",
            c.x, c.y, c.z
        );
    }

    println!("\n[toon] Posterize (4 levels)");
    let src = Vec3::new(0.13, 0.47, 0.82);
    let post = posterize_color(src, 4);
    println!("  src  = ({:.2}, {:.2}, {:.2})", src.x, src.y, src.z);
    println!("  post = ({:.2}, {:.2}, {:.2})", post.x, post.y, post.z);

    println!("\n[outline] Soft distance-field outline (inner=0.02, outer=0.10)");
    for &d in &[-0.15_f32, -0.05, 0.0, 0.05, 0.15] {
        let a = distance_field_outline_soft(d, 0.02, 0.10);
        let base = Vec3::new(0.9, 0.9, 0.9);
        let color = composite_outline(base, Vec3::ZERO, a);
        println!(
            "  sdf = {d:+.3}  alpha = {a:.3}  color = ({:.2}, {:.2}, {:.2})",
            color.x, color.y, color.z
        );
    }

    println!("\n[sky] Gradient bands (3-color palette)");
    let palette = [
        Vec3::new(0.90, 0.60, 0.40),
        Vec3::new(0.60, 0.70, 0.90),
        Vec3::new(0.20, 0.35, 0.70),
    ];
    for &y in &[-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
        let c = sky_gradient_bands(Vec3::new(0.0, y, 0.0), &palette);
        println!(
            "  dir.y = {y:+.2}  sky = ({:.2}, {:.2}, {:.2})",
            c.x, c.y, c.z
        );
    }

    println!("\n[sky] Cloud coverage (coverage sweep, noise = 0.5)");
    for i in 0..=10 {
        let coverage = i as f32 / 10.0;
        let d = puffy_cloud_layer(0.5, coverage, 0.1);
        println!("  coverage = {coverage:.2}  density = {d:.3}");
    }

    println!("\n[sky] Distance color quantize (near = warm, far = cool, 4 bands)");
    let near = Vec3::new(0.85, 0.65, 0.45);
    let far = Vec3::new(0.35, 0.45, 0.70);
    for i in 0..=10 {
        let dist = i as f32 * 10.0;
        let c = distance_color_quantize(dist, 100.0, near, far, 4);
        println!(
            "  dist = {dist:>5.1}  color = ({:.2}, {:.2}, {:.2})",
            c.x, c.y, c.z
        );
    }

    println!("\n[sky] Light shaft (density = 8)");
    let to_sun = Vec3::new(0.3, 0.7, 0.5).normalize();
    for offset_deg in [0.0_f32, 5.0, 15.0, 45.0, 90.0] {
        let a = offset_deg.to_radians();
        let (s, c) = a.sin_cos();
        let view = Vec3::new(
            to_sun.x * c - to_sun.z * s,
            to_sun.y,
            to_sun.x * s + to_sun.z * c,
        );
        let i = light_shaft_beam(view, to_sun, 8.0);
        println!("  offset = {offset_deg:>4.1}deg  intensity = {i:.4}");
    }

    println!("\n[sky] Sun disc (radius = 0.02 rad, softness = 0.03 rad)");
    for offset_deg in [0.0_f32, 0.5, 1.5, 3.0, 5.0] {
        let a = offset_deg.to_radians();
        let (s, c) = a.sin_cos();
        let view = Vec3::new(
            to_sun.x * c - to_sun.z * s,
            to_sun.y,
            to_sun.x * s + to_sun.z * c,
        );
        let i = sun_disc(view, to_sun, 0.02, 0.03);
        println!("  offset = {offset_deg:>4.1}deg  intensity = {i:.4}");
    }

    println!("\nDone.");
}
