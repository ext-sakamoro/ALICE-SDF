//! NPR primitives demonstration
//!
//! Exercises the full 9-category NPR primitive set (toon, outline, sky,
//! rim, hatch, distortion, palette, composition, motion) at a small grid
//! of inputs and prints the results.
//!
//! # Running
//! ```bash
//! cargo run --example npr_toon_demo
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::npr::composition::{bloom_toon, chromatic_offsets, vignette};
use alice_sdf::npr::distortion::{hand_drawn_jitter, line_boil, sketch_wobble};
use alice_sdf::npr::hatch::{cross_hatch, hatch_lines, paper_grain, pencil_shade};
use alice_sdf::npr::motion::{impact_flash, speed_line};
use alice_sdf::npr::outline::{composite_outline, distance_field_outline_soft};
use alice_sdf::npr::palette::{palette_gradient, season_palette, time_of_day};
use alice_sdf::npr::rim::{fresnel_rim, procedural_matcap, stylized_specular};
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
            to_sun.z.mul_add(-s, to_sun.x * c),
            to_sun.y,
            to_sun.z.mul_add(c, to_sun.x * s),
        );
        let i = light_shaft_beam(view, to_sun, 8.0);
        println!("  offset = {offset_deg:>4.1}deg  intensity = {i:.4}");
    }

    println!("\n[sky] Sun disc (radius = 0.02 rad, softness = 0.03 rad)");
    for offset_deg in [0.0_f32, 0.5, 1.5, 3.0, 5.0] {
        let a = offset_deg.to_radians();
        let (s, c) = a.sin_cos();
        let view = Vec3::new(
            to_sun.z.mul_add(-s, to_sun.x * c),
            to_sun.y,
            to_sun.z.mul_add(c, to_sun.x * s),
        );
        let i = sun_disc(view, to_sun, 0.02, 0.03);
        println!("  offset = {offset_deg:>4.1}deg  intensity = {i:.4}");
    }

    println!("\n[rim] Fresnel rim (power = 2, intensity = 1)");
    for i in 0..=5 {
        let ndv = i as f32 / 5.0;
        println!("  n.v = {ndv:.2}  rim = {:.3}", fresnel_rim(ndv, 2.0, 1.0));
    }

    println!("\n[rim] Procedural matcap (2x2 palette)");
    let bl = Vec3::new(0.1, 0.1, 0.3);
    let br = Vec3::new(0.6, 0.2, 0.2);
    let tl = Vec3::new(0.2, 0.5, 0.7);
    let tr = Vec3::new(0.95, 0.9, 0.85);
    for &(nx, ny) in &[(-1.0_f32, -1.0), (0.0, 0.0), (1.0, 1.0), (-1.0, 1.0)] {
        let c = procedural_matcap(Vec3::new(nx, ny, 0.0), bl, br, tl, tr);
        println!(
            "  n = ({nx:+.1}, {ny:+.1})  color = ({:.2}, {:.2}, {:.2})",
            c.x, c.y, c.z
        );
    }

    println!("\n[rim] Stylized specular (sharp anime highlight)");
    for i in 0..=5 {
        let ndh = i as f32 / 5.0;
        let hard = stylized_specular(ndh, 8.0, 0.0);
        let soft = stylized_specular(ndh, 8.0, 0.1);
        println!("  n.h = {ndh:.2}  hard = {hard:.2}  soft = {soft:.3}");
    }

    println!("\n[hatch] Hatch lines (angle = 0.4 rad, density = 12, thickness = 0.1)");
    for uvy in [0.0_f32, 0.02, 0.05, 0.08, 0.1] {
        println!(
            "  uv = (0.5, {uvy:.2})  mask = {:.1}",
            hatch_lines(0.5, uvy, 0.4, 12.0, 0.1)
        );
    }

    println!("\n[hatch] Cross-hatch coverage sweep");
    let mut covered = 0;
    let mut total = 0;
    for i in 0..40 {
        for j in 0..40 {
            let uvx = i as f32 / 40.0;
            let uvy = j as f32 / 40.0;
            let m = cross_hatch(uvx, uvy, 0.4, -0.4, 8.0, 0.08);
            if m > 0.5 {
                covered += 1;
            }
            total += 1;
        }
    }
    println!(
        "  covered {covered}/{total} = {:.1}%",
        covered as f32 * 100.0 / total as f32
    );

    println!("\n[hatch] Paper grain and pencil shade");
    for &(noise, nl) in &[(0.2_f32, 0.9), (0.5, 0.5), (0.8, 0.1)] {
        println!(
            "  noise = {noise:.2} n.l = {nl:.2}  grain = {:.3}  pencil = {:.3}",
            paper_grain(noise, 0.2),
            pencil_shade(nl, 1.0)
        );
    }

    println!("\n[distortion] Sketch wobble and line boil (position = (1, 2, 3))");
    let p = Vec3::new(1.0, 2.0, 3.0);
    let wobble = sketch_wobble(p, 0.1, 3.0);
    let boil = line_boil(p, 0.5, 0.1, 3.0);
    let jitter = hand_drawn_jitter(p, Vec3::new(0.5, -0.3, 0.7), 0.1);
    println!(
        "  wobble = ({:.3}, {:.3}, {:.3})",
        wobble.x, wobble.y, wobble.z
    );
    println!("  boil   = ({:.3}, {:.3}, {:.3})", boil.x, boil.y, boil.z);
    println!(
        "  jitter = ({:.3}, {:.3}, {:.3})",
        jitter.x, jitter.y, jitter.z
    );

    println!("\n[palette] Time-of-day sweep");
    let night = Vec3::new(0.05, 0.05, 0.15);
    let dusk = Vec3::new(0.9, 0.5, 0.3);
    let noon = Vec3::new(0.7, 0.8, 1.0);
    for &alt in &[-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
        let c = time_of_day(alt, night, dusk, noon);
        println!(
            "  alt = {alt:+.2}  color = ({:.2}, {:.2}, {:.2})",
            c.x, c.y, c.z
        );
    }

    println!("\n[palette] Season sweep");
    let spring = Vec3::new(0.9, 0.8, 0.7);
    let summer = Vec3::new(0.3, 0.9, 0.2);
    let autumn = Vec3::new(0.9, 0.5, 0.1);
    let winter = Vec3::new(0.8, 0.9, 1.0);
    for i in 0..=4 {
        let t = i as f32 / 4.0;
        let c = season_palette(t, spring, summer, autumn, winter);
        println!("  t = {t:.2}  color = ({:.2}, {:.2}, {:.2})", c.x, c.y, c.z);
    }

    println!("\n[palette] Palette gradient (5-anchor)");
    let anchors = [
        Vec3::new(0.1, 0.05, 0.2),
        Vec3::new(0.5, 0.1, 0.3),
        Vec3::new(0.9, 0.4, 0.2),
        Vec3::new(0.95, 0.85, 0.5),
        Vec3::new(0.8, 0.95, 0.95),
    ];
    for i in 0..=5 {
        let t = i as f32 / 5.0;
        let c = palette_gradient(t, &anchors);
        println!("  t = {t:.2}  color = ({:.2}, {:.2}, {:.2})", c.x, c.y, c.z);
    }

    println!("\n[composition] Vignette (radius = 0.3, softness = 0.2)");
    for &(uvx, uvy) in &[(0.5_f32, 0.5), (0.7, 0.5), (0.9, 0.5), (0.0, 0.0)] {
        println!(
            "  uv = ({uvx:.1}, {uvy:.1})  mult = {:.3}",
            vignette(uvx, uvy, 0.3, 0.2)
        );
    }

    println!("\n[composition] Bloom toon (threshold = 0.5, intensity = 0.8)");
    for &c in &[
        Vec3::new(0.3, 0.3, 0.3),
        Vec3::new(0.6, 0.6, 0.6),
        Vec3::new(0.95, 0.3, 0.3),
    ] {
        let out = bloom_toon(c, 0.5, 0.8);
        println!(
            "  in = ({:.2}, {:.2}, {:.2})  bloom = ({:.2}, {:.2}, {:.2})",
            c.x, c.y, c.z, out.x, out.y, out.z
        );
    }

    println!("\n[composition] Chromatic offsets (uv = (0.7, 0.3), strength = 0.05)");
    let offs = chromatic_offsets(0.7, 0.3, 0.05);
    for (label, (u, v)) in ["red", "green", "blue"].iter().zip(offs.iter()) {
        println!("  {label:>5}: ({u:.4}, {v:.4})");
    }

    println!("\n[motion] Speed line coverage (12 lines, thickness = 0.05)");
    let mut hits = 0;
    let mut samples = 0;
    for i in 0..40 {
        for j in 0..40 {
            let uvx = i as f32 / 40.0;
            let uvy = j as f32 / 40.0;
            if speed_line(uvx, uvy, 0.5, 0.5, 12, 0.05) > 0.5 {
                hits += 1;
            }
            samples += 1;
        }
    }
    println!(
        "  hits {hits}/{samples} = {:.1}%",
        hits as f32 * 100.0 / samples as f32
    );

    println!("\n[motion] Impact flash decay (decay = 0.3, intensity = 1)");
    for &t in &[0.0_f32, 0.1, 0.3, 0.6, 1.0] {
        println!("  t = {t:.2}  intensity = {:.4}", impact_flash(t, 0.3, 1.0));
    }

    println!("\nDone.");
}
