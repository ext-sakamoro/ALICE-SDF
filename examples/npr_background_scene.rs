//! Shadertoy-style raymarching demo composing NPR primitives
//!
//! Renders a small sphere against a stylized sky using
//! `sky_gradient_bands`, `puffy_cloud_layer`, `sun_disc`,
//! `soft_toon_shade_from_node`, `distance_field_outline_soft`,
//! `vignette`, and `HashNoise`. Outputs an ASCII-art frame so the
//! composition can be verified without a GPU.
//!
//! # Running
//! ```bash
//! cargo run --example npr_background_scene
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use glam::Vec3;

const WIDTH: usize = 80;
const HEIGHT: usize = 32;
const MAX_STEPS: u32 = 64;
const MAX_DIST: f32 = 20.0;
const SURFACE_EPS: f32 = 1e-3;
const RAMP: &[u8] = b" .:-=+*#%@";

fn sample_pixel(
    uv_x: f32,
    uv_y: f32,
    sphere: &SdfNode,
    cam_pos: Vec3,
    to_sun: Vec3,
    palette: &[Vec3],
    noise: HashNoise,
) -> Vec3 {
    // Camera ray: image plane at z = 0, camera at z = -3 looking down +z
    let aspect = WIDTH as f32 / HEIGHT as f32 * 0.5; // ASCII cells are ~2x tall
    let sx = (uv_x - 0.5) * 2.0 * aspect;
    let sy = (0.5 - uv_y) * 2.0;
    let ray_dir = Vec3::new(sx, sy, 1.0).normalize();

    // Raymarch against the sphere
    let mut t = 0.0_f32;
    let mut hit = false;
    let mut point = cam_pos;
    for _ in 0..MAX_STEPS {
        point = cam_pos + ray_dir * t;
        let d = eval(sphere, point);
        if d < SURFACE_EPS {
            hit = true;
            break;
        }
        t += d.max(SURFACE_EPS);
        if t > MAX_DIST {
            break;
        }
    }

    if hit {
        // Foreground: soft toon shading + soft distance-field outline
        let brightness = soft_toon_shade_from_node(sphere, point, to_sun, 3, 0.05);
        let shadow = Vec3::new(0.20, 0.18, 0.35);
        let light = Vec3::new(0.95, 0.88, 0.75);
        let shaded = shadow.lerp(light, brightness);
        // Outline: distance grows fast just past the silhouette
        let outline_mask = distance_field_outline_soft(eval(sphere, point), 0.005, 0.03);
        composite_outline(shaded, Vec3::new(0.02, 0.02, 0.05), outline_mask)
    } else {
        // Background: layered sky + cloud + sun
        let sky = sky_gradient_bands(ray_dir, palette);
        let cloud_noise = noise.sample_scalar(ray_dir * 4.0);
        let cloud_density = puffy_cloud_layer(cloud_noise, 0.55, 0.15);
        let cloud_color = Vec3::new(0.95, 0.95, 0.98);
        let with_cloud = sky.lerp(cloud_color, cloud_density * (ray_dir.y.max(0.0)));
        let sun = sun_disc(ray_dir, to_sun, 0.03, 0.05);
        let shaft = light_shaft_beam(ray_dir, to_sun, 12.0) * 0.15;
        let sun_color = Vec3::new(1.0, 0.95, 0.8);
        with_cloud + sun_color * (sun + shaft)
    }
}

fn render_ascii() -> String {
    let sphere = SdfNode::sphere(1.0);
    let cam_pos = Vec3::new(0.0, 0.0, -3.0);
    let to_sun = Vec3::new(0.4, 0.7, -0.6).normalize();
    let palette = [
        Vec3::new(0.90, 0.60, 0.40),
        Vec3::new(0.65, 0.70, 0.85),
        Vec3::new(0.20, 0.30, 0.60),
    ];
    let noise = HashNoise::new(1337).with_frequency(1.0);

    let mut out = String::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let uv_x = (x as f32 + 0.5) / WIDTH as f32;
            let uv_y = (y as f32 + 0.5) / HEIGHT as f32;
            let color = sample_pixel(uv_x, uv_y, &sphere, cam_pos, to_sun, &palette, noise);
            // Apply vignette to composite
            let vig = vignette(uv_x, uv_y, 0.42, 0.35);
            let final_color = color * vig;
            // Luminance to ramp index
            let lum = final_color.z.mul_add(
                0.0722,
                final_color.y.mul_add(0.7152, final_color.x * 0.2126),
            );
            let lum_c = lum.clamp(0.0, 1.0);
            let idx = (lum_c * (RAMP.len() as f32 - 1.0)).round() as usize;
            out.push(RAMP[idx.min(RAMP.len() - 1)] as char);
        }
        out.push('\n');
    }
    out
}

fn sampled_stats() {
    let sphere = SdfNode::sphere(1.0);
    let cam_pos = Vec3::new(0.0, 0.0, -3.0);
    let to_sun = Vec3::new(0.4, 0.7, -0.6).normalize();
    let palette = [
        Vec3::new(0.90, 0.60, 0.40),
        Vec3::new(0.65, 0.70, 0.85),
        Vec3::new(0.20, 0.30, 0.60),
    ];
    let noise = HashNoise::new(1337);

    let mut hit_count = 0_u32;
    let mut total = 0_u32;
    let mut sum = Vec3::ZERO;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let uv_x = (x as f32 + 0.5) / WIDTH as f32;
            let uv_y = (y as f32 + 0.5) / HEIGHT as f32;
            let color = sample_pixel(uv_x, uv_y, &sphere, cam_pos, to_sun, &palette, noise);
            // Rough hit detection: dark foreground vs bright sky
            if color.x < 0.6 && color.y < 0.6 && color.z < 0.6 {
                hit_count += 1;
            }
            sum += color;
            total += 1;
        }
    }
    let mean = sum / total as f32;
    println!(
        "\nStats: {hit_count}/{total} dark pixels ({:.1}%), mean color = ({:.2}, {:.2}, {:.2})",
        hit_count as f32 * 100.0 / total as f32,
        mean.x,
        mean.y,
        mean.z
    );
}

fn main() {
    println!("ALICE-SDF — NPR background scene");
    println!("=================================");
    println!(
        "Legend: `{}` (dark -> bright)",
        std::str::from_utf8(RAMP).unwrap_or(" ")
    );
    println!();
    let frame = render_ascii();
    print!("{frame}");
    sampled_stats();
}
