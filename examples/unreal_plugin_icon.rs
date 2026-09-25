//! The UE5 plugin's browser icon, raymarched by alice-sdf itself.
//!
//! `unreal-plugin/Resources/Icon128.png` is what the Plugins browser shows.
//! Rather than a drawn asset, it is a render of an SDF scene through this
//! crate's own evaluator and normals — so the icon cannot drift away from
//! what the library actually produces, and regenerating it is one command.
//!
//! # Running
//! ```bash
//! cargo run --example unreal_plugin_icon --features image -- unreal-plugin/Resources/Icon128.png
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::{eval, eval_normal};
use alice_sdf::prelude::*;

/// Plugin icons are 128×128 (Epic's plugin browser scales them down).
const SIZE: u32 = 128;
/// Supersampling factor: 4×4 samples per pixel, resolved in linear space.
const SS: u32 = 4;

/// The scene: a smooth union of a sphere and a box with a torus cut out —
/// union, blend and subtraction in one shape, which is what the plugin is for.
fn scene() -> SdfNode {
    let body = SdfNode::sphere(0.60).smooth_union(
        SdfNode::box3d(0.40, 0.40, 0.40).translate(0.46, -0.30, 0.12),
        0.26,
    );
    // A ring cut through the front of the body: the torus law is in the XZ
    // plane (axis = Y), so a quarter turn about X points its axis at the
    // camera and the subtraction carves a visible circular groove.
    let ring = SdfNode::torus(0.42, 0.11)
        .rotate(glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
        .translate(-0.04, 0.02, 0.30);
    body.smooth_subtract(ring, 0.05)
}

/// Sphere-traced hit distance, or `None` when the ray escapes.
fn trace(node: &SdfNode, ro: Vec3, rd: Vec3) -> Option<f32> {
    let mut t = 0.0f32;
    for _ in 0..96 {
        let p = ro + rd * t;
        let d = eval(node, p);
        if d < 1.0e-4 {
            return Some(t);
        }
        t += d.max(1.0e-4);
        if t > 6.0 {
            break;
        }
    }
    None
}

/// Soft shadow along the light direction (IQ's penumbra estimate).
fn shadow(node: &SdfNode, p: Vec3, light: Vec3) -> f32 {
    let mut res = 1.0f32;
    let mut t = 0.02f32;
    for _ in 0..48 {
        let d = eval(node, p + light * t);
        if d < 1.0e-4 {
            return 0.0;
        }
        res = res.min(12.0 * d / t);
        t += d.clamp(0.01, 0.30);
        if t > 4.0 {
            break;
        }
    }
    res.clamp(0.0, 1.0)
}

/// sRGB transfer for a linear channel.
fn encode_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "unreal-plugin/Resources/Icon128.png".to_string());

    let node = scene();
    let camera = Vec3::new(0.20, 0.40, 1.95);
    let target = Vec3::new(0.12, -0.04, 0.0);
    let forward = (target - camera).normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward);
    let light = Vec3::new(-0.45, 0.82, 0.36).normalize();

    let mut rgba = vec![0u8; (SIZE * SIZE * 4) as usize];
    let inv_ss = 1.0 / (SS * SS) as f32;

    for py in 0..SIZE {
        for px in 0..SIZE {
            let mut colour = Vec3::ZERO;
            let mut alpha = 0.0f32;

            for sy in 0..SS {
                for sx in 0..SS {
                    let u = (px as f32 + (sx as f32 + 0.5) / SS as f32) / SIZE as f32 * 2.0 - 1.0;
                    let v = 1.0 - (py as f32 + (sy as f32 + 0.5) / SS as f32) / SIZE as f32 * 2.0;
                    let rd = (forward + right * (u * 0.55) + up * (v * 0.55)).normalize();

                    let Some(t) = trace(&node, camera, rd) else {
                        // Background: a dark vertical gradient (an opaque icon
                        // reads better in the plugin browser than a cut-out).
                        let g = (v * 0.5 + 0.5).clamp(0.0, 1.0);
                        colour += Vec3::new(0.035, 0.040, 0.075).lerp(
                            Vec3::new(0.085, 0.075, 0.150),
                            g,
                        );
                        alpha += 1.0;
                        continue;
                    };
                    let p = camera + rd * t;
                    let n = eval_normal(&node, p);

                    // Two-tone lighting: a violet key light with a cyan rim,
                    // the palette the ALICE material previews use.
                    let diffuse = n.dot(light).max(0.0) * shadow(&node, p, light);
                    let rim = (1.0 + n.dot(rd)).clamp(0.0, 1.0).powf(2.6);
                    let key = Vec3::new(0.62, 0.42, 0.98);
                    let fill = Vec3::new(0.10, 0.13, 0.28);
                    let rim_col = Vec3::new(0.25, 0.85, 0.95);

                    let half = (light - rd).normalize();
                    let spec = n.dot(half).max(0.0).powf(46.0) * shadow(&node, p, light);

                    colour += key * diffuse + fill + rim_col * rim * 0.70 + Vec3::ONE * spec * 0.7;
                    alpha += 1.0;
                }
            }

            let idx = ((py * SIZE + px) * 4) as usize;
            if alpha > 0.0 {
                // Premultiplied-free resolve: average the covered samples only,
                // so edge pixels keep the shape's colour and fade in alpha.
                let c = colour * (1.0 / alpha);
                rgba[idx] = encode_srgb(c.x);
                rgba[idx + 1] = encode_srgb(c.y);
                rgba[idx + 2] = encode_srgb(c.z);
                rgba[idx + 3] = (alpha * inv_ss * 255.0 + 0.5) as u8;
            }
        }
    }

    if let Some(parent) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(parent).expect("create Resources dir");
    }
    image::save_buffer(&out, &rgba, SIZE, SIZE, image::ColorType::Rgba8)
        .unwrap_or_else(|e| panic!("write {out}: {e}"));

    // Every pixel is opaque now (background included); the check is that the
    // lit shape covers a sensible part of the frame.
    let covered = rgba
        .chunks_exact(4)
        .filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 240)
        .count();
    println!(
        "wrote {out} ({SIZE}x{SIZE}, {SS}x{SS} samples/pixel, {covered} pixels covered by the SDF)"
    );
    assert!(
        (3000..14000).contains(&covered),
        "the shape covers {covered} of {} pixels — camera or scene drifted",
        SIZE * SIZE
    );
}
