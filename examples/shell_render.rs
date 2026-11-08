//! Render a mesh thickened by a radius: four views of a synthetic torus through `ShellRenderer`
//! (on a GPU when one is available, otherwise on the CPU), written as PNG images of the hit
//! distance and of the shell's normal.
//!
//! ```text
//! cargo run --release --example shell_render --features gpu,image -- <out dir> [size] [radius] [--cpu]
//! ```
//!
//! `--cpu` renders on the CPU even when a GPU is available. The centre pixel of each view is
//! checked against `MeshBvh::shell_first_hit` (the same answer on the CPU, one ray).
//!
//! Writes `<out dir>/view<k>_t.png` (distance to the first hit, near is light) and
//! `<out dir>/view<k>_normal.png` (the shell's outward normal, `n * 0.5 + 0.5` as RGB), and prints
//! one line per view.
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::{MeshBvh, ShellFeature, ShellRay, ShellRenderer};
use glam::Vec3;
use std::path::PathBuf;

/// A torus of `segments` x `sides` quads (two triangles each)
fn torus(major: f32, minor: f32, segments: u32, sides: u32) -> (Vec<Vec3>, Vec<u32>) {
    let mut v = Vec::new();
    for i in 0..segments {
        let a = std::f32::consts::TAU * i as f32 / segments as f32;
        for j in 0..sides {
            let b = std::f32::consts::TAU * j as f32 / sides as f32;
            let ring = major + minor * alice_det_math::cos(b);
            v.push(Vec3::new(
                ring * alice_det_math::cos(a),
                minor * alice_det_math::sin(b),
                ring * alice_det_math::sin(a),
            ));
        }
    }
    let mut ix = Vec::new();
    for i in 0..segments {
        for j in 0..sides {
            let p = i * sides + j;
            let q = ((i + 1) % segments) * sides + j;
            let p1 = i * sides + (j + 1) % sides;
            let q1 = ((i + 1) % segments) * sides + (j + 1) % sides;
            ix.extend_from_slice(&[p, q, p1, p1, q, q1]);
        }
    }
    (v, ix)
}

/// `size` x `size` rays from a pinhole camera at `eye` looking at the origin
fn view(eye: Vec3, size: u32) -> Vec<ShellRay> {
    let fwd = (-eye).normalize();
    let up0 = if fwd.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
    let right = fwd.cross(up0).normalize();
    let up = right.cross(fwd);
    let mut rays = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        for x in 0..size {
            let sx = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
            let sy = 1.0 - (y as f32 + 0.5) / size as f32 * 2.0;
            let dir = (fwd + right * (sx * 0.7) + up * (sy * 0.7)).normalize();
            rays.push(ShellRay {
                origin: eye,
                dir,
                tmax: 100.0,
            });
        }
    }
    rays
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let all: Vec<String> = std::env::args().skip(1).collect();
    let on_cpu = all.iter().any(|a| a == "--cpu");
    let mut args = all.into_iter().filter(|a| a != "--cpu");
    let out = PathBuf::from(args.next().unwrap_or_else(|| "shell_render_out".into()));
    let size: u32 = args.next().map_or(Ok(256), |s| s.parse())?;
    let r: f32 = args.next().map_or(Ok(0.01), |s| s.parse())?;
    std::fs::create_dir_all(&out)?;

    let (v, ix) = torus(1.0, 0.35, 96, 48);
    let bvh = MeshBvh::build(&v, &ix, 4);
    let renderer = if on_cpu {
        ShellRenderer::cpu(&bvh, r)
    } else {
        ShellRenderer::new(&bvh, r)?
    };
    println!(
        "[shell_render] {} triangles, r {r}, {size}x{size}, backend {:?}",
        bvh.triangles.len(),
        renderer.backend()
    );

    let eyes = [
        Vec3::new(0.0, 0.0, 3.5),
        Vec3::new(3.5, 0.0, 0.0),
        Vec3::new(0.0, 3.5, 0.01),
        Vec3::new(2.4, 1.8, 2.4),
    ];
    for (k, eye) in eyes.iter().enumerate() {
        let rays = view(*eye, size);
        let started = std::time::Instant::now();
        let hits = renderer.first_hits(&rays)?;
        let elapsed = started.elapsed();
        let found: Vec<f32> = hits.iter().flatten().map(|h| h.t).collect();
        let (lo, hi) = found
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &t| {
                (a.min(t), b.max(t))
            });
        let span = (hi - lo).max(f32::MIN_POSITIVE);
        let (mut depth, mut normal) = (Vec::new(), Vec::new());
        for (ray, hit) in rays.iter().zip(&hits) {
            match hit {
                Some(h) => {
                    let g = (255.0 * (1.0 - 0.8 * (h.t - lo) / span)) as u8;
                    depth.extend_from_slice(&[g, g, g]);
                    // the shell's normal: away from the nearest point of the triangle, or the
                    // face's when the ray starts inside the shell
                    let tri = &bvh.triangles[h.triangle as usize];
                    let p = ray.origin + ray.dir * h.t;
                    let n = if h.feature == ShellFeature::Inside {
                        tri.normal
                    } else {
                        (p - tri.closest_point(p)).normalize_or(tri.normal)
                    };
                    let c = (n * 0.5 + Vec3::splat(0.5)) * 255.0;
                    normal.extend_from_slice(&[c.x as u8, c.y as u8, c.z as u8]);
                }
                None => {
                    depth.extend_from_slice(&[0, 0, 0]);
                    normal.extend_from_slice(&[0, 0, 0]);
                }
            }
        }
        // the centre pixel against the CPU method, one ray: the same triangle and piece
        let mid = (size / 2 * size + size / 2) as usize;
        let one = bvh.shell_first_hit(rays[mid].origin, rays[mid].dir, r, rays[mid].tmax);
        let same =
            one.map(|h| (h.triangle, h.feature)) == hits[mid].map(|h| (h.triangle, h.feature));
        if !same {
            return Err(format!(
                "view {k}: centre pixel {:?} against the CPU's {one:?}",
                hits[mid]
            )
            .into());
        }
        let t_path = out.join(format!("view{k}_t.png"));
        let n_path = out.join(format!("view{k}_normal.png"));
        image::save_buffer(&t_path, &depth, size, size, image::ColorType::Rgb8)?;
        image::save_buffer(&n_path, &normal, size, size, image::ColorType::Rgb8)?;
        println!(
            "[shell_render] view {k}: {} hits of {} rays in {:.1} ms, t {lo:.4}..{hi:.4}, wrote {} and {}",
            found.len(),
            rays.len(),
            elapsed.as_secs_f64() * 1e3,
            t_path.display(),
            n_path.display()
        );
    }
    Ok(())
}
