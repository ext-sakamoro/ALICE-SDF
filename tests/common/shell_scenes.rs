//! Meshes and rays for the shell first-hit tests: a closed sphere, a soup with collinear,
//! zero-length-edge and sliver triangles, pinhole views, and rays grazing edges far from their origin
//!
//! Author: Moroya Sakamoto
#![allow(dead_code)]

use alice_sdf::mesh::ShellRay;
use glam::Vec3;

pub fn sphere(stacks: u32, slices: u32) -> (Vec<Vec3>, Vec<u32>) {
    let mut v = Vec::new();
    for i in 0..=stacks {
        let th = std::f32::consts::PI * i as f32 / stacks as f32;
        for j in 0..slices {
            let ph = std::f32::consts::TAU * j as f32 / slices as f32;
            v.push(Vec3::new(
                th.sin() * ph.cos(),
                th.cos(),
                th.sin() * ph.sin(),
            ));
        }
    }
    let mut ix = Vec::new();
    for i in 0..stacks {
        for j in 0..slices {
            let a = i * slices + j;
            let b = i * slices + (j + 1) % slices;
            let (c, d) = (a + slices, b + slices);
            ix.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    (v, ix)
}

pub fn soup(n: u32, seed: u64) -> (Vec<Vec3>, Vec<u32>) {
    let mut x = seed;
    let mut u = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 40) as f32 / (1u64 << 24) as f32
    };
    let mut v = Vec::new();
    for k in 0..n {
        let c = Vec3::new(u() * 2. - 1., u() * 2. - 1., u() * 2. - 1.);
        let a = c + Vec3::new(u() - 0.5, u() - 0.5, u() - 0.5) * 0.4;
        let b = c + Vec3::new(u() - 0.5, u() - 0.5, u() - 0.5) * 0.4;
        let third = match k % 7 {
            0 => (a + b) * 0.5,                         // collinear
            1 => a,                                     // a zero-length edge
            2 => a + (b - a) * 0.5 + Vec3::splat(1e-4), // a sliver
            _ => c + Vec3::new(u() - 0.5, u() - 0.5, u() - 0.5) * 0.4,
        };
        v.extend_from_slice(&[a, b, third]);
    }
    let ix = (0..v.len() as u32).collect();
    (v, ix)
}

/// `size`² rays from a pinhole camera at `eye` looking at the origin
pub fn view(eye: Vec3, size: u32) -> Vec<ShellRay> {
    let fwd = (-eye).normalize();
    let up0 = if fwd.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
    let right = fwd.cross(up0).normalize();
    let up = right.cross(fwd);
    let mut rays = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        for x in 0..size {
            let sx = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
            let sy = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;
            let dir = (fwd + right * (sx * 0.6) + up * (sy * 0.6)).normalize();
            rays.push(ShellRay {
                origin: eye,
                dir,
                tmax: 100.0,
            });
        }
    }
    rays
}

/// Rays at slope `s` past each edge of the mesh at height near `r`, closest approach at the edge's
/// middle, starting far back (the cases where the textbook discriminant cancels)
pub fn grazing(v: &[Vec3], ix: &[u32], r: f32) -> Vec<ShellRay> {
    let mut rays = Vec::new();
    for (k, t) in ix.chunks_exact(3).enumerate().take(400) {
        let (a, b) = (v[t[0] as usize], v[t[1] as usize]);
        let e = b - a;
        if e.length() < 1e-3 {
            continue;
        }
        let ed = e.normalize();
        let side = if ed.cross(Vec3::Y).length() > 0.1 {
            ed.cross(Vec3::Y).normalize()
        } else {
            ed.cross(Vec3::X).normalize()
        };
        let mid = (a + b) * 0.5;
        for (j, h) in [0.9f32, 0.999, 1.0, 1.001, 1.1].iter().enumerate() {
            let s = [0.01f32, 0.005, 0.004][(k + j) % 3];
            let dir = (ed + side * s).normalize();
            let back = 50.0 + (k % 5) as f32 * 100.0;
            let origin = mid + side * (h * r) - dir * back;
            rays.push(ShellRay {
                origin,
                dir,
                tmax: back * 2.0,
            });
        }
    }
    rays
}
