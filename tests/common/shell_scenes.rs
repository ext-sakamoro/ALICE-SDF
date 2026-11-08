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
                alice_det_math::sin(th) * alice_det_math::cos(ph),
                alice_det_math::cos(th),
                alice_det_math::sin(th) * alice_det_math::sin(ph),
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

/// Long thin triangles (40-80 x 1-4 mm) about 1400 mm from the origin and 25 rays per triangle from
/// 900-1400 mm back, aimed at points of it: the scale where an f32 world coordinate's ulp is a
/// sizeable part of a 0.002 mm shell
pub fn far_thin(seed: u64) -> (Vec<Vec3>, Vec<u32>, Vec<ShellRay>) {
    let mut x = seed;
    let mut u = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 40) as f32 / (1u64 << 24) as f32
    };
    let (mut v, mut ix, mut aims) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..60u32 {
        let base = Vec3::new(
            u() * 1200.0 - 600.0,
            u() * 500.0 - 170.0,
            900.0 + u() * 500.0,
        );
        let ax = Vec3::new(u() - 0.5, u() - 0.5, u() - 0.5).normalize();
        let pp = Vec3::new(u() - 0.5, u() - 0.5, u() - 0.5)
            .cross(ax)
            .normalize();
        let (l, w) = (40.0 + 40.0 * u(), 1.0 + 3.0 * u());
        v.extend([
            base,
            base + ax * l,
            base + ax * (l - w * (u() - 0.5)) + pp * w,
        ]);
        ix.extend([3 * k, 3 * k + 1, 3 * k + 2]);
    }
    for t in ix.chunks_exact(3) {
        let (p0, p1, p2) = (v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]);
        for _ in 0..25 {
            let (a, b0) = (u(), u());
            let p = p0 + (p1 - p0) * a + (p2 - p0) * (b0 * (1.0 - a));
            let n = (p1 - p0).cross(p2 - p0).normalize();
            let tilt = 0.1 + 0.8 * u();
            let side = Vec3::new(u() - 0.5, u() - 0.5, u() - 0.5)
                .cross(n)
                .normalize();
            let d = (-n * tilt + side * (1.0 - tilt * tilt).sqrt()).normalize();
            let o = p - d * (900.0 + 500.0 * u());
            aims.push(ShellRay {
                origin: o,
                dir: d,
                tmax: 3000.0,
            });
        }
    }
    (v, ix, aims)
}
