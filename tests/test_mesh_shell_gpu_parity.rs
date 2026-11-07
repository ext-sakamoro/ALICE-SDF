//! `ShellRenderer` on a GPU against its CPU path (`MeshBvh::shell_first_hit`) on the same rays
//!
//! Four camera views of two synthetic meshes (a closed sphere, a random soup with thin and
//! degenerate triangles) and a set of rays grazing edges far from their origin. Whether a ray hits,
//! the triangle and the piece it hits must be the same; `t` is reported as the number of rays
//! bit-identical and the largest difference in units in the last place.
//!
//! Without a device it skips, unless `ALICE_SDF_REQUIRE_GPU=1`.
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gpu")]

use alice_sdf::mesh::{
    triangle_shell_hit, MeshBvh, ShellBackend, ShellHit, ShellRay, ShellRenderer,
};
use glam::Vec3;

fn sphere(stacks: u32, slices: u32) -> (Vec<Vec3>, Vec<u32>) {
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

fn soup(n: u32, seed: u64) -> (Vec<Vec3>, Vec<u32>) {
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
fn view(eye: Vec3, size: u32) -> Vec<ShellRay> {
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
fn grazing(v: &[Vec3], ix: &[u32], r: f32) -> Vec<ShellRay> {
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

fn ulps(a: f32, b: f32) -> u64 {
    let key = |x: f32| {
        let i = x.to_bits() as i32;
        if i < 0 {
            i64::from(i32::MIN) - i64::from(i)
        } else {
            i64::from(i)
        }
    };
    (key(a) - key(b)).unsigned_abs()
}

#[derive(Default, Debug)]
struct Stats {
    rays: usize,
    hits: usize,
    set: usize,
    tri: usize,
    feature: usize,
    t_equal: usize,
    max_ulps: u64,
    worst: Option<(usize, f32, f32)>,
}

fn compare(gpu: &[Option<ShellHit>], cpu: &[Option<ShellHit>], s: &mut Stats) {
    assert_eq!(gpu.len(), cpu.len());
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        s.rays += 1;
        match (g, c) {
            (Some(g), Some(c)) => {
                s.hits += 1;
                if g.triangle != c.triangle {
                    s.tri += 1;
                } else if g.feature != c.feature {
                    s.feature += 1;
                }
                if g.t.to_bits() == c.t.to_bits() {
                    s.t_equal += 1;
                } else {
                    let u = ulps(g.t, c.t);
                    if u > s.max_ulps {
                        s.max_ulps = u;
                        s.worst = Some((i, g.t, c.t));
                    }
                }
            }
            (None, None) => {}
            _ => s.set += 1,
        }
    }
}

/// For the first rays whose answers differ by more than rounding: the GPU's triangle on the CPU,
/// the CPU's triangle on the CPU, and the brute-force answer
fn explain(
    bvh: &MeshBvh,
    r: f32,
    rays: &[ShellRay],
    gpu: &[Option<ShellHit>],
    cpu: &[Option<ShellHit>],
) {
    let mut shown = 0;
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        let far = match (g, c) {
            (Some(g), Some(c)) => ulps(g.t, c.t) > 1000,
            (None, None) => false,
            _ => true,
        };
        if !far || shown >= 4 {
            continue;
        }
        shown += 1;
        let ray = rays[i];
        let on = |h: &Option<ShellHit>| {
            h.map(|h| {
                triangle_shell_hit(ray.origin, ray.dir, &bvh.triangles[h.triangle as usize], r)
            })
        };
        let mut brute: Option<(f32, usize)> = None;
        for (k, t) in bvh.triangles.iter().enumerate() {
            if let Some((s, _)) = triangle_shell_hit(ray.origin, ray.dir, t, r) {
                if s <= ray.tmax && brute.is_none_or(|(b, _)| s < b) {
                    brute = Some((s, k));
                }
            }
        }
        let tri = |h: &Option<ShellHit>| {
            h.map(|h| {
                let t = &bvh.triangles[h.triangle as usize];
                [t.v0, t.v1, t.v2]
            })
        };
        println!("  ray {i} o {:?} d {:?}\n    gpu {g:?} -> cpu on it {:?} verts {:?}\n    cpu {c:?} -> {:?}\n    brute {brute:?}", ray.origin, ray.dir, on(g), tri(g), on(c));
    }
}

#[test]
fn gpu_first_hits_match_the_cpu_path() {
    let (sv, si) = sphere(48, 96);
    let (pv, pi) = soup(3000, 0x2545F4914F6CDD1D);
    let meshes = [
        ("sphere", MeshBvh::build(&sv, &si, 4), sv, si, 0.002f32),
        ("soup", MeshBvh::build(&pv, &pi, 4), pv, pi, 0.01),
    ];
    let eyes = [
        Vec3::new(0., 0., 3.),
        Vec3::new(3., 0., 0.),
        Vec3::new(0., 3., 0.01),
        Vec3::new(2., 1.5, 2.),
    ];
    let mut all = Stats::default();
    for (name, bvh, v, ix, r) in &meshes {
        let renderer = ShellRenderer::new(bvh, *r).expect("renderer");
        if renderer.backend() == ShellBackend::Cpu {
            if std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_some() {
                panic!("ALICE_SDF_REQUIRE_GPU is set but no GPU device was found");
            }
            eprintln!("gpu-test: skipped (no GPU device)");
            return;
        }
        let mut sets: Vec<(String, Vec<ShellRay>)> = eyes
            .iter()
            .enumerate()
            .map(|(k, e)| (format!("view {k}"), view(*e, 192)))
            .collect();
        sets.push(("grazing".into(), grazing(v, ix, *r)));
        for (label, rays) in &sets {
            let gpu = renderer.first_hits(rays).expect("gpu");
            let cpu = renderer.cpu_hits(rays);
            let mut s = Stats::default();
            compare(&gpu, &cpu, &mut s);
            println!("{name} {label}: {s:?}");
            explain(bvh, *r, rays, &gpu, &cpu);
            all.rays += s.rays;
            all.hits += s.hits;
            all.set += s.set;
            all.tri += s.tri;
            all.feature += s.feature;
            all.t_equal += s.t_equal;
            all.max_ulps = all.max_ulps.max(s.max_ulps);
        }
    }
    println!("gpu-test: ran");
    println!("all: {all:?}");
    assert!(
        all.rays > 0 && all.hits > 1000,
        "compared too little: {all:?}"
    );
    assert_eq!(
        (all.set, all.tri, all.feature),
        (0, 0, 0),
        "hit set, triangle or feature differs: {all:?}"
    );
}
