//! Prototype driver: `mesh_gpu_proto <dir> <rays name|grid> <shell> <tile>`. Prints compile time, ms
//! per frame over repeated runs, bits equal across runs, and the difference from the CPU on the same
//! rays. Runs on one device; the limit is the caller's.
use alice_sdf::mesh::gpu_bvh::{flatten, RayOut, Tracer};
use alice_sdf::mesh::MeshBvh;
use glam::Vec3;
use std::time::Instant;

fn f32s(p: &str) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
fn u32s(p: &str) -> Vec<u32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// A generated body of about 40,000 triangles, in mm: a bumpy torus knot tube
fn synthetic() -> (Vec<Vec3>, Vec<u32>) {
    let (nu, nv) = (500_u32, 40_u32);
    let mut v = Vec::new();
    let mut t = Vec::new();
    let curve = |u: f32| {
        let (p, q) = (2.0 * u, 3.0 * u);
        let r = 300.0 + 120.0 * q.cos();
        Vec3::new(r * p.cos(), r * p.sin(), 120.0 * q.sin())
    };
    for i in 0..nu {
        let u = i as f32 / nu as f32 * std::f32::consts::TAU;
        let c = curve(u);
        let n = (curve(u + 0.001) - c).normalize();
        let a = n.cross(Vec3::Z).normalize();
        let b = n.cross(a);
        for j in 0..nv {
            let w = j as f32 / nv as f32 * std::f32::consts::TAU;
            let r = 60.0 + 12.0 * (w * 5.0 + u * 40.0).sin();
            v.push(c + a * (r * w.cos()) + b * (r * w.sin()));
            let (i1, j1) = ((i + 1) % nu, (j + 1) % nv);
            let at = |x: u32, y: u32| x * nv + y;
            t.extend_from_slice(&[
                at(i, j),
                at(i1, j),
                at(i1, j1),
                at(i, j),
                at(i1, j1),
                at(i, j1),
            ]);
        }
    }
    (v, t)
}

/// A 1300 x 680 front view of the mesh's box: orthographic, along -z, every pixel a ray
fn grid(v: &[Vec3]) -> Vec<f32> {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for p in v {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let size = hi - lo;
    let step = (size.x / 1300.0).max(size.y / 680.0);
    let c = (lo + hi) * 0.5;
    let z = hi.z + 10.0;
    let end = size.z + 20.0;
    let mut r = Vec::with_capacity(1300 * 680 * 7);
    for y in 0..680 {
        for x in 0..1300 {
            let px = c.x + (x as f32 + 0.5 - 650.0) * step;
            let py = c.y + (340.0 - y as f32 - 0.5) * step;
            r.extend_from_slice(&[px, py, z, 0.0, 0.0, -1.0, end]);
        }
    }
    r
}

fn cpu(bvh: &MeshBvh, r: &[f32], shell: f32, hit: f32, max_evals: u32) -> Vec<RayOut> {
    use rayon::prelude::*;
    r.par_chunks_exact(7)
        .map(|c| {
            let o = Vec3::new(c[0], c[1], c[2]);
            let d = Vec3::new(c[3], c[4], c[5]);
            let (mut s, mut e) = (0.0_f32, 0_u32);
            let mut out = RayOut {
                s: 0.0,
                tri: u32::MAX,
                evals: 0,
                hit: 0,
            };
            while s < c[6] && e < max_evals {
                let p = o + d * s;
                let q = bvh.unsigned_distance(p) - shell;
                e += 1;
                if q < hit {
                    out.hit = 1;
                    out.s = s;
                    if let Some((k, _, _)) = bvh.closest_triangle(p) {
                        out.tri = k as u32;
                    }
                    break;
                }
                s += q;
            }
            out.evals = e;
            out
        })
        .collect()
}

fn ulp(a: f32, b: f32) -> u32 {
    let k = |x: f32| {
        let u = x.to_bits() as i64;
        if u >> 31 == 1 {
            -(u & 0x7fff_ffff)
        } else {
            u
        }
    };
    (k(a) - k(b)).unsigned_abs() as u32
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (dir, name, shell, tile): (&str, &str, f32, u32) =
        (&a[1], &a[2], a[3].parse().unwrap(), a[4].parse().unwrap());
    let (hit, max_evals) = (0.02_f32, 2000_u32);
    let (v, t) = if dir == "syn" {
        synthetic()
    } else {
        (
            f32s(&format!("{dir}/V.bin"))
                .chunks_exact(3)
                .map(|c| Vec3::new(c[0], c[1], c[2]))
                .collect(),
            u32s(&format!("{dir}/T.bin")),
        )
    };
    let started = Instant::now();
    let bvh = MeshBvh::build(&v, &t, 4);
    let flat = flatten(&bvh);
    println!(
        "tris {} nodes {} depth {} data {} KiB build+flatten {:.3} s",
        bvh.triangles.len(),
        flat.nodes.len() / 8,
        flat.depth,
        (flat.nodes.len() * 4 + flat.leaf_tris.len() * 4 + flat.tris.len() * 4) / 1024,
        started.elapsed().as_secs_f64()
    );
    let rays = if name == "grid" {
        grid(&v)
    } else {
        f32s(&format!("{dir}/rays_{name}.bin"))
    };
    let n = (rays.len() / 7) as u32;
    let tracer = Tracer::new(&flat, &rays, shell, hit, max_evals, tile).expect("kernel");
    println!("compile {:.3} s rays {n} tile {tile}", tracer.compile_s);
    let mut first: Option<Vec<RayOut>> = None;
    let mut same = true;
    let runs: usize = std::env::var("MG_RUNS").map_or(5, |v| v.parse().unwrap());
    for run in 0..runs {
        let started = Instant::now();
        let (got, longest) = tracer.trace(n).expect("trace");
        let ms = started.elapsed().as_secs_f64() * 1e3;
        let empty = got.iter().filter(|r| r.evals & 0x8000_0000 != 0).count();
        let misread = got.iter().filter(|r| r.evals & 0x4000_0000 != 0).count();
        if misread > 0 {
            println!("  run {run}: {misread} rays read a word that does not match its sum");
        }
        let state = |p: &str| {
            std::fs::read_to_string(p).map_or(0, |v| v.trim().parse::<u64>().unwrap_or(0))
        };
        println!(
            "  state run {run}: gpu {} MHz, temp {} C",
            state("/sys/devices/platform/gpu.0/devfreq/17000000.gpu/cur_freq") / 1_000_000,
            state("/sys/class/thermal/thermal_zone1/temp") / 1000
        );
        if empty > 0 {
            println!("  run {run}: {empty} rays had a walk that found no triangle");
        }
        let got: Vec<RayOut> = got
            .into_iter()
            .map(|mut r| {
                r.evals &= 0x3fff_ffff;
                r
            })
            .collect();
        println!(
            "run {run}: {ms:.1} ms, longest tile {:.1} ms",
            longest * 1e3
        );
        match &first {
            None => first = Some(got),
            Some(f) => {
                same &= *f == got;
                let differing: Vec<usize> = (0..f.len()).filter(|&i| f[i] != got[i]).collect();
                if !differing.is_empty() {
                    let tiles: std::collections::BTreeSet<usize> =
                        differing.iter().map(|i| i / tile as usize).collect();
                    println!(
                        "  run {run} differs from run 0 at {} rays in {} tiles",
                        differing.len(),
                        tiles.len()
                    );
                    for &i in differing.iter().take(4) {
                        println!(
                            "    ray {i} tile {} pos {}: {:?} ({:08x}) vs {:?} ({:08x})",
                            i / tile as usize,
                            i % tile as usize,
                            f[i],
                            f[i].s.to_bits(),
                            got[i],
                            got[i].s.to_bits()
                        );
                    }
                }
            }
        }
    }
    let gpu = first.unwrap();
    println!("bits equal across runs: {same}");
    let started = Instant::now();
    let c = cpu(&bvh, &rays, shell, hit, max_evals);
    println!(
        "cpu {:.1} ms ({} threads)",
        started.elapsed().as_secs_f64() * 1e3,
        rayon::current_num_threads()
    );
    let (mut hits, mut hit_diff, mut tri_diff, mut ev_diff, mut s_diff, mut most, mut most_mm) =
        (0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0.0_f32);
    let (mut evals, mut max_e) = (0u64, 0u32);
    let mut ulps = Vec::new();
    for (g, c) in gpu.iter().zip(&c) {
        evals += u64::from(c.evals);
        max_e = max_e.max(g.evals);
        hits += c.hit;
        hit_diff += u32::from(g.hit != c.hit);
        tri_diff += u32::from(g.tri != c.tri);
        ev_diff += u32::from(g.evals != c.evals);
        if g.s.to_bits() != c.s.to_bits() {
            s_diff += 1;
        }
        if g.hit == 1 && c.hit == 1 {
            let u = ulp(g.s, c.s);
            most = most.max(u);
            most_mm = most_mm.max((g.s - c.s).abs());
            ulps.push(u);
        }
    }
    ulps.sort_unstable();
    let p99 = ulps.get(ulps.len() * 99 / 100).copied().unwrap_or(0);
    println!(
        "cpu hits {hits}, evals/ray {:.2}, gpu max evals {max_e}",
        evals as f64 / f64::from(n)
    );
    println!("gpu vs cpu of {n}: hit differs {hit_diff}, tri differs {tri_diff}, evals differ {ev_diff}, s bits differ {s_diff}, s ulp p99 {p99} max {most}, s max diff {most_mm}");
    println!("ALL-DONE");
}
