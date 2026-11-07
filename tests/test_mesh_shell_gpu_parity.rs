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

#[path = "common/shell_oracle.rs"]
mod shell_oracle;
#[path = "common/shell_scenes.rs"]
mod shell_scenes;
use shell_scenes::{grazing, soup, sphere, view};

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
    let (mut judged, mut wrong_n, mut wrong) = (0usize, 0usize, Vec::<String>::new());
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
            // the GPU against the f64 truth (no shared code) on every 97th ray (the truth scans every triangle at every step)
            let truth = shell_oracle::triangles(v, ix);
            for (k, ray) in rays.iter().enumerate().step_by(97) {
                judged += 1;
                if let Some(why) = shell_oracle::judge(
                    &truth,
                    ray.origin,
                    ray.dir,
                    ray.tmax,
                    *r,
                    gpu[k].map(|h| h.t),
                ) {
                    if wrong.len() < 6 {
                        wrong.push(format!("{name} {label} ray {k}: {why}"));
                    }
                    wrong_n += 1;
                }
            }
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
    println!("gpu against the f64 truth: {judged} rays judged, {wrong_n} wrong {wrong:#?}");
    assert!(judged > 1000, "judged too few: {judged}");
    assert_eq!(
        wrong_n, 0,
        "the GPU disagrees with the f64 truth: {wrong:#?}"
    );
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
