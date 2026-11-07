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

#[path = "common/shell_margin.rs"]
mod shell_margin;
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

/// The ambiguous band per view may fall but not rise: `tests/data/shell_band_ratchet.txt` holds the
/// largest count each view may have (`<mesh> <view>: <count>`). A view with no line fails.
fn ratchet(view: &str, count: usize, bad: &mut Vec<String>) {
    let text = include_str!("data/shell_band_ratchet.txt");
    let pinned = text.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        (k.trim() == view)
            .then(|| v.trim().parse::<usize>().ok())
            .flatten()
    });
    match pinned {
        None => bad.push(format!("{view}: no line in the ratchet (band {count})")),
        Some(p) if count > p => bad.push(format!("{view}: band {count} rose above {p}")),
        Some(p) if count < p => {
            println!("  ratchet: {view} band {count}, below {p}: lower the line")
        }
        Some(_) => {}
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
    let (mut outside, mut over_bound, mut worst_ratio) = (0usize, 0usize, 0f64);
    let mut outside_why = Vec::<String>::new();
    let mut ratchet_bad = Vec::<String>::new();
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
            // the ambiguous band, from the WGSL accuracy rules (common/shell_margin.rs): outside it the
            // hit, triangle and piece are the same and t lies within its derived bound
            let (mut in_band, mut differ_in_band) = (0usize, 0usize);
            let mut band_rays = Vec::new();
            for (k, ray) in rays.iter().enumerate() {
                let band = shell_margin::band(bvh, ray, *r);
                in_band += band.is_some() as usize;
                if band.is_some() {
                    band_rays.push(k);
                }
                let (g, c) = (gpu[k], cpu[k]);
                let same = match (g, c) {
                    (Some(g), Some(c)) => g.triangle == c.triangle && g.feature == c.feature,
                    (None, None) => true,
                    _ => false,
                };
                if !same {
                    if band.is_some() {
                        differ_in_band += 1;
                    } else {
                        outside += 1;
                        if outside_why.len() < 6 {
                            outside_why
                                .push(format!("{name} {label} ray {k}: gpu {g:?} cpu {c:?}"));
                        }
                    }
                    continue;
                }
                if let (Some(g), Some(c)) = (g, c) {
                    if g.t.to_bits() != c.t.to_bits() && band.is_none() {
                        let bound =
                            shell_margin::t_bound(bvh, ray, *r, c.triangle, c.feature as u32, c.t)
                                .expect("the CPU's answer is one of its pieces");
                        let diff = f64::from((g.t - c.t).abs());
                        worst_ratio = worst_ratio.max(diff / bound.max(f64::MIN_POSITIVE));
                        if diff > bound {
                            over_bound += 1;
                            if outside_why.len() < 6 {
                                outside_why.push(format!("{name} {label} ray {k}: t {} vs {} differ by {diff:e}, bound {bound:e}", g.t, c.t));
                            }
                        }
                    }
                }
            }
            println!(
                "  band {in_band} of {} rays, {differ_in_band} of them differ",
                rays.len()
            );
            ratchet(&format!("{name} {label}"), in_band, &mut ratchet_bad);
            // the f64 truth (no shared code): the GPU and the CPU on every band ray, where they may
            // differ, and the GPU on every 97th ray besides (the truth scans every triangle at every step)
            let truth = shell_oracle::triangles(v, ix);
            let sample = (0..rays.len()).step_by(97).map(|k| (k, false));
            let band_both = band_rays.iter().flat_map(|&k| [(k, false), (k, true)]);
            for (k, on_cpu) in sample.chain(band_both) {
                let ray = rays[k];
                let got = if on_cpu { cpu[k] } else { gpu[k] };
                judged += 1;
                if let Some(why) =
                    shell_oracle::judge(&truth, ray.origin, ray.dir, ray.tmax, *r, got.map(|h| h.t))
                {
                    if wrong.len() < 6 {
                        let side = if on_cpu { "cpu" } else { "gpu" };
                        wrong.push(format!("{name} {label} ray {k} {side}: {why}"));
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
    println!(
        "outside the band: {outside} differ, {over_bound} t over the bound, worst t / bound {worst_ratio:.3}"
    );
    assert_eq!((outside, over_bound), (0, 0), "{outside_why:#?}");
    assert!(ratchet_bad.is_empty(), "{ratchet_bad:#?}");
}
