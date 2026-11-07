//! The shell first hit (`MeshBvh::shell_first_hit`, and a brute-force scan of
//! `triangle_shell_hit`) against an f64 truth that shares no code with them
//! (`common/shell_oracle.rs`: Ericson's closest point, a minimum over every triangle, sphere traced)
//!
//! Scenes: a soup rich in collinear, zero-length-edge and sliver triangles, a closed sphere, and
//! the ray that once took a face hit 1.69 away from a near-collinear triangle.
//!
//! Author: Moroya Sakamoto

#[path = "common/shell_oracle.rs"]
mod shell_oracle;
#[path = "common/shell_scenes.rs"]
mod shell_scenes;

use alice_sdf::mesh::{triangle_shell_hit, MeshBvh, ShellFrame, ShellRay};
use glam::Vec3;
use shell_scenes::{grazing, soup, sphere, view};

fn brute(bvh: &MeshBvh, ray: &ShellRay, r: f32) -> Option<f32> {
    let mut best: Option<f32> = None;
    for t in &bvh.triangles {
        if let Some((s, _)) = triangle_shell_hit(ray.origin, ray.dir, t, r) {
            if s <= ray.tmax && best.is_none_or(|b| s < b) {
                best = Some(s);
            }
        }
    }
    best
}

/// Judge the BVH and the brute force on every ray; returns (rays judged, hits, failures)
fn run(
    name: &str,
    v: &[Vec3],
    ix: &[u32],
    r: f32,
    rays: &[ShellRay],
) -> (usize, usize, Vec<String>) {
    let bvh = MeshBvh::build(v, ix, 4);
    let truth = shell_oracle::triangles(v, ix);
    let (mut hits, mut bad) = (0, Vec::new());
    for (i, ray) in rays.iter().enumerate() {
        // each frame's answer as t_base + t_local in f64 (one f32 sum would round it again)
        let walk = |frame| {
            bvh.shell_first_hit_in(frame, ray.origin, ray.dir, r, ray.tmax)
                .map(|h| f64::from(h.t_base) + f64::from(h.t_local))
        };
        let world = walk(ShellFrame::World);
        hits += world.is_some() as usize;
        for (method, got) in [
            ("bvh world", world),
            ("bvh leaf", walk(ShellFrame::Leaf)),
            ("bvh triangle", walk(ShellFrame::Triangle)),
            ("brute", brute(&bvh, ray, r).map(f64::from)),
        ] {
            if let Some(why) = shell_oracle::judge(&truth, ray.origin, ray.dir, ray.tmax, r, got) {
                bad.push(format!("{name} ray {i} {method} {got:?}: {why}"));
            }
        }
    }
    (rays.len(), hits, bad)
}

#[test]
fn a_near_collinear_triangle_takes_no_face_hit_far_from_it() {
    // the triangle (a soup case `third = (a + b) * 0.5`, normal length 4e-9 after rounding) and the
    // ray that took a "face hit" 1.69 from it at t = 0.768
    let v = [
        Vec3::new(0.43042785, -0.105567336, 0.8426804),
        Vec3::new(0.5522124, -0.06037624, 0.53305507),
        Vec3::new(0.49132013, -0.08297179, 0.68786776),
    ];
    let ray = ShellRay {
        origin: Vec3::new(0.0, 0.0, 3.0),
        dir: Vec3::new(-0.3364487, -0.45972764, -0.8218594),
        tmax: 100.0,
    };
    let (n, _, bad) = run("collinear", &v, &[0, 1, 2], 0.01, &[ray]);
    assert_eq!(n, 1);
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn soup_and_sphere_agree_with_the_f64_truth() {
    let (sv, si) = sphere(16, 32);
    let (pv, pi) = soup(400, 0x2545F4914F6CDD1D);
    let eyes = [
        Vec3::new(0., 0., 3.),
        Vec3::new(3., 0., 0.),
        Vec3::new(0., 3., 0.01),
        Vec3::new(2., 1.5, 2.),
    ];
    let mut total = (0, 0);
    let mut bad = Vec::new();
    for (name, v, ix, r) in [("soup", &pv, &pi, 0.01f32), ("sphere", &sv, &si, 0.002)] {
        let mut rays: Vec<ShellRay> = eyes.iter().flat_map(|e| view(*e, 20)).collect();
        rays.extend(grazing(v, ix, r).into_iter().take(150));
        let (n, h, b) = run(name, v, ix, r, &rays);
        println!("{name}: {n} rays, {h} hits, {} failures", b.len());
        total.0 += n;
        total.1 += h;
        bad.extend(b);
    }
    // the truth judged something, and both outcomes occurred
    assert!(
        total.0 > 3000 && total.1 > 500 && total.1 < total.0,
        "{total:?}"
    );
    assert!(
        bad.is_empty(),
        "{} failures, first: {:#?}",
        bad.len(),
        &bad[..bad.len().min(6)]
    );
}

/// Long thin triangles (40-80 x 1-4 mm) about 1400 mm from the origin and rays from 900-1400 mm
/// back, aimed at points of the triangles: the scale where an f32 world coordinate's ulp is a
/// sizeable part of a 0.002 mm shell. Each frame's hits against the f64 distance.
#[test]
fn far_thin_triangles_in_each_frame() {
    let mut x = 0x5EED_F00Du64;
    let mut u = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 40) as f32 / (1u64 << 24) as f32
    };
    let r = 0.002f32;
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
    let bvh = MeshBvh::build(&v, &ix, 4);
    let truth = shell_oracle::triangles(&v, &ix);
    for t in &bvh.triangles {
        for _ in 0..25 {
            let (a, b0) = (u(), u());
            let p = t.v0 + (t.v1 - t.v0) * a + (t.v2 - t.v0) * (b0 * (1.0 - a));
            let n = (t.v1 - t.v0).cross(t.v2 - t.v0).normalize();
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
    let mut report = Vec::new();
    for frame in [ShellFrame::World, ShellFrame::Leaf, ShellFrame::Triangle] {
        let (mut worst, mut hits, mut bad) = (0f64, 0, Vec::new());
        for (i, ray) in aims.iter().enumerate() {
            let h = bvh.shell_first_hit_in(frame, ray.origin, ray.dir, r, ray.tmax);
            let got = h.map(|h| f64::from(h.t_base) + f64::from(h.t_local));
            if let Some(t) = got {
                hits += 1;
                let at = ray.origin.as_dvec3() + ray.dir.as_dvec3() * t;
                worst = worst.max((shell_oracle::distance(&truth, at) - f64::from(r)).abs());
            }
            if let Some(why) = shell_oracle::judge(&truth, ray.origin, ray.dir, ray.tmax, r, got) {
                bad.push(format!("{frame:?} ray {i}: {why}"));
            }
        }
        println!(
            "far thin, {frame:?}: {hits} hits of {}, worst |d - r| {worst:.3e} ({:.2}% of r)",
            aims.len(),
            worst / f64::from(r) * 100.0
        );
        report.push((frame, hits, worst));
        assert!(
            bad.is_empty(),
            "{} failures, first: {:#?}",
            bad.len(),
            &bad[..bad.len().min(4)]
        );
    }
    assert!(
        report.iter().all(|(_, h, _)| *h * 10 > aims.len() * 9),
        "{report:?}"
    );
}
