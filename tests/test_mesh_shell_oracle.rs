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

use alice_sdf::mesh::{triangle_shell_hit, MeshBvh, ShellRay};
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
        let walk = bvh
            .shell_first_hit(ray.origin, ray.dir, r, ray.tmax)
            .map(|h| h.t);
        hits += walk.is_some() as usize;
        for (method, got) in [("bvh", walk), ("brute", brute(&bvh, ray, r))] {
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
