//! A shell ray that starts at `tmin` instead of 0 (red until `ShellRay::tmin` and
//! `MeshBvh::shell_first_hit_between` exist).
//!
//! Moving the origin to `o + s·d` and rounding it to f32 changes the ray: grazing rays far from
//! the mesh then hit another triangle or another layer. A start parameter keeps the origin's bits.
//!
//! oracle, two parts:
//! - a start before the first hit changes nothing: the hit with `tmin = s < t₀` is the hit with
//!   `tmin = 0`, every field bit for bit, on the CPU and on the GPU (the same ray, the same
//!   arithmetic, only hits with `t < tmin` are dropped)
//! - a start past the first shell finds the next one: the f64 trace of the shell from the point at
//!   `tmin` (`shell_oracle::judge`, with its derived η) accepts the answer, and `t ≥ tmin`
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gpu")]

use alice_sdf::mesh::{MeshBvh, ShellBackend, ShellRay, ShellRenderer};
use glam::Vec3;

#[path = "common/shell_oracle.rs"]
mod shell_oracle;
#[path = "common/shell_scenes.rs"]
mod shell_scenes;

const R: f32 = 0.01;

fn scene() -> (MeshBvh, Vec<Vec3>, Vec<u32>, Vec<ShellRay>) {
    let (v, ix) = shell_scenes::sphere(48, 96);
    let bvh = MeshBvh::build(&v, &ix, 4);
    let mut rays = shell_scenes::view(Vec3::new(0.3, 0.2, 3.5), 64);
    rays.extend(shell_scenes::view(Vec3::new(-2.6, 1.4, 1.9), 64));
    (bvh, v, ix, rays)
}

#[test]
fn a_start_before_the_first_hit_changes_nothing_on_the_cpu() {
    let (bvh, _, _, rays) = scene();
    let mut compared = 0usize;
    for ray in &rays {
        let Some(h0) = bvh.shell_first_hit(ray.origin, ray.dir, R, ray.tmax) else {
            continue;
        };
        for s in [h0.t * 0.5, h0.t * 0.999] {
            let h = bvh.shell_first_hit_between(ray.origin, ray.dir, R, s, ray.tmax);
            assert_eq!(h, Some(h0), "tmin {s} before the hit at {}", h0.t);
            compared += 1;
        }
    }
    assert!(compared > 1000, "only {compared} rays compared");
}

#[test]
fn a_start_before_the_first_hit_changes_nothing_on_the_gpu() {
    let (bvh, _, _, rays) = scene();
    let renderer = ShellRenderer::new(&bvh, R).expect("renderer");
    if renderer.backend() == ShellBackend::Cpu {
        assert!(
            std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
            "ALICE_SDF_REQUIRE_GPU is set but no GPU device was found"
        );
        eprintln!("skipped (no GPU device)");
        return;
    }
    let base = renderer.first_hits(&rays).expect("gpu");
    let started: Vec<ShellRay> = rays
        .iter()
        .zip(&base)
        .map(|(r, h)| ShellRay {
            tmin: h.map_or(0.0, |h| h.t * 0.5),
            ..*r
        })
        .collect();
    let again = renderer.first_hits(&started).expect("gpu");
    let compared = base.iter().filter(|h| h.is_some()).count();
    assert!(compared > 1000, "only {compared} rays hit");
    assert_eq!(base, again);
}

#[test]
fn a_start_past_the_first_shell_finds_the_next_one() {
    let (bvh, v, ix, rays) = scene();
    let tris = shell_oracle::triangles(&v, &ix);
    let (mut judged, mut wrong) = (0usize, Vec::new());
    for ray in rays.iter().step_by(7) {
        let Some(h0) = bvh.shell_first_hit(ray.origin, ray.dir, R, ray.tmax) else {
            continue;
        };
        // past the front layer (its thickness is 2r along the normal; 4r along the ray is beyond
        // it for the view's angles)
        let s = h0.t + 4.0 * R / ray.dir.length();
        let got = bvh.shell_first_hit_between(ray.origin, ray.dir, R, s, ray.tmax);
        if let Some(h) = got {
            assert!(h.t >= s, "a hit at {} before tmin {s}", h.t);
        }
        let start = ray.origin + ray.dir * s;
        if let Some(why) = shell_oracle::judge(
            &tris,
            start,
            ray.dir,
            ray.tmax - s,
            R,
            got.map(|h| f64::from(h.t) - f64::from(s)),
        ) {
            wrong.push(format!("ray from {} tmin {s}: {why}", ray.origin));
        }
        judged += 1;
    }
    assert!(judged > 300, "only {judged} rays judged");
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
