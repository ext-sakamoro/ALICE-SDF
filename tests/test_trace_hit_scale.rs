//! The tracer's hit test in the field's own units.
//!
//! A field divided by a local slope bound (a grid warp's Λ) is a sound step, but it is no longer a
//! distance at the scale the hit threshold is set for: a ray passing at height h over a surface
//! whose warp compresses by Λ elsewhere sees a value h / Λ and reports a hit while still h away.
//! These tests fix what the hit test must keep and what it must stop:
//!
//! - no holes: every ray that a dense fixed-step march of the closed-form shape hits, the tracer
//!   hits, and not past where the shape is met (the hit rule may only reject near-misses, never real
//!   surfaces);
//! - no near-misses: where the warp is not folded, a hit point lies within μ·ε of the shape
//!   (μ = the largest stretch of the rest → posed map, ε = the tracer's threshold).
//!
//! The scene is closed form: a sheet over [0, 1]² posed by squeezing y to a tenth, so the posed →
//! rest map is diag(1, 10, 1) (Λ ≈ 10) and the rest → posed map is diag(1, 0.1, 1) (μ = 1). The
//! child is the slab [0, 1]² × [−0.02, 0.02]; posed, it is the box [0, 1] × [0, 0.1] × [−0.02, 0.02].
//!
//! Every evaluation goes through `SdfNode::grid_warp` and `raymarch`, the production path.
//!
//! Author: Moroya Sakamoto

use alice_sdf::raycast::{raymarch, RaymarchConfig};
use alice_sdf::transforms::{grid_warp_from_triangles, GridWarp, WarpGrid};
use alice_sdf::SdfNode;
use glam::Vec3;

/// A flat n × n sheet in the plane z = 0 over [0, 1]², two triangles per cell.
fn sheet(n: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut v = Vec::new();
    for j in 0..n {
        for i in 0..n {
            v.push(Vec3::new(
                i as f32 / (n - 1) as f32,
                j as f32 / (n - 1) as f32,
                0.0,
            ));
        }
    }
    let mut t = Vec::new();
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let a = (j * n + i) as u32;
            let (b, c, d) = (a + 1, a + n as u32, a + n as u32 + 1);
            t.push([a, b, d]);
            t.push([a, d, c]);
        }
    }
    (v, t)
}

/// The slab on a sheet squeezed to a tenth in y.
fn squeezed() -> SdfNode {
    let (rest, tris) = sheet(9);
    let posed: Vec<Vec3> = rest
        .iter()
        .map(|p| Vec3::new(p.x, 0.1 * p.y, p.z))
        .collect();
    let grid = WarpGrid {
        box_min: Vec3::new(-0.3, -0.3, -0.3),
        box_max: Vec3::new(1.3, 0.4, 0.3),
        cell: 0.025,
        reach: 0.3,
        shell: None,
        overhang: None,
    };
    let warp =
        GridWarp::new(grid_warp_from_triangles(&rest, &posed, &tris, &grid).expect("builds"))
            .expect("valid");
    SdfNode::box3d(1.0, 1.0, 0.04)
        .translate(0.5, 0.5, 0.0)
        .grid_warp(warp)
}

/// The posed shape's distance, closed form: the box [0, 1] × [0, 0.1] × [−0.02, 0.02].
fn posed_box(p: Vec3) -> f32 {
    let q = (p - Vec3::new(0.5, 0.05, 0.0)).abs() - Vec3::new(0.5, 0.05, 0.02);
    q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
}

/// The first t where the closed-form box is reached, by fixed steps of `h` (then bisected).
fn reference_hit(o: Vec3, d: Vec3, t_max: f32, h: f32) -> Option<f32> {
    if posed_box(o) <= 0.0 {
        return Some(0.0);
    }
    let steps = (t_max / h).ceil() as u32;
    for k in 0..steps {
        let (t, t1) = (k as f32 * h, (k + 1) as f32 * h);
        if posed_box(o + d * t1) <= 0.0 {
            let (mut a, mut b) = (t, t1);
            for _ in 0..40 {
                let m = f32::midpoint(a, b);
                if posed_box(o + d * m) <= 0.0 {
                    b = m;
                } else {
                    a = m;
                }
            }
            return Some(b);
        }
    }
    None
}

/// Deterministic pseudo-random numbers in [0, 1) (xorshift32).
fn rng(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    (*state >> 8) as f32 / (1u32 << 24) as f32
}

#[test]
fn a_ray_passing_over_a_squeezed_warp_does_not_hit_farther_than_mu_eps() {
    // Rays parallel to the slab's top face at heights from 1.5 ε to 9 ε above it, along x and at
    // angles in the face's plane. The squeeze divides the value there by Λ ≈ 10 and leaves the
    // distance as is (the face's normal is kept), so a hit test on the raw value reports these as
    // hits up to 10 ε away. μ = 1 here: a hit must be within ε (+ a small tolerance) of the box.
    let node = squeezed();
    let eps = RaymarchConfig::default().epsilon;
    let mut checked = 0;
    let mut far = Vec::new();
    for k in 0..12 {
        let h = eps * (1.5 + 7.5 * k as f32 / 11.0);
        for a in 0..9 {
            let ang = (a as f32 - 4.0) * 0.012;
            let (sn, cs) = alice_det_math::sin_cos(ang);
            let d = Vec3::new(cs, sn, 0.0);
            let o = Vec3::new(0.15, 0.05 - 0.3 * sn, 0.02 + h);
            if let Some(hit) = raymarch(&node, o, d, 0.7) {
                let dist = posed_box(hit.point);
                if dist > eps * 1.05 + 1e-6 {
                    far.push((h, ang, dist));
                }
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 108);
    assert!(
        far.is_empty(),
        "{} of {checked} rays hit farther than μ·ε = {eps} from the shape, e.g. (height, angle, distance) {:?}",
        far.len(),
        &far[..far.len().min(4)]
    );
}

#[test]
fn every_ray_the_shape_meets_is_hit_where_it_meets_it() {
    // Random rays from around the posed box toward points near it; the reference marches the
    // closed-form box in steps of 2e-4 and bisects. The tracer must hit every ray the reference
    // hits, and not past where the reference meets it (a hit before it is a near-miss, the test
    // above); the guard below needs some of each kind of ray.
    let node = squeezed();
    let mut s = 0x2545_f491u32;
    let (mut met, mut missed_by_both, mut checked) = (0, 0, 0);
    let mut holes = Vec::new();
    for _ in 0..240 {
        let o = Vec3::new(
            -0.2 + 1.4 * rng(&mut s),
            -0.2 + 0.5 * rng(&mut s),
            if rng(&mut s) < 0.5 { -0.25 } else { 0.25 },
        );
        let target = Vec3::new(
            -0.1 + 1.2 * rng(&mut s),
            -0.05 + 0.2 * rng(&mut s),
            -0.03 + 0.06 * rng(&mut s),
        );
        let d = (target - o).normalize();
        let reference = reference_hit(o, d, 1.0, 2e-4);
        let traced = raymarch(&node, o, d, 1.0);
        checked += 1;
        match (reference, traced) {
            (Some(t), Some(hit)) => {
                met += 1;
                // a hole is a hit past the surface (stepping through it); an early hit is a
                // near-miss, which the test above owns
                if hit.distance > t + 2e-3 {
                    holes.push((o, d, t, Some(hit.distance)));
                }
            }
            (Some(t), None) => holes.push((o, d, t, None)),
            (None, _) => missed_by_both += 1,
        }
    }
    assert_eq!(checked, 240);
    assert!(
        met > 60 && missed_by_both > 20,
        "met {met}, missed {missed_by_both}"
    );
    assert!(
        holes.is_empty(),
        "{} rays the shape meets were not hit there, e.g. (origin, dir, reference t, traced t) {:?}",
        holes.len(),
        &holes[..holes.len().min(3)]
    );
}
