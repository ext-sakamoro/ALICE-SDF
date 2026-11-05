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

mod common;

use alice_sdf::eval::{eval, eval_scaled};
use alice_sdf::interval::eval_lipschitz;
use alice_sdf::raycast::{raymarch, raymarch_detailed, raymarch_with_config, RaymarchConfig};
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
    // The same holds when the warp is one operand of a union or of a smooth union with a sphere
    // far from the rays (the value there is the warp's, so its scale must carry through).
    let far_ball = || SdfNode::sphere(0.3).translate(0.5, 0.05, 2.0);
    let scenes = [
        ("warp", squeezed()),
        ("union", squeezed().union(far_ball())),
        ("smooth union", squeezed().smooth_union(far_ball(), 0.05)),
    ];
    let eps = RaymarchConfig::default().epsilon;
    let mut checked = 0;
    let mut far = Vec::new();
    for (name, node) in &scenes {
        for k in 0..12 {
            let h = eps * (1.5 + 7.5 * k as f32 / 11.0);
            for a in 0..9 {
                let ang = (a as f32 - 4.0) * 0.012;
                let (sn, cs) = alice_det_math::sin_cos(ang);
                let d = Vec3::new(cs, sn, 0.0);
                let o = Vec3::new(0.15, 0.05 - 0.3 * sn, 0.02 + h);
                if let Some(hit) = raymarch(node, o, d, 0.7) {
                    let dist = posed_box(hit.point);
                    if dist > eps * 1.05 + 1e-6 {
                        far.push((*name, h, ang, dist));
                    }
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 3 * 108);
    assert!(
        far.is_empty(),
        "{} of {checked} rays hit farther than μ·ε = {eps} from the shape, e.g. (scene, height, angle, distance) {:?}",
        far.len(),
        &far[..far.len().min(4)]
    );
}

#[test]
fn every_ray_the_shape_meets_is_hit_where_it_meets_it() {
    // Random rays from around the posed box toward points near it; the reference marches the
    // closed-form box in steps of 2e-4 and bisects. The tracer must hit every ray the reference
    // hits, and not past where the reference meets it (a hit before it is a near-miss, the test
    // above); the guard below needs some of each kind of ray. The budget is raised: the stricter
    // band costs steps near the surface (the field is the distance / Λ, so each step closes about
    // 1 / Λ of what is left), which is a cost, not a hole; the default budget is what `raymarch`
    // builds, with more steps.
    let node = squeezed();
    let mut config = RaymarchConfig::default().with_bound(eval_lipschitz(&node));
    config.max_steps = 4096;
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
        let traced = raymarch_with_config(&node, o, d, 1.0, &config);
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

#[test]
fn the_value_is_eval_bit_for_bit_and_the_scale_is_one_without_a_local_factor() {
    // d must be `eval` exactly on every tree (the step and every golden built on it stay), and a
    // tree with no node that divides or multiplies by a local factor (grid warp, non-uniform
    // scale, lattice deform, projective transform) must have s = 1 (its hit test is unchanged).
    let factor = |name: &str| {
        name.starts_with("scale_xyz")
            || name.starts_with("projective")
            || name.starts_with("lattice")
            || name.starts_with("grid_warp")
    };
    let mut trees = common::corpus::corpus();
    trees.push((
        "projective_nonidentity",
        common::corpus::projective_nonidentity(),
    ));
    trees.push(("grid_warp_squeezed", squeezed()));
    let mut st = 0x9e37_79b9u32;
    let (mut compared, mut scaled_seen) = (0usize, 0usize);
    for (name, node) in &trees {
        for _ in 0..64 {
            let p = Vec3::new(
                -1.5 + 3.0 * rng(&mut st),
                -1.5 + 3.0 * rng(&mut st),
                -1.5 + 3.0 * rng(&mut st),
            );
            let (d, scale) = eval_scaled(node, p);
            assert_eq!(
                d.to_bits(),
                eval(node, p).to_bits(),
                "{name} at {p:?}: {d} vs eval {}",
                eval(node, p)
            );
            assert!(
                scale.is_finite() && scale > 0.0,
                "{name} at {p:?}: scale {scale}"
            );
            if factor(name) {
                if scale != 1.0 {
                    scaled_seen += 1;
                }
            } else {
                assert_eq!(
                    scale, 1.0,
                    "{name} at {p:?}: no local factor, scale {scale}"
                );
            }
            compared += 1;
        }
    }
    assert!(compared > 90 * 64, "{compared}");
    assert!(
        scaled_seen > 20,
        "the factor nodes never gave a scale: {scaled_seen}"
    );
}

#[test]
fn inside_a_blend_the_scale_is_the_smaller_operands() {
    // Where a smooth union mixes both operands (the value is neither one's), the scale is the
    // smaller of the two: the filter never gets stricter than either operand's own, so the blend
    // costs no more steps than its parts (any finite scale keeps every surface; this pins the
    // rule, which rays alone cannot observe).
    let warp = squeezed();
    let ball = SdfNode::sphere(0.05).translate(0.5, 0.05, 0.1);
    let node = warp.clone().smooth_union(ball.clone(), 0.2);
    let mut st = 0x1234_5678u32;
    let mut inside = 0;
    for _ in 0..400 {
        let p = Vec3::new(
            0.35 + 0.3 * rng(&mut st),
            -0.05 + 0.2 * rng(&mut st),
            0.0 + 0.12 * rng(&mut st),
        );
        let ((da, sa), (db, sb)) = (eval_scaled(&warp, p), eval_scaled(&ball, p));
        let (v, s) = eval_scaled(&node, p);
        if v.abs() != da.abs() && v.abs() != db.abs() {
            assert_eq!(s, sa.min(sb), "at {p:?}: {s} vs operands {sa}, {sb}");
            if sa != sb {
                inside += 1;
            }
        }
    }
    assert!(
        inside > 50,
        "{inside} points in the blend with different operand scales"
    );
}

#[test]
fn a_warped_scene_needs_more_steps_and_running_out_is_reported() {
    // The cost of the stricter hit test, measured and pinned: in the squeezed scene every ray the
    // closed-form box meets is hit by step 297 (a budget of 298), more than `raymarch` gives (257
    // after `with_bound`); with that budget some rays run out, and each one that is not hit says
    // so (`RaymarchResult::ran_out`) rather than looking like a ray that met nothing.
    let node = squeezed();
    let default = RaymarchConfig::default().with_bound(eval_lipschitz(&node));
    let mut enough = default;
    enough.max_steps = 298;
    let mut s = 0x2545_f491u32;
    let (mut met, mut ran_out, mut silent, mut short, mut clean) = (0, 0, 0, 0, 0);
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
        if reference_hit(o, d, 1.0, 2e-4).is_none() {
            // a ray that meets nothing is a miss, not an unresolved ray
            let r = raymarch_detailed(&node, o, d, 1.0, &default);
            if !r.hit {
                assert!(
                    !r.ran_out(&default, 1.0),
                    "a clean miss reported as running out"
                );
                clean += 1;
            }
            continue;
        }
        met += 1;
        if !raymarch_detailed(&node, o, d, 1.0, &enough).hit {
            short += 1;
        }
        let r = raymarch_detailed(&node, o, d, 1.0, &default);
        if !r.hit {
            if r.ran_out(&default, 1.0) {
                ran_out += 1;
            } else {
                silent += 1;
            }
        }
    }
    assert!(met > 60 && clean > 20, "met {met}, clean misses {clean}");
    assert_eq!(short, 0, "{short} met rays not hit with a budget of 298");
    assert!(
        ran_out > 0,
        "the default budget was enough: the cost is not pinned"
    );
    assert_eq!(silent, 0, "{silent} met rays missed without running out");
}
