//! Loft: closed sections swept along a spline axis, the section interpolated between stations.
//!
//! A section is the polar curve r(θ) = a0 + Σ (a_k cos kθ + b_k sin kθ); along the axis the
//! coefficient rows are a uniform cubic B-spline (end rows reflected, so the end rows are met).
//!
//! Oracles, each independent of the implementation:
//! - a straight axis with a constant circle is the capped cylinder: its closed form, exactly;
//! - a straight axis whose circle grows linearly is the capped cone: the sign agrees with the
//!   cone's closed form, and outside `D / L ≤ f ≤ D` with `L = 1 + 1.1·slope` (the declared bound
//!   for a straight axis: the field never overshoots the surface);
//! - a sphere as a loft (rows sampled from √(R² − y²)): the field vanishes on the sphere within
//!   the B-spline's smoothing, and has the sphere's sign just inside and outside;
//! - on a curved axis with sections that change size and shape (cos 2θ, sin θ, sin 3θ terms),
//!   the field never grows faster than 1 per unit length (finite differences over random pairs,
//!   points next to the axis included) and `eval_lipschitz` says 1;
//! - the sign: points on the axis are negative, points far away positive;
//! - every malformed input is refused by name;
//! - the compiled bytecode gives the same value as the tree, bit for bit.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm serves as an independent reference"
)]

use alice_sdf::interval::eval_lipschitz;
use alice_sdf::prelude::*;
use glam::Vec2;

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// The capped cylinder along +y from y = 0 to h, radius r: closed form
fn cylinder_reference(p: Vec3, r: f32, h: f32) -> f32 {
    let d = Vec2::new(Vec2::new(p.x, p.z).length() - r, (p.y - h / 2.0).abs() - h / 2.0);
    d.x.max(d.y).min(0.0) + d.max(Vec2::ZERO).length()
}

/// The capped cone along +y from y = 0 (radius ra) to y = h (radius rb): closed form (exact
/// distance to the solid, by the nearest of the side segment and the two discs in the (ρ, y)
/// half plane)
fn cone_reference(p: Vec3, ra: f32, rb: f32, h: f32) -> f32 {
    let q = Vec2::new(Vec2::new(p.x, p.z).length(), p.y);
    let seg = |a: Vec2, b: Vec2| -> f32 {
        let e = b - a;
        let t = ((q - a).dot(e) / e.dot(e)).clamp(0.0, 1.0);
        (q - (a + e * t)).length()
    };
    let d = seg(Vec2::new(ra, 0.0), Vec2::new(rb, h))
        .min(seg(Vec2::ZERO, Vec2::new(ra, 0.0)))
        .min(seg(Vec2::new(0.0, h), Vec2::new(rb, h)));
    let inside = q.y >= 0.0 && q.y <= h && q.x <= ra + (rb - ra) * q.y / h;
    if inside {
        -d
    } else {
        d
    }
}

/// A coefficient row of a circle of radius r with K harmonics (all others zero)
fn circle_row(r: f32, k: usize) -> Vec<f32> {
    let mut v = vec![0.0; 2 * k + 1];
    v[0] = r;
    v
}

/// A straight axis along +y from 0 to h
fn straight_axis(h: f32) -> Vec<Vec3> {
    vec![Vec3::ZERO, Vec3::new(0.0, h, 0.0)]
}

#[test]
fn a_straight_loft_of_a_constant_circle_is_the_capped_cylinder() {
    let (h, r0) = (1.5, 0.3);
    let node = SdfNode::loft(straight_axis(h), Vec3::X, 4, vec![circle_row(r0, 4), circle_row(r0, 4)]).expect("cylinder");
    let mut r = lcg(1);
    let mut worst = 0.0f32;
    for _ in 0..4000 {
        let p = Vec3::new(r() * 1.4 - 0.7, r() * 2.4 - 0.45, r() * 1.4 - 0.7);
        worst = worst.max((eval(&node, p) - cylinder_reference(p, r0, h)).abs());
    }
    assert!(worst < 2e-5, "largest difference {worst}");
}

#[test]
fn a_straight_loft_of_a_linearly_growing_circle_is_bounded_by_the_capped_cone() {
    let (h, ra, rb) = (1.0, 0.2, 0.45);
    let node = SdfNode::loft(straight_axis(h), Vec3::X, 2, vec![circle_row(ra, 2), circle_row(rb, 2)]).expect("cone");
    let l = 1.0 + 1.1 * (rb - ra) / h;
    let mut r = lcg(2);
    let (mut sign_bad, mut over, mut under, mut compared) = (0, 0, 0, 0);
    for _ in 0..4000 {
        let p = Vec3::new(r() * 1.4 - 0.7, r() * 1.8 - 0.4, r() * 1.4 - 0.7);
        let d = cone_reference(p, ra, rb, h);
        let f = eval(&node, p);
        if d.abs() > 1e-3 {
            compared += 1;
            sign_bad += usize::from((f > 0.0) != (d > 0.0));
            if d > 0.0 {
                over += usize::from(f > d + 1e-5);
                under += usize::from(f < d / l - 1e-5);
            }
        }
    }
    assert!(compared > 1000, "compared {compared}");
    assert_eq!((sign_bad, over, under), (0, 0, 0), "sign / over / under among {compared}");
}

#[test]
fn a_sphere_as_a_loft_vanishes_on_the_sphere() {
    // axis from y = −0.9 to 0.9 in 12 even steps, rows √(1 − y²) at the axis points
    let n = 13;
    let ys: Vec<f32> = (0..n).map(|i| -0.9 + 1.8 * i as f32 / (n - 1) as f32).collect();
    let axis: Vec<Vec3> = ys.iter().map(|&y| Vec3::new(0.0, y, 0.0)).collect();
    let rows: Vec<Vec<f32>> = ys.iter().map(|&y| circle_row((1.0 - y * y).sqrt(), 1)).collect();
    let node = SdfNode::loft(axis, Vec3::X, 1, rows).expect("sphere");
    let mut r = lcg(4);
    let (mut worst, mut sign_bad, mut compared) = (0.0f32, 0, 0);
    for _ in 0..3000 {
        let dir = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5).normalize();
        if dir.y.abs() > 0.75 {
            continue;
        }
        compared += 1;
        worst = worst.max(eval(&node, dir).abs());
        sign_bad += usize::from(eval(&node, dir * 1.03) <= 0.0) + usize::from(eval(&node, dir * 0.97) >= 0.0);
    }
    assert!(compared > 1000, "compared {compared}");
    assert!(worst < 1e-2, "largest |f| on the sphere {worst}");
    assert_eq!(sign_bad, 0, "wrong sign 3 % off the sphere among {compared}");
}

/// A curved axis with sections that grow, shrink, squash and lean
fn curved() -> SdfNode {
    let axis = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.05, 0.3, 0.02),
        Vec3::new(0.12, 0.6, 0.0),
        Vec3::new(0.15, 0.9, -0.05),
    ];
    // [a0, a1, b1, a2, b2, a3, b3]
    let rows = vec![
        vec![0.10, 0.0, 0.01, 0.02, 0.0, 0.0, 0.004],
        vec![0.09, 0.005, 0.0, -0.015, 0.0, 0.0, 0.0],
        vec![0.08, 0.0, -0.01, 0.01, 0.005, 0.003, 0.0],
        vec![0.06, 0.0, 0.0, 0.005, 0.0, 0.0, -0.002],
    ];
    SdfNode::loft(axis, Vec3::Z, 3, rows).expect("a curved loft")
}

#[test]
fn the_field_is_one_lipschitz_on_a_curved_varying_loft() {
    let node = curved();
    let l = eval_lipschitz(&node);
    assert!(l.is_finite() && l <= 1.0, "eval_lipschitz {l}");
    let mut r = lcg(3);
    let mut worst = 0.0f32;
    for i in 0..20000 {
        // every fourth pair next to the axis (the polar centre)
        let p = if i % 4 == 0 {
            Vec3::new(0.05 * r() + 0.04, r() * 0.8, 0.02 * r() - 0.01)
        } else {
            Vec3::new(r() * 0.8 - 0.3, r() * 1.4 - 0.25, r() * 0.6 - 0.3)
        };
        let dir = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5).normalize();
        let step = 1e-3 + r() * 0.05;
        let q = p + dir * step;
        worst = worst.max((eval(&node, q) - eval(&node, p)).abs() / step);
    }
    assert!(worst <= 1.0 + 2e-3, "largest difference quotient {worst}");
}

#[test]
fn inside_on_the_axis_and_outside_far_away() {
    let node = curved();
    for s in [0.05f32, 0.4, 0.75] {
        let p = Vec3::new(0.15 * s, 0.9 * s, 0.0);
        assert!(eval(&node, p) < 0.0, "axis point {p:?} reads {}", eval(&node, p));
    }
    for p in [Vec3::new(1.0, 0.5, 0.0), Vec3::new(0.0, -0.5, 0.0), Vec3::new(0.0, 1.5, 0.0)] {
        assert!(eval(&node, p) > 0.2, "far point {p:?} reads {}", eval(&node, p));
    }
}

#[test]
fn malformed_input_is_refused_by_name() {
    let row = circle_row(0.2, 2);
    let ax = vec![Vec3::ZERO, Vec3::Y];
    let err = |r: Result<SdfNode, alice_sdf::primitives::loft::LoftError>| r.err().expect("refused").to_string();
    assert!(err(SdfNode::loft(vec![Vec3::ZERO], Vec3::X, 2, vec![row.clone(), row.clone()])).contains("axis"));
    assert!(err(SdfNode::loft(ax.clone(), Vec3::X, 2, vec![row.clone()])).contains("row"));
    assert!(err(SdfNode::loft(ax.clone(), Vec3::X, 2, vec![row.clone(), circle_row(0.2, 3)])).contains("row"));
    assert!(err(SdfNode::loft(ax.clone(), Vec3::X, 65, vec![circle_row(0.2, 65), circle_row(0.2, 65)])).contains("harmonics"));
    assert!(err(SdfNode::loft(ax.clone(), Vec3::Y, 2, vec![row.clone(), row.clone()])).contains("up"));
    let mut neg = row.clone();
    neg[1] = 0.3; // a0 = 0.2 with a1 = 0.3 passes through the centre
    assert!(err(SdfNode::loft(ax.clone(), Vec3::X, 2, vec![neg, row.clone()])).contains("radius"));
    let nan = vec![Vec3::ZERO, Vec3::new(f32::NAN, 1.0, 0.0)];
    assert!(err(SdfNode::loft(nan, Vec3::X, 2, vec![row.clone(), row])).contains("finite"));
}

#[test]
fn the_compiled_bytecode_gives_the_trees_value() {
    let node = curved();
    let compiled = alice_sdf::compiled::CompiledSdf::compile(&node);
    let mut r = lcg(5);
    for _ in 0..2000 {
        let p = Vec3::new(r() * 0.8 - 0.3, r() * 1.4 - 0.25, r() * 0.6 - 0.3);
        let a = eval(&node, p);
        let b = alice_sdf::compiled::eval_compiled(&compiled, p);
        assert_eq!(a.to_bits(), b.to_bits(), "at {p:?}: tree {a}, bytecode {b}");
    }
}

#[test]
fn a_cylinder_with_hemispherical_domes_is_the_capsule() {
    // radius 0.3 from y = 0 to 0.9, domes of height 0.3 at both ends: the capsule y ∈ [0, 0.9]
    let (h, r0) = (0.9, 0.3);
    let node = SdfNode::loft_capped(straight_axis(h), Vec3::X, 3, vec![circle_row(r0, 3), circle_row(r0, 3)], [r0, r0])
        .expect("capsule");
    let mut r = lcg(6);
    let (mut worst, mut sign_bad) = (0.0f32, 0);
    for _ in 0..4000 {
        let p = Vec3::new(r() * 1.4 - 0.7, r() * 2.2 - 0.65, r() * 1.4 - 0.7);
        let y = p.y.clamp(0.0, h);
        let want = (p - Vec3::new(0.0, y, 0.0)).length() - r0;
        let f = eval(&node, p);
        if want > 0.0 {
            worst = worst.max((f - want).abs());
        }
        sign_bad += usize::from(want.abs() > 1e-4 && (f > 0.0) != (want > 0.0));
    }
    assert!(worst < 2e-5, "largest difference outside {worst}");
    assert_eq!(sign_bad, 0);
}

#[test]
fn a_dome_closes_a_varying_section_and_stays_one_lipschitz() {
    // the curved loft's sections with domes: the tips beyond the end planes are inside, the field
    // never grows faster than 1, and the dome reaches its height on the end tangent
    let axis = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.05, 0.3, 0.02), Vec3::new(0.12, 0.6, 0.0)];
    let rows = vec![
        vec![0.10, 0.0, 0.01, 0.02, 0.0, 0.0, 0.004],
        vec![0.09, 0.005, 0.0, -0.015, 0.0, 0.0, 0.0],
        vec![0.08, 0.0, -0.01, 0.01, 0.005, 0.003, 0.0],
    ];
    let node = SdfNode::loft_capped(axis, Vec3::Z, 3, rows, [0.05, 0.12]).expect("domed");
    // the end tangent of a Catmull-Rom with reflected ends is A₁ − A₀ at the start, Aₙ₋₁ − Aₙ₋₂ at the end
    let t0 = (Vec3::new(0.0, 0.0, 0.0) - Vec3::new(0.05, 0.3, 0.02)).normalize();
    let t1 = (Vec3::new(0.12, 0.6, 0.0) - Vec3::new(0.05, 0.3, 0.02)).normalize();
    assert!(eval(&node, t0 * 0.045) < 0.0 && eval(&node, t0 * 0.055) > 0.0, "start dome height");
    let e = Vec3::new(0.12, 0.6, 0.0);
    assert!(eval(&node, e + t1 * 0.11) < 0.0 && eval(&node, e + t1 * 0.13) > 0.0, "end dome height");
    let mut r = lcg(7);
    let mut worst = 0.0f32;
    for _ in 0..20000 {
        let p = Vec3::new(r() * 0.6 - 0.25, r() * 1.0 - 0.2, r() * 0.5 - 0.25);
        let dir = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5).normalize();
        let step = 1e-3 + r() * 0.05;
        worst = worst.max((eval(&node, p + dir * step) - eval(&node, p)).abs() / step);
    }
    assert!(worst <= 1.0 + 2e-3, "largest difference quotient {worst}");
}

/// Sphere-tracing steps to reach |f| < 1e-5 from `o` along `d` (at most 4096)
fn march_steps(node: &SdfNode, o: Vec3, d: Vec3) -> u32 {
    let mut t = 0.0f32;
    for k in 0..4096 {
        let f = eval(node, o + d * t);
        if f.abs() < 1e-5 {
            return k;
        }
        t += f;
    }
    4096
}

#[test]
#[ignore = "measurement: march step counts of the global bound (run with --ignored --nocapture)"]
fn march_steps_of_the_global_bound() {
    let n = 13;
    let ys: Vec<f32> = (0..n).map(|i| -0.9 + 1.8 * i as f32 / (n - 1) as f32).collect();
    let axis: Vec<Vec3> = ys.iter().map(|&y| Vec3::new(0.0, y, 0.0)).collect();
    let rows: Vec<Vec<f32>> = ys.iter().map(|&y| circle_row((1.0 - y * y).sqrt(), 1)).collect();
    let sphere = SdfNode::loft(axis, Vec3::X, 1, rows).expect("sphere");
    let exact = SdfNode::sphere(1.0);
    let cone = SdfNode::loft(straight_axis(1.0), Vec3::X, 2, vec![circle_row(0.2, 2), circle_row(0.45, 2)]).expect("cone");
    let mut r = lcg(8);
    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..200 {
        let o = Vec3::new(3.0, r() * 1.2 - 0.6, r() * 1.2 - 0.6);
        let d = (Vec3::new(0.0, r() * 1.0 - 0.5, 0.0) - o).normalize();
        a.push(march_steps(&sphere, o, d));
        b.push(march_steps(&exact, o, d));
        let oc = Vec3::new(3.0, r() * 0.8 + 0.1, 0.0);
        c.push(march_steps(&cone, oc, (Vec3::new(0.0, oc.y, 0.0) - oc).normalize()));
    }
    for (name, v) in [("sphere as a loft", &mut a), ("exact sphere", &mut b), ("cone loft", &mut c)] {
        v.sort_unstable();
        println!("{name}: median {} p95 {} max {}", v[v.len() / 2], v[v.len() * 95 / 100], v[v.len() - 1]);
    }
}

#[test]
fn the_field_is_one_lipschitz_next_to_the_polar_centre() {
    // a lobed section (a0 = 0.2, a3 = 0.06, b2 = 0.03): near the centre the polar field's angular
    // term grows like r'(θ)/ρ, which the core must hold
    let rows = vec![vec![0.2, 0.0, 0.0, 0.0, 0.03, 0.06, 0.0], vec![0.18, 0.0, 0.0, 0.0, 0.03, 0.06, 0.0]];
    let node = SdfNode::loft(straight_axis(1.0), Vec3::X, 3, rows).expect("lobed");
    let mut r = lcg(9);
    let mut worst = 0.0f32;
    for _ in 0..20000 {
        let p = Vec3::new(r() * 0.1 - 0.05, r() * 0.8 + 0.1, r() * 0.1 - 0.05);
        let dir = Vec3::new(r() - 0.5, 0.0, r() - 0.5).normalize();
        let step = 1e-3 + r() * 0.02;
        worst = worst.max((eval(&node, p + dir * step) - eval(&node, p)).abs() / step);
    }
    assert!(worst <= 1.0 + 2e-3, "largest difference quotient {worst}");
}

/// The sphere as a loft (13 rows over |y| ≤ 0.9): the local bound varies from span to span
fn sphere_loft() -> SdfNode {
    let n = 13;
    let ys: Vec<f32> = (0..n).map(|i| -0.9 + 1.8 * i as f32 / (n - 1) as f32).collect();
    let axis: Vec<Vec3> = ys.iter().map(|&y| Vec3::new(0.0, y, 0.0)).collect();
    let rows: Vec<Vec<f32>> = ys.iter().map(|&y| circle_row((1.0 - y * y).sqrt(), 1)).collect();
    SdfNode::loft(axis, Vec3::X, 1, rows).expect("sphere")
}

#[test]
fn the_local_bound_is_one_lipschitz_across_span_joins_and_far_away() {
    let node = sphere_loft();
    let mut r = lcg(10);
    let (mut worst, mut compared) = (0.0f32, 0);
    for i in 0..30000 {
        // a third on a span join (y = −0.9 + 0.15 j) near the surface, a third far (up to 3), a third anywhere
        let p = match i % 3 {
            0 => {
                let y = -0.9 + 0.15 * (r() * 12.0).floor() + (r() - 0.5) * 0.01;
                let a = r() * std::f32::consts::TAU;
                let rho = (1.0 - y * y).max(0.0).sqrt() + (r() - 0.5) * 0.2;
                Vec3::new(rho * a.cos(), y, rho * a.sin())
            }
            1 => Vec3::new(r() * 6.0 - 3.0, r() * 6.0 - 3.0, r() * 6.0 - 3.0),
            _ => Vec3::new(r() * 2.4 - 1.2, r() * 2.4 - 1.2, r() * 2.4 - 1.2),
        };
        let dir = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5).normalize();
        let step = 1e-3 + r() * 0.1;
        worst = worst.max((eval(&node, p + dir * step) - eval(&node, p)).abs() / step);
        compared += 1;
    }
    // along the axis, where the local bound changes fastest, out to where the tube takes over:
    // the joins between y = ±0.45 and ±0.9, ρ from the surface to 0.5 beyond it
    for _ in 0..20000 {
        let y = (0.45 + 0.45 * r()) * if r() < 0.5 { -1.0 } else { 1.0 };
        let a = r() * std::f32::consts::TAU;
        let rho = (1.0 - y * y).max(0.0).sqrt() + r() * 0.5;
        let p = Vec3::new(rho * a.cos(), y, rho * a.sin());
        let step = 1e-3 + r() * 0.03;
        let q = p + Vec3::Y * step;
        worst = worst.max((eval(&node, q) - eval(&node, p)).abs() / step);
        compared += 1;
    }
    assert!(compared > 0);
    assert!(worst <= 1.0 + 2e-3, "largest difference quotient {worst} over {compared} pairs");
}

#[test]
fn the_local_bound_keeps_the_sign_and_never_passes_the_surface() {
    // against the exact sphere, away from the loft's smoothing (1e-2) and its flat ends (|y| ≤ 0.85)
    let node = sphere_loft();
    let mut r = lcg(11);
    let (mut sign_bad, mut over, mut compared) = (0, 0, 0);
    for _ in 0..20000 {
        let p = Vec3::new(r() * 6.0 - 3.0, r() * 1.7 - 0.85, r() * 6.0 - 3.0);
        let d = p.length() - 1.0;
        if d.abs() < 0.05 {
            continue;
        }
        compared += 1;
        let f = eval(&node, p);
        sign_bad += usize::from((f > 0.0) != (d > 0.0));
        over += usize::from(d > 0.0 && f > d + 1e-2);
    }
    assert!(compared > 10000, "compared {compared}");
    assert_eq!((sign_bad, over), (0, 0), "wrong sign / past the surface among {compared}");
}

#[test]
fn the_clamp_holds_the_field_where_the_local_bound_jumps() {
    // the radius steps from 0.2 to 0.6 within one span, so L(s) climbs steeply there; outside the
    // thin part at ρ ≈ 1, |F| is far beyond the band, so without the clamp F/L(s) falls faster than 1
    // per unit length toward the step
    let mut rows = Vec::new();
    for i in 0..10 {
        rows.push(circle_row(if i < 5 { 0.2 } else { 0.6 }, 1));
    }
    let axis: Vec<Vec3> = (0..10).map(|i| Vec3::new(0.0, 0.1 * i as f32, 0.0)).collect();
    let node = SdfNode::loft(axis, Vec3::X, 1, rows).expect("stepped");
    let mut r = lcg(12);
    let mut worst = 0.0f32;
    for _ in 0..20000 {
        let a = r() * std::f32::consts::TAU;
        let rho = 0.6 + r() * 0.8;
        let p = Vec3::new(rho * a.cos(), 0.1 + r() * 0.7, rho * a.sin());
        let step = 1e-3 + r() * 0.02;
        worst = worst.max((eval(&node, p + Vec3::Y * step) - eval(&node, p)).abs() / step);
    }
    assert!(worst <= 1.0 + 2e-3, "largest difference quotient {worst}");
}

/// The exact distance to a loft of circles on a straight axis along +y (axis from y0 to y1 in n − 1
/// even steps, circle radii `radii` as the control rows), written here: the profile R(y) is the
/// uniform cubic B-spline of the radii with reflected end rows, the solid is ρ ≤ R(y) between the
/// end planes; the distance in the (ρ, y) half plane to the profile and the two end segments
fn circle_loft_distance(p: Vec3, y0: f32, y1: f32, radii: &[f32]) -> f32 {
    let k = radii.len();
    let get = |i: isize| -> f32 {
        if i < 0 {
            2.0 * radii[0] - radii[1]
        } else if i >= k as isize {
            2.0 * radii[k - 1] - radii[k - 2]
        } else {
            radii[i as usize]
        }
    };
    let profile = |y: f32| -> f32 {
        let w = ((y - y0) / (y1 - y0) * (k - 1) as f32).clamp(0.0, (k - 1) as f32);
        let i = (w as usize).min(k - 2);
        let t = w - i as f32;
        let u = 1.0 - t;
        let i = i as isize;
        u * u * u / 6.0 * get(i - 1)
            + (3.0 * t * t * t - 6.0 * t * t + 4.0) / 6.0 * get(i)
            + (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) / 6.0 * get(i + 1)
            + t * t * t / 6.0 * get(i + 2)
    };
    let q = Vec2::new(Vec2::new(p.x, p.z).length(), p.y);
    let n = 20000;
    let mut best = f32::INFINITY;
    let mut prev = Vec2::new(profile(y0), y0);
    for j in 1..=n {
        let y = y0 + (y1 - y0) * j as f32 / n as f32;
        let cur = Vec2::new(profile(y), y);
        let e = cur - prev;
        let t = ((q - prev).dot(e) / e.dot(e)).clamp(0.0, 1.0);
        best = best.min((q - (prev + e * t)).length());
        prev = cur;
    }
    for (y, r) in [(y0, profile(y0)), (y1, profile(y1))] {
        let t = (q.x / r).clamp(0.0, 1.0);
        best = best.min((q - Vec2::new(r * t, y)).length());
    }
    let inside = q.y >= y0 && q.y <= y1 && q.x <= profile(q.y);
    if inside {
        -best
    } else {
        best
    }
}

#[test]
fn the_local_bound_never_passes_the_lofts_own_surface() {
    // the sphere's rows and the stepped rows: outside, f never exceeds the exact distance to the
    // loft's own surface (computed here), and inside it never goes below minus that distance
    let n = 13;
    let ys: Vec<f32> = (0..n).map(|i| -0.9 + 1.8 * i as f32 / (n - 1) as f32).collect();
    let sphere_r: Vec<f32> = ys.iter().map(|&y| (1.0 - y * y).sqrt()).collect();
    let step_r: Vec<f32> = (0..10).map(|i| if i < 5 { 0.2 } else { 0.6 }).collect();
    for (radii, y0, y1) in [(sphere_r, -0.9f32, 0.9f32), (step_r, 0.0, 0.9)] {
        let k = radii.len();
        let axis: Vec<Vec3> = (0..k).map(|i| Vec3::new(0.0, y0 + (y1 - y0) * i as f32 / (k - 1) as f32, 0.0)).collect();
        let node = SdfNode::loft(axis, Vec3::X, 1, radii.iter().map(|&r| circle_row(r, 1)).collect()).expect("circle loft");
        let mut r = lcg(13);
        let (mut over, mut compared, mut tightest) = (0, 0, f32::INFINITY);
        for i in 0..6000 {
            // half anywhere, half within 0.03 of the rows' own radius (where the local bound is tight)
            let p = if i % 2 == 0 {
                Vec3::new(r() * 3.0 - 1.5, y0 - 0.3 + r() * (y1 - y0 + 0.6), r() * 3.0 - 1.5)
            } else {
                let j = (r() * k as f32).floor().min((k - 1) as f32) as usize;
                let y = y0 + (y1 - y0) * (j as f32 + (r() - 0.5) * 0.5).clamp(0.0, (k - 1) as f32) / (k - 1) as f32;
                let a = r() * std::f32::consts::TAU;
                let rho = radii[j] + (r() - 0.5) * 0.06;
                Vec3::new(rho * a.cos(), y, rho * a.sin())
            };
            let d = circle_loft_distance(p, y0, y1, &radii);
            if d.abs() < 1e-3 {
                continue;
            }
            compared += 1;
            let f = eval(&node, p);
            over += usize::from(f.abs() > d.abs() + 1e-4 || (f > 0.0) != (d > 0.0));
            if d > 0.0 {
                tightest = tightest.min(f / d);
            }
        }
        assert!(compared > 4000, "compared {compared}");
        assert_eq!(over, 0, "past the loft's own surface or the wrong sign among {compared} (least f/D outside {tightest})");
    }
}

#[test]
fn the_local_bounds_margin_is_never_negative() {
    // every loft built in these oracles and the corpus, plus a gently bent limb-like loft whose
    // sections reach past half the axis's radius of curvature (where the margin used to come out
    // negative): m ≥ 0, and > 0 wherever the local bound varies
    let bent: Vec<Vec3> = (0..6)
        .map(|i| {
            let a = 0.35 * i as f32;
            Vec3::new(0.6 * (1.0 - a.cos()), 0.6 * a.sin(), 0.0)
        })
        .collect();
    let rows: Vec<Vec<f32>> = (0..6).map(|i| circle_row(0.13 + 0.05 * (i as f32 * 0.9).sin(), 2)).collect();
    let limb = SdfNode::loft(bent, Vec3::Z, 2, rows).expect("a bent limb");
    let n = 13;
    let ys: Vec<f32> = (0..n).map(|i| -0.9 + 1.8 * i as f32 / (n - 1) as f32).collect();
    let sphere = SdfNode::loft(
        ys.iter().map(|&y| Vec3::new(0.0, y, 0.0)).collect(),
        Vec3::X,
        1,
        ys.iter().map(|&y| circle_row((1.0 - y * y).sqrt(), 1)).collect(),
    )
    .expect("sphere");
    let mut checked = 0;
    for node in [limb.clone(), sphere, curved()] {
        if let SdfNode::Loft { ref bounds, .. } = node {
            checked += 1;
            assert!(bounds.m >= 0.0, "margin {}", bounds.m);
            let varies = bounds.knots.windows(2).any(|w| w[0] != w[1]);
            assert!(!varies || bounds.m > 0.0, "margin {} with a varying bound", bounds.m);
        }
    }
    assert_eq!(checked, 3);
    // the bent limb's field stays 1-Lipschitz
    let mut r = lcg(14);
    let mut worst = 0.0f32;
    for _ in 0..20000 {
        let p = Vec3::new(r() * 1.6 - 0.4, r() * 1.4 - 0.2, r() * 0.8 - 0.4);
        let dir = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5).normalize();
        let step = 1e-3 + r() * 0.05;
        worst = worst.max((eval(&limb, p + dir * step) - eval(&limb, p)).abs() / step);
    }
    assert!(worst <= 1.0 + 2e-3, "largest difference quotient {worst}");
}
