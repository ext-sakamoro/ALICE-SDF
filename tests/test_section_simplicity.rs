//! Closed sections must be simple: a loop that crosses itself has no consistent inside, so the
//! even-odd sign flips across the crossing far from any surface and the field jumps (a 1-Lipschitz
//! claim with a discontinuous value). Construction refuses
//! - a self-crossing polygon loop of a `SectionStack`,
//! - a self-crossing spline loop of a `SmoothSectionStack`,
//! - a curve loft whose sections cross themselves at a crossing row, and
//! - a pair of simple curve rows whose blend between them crosses itself.
//!
//! The crossings are found here by an independent check (the sampled spline's segments in f64), so
//! each refusal is tested against a shape known to cross.

use alice_sdf::prelude::*;
use glam::Vec2;

/// The closed uniform cubic B-spline of `cp` at `t` in [0, m)
fn spline(cp: &[Vec2], t: f64) -> (f64, f64) {
    let m = cp.len();
    let i = t.floor() as usize % m;
    let u = t - t.floor();
    let (v, u3) = (1.0 - u, u * u * u);
    let b = [
        v * v * v / 6.0,
        (3.0 * u3 - 6.0 * u * u + 4.0) / 6.0,
        (-3.0 * u3 + 3.0 * u * u + 3.0 * u + 1.0) / 6.0,
        u3 / 6.0,
    ];
    let (mut x, mut y) = (0.0, 0.0);
    for (k, w) in b.iter().enumerate() {
        let p = cp[(i + m + k - 1) % m];
        x += w * f64::from(p.x);
        y += w * f64::from(p.y);
    }
    (x, y)
}

fn polyline_crosses(pts: &[(f64, f64)]) -> bool {
    let n = pts.len();
    let cross = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| {
        (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
    };
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        for j in i + 2..n {
            if (j + 1) % n == i {
                continue;
            }
            let (c, d) = (pts[j], pts[(j + 1) % n]);
            let (d1, d2, d3, d4) = (
                cross(a, b, c),
                cross(a, b, d),
                cross(c, d, a),
                cross(c, d, b),
            );
            if d1 * d2 < 0.0 && d3 * d4 < 0.0 {
                return true;
            }
        }
    }
    false
}

fn spline_crosses(cp: &[Vec2]) -> bool {
    let n = cp.len() * 32;
    let pts: Vec<_> = (0..n)
        .map(|k| spline(cp, k as f64 * cp.len() as f64 / n as f64))
        .collect();
    polyline_crosses(&pts)
}

fn circle(r: f32, m: usize) -> Vec<Vec2> {
    (0..m)
        .map(|j| {
            let (s, c) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / m as f32);
            Vec2::new(c, s) * r
        })
        .collect()
}

/// A limaçon with an inner loop, r = 0.12 + 0.3 cos θ: positive area (the outer loop dominates,
/// so the counter-clockwise checks pass) and a crossing where the inner loop meets the outer one
fn limacon(m: usize) -> Vec<Vec2> {
    (0..m)
        .map(|j| {
            let (s, c) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / m as f32);
            Vec2::new(c, s) * (0.12 + 0.3 * c)
        })
        .collect()
}

fn levels(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.1 * i as f32).collect()
}

#[test]
fn the_limacon_crosses_itself() {
    // precondition of the refusals below, by the independent check
    assert!(spline_crosses(&limacon(24)));
    let area: f32 = (0..24)
        .map(|j| limacon(24)[j].perp_dot(limacon(24)[(j + 1) % 24]))
        .sum();
    assert!(
        area > 0.0,
        "the loop runs counter-clockwise, so only the crossing check can refuse it"
    );
    let poly: Vec<(f64, f64)> = limacon(24)
        .iter()
        .map(|p| (f64::from(p.x), f64::from(p.y)))
        .collect();
    assert!(polyline_crosses(&poly));
    assert!(
        !spline_crosses(&circle(0.2, 24)),
        "the checker passes a simple loop"
    );
}

#[test]
fn a_section_stack_refuses_a_self_crossing_loop() {
    let mut loops: Vec<Vec<Vec<Vec2>>> = (0..4).map(|_| vec![circle(0.2, 24)]).collect();
    loops[2] = vec![limacon(24)];
    assert!(SdfNode::section_stack(levels(4), loops).is_err());
}

#[test]
fn a_smooth_section_stack_refuses_a_self_crossing_loop() {
    let mut loops: Vec<Vec<Vec<Vec2>>> = (0..5).map(|_| vec![circle(0.2, 24)]).collect();
    loops[2] = vec![limacon(24)];
    assert!(SdfNode::smooth_section_stack(levels(5), loops, 0.01, 0.1).is_err());
}

#[test]
fn a_curve_loft_refuses_a_self_crossing_row() {
    let axis = vec![
        Vec3::ZERO,
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let rows = vec![circle(0.2, 24), limacon(24), circle(0.2, 24)];
    assert!(SdfNode::loft_curves(axis, Vec3::X, rows).is_err());
}

/// Two simple star-shaped rows (radii per control point, the second turned by 4.49 rad) whose
/// blend half way between them crosses itself; found by a search with the independent check below
fn crossing_pair() -> (Vec<Vec2>, Vec<Vec2>) {
    let star = |rs: [f32; 8], phase: f32| -> Vec<Vec2> {
        (0..8)
            .map(|j| {
                let (s, c) =
                    alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / 8.0 + phase);
                Vec2::new(c, s) * rs[j]
            })
            .collect()
    };
    (
        star([0.07, 0.07, 0.27, 0.09, 0.06, 0.26, 0.08, 0.26], 0.0),
        star([0.22, 0.26, 0.29, 0.19, 0.25, 0.06, 0.24, 0.18], 4.49),
    )
}

#[test]
fn a_curve_loft_refuses_rows_whose_blend_crosses_itself() {
    let (a, b) = crossing_pair();
    let mid: Vec<Vec2> = a.iter().zip(&b).map(|(x, y)| (*x + *y) * 0.5).collect();
    assert!(
        !spline_crosses(&a) && !spline_crosses(&b),
        "both rows are simple"
    );
    assert!(
        spline_crosses(&mid),
        "the half-way blend crosses itself (precondition)"
    );
    let axis = vec![
        Vec3::ZERO,
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.5, 0.0),
    ];
    let rows = vec![a.clone(), a, b.clone(), b];
    assert!(SdfNode::loft_curves(axis, Vec3::X, rows).is_err());
}

#[test]
fn simple_rows_and_loops_are_accepted() {
    let axis = vec![
        Vec3::ZERO,
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    assert!(SdfNode::loft_curves(
        axis,
        Vec3::X,
        vec![circle(0.2, 24), circle(0.15, 24), circle(0.1, 24)]
    )
    .is_ok());
    assert!(
        SdfNode::section_stack(levels(4), (0..4).map(|_| vec![circle(0.2, 24)]).collect()).is_ok()
    );
    assert!(SdfNode::smooth_section_stack(
        levels(5),
        (0..5).map(|_| vec![circle(0.2, 24)]).collect(),
        0.01,
        0.1
    )
    .is_ok());
}
