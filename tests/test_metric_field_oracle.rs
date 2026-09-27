//! Oracles for the metric ball and the metric blend.
//!
//! The corpus tests already prove these two nodes agree across every
//! evaluation path, that their Lipschitz claims bound the real difference
//! quotients, that their boxes are conservative and that their intervals
//! contain the point values. What is left is what they *mean*, which is
//! what this file pins.
//!
//! # Closed forms
//!
//! * `‖p‖∞ − r` vanishes exactly on the faces of the cube `[−r, r]³`, and
//!   `‖p‖₁ − r` exactly on the eight planes `±x ±y ±z = r` of the
//!   octahedron. Those are the shapes, not approximations of them.
//! * With the Euclidean weights the field *is* the sphere field, bit for
//!   bit — a sphere and a cube are the same expression measured with two
//!   different norms, and that claim is executable rather than rhetorical.
//! * A blend weighted by `smoothstep` is exactly `0` below its low edge and
//!   exactly `1` above the high one, and `(1−t)a + tb` is endpoint-exact, so
//!   outside the bubble the field has to be the outer field *bit for bit*.
//! * The blend's gradient picks up `|f_in − f_out| · |∇w|`, and the cubic
//!   weight's steepest slope is `1.5/W`, so the excess tension a bubble
//!   creates scales as `1/W`: widening the skin by a factor halves it. That
//!   is the whole balance rule of the mechanic, and it is measurable.

use alice_det_math::metric::MetricWeights;
use alice_sdf::eval::eval;
use alice_sdf::interval::eval_lipschitz;
use alice_sdf::measure::measure_tension;
use alice_sdf::types::{Aabb, SdfNode};
use glam::Vec3;

fn grid(half: f32, n: i32) -> Vec<Vec3> {
    let mut out = Vec::new();
    for i in -n..=n {
        for j in -n..=n {
            for k in -n..=n {
                out.push(Vec3::new(
                    half * i as f32 / n as f32,
                    half * j as f32 / n as f32,
                    half * k as f32 / n as f32,
                ));
            }
        }
    }
    out
}

#[test]
fn the_cube_metric_ball_is_exactly_the_cube() {
    let ball = SdfNode::metric_ball(1.0, MetricWeights::LINF);
    // on the faces
    for p in [
        Vec3::new(1.0, 0.4, -0.9),
        Vec3::new(-0.2, 1.0, 0.7),
        Vec3::new(0.0, -0.5, -1.0),
    ] {
        assert!(eval(&ball, p).abs() < 1e-6, "{p:?} should be on a face");
    }
    // the corner is inside its own metric but the furthest Euclidean point
    let corner = Vec3::splat(1.0);
    assert!(eval(&ball, corner).abs() < 1e-6);
    assert!((corner.length() - 3f32.sqrt()).abs() < 1e-6);
    // and the box of the ball is the cube itself, not the √3 sphere
    assert!((MetricWeights::LINF.axis_extent(1.0) - 1.0).abs() < 1e-6);
}

#[test]
fn the_octahedral_metric_ball_is_exactly_the_octahedron() {
    let ball = SdfNode::metric_ball(1.0, MetricWeights::L1);
    for p in [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.5, 0.25, 0.25),
    ] {
        assert!(eval(&ball, p).abs() < 1e-6, "{p:?} should be on a facet");
    }
    assert!(eval(&ball, Vec3::splat(0.5)) > 0.0, "the corner is outside");
}

#[test]
fn the_euclidean_weights_reproduce_the_sphere_bit_for_bit() {
    // the claim that a sphere is just one choice of norm, made executable
    let sphere = SdfNode::sphere(0.75);
    let ball = SdfNode::metric_ball(0.75, MetricWeights::L2);
    for p in grid(2.0, 6) {
        assert_eq!(
            eval(&ball, p).to_bits(),
            eval(&sphere, p).to_bits(),
            "the Euclidean metric ball diverged from the sphere at {p:?}"
        );
    }
}

#[test]
fn the_lipschitz_claim_is_the_closed_form_of_the_norm() {
    assert_eq!(
        eval_lipschitz(&SdfNode::metric_ball(1.0, MetricWeights::L2)),
        1.0
    );
    assert!((eval_lipschitz(&SdfNode::metric_ball(1.0, MetricWeights::LINF)) - 1.0).abs() < 1e-6);
    assert!(
        (eval_lipschitz(&SdfNode::metric_ball(1.0, MetricWeights::L1)) - 3f32.sqrt()).abs() < 1e-5
    );
    // the octahedral field over-reports, which is the mode that pierces
    // surfaces — and the marcher's divisor is exactly that constant
    let f = alice_sdf::fidelity::distance_fidelity(&SdfNode::metric_ball(1.0, MetricWeights::L1));
    assert!(f.can_overshoot());
    assert!((f.safe_step_scale().unwrap() - 3f32.sqrt()).abs() < 1e-5);
}

#[test]
fn outside_the_bubble_the_world_is_untouched_bit_for_bit() {
    let outer = SdfNode::box3d(0.5, 0.4, 0.3);
    let inner = SdfNode::metric_ball(0.5, MetricWeights::L1);
    let blend = SdfNode::metric_blend(inner, outer.clone(), Vec3::ZERO, 1.0, 0.5);
    for p in grid(3.0, 7) {
        if p.length() >= 1.5 + 1e-4 {
            assert_eq!(
                eval(&blend, p).to_bits(),
                eval(&outer, p).to_bits(),
                "a bubble perturbed the world outside it at {p:?}"
            );
        }
    }
}

#[test]
fn inside_the_bubble_the_field_is_the_inner_one_bit_for_bit() {
    let outer = SdfNode::box3d(0.5, 0.4, 0.3);
    let inner = SdfNode::metric_ball(0.5, MetricWeights::LINF);
    let blend = SdfNode::metric_blend(inner.clone(), outer, Vec3::ZERO, 1.0, 0.5);
    // the bubble is a Euclidean ball, so the cube grid's corners
    // (0.9·√3 ≈ 1.56) are outside it — filter by the radius, not the box
    for p in grid(0.9, 5).into_iter().filter(|p| p.length() <= 1.0) {
        assert_eq!(
            eval(&blend, p).to_bits(),
            eval(&inner, p).to_bits(),
            "the inner field was not reproduced at {p:?}"
        );
    }
}

#[test]
fn a_displaced_bubble_follows_its_centre() {
    let outer = SdfNode::sphere(0.3);
    let inner = SdfNode::metric_ball(0.3, MetricWeights::LINF);
    let centre = Vec3::new(2.0, -1.0, 0.5);
    let blend = SdfNode::metric_blend(inner.clone(), outer.clone(), centre, 0.8, 0.3);
    assert_eq!(
        eval(&blend, centre).to_bits(),
        eval(&inner, centre).to_bits()
    );
    let far = centre + Vec3::new(3.0, 0.0, 0.0);
    assert_eq!(eval(&blend, far).to_bits(), eval(&outer, far).to_bits());
}

#[test]
fn the_tension_a_bubble_creates_falls_as_one_over_the_skin_width() {
    // Two fields that disagree by a constant offset, so the excess gradient
    // over the skin is `|Δd| · |∇w|` with a known `|Δd|`; the cubic weight's
    // steepest slope is 1.5/W, so the measured tension has to fall like 1/W.
    let outer = SdfNode::sphere(1.0);
    let inner = SdfNode::sphere(0.5); // differs from `outer` by 0.5 everywhere
    let region = Aabb::new(Vec3::splat(-3.0), Vec3::splat(3.0));

    let mut measured = Vec::new();
    for skin in [0.25_f32, 0.5, 1.0, 2.0] {
        let blend = SdfNode::metric_blend(inner.clone(), outer.clone(), Vec3::ZERO, 1.0, skin);
        let t = measure_tension(&blend, region, 200_000, 0xbeef_1234, 1e-3);
        measured.push((skin, t.max_quotient));
    }

    // monotone in the skin width
    for w in measured.windows(2) {
        assert!(
            w[1].1 <= w[0].1 + 1e-2,
            "widening the skin should not raise the tension: {measured:?}"
        );
    }
    // and the excess above a 1-Lipschitz field scales like 1/W: halving the
    // skin roughly doubles it (loose bounds — this is a sampled maximum)
    let excess: Vec<f32> = measured.iter().map(|(_, q)| (q - 1.0).max(0.0)).collect();
    assert!(
        excess[0] > 2.0 * excess[2],
        "quartering the skin should multiply the excess several-fold: {measured:?}"
    );
    // the widest skin is safe, the narrowest is not
    assert!(
        measured[0].1 > 1.0,
        "a 0.25 skin over a 0.5 step must tear: {measured:?}"
    );
    assert!(
        measured[3].1 <= 1.05,
        "a 2.0 skin over a 0.5 step must not: {measured:?}"
    );
}

#[test]
fn the_blend_claims_no_lipschitz_bound() {
    // and it must not: the bound would depend on the two fields, not on this
    // node, so claiming one would be a guess
    let blend = SdfNode::metric_blend(
        SdfNode::sphere(0.5),
        SdfNode::sphere(1.0),
        Vec3::ZERO,
        1.0,
        0.1,
    );
    assert!(!eval_lipschitz(&blend).is_finite());
    assert_eq!(
        alice_sdf::fidelity::distance_fidelity(&blend),
        alice_sdf::fidelity::Fidelity::Unbounded
    );
}

#[test]
fn a_metric_ball_is_a_convex_body() {
    // the defining property of a metric: the midpoint of two points of the
    // ball is in the ball
    for w in [
        MetricWeights::L1,
        MetricWeights::L2,
        MetricWeights::LINF,
        MetricWeights::new(0.4, 0.4, 0.2).unwrap(),
    ] {
        let ball = SdfNode::metric_ball(1.0, w);
        let pts: Vec<Vec3> = grid(1.5, 5)
            .into_iter()
            .filter(|p| eval(&ball, *p) <= 0.0)
            .collect();
        for (i, a) in pts.iter().enumerate().take(400) {
            let b = pts[(i * 37 + 11) % pts.len()];
            let mid = (*a + b) * 0.5;
            assert!(
                eval(&ball, mid) <= 1e-5,
                "midpoint of two interior points escaped for {:?}",
                w.weights()
            );
        }
    }
}
