//! Metric ball — the unit ball of a norm, as a primitive.
//!
//! Every other primitive in this crate is written against the Euclidean norm:
//! `‖p‖₂ − r` is a sphere. Write the same expression with a different norm
//! and it is a different shape — `‖p‖∞ − r` is a cube, `‖p‖₁ − r` an
//! octahedron — while the *expression* has not changed at all. This
//! primitive makes the norm a parameter, so the shape is chosen by three
//! weights rather than by picking a different variant.
//!
//! The norm is a non-negative combination of the three bases,
//! `g(p) = w₁‖p‖₁ + w₂‖p‖₂ + w∞‖p‖∞`, which is a metric for exactly the
//! non-negative weights (a negative one dents the unit ball inwards and the
//! triangle inequality fails). [`MetricWeights`] enforces that at
//! construction, so this function never has to.
//!
//! # What the field is worth
//!
//! `g(p) − r` is a true distance in *its own* metric, and in Euclidean terms
//! its gradient is bounded by [`MetricWeights::lipschitz`] — exactly, not
//! approximately. For the cube and the sphere that bound is `1` and the
//! field never over-reports; for the octahedron it is `√3`, so a marcher has
//! to divide its step by that (which [`eval_lipschitz`](crate::interval::eval_lipschitz)
//! reports and `RaymarchConfig` already divides by). The tight axis-aligned
//! box of the ball is a *different* number, [`MetricWeights::axis_extent`].

use alice_det_math::metric::{norm_l1, norm_l2, norm_linf};
use glam::Vec3;

/// SDF of the ball `{p : g(p) ≤ radius}` for the weighted norm `g`.
///
/// The weights are taken as three scalars rather than a
/// [`MetricWeights`](alice_det_math::metric::MetricWeights) so the node stays
/// a plain bag of `f32` (serialisable, FFI-able, JIT-encodable like every
/// other primitive). They are validated where the node is built —
/// [`SdfNode::metric_ball`](crate::types::SdfNode::metric_ball) only accepts
/// a `MetricWeights` — so no validation happens per sample.
///
/// The summation order is the same left-to-right order as
/// `MetricWeights::norm`, and `tests/test_metric_ball_oracle.rs` asserts the
/// two agree bit for bit; a reordering here would be a silent divergence
/// between this primitive and every other consumer of that law.
#[inline]
#[must_use]
pub fn sdf_metric_ball(p: Vec3, radius: f32, w_l1: f32, w_l2: f32, w_linf: f32) -> f32 {
    let v = [p.x, p.y, p.z];
    w_l1 * norm_l1(v) + w_l2 * norm_l2(v) + w_linf * norm_linf(v) - radius
}

#[cfg(test)]
mod tests {
    use super::*;
    use alice_det_math::metric::MetricWeights;

    #[test]
    fn matches_the_canonical_norm_bit_for_bit() {
        let w = MetricWeights::new(0.3, 0.5, 0.2).unwrap();
        let (a, b, c) = w.weights();
        for p in [
            Vec3::new(1.0, 2.0, -3.0),
            Vec3::new(-0.25, 0.0, 7.5),
            Vec3::ZERO,
        ] {
            let got = sdf_metric_ball(p, 0.0, a, b, c);
            assert_eq!(got.to_bits(), w.norm([p.x, p.y, p.z]).to_bits());
        }
    }

    #[test]
    fn the_euclidean_weights_reproduce_a_sphere() {
        for p in [Vec3::new(3.0, 4.0, 0.0), Vec3::new(1.0, 1.0, 1.0)] {
            let got = sdf_metric_ball(p, 1.0, 0.0, 1.0, 0.0);
            assert_eq!(got.to_bits(), (p.length() - 1.0).to_bits());
        }
    }

    #[test]
    fn the_cube_metric_is_a_cube() {
        // ‖p‖∞ − r vanishes on the faces of the cube [-r, r]³
        assert!((sdf_metric_ball(Vec3::new(1.0, 0.3, -0.7), 1.0, 0.0, 0.0, 1.0)).abs() < 1e-6);
        assert!(sdf_metric_ball(Vec3::new(0.9, 0.9, 0.9), 1.0, 0.0, 0.0, 1.0) < 0.0);
        assert!(sdf_metric_ball(Vec3::new(1.1, 0.0, 0.0), 1.0, 0.0, 0.0, 1.0) > 0.0);
    }
}
