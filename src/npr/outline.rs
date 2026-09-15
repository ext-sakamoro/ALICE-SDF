//! Outline primitives derived from SDF value, curvature, or depth
//!
//! Outline generation without mesh-based inverted-hull passes: each function
//! returns a scalar mask in `[0, 1]` that can be composed with a color pass.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Hard outline mask: 1.0 inside `|sdf| < width`, otherwise 0.0
#[inline]
#[must_use]
pub fn distance_field_outline(sdf: f32, width: f32) -> f32 {
    if sdf.abs() < width.max(0.0) {
        1.0
    } else {
        0.0
    }
}

/// Soft outline mask: 1.0 at `sdf == 0`, tapering to 0.0 at `|sdf| >= width_outer`
///
/// Between `width_inner` and `width_outer` the mask uses a smoothstep falloff.
/// `width_outer` must be `>= width_inner`.
#[inline]
#[must_use]
pub fn distance_field_outline_soft(sdf: f32, width_inner: f32, width_outer: f32) -> f32 {
    let d = sdf.abs();
    let inner = width_inner.max(0.0);
    let outer = width_outer.max(inner + 1e-6);
    if d <= inner {
        1.0
    } else if d >= outer {
        0.0
    } else {
        let t = ((d - inner) / (outer - inner)).clamp(0.0, 1.0);
        (t * t).mul_add(-2.0f32.mul_add(-t, 3.0), 1.0)
    }
}

/// Curvature-based outline: strong where surface curvature is high
///
/// `mean_curvature` is expected in inverse-length units. `sensitivity`
/// scales the raw curvature and `threshold` cuts off low-curvature regions.
#[inline]
#[must_use]
pub fn curvature_outline(mean_curvature: f32, sensitivity: f32, threshold: f32) -> f32 {
    let k = mean_curvature.abs() * sensitivity.max(0.0);
    (k - threshold).clamp(0.0, 1.0)
}

/// Depth-gradient outline: 1.0 where `|depth_gradient|` exceeds `threshold`
///
/// Intended for use with a screen-space depth derivative (e.g. `fwidth(z)`).
#[inline]
#[must_use]
pub fn depth_step_outline(depth_gradient: f32, threshold: f32) -> f32 {
    if depth_gradient.abs() > threshold.max(0.0) {
        1.0
    } else {
        0.0
    }
}

/// Composite an outline color over a base color using an alpha mask
///
/// `alpha` is clamped to `[0, 1]`.
#[inline]
#[must_use]
pub fn composite_outline(base: Vec3, outline: Vec3, alpha: f32) -> Vec3 {
    let a = alpha.clamp(0.0, 1.0);
    base * (1.0 - a) + outline * a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_outline_inside_band() {
        assert!((distance_field_outline(0.05, 0.1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hard_outline_outside_band() {
        assert!(distance_field_outline(0.2, 0.1).abs() < 1e-6);
    }

    #[test]
    fn hard_outline_negative_side() {
        // Interior of the shape (sdf < 0) still counts if within width
        assert!((distance_field_outline(-0.05, 0.1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn soft_outline_peak_at_zero() {
        assert!((distance_field_outline_soft(0.0, 0.02, 0.1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn soft_outline_far_beyond_outer() {
        assert!(distance_field_outline_soft(1.0, 0.02, 0.1).abs() < 1e-6);
    }

    #[test]
    fn soft_outline_inside_inner() {
        assert!((distance_field_outline_soft(0.01, 0.02, 0.1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn soft_outline_between_inner_and_outer() {
        let mid = distance_field_outline_soft(0.06, 0.02, 0.1);
        assert!((0.0..=1.0).contains(&mid));
        assert!(mid > 0.0 && mid < 1.0, "expected soft mid value, got {mid}");
    }

    #[test]
    fn curvature_outline_below_threshold_is_zero() {
        assert!(curvature_outline(0.1, 1.0, 0.5).abs() < 1e-6);
    }

    #[test]
    fn curvature_outline_above_threshold_positive() {
        assert!(curvature_outline(1.0, 1.0, 0.2) > 0.0);
    }

    #[test]
    fn depth_step_outline_below_threshold_zero() {
        assert!(depth_step_outline(0.05, 0.1).abs() < 1e-6);
    }

    #[test]
    fn depth_step_outline_above_threshold_one() {
        assert!((depth_step_outline(0.5, 0.1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn composite_full_alpha_returns_outline_color() {
        assert_eq!(composite_outline(Vec3::ZERO, Vec3::ONE, 1.0), Vec3::ONE);
    }

    #[test]
    fn composite_zero_alpha_returns_base_color() {
        assert_eq!(composite_outline(Vec3::ONE, Vec3::ZERO, 0.0), Vec3::ONE);
    }

    #[test]
    fn composite_half_alpha_midway() {
        let out = composite_outline(Vec3::ZERO, Vec3::ONE, 0.5);
        assert!((out - Vec3::splat(0.5)).length() < 1e-6);
    }
}
