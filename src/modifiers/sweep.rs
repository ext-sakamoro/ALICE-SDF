//! Sweep modifier: extrude 2D cross-section along a quadratic Bezier curve
//!
//! Sweeps a child SDF (2D cross-section) along a quadratic Bezier path
//! in the XZ plane. The child is evaluated at (perpendicular_distance, y, 0).
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Evaluate a quadratic Bezier curve at parameter t
#[inline(always)]
fn bezier_eval(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) -> Vec2 {
    let omt = 1.0 - t;
    p0 * (omt * omt) + p1 * (2.0 * omt * t) + p2 * (t * t)
}

/// Evaluate the tangent of a quadratic Bezier at parameter t
#[inline(always)]
fn bezier_tangent(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) -> Vec2 {
    // B'(t) = 2(1-t)(p1-p0) + 2t(p2-p1)
    (p1 - p0) * (2.0 * (1.0 - t)) + (p2 - p1) * (2.0 * t)
}

/// Find the closest parameter t on the quadratic Bezier to point q.
/// Uses Newton's method with 5 iterations, starting from the best of 5 samples.
#[inline]
fn closest_t_on_bezier(q: Vec2, p0: Vec2, p1: Vec2, p2: Vec2) -> f32 {
    // Coarse search: sample 5 points to find good initial guess
    let mut best_t = 0.0f32;
    let mut best_d2 = f32::MAX;
    for i in 0..5 {
        let t = i as f32 * 0.25;
        let b = bezier_eval(p0, p1, p2, t);
        let d2 = (q - b).length_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best_t = t;
        }
    }

    // Newton iterations: minimize F(t) = |q - B(t)|²
    // F'(t) = -2 dot(q - B(t), B'(t))
    // F''(t) = 2 [dot(B'(t), B'(t)) - dot(q - B(t), B''(t))]
    let b_dd = 2.0 * (p0 - 2.0 * p1 + p2); // B''(t) is constant for quadratic

    let mut t = best_t;
    for _ in 0..5 {
        let bt = bezier_eval(p0, p1, p2, t);
        let bt_d = bezier_tangent(p0, p1, p2, t);
        let diff = bt - q;
        let numerator = diff.dot(bt_d);
        let denominator = bt_d.dot(bt_d) + diff.dot(b_dd);
        if denominator.abs() < 1e-10 {
            break;
        }
        t -= numerator / denominator;
        t = t.clamp(0.0, 1.0);
    }
    t
}

/// Sweep Bezier modifier: maps 3D point to 2D cross-section coordinates.
///
/// The Bezier curve is defined in the XZ plane. For each query point:
/// 1. Project to XZ: q = (p.x, p.z)
/// 2. Find closest point on Bezier curve
/// 3. Return (perpendicular_distance, p.y, 0.0) for child evaluation
///
/// This is analogous to Revolution but along a curved path instead of a circle.
#[inline]
pub fn modifier_sweep_bezier(p: Vec3, p0: Vec2, p1: Vec2, p2: Vec2) -> Vec3 {
    let q = Vec2::new(p.x, p.z);
    let t = closest_t_on_bezier(q, p0, p1, p2);
    let closest = bezier_eval(p0, p1, p2, t);
    let d = (q - closest).length();
    Vec3::new(d, p.y, 0.0)
}

/// SIMD-friendly version: returns (perpendicular_distance, y) without Vec3 allocation
#[inline]
pub fn sweep_bezier_dist_y(
    px: f32,
    py: f32,
    pz: f32,
    p0x: f32,
    p0z: f32,
    p1x: f32,
    p1z: f32,
    p2x: f32,
    p2z: f32,
) -> (f32, f32) {
    let p0 = Vec2::new(p0x, p0z);
    let p1 = Vec2::new(p1x, p1z);
    let p2 = Vec2::new(p2x, p2z);
    let q = Vec2::new(px, pz);
    let t = closest_t_on_bezier(q, p0, p1, p2);
    let closest = bezier_eval(p0, p1, p2, t);
    let d = (q - closest).length();
    (d, py)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bezier_eval_endpoints() {
        let p0 = Vec2::new(0.0, 0.0);
        let p1 = Vec2::new(0.5, 1.0);
        let p2 = Vec2::new(1.0, 0.0);

        let b0 = bezier_eval(p0, p1, p2, 0.0);
        let b1 = bezier_eval(p0, p1, p2, 1.0);
        assert!((b0 - p0).length() < 1e-6);
        assert!((b1 - p2).length() < 1e-6);
    }

    #[test]
    fn test_closest_t_at_control_points() {
        let p0 = Vec2::new(0.0, 0.0);
        let p1 = Vec2::new(0.5, 1.0);
        let p2 = Vec2::new(1.0, 0.0);

        // Query at start
        let t = closest_t_on_bezier(p0, p0, p1, p2);
        assert!(t < 0.05, "t at start should be ~0, got {}", t);

        // Query at end
        let t = closest_t_on_bezier(p2, p0, p1, p2);
        assert!(t > 0.95, "t at end should be ~1, got {}", t);
    }

    #[test]
    fn test_sweep_straight_line() {
        // Straight bezier (control point on line)
        let p0 = Vec2::new(-1.0, 0.0);
        let p1 = Vec2::new(0.0, 0.0);
        let p2 = Vec2::new(1.0, 0.0);

        // Point above the line at x=0
        let result = modifier_sweep_bezier(Vec3::new(0.0, 2.0, 0.5), p0, p1, p2);
        assert!(
            (result.x - 0.5).abs() < 0.05,
            "perp dist should be ~0.5, got {}",
            result.x
        );
        assert!((result.y - 2.0).abs() < 1e-6, "y should be preserved");
    }

    #[test]
    fn test_sweep_on_curve() {
        // Point exactly on the bezier curve
        let p0 = Vec2::new(0.0, 0.0);
        let p1 = Vec2::new(0.5, 1.0);
        let p2 = Vec2::new(1.0, 0.0);

        let mid = bezier_eval(p0, p1, p2, 0.5);
        let result = modifier_sweep_bezier(Vec3::new(mid.x, 3.0, mid.y), p0, p1, p2);
        assert!(
            result.x < 0.01,
            "perp dist on curve should be ~0, got {}",
            result.x
        );
    }
}
