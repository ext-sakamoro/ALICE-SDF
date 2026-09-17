//! Sweep modifier: extrude 2D cross-section along a quadratic Bezier curve
//!
//! Sweeps a child SDF (2D cross-section) along a quadratic Bezier path
//! in the XZ plane. The child is evaluated at (perpendicular_distance, y, 0).
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Evaluate a quadratic Bezier curve at parameter t
#[cfg(test)]
fn bezier_eval(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) -> Vec2 {
    let omt = 1.0 - t;
    p0 * (omt * omt) + p1 * (2.0 * omt * t) + p2 * (t * t)
}

/// Distance from `q` to the quadratic Bézier `A B C` (Inigo Quilez's `sdBezier`).
///
/// The nearest parameter is a root of a depressed cubic, solved in closed
/// form (one real root via Cardano, or three via the trigonometric form, of
/// which the two candidates are compared). Exact and continuous;
/// the pre-1.11.0 "5 samples + Newton" search jumped between local minima
/// (difference quotients 800× the sample spacing, Lipschitz property test).
///
/// A degenerate curve (`A − 2B + C ≈ 0`, a straight segment) is measured as
/// the segment `A C`.
#[inline]
pub fn bezier_distance_2d(q: Vec2, p0: Vec2, p1: Vec2, p2: Vec2) -> f32 {
    let a = p1 - p0;
    let b = p0 - p1 * 2.0 + p2;
    let c = a * 2.0;
    let d = p0 - q;
    let bb = b.dot(b);
    if bb < 1e-12 {
        // straight segment A → C
        let ac = p2 - p0;
        let t = (-d.dot(ac) / ac.dot(ac).max(1e-12)).clamp(0.0, 1.0);
        return (d + ac * t).length();
    }
    let kk = 1.0 / bb;
    let kx = kk * a.dot(b);
    let ky = kk * (2.0f32 * a.dot(a) + d.dot(b)) / 3.0;
    let kz = kk * d.dot(a);
    let p = ky - kx * kx;
    let p3 = p * p * p;
    let qq = kx * (3.0f32 * -ky + (2.0 * kx * kx)) + kz;
    let h = 4.0f32 * p3 + (qq * qq);
    let dot2 = |v: Vec2| v.dot(v);
    let res = if h >= 0.0 {
        let h = h.sqrt();
        let x0 = (h - qq) * 0.5;
        let x1 = (-h - qq) * 0.5;
        let u = x0.signum() * alice_det_math::cbrt(x0.abs());
        let v = x1.signum() * alice_det_math::cbrt(x1.abs());
        let t = (u + v - kx).clamp(0.0, 1.0);
        dot2(d + (c + b * t) * t)
    } else {
        let z = (-p).sqrt();
        let v = alice_det_math::acos((qq / (p * z * 2.0)).clamp(-1.0, 1.0)) / 3.0;
        let m = alice_det_math::cos(v);
        let n = alice_det_math::sin(v) * 1.732_050_8;
        let t0 = ((m + m) * z + -kx).clamp(0.0, 1.0);
        let t1 = ((-n - m) * z + -kx).clamp(0.0, 1.0);
        // the third root cannot be the closest
        dot2(d + (c + b * t0) * t0).min(dot2(d + (c + b * t1) * t1))
    };
    res.sqrt()
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
    let d = bezier_distance_2d(Vec2::new(p.x, p.z), p0, p1, p2);
    Vec3::new(d, p.y, 0.0)
}

/// SIMD-friendly version: returns (perpendicular_distance, y) without Vec3 allocation
#[allow(clippy::too_many_arguments)]
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
    let d = bezier_distance_2d(Vec2::new(px, pz), p0, p1, p2);
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
    fn test_distance_at_control_points_and_brute_force() {
        let p0 = Vec2::new(0.0, 0.0);
        let p1 = Vec2::new(0.5, 1.0);
        let p2 = Vec2::new(1.0, 0.0);
        assert!(bezier_distance_2d(p0, p0, p1, p2) < 1e-6);
        assert!(bezier_distance_2d(p2, p0, p1, p2) < 1e-6);
        // closed form == brute-force minimum over 100k samples (both cubic branches)
        for (x, y) in [
            (0.5, 0.2),
            (0.5, 0.9),
            (-0.3, 0.4),
            (1.4, -0.2),
            (0.2, 0.6),
            (0.8, 0.45),
        ] {
            let q = Vec2::new(x, y);
            let brute = (0..=100_000)
                .map(|i| (q - bezier_eval(p0, p1, p2, i as f32 * 1e-5)).length())
                .fold(f32::MAX, f32::min);
            let d = bezier_distance_2d(q, p0, p1, p2);
            assert!(
                (d - brute).abs() < 1e-4,
                "{q:?}: closed {d} vs brute {brute}"
            );
        }
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
