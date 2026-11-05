//! Bezier SDF — distance to a quadratic Bezier curve with radius
//!
//! Ported from Baker's `sdBezier(pos, A, B, C, rad)` (IQ's formulation).
//! Uses an analytical cubic solver to find the closest point on the curve.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::LawMinMax;
use glam::Vec3;

/// Newton steps applied to every candidate parameter (see [`refine`])
const NEWTON_STEPS: usize = 2;

/// |q(t)|² with q(t) = B(t) − pos = dv + (cv + b·t)·t
#[inline]
fn d2_at(dv: Vec3, cv: Vec3, b: Vec3, t: f32) -> f32 {
    let q = dv + (cv + b * t) * t;
    q.dot(q)
}

/// [`NEWTON_STEPS`] clamped Newton steps on q(t)·q′(t) = 0 (q′ = cv + 2b·t, q″ = 2b) from `t`. A step
/// is skipped where (|q|²)″/2 = |q′|² + q·q″ is not positive (Newton would climb). The closed form below
/// picks a wrong root in f32 when |a − 2b + c| is small (cancellation after kk = 1/|a − 2b + c|²), which
/// made the law state a distance larger than the true one (measured: 0.004022 for a point 0.000126 from
/// a stroke with |a − 2b + c| = 0.06); the steps move every candidate to its nearest stationary point.
#[inline]
fn refine(dv: Vec3, cv: Vec3, b: Vec3, mut t: f32) -> f32 {
    for _ in 0..NEWTON_STEPS {
        let q = dv + (cv + b * t) * t;
        let dq = cv + b * (2.0 * t);
        let g = q.dot(dq);
        let gp = dq.dot(dq) + 2.0 * q.dot(b);
        if gp > 0.0 {
            t = (t - g / gp).clamp(0.0, 1.0);
        }
    }
    t
}

/// Distance from point `pos` to a quadratic Bezier curve `A→B→C`
/// with tube `radius`.
///
/// The Bezier is defined by three control points:
/// - `a`: start point
/// - `b`: control point
/// - `c`: end point
///
/// The result is the distance to the curve minus `radius`,
/// creating a tube of the given radius around the curve.
#[inline]
pub fn sdf_bezier(pos: Vec3, a: Vec3, b: Vec3, c: Vec3, radius: f32) -> f32 {
    let ab = b - a;
    let ba2c = a - 2.0 * b + c;
    let cv = ab * 2.0;
    let dv = a - pos;

    // Degenerate case: collinear control points → capsule fallback
    let ba2c_dot = ba2c.dot(ba2c);
    if ba2c_dot < 1e-10 {
        let ac = c - a;
        let ac_dot = ac.dot(ac);
        if ac_dot < 1e-10 {
            return (pos - a).length() - radius;
        }
        let t = ((pos - a).dot(ac) / ac_dot).clamp(0.0, 1.0);
        return (pos - a - ac * t).length() - radius;
    }

    let kk = 1.0 / ba2c_dot;
    let kx = kk * ab.dot(ba2c);
    let ky = kk * (2.0f32 * ab.dot(ab) + dv.dot(ba2c)) / 3.0;
    let kz = kk * dv.dot(ab);

    let p2 = ky - kx * kx;
    let p3 = p2 * p2 * p2;
    let q2 = kx * ((2.0 * kx) * kx + -(3.0 * ky)) + kz;
    let h = 4.0f32 * p3 + (q2 * q2);

    let res = if h >= 0.0 {
        // One real root
        let h_sqrt = h.sqrt();
        let x0 = (h_sqrt - q2) * 0.5;
        let x1 = (-h_sqrt - q2) * 0.5;
        let uv_x = x0.signum() * alice_det_math::cbrt(x0.abs());
        let uv_y = x1.signum() * alice_det_math::cbrt(x1.abs());
        let t = (uv_x + uv_y - kx).clamp(0.0, 1.0);
        let tr = refine(dv, cv, ba2c, t);
        d2_at(dv, cv, ba2c, t).law_min(d2_at(dv, cv, ba2c, tr))
    } else {
        // Three real roots — take closest
        let z = (-p2).sqrt();
        let v = alice_det_math::acos(q2 / (p2 * z * 2.0)) / 3.0;
        let m = alice_det_math::cos(v);
        let n = alice_det_math::sin(v) * 1.732_050_8;

        let t0 = ((m + m) * z + -kx).clamp(0.0, 1.0);
        let t1 = ((-n - m) * z + -kx).clamp(0.0, 1.0);

        let d0 = d2_at(dv, cv, ba2c, t0).law_min(d2_at(dv, cv, ba2c, refine(dv, cv, ba2c, t0)));
        let d1 = d2_at(dv, cv, ba2c, t1).law_min(d2_at(dv, cv, ba2c, refine(dv, cv, ba2c, t1)));
        d0.law_min(d1)
    };
    // every candidate is a point of the curve, so the minimum never states less than the true
    // distance beyond rounding; the refinement only removes over-statement (measured over near-straight
    // to strongly bent quadratics: worst over-statement 5e-7 = f32 rounding, was 1.17e-3; 1 step is
    // enough on that sweep, the second is margin)
    let res = res.sqrt();

    res - radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bezier_on_start() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(0.5, 1.0, 0.0);
        let c = Vec3::new(1.0, 0.0, 0.0);
        let radius = 0.1;

        // On start point, distance should be ~-radius (inside tube)
        let d = sdf_bezier(a, a, b, c, radius);
        assert!((d + radius).abs() < 0.01, "At start: {}", d);
    }

    #[test]
    fn test_bezier_on_end() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(0.5, 1.0, 0.0);
        let c = Vec3::new(1.0, 0.0, 0.0);
        let radius = 0.1;

        // On end point, distance should be ~-radius (inside tube)
        let d = sdf_bezier(c, a, b, c, radius);
        assert!((d + radius).abs() < 0.01, "At end: {}", d);
    }

    #[test]
    fn test_bezier_far_away() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(0.5, 1.0, 0.0);
        let c = Vec3::new(1.0, 0.0, 0.0);
        let radius = 0.1;

        // Far away point
        let d = sdf_bezier(Vec3::new(0.0, 10.0, 0.0), a, b, c, radius);
        assert!(d > 5.0, "Far away: {}", d);
    }

    #[test]
    fn test_bezier_straight_line() {
        // Collinear control points → straight line
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(0.5, 0.0, 0.0);
        let c = Vec3::new(1.0, 0.0, 0.0);
        let radius = 0.2;

        // Point perpendicular to midpoint, at distance 0.5
        let d = sdf_bezier(Vec3::new(0.5, 0.5, 0.0), a, b, c, radius);
        assert!((d - 0.3).abs() < 0.05, "Perpendicular: {}", d);
    }
}
