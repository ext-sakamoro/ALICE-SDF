//! Bezier SDF — distance to a quadratic Bezier curve with radius
//!
//! Ported from Baker's `sdBezier(pos, A, B, C, rad)` (IQ's formulation).
//! Uses an analytical cubic solver to find the closest point on the curve.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

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
    let ky = kk * (2.0 * ab.dot(ab) + dv.dot(ba2c)) / 3.0;
    let kz = kk * dv.dot(ab);

    let p2 = ky - kx * kx;
    let p3 = p2 * p2 * p2;
    let q2 = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    let h = q2 * q2 + 4.0 * p3;

    let res = if h >= 0.0 {
        // One real root
        let h_sqrt = h.sqrt();
        let x0 = (h_sqrt - q2) * 0.5;
        let x1 = (-h_sqrt - q2) * 0.5;
        let uv_x = x0.signum() * x0.abs().cbrt();
        let uv_y = x1.signum() * x1.abs().cbrt();
        let t = (uv_x + uv_y - kx).clamp(0.0, 1.0);
        (dv + (cv + ba2c * t) * t).length()
    } else {
        // Three real roots — take closest
        let z = (-p2).sqrt();
        let v = (q2 / (p2 * z * 2.0)).acos() / 3.0;
        let m = v.cos();
        let n = v.sin() * 1.732050808;

        let t0 = ((m + m) * z - kx).clamp(0.0, 1.0);
        let t1 = ((-n - m) * z - kx).clamp(0.0, 1.0);

        let d0 = (dv + (cv + ba2c * t0) * t0).length();
        let d1 = (dv + (cv + ba2c * t1) * t1).length();
        d0.min(d1)
    };

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
