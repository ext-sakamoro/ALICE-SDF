//! Helix SDF (Deep Fried Edition)
//!
//! 3D helix (spiral tube) shape along Y-axis.
//!
//! Author: Moroya Sakamoto

use crate::crispy::round_half_up;
use glam::{Vec2, Vec3};

/// Newton refinements of the nearest curve parameter (each starts at the
/// same-azimuth point of a wrap; 6 steps converge to f32 precision for the
/// slopes a helix tube is drawn with).
const HELIX_NEWTON_STEPS: u32 = 6;

/// Squared distance from `(r, theta, y)` (cylindrical) to the helix curve
/// `H(φ) = (R cos φ, pitch·φ/τ, R sin φ)`: the nearest φ minimises
/// `r² + R² − 2rR cos(φ − θ) + (y − cφ)²` (`c = pitch/τ`), found by Newton
/// from the same-azimuth candidates of the nearest three wraps.
///
/// Until 1.10.3 the law measured the distance to the same-azimuth point
/// only — an over-estimate that also depended on the (undefined) azimuth on
/// the axis, so the field jumped by up to 15× the sample spacing there.
#[inline(always)]
fn helix_curve_dist2(r: f32, theta: f32, py: f32, major_r: f32, pitch: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let c = pitch / tau;
    let two_rr = 2.0 * r * major_r;
    let k = round_half_up((py - theta * c) / pitch);
    let mut best = f32::MAX;
    for dk in [-1.0_f32, 0.0, 1.0] {
        let mut phi = (k + dk).mul_add(tau, theta);
        for _ in 0..HELIX_NEWTON_STEPS {
            let (s, co) = (phi - theta).sin_cos();
            let dy = c.mul_add(-phi, py);
            let f1 = two_rr.mul_add(s, -2.0 * c * dy);
            let f2 = two_rr.mul_add(co, 2.0 * c * c).max(1e-6);
            phi -= (f1 / f2).clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        }
        let co = (phi - theta).cos();
        let dy = c.mul_add(-phi, py);
        let d2 = dy.mul_add(dy, r.mul_add(r, major_r * major_r) - two_rr * co);
        best = best.min(d2);
    }
    best
}

/// SDF for a helix (spiral tube) along Y-axis
///
/// Tube of radius `minor_r` around the helix curve of radius `major_r`
/// and `pitch` per revolution, capped at `±half_height`. The distance to the
/// curve is the true nearest point (see [`helix_curve_dist2`]), so the
/// field is a distance bound with Lipschitz constant 1.
///
/// - `major_r`: major radius (distance from Y-axis to helix center)
/// - `minor_r`: minor radius (tube thickness)
/// - `pitch`: vertical distance per full revolution
/// - `half_height`: half the height along Y (caps the helix)
#[inline(always)]
pub fn sdf_helix(p: Vec3, major_r: f32, minor_r: f32, pitch: f32, half_height: f32) -> f32 {
    let r_xz = Vec2::new(p.x, p.z).length();
    // atan2(0, 0) is NaN on some GPUs (0 in libm): the azimuth is irrelevant
    // on the axis, pin it to 0 on every path
    let theta = if r_xz > 0.0 { p.z.atan2(p.x) } else { 0.0 };
    let d_tube = helix_curve_dist2(r_xz, theta, p.y, major_r, pitch).sqrt() - minor_r;
    let d_cap = p.y.abs() - half_height;
    d_tube.max(d_cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helix_on_tube() {
        // Point on the helix tube at (major_r, 0, 0)
        let d = sdf_helix(Vec3::new(1.0, 0.0, 0.0), 1.0, 0.2, 1.0, 2.0);
        assert!(d < 0.0, "Point on helix tube should be inside, got {}", d);
    }

    #[test]
    fn test_helix_curve_distance_matches_brute_force() {
        // dense sampling of the curve parameter is the oracle
        for (major_r, pitch) in [(1.0_f32, 0.5_f32), (0.6, 2.0), (1.5, 0.2)] {
            let c = pitch / std::f32::consts::TAU;
            for (x, y, z) in [
                (0.0, 0.3, 0.0),
                (1e-4, 0.7, 2e-4),
                (0.5, 0.1, 0.9),
                (1.4, -0.6, -0.3),
                (0.2, 1.1, -1.2),
                (0.9, 0.05, 0.1),
            ] {
                let q = Vec3::new(x, y, z);
                // sample ±3 turns around the height-matched parameter y / c
                let centre = y / c;
                let brute = (-40_000..=40_000)
                    .map(|i| {
                        let phi = (i as f32).mul_add(5e-4, centre);
                        (q - Vec3::new(major_r * phi.cos(), c * phi, major_r * phi.sin())).length()
                    })
                    .fold(f32::MAX, f32::min);
                let r = Vec2::new(x, z).length();
                let d = helix_curve_dist2(r, z.atan2(x), y, major_r, pitch).sqrt();
                assert!(
                    (d - brute).abs() < 2e-3,
                    "R={major_r} pitch={pitch} q={q:?}: newton {d} vs brute {brute}"
                );
            }
        }
    }

    #[test]
    fn test_helix_continuous_on_axis() {
        // the azimuth is undefined on the axis: neighbouring samples must agree
        let a = sdf_helix(Vec3::new(1e-5, 0.37, 0.0), 1.0, 0.2, 0.5, 2.0);
        let b = sdf_helix(Vec3::new(0.0, 0.37, 1e-5), 1.0, 0.2, 0.5, 2.0);
        let c = sdf_helix(Vec3::new(-1e-5, 0.37, -1e-5), 1.0, 0.2, 0.5, 2.0);
        assert!((a - b).abs() < 1e-4 && (a - c).abs() < 1e-4, "{a} {b} {c}");
        // and equal the analytic axis distance R - minor_r
        assert!((a - 0.8).abs() < 1e-3, "{a}");
    }

    #[test]
    fn test_helix_far_outside() {
        let d = sdf_helix(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.2, 1.0, 2.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_helix_center_outside() {
        // Center of helix (on the axis) should be outside
        let d = sdf_helix(Vec3::ZERO, 1.0, 0.2, 1.0, 2.0);
        assert!(d > 0.0, "Center (on axis) should be outside, got {}", d);
    }

    #[test]
    fn test_helix_wrap_at_pitch() {
        // At theta=0 (x=major_r, z=0), helix wraps are at y=k*pitch
        // So point (major_r, pitch, 0) should be on the tube (k=1 wrap)
        let d = sdf_helix(Vec3::new(1.0, 1.0, 0.0), 1.0, 0.2, 1.0, 2.0);
        assert!(d < 0.0, "Point on helix wrap should be inside, got {}", d);
    }
}
