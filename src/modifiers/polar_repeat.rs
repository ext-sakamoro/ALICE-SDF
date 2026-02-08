//! Polar repetition modifier — repeat around Y-axis
//!
//! Ported from Baker's `opPolarRepeat(p, count)`.
//! Repeats the child SDF `count` times evenly around the Y-axis.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Repeat point around Y-axis by `count` copies.
///
/// Returns the transformed point (pass to child SDF).
/// Uses atan2 + modulo to fold the angle into one sector.
#[inline(always)]
pub fn modifier_polar_repeat(p: Vec3, count: u32) -> Vec3 {
    let count_f = count as f32;
    let angle = std::f32::consts::TAU / count_f;
    let a = p.z.atan2(p.x) + angle * 0.5;
    let r = (p.x * p.x + p.z * p.z).sqrt();
    // Stable modulo: add large multiple to avoid negative values
    let a_mod = ((a + 100.0 * angle) % angle) - angle * 0.5;
    Vec3::new(r * a_mod.cos(), p.y, r * a_mod.sin())
}

/// Polar repeat — Division Exorcism edition.
///
/// Takes precomputed `sector = TAU / count` and `recip_sector = count / TAU`
/// to eliminate division and replace `%` with the round-trick:
/// `a - sector * round(a * recip_sector)` instead of `a % sector`.
#[inline(always)]
pub fn modifier_polar_repeat_rk(p: Vec3, sector: f32, recip_sector: f32) -> Vec3 {
    let a = p.z.atan2(p.x);
    let r = (p.x * p.x + p.z * p.z).sqrt();

    // Round trick: a - sector * round(a / sector)
    // = a - sector * round(a * recip_sector)
    let sector_angle = a - sector * (a * recip_sector).round();

    let (s, c) = sector_angle.sin_cos();
    Vec3::new(r * c, p.y, r * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polar_repeat_on_axis() {
        // On Y-axis, r=0, result should be on Y-axis
        let p = Vec3::new(0.0, 5.0, 0.0);
        let q = modifier_polar_repeat(p, 6);
        assert!(q.x.abs() < 0.001, "x should be ~0: {}", q.x);
        assert!((q.y - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_polar_repeat_preserves_radius() {
        // Radius should be preserved
        let p = Vec3::new(3.0, 0.0, 4.0);
        let q = modifier_polar_repeat(p, 8);
        let r_original = (p.x * p.x + p.z * p.z).sqrt();
        let r_result = (q.x * q.x + q.z * q.z).sqrt();
        assert!((r_original - r_result).abs() < 0.001);
    }

    #[test]
    fn test_polar_repeat_preserves_y() {
        let p = Vec3::new(1.0, 7.0, 2.0);
        let q = modifier_polar_repeat(p, 4);
        assert!((q.y - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_polar_repeat_symmetry() {
        // Points at equal angles should map to the same result
        let r = 5.0;
        let count = 6u32;
        let angle = std::f32::consts::TAU / count as f32;

        let p1 = Vec3::new(r * 0.0_f32.cos(), 0.0, r * 0.0_f32.sin());
        let p2 = Vec3::new(r * angle.cos(), 0.0, r * angle.sin());

        let q1 = modifier_polar_repeat(p1, count);
        let q2 = modifier_polar_repeat(p2, count);

        assert!(
            (q1 - q2).length() < 0.01,
            "Symmetric points should map to same result: {:?} vs {:?}",
            q1,
            q2
        );
    }
}
