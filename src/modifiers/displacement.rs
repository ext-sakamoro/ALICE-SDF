//! Displacement modifier — add sin-based noise to distance field
//!
//! Ported from Baker's `opDisplacement(d, p, strength)`.
//! Uses `sin(5x)*sin(5y)*sin(5z)*strength` displacement pattern.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Apply sin-based displacement to a distance value.
///
/// This is a post-processing modifier: evaluate the child SDF first,
/// then add the displacement to the distance.
///
/// Formula: `d + sin(5*p.x) * sin(5*p.y) * sin(5*p.z) * strength`
#[inline(always)]
pub fn modifier_displacement(d: f32, p: Vec3, strength: f32) -> f32 {
    let disp = (p.x * 5.0).sin() * (p.y * 5.0).sin() * (p.z * 5.0).sin();
    d + disp * strength
}

/// Apply sin-based displacement with custom amplitude + per-axis frequency.
///
/// Formula: `d + sin(fx*p.x) * sin(fy*p.y) * sin(fz*p.z) * amplitude`
/// (= 異方性 Vec3 frequency、 例: `Vec3::new(10, 40, 10)` で軸別周波数で細長菱形鱗)
#[inline(always)]
pub fn modifier_sine_displacement(d: f32, p: Vec3, amplitude: f32, frequency: Vec3) -> f32 {
    let disp = (p.x * frequency.x).sin() * (p.y * frequency.y).sin() * (p.z * frequency.z).sin();
    d + disp * amplitude
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_displacement_zero_strength() {
        let d = 1.0;
        let p = Vec3::new(0.3, 0.7, 1.1);
        let result = modifier_displacement(d, p, 0.0);
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_displacement_bounded() {
        // Displacement should be bounded by strength
        for i in 0..100 {
            let x = (i as f32) * 0.137;
            let y = (i as f32) * 0.291;
            let z = (i as f32) * 0.473;
            let p = Vec3::new(x, y, z);
            let strength = 0.5;
            let d = 1.0;
            let result = modifier_displacement(d, p, strength);
            assert!(result >= d - strength, "Below bound: {}", result);
            assert!(result <= d + strength, "Above bound: {}", result);
        }
    }

    #[test]
    fn test_displacement_at_origin() {
        // sin(0)*sin(0)*sin(0) = 0, so displacement at origin is 0
        let d = 2.0;
        let result = modifier_displacement(d, Vec3::ZERO, 1.0);
        assert!((result - 2.0).abs() < 0.001);
    }
}
