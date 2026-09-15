//! Toon shading band primitives
//!
//! Discrete band shading and posterize operators for stylized rendering.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Hard N-band step ramp on a diffuse-like scalar
///
/// Input `n_dot_l` is clamped to `[0, 1]` and quantized into `bands`
/// discrete steps. Returns a value in `[0, 1]`.
///
/// # Panics
/// Panics if `bands == 0`.
#[inline]
#[must_use]
pub fn toon_ramp(n_dot_l: f32, bands: u32) -> f32 {
    assert!(bands > 0, "bands must be > 0");
    let clamped = n_dot_l.clamp(0.0, 1.0);
    let bands_f = bands as f32;
    let idx = (clamped * bands_f).floor().min(bands_f - 1.0);
    idx / (bands_f - 1.0).max(1.0)
}

/// Soft N-band ramp with smoothstep transitions at each band edge
///
/// `smoothness` is the half-width of each transition in normalized band
/// units and is clamped to `[0.001, 0.5]`.
///
/// # Panics
/// Panics if `bands == 0`.
#[inline]
#[must_use]
pub fn soft_toon_ramp(n_dot_l: f32, bands: u32, smoothness: f32) -> f32 {
    assert!(bands > 0, "bands must be > 0");
    let clamped = n_dot_l.clamp(0.0, 1.0);
    let bands_f = bands as f32;
    let scaled = clamped * bands_f;
    let idx = scaled.floor().min(bands_f - 1.0);
    let frac = (scaled - idx).clamp(0.0, 1.0);
    let s = smoothness.clamp(0.001, 0.5);
    let t = smoothstep(0.5 - s, 0.5 + s, frac);
    ((idx + t) / bands_f).clamp(0.0, 1.0)
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let denom = (edge1 - edge0).abs().max(1e-6);
    let t = ((x - edge0) / denom).clamp(0.0, 1.0);
    t * t * 2.0f32.mul_add(-t, 3.0)
}

/// Two-tone shading: `shadow` when `n_dot_l < threshold`, otherwise `light`
///
/// `n_dot_l` is clamped to `[0, 1]` before comparison.
#[inline]
#[must_use]
pub fn two_tone(n_dot_l: f32, shadow: Vec3, light: Vec3, threshold: f32) -> Vec3 {
    let t = n_dot_l.clamp(0.0, 1.0);
    if t < threshold {
        shadow
    } else {
        light
    }
}

/// Posterize a color to `levels` discrete steps per channel
///
/// # Panics
/// Panics if `levels < 2`.
#[inline]
#[must_use]
pub fn posterize_color(color: Vec3, levels: u32) -> Vec3 {
    assert!(levels >= 2, "levels must be >= 2");
    let steps = levels as f32;
    let denom = (steps - 1.0).max(1.0);
    let quantize = |c: f32| ((c.clamp(0.0, 1.0) * steps).floor() / denom).clamp(0.0, 1.0);
    Vec3::new(quantize(color.x), quantize(color.y), quantize(color.z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toon_ramp_dark_returns_zero() {
        assert!(toon_ramp(0.0, 3).abs() < 1e-6);
    }

    #[test]
    fn toon_ramp_bright_returns_one() {
        assert!((toon_ramp(1.0, 3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn toon_ramp_same_band_same_output() {
        let a = toon_ramp(0.10, 3);
        let b = toon_ramp(0.20, 3);
        assert!(
            (a - b).abs() < 1e-6,
            "0.10 and 0.20 should sit in the same band"
        );
    }

    #[test]
    fn toon_ramp_next_band_differs() {
        let a = toon_ramp(0.10, 3);
        let b = toon_ramp(0.50, 3);
        assert!(
            (a - b).abs() > 1e-3,
            "0.10 and 0.50 should land in different bands"
        );
    }

    #[test]
    fn soft_toon_ramp_stays_in_unit_interval() {
        for i in 0..=100 {
            let x = i as f32 / 100.0;
            let out = soft_toon_ramp(x, 4, 0.1);
            assert!((0.0..=1.0).contains(&out), "out of range at x={x}: {out}");
        }
    }

    #[test]
    fn two_tone_shadow_side() {
        assert_eq!(two_tone(0.2, Vec3::ZERO, Vec3::ONE, 0.5), Vec3::ZERO);
    }

    #[test]
    fn two_tone_light_side() {
        assert_eq!(two_tone(0.8, Vec3::ZERO, Vec3::ONE, 0.5), Vec3::ONE);
    }

    #[test]
    fn posterize_two_levels_is_binary() {
        let out = posterize_color(Vec3::new(0.25, 0.5, 0.75), 2);
        for c in [out.x, out.y, out.z] {
            assert!(c == 0.0 || c == 1.0, "expected 0 or 1, got {c}");
        }
    }

    #[test]
    fn posterize_reduces_unique_values() {
        // With 4 levels, distinct inputs collapse into at most 4 outputs
        let mut outputs = std::collections::HashSet::new();
        for i in 0..100 {
            let x = i as f32 / 100.0;
            let out = posterize_color(Vec3::splat(x), 4);
            outputs.insert((out.x * 1000.0) as i32);
        }
        assert!(
            outputs.len() <= 4,
            "expected <= 4 unique levels, got {}",
            outputs.len()
        );
    }

    #[test]
    #[should_panic(expected = "bands must be > 0")]
    fn toon_ramp_zero_bands_panics() {
        let _ = toon_ramp(0.5, 0);
    }

    #[test]
    #[should_panic(expected = "levels must be >= 2")]
    fn posterize_one_level_panics() {
        let _ = posterize_color(Vec3::ZERO, 1);
    }
}
