//! Composition primitives: vignette, bloom, chromatic aberration
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Vignette darkening multiplier as a function of UV distance from center
///
/// UV space is `[0, 1]` with center at `(0.5, 0.5)`. Returns 1.0 inside
/// `radius`, tapered to 0.0 at `radius + softness` via smoothstep.
#[inline]
#[must_use]
pub fn vignette(uv_x: f32, uv_y: f32, radius: f32, softness: f32) -> f32 {
    let dx = uv_x - 0.5;
    let dy = uv_y - 0.5;
    let d = dx.hypot(dy);
    let inner = radius.max(0.0);
    let outer = (inner + softness.max(0.0)).max(inner + 1e-6);
    if d <= inner {
        1.0
    } else if d >= outer {
        0.0
    } else {
        let t = ((d - inner) / (outer - inner)).clamp(0.0, 1.0);
        (t * t).mul_add(-2.0f32.mul_add(-t, 3.0), 1.0)
    }
}

/// Toon bloom: pass through the color scaled by `intensity` when any
/// channel exceeds `threshold`; otherwise return zero
///
/// Intended to feed a downstream Gaussian blur pass for the bloom halo.
#[inline]
#[must_use]
pub fn bloom_toon(color: Vec3, threshold: f32, intensity: f32) -> Vec3 {
    let max_ch = color.x.max(color.y).max(color.z);
    if max_ch > threshold {
        color * intensity
    } else {
        Vec3::ZERO
    }
}

/// Chromatic aberration offsets: three UV positions for R/G/B sampling
///
/// Returns `[red_uv, green_uv, blue_uv]` where each entry is `(u, v)`.
/// Red is pushed outward from `(0.5, 0.5)`, blue pulled inward, green
/// stays at the input UV. `strength` scales the offset (positive typical).
#[inline]
#[must_use]
pub fn chromatic_offsets(uv_x: f32, uv_y: f32, strength: f32) -> [(f32, f32); 3] {
    let dx = uv_x - 0.5;
    let dy = uv_y - 0.5;
    [
        (uv_x + dx * strength, uv_y + dy * strength),
        (uv_x, uv_y),
        (uv_x - dx * strength, uv_y - dy * strength),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vignette_center_is_full() {
        assert!((vignette(0.5, 0.5, 0.3, 0.1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vignette_corner_is_zero() {
        // Corner distance from center is sqrt(0.5) ~= 0.707; radius+softness < that
        assert!(vignette(0.0, 0.0, 0.3, 0.1).abs() < 1e-6);
    }

    #[test]
    fn vignette_bounded_across_uv() {
        for i in 0..=10 {
            for j in 0..=10 {
                let uvx = i as f32 / 10.0;
                let uvy = j as f32 / 10.0;
                let out = vignette(uvx, uvy, 0.3, 0.2);
                assert!((0.0..=1.0).contains(&out), "out of range: {out}");
            }
        }
    }

    #[test]
    fn bloom_below_threshold_is_zero() {
        let out = bloom_toon(Vec3::new(0.4, 0.4, 0.4), 0.5, 1.0);
        assert_eq!(out, Vec3::ZERO);
    }

    #[test]
    fn bloom_above_threshold_scales_color() {
        let color = Vec3::new(0.9, 0.9, 0.9);
        let out = bloom_toon(color, 0.5, 0.5);
        assert!((out - color * 0.5).length() < 1e-6);
    }

    #[test]
    fn bloom_partial_channel_above_threshold() {
        // Only red exceeds threshold; whole color still passes
        let color = Vec3::new(0.9, 0.1, 0.1);
        let out = bloom_toon(color, 0.5, 1.0);
        assert!((out - color).length() < 1e-6);
    }

    #[test]
    fn chromatic_offsets_center_unchanged() {
        let offs = chromatic_offsets(0.5, 0.5, 0.02);
        for (u, v) in &offs {
            assert!((*u - 0.5).abs() < 1e-6 && (*v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn chromatic_offsets_green_matches_input() {
        let offs = chromatic_offsets(0.7, 0.3, 0.05);
        let green = offs[1];
        assert!((green.0 - 0.7).abs() < 1e-6 && (green.1 - 0.3).abs() < 1e-6);
    }

    #[test]
    fn chromatic_offsets_red_and_blue_symmetric_around_input() {
        let (uvx, uvy) = (0.7_f32, 0.3_f32);
        let strength = 0.05_f32;
        let offs = chromatic_offsets(uvx, uvy, strength);
        let red = offs[0];
        let blue = offs[2];
        let mid_x = f32::midpoint(red.0, blue.0);
        let mid_y = f32::midpoint(red.1, blue.1);
        assert!((mid_x - uvx).abs() < 1e-6 && (mid_y - uvy).abs() < 1e-6);
    }
}
