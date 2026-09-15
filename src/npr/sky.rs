//! Sky, cloud, and atmospheric primitives
//!
//! Procedural sky building blocks: banded gradient, cloud coverage from a
//! noise sample, distance-based stepped color, light shaft intensity, and
//! sun disc. All inputs are scalars or vectors; no atlas is used.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Palette-driven sky gradient, sampled by view-direction elevation
///
/// The `y` component of `dir` selects the palette position: `y = -1`
/// picks the first entry, `y = +1` picks the last. Between palette
/// entries the color is linearly interpolated.
#[inline]
#[must_use]
pub fn sky_gradient_bands(dir: Vec3, palette: &[Vec3]) -> Vec3 {
    if palette.is_empty() {
        return Vec3::ZERO;
    }
    if palette.len() == 1 {
        return palette[0];
    }
    let n = palette.len();
    let t = f32::midpoint(dir.y, 1.0).clamp(0.0, 1.0);
    let scaled = t * (n as f32 - 1.0);
    let idx = (scaled.floor() as usize).min(n - 1);
    let next = (idx + 1).min(n - 1);
    let frac = (scaled - idx as f32).clamp(0.0, 1.0);
    palette[idx].lerp(palette[next], frac)
}

/// Cloud coverage mask from a scalar noise sample
///
/// `noise_sample` and `coverage` are clamped to `[0, 1]`. `softness`
/// controls the sharpness of the coverage transition (smaller = harder).
/// Returns a density in `[0, 1]`.
#[inline]
#[must_use]
pub fn puffy_cloud_layer(noise_sample: f32, coverage: f32, softness: f32) -> f32 {
    let n = noise_sample.clamp(0.0, 1.0);
    let c = coverage.clamp(0.0, 1.0);
    let s = softness.max(1e-4);
    ((n - (1.0 - c)) / s).clamp(0.0, 1.0)
}

/// Distance-based color quantize (stepped atmospheric fade)
///
/// Interpolates between `near` and `far` in `bands` discrete steps of
/// `distance / max_distance`.
///
/// # Panics
/// Panics if `bands == 0`.
#[inline]
#[must_use]
pub fn distance_color_quantize(
    distance: f32,
    max_distance: f32,
    near: Vec3,
    far: Vec3,
    bands: u32,
) -> Vec3 {
    assert!(bands > 0, "bands must be > 0");
    let t = (distance / max_distance.max(1e-6)).clamp(0.0, 1.0);
    let b = bands as f32;
    let stepped = ((t * b).floor() / (b - 1.0).max(1.0)).clamp(0.0, 1.0);
    near.lerp(far, stepped)
}

/// Light-shaft intensity along the view direction
///
/// Returns the fraction of the shaft visible along `view` when the sun is
/// at `to_sun`. `density` sharpens the shaft (higher = tighter).
#[inline]
#[must_use]
pub fn light_shaft_beam(view: Vec3, to_sun: Vec3, density: f32) -> f32 {
    let v = view.normalize_or_zero();
    let s = to_sun.normalize_or_zero();
    let cos_theta = v.dot(s).max(0.0);
    cos_theta.powf(density.max(1.0))
}

/// Sun disc intensity within an angular `radius` around `to_sun`
///
/// `softness` extends the taper past the disc edge. Both `radius` and
/// `softness` are in radians.
#[inline]
#[must_use]
pub fn sun_disc(view: Vec3, to_sun: Vec3, radius: f32, softness: f32) -> f32 {
    let v = view.normalize_or_zero();
    let s = to_sun.normalize_or_zero();
    let cos_theta = v.dot(s);
    let r = radius.max(0.0);
    let soft = softness.max(0.0);
    let cos_inner = r.cos();
    let cos_outer = (r + soft).cos();
    if cos_theta >= cos_inner {
        1.0
    } else if cos_theta <= cos_outer {
        0.0
    } else {
        let denom = (cos_inner - cos_outer).max(1e-6);
        ((cos_theta - cos_outer) / denom).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sky_gradient_horizon_picks_first_entry() {
        let palette = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)];
        let c = sky_gradient_bands(Vec3::new(0.0, -1.0, 0.0), &palette);
        assert!((c - palette[0]).length() < 1e-4);
    }

    #[test]
    fn sky_gradient_zenith_picks_last_entry() {
        let palette = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)];
        let c = sky_gradient_bands(Vec3::new(0.0, 1.0, 0.0), &palette);
        assert!((c - palette[1]).length() < 1e-4);
    }

    #[test]
    fn sky_gradient_empty_palette_returns_zero() {
        assert_eq!(sky_gradient_bands(Vec3::ZERO, &[]), Vec3::ZERO);
    }

    #[test]
    fn sky_gradient_single_entry_returns_it() {
        let single = [Vec3::new(0.5, 0.25, 0.1)];
        assert_eq!(sky_gradient_bands(Vec3::Y, &single), single[0]);
    }

    #[test]
    fn puffy_cloud_no_coverage_is_clear() {
        assert!(puffy_cloud_layer(0.3, 0.0, 0.1).abs() < 1e-6);
    }

    #[test]
    fn puffy_cloud_full_coverage_is_opaque() {
        assert!((puffy_cloud_layer(0.9, 1.0, 0.1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn puffy_cloud_output_bounded() {
        for n in 0..=10 {
            for c in 0..=10 {
                let d = puffy_cloud_layer(n as f32 / 10.0, c as f32 / 10.0, 0.05);
                assert!((0.0..=1.0).contains(&d), "out of range at n={n} c={c}: {d}");
            }
        }
    }

    #[test]
    fn distance_color_quantize_near_returns_near() {
        let c = distance_color_quantize(0.0, 100.0, Vec3::X, Vec3::Y, 4);
        assert!((c - Vec3::X).length() < 1e-4);
    }

    #[test]
    fn distance_color_quantize_far_returns_far() {
        let c = distance_color_quantize(100.0, 100.0, Vec3::X, Vec3::Y, 4);
        assert!((c - Vec3::Y).length() < 1e-4);
    }

    #[test]
    fn distance_color_quantize_beyond_max_clamped() {
        let c = distance_color_quantize(200.0, 100.0, Vec3::X, Vec3::Y, 4);
        assert!((c - Vec3::Y).length() < 1e-4);
    }

    #[test]
    fn light_shaft_aligned_returns_peak() {
        let sun = Vec3::new(0.0, 1.0, 0.0);
        assert!((light_shaft_beam(sun, sun, 4.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn light_shaft_opposite_returns_zero() {
        let sun = Vec3::new(0.0, 1.0, 0.0);
        assert!(light_shaft_beam(-sun, sun, 4.0).abs() < 1e-4);
    }

    #[test]
    fn light_shaft_perpendicular_returns_zero() {
        let sun = Vec3::new(0.0, 1.0, 0.0);
        assert!(light_shaft_beam(Vec3::X, sun, 4.0).abs() < 1e-4);
    }

    #[test]
    fn sun_disc_center_full_intensity() {
        let sun = Vec3::new(0.0, 1.0, 0.0);
        assert!((sun_disc(sun, sun, 0.05, 0.01) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn sun_disc_off_axis_is_zero() {
        let sun = Vec3::new(0.0, 1.0, 0.0);
        assert!(sun_disc(Vec3::X, sun, 0.05, 0.01).abs() < 1e-4);
    }

    #[test]
    fn sun_disc_edge_softness_tapered() {
        // Just outside the hard radius but inside the soft band
        let sun = Vec3::new(0.0, 1.0, 0.0);
        let radius = 0.05_f32;
        let softness = 0.03_f32;
        // Rotate view slightly off sun in XY plane
        let angle = radius + softness * 0.5;
        let (s, c) = angle.sin_cos();
        let view = Vec3::new(s, c, 0.0);
        let intensity = sun_disc(view, sun, radius, softness);
        assert!(
            intensity > 0.0 && intensity < 1.0,
            "expected tapered value, got {intensity}"
        );
    }

    #[test]
    #[should_panic(expected = "bands must be > 0")]
    fn distance_color_quantize_zero_bands_panics() {
        let _ = distance_color_quantize(1.0, 10.0, Vec3::ZERO, Vec3::ONE, 0);
    }
}
