//! Hatch, cross-hatch, paper grain, and pencil-shade primitives
//!
//! Author: Moroya Sakamoto

/// Straight hatch line mask
///
/// Returns 1.0 when the UV falls on a line, 0.0 otherwise. `angle_rad` is
/// the line direction (0 = horizontal). `density` sets lines per UV unit
/// and `thickness` is the half-width of each line in fractional UV.
#[inline]
#[must_use]
pub fn hatch_lines(uv_x: f32, uv_y: f32, angle_rad: f32, density: f32, thickness: f32) -> f32 {
    let (s, c) = angle_rad.sin_cos();
    let projected = uv_x * (-s) + uv_y * c;
    let d = density.max(1e-6);
    let raw = projected * d;
    let phase = raw - raw.floor();
    let dist = (phase - 0.5).abs();
    let t = thickness.clamp(0.0, 0.5);
    if dist > 0.5 - t {
        1.0
    } else {
        0.0
    }
}

/// Cross-hatch: union of two hatch layers at different angles
#[inline]
#[must_use]
pub fn cross_hatch(
    uv_x: f32,
    uv_y: f32,
    angle_a: f32,
    angle_b: f32,
    density: f32,
    thickness: f32,
) -> f32 {
    let a = hatch_lines(uv_x, uv_y, angle_a, density, thickness);
    let b = hatch_lines(uv_x, uv_y, angle_b, density, thickness);
    a.max(b)
}

/// Paper grain multiplier from a caller-supplied noise sample
///
/// `noise_sample` in `[0, 1]`. Returns a multiplier in
/// `[1 - intensity, 1 + intensity]`.
#[inline]
#[must_use]
pub fn paper_grain(noise_sample: f32, intensity: f32) -> f32 {
    let n = noise_sample.clamp(0.0, 1.0);
    let i = intensity.clamp(0.0, 1.0);
    1.0 + (n * 2.0 - 1.0) * i
}

/// Pencil shading density: high on dark side, low on bright side
///
/// Returns a hatch density mask in `[0, max_density]` driven by `1 - n.l`.
#[inline]
#[must_use]
pub fn pencil_shade(n_dot_l: f32, max_density: f32) -> f32 {
    let brightness = n_dot_l.clamp(0.0, 1.0);
    (1.0 - brightness) * max_density.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hatch_lines_output_binary() {
        for i in 0..=20 {
            for j in 0..=20 {
                let uvx = i as f32 / 20.0;
                let uvy = j as f32 / 20.0;
                let out = hatch_lines(uvx, uvy, 0.0, 8.0, 0.1);
                assert!(out == 0.0 || out == 1.0, "expected binary, got {out}");
            }
        }
    }

    #[test]
    fn hatch_lines_thicker_covers_more() {
        // At angle=0 the projected coordinate is uv_y, so sweep uv_y
        let mut thin_count = 0;
        let mut thick_count = 0;
        for i in 0..100 {
            let uv = i as f32 / 100.0;
            if hatch_lines(0.0, uv, 0.0, 10.0, 0.05) > 0.5 {
                thin_count += 1;
            }
            if hatch_lines(0.0, uv, 0.0, 10.0, 0.20) > 0.5 {
                thick_count += 1;
            }
        }
        assert!(thick_count > thin_count, "thicker should cover more");
    }

    #[test]
    fn cross_hatch_is_union_of_layers() {
        let angle_a = 0.0_f32;
        let angle_b = std::f32::consts::FRAC_PI_2;
        for i in 0..20 {
            for j in 0..20 {
                let uvx = i as f32 / 20.0;
                let uvy = j as f32 / 20.0;
                let a = hatch_lines(uvx, uvy, angle_a, 8.0, 0.1);
                let b = hatch_lines(uvx, uvy, angle_b, 8.0, 0.1);
                let combined = cross_hatch(uvx, uvy, angle_a, angle_b, 8.0, 0.1);
                assert!((combined - a.max(b)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn paper_grain_zero_intensity_neutral() {
        assert!((paper_grain(0.5, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn paper_grain_output_bounded() {
        for n in 0..=10 {
            let s = n as f32 / 10.0;
            let out = paper_grain(s, 0.3);
            assert!((0.7..=1.3).contains(&out), "out of range: {out}");
        }
    }

    #[test]
    fn pencil_shade_bright_returns_zero() {
        assert!(pencil_shade(1.0, 1.0).abs() < 1e-6);
    }

    #[test]
    fn pencil_shade_dark_returns_max() {
        assert!((pencil_shade(0.0, 0.8) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn pencil_shade_monotonic() {
        let a = pencil_shade(0.2, 1.0);
        let b = pencil_shade(0.6, 1.0);
        assert!(a > b, "darker should have higher density");
    }
}
