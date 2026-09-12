//! Motion and VFX primitives
//!
//! Author: Moroya Sakamoto

use std::f32::consts::TAU;

/// Radial speed line mask emanating from a focal UV
///
/// Returns 1.0 when the UV falls on one of `count` evenly-spaced radial
/// lines, 0.0 otherwise. `thickness` is the fractional half-width in the
/// angular phase space `[0, 1)`.
#[inline]
#[must_use]
pub fn speed_line(
    uv_x: f32,
    uv_y: f32,
    focus_x: f32,
    focus_y: f32,
    count: u32,
    thickness: f32,
) -> f32 {
    let dx = uv_x - focus_x;
    let dy = uv_y - focus_y;
    if dx.abs() < 1e-6 && dy.abs() < 1e-6 {
        return 0.0;
    }
    if count == 0 {
        return 0.0;
    }
    let angle = dy.atan2(dx);
    let n = count as f32;
    let phase_raw = (angle + std::f32::consts::PI) * n / TAU;
    let phase = phase_raw - phase_raw.floor();
    let dist = (phase - 0.5).abs();
    let t = thickness.clamp(0.0, 0.5);
    if dist > 0.5 - t {
        1.0
    } else {
        0.0
    }
}

/// Impact flash intensity with exponential decay
///
/// Peaks at `intensity` when `time == 0`, decays with time constant `decay`.
/// Returns 0.0 for negative time.
#[inline]
#[must_use]
pub fn impact_flash(time: f32, decay: f32, intensity: f32) -> f32 {
    if time < 0.0 {
        return 0.0;
    }
    let d = decay.max(1e-4);
    intensity * (-time / d).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_line_at_focus_returns_zero() {
        assert!(speed_line(0.5, 0.5, 0.5, 0.5, 12, 0.05).abs() < 1e-6);
    }

    #[test]
    fn speed_line_zero_count_returns_zero() {
        assert!(speed_line(0.7, 0.5, 0.5, 0.5, 0, 0.05).abs() < 1e-6);
    }

    #[test]
    fn speed_line_output_binary() {
        for i in 0..=20 {
            for j in 0..=20 {
                let uvx = i as f32 / 20.0;
                let uvy = j as f32 / 20.0;
                let out = speed_line(uvx, uvy, 0.5, 0.5, 24, 0.05);
                assert!(out == 0.0 || out == 1.0, "expected binary, got {out}");
            }
        }
    }

    #[test]
    fn speed_line_thicker_covers_more() {
        let mut thin = 0;
        let mut thick = 0;
        for i in 0..40 {
            for j in 0..40 {
                let uvx = i as f32 / 40.0;
                let uvy = j as f32 / 40.0;
                if speed_line(uvx, uvy, 0.5, 0.5, 24, 0.02) > 0.5 {
                    thin += 1;
                }
                if speed_line(uvx, uvy, 0.5, 0.5, 24, 0.10) > 0.5 {
                    thick += 1;
                }
            }
        }
        assert!(thick > thin, "thicker should cover more");
    }

    #[test]
    fn impact_flash_at_zero_is_peak() {
        assert!((impact_flash(0.0, 0.5, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn impact_flash_negative_time_is_zero() {
        assert!(impact_flash(-1.0, 0.5, 1.0).abs() < 1e-6);
    }

    #[test]
    fn impact_flash_decays_monotonically() {
        let a = impact_flash(0.1, 0.5, 1.0);
        let b = impact_flash(0.5, 0.5, 1.0);
        let c = impact_flash(1.0, 0.5, 1.0);
        assert!(a > b && b > c);
    }

    #[test]
    fn impact_flash_intensity_scales_output() {
        let a = impact_flash(0.2, 0.5, 1.0);
        let b = impact_flash(0.2, 0.5, 0.5);
        assert!((a - b * 2.0).abs() < 1e-4);
    }
}
