//! Domain distortion primitives for hand-drawn feel
//!
//! Each primitive returns a distorted position; compose by applying to the
//! shading point before feeding into an SDF or UV lookup.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Jitter a position by a caller-supplied 3D noise direction
///
/// `noise_direction` should be in `[-1, 1]` per component. Returns
/// `position + noise_direction * amplitude`.
#[inline]
#[must_use]
pub fn hand_drawn_jitter(position: Vec3, noise_direction: Vec3, amplitude: f32) -> Vec3 {
    position + noise_direction * amplitude
}

/// Sinusoidal wobble: cross-axis coupled sine offsets
///
/// Offsets each axis by a sine of another axis, scaled by `amplitude`.
/// `frequency` controls the wavelength (higher = tighter).
#[inline]
#[must_use]
pub fn sketch_wobble(position: Vec3, amplitude: f32, frequency: f32) -> Vec3 {
    let f = frequency.max(0.0);
    let offset = Vec3::new(
        (position.y * f).sin(),
        (position.z * f).sin(),
        (position.x * f).sin(),
    );
    position + offset * amplitude
}

/// Time-varying wobble for line-boil animation
///
/// Adds per-axis sinusoidal offsets driven by both position and time.
/// The per-axis phase constants (`3.0`, `3.7`, `4.1`) desynchronize the
/// axes so the motion does not appear locked.
#[inline]
#[must_use]
pub fn line_boil(position: Vec3, time: f32, amplitude: f32, frequency: f32) -> Vec3 {
    let f = frequency.max(0.0);
    let offset = Vec3::new(
        time.mul_add(3.0, position.y * f).sin(),
        time.mul_add(3.7, position.z * f).sin(),
        time.mul_add(4.1, position.x * f).sin(),
    );
    position + offset * amplitude
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_drawn_jitter_zero_amplitude_no_change() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let out = hand_drawn_jitter(p, Vec3::new(0.5, -0.3, 0.7), 0.0);
        assert!((out - p).length() < 1e-6);
    }

    #[test]
    fn hand_drawn_jitter_zero_noise_no_change() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let out = hand_drawn_jitter(p, Vec3::ZERO, 0.5);
        assert!((out - p).length() < 1e-6);
    }

    #[test]
    fn hand_drawn_jitter_scales_linearly() {
        let p = Vec3::ZERO;
        let n = Vec3::new(1.0, 0.0, 0.0);
        let out = hand_drawn_jitter(p, n, 0.5);
        assert!((out - Vec3::new(0.5, 0.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn sketch_wobble_zero_amplitude_no_change() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let out = sketch_wobble(p, 0.0, 5.0);
        assert!((out - p).length() < 1e-6);
    }

    #[test]
    fn sketch_wobble_zero_frequency_returns_unchanged() {
        // frequency 0 -> sin(0) = 0, so no offset
        let p = Vec3::new(1.0, 2.0, 3.0);
        let out = sketch_wobble(p, 0.5, 0.0);
        assert!((out - p).length() < 1e-6);
    }

    #[test]
    fn sketch_wobble_deterministic() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let a = sketch_wobble(p, 0.3, 2.0);
        let b = sketch_wobble(p, 0.3, 2.0);
        assert!((a - b).length() < 1e-6);
    }

    #[test]
    fn line_boil_time_changes_output() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let a = line_boil(p, 0.0, 0.5, 2.0);
        let b = line_boil(p, 1.0, 0.5, 2.0);
        assert!((a - b).length() > 1e-3, "time should change output");
    }

    #[test]
    fn line_boil_zero_amplitude_no_change() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let out = line_boil(p, 5.0, 0.0, 2.0);
        assert!((out - p).length() < 1e-6);
    }
}
