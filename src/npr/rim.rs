//! Rim, fresnel, and stylized specular primitives
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Fresnel rim intensity from `normal . view`
///
/// Peaks at grazing angles (`n_dot_v -> 0`). `power` controls sharpness
/// and `intensity` scales the peak.
#[inline]
#[must_use]
pub fn fresnel_rim(n_dot_v: f32, power: f32, intensity: f32) -> f32 {
    let base = (1.0 - n_dot_v.clamp(0.0, 1.0)).max(0.0);
    base.powf(power.max(0.0)) * intensity
}

/// Procedural matcap: bilinear sample of a 2x2 palette by view-space normal
///
/// `normal_view` is the normal in view space (unit); its `x` and `y`
/// components in `[-1, 1]` index the palette corners. No texture atlas.
#[inline]
#[must_use]
pub fn procedural_matcap(
    normal_view: Vec3,
    bottom_left: Vec3,
    bottom_right: Vec3,
    top_left: Vec3,
    top_right: Vec3,
) -> Vec3 {
    let u = (normal_view.x * 0.5 + 0.5).clamp(0.0, 1.0);
    let v = (normal_view.y * 0.5 + 0.5).clamp(0.0, 1.0);
    let bottom = bottom_left.lerp(bottom_right, u);
    let top = top_left.lerp(top_right, u);
    bottom.lerp(top, v)
}

/// Stylized specular with optional soft edge
///
/// `n_dot_h` is dot of surface normal and half-vector. `sharpness` is the
/// Blinn-Phong-like exponent. When `soft_edge > 0`, the specular boundary
/// uses a smoothstep of that half-width; when `soft_edge == 0` the output
/// is a hard step at `raw > 0.5`.
#[inline]
#[must_use]
pub fn stylized_specular(n_dot_h: f32, sharpness: f32, soft_edge: f32) -> f32 {
    let ndh = n_dot_h.clamp(0.0, 1.0);
    let raw = ndh.powf(sharpness.max(1.0));
    let s = soft_edge.clamp(0.0, 0.5);
    if s < 1e-6 {
        if raw > 0.5 {
            1.0
        } else {
            0.0
        }
    } else {
        let edge0 = 0.5 - s;
        let edge1 = 0.5 + s;
        let t = ((raw - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresnel_grazing_is_max() {
        assert!((fresnel_rim(0.0, 2.0, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fresnel_head_on_is_zero() {
        assert!(fresnel_rim(1.0, 2.0, 1.0).abs() < 1e-6);
    }

    #[test]
    fn fresnel_intensity_scales_output() {
        let a = fresnel_rim(0.5, 2.0, 1.0);
        let b = fresnel_rim(0.5, 2.0, 0.5);
        assert!((a - b * 2.0).abs() < 1e-4);
    }

    #[test]
    fn matcap_center_averages_corners() {
        let bl = Vec3::new(1.0, 0.0, 0.0);
        let br = Vec3::new(0.0, 1.0, 0.0);
        let tl = Vec3::new(0.0, 0.0, 1.0);
        let tr = Vec3::new(1.0, 1.0, 1.0);
        let c = procedural_matcap(Vec3::ZERO, bl, br, tl, tr);
        let expected = (bl + br + tl + tr) * 0.25;
        assert!((c - expected).length() < 1e-4);
    }

    #[test]
    fn matcap_corner_returns_corner_color() {
        let bl = Vec3::new(1.0, 0.0, 0.0);
        let br = Vec3::new(0.0, 1.0, 0.0);
        let tl = Vec3::new(0.0, 0.0, 1.0);
        let tr = Vec3::new(1.0, 1.0, 1.0);
        // normal = (-1, -1, 0) -> bottom-left
        let c = procedural_matcap(Vec3::new(-1.0, -1.0, 0.0), bl, br, tl, tr);
        assert!((c - bl).length() < 1e-4);
    }

    #[test]
    fn stylized_specular_hard_step_above_threshold() {
        let out = stylized_specular(1.0, 2.0, 0.0);
        assert!((out - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stylized_specular_hard_step_below_threshold() {
        let out = stylized_specular(0.1, 8.0, 0.0);
        assert!(out.abs() < 1e-6);
    }

    #[test]
    fn stylized_specular_soft_edge_bounded() {
        for i in 0..=20 {
            let ndh = i as f32 / 20.0;
            let out = stylized_specular(ndh, 4.0, 0.1);
            assert!(
                (0.0..=1.0).contains(&out),
                "out of range at ndh={ndh}: {out}"
            );
        }
    }
}
