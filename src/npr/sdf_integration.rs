//! Adapters that connect NPR primitives to `SdfNode` and the autodiff pipeline
//!
//! These helpers evaluate an [`SdfNode`] at a point and feed the resulting
//! distance / normal / curvature into the corresponding NPR primitive so
//! callers do not have to wire the calls themselves.
//!
//! Author: Moroya Sakamoto

use crate::autodiff::mean_curvature;
use crate::eval::{eval, eval_normal};
use crate::npr::outline::{curvature_outline, distance_field_outline_soft};
use crate::npr::toon::{soft_toon_ramp, toon_ramp};
use crate::types::SdfNode;
use glam::Vec3;

/// Curvature outline mask driven by the SDF's autodiff Hessian
///
/// Computes the mean curvature of `node` at `point` and feeds it into
/// [`curvature_outline`]. `epsilon` is the finite-difference step used by
/// [`mean_curvature`]; typical values are 1e-3 to 1e-4.
#[inline]
#[must_use]
pub fn curvature_outline_from_node(
    node: &SdfNode,
    point: Vec3,
    epsilon: f32,
    sensitivity: f32,
    threshold: f32,
) -> f32 {
    let k = mean_curvature(node, point, epsilon);
    curvature_outline(k, sensitivity, threshold)
}

/// Distance-field outline mask driven by the SDF's evaluated distance
///
/// Evaluates the SDF at `point` and feeds the signed distance into
/// [`distance_field_outline_soft`].
#[inline]
#[must_use]
pub fn distance_outline_from_node(
    node: &SdfNode,
    point: Vec3,
    width_inner: f32,
    width_outer: f32,
) -> f32 {
    let sdf = eval(node, point);
    distance_field_outline_soft(sdf, width_inner, width_outer)
}

/// N-band toon shading evaluated from an SDF's surface normal
///
/// Computes the surface normal at `point` and returns the toon ramp value
/// of `dot(normal, light)`.
#[inline]
#[must_use]
pub fn toon_shade_from_node(node: &SdfNode, point: Vec3, light: Vec3, bands: u32) -> f32 {
    let n = eval_normal(node, point);
    let l = light.normalize_or_zero();
    let n_dot_l = n.dot(l);
    toon_ramp(n_dot_l, bands)
}

/// Soft N-band toon shading evaluated from an SDF's surface normal
#[inline]
#[must_use]
pub fn soft_toon_shade_from_node(
    node: &SdfNode,
    point: Vec3,
    light: Vec3,
    bands: u32,
    smoothness: f32,
) -> f32 {
    let n = eval_normal(node, point);
    let l = light.normalize_or_zero();
    let n_dot_l = n.dot(l);
    soft_toon_ramp(n_dot_l, bands, smoothness)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_sphere() -> SdfNode {
        SdfNode::sphere(1.0)
    }

    #[test]
    fn distance_outline_on_sphere_surface_peaks() {
        let node = unit_sphere();
        // Point on the unit sphere surface (sdf = 0)
        let p = Vec3::new(1.0, 0.0, 0.0);
        let mask = distance_outline_from_node(&node, p, 0.01, 0.1);
        assert!(
            (mask - 1.0).abs() < 1e-3,
            "surface should be at outline peak, got {mask}"
        );
    }

    #[test]
    fn distance_outline_far_from_surface_is_zero() {
        let node = unit_sphere();
        let p = Vec3::new(3.0, 0.0, 0.0);
        let mask = distance_outline_from_node(&node, p, 0.01, 0.1);
        assert!(mask.abs() < 1e-4);
    }

    #[test]
    fn curvature_outline_on_sphere_positive() {
        let node = unit_sphere();
        // Sphere has non-zero mean curvature at surface
        let p = Vec3::new(1.0, 0.0, 0.0);
        let mask = curvature_outline_from_node(&node, p, 1e-3, 0.5, 0.1);
        assert!(mask >= 0.0, "mask should be non-negative, got {mask}");
    }

    #[test]
    fn toon_shade_lit_side_bright() {
        let node = unit_sphere();
        // Light coming from +X: point at (1, 0, 0) has normal ~+X, so n.l = 1
        let p = Vec3::new(1.0, 0.0, 0.0);
        let light = Vec3::new(1.0, 0.0, 0.0);
        let brightness = toon_shade_from_node(&node, p, light, 3);
        assert!(
            (brightness - 1.0).abs() < 1e-3,
            "expected max brightness, got {brightness}"
        );
    }

    #[test]
    fn toon_shade_shadow_side_dark() {
        let node = unit_sphere();
        // Light coming from +X: point at (-1, 0, 0) has normal ~-X, so n.l = -1 (clamped to 0)
        let p = Vec3::new(-1.0, 0.0, 0.0);
        let light = Vec3::new(1.0, 0.0, 0.0);
        let brightness = toon_shade_from_node(&node, p, light, 3);
        assert!(brightness.abs() < 1e-3, "expected dark, got {brightness}");
    }

    #[test]
    fn soft_toon_shade_bounded() {
        let node = unit_sphere();
        let light = Vec3::new(1.0, 0.5, 0.3).normalize();
        for i in 0..12 {
            let a = i as f32 * std::f32::consts::TAU / 12.0;
            let p = Vec3::new(a.cos(), a.sin(), 0.0);
            let b = soft_toon_shade_from_node(&node, p, light, 4, 0.05);
            assert!((0.0..=1.0).contains(&b), "out of range at angle {a}: {b}");
        }
    }
}
