//! Torus primitive SDF (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Forced Inlining**: Zero call overhead.
//! - **Vectorized Operations**: Uses glam's optimized Vec2 operations.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::{Real, Vec3R};
use glam::{Vec2, Vec3};

/// Torus in the XZ plane (generic over [`Real`]).
#[inline(always)]
pub fn sdf_torus_r<R: Real>(p: Vec3R<R>, major_radius: f32, minor_radius: f32) -> R {
    let qx = (p.x * p.x + p.z * p.z).sqrt() - R::splat(major_radius);
    (qx * qx + p.y * p.y).sqrt() - R::splat(minor_radius)
}

/// Torus in the XZ plane.
#[inline(always)]
pub fn sdf_torus(point: Vec3, major_radius: f32, minor_radius: f32) -> f32 {
    sdf_torus_r::<f32>(point.into(), major_radius, minor_radius)
}

/// Signed distance to a torus with arbitrary orientation
///
/// The torus is rotated so its axis aligns with `axis`
#[allow(dead_code)]
#[inline(always)]
pub fn sdf_torus_oriented(point: Vec3, axis: Vec3, major_radius: f32, minor_radius: f32) -> f32 {
    let axis = axis.normalize();

    // Project point onto the plane perpendicular to axis
    let projected_dist = point.dot(axis);
    let in_plane = point - axis * projected_dist;
    let radial_dist = in_plane.length();

    let q = Vec2::new(radial_dist - major_radius, projected_dist);
    q.length() - minor_radius
}

/// Signed distance to a capped torus (partial torus)
///
/// # Arguments
/// * `point` - Point to evaluate
/// * `major_radius` - Distance from center to tube center
/// * `minor_radius` - Tube radius
/// * `angle` - Half-angle of the cap in radians (0 to PI)
#[inline(always)]
pub fn sdf_torus_capped(point: Vec3, major_radius: f32, minor_radius: f32, angle: f32) -> f32 {
    let sc = Vec2::new(alice_det_math::sin(angle), alice_det_math::cos(angle));
    let p_xz = Vec2::new(point.x.abs(), point.z);

    // Branchless selection would require select intrinsic
    // For now, this branch is predictable per-call
    let k = if sc.y * p_xz.x > sc.x * p_xz.y {
        p_xz.dot(sc)
    } else {
        p_xz.length()
    };

    let q = Vec2::new(
        ((2.0 * major_radius) * -k + (major_radius * major_radius + p_xz.length_squared())).sqrt(),
        point.y,
    );
    q.length() - minor_radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_torus_center() {
        // At the very center of the torus hole
        let d = sdf_torus(Vec3::ZERO, 2.0, 0.5);
        // Distance should be major_radius - minor_radius = 1.5
        assert!((d - 1.5).abs() < 0.0001);
    }

    #[test]
    fn test_torus_tube_center() {
        // Inside the tube at (major_radius, 0, 0)
        let d = sdf_torus(Vec3::new(2.0, 0.0, 0.0), 2.0, 0.5);
        // Should be exactly on the tube center, distance = -minor_radius
        assert!((d + 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_torus_surface_outer() {
        // On outer surface
        let d = sdf_torus(Vec3::new(2.5, 0.0, 0.0), 2.0, 0.5);
        assert!(d.abs() < 0.0001);
    }

    #[test]
    fn test_torus_surface_inner() {
        // On inner surface
        let d = sdf_torus(Vec3::new(1.5, 0.0, 0.0), 2.0, 0.5);
        assert!(d.abs() < 0.0001);
    }

    #[test]
    fn test_torus_surface_top() {
        // On top surface
        let d = sdf_torus(Vec3::new(2.0, 0.5, 0.0), 2.0, 0.5);
        assert!(d.abs() < 0.0001);
    }

    #[test]
    fn test_torus_oriented() {
        // Torus with Y axis (same as default)
        let d = sdf_torus_oriented(Vec3::new(2.0, 0.0, 0.0), Vec3::Y, 2.0, 0.5);
        assert!((d + 0.5).abs() < 0.0001);
    }
}
