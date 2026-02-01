//! Cylinder primitive SDF (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Branchless Capped Cylinder**: Rewrote Inigo Quilez's implementation
//!   to use branchless selection logic (max/min) instead of if/else chains.
//! - **Forced Inlining**: Zero call overhead.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Signed distance to a vertical cylinder (Deep Fried)
///
/// # Arguments
/// * `point` - Point to evaluate
/// * `radius` - Cylinder radius
/// * `half_height` - Half of the cylinder height
///
/// # Returns
/// Signed distance (negative inside, positive outside)
#[inline(always)]
pub fn sdf_cylinder(point: Vec3, radius: f32, half_height: f32) -> f32 {
    let d = Vec2::new(
        Vec2::new(point.x, point.z).length() - radius,
        point.y.abs() - half_height,
    );
    // Branchless: combine interior (negative) and exterior (positive)
    d.x.max(d.y).min(0.0) + d.max(Vec2::ZERO).length()
}

/// Signed distance to a capped cylinder between two points (Deep Fried Branchless)
///
/// # Deep Fried Optimization
///
/// Original code had 4 branches (if/else if/else if/else).
/// Replaced with branchless selection using max(0.0) pattern:
/// ```text
/// // Before: 4 branches
/// if x.max(y) < 0.0 { -x2.min(y2) }
/// else if x > 0.0 && y > 0.0 { x2 + y2 }
/// else if x > 0.0 { x2 }
/// else { y2 }
///
/// // After: Branchless
/// inner = -x2.min(y2)
/// outer = x.max(0.0)² + y.max(0.0)² * baba
/// d = select(x.max(y) < 0.0, inner, outer)
/// ```
#[inline(always)]
pub fn sdf_cylinder_capped(point: Vec3, a: Vec3, b: Vec3, radius: f32) -> f32 {
    let ba = b - a;
    let pa = point - a;
    let baba = ba.dot(ba);
    let paba = pa.dot(ba);

    let x = (pa * baba - ba * paba).length() - radius * baba;
    let y = (paba - baba * 0.5).abs() - baba * 0.5;

    let x2 = x * x;
    let y2 = y * y * baba;

    // Deep Fried Branchless Logic
    // Interior case: both x and y are negative (inside cylinder)
    let dist_sq_inner = -x2.min(y2);

    // Exterior case: at least one of x or y is positive
    // x.max(0.0)² contributes if outside radially
    // y.max(0.0)² * baba contributes if outside axially
    let dist_sq_outer = x.max(0.0).powi(2) + y.max(0.0).powi(2) * baba;

    // Select based on whether we're inside or outside
    let inside = x.max(y) < 0.0;
    let d = if inside { dist_sq_inner } else { dist_sq_outer };

    d.signum() * d.abs().sqrt() / baba
}

/// Signed distance to an infinite cylinder along Y-axis
#[inline(always)]
pub fn sdf_cylinder_infinite(point: Vec3, radius: f32) -> f32 {
    // Only XZ distance matters - direct scalar math
    (point.x * point.x + point.z * point.z).sqrt() - radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cylinder_origin() {
        let d = sdf_cylinder(Vec3::ZERO, 1.0, 1.0);
        assert!(d < 0.0); // Inside
    }

    #[test]
    fn test_cylinder_surface_side() {
        // On the side surface
        let d = sdf_cylinder(Vec3::new(1.0, 0.0, 0.0), 1.0, 1.0);
        assert!(d.abs() < 0.0001);
    }

    #[test]
    fn test_cylinder_surface_top() {
        // On the top cap
        let d = sdf_cylinder(Vec3::new(0.0, 1.0, 0.0), 1.0, 1.0);
        assert!(d.abs() < 0.0001);
    }

    #[test]
    fn test_cylinder_outside_side() {
        let d = sdf_cylinder(Vec3::new(2.0, 0.0, 0.0), 1.0, 1.0);
        assert!((d - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cylinder_outside_top() {
        let d = sdf_cylinder(Vec3::new(0.0, 2.0, 0.0), 1.0, 1.0);
        assert!((d - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cylinder_infinite() {
        // Infinite cylinder - only radial distance matters
        let d = sdf_cylinder_infinite(Vec3::new(2.0, 100.0, 0.0), 1.0);
        assert!((d - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cylinder_capped_vertical() {
        // Vertical capped cylinder from (0,0,0) to (0,2,0)
        let a = Vec3::ZERO;
        let b = Vec3::new(0.0, 2.0, 0.0);
        let radius = 0.5;

        // At center
        let d = sdf_cylinder_capped(Vec3::new(0.0, 1.0, 0.0), a, b, radius);
        assert!(d < 0.0); // Inside

        // On side surface
        let d = sdf_cylinder_capped(Vec3::new(0.5, 1.0, 0.0), a, b, radius);
        assert!(d.abs() < 0.01);
    }
}
