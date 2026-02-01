//! Capsule primitive SDF (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Vectorized Dot Products**: Uses glam's optimized dot products.
//! - **Forced Inlining**: Zero call overhead.
//! - **Optimized Axis-Aligned Variants**: Simplified math for common cases.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Signed distance to a capsule (line segment with radius)
///
/// # Arguments
/// * `point` - Point to evaluate
/// * `a` - Start point of capsule axis
/// * `b` - End point of capsule axis
/// * `radius` - Capsule radius
///
/// # Returns
/// Signed distance (negative inside, positive outside)
#[inline(always)]
pub fn sdf_capsule(point: Vec3, a: Vec3, b: Vec3, radius: f32) -> f32 {
    let pa = point - a;
    let ba = b - a;
    // Project point onto line segment, clamp t to [0, 1]
    let h = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
    (pa - ba * h).length() - radius
}

/// Signed distance to a vertical capsule centered at origin (Deep Fried)
///
/// # Deep Fried Optimization
/// Simplified vector math since axis is Y - avoids general dot products.
///
/// # Arguments
/// * `point` - Point to evaluate
/// * `half_height` - Half the height (excluding caps)
/// * `radius` - Capsule radius
#[inline(always)]
pub fn sdf_capsule_vertical(point: Vec3, half_height: f32, radius: f32) -> f32 {
    // Optimized: directly compute clamped Y offset
    let p_y = point.y - point.y.clamp(-half_height, half_height);
    // length of (point.x, p_y, point.z) - avoid Vec3 allocation
    (point.x * point.x + p_y * p_y + point.z * point.z).sqrt() - radius
}

/// Signed distance to a horizontal capsule along X-axis centered at origin (Deep Fried)
///
/// # Deep Fried Optimization
/// Simplified vector math since axis is X.
#[inline(always)]
pub fn sdf_capsule_horizontal(point: Vec3, half_length: f32, radius: f32) -> f32 {
    // Optimized: directly compute clamped X offset
    let p_x = point.x - point.x.clamp(-half_length, half_length);
    // length of (p_x, point.y, point.z)
    (p_x * p_x + point.y * point.y + point.z * point.z).sqrt() - radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capsule_at_endpoint() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 2.0, 0.0);
        let radius = 0.5;

        // At endpoint A center
        let d = sdf_capsule(a, a, b, radius);
        assert!((d + radius).abs() < 0.0001);

        // At endpoint B center
        let d = sdf_capsule(b, a, b, radius);
        assert!((d + radius).abs() < 0.0001);
    }

    #[test]
    fn test_capsule_at_middle() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 2.0, 0.0);
        let radius = 0.5;

        // Middle of capsule
        let d = sdf_capsule(Vec3::new(0.0, 1.0, 0.0), a, b, radius);
        assert!((d + radius).abs() < 0.0001);
    }

    #[test]
    fn test_capsule_surface() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 2.0, 0.0);
        let radius = 0.5;

        // On surface (side)
        let d = sdf_capsule(Vec3::new(0.5, 1.0, 0.0), a, b, radius);
        assert!(d.abs() < 0.0001);

        // On surface (top cap)
        let d = sdf_capsule(Vec3::new(0.0, 2.5, 0.0), a, b, radius);
        assert!(d.abs() < 0.0001);
    }

    #[test]
    fn test_capsule_outside() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 2.0, 0.0);
        let radius = 0.5;

        // Outside (side)
        let d = sdf_capsule(Vec3::new(1.5, 1.0, 0.0), a, b, radius);
        assert!((d - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_capsule_vertical() {
        let d = sdf_capsule_vertical(Vec3::ZERO, 1.0, 0.5);
        assert!((d + 0.5).abs() < 0.0001);

        let d = sdf_capsule_vertical(Vec3::new(0.5, 0.0, 0.0), 1.0, 0.5);
        assert!(d.abs() < 0.0001);
    }

    #[test]
    fn test_capsule_horizontal() {
        let d = sdf_capsule_horizontal(Vec3::ZERO, 1.0, 0.5);
        assert!((d + 0.5).abs() < 0.0001);

        let d = sdf_capsule_horizontal(Vec3::new(1.5, 0.0, 0.0), 1.0, 0.5);
        assert!(d.abs() < 0.0001);
    }
}
