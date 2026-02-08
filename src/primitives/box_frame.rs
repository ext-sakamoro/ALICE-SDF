//! BoxFrame SDF (Deep Fried Edition)
//!
//! Wireframe box (only the edges). Based on IQ's sdBoxFrame.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// SDF for a box frame (wireframe box)
///
/// - `half_extents`: half-size along each axis
/// - `edge`: edge thickness
#[inline(always)]
pub fn sdf_box_frame(p: Vec3, half_extents: Vec3, edge: f32) -> f32 {
    let p = p.abs() - half_extents;
    let q = (p + edge).abs() - edge;

    let d1 = {
        let v = Vec3::new(p.x, q.y, q.z).max(Vec3::ZERO);
        v.length() + p.x.max(q.y.max(q.z)).min(0.0)
    };
    let d2 = {
        let v = Vec3::new(q.x, p.y, q.z).max(Vec3::ZERO);
        v.length() + q.x.max(p.y.max(q.z)).min(0.0)
    };
    let d3 = {
        let v = Vec3::new(q.x, q.y, p.z).max(Vec3::ZERO);
        v.length() + q.x.max(q.y.max(p.z)).min(0.0)
    };
    d1.min(d2).min(d3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_frame_center_hollow() {
        // Center of a box frame is in the hollow interior (positive distance)
        let d = sdf_box_frame(Vec3::ZERO, Vec3::splat(1.0), 0.1);
        assert!(d > 0.0, "Center should be outside frame (hollow), got {}", d);
    }

    #[test]
    fn test_box_frame_inside_edge() {
        // Point slightly inside the edge volume should be inside or on surface
        let d = sdf_box_frame(Vec3::new(0.0, 0.95, 0.95), Vec3::splat(1.0), 0.1);
        assert!(d <= 0.0, "Near-edge point should be inside or on surface, got {}", d);
    }

    #[test]
    fn test_box_frame_on_edge() {
        // Point on an edge should be near zero
        let d = sdf_box_frame(Vec3::new(1.0, 1.0, 0.0), Vec3::splat(1.0), 0.1);
        assert!(d.abs() < 0.15, "Edge point should be near surface, got {}", d);
    }

    #[test]
    fn test_box_frame_far_outside() {
        let d = sdf_box_frame(Vec3::new(5.0, 0.0, 0.0), Vec3::splat(1.0), 0.1);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_box_frame_symmetry() {
        let b = Vec3::new(1.0, 0.8, 0.6);
        let d1 = sdf_box_frame(Vec3::new(0.5, 0.3, 0.2), b, 0.1);
        let d2 = sdf_box_frame(Vec3::new(-0.5, 0.3, 0.2), b, 0.1);
        assert!((d1 - d2).abs() < 1e-6);
    }
}
