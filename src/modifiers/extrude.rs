//! Extrude modifier (Deep Fried Edition)
//!
//! Extrudes a shape along the Z-axis, creating a 3D shape from
//! a 2D cross-section evaluated in the XY plane.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Compute extrusion distance component
///
/// The child SDF is evaluated at (p.x, p.y, 0). The extrusion
/// caps the shape at z = ±half_height.
///
/// # Arguments
/// * `child_distance` - Distance from the child SDF (evaluated in XY plane)
/// * `p_z` - Z coordinate of the evaluation point
/// * `half_height` - Half the extrusion depth
///
/// # Returns
/// Combined distance accounting for both the 2D shape and Z capping
#[inline(always)]
pub fn modifier_extrude(child_distance: f32, p_z: f32, half_height: f32) -> f32 {
    let w_x = child_distance;
    let w_y = p_z.abs() - half_height;
    w_x.max(w_y).min(0.0) + glam::Vec2::new(w_x.max(0.0), w_y.max(0.0)).length()
}

/// Transform point for extrusion evaluation
///
/// Maps the point to the XY plane by zeroing out Z.
/// The child SDF should be evaluated at this transformed point.
#[inline(always)]
pub fn modifier_extrude_point(p: Vec3) -> Vec3 {
    Vec3::new(p.x, p.y, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extrude_inside() {
        // If child says -1.0 (inside) and z is within bounds
        let d = modifier_extrude(-1.0, 0.0, 1.0);
        assert!(d < 0.0, "Should be inside extruded shape, got {}", d);
    }

    #[test]
    fn test_extrude_outside_z() {
        // If child says -1.0 (inside) but z is beyond bounds
        let d = modifier_extrude(-1.0, 2.0, 0.5);
        assert!(d > 0.0, "Should be outside due to Z, got {}", d);
    }

    #[test]
    fn test_extrude_outside_xy() {
        // If child says 1.0 (outside) and z is within bounds
        let d = modifier_extrude(1.0, 0.0, 1.0);
        assert!(d > 0.0, "Should be outside due to XY, got {}", d);
    }

    #[test]
    fn test_extrude_on_surface() {
        // If child says 0.0 (on surface) and z is at boundary
        let d = modifier_extrude(0.0, 1.0, 1.0);
        assert!(d.abs() < 0.001, "Should be on surface, got {}", d);
    }
}
