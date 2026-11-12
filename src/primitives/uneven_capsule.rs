//! Uneven Capsule prism SDF (Deep Fried Edition)
//!
//! 2D uneven capsule shape in XY plane, extruded along Z-axis.
//!
//! Method also known from Inigo Quilez's articles as 2D Uneven Capsule (iquilezles.org).
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::LawMinMax;
use glam::{Vec2, Vec3};

/// The 2D uneven capsule (IQ formula) in its own frame: the circle of radius `r1` at the origin, the
/// circle of radius `r2` at `(0, h)`, and the two lines tangent to both; `px` is `|x|`
///
/// Exact when `|r1 - r2| < h` (neither circle contains the other); the caller handles the other case
/// Shared by the prism ([`sdf_uneven_capsule`]) and the 2D node (`Sdf2dNode::UnevenCapsule`)
#[inline(always)]
pub fn sdf_uneven_capsule_2d_core(px: f32, py: f32, r1: f32, r2: f32, h: f32) -> f32 {
    let b = (r1 - r2) / h;
    let a = (1.0 - b * b).law_max(0.0).sqrt();
    let k = Vec2::new(-b, a).dot(Vec2::new(px, py));
    if k < 0.0 {
        Vec2::new(px, py).length() - r1
    } else if k > a * h {
        Vec2::new(px, py - h).length() - r2
    } else {
        Vec2::new(px, py).dot(Vec2::new(a, b)) - r1
    }
}

/// SDF for an uneven capsule prism
///
/// Two circles of different radii connected by tangent lines, in XY plane,
/// extruded along Z.
/// - `r1`: bottom circle radius
/// - `r2`: top circle radius
/// - `cap_height`: half-height between circle centers
/// - `half_depth`: half the extrusion depth along Z
#[inline(always)]
pub fn sdf_uneven_capsule(p: Vec3, r1: f32, r2: f32, cap_height: f32, half_depth: f32) -> f32 {
    // 2D Uneven Capsule in XY (IQ formula)
    let d_2d = sdf_uneven_capsule_2d_core(p.x.abs(), p.y, r1, r2, cap_height * 2.0);
    // Extrude along Z
    let d_z = p.z.abs() - half_depth;
    let w = Vec2::new(d_2d.law_max(0.0), d_z.law_max(0.0));
    d_2d.law_max(d_z).law_min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uneven_capsule_center_inside() {
        let d = sdf_uneven_capsule(Vec3::ZERO, 0.5, 0.3, 0.5, 0.5);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_uneven_capsule_far_outside() {
        let d = sdf_uneven_capsule(Vec3::new(5.0, 0.0, 0.0), 0.5, 0.3, 0.5, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_uneven_capsule_symmetry_x() {
        let d1 = sdf_uneven_capsule(Vec3::new(0.2, 0.1, 0.1), 0.5, 0.3, 0.5, 0.5);
        let d2 = sdf_uneven_capsule(Vec3::new(-0.2, 0.1, 0.1), 0.5, 0.3, 0.5, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }

    #[test]
    fn test_uneven_capsule_symmetry_z() {
        let d1 = sdf_uneven_capsule(Vec3::new(0.1, 0.1, 0.2), 0.5, 0.3, 0.5, 0.5);
        let d2 = sdf_uneven_capsule(Vec3::new(0.1, 0.1, -0.2), 0.5, 0.3, 0.5, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Z");
    }
    /// The prism before the 2D core was factored out, kept here to pin its bits
    fn prism_before(p: Vec3, r1: f32, r2: f32, cap_height: f32, half_depth: f32) -> f32 {
        let px = p.x.abs();
        let h = cap_height * 2.0;
        let b = (r1 - r2) / h;
        let a = (1.0 - b * b).law_max(0.0).sqrt();
        let k = Vec2::new(-b, a).dot(Vec2::new(px, p.y));
        let d_2d = if k < 0.0 {
            Vec2::new(px, p.y).length() - r1
        } else if k > a * h {
            Vec2::new(px, p.y - h).length() - r2
        } else {
            Vec2::new(px, p.y).dot(Vec2::new(a, b)) - r1
        };
        let d_z = p.z.abs() - half_depth;
        let w = Vec2::new(d_2d.law_max(0.0), d_z.law_max(0.0));
        d_2d.law_max(d_z).law_min(0.0) + w.length()
    }

    #[test]
    fn test_uneven_capsule_bits_unchanged_by_the_2d_core() {
        let mut n = 0;
        for &(r1, r2, ch, hd) in &[
            (0.5, 0.3, 0.5, 0.5),
            (0.2, 0.6, 0.8, 0.1),
            (0.4, 0.4, 0.3, 1.0),
        ] {
            for i in 0..17 {
                for j in 0..17 {
                    for k in 0..5 {
                        let p = Vec3::new(
                            i as f32 * 0.17 - 1.4,
                            j as f32 * 0.19 - 1.3,
                            k as f32 * 0.3 - 0.61,
                        );
                        let (got, was) = (
                            sdf_uneven_capsule(p, r1, r2, ch, hd),
                            prism_before(p, r1, r2, ch, hd),
                        );
                        assert_eq!(got.to_bits(), was.to_bits(), "at {p:?}");
                        n += 1;
                    }
                }
            }
        }
        assert_eq!(n, 3 * 17 * 17 * 5);
    }
}
