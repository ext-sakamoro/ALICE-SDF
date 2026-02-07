//! Primitive SDF shapes (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Enum Dispatch**: Replaced string matching with fast Enum matching (integer comparison).
//! - **Unchecked Variants**: Added `_unchecked` variants for hot paths where
//!   parameter length is guaranteed by the caller (e.g. from compiled bytecode).
//!
//! Author: Moroya Sakamoto

mod sphere;
mod box3d;
mod cylinder;
mod torus;
mod plane;
mod capsule;
mod cone;
mod ellipsoid;
mod rounded_cone;
mod pyramid;
mod octahedron;
mod hex_prism;
mod link;
mod triangle;
mod bezier;

pub use sphere::{sdf_sphere, sdf_sphere_at};
pub use box3d::{sdf_box3d, sdf_box3d_at, sdf_rounded_box3d};
pub use cylinder::{sdf_cylinder, sdf_cylinder_capped, sdf_cylinder_infinite};
pub use torus::{sdf_torus, sdf_torus_capped};
pub use plane::{sdf_plane, sdf_plane_xy, sdf_plane_xz, sdf_plane_yz, sdf_plane_from_points};
pub use capsule::{sdf_capsule, sdf_capsule_vertical, sdf_capsule_horizontal};
pub use cone::sdf_cone;
pub use ellipsoid::sdf_ellipsoid;
pub use rounded_cone::sdf_rounded_cone;
pub use pyramid::sdf_pyramid;
pub use octahedron::sdf_octahedron;
pub use hex_prism::sdf_hex_prism;
pub use link::sdf_link;
pub use triangle::sdf_triangle;
pub use bezier::sdf_bezier;

use glam::Vec3;

/// Primitive type identifier for fast dispatch
///
/// #[repr(u8)] ensures it compiles to a single byte, friendly for serialization/bytecode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PrimitiveType {
    /// Sphere
    Sphere,
    /// Axis-aligned box
    Box3d,
    /// Cylinder along Y-axis
    Cylinder,
    /// Torus in XZ plane
    Torus,
    /// Infinite plane
    Plane,
    /// Capsule between two points
    Capsule,
    /// Cone along Y-axis
    Cone,
    /// Ellipsoid
    Ellipsoid,
    /// Rounded cone along Y-axis
    RoundedCone,
    /// Square-base pyramid
    Pyramid,
    /// Regular octahedron
    Octahedron,
    /// Hexagonal prism
    HexPrism,
    /// Chain link
    Link,
    /// Triangle (3 vertices)
    Triangle,
    /// Quadratic Bezier curve with radius
    Bezier,
}

/// Evaluate a primitive SDF using fast Enum dispatch (Safe version)
///
/// # Arguments
/// * `prim` - Primitive type enum (fast integer comparison)
/// * `point` - Point to evaluate
/// * `params` - Parameters array
#[inline(always)]
pub fn eval_primitive(prim: PrimitiveType, point: Vec3, params: &[f32]) -> Option<f32> {
    match prim {
        PrimitiveType::Sphere => {
            if params.len() >= 1 {
                Some(sdf_sphere(point, params[0]))
            } else {
                None
            }
        }
        PrimitiveType::Box3d => {
            if params.len() >= 3 {
                Some(sdf_box3d(point, Vec3::new(params[0], params[1], params[2])))
            } else {
                None
            }
        }
        PrimitiveType::Cylinder => {
            if params.len() >= 2 {
                Some(sdf_cylinder(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::Torus => {
            if params.len() >= 2 {
                Some(sdf_torus(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::Plane => {
            if params.len() >= 4 {
                Some(sdf_plane(point, Vec3::new(params[0], params[1], params[2]), params[3]))
            } else {
                None
            }
        }
        PrimitiveType::Capsule => {
            if params.len() >= 7 {
                Some(sdf_capsule(
                    point,
                    Vec3::new(params[0], params[1], params[2]),
                    Vec3::new(params[3], params[4], params[5]),
                    params[6],
                ))
            } else {
                None
            }
        }
        PrimitiveType::Cone => {
            if params.len() >= 2 {
                Some(sdf_cone(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::Ellipsoid => {
            if params.len() >= 3 {
                Some(sdf_ellipsoid(point, Vec3::new(params[0], params[1], params[2])))
            } else {
                None
            }
        }
        PrimitiveType::RoundedCone => {
            if params.len() >= 3 {
                Some(sdf_rounded_cone(point, params[0], params[1], params[2]))
            } else {
                None
            }
        }
        PrimitiveType::Pyramid => {
            if params.len() >= 1 {
                Some(sdf_pyramid(point, params[0]))
            } else {
                None
            }
        }
        PrimitiveType::Octahedron => {
            if params.len() >= 1 {
                Some(sdf_octahedron(point, params[0]))
            } else {
                None
            }
        }
        PrimitiveType::HexPrism => {
            if params.len() >= 2 {
                Some(sdf_hex_prism(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::Link => {
            if params.len() >= 3 {
                Some(sdf_link(point, params[0], params[1], params[2]))
            } else {
                None
            }
        }
        PrimitiveType::Triangle => {
            if params.len() >= 9 {
                Some(sdf_triangle(
                    point,
                    Vec3::new(params[0], params[1], params[2]),
                    Vec3::new(params[3], params[4], params[5]),
                    Vec3::new(params[6], params[7], params[8]),
                ))
            } else {
                None
            }
        }
        PrimitiveType::Bezier => {
            if params.len() >= 10 {
                Some(sdf_bezier(
                    point,
                    Vec3::new(params[0], params[1], params[2]),
                    Vec3::new(params[3], params[4], params[5]),
                    Vec3::new(params[6], params[7], params[8]),
                    params[9],
                ))
            } else {
                None
            }
        }
    }
}

/// Evaluate a primitive SDF using fast Enum dispatch (Unsafe/Deep Fried version)
///
/// # Deep Fried Optimization
///
/// - **No Bounds Checks**: Uses `get_unchecked` for parameter access.
/// - **No Option Wrapping**: Returns `f32` directly.
///
/// # Safety
/// Caller must ensure `params` has enough elements for the given primitive type.
/// This is intended for use in the inner loop where validation happened upstream.
#[inline(always)]
pub unsafe fn eval_primitive_unchecked(prim: PrimitiveType, point: Vec3, params: &[f32]) -> f32 {
    match prim {
        PrimitiveType::Sphere => sdf_sphere(point, *params.get_unchecked(0)),
        PrimitiveType::Box3d => sdf_box3d(
            point,
            Vec3::new(*params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2))
        ),
        PrimitiveType::Cylinder => sdf_cylinder(
            point,
            *params.get_unchecked(0),
            *params.get_unchecked(1)
        ),
        PrimitiveType::Torus => sdf_torus(
            point,
            *params.get_unchecked(0),
            *params.get_unchecked(1)
        ),
        PrimitiveType::Plane => sdf_plane(
            point,
            Vec3::new(*params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2)),
            *params.get_unchecked(3)
        ),
        PrimitiveType::Capsule => sdf_capsule(
            point,
            Vec3::new(*params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2)),
            Vec3::new(*params.get_unchecked(3), *params.get_unchecked(4), *params.get_unchecked(5)),
            *params.get_unchecked(6)
        ),
        PrimitiveType::Cone => sdf_cone(
            point,
            *params.get_unchecked(0),
            *params.get_unchecked(1)
        ),
        PrimitiveType::Ellipsoid => sdf_ellipsoid(
            point,
            Vec3::new(*params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2))
        ),
        PrimitiveType::RoundedCone => sdf_rounded_cone(
            point,
            *params.get_unchecked(0),
            *params.get_unchecked(1),
            *params.get_unchecked(2)
        ),
        PrimitiveType::Pyramid => sdf_pyramid(point, *params.get_unchecked(0)),
        PrimitiveType::Octahedron => sdf_octahedron(point, *params.get_unchecked(0)),
        PrimitiveType::HexPrism => sdf_hex_prism(
            point,
            *params.get_unchecked(0),
            *params.get_unchecked(1)
        ),
        PrimitiveType::Link => sdf_link(
            point,
            *params.get_unchecked(0),
            *params.get_unchecked(1),
            *params.get_unchecked(2)
        ),
        PrimitiveType::Triangle => sdf_triangle(
            point,
            Vec3::new(*params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2)),
            Vec3::new(*params.get_unchecked(3), *params.get_unchecked(4), *params.get_unchecked(5)),
            Vec3::new(*params.get_unchecked(6), *params.get_unchecked(7), *params.get_unchecked(8)),
        ),
        PrimitiveType::Bezier => sdf_bezier(
            point,
            Vec3::new(*params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2)),
            Vec3::new(*params.get_unchecked(3), *params.get_unchecked(4), *params.get_unchecked(5)),
            Vec3::new(*params.get_unchecked(6), *params.get_unchecked(7), *params.get_unchecked(8)),
            *params.get_unchecked(9),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_primitive_enum() {
        // Sphere at origin with radius 1
        let d = eval_primitive(PrimitiveType::Sphere, Vec3::ZERO, &[1.0]).unwrap();
        assert!((d + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_eval_primitive_unchecked() {
        let params = [1.0];
        // Safety: params length is 1, Sphere needs 1.
        let d = unsafe { eval_primitive_unchecked(PrimitiveType::Sphere, Vec3::ZERO, &params) };
        assert!((d + 1.0).abs() < 0.001);
    }
}
