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

pub use sphere::{sdf_sphere, sdf_sphere_at};
pub use box3d::{sdf_box3d, sdf_box3d_at, sdf_rounded_box3d};
pub use cylinder::{sdf_cylinder, sdf_cylinder_capped, sdf_cylinder_infinite};
pub use torus::{sdf_torus, sdf_torus_capped};
pub use plane::{sdf_plane, sdf_plane_xy, sdf_plane_xz, sdf_plane_yz, sdf_plane_from_points};
pub use capsule::{sdf_capsule, sdf_capsule_vertical, sdf_capsule_horizontal};

use glam::Vec3;

/// Primitive type identifier for fast dispatch
///
/// #[repr(u8)] ensures it compiles to a single byte, friendly for serialization/bytecode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PrimitiveType {
    Sphere,
    Box3d,
    Cylinder,
    Torus,
    Plane,
    Capsule,
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
