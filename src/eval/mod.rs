//! SDF Evaluation (Deep Fried Edition)
//!
//! Functions for evaluating SDF trees at points.
//!
//! # Deep Fried Optimizations
//! - **Forced Inlining**: `eval` is marked `#[inline]` to allow recursion unrolling by LLVM.
//! - **Direct Dispatch**: Primitives are called directly without wrapper overhead.
//! - **Swizzle Normals**: Uses Vec3 swizzles for efficient gradient computation.
//!
//! Author: Moroya Sakamoto

pub mod parallel;

pub use parallel::{eval_batch, eval_batch_parallel, eval_grid, eval_grid_with_normals};

use crate::modifiers::*;
use crate::operations::*;
use crate::primitives::*;
use crate::transforms::*;
use crate::types::SdfNode;
use glam::Vec3;

/// Evaluate an SDF tree at a single point (Deep Fried)
///
/// Recursively traverses the tree and computes the signed distance.
/// Marked `#[inline]` to encourage the compiler to inline small tree traversals.
///
/// # Arguments
/// * `node` - The SDF tree root
/// * `point` - Point to evaluate
///
/// # Returns
/// Signed distance to the surface
#[inline]
pub fn eval(node: &SdfNode, point: Vec3) -> f32 {
    match node {
        // === Primitives (Leaf Nodes) ===
        // These are the hot paths at the bottom of the recursion
        SdfNode::Sphere { radius } => sdf_sphere(point, *radius),
        SdfNode::Box3d { half_extents } => sdf_box3d(point, *half_extents),
        SdfNode::Cylinder {
            radius,
            half_height,
        } => sdf_cylinder(point, *radius, *half_height),
        SdfNode::Torus {
            major_radius,
            minor_radius,
        } => sdf_torus(point, *major_radius, *minor_radius),
        SdfNode::Plane { normal, distance } => sdf_plane(point, *normal, *distance),
        SdfNode::Capsule {
            point_a,
            point_b,
            radius,
        } => sdf_capsule(point, *point_a, *point_b, *radius),
        SdfNode::Cone {
            radius,
            half_height,
        } => sdf_cone(point, *radius, *half_height),
        SdfNode::Ellipsoid { radii } => sdf_ellipsoid(point, *radii),
        SdfNode::RoundedCone { r1, r2, half_height } => sdf_rounded_cone(point, *r1, *r2, *half_height),
        SdfNode::Pyramid { half_height } => sdf_pyramid(point, *half_height),
        SdfNode::Octahedron { size } => sdf_octahedron(point, *size),
        SdfNode::HexPrism { hex_radius, half_height } => sdf_hex_prism(point, *hex_radius, *half_height),
        SdfNode::Link { half_length, r1, r2 } => sdf_link(point, *half_length, *r1, *r2),

        // === Operations ===
        // Recurse first, then combine. Compiler can reorder these instruction streams.
        SdfNode::Union { a, b } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_union(d1, d2)
        }
        SdfNode::Intersection { a, b } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_intersection(d1, d2)
        }
        SdfNode::Subtraction { a, b } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_subtraction(d1, d2)
        }
        SdfNode::SmoothUnion { a, b, k } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_smooth_union(d1, d2, *k)
        }
        SdfNode::SmoothIntersection { a, b, k } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_smooth_intersection(d1, d2, *k)
        }
        SdfNode::SmoothSubtraction { a, b, k } => {
            let d1 = eval(a, point);
            let d2 = eval(b, point);
            sdf_smooth_subtraction(d1, d2, *k)
        }

        // === Transforms ===
        // Transform point, then recurse.
        SdfNode::Translate { child, offset } => {
            // Direct inline: point - offset (no function call)
            eval(child, point - *offset)
        }
        SdfNode::Rotate { child, rotation } => {
            // Conjugate multiplication for inverse rotation (Deep Fried)
            let p = rotation.conjugate() * point;
            eval(child, p)
        }
        SdfNode::Scale { child, factor } => {
            // Division then multiply result
            eval(child, point / *factor) * factor
        }
        SdfNode::ScaleNonUniform { child, factors } => {
            let (p, mult) = transform_scale_nonuniform(point, *factors);
            eval(child, p) * mult
        }

        // === Modifiers ===
        SdfNode::Twist { child, strength } => {
            let p = modifier_twist(point, *strength);
            eval(child, p)
        }
        SdfNode::Bend { child, curvature } => {
            let p = modifier_bend(point, *curvature);
            eval(child, p)
        }
        SdfNode::RepeatInfinite { child, spacing } => {
            let p = modifier_repeat_infinite(point, *spacing);
            eval(child, p)
        }
        SdfNode::RepeatFinite {
            child,
            count,
            spacing,
        } => {
            let p = modifier_repeat_finite(point, *count, *spacing);
            eval(child, p)
        }
        SdfNode::Noise {
            child,
            amplitude,
            frequency,
            seed,
        } => {
            let d = eval(child, point);
            modifier_noise_perlin(d, point, *amplitude, *frequency, *seed)
        }

        // Direct arithmetic operations (no function call overhead)
        SdfNode::Round { child, radius } => eval(child, point) - radius,
        SdfNode::Onion { child, thickness } => eval(child, point).abs() - thickness,
        SdfNode::Elongate { child, amount } => {
            let q = point - point.clamp(-*amount, *amount);
            eval(child, q)
        }
        SdfNode::Mirror { child, axes } => {
            let p = modifier_mirror(point, *axes);
            eval(child, p)
        }
        SdfNode::Revolution { child, offset } => {
            let p = modifier_revolution(point, *offset);
            eval(child, p)
        }
        SdfNode::Extrude { child, half_height } => {
            let p_flat = modifier_extrude_point(point);
            let d = eval(child, p_flat);
            modifier_extrude(d, point.z, *half_height)
        }

        // Material assignment is transparent for distance evaluation
        SdfNode::WithMaterial { child, .. } => eval(child, point),
    }
}

/// Evaluate which material ID applies at a given point
///
/// Walks the SDF tree and returns the material_id of the closest
/// surface node that has a material assigned. Returns 0 (default) if
/// no material is assigned.
#[inline]
pub fn eval_material(node: &SdfNode, point: Vec3) -> u32 {
    match node {
        SdfNode::WithMaterial { child, material_id } => {
            // This subtree has a material; return it
            // (nested WithMaterial: inner wins if closer)
            let inner = eval_material(child, point);
            if inner != 0 { inner } else { *material_id }
        }

        // Operations: return material of the closer child
        SdfNode::Union { a, b }
        | SdfNode::SmoothUnion { a, b, .. } => {
            let da = eval(a, point);
            let db = eval(b, point);
            if da <= db { eval_material(a, point) } else { eval_material(b, point) }
        }
        SdfNode::Intersection { a, b }
        | SdfNode::SmoothIntersection { a, b, .. } => {
            let da = eval(a, point);
            let db = eval(b, point);
            if da >= db { eval_material(a, point) } else { eval_material(b, point) }
        }
        SdfNode::Subtraction { a, b }
        | SdfNode::SmoothSubtraction { a, b, .. } => {
            let da = eval(a, point);
            let db = eval(b, point);
            if da >= -db { eval_material(a, point) } else { eval_material(b, point) }
        }

        // Transforms: transform point, recurse
        SdfNode::Translate { child, offset } => eval_material(child, point - *offset),
        SdfNode::Rotate { child, rotation } => eval_material(child, rotation.conjugate() * point),
        SdfNode::Scale { child, factor } => eval_material(child, point / *factor),
        SdfNode::ScaleNonUniform { child, factors } => {
            let p = point / *factors;
            eval_material(child, p)
        }

        // Modifiers: transform point, recurse
        SdfNode::Twist { child, strength } => eval_material(child, modifier_twist(point, *strength)),
        SdfNode::Bend { child, curvature } => eval_material(child, modifier_bend(point, *curvature)),
        SdfNode::RepeatInfinite { child, spacing } => eval_material(child, modifier_repeat_infinite(point, *spacing)),
        SdfNode::RepeatFinite { child, count, spacing } => eval_material(child, modifier_repeat_finite(point, *count, *spacing)),
        SdfNode::Noise { child, .. } => eval_material(child, point),
        SdfNode::Round { child, .. } | SdfNode::Onion { child, .. } => eval_material(child, point),
        SdfNode::Elongate { child, amount } => eval_material(child, point - point.clamp(-*amount, *amount)),
        SdfNode::Mirror { child, axes } => eval_material(child, modifier_mirror(point, *axes)),
        SdfNode::Revolution { child, offset } => eval_material(child, modifier_revolution(point, *offset)),
        SdfNode::Extrude { child, .. } => eval_material(child, modifier_extrude_point(point)),

        // Primitives: no material assigned
        _ => 0,
    }
}

/// Compute the surface normal at a point using finite differences (Deep Fried)
///
/// Uses swizzle operations for efficient gradient vector construction.
///
/// # Arguments
/// * `node` - The SDF tree
/// * `point` - Point on or near the surface
/// * `epsilon` - Small offset for gradient estimation
///
/// # Returns
/// Normalized surface normal
#[inline(always)]
pub fn normal(node: &SdfNode, point: Vec3, epsilon: f32) -> Vec3 {
    let ex = Vec3::new(epsilon, 0.0, 0.0);
    let ey = Vec3::new(0.0, epsilon, 0.0);
    let ez = Vec3::new(0.0, 0.0, epsilon);

    let grad = Vec3::new(
        eval(node, point + ex) - eval(node, point - ex),
        eval(node, point + ey) - eval(node, point - ey),
        eval(node, point + ez) - eval(node, point - ez),
    );

    // NaN guard: if gradient is zero/degenerate, return safe default
    let len_sq = grad.length_squared();
    if len_sq < 1e-20 {
        return Vec3::Y; // Safe fallback: up vector
    }
    grad / len_sq.sqrt()
}

/// Compute the gradient of the SDF at a point (Deep Fried)
///
/// Similar to normal but not normalized.
#[inline(always)]
pub fn gradient(node: &SdfNode, point: Vec3, epsilon: f32) -> Vec3 {
    let ex = Vec3::new(epsilon, 0.0, 0.0);
    let ey = Vec3::new(0.0, epsilon, 0.0);
    let ez = Vec3::new(0.0, 0.0, epsilon);

    let inv_2e = 1.0 / (2.0 * epsilon);

    Vec3::new(
        eval(node, point + ex) - eval(node, point - ex),
        eval(node, point + ey) - eval(node, point - ey),
        eval(node, point + ez) - eval(node, point - ez),
    ) * inv_2e
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_eval_sphere() {
        let sphere = SdfNode::sphere(1.0);
        assert!((eval(&sphere, Vec3::ZERO) + 1.0).abs() < 0.0001);
        assert!((eval(&sphere, Vec3::new(1.0, 0.0, 0.0))).abs() < 0.0001);
        assert!((eval(&sphere, Vec3::new(2.0, 0.0, 0.0)) - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_eval_union() {
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(1.0).translate(3.0, 0.0, 0.0);
        let union = a.union(b);

        // At origin, distance to union is distance to left sphere
        assert!((eval(&union, Vec3::ZERO) + 1.0).abs() < 0.0001);

        // Between spheres (at midpoint x=1.5)
        assert!(eval(&union, Vec3::new(1.5, 0.0, 0.0)) > 0.0);
    }

    #[test]
    fn test_eval_translated() {
        let sphere = SdfNode::sphere(1.0).translate(2.0, 0.0, 0.0);
        assert!((eval(&sphere, Vec3::new(2.0, 0.0, 0.0)) + 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_eval_rotated() {
        let box3d = SdfNode::box3d(2.0, 1.0, 1.0).rotate_euler(0.0, PI / 4.0, 0.0);
        // After 45° rotation, the box should extend diagonally
        let d = eval(&box3d, Vec3::ZERO);
        assert!(d < 0.0);
    }

    #[test]
    fn test_eval_scaled() {
        let sphere = SdfNode::sphere(1.0).scale(2.0);
        // Scaled by 2, so radius is now 2
        assert!((eval(&sphere, Vec3::new(2.0, 0.0, 0.0))).abs() < 0.0001);
    }

    #[test]
    fn test_normal() {
        let sphere = SdfNode::sphere(1.0);
        let n = normal(&sphere, Vec3::new(1.0, 0.0, 0.0), 0.001);
        let expected = Vec3::new(1.0, 0.0, 0.0);
        assert!((n - expected).length() < 0.01);
    }

    #[test]
    fn test_gradient() {
        let sphere = SdfNode::sphere(1.0);
        let g = gradient(&sphere, Vec3::new(1.0, 0.0, 0.0), 0.001);
        // Gradient magnitude should be ~1 for proper SDF
        assert!((g.length() - 1.0).abs() < 0.1);
    }
}
