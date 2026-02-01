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
    // Use direct construction which is clearer and equally fast
    let ex = Vec3::new(epsilon, 0.0, 0.0);
    let ey = Vec3::new(0.0, epsilon, 0.0);
    let ez = Vec3::new(0.0, 0.0, epsilon);

    Vec3::new(
        eval(node, point + ex) - eval(node, point - ex),
        eval(node, point + ey) - eval(node, point - ey),
        eval(node, point + ez) - eval(node, point - ez),
    )
    .normalize()
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
