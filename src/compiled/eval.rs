//! Stack-based SDF evaluation for compiled bytecode
//!
//! Thin entry points over the unified stack machine in
//! `eval_core` (instantiated with `R = f32`). Batch / normal helpers live here.
//!
//! Author: Moroya Sakamoto

use super::compiler::CompiledSdf;
use super::eval_core::eval_bytecode;
use glam::Vec3;

/// Evaluate a compiled SDF at a point
///
/// This is the main entry point for compiled SDF evaluation.
/// It uses a stack-based approach instead of recursion.
#[inline]
pub fn eval_compiled(sdf: &CompiledSdf, point: Vec3) -> f32 {
    eval_bytecode::<f32>(&sdf.instructions, &sdf.aux_data, point.into())
}

/// Evaluate compiled SDF and compute normal using finite differences
#[inline]
pub fn eval_compiled_normal(sdf: &CompiledSdf, point: Vec3, epsilon: f32) -> Vec3 {
    let e = epsilon;

    // Tetrahedral method: 4 evaluations instead of 6
    let v0 = eval_compiled(sdf, point + Vec3::new(e, -e, -e)); // (+,-,-)
    let v1 = eval_compiled(sdf, point + Vec3::new(-e, -e, e)); // (-,-,+)
    let v2 = eval_compiled(sdf, point + Vec3::new(-e, e, -e)); // (-,+,-)
    let v3 = eval_compiled(sdf, point + Vec3::new(e, e, e)); // (+,+,+)

    Vec3::new(v0 - v1 - v2 + v3, -v0 - v1 + v2 + v3, -v0 + v1 - v2 + v3).normalize()
}

/// Combined distance + normal from 4 evaluations (tetrahedral method).
///
/// The distance at center is approximated as the average of the 4 tetrahedral
/// offset distances. This avoids the 5th eval needed when calling
/// `eval_compiled` + `eval_compiled_normal` separately.
///
/// Accuracy: distance error ≈ O(epsilon²), negligible for collision detection.
pub fn eval_compiled_distance_and_normal(
    sdf: &CompiledSdf,
    point: Vec3,
    epsilon: f32,
) -> (f32, Vec3) {
    let e = epsilon;

    let v0 = eval_compiled(sdf, point + Vec3::new(e, -e, -e));
    let v1 = eval_compiled(sdf, point + Vec3::new(-e, -e, e));
    let v2 = eval_compiled(sdf, point + Vec3::new(-e, e, -e));
    let v3 = eval_compiled(sdf, point + Vec3::new(e, e, e));

    // Distance ≈ average of the 4 offset samples
    let dist = (v0 + v1 + v2 + v3) * 0.25;

    let normal = Vec3::new(v0 - v1 - v2 + v3, -v0 - v1 + v2 + v3, -v0 + v1 - v2 + v3).normalize();

    (dist, normal)
}

/// Batch evaluate compiled SDF at multiple points
pub fn eval_compiled_batch(sdf: &CompiledSdf, points: &[Vec3]) -> Vec<f32> {
    points.iter().map(|p| eval_compiled(sdf, *p)).collect()
}

/// Parallel batch evaluate compiled SDF at multiple points
pub fn eval_compiled_batch_parallel(sdf: &CompiledSdf, points: &[Vec3]) -> Vec<f32> {
    use rayon::prelude::*;
    points.par_iter().map(|p| eval_compiled(sdf, *p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::eval;
    use crate::types::SdfNode;

    #[test]
    fn test_eval_sphere() {
        let node = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&node);

        // At origin: distance = -1 (inside)
        let d = eval_compiled(&compiled, Vec3::ZERO);
        assert!((d + 1.0).abs() < 0.0001);

        // On surface: distance = 0
        let d = eval_compiled(&compiled, Vec3::new(1.0, 0.0, 0.0));
        assert!(d.abs() < 0.0001);

        // Outside: distance = 1
        let d = eval_compiled(&compiled, Vec3::new(2.0, 0.0, 0.0));
        assert!((d - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_eval_box() {
        let node = SdfNode::box3d(1.0, 1.0, 1.0);
        let compiled = CompiledSdf::compile(&node);

        // At origin: inside
        let d = eval_compiled(&compiled, Vec3::ZERO);
        assert!(d < 0.0);

        // On surface - compare with interpreted
        let p = Vec3::new(1.0, 0.0, 0.0);
        let d_interpreted = eval(&node, p);
        let d_compiled = eval_compiled(&compiled, p);
        assert!(
            (d_interpreted - d_compiled).abs() < 0.0001,
            "Box at (1,0,0): interpreted={}, compiled={}",
            d_interpreted,
            d_compiled
        );
    }

    #[test]
    fn test_eval_union() {
        let node = SdfNode::sphere(1.0).union(SdfNode::sphere(1.0).translate(3.0, 0.0, 0.0));
        let compiled = CompiledSdf::compile(&node);

        // At origin: inside first sphere
        let d = eval_compiled(&compiled, Vec3::ZERO);
        assert!((d + 1.0).abs() < 0.0001);

        // At (3, 0, 0): inside second sphere
        let d = eval_compiled(&compiled, Vec3::new(3.0, 0.0, 0.0));
        assert!((d + 1.0).abs() < 0.0001);

        // Between spheres: outside
        let d = eval_compiled(&compiled, Vec3::new(1.5, 0.0, 0.0));
        assert!(d > 0.0);
    }

    #[test]
    fn test_eval_translate() {
        let node = SdfNode::sphere(1.0).translate(2.0, 0.0, 0.0);
        let compiled = CompiledSdf::compile(&node);

        // At (2, 0, 0): center of translated sphere
        let d = eval_compiled(&compiled, Vec3::new(2.0, 0.0, 0.0));
        assert!((d + 1.0).abs() < 0.0001);

        // At origin: outside
        let d = eval_compiled(&compiled, Vec3::ZERO);
        assert!((d - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_eval_scale() {
        let node = SdfNode::sphere(1.0).scale(2.0);
        let compiled = CompiledSdf::compile(&node);

        // At (2, 0, 0): on surface of scaled sphere
        let d = eval_compiled(&compiled, Vec3::new(2.0, 0.0, 0.0));
        assert!(d.abs() < 0.01);
    }

    #[test]
    fn test_compare_with_interpreted() {
        // Compare compiled vs interpreted results
        let node = SdfNode::sphere(1.0)
            .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5).translate(1.0, 0.0, 0.0), 0.2);
        let compiled = CompiledSdf::compile(&node);

        let test_points = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-1.0, -1.0, -1.0),
        ];

        for p in test_points {
            let d_interpreted = eval(&node, p);
            let d_compiled = eval_compiled(&compiled, p);
            assert!(
                (d_interpreted - d_compiled).abs() < 0.001,
                "Mismatch at {:?}: interpreted={}, compiled={}",
                p,
                d_interpreted,
                d_compiled
            );
        }
    }

    #[test]
    fn test_eval_normal() {
        let node = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&node);

        let n = eval_compiled_normal(&compiled, Vec3::new(1.0, 0.0, 0.0), 0.001);
        assert!((n.x - 1.0).abs() < 0.01);
        assert!(n.y.abs() < 0.01);
        assert!(n.z.abs() < 0.01);
    }

    #[test]
    fn test_eval_noise() {
        let base_node = SdfNode::sphere(1.0);
        let noise_node = SdfNode::sphere(1.0).noise(0.1, 2.0, 42);

        let compiled_base = CompiledSdf::compile(&base_node);
        let compiled_noise = CompiledSdf::compile(&noise_node);

        // Test at several points
        let test_points = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
        ];

        for p in test_points {
            let d_base = eval_compiled(&compiled_base, p);
            let d_noise = eval_compiled(&compiled_noise, p);

            // Noise should modify the distance by at most amplitude (0.1)
            let diff = (d_noise - d_base).abs();
            assert!(
                diff <= 0.1 + 0.001, // tolerance for floating point
                "Noise effect too large at {:?}: base={}, noise={}, diff={}",
                p,
                d_base,
                d_noise,
                diff
            );
        }

        // Verify noise is deterministic (same seed gives same result)
        let p = Vec3::new(0.7, 0.3, 0.5);
        let d1 = eval_compiled(&compiled_noise, p);
        let d2 = eval_compiled(&compiled_noise, p);
        assert_eq!(d1, d2, "Noise should be deterministic");
    }
}
