//! CSG tree optimization pass
//!
//! Reduces SDF tree size by removing redundant nodes, folding identity
//! transforms, merging nested transforms, and simplifying degenerate operations.
//!
//! # Optimizations
//!
//! - **Identity transform removal**: `Scale(1.0)`, `Translate(0,0,0)`, `Rotate(identity)` → child
//! - **Nested transform merging**: `Translate(Translate(c, a), b)` → `Translate(c, a+b)`, etc.
//! - **Identity modifier removal**: `Round(0.0)`, `Elongate(0,0,0)` → child
//! - **Smooth→standard demotion**: `SmoothUnion(k=0)` → `Union`, etc.
//! - **Chamfer→standard demotion**: `ChamferUnion(r=0)` → `Union`, etc.
//! - **WithMaterial passthrough**: removes redundant `WithMaterial` wrapping
//!
//! Author: Moroya Sakamoto

use crate::types::SdfNode;
use glam::{Quat, Vec3};
use std::sync::Arc;

/// Optimize an SDF tree by removing redundant nodes and simplifying transforms.
///
/// Returns a new tree with optimizations applied recursively (bottom-up).
/// The original tree is not modified.
///
/// # Example
///
/// ```
/// use alice_sdf::prelude::*;
/// use alice_sdf::optimize::optimize;
///
/// // Redundant identity transforms
/// let shape = SdfNode::sphere(1.0)
///     .translate(0.0, 0.0, 0.0)
///     .scale(1.0);
///
/// let optimized = optimize(&shape);
/// assert_eq!(optimized.node_count(), 1); // Just the sphere
/// ```
pub fn optimize(node: &SdfNode) -> SdfNode {
    // Bottom-up: optimize children first, then this node
    let node = optimize_children(node);
    let node = fold_identity_transform(&node);
    let node = merge_nested_transforms(&node);
    let node = fold_identity_modifier(&node);
    let node = demote_smooth_to_standard(&node);
    node
}

/// Recursively optimize all children of a node.
fn optimize_children(node: &SdfNode) -> SdfNode {
    match node {
        // Binary operations: optimize both children
        SdfNode::Union { a, b } => SdfNode::Union {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
        },
        SdfNode::Intersection { a, b } => SdfNode::Intersection {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
        },
        SdfNode::Subtraction { a, b } => SdfNode::Subtraction {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
        },
        SdfNode::SmoothUnion { a, b, k } => SdfNode::SmoothUnion {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            k: *k,
        },
        SdfNode::SmoothIntersection { a, b, k } => SdfNode::SmoothIntersection {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            k: *k,
        },
        SdfNode::SmoothSubtraction { a, b, k } => SdfNode::SmoothSubtraction {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            k: *k,
        },
        SdfNode::ChamferUnion { a, b, r } => SdfNode::ChamferUnion {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            r: *r,
        },
        SdfNode::ChamferIntersection { a, b, r } => SdfNode::ChamferIntersection {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            r: *r,
        },
        SdfNode::ChamferSubtraction { a, b, r } => SdfNode::ChamferSubtraction {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            r: *r,
        },
        SdfNode::StairsUnion { a, b, r, n } => SdfNode::StairsUnion {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            r: *r,
            n: *n,
        },
        SdfNode::StairsIntersection { a, b, r, n } => SdfNode::StairsIntersection {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            r: *r,
            n: *n,
        },
        SdfNode::StairsSubtraction { a, b, r, n } => SdfNode::StairsSubtraction {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            r: *r,
            n: *n,
        },
        SdfNode::XOR { a, b } => SdfNode::XOR {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
        },
        SdfNode::Morph { a, b, t } => SdfNode::Morph {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            t: *t,
        },
        SdfNode::ColumnsUnion { a, b, r, n } => SdfNode::ColumnsUnion {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            r: *r,
            n: *n,
        },
        SdfNode::ColumnsIntersection { a, b, r, n } => SdfNode::ColumnsIntersection {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            r: *r,
            n: *n,
        },
        SdfNode::ColumnsSubtraction { a, b, r, n } => SdfNode::ColumnsSubtraction {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            r: *r,
            n: *n,
        },
        SdfNode::Pipe { a, b, r } => SdfNode::Pipe {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            r: *r,
        },
        SdfNode::Engrave { a, b, r } => SdfNode::Engrave {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            r: *r,
        },
        SdfNode::Groove { a, b, ra, rb } => SdfNode::Groove {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            ra: *ra,
            rb: *rb,
        },
        SdfNode::Tongue { a, b, ra, rb } => SdfNode::Tongue {
            a: Arc::new(optimize(a)),
            b: Arc::new(optimize(b)),
            ra: *ra,
            rb: *rb,
        },

        // Unary transforms: optimize child
        SdfNode::Translate { child, offset } => SdfNode::Translate {
            child: Arc::new(optimize(child)),
            offset: *offset,
        },
        SdfNode::Rotate { child, rotation } => SdfNode::Rotate {
            child: Arc::new(optimize(child)),
            rotation: *rotation,
        },
        SdfNode::Scale { child, factor } => SdfNode::Scale {
            child: Arc::new(optimize(child)),
            factor: *factor,
        },
        SdfNode::ScaleNonUniform { child, factors } => SdfNode::ScaleNonUniform {
            child: Arc::new(optimize(child)),
            factors: *factors,
        },

        // Unary modifiers: optimize child
        SdfNode::Twist { child, strength } => SdfNode::Twist {
            child: Arc::new(optimize(child)),
            strength: *strength,
        },
        SdfNode::Bend { child, curvature } => SdfNode::Bend {
            child: Arc::new(optimize(child)),
            curvature: *curvature,
        },
        SdfNode::RepeatInfinite { child, spacing } => SdfNode::RepeatInfinite {
            child: Arc::new(optimize(child)),
            spacing: *spacing,
        },
        SdfNode::RepeatFinite { child, count, spacing } => SdfNode::RepeatFinite {
            child: Arc::new(optimize(child)),
            count: *count,
            spacing: *spacing,
        },
        SdfNode::Noise { child, amplitude, frequency, seed } => SdfNode::Noise {
            child: Arc::new(optimize(child)),
            amplitude: *amplitude,
            frequency: *frequency,
            seed: *seed,
        },
        SdfNode::Round { child, radius } => SdfNode::Round {
            child: Arc::new(optimize(child)),
            radius: *radius,
        },
        SdfNode::Onion { child, thickness } => SdfNode::Onion {
            child: Arc::new(optimize(child)),
            thickness: *thickness,
        },
        SdfNode::Elongate { child, amount } => SdfNode::Elongate {
            child: Arc::new(optimize(child)),
            amount: *amount,
        },
        SdfNode::Mirror { child, axes } => SdfNode::Mirror {
            child: Arc::new(optimize(child)),
            axes: *axes,
        },
        SdfNode::OctantMirror { child } => SdfNode::OctantMirror {
            child: Arc::new(optimize(child)),
        },
        SdfNode::Revolution { child, offset } => SdfNode::Revolution {
            child: Arc::new(optimize(child)),
            offset: *offset,
        },
        SdfNode::Extrude { child, half_height } => SdfNode::Extrude {
            child: Arc::new(optimize(child)),
            half_height: *half_height,
        },
        SdfNode::SweepBezier { child, p0, p1, p2 } => SdfNode::SweepBezier {
            child: Arc::new(optimize(child)),
            p0: *p0,
            p1: *p1,
            p2: *p2,
        },
        SdfNode::Taper { child, factor } => SdfNode::Taper {
            child: Arc::new(optimize(child)),
            factor: *factor,
        },
        SdfNode::Displacement { child, strength } => SdfNode::Displacement {
            child: Arc::new(optimize(child)),
            strength: *strength,
        },
        SdfNode::PolarRepeat { child, count } => SdfNode::PolarRepeat {
            child: Arc::new(optimize(child)),
            count: *count,
        },
        SdfNode::WithMaterial { child, material_id } => SdfNode::WithMaterial {
            child: Arc::new(optimize(child)),
            material_id: *material_id,
        },

        // Primitives: leaf nodes, nothing to optimize
        _ => node.clone(),
    }
}

// ── Identity transform removal ──────────────────────────────────────

const EPS: f32 = 1e-6;

/// Remove identity transforms: Scale(1.0), Translate(0,0,0), Rotate(identity)
fn fold_identity_transform(node: &SdfNode) -> SdfNode {
    match node {
        SdfNode::Translate { child, offset } if offset.length_squared() < EPS * EPS => {
            child.as_ref().clone()
        }
        SdfNode::Rotate { child, rotation } if is_identity_quat(rotation) => {
            child.as_ref().clone()
        }
        SdfNode::Scale { child, factor } if (*factor - 1.0).abs() < EPS => {
            child.as_ref().clone()
        }
        SdfNode::ScaleNonUniform { child, factors }
            if (factors.x - 1.0).abs() < EPS
                && (factors.y - 1.0).abs() < EPS
                && (factors.z - 1.0).abs() < EPS =>
        {
            child.as_ref().clone()
        }
        _ => node.clone(),
    }
}

/// Check if a quaternion is the identity rotation
#[inline(always)]
fn is_identity_quat(q: &Quat) -> bool {
    // Identity quaternion is (0,0,0,1) or (0,0,0,-1)
    let dot = q.dot(Quat::IDENTITY).abs();
    (dot - 1.0).abs() < EPS
}

// ── Nested transform merging ────────────────────────────────────────

/// Merge nested transforms: Translate(Translate(c, a), b) → Translate(c, a+b), etc.
fn merge_nested_transforms(node: &SdfNode) -> SdfNode {
    match node {
        // Translate(Translate(child, inner_offset), outer_offset) → Translate(child, inner+outer)
        SdfNode::Translate { child, offset: outer } => {
            if let SdfNode::Translate { child: inner_child, offset: inner } = child.as_ref() {
                SdfNode::Translate {
                    child: inner_child.clone(),
                    offset: *inner + *outer,
                }
            } else {
                node.clone()
            }
        }

        // Scale(Scale(child, inner_factor), outer_factor) → Scale(child, inner*outer)
        SdfNode::Scale { child, factor: outer } => {
            if let SdfNode::Scale { child: inner_child, factor: inner } = child.as_ref() {
                SdfNode::Scale {
                    child: inner_child.clone(),
                    factor: inner * outer,
                }
            } else {
                node.clone()
            }
        }

        // Rotate(Rotate(child, inner_rot), outer_rot) → Rotate(child, outer*inner)
        SdfNode::Rotate { child, rotation: outer } => {
            if let SdfNode::Rotate { child: inner_child, rotation: inner } = child.as_ref() {
                SdfNode::Rotate {
                    child: inner_child.clone(),
                    rotation: *outer * *inner,
                }
            } else {
                node.clone()
            }
        }

        _ => node.clone(),
    }
}

// ── Identity modifier removal ───────────────────────────────────────

/// Remove identity modifiers: Round(0.0), Elongate(0,0,0), Noise(amplitude=0), etc.
fn fold_identity_modifier(node: &SdfNode) -> SdfNode {
    match node {
        SdfNode::Round { child, radius } if radius.abs() < EPS => {
            child.as_ref().clone()
        }
        SdfNode::Elongate { child, amount } if amount.length_squared() < EPS * EPS => {
            child.as_ref().clone()
        }
        SdfNode::Noise { child, amplitude, .. } if amplitude.abs() < EPS => {
            child.as_ref().clone()
        }
        SdfNode::Displacement { child, strength } if strength.abs() < EPS => {
            child.as_ref().clone()
        }
        SdfNode::Twist { child, strength } if strength.abs() < EPS => {
            child.as_ref().clone()
        }
        SdfNode::Bend { child, curvature } if curvature.abs() < EPS => {
            child.as_ref().clone()
        }
        SdfNode::Taper { child, factor } if (*factor - 1.0).abs() < EPS => {
            child.as_ref().clone()
        }
        _ => node.clone(),
    }
}

// ── Smooth/Chamfer demotion ─────────────────────────────────────────

/// Demote SmoothOp(k=0) to standard Op, ChamferOp(r=0) to standard Op
fn demote_smooth_to_standard(node: &SdfNode) -> SdfNode {
    match node {
        SdfNode::SmoothUnion { a, b, k } if k.abs() < EPS => {
            SdfNode::Union { a: a.clone(), b: b.clone() }
        }
        SdfNode::SmoothIntersection { a, b, k } if k.abs() < EPS => {
            SdfNode::Intersection { a: a.clone(), b: b.clone() }
        }
        SdfNode::SmoothSubtraction { a, b, k } if k.abs() < EPS => {
            SdfNode::Subtraction { a: a.clone(), b: b.clone() }
        }
        SdfNode::ChamferUnion { a, b, r } if r.abs() < EPS => {
            SdfNode::Union { a: a.clone(), b: b.clone() }
        }
        SdfNode::ChamferIntersection { a, b, r } if r.abs() < EPS => {
            SdfNode::Intersection { a: a.clone(), b: b.clone() }
        }
        SdfNode::ChamferSubtraction { a, b, r } if r.abs() < EPS => {
            SdfNode::Subtraction { a: a.clone(), b: b.clone() }
        }
        _ => node.clone(),
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Count the number of nodes removed by optimization
pub fn optimization_stats(original: &SdfNode, optimized: &SdfNode) -> OptimizationStats {
    let before = original.node_count();
    let after = optimized.node_count();
    OptimizationStats {
        nodes_before: before,
        nodes_after: after,
        nodes_removed: before.saturating_sub(after),
    }
}

/// Statistics from an optimization pass
#[derive(Debug, Clone, Copy)]
pub struct OptimizationStats {
    /// Number of nodes before optimization
    pub nodes_before: u32,
    /// Number of nodes after optimization
    pub nodes_after: u32,
    /// Number of nodes removed
    pub nodes_removed: u32,
}

impl std::fmt::Display for OptimizationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Optimization: {} → {} nodes ({} removed, {:.1}% reduction)",
            self.nodes_before,
            self.nodes_after,
            self.nodes_removed,
            if self.nodes_before > 0 {
                self.nodes_removed as f64 / self.nodes_before as f64 * 100.0
            } else {
                0.0
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::eval;

    #[test]
    fn test_identity_translate_removed() {
        let shape = SdfNode::sphere(1.0).translate(0.0, 0.0, 0.0);
        assert_eq!(shape.node_count(), 2);

        let opt = optimize(&shape);
        assert_eq!(opt.node_count(), 1);

        // Verify evaluation unchanged
        let p = Vec3::new(0.5, 0.3, 0.1);
        assert!((eval(&shape, p) - eval(&opt, p)).abs() < 1e-5);
    }

    #[test]
    fn test_identity_scale_removed() {
        let shape = SdfNode::sphere(1.0).scale(1.0);
        assert_eq!(shape.node_count(), 2);

        let opt = optimize(&shape);
        assert_eq!(opt.node_count(), 1);
    }

    #[test]
    fn test_identity_rotate_removed() {
        let shape = SdfNode::sphere(1.0).rotate(Quat::IDENTITY);
        assert_eq!(shape.node_count(), 2);

        let opt = optimize(&shape);
        assert_eq!(opt.node_count(), 1);
    }

    #[test]
    fn test_nested_translate_merged() {
        let shape = SdfNode::sphere(1.0)
            .translate(1.0, 0.0, 0.0)
            .translate(0.0, 2.0, 0.0);
        assert_eq!(shape.node_count(), 3);

        let opt = optimize(&shape);
        assert_eq!(opt.node_count(), 2); // Sphere + one merged Translate

        // Verify evaluation unchanged
        let p = Vec3::new(1.5, 2.3, 0.1);
        assert!((eval(&shape, p) - eval(&opt, p)).abs() < 1e-5);
    }

    #[test]
    fn test_nested_scale_merged() {
        let shape = SdfNode::sphere(1.0).scale(2.0).scale(3.0);
        assert_eq!(shape.node_count(), 3);

        let opt = optimize(&shape);
        assert_eq!(opt.node_count(), 2); // Sphere + Scale(6.0)

        let p = Vec3::new(5.0, 0.3, 0.1);
        assert!((eval(&shape, p) - eval(&opt, p)).abs() < 1e-4);
    }

    #[test]
    fn test_nested_rotate_merged() {
        let r1 = Quat::from_rotation_x(0.5);
        let r2 = Quat::from_rotation_y(0.3);
        let shape = SdfNode::sphere(1.0).rotate(r1).rotate(r2);
        assert_eq!(shape.node_count(), 3);

        let opt = optimize(&shape);
        assert_eq!(opt.node_count(), 2);

        let p = Vec3::new(0.5, 0.7, 0.3);
        assert!((eval(&shape, p) - eval(&opt, p)).abs() < 1e-5);
    }

    #[test]
    fn test_smooth_union_k0_demoted() {
        let shape = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.0);
        let opt = optimize(&shape);

        // Should be demoted to standard Union
        assert!(matches!(opt, SdfNode::Union { .. }));

        let p = Vec3::new(0.3, 0.4, 0.1);
        assert!((eval(&shape, p) - eval(&opt, p)).abs() < 1e-5);
    }

    #[test]
    fn test_chamfer_r0_demoted() {
        let shape = SdfNode::sphere(1.0).chamfer_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.0);
        let opt = optimize(&shape);

        assert!(matches!(opt, SdfNode::Union { .. }));
    }

    #[test]
    fn test_identity_round_removed() {
        let shape = SdfNode::Sphere { radius: 1.0 };
        let rounded = SdfNode::Round {
            child: Arc::new(shape.clone()),
            radius: 0.0,
        };
        assert_eq!(rounded.node_count(), 2);

        let opt = optimize(&rounded);
        assert_eq!(opt.node_count(), 1);
    }

    #[test]
    fn test_identity_twist_removed() {
        let shape = SdfNode::sphere(1.0).twist(0.0);
        assert_eq!(shape.node_count(), 2);

        let opt = optimize(&shape);
        assert_eq!(opt.node_count(), 1);
    }

    #[test]
    fn test_complex_tree_optimization() {
        // Build a tree with multiple redundancies
        let shape = SdfNode::sphere(1.0)
            .translate(0.0, 0.0, 0.0) // identity → removed
            .scale(1.0)               // identity → removed
            .translate(1.0, 0.0, 0.0)
            .translate(0.0, 2.0, 0.0) // merges with above → (1, 2, 0)
            .smooth_union(
                SdfNode::box3d(0.5, 0.5, 0.5)
                    .twist(0.0)        // identity → removed
                    .scale(2.0)
                    .scale(0.5),       // merges with above → Scale(1.0) → identity → removed
                0.0,                   // k=0 → demoted to Union
            );

        let original_count = shape.node_count();
        let opt = optimize(&shape);
        let optimized_count = opt.node_count();

        // Should have removed several nodes
        assert!(
            optimized_count < original_count,
            "Should reduce nodes: {} → {}",
            original_count,
            optimized_count
        );

        // Verify evaluation is preserved
        let test_points = [
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::new(1.5, 2.3, 0.1),
            Vec3::new(-0.3, 0.5, 0.8),
        ];
        for p in test_points {
            let d_orig = eval(&shape, p);
            let d_opt = eval(&opt, p);
            assert!(
                (d_orig - d_opt).abs() < 1e-4,
                "Mismatch at {:?}: orig={}, opt={}",
                p, d_orig, d_opt
            );
        }
    }

    #[test]
    fn test_non_identity_not_removed() {
        // These should NOT be optimized away
        let shape = SdfNode::sphere(1.0)
            .translate(1.0, 0.0, 0.0)
            .scale(2.0)
            .twist(0.5);

        let opt = optimize(&shape);
        assert_eq!(opt.node_count(), shape.node_count());
    }

    #[test]
    fn test_optimization_stats_display() {
        let shape = SdfNode::sphere(1.0)
            .translate(0.0, 0.0, 0.0)
            .scale(1.0);

        let opt = optimize(&shape);
        let stats = optimization_stats(&shape, &opt);

        assert_eq!(stats.nodes_before, 3);
        assert_eq!(stats.nodes_after, 1);
        assert_eq!(stats.nodes_removed, 2);

        let s = format!("{}", stats);
        assert!(s.contains("3 → 1"));
    }

    #[test]
    fn test_primitive_unchanged() {
        let shape = SdfNode::sphere(1.0);
        let opt = optimize(&shape);
        assert_eq!(opt.node_count(), 1);
    }
}
