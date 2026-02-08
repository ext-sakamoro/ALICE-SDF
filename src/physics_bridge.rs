//! Physics Bridge: ALICE-SDF → ALICE-Physics Integration
//!
//! Implements `alice_physics::SdfField` for `CompiledSdf`, enabling
//! SDF shapes to be used as colliders in the physics engine.
//!
//! # Usage
//!
//! ```ignore
//! use alice_sdf::prelude::*;
//! use alice_sdf::physics_bridge::CompiledSdfField;
//! use alice_physics::prelude::*;
//!
//! // Create SDF shape
//! let node = SdfNode::sphere(1.0)
//!     .smooth_union(SdfNode::box3d(0.5, 2.0, 0.5), 0.2);
//! let compiled = CompiledSdf::compile(&node);
//!
//! // Wrap as physics collider
//! let field = CompiledSdfField::new(compiled);
//! let collider = SdfCollider::new_static(
//!     Box::new(field),
//!     Vec3Fix::ZERO,
//!     QuatFix::IDENTITY,
//! );
//!
//! // Add to physics world
//! world.add_sdf_collider(collider);
//! ```
//!
//! # Performance
//!
//! - Single-point evaluation: `eval_compiled` (~10ns per query)
//! - Normal computation: `eval_compiled_normal` (tetrahedral, 4 evals)
//! - Both use the compiled bytecode VM — no tree traversal overhead
//!
//! Author: Moroya Sakamoto

use crate::compiled::{
    eval_compiled, eval_compiled_distance_and_normal, eval_compiled_normal, CompiledSdf,
};
use alice_physics::SdfField;
use glam::Vec3;
use std::sync::Arc;

/// Wrapper around `CompiledSdf` that implements `alice_physics::SdfField`.
///
/// Thread-safe via `Arc<CompiledSdf>` — can be shared across physics threads.
pub struct CompiledSdfField {
    sdf: Arc<CompiledSdf>,
    /// Epsilon for gradient/normal computation (default: 0.001)
    pub epsilon: f32,
}

impl CompiledSdfField {
    /// Create a new physics-compatible SDF field from a compiled SDF.
    pub fn new(sdf: CompiledSdf) -> Self {
        Self {
            sdf: Arc::new(sdf),
            epsilon: 0.001,
        }
    }

    /// Create from an existing `Arc<CompiledSdf>` (zero-cost sharing).
    pub fn from_arc(sdf: Arc<CompiledSdf>) -> Self {
        Self {
            sdf,
            epsilon: 0.001,
        }
    }

    /// Set the epsilon used for gradient computation.
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Get a reference to the underlying compiled SDF.
    pub fn compiled(&self) -> &CompiledSdf {
        &self.sdf
    }

    /// Get the Arc reference (for sharing across threads).
    pub fn arc(&self) -> Arc<CompiledSdf> {
        Arc::clone(&self.sdf)
    }
}

// Safety: CompiledSdf is read-only after compilation
unsafe impl Send for CompiledSdfField {}
unsafe impl Sync for CompiledSdfField {}

impl SdfField for CompiledSdfField {
    #[inline]
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        eval_compiled(&self.sdf, Vec3::new(x, y, z))
    }

    #[inline]
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let n = eval_compiled_normal(&self.sdf, Vec3::new(x, y, z), self.epsilon);
        (n.x, n.y, n.z)
    }

    /// Combined distance + normal from 4 evaluations (tetrahedral method).
    ///
    /// 20% faster than separate `distance` + `normal` calls (4 evals instead of 5).
    #[inline]
    fn distance_and_normal(&self, x: f32, y: f32, z: f32) -> (f32, (f32, f32, f32)) {
        let p = Vec3::new(x, y, z);
        let (dist, n) = eval_compiled_distance_and_normal(&self.sdf, p, self.epsilon);
        (dist, (n.x, n.y, n.z))
    }
}

/// Convenience function: compile an SDF node and wrap as a physics field.
pub fn sdf_to_physics_field(node: &crate::types::SdfNode) -> CompiledSdfField {
    let compiled = CompiledSdf::compile(node);
    CompiledSdfField::new(compiled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;
    use alice_physics::sdf_collider::{collide_point_sdf, SdfCollider};
    use alice_physics::{Fix128, QuatFix, Vec3Fix};

    #[test]
    fn test_compiled_sdf_as_physics_field() {
        let node = SdfNode::Sphere { radius: 1.0 };
        let field = sdf_to_physics_field(&node);

        // Distance at origin should be -1.0 (inside unit sphere)
        let d = field.distance(0.0, 0.0, 0.0);
        assert!((d - (-1.0)).abs() < 0.01, "d={}", d);

        // Distance at (2, 0, 0) should be +1.0 (outside)
        let d2 = field.distance(2.0, 0.0, 0.0);
        assert!((d2 - 1.0).abs() < 0.01, "d2={}", d2);

        // Normal at (1, 0, 0) should be (1, 0, 0)
        let (nx, ny, nz) = field.normal(1.0, 0.0, 0.0);
        assert!(nx > 0.99, "nx={}", nx);
        assert!(ny.abs() < 0.01, "ny={}", ny);
        assert!(nz.abs() < 0.01, "nz={}", nz);
    }

    #[test]
    fn test_sphere_collider_via_bridge() {
        // Create a complex SDF: sphere with box subtracted
        let node = SdfNode::Sphere { radius: 2.0 };
        let field = sdf_to_physics_field(&node);

        let collider = SdfCollider::new_static(Box::new(field), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // Point inside sphere — should collide
        let point_inside = Vec3Fix::from_f32(0.5, 0.0, 0.0);
        let result = collide_point_sdf(point_inside, &collider);
        assert!(result.is_some(), "Point inside should collide");

        // Point far outside sphere — should not collide
        let point_outside = Vec3Fix::from_f32(5.0, 0.0, 0.0);
        let result2 = collide_point_sdf(point_outside, &collider);
        assert!(result2.is_none(), "Point outside should not collide");
    }

    #[test]
    fn test_csg_collider_via_bridge() {
        // CSG: sphere with box hole
        let node = SdfNode::sphere(2.0).subtract(SdfNode::box3d(0.5, 0.5, 0.5));
        let field = sdf_to_physics_field(&node);

        let collider = SdfCollider::new_static(Box::new(field), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // Origin is inside the subtracted box — should be outside the CSG shape
        let origin = Vec3Fix::ZERO;
        let result = collide_point_sdf(origin, &collider);
        assert!(
            result.is_none(),
            "Origin inside box hole should not collide"
        );

        // Point on sphere shell (1.5, 0, 0) — inside the SDF
        let shell_point = Vec3Fix::from_f32(1.5, 0.0, 0.0);
        let result2 = collide_point_sdf(shell_point, &collider);
        assert!(
            result2.is_some(),
            "Point inside sphere shell should collide"
        );
    }

    #[test]
    fn test_physics_world_with_sdf() {
        use alice_physics::{PhysicsConfig, PhysicsWorld, RigidBody};

        let node = SdfNode::Plane {
            normal: glam::Vec3::Y,
            distance: 0.0,
        };
        let field = sdf_to_physics_field(&node);

        let config = PhysicsConfig::default();
        let mut world = PhysicsWorld::new(config);
        world.set_sdf_collision_radius(Fix128::from_f32(0.5));

        // Add falling body
        let body = RigidBody::new_dynamic(Vec3Fix::from_f32(0.0, 5.0, 0.0), Fix128::ONE);
        world.add_body(body);

        // Add SDF ground
        world.add_sdf_collider(SdfCollider::new_static(
            Box::new(field),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        ));

        // Simulate
        let dt = Fix128::from_f32(1.0 / 60.0);
        for _ in 0..120 {
            world.step(dt);
        }

        // Body should be near ground
        let y = world.bodies[0].position.y.to_f32();
        assert!(y < 5.0, "Body should have fallen");
        assert!(y > -1.0, "Body should not have fallen through, y={}", y);
    }
}
