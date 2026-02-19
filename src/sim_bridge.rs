//! Simulation Bridge: ALICE-SDF ↔ Simulation Modifiers
//!
//! Connects ALICE-SDF's `CompiledSdf` with ALICE-Physics simulation
//! modifiers (thermal, pressure, erosion, fracture, phase change).
//!
//! # Architecture
//!
//! ```text
//! CompiledSdf ──→ CompiledSdfField ──→ ModifiedSdf
//!                  (physics_bridge)      ├── ThermalModifier
//!                                        ├── PressureModifier
//!                                        ├── ErosionModifier
//!                                        ├── FractureModifier
//!                                        └── PhaseChangeModifier
//! ```
//!
//! The `SimulatedSdf` wrapper provides a convenient API for
//! combining an ALICE-SDF shape with physics simulation modifiers.
//!
//! # Example
//!
//! ```ignore
//! use alice_sdf::prelude::*;
//! use alice_sdf::sim_bridge::SimulatedSdf;
//! use alice_physics::thermal::{ThermalModifier, ThermalConfig};
//!
//! // Create SDF shape
//! let node = SdfNode::sphere(1.0);
//! let compiled = CompiledSdf::compile(&node);
//!
//! // Wrap with simulation (bounds auto-estimated or user-provided)
//! let mut sim = SimulatedSdf::new(compiled);
//! sim.add_thermal(ThermalConfig::default(), 16);
//!
//! // Query modified distance
//! let d = sim.distance(1.0, 0.0, 0.0);
//! ```
//!
//! Author: Moroya Sakamoto

use crate::compiled::CompiledSdf;
use crate::physics_bridge::CompiledSdfField;
use alice_physics::sdf_collider::SdfField;
use alice_physics::sim_modifier::{ModifiedSdf, PhysicsModifier};
use std::sync::Arc;

/// Default simulation bounds (half-extent)
const DEFAULT_HALF_EXTENT: f32 = 5.0;

// ============================================================================
// SimulatedSdf
// ============================================================================

/// SDF with physics simulation modifiers attached.
///
/// Wraps a `CompiledSdf` and chains physics modifiers that modify
/// the distance field based on simulation state (heat, pressure, etc.).
pub struct SimulatedSdf {
    /// The underlying modified SDF (CompiledSdf + modifiers)
    inner: ModifiedSdf,
    /// Reference to the compiled SDF (for re-use)
    compiled: Arc<CompiledSdf>,
    /// Simulation bounds (min corner)
    bounds_min: (f32, f32, f32),
    /// Simulation bounds (max corner)
    bounds_max: (f32, f32, f32),
}

impl SimulatedSdf {
    /// Create a new simulated SDF from a compiled SDF.
    ///
    /// Uses default bounds (-5..5). For accurate bounds, use `with_bounds`.
    pub fn new(compiled: CompiledSdf) -> Self {
        let arc = Arc::new(compiled);
        let field = CompiledSdfField::from_arc(Arc::clone(&arc));
        let modified = ModifiedSdf::new(Box::new(field));
        let e = DEFAULT_HALF_EXTENT;
        Self {
            inner: modified,
            compiled: arc,
            bounds_min: (-e, -e, -e),
            bounds_max: (e, e, e),
        }
    }

    /// Create from an existing `Arc<CompiledSdf>`.
    pub fn from_arc(arc: Arc<CompiledSdf>) -> Self {
        let field = CompiledSdfField::from_arc(Arc::clone(&arc));
        let modified = ModifiedSdf::new(Box::new(field));
        let e = DEFAULT_HALF_EXTENT;
        Self {
            inner: modified,
            compiled: arc,
            bounds_min: (-e, -e, -e),
            bounds_max: (e, e, e),
        }
    }

    /// Set custom simulation bounds.
    pub fn with_bounds(mut self, min: (f32, f32, f32), max: (f32, f32, f32)) -> Self {
        self.bounds_min = min;
        self.bounds_max = max;
        self
    }

    /// Get the underlying compiled SDF.
    pub fn compiled(&self) -> &CompiledSdf {
        &self.compiled
    }

    /// Get simulation bounds.
    pub fn bounds(&self) -> ((f32, f32, f32), (f32, f32, f32)) {
        (self.bounds_min, self.bounds_max)
    }

    /// Add a generic physics modifier.
    pub fn add_modifier(&mut self, modifier: Box<dyn PhysicsModifier>) {
        self.inner.add_modifier(modifier);
    }

    /// Number of active modifiers.
    pub fn modifier_count(&self) -> usize {
        self.inner.modifier_count()
    }

    /// Remove all modifiers.
    pub fn clear_modifiers(&mut self) {
        self.inner.clear_modifiers();
    }

    /// Update all modifier simulations by dt seconds.
    pub fn update(&mut self, dt: f32) {
        self.inner.update(dt);
    }

    /// Evaluate the simulated distance at a point.
    pub fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        self.inner.distance(x, y, z)
    }

    /// Evaluate the simulated normal at a point.
    pub fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        self.inner.normal(x, y, z)
    }

    /// Get mutable access to a modifier by index.
    pub fn modifier_mut(&mut self, index: usize) -> Option<&mut dyn PhysicsModifier> {
        self.inner.modifier_mut(index)
    }

    // ========================================================================
    // Convenience methods for common modifiers
    // ========================================================================

    /// Add a thermal modifier with given config and field resolution.
    ///
    /// Returns the modifier index for later access.
    pub fn add_thermal(
        &mut self,
        config: alice_physics::thermal::ThermalConfig,
        resolution: usize,
    ) -> usize {
        let modifier = alice_physics::thermal::ThermalModifier::new(
            config,
            resolution,
            self.bounds_min,
            self.bounds_max,
        );
        let idx = self.modifier_count();
        self.add_modifier(Box::new(modifier));
        idx
    }

    /// Add a pressure modifier.
    pub fn add_pressure(
        &mut self,
        config: alice_physics::pressure::PressureConfig,
        resolution: usize,
    ) -> usize {
        let modifier = alice_physics::pressure::PressureModifier::new(
            config,
            resolution,
            self.bounds_min,
            self.bounds_max,
        );
        let idx = self.modifier_count();
        self.add_modifier(Box::new(modifier));
        idx
    }

    /// Add an erosion modifier.
    pub fn add_erosion(
        &mut self,
        config: alice_physics::erosion::ErosionConfig,
        resolution: usize,
    ) -> usize {
        let modifier = alice_physics::erosion::ErosionModifier::new(
            config,
            resolution,
            self.bounds_min,
            self.bounds_max,
        );
        let idx = self.modifier_count();
        self.add_modifier(Box::new(modifier));
        idx
    }

    /// Add a fracture modifier.
    pub fn add_fracture(
        &mut self,
        config: alice_physics::fracture::FractureConfig,
        resolution: usize,
    ) -> usize {
        let modifier = alice_physics::fracture::FractureModifier::new(
            config,
            resolution,
            self.bounds_min,
            self.bounds_max,
        );
        let idx = self.modifier_count();
        self.add_modifier(Box::new(modifier));
        idx
    }

    /// Add a phase change modifier.
    pub fn add_phase_change(
        &mut self,
        config: alice_physics::phase_change::PhaseChangeConfig,
        resolution: usize,
    ) -> usize {
        let modifier = alice_physics::phase_change::PhaseChangeModifier::new(
            config,
            resolution,
            self.bounds_min,
            self.bounds_max,
        );
        let idx = self.modifier_count();
        self.add_modifier(Box::new(modifier));
        idx
    }
}

/// Convenience function: create a simulated SDF from an SDF node.
pub fn simulate_sdf(node: &crate::types::SdfNode) -> SimulatedSdf {
    let compiled = CompiledSdf::compile(node);
    SimulatedSdf::new(compiled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;
    use alice_physics::pressure::PressureConfig;
    use alice_physics::thermal::ThermalConfig;

    #[test]
    fn test_simulated_sdf_no_modifiers() {
        let node = SdfNode::sphere(1.0);
        let sim = simulate_sdf(&node);

        let d = sim.distance(2.0, 0.0, 0.0);
        assert!((d - 1.0).abs() < 0.01, "No modifiers: d={}", d);
    }

    #[test]
    fn test_simulated_sdf_with_thermal() {
        let node = SdfNode::sphere(1.0);
        let mut sim = simulate_sdf(&node);

        let _idx = sim.add_thermal(ThermalConfig::default(), 8);
        assert_eq!(sim.modifier_count(), 1);

        // Without heat, distance unchanged
        let d = sim.distance(2.0, 0.0, 0.0);
        assert!((d - 1.0).abs() < 0.1, "No heat: d={}", d);
    }

    #[test]
    fn test_simulated_sdf_with_pressure() {
        let node = SdfNode::sphere(1.0);
        let mut sim = simulate_sdf(&node);

        sim.add_pressure(PressureConfig::default(), 8);
        assert_eq!(sim.modifier_count(), 1);

        // Step simulation
        sim.update(0.016);

        let d = sim.distance(2.0, 0.0, 0.0);
        assert!((d - 1.0).abs() < 0.1, "No pressure: d={}", d);
    }

    #[test]
    fn test_simulated_sdf_multiple_modifiers() {
        let node = SdfNode::sphere(1.0);
        let mut sim = simulate_sdf(&node);

        sim.add_thermal(ThermalConfig::default(), 8);
        sim.add_pressure(PressureConfig::default(), 8);
        assert_eq!(sim.modifier_count(), 2);

        // Step several times
        for _ in 0..10 {
            sim.update(0.016);
        }

        // Distance should still be reasonable
        let d = sim.distance(2.0, 0.0, 0.0);
        assert!(d > 0.0, "Should still be outside: d={}", d);
    }

    #[test]
    fn test_simulated_sdf_custom_bounds() {
        let node = SdfNode::sphere(1.0);
        let sim = SimulatedSdf::new(CompiledSdf::compile(&node))
            .with_bounds((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        let (bmin, bmax) = sim.bounds();
        assert_eq!(bmin, (-2.0, -2.0, -2.0));
        assert_eq!(bmax, (2.0, 2.0, 2.0));
    }
}
