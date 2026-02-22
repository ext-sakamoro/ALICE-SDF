//! Variable-thickness offset surface (shell) modifier
//!
//! Creates a hollow shell around an SDF surface with independently
//! configurable inner and outer offsets. Unlike the `Onion` modifier
//! which applies uniform thickness, `ShellConfig` allows asymmetric
//! inner/outer expansion.
//!
//! # Example
//!
//! ```rust
//! use alice_sdf::shell::{ShellConfig, eval_shell};
//! use alice_sdf::types::SdfNode;
//! use glam::Vec3;
//!
//! let sphere = SdfNode::sphere(1.0);
//! let config = ShellConfig::new(0.1, 0.2); // inner=0.1, outer=0.2
//! let d = eval_shell(&sphere, Vec3::new(1.15, 0.0, 0.0), &config);
//! // Point is inside the shell wall
//! ```
//!
//! Author: Moroya Sakamoto

use crate::eval::eval;
use crate::types::SdfNode;
use glam::Vec3;
use std::sync::Arc;

/// Configuration for the shell modifier.
///
/// `inner_offset` controls how far inward the shell extends from the original surface.
/// `outer_offset` controls how far outward the shell extends.
///
/// The resulting shell occupies the region: `-inner_offset <= sdf(p) <= outer_offset`.
#[derive(Debug, Clone, Copy)]
pub struct ShellConfig {
    /// Inward shell thickness (positive = shrinks inward from surface).
    pub inner_offset: f32,
    /// Outward shell thickness (positive = expands outward from surface).
    pub outer_offset: f32,
}

impl ShellConfig {
    /// Create a new shell configuration.
    #[inline]
    pub fn new(inner_offset: f32, outer_offset: f32) -> Self {
        Self {
            inner_offset,
            outer_offset,
        }
    }

    /// Uniform thickness shell (same inner and outer offset).
    #[inline]
    pub fn uniform(thickness: f32) -> Self {
        Self {
            inner_offset: thickness,
            outer_offset: thickness,
        }
    }

    /// Total wall thickness.
    #[inline]
    pub fn wall_thickness(self) -> f32 {
        self.inner_offset + self.outer_offset
    }
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            inner_offset: 0.1,
            outer_offset: 0.1,
        }
    }
}

/// Evaluate the shell SDF at a point.
///
/// The shell SDF is defined as: `max(sdf(p) - outer, -(sdf(p) + inner))`
///
/// - Positive: outside the shell
/// - Negative: inside the shell wall
/// - Zero: on either the inner or outer surface
#[inline]
pub fn eval_shell(node: &SdfNode, point: Vec3, config: &ShellConfig) -> f32 {
    let d = eval(node, point);
    let outer_surface = d - config.outer_offset;
    let inner_surface = -(d + config.inner_offset);
    outer_surface.max(inner_surface)
}

/// Evaluate shell SDF for a batch of points.
pub fn eval_shell_batch(node: &SdfNode, points: &[Vec3], config: &ShellConfig) -> Vec<f32> {
    points
        .iter()
        .map(|&p| eval_shell(node, p, config))
        .collect()
}

/// Create an SdfNode that wraps a child in a shell.
///
/// This constructs a CSG subtraction: `onion_outer - onion_inner`.
/// For simple shells, prefer `eval_shell()` directly for better performance.
pub fn shell_node(child: Arc<SdfNode>, config: ShellConfig) -> SdfNode {
    // Shell = intersection of "enlarged" and "inverted shrunk"
    // Equivalent to: sdf(p) <= outer AND sdf(p) >= -inner
    // = max(sdf(p) - outer, -(sdf(p) + inner))
    //
    // Approximate via Onion: |sdf(p)| - thickness
    // where thickness = (inner + outer) / 2, then offset by (outer - inner) / 2
    let half_thick = (config.inner_offset + config.outer_offset) * 0.5;
    let center_offset = (config.outer_offset - config.inner_offset) * 0.5;

    if center_offset.abs() < 1e-6 {
        // Symmetric: use Onion directly
        SdfNode::Onion {
            child,
            thickness: half_thick,
        }
    } else {
        // Asymmetric: Onion + round offset
        SdfNode::Round {
            child: Arc::new(SdfNode::Onion {
                child,
                thickness: half_thick,
            }),
            radius: -center_offset,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_outside() {
        let sphere = SdfNode::sphere(1.0);
        let config = ShellConfig::new(0.1, 0.1);
        // Point well outside outer surface
        let d = eval_shell(&sphere, Vec3::new(2.0, 0.0, 0.0), &config);
        assert!(d > 0.0);
    }

    #[test]
    fn shell_inside_wall() {
        let sphere = SdfNode::sphere(1.0);
        let config = ShellConfig::new(0.1, 0.1);
        // Point on the original surface: sdf=0, shell = max(-0.1, -0.1) = -0.1
        let d = eval_shell(&sphere, Vec3::new(1.0, 0.0, 0.0), &config);
        assert!(d < 0.0);
    }

    #[test]
    fn shell_hollow_center() {
        let sphere = SdfNode::sphere(1.0);
        let config = ShellConfig::new(0.1, 0.1);
        // Origin: sdf = -1.0, shell = max(-1.1, -(-1.0 + 0.1)) = max(-1.1, 0.9) = 0.9
        let d = eval_shell(&sphere, Vec3::ZERO, &config);
        assert!(d > 0.0, "Center should be outside shell (hollow)");
    }

    #[test]
    fn shell_outer_boundary() {
        let sphere = SdfNode::sphere(1.0);
        let config = ShellConfig::new(0.1, 0.2);
        // At distance 1.2 from center: sdf=0.2, shell = max(0.0, -0.3) = 0.0
        let d = eval_shell(&sphere, Vec3::new(1.2, 0.0, 0.0), &config);
        assert!(d.abs() < 1e-4, "Should be on outer surface, got {}", d);
    }

    #[test]
    fn shell_inner_boundary() {
        let sphere = SdfNode::sphere(1.0);
        let config = ShellConfig::new(0.1, 0.2);
        // At distance 0.9 from center: sdf=-0.1, shell = max(-0.3, 0.0) = 0.0
        let d = eval_shell(&sphere, Vec3::new(0.9, 0.0, 0.0), &config);
        assert!(d.abs() < 1e-4, "Should be on inner surface, got {}", d);
    }

    #[test]
    fn shell_asymmetric() {
        let sphere = SdfNode::sphere(1.0);
        let config = ShellConfig::new(0.05, 0.3);
        // Total wall = 0.35
        assert!((config.wall_thickness() - 0.35).abs() < 1e-6);

        // Inside wall at original surface
        let d = eval_shell(&sphere, Vec3::new(1.0, 0.0, 0.0), &config);
        assert!(d < 0.0);
    }

    #[test]
    fn shell_zero_thickness() {
        let sphere = SdfNode::sphere(1.0);
        let config = ShellConfig::new(0.0, 0.0);
        // Zero-thickness shell: same as original surface
        let d = eval_shell(&sphere, Vec3::new(1.0, 0.0, 0.0), &config);
        assert!(d.abs() < 1e-5);
    }

    #[test]
    fn shell_batch() {
        let sphere = SdfNode::sphere(1.0);
        let config = ShellConfig::new(0.1, 0.1);
        let points = vec![
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,
        ];
        let results = eval_shell_batch(&sphere, &points, &config);
        assert_eq!(results.len(), 3);
        assert!(results[0] > 0.0); // outside
        assert!(results[1] < 0.0); // in wall
        assert!(results[2] > 0.0); // hollow center
    }

    #[test]
    fn shell_uniform_config() {
        let config = ShellConfig::uniform(0.5);
        assert_eq!(config.inner_offset, 0.5);
        assert_eq!(config.outer_offset, 0.5);
        assert!((config.wall_thickness() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn shell_node_symmetric() {
        let child = Arc::new(SdfNode::sphere(1.0));
        let config = ShellConfig::uniform(0.1);
        let node = shell_node(child, config);
        // Should produce an Onion node for symmetric case
        match node {
            SdfNode::Onion { thickness, .. } => {
                assert!((thickness - 0.1).abs() < 1e-6);
            }
            _ => panic!("Expected Onion node for symmetric shell"),
        }
    }

    #[test]
    fn shell_default_config() {
        let config = ShellConfig::default();
        assert_eq!(config.inner_offset, 0.1);
        assert_eq!(config.outer_offset, 0.1);
    }
}
