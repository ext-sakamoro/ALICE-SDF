//! Core types for ALICE-SDF
//!
//! Defines the SdfNode tree structure and related types.
//!
//! Author: Moroya Sakamoto

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Signed Distance Function Node
///
/// Represents a node in the SDF tree. Each node can be:
/// - A primitive shape (sphere, box, etc.)
/// - An operation combining two shapes (union, intersection, etc.)
/// - A transform applied to a child node
/// - A modifier deforming a child node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SdfNode {
    // === Primitives ===
    /// Sphere with radius
    Sphere { radius: f32 },

    /// Axis-aligned box with half-extents
    Box3d { half_extents: Vec3 },

    /// Cylinder along Y-axis with radius and half-height
    Cylinder { radius: f32, half_height: f32 },

    /// Torus in XZ plane with major and minor radius
    Torus { major_radius: f32, minor_radius: f32 },

    /// Infinite plane with normal and distance from origin
    Plane { normal: Vec3, distance: f32 },

    /// Capsule between two points with radius
    Capsule { point_a: Vec3, point_b: Vec3, radius: f32 },

    // === Operations ===
    /// Union of two shapes (min distance)
    Union { a: Arc<SdfNode>, b: Arc<SdfNode> },

    /// Intersection of two shapes (max distance)
    Intersection { a: Arc<SdfNode>, b: Arc<SdfNode> },

    /// Subtraction: a minus b (max of a and -b)
    Subtraction { a: Arc<SdfNode>, b: Arc<SdfNode> },

    /// Smooth union with blending factor k
    SmoothUnion { a: Arc<SdfNode>, b: Arc<SdfNode>, k: f32 },

    /// Smooth intersection with blending factor k
    SmoothIntersection { a: Arc<SdfNode>, b: Arc<SdfNode>, k: f32 },

    /// Smooth subtraction with blending factor k
    SmoothSubtraction { a: Arc<SdfNode>, b: Arc<SdfNode>, k: f32 },

    // === Transforms ===
    /// Translation
    Translate { child: Arc<SdfNode>, offset: Vec3 },

    /// Rotation (quaternion)
    Rotate { child: Arc<SdfNode>, rotation: Quat },

    /// Uniform scale
    Scale { child: Arc<SdfNode>, factor: f32 },

    /// Non-uniform scale (stretches the shape)
    ScaleNonUniform { child: Arc<SdfNode>, factors: Vec3 },

    // === Modifiers ===
    /// Twist around Y-axis (radians per unit height)
    Twist { child: Arc<SdfNode>, strength: f32 },

    /// Bend around Y-axis
    Bend { child: Arc<SdfNode>, curvature: f32 },

    /// Infinite repetition with spacing
    RepeatInfinite { child: Arc<SdfNode>, spacing: Vec3 },

    /// Finite repetition with count and spacing
    RepeatFinite {
        child: Arc<SdfNode>,
        count: [u32; 3],
        spacing: Vec3,
    },

    /// Perlin noise displacement
    Noise {
        child: Arc<SdfNode>,
        amplitude: f32,
        frequency: f32,
        seed: u32,
    },

    /// Round edges by subtracting radius
    Round { child: Arc<SdfNode>, radius: f32 },

    /// Onion: creates a shell with thickness
    Onion { child: Arc<SdfNode>, thickness: f32 },

    /// Elongate along an axis
    Elongate { child: Arc<SdfNode>, amount: Vec3 },
}

impl SdfNode {
    // === Primitive constructors ===

    /// Create a sphere with the given radius
    #[inline]
    pub fn sphere(radius: f32) -> Self {
        SdfNode::Sphere { radius }
    }

    /// Create an axis-aligned box with the given dimensions
    #[inline]
    pub fn box3d(width: f32, height: f32, depth: f32) -> Self {
        SdfNode::Box3d {
            half_extents: Vec3::new(width * 0.5, height * 0.5, depth * 0.5),
        }
    }

    /// Create a cylinder along Y-axis
    #[inline]
    pub fn cylinder(radius: f32, height: f32) -> Self {
        SdfNode::Cylinder {
            radius,
            half_height: height * 0.5,
        }
    }

    /// Create a torus in the XZ plane
    #[inline]
    pub fn torus(major_radius: f32, minor_radius: f32) -> Self {
        SdfNode::Torus {
            major_radius,
            minor_radius,
        }
    }

    /// Create an infinite plane
    #[inline]
    pub fn plane(normal: Vec3, distance: f32) -> Self {
        SdfNode::Plane {
            normal: normal.normalize(),
            distance,
        }
    }

    /// Create a capsule between two points
    #[inline]
    pub fn capsule(a: Vec3, b: Vec3, radius: f32) -> Self {
        SdfNode::Capsule {
            point_a: a,
            point_b: b,
            radius,
        }
    }

    // === Operation methods ===

    /// Union with another shape
    #[inline]
    pub fn union(self, other: SdfNode) -> Self {
        SdfNode::Union {
            a: Arc::new(self),
            b: Arc::new(other),
        }
    }

    /// Intersection with another shape
    #[inline]
    pub fn intersection(self, other: SdfNode) -> Self {
        SdfNode::Intersection {
            a: Arc::new(self),
            b: Arc::new(other),
        }
    }

    /// Subtract another shape from this one
    #[inline]
    pub fn subtract(self, other: SdfNode) -> Self {
        SdfNode::Subtraction {
            a: Arc::new(self),
            b: Arc::new(other),
        }
    }

    /// Smooth union with another shape
    #[inline]
    pub fn smooth_union(self, other: SdfNode, k: f32) -> Self {
        SdfNode::SmoothUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Smooth intersection with another shape
    #[inline]
    pub fn smooth_intersection(self, other: SdfNode, k: f32) -> Self {
        SdfNode::SmoothIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Smooth subtraction of another shape
    #[inline]
    pub fn smooth_subtract(self, other: SdfNode, k: f32) -> Self {
        SdfNode::SmoothSubtraction {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    // === Transform methods ===

    /// Translate by offset
    #[inline]
    pub fn translate(self, x: f32, y: f32, z: f32) -> Self {
        SdfNode::Translate {
            child: Arc::new(self),
            offset: Vec3::new(x, y, z),
        }
    }

    /// Translate by vector
    #[inline]
    pub fn translate_vec(self, offset: Vec3) -> Self {
        SdfNode::Translate {
            child: Arc::new(self),
            offset,
        }
    }

    /// Rotate by quaternion
    #[inline]
    pub fn rotate(self, rotation: Quat) -> Self {
        SdfNode::Rotate {
            child: Arc::new(self),
            rotation,
        }
    }

    /// Rotate by Euler angles (radians)
    #[inline]
    pub fn rotate_euler(self, x: f32, y: f32, z: f32) -> Self {
        SdfNode::Rotate {
            child: Arc::new(self),
            rotation: Quat::from_euler(glam::EulerRot::XYZ, x, y, z),
        }
    }

    /// Uniform scale
    #[inline]
    pub fn scale(self, factor: f32) -> Self {
        SdfNode::Scale {
            child: Arc::new(self),
            factor,
        }
    }

    /// Non-uniform scale
    #[inline]
    pub fn scale_xyz(self, x: f32, y: f32, z: f32) -> Self {
        SdfNode::ScaleNonUniform {
            child: Arc::new(self),
            factors: Vec3::new(x, y, z),
        }
    }

    // === Modifier methods ===

    /// Twist around Y-axis
    #[inline]
    pub fn twist(self, strength: f32) -> Self {
        SdfNode::Twist {
            child: Arc::new(self),
            strength,
        }
    }

    /// Bend around Y-axis
    #[inline]
    pub fn bend(self, curvature: f32) -> Self {
        SdfNode::Bend {
            child: Arc::new(self),
            curvature,
        }
    }

    /// Infinite repetition
    #[inline]
    pub fn repeat_infinite(self, spacing_x: f32, spacing_y: f32, spacing_z: f32) -> Self {
        SdfNode::RepeatInfinite {
            child: Arc::new(self),
            spacing: Vec3::new(spacing_x, spacing_y, spacing_z),
        }
    }

    /// Finite repetition
    #[inline]
    pub fn repeat_finite(self, count: [u32; 3], spacing: Vec3) -> Self {
        SdfNode::RepeatFinite {
            child: Arc::new(self),
            count,
            spacing,
        }
    }

    /// Perlin noise displacement
    #[inline]
    pub fn noise(self, amplitude: f32, frequency: f32, seed: u32) -> Self {
        SdfNode::Noise {
            child: Arc::new(self),
            amplitude,
            frequency,
            seed,
        }
    }

    /// Round edges
    #[inline]
    pub fn round(self, radius: f32) -> Self {
        SdfNode::Round {
            child: Arc::new(self),
            radius,
        }
    }

    /// Create a shell (onion)
    #[inline]
    pub fn onion(self, thickness: f32) -> Self {
        SdfNode::Onion {
            child: Arc::new(self),
            thickness,
        }
    }

    /// Elongate along an axis
    #[inline]
    pub fn elongate(self, x: f32, y: f32, z: f32) -> Self {
        SdfNode::Elongate {
            child: Arc::new(self),
            amount: Vec3::new(x, y, z),
        }
    }

    /// Count total nodes in the tree
    pub fn node_count(&self) -> u32 {
        match self {
            // Primitives: 1 node
            SdfNode::Sphere { .. }
            | SdfNode::Box3d { .. }
            | SdfNode::Cylinder { .. }
            | SdfNode::Torus { .. }
            | SdfNode::Plane { .. }
            | SdfNode::Capsule { .. } => 1,

            // Operations: 1 + children
            SdfNode::Union { a, b }
            | SdfNode::Intersection { a, b }
            | SdfNode::Subtraction { a, b }
            | SdfNode::SmoothUnion { a, b, .. }
            | SdfNode::SmoothIntersection { a, b, .. }
            | SdfNode::SmoothSubtraction { a, b, .. } => 1 + a.node_count() + b.node_count(),

            // Transforms and modifiers: 1 + child
            SdfNode::Translate { child, .. }
            | SdfNode::Rotate { child, .. }
            | SdfNode::Scale { child, .. }
            | SdfNode::ScaleNonUniform { child, .. }
            | SdfNode::Twist { child, .. }
            | SdfNode::Bend { child, .. }
            | SdfNode::RepeatInfinite { child, .. }
            | SdfNode::RepeatFinite { child, .. }
            | SdfNode::Noise { child, .. }
            | SdfNode::Round { child, .. }
            | SdfNode::Onion { child, .. }
            | SdfNode::Elongate { child, .. } => 1 + child.node_count(),
        }
    }
}

/// SDF Tree - top-level container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdfTree {
    /// Version string
    pub version: String,
    /// Root node
    pub root: SdfNode,
    /// Optional metadata
    pub metadata: Option<SdfMetadata>,
}

impl SdfTree {
    /// Create a new SDF tree
    pub fn new(root: SdfNode) -> Self {
        SdfTree {
            version: env!("CARGO_PKG_VERSION").to_string(),
            root,
            metadata: None,
        }
    }

    /// Create with metadata
    pub fn with_metadata(root: SdfNode, metadata: SdfMetadata) -> Self {
        SdfTree {
            version: env!("CARGO_PKG_VERSION").to_string(),
            root,
            metadata: Some(metadata),
        }
    }

    /// Get total node count
    pub fn node_count(&self) -> u32 {
        self.root.node_count()
    }
}

/// Optional metadata for SDF trees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdfMetadata {
    /// Name of the model
    pub name: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Author
    pub author: Option<String>,
    /// Bounding box hint (min, max)
    pub bounds: Option<(Vec3, Vec3)>,
    /// Custom key-value pairs
    pub custom: Option<std::collections::HashMap<String, String>>,
}

impl Default for SdfMetadata {
    fn default() -> Self {
        SdfMetadata {
            name: None,
            description: None,
            author: None,
            bounds: None,
            custom: None,
        }
    }
}

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Create a new AABB
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Aabb { min, max }
    }

    /// Create from center and half-extents
    pub fn from_center_extents(center: Vec3, half_extents: Vec3) -> Self {
        Aabb {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    /// Get center point
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Get size
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// Get half-extents
    pub fn half_extents(&self) -> Vec3 {
        self.size() * 0.5
    }

    /// Check if point is inside
    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Expand to include another AABB
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

/// Ray for raycasting
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    /// Create a new ray
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Ray {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Get point along ray at distance t
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// Hit result from raycasting
#[derive(Debug, Clone, Copy)]
pub struct Hit {
    /// Distance along ray
    pub distance: f32,
    /// Hit point
    pub point: Vec3,
    /// Surface normal (approximate)
    pub normal: Vec3,
    /// Number of marching steps
    pub steps: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_creation() {
        let sphere = SdfNode::sphere(1.0);
        assert_eq!(sphere.node_count(), 1);
    }

    #[test]
    fn test_union() {
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::box3d(2.0, 2.0, 2.0);
        let union = a.union(b);
        assert_eq!(union.node_count(), 3);
    }

    #[test]
    fn test_transform_chain() {
        let shape = SdfNode::sphere(1.0)
            .translate(1.0, 0.0, 0.0)
            .rotate_euler(0.0, std::f32::consts::PI, 0.0)
            .scale(2.0);
        assert_eq!(shape.node_count(), 4);
    }

    #[test]
    fn test_sdf_tree() {
        let tree = SdfTree::new(SdfNode::sphere(1.0));
        assert_eq!(tree.node_count(), 1);
    }

    #[test]
    fn test_aabb() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(aabb.contains(Vec3::ZERO));
        assert!(!aabb.contains(Vec3::new(2.0, 0.0, 0.0)));
    }
}
