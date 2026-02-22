//! Container types: SdfTree, SdfMetadata, Aabb, Ray, Hit
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use serde::{Deserialize, Serialize};

use super::SdfNode;

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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Aabb {
    /// Minimum corner
    pub min: Vec3,
    /// Maximum corner
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
    /// Ray origin point
    pub origin: Vec3,
    /// Ray direction (normalized)
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
