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
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            root,
            metadata: None,
        }
    }

    /// Create with metadata
    pub fn with_metadata(root: SdfNode, metadata: SdfMetadata) -> Self {
        Self {
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
    #[inline]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Create an empty (inverted) AABB: `min = +∞`, `max = −∞`, so that the
    /// first [`expand_point`](Self::expand_point) makes it the point itself
    #[inline]
    pub const fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    /// Expand in place to include a point
    #[inline]
    pub fn expand_point(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Expand in place to include another AABB (the in-place form of
    /// [`union`](Self::union))
    #[inline]
    pub fn expand_aabb(&mut self, other: &Self) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Surface area `2·(dx·dy + dy·dz + dz·dx)` (for SAH)
    #[inline]
    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Longest axis (0 = X, 1 = Y, 2 = Z; ties go to the later axis)
    #[inline]
    pub fn longest_axis(&self) -> usize {
        let d = self.max - self.min;
        if d.x > d.y && d.x > d.z {
            0
        } else if d.y > d.z {
            1
        } else {
            2
        }
    }

    /// Signed distance to the box (negative inside, positive outside)
    #[inline]
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        let q = (point - self.center()).abs() - (self.max - self.min) * 0.5;
        q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
    }

    /// Create from center and half-extents
    pub fn from_center_extents(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    /// Get center point
    #[inline]
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
    pub fn union(&self, other: &Self) -> Self {
        Self {
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
        Self {
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
