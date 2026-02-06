//! AABB (Axis-Aligned Bounding Box) for BVH pruning
//!
//! This module provides efficient AABB computation and distance functions
//! for accelerating SDF evaluation through spatial pruning.
//!
//! Author: Moroya Sakamoto

use glam::{Quat, Vec3};

/// SIMD-friendly AABB structure (32 bytes, cache-aligned)
///
/// Stores the minimum and maximum corners of an axis-aligned bounding box.
/// Designed for efficient SIMD operations and cache-friendly memory access.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct AabbPacked {
    /// Minimum X coordinate
    pub min_x: f32,
    /// Minimum Y coordinate
    pub min_y: f32,
    /// Minimum Z coordinate
    pub min_z: f32,
    /// Padding for alignment
    pub _pad1: f32,
    /// Maximum X coordinate
    pub max_x: f32,
    /// Maximum Y coordinate
    pub max_y: f32,
    /// Maximum Z coordinate
    pub max_z: f32,
    /// Padding for alignment
    pub _pad2: f32,
}

impl AabbPacked {
    /// Create an empty (invalid) AABB
    #[inline]
    pub fn empty() -> Self {
        AabbPacked {
            min_x: f32::MAX,
            min_y: f32::MAX,
            min_z: f32::MAX,
            _pad1: 0.0,
            max_x: f32::MIN,
            max_y: f32::MIN,
            max_z: f32::MIN,
            _pad2: 0.0,
        }
    }

    /// Create an AABB that contains everything (infinite)
    #[inline]
    pub fn infinite() -> Self {
        AabbPacked {
            min_x: f32::MIN,
            min_y: f32::MIN,
            min_z: f32::MIN,
            _pad1: 0.0,
            max_x: f32::MAX,
            max_y: f32::MAX,
            max_z: f32::MAX,
            _pad2: 0.0,
        }
    }

    /// Create an AABB from min and max corners
    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        AabbPacked {
            min_x: min.x,
            min_y: min.y,
            min_z: min.z,
            _pad1: 0.0,
            max_x: max.x,
            max_y: max.y,
            max_z: max.z,
            _pad2: 0.0,
        }
    }

    /// Create an AABB centered at origin with given half-size
    #[inline]
    pub fn from_half_size(half_size: Vec3) -> Self {
        AabbPacked::new(-half_size, half_size)
    }

    /// Create an AABB for a sphere
    #[inline]
    pub fn from_sphere(center: Vec3, radius: f32) -> Self {
        let r = Vec3::splat(radius);
        AabbPacked::new(center - r, center + r)
    }

    /// Get minimum corner as Vec3
    #[inline]
    pub fn min(&self) -> Vec3 {
        Vec3::new(self.min_x, self.min_y, self.min_z)
    }

    /// Get maximum corner as Vec3
    #[inline]
    pub fn max(&self) -> Vec3 {
        Vec3::new(self.max_x, self.max_y, self.max_z)
    }

    /// Get center of the AABB
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min() + self.max()) * 0.5
    }

    /// Get half-size (extents) of the AABB
    #[inline]
    pub fn half_size(&self) -> Vec3 {
        (self.max() - self.min()) * 0.5
    }

    /// Check if the AABB is valid (non-empty)
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.min_x <= self.max_x && self.min_y <= self.max_y && self.min_z <= self.max_z
    }

    /// Compute signed distance from a point to the AABB surface
    ///
    /// Returns:
    /// - Negative if point is inside
    /// - Zero if point is on surface
    /// - Positive if point is outside
    #[inline]
    pub fn distance_to_point(&self, p: Vec3) -> f32 {
        let center = self.center();
        let half = self.half_size();

        // q = abs(p - center) - half_size
        let q = (p - center).abs() - half;

        // Same formula as box SDF
        let outside = q.max(Vec3::ZERO).length();
        let inside = q.x.max(q.y).max(q.z).min(0.0);

        outside + inside
    }

    /// Fast conservative distance estimate (may underestimate)
    ///
    /// Faster than exact distance but may return smaller values.
    /// Safe for BVH pruning (won't skip valid nodes).
    #[inline]
    pub fn distance_to_point_fast(&self, p: Vec3) -> f32 {
        let dx = (self.min_x - p.x).max(p.x - self.max_x).max(0.0);
        let dy = (self.min_y - p.y).max(p.y - self.max_y).max(0.0);
        let dz = (self.min_z - p.z).max(p.z - self.max_z).max(0.0);

        // Return L-infinity norm (max component) - conservative underestimate
        dx.max(dy).max(dz)
    }

    /// Union of two AABBs (smallest AABB containing both)
    #[inline]
    pub fn union(&self, other: &AabbPacked) -> AabbPacked {
        AabbPacked {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            min_z: self.min_z.min(other.min_z),
            _pad1: 0.0,
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
            max_z: self.max_z.max(other.max_z),
            _pad2: 0.0,
        }
    }

    /// Intersection of two AABBs
    #[inline]
    pub fn intersection(&self, other: &AabbPacked) -> AabbPacked {
        AabbPacked {
            min_x: self.min_x.max(other.min_x),
            min_y: self.min_y.max(other.min_y),
            min_z: self.min_z.max(other.min_z),
            _pad1: 0.0,
            max_x: self.max_x.min(other.max_x),
            max_y: self.max_y.min(other.max_y),
            max_z: self.max_z.min(other.max_z),
            _pad2: 0.0,
        }
    }

    /// Expand AABB by a scalar amount in all directions
    #[inline]
    pub fn expand(&self, amount: f32) -> AabbPacked {
        AabbPacked {
            min_x: self.min_x - amount,
            min_y: self.min_y - amount,
            min_z: self.min_z - amount,
            _pad1: 0.0,
            max_x: self.max_x + amount,
            max_y: self.max_y + amount,
            max_z: self.max_z + amount,
            _pad2: 0.0,
        }
    }

    /// Translate the AABB
    #[inline]
    pub fn translate(&self, offset: Vec3) -> AabbPacked {
        AabbPacked {
            min_x: self.min_x + offset.x,
            min_y: self.min_y + offset.y,
            min_z: self.min_z + offset.z,
            _pad1: 0.0,
            max_x: self.max_x + offset.x,
            max_y: self.max_y + offset.y,
            max_z: self.max_z + offset.z,
            _pad2: 0.0,
        }
    }

    /// Scale the AABB uniformly from origin
    #[inline]
    pub fn scale(&self, factor: f32) -> AabbPacked {
        AabbPacked {
            min_x: self.min_x * factor,
            min_y: self.min_y * factor,
            min_z: self.min_z * factor,
            _pad1: 0.0,
            max_x: self.max_x * factor,
            max_y: self.max_y * factor,
            max_z: self.max_z * factor,
            _pad2: 0.0,
        }
    }

    /// Scale the AABB non-uniformly from origin
    #[inline]
    pub fn scale_nonuniform(&self, factors: Vec3) -> AabbPacked {
        let min = self.min() * factors;
        let max = self.max() * factors;
        // Handle negative scaling
        AabbPacked::new(min.min(max), min.max(max))
    }

    /// Transform AABB by rotation (conservative - may be larger than necessary)
    ///
    /// Computes the AABB of the rotated AABB, which is always larger or equal
    /// to the actual rotated bounding box.
    #[inline]
    pub fn rotate(&self, rotation: Quat) -> AabbPacked {
        let center = self.center();
        let half = self.half_size();

        // Rotate all 8 corners and find new AABB
        let corners = [
            Vec3::new(-half.x, -half.y, -half.z),
            Vec3::new(half.x, -half.y, -half.z),
            Vec3::new(-half.x, half.y, -half.z),
            Vec3::new(half.x, half.y, -half.z),
            Vec3::new(-half.x, -half.y, half.z),
            Vec3::new(half.x, -half.y, half.z),
            Vec3::new(-half.x, half.y, half.z),
            Vec3::new(half.x, half.y, half.z),
        ];

        let mut new_min = Vec3::splat(f32::MAX);
        let mut new_max = Vec3::splat(f32::MIN);

        for corner in corners {
            let rotated = rotation * corner + center;
            new_min = new_min.min(rotated);
            new_max = new_max.max(rotated);
        }

        AabbPacked::new(new_min, new_max)
    }
}

impl Default for AabbPacked {
    fn default() -> Self {
        AabbPacked::empty()
    }
}

/// Compute AABB for primitive SDF nodes
pub mod primitives {
    use super::*;

    /// AABB for a sphere centered at origin
    #[inline]
    pub fn sphere_aabb(radius: f32) -> AabbPacked {
        AabbPacked::from_sphere(Vec3::ZERO, radius)
    }

    /// AABB for a box centered at origin
    #[inline]
    pub fn box_aabb(half_extents: Vec3) -> AabbPacked {
        AabbPacked::new(-half_extents, half_extents)
    }

    /// AABB for a cylinder along Y axis
    #[inline]
    pub fn cylinder_aabb(radius: f32, half_height: f32) -> AabbPacked {
        AabbPacked::new(
            Vec3::new(-radius, -half_height, -radius),
            Vec3::new(radius, half_height, radius),
        )
    }

    /// AABB for a torus in XZ plane
    #[inline]
    pub fn torus_aabb(major_radius: f32, minor_radius: f32) -> AabbPacked {
        let r = major_radius + minor_radius;
        AabbPacked::new(
            Vec3::new(-r, -minor_radius, -r),
            Vec3::new(r, minor_radius, r),
        )
    }

    /// AABB for an infinite plane (returns infinite AABB)
    #[inline]
    pub fn plane_aabb() -> AabbPacked {
        AabbPacked::infinite()
    }

    /// AABB for a capsule
    #[inline]
    pub fn capsule_aabb(point_a: Vec3, point_b: Vec3, radius: f32) -> AabbPacked {
        let min = point_a.min(point_b) - Vec3::splat(radius);
        let max = point_a.max(point_b) + Vec3::splat(radius);
        AabbPacked::new(min, max)
    }

    /// AABB for a rounded cone along Y axis
    #[allow(dead_code)]
    #[inline]
    pub fn rounded_cone_aabb(r1: f32, r2: f32, half_height: f32) -> AabbPacked {
        let max_r = r1.max(r2);
        AabbPacked::new(
            Vec3::new(-max_r, -half_height - r1, -max_r),
            Vec3::new(max_r, half_height + r2, max_r),
        )
    }

    /// AABB for a 4-sided pyramid along Y axis
    #[allow(dead_code)]
    #[inline]
    pub fn pyramid_aabb(half_height: f32) -> AabbPacked {
        AabbPacked::new(
            Vec3::new(-0.5, -half_height, -0.5),
            Vec3::new(0.5, half_height, 0.5),
        )
    }

    /// AABB for a regular octahedron
    #[allow(dead_code)]
    #[inline]
    pub fn octahedron_aabb(size: f32) -> AabbPacked {
        AabbPacked::new(Vec3::splat(-size), Vec3::splat(size))
    }

    /// AABB for a hexagonal prism (hex in XY, extruded along Z)
    #[allow(dead_code)]
    #[inline]
    pub fn hex_prism_aabb(hex_radius: f32, half_height: f32) -> AabbPacked {
        AabbPacked::new(
            Vec3::new(-hex_radius, -hex_radius, -half_height),
            Vec3::new(hex_radius, hex_radius, half_height),
        )
    }

    /// AABB for a chain link shape
    #[allow(dead_code)]
    #[inline]
    pub fn link_aabb(half_length: f32, r1: f32, r2: f32) -> AabbPacked {
        let r = r1 + r2;
        AabbPacked::new(
            Vec3::new(-r, -(half_length + r2), -r2),
            Vec3::new(r, half_length + r2, r2),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_aabb_size_and_alignment() {
        assert_eq!(mem::size_of::<AabbPacked>(), 32);
        assert_eq!(mem::align_of::<AabbPacked>(), 32);
    }

    #[test]
    fn test_aabb_from_sphere() {
        let aabb = AabbPacked::from_sphere(Vec3::ZERO, 1.0);
        assert_eq!(aabb.min(), Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(aabb.max(), Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_aabb_distance_inside() {
        let aabb = AabbPacked::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let d = aabb.distance_to_point(Vec3::ZERO);
        assert!(d < 0.0); // Inside
    }

    #[test]
    fn test_aabb_distance_surface() {
        let aabb = AabbPacked::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let d = aabb.distance_to_point(Vec3::new(1.0, 0.0, 0.0));
        assert!(d.abs() < 0.0001); // On surface
    }

    #[test]
    fn test_aabb_distance_outside() {
        let aabb = AabbPacked::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let d = aabb.distance_to_point(Vec3::new(2.0, 0.0, 0.0));
        assert!((d - 1.0).abs() < 0.0001); // 1 unit outside
    }

    #[test]
    fn test_aabb_union() {
        let a = AabbPacked::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let b = AabbPacked::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let u = a.union(&b);
        assert_eq!(u.min(), Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(u.max(), Vec3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn test_aabb_translate() {
        let aabb = AabbPacked::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let translated = aabb.translate(Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(translated.min(), Vec3::new(1.0, -1.0, -1.0));
        assert_eq!(translated.max(), Vec3::new(3.0, 1.0, 1.0));
    }

    #[test]
    fn test_aabb_scale() {
        let aabb = AabbPacked::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let scaled = aabb.scale(2.0);
        assert_eq!(scaled.min(), Vec3::new(-2.0, -2.0, -2.0));
        assert_eq!(scaled.max(), Vec3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn test_primitive_aabbs() {
        let sphere = primitives::sphere_aabb(1.0);
        assert_eq!(sphere.center(), Vec3::ZERO);
        assert_eq!(sphere.half_size(), Vec3::splat(1.0));

        let cylinder = primitives::cylinder_aabb(0.5, 1.0);
        assert_eq!(cylinder.half_size(), Vec3::new(0.5, 1.0, 0.5));
    }
}
