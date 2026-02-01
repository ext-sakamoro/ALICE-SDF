//! BVH (Bounding Volume Hierarchy) for mesh acceleration
//!
//! Provides O(log n) distance queries for triangle meshes.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use rayon::prelude::*;

/// Axis-Aligned Bounding Box
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Create an empty (inverted) AABB
    #[inline]
    pub fn empty() -> Self {
        Aabb {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    /// Create AABB from min/max
    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Aabb { min, max }
    }

    /// Expand AABB to include a point
    #[inline]
    pub fn expand_point(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Expand AABB to include another AABB
    #[inline]
    pub fn expand_aabb(&mut self, other: &Aabb) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Get center of AABB
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Get surface area (for SAH)
    #[inline]
    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Get longest axis (0=X, 1=Y, 2=Z)
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

    /// Signed distance to AABB (negative inside, positive outside)
    #[inline]
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        let q = (point - self.center()).abs() - (self.max - self.min) * 0.5;
        q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
    }
}

/// Triangle with precomputed data for fast distance queries
#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub normal: Vec3,
    pub aabb: Aabb,
}

impl Triangle {
    /// Create triangle from vertices
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3) -> Self {
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let normal = e1.cross(e2).normalize_or_zero();

        let mut aabb = Aabb::empty();
        aabb.expand_point(v0);
        aabb.expand_point(v1);
        aabb.expand_point(v2);

        Triangle { v0, v1, v2, normal, aabb }
    }

    /// Signed distance to triangle
    /// Uses Inigo Quilez's method for exact unsigned distance,
    /// then determines sign using pseudo-normal method
    #[inline]
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        let v0 = self.v0;
        let v1 = self.v1;
        let v2 = self.v2;

        let v10 = v1 - v0;
        let v21 = v2 - v1;
        let v02 = v0 - v2;

        let p0 = point - v0;
        let p1 = point - v1;
        let p2 = point - v2;

        let n = v10.cross(v02);

        // Determine if point projects inside triangle
        let sign_p0 = v10.cross(n).dot(p0);
        let sign_p1 = v21.cross(n).dot(p1);
        let sign_p2 = v02.cross(n).dot(p2);

        let unsigned_dist = if sign_p0 >= 0.0 && sign_p1 >= 0.0 && sign_p2 >= 0.0 {
            // Point projects inside triangle - distance to plane
            let h = n.dot(p0);
            let n_len_sq = n.length_squared();
            if n_len_sq > 1e-10 {
                (h * h / n_len_sq).sqrt()
            } else {
                0.0
            }
        } else {
            // Point projects outside - find closest point on edges
            let clamp01 = |t: f32| t.clamp(0.0, 1.0);

            // Edge 0-1
            let t0 = clamp01(v10.dot(p0) / v10.length_squared().max(1e-10));
            let d0 = (p0 - v10 * t0).length_squared();

            // Edge 1-2
            let t1 = clamp01(v21.dot(p1) / v21.length_squared().max(1e-10));
            let d1 = (p1 - v21 * t1).length_squared();

            // Edge 2-0
            let t2 = clamp01((-v02).dot(p2) / v02.length_squared().max(1e-10));
            let d2 = (p2 + v02 * t2).length_squared();

            d0.min(d1).min(d2).sqrt()
        };

        // Determine sign using triangle normal
        // Positive = outside (same side as normal), Negative = inside
        let sign = if self.normal.dot(point - v0) >= 0.0 { 1.0 } else { -1.0 };

        unsigned_dist * sign
    }

    /// Unsigned distance to triangle (always positive)
    #[inline]
    pub fn unsigned_distance(&self, point: Vec3) -> f32 {
        self.signed_distance(point).abs()
    }
}

/// BVH Node
#[derive(Debug)]
pub enum BvhNode {
    /// Leaf node containing triangle indices
    Leaf {
        aabb: Aabb,
        triangles: Vec<usize>,
    },
    /// Internal node with two children
    Internal {
        aabb: Aabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
}

impl BvhNode {
    /// Get AABB of this node
    #[inline]
    pub fn aabb(&self) -> &Aabb {
        match self {
            BvhNode::Leaf { aabb, .. } => aabb,
            BvhNode::Internal { aabb, .. } => aabb,
        }
    }
}

/// BVH for triangle mesh
pub struct MeshBvh {
    pub triangles: Vec<Triangle>,
    pub root: Option<BvhNode>,
    pub max_triangles_per_leaf: usize,
}

impl MeshBvh {
    /// Build BVH from mesh data
    pub fn build(vertices: &[Vec3], indices: &[u32], max_triangles_per_leaf: usize) -> Self {
        // Create triangles
        let triangles: Vec<Triangle> = indices
            .chunks(3)
            .filter_map(|chunk| {
                if chunk.len() == 3 {
                    let v0 = vertices[chunk[0] as usize];
                    let v1 = vertices[chunk[1] as usize];
                    let v2 = vertices[chunk[2] as usize];
                    Some(Triangle::new(v0, v1, v2))
                } else {
                    None
                }
            })
            .collect();

        if triangles.is_empty() {
            return MeshBvh {
                triangles,
                root: None,
                max_triangles_per_leaf,
            };
        }

        let indices: Vec<usize> = (0..triangles.len()).collect();
        let root = Self::build_node(&triangles, indices, max_triangles_per_leaf);

        MeshBvh {
            triangles,
            root: Some(root),
            max_triangles_per_leaf,
        }
    }

    /// Recursively build BVH nodes
    fn build_node(
        triangles: &[Triangle],
        indices: Vec<usize>,
        max_per_leaf: usize,
    ) -> BvhNode {
        // Compute AABB for all triangles
        let mut aabb = Aabb::empty();
        for &idx in &indices {
            aabb.expand_aabb(&triangles[idx].aabb);
        }

        // If few triangles, create leaf
        if indices.len() <= max_per_leaf {
            return BvhNode::Leaf {
                aabb,
                triangles: indices,
            };
        }

        // Split along longest axis using median
        let axis = aabb.longest_axis();
        let mut sorted_indices = indices;
        sorted_indices.sort_by(|&a, &b| {
            let ca = triangles[a].aabb.center();
            let cb = triangles[b].aabb.center();
            let va = match axis {
                0 => ca.x,
                1 => ca.y,
                _ => ca.z,
            };
            let vb = match axis {
                0 => cb.x,
                1 => cb.y,
                _ => cb.z,
            };
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = sorted_indices.len() / 2;
        let (left_indices, right_indices) = sorted_indices.split_at(mid);

        let left = Self::build_node(triangles, left_indices.to_vec(), max_per_leaf);
        let right = Self::build_node(triangles, right_indices.to_vec(), max_per_leaf);

        BvhNode::Internal {
            aabb,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Query signed distance to mesh at a point
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        match &self.root {
            None => f32::INFINITY,
            Some(root) => self.signed_distance_recursive(root, point, f32::INFINITY),
        }
    }

    /// Recursive signed distance query with early termination
    fn signed_distance_recursive(&self, node: &BvhNode, point: Vec3, mut best: f32) -> f32 {
        match node {
            BvhNode::Leaf { triangles, .. } => {
                for &idx in triangles {
                    let d = self.triangles[idx].signed_distance(point);
                    if d.abs() < best.abs() {
                        best = d;
                    }
                }
                best
            }
            BvhNode::Internal { aabb, left, right, .. } => {
                // Early termination: if AABB is farther than best, skip
                let aabb_dist = aabb.signed_distance(point);
                if aabb_dist > best.abs() {
                    return best;
                }

                // Query children, closest first
                let left_dist = left.aabb().signed_distance(point);
                let right_dist = right.aabb().signed_distance(point);

                if left_dist < right_dist {
                    best = self.signed_distance_recursive(left, point, best);
                    if right_dist <= best.abs() {
                        best = self.signed_distance_recursive(right, point, best);
                    }
                } else {
                    best = self.signed_distance_recursive(right, point, best);
                    if left_dist <= best.abs() {
                        best = self.signed_distance_recursive(left, point, best);
                    }
                }
                best
            }
        }
    }

    /// Query unsigned distance to mesh at a point
    #[inline]
    pub fn unsigned_distance(&self, point: Vec3) -> f32 {
        self.signed_distance(point).abs()
    }

    /// Batch query signed distances (parallel)
    pub fn signed_distance_batch(&self, points: &[Vec3]) -> Vec<f32> {
        points
            .par_iter()
            .map(|&p| self.signed_distance(p))
            .collect()
    }

    /// Batch query unsigned distances (parallel)
    pub fn unsigned_distance_batch(&self, points: &[Vec3]) -> Vec<f32> {
        points
            .par_iter()
            .map(|&p| self.unsigned_distance(p))
            .collect()
    }

    /// Get total triangle count
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Get mesh bounds
    pub fn bounds(&self) -> Option<Aabb> {
        self.root.as_ref().map(|r| *r.aabb())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_basic() {
        let mut aabb = Aabb::empty();
        aabb.expand_point(Vec3::new(0.0, 0.0, 0.0));
        aabb.expand_point(Vec3::new(1.0, 1.0, 1.0));

        assert_eq!(aabb.min, Vec3::ZERO);
        assert_eq!(aabb.max, Vec3::ONE);
        assert_eq!(aabb.center(), Vec3::splat(0.5));
    }

    #[test]
    fn test_aabb_signed_distance() {
        let aabb = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));

        // Inside
        let d_inside = aabb.signed_distance(Vec3::ZERO);
        assert!(d_inside < 0.0);

        // Outside
        let d_outside = aabb.signed_distance(Vec3::new(2.0, 0.0, 0.0));
        assert!(d_outside > 0.0);
        assert!((d_outside - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_triangle_distance() {
        let tri = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        );

        // Point on plane, inside triangle
        let d_inside = tri.signed_distance(Vec3::new(0.5, 0.3, 0.0));
        assert!(d_inside.abs() < 0.01);

        // Point above triangle (positive side of normal)
        let d_above = tri.signed_distance(Vec3::new(0.5, 0.3, 1.0));
        assert!(d_above > 0.0);
        assert!((d_above - 1.0).abs() < 0.01);

        // Point below triangle (negative side of normal)
        let d_below = tri.signed_distance(Vec3::new(0.5, 0.3, -1.0));
        assert!(d_below < 0.0);
    }

    #[test]
    fn test_bvh_build() {
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 1.0),
        ];
        let indices = vec![0, 1, 2, 0, 2, 3, 1, 2, 3, 0, 1, 3];

        let bvh = MeshBvh::build(&vertices, &indices, 2);
        assert_eq!(bvh.triangle_count(), 4);
        assert!(bvh.root.is_some());
    }

    #[test]
    fn test_bvh_query() {
        // Simple quad (2 triangles)
        let vertices = vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];

        let bvh = MeshBvh::build(&vertices, &indices, 4);

        // Point on the quad surface
        let d_surface = bvh.signed_distance(Vec3::new(0.0, 0.0, 0.0));
        assert!(d_surface.abs() < 0.01);

        // Point above
        let d_above = bvh.signed_distance(Vec3::new(0.0, 0.0, 1.0));
        assert!((d_above.abs() - 1.0).abs() < 0.01);
    }
}
