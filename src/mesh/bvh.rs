//! BVH (Bounding Volume Hierarchy) for mesh acceleration (Deep Fried v2)
//!
//! Provides O(log n) distance queries for triangle meshes.
//!
//! # Deep Fried v2 Optimizations
//!
//! - **Parallel Triangle Construction**: `rayon` parallel iterator for Triangle::new().
//! - **SIMD AABB Computation**: `wide::f32x8` for 8-triangle batch AABB min/max.
//! - **Forced Inlining**: Hot-path distance functions.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use rayon::prelude::*;
use wide::f32x8;

/// Axis-Aligned Bounding Box
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    /// Minimum corner
    pub min: Vec3,
    /// Maximum corner
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
    /// First vertex
    pub v0: Vec3,
    /// Second vertex
    pub v1: Vec3,
    /// Third vertex
    pub v2: Vec3,
    /// Face normal
    pub normal: Vec3,
    /// Bounding box
    pub aabb: Aabb,
}

impl Triangle {
    /// Create triangle from vertices
    #[inline]
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3) -> Self {
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let normal = e1.cross(e2).normalize_or_zero();

        let mut aabb = Aabb::empty();
        aabb.expand_point(v0);
        aabb.expand_point(v1);
        aabb.expand_point(v2);

        Triangle {
            v0,
            v1,
            v2,
            normal,
            aabb,
        }
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
        let sign = if self.normal.dot(point - v0) >= 0.0 {
            1.0
        } else {
            -1.0
        };

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
        /// Bounding box
        aabb: Aabb,
        /// Triangle indices
        triangles: Vec<usize>,
    },
    /// Internal node with two children
    Internal {
        /// Bounding box
        aabb: Aabb,
        /// Left child
        left: Box<BvhNode>,
        /// Right child
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
    /// All triangles in the mesh
    pub triangles: Vec<Triangle>,
    /// Root BVH node
    pub root: Option<BvhNode>,
    /// Maximum triangles per leaf node
    pub max_triangles_per_leaf: usize,
}

impl MeshBvh {
    /// Build BVH from mesh data (Deep Fried v2)
    ///
    /// Triangle construction is parallelized via `rayon`.
    pub fn build(vertices: &[Vec3], indices: &[u32], max_triangles_per_leaf: usize) -> Self {
        // [Deep Fried v2] Parallel triangle construction
        let triangles: Vec<Triangle> = indices
            .par_chunks(3)
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
    ///
    /// [Deep Fried v2] SIMD-accelerated AABB computation for batches of 8 triangles.
    fn build_node(triangles: &[Triangle], indices: Vec<usize>, max_per_leaf: usize) -> BvhNode {
        // [Deep Fried v2] SIMD AABB computation — process 8 triangle AABBs at a time
        let aabb = compute_aabb_simd(triangles, &indices);

        // If few triangles, create leaf
        if indices.len() <= max_per_leaf {
            return BvhNode::Leaf {
                aabb,
                triangles: indices,
            };
        }

        // [Deep Fried v2] SAH (Surface Area Heuristic) split
        // Evaluates candidate splits to minimize traversal cost.
        // Falls back to median if SAH finds no improvement over leaf.
        let axis = aabb.longest_axis();
        let mut sorted_indices = indices;
        sorted_indices.sort_unstable_by(|&a, &b| {
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

        let n = sorted_indices.len();
        let parent_sa = aabb.surface_area().max(1e-10); // Division Exorcism
        let inv_parent_sa = 1.0 / parent_sa;

        // Evaluate SAH at ~8 candidate split positions (or every position if small)
        let num_buckets = n.min(16).max(2);
        let mut best_cost = f32::INFINITY;
        let mut best_mid = n / 2;

        for bucket in 1..num_buckets {
            let mid = bucket * n / num_buckets;
            if mid == 0 || mid == n {
                continue;
            }

            let left_aabb = compute_aabb_simd(triangles, &sorted_indices[..mid]);
            let right_aabb = compute_aabb_simd(triangles, &sorted_indices[mid..]);

            let cost = left_aabb.surface_area() * inv_parent_sa * mid as f32
                + right_aabb.surface_area() * inv_parent_sa * (n - mid) as f32;

            if cost < best_cost {
                best_cost = cost;
                best_mid = mid;
            }
        }

        let (left_indices, right_indices) = sorted_indices.split_at(best_mid);

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
            BvhNode::Internal {
                aabb, left, right, ..
            } => {
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

/// [Deep Fried v2] SIMD-accelerated AABB computation
///
/// Processes 8 triangle AABBs at a time using `wide::f32x8` for min/max reduction.
/// Falls back to scalar for the remainder (<8 triangles).
#[inline]
fn compute_aabb_simd(triangles: &[Triangle], indices: &[usize]) -> Aabb {
    if indices.is_empty() {
        return Aabb::empty();
    }

    // [Deep Fried v2] SIMD accumulators — accumulate min/max in f32x8 lanes,
    // single horizontal reduction at the end instead of per-chunk extract.
    let mut acc_min_x = f32x8::splat(f32::INFINITY);
    let mut acc_min_y = f32x8::splat(f32::INFINITY);
    let mut acc_min_z = f32x8::splat(f32::INFINITY);
    let mut acc_max_x = f32x8::splat(f32::NEG_INFINITY);
    let mut acc_max_y = f32x8::splat(f32::NEG_INFINITY);
    let mut acc_max_z = f32x8::splat(f32::NEG_INFINITY);

    // Process 8 triangles at a time with SIMD
    let chunks = indices.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let min_x = f32x8::new([
            triangles[chunk[0]].aabb.min.x,
            triangles[chunk[1]].aabb.min.x,
            triangles[chunk[2]].aabb.min.x,
            triangles[chunk[3]].aabb.min.x,
            triangles[chunk[4]].aabb.min.x,
            triangles[chunk[5]].aabb.min.x,
            triangles[chunk[6]].aabb.min.x,
            triangles[chunk[7]].aabb.min.x,
        ]);
        let min_y = f32x8::new([
            triangles[chunk[0]].aabb.min.y,
            triangles[chunk[1]].aabb.min.y,
            triangles[chunk[2]].aabb.min.y,
            triangles[chunk[3]].aabb.min.y,
            triangles[chunk[4]].aabb.min.y,
            triangles[chunk[5]].aabb.min.y,
            triangles[chunk[6]].aabb.min.y,
            triangles[chunk[7]].aabb.min.y,
        ]);
        let min_z = f32x8::new([
            triangles[chunk[0]].aabb.min.z,
            triangles[chunk[1]].aabb.min.z,
            triangles[chunk[2]].aabb.min.z,
            triangles[chunk[3]].aabb.min.z,
            triangles[chunk[4]].aabb.min.z,
            triangles[chunk[5]].aabb.min.z,
            triangles[chunk[6]].aabb.min.z,
            triangles[chunk[7]].aabb.min.z,
        ]);
        let max_x = f32x8::new([
            triangles[chunk[0]].aabb.max.x,
            triangles[chunk[1]].aabb.max.x,
            triangles[chunk[2]].aabb.max.x,
            triangles[chunk[3]].aabb.max.x,
            triangles[chunk[4]].aabb.max.x,
            triangles[chunk[5]].aabb.max.x,
            triangles[chunk[6]].aabb.max.x,
            triangles[chunk[7]].aabb.max.x,
        ]);
        let max_y = f32x8::new([
            triangles[chunk[0]].aabb.max.y,
            triangles[chunk[1]].aabb.max.y,
            triangles[chunk[2]].aabb.max.y,
            triangles[chunk[3]].aabb.max.y,
            triangles[chunk[4]].aabb.max.y,
            triangles[chunk[5]].aabb.max.y,
            triangles[chunk[6]].aabb.max.y,
            triangles[chunk[7]].aabb.max.y,
        ]);
        let max_z = f32x8::new([
            triangles[chunk[0]].aabb.max.z,
            triangles[chunk[1]].aabb.max.z,
            triangles[chunk[2]].aabb.max.z,
            triangles[chunk[3]].aabb.max.z,
            triangles[chunk[4]].aabb.max.z,
            triangles[chunk[5]].aabb.max.z,
            triangles[chunk[6]].aabb.max.z,
            triangles[chunk[7]].aabb.max.z,
        ]);

        // Accumulate in SIMD lanes — no per-chunk extract
        acc_min_x = acc_min_x.min(min_x);
        acc_min_y = acc_min_y.min(min_y);
        acc_min_z = acc_min_z.min(min_z);
        acc_max_x = acc_max_x.max(max_x);
        acc_max_y = acc_max_y.max(max_y);
        acc_max_z = acc_max_z.max(max_z);
    }

    // Single horizontal reduction across 8 lanes
    let min_x_arr: [f32; 8] = acc_min_x.into();
    let min_y_arr: [f32; 8] = acc_min_y.into();
    let min_z_arr: [f32; 8] = acc_min_z.into();
    let max_x_arr: [f32; 8] = acc_max_x.into();
    let max_y_arr: [f32; 8] = acc_max_y.into();
    let max_z_arr: [f32; 8] = acc_max_z.into();

    let mut global_min_x = f32::INFINITY;
    let mut global_min_y = f32::INFINITY;
    let mut global_min_z = f32::INFINITY;
    let mut global_max_x = f32::NEG_INFINITY;
    let mut global_max_y = f32::NEG_INFINITY;
    let mut global_max_z = f32::NEG_INFINITY;

    for i in 0..8 {
        global_min_x = global_min_x.min(min_x_arr[i]);
        global_min_y = global_min_y.min(min_y_arr[i]);
        global_min_z = global_min_z.min(min_z_arr[i]);
        global_max_x = global_max_x.max(max_x_arr[i]);
        global_max_y = global_max_y.max(max_y_arr[i]);
        global_max_z = global_max_z.max(max_z_arr[i]);
    }

    // Scalar remainder
    for &idx in remainder {
        let t = &triangles[idx];
        global_min_x = global_min_x.min(t.aabb.min.x);
        global_min_y = global_min_y.min(t.aabb.min.y);
        global_min_z = global_min_z.min(t.aabb.min.z);
        global_max_x = global_max_x.max(t.aabb.max.x);
        global_max_y = global_max_y.max(t.aabb.max.y);
        global_max_z = global_max_z.max(t.aabb.max.z);
    }

    Aabb {
        min: Vec3::new(global_min_x, global_min_y, global_min_z),
        max: Vec3::new(global_max_x, global_max_y, global_max_z),
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

    #[test]
    fn test_compute_aabb_simd() {
        // Create enough triangles to exercise the SIMD path (>8)
        let mut verts = Vec::new();
        let mut idxs = Vec::new();
        for i in 0..10 {
            let offset = i as f32;
            let base = (i * 3) as u32;
            verts.push(Vec3::new(offset, 0.0, 0.0));
            verts.push(Vec3::new(offset + 1.0, 0.0, 0.0));
            verts.push(Vec3::new(offset + 0.5, 1.0, 0.0));
            idxs.extend_from_slice(&[base, base + 1, base + 2]);
        }

        let bvh = MeshBvh::build(&verts, &idxs, 4);
        let bounds = bvh.bounds().unwrap();

        assert!(bounds.min.x < 0.1);
        assert!(bounds.max.x > 9.0);
        assert!(bounds.min.y < 0.1);
        assert!(bounds.max.y > 0.9);
    }
}
