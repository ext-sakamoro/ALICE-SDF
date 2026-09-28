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
    pub const fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    /// Create AABB from min/max
    #[inline]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Expand AABB to include a point
    #[inline]
    pub fn expand_point(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Expand AABB to include another AABB
    #[inline]
    pub fn expand_aabb(&mut self, other: &Self) {
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
        2.0 * d.z.mul_add(d.x, d.x.mul_add(d.y, d.y * d.z))
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

        Self {
            v0,
            v1,
            v2,
            normal,
            aabb,
        }
    }

    /// Closest point on the triangle (including its edges and vertices).
    ///
    /// Voronoi-region form (Ericson, *Real-Time Collision Detection* §5.1.5):
    /// exact for every query point, including the edge and vertex regions where
    /// a plane projection would answer a point that is not on the triangle.
    /// Uses only add / sub / mul / div / compare, so the result is bit-identical
    /// across targets.
    #[inline]
    pub fn closest_point(&self, point: Vec3) -> Vec3 {
        let (a, b, c) = (self.v0, self.v1, self.v2);
        let ab = b - a;
        let ac = c - a;

        // vertex region A
        let ap = point - a;
        let d1 = ab.dot(ap);
        let d2 = ac.dot(ap);
        if d1 <= 0.0 && d2 <= 0.0 {
            return a;
        }

        // vertex region B
        let bp = point - b;
        let d3 = ab.dot(bp);
        let d4 = ac.dot(bp);
        if d3 >= 0.0 && d4 <= d3 {
            return b;
        }

        // edge region AB
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let denom = d1 - d3;
            if denom != 0.0 {
                return a + ab * (d1 / denom);
            }
            return a;
        }

        // vertex region C
        let cp = point - c;
        let d5 = ab.dot(cp);
        let d6 = ac.dot(cp);
        if d6 >= 0.0 && d5 <= d6 {
            return c;
        }

        // edge region AC
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let denom = d2 - d6;
            if denom != 0.0 {
                return a + ac * (d2 / denom);
            }
            return a;
        }

        // edge region BC
        let va = d3 * d6 - d5 * d4;
        let bc_num = d4 - d3;
        let bc_den = d5 - d6;
        if va <= 0.0 && bc_num >= 0.0 && bc_den >= 0.0 {
            let denom = bc_num + bc_den;
            if denom != 0.0 {
                return b + (c - b) * (bc_num / denom);
            }
            return b;
        }

        // face region — barycentric interior
        let denom = va + vb + vc;
        if denom == 0.0 {
            // Degenerate (zero-area) triangle: the Voronoi regions above do not
            // partition the plane, so answer the best of the three segments.
            return closest_on_segment(point, a, b)
                .into_iter()
                .chain(closest_on_segment(point, b, c))
                .chain(closest_on_segment(point, c, a))
                .fold((f32::INFINITY, a), |(best, q), cand| {
                    let d = (point - cand).length_squared();
                    if d < best {
                        (d, cand)
                    } else {
                        (best, q)
                    }
                })
                .1;
        }
        let inv = 1.0 / denom;
        a + ab * (vb * inv) + ac * (vc * inv)
    }

    /// Signed distance to triangle.
    ///
    /// The magnitude is the exact distance to the triangle; the sign comes from
    /// the face normal, so it is a function of the triangle's **winding**. For a
    /// mesh, prefer [`crate::mesh::ExteriorField`], whose sign does not depend
    /// on winding at all.
    #[inline]
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        let unsigned_dist = self.unsigned_distance(point);

        // Positive = outside (same side as normal), Negative = inside
        let sign = if self.normal.dot(point - self.v0) >= 0.0 {
            1.0
        } else {
            -1.0
        };

        unsigned_dist * sign
    }

    /// Unsigned distance to triangle (always positive)
    #[inline]
    pub fn unsigned_distance(&self, point: Vec3) -> f32 {
        (point - self.closest_point(point)).length()
    }
}

/// Closest point on the segment `a`..`b`, as a one-element iterator so the
/// degenerate branch above can chain the three edges without allocating.
#[inline]
fn closest_on_segment(point: Vec3, a: Vec3, b: Vec3) -> [Vec3; 1] {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq == 0.0 {
        return [a];
    }
    [a + ab * (ab.dot(point - a) / len_sq).clamp(0.0, 1.0)]
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
        left: Box<Self>,
        /// Right child
        right: Box<Self>,
    },
}

impl BvhNode {
    /// Get AABB of this node
    #[inline]
    pub const fn aabb(&self) -> &Aabb {
        match self {
            Self::Leaf { aabb, .. } => aabb,
            Self::Internal { aabb, .. } => aabb,
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
            return Self {
                triangles,
                root: None,
                max_triangles_per_leaf,
            };
        }

        let indices: Vec<usize> = (0..triangles.len()).collect();
        let root = Self::build_node(&triangles, indices, max_triangles_per_leaf);

        Self {
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
        let num_buckets = n.clamp(2, 16);
        let mut best_cost = f32::INFINITY;
        let mut best_mid = n / 2;

        for bucket in 1..num_buckets {
            let mid = bucket * n / num_buckets;
            if mid == 0 || mid == n {
                continue;
            }

            let left_aabb = compute_aabb_simd(triangles, &sorted_indices[..mid]);
            let right_aabb = compute_aabb_simd(triangles, &sorted_indices[mid..]);

            let cost = (left_aabb.surface_area() * inv_parent_sa).mul_add(
                mid as f32,
                right_aabb.surface_area() * inv_parent_sa * (n - mid) as f32,
            );

            if cost < best_cost {
                best_cost = cost;
                best_mid = mid;
            }
        }

        let right_indices = sorted_indices.split_off(best_mid);
        let left_indices = sorted_indices;

        let left = Self::build_node(triangles, left_indices, max_per_leaf);
        let right = Self::build_node(triangles, right_indices, max_per_leaf);

        BvhNode::Internal {
            aabb,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Query signed distance to mesh at a point
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        self.root.as_ref().map_or(f32::INFINITY, |root| {
            self.signed_distance_recursive(root, point, f32::INFINITY)
        })
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

    /// Query unsigned distance to mesh at a point.
    ///
    /// Independent of triangle winding, and therefore the part of the mesh
    /// field that is always well defined. [`Self::signed_distance`] only adds a
    /// sign to this value.
    #[inline]
    pub fn unsigned_distance(&self, point: Vec3) -> f32 {
        self.closest_point_and_dist_sq(point)
            .map_or(f32::INFINITY, |(_, d_sq)| d_sq.sqrt())
    }

    /// Closest point on the mesh surface, or `None` for an empty mesh.
    #[inline]
    pub fn closest_point(&self, point: Vec3) -> Option<Vec3> {
        self.closest_point_and_dist_sq(point).map(|(q, _)| q)
    }

    fn closest_point_and_dist_sq(&self, point: Vec3) -> Option<(Vec3, f32)> {
        let root = self.root.as_ref()?;
        let mut best = (Vec3::ZERO, f32::INFINITY);
        Self::closest_recursive(&self.triangles, root, point, &mut best);
        if best.1.is_finite() {
            Some(best)
        } else {
            None
        }
    }

    /// Nearest-child-first traversal with an exact lower bound (the AABB
    /// distance) as the pruning test.
    fn closest_recursive(
        triangles: &[Triangle],
        node: &BvhNode,
        point: Vec3,
        best: &mut (Vec3, f32),
    ) {
        match node {
            BvhNode::Leaf {
                triangles: idxs, ..
            } => {
                for &idx in idxs {
                    let q = triangles[idx].closest_point(point);
                    let d_sq = (point - q).length_squared();
                    if d_sq < best.1 {
                        *best = (q, d_sq);
                    }
                }
            }
            BvhNode::Internal { left, right, .. } => {
                // `Aabb::signed_distance` is negative inside, so clamping at 0
                // turns it into a lower bound on the distance to any triangle
                // held below this node.
                let bound = |n: &BvhNode| {
                    let d = n.aabb().signed_distance(point).max(0.0);
                    d * d
                };
                let (l_bound, r_bound) = (bound(left), bound(right));
                let (first, first_bound, second, second_bound) = if l_bound <= r_bound {
                    (left, l_bound, right, r_bound)
                } else {
                    (right, r_bound, left, l_bound)
                };
                if first_bound < best.1 {
                    Self::closest_recursive(triangles, first, point, best);
                }
                if second_bound < best.1 {
                    Self::closest_recursive(triangles, second, point, best);
                }
            }
        }
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
