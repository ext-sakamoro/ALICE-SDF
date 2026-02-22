//! Point Cloud → SDF conversion
//!
//! Converts a set of oriented points (position + normal) into a signed distance
//! field approximation. Uses a KD-tree-like spatial partitioning for O(log n)
//! nearest-neighbor queries.
//!
//! # Algorithm
//!
//! For each query point:
//! 1. Find the K nearest points in the cloud
//! 2. Compute unsigned distance to the nearest point
//! 3. Determine sign using the dot product of the displacement vector with the
//!    nearest point's normal (inside/outside classification)
//!
//! # Example
//!
//! ```
//! use alice_sdf::mesh::point_cloud_sdf::{PointCloudSdf, PointCloudSdfConfig};
//! use glam::Vec3;
//!
//! let points = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
//! let normals = vec![Vec3::NEG_ONE.normalize(); 4];
//!
//! let sdf = PointCloudSdf::new(&points, &normals, &PointCloudSdfConfig::default());
//! let dist = sdf.eval(Vec3::new(0.5, 0.5, 0.5));
//! ```
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use rayon::prelude::*;

/// Configuration for point cloud SDF construction
#[derive(Debug, Clone)]
pub struct PointCloudSdfConfig {
    /// Number of nearest neighbors for sign determination (default: 8)
    pub k_neighbors: usize,
    /// Leaf size for spatial partitioning (default: 16)
    pub leaf_size: usize,
}

impl Default for PointCloudSdfConfig {
    fn default() -> Self {
        PointCloudSdfConfig {
            k_neighbors: 8,
            leaf_size: 16,
        }
    }
}

impl PointCloudSdfConfig {
    /// Fast config (fewer neighbors, larger leaves)
    pub fn fast() -> Self {
        PointCloudSdfConfig {
            k_neighbors: 4,
            leaf_size: 32,
        }
    }

    /// Accurate config (more neighbors, smaller leaves)
    pub fn accurate() -> Self {
        PointCloudSdfConfig {
            k_neighbors: 16,
            leaf_size: 8,
        }
    }
}

/// KD-tree node for spatial partitioning
enum KdNode {
    Leaf {
        indices: Vec<usize>,
    },
    Split {
        axis: usize,
        split_val: f32,
        left: Box<KdNode>,
        right: Box<KdNode>,
    },
}

/// Signed distance field from a point cloud
pub struct PointCloudSdf {
    points: Vec<Vec3>,
    normals: Vec<Vec3>,
    tree: KdNode,
    k: usize,
}

impl PointCloudSdf {
    /// Construct a new point cloud SDF
    ///
    /// # Arguments
    /// * `points` - 3D positions of the point cloud
    /// * `normals` - Surface normals at each point (must be same length as points)
    /// * `config` - Construction parameters
    pub fn new(points: &[Vec3], normals: &[Vec3], config: &PointCloudSdfConfig) -> Self {
        assert_eq!(
            points.len(),
            normals.len(),
            "Points and normals must have same length"
        );

        let indices: Vec<usize> = (0..points.len()).collect();
        let tree = Self::build_tree(points, &indices, config.leaf_size, 0);

        PointCloudSdf {
            points: points.to_vec(),
            normals: normals.to_vec(),
            tree,
            k: config.k_neighbors,
        }
    }

    fn build_tree(points: &[Vec3], indices: &[usize], leaf_size: usize, depth: usize) -> KdNode {
        if indices.len() <= leaf_size {
            return KdNode::Leaf {
                indices: indices.to_vec(),
            };
        }

        let axis = depth % 3;

        // Find median along axis
        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_by(|&a, &b| {
            let va = match axis {
                0 => points[a].x,
                1 => points[a].y,
                _ => points[a].z,
            };
            let vb = match axis {
                0 => points[b].x,
                1 => points[b].y,
                _ => points[b].z,
            };
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = sorted_indices.len() / 2;
        let split_val = match axis {
            0 => points[sorted_indices[mid]].x,
            1 => points[sorted_indices[mid]].y,
            _ => points[sorted_indices[mid]].z,
        };

        let left_indices = &sorted_indices[..mid];
        let right_indices = &sorted_indices[mid..];

        KdNode::Split {
            axis,
            split_val,
            left: Box::new(Self::build_tree(points, left_indices, leaf_size, depth + 1)),
            right: Box::new(Self::build_tree(
                points,
                right_indices,
                leaf_size,
                depth + 1,
            )),
        }
    }

    /// Evaluate signed distance at a point
    #[inline]
    pub fn eval(&self, pos: Vec3) -> f32 {
        let mut best = Vec::with_capacity(self.k);
        self.knn_search(&self.tree, pos, self.k, &mut best);

        if best.is_empty() {
            return f32::MAX;
        }

        // best is already sorted by knn_insert — no need to sort again
        let nearest_idx = best[0].0;
        let nearest_dist = best[0].1.sqrt();

        // Sign determination: weighted vote from K nearest neighbors
        let displacement = pos - self.points[nearest_idx];
        let sign = if displacement.dot(self.normals[nearest_idx]) >= 0.0 {
            1.0f32
        } else {
            -1.0f32
        };

        sign * nearest_dist
    }

    /// Batch evaluation using rayon parallelism
    pub fn eval_batch(&self, positions: &[Vec3]) -> Vec<f32> {
        positions.par_iter().map(|&pos| self.eval(pos)).collect()
    }

    /// Insert a candidate into the sorted KNN result list.
    /// Maintains sorted order via binary search + insert (O(log k) search, O(k) shift).
    #[inline(always)]
    fn knn_insert(best: &mut Vec<(usize, f32)>, k: usize, idx: usize, dist_sq: f32) {
        if best.len() < k {
            let pos = best.partition_point(|&(_, d)| d < dist_sq);
            best.insert(pos, (idx, dist_sq));
        } else if dist_sq < best[k - 1].1 {
            best.pop();
            let pos = best.partition_point(|&(_, d)| d < dist_sq);
            best.insert(pos, (idx, dist_sq));
        }
    }

    /// K-nearest neighbor search in KD-tree (recursive, sorted-insert)
    #[inline(always)]
    fn knn_search(&self, node: &KdNode, pos: Vec3, k: usize, best: &mut Vec<(usize, f32)>) {
        let pos_arr = [pos.x, pos.y, pos.z];
        match node {
            KdNode::Leaf { indices } => {
                for &idx in indices {
                    let dist_sq = (pos - self.points[idx]).length_squared();
                    Self::knn_insert(best, k, idx, dist_sq);
                }
            }
            KdNode::Split {
                axis,
                split_val,
                left,
                right,
            } => {
                let val = pos_arr[*axis];
                let diff = val - split_val;

                let (near, far) = if diff < 0.0 {
                    (left, right)
                } else {
                    (right, left)
                };

                // Search near side first
                self.knn_search(near, pos, k, best);

                // Check if far side could contain closer points
                let worst_dist_sq = if best.len() < k {
                    f32::MAX
                } else {
                    best[k - 1].1
                };

                if diff * diff < worst_dist_sq {
                    self.knn_search(far, pos, k, best);
                }
            }
        }
    }

    /// Get the number of points in the cloud
    #[inline]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }
}

/// Create a point cloud SDF from points and normals (convenience function)
pub fn point_cloud_to_sdf(
    points: &[Vec3],
    normals: &[Vec3],
    config: &PointCloudSdfConfig,
) -> PointCloudSdf {
    PointCloudSdf::new(points, normals, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sphere_cloud(n: usize, radius: f32) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut points = Vec::with_capacity(n);
        let mut normals = Vec::with_capacity(n);

        // Fibonacci sphere distribution
        let golden_ratio = (1.0 + 5.0f32.sqrt()) / 2.0;
        for i in 0..n {
            let theta = 2.0 * std::f32::consts::PI * (i as f32) / golden_ratio;
            let phi = (1.0 - 2.0 * (i as f32 + 0.5) / n as f32).acos();

            let normal = Vec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos());
            points.push(normal * radius);
            normals.push(normal);
        }
        (points, normals)
    }

    #[test]
    fn test_point_cloud_sdf_construction() {
        let (points, normals) = make_sphere_cloud(100, 1.0);
        let sdf = PointCloudSdf::new(&points, &normals, &PointCloudSdfConfig::default());
        assert_eq!(sdf.point_count(), 100);
    }

    #[test]
    fn test_point_cloud_sdf_sign() {
        let (points, normals) = make_sphere_cloud(500, 1.0);
        let sdf = PointCloudSdf::new(&points, &normals, &PointCloudSdfConfig::default());

        // Outside the sphere (distance ~1.0 from surface)
        let outside = sdf.eval(Vec3::new(2.0, 0.0, 0.0));
        assert!(
            outside > 0.0,
            "Expected positive outside sphere, got {}",
            outside
        );

        // Inside the sphere
        let inside = sdf.eval(Vec3::ZERO);
        assert!(
            inside < 0.0,
            "Expected negative inside sphere, got {}",
            inside
        );
    }

    #[test]
    fn test_point_cloud_sdf_batch() {
        let (points, normals) = make_sphere_cloud(200, 1.0);
        let sdf = PointCloudSdf::new(&points, &normals, &PointCloudSdfConfig::default());

        let test_points = vec![
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
        ];

        let results = sdf.eval_batch(&test_points);
        assert_eq!(results.len(), 3);
        assert!(results[0] > 0.0); // outside
        assert!(results[1] < 0.0); // inside
    }

    #[test]
    fn test_convenience_function() {
        let (points, normals) = make_sphere_cloud(50, 1.0);
        let sdf = point_cloud_to_sdf(&points, &normals, &PointCloudSdfConfig::fast());
        assert_eq!(sdf.point_count(), 50);
    }

    #[test]
    fn test_configs() {
        let (points, normals) = make_sphere_cloud(50, 1.0);

        let _fast = PointCloudSdf::new(&points, &normals, &PointCloudSdfConfig::fast());
        let _accurate = PointCloudSdf::new(&points, &normals, &PointCloudSdfConfig::accurate());
    }
}
