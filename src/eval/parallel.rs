//! Parallel SDF evaluation (Deep Fried Edition)
//!
//! Batch evaluation using rayon for CPU parallelism.
//!
//! # Deep Fried Optimizations
//! - **Division Exorcism**: Replaced grid coordinate calculation `i % res` with
//!   nested loops to avoid integer division latency.
//! - **Z-Slice Parallelism**: Processes Z-slices independently for cache efficiency.
//! - **Pre-calculated Step**: Step vector computed once, not per-point.
//!
//! Author: Moroya Sakamoto

use crate::eval::eval;
use crate::types::SdfNode;
use glam::Vec3;
use rayon::prelude::*;

/// Evaluate SDF at multiple points (single-threaded)
///
/// # Arguments
/// * `node` - The SDF tree
/// * `points` - Slice of points to evaluate
///
/// # Returns
/// Vector of distances
#[inline]
pub fn eval_batch(node: &SdfNode, points: &[Vec3]) -> Vec<f32> {
    points.iter().map(|&p| eval(node, p)).collect()
}

/// Evaluate SDF at multiple points (parallel)
///
/// Uses rayon for parallel iteration over points.
///
/// # Arguments
/// * `node` - The SDF tree
/// * `points` - Slice of points to evaluate
///
/// # Returns
/// Vector of distances
#[inline]
pub fn eval_batch_parallel(node: &SdfNode, points: &[Vec3]) -> Vec<f32> {
    points.par_iter().map(|&p| eval(node, p)).collect()
}

/// Evaluate SDF on a 3D grid (Deep Fried)
///
/// Uses Z-slice parallelism to avoid integer division for coordinate reconstruction.
///
/// # Arguments
/// * `node` - The SDF tree
/// * `min` - Minimum corner of the grid
/// * `max` - Maximum corner of the grid
/// * `resolution` - Number of samples along each axis
///
/// # Returns
/// Flattened grid of distances (in X-major order: x + y*res + z*res*res)
pub fn eval_grid(node: &SdfNode, min: Vec3, max: Vec3, resolution: usize) -> Vec<f32> {
    let size = max - min;
    // Pre-calculate step vector (avoid division in hot loop)
    let step = size / (resolution as f32 - 1.0);

    // Output buffer
    let total_size = resolution * resolution * resolution;
    let mut buffer = vec![0.0f32; total_size];

    // Parallelize by Z-slices (avoids integer division)
    let slice_size = resolution * resolution;

    buffer
        .par_chunks_mut(slice_size)
        .enumerate()
        .for_each(|(z, slice)| {
            let z_pos = min.z + z as f32 * step.z;

            // Iterate Y rows within the Z slice
            for y in 0..resolution {
                let y_pos = min.y + y as f32 * step.y;
                let row_offset = y * resolution;

                // Inner loop X (hot path - no division)
                for x in 0..resolution {
                    let x_pos = min.x + x as f32 * step.x;
                    let p = Vec3::new(x_pos, y_pos, z_pos);

                    // Direct slice access (bounds already checked by loop)
                    slice[row_offset + x] = eval(node, p);
                }
            }
        });

    buffer
}

/// Evaluate SDF on a 3D grid with normals (Deep Fried)
///
/// Returns both distances and surface normals for each point.
///
/// # Arguments
/// * `node` - The SDF tree
/// * `min` - Minimum corner of the grid
/// * `max` - Maximum corner of the grid
/// * `resolution` - Number of samples along each axis
/// * `epsilon` - Gradient estimation epsilon
///
/// # Returns
/// (distances, normals) tuple
pub fn eval_grid_with_normals(
    node: &SdfNode,
    min: Vec3,
    max: Vec3,
    resolution: usize,
    epsilon: f32,
) -> (Vec<f32>, Vec<Vec3>) {
    let size = max - min;
    let step = size / (resolution as f32 - 1.0);

    let total_size = resolution * resolution * resolution;
    let mut distances = vec![0.0f32; total_size];
    let mut normals = vec![Vec3::ZERO; total_size];

    let slice_size = resolution * resolution;

    // Zip parallel chunks of both buffers
    distances
        .par_chunks_mut(slice_size)
        .zip(normals.par_chunks_mut(slice_size))
        .enumerate()
        .for_each(|(z, (dist_slice, norm_slice))| {
            let z_pos = min.z + z as f32 * step.z;

            for y in 0..resolution {
                let y_pos = min.y + y as f32 * step.y;
                let row_offset = y * resolution;

                for x in 0..resolution {
                    let x_pos = min.x + x as f32 * step.x;
                    let p = Vec3::new(x_pos, y_pos, z_pos);

                    let idx = row_offset + x;
                    dist_slice[idx] = eval(node, p);
                    norm_slice[idx] = crate::eval::normal(node, p, epsilon);
                }
            }
        });

    (distances, normals)
}

/// Get grid index from 3D coordinates (Deep Fried)
#[inline(always)]
pub fn grid_index(x: usize, y: usize, z: usize, resolution: usize) -> usize {
    x + y * resolution + z * resolution * resolution
}

/// Get 3D coordinates from grid index (Deep Fried)
///
/// Note: Contains integer division - prefer nested loops when iterating.
#[inline(always)]
pub fn grid_coords(index: usize, resolution: usize) -> (usize, usize, usize) {
    let x = index % resolution;
    let y = (index / resolution) % resolution;
    let z = index / (resolution * resolution);
    (x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_batch() {
        let sphere = SdfNode::sphere(1.0);
        let points = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];

        let distances = eval_batch(&sphere, &points);

        assert_eq!(distances.len(), 3);
        assert!((distances[0] + 1.0).abs() < 0.0001);
        assert!(distances[1].abs() < 0.0001);
        assert!((distances[2] - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_eval_batch_parallel() {
        let sphere = SdfNode::sphere(1.0);
        let points: Vec<Vec3> = (0..1000)
            .map(|i| Vec3::new(i as f32 * 0.01, 0.0, 0.0))
            .collect();

        let distances = eval_batch_parallel(&sphere, &points);

        assert_eq!(distances.len(), 1000);
    }

    #[test]
    fn test_eval_grid() {
        let sphere = SdfNode::sphere(1.0);
        let min = Vec3::splat(-2.0);
        let max = Vec3::splat(2.0);
        let resolution = 10;

        let grid = eval_grid(&sphere, min, max, resolution);

        assert_eq!(grid.len(), resolution * resolution * resolution);

        // Center should be inside
        let center_idx = grid_index(resolution / 2, resolution / 2, resolution / 2, resolution);
        assert!(grid[center_idx] < 0.0);
    }

    #[test]
    fn test_grid_indexing() {
        let res = 10;

        for i in 0..res * res * res {
            let (x, y, z) = grid_coords(i, res);
            let back = grid_index(x, y, z, res);
            assert_eq!(i, back);
        }
    }

    #[test]
    fn test_eval_grid_with_normals() {
        let sphere = SdfNode::sphere(1.0);
        let min = Vec3::splat(-2.0);
        let max = Vec3::splat(2.0);
        let resolution = 5;

        let (distances, normals) = eval_grid_with_normals(&sphere, min, max, resolution, 0.001);

        assert_eq!(distances.len(), normals.len());
        assert_eq!(distances.len(), resolution * resolution * resolution);
    }
}
