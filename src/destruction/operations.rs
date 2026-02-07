//! Destruction Operations: Carve, Explode, Batch Carve (Deep Fried Edition)
//!
//! Implements CSG-style destructive operations on MutableVoxelGrid.
//! `max(old_dist, -carve_dist)` for subtractive carving.
//!
//! Author: Moroya Sakamoto

use glam::{Vec3, Quat};

use super::MutableVoxelGrid;
use crate::types::SdfNode;
use crate::eval::eval;

/// Shape used for carving operations
#[derive(Clone, Debug)]
pub enum CarveShape {
    /// Spherical carve
    Sphere {
        /// Center position in world space
        center: Vec3,
        /// Carve radius
        radius: f32,
    },
    /// Box-shaped carve
    Box {
        /// Center position in world space
        center: Vec3,
        /// Half-extents of the box
        half_extents: Vec3,
        /// Rotation of the box
        rotation: Quat,
    },
    /// Arbitrary SDF carve
    Sdf(SdfNode),
}

impl CarveShape {
    /// Evaluate the carve shape's signed distance at a point
    fn eval(&self, point: Vec3) -> f32 {
        match self {
            CarveShape::Sphere { center, radius } => {
                (point - *center).length() - radius
            }
            CarveShape::Box { center, half_extents, rotation } => {
                let local = rotation.inverse() * (point - *center);
                let q = local.abs() - *half_extents;
                q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
            }
            CarveShape::Sdf(node) => {
                eval(node, point)
            }
        }
    }

    /// Compute the world-space AABB of this shape (for grid bounds check)
    fn aabb(&self) -> (Vec3, Vec3) {
        match self {
            CarveShape::Sphere { center, radius } => {
                (
                    *center - Vec3::splat(*radius),
                    *center + Vec3::splat(*radius),
                )
            }
            CarveShape::Box { center, half_extents, rotation } => {
                // Conservative AABB of rotated box
                let axes = [
                    *rotation * Vec3::new(half_extents.x, 0.0, 0.0),
                    *rotation * Vec3::new(0.0, half_extents.y, 0.0),
                    *rotation * Vec3::new(0.0, 0.0, half_extents.z),
                ];
                let extent = Vec3::new(
                    axes[0].x.abs() + axes[1].x.abs() + axes[2].x.abs(),
                    axes[0].y.abs() + axes[1].y.abs() + axes[2].y.abs(),
                    axes[0].z.abs() + axes[1].z.abs() + axes[2].z.abs(),
                );
                (*center - extent, *center + extent)
            }
            CarveShape::Sdf(_node) => {
                // Conservative: full grid bounds
                (Vec3::splat(f32::NEG_INFINITY), Vec3::splat(f32::INFINITY))
            }
        }
    }
}

/// Result of a destruction operation
#[derive(Debug)]
pub struct DestructionResult {
    /// Chunk coordinates that were modified
    pub dirty_chunks: Vec<[u32; 3]>,
    /// Approximate volume of removed material (in world units cubed)
    pub removed_volume: f32,
    /// Number of voxels that were modified
    pub modified_voxels: u32,
}

/// Carve a shape from the voxel grid (subtraction)
///
/// Uses `max(old_distance, -carve_distance)` to remove material.
/// Only voxels within the shape's AABB are evaluated for efficiency.
pub fn carve(grid: &mut MutableVoxelGrid, shape: &CarveShape) -> DestructionResult {
    let (shape_min, shape_max) = shape.aabb();
    let vs = grid.voxel_size();
    let voxel_volume = vs.x * vs.y * vs.z;

    // Clamp shape AABB to grid bounds, then compute grid coordinate ranges
    let clamped_min = shape_min.max(grid.bounds_min);
    let clamped_max = shape_max.min(grid.bounds_max);

    // Check for valid overlap
    if clamped_min.x >= clamped_max.x || clamped_min.y >= clamped_max.y || clamped_min.z >= clamped_max.z {
        return DestructionResult {
            dirty_chunks: Vec::new(),
            removed_volume: 0.0,
            modified_voxels: 0,
        };
    }

    // Compute grid ranges directly (avoid world_to_grid edge issues)
    let size = grid.bounds_max - grid.bounds_min;
    let inv_size = Vec3::new(1.0 / size.x, 1.0 / size.y, 1.0 / size.z);
    let start = [
        ((clamped_min.x - grid.bounds_min.x) * inv_size.x * grid.resolution[0] as f32).floor().max(0.0) as u32,
        ((clamped_min.y - grid.bounds_min.y) * inv_size.y * grid.resolution[1] as f32).floor().max(0.0) as u32,
        ((clamped_min.z - grid.bounds_min.z) * inv_size.z * grid.resolution[2] as f32).floor().max(0.0) as u32,
    ];
    let end = [
        ((clamped_max.x - grid.bounds_min.x) * inv_size.x * grid.resolution[0] as f32).ceil().min(grid.resolution[0] as f32) as u32,
        ((clamped_max.y - grid.bounds_min.y) * inv_size.y * grid.resolution[1] as f32).ceil().min(grid.resolution[1] as f32) as u32,
        ((clamped_max.z - grid.bounds_min.z) * inv_size.z * grid.resolution[2] as f32).ceil().min(grid.resolution[2] as f32) as u32,
    ];

    let mut removed_volume = 0.0f32;
    let mut modified_voxels = 0u32;

    for z in start[2]..end[2] {
        for y in start[1]..end[1] {
            for x in start[0]..end[0] {
                let world_pos = grid.grid_to_world(x, y, z);
                let carve_d = shape.eval(world_pos);

                let idx = grid.voxel_index(x, y, z);
                let old_d = grid.distances[idx];

                // Subtraction: max(old, -carve)
                let new_d = old_d.max(-carve_d);

                if (new_d - old_d).abs() > 1e-6 {
                    // Track volume removed (was inside, now outside or less inside)
                    if old_d < 0.0 && new_d > old_d {
                        removed_volume += voxel_volume * (new_d - old_d).min(1.0);
                    }

                    grid.distances[idx] = new_d;
                    grid.mark_dirty(x, y, z);
                    modified_voxels += 1;
                }
            }
        }
    }

    let dirty_chunks = grid.dirty_chunks();

    DestructionResult {
        dirty_chunks,
        removed_volume,
        modified_voxels,
    }
}

/// Carve multiple shapes in a batch (more efficient than individual carves)
pub fn carve_batch(grid: &mut MutableVoxelGrid, shapes: &[CarveShape]) -> DestructionResult {
    let mut total_removed = 0.0f32;
    let mut total_modified = 0u32;

    grid.clear_dirty();

    for shape in shapes {
        let result = carve(grid, shape);
        total_removed += result.removed_volume;
        total_modified += result.modified_voxels;
    }

    let dirty_chunks = grid.dirty_chunks();

    DestructionResult {
        dirty_chunks,
        removed_volume: total_removed,
        modified_voxels: total_modified,
    }
}

/// Explosion carve: multiple spheres radiating from a center point
///
/// Creates a rough, natural-looking crater effect.
pub fn explode(
    grid: &mut MutableVoxelGrid,
    center: Vec3,
    radius: f32,
    fragment_count: u32,
    seed: u64,
) -> DestructionResult {
    let mut shapes = Vec::with_capacity(1 + fragment_count as usize);

    // Main crater
    shapes.push(CarveShape::Sphere { center, radius });

    // Fragment craters around the main one
    let mut rng_state = seed;
    for _ in 0..fragment_count {
        // Simple LCG pseudo-random
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let rx = ((rng_state >> 16) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let ry = ((rng_state >> 16) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let rz = ((rng_state >> 16) as f32 / u32::MAX as f32) * 2.0 - 1.0;

        let dir = Vec3::new(rx, ry, rz).normalize_or_zero();
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let dist = ((rng_state >> 16) as f32 / u32::MAX as f32) * radius * 0.8;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let frag_radius = radius * (0.2 + ((rng_state >> 16) as f32 / u32::MAX as f32) * 0.5);

        shapes.push(CarveShape::Sphere {
            center: center + dir * dist,
            radius: frag_radius,
        });
    }

    carve_batch(grid, &shapes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;

    fn make_test_grid() -> MutableVoxelGrid {
        let sphere = SdfNode::sphere(1.5);
        MutableVoxelGrid::from_sdf(
            &sphere, [32, 32, 32],
            Vec3::splat(-2.0), Vec3::splat(2.0),
        )
    }

    #[test]
    fn test_carve_sphere() {
        let mut grid = make_test_grid();
        let result = carve(&mut grid, &CarveShape::Sphere {
            center: Vec3::ZERO,
            radius: 0.5,
        });

        assert!(result.modified_voxels > 0);
        assert!(result.removed_volume > 0.0);
        assert!(!result.dirty_chunks.is_empty());

        // Center should now be outside (positive distance)
        let center_d = grid.get_distance(16, 16, 16);
        assert!(center_d >= 0.0, "Center should be carved out, got {}", center_d);
    }

    #[test]
    fn test_carve_box() {
        let mut grid = make_test_grid();
        let result = carve(&mut grid, &CarveShape::Box {
            center: Vec3::ZERO,
            half_extents: Vec3::splat(0.3),
            rotation: Quat::IDENTITY,
        });

        assert!(result.modified_voxels > 0);
        assert!(result.removed_volume > 0.0);
    }

    #[test]
    fn test_carve_sdf() {
        let mut grid = make_test_grid();
        let carve_shape = SdfNode::cylinder(0.3, 2.0);
        let result = carve(&mut grid, &CarveShape::Sdf(carve_shape));

        assert!(result.modified_voxels > 0);
    }

    #[test]
    fn test_carve_batch() {
        let mut grid = make_test_grid();
        let shapes = vec![
            CarveShape::Sphere { center: Vec3::new(0.5, 0.0, 0.0), radius: 0.3 },
            CarveShape::Sphere { center: Vec3::new(-0.5, 0.0, 0.0), radius: 0.3 },
        ];

        let result = carve_batch(&mut grid, &shapes);
        assert!(result.modified_voxels > 0);
    }

    #[test]
    fn test_carve_outside_bounds() {
        let mut grid = make_test_grid();
        let result = carve(&mut grid, &CarveShape::Sphere {
            center: Vec3::splat(100.0),
            radius: 0.5,
        });

        assert_eq!(result.modified_voxels, 0);
    }

    #[test]
    fn test_explode() {
        let mut grid = make_test_grid();
        let result = explode(&mut grid, Vec3::ZERO, 0.5, 5, 12345);

        assert!(result.modified_voxels > 0);
        assert!(result.removed_volume > 0.0);
    }

    #[test]
    fn test_carve_preserves_distance_property() {
        let mut grid = make_test_grid();

        // Distance at edge should be near 1.5 (sphere radius) minus distance to center
        let edge_before = grid.get_distance(16, 16, 16); // center, should be ~-1.5

        carve(&mut grid, &CarveShape::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 0.5,
        });

        let edge_after = grid.get_distance(16, 16, 16);
        // After carving a 0.5 radius sphere at center, distance should be ~0.5 (max(-1.5, -(-0.5)))
        assert!(edge_after > edge_before, "Distance should increase after carving");
    }
}
