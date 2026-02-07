//! CPU-based SDF Volume Baking (Deep Fried Edition)
//!
//! Bakes SDF to 3D volume texture using Rayon Z-slab parallelism.
//! Matches the `eval_grid` pattern from `src/eval/parallel.rs`.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use rayon::prelude::*;

use crate::eval::{eval, normal};
use crate::types::SdfNode;
use super::{BakeConfig, Volume3D, VoxelDistGrad};

/// Bake SDF to a distance-only volume (Deep Fried)
///
/// Uses Z-slab parallelism for optimal cache utilization.
///
/// # Arguments
/// * `node` - The SDF tree to bake
/// * `config` - Baking configuration
///
/// # Returns
/// Volume3D<f32> containing signed distances
pub fn bake_volume(node: &SdfNode, config: &BakeConfig) -> Volume3D<f32> {
    let res = config.resolution;
    let total = res[0] as usize * res[1] as usize * res[2] as usize;

    // Apply padding to bounds
    let padding_world = if config.padding > 0 {
        let size = config.bounds_max - config.bounds_min;
        Vec3::new(
            size.x / res[0] as f32 * config.padding as f32,
            size.y / res[1] as f32 * config.padding as f32,
            size.z / res[2] as f32 * config.padding as f32,
        )
    } else {
        Vec3::ZERO
    };
    let world_min = config.bounds_min - padding_world;
    let world_max = config.bounds_max + padding_world;

    let size = world_max - world_min;
    // Pre-calculate step vector (avoid division in hot loop)
    let step = Vec3::new(
        size.x / (res[0] as f32 - 1.0).max(1.0),
        size.y / (res[1] as f32 - 1.0).max(1.0),
        size.z / (res[2] as f32 - 1.0).max(1.0),
    );

    let mut data = vec![0.0f32; total];
    let slice_size = res[0] as usize * res[1] as usize;

    // Parallelize by Z-slices (Deep Fried: no integer division in hot path)
    data.par_chunks_mut(slice_size)
        .enumerate()
        .for_each(|(z, slice)| {
            let z_pos = world_min.z + z as f32 * step.z;

            for y in 0..res[1] as usize {
                let y_pos = world_min.y + y as f32 * step.y;
                let row_offset = y * res[0] as usize;

                for x in 0..res[0] as usize {
                    let x_pos = world_min.x + x as f32 * step.x;
                    let p = Vec3::new(x_pos, y_pos, z_pos);
                    slice[row_offset + x] = eval(node, p);
                }
            }
        });

    let mut volume = Volume3D {
        data,
        resolution: res,
        world_min,
        world_max,
        mips: Vec::new(),
    };

    if config.generate_mips {
        volume.mips = super::mipchain::generate_mip_chain(&volume);
    }

    volume
}

/// Bake SDF to a distance + gradient volume (Deep Fried)
///
/// Each voxel stores distance and surface normal.
///
/// # Arguments
/// * `node` - The SDF tree to bake
/// * `config` - Baking configuration
/// * `gradient_epsilon` - Epsilon for finite-difference normal estimation
///
/// # Returns
/// Volume3D<VoxelDistGrad> containing distance + normal per voxel
pub fn bake_volume_with_normals(
    node: &SdfNode,
    config: &BakeConfig,
    gradient_epsilon: f32,
) -> Volume3D<VoxelDistGrad> {
    let res = config.resolution;
    let total = res[0] as usize * res[1] as usize * res[2] as usize;

    let padding_world = if config.padding > 0 {
        let size = config.bounds_max - config.bounds_min;
        Vec3::new(
            size.x / res[0] as f32 * config.padding as f32,
            size.y / res[1] as f32 * config.padding as f32,
            size.z / res[2] as f32 * config.padding as f32,
        )
    } else {
        Vec3::ZERO
    };
    let world_min = config.bounds_min - padding_world;
    let world_max = config.bounds_max + padding_world;

    let size = world_max - world_min;
    let step = Vec3::new(
        size.x / (res[0] as f32 - 1.0).max(1.0),
        size.y / (res[1] as f32 - 1.0).max(1.0),
        size.z / (res[2] as f32 - 1.0).max(1.0),
    );

    let mut data = vec![VoxelDistGrad::default(); total];
    let slice_size = res[0] as usize * res[1] as usize;

    data.par_chunks_mut(slice_size)
        .enumerate()
        .for_each(|(z, slice)| {
            let z_pos = world_min.z + z as f32 * step.z;

            for y in 0..res[1] as usize {
                let y_pos = world_min.y + y as f32 * step.y;
                let row_offset = y * res[0] as usize;

                for x in 0..res[0] as usize {
                    let x_pos = world_min.x + x as f32 * step.x;
                    let p = Vec3::new(x_pos, y_pos, z_pos);

                    let d = eval(node, p);
                    let n = normal(node, p, gradient_epsilon);

                    slice[row_offset + x] = VoxelDistGrad {
                        distance: d,
                        nx: n.x,
                        ny: n.y,
                        nz: n.z,
                    };
                }
            }
        });

    let mut volume = Volume3D {
        data,
        resolution: res,
        world_min,
        world_max,
        mips: Vec::new(),
    };

    if config.generate_mips {
        volume.mips = super::mipchain::generate_mip_chain_distgrad(&volume);
    }

    volume
}

/// Bake SDF using compiled evaluator for maximum CPU performance
///
/// Uses the compiled VM path which is ~10x faster than interpreted eval.
pub fn bake_volume_compiled(
    compiled: &crate::compiled::CompiledSdf,
    config: &BakeConfig,
) -> Volume3D<f32> {
    use crate::compiled::eval_compiled;

    let res = config.resolution;
    let total = res[0] as usize * res[1] as usize * res[2] as usize;

    let padding_world = if config.padding > 0 {
        let size = config.bounds_max - config.bounds_min;
        Vec3::new(
            size.x / res[0] as f32 * config.padding as f32,
            size.y / res[1] as f32 * config.padding as f32,
            size.z / res[2] as f32 * config.padding as f32,
        )
    } else {
        Vec3::ZERO
    };
    let world_min = config.bounds_min - padding_world;
    let world_max = config.bounds_max + padding_world;

    let size = world_max - world_min;
    let step = Vec3::new(
        size.x / (res[0] as f32 - 1.0).max(1.0),
        size.y / (res[1] as f32 - 1.0).max(1.0),
        size.z / (res[2] as f32 - 1.0).max(1.0),
    );

    let mut data = vec![0.0f32; total];
    let slice_size = res[0] as usize * res[1] as usize;

    data.par_chunks_mut(slice_size)
        .enumerate()
        .for_each(|(z, slice)| {
            let z_pos = world_min.z + z as f32 * step.z;

            for y in 0..res[1] as usize {
                let y_pos = world_min.y + y as f32 * step.y;
                let row_offset = y * res[0] as usize;

                for x in 0..res[0] as usize {
                    let x_pos = world_min.x + x as f32 * step.x;
                    let p = Vec3::new(x_pos, y_pos, z_pos);
                    slice[row_offset + x] = eval_compiled(compiled, p);
                }
            }
        });

    Volume3D {
        data,
        resolution: res,
        world_min,
        world_max,
        mips: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bake_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let config = BakeConfig {
            resolution: [8, 8, 8],
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            ..Default::default()
        };

        let volume = bake_volume(&sphere, &config);

        assert_eq!(volume.voxel_count(), 512);
        assert_eq!(volume.data.len(), 512);

        // Center should be inside (negative distance)
        let center = volume.get(3, 3, 3);
        assert!(center < 0.0, "Center should be inside sphere, got {}", center);

        // Corner should be outside (positive distance)
        let corner = volume.get(0, 0, 0);
        assert!(corner > 0.0, "Corner should be outside sphere, got {}", corner);
    }

    #[test]
    fn test_bake_with_normals() {
        let sphere = SdfNode::sphere(1.0);
        let config = BakeConfig {
            resolution: [8, 8, 8],
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            channels: crate::volume::BakeChannels::DistanceGradient,
            ..Default::default()
        };

        let volume = bake_volume_with_normals(&sphere, &config, 0.001);

        assert_eq!(volume.voxel_count(), 512);

        // Check that normals are approximately normalized near surface
        let voxel = volume.get(7, 3, 3); // Near surface in +X direction
        let n_len = (voxel.nx * voxel.nx + voxel.ny * voxel.ny + voxel.nz * voxel.nz).sqrt();
        assert!(n_len > 0.9 && n_len < 1.1, "Normal should be ~normalized, got length {}", n_len);
    }

    #[test]
    fn test_bake_compiled() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = crate::compiled::CompiledSdf::compile(&sphere);
        let config = BakeConfig {
            resolution: [8, 8, 8],
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            ..Default::default()
        };

        let vol_interp = bake_volume(&sphere, &config);
        let vol_compiled = bake_volume_compiled(&compiled, &config);

        // Should produce same results (within float tolerance)
        for i in 0..vol_interp.data.len() {
            assert!(
                (vol_interp.data[i] - vol_compiled.data[i]).abs() < 0.001,
                "Mismatch at index {}: interp={}, compiled={}",
                i, vol_interp.data[i], vol_compiled.data[i]
            );
        }
    }

    #[test]
    fn test_bake_with_padding() {
        let sphere = SdfNode::sphere(1.0);
        let config = BakeConfig {
            resolution: [8, 8, 8],
            bounds_min: Vec3::splat(-1.0),
            bounds_max: Vec3::splat(1.0),
            padding: 2,
            ..Default::default()
        };

        let volume = bake_volume(&sphere, &config);

        // Bounds should be expanded
        assert!(volume.world_min.x < -1.0);
        assert!(volume.world_max.x > 1.0);
    }

    #[test]
    fn test_bake_with_mips() {
        let sphere = SdfNode::sphere(1.0);
        let config = BakeConfig {
            resolution: [8, 8, 8],
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            generate_mips: true,
            ..Default::default()
        };

        let volume = bake_volume(&sphere, &config);

        assert!(volume.mips.len() >= 2, "Should have at least 2 mip levels");
        assert_eq!(volume.mips[0].len(), 4 * 4 * 4); // 4^3
        assert_eq!(volume.mips[1].len(), 2 * 2 * 2); // 2^3
    }
}
