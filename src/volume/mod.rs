//! SDF 3D Volume Texture Baking (Deep Fried Edition)
//!
//! Bake SDF to 3D volume textures for GPU distance field queries.
//! Used by UE5 Lumen, distance field shadows, and real-time GI.
//!
//! # Features
//!
//! - **CPU Baking**: Rayon Z-slab parallel grid evaluation
//! - **GPU Baking**: 3D compute shader dispatch via `GpuEvaluator`
//! - **Trilinear Sampling**: 8-corner lookup + 3-axis lerp
//! - **Mip Chain**: Min-downsample for hierarchical queries
//! - **Export**: Raw binary and DDS 3D texture formats
//!
//! # Deep Fried Optimizations
//!
//! - **Z-Slab Parallelism**: Matches `eval_grid` pattern for cache efficiency
//! - **Pre-computed Step**: Step vector computed once, not per-voxel
//! - **Min Downsample Mips**: Preserves SDF distance property
//!
//! Author: Moroya Sakamoto

pub mod bake;
pub mod export;
pub mod mipchain;

#[cfg(feature = "gpu")]
pub mod gpu_bake;

use glam::Vec3;

/// Channels to bake into the volume
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BakeChannels {
    /// Distance only (1 channel, f32)
    Distance,
    /// Distance + gradient/normal (4 channels: dist, nx, ny, nz)
    DistanceGradient,
    /// Distance + material ID (2 channels: dist, material_id as f32)
    DistanceMaterial,
}

/// Configuration for volume baking
#[derive(Debug, Clone)]
pub struct BakeConfig {
    /// Resolution along each axis [x, y, z]
    pub resolution: [u32; 3],
    /// World-space minimum bounds
    pub bounds_min: Vec3,
    /// World-space maximum bounds
    pub bounds_max: Vec3,
    /// Channels to bake
    pub channels: BakeChannels,
    /// Generate mip chain after baking
    pub generate_mips: bool,
    /// Padding voxels around the boundary (filled with large distance)
    pub padding: u32,
}

impl Default for BakeConfig {
    fn default() -> Self {
        Self {
            resolution: [64, 64, 64],
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            channels: BakeChannels::Distance,
            generate_mips: false,
            padding: 0,
        }
    }
}

/// Voxel with distance + gradient (16 bytes, GPU-friendly)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct VoxelDistGrad {
    /// Signed distance
    pub distance: f32,
    /// Normal X
    pub nx: f32,
    /// Normal Y
    pub ny: f32,
    /// Normal Z
    pub nz: f32,
}

#[cfg(feature = "gpu")]
unsafe impl bytemuck::Pod for VoxelDistGrad {}
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Zeroable for VoxelDistGrad {}

/// 3D volume texture
///
/// Stores voxel data in a flat array with Z-major ordering:
/// `index = x + y * resolution.x + z * resolution.x * resolution.y`
#[derive(Debug, Clone)]
pub struct Volume3D<T: Copy + Default> {
    /// Flat voxel data in Z-major order
    pub data: Vec<T>,
    /// Resolution along each axis [x, y, z]
    pub resolution: [u32; 3],
    /// World-space minimum bounds
    pub world_min: Vec3,
    /// World-space maximum bounds
    pub world_max: Vec3,
    /// Mip chain (if generated)
    pub mips: Vec<Vec<T>>,
}

impl<T: Copy + Default> Volume3D<T> {
    /// Create a new volume filled with default values
    pub fn new(resolution: [u32; 3], world_min: Vec3, world_max: Vec3) -> Self {
        let total = resolution[0] as usize * resolution[1] as usize * resolution[2] as usize;
        Self {
            data: vec![T::default(); total],
            resolution,
            world_min,
            world_max,
            mips: Vec::new(),
        }
    }

    /// Total number of voxels
    #[inline(always)]
    pub fn voxel_count(&self) -> usize {
        self.resolution[0] as usize * self.resolution[1] as usize * self.resolution[2] as usize
    }

    /// World-space size
    #[inline(always)]
    pub fn world_size(&self) -> Vec3 {
        self.world_max - self.world_min
    }

    /// Voxel size in world-space
    #[inline(always)]
    pub fn voxel_size(&self) -> Vec3 {
        let size = self.world_size();
        Vec3::new(
            size.x / (self.resolution[0] as f32 - 1.0).max(1.0),
            size.y / (self.resolution[1] as f32 - 1.0).max(1.0),
            size.z / (self.resolution[2] as f32 - 1.0).max(1.0),
        )
    }

    /// Get flat index from 3D coordinates (bounds-unchecked)
    #[inline(always)]
    pub fn index(&self, x: u32, y: u32, z: u32) -> usize {
        x as usize
            + y as usize * self.resolution[0] as usize
            + z as usize * self.resolution[0] as usize * self.resolution[1] as usize
    }

    /// Get voxel at 3D coordinates
    #[inline(always)]
    pub fn get(&self, x: u32, y: u32, z: u32) -> T {
        self.data[self.index(x, y, z)]
    }

    /// Set voxel at 3D coordinates
    #[inline(always)]
    pub fn set(&mut self, x: u32, y: u32, z: u32, value: T) {
        let idx = self.index(x, y, z);
        self.data[idx] = value;
    }

    /// Get world-space position for voxel coordinates
    #[inline(always)]
    pub fn voxel_to_world(&self, x: u32, y: u32, z: u32) -> Vec3 {
        let step = self.voxel_size();
        Vec3::new(
            self.world_min.x + x as f32 * step.x,
            self.world_min.y + y as f32 * step.y,
            self.world_min.z + z as f32 * step.z,
        )
    }

    /// Number of mip levels
    pub fn mip_count(&self) -> usize {
        self.mips.len() + 1 // +1 for level 0 (self.data)
    }
}

impl Volume3D<f32> {
    /// Trilinear sampling at world-space position
    ///
    /// Returns interpolated distance value. Points outside the volume
    /// are clamped to the nearest boundary voxel.
    pub fn sample_trilinear(&self, pos: Vec3) -> f32 {
        let size = self.world_size();
        let res = [
            self.resolution[0] as f32,
            self.resolution[1] as f32,
            self.resolution[2] as f32,
        ];

        // Normalize to [0, res-1] coordinates
        let fx = ((pos.x - self.world_min.x) / size.x * (res[0] - 1.0)).clamp(0.0, res[0] - 1.0);
        let fy = ((pos.y - self.world_min.y) / size.y * (res[1] - 1.0)).clamp(0.0, res[1] - 1.0);
        let fz = ((pos.z - self.world_min.z) / size.z * (res[2] - 1.0)).clamp(0.0, res[2] - 1.0);

        // Integer coordinates
        let x0 = fx as u32;
        let y0 = fy as u32;
        let z0 = fz as u32;
        let x1 = (x0 + 1).min(self.resolution[0] - 1);
        let y1 = (y0 + 1).min(self.resolution[1] - 1);
        let z1 = (z0 + 1).min(self.resolution[2] - 1);

        // Fractional parts
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let tz = fz - z0 as f32;

        // 8-corner lookup
        let c000 = self.get(x0, y0, z0);
        let c100 = self.get(x1, y0, z0);
        let c010 = self.get(x0, y1, z0);
        let c110 = self.get(x1, y1, z0);
        let c001 = self.get(x0, y0, z1);
        let c101 = self.get(x1, y0, z1);
        let c011 = self.get(x0, y1, z1);
        let c111 = self.get(x1, y1, z1);

        // 3-axis lerp
        let c00 = c000 * (1.0 - tx) + c100 * tx;
        let c10 = c010 * (1.0 - tx) + c110 * tx;
        let c01 = c001 * (1.0 - tx) + c101 * tx;
        let c11 = c011 * (1.0 - tx) + c111 * tx;

        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;

        c0 * (1.0 - tz) + c1 * tz
    }
}

impl Volume3D<VoxelDistGrad> {
    /// Trilinear sampling of distance + gradient at world-space position
    ///
    /// Returns interpolated distance and normalized gradient.
    pub fn sample_trilinear(&self, pos: Vec3) -> VoxelDistGrad {
        let size = self.world_size();
        let res = [
            self.resolution[0] as f32,
            self.resolution[1] as f32,
            self.resolution[2] as f32,
        ];

        let fx = ((pos.x - self.world_min.x) / size.x * (res[0] - 1.0)).clamp(0.0, res[0] - 1.0);
        let fy = ((pos.y - self.world_min.y) / size.y * (res[1] - 1.0)).clamp(0.0, res[1] - 1.0);
        let fz = ((pos.z - self.world_min.z) / size.z * (res[2] - 1.0)).clamp(0.0, res[2] - 1.0);

        let x0 = fx as u32;
        let y0 = fy as u32;
        let z0 = fz as u32;
        let x1 = (x0 + 1).min(self.resolution[0] - 1);
        let y1 = (y0 + 1).min(self.resolution[1] - 1);
        let z1 = (z0 + 1).min(self.resolution[2] - 1);

        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let tz = fz - z0 as f32;

        // 8-corner lookup
        let corners = [
            self.get(x0, y0, z0),
            self.get(x1, y0, z0),
            self.get(x0, y1, z0),
            self.get(x1, y1, z0),
            self.get(x0, y0, z1),
            self.get(x1, y0, z1),
            self.get(x0, y1, z1),
            self.get(x1, y1, z1),
        ];

        // Interpolate each component
        let lerp = |a: f32, b: f32, t: f32| a * (1.0 - t) + b * t;

        let interp_component = |f: fn(&VoxelDistGrad) -> f32| -> f32 {
            let c00 = lerp(f(&corners[0]), f(&corners[1]), tx);
            let c10 = lerp(f(&corners[2]), f(&corners[3]), tx);
            let c01 = lerp(f(&corners[4]), f(&corners[5]), tx);
            let c11 = lerp(f(&corners[6]), f(&corners[7]), tx);
            let c0 = lerp(c00, c10, ty);
            let c1 = lerp(c01, c11, ty);
            lerp(c0, c1, tz)
        };

        VoxelDistGrad {
            distance: interp_component(|v| v.distance),
            nx: interp_component(|v| v.nx),
            ny: interp_component(|v| v.ny),
            nz: interp_component(|v| v.nz),
        }
    }
}

// Re-exports
pub use bake::{bake_volume, bake_volume_with_normals};
pub use export::{export_dds_3d, export_raw};
pub use mipchain::generate_mip_chain;

#[cfg(feature = "gpu")]
pub use gpu_bake::gpu_bake_volume;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_creation() {
        let vol: Volume3D<f32> = Volume3D::new([4, 4, 4], Vec3::splat(-1.0), Vec3::splat(1.0));
        assert_eq!(vol.voxel_count(), 64);
        assert_eq!(vol.data.len(), 64);
    }

    #[test]
    fn test_volume_indexing() {
        let mut vol: Volume3D<f32> = Volume3D::new([4, 4, 4], Vec3::splat(-1.0), Vec3::splat(1.0));

        vol.set(1, 2, 3, 42.0);
        assert_eq!(vol.get(1, 2, 3), 42.0);
        assert_eq!(vol.index(1, 2, 3), 1 + 2 * 4 + 3 * 16);
    }

    #[test]
    fn test_trilinear_sampling_corners() {
        let mut vol: Volume3D<f32> = Volume3D::new([2, 2, 2], Vec3::ZERO, Vec3::ONE);

        // Set corners
        vol.set(0, 0, 0, 0.0);
        vol.set(1, 0, 0, 1.0);
        vol.set(0, 1, 0, 0.0);
        vol.set(1, 1, 0, 1.0);
        vol.set(0, 0, 1, 0.0);
        vol.set(1, 0, 1, 1.0);
        vol.set(0, 1, 1, 0.0);
        vol.set(1, 1, 1, 1.0);

        // Corner should return exact value
        let d = vol.sample_trilinear(Vec3::ZERO);
        assert!((d - 0.0).abs() < 1e-5);

        let d = vol.sample_trilinear(Vec3::ONE);
        assert!((d - 1.0).abs() < 1e-5);

        // Center should interpolate
        let d = vol.sample_trilinear(Vec3::splat(0.5));
        assert!((d - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_trilinear_clamping() {
        let vol: Volume3D<f32> = Volume3D::new([4, 4, 4], Vec3::splat(-1.0), Vec3::splat(1.0));

        // Should not panic for out-of-bounds
        let _ = vol.sample_trilinear(Vec3::splat(-10.0));
        let _ = vol.sample_trilinear(Vec3::splat(10.0));
    }

    #[test]
    fn test_voxel_to_world() {
        let vol: Volume3D<f32> = Volume3D::new([3, 3, 3], Vec3::splat(-1.0), Vec3::splat(1.0));

        let p = vol.voxel_to_world(0, 0, 0);
        assert!((p - Vec3::splat(-1.0)).length() < 1e-5);

        let p = vol.voxel_to_world(2, 2, 2);
        assert!((p - Vec3::splat(1.0)).length() < 1e-5);

        let p = vol.voxel_to_world(1, 1, 1);
        assert!((p - Vec3::ZERO).length() < 1e-5);
    }
}
