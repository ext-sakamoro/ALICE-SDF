//! Mip Chain Generation for 3D Volume Textures
//!
//! Generates hierarchical mip levels using min-downsample,
//! which preserves the SDF distance property (closest surface wins).
//!
//! Author: Moroya Sakamoto

use super::{Volume3D, VoxelDistGrad};

/// Generate mip chain for a distance volume using min-downsample
///
/// Each mip level is half the resolution of the previous level.
/// The minimum distance of the 8 children is used, preserving
/// the SDF property that distance decreases toward the surface.
///
/// # Arguments
/// * `volume` - The base (mip 0) volume
///
/// # Returns
/// Vector of mip levels (mip 1, mip 2, ..., mip N where any axis <= 1)
pub fn generate_mip_chain(volume: &Volume3D<f32>) -> Vec<Vec<f32>> {
    let mut mips = Vec::new();
    let mut prev_data = &volume.data;
    let mut prev_res = volume.resolution;
    let mut owned_data;

    loop {
        let next_res = [
            (prev_res[0] / 2).max(1),
            (prev_res[1] / 2).max(1),
            (prev_res[2] / 2).max(1),
        ];

        // Stop when we can't downsample further
        if next_res[0] == prev_res[0] && next_res[1] == prev_res[1] && next_res[2] == prev_res[2] {
            break;
        }

        let total = next_res[0] as usize * next_res[1] as usize * next_res[2] as usize;
        let mut mip_data = vec![f32::MAX; total];

        let prev_sx = prev_res[0] as usize;
        let prev_sy = prev_res[1] as usize;

        for z in 0..next_res[2] as usize {
            for y in 0..next_res[1] as usize {
                for x in 0..next_res[0] as usize {
                    // Sample 8 children from previous level
                    let x0 = (x * 2).min(prev_res[0] as usize - 1);
                    let y0 = (y * 2).min(prev_res[1] as usize - 1);
                    let z0 = (z * 2).min(prev_res[2] as usize - 1);
                    let x1 = (x0 + 1).min(prev_res[0] as usize - 1);
                    let y1 = (y0 + 1).min(prev_res[1] as usize - 1);
                    let z1 = (z0 + 1).min(prev_res[2] as usize - 1);

                    let idx = |ix: usize, iy: usize, iz: usize| -> usize {
                        ix + iy * prev_sx + iz * prev_sx * prev_sy
                    };

                    // Min of 8 children (preserves SDF distance property)
                    let min_val = prev_data[idx(x0, y0, z0)]
                        .min(prev_data[idx(x1, y0, z0)])
                        .min(prev_data[idx(x0, y1, z0)])
                        .min(prev_data[idx(x1, y1, z0)])
                        .min(prev_data[idx(x0, y0, z1)])
                        .min(prev_data[idx(x1, y0, z1)])
                        .min(prev_data[idx(x0, y1, z1)])
                        .min(prev_data[idx(x1, y1, z1)]);

                    let out_idx = x
                        + y * next_res[0] as usize
                        + z * next_res[0] as usize * next_res[1] as usize;
                    mip_data[out_idx] = min_val;
                }
            }
        }

        mips.push(mip_data.clone());
        owned_data = mip_data;
        prev_data = &owned_data;
        prev_res = next_res;
    }

    mips
}

/// Generate mip chain for a distance+gradient volume
///
/// Distance uses min-downsample. Gradient is taken from the child
/// with the minimum distance (follows the nearest surface).
pub fn generate_mip_chain_distgrad(volume: &Volume3D<VoxelDistGrad>) -> Vec<Vec<VoxelDistGrad>> {
    let mut mips = Vec::new();
    let mut prev_data = &volume.data;
    let mut prev_res = volume.resolution;
    let mut owned_data;

    loop {
        let next_res = [
            (prev_res[0] / 2).max(1),
            (prev_res[1] / 2).max(1),
            (prev_res[2] / 2).max(1),
        ];

        if next_res[0] == prev_res[0] && next_res[1] == prev_res[1] && next_res[2] == prev_res[2] {
            break;
        }

        let total = next_res[0] as usize * next_res[1] as usize * next_res[2] as usize;
        let mut mip_data = vec![VoxelDistGrad::default(); total];

        let prev_sx = prev_res[0] as usize;
        let prev_sy = prev_res[1] as usize;

        for z in 0..next_res[2] as usize {
            for y in 0..next_res[1] as usize {
                for x in 0..next_res[0] as usize {
                    let x0 = (x * 2).min(prev_res[0] as usize - 1);
                    let y0 = (y * 2).min(prev_res[1] as usize - 1);
                    let z0 = (z * 2).min(prev_res[2] as usize - 1);
                    let x1 = (x0 + 1).min(prev_res[0] as usize - 1);
                    let y1 = (y0 + 1).min(prev_res[1] as usize - 1);
                    let z1 = (z0 + 1).min(prev_res[2] as usize - 1);

                    let idx = |ix: usize, iy: usize, iz: usize| -> usize {
                        ix + iy * prev_sx + iz * prev_sx * prev_sy
                    };

                    // Find child with minimum distance
                    let children = [
                        prev_data[idx(x0, y0, z0)],
                        prev_data[idx(x1, y0, z0)],
                        prev_data[idx(x0, y1, z0)],
                        prev_data[idx(x1, y1, z0)],
                        prev_data[idx(x0, y0, z1)],
                        prev_data[idx(x1, y0, z1)],
                        prev_data[idx(x0, y1, z1)],
                        prev_data[idx(x1, y1, z1)],
                    ];

                    let mut min_child = children[0];
                    for &child in &children[1..] {
                        if child.distance < min_child.distance {
                            min_child = child;
                        }
                    }

                    let out_idx = x
                        + y * next_res[0] as usize
                        + z * next_res[0] as usize * next_res[1] as usize;
                    mip_data[out_idx] = min_child;
                }
            }
        }

        mips.push(mip_data.clone());
        owned_data = mip_data;
        prev_data = &owned_data;
        prev_res = next_res;
    }

    mips
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_mip_chain_dimensions() {
        let vol: Volume3D<f32> = Volume3D::new([8, 8, 8], Vec3::splat(-1.0), Vec3::splat(1.0));

        let mips = generate_mip_chain(&vol);

        assert_eq!(mips.len(), 3); // 8->4->2->1
        assert_eq!(mips[0].len(), 4 * 4 * 4);
        assert_eq!(mips[1].len(), 2 * 2 * 2);
        assert_eq!(mips[2].len(), 1 * 1 * 1);
    }

    #[test]
    fn test_mip_chain_min_downsample() {
        let mut vol: Volume3D<f32> = Volume3D::new([4, 4, 4], Vec3::splat(-1.0), Vec3::splat(1.0));

        // Fill with positive values
        for v in vol.data.iter_mut() {
            *v = 10.0;
        }

        // Set one voxel to negative (inside surface)
        vol.set(0, 0, 0, -1.0);

        let mips = generate_mip_chain(&vol);

        // First mip should carry the minimum (-1.0) to the parent
        assert!(
            mips[0][0] < 0.0,
            "Min downsample should preserve negative distance"
        );
    }

    #[test]
    fn test_mip_chain_distgrad() {
        let vol: Volume3D<VoxelDistGrad> =
            Volume3D::new([4, 4, 4], Vec3::splat(-1.0), Vec3::splat(1.0));

        let mips = generate_mip_chain_distgrad(&vol);

        assert_eq!(mips.len(), 2); // 4->2->1
        assert_eq!(mips[0].len(), 2 * 2 * 2);
        assert_eq!(mips[1].len(), 1 * 1 * 1);
    }

    #[test]
    fn test_non_power_of_two() {
        let vol: Volume3D<f32> = Volume3D::new([6, 6, 6], Vec3::splat(-1.0), Vec3::splat(1.0));

        let mips = generate_mip_chain(&vol);

        // 6->3->1
        assert_eq!(mips.len(), 2);
        assert_eq!(mips[0].len(), 3 * 3 * 3);
        assert_eq!(mips[1].len(), 1 * 1 * 1);
    }
}
