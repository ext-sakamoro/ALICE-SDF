//! Mip Chain Generation for 3D Volume Textures
//!
//! Generates hierarchical mip levels using min-downsample,
//! which preserves the SDF distance property (closest surface wins).
//!
//! Author: Moroya Sakamoto

use super::{Volume3D, VoxelDistGrad};

/// Span of previous-level voxels that mip voxel `i` stands for on one axis.
///
/// Spans are `[2i, 2i+2)` except the last, which absorbs the leftover voxel
/// when `prev` is odd, so the spans **partition** `0..prev`. A span that left a
/// voxel out would make the mip report a larger distance than some voxel it
/// covers, and a min chain is only usable for skipping empty space while every
/// level is a lower bound of its own footprint. Identical to `[2i, 2i+2)` on
/// every axis whose resolution is even, so power-of-two chains are unchanged.
#[inline]
fn child_span(i: usize, next: usize, prev: usize) -> (usize, usize) {
    // `next` is `max(prev / 2, 1)`, so `2 * next <= prev` whenever `prev > 1`
    // and the start below always lands inside the previous level.
    debug_assert!(prev <= 1 || 2 * next <= prev);
    let start = (i * 2).min(prev.saturating_sub(1));
    let end = if i + 1 >= next {
        prev
    } else {
        ((i + 1) * 2).min(prev)
    };
    (start, end.max(start + 1))
}

/// Generate mip chain for a distance volume using min-downsample
///
/// Each mip level is half the resolution of the previous level.
/// The minimum distance of the children covering the voxel is used (2 per axis,
/// 3 on an axis whose previous resolution is odd), preserving the SDF property
/// that distance decreases toward the surface.
///
/// # Arguments
/// * `volume` - The base (mip 0) volume
///
/// # Returns
/// Vector of mip levels (mip 1, mip 2, ..., mip N where any axis <= 1)
pub fn generate_mip_chain(volume: &Volume3D<f32>) -> Vec<Vec<f32>> {
    // Index-based: mip[-1] = volume.data (base), mip[i] reads from mip[i-1].
    // Avoids clone-before-move by keeping data in-place in the output vec.
    let mut mips: Vec<Vec<f32>> = Vec::new();
    let mut res_chain = vec![volume.resolution];

    // Pre-compute resolution chain
    loop {
        let prev = *res_chain.last().unwrap();
        let next = [
            (prev[0] / 2).max(1),
            (prev[1] / 2).max(1),
            (prev[2] / 2).max(1),
        ];
        if next == prev {
            break;
        }
        res_chain.push(next);
    }

    // Build each mip level reading from the previous level already in `mips`
    // (or from `volume.data` for the first level).
    for level in 1..res_chain.len() {
        let prev_res = res_chain[level - 1];
        let next_res = res_chain[level];

        let prev_data: &[f32] = if level == 1 {
            &volume.data
        } else {
            &mips[level - 2]
        };

        let total = next_res[0] as usize * next_res[1] as usize * next_res[2] as usize;
        let mut mip_data = vec![f32::MAX; total];

        let prev_sx = prev_res[0] as usize;
        let prev_sy = prev_res[1] as usize;

        for z in 0..next_res[2] as usize {
            let (z0, z1) = child_span(z, next_res[2] as usize, prev_res[2] as usize);
            for y in 0..next_res[1] as usize {
                let (y0, y1) = child_span(y, next_res[1] as usize, prev_res[1] as usize);
                for x in 0..next_res[0] as usize {
                    let (x0, x1) = child_span(x, next_res[0] as usize, prev_res[0] as usize);

                    // Min over the children covering this voxel (preserves the
                    // SDF distance property: the mip is a lower bound of them)
                    let mut min_val = f32::MAX;
                    for cz in z0..z1 {
                        let plane = cz * prev_sx * prev_sy;
                        for cy in y0..y1 {
                            let row = plane + cy * prev_sx;
                            for cx in x0..x1 {
                                min_val = min_val.min(prev_data[row + cx]);
                            }
                        }
                    }

                    let out_idx = x
                        + y * next_res[0] as usize
                        + z * next_res[0] as usize * next_res[1] as usize;
                    mip_data[out_idx] = min_val;
                }
            }
        }

        mips.push(mip_data);
    }

    mips
}

/// Generate mip chain for a distance+gradient volume
///
/// Distance uses min-downsample over the children covering the voxel (see
/// [`child_span`]). Gradient is taken from the child with the minimum distance
/// (follows the nearest surface).
pub fn generate_mip_chain_distgrad(volume: &Volume3D<VoxelDistGrad>) -> Vec<Vec<VoxelDistGrad>> {
    let mut mips: Vec<Vec<VoxelDistGrad>> = Vec::new();
    let mut res_chain = vec![volume.resolution];

    loop {
        let prev = *res_chain.last().unwrap();
        let next = [
            (prev[0] / 2).max(1),
            (prev[1] / 2).max(1),
            (prev[2] / 2).max(1),
        ];
        if next == prev {
            break;
        }
        res_chain.push(next);
    }

    for level in 1..res_chain.len() {
        let prev_res = res_chain[level - 1];
        let next_res = res_chain[level];

        let prev_data: &[VoxelDistGrad] = if level == 1 {
            &volume.data
        } else {
            &mips[level - 2]
        };

        let total = next_res[0] as usize * next_res[1] as usize * next_res[2] as usize;
        let mut mip_data = vec![VoxelDistGrad::default(); total];

        let prev_sx = prev_res[0] as usize;
        let prev_sy = prev_res[1] as usize;

        for z in 0..next_res[2] as usize {
            let (z0, z1) = child_span(z, next_res[2] as usize, prev_res[2] as usize);
            for y in 0..next_res[1] as usize {
                let (y0, y1) = child_span(y, next_res[1] as usize, prev_res[1] as usize);
                for x in 0..next_res[0] as usize {
                    let (x0, x1) = child_span(x, next_res[0] as usize, prev_res[0] as usize);

                    // Child with the minimum distance among those covering this
                    // voxel, so the mip is a lower bound of its own footprint
                    let mut min_child = prev_data[x0 + y0 * prev_sx + z0 * prev_sx * prev_sy];
                    for cz in z0..z1 {
                        let plane = cz * prev_sx * prev_sy;
                        for cy in y0..y1 {
                            let row = plane + cy * prev_sx;
                            for cx in x0..x1 {
                                let child = prev_data[row + cx];
                                if child.distance < min_child.distance {
                                    min_child = child;
                                }
                            }
                        }
                    }

                    let out_idx = x
                        + y * next_res[0] as usize
                        + z * next_res[0] as usize * next_res[1] as usize;
                    mip_data[out_idx] = min_child;
                }
            }
        }

        mips.push(mip_data);
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
        assert_eq!(mips[2].len(), 1);
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
        assert_eq!(mips[1].len(), 1);
    }

    #[test]
    fn test_non_power_of_two() {
        let vol: Volume3D<f32> = Volume3D::new([6, 6, 6], Vec3::splat(-1.0), Vec3::splat(1.0));

        let mips = generate_mip_chain(&vol);

        // 6->3->1
        assert_eq!(mips.len(), 2);
        assert_eq!(mips[0].len(), 3 * 3 * 3);
        assert_eq!(mips[1].len(), 1);
    }
}
