//! Voronoi Fracture: Split voxel grid into convex pieces (Deep Fried Edition)
//!
//! Uses Voronoi tessellation to split a voxel grid into multiple pieces,
//! each becoming an independent mesh that can be used as a physics object.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

use super::MutableVoxelGrid;
use crate::mesh::{Mesh, Vertex};

/// Configuration for Voronoi fracture
#[derive(Debug, Clone)]
pub struct FractureConfig {
    /// Number of fracture pieces
    pub piece_count: u32,
    /// Random seed for Voronoi point placement
    pub seed: u64,
    /// Minimum piece size (skip pieces smaller than this)
    pub min_piece_size: f32,
    /// Noise amplitude for fracture planes (0.0 = clean cuts)
    pub noise_amplitude: f32,
}

impl Default for FractureConfig {
    fn default() -> Self {
        Self {
            piece_count: 8,
            seed: 42,
            min_piece_size: 0.05,
            noise_amplitude: 0.0,
        }
    }
}

/// A single fracture piece
#[derive(Debug)]
pub struct FracturePiece {
    /// Index of the Voronoi cell
    pub cell_index: u32,
    /// Center of mass (approximate)
    pub center: Vec3,
    /// The mesh for this piece
    pub mesh: Mesh,
    /// Volume of this piece (in voxels)
    pub voxel_count: u32,
}

/// Perform Voronoi fracture on a voxel grid
///
/// Places `piece_count` random seed points within the grid bounds,
/// assigns each interior voxel to its nearest seed, then generates
/// a mesh for each piece using Marching Cubes.
pub fn voronoi_fracture(
    grid: &MutableVoxelGrid,
    center: Vec3,
    radius: f32,
    config: &FractureConfig,
) -> Vec<FracturePiece> {
    // Generate Voronoi seed points within the fracture region
    let seeds = generate_seeds(center, radius, config.piece_count, config.seed);

    // Assign each voxel to nearest seed
    let [rx, ry, rz] = grid.resolution;
    let total = (rx * ry * rz) as usize;
    let mut assignments = vec![u32::MAX; total];
    let mut cell_counts = vec![0u32; config.piece_count as usize];

    let vs = grid.voxel_size();
    let radius_sq = radius * radius;

    for z in 0..rz {
        for y in 0..ry {
            for x in 0..rx {
                let idx = grid.voxel_index(x, y, z);
                let d = grid.distances[idx];

                // Only assign interior voxels (negative distance = inside)
                if d >= 0.0 {
                    continue;
                }

                let world = grid.grid_to_world(x, y, z);

                // Only fracture within the radius
                if (world - center).length_squared() > radius_sq {
                    continue;
                }

                // Find nearest seed with optional noise
                let mut best_dist = f32::MAX;
                let mut best_seed = 0u32;

                for (si, seed_pos) in seeds.iter().enumerate() {
                    let mut d2 = (world - *seed_pos).length_squared();

                    // Add noise to fracture boundaries
                    if config.noise_amplitude > 0.0 {
                        let noise = simple_hash_noise(world, si as u32) * config.noise_amplitude;
                        d2 += noise;
                    }

                    if d2 < best_dist {
                        best_dist = d2;
                        best_seed = si as u32;
                    }
                }

                assignments[idx] = best_seed;
                cell_counts[best_seed as usize] += 1;
            }
        }
    }

    // Generate mesh for each piece
    let mut pieces = Vec::new();
    let min_voxels = (config.min_piece_size / vs.x).max(1.0) as u32;

    for cell in 0..config.piece_count {
        if cell_counts[cell as usize] < min_voxels {
            continue;
        }

        let (mesh, center_of_mass) = extract_piece_mesh(grid, &assignments, cell);

        if mesh.vertices.is_empty() {
            continue;
        }

        pieces.push(FracturePiece {
            cell_index: cell,
            center: center_of_mass,
            mesh,
            voxel_count: cell_counts[cell as usize],
        });
    }

    pieces
}

/// Generate random seed points within a sphere
fn generate_seeds(center: Vec3, radius: f32, count: u32, seed: u64) -> Vec<Vec3> {
    let mut seeds = Vec::with_capacity(count as usize);
    let mut rng = seed;

    for _ in 0..count {
        // Rejection sampling for uniform distribution in sphere
        loop {
            rng = lcg_next(rng);
            let rx = lcg_float(rng) * 2.0 + -1.0;
            rng = lcg_next(rng);
            let ry = lcg_float(rng) * 2.0 + -1.0;
            rng = lcg_next(rng);
            let rz = lcg_float(rng) * 2.0 + -1.0;

            let p = Vec3::new(rx, ry, rz);
            if p.length_squared() <= 1.0 {
                seeds.push(center + p * radius);
                break;
            }
        }
    }

    seeds
}

/// Extract a mesh for a single Voronoi cell
///
/// Creates an isosurface at the boundary of the cell using a simple
/// surface-finding approach on the assignment grid.
fn extract_piece_mesh(grid: &MutableVoxelGrid, assignments: &[u32], cell: u32) -> (Mesh, Vec3) {
    let [rx, ry, rz] = grid.resolution;
    let vs = grid.voxel_size();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut center_sum = Vec3::ZERO;
    let mut center_count = 0u32;

    // Find boundary voxels: voxels assigned to this cell with at least one
    // neighbor assigned to a different cell (or outside)
    for z in 1..rz.saturating_sub(1) {
        for y in 1..ry.saturating_sub(1) {
            for x in 1..rx.saturating_sub(1) {
                let idx = grid.voxel_index(x, y, z);
                if assignments[idx] != cell {
                    continue;
                }

                let world = grid.grid_to_world(x, y, z);
                center_sum += world;
                center_count += 1;

                // Check 6-connected neighbors
                let neighbors = [
                    grid.voxel_index(x + 1, y, z),
                    grid.voxel_index(x.saturating_sub(1), y, z),
                    grid.voxel_index(x, y + 1, z),
                    grid.voxel_index(x, y.saturating_sub(1), z),
                    grid.voxel_index(x, y, z + 1),
                    grid.voxel_index(x, y, z.saturating_sub(1)),
                ];
                let normals = [
                    Vec3::X,
                    Vec3::NEG_X,
                    Vec3::Y,
                    Vec3::NEG_Y,
                    Vec3::Z,
                    Vec3::NEG_Z,
                ];

                for (ni, &nidx) in neighbors.iter().enumerate() {
                    if assignments[nidx] != cell {
                        // Boundary face! Generate a quad (2 triangles)
                        //
                        // The extent along each direction is the voxel size of
                        // THAT axis. Using `vs.x` for all three only works on
                        // cubic voxels; on an anisotropic grid it both offsets
                        // the face and mis-sizes the quad (a 24x48x24 grid
                        // overstates the surface area of a sphere by 68%).
                        // Pinned by the Cauchy projection oracle in
                        // `tests/test_terrain_destruction_oracle.rs`.
                        let normal = normals[ni];
                        let face_center = world + normal * (extent_along(normal, vs) * 0.5);

                        let (u, v) = make_tangent_basis(normal);
                        let hu = extent_along(u, vs) * 0.5;
                        let hv = extent_along(v, vs) * 0.5;

                        let p0 = face_center - u * hu - v * hv;
                        let p1 = face_center + u * hu - v * hv;
                        let p2 = face_center + u * hu + v * hv;
                        let p3 = face_center - u * hu + v * hv;

                        let vi = vertices.len() as u32;
                        vertices.push(Vertex {
                            position: p0,
                            normal,
                            ..Default::default()
                        });
                        vertices.push(Vertex {
                            position: p1,
                            normal,
                            ..Default::default()
                        });
                        vertices.push(Vertex {
                            position: p2,
                            normal,
                            ..Default::default()
                        });
                        vertices.push(Vertex {
                            position: p3,
                            normal,
                            ..Default::default()
                        });

                        indices.push(vi);
                        indices.push(vi + 1);
                        indices.push(vi + 2);
                        indices.push(vi);
                        indices.push(vi + 2);
                        indices.push(vi + 3);
                    }
                }
            }
        }
    }

    let center = if center_count > 0 {
        center_sum / center_count as f32
    } else {
        Vec3::ZERO
    };

    (Mesh { vertices, indices }, center)
}

/// Voxel extent along an axis-aligned direction
///
/// `|d| = 1` and `d` is one of the six signed basis vectors, so this picks the
/// voxel size of that axis.
#[inline]
fn extent_along(d: Vec3, voxel_size: Vec3) -> f32 {
    (d.abs() * voxel_size).element_sum()
}

/// Create a **right-handed** orthonormal tangent basis for a face normal
///
/// `u x v == normal`, so a quad wound `p0 -> p1 -> p2 -> p3` over
/// `(±u, ±v)` comes out counter-clockwise when seen from outside, which is the
/// crate-wide mesh convention (outward winding, positive signed volume — see
/// `tests/test_mesh_orientation.rs`). With `v = u x normal` the basis is
/// left-handed and every fracture piece is emitted inside-out (measured signed
/// volume -14.41 for a sphere whose volume is +14.14).
fn make_tangent_basis(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let u = normal.cross(up).normalize();
    let v = normal.cross(u).normalize();
    (u, v)
}

/// Simple hash-based noise for fracture boundary perturbation
fn simple_hash_noise(pos: Vec3, seed: u32) -> f32 {
    let ix = (pos.x * 100.0) as u32;
    let iy = (pos.y * 100.0) as u32;
    let iz = (pos.z * 100.0) as u32;
    let h = ix
        .wrapping_mul(374761393)
        .wrapping_add(iy.wrapping_mul(668265263))
        .wrapping_add(iz.wrapping_mul(1274126177))
        .wrapping_add(seed.wrapping_mul(1103515245));
    let h = h ^ (h >> 13);
    let h = h.wrapping_mul(1103515245);
    (h as f32) / (u32::MAX as f32)
}

#[inline]
const fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

#[inline]
fn lcg_float(state: u64) -> f32 {
    ((state >> 16) as u32 as f32) / (u32::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;

    fn make_test_grid() -> MutableVoxelGrid {
        let sphere = SdfNode::sphere(1.5);
        MutableVoxelGrid::from_sdf(&sphere, [16, 16, 16], Vec3::splat(-2.0), Vec3::splat(2.0))
    }

    #[test]
    fn test_voronoi_fracture() {
        let grid = make_test_grid();
        let config = FractureConfig {
            piece_count: 4,
            seed: 42,
            min_piece_size: 0.01,
            ..Default::default()
        };

        let pieces = voronoi_fracture(&grid, Vec3::ZERO, 1.5, &config);

        assert!(!pieces.is_empty(), "Should produce fracture pieces");

        let total_voxels: u32 = pieces.iter().map(|p| p.voxel_count).sum();
        assert!(total_voxels > 0);

        for piece in &pieces {
            assert!(!piece.mesh.vertices.is_empty());
        }
    }

    #[test]
    fn test_fracture_deterministic() {
        let grid = make_test_grid();
        let config = FractureConfig {
            piece_count: 3,
            seed: 999,
            min_piece_size: 0.01,
            ..Default::default()
        };

        let a = voronoi_fracture(&grid, Vec3::ZERO, 1.0, &config);
        let b = voronoi_fracture(&grid, Vec3::ZERO, 1.0, &config);

        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.voxel_count, pb.voxel_count);
            assert_eq!(pa.center, pb.center);
        }
    }

    #[test]
    fn test_fracture_small_radius() {
        let grid = make_test_grid();
        let config = FractureConfig {
            piece_count: 4,
            seed: 42,
            min_piece_size: 0.01,
            ..Default::default()
        };

        let pieces = voronoi_fracture(&grid, Vec3::ZERO, 0.3, &config);
        // Small region, fewer pieces expected
        assert!(pieces.len() <= 4);
    }

    #[test]
    fn test_generate_seeds() {
        let seeds = generate_seeds(Vec3::ZERO, 1.0, 10, 42);
        assert_eq!(seeds.len(), 10);

        for seed in &seeds {
            assert!(seed.length() <= 1.0 + 0.001);
        }
    }

    /// oracle: the basis must be orthonormal AND right-handed (`u x v = n`),
    /// otherwise every quad built from it is wound inside-out
    #[test]
    fn test_make_tangent_basis() {
        for normal in [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ] {
            let (u, v) = make_tangent_basis(normal);
            assert!(u.dot(normal).abs() < 0.001, "{normal:?}");
            assert!(v.dot(normal).abs() < 0.001, "{normal:?}");
            assert!(u.dot(v).abs() < 0.001, "{normal:?}");
            assert!((u.length() - 1.0).abs() < 0.001, "{normal:?}");
            assert!((v.length() - 1.0).abs() < 0.001, "{normal:?}");
            assert!(
                (u.cross(v) - normal).length() < 0.001,
                "basis is left handed for {normal:?}: u x v = {:?}",
                u.cross(v)
            );
        }
    }

    /// oracle: on an anisotropic grid the face extents are the voxel sizes of
    /// the respective axes, not `voxel_size.x` three times
    #[test]
    fn test_extent_along_picks_the_axis_component() {
        let vs = Vec3::new(0.25, 0.5, 1.0);
        assert!((extent_along(Vec3::X, vs) - 0.25).abs() < 1e-6);
        assert!((extent_along(Vec3::NEG_X, vs) - 0.25).abs() < 1e-6);
        assert!((extent_along(Vec3::Y, vs) - 0.5).abs() < 1e-6);
        assert!((extent_along(Vec3::NEG_Z, vs) - 1.0).abs() < 1e-6);
    }
}
