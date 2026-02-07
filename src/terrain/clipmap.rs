//! Clipmap LOD: Nested grids centered on camera (Deep Fried Edition)
//!
//! Implements geoclipmapping for terrain rendering. Each level covers
//! 2x the area of the previous level at half the vertex density.
//! The camera position determines which level is active at each distance.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

use super::Heightmap;
use crate::mesh::{Mesh, Vertex};

/// A complete clipmap terrain with multiple LOD levels
pub struct ClipmapTerrain {
    /// LOD levels (0 = finest, N = coarsest)
    pub levels: Vec<ClipmapLevel>,
    /// Last camera position used for generation
    pub camera_pos: Vec3,
    /// Grid resolution per level (vertices per edge)
    pub grid_resolution: u32,
}

/// A single clipmap LOD level
#[derive(Debug, Clone)]
pub struct ClipmapLevel {
    /// LOD level index (0 = finest)
    pub level: u32,
    /// World-space spacing between vertices at this level
    pub spacing: f32,
    /// Snapped grid origin (world X)
    pub origin_x: f32,
    /// Snapped grid origin (world Z)
    pub origin_z: f32,
    /// Grid resolution (vertices per edge)
    pub resolution: u32,
}

/// A mesh generated from a clipmap level
pub struct ClipmapMesh {
    /// LOD level
    pub level: u32,
    /// The terrain mesh for this level
    pub mesh: Mesh,
}

impl ClipmapTerrain {
    /// Create a new clipmap terrain
    ///
    /// - `num_levels`: number of LOD levels
    /// - `grid_resolution`: vertices per level edge (e.g. 64)
    /// - `base_spacing`: vertex spacing at level 0 (finest)
    pub fn new(num_levels: u32, grid_resolution: u32, base_spacing: f32) -> Self {
        let mut levels = Vec::with_capacity(num_levels as usize);
        for i in 0..num_levels {
            let spacing = base_spacing * (1 << i) as f32;
            levels.push(ClipmapLevel {
                level: i,
                spacing,
                origin_x: 0.0,
                origin_z: 0.0,
                resolution: grid_resolution,
            });
        }

        ClipmapTerrain {
            levels,
            camera_pos: Vec3::ZERO,
            grid_resolution,
        }
    }

    /// Update clipmap grid origins based on camera position
    ///
    /// Snaps each level's origin to prevent texture swimming.
    pub fn update(&mut self, camera_pos: Vec3) {
        self.camera_pos = camera_pos;

        for level in &mut self.levels {
            // Snap to grid spacing to prevent popping
            let half_extent = level.spacing * level.resolution as f32 * 0.5;
            level.origin_x = ((camera_pos.x - half_extent) / level.spacing).floor() * level.spacing;
            level.origin_z = ((camera_pos.z - half_extent) / level.spacing).floor() * level.spacing;
        }
    }

    /// Generate meshes for all levels
    pub fn generate_meshes(&self, heightmap: &Heightmap) -> Vec<ClipmapMesh> {
        self.levels
            .iter()
            .map(|level| {
                let mesh = generate_level_mesh(level, heightmap);
                ClipmapMesh {
                    level: level.level,
                    mesh,
                }
            })
            .collect()
    }

    /// Generate mesh for a single level
    pub fn generate_level_mesh(&self, level_idx: u32, heightmap: &Heightmap) -> Option<ClipmapMesh> {
        self.levels.get(level_idx as usize).map(|level| {
            let mesh = generate_level_mesh(level, heightmap);
            ClipmapMesh {
                level: level.level,
                mesh,
            }
        })
    }

    /// Number of LOD levels
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Total vertex count across all levels
    pub fn total_vertices(&self) -> usize {
        self.levels.len() * (self.grid_resolution * self.grid_resolution) as usize
    }
}

/// Generate a terrain mesh for a single clipmap level
fn generate_level_mesh(level: &ClipmapLevel, heightmap: &Heightmap) -> Mesh {
    let res = level.resolution;
    let spacing = level.spacing;
    let mut vertices = Vec::with_capacity((res * res) as usize);
    let mut indices = Vec::with_capacity(((res - 1) * (res - 1) * 6) as usize);

    // Generate vertices
    for gz in 0..res {
        for gx in 0..res {
            let wx = level.origin_x + gx as f32 * spacing;
            let wz = level.origin_z + gz as f32 * spacing;
            let wy = heightmap.sample(wx, wz);
            let normal = heightmap.normal_at(wx, wz);

            let uv_x = gx as f32 / (res - 1) as f32;
            let uv_z = gz as f32 / (res - 1) as f32;

            vertices.push(Vertex {
                position: Vec3::new(wx, wy, wz),
                normal,
                uv: glam::Vec2::new(uv_x, uv_z),
                ..Default::default()
            });
        }
    }

    // Generate indices (two triangles per quad)
    for gz in 0..(res - 1) {
        for gx in 0..(res - 1) {
            let i00 = gz * res + gx;
            let i10 = gz * res + gx + 1;
            let i01 = (gz + 1) * res + gx;
            let i11 = (gz + 1) * res + gx + 1;

            indices.push(i00);
            indices.push(i01);
            indices.push(i10);

            indices.push(i10);
            indices.push(i01);
            indices.push(i11);
        }
    }

    Mesh { vertices, indices }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_heightmap() -> Heightmap {
        let mut hm = Heightmap::new(32, 32, 100.0, 100.0);
        hm.generate_fbm(3, 0.5, 2.0, 42);
        hm.normalize();
        hm.scale_heights(10.0);
        hm
    }

    #[test]
    fn test_clipmap_new() {
        let clipmap = ClipmapTerrain::new(4, 32, 1.0);
        assert_eq!(clipmap.level_count(), 4);
        assert_eq!(clipmap.levels[0].spacing, 1.0);
        assert_eq!(clipmap.levels[1].spacing, 2.0);
        assert_eq!(clipmap.levels[2].spacing, 4.0);
        assert_eq!(clipmap.levels[3].spacing, 8.0);
    }

    #[test]
    fn test_clipmap_update() {
        let mut clipmap = ClipmapTerrain::new(3, 16, 1.0);
        clipmap.update(Vec3::new(50.0, 0.0, 50.0));

        // Origins should be snapped to grid
        for level in &clipmap.levels {
            let remainder_x = level.origin_x % level.spacing;
            let remainder_z = level.origin_z % level.spacing;
            assert!(remainder_x.abs() < 0.001, "Origin X not snapped at level {}", level.level);
            assert!(remainder_z.abs() < 0.001, "Origin Z not snapped at level {}", level.level);
        }
    }

    #[test]
    fn test_generate_meshes() {
        let hm = make_test_heightmap();
        let mut clipmap = ClipmapTerrain::new(3, 8, 1.0);
        clipmap.update(Vec3::new(50.0, 0.0, 50.0));

        let meshes = clipmap.generate_meshes(&hm);
        assert_eq!(meshes.len(), 3);

        for cm in &meshes {
            assert_eq!(cm.mesh.vertices.len(), 64); // 8x8
            assert_eq!(cm.mesh.indices.len(), 7 * 7 * 6); // (8-1)^2 * 6
        }
    }

    #[test]
    fn test_level_mesh_has_normals() {
        let hm = make_test_heightmap();
        let mut clipmap = ClipmapTerrain::new(1, 8, 2.0);
        clipmap.update(Vec3::new(50.0, 0.0, 50.0));

        let meshes = clipmap.generate_meshes(&hm);
        for v in &meshes[0].mesh.vertices {
            let len = v.normal.length();
            assert!((len - 1.0).abs() < 0.1, "Normal should be unit length, got {}", len);
        }
    }

    #[test]
    fn test_total_vertices() {
        let clipmap = ClipmapTerrain::new(4, 32, 1.0);
        assert_eq!(clipmap.total_vertices(), 4 * 32 * 32);
    }

    #[test]
    fn test_generate_single_level() {
        let hm = make_test_heightmap();
        let mut clipmap = ClipmapTerrain::new(3, 8, 1.0);
        clipmap.update(Vec3::new(50.0, 0.0, 50.0));

        let mesh = clipmap.generate_level_mesh(1, &hm);
        assert!(mesh.is_some());
        assert_eq!(mesh.unwrap().mesh.vertices.len(), 64);

        let invalid = clipmap.generate_level_mesh(99, &hm);
        assert!(invalid.is_none());
    }
}
