//! Terrain System (Deep Fried Edition)
//!
//! Large-scale heightmap + SDF hybrid terrain with erosion simulation,
//! biome splatting, clipmap LOD, and cave/overhang generation.
//!
//! # Architecture
//!
//! - **Heightmap**: 2D grid of heights with bicubic interpolation
//! - **Clipmap**: Nested LOD grids centered on camera for efficient rendering
//! - **Splatmap**: Multi-layer material blending (up to 16 layers)
//! - **Erosion**: Particle-based hydraulic + thermal erosion
//! - **Caves**: SDF-based subterranean features combined with heightmap
//!
//! # Usage
//!
//! ```rust,ignore
//! use alice_sdf::terrain::*;
//!
//! let mut hm = Heightmap::new(256, 256, 100.0, 100.0);
//! hm.generate_fbm(6, 0.5, 2.0, 42);
//! hm.apply_erosion(&ErosionConfig::default());
//!
//! let clipmap = ClipmapTerrain::new(&hm, 4, 64);
//! let meshes = clipmap.generate_meshes(camera_pos);
//! ```
//!
//! Author: Moroya Sakamoto

pub mod caves;
pub mod clipmap;
pub mod erosion;
pub mod heightmap;
pub mod splatmap;

use glam::Vec3;

pub use caves::{generate_cave_sdf, CaveConfig};
pub use clipmap::{ClipmapLevel, ClipmapMesh, ClipmapTerrain};
pub use erosion::{erode, ErosionConfig};
pub use heightmap::Heightmap;
#[cfg(feature = "image")]
pub use heightmap::HeightmapImageConfig;
pub use splatmap::{SplatLayer, Splatmap};

/// Configuration for terrain generation
#[derive(Debug, Clone)]
pub struct TerrainConfig {
    /// Terrain width in world units (X axis)
    pub width: f32,
    /// Terrain depth in world units (Z axis)
    pub depth: f32,
    /// Maximum terrain height (Y axis)
    pub height_scale: f32,
    /// Heightmap resolution (width)
    pub resolution_x: u32,
    /// Heightmap resolution (depth)
    pub resolution_z: u32,
    /// Number of clipmap LOD levels
    pub clipmap_levels: u32,
    /// Vertices per clipmap level edge
    pub clipmap_resolution: u32,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        TerrainConfig {
            width: 1000.0,
            depth: 1000.0,
            height_scale: 100.0,
            resolution_x: 512,
            resolution_z: 512,
            clipmap_levels: 5,
            clipmap_resolution: 64,
        }
    }
}

/// Evaluate terrain as an SDF at a world position
///
/// Combines heightmap distance with optional cave SDF.
/// Returns signed distance (negative = inside terrain).
pub fn terrain_sdf(
    heightmap: &Heightmap,
    point: Vec3,
    cave_sdf: Option<&crate::types::SdfNode>,
) -> f32 {
    let height = heightmap.sample(point.x, point.z);
    let terrain_dist = point.y - height;

    match cave_sdf {
        Some(cave) => {
            let cave_dist = crate::eval::eval(cave, point);
            // Terrain is the base, caves subtract from it
            // Inside terrain = negative, inside cave = negative
            // Result: max(terrain_dist, -cave_dist) would carve caves
            // But we want: terrain_dist where no cave, carved where cave
            terrain_dist.max(-cave_dist)
        }
        None => terrain_dist,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terrain_config_default() {
        let config = TerrainConfig::default();
        assert_eq!(config.width, 1000.0);
        assert_eq!(config.depth, 1000.0);
        assert_eq!(config.height_scale, 100.0);
        assert_eq!(config.clipmap_levels, 5);
    }

    #[test]
    fn test_terrain_sdf_flat() {
        let hm = Heightmap::new(8, 8, 10.0, 10.0);
        // Flat heightmap at y=0, point above should be positive
        let d = terrain_sdf(&hm, Vec3::new(5.0, 1.0, 5.0), None);
        assert!(d > 0.0);

        // Point below should be negative
        let d = terrain_sdf(&hm, Vec3::new(5.0, -1.0, 5.0), None);
        assert!(d < 0.0);
    }

    #[test]
    fn test_terrain_sdf_with_cave() {
        let mut hm = Heightmap::new(8, 8, 10.0, 10.0);
        // Set all heights to 5.0
        for z in 0..8 {
            for x in 0..8 {
                hm.set_height(x, z, 5.0);
            }
        }

        // Cave sphere at (5, 2, 5) with radius 1
        let cave = crate::types::SdfNode::sphere(1.0).translate(5.0, 2.0, 5.0);
        let d = terrain_sdf(&hm, Vec3::new(5.0, 2.0, 5.0), Some(&cave));
        // Inside the cave should be positive (carved out)
        assert!(d > 0.0, "Inside cave should be positive, got {}", d);
    }
}
