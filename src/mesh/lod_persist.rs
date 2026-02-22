//! LOD chain persistence (Deep Fried Edition)
//!
//! Serializes and deserializes complete LOD chains including
//! transition distances and AABB bounds. Integrates with the
//! ABM format and the existing LOD generation pipeline.
//!
//! Author: Moroya Sakamoto

use crate::io::abm::{load_abm_with_lods, save_abm_with_lods};
use crate::io::IoError;
use crate::mesh::Mesh;
use serde::{Deserialize, Serialize};
use std::path::Path;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A complete LOD chain with metadata for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LodChainPersist {
    /// LOD meshes, index 0 = highest detail
    pub meshes: Vec<LodLevelPersist>,
    /// Source SDF hash for cache invalidation
    pub sdf_hash: u64,
    /// Generation parameters
    pub config: LodChainConfig,
}

/// A single LOD level with persistence metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LodLevelPersist {
    /// The mesh at this LOD
    pub mesh: Mesh,
    /// Screen-space transition distance (when to switch to this LOD)
    pub transition_distance: f32,
    /// AABB min bounds
    pub aabb_min: [f32; 3],
    /// AABB max bounds
    pub aabb_max: [f32; 3],
    /// Vertex count (for quick query without loading mesh)
    pub vertex_count: u32,
    /// Triangle count
    pub triangle_count: u32,
    /// Reduction ratio from LOD 0 (1.0 for LOD 0, <1.0 for others)
    pub reduction_ratio: f32,
}

/// Configuration used to generate the LOD chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LodChainConfig {
    /// Number of LOD levels
    pub lod_count: u32,
    /// Base resolution for LOD 0
    pub base_resolution: u32,
    /// Resolution reduction factor per LOD level (e.g. 0.5)
    pub resolution_factor: f32,
    /// Target triangle reduction per LOD (e.g. 0.5 = 50% reduction each level)
    pub triangle_reduction: f32,
    /// Transition distance multiplier
    pub distance_multiplier: f32,
}

impl Default for LodChainConfig {
    fn default() -> Self {
        Self {
            lod_count: 4,
            base_resolution: 128,
            resolution_factor: 0.5,
            triangle_reduction: 0.5,
            distance_multiplier: 2.0,
        }
    }
}

/// Summary statistics for a LOD chain
#[derive(Debug, Clone)]
pub struct LodChainSummary {
    /// Number of LOD levels
    pub level_count: usize,
    /// Total vertices across all LODs
    pub total_vertices: usize,
    /// Total triangles across all LODs
    pub total_triangles: usize,
    /// Total memory usage in bytes (approximate)
    pub total_memory_bytes: usize,
    /// Vertex count at LOD 0
    pub lod0_vertices: usize,
    /// Triangle count at LOD 0
    pub lod0_triangles: usize,
}

// ---------------------------------------------------------------------------
// AABB computation
// ---------------------------------------------------------------------------

/// Compute axis-aligned bounding box from mesh vertices
fn compute_aabb(mesh: &Mesh) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for v in &mesh.vertices {
        min[0] = min[0].min(v.position.x);
        min[1] = min[1].min(v.position.y);
        min[2] = min[2].min(v.position.z);
        max[0] = max[0].max(v.position.x);
        max[1] = max[1].max(v.position.y);
        max[2] = max[2].max(v.position.z);
    }
    (min, max)
}

// ---------------------------------------------------------------------------
// LodChainPersist implementation
// ---------------------------------------------------------------------------

impl LodChainPersist {
    /// Create a new LOD chain from a list of meshes and transition distances
    ///
    /// # Arguments
    /// * `meshes` - LOD meshes from highest to lowest detail
    /// * `transition_distances` - Screen-space distance for each LOD level.
    ///   Length must equal `meshes.len()`. The first entry (LOD 0) is typically 0.0.
    /// * `sdf_hash` - Hash of the source SDF for cache invalidation
    /// * `config` - Generation parameters
    pub fn new(
        meshes: Vec<Mesh>,
        transition_distances: Vec<f32>,
        sdf_hash: u64,
        config: LodChainConfig,
    ) -> Self {
        assert_eq!(
            meshes.len(),
            transition_distances.len(),
            "meshes and transition_distances must have the same length"
        );

        let lod0_tri_count = if meshes.is_empty() {
            0
        } else {
            meshes[0].triangle_count()
        };

        let levels: Vec<LodLevelPersist> = meshes
            .into_iter()
            .enumerate()
            .zip(transition_distances)
            .map(|((i, mesh), dist)| {
                let (aabb_min, aabb_max) = if mesh.vertices.is_empty() {
                    ([0.0; 3], [0.0; 3])
                } else {
                    compute_aabb(&mesh)
                };
                let vertex_count = mesh.vertices.len() as u32;
                let triangle_count = mesh.triangle_count() as u32;
                let reduction_ratio = if i == 0 || lod0_tri_count == 0 {
                    1.0
                } else {
                    triangle_count as f32 / lod0_tri_count as f32
                };
                LodLevelPersist {
                    mesh,
                    transition_distance: dist,
                    aabb_min,
                    aabb_max,
                    vertex_count,
                    triangle_count,
                    reduction_ratio,
                }
            })
            .collect();

        LodChainPersist {
            meshes: levels,
            sdf_hash,
            config,
        }
    }

    /// Get the appropriate LOD level index for a given screen-space distance.
    ///
    /// Returns the highest-detail LOD whose transition distance is <= the given
    /// distance. Falls back to the last (lowest detail) level if distance
    /// exceeds all thresholds.
    pub fn select_lod(&self, distance: f32) -> usize {
        // Find the last (highest index) LOD whose transition_distance <= distance.
        // rposition scans from the end, so for sorted transition distances this
        // is effectively O(1) for the common case of selecting the coarsest LOD.
        self.meshes
            .iter()
            .rposition(|l| distance >= l.transition_distance)
            .unwrap_or(0)
    }

    /// Get the mesh at a specific LOD level
    pub fn mesh(&self, lod: usize) -> Option<&Mesh> {
        self.meshes.get(lod).map(|l| &l.mesh)
    }

    /// Number of LOD levels
    pub fn level_count(&self) -> usize {
        self.meshes.len()
    }

    /// Total memory usage across all LODs (approximate, in bytes)
    pub fn total_memory_bytes(&self) -> usize {
        self.meshes
            .iter()
            .map(|l| {
                l.mesh.vertices.len() * std::mem::size_of::<crate::mesh::Vertex>()
                    + l.mesh.indices.len() * std::mem::size_of::<u32>()
            })
            .sum()
    }

    /// Summary statistics
    pub fn summary(&self) -> LodChainSummary {
        let total_vertices: usize = self.meshes.iter().map(|l| l.vertex_count as usize).sum();
        let total_triangles: usize = self.meshes.iter().map(|l| l.triangle_count as usize).sum();
        let total_memory_bytes = self.total_memory_bytes();
        let (lod0_vertices, lod0_triangles) = self
            .meshes
            .first()
            .map(|l| (l.vertex_count as usize, l.triangle_count as usize))
            .unwrap_or((0, 0));

        LodChainSummary {
            level_count: self.meshes.len(),
            total_vertices,
            total_triangles,
            total_memory_bytes,
            lod0_vertices,
            lod0_triangles,
        }
    }
}

// ---------------------------------------------------------------------------
// Sidecar metadata (JSON)
// ---------------------------------------------------------------------------

/// Metadata stored in the `.abm.meta.json` sidecar file
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LodChainMetadata {
    /// Source SDF hash for cache invalidation
    sdf_hash: u64,
    /// Generation configuration
    config: LodChainConfig,
    /// Per-level statistics
    levels: Vec<LodLevelMetadata>,
}

/// Per-level metadata in the sidecar file
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LodLevelMetadata {
    /// LOD level index
    level: usize,
    /// Transition distance
    transition_distance: f32,
    /// AABB min
    aabb_min: [f32; 3],
    /// AABB max
    aabb_max: [f32; 3],
    /// Vertex count
    vertex_count: u32,
    /// Triangle count
    triangle_count: u32,
    /// Reduction ratio from LOD 0
    reduction_ratio: f32,
}

// ---------------------------------------------------------------------------
// Public API: save / load
// ---------------------------------------------------------------------------

/// Save LOD chain to ABM file with a JSON metadata sidecar.
///
/// The mesh data is stored in the ABM binary format via `save_abm_with_lods`.
/// A `.abm.meta.json` sidecar file is written alongside for metadata
/// (sdf_hash, config, per-level stats).
pub fn save_lod_chain(chain: &LodChainPersist, path: impl AsRef<Path>) -> Result<(), IoError> {
    let path = path.as_ref();

    // Extract meshes and transition distances for ABM
    let meshes: Vec<Mesh> = chain.meshes.iter().map(|l| l.mesh.clone()).collect();

    // save_abm_with_lods expects transition_distances of length meshes.len() - 1
    // (distances for LOD levels after the first)
    let transition_distances: Vec<f32> = if chain.meshes.len() > 1 {
        chain.meshes[1..]
            .iter()
            .map(|l| l.transition_distance)
            .collect()
    } else {
        Vec::new()
    };

    save_abm_with_lods(&meshes, &transition_distances, path)?;

    // Write sidecar metadata
    save_lod_chain_metadata(chain, path)?;

    Ok(())
}

/// Load LOD chain from ABM file and its JSON metadata sidecar.
pub fn load_lod_chain(path: impl AsRef<Path>) -> Result<LodChainPersist, IoError> {
    let path = path.as_ref();

    // Load meshes and transition distances from ABM
    let (meshes, transition_distances) = load_abm_with_lods(path)?;

    // Load sidecar metadata
    let meta_path = sidecar_path(path);
    let config = load_lod_chain_metadata(&meta_path)?;

    // Read full metadata for per-level stats
    let metadata = read_sidecar_metadata(&meta_path)?;

    // Reconstruct LodChainPersist
    let lod0_tri_count = if meshes.is_empty() {
        0
    } else {
        meshes[0].triangle_count()
    };

    // Build transition_distances with LOD 0 = 0.0 prepended
    let mut full_distances = vec![0.0_f32];
    full_distances.extend_from_slice(&transition_distances);

    let levels: Vec<LodLevelPersist> = meshes
        .into_iter()
        .enumerate()
        .map(|(i, mesh)| {
            // Use sidecar metadata if available, otherwise recompute
            if let Some(level_meta) = metadata.levels.get(i) {
                LodLevelPersist {
                    vertex_count: mesh.vertices.len() as u32,
                    triangle_count: mesh.triangle_count() as u32,
                    transition_distance: level_meta.transition_distance,
                    aabb_min: level_meta.aabb_min,
                    aabb_max: level_meta.aabb_max,
                    reduction_ratio: level_meta.reduction_ratio,
                    mesh,
                }
            } else {
                let (aabb_min, aabb_max) = if mesh.vertices.is_empty() {
                    ([0.0; 3], [0.0; 3])
                } else {
                    compute_aabb(&mesh)
                };
                let tri_count = mesh.triangle_count();
                let reduction_ratio = if i == 0 || lod0_tri_count == 0 {
                    1.0
                } else {
                    tri_count as f32 / lod0_tri_count as f32
                };
                LodLevelPersist {
                    vertex_count: mesh.vertices.len() as u32,
                    triangle_count: tri_count as u32,
                    transition_distance: *full_distances.get(i).unwrap_or(&0.0),
                    aabb_min,
                    aabb_max,
                    reduction_ratio,
                    mesh,
                }
            }
        })
        .collect();

    Ok(LodChainPersist {
        meshes: levels,
        sdf_hash: metadata.sdf_hash,
        config,
    })
}

/// Save LOD chain metadata as JSON sidecar (`.abm.meta.json`)
pub fn save_lod_chain_metadata(
    chain: &LodChainPersist,
    path: impl AsRef<Path>,
) -> Result<(), IoError> {
    let meta_path = sidecar_path(path.as_ref());

    let levels: Vec<LodLevelMetadata> = chain
        .meshes
        .iter()
        .enumerate()
        .map(|(i, l)| LodLevelMetadata {
            level: i,
            transition_distance: l.transition_distance,
            aabb_min: l.aabb_min,
            aabb_max: l.aabb_max,
            vertex_count: l.vertex_count,
            triangle_count: l.triangle_count,
            reduction_ratio: l.reduction_ratio,
        })
        .collect();

    let metadata = LodChainMetadata {
        sdf_hash: chain.sdf_hash,
        config: chain.config.clone(),
        levels,
    };

    let file = std::fs::File::create(&meta_path)?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &metadata).map_err(|e| {
        IoError::Serialization(format!("Failed to write LOD chain metadata: {}", e))
    })?;

    Ok(())
}

/// Load LOD chain configuration from JSON sidecar
pub fn load_lod_chain_metadata(path: impl AsRef<Path>) -> Result<LodChainConfig, IoError> {
    let metadata = read_sidecar_metadata(path)?;
    Ok(metadata.config)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the sidecar path for a given ABM file path
fn sidecar_path(abm_path: &Path) -> std::path::PathBuf {
    let mut meta_path = abm_path.as_os_str().to_owned();
    meta_path.push(".meta.json");
    std::path::PathBuf::from(meta_path)
}

/// Read sidecar metadata from JSON file
fn read_sidecar_metadata(path: impl AsRef<Path>) -> Result<LodChainMetadata, IoError> {
    let file = std::fs::File::open(path.as_ref())?;
    let reader = std::io::BufReader::new(file);
    serde_json::from_reader(reader)
        .map_err(|e| IoError::Serialization(format!("Failed to read LOD chain metadata: {}", e)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Vertex;
    use glam::{Vec2, Vec3, Vec4};

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("alice_sdf_lod_persist_{}", name));
        path
    }

    /// Build a simple triangle mesh with N copies of a triangle
    fn make_test_mesh(num_triangles: usize) -> Mesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for i in 0..num_triangles {
            let base = (i * 3) as u32;
            let offset = i as f32 * 2.0;
            vertices.push(Vertex::with_all(
                Vec3::new(offset, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec2::new(0.0, 0.0),
                Vec2::ZERO,
                Vec4::new(1.0, 0.0, 0.0, 1.0),
                [1.0, 1.0, 1.0, 1.0],
                0,
            ));
            vertices.push(Vertex::with_all(
                Vec3::new(offset + 1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec2::new(1.0, 0.0),
                Vec2::ZERO,
                Vec4::new(1.0, 0.0, 0.0, 1.0),
                [1.0, 1.0, 1.0, 1.0],
                0,
            ));
            vertices.push(Vertex::with_all(
                Vec3::new(offset + 0.5, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec2::new(0.5, 1.0),
                Vec2::ZERO,
                Vec4::new(1.0, 0.0, 0.0, 1.0),
                [1.0, 1.0, 1.0, 1.0],
                0,
            ));
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
        }

        Mesh { vertices, indices }
    }

    #[test]
    fn test_lod_chain_creation() {
        let lod0 = make_test_mesh(100);
        let lod1 = make_test_mesh(50);
        let lod2 = make_test_mesh(25);

        let chain = LodChainPersist::new(
            vec![lod0, lod1, lod2],
            vec![0.0, 10.0, 20.0],
            0xDEADBEEF,
            LodChainConfig::default(),
        );

        assert_eq!(chain.level_count(), 3);
        assert_eq!(chain.meshes[0].vertex_count, 300);
        assert_eq!(chain.meshes[0].triangle_count, 100);
        assert_eq!(chain.meshes[1].triangle_count, 50);
        assert_eq!(chain.meshes[2].triangle_count, 25);
        assert_eq!(chain.sdf_hash, 0xDEADBEEF);
    }

    #[test]
    fn test_select_lod() {
        let lod0 = make_test_mesh(100);
        let lod1 = make_test_mesh(50);
        let lod2 = make_test_mesh(25);

        let chain = LodChainPersist::new(
            vec![lod0, lod1, lod2],
            vec![0.0, 10.0, 20.0],
            0,
            LodChainConfig::default(),
        );

        // Distance 0 -> LOD 0
        assert_eq!(chain.select_lod(0.0), 0);
        // Distance 5 -> still LOD 0 (no higher threshold exceeded)
        assert_eq!(chain.select_lod(5.0), 0);
        // Distance 10 -> LOD 1
        assert_eq!(chain.select_lod(10.0), 1);
        // Distance 15 -> still LOD 1
        assert_eq!(chain.select_lod(15.0), 1);
        // Distance 20 -> LOD 2
        assert_eq!(chain.select_lod(20.0), 2);
        // Distance 100 -> LOD 2 (furthest)
        assert_eq!(chain.select_lod(100.0), 2);
    }

    #[test]
    fn test_mesh_accessor() {
        let lod0 = make_test_mesh(10);
        let chain = LodChainPersist::new(vec![lod0], vec![0.0], 0, LodChainConfig::default());

        assert!(chain.mesh(0).is_some());
        assert_eq!(chain.mesh(0).unwrap().triangle_count(), 10);
        assert!(chain.mesh(1).is_none());
    }

    #[test]
    fn test_aabb_computation() {
        let mesh = make_test_mesh(2);
        let (aabb_min, aabb_max) = compute_aabb(&mesh);

        // First triangle: (0,0,0) (1,0,0) (0.5,1,0)
        // Second triangle: (2,0,0) (3,0,0) (2.5,1,0)
        assert!(aabb_min[0] <= 0.0);
        assert!(aabb_min[1] <= 0.0);
        assert!(aabb_min[2] <= 0.0);
        assert!(aabb_max[0] >= 3.0);
        assert!(aabb_max[1] >= 1.0);
        assert!((aabb_max[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduction_ratio() {
        let lod0 = make_test_mesh(100);
        let lod1 = make_test_mesh(50);
        let lod2 = make_test_mesh(25);

        let chain = LodChainPersist::new(
            vec![lod0, lod1, lod2],
            vec![0.0, 10.0, 20.0],
            0,
            LodChainConfig::default(),
        );

        assert!((chain.meshes[0].reduction_ratio - 1.0).abs() < 1e-6);
        assert!((chain.meshes[1].reduction_ratio - 0.5).abs() < 1e-6);
        assert!((chain.meshes[2].reduction_ratio - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_memory_calculation() {
        let lod0 = make_test_mesh(10);
        let chain = LodChainPersist::new(vec![lod0], vec![0.0], 0, LodChainConfig::default());

        let mem = chain.total_memory_bytes();
        // 30 vertices * size_of::<Vertex>() + 30 indices * 4 bytes
        let expected = 30 * std::mem::size_of::<Vertex>() + 30 * std::mem::size_of::<u32>();
        assert_eq!(mem, expected);
    }

    #[test]
    fn test_summary_statistics() {
        let lod0 = make_test_mesh(100);
        let lod1 = make_test_mesh(50);

        let chain = LodChainPersist::new(
            vec![lod0, lod1],
            vec![0.0, 10.0],
            42,
            LodChainConfig::default(),
        );

        let summary = chain.summary();
        assert_eq!(summary.level_count, 2);
        assert_eq!(summary.total_vertices, 300 + 150);
        assert_eq!(summary.total_triangles, 100 + 50);
        assert_eq!(summary.lod0_vertices, 300);
        assert_eq!(summary.lod0_triangles, 100);
        assert!(summary.total_memory_bytes > 0);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let lod0 = make_test_mesh(10);
        let lod1 = make_test_mesh(5);
        let lod2 = make_test_mesh(2);

        let chain = LodChainPersist::new(
            vec![lod0, lod1, lod2],
            vec![0.0, 10.0, 25.0],
            0xCAFEBABE,
            LodChainConfig {
                lod_count: 3,
                base_resolution: 64,
                resolution_factor: 0.5,
                triangle_reduction: 0.5,
                distance_multiplier: 2.5,
            },
        );

        let path = temp_path("roundtrip.abm");

        save_lod_chain(&chain, &path).unwrap();
        let loaded = load_lod_chain(&path).unwrap();

        // Verify structure
        assert_eq!(loaded.level_count(), 3);
        assert_eq!(loaded.sdf_hash, 0xCAFEBABE);
        assert_eq!(loaded.config.lod_count, 3);
        assert_eq!(loaded.config.base_resolution, 64);
        assert!((loaded.config.distance_multiplier - 2.5).abs() < 1e-6);

        // Verify per-level data
        assert_eq!(loaded.meshes[0].vertex_count, 30);
        assert_eq!(loaded.meshes[0].triangle_count, 10);
        assert!((loaded.meshes[0].transition_distance - 0.0).abs() < 1e-6);
        assert!((loaded.meshes[0].reduction_ratio - 1.0).abs() < 1e-6);

        assert_eq!(loaded.meshes[1].vertex_count, 15);
        assert_eq!(loaded.meshes[1].triangle_count, 5);
        assert!((loaded.meshes[1].transition_distance - 10.0).abs() < 1e-6);

        assert_eq!(loaded.meshes[2].vertex_count, 6);
        assert_eq!(loaded.meshes[2].triangle_count, 2);
        assert!((loaded.meshes[2].transition_distance - 25.0).abs() < 1e-6);

        // Verify mesh vertex data round-trips
        let orig_pos = chain.meshes[0].mesh.vertices[0].position;
        let load_pos = loaded.meshes[0].mesh.vertices[0].position;
        assert!((orig_pos - load_pos).length() < 1e-6);

        // Cleanup
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(sidecar_path(&path)).ok();
    }

    #[test]
    fn test_save_load_single_lod() {
        let lod0 = make_test_mesh(8);

        let chain = LodChainPersist::new(vec![lod0], vec![0.0], 123, LodChainConfig::default());

        let path = temp_path("single_lod.abm");

        save_lod_chain(&chain, &path).unwrap();
        let loaded = load_lod_chain(&path).unwrap();

        assert_eq!(loaded.level_count(), 1);
        assert_eq!(loaded.meshes[0].triangle_count, 8);
        assert_eq!(loaded.sdf_hash, 123);

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(sidecar_path(&path)).ok();
    }

    #[test]
    fn test_metadata_sidecar_roundtrip() {
        let lod0 = make_test_mesh(10);
        let lod1 = make_test_mesh(5);

        let chain = LodChainPersist::new(
            vec![lod0, lod1],
            vec![0.0, 15.0],
            999,
            LodChainConfig {
                lod_count: 2,
                base_resolution: 256,
                resolution_factor: 0.6,
                triangle_reduction: 0.4,
                distance_multiplier: 3.0,
            },
        );

        let abm_path = temp_path("meta_test.abm");
        let meta_path = sidecar_path(&abm_path);

        save_lod_chain_metadata(&chain, &abm_path).unwrap();
        let config = load_lod_chain_metadata(&meta_path).unwrap();

        assert_eq!(config.lod_count, 2);
        assert_eq!(config.base_resolution, 256);
        assert!((config.resolution_factor - 0.6).abs() < 1e-6);
        assert!((config.triangle_reduction - 0.4).abs() < 1e-6);
        assert!((config.distance_multiplier - 3.0).abs() < 1e-6);

        std::fs::remove_file(&meta_path).ok();
    }

    #[test]
    fn test_aabb_stored_in_levels() {
        let lod0 = make_test_mesh(3);

        let chain = LodChainPersist::new(vec![lod0], vec![0.0], 0, LodChainConfig::default());

        let level = &chain.meshes[0];
        // 3 triangles at offsets 0, 2, 4 -> x range [0, 5.0], y range [0, 1.0]
        assert!(level.aabb_min[0] <= 0.0);
        assert!(level.aabb_min[1] <= 0.0);
        assert!(level.aabb_max[0] >= 5.0);
        assert!(level.aabb_max[1] >= 1.0);
    }

    #[test]
    fn test_default_config() {
        let config = LodChainConfig::default();
        assert_eq!(config.lod_count, 4);
        assert_eq!(config.base_resolution, 128);
        assert!((config.resolution_factor - 0.5).abs() < 1e-6);
        assert!((config.triangle_reduction - 0.5).abs() < 1e-6);
        assert!((config.distance_multiplier - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_chain() {
        let chain = LodChainPersist::new(vec![], vec![], 0, LodChainConfig::default());
        assert_eq!(chain.level_count(), 0);
        assert_eq!(chain.total_memory_bytes(), 0);
        let summary = chain.summary();
        assert_eq!(summary.level_count, 0);
        assert_eq!(summary.total_vertices, 0);
        assert_eq!(summary.total_triangles, 0);
        assert_eq!(summary.lod0_vertices, 0);
        assert_eq!(summary.lod0_triangles, 0);
    }
}
