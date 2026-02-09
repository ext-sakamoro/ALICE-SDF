//! Chunked incremental mesh cache (Deep Fried Edition)
//!
//! Provides spatial partitioning of SDF evaluation space into chunks.
//! Only dirty chunks are re-meshed when the SDF changes, enabling
//! incremental updates for real-time editing workflows.
//!
//! # Design
//! - Space is divided into a uniform grid of chunks
//! - Each chunk is independently meshable
//! - Dirty tracking: only modified chunks are re-generated
//! - Chunk meshes can be persisted individually as .abm files
//!
//! Author: Moroya Sakamoto

use crate::io::abm::{load_abm, save_abm};
use crate::io::IoError;
use crate::mesh::{Mesh, Vertex};
use glam::Vec3;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// ChunkCoord
// ---------------------------------------------------------------------------

/// 3D chunk coordinate in the spatial grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCoord {
    /// X index
    pub x: i32,
    /// Y index
    pub y: i32,
    /// Z index
    pub z: i32,
}

impl ChunkCoord {
    /// Create a new chunk coordinate.
    #[inline(always)]
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

impl std::fmt::Display for ChunkCoord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

// ---------------------------------------------------------------------------
// ChunkedCacheConfig
// ---------------------------------------------------------------------------

/// Configuration for the chunked mesh cache.
#[derive(Debug, Clone)]
pub struct ChunkedCacheConfig {
    /// Size of each chunk in world units.
    pub chunk_size: f32,
    /// Resolution (voxels per axis) used when meshing a single chunk.
    pub chunk_resolution: usize,
    /// Optional disk persistence directory for `.abm` files.
    pub cache_dir: Option<PathBuf>,
    /// Maximum number of chunk meshes kept in memory.
    pub max_cached_chunks: usize,
}

impl Default for ChunkedCacheConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1.0,
            chunk_resolution: 32,
            cache_dir: None,
            max_cached_chunks: 512,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: ChunkEntry
// ---------------------------------------------------------------------------

/// A single cached chunk entry.
struct ChunkEntry {
    /// The generated mesh for this chunk.
    mesh: Arc<Mesh>,
    /// Hash of the SDF at the time this mesh was generated.
    sdf_hash: u64,
    /// Whether this entry has been modified since last persist.
    dirty: bool,
}

// ---------------------------------------------------------------------------
// ChunkedMeshCache
// ---------------------------------------------------------------------------

/// Thread-safe chunked incremental mesh cache.
///
/// Divides the SDF evaluation space into a uniform grid of chunks. Each
/// chunk is independently cached and can be marked dirty when the underlying
/// SDF changes — either globally or within a specific region.
pub struct ChunkedMeshCache {
    config: ChunkedCacheConfig,
    chunks: RwLock<HashMap<ChunkCoord, ChunkEntry>>,
    /// Current global SDF hash used for invalidation.
    current_sdf_hash: RwLock<u64>,
}

impl ChunkedMeshCache {
    /// Create a new chunked mesh cache.
    pub fn new(config: ChunkedCacheConfig) -> Self {
        Self {
            config,
            chunks: RwLock::new(HashMap::new()),
            current_sdf_hash: RwLock::new(0),
        }
    }

    // -- Query ---------------------------------------------------------------

    /// Get the chunk mesh at the given coordinate, if cached and valid.
    pub fn get_chunk(&self, coord: &ChunkCoord) -> Option<Arc<Mesh>> {
        let chunks = self.chunks.read().unwrap();
        chunks.get(coord).map(|e| Arc::clone(&e.mesh))
    }

    /// Insert or update a chunk mesh.
    pub fn set_chunk(&self, coord: ChunkCoord, mesh: Mesh, sdf_hash: u64) {
        let mut chunks = self.chunks.write().unwrap();

        // If we are at capacity and the coord is new, evict nothing (HashMap
        // simply grows).  A hard cap is checked here to avoid unbounded memory.
        if chunks.len() >= self.config.max_cached_chunks && !chunks.contains_key(&coord) {
            // Remove an arbitrary entry (effectively random eviction).
            if let Some(&evict_key) = chunks.keys().next() {
                chunks.remove(&evict_key);
            }
        }

        chunks.insert(
            coord,
            ChunkEntry {
                mesh: Arc::new(mesh),
                sdf_hash,
                dirty: true,
            },
        );
    }

    // -- Invalidation --------------------------------------------------------

    /// Mark **all** chunks as dirty (e.g. the SDF changed globally).
    pub fn invalidate_all(&self) {
        let mut chunks = self.chunks.write().unwrap();
        for entry in chunks.values_mut() {
            entry.dirty = true;
        }
    }

    /// Mark chunks overlapping a world-space bounding box as dirty.
    pub fn invalidate_region(&self, min: Vec3, max: Vec3) {
        let coords = self.chunks_in_bounds(min, max);
        let mut chunks = self.chunks.write().unwrap();
        for c in &coords {
            if let Some(entry) = chunks.get_mut(c) {
                entry.dirty = true;
            }
        }
    }

    /// Get the coordinates of all currently dirty chunks.
    pub fn dirty_chunks(&self) -> Vec<ChunkCoord> {
        let chunks = self.chunks.read().unwrap();
        chunks
            .iter()
            .filter(|(_, e)| e.dirty)
            .map(|(&c, _)| c)
            .collect()
    }

    /// Update the global SDF hash. Returns the list of chunks that were
    /// invalidated (those whose stored hash differs from `new_hash`).
    pub fn update_sdf_hash(&self, new_hash: u64) -> Vec<ChunkCoord> {
        {
            let mut h = self.current_sdf_hash.write().unwrap();
            *h = new_hash;
        }

        let mut invalidated = Vec::new();
        let mut chunks = self.chunks.write().unwrap();
        for (&coord, entry) in chunks.iter_mut() {
            if entry.sdf_hash != new_hash {
                entry.dirty = true;
                invalidated.push(coord);
            }
        }
        invalidated
    }

    // -- Enumeration ---------------------------------------------------------

    /// Get all cached chunk coordinates.
    pub fn cached_chunks(&self) -> Vec<ChunkCoord> {
        let chunks = self.chunks.read().unwrap();
        chunks.keys().copied().collect()
    }

    /// Number of cached chunks.
    #[inline]
    pub fn chunk_count(&self) -> usize {
        let chunks = self.chunks.read().unwrap();
        chunks.len()
    }

    /// Total approximate memory usage of all cached chunk meshes (bytes).
    pub fn memory_usage(&self) -> usize {
        let chunks = self.chunks.read().unwrap();
        chunks
            .values()
            .map(|e| estimate_mesh_bytes(&e.mesh))
            .sum()
    }

    // -- Coordinate helpers --------------------------------------------------

    /// Convert a world-space position to its containing chunk coordinate.
    #[inline]
    pub fn world_to_chunk(&self, pos: Vec3) -> ChunkCoord {
        let s = self.config.chunk_size;
        ChunkCoord {
            x: pos.x.div_euclid(s).floor() as i32,
            y: pos.y.div_euclid(s).floor() as i32,
            z: pos.z.div_euclid(s).floor() as i32,
        }
    }

    /// Get the axis-aligned bounding box `(min, max)` for a chunk coordinate.
    #[inline]
    pub fn chunk_bounds(&self, coord: &ChunkCoord) -> (Vec3, Vec3) {
        let s = self.config.chunk_size;
        let min = Vec3::new(coord.x as f32 * s, coord.y as f32 * s, coord.z as f32 * s);
        let max = min + Vec3::splat(s);
        (min, max)
    }

    /// Compute all chunk coordinates required to cover the given bounding box.
    pub fn chunks_in_bounds(&self, min: Vec3, max: Vec3) -> Vec<ChunkCoord> {
        let lo = self.world_to_chunk(min);
        let hi = self.world_to_chunk(max);
        let mut coords = Vec::new();
        for x in lo.x..=hi.x {
            for y in lo.y..=hi.y {
                for z in lo.z..=hi.z {
                    coords.push(ChunkCoord::new(x, y, z));
                }
            }
        }
        coords
    }

    // -- Persistence ---------------------------------------------------------

    /// Persist all dirty chunks to disk as individual `.abm` files.
    ///
    /// Requires `cache_dir` to be set in the configuration. Returns the
    /// number of chunks written.
    pub fn persist_dirty(&self) -> Result<usize, IoError> {
        let dir = self.config.cache_dir.as_ref().ok_or_else(|| {
            IoError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "cache_dir not configured",
            ))
        })?;
        std::fs::create_dir_all(dir)?;

        let mut chunks = self.chunks.write().unwrap();
        let mut written = 0usize;
        for (&coord, entry) in chunks.iter_mut() {
            if entry.dirty {
                let path = dir.join(format!("{}_{}_{}_.abm", coord.x, coord.y, coord.z));
                save_abm(&entry.mesh, &path)?;
                entry.dirty = false;
                written += 1;
            }
        }
        Ok(written)
    }

    /// Load a single chunk from the disk cache. Returns `None` if the file
    /// does not exist.
    pub fn load_chunk(&self, coord: &ChunkCoord) -> Result<Option<Arc<Mesh>>, IoError> {
        let dir = self.config.cache_dir.as_ref().ok_or_else(|| {
            IoError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "cache_dir not configured",
            ))
        })?;
        let path = dir.join(format!("{}_{}_{}_.abm", coord.x, coord.y, coord.z));
        if !path.exists() {
            return Ok(None);
        }
        let mesh = load_abm(&path)?;
        let sdf_hash = *self.current_sdf_hash.read().unwrap();
        let arc = Arc::new(mesh);

        let mut chunks = self.chunks.write().unwrap();
        chunks.insert(
            *coord,
            ChunkEntry {
                mesh: Arc::clone(&arc),
                sdf_hash,
                dirty: false,
            },
        );
        Ok(Some(arc))
    }

    // -- Bulk operations -----------------------------------------------------

    /// Clear all cached chunks.
    pub fn clear(&self) {
        let mut chunks = self.chunks.write().unwrap();
        chunks.clear();
    }

    /// Merge all cached chunk meshes into a single combined [`Mesh`].
    ///
    /// Vertex positions are preserved in world space. Index offsets are
    /// adjusted so the resulting mesh is self-consistent.
    pub fn merge_all(&self) -> Mesh {
        let chunks = self.chunks.read().unwrap();
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        for entry in chunks.values() {
            let base = vertices.len() as u32;
            vertices.extend_from_slice(&entry.mesh.vertices);
            indices.extend(entry.mesh.indices.iter().map(|&i| i + base));
        }
        Mesh { vertices, indices }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Estimate mesh memory footprint in bytes.
#[inline(always)]
fn estimate_mesh_bytes(mesh: &Mesh) -> usize {
    let vertex_size = std::mem::size_of::<Vertex>();
    mesh.vertices.len() * vertex_size + mesh.indices.len() * std::mem::size_of::<u32>()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Vertex;
    use glam::Vec3;

    /// Helper: create a simple test mesh.
    fn make_mesh(num_verts: usize, num_tris: usize) -> Mesh {
        let vertices: Vec<Vertex> = (0..num_verts)
            .map(|i| {
                let f = i as f32;
                Vertex::new(Vec3::new(f, f, f), Vec3::Y)
            })
            .collect();
        let indices: Vec<u32> = (0..num_tris * 3).map(|i| (i % num_verts) as u32).collect();
        Mesh { vertices, indices }
    }

    #[test]
    fn test_world_to_chunk_positive() {
        let cache = ChunkedMeshCache::new(ChunkedCacheConfig {
            chunk_size: 2.0,
            ..Default::default()
        });
        let c = cache.world_to_chunk(Vec3::new(3.5, 1.0, 5.9));
        assert_eq!(c, ChunkCoord::new(1, 0, 2));
    }

    #[test]
    fn test_world_to_chunk_negative() {
        let cache = ChunkedMeshCache::new(ChunkedCacheConfig {
            chunk_size: 1.0,
            ..Default::default()
        });
        let c = cache.world_to_chunk(Vec3::new(-0.5, -1.5, -0.01));
        assert_eq!(c, ChunkCoord::new(-1, -2, -1));
    }

    #[test]
    fn test_world_to_chunk_on_boundary() {
        let cache = ChunkedMeshCache::new(ChunkedCacheConfig {
            chunk_size: 1.0,
            ..Default::default()
        });
        // Exactly on a boundary should map to the lower chunk.
        let c = cache.world_to_chunk(Vec3::new(1.0, 0.0, -1.0));
        assert_eq!(c, ChunkCoord::new(1, 0, -1));
    }

    #[test]
    fn test_chunk_bounds() {
        let cache = ChunkedMeshCache::new(ChunkedCacheConfig {
            chunk_size: 2.0,
            ..Default::default()
        });
        let (min, max) = cache.chunk_bounds(&ChunkCoord::new(1, -1, 0));
        assert!((min.x - 2.0).abs() < 1e-6);
        assert!((min.y - (-2.0)).abs() < 1e-6);
        assert!((min.z - 0.0).abs() < 1e-6);
        assert!((max.x - 4.0).abs() < 1e-6);
        assert!((max.y - 0.0).abs() < 1e-6);
        assert!((max.z - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_chunks_in_bounds() {
        let cache = ChunkedMeshCache::new(ChunkedCacheConfig {
            chunk_size: 1.0,
            ..Default::default()
        });
        let coords = cache.chunks_in_bounds(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.5, 0.5, 0.5));
        // Should cover (0,0,0) and (1,0,0)
        assert!(coords.contains(&ChunkCoord::new(0, 0, 0)));
        assert!(coords.contains(&ChunkCoord::new(1, 0, 0)));
    }

    #[test]
    fn test_insert_and_get() {
        let cache = ChunkedMeshCache::new(Default::default());
        let coord = ChunkCoord::new(0, 0, 0);
        let mesh = make_mesh(4, 2);

        cache.set_chunk(coord, mesh, 42);
        let retrieved = cache.get_chunk(&coord);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().vertex_count(), 4);
    }

    #[test]
    fn test_get_missing_returns_none() {
        let cache = ChunkedMeshCache::new(Default::default());
        assert!(cache.get_chunk(&ChunkCoord::new(99, 99, 99)).is_none());
    }

    #[test]
    fn test_invalidate_region() {
        let cache = ChunkedMeshCache::new(ChunkedCacheConfig {
            chunk_size: 1.0,
            ..Default::default()
        });

        // Insert several chunks and mark them clean.
        for x in 0..3 {
            for z in 0..3 {
                cache.set_chunk(ChunkCoord::new(x, 0, z), make_mesh(4, 2), 1);
            }
        }
        // Manually mark all clean first.
        {
            let mut chunks = cache.chunks.write().unwrap();
            for e in chunks.values_mut() {
                e.dirty = false;
            }
        }

        // Invalidate a region covering chunk (1,0,1) only.
        cache.invalidate_region(Vec3::new(1.1, 0.0, 1.1), Vec3::new(1.9, 0.9, 1.9));

        let dirty = cache.dirty_chunks();
        assert_eq!(dirty.len(), 1);
        assert_eq!(dirty[0], ChunkCoord::new(1, 0, 1));
    }

    #[test]
    fn test_invalidate_all() {
        let cache = ChunkedMeshCache::new(Default::default());
        cache.set_chunk(ChunkCoord::new(0, 0, 0), make_mesh(4, 2), 1);
        cache.set_chunk(ChunkCoord::new(1, 0, 0), make_mesh(4, 2), 1);
        // Mark clean.
        {
            let mut chunks = cache.chunks.write().unwrap();
            for e in chunks.values_mut() {
                e.dirty = false;
            }
        }
        assert_eq!(cache.dirty_chunks().len(), 0);

        cache.invalidate_all();
        assert_eq!(cache.dirty_chunks().len(), 2);
    }

    #[test]
    fn test_merge_all() {
        let cache = ChunkedMeshCache::new(Default::default());
        // Chunk A: 4 verts, indices [0,1,2, 1,2,3]
        let mut mesh_a = make_mesh(4, 0);
        mesh_a.indices = vec![0, 1, 2, 1, 2, 3];
        cache.set_chunk(ChunkCoord::new(0, 0, 0), mesh_a, 1);

        // Chunk B: 3 verts, indices [0,1,2]
        let mut mesh_b = make_mesh(3, 0);
        mesh_b.indices = vec![0, 1, 2];
        cache.set_chunk(ChunkCoord::new(1, 0, 0), mesh_b, 1);

        let merged = cache.merge_all();
        assert_eq!(merged.vertices.len(), 7); // 4 + 3
        assert_eq!(merged.indices.len(), 9); // 6 + 3

        // Indices from chunk B should be offset by 4 (or by the vertex count
        // of whatever chunk was iterated first -- HashMap order is unspecified).
        // Just verify all indices are in-bounds.
        for &idx in &merged.indices {
            assert!((idx as usize) < merged.vertices.len());
        }
    }

    #[test]
    fn test_memory_usage() {
        let cache = ChunkedMeshCache::new(Default::default());
        assert_eq!(cache.memory_usage(), 0);

        let mesh = make_mesh(10, 4);
        let expected =
            10 * std::mem::size_of::<Vertex>() + 12 * std::mem::size_of::<u32>();
        cache.set_chunk(ChunkCoord::new(0, 0, 0), mesh, 1);
        assert_eq!(cache.memory_usage(), expected);
    }

    #[test]
    fn test_update_sdf_hash_invalidation() {
        let cache = ChunkedMeshCache::new(Default::default());
        cache.set_chunk(ChunkCoord::new(0, 0, 0), make_mesh(4, 2), 10);
        cache.set_chunk(ChunkCoord::new(1, 0, 0), make_mesh(4, 2), 10);
        cache.set_chunk(ChunkCoord::new(2, 0, 0), make_mesh(4, 2), 20);
        // Mark clean.
        {
            let mut chunks = cache.chunks.write().unwrap();
            for e in chunks.values_mut() {
                e.dirty = false;
            }
        }

        // Update hash to 20 — chunks with hash 10 should be invalidated.
        let invalidated = cache.update_sdf_hash(20);
        assert_eq!(invalidated.len(), 2);
        // Chunk (2,0,0) already matches hash 20, so NOT invalidated.
        assert!(!invalidated.contains(&ChunkCoord::new(2, 0, 0)));
    }

    #[test]
    fn test_clear() {
        let cache = ChunkedMeshCache::new(Default::default());
        cache.set_chunk(ChunkCoord::new(0, 0, 0), make_mesh(4, 2), 1);
        cache.set_chunk(ChunkCoord::new(1, 0, 0), make_mesh(4, 2), 1);
        assert_eq!(cache.chunk_count(), 2);

        cache.clear();
        assert_eq!(cache.chunk_count(), 0);
        assert!(cache.get_chunk(&ChunkCoord::new(0, 0, 0)).is_none());
    }

    #[test]
    fn test_cached_chunks() {
        let cache = ChunkedMeshCache::new(Default::default());
        cache.set_chunk(ChunkCoord::new(0, 0, 0), make_mesh(4, 2), 1);
        cache.set_chunk(ChunkCoord::new(1, 2, 3), make_mesh(4, 2), 1);

        let cached = cache.cached_chunks();
        assert_eq!(cached.len(), 2);
        assert!(cached.contains(&ChunkCoord::new(0, 0, 0)));
        assert!(cached.contains(&ChunkCoord::new(1, 2, 3)));
    }

    #[test]
    fn test_persist_dirty_and_load_chunk() {
        let dir = std::env::temp_dir().join("alice_sdf_chunked_test");
        let _ = std::fs::remove_dir_all(&dir);

        let config = ChunkedCacheConfig {
            cache_dir: Some(dir.clone()),
            ..Default::default()
        };
        let cache = ChunkedMeshCache::new(config);

        // Insert a mesh and persist.
        let coord = ChunkCoord::new(1, -2, 3);
        cache.set_chunk(coord, make_mesh(6, 3), 42);
        let written = cache.persist_dirty().unwrap();
        assert_eq!(written, 1);

        // Dirty should now be empty.
        assert_eq!(cache.dirty_chunks().len(), 0);

        // Create a fresh cache and load from disk.
        let config2 = ChunkedCacheConfig {
            cache_dir: Some(dir.clone()),
            ..Default::default()
        };
        let cache2 = ChunkedMeshCache::new(config2);
        let loaded = cache2.load_chunk(&coord).unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().vertex_count(), 6);

        // Cleanup.
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_persist_without_cache_dir_errors() {
        let cache = ChunkedMeshCache::new(ChunkedCacheConfig {
            cache_dir: None,
            ..Default::default()
        });
        cache.set_chunk(ChunkCoord::new(0, 0, 0), make_mesh(4, 2), 1);
        assert!(cache.persist_dirty().is_err());
    }

    #[test]
    fn test_max_cached_chunks_eviction() {
        let cache = ChunkedMeshCache::new(ChunkedCacheConfig {
            max_cached_chunks: 3,
            ..Default::default()
        });
        for i in 0..5 {
            cache.set_chunk(ChunkCoord::new(i, 0, 0), make_mesh(4, 2), 1);
        }
        // Should never exceed max.
        assert!(cache.chunk_count() <= 3);
    }
}
