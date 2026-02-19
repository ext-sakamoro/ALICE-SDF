//! SDF hash-based mesh cache with LRU eviction
//!
//! Maps SDF tree hashes to cached meshes for fast retrieval when the same
//! SDF is meshed repeatedly (e.g. editor preview, LOD regeneration).
//!
//! Also provides chunked incremental mesh caching via [`chunked`].
//!
//! # Features
//!
//! - **LRU eviction**: Oldest entries are evicted when the cache is full.
//! - **Thread-safe**: Uses `RwLock` for concurrent reads, exclusive writes.
//! - **Deterministic hashing**: Same SDF tree + bounds + resolution = same key.
//! - **Zero-copy sharing**: Cached meshes are returned as `Arc<Mesh>`.
//! - **`get_or_generate`**: Atomic check-and-insert to avoid redundant work.
//!
//! # Example
//!
//! ```rust
//! use alice_sdf::cache::{MeshCache, CacheConfig, compute_cache_key};
//! use alice_sdf::prelude::*;
//!
//! let cache = MeshCache::new(CacheConfig::default());
//! let sphere = SdfNode::sphere(1.0);
//! let min = Vec3::splat(-2.0);
//! let max = Vec3::splat(2.0);
//! let config = MarchingCubesConfig { resolution: 32, ..Default::default() };
//!
//! let key = compute_cache_key(&sphere, min, max, config.resolution);
//! let mesh = cache.get_or_generate(key, || {
//!     sdf_to_mesh(&sphere, min, max, &config)
//! });
//! ```
//!
//! Author: Moroya Sakamoto

pub mod chunked;
pub use chunked::{ChunkCoord, ChunkedCacheConfig, ChunkedMeshCache};

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::mesh::Mesh;
use crate::types::SdfNode;

/// Configuration for the mesh cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of cached meshes in memory.
    pub max_entries: usize,
    /// Optional disk cache directory for .abm persistence.
    pub disk_cache_dir: Option<PathBuf>,
    /// Whether to automatically persist to disk on eviction.
    pub persist_on_evict: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 64,
            disk_cache_dir: None,
            persist_on_evict: false,
        }
    }
}

/// Cache key computed from SDF tree + generation parameters.
///
/// Two identical SDF trees with the same bounds and resolution will produce
/// the same `MeshCacheKey`, enabling cache hits across frames or sessions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MeshCacheKey {
    /// Hash of the SDF node tree.
    pub sdf_hash: u64,
    /// Resolution used for mesh generation.
    pub resolution: u32,
    /// Quantized bounds hash (to avoid float comparison issues).
    pub bounds_hash: u64,
}

/// Internal cache entry holding the mesh and its approximate memory footprint.
struct CacheEntry {
    mesh: Arc<Mesh>,
    /// Approximate size in bytes (vertices * vertex_size + indices * 4).
    size_bytes: usize,
}

/// Internal LRU-ordered map.
///
/// `order` tracks access recency: front = oldest, back = newest.
struct LruMap {
    map: HashMap<MeshCacheKey, CacheEntry>,
    /// LRU order: front = oldest, back = newest.
    order: Vec<MeshCacheKey>,
}

impl LruMap {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Move `key` to the back (most-recently-used position).
    #[inline(always)]
    fn touch(&mut self, key: &MeshCacheKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
            self.order.push(key.clone());
        }
    }

    /// Insert a new entry, returning the evicted key (if any) when over capacity.
    fn insert(
        &mut self,
        key: MeshCacheKey,
        entry: CacheEntry,
        max_entries: usize,
    ) -> Option<MeshCacheKey> {
        // If key already exists, update in-place and touch.
        if self.map.contains_key(&key) {
            self.map.insert(key.clone(), entry);
            self.touch(&key);
            return None;
        }

        // Evict oldest if at capacity.
        let evicted = if self.map.len() >= max_entries && max_entries > 0 {
            if let Some(oldest_key) = self.order.first().cloned() {
                self.map.remove(&oldest_key);
                self.order.remove(0);
                Some(oldest_key)
            } else {
                None
            }
        } else {
            None
        };

        self.map.insert(key.clone(), entry);
        self.order.push(key);
        evicted
    }

    /// Remove an entry by key.
    fn remove(&mut self, key: &MeshCacheKey) -> Option<Arc<Mesh>> {
        if let Some(entry) = self.map.remove(key) {
            if let Some(pos) = self.order.iter().position(|k| k == key) {
                self.order.remove(pos);
            }
            Some(entry.mesh)
        } else {
            None
        }
    }

    /// Get a reference to a cached mesh, touching it for LRU.
    fn get(&mut self, key: &MeshCacheKey) -> Option<Arc<Mesh>> {
        if self.map.contains_key(key) {
            self.touch(key);
            Some(Arc::clone(&self.map[key].mesh))
        } else {
            None
        }
    }

    /// Clear all entries.
    fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    /// Number of entries.
    #[inline(always)]
    fn len(&self) -> usize {
        self.map.len()
    }

    /// Total memory usage across all cached meshes.
    fn memory_usage(&self) -> usize {
        self.map.values().map(|e| e.size_bytes).sum()
    }
}

/// Thread-safe LRU mesh cache.
///
/// Stores generated meshes keyed by SDF tree hash, bounds, and resolution.
/// Uses `RwLock` internally so multiple readers can access the cache
/// concurrently, while writes (insert/evict) take exclusive access.
pub struct MeshCache {
    config: CacheConfig,
    entries: RwLock<LruMap>,
}

impl MeshCache {
    /// Create a new mesh cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: RwLock::new(LruMap::new()),
        }
    }

    /// Get a cached mesh by key. Returns `None` if not cached.
    ///
    /// This promotes the entry to most-recently-used position.
    pub fn get(&self, key: &MeshCacheKey) -> Option<Arc<Mesh>> {
        // We need write access because `get` updates LRU order.
        let mut entries = self
            .entries
            .write()
            .expect("MeshCache: RwLock poisoned on entries.write() in get()");
        entries.get(key)
    }

    /// Insert a mesh into the cache, evicting the oldest entry if full.
    ///
    /// Returns an `Arc<Mesh>` pointing to the cached copy.
    pub fn insert(&self, key: MeshCacheKey, mesh: Mesh) -> Arc<Mesh> {
        let size_bytes = estimate_mesh_size(&mesh);
        let arc_mesh = Arc::new(mesh);
        let entry = CacheEntry {
            mesh: Arc::clone(&arc_mesh),
            size_bytes,
        };

        let mut entries = self
            .entries
            .write()
            .expect("MeshCache: RwLock poisoned on entries.write() in insert()");
        let _evicted = entries.insert(key, entry, self.config.max_entries);
        // Future: if persist_on_evict && disk_cache_dir is set, write evicted mesh to disk.

        arc_mesh
    }

    /// Remove an entry from the cache.
    pub fn remove(&self, key: &MeshCacheKey) -> Option<Arc<Mesh>> {
        let mut entries = self
            .entries
            .write()
            .expect("MeshCache: RwLock poisoned on entries.write() in remove()");
        entries.remove(key)
    }

    /// Clear all cached meshes.
    pub fn clear(&self) {
        let mut entries = self
            .entries
            .write()
            .expect("MeshCache: RwLock poisoned on entries.write() in clear()");
        entries.clear();
    }

    /// Number of cached entries.
    #[inline(always)]
    pub fn len(&self) -> usize {
        let entries = self
            .entries
            .read()
            .expect("MeshCache: RwLock poisoned on entries.read() in len()");
        entries.len()
    }

    /// Check if the cache is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total approximate memory usage of all cached meshes in bytes.
    pub fn memory_usage(&self) -> usize {
        let entries = self
            .entries
            .read()
            .expect("MeshCache: RwLock poisoned on entries.read() in memory_usage()");
        entries.memory_usage()
    }

    /// Get a cached mesh or generate it using the provided closure.
    ///
    /// If the key is found in the cache, returns the cached `Arc<Mesh>`.
    /// Otherwise, calls `generator()` to produce a new mesh, inserts it
    /// into the cache, and returns it.
    ///
    /// **Note**: The generator is called outside the lock to avoid holding
    /// the lock during potentially expensive mesh generation.
    pub fn get_or_generate<F>(&self, key: MeshCacheKey, generator: F) -> Arc<Mesh>
    where
        F: FnOnce() -> Mesh,
    {
        // Fast path: check if already cached.
        {
            let mut entries = self
                .entries
                .write()
                .expect("MeshCache: RwLock poisoned on entries.write() in get_or_generate()");
            if let Some(mesh) = entries.get(&key) {
                return mesh;
            }
        }

        // Slow path: generate mesh outside the lock.
        let mesh = generator();

        // Insert into cache (another thread may have inserted in the meantime).
        let size_bytes = estimate_mesh_size(&mesh);
        let arc_mesh = Arc::new(mesh);
        let entry = CacheEntry {
            mesh: Arc::clone(&arc_mesh),
            size_bytes,
        };

        let mut entries = self
            .entries
            .write()
            .expect("MeshCache: RwLock poisoned on entries.write() in get_or_generate()");
        // Double-check: if another thread inserted while we were generating,
        // return the existing entry instead.
        if let Some(existing) = entries.get(&key) {
            return existing;
        }
        entries.insert(key, entry, self.config.max_entries);

        arc_mesh
    }
}

/// Compute a deterministic hash of an SDF node tree.
///
/// Uses JSON serialization of the node for a stable, content-based hash.
/// The same logical SDF tree will always produce the same hash.
pub fn hash_sdf_node(node: &SdfNode) -> u64 {
    let json = serde_json::to_string(node).unwrap_or_default();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    json.hash(&mut hasher);
    hasher.finish()
}

/// Compute a cache key from an SDF node, bounds, and resolution.
///
/// Bounds are quantized to fixed-point (multiply by 1000, cast to i32) to
/// avoid floating-point comparison issues.
pub fn compute_cache_key(
    node: &SdfNode,
    min_bounds: glam::Vec3,
    max_bounds: glam::Vec3,
    resolution: usize,
) -> MeshCacheKey {
    let sdf_hash = hash_sdf_node(node);
    let bounds_hash = hash_bounds(min_bounds, max_bounds);
    MeshCacheKey {
        sdf_hash,
        resolution: resolution as u32,
        bounds_hash,
    }
}

/// Hash bounds by quantizing floats to fixed-point integers.
#[inline(always)]
fn hash_bounds(min_bounds: glam::Vec3, max_bounds: glam::Vec3) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    quantize_f32(min_bounds.x).hash(&mut hasher);
    quantize_f32(min_bounds.y).hash(&mut hasher);
    quantize_f32(min_bounds.z).hash(&mut hasher);
    quantize_f32(max_bounds.x).hash(&mut hasher);
    quantize_f32(max_bounds.y).hash(&mut hasher);
    quantize_f32(max_bounds.z).hash(&mut hasher);
    hasher.finish()
}

/// Quantize a float to fixed-point (multiply by 1000, cast to i32).
#[inline(always)]
fn quantize_f32(value: f32) -> i32 {
    (value * 1000.0) as i32
}

/// Estimate the memory footprint of a mesh in bytes.
#[inline(always)]
fn estimate_mesh_size(mesh: &Mesh) -> usize {
    let vertex_size = std::mem::size_of::<crate::mesh::Vertex>();
    mesh.vertices.len() * vertex_size + mesh.indices.len() * std::mem::size_of::<u32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Vertex;
    use glam::Vec3;

    /// Helper: create a simple test mesh with the given number of vertices and triangles.
    fn make_test_mesh(num_verts: usize, num_tris: usize) -> Mesh {
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
    fn test_insert_and_retrieve() {
        let cache = MeshCache::new(CacheConfig {
            max_entries: 4,
            ..Default::default()
        });

        let key = MeshCacheKey {
            sdf_hash: 42,
            resolution: 32,
            bounds_hash: 100,
        };
        let mesh = make_test_mesh(8, 4);
        let expected_verts = mesh.vertices.len();

        cache.insert(key.clone(), mesh);

        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().vertex_count(), expected_verts);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = MeshCache::new(CacheConfig {
            max_entries: 3,
            ..Default::default()
        });

        // Insert 3 entries (fills cache).
        for i in 0..3u64 {
            let key = MeshCacheKey {
                sdf_hash: i,
                resolution: 16,
                bounds_hash: 0,
            };
            cache.insert(key, make_test_mesh(4, 2));
        }
        assert_eq!(cache.len(), 3);

        // Insert a 4th entry -- should evict key with sdf_hash=0 (oldest).
        let key4 = MeshCacheKey {
            sdf_hash: 99,
            resolution: 16,
            bounds_hash: 0,
        };
        cache.insert(key4, make_test_mesh(4, 2));
        assert_eq!(cache.len(), 3);

        // The oldest (sdf_hash=0) should be evicted.
        let evicted_key = MeshCacheKey {
            sdf_hash: 0,
            resolution: 16,
            bounds_hash: 0,
        };
        assert!(cache.get(&evicted_key).is_none());

        // The rest should still be present.
        for i in 1..3u64 {
            let key = MeshCacheKey {
                sdf_hash: i,
                resolution: 16,
                bounds_hash: 0,
            };
            assert!(cache.get(&key).is_some());
        }
        let key99 = MeshCacheKey {
            sdf_hash: 99,
            resolution: 16,
            bounds_hash: 0,
        };
        assert!(cache.get(&key99).is_some());
    }

    #[test]
    fn test_lru_access_updates_order() {
        let cache = MeshCache::new(CacheConfig {
            max_entries: 3,
            ..Default::default()
        });

        // Insert keys 0, 1, 2.
        for i in 0..3u64 {
            let key = MeshCacheKey {
                sdf_hash: i,
                resolution: 16,
                bounds_hash: 0,
            };
            cache.insert(key, make_test_mesh(4, 2));
        }

        // Access key 0 to make it most-recently-used.
        let key0 = MeshCacheKey {
            sdf_hash: 0,
            resolution: 16,
            bounds_hash: 0,
        };
        assert!(cache.get(&key0).is_some());

        // Insert key 3 -- should evict key 1 (now the oldest).
        let key3 = MeshCacheKey {
            sdf_hash: 3,
            resolution: 16,
            bounds_hash: 0,
        };
        cache.insert(key3, make_test_mesh(4, 2));

        let evicted_key = MeshCacheKey {
            sdf_hash: 1,
            resolution: 16,
            bounds_hash: 0,
        };
        assert!(cache.get(&evicted_key).is_none());

        // Key 0 should still be present (was touched).
        assert!(cache.get(&key0).is_some());
    }

    #[test]
    fn test_get_or_generate_caches() {
        let cache = MeshCache::new(CacheConfig::default());
        let key = MeshCacheKey {
            sdf_hash: 1,
            resolution: 16,
            bounds_hash: 0,
        };

        let call_count = std::sync::atomic::AtomicU32::new(0);

        // First call: generator runs.
        let mesh1 = cache.get_or_generate(key.clone(), || {
            call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            make_test_mesh(10, 5)
        });
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(mesh1.vertex_count(), 10);

        // Second call: generator should NOT run.
        let mesh2 = cache.get_or_generate(key.clone(), || {
            call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            make_test_mesh(20, 10)
        });
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(mesh2.vertex_count(), 10); // Same mesh as first call.
    }

    #[test]
    fn test_cache_key_determinism() {
        let node = SdfNode::sphere(1.0);
        let hash1 = hash_sdf_node(&node);
        let hash2 = hash_sdf_node(&node);
        assert_eq!(hash1, hash2, "Same SDF tree must produce the same hash");

        // Different tree should (very likely) produce different hash.
        let node2 = SdfNode::sphere(2.0);
        let hash3 = hash_sdf_node(&node2);
        assert_ne!(
            hash1, hash3,
            "Different SDF trees should produce different hashes"
        );
    }

    #[test]
    fn test_compute_cache_key_determinism() {
        let node = SdfNode::sphere(1.0).subtract(SdfNode::box3d(0.5, 0.5, 0.5));
        let min = Vec3::splat(-2.0);
        let max = Vec3::splat(2.0);

        let key1 = compute_cache_key(&node, min, max, 32);
        let key2 = compute_cache_key(&node, min, max, 32);
        assert_eq!(key1, key2);

        // Different resolution => different key.
        let key3 = compute_cache_key(&node, min, max, 64);
        assert_ne!(key1, key3);

        // Different bounds => different key.
        let key4 = compute_cache_key(&node, Vec3::splat(-3.0), max, 32);
        assert_ne!(key1, key4);
    }

    #[test]
    fn test_thread_safety_concurrent_reads() {
        use std::sync::Arc as StdArc;

        let cache = StdArc::new(MeshCache::new(CacheConfig {
            max_entries: 16,
            ..Default::default()
        }));

        // Pre-populate.
        for i in 0..8u64 {
            let key = MeshCacheKey {
                sdf_hash: i,
                resolution: 16,
                bounds_hash: 0,
            };
            cache.insert(key, make_test_mesh(4, 2));
        }

        // Spawn threads that all read concurrently.
        let mut handles = Vec::new();
        for i in 0..8u64 {
            let cache_clone = StdArc::clone(&cache);
            handles.push(std::thread::spawn(move || {
                let key = MeshCacheKey {
                    sdf_hash: i,
                    resolution: 16,
                    bounds_hash: 0,
                };
                let result = cache_clone.get(&key);
                assert!(result.is_some(), "Key {} should be in cache", i);
                result.unwrap().vertex_count()
            }));
        }

        for handle in handles {
            let count = handle.join().unwrap();
            assert_eq!(count, 4);
        }
    }

    #[test]
    fn test_clear() {
        let cache = MeshCache::new(CacheConfig::default());
        let key = MeshCacheKey {
            sdf_hash: 1,
            resolution: 16,
            bounds_hash: 0,
        };
        cache.insert(key.clone(), make_test_mesh(4, 2));
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_remove() {
        let cache = MeshCache::new(CacheConfig::default());

        let key1 = MeshCacheKey {
            sdf_hash: 1,
            resolution: 16,
            bounds_hash: 0,
        };
        let key2 = MeshCacheKey {
            sdf_hash: 2,
            resolution: 16,
            bounds_hash: 0,
        };
        cache.insert(key1.clone(), make_test_mesh(4, 2));
        cache.insert(key2.clone(), make_test_mesh(8, 4));
        assert_eq!(cache.len(), 2);

        let removed = cache.remove(&key1);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().vertex_count(), 4);
        assert_eq!(cache.len(), 1);
        assert!(cache.get(&key1).is_none());
        assert!(cache.get(&key2).is_some());

        // Remove non-existent key.
        let removed_none = cache.remove(&key1);
        assert!(removed_none.is_none());
    }

    #[test]
    fn test_memory_usage() {
        let cache = MeshCache::new(CacheConfig::default());

        let key = MeshCacheKey {
            sdf_hash: 1,
            resolution: 16,
            bounds_hash: 0,
        };
        let mesh = make_test_mesh(100, 50);
        let expected_size = mesh.vertices.len() * std::mem::size_of::<Vertex>()
            + mesh.indices.len() * std::mem::size_of::<u32>();

        cache.insert(key, mesh);

        assert_eq!(cache.memory_usage(), expected_size);
    }

    #[test]
    fn test_empty_cache() {
        let cache = MeshCache::new(CacheConfig::default());
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.memory_usage(), 0);
    }

    #[test]
    fn test_insert_duplicate_key_updates() {
        let cache = MeshCache::new(CacheConfig {
            max_entries: 4,
            ..Default::default()
        });

        let key = MeshCacheKey {
            sdf_hash: 42,
            resolution: 32,
            bounds_hash: 100,
        };

        cache.insert(key.clone(), make_test_mesh(4, 2));
        assert_eq!(cache.len(), 1);

        // Insert with same key but different mesh.
        cache.insert(key.clone(), make_test_mesh(16, 8));
        assert_eq!(cache.len(), 1); // Should not increase count.

        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved.vertex_count(), 16); // Should be the updated mesh.
    }
}
