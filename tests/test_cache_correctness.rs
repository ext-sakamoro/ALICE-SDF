//! Integration tests: Cache correctness
//!
//! Verifies MeshCache key matching, LRU eviction, and ChunkedMeshCache behavior.
//!
//! Author: Moroya Sakamoto

use alice_sdf::prelude::*;
use alice_sdf::{
    compute_cache_key, hash_sdf_node, CacheConfig, ChunkCoord, ChunkedCacheConfig,
    ChunkedMeshCache, MeshCache, MeshCacheKey,
};
use std::sync::Arc;

mod common;
use common::*;

// ============================================================================
// SDF hashing
// ============================================================================

#[test]
fn hash_same_shape_is_stable() {
    let shape = test_sphere();
    let h1 = hash_sdf_node(&shape);
    let h2 = hash_sdf_node(&shape);
    assert_eq!(h1, h2, "Same shape should produce same hash");
}

#[test]
fn hash_different_shapes_differ() {
    let h_sphere = hash_sdf_node(&test_sphere());
    let h_box = hash_sdf_node(&test_box());
    assert_ne!(
        h_sphere, h_box,
        "Different shapes should have different hashes"
    );
}

#[test]
fn cache_key_stable_for_same_params() {
    let shape = test_sphere();
    let min = Vec3::splat(-1.5);
    let max = Vec3::splat(1.5);

    let k1 = compute_cache_key(&shape, min, max, 32);
    let k2 = compute_cache_key(&shape, min, max, 32);

    assert_eq!(k1.sdf_hash, k2.sdf_hash);
    assert_eq!(k1.resolution, k2.resolution);
    assert_eq!(k1.bounds_hash, k2.bounds_hash);
}

#[test]
fn cache_key_differs_with_resolution() {
    let shape = test_sphere();
    let min = Vec3::splat(-1.5);
    let max = Vec3::splat(1.5);

    let k1 = compute_cache_key(&shape, min, max, 32);
    let k2 = compute_cache_key(&shape, min, max, 64);

    assert_ne!(
        k1.resolution, k2.resolution,
        "Different resolutions should produce different keys"
    );
}

// ============================================================================
// MeshCache
// ============================================================================

#[test]
fn mesh_cache_insert_and_retrieve() {
    let cache = MeshCache::new(CacheConfig {
        max_entries: 10,
        disk_cache_dir: None,
        persist_on_evict: false,
    });

    let mesh = Mesh {
        vertices: vec![Vertex {
            position: Vec3::ZERO,
            normal: Vec3::Y,
            uv: glam::Vec2::ZERO,
            uv2: glam::Vec2::ZERO,
            tangent: glam::Vec4::ZERO,
            color: [1.0, 1.0, 1.0, 1.0],
            material_id: 0,
        }],
        indices: vec![0, 0, 0],
    };

    let key = MeshCacheKey {
        sdf_hash: 42,
        resolution: 32,
        bounds_hash: 100,
    };

    cache.insert(key.clone(), mesh);
    let retrieved = cache.get(&key);
    assert!(retrieved.is_some(), "Should retrieve inserted mesh");
    assert_eq!(
        retrieved.unwrap().vertices.len(),
        1,
        "Retrieved mesh should match"
    );
}

#[test]
fn mesh_cache_miss_on_unknown_key() {
    let cache = MeshCache::new(CacheConfig {
        max_entries: 10,
        disk_cache_dir: None,
        persist_on_evict: false,
    });

    let key = MeshCacheKey {
        sdf_hash: 99999,
        resolution: 32,
        bounds_hash: 0,
    };

    assert!(cache.get(&key).is_none(), "Unknown key should return None");
}

#[test]
fn mesh_cache_lru_eviction() {
    let cache = MeshCache::new(CacheConfig {
        max_entries: 2,
        disk_cache_dir: None,
        persist_on_evict: false,
    });

    let make_mesh = || Mesh {
        vertices: vec![],
        indices: vec![],
    };

    let key1 = MeshCacheKey {
        sdf_hash: 1,
        resolution: 32,
        bounds_hash: 0,
    };
    let key2 = MeshCacheKey {
        sdf_hash: 2,
        resolution: 32,
        bounds_hash: 0,
    };
    let key3 = MeshCacheKey {
        sdf_hash: 3,
        resolution: 32,
        bounds_hash: 0,
    };

    cache.insert(key1.clone(), make_mesh());
    cache.insert(key2.clone(), make_mesh());
    cache.insert(key3.clone(), make_mesh()); // Should evict key1

    assert!(cache.get(&key1).is_none(), "Key1 should be evicted (LRU)");
    assert!(cache.get(&key2).is_some(), "Key2 should still be present");
    assert!(cache.get(&key3).is_some(), "Key3 should still be present");
}

#[test]
fn mesh_cache_get_or_generate() {
    let cache = MeshCache::new(CacheConfig {
        max_entries: 10,
        disk_cache_dir: None,
        persist_on_evict: false,
    });

    let key = MeshCacheKey {
        sdf_hash: 42,
        resolution: 16,
        bounds_hash: 0,
    };
    let mut generated = false;

    let _mesh = cache.get_or_generate(key.clone(), || {
        generated = true;
        Mesh {
            vertices: vec![],
            indices: vec![],
        }
    });
    assert!(generated, "Generator should be called on first access");

    generated = false;
    let _mesh = cache.get_or_generate(key, || {
        generated = true;
        Mesh {
            vertices: vec![],
            indices: vec![],
        }
    });
    assert!(!generated, "Generator should NOT be called on cache hit");
}

#[test]
fn mesh_cache_clear() {
    let cache = MeshCache::new(CacheConfig {
        max_entries: 10,
        disk_cache_dir: None,
        persist_on_evict: false,
    });

    let key = MeshCacheKey {
        sdf_hash: 1,
        resolution: 32,
        bounds_hash: 0,
    };
    cache.insert(
        key.clone(),
        Mesh {
            vertices: vec![],
            indices: vec![],
        },
    );

    cache.clear();
    assert!(
        cache.get(&key).is_none(),
        "Cache should be empty after clear"
    );
}

// ============================================================================
// ChunkedMeshCache
// ============================================================================

#[test]
fn chunked_cache_insert_and_retrieve() {
    let config = ChunkedCacheConfig {
        chunk_size: 1.0,
        chunk_resolution: 16,
        cache_dir: None,
        max_cached_chunks: 100,
    };
    let cache = ChunkedMeshCache::new(config);

    let coord = ChunkCoord { x: 0, y: 0, z: 0 };
    let mesh = Mesh {
        vertices: vec![],
        indices: vec![],
    };

    cache.set_chunk(coord, mesh, 0);
    let retrieved = cache.get_chunk(&coord);
    assert!(retrieved.is_some(), "Should retrieve inserted chunk");
}
