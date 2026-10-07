//! Mesh caches: `MeshCache` (whole meshes, LRU) and `ChunkedMeshCache`
//! (per-chunk meshes, FIFO, dirty tracking and `.abm` persistence).
//!
//! ```sh
//! cargo run --example mesh_cache
//! ```

use alice_sdf::cache::{
    compute_cache_key, hash_sdf_node, CacheConfig, ChunkCoord, ChunkedCacheConfig,
    ChunkedMeshCache, MeshCache,
};
use alice_sdf::prelude::*;

fn main() {
    // --- MeshCache: regenerate only when the shape, bounds or resolution change.
    let cache = MeshCache::new(CacheConfig {
        max_entries: 2,
        ..CacheConfig::default()
    });
    let (lo, hi) = (Vec3::splat(-1.5), Vec3::splat(1.5));
    let mut generated = 0;
    for (radius, res) in [(1.0, 16), (1.0, 16), (0.8, 16), (1.0, 24), (1.0, 16)] {
        let node = SdfNode::sphere(radius);
        let key = compute_cache_key(&node, lo, hi, res);
        let mesh = cache.get_or_generate(key, || {
            generated += 1;
            let config = MarchingCubesConfig {
                resolution: res,
                ..Default::default()
            };
            sdf_to_mesh(&node, lo, hi, &config)
        });
        println!(
            "r={radius} res={res}: {} triangles, hash {:016x}",
            mesh.indices.len() / 3,
            hash_sdf_node(&node)
        );
    }
    // (1,16) hit once; (0.8,16) and (1,24) pushed (1,16) out (capacity 2).
    println!(
        "generated {generated}, cached {}, {} bytes",
        cache.len(),
        cache.memory_usage()
    );
    assert_eq!(generated, 4);
    let key = compute_cache_key(&SdfNode::sphere(1.0), lo, hi, 24);
    let hit = cache.get(&key).expect("still cached");
    let removed = cache.remove(&key).expect("removed");
    assert!(std::sync::Arc::ptr_eq(&hit, &removed));
    cache.insert(key, (*removed).clone());
    cache.clear();
    assert!(cache.is_empty());

    // --- ChunkedMeshCache: mesh per chunk, re-mesh only what an edit touches.
    let dir = std::env::temp_dir().join(format!("alice_sdf_mesh_cache_{}", std::process::id()));
    let chunks = ChunkedMeshCache::new(ChunkedCacheConfig {
        chunk_size: 1.0,
        max_cached_chunks: 8,
        cache_dir: Some(dir.clone()),
        ..ChunkedCacheConfig::default()
    });
    let node = SdfNode::sphere(1.2);
    let hash = hash_sdf_node(&node);
    chunks.update_sdf_hash(hash);
    for c in chunks.chunks_in_bounds(Vec3::splat(-1.0), Vec3::splat(0.5)) {
        let (cmin, cmax) = chunks.chunk_bounds(&c);
        let config = MarchingCubesConfig {
            resolution: 8,
            ..Default::default()
        };
        chunks.set_chunk(c, sdf_to_mesh(&node, cmin, cmax, &config), hash);
    }
    println!(
        "{} chunks {:?}..., {} dirty",
        chunks.chunk_count(),
        &chunks.cached_chunks()[..2],
        chunks.dirty_chunks().len()
    );
    let written = chunks.persist_dirty().expect("persist");
    println!(
        "persisted {written} chunks, now {} dirty",
        chunks.dirty_chunks().len()
    );
    assert_eq!(written, chunks.chunk_count());

    // An edit near the origin dirties the chunks it overlaps.
    chunks.invalidate_region(Vec3::splat(-0.1), Vec3::splat(0.1));
    println!("after a local edit: {} dirty", chunks.dirty_chunks().len());
    // A new shape dirties everything meshed from the old one.
    let stale = chunks.update_sdf_hash(hash ^ 1);
    println!("after a shape change: {} stale chunks", stale.len());
    chunks.invalidate_all();

    let c = chunks.world_to_chunk(Vec3::new(-0.5, -0.5, -0.5));
    let in_memory = chunks.get_chunk(&c).expect("cached");
    chunks.clear();
    let reloaded = chunks.load_chunk(&c).expect("load").expect("on disk");
    assert_eq!(in_memory.indices, reloaded.indices);
    println!(
        "chunk {c} reloaded from disk: {} triangles; merged mesh {} vertices",
        reloaded.indices.len() / 3,
        chunks.merge_all().vertices.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = ChunkCoord::new(0, 0, 0);
}
