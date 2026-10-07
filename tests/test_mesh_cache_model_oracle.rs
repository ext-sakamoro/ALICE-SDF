//! Model-based oracles for `cache::MeshCache` (LRU) and
//! `cache::ChunkedMeshCache` (FIFO, dirty tracking, `.abm` persistence).
//!
//! A deterministic sequence of operations is applied both to the cache and to
//! a small model written here from the documented rules:
//!
//! * `MeshCache`: least-recently-used eviction at `max_entries`, `get` and a
//!   hitting `get_or_generate` refresh recency, `insert` of an existing key
//!   refreshes it without evicting; memory is
//!   `vertices * size_of::<Vertex>() + indices * 4` per entry;
//! * `ChunkedMeshCache`: first-in-first-out eviction at `max_cached_chunks`
//!   (`set_chunk` of an existing chunk moves it to the back), new / reloaded
//!   chunks are dirty / clean, `invalidate_*` and `update_sdf_hash` mark dirty,
//!   `persist_dirty` writes and cleans; chunk coordinates are
//!   `floor(p / chunk_size)`.
//!
//! Every comparison loop counts and fails on zero.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use alice_sdf::cache::{
    compute_cache_key, CacheConfig, ChunkCoord, ChunkedCacheConfig, ChunkedMeshCache, MeshCache,
    MeshCacheKey,
};
use alice_sdf::mesh::{Mesh, Vertex};
use alice_sdf::prelude::*;

/// 64-bit LCG, deterministic across platforms.
struct Lcg(u64);
impl Lcg {
    const fn next(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % n
    }
}

/// A mesh whose vertex count identifies it (`tag + 1` vertices, `tag` triangles).
fn tagged_mesh(tag: usize) -> Mesh {
    let vertices = (0..=tag)
        .map(|i| Vertex::new(Vec3::new(tag as f32, i as f32, 0.5), Vec3::Y))
        .collect();
    let indices = (0..tag * 3).map(|i| (i % (tag + 1)) as u32).collect();
    Mesh { vertices, indices }
}

const fn mesh_bytes(tag: usize) -> usize {
    (tag + 1) * std::mem::size_of::<Vertex>() + tag * 3 * 4
}

const fn key(i: usize) -> MeshCacheKey {
    MeshCacheKey {
        sdf_hash: i as u64 * 7919,
        resolution: 16,
        bounds_hash: 3,
    }
}

#[test]
fn mesh_cache_follows_the_lru_model() {
    let cap = 5;
    let cache = MeshCache::new(CacheConfig {
        max_entries: cap,
        ..CacheConfig::default()
    });
    assert!(cache.is_empty());
    // model: recency order (front = least recent) and the tag stored per key.
    let mut order: Vec<usize> = Vec::new();
    let mut tags: HashMap<usize, usize> = HashMap::new();
    let touch = |order: &mut Vec<usize>, k: usize| {
        order.retain(|&x| x != k);
        order.push(k);
    };
    let mut rng = Lcg(12345);
    let mut steps = 0;
    let mut generated = 0;
    let mut hits = 0;
    for step in 0..600 {
        let k = rng.next(12) as usize;
        match rng.next(4) {
            0 => {
                let tag = step % 17 + 1;
                let arc = cache.insert(key(k), tagged_mesh(tag));
                assert_eq!(arc.vertices.len(), tag + 1);
                if !tags.contains_key(&k) && order.len() >= cap {
                    let old = order.remove(0);
                    tags.remove(&old);
                }
                tags.insert(k, tag);
                touch(&mut order, k);
            }
            1 => {
                let got = cache.get(&key(k)).map(|m| m.vertices.len() - 1);
                assert_eq!(got, tags.get(&k).copied(), "get {k} at {step}");
                if got.is_some() {
                    touch(&mut order, k);
                    hits += 1;
                }
            }
            2 => {
                let got = cache.remove(&key(k)).map(|m| m.vertices.len() - 1);
                assert_eq!(got, tags.remove(&k), "remove {k} at {step}");
                order.retain(|&x| x != k);
            }
            _ => {
                let tag = step % 13 + 1;
                let mut called = false;
                let m = cache.get_or_generate(key(k), || {
                    called = true;
                    tagged_mesh(tag)
                });
                assert_eq!(called, !tags.contains_key(&k), "generator at {step}");
                if called {
                    generated += 1;
                    if order.len() >= cap {
                        let old = order.remove(0);
                        tags.remove(&old);
                    }
                    tags.insert(k, tag);
                }
                assert_eq!(m.vertices.len() - 1, tags[&k]);
                touch(&mut order, k);
            }
        }
        assert_eq!(cache.len(), tags.len(), "len at {step}");
        assert_eq!(cache.is_empty(), tags.is_empty());
        let mem: usize = tags.values().map(|&t| mesh_bytes(t)).sum();
        assert_eq!(cache.memory_usage(), mem, "memory at {step}");
        steps += 1;
    }
    assert_eq!(steps, 600);
    assert!(generated > 0 && hits > 0);
    // A cached mesh is shared, not copied.
    let k = *order.last().unwrap();
    assert!(Arc::ptr_eq(
        &cache.get(&key(k)).unwrap(),
        &cache.get(&key(k)).unwrap()
    ));
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.memory_usage(), 0);
}

#[test]
fn cache_key_quantizes_bounds_to_thousandths() {
    // Documented: bounds are multiplied by 1000 and cast to i32 (truncation).
    let s = SdfNode::sphere(1.0);
    let a = compute_cache_key(&s, Vec3::splat(-1.0), Vec3::splat(1.0001), 32);
    let b = compute_cache_key(&s, Vec3::splat(-1.0), Vec3::splat(1.0004), 32);
    let c = compute_cache_key(&s, Vec3::splat(-1.0), Vec3::splat(1.002), 32);
    assert_eq!(a, b);
    assert_ne!(a.bounds_hash, c.bounds_hash);
    assert_eq!((a.resolution, a.sdf_hash), (c.resolution, c.sdf_hash));
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("alice_sdf_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

#[test]
fn chunked_cache_follows_the_fifo_and_dirty_model() {
    let dir = temp_dir("chunked_model");
    let cap = 6;
    let cache = ChunkedMeshCache::new(ChunkedCacheConfig {
        chunk_size: 0.5,
        max_cached_chunks: cap,
        cache_dir: Some(dir.clone()),
        ..ChunkedCacheConfig::default()
    });
    let coord = |i: usize| ChunkCoord::new(i as i32 - 4, (i % 3) as i32, -(i as i32 % 2));
    // model
    let mut order: Vec<ChunkCoord> = Vec::new();
    let mut tag: HashMap<ChunkCoord, usize> = HashMap::new();
    let mut hash: HashMap<ChunkCoord, u64> = HashMap::new();
    let mut dirty: HashSet<ChunkCoord> = HashSet::new();
    let mut on_disk: HashMap<ChunkCoord, usize> = HashMap::new();
    let mut current_hash = 0u64;
    let evict = |order: &mut Vec<ChunkCoord>,
                 tag: &mut HashMap<ChunkCoord, usize>,
                 hash: &mut HashMap<ChunkCoord, u64>,
                 dirty: &mut HashSet<ChunkCoord>| {
        let old = order.remove(0);
        tag.remove(&old);
        hash.remove(&old);
        dirty.remove(&old);
    };
    let mut rng = Lcg(99);
    let mut loads = 0;
    for step in 0..500 {
        let c = coord(rng.next(10) as usize);
        match rng.next(7) {
            0 | 1 => {
                let t = step % 11 + 1;
                let h = rng.next(3);
                cache.set_chunk(c, tagged_mesh(t), h);
                if tag.contains_key(&c) {
                    order.retain(|&x| x != c);
                } else if order.len() >= cap {
                    evict(&mut order, &mut tag, &mut hash, &mut dirty);
                }
                order.push(c);
                tag.insert(c, t);
                hash.insert(c, h);
                dirty.insert(c);
            }
            2 => {
                let got = cache.get_chunk(&c).map(|m| m.vertices.len() - 1);
                assert_eq!(got, tag.get(&c).copied(), "get {c} at {step}");
            }
            3 => {
                let n = cache.persist_dirty().expect("persist");
                assert_eq!(n, dirty.len(), "persisted at {step}");
                for d in dirty.drain() {
                    on_disk.insert(d, tag[&d]);
                }
            }
            4 => {
                let got = cache
                    .load_chunk(&c)
                    .expect("load")
                    .map(|m| m.vertices.len() - 1);
                assert_eq!(got, on_disk.get(&c).copied(), "load {c} at {step}");
                if let Some(t) = got {
                    loads += 1;
                    if !tag.contains_key(&c) {
                        if order.len() >= cap {
                            evict(&mut order, &mut tag, &mut hash, &mut dirty);
                        }
                        order.push(c);
                    }
                    tag.insert(c, t);
                    hash.insert(c, current_hash);
                    dirty.remove(&c);
                }
            }
            5 => {
                current_hash = rng.next(3);
                let mut got = cache.update_sdf_hash(current_hash);
                let mut want: Vec<ChunkCoord> = hash
                    .iter()
                    .filter(|(_, &h)| h != current_hash)
                    .map(|(&k, _)| k)
                    .collect();
                for k in &want {
                    dirty.insert(*k);
                }
                got.sort_by_key(|k| (k.x, k.y, k.z));
                want.sort_by_key(|k| (k.x, k.y, k.z));
                assert_eq!(got, want, "invalidated at {step}");
            }
            _ => {
                // A box around chunk c: every cached chunk it touches is dirty.
                let (lo, hi) = cache.chunk_bounds(&c);
                cache.invalidate_region(lo + Vec3::splat(0.1), hi + Vec3::splat(0.2));
                for k in tag.keys() {
                    let inside = (c.x..=c.x + 1).contains(&k.x)
                        && (c.y..=c.y + 1).contains(&k.y)
                        && (c.z..=c.z + 1).contains(&k.z);
                    if inside {
                        dirty.insert(*k);
                    }
                }
            }
        }
        assert!(
            cache.chunk_count() <= cap,
            "{} chunks at {step}",
            cache.chunk_count()
        );
        assert_eq!(cache.chunk_count(), tag.len(), "count at {step}");
        let mut cached = cache.cached_chunks();
        cached.sort_by_key(|k| (k.x, k.y, k.z));
        let mut want: Vec<ChunkCoord> = tag.keys().copied().collect();
        want.sort_by_key(|k| (k.x, k.y, k.z));
        assert_eq!(cached, want, "cached set at {step}");
        let got_dirty: HashSet<ChunkCoord> = cache.dirty_chunks().into_iter().collect();
        assert_eq!(got_dirty, dirty, "dirty set at {step}");
        let mem: usize = tag.values().map(|&t| mesh_bytes(t)).sum();
        assert_eq!(cache.memory_usage(), mem);
    }
    assert!(loads > 0);

    cache.invalidate_all();
    assert_eq!(cache.dirty_chunks().len(), tag.len());

    // merge_all: the triangles of every cached chunk, as a multiset of
    // position triples (chunk order is not specified).
    let merged = cache.merge_all();
    let tri_key = |m: &Mesh, t: usize| -> [[u32; 3]; 3] {
        [0, 1, 2].map(|j| {
            m.vertices[m.indices[t * 3 + j] as usize]
                .position
                .to_array()
                .map(f32::to_bits)
        })
    };
    let mut got: Vec<_> = (0..merged.indices.len() / 3)
        .map(|t| tri_key(&merged, t))
        .collect();
    let mut want = Vec::new();
    for &t in tag.values() {
        let m = tagged_mesh(t);
        want.extend((0..t).map(|i| tri_key(&m, i)));
    }
    assert!(!want.is_empty());
    got.sort_unstable();
    want.sort_unstable();
    assert_eq!(got, want);
    assert_eq!(
        merged.vertices.len(),
        tag.values().map(|t| t + 1).sum::<usize>()
    );

    cache.clear();
    assert_eq!(cache.chunk_count(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn chunk_coordinates_are_floor_of_position_over_size() {
    let cache = ChunkedMeshCache::new(ChunkedCacheConfig {
        chunk_size: 0.75,
        ..ChunkedCacheConfig::default()
    });
    let mut compared = 0;
    for &x in &[-2.3f32, -0.75, -0.0001, 0.0, 0.7499, 0.75, 3.1] {
        let c = cache.world_to_chunk(Vec3::new(x, -x, 2.0 * x));
        let want = [x, -x, 2.0 * x].map(|v| (v / 0.75).floor() as i32);
        assert_eq!([c.x, c.y, c.z], want, "{x}");
        let (lo, hi) = cache.chunk_bounds(&c);
        assert_eq!(
            lo,
            Vec3::new(want[0] as f32, want[1] as f32, want[2] as f32) * 0.75
        );
        assert_eq!(hi, lo + Vec3::splat(0.75));
        compared += 1;
    }
    assert_eq!(compared, 7);
    // chunks_in_bounds is the full integer box between the two corner chunks.
    let cs = cache.chunks_in_bounds(Vec3::new(-0.1, 0.0, 0.0), Vec3::new(1.6, 0.8, 0.1));
    assert_eq!(cs.len(), 4 * 2);
}

#[test]
fn persisted_chunks_reload_with_identical_geometry() {
    let dir = temp_dir("chunked_persist");
    let config = ChunkedCacheConfig {
        cache_dir: Some(dir.clone()),
        ..ChunkedCacheConfig::default()
    };
    let writer = ChunkedMeshCache::new(config.clone());
    let mesh = sdf_to_mesh(
        &SdfNode::sphere(0.4),
        Vec3::splat(-0.5),
        Vec3::splat(0.5),
        &MarchingCubesConfig {
            resolution: 8,
            ..Default::default()
        },
    );
    assert!(!mesh.indices.is_empty());
    let c = ChunkCoord::new(1, -2, 3);
    writer.set_chunk(c, mesh.clone(), 7);
    assert_eq!(writer.persist_dirty().unwrap(), 1);
    assert_eq!(writer.persist_dirty().unwrap(), 0);

    let reader = ChunkedMeshCache::new(config);
    assert!(reader
        .load_chunk(&ChunkCoord::new(0, 0, 0))
        .unwrap()
        .is_none());
    let back = reader.load_chunk(&c).unwrap().expect("persisted chunk");
    assert_eq!(back.indices, mesh.indices);
    let mut compared = 0;
    for (a, b) in back.vertices.iter().zip(&mesh.vertices) {
        assert_eq!(
            a.position.to_array().map(f32::to_bits),
            b.position.to_array().map(f32::to_bits)
        );
        compared += 1;
    }
    assert_eq!(compared, mesh.vertices.len());
    assert!(reader.dirty_chunks().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// `merge_all` concatenates chunks in ascending `(x, y, z)` coordinate order
/// (5.0.0): the merged mesh is the same, bit for bit, whatever the insertion
/// order and whichever `HashMap` seed each cache got. The expected mesh is
/// built here from the sorted coordinates.
#[test]
fn merge_all_is_ordered_by_chunk_coordinate_and_repeatable() {
    let coords: Vec<ChunkCoord> = (0..40)
        .map(|i| ChunkCoord::new(i % 4 - 2, (i / 4) % 3 - 1, i / 12 - 1))
        .collect();
    let tag_of = |c: &ChunkCoord| ((c.x + 2) * 100 + (c.y + 1) * 10 + (c.z + 1) + 1) as usize;
    let config = || ChunkedCacheConfig {
        max_cached_chunks: 1000,
        ..ChunkedCacheConfig::default()
    };

    // expected: sorted (x, y, z), each chunk's indices shifted by the vertex
    // count before it
    let mut sorted = coords.clone();
    sorted.sort_by_key(|c| (c.x, c.y, c.z));
    sorted.dedup();
    let mut want_v: Vec<Vertex> = Vec::new();
    let mut want_i: Vec<u32> = Vec::new();
    for c in &sorted {
        let m = tagged_mesh(tag_of(c));
        let base = want_v.len() as u32;
        want_v.extend_from_slice(&m.vertices);
        want_i.extend(m.indices.iter().map(|&i| i + base));
    }
    assert!(sorted.len() > 1);

    let bits = |m: &Mesh| -> (Vec<[u32; 3]>, Vec<u32>) {
        (
            m.vertices
                .iter()
                .map(|v| v.position.to_array().map(f32::to_bits))
                .collect(),
            m.indices.clone(),
        )
    };
    let want = bits(&Mesh {
        vertices: want_v,
        indices: want_i,
    });

    let mut rng = Lcg(0x5eed);
    let mut compared = 0;
    for round in 0..6 {
        // a fresh cache (fresh HashMap seed) and a shuffled insertion order
        let mut order = coords.clone();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.next(i as u64 + 1) as usize);
        }
        if round == 1 {
            order.reverse();
        }
        let cache = ChunkedMeshCache::new(config());
        for c in &order {
            cache.set_chunk(*c, tagged_mesh(tag_of(c)), 1);
        }
        let first = bits(&cache.merge_all());
        let second = bits(&cache.merge_all());
        assert_eq!(first, second, "round {round}: two merges of one cache");
        assert_eq!(first, want, "round {round}: coordinate order");
        compared += 1;
    }
    assert_eq!(compared, 6);
}
