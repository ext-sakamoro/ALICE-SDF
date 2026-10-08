//! Oracles for the sparse voxel octree API that `examples/svo_octree.rs` wires:
//! build (interpreted and compiled), linearisation and its per-level tables,
//! compaction, nearest-surface and ray queries, chunk splitting, the chunk wire
//! format and the LRU streaming cache.
//!
//! Expected values come from definitions written in this file:
//!
//! | target | oracle |
//! |---|---|
//! | `build` / `build_svo` / `build_svo_compiled` | every node stores the field **at its own centre**: `node.distance == eval(node_centre)` (compiled: `eval_compiled`) bit for bit, and so does `query_point` at a leaf centre |
//! | `linearize_svo` / `nodes_at_level` | the depth of every node found by an independent walk from the root: level `l` is the contiguous run of depth-`l` nodes, `level_offsets` are the prefix sums of `level_counts` |
//! | `LinearizedSvo::as_bytes` | the `#[repr(C)]` field layout of `SvoNode` (distance, n, mask, leaf, material, first child, padding) read back little-endian |
//! | `compact_svo` | after cutting a subtree loose, the node count drops by exactly the subtree size and every query is unchanged |
//! | `nearest_surface` | the distance is the descent's leaf distance; on a sphere the returned point lies on the surface to the leaf error |
//! | `split_into_chunks` | the chunks partition the nodes, each chunk's bounds is its octant cell, and an octree rebuilt from a chunk answers `query_point` like the whole tree inside that cell |
//! | `SvoChunk::to_bytes` / `from_bytes` | the documented header (id, count, depth, bounds) and 32-byte node records, bit-exact round trip, truncated input rejected |
//! | `SvoStreamingCache` | an independent LRU simulation; `memory_used` is the sum over the cached chunks |
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
#![cfg(feature = "svo")]

use std::collections::BTreeMap;

use alice_sdf::compiled::{eval_compiled, CompiledSdf};
use alice_sdf::prelude::*;
use alice_sdf::svo::linearize::{compact_svo, validate_linearized};
use alice_sdf::svo::streaming::split_into_chunks;
use alice_sdf::svo::{
    build_svo, build_svo_compiled, child_center, linearize_svo, svo_nearest_surface, svo_ray_query,
    SparseVoxelOctree, SvoBuildConfig, SvoChunk, SvoNode, SvoStreamingCache,
};

const B: f32 = 2.0;

fn scene() -> SdfNode {
    SdfNode::sphere(0.9).smooth_union(SdfNode::box3d(1.0, 0.6, 0.8).translate(0.5, 0.2, 0.0), 0.2)
}

fn cfg(depth: u32) -> SvoBuildConfig {
    SvoBuildConfig {
        max_depth: depth,
        bounds_min: Vec3::splat(-B),
        bounds_max: Vec3::splat(B),
        use_compiled: false,
        ..Default::default()
    }
}

/// One node found by walking from the root: index, depth, cell centre, half size.
#[derive(Clone, Copy)]
struct Visit {
    idx: usize,
    depth: u32,
    center: Vec3,
    half: Vec3,
}

/// Breadth-first walk through `child_index`, independent of the build's
/// bookkeeping.
fn walk(svo: &SparseVoxelOctree) -> Vec<Visit> {
    let mut out = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(Visit {
        idx: 0,
        depth: 0,
        center: svo.bounds.center(),
        half: svo.bounds.half_size(),
    });
    while let Some(v) = queue.pop_front() {
        out.push(v);
        let n = svo.nodes[v.idx];
        if n.is_leaf == 0 {
            for o in 0..8u8 {
                if let Some(c) = n.child_index(o) {
                    queue.push_back(Visit {
                        idx: c as usize,
                        depth: v.depth + 1,
                        center: child_center(v.center, v.half, o),
                        half: v.half * 0.5,
                    });
                }
            }
        }
    }
    out
}

#[test]
fn every_node_stores_the_field_at_its_own_centre() {
    let node = scene();
    let config = cfg(5);
    let interp = build_svo(&node, &config);
    let via_method = SparseVoxelOctree::build(&node, &config);
    assert_eq!(interp.nodes.len(), via_method.nodes.len());

    let compiled = CompiledSdf::compile(&node);
    let comp = build_svo_compiled(&compiled, &config);

    let mut leaves = 0;
    for (svo, field) in [
        (
            &interp,
            Box::new(|p| eval(&node, p)) as Box<dyn Fn(Vec3) -> f32>,
        ),
        (&comp, Box::new(|p| eval_compiled(&compiled, p))),
    ] {
        let visits = walk(svo);
        assert_eq!(
            visits.len(),
            svo.nodes.len(),
            "every node is reachable once"
        );
        for v in &visits {
            let n = svo.nodes[v.idx];
            assert_eq!(
                n.distance.to_bits(),
                field(v.center).to_bits(),
                "node {}",
                v.idx
            );
            assert_eq!(n.child_count(), n.child_mask.count_ones());
            assert_eq!(n.normal(), Vec3::new(n.nx, n.ny, n.nz));
            if n.is_leaf == 1 {
                leaves += 1;
                assert_eq!(svo.query_point(v.center).to_bits(), n.distance.to_bits());
            }
        }
    }
    assert!(leaves > 500, "scene too coarse: {leaves} leaves");
    // the method wrapper is the free function
    for (a, b) in interp.nodes.iter().zip(&via_method.nodes) {
        assert_eq!(a.distance.to_bits(), b.distance.to_bits());
        assert_eq!((a.child_mask, a.first_child), (b.child_mask, b.first_child));
    }
}

#[test]
fn linearised_levels_are_the_depth_runs_of_an_independent_walk() {
    let svo = SparseVoxelOctree::build(&scene(), &cfg(5));
    let lin = svo.linearize();
    let free = linearize_svo(&svo);
    assert_eq!(lin.level_counts, free.level_counts);
    validate_linearized(&lin).unwrap();

    let mut depth_of = vec![u32::MAX; svo.nodes.len()];
    for v in walk(&svo) {
        depth_of[v.idx] = v.depth;
    }
    let max_depth = *depth_of.iter().max().unwrap();
    assert_eq!(lin.depth, max_depth);
    assert_eq!(lin.level_counts.len() as u32, max_depth + 1);
    let mut offset = 0u32;
    for l in 0..=max_depth {
        let want: Vec<usize> = (0..svo.nodes.len()).filter(|&i| depth_of[i] == l).collect();
        assert_eq!(lin.level_offsets[l as usize], offset, "offset of level {l}");
        assert_eq!(
            lin.level_counts[l as usize] as usize,
            want.len(),
            "count of level {l}"
        );
        assert_eq!(
            want.first().copied(),
            Some(offset as usize),
            "level {l} starts at its offset"
        );
        assert_eq!(
            *want.last().unwrap() + 1 - want[0],
            want.len(),
            "level {l} is contiguous"
        );
        let slice = lin.nodes_at_level(l);
        assert_eq!(slice.len(), want.len());
        for (s, &i) in slice.iter().zip(&want) {
            assert_eq!(s.distance.to_bits(), svo.nodes[i].distance.to_bits());
        }
        offset += want.len() as u32;
    }
    assert!(lin.nodes_at_level(max_depth + 1).is_empty());
    assert_eq!(lin.node_count(), svo.node_count());
    assert_eq!(lin.memory_bytes(), 32 * svo.node_count());
    assert_eq!(svo.memory_bytes(), 32 * svo.node_count());
}

#[test]
fn as_bytes_is_the_repr_c_node_layout() {
    assert_eq!(std::mem::size_of::<SvoNode>(), 32);
    let svo = SparseVoxelOctree::build(&scene(), &cfg(3));
    let lin = svo.linearize();
    let bytes = lin.as_bytes();
    assert_eq!(bytes.len(), 32 * lin.node_count());
    let f = |b: &[u8], o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let u = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    for (i, n) in lin.nodes.iter().enumerate() {
        let r = &bytes[32 * i..32 * i + 32];
        assert_eq!(f(r, 0).to_bits(), n.distance.to_bits());
        assert_eq!(f(r, 4).to_bits(), n.nx.to_bits());
        assert_eq!(f(r, 8).to_bits(), n.ny.to_bits());
        assert_eq!(f(r, 12).to_bits(), n.nz.to_bits());
        assert_eq!(r[16], n.child_mask);
        assert_eq!(r[17], n.is_leaf);
        assert_eq!(u16::from_le_bytes([r[18], r[19]]), n.material_id);
        assert_eq!(u(r, 20), n.first_child);
    }
}

#[test]
fn compaction_drops_exactly_the_detached_subtree_and_keeps_every_query() {
    let svo = SparseVoxelOctree::build(&scene(), &cfg(5));
    // a freshly built tree has no unreachable node
    let same = compact_svo(&svo);
    assert_eq!(same.node_count(), svo.node_count());

    // detach the subtree of the first interior node at depth 2
    let visits = walk(&svo);
    let cut = visits
        .iter()
        .find(|v| v.depth == 2 && svo.nodes[v.idx].is_leaf == 0)
        .expect("an interior node at depth 2")
        .idx;
    let mut subtree = 0usize;
    let mut stack = vec![cut];
    while let Some(i) = stack.pop() {
        subtree += 1;
        let n = svo.nodes[i];
        if n.is_leaf == 0 {
            (0..8)
                .filter_map(|o| n.child_index(o))
                .for_each(|c| stack.push(c as usize));
        }
    }
    let mut cut_svo = SparseVoxelOctree {
        nodes: svo.nodes.clone(),
        bounds: svo.bounds,
        max_depth: svo.max_depth,
        leaf_count: svo.leaf_count,
        interior_count: svo.interior_count,
    };
    cut_svo.nodes[cut].is_leaf = 1;
    cut_svo.nodes[cut].child_mask = 0;

    let compact = compact_svo(&cut_svo);
    assert_eq!(compact.node_count(), svo.node_count() - (subtree - 1));
    assert_eq!(
        compact.node_count(),
        (compact.leaf_count + compact.interior_count) as usize
    );
    let mut r = 0x1234_5678u64;
    for _ in 0..4000 {
        let mut c = || {
            r = r
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((r >> 40) as f32 / (1u64 << 24) as f32).mul_add(2.0 * B, -B)
        };
        let p = Vec3::new(c(), c(), c());
        assert_eq!(
            compact.query_point(p).to_bits(),
            cut_svo.query_point(p).to_bits(),
            "{p}"
        );
    }

    // degenerate: an empty octree compacts to an empty octree
    let empty = SparseVoxelOctree {
        nodes: Vec::new(),
        bounds: svo.bounds,
        max_depth: 0,
        leaf_count: 0,
        interior_count: 0,
    };
    assert_eq!(compact_svo(&empty).node_count(), 0);
    assert_eq!(empty.query_point(Vec3::ZERO), f32::MAX);
    assert_eq!(
        svo_nearest_surface(&empty, Vec3::ONE),
        (f32::MAX, Vec3::ONE)
    );
    assert!(svo_ray_query(&empty, Vec3::ZERO, Vec3::X, 10.0).is_none());
}

#[test]
fn nearest_surface_lands_on_the_sphere_to_the_leaf_error() {
    let r = 1.0;
    let depth = 6;
    let svo = SparseVoxelOctree::build(&SdfNode::sphere(r), &cfg(depth));
    // finest leaf half diagonal: the stored centre value differs from the
    // field at the query point by at most this much (1-Lipschitz field)
    let h = (B * 3f32.sqrt()) / (1u32 << depth) as f32;
    let mut checked = 0;
    for i in 0..200 {
        let t = i as f32 * 0.731;
        let dir = Vec3::new(
            t.cos() * (0.3 * t).sin(),
            (0.3 * t).cos(),
            t.sin() * (0.3 * t).sin(),
        )
        .normalize();
        // within 0.06 of the sphere: the leaf centre is within 0.06 + h of
        // it, inside the 2·node_size band where the build stores a normal
        let p = dir * (r + 0.03 * ((i % 5) as f32 - 2.0));
        let (d, s) = svo.nearest_surface(p);
        let (d2, s2) = svo_nearest_surface(&svo, p);
        assert_eq!((d.to_bits(), s), (d2.to_bits(), s2));
        assert_eq!(
            d.to_bits(),
            svo.query_point(p).to_bits(),
            "same descent as query_point"
        );
        assert!((d - (p.length() - r)).abs() <= h, "distance error at {p}");
        // s = p − n·d with n the leaf's FD normal (≈ radial): | |s| − r | is
        // the distance error plus the tangential normal error times |d|
        assert!(
            (s.length() - r).abs() <= h + 0.05 * d.abs() + 1e-4,
            "surface point {s} for {p}"
        );
        checked += 1;
    }
    assert_eq!(checked, 200);
}

#[test]
fn ray_query_wrapper_reports_a_consistent_hit_on_the_sphere() {
    let svo = SparseVoxelOctree::build(&SdfNode::sphere(1.0), &cfg(6));
    let h = (B * 3f32.sqrt()) / 64.0;
    let origin = Vec3::new(-3.0, 0.1, 0.2);
    let dir = Vec3::new(1.0, 0.0, 0.0);
    let hit = svo
        .ray_query(origin, dir, 10.0)
        .expect("ray hits the sphere");
    let free = svo_ray_query(&svo, origin, dir, 10.0).unwrap();
    assert_eq!(hit.distance.to_bits(), free.distance.to_bits());
    assert_eq!(hit.position, origin + dir * hit.distance);
    // analytic entry: x = −sqrt(1 − y² − z²)
    let t_true = 3.0 - (1.0f32 - 0.01 - 0.04).sqrt();
    assert!(
        (hit.distance - t_true).abs() <= h,
        "t {} vs {t_true}",
        hit.distance
    );
    assert!(hit.normal.dot(Vec3::NEG_X) > 0.9, "normal {}", hit.normal);
    assert!(hit.depth <= 6);
    // a ray that passes above the sphere misses
    assert!(svo
        .ray_query(Vec3::new(-3.0, 1.6, 0.0), dir, 10.0)
        .is_none());
}

#[test]
fn chunks_partition_the_tree_and_answer_like_it_inside_their_cell() {
    let svo = SparseVoxelOctree::build(&scene(), &cfg(5));
    for split in [0u32, 1, 2] {
        let chunks = split_into_chunks(&svo, split);
        let (roots, subs): (Vec<&SvoChunk>, Vec<&SvoChunk>) =
            chunks.iter().partition(|c| c.chunk_id == u32::MAX);
        if split == 0 {
            assert!(roots.is_empty());
            assert_eq!(subs.len(), 1);
        } else {
            assert_eq!(roots.len(), 1);
            assert_eq!(chunks[0].chunk_id, u32::MAX, "root chunk first");
        }
        let total: usize = chunks.iter().map(|c| c.nodes.len()).sum();
        assert_eq!(
            total,
            svo.node_count(),
            "split {split}: chunks partition the nodes"
        );
        for (k, c) in subs.iter().enumerate() {
            assert_eq!(c.chunk_id, k as u32);
            assert_eq!(c.memory_bytes(), 32 * c.nodes.len());
            let lo = c.bounds.min();
            let hi = c.bounds.max();
            let edge = (2.0 * B) / (1u32 << c.start_depth) as f32;
            assert!(((hi - lo) - Vec3::splat(edge)).abs().max_element() < 1e-6);
            let sub = SparseVoxelOctree {
                nodes: c.nodes.clone(),
                bounds: c.bounds,
                max_depth: svo.max_depth - c.start_depth,
                leaf_count: 0,
                interior_count: 0,
            };
            for i in 0..27 {
                let f =
                    Vec3::new((i % 3) as f32, ((i / 3) % 3) as f32, (i / 9) as f32) * 0.37 + 0.13;
                let p = lo + (hi - lo) * f;
                assert_eq!(
                    sub.query_point(p).to_bits(),
                    svo.query_point(p).to_bits(),
                    "split {split} chunk {k} at {p}"
                );
            }
        }
    }
}

#[test]
fn chunk_wire_format_round_trips_and_follows_its_header_layout() {
    let svo = SparseVoxelOctree::build(&scene(), &cfg(4));
    let chunks = split_into_chunks(&svo, 1);
    for c in &chunks {
        let bytes = c.to_bytes();
        assert_eq!(bytes.len(), 36 + 32 * c.nodes.len());
        assert_eq!(
            u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            c.chunk_id
        );
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            c.nodes.len() as u32
        );
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            c.start_depth
        );
        let b = [
            c.bounds.min_x,
            c.bounds.min_y,
            c.bounds.min_z,
            c.bounds.max_x,
            c.bounds.max_y,
            c.bounds.max_z,
        ];
        for (k, v) in b.iter().enumerate() {
            let o = 12 + 4 * k;
            assert_eq!(
                f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()).to_bits(),
                v.to_bits()
            );
        }
        let back = SvoChunk::from_bytes(&bytes).expect("round trip");
        assert_eq!(back.to_bytes(), bytes, "bit-exact");
        assert_eq!(
            (back.chunk_id, back.start_depth),
            (c.chunk_id, c.start_depth)
        );
        for (a, n) in back.nodes.iter().zip(&c.nodes) {
            assert_eq!(
                [a.distance, a.nx, a.ny, a.nz].map(f32::to_bits),
                [n.distance, n.nx, n.ny, n.nz].map(f32::to_bits)
            );
            assert_eq!(
                (a.child_mask, a.is_leaf, a.material_id, a.first_child),
                (n.child_mask, n.is_leaf, n.material_id, n.first_child)
            );
        }
        // node record i starts at 36 + 32·i with the distance
        if let Some(n) = c.nodes.last() {
            let o = 36 + 32 * (c.nodes.len() - 1);
            assert_eq!(&bytes[o..o + 4], &n.distance.to_le_bytes());
            assert_eq!(&bytes[o + 4..o + 8], &n.nx.to_le_bytes());
        }
        assert!(SvoChunk::from_bytes(&bytes[..bytes.len() - 1]).is_none() || c.nodes.is_empty());
    }
    assert!(SvoChunk::from_bytes(&[0u8; 35]).is_none());
    let empty = SvoChunk::new(5, Vec::new(), svo.bounds, 0);
    let e = SvoChunk::from_bytes(&empty.to_bytes()).unwrap();
    assert_eq!((e.chunk_id, e.nodes.len()), (5, 0));
}

fn chunk(id: u32, nodes: usize) -> SvoChunk {
    let bounds = alice_sdf::compiled::AabbPacked::new(Vec3::ZERO, Vec3::ONE);
    SvoChunk::new(
        id,
        vec![SvoNode::leaf(id as f32, Vec3::Y); nodes],
        bounds,
        1,
    )
}

#[test]
fn streaming_cache_follows_an_independent_lru_model() {
    // model: id → (bytes, last access); evict the smallest last access
    let mut cache = SvoStreamingCache::new(3);
    let mut model: BTreeMap<u32, (usize, u64)> = BTreeMap::new();
    let mut clock = 0u64;
    let (mut hits, mut misses) = (0u64, 0u64);
    let ops: &[(bool, u32, usize)] = &[
        (true, 1, 4),
        (true, 2, 2),
        (false, 1, 0),
        (true, 3, 5),
        (true, 4, 1), // evicts 2 (1 was touched)
        (false, 2, 0),
        (false, 3, 0),
        (true, 5, 3), // evicts 1
        (true, 3, 7), // replaces 3 in place: nothing else evicted
        (false, 4, 0),
        (false, 1, 0),
    ];
    for &(insert, id, n) in ops {
        clock += 1;
        if insert {
            model.remove(&id);
            while model.len() >= 3 {
                let lru = *model.iter().min_by_key(|(_, v)| v.1).unwrap().0;
                model.remove(&lru);
            }
            model.insert(id, (32 * n, clock));
            cache.insert(chunk(id, n));
        } else {
            let got = cache.get(id).map(|c| c.chunk_id);
            if let Some(e) = model.get_mut(&id) {
                e.1 = clock;
                hits += 1;
                assert_eq!(got, Some(id));
            } else {
                misses += 1;
                assert_eq!(got, None, "id {id} should have been evicted");
            }
        }
        assert_eq!(cache.len(), model.len());
        assert_eq!(
            cache.memory_used(),
            model.values().map(|v| v.0).sum::<usize>(),
            "after {id}"
        );
    }
    assert!((cache.hit_rate() - hits as f64 / (hits + misses) as f64).abs() < 1e-12);
    assert_eq!(cache.remove(5).map(|c| c.chunk_id), Some(5));
    assert!(cache.remove(5).is_none());
    assert_eq!(
        cache.memory_used(),
        model
            .iter()
            .filter(|(k, _)| **k != 5)
            .map(|(_, v)| v.0)
            .sum::<usize>()
    );
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.memory_used(), 0);
    assert_eq!(SvoStreamingCache::new(2).hit_rate(), 0.0);
}

#[test]
fn memory_budget_cache_never_holds_more_than_its_budget() {
    let mut cache = SvoStreamingCache::with_memory_budget(32 * 10);
    for (id, n) in [(1u32, 4usize), (2, 4), (3, 4), (4, 1), (2, 6), (5, 9)] {
        cache.insert(chunk(id, n));
        assert!(
            cache.memory_used() <= 32 * 10,
            "after {id}: {}",
            cache.memory_used()
        );
        assert!(cache.get(id).is_some(), "just inserted {id}");
    }
    // the last insert (9 nodes) leaves room for at most one more node
    assert_eq!(
        cache.memory_used(),
        32 * 9 + if cache.get(4).is_some() { 32 } else { 0 }
    );
}
