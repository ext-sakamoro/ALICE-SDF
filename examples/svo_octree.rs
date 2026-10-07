//! Sparse voxel octree: build, query, flatten for the GPU, split into chunks
//! and stream them through an LRU cache.
//!
//! Run: `cargo run --example svo_octree --features svo`
//!
//! Every printed value is checked against the analytic sphere it was built
//! from (the properties themselves are pinned in `tests/test_svo_api_oracle.rs`).
//!
//! Author: Moroya Sakamoto

use alice_sdf::prelude::*;
use alice_sdf::svo::linearize::{compact_svo, validate_linearized};
use alice_sdf::svo::streaming::split_into_chunks;
use alice_sdf::svo::{
    build_svo, linearize_svo, svo_nearest_surface, svo_ray_query, SparseVoxelOctree,
    SvoBuildConfig, SvoChunk, SvoRayHit, SvoStreamingCache,
};

fn main() {
    let radius = 1.0;
    let sphere = SdfNode::sphere(radius);
    let config = SvoBuildConfig {
        max_depth: 6,
        bounds_min: Vec3::splat(-2.0),
        bounds_max: Vec3::splat(2.0),
        use_compiled: false,
        ..Default::default()
    };
    // finest leaf half diagonal: the bound on |stored − true| near the surface
    let h = 2.0 * 3f32.sqrt() / 64.0;

    // --- build -----------------------------------------------------------
    let svo = SparseVoxelOctree::build(&sphere, &config);
    let same = build_svo(&sphere, &config);
    assert_eq!(svo.node_count(), same.node_count());
    let root = svo.nodes[0];
    println!(
        "octree: {} nodes ({} leaves, {} interior), root has {} children, {} bytes",
        svo.node_count(),
        svo.leaf_count,
        svo.interior_count,
        root.child_count(),
        svo.memory_bytes()
    );
    assert_eq!(root.child_count(), 8);

    // --- point / nearest-surface / ray queries ----------------------------
    for p in [
        Vec3::new(1.02, 0.0, 0.0),
        Vec3::new(0.3, -0.7, 0.6),
        Vec3::new(0.0, 0.0, 0.97),
    ] {
        let d = svo.query_point(p);
        let (nd, surface) = svo.nearest_surface(p);
        assert_eq!(svo_nearest_surface(&svo, p), (nd, surface));
        let exact = p.length() - radius;
        println!(
            "  query {p}: svo {d:+.4}, analytic {exact:+.4}, surface point |s| = {:.4}",
            surface.length()
        );
        assert!((d - exact).abs() <= h);
        assert!((surface.length() - radius).abs() <= h + 0.05 * d.abs() + 1e-4);
    }
    let origin = Vec3::new(-3.0, 0.2, 0.0);
    let hit: SvoRayHit = svo
        .ray_query(origin, Vec3::X, 10.0)
        .expect("ray hits the sphere");
    let free = svo_ray_query(&svo, origin, Vec3::X, 10.0).expect("same ray");
    let t_true = 3.0 - (1.0f32 - 0.04).sqrt();
    println!(
        "  ray: t = {:.4} (analytic {t_true:.4}), normal {:.3}, depth {}",
        hit.distance, hit.normal, hit.depth
    );
    assert_eq!(hit.distance, free.distance);
    assert!((hit.distance - t_true).abs() <= h);
    let leaf = svo
        .nodes
        .iter()
        .find(|n| n.is_leaf == 1 && n.normal() != Vec3::ZERO)
        .unwrap();
    assert!((leaf.normal().length() - 1.0).abs() < 1e-3);

    // --- GPU layout -------------------------------------------------------
    let lin = svo.linearize();
    validate_linearized(&lin).expect("valid child indices");
    let lin2 = linearize_svo(&svo);
    assert_eq!(lin.level_counts, lin2.level_counts);
    println!(
        "linearized: {} nodes, {} levels {:?}, {} bytes for upload",
        lin.node_count(),
        lin.depth + 1,
        lin.level_counts,
        lin.as_bytes().len()
    );
    assert_eq!(lin.as_bytes().len(), lin.memory_bytes());
    let total: usize = (0..=lin.depth).map(|l| lin.nodes_at_level(l).len()).sum();
    assert_eq!(total, lin.node_count());
    let compact = compact_svo(&svo);
    assert_eq!(
        compact.node_count(),
        svo.node_count(),
        "a fresh build has no garbage"
    );

    // --- chunks and streaming --------------------------------------------
    let chunks = split_into_chunks(&svo, 2);
    let nodes_in_chunks: usize = chunks.iter().map(|c| c.nodes.len()).sum();
    println!(
        "chunks at depth 2: {} (including the root chunk), {nodes_in_chunks} nodes",
        chunks.len()
    );
    assert_eq!(nodes_in_chunks, svo.node_count());

    let mut cache = SvoStreamingCache::new(8);
    let mut budget = SvoStreamingCache::with_memory_budget(64 * 1024);
    for c in chunks.iter().filter(|c| c.chunk_id != u32::MAX) {
        // what a disk round trip would do
        let back = SvoChunk::from_bytes(&c.to_bytes()).expect("round trip");
        assert_eq!(back.memory_bytes(), c.memory_bytes());
        cache.insert(back.clone());
        budget.insert(SvoChunk::new(
            back.chunk_id,
            back.nodes,
            back.bounds,
            back.start_depth,
        ));
    }
    let hits = (0..64).filter(|&id| cache.get(id).is_some()).count();
    println!(
        "LRU cache: {} chunks, {} bytes, hit rate {:.3}; budget cache {} chunks in {} bytes",
        cache.len(),
        cache.memory_used(),
        cache.hit_rate(),
        budget.len(),
        budget.memory_used()
    );
    assert_eq!(cache.len(), 8);
    assert_eq!(hits, 8);
    assert!(budget.memory_used() <= 64 * 1024);
}
