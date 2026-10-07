//! Mesh reordering — vertex cache, vertex fetch, overdraw, spatial order and
//! triangle strips
//!
//! Meshes the unit sphere, shuffles its triangles, then runs the usual chain
//! `optimize_spatial_order → optimize_vertex_cache → optimize_overdraw →
//! optimize_vertex_fetch`, printing the cache metrics (`compute_acmr`,
//! `compute_atvr`) at each step, and converts the result to triangle strips
//! and back. Each step is checked to keep the set of triangles. The oracle
//! (simulated caches, Morton interleave, strip round trip, cluster order) is
//! `tests/test_mesh_reorder_oracle.rs`.
//!
//! # Running
//! ```bash
//! cargo run --example mesh_reorder
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::optimize::{
    compute_acmr, compute_atvr, optimize_vertex_cache, optimize_vertex_fetch,
};
use alice_sdf::mesh::overdraw::optimize_overdraw;
use alice_sdf::mesh::spatial_order::{morton_3d, optimize_spatial_order};
use alice_sdf::mesh::stripifier::{stripify, unstripify};
use alice_sdf::mesh::{sdf_to_mesh, MarchingCubesConfig, Mesh};
use alice_sdf::SdfNode;
use glam::Vec3;
use std::collections::HashMap;

/// triangles by vertex position, each rotated to a fixed start (winding kept)
fn triangle_set(m: &Mesh) -> HashMap<[[u32; 3]; 3], usize> {
    let mut set = HashMap::new();
    for t in m.indices.chunks_exact(3) {
        let p = [0, 1, 2].map(|k| {
            m.vertices[t[k] as usize]
                .position
                .to_array()
                .map(f32::to_bits)
        });
        let r = (0..3).min_by_key(|&r| p[r]).unwrap();
        *set.entry([p[r], p[(r + 1) % 3], p[(r + 2) % 3]])
            .or_insert(0) += 1;
    }
    set
}

fn report(step: &str, m: &Mesh) {
    println!(
        "  {step:<22} ACMR {:.3}  ATVR {:.3}",
        compute_acmr(m, 32),
        compute_atvr(m, 16)
    );
}

fn main() {
    println!("ALICE-SDF — mesh reordering");
    println!("===========================");

    let mut mesh = sdf_to_mesh(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &MarchingCubesConfig {
            resolution: 24,
            ..Default::default()
        },
    );
    // shuffle the triangles (deterministic LCG)
    let mut tris: Vec<[u32; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| [t[0], t[1], t[2]])
        .collect();
    let mut s = 0x2545_F491_u64;
    for i in (1..tris.len()).rev() {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        tris.swap(i, (s >> 33) as usize % (i + 1));
    }
    mesh.indices = tris.concat();
    let original = triangle_set(&mesh);
    println!(
        "{} vertices, {} triangles (shuffled)",
        mesh.vertex_count(),
        mesh.triangle_count()
    );
    report("shuffled", &mesh);
    let acmr_shuffled = compute_acmr(&mesh, 32);

    optimize_spatial_order(&mut mesh);
    assert_eq!(triangle_set(&mesh), original);
    report("spatial order", &mesh);
    println!(
        "    morton_3d(1, 2, 3) = {:#b} (x bits at 0, 3, …; y at 1, 4, …; z at 2, 5, …)",
        morton_3d(1, 2, 3)
    );
    assert_eq!(morton_3d(1, 2, 3), 0b110_101);

    optimize_vertex_cache(&mut mesh);
    assert_eq!(triangle_set(&mesh), original);
    report("vertex cache", &mesh);
    let acmr_opt = compute_acmr(&mesh, 32);
    assert!(acmr_opt < acmr_shuffled, "{acmr_shuffled} -> {acmr_opt}");

    let before_overdraw = mesh.indices.clone();
    optimize_overdraw(&mut mesh, 1.05);
    assert_eq!(triangle_set(&mesh), original);
    report("overdraw", &mesh);
    println!(
        "    triangle order changed by the 6-axis overdraw pass: {}",
        mesh.indices != before_overdraw
    );

    optimize_vertex_fetch(&mut mesh);
    assert_eq!(triangle_set(&mesh), original);
    report("vertex fetch", &mesh);
    let first_use: Vec<u32> = {
        let mut seen = vec![false; mesh.vertices.len()];
        mesh.indices
            .iter()
            .filter(|&&i| !std::mem::replace(&mut seen[i as usize], true))
            .copied()
            .collect()
    };
    assert!(first_use.iter().enumerate().all(|(k, &i)| i as usize == k));

    for restart in [None, Some(u32::MAX)] {
        let strip = stripify(&mesh.indices, mesh.vertices.len(), restart);
        let list = unstripify(&strip, restart);
        let mut back = mesh.clone();
        back.indices = list;
        assert_eq!(triangle_set(&back), original);
        println!(
            "  strip ({}): {} list indices -> {} strip indices, unrolled back to the same triangles",
            if restart.is_some() { "restart" } else { "degenerate joins" },
            mesh.indices.len(),
            strip.len()
        );
    }
}
