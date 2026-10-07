//! Index reordering vs the triangle set it must keep, and the metric it must
//! improve (`mesh::stripifier`, `mesh::optimize`, `mesh::overdraw`,
//! `mesh::spatial_order`).
//!
//! - Strips: `unstripify(stripify(t))` is the same multiset of triangles as
//!   `t` with each triangle's winding kept (a triangle may start at any of its
//!   three vertices), for both join modes (primitive restart / degenerate
//!   triangles). The strip length is within `stripify_bound`, and the list
//!   length within `unstripify_bound`; a strip shorter than 3 indices holds no
//!   triangle.
//! - Vertex cache: `compute_acmr` is checked against an LRU cache simulated
//!   here, and against closed forms (3 for a soup of disjoint triangles;
//!   `V / T` when the cache holds every vertex). `optimize_vertex_cache` keeps
//!   the triangle set and lowers ACMR on a shuffled grid, never below the
//!   lower bound `V / T`.
//! - `compute_atvr` is checked against a FIFO simulated here (1 when the cache
//!   holds every vertex). `optimize_vertex_fetch` renumbers vertices in order
//!   of first use (so the first occurrences read 0, 1, 2, ...), keeps every
//!   triangle's positions and drops unreferenced vertices.
//! - Overdraw with one view direction sorts clusters front to back: the
//!   triangle set is kept and the first triangle of each emitted cluster has a
//!   non-decreasing depth along the view.
//! - Spatial order: `morton_3d` is the 3-bit interleave written as a loop over
//!   bits; `optimize_spatial_order` keeps the vertex multiset and every
//!   triangle's positions, and leaves the vertices in non-decreasing Morton
//!   order of their 10-bit quantized positions.
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::optimize::{
    compute_acmr, compute_atvr, optimize_vertex_cache, optimize_vertex_fetch,
};
use alice_sdf::mesh::overdraw::{default_view_directions, optimize_overdraw_with_views};
use alice_sdf::mesh::spatial_order::{morton_3d, optimize_spatial_order};
use alice_sdf::mesh::stripifier::{
    stripify, stripify_bound, try_stripify, unstripify, unstripify_bound,
};
use alice_sdf::mesh::{Mesh, Vertex};
use glam::Vec3;
use std::collections::{HashMap, VecDeque};

struct Rng(u64);
impl Rng {
    const fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

/// canonical rotation that keeps the winding
fn canon(t: &[u32]) -> [u32; 3] {
    let r = (0..3).min_by_key(|&r| t[r]).unwrap();
    [t[r], t[(r + 1) % 3], t[(r + 2) % 3]]
}

fn tri_multiset(idx: &[u32]) -> HashMap<[u32; 3], usize> {
    let mut m = HashMap::new();
    for t in idx.chunks_exact(3) {
        *m.entry(canon(t)).or_insert(0) += 1;
    }
    m
}

fn grid(n: u32) -> Mesh {
    let mut mesh = Mesh::new();
    for y in 0..=n {
        for x in 0..=n {
            let p = Vec3::new(x as f32, y as f32, ((x * 7 + y * 3) % 5) as f32 * 0.1);
            mesh.vertices.push(Vertex::new(p, Vec3::Z));
        }
    }
    for y in 0..n {
        for x in 0..n {
            let i = y * (n + 1) + x;
            mesh.indices
                .extend([i, i + 1, i + n + 1, i + 1, i + n + 2, i + n + 1]);
        }
    }
    mesh
}

fn shuffle_triangles(mesh: &mut Mesh, seed: u64) {
    let mut rng = Rng(seed);
    let mut tris: Vec<[u32; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| [t[0], t[1], t[2]])
        .collect();
    for i in (1..tris.len()).rev() {
        let j = rng.next() as usize % (i + 1);
        tris.swap(i, j);
    }
    mesh.indices = tris.concat();
}

#[test]
fn strips_unroll_to_the_same_triangles_with_the_winding_kept() {
    let mut cases: Vec<(Vec<u32>, usize)> = vec![
        (vec![0, 1, 2], 3),
        (vec![0, 1, 2, 2, 1, 3], 4),
        (vec![0, 1, 2, 3, 4, 5], 6),
    ];
    let g = grid(12);
    cases.push((g.indices.clone(), g.vertices.len()));
    let mut s = grid(9);
    shuffle_triangles(&mut s, 7);
    cases.push((s.indices.clone(), s.vertices.len()));
    let mut n = 0;
    for (idx, vc) in &cases {
        for restart in [None, Some(u32::MAX)] {
            let strip = stripify(idx, *vc, restart);
            assert_eq!(strip, try_stripify(idx, *vc, restart).unwrap());
            assert!(
                strip.len() <= stripify_bound(idx.len()),
                "{} > bound",
                strip.len()
            );
            let list = unstripify(&strip, restart);
            assert!(list.len() <= unstripify_bound(strip.len()));
            assert_eq!(
                tri_multiset(&list),
                tri_multiset(idx),
                "restart {restart:?}"
            );
            n += idx.len() / 3;
        }
    }
    assert!(n > 500, "compared {n} triangles");
    assert!(try_stripify(&[0, 1], 3, None).is_err());
    assert!(try_stripify(&[0, 1, 9], 3, None).is_err());
}

#[test]
fn a_strip_shorter_than_three_indices_holds_no_triangle() {
    for len in 0..3usize {
        assert_eq!(unstripify_bound(len), 0, "bound({len})");
        let strip: Vec<u32> = (0..len as u32).collect();
        assert!(unstripify(&strip, None).is_empty());
        assert!(unstripify(&strip, Some(99)).is_empty());
    }
    assert_eq!(unstripify_bound(3), 3);
    assert_eq!(unstripify_bound(10), 24);
}

fn lru_acmr(idx: &[u32], cache: usize) -> f64 {
    let mut c: VecDeque<u32> = VecDeque::new();
    let mut misses = 0usize;
    for &i in idx {
        if let Some(p) = c.iter().position(|&v| v == i) {
            c.remove(p);
        } else {
            misses += 1;
        }
        c.push_front(i);
        c.truncate(cache);
    }
    misses as f64 / (idx.len() / 3) as f64
}

fn fifo_atvr(idx: &[u32], cache: usize) -> f64 {
    let mut c: VecDeque<u32> = VecDeque::new();
    let mut misses = 0usize;
    let mut seen = std::collections::HashSet::new();
    for &i in idx {
        seen.insert(i);
        if !c.contains(&i) {
            misses += 1;
            c.push_front(i);
            c.truncate(cache);
        }
    }
    misses as f64 / seen.len() as f64
}

#[test]
fn acmr_and_atvr_are_the_simulated_cache_miss_ratios() {
    let mut n = 0;
    let mut g = grid(10);
    for (seed, cache) in [(1u64, 4usize), (2, 16), (3, 32), (4, 1000)] {
        shuffle_triangles(&mut g, seed);
        let acmr = f64::from(compute_acmr(&g, cache));
        assert!(
            (acmr - lru_acmr(&g.indices, cache)).abs() < 1e-6,
            "acmr cache {cache}"
        );
        let atvr = f64::from(compute_atvr(&g, cache));
        assert!(
            (atvr - fifo_atvr(&g.indices, cache)).abs() < 1e-6,
            "atvr cache {cache}"
        );
        n += 2;
    }
    // closed forms
    let t = g.indices.len() / 3;
    let v = g.vertices.len();
    let all = f64::from(compute_acmr(&g, 10_000));
    assert!((all - v as f64 / t as f64).abs() < 1e-6);
    assert_eq!(compute_atvr(&g, 10_000), 1.0);
    let mut soup = Mesh::new();
    soup.vertices = vec![Vertex::new(Vec3::ZERO, Vec3::Z); 30];
    soup.indices = (0..30).collect();
    assert_eq!(compute_acmr(&soup, 32), 3.0);
    assert_eq!(compute_acmr(&Mesh::new(), 32), 0.0);
    assert_eq!(compute_atvr(&Mesh::new(), 32), 0.0);
    assert_eq!(n, 8);
}

#[test]
fn vertex_cache_optimisation_keeps_triangles_and_lowers_acmr() {
    let mut n = 0;
    for (size, seed) in [(8u32, 11u64), (16, 12), (24, 13)] {
        let mut m = grid(size);
        shuffle_triangles(&mut m, seed);
        let before = tri_multiset(&m.indices);
        let acmr_before = lru_acmr(&m.indices, 32);
        optimize_vertex_cache(&mut m);
        assert_eq!(tri_multiset(&m.indices), before);
        let acmr_after = lru_acmr(&m.indices, 32);
        let lower = m.vertices.len() as f64 / (m.indices.len() / 3) as f64;
        println!("grid {size}: ACMR {acmr_before:.3} -> {acmr_after:.3} (lower bound {lower:.3})");
        assert!(
            acmr_after < acmr_before * 0.8,
            "{acmr_before} -> {acmr_after}"
        );
        assert!(acmr_after >= lower - 1e-9);
        n += 1;
    }
    assert_eq!(n, 3);
}

#[test]
fn vertex_fetch_optimisation_numbers_vertices_in_first_use_order() {
    let mut m = grid(10);
    shuffle_triangles(&mut m, 21);
    // two vertices no triangle uses
    m.vertices.push(Vertex::new(Vec3::splat(99.0), Vec3::Z));
    m.vertices
        .insert(0, Vertex::new(Vec3::splat(-99.0), Vec3::Z));
    for i in &mut m.indices {
        *i += 1;
    }
    let pos_before: Vec<[[u32; 3]; 3]> = m
        .indices
        .chunks_exact(3)
        .map(|t| {
            t.iter()
                .map(|&i| m.vertices[i as usize].position.to_array().map(f32::to_bits))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect();
    let used = m.vertices.len() - 2;
    optimize_vertex_fetch(&mut m);
    assert_eq!(m.vertices.len(), used);
    let mut next = 0u32;
    for &i in &m.indices {
        assert!(i <= next, "index {i} before {next}");
        if i == next {
            next += 1;
        }
    }
    assert_eq!(next as usize, used);
    let pos_after: Vec<[[u32; 3]; 3]> = m
        .indices
        .chunks_exact(3)
        .map(|t| {
            t.iter()
                .map(|&i| m.vertices[i as usize].position.to_array().map(f32::to_bits))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect();
    assert_eq!(pos_after, pos_before);
}

#[test]
fn overdraw_with_one_view_emits_clusters_front_to_back() {
    // A soup of disjoint triangles misses the cache on all 3 vertices of every
    // triangle, so a cluster closes after ceil(32 / 3) = 11 triangles (the
    // 32-entry cache's worth of misses). Build 12 such clusters, each at its
    // own depth along the view, in a scrambled order: the emitted order must
    // be the clusters sorted by depth, each cluster's triangles kept in order.
    let view = Vec3::new(0.3, 1.0, 0.1).normalize();
    let depths = [
        5.0f32, -2.0, 9.0, 0.5, 7.5, -6.0, 3.0, 11.0, -1.0, 4.0, 8.0, -4.5,
    ];
    let side = Vec3::new(1.0, -0.3, 0.0).normalize();
    let up = view.cross(side).normalize();
    let mut m = Mesh::new();
    for (c, &d) in depths.iter().enumerate() {
        for t in 0..11 {
            let o = view * d + side * (t as f32) + up * (c as f32);
            for k in [Vec3::ZERO, side * 0.5, up * 0.5] {
                m.indices.push(m.vertices.len() as u32);
                m.vertices.push(Vertex::new(o + k, view));
            }
        }
    }
    let before = m.indices.clone();
    optimize_overdraw_with_views(&mut m, 1.0, &[view]);
    let mut order: Vec<usize> = (0..depths.len()).collect();
    order.sort_by(|&a, &b| depths[a].total_cmp(&depths[b]));
    let want: Vec<u32> = order
        .iter()
        .flat_map(|&c| before[c * 33..c * 33 + 33].iter().copied())
        .collect();
    assert_eq!(m.indices, want);
    assert_eq!(default_view_directions().len(), 6);
}

fn morton_reference(x: u32, y: u32, z: u32) -> u32 {
    let mut code = 0;
    for b in 0..10 {
        code |= ((x >> b) & 1) << (3 * b);
        code |= ((y >> b) & 1) << (3 * b + 1);
        code |= ((z >> b) & 1) << (3 * b + 2);
    }
    code
}

#[test]
fn morton_code_is_the_bit_interleave() {
    let mut rng = Rng(0xABCD);
    let mut n = 0;
    for x in [0u32, 1, 2, 511, 512, 1023] {
        for y in [0u32, 1, 3, 700, 1023] {
            for z in [0u32, 5, 1022, 1023] {
                assert_eq!(morton_3d(x, y, z), morton_reference(x, y, z));
                n += 1;
            }
        }
    }
    for _ in 0..10_000 {
        let (x, y, z) = (rng.next() % 1024, rng.next() % 1024, rng.next() % 1024);
        assert_eq!(morton_3d(x, y, z), morton_reference(x, y, z));
        n += 1;
    }
    // bits above 10 are dropped
    assert_eq!(morton_3d(1024 + 5, 2048, 0), morton_reference(5, 0, 0));
    assert_eq!(n, 120 + 10_000);
}

#[test]
fn spatial_order_sorts_vertices_by_morton_code_and_keeps_triangles() {
    let mut m = grid(14);
    let mut rng = Rng(77);
    // scramble the vertex order first
    let n = m.vertices.len();
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        perm.swap(i, rng.next() as usize % (i + 1));
    }
    let mut inv = vec![0u32; n];
    for (new, &old) in perm.iter().enumerate() {
        inv[old] = new as u32;
    }
    m.vertices = perm.iter().map(|&o| m.vertices[o]).collect();
    for i in &mut m.indices {
        *i = inv[*i as usize];
    }
    let tri_pos = |m: &Mesh| -> Vec<[[u32; 3]; 3]> {
        m.indices
            .chunks_exact(3)
            .map(|t| {
                [0, 1, 2].map(|k| {
                    m.vertices[t[k] as usize]
                        .position
                        .to_array()
                        .map(f32::to_bits)
                })
            })
            .collect()
    };
    let before_tris = tri_pos(&m);
    let mut before_verts: Vec<[u32; 3]> = m
        .vertices
        .iter()
        .map(|v| v.position.to_array().map(f32::to_bits))
        .collect();
    optimize_spatial_order(&mut m);
    assert_eq!(tri_pos(&m), before_tris);
    let mut after_verts: Vec<[u32; 3]> = m
        .vertices
        .iter()
        .map(|v| v.position.to_array().map(f32::to_bits))
        .collect();
    before_verts.sort_unstable();
    after_verts.sort_unstable();
    assert_eq!(after_verts, before_verts);

    let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for v in &m.vertices {
        lo = lo.min(v.position);
        hi = hi.max(v.position);
    }
    let q =
        |c: f32, l: f32, h: f32| -> u32 { ((c - l) / (h - l) * 1023.0).clamp(0.0, 1023.0) as u32 };
    let codes: Vec<u32> = m
        .vertices
        .iter()
        .map(|v| {
            let p = v.position;
            morton_reference(q(p.x, lo.x, hi.x), q(p.y, lo.y, hi.y), q(p.z, lo.z, hi.z))
        })
        .collect();
    assert!(
        codes.windows(2).all(|w| w[0] <= w[1]),
        "not in Morton order"
    );
    assert_eq!(codes.len(), n);
}
