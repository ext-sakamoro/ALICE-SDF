//! View-independent overdraw ordering (`mesh::overdraw::optimize_overdraw`) and
//! the vertex-numbering invariance of `mesh::optimize::compute_atvr`.
//!
//! - Overdraw is measured here by an independent software rasterizer: an
//!   orthographic view along each of 26 directions (cube faces, edges and
//!   corners), back faces culled by the outward normal, a strict depth test,
//!   and overdraw = shaded fragments / covered pixels. On a concave mesh the
//!   reordered index buffer must draw fewer fragments, on a convex one it can
//!   not do better than 1 and must not do worse, and the triangle multiset (with
//!   winding) is kept.
//! - The sort rule is pinned on a soup of disjoint triangles, where every
//!   triangle misses the cache on all three vertices and so is its own cluster:
//!   the output must be the triangles in decreasing order of
//!   `n · (c − m)` (unit face normal, face centroid, mean of the referenced
//!   vertices), recomputed here in f64.
//! - `threshold` bounds the vertex-cache cost: ACMR of a 16-entry FIFO cache
//!   simulated here, after vs before, for thresholds 1.0 / 1.05 / 2.0.
//! - The view-dependent variant with the six axes leaves the order unchanged
//!   (opposite directions cancel), a single view reorders.
//! - ATVR depends only on which vertices the index list references in which
//!   order, not on their numbers: it is unchanged by a random renumbering and
//!   by `optimize_vertex_fetch`.
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::optimize::{compute_atvr, optimize_vertex_cache, optimize_vertex_fetch};
use alice_sdf::mesh::overdraw::optimize_overdraw;
use alice_sdf::mesh::{sdf_to_mesh, MarchingCubesConfig, Mesh, Vertex};
use alice_sdf::types::SdfNode;
use glam::{DVec2, DVec3, Vec3};
use std::collections::HashMap;

struct Rng(u64);
impl Rng {
    const fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

fn mc_mesh(node: &SdfNode, half: f32, resolution: usize) -> Mesh {
    let mut m = sdf_to_mesh(
        node,
        Vec3::splat(-half),
        Vec3::splat(half),
        &MarchingCubesConfig {
            resolution,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        },
    );
    optimize_vertex_cache(&mut m);
    m
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

/// 26 directions: cube faces, edges and corners
fn view_dirs() -> Vec<DVec3> {
    let mut v = Vec::new();
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                if (x, y, z) != (0, 0, 0) {
                    v.push(DVec3::new(f64::from(x), f64::from(y), f64::from(z)).normalize());
                }
            }
        }
    }
    v
}

/// Outward face normal: the winding normal, flipped to agree with the sum of
/// the vertex normals (independent of the mesher's winding convention).
fn outward_normal(m: &Mesh, t: &[u32]) -> DVec3 {
    let p = |i: u32| m.vertices[i as usize].position.as_dvec3();
    let n = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
    let vn: DVec3 = t
        .iter()
        .map(|&i| m.vertices[i as usize].normal.as_dvec3())
        .sum();
    if n.dot(vn) < 0.0 {
        -n
    } else {
        n
    }
}

/// (shaded fragments, covered pixels) over all views, `res`² pixels per view.
fn overdraw(m: &Mesh, res: usize) -> (u64, u64) {
    let mut shaded = 0u64;
    let mut covered = 0u64;
    let lo = m
        .vertices
        .iter()
        .fold(DVec3::splat(f64::MAX), |a, v| a.min(v.position.as_dvec3()));
    let hi = m
        .vertices
        .iter()
        .fold(DVec3::splat(f64::MIN), |a, v| a.max(v.position.as_dvec3()));
    let center = (lo + hi) * 0.5;
    let radius = (hi - lo).length() * 0.5 * 1.01;
    for d in view_dirs() {
        // viewer at +inf along d; depth = -p·d (smaller is nearer)
        let helper = if d.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
        let u = d.cross(helper).normalize();
        let v = d.cross(u);
        let to_px = |p: DVec3| {
            let q = p - center;
            DVec2::new(
                (q.dot(u) / radius * 0.5 + 0.5) * res as f64,
                (q.dot(v) / radius * 0.5 + 0.5) * res as f64,
            )
        };
        let mut zbuf = vec![f64::INFINITY; res * res];
        for t in m.indices.chunks_exact(3) {
            if outward_normal(m, t).dot(d) <= 0.0 {
                continue; // back face
            }
            let p: Vec<DVec3> = t
                .iter()
                .map(|&i| m.vertices[i as usize].position.as_dvec3())
                .collect();
            let s: Vec<DVec2> = p.iter().map(|&q| to_px(q)).collect();
            let z: Vec<f64> = p.iter().map(|q| -q.dot(d)).collect();
            let area = (s[1] - s[0]).perp_dot(s[2] - s[0]);
            if area.abs() < 1e-12 {
                continue;
            }
            let bmin = s[0].min(s[1]).min(s[2]);
            let bmax = s[0].max(s[1]).max(s[2]);
            let x0 = bmin.x.floor().max(0.0) as usize;
            let y0 = bmin.y.floor().max(0.0) as usize;
            let x1 = (bmax.x.ceil() as usize).min(res);
            let y1 = (bmax.y.ceil() as usize).min(res);
            for py in y0..y1 {
                for px in x0..x1 {
                    let c = DVec2::new(px as f64 + 0.5, py as f64 + 0.5);
                    let w0 = (s[2] - s[1]).perp_dot(c - s[1]) / area;
                    let w1 = (s[0] - s[2]).perp_dot(c - s[2]) / area;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    let depth = w0 * z[0] + w1 * z[1] + w2 * z[2];
                    let k = py * res + px;
                    if depth < zbuf[k] {
                        zbuf[k] = depth;
                        shaded += 1;
                    }
                }
            }
        }
        covered += zbuf.iter().filter(|z| z.is_finite()).count() as u64;
    }
    (shaded, covered)
}

/// ACMR of a FIFO cache of `size` entries (hits do not reorder)
fn fifo_acmr(idx: &[u32], size: usize) -> f64 {
    let mut cache: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
    let mut misses = 0usize;
    for &i in idx {
        if !cache.contains(&i) {
            misses += 1;
            cache.push_back(i);
            if cache.len() > size {
                cache.pop_front();
            }
        }
    }
    misses as f64 / (idx.len() / 3) as f64
}

#[test]
fn reordering_a_concave_mesh_draws_fewer_fragments() {
    let shapes = [
        ("torus", SdfNode::torus(1.0, 0.35), 1.6f32),
        (
            "three spheres",
            SdfNode::sphere(0.6)
                .translate(-0.9, 0.0, 0.0)
                .union(SdfNode::sphere(0.6).translate(0.9, 0.0, 0.0))
                .union(SdfNode::sphere(0.6).translate(0.0, 0.9, 0.3)),
            1.8,
        ),
    ];
    let mut compared = 0;
    for (name, node, half) in &shapes {
        let before = mc_mesh(node, *half, 40);
        assert!(before.triangle_count() > 1000, "{name}");
        let mut after = before.clone();
        optimize_overdraw(&mut after, 1.05);
        assert_eq!(after.vertices.len(), before.vertices.len());
        assert_eq!(tri_multiset(&after.indices), tri_multiset(&before.indices));

        let (sb, cb) = overdraw(&before, 128);
        let (sa, ca) = overdraw(&after, 128);
        assert_eq!(cb, ca, "{name}: coverage depends only on the triangle set");
        let (ob, oa) = (sb as f64 / cb as f64, sa as f64 / ca as f64);
        println!("{name}: overdraw {ob:.4} -> {oa:.4}");
        assert!(
            ob > 1.01,
            "{name}: the fixture must have overdraw to remove"
        );
        assert!(
            oa < ob - 0.01,
            "{name}: overdraw {ob:.4} -> {oa:.4} (must drop)"
        );
        compared += 1;
    }
    assert_eq!(compared, shapes.len());
}

#[test]
fn reordering_a_convex_mesh_keeps_triangles_and_overdraw_of_one() {
    let before = mc_mesh(&SdfNode::sphere(1.0), 1.5, 32);
    let mut after = before.clone();
    optimize_overdraw(&mut after, 1.05);
    assert_eq!(tri_multiset(&after.indices), tri_multiset(&before.indices));
    let (sb, cb) = overdraw(&before, 128);
    let (sa, ca) = overdraw(&after, 128);
    assert_eq!(cb, ca);
    // front faces of a convex surface do not overlap (only shared edges can
    // shade a pixel twice), so the order cannot matter much either way
    assert!((sb as f64 / cb as f64) < 1.01);
    assert!(sa <= sb + cb / 1000, "{sa} vs {sb}");
}

#[test]
fn disjoint_triangles_are_sorted_by_normal_dot_offset_from_the_centroid() {
    // Every triangle references three fresh vertices, so each is its own
    // cluster. Triangles sit on random points with random normals, scaled so
    // the keys are well separated.
    let mut rng = Rng(0x5EED_1234);
    let mut unit = || f64::from(rng.next() % 2001) / 1000.0 - 1.0;
    let mut m = Mesh::new();
    for _ in 0..60 {
        let c = DVec3::new(unit(), unit(), unit()) * 3.0;
        let n = DVec3::new(unit(), unit(), unit()).normalize_or(DVec3::Z);
        let helper = if n.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
        let a = n.cross(helper).normalize() * 0.1;
        let b = n.cross(a);
        // winding (p0, p1, p2) has normal +n
        for p in [c - a - b, c + a - b, c + b] {
            m.indices.push(m.vertices.len() as u32);
            m.vertices.push(Vertex::new(p.as_vec3(), n.as_vec3()));
        }
    }
    let before = m.indices.clone();
    optimize_overdraw(&mut m, 1.05);

    let p = |i: u32| {
        let v = m.vertices[i as usize].position;
        v.as_dvec3()
    };
    let mean: DVec3 = before.iter().map(|&i| p(i)).sum::<DVec3>() / before.len() as f64;
    let key = |t: &[u32]| {
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        let n = (b - a).cross(c - a).normalize();
        n.dot((a + b + c) / 3.0 - mean)
    };
    let mut tris: Vec<&[u32]> = before.chunks_exact(3).collect();
    let keys: Vec<f64> = tris.iter().map(|t| key(t)).collect();
    // keys are separated by more than f32 rounding
    let mut sorted = keys;
    sorted.sort_by(f64::total_cmp);
    assert!(sorted.windows(2).all(|w| w[1] - w[0] > 1e-4));
    tris.sort_by(|a, b| key(b).total_cmp(&key(a)));
    let want: Vec<u32> = tris.concat();
    assert_eq!(m.indices, want);
}

#[test]
fn threshold_bounds_the_vertex_cache_cost() {
    let base = mc_mesh(&SdfNode::torus(1.0, 0.35), 1.6, 40);
    let acmr_before = fifo_acmr(&base.indices, 16);
    let mut compared = 0;
    for threshold in [1.0f32, 1.05, 2.0] {
        let mut m = base.clone();
        optimize_overdraw(&mut m, threshold);
        let acmr_after = fifo_acmr(&m.indices, 16);
        println!("threshold {threshold}: ACMR {acmr_before:.4} -> {acmr_after:.4}");
        assert!(
            acmr_after <= acmr_before * f64::from(threshold) * 1.02,
            "threshold {threshold}: ACMR {acmr_before} -> {acmr_after}"
        );
        compared += 1;
    }
    assert_eq!(compared, 3);
    // threshold 0 does not split the hard clusters (whose first triangle
    // misses on all three vertices anyway), so the cost stays within the 1.0 bound
    let mut m = base.clone();
    optimize_overdraw(&mut m, 0.0);
    assert_eq!(tri_multiset(&m.indices), tri_multiset(&base.indices));
    assert!(fifo_acmr(&m.indices, 16) <= acmr_before * 1.02);
}

#[test]
fn atvr_does_not_depend_on_vertex_numbering() {
    let mut m = mc_mesh(&SdfNode::torus(1.0, 0.35), 1.6, 24);
    let mut compared = 0;
    for cache in [4usize, 8, 16, 32] {
        let atvr = compute_atvr(&m, cache);
        assert!(atvr >= 1.0);
        // random renumbering
        let n = m.vertices.len();
        let mut perm: Vec<u32> = (0..n as u32).collect();
        let mut rng = Rng(0xA7B0 + cache as u64);
        for i in (1..n).rev() {
            perm.swap(i, rng.next() as usize % (i + 1));
        }
        let mut r = m.clone();
        for (old, &new) in perm.iter().enumerate() {
            r.vertices[new as usize] = m.vertices[old];
        }
        for i in &mut r.indices {
            *i = perm[*i as usize];
        }
        assert_eq!(compute_atvr(&r, cache).to_bits(), atvr.to_bits());
        let mut f = r.clone();
        optimize_vertex_fetch(&mut f);
        assert_eq!(compute_atvr(&f, cache).to_bits(), atvr.to_bits());
        compared += 1;
    }
    assert_eq!(compared, 4);
    m.indices.clear();
    assert_eq!(compute_atvr(&m, 8), 0.0);
}

#[test]
fn opposite_view_directions_cancel_in_the_view_dependent_variant() {
    // rank along v plus rank along -v is (clusters - 1) for every cluster, so
    // the six axes give equal sums and the stable sort keeps the order.
    // Fixture: 66 disjoint triangles at scattered positions; each misses the
    // 32-entry cache on all three vertices, so a cluster closes every 11
    // triangles (6 clusters).
    use alice_sdf::mesh::overdraw::{default_view_directions, optimize_overdraw_with_views};
    let mut rng = Rng(0x0DD5_1DE5);
    let mut unit = || f32::from((rng.next() % 2001) as u16) / 1000.0 - 1.0;
    let mut base = Mesh::new();
    for _ in 0..66 {
        let o = Vec3::new(unit(), unit(), unit()) * 5.0;
        for k in [Vec3::ZERO, Vec3::X * 0.1, Vec3::Z * 0.1] {
            base.indices.push(base.vertices.len() as u32);
            base.vertices.push(Vertex::new(o + k, Vec3::Y));
        }
    }
    let mut m = base.clone();
    optimize_overdraw_with_views(&mut m, 1.05, &default_view_directions());
    assert_eq!(m.indices, base.indices);
    // a single view does reorder it
    optimize_overdraw_with_views(&mut m, 1.05, &[Vec3::Y]);
    assert_ne!(m.indices, base.indices);
    assert_eq!(tri_multiset(&m.indices), tri_multiset(&base.indices));
}
