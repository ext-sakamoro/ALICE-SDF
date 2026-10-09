//! Marching cubes shares each vertex between the cells around its lattice
//! edge, on the CPU and the GPU, without welding by position.
//!
//! Oracles (none of them reads the expected value from the code under test):
//!
//! - **Closed 2-manifold of the known genus**: every edge of the index buffer
//!   is used by exactly two triangles, the surface is one connected component
//!   and `V − E + F` is the scene's closed-form Euler characteristic
//!   (2 − 2·genus), counted by an edge / union-find implementation here.
//! - **Lattice points on the surface**: a sphere of integer radius on an
//!   integer (and half-integer) lattice has lattice points exactly on the
//!   surface. With "inside ⇔ value < iso" the vertex count is the number of
//!   lattice edges with exactly one endpoint strictly inside, counted here in
//!   integer arithmetic (`x² + y² + z² < r²`).
//! - **One position per lattice edge**: the up to four cells around an edge
//!   reach it through different local edges and endpoint orders (Bourke cube
//!   numbering, restated below); the mesh holds exactly one vertex per
//!   sign-changing edge, bit-equal to the linear interpolation from the
//!   edge's lower endpoint (re-implemented here), for whichever cell lists it.
//! - **Normals**: on a sphere the vertex normal is within 1e-3 rad of the
//!   analytic gradient `p / |p|`.
//! - **Volume**: sphere and box signed volume within 1 % of the closed form.
//! - **GPU = CPU topology**: the GPU mesh has the CPU mesh's vertex and
//!   triangle counts (`gpu-mesh`; without an adapter the GPU part is skipped
//!   unless `ALICE_SDF_REQUIRE_GPU` is set).
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::CompiledSdf;
use alice_sdf::eval::eval;
use alice_sdf::mesh::{
    marching_cubes, marching_cubes_compiled, sdf_to_mesh, sdf_to_mesh_compiled,
    MarchingCubesConfig, Mesh,
};
use alice_sdf::prelude::*;
use alice_sdf::tight_aabb::compute_tight_aabb;
use std::collections::{HashMap, HashSet};

struct Scene {
    name: &'static str,
    node: SdfNode,
    lo: Vec3,
    hi: Vec3,
    res: usize,
    /// Euler characteristic of the surface (2 − 2·genus per component)
    chi: i64,
}

/// Cubic grid around the tight AABB with `margin` empty cells on each side.
fn around(node: &SdfNode, res: usize, margin: f32) -> (Vec3, Vec3) {
    let t = compute_tight_aabb(node);
    let c = (t.min + t.max) * 0.5;
    let m = (t.max - t.min).max_element() * 0.5;
    let half = m / (1.0 - 2.0 * margin / res as f32);
    (c - Vec3::splat(half), c + Vec3::splat(half))
}

fn scenes() -> Vec<Scene> {
    let mk = |name, node: SdfNode, res: usize, margin: f32, chi: i64| {
        let (lo, hi) = around(&node, res, margin);
        Scene {
            name,
            node,
            lo,
            hi,
            res,
            chi,
        }
    };
    vec![
        mk("sphere", SdfNode::sphere(0.8), 64, 3.0, 2),
        mk(
            "box",
            SdfNode::box3d(1.2, 0.8, 1.0).translate(0.1, 0.0, -0.05),
            64,
            3.0,
            2,
        ),
        // the sphere (r 0.75) pierces all six faces of the cube (half side
        // 0.6) but none of its edges (0.6·√2 > 0.75): genus 5, χ = −8
        mk(
            "box_minus_sphere",
            SdfNode::box3d(1.2, 1.2, 1.2).subtract(SdfNode::sphere(0.75)),
            64,
            3.0,
            -8,
        ),
        mk(
            "rotated_band",
            SdfNode::box3d(1.6, 0.16, 0.5).rotate_euler(0.3, 0.5, 0.7),
            64,
            2.0,
            2,
        ),
        // the cut plane x = 0 is a lattice plane: its lattice points inside
        // the ball have the value 0 exactly on every evaluator (CPU and GPU),
        // so the GPU and CPU inside rules must agree there
        Scene {
            name: "half_ball_cut_on_lattice",
            node: SdfNode::sphere(0.8).intersection(SdfNode::plane(Vec3::X, 0.0)),
            lo: Vec3::splat(-1.0),
            hi: Vec3::splat(1.0),
            res: 32,
            chi: 2,
        },
        Scene {
            name: "int_sphere_r5_cell1",
            node: SdfNode::sphere(5.0),
            lo: Vec3::splat(-8.0),
            hi: Vec3::splat(8.0),
            res: 16,
            chi: 2,
        },
        Scene {
            name: "int_sphere_r5_cell_half",
            node: SdfNode::sphere(5.0),
            lo: Vec3::splat(-8.0),
            hi: Vec3::splat(8.0),
            res: 32,
            chi: 2,
        },
    ]
}

fn cfg(res: usize) -> MarchingCubesConfig {
    MarchingCubesConfig {
        resolution: res,
        ..Default::default()
    }
}

/// `(boundary edges, non-manifold edges, Euler characteristic, components)`
/// of the index buffer.
fn topology(m: &Mesh) -> (usize, usize, i64, usize) {
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    for t in m.indices.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *edges.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let boundary = edges.values().filter(|&&n| n == 1).count();
    let non_manifold = edges.values().filter(|&&n| n > 2).count();
    // union-find over the referenced vertices
    let mut parent: Vec<u32> = (0..m.vertices.len() as u32).collect();
    const fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    let used: HashSet<u32> = m.indices.iter().copied().collect();
    for t in m.indices.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2])] {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra as usize] = rb;
            }
        }
    }
    let components: HashSet<u32> = used.iter().map(|&v| find(&mut parent, v)).collect();
    let chi = used.len() as i64 - edges.len() as i64 + (m.indices.len() / 3) as i64;
    (boundary, non_manifold, chi, components.len())
}

fn assert_closed(label: &str, m: &Mesh, expected_chi: i64) {
    let (b, nm, chi, comp) = topology(m);
    println!(
        "{label}: V {} F {} boundary {b} non-manifold {nm} chi {chi} components {comp}",
        m.vertices.len(),
        m.indices.len() / 3
    );
    assert!(m.indices.len() >= 3, "{label}: empty mesh");
    assert_eq!(b, 0, "{label}: {b} boundary edges");
    assert_eq!(nm, 0, "{label}: {nm} non-manifold edges");
    assert_eq!(comp, 1, "{label}: {comp} components");
    assert_eq!(chi, expected_chi, "{label}: Euler characteristic");
    for &i in &m.indices {
        assert!(
            (i as usize) < m.vertices.len(),
            "{label}: index out of range"
        );
    }
}

#[test]
fn cpu_meshes_are_closed_manifolds_without_welding() {
    let mut checked = 0;
    for s in scenes() {
        let c = cfg(s.res);
        let compiled = CompiledSdf::compile(&s.node);
        for (label, m) in [
            ("marching_cubes", marching_cubes(&s.node, s.lo, s.hi, &c)),
            ("sdf_to_mesh", sdf_to_mesh(&s.node, s.lo, s.hi, &c)),
            (
                "marching_cubes_compiled",
                marching_cubes_compiled(&compiled, s.lo, s.hi, &c),
            ),
            (
                "sdf_to_mesh_compiled",
                sdf_to_mesh_compiled(&compiled, s.lo, s.hi, &c),
            ),
        ] {
            assert_closed(&format!("{}/{label}", s.name), &m, s.chi);
            checked += 1;
        }
    }
    assert!(checked > 0, "no mesh checked");
}

/// Tree and compiled paths evaluate the same lattice and must extract the
/// same surface: identical position bits and identical index buffers.
#[test]
fn tree_and_compiled_paths_give_the_same_mesh() {
    let mut checked = 0;
    for s in scenes() {
        let c = cfg(s.res);
        let a = marching_cubes(&s.node, s.lo, s.hi, &c);
        let b = marching_cubes_compiled(&CompiledSdf::compile(&s.node), s.lo, s.hi, &c);
        assert_eq!(a.indices, b.indices, "{}: index buffers differ", s.name);
        assert_eq!(a.vertices.len(), b.vertices.len(), "{}", s.name);
        for (u, v) in a.vertices.iter().zip(&b.vertices) {
            assert_eq!(
                u.position.to_array().map(f32::to_bits),
                v.position.to_array().map(f32::to_bits),
                "{}: positions differ",
                s.name
            );
            checked += 1;
        }
    }
    assert!(checked > 0);
}

/// Number of lattice edges of `{lo + k·cell}` with exactly one endpoint
/// strictly inside the sphere of radius `r` at the origin, in integers:
/// coordinates are `k / q` and the test is `(qx)² + (qy)² + (qz)² < (qr)²`.
fn sphere_crossing_edges(lo_q: i64, n: i64, r_q: i64) -> usize {
    let inside = |x: i64, y: i64, z: i64| x * x + y * y + z * z < r_q * r_q;
    let mut count = 0;
    for z in 0..=n {
        for y in 0..=n {
            for x in 0..=n {
                let (px, py, pz) = (lo_q + x, lo_q + y, lo_q + z);
                let a = inside(px, py, pz);
                if x < n && a != inside(px + 1, py, pz) {
                    count += 1;
                }
                if y < n && a != inside(px, py + 1, pz) {
                    count += 1;
                }
                if z < n && a != inside(px, py, pz + 1) {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Lattice points exactly on the surface are outside, and the mesh still
/// has one vertex per sign-changing edge (none merged, none duplicated).
#[test]
fn lattice_points_on_the_surface_follow_the_strict_inside_rule() {
    // (res, q): cell = 16 / res = 1 / q
    for (res, q) in [(16usize, 1i64), (32, 2)] {
        let expected = sphere_crossing_edges(-8 * q, res as i64, 5 * q);
        // the lattice really has points on the surface (else the test is vacuous)
        let on_surface = (0..=res as i64)
            .flat_map(|x| {
                (0..=res as i64).flat_map(move |y| (0..=res as i64).map(move |z| (x, y, z)))
            })
            .filter(|&(x, y, z)| {
                let (a, b, c) = (x - 8 * q, y - 8 * q, z - 8 * q);
                a * a + b * b + c * c == 25 * q * q
            })
            .count();
        assert!(on_surface > 0, "res {res}: no lattice point on the surface");
        let node = SdfNode::sphere(5.0);
        let m = marching_cubes(&node, Vec3::splat(-8.0), Vec3::splat(8.0), &cfg(res));
        println!(
            "res {res}: lattice points on the surface {on_surface}, expected vertices {expected}, got {}",
            m.vertices.len()
        );
        assert_eq!(m.vertices.len(), expected, "res {res}");
        // closed genus 0: F = 2V − 4
        assert_eq!(m.indices.len() / 3, 2 * expected - 4, "res {res}");
    }
}

/// Lattice points exactly on the surface keep one vertex per sign-changing
/// edge: closed form of the same-position vertices and zero-area triangles.
///
/// A surface lattice point `P` (value 0, outside) with `k` strictly inside
/// axis neighbours is the crossing of `k` edges, so it carries `k` vertices
/// at its position: `Σ (k − 1)` surplus vertices and `Σ k(k − 1)/2`
/// same-position pairs. Merging each group by position and dropping the
/// triangles that collapse must leave a closed genus-0 surface
/// (`F' = 2V' − 4`); with `F = 2V − 4` before, the zero-area triangles are
/// `F − F' = 2 Σ (k − 1)`.
#[test]
fn same_position_vertices_and_zero_area_triangles_follow_the_closed_form() {
    for (res, q) in [(16usize, 1i64), (32, 2)] {
        let n = res as i64;
        let (lo_q, r2) = (-8 * q, 25 * q * q);
        let norm2 = |x: i64, y: i64, z: i64| {
            let (a, b, c) = (lo_q + x, lo_q + y, lo_q + z);
            a * a + b * b + c * c
        };
        let (mut surplus, mut pairs) = (0usize, 0usize);
        for z in 0..=n {
            for y in 0..=n {
                for x in 0..=n {
                    if norm2(x, y, z) != r2 {
                        continue;
                    }
                    let k = [
                        (1, 0, 0),
                        (-1, 0, 0),
                        (0, 1, 0),
                        (0, -1, 0),
                        (0, 0, 1),
                        (0, 0, -1),
                    ]
                    .iter()
                    .map(|&(dx, dy, dz)| (x + dx, y + dy, z + dz))
                    .filter(|&(a, b, c)| {
                        (0..=n).contains(&a) && (0..=n).contains(&b) && (0..=n).contains(&c)
                    })
                    .filter(|&(a, b, c)| norm2(a, b, c) < r2)
                    .count();
                    surplus += k.saturating_sub(1);
                    pairs += k * k.saturating_sub(1) / 2;
                }
            }
        }
        assert!(
            surplus > 0,
            "res {res}: no surface lattice point with two inside neighbours"
        );

        let m = marching_cubes(
            &SdfNode::sphere(5.0),
            Vec3::splat(-8.0),
            Vec3::splat(8.0),
            &cfg(res),
        );
        let key = |i: u32| m.vertices[i as usize].position.to_array().map(f32::to_bits);
        let mut groups: HashMap<[u32; 3], usize> = HashMap::new();
        for i in 0..m.vertices.len() as u32 {
            *groups.entry(key(i)).or_default() += 1;
        }
        let got_surplus = m.vertices.len() - groups.len();
        let got_pairs: usize = groups.values().map(|&k| k * (k - 1) / 2).sum();
        let zero_area: Vec<bool> = m
            .indices
            .chunks_exact(3)
            .map(|t| key(t[0]) == key(t[1]) || key(t[1]) == key(t[2]) || key(t[0]) == key(t[2]))
            .collect();
        let got_zero = zero_area.iter().filter(|&&z| z).count();
        println!(
            "res {res}: surplus vertices {got_surplus} (closed form {surplus}), same-position pairs {got_pairs} ({pairs}), zero-area triangles {got_zero} ({})",
            2 * surplus
        );
        assert_eq!(got_surplus, surplus, "res {res}: surplus vertices");
        assert_eq!(got_pairs, pairs, "res {res}: same-position pairs");
        assert_eq!(got_zero, 2 * surplus, "res {res}: zero-area triangles");

        // merged by position, the collapsed triangles dropped: still closed, genus 0
        let mut id: HashMap<[u32; 3], u32> = HashMap::new();
        let mut merged = Mesh::new();
        let remap: Vec<u32> = (0..m.vertices.len() as u32)
            .map(|i| {
                *id.entry(key(i)).or_insert_with(|| {
                    merged.vertices.push(m.vertices[i as usize]);
                    (merged.vertices.len() - 1) as u32
                })
            })
            .collect();
        for (t, z) in m.indices.chunks_exact(3).zip(&zero_area) {
            if !z {
                merged.indices.extend(t.iter().map(|&i| remap[i as usize]));
            }
        }
        assert_closed(&format!("res {res}/merged by position"), &merged, 2);
    }
}

/// Bourke / Lorensen cube corners (x, y, z offsets) and edges.
const CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 1],
];
const EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

/// Iso-0 crossing on the segment `p0 → p1` (values `v0`, `v1`), evaluated
/// from `p0`: `t = −v0 / (v1 − v0)` with the denominator kept away from 0
/// and `t` clamped to `[0, 1]`.
fn lerp_from(p0: Vec3, p1: Vec3, v0: f32, v1: f32) -> Vec3 {
    let d = v1 - v0;
    let d = f32::copysign(d.abs().max(1e-10), d);
    let t = ((0.0 - v0) / d).clamp(0.0, 1.0);
    p0 + (p1 - p0) * t
}

/// Every cell around a sign-changing lattice edge refers to one vertex,
/// interpolated from the edge's lower endpoint, whatever its local order.
#[test]
fn every_cell_sees_the_same_vertex_for_a_shared_edge() {
    let mut compared = 0usize;
    let mut order_sensitive = 0usize;
    for s in scenes()
        .into_iter()
        .filter(|s| s.res <= 32 || s.name == "sphere")
    {
        let res = s.res.min(32);
        let (lo, hi) = (s.lo, s.hi);
        let cell = (hi - lo) / res as f32;
        let g = res + 1;
        let point = |c: [usize; 3]| {
            lo + Vec3::new(
                c[0] as f32 * cell.x,
                c[1] as f32 * cell.y,
                c[2] as f32 * cell.z,
            )
        };
        let values: Vec<f32> = (0..g * g * g)
            .map(|i| eval(&s.node, point([i % g, (i / g) % g, i / (g * g)])))
            .collect();
        let at = |c: [usize; 3]| values[c[0] + c[1] * g + c[2] * g * g];
        let mesh = marching_cubes(&s.node, lo, hi, &cfg(res));
        let mesh_bits: HashMap<[u32; 3], usize> = mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(i, v)| (v.position.to_array().map(f32::to_bits), i))
            .collect();
        // lattice edge -> vertex index seen from each cell
        let mut by_edge: HashMap<([usize; 3], usize), Vec<usize>> = HashMap::new();
        for z in 0..res {
            for y in 0..res {
                for x in 0..res {
                    for ends in EDGES {
                        let [a, b] = ends.map(|k| {
                            let o = CORNERS[k];
                            [x + o[0], y + o[1], z + o[2]]
                        });
                        let (va, vb) = (at(a), at(b));
                        if (va < 0.0) == (vb < 0.0) {
                            continue;
                        }
                        let lower_first = a <= b;
                        let (p0, p1, v0, v1) = if lower_first {
                            (point(a), point(b), va, vb)
                        } else {
                            (point(b), point(a), vb, va)
                        };
                        let canonical = lerp_from(p0, p1, v0, v1);
                        if lerp_from(point(a), point(b), va, vb) != canonical {
                            order_sensitive += 1;
                        }
                        let bits = canonical.to_array().map(f32::to_bits);
                        let Some(&vi) = mesh_bits.get(&bits) else {
                            panic!(
                                "{}: cell {:?} edge {:?}-{:?}: {canonical:?} is not a mesh vertex",
                                s.name,
                                [x, y, z],
                                a,
                                b
                            );
                        };
                        let lower = [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])];
                        let axis = (0..3).find(|&k| a[k] != b[k]).unwrap_or(0);
                        by_edge.entry((lower, axis)).or_default().push(vi);
                        compared += 1;
                    }
                }
            }
        }
        let mut shared = 0;
        for (edge, seen) in &by_edge {
            assert!(
                seen.iter().all(|&v| v == seen[0]),
                "{}: edge {edge:?} maps to vertices {seen:?}",
                s.name
            );
            shared += usize::from(seen.len() > 1);
        }
        assert_eq!(
            by_edge.len(),
            mesh.vertices.len(),
            "{}: one vertex per edge",
            s.name
        );
        assert!(shared > 0, "{}: no edge shared by two cells", s.name);
    }
    println!(
        "compared {compared} per-cell edge lookups, {order_sensitive} would differ if interpolated in the cell's local order"
    );
    assert!(compared > 0, "no edge compared");
}

fn max_normal_angle_on_sphere(m: &Mesh) -> f32 {
    m.vertices
        .iter()
        .map(|v| v.normal.angle_between(v.position.normalize()))
        .fold(0.0f32, f32::max)
}

#[test]
fn vertex_normals_are_the_gradient_at_the_vertex() {
    let node = SdfNode::sphere(0.8);
    let (lo, hi) = around(&node, 64, 3.0);
    let c = cfg(64);
    for (label, m) in [
        ("marching_cubes", marching_cubes(&node, lo, hi, &c)),
        (
            "marching_cubes_compiled",
            marching_cubes_compiled(&CompiledSdf::compile(&node), lo, hi, &c),
        ),
    ] {
        assert!(!m.vertices.is_empty());
        let worst = max_normal_angle_on_sphere(&m);
        println!("{label}: worst normal angle {worst:.2e} rad");
        assert!(worst < 1e-3, "{label}: normal {worst} rad from p/|p|");
    }
}

fn signed_volume(m: &Mesh) -> f64 {
    m.indices
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [0, 1, 2].map(|k| m.vertices[t[k] as usize].position.as_dvec3());
            a.dot(b.cross(c)) / 6.0
        })
        .sum()
}

#[test]
fn volume_matches_the_closed_form() {
    let r = 0.8f64;
    let cases = [
        (
            "sphere",
            SdfNode::sphere(r as f32),
            4.0 / 3.0 * std::f64::consts::PI * r * r * r,
        ),
        (
            "box",
            SdfNode::box3d(1.2, 0.8, 1.0).translate(0.1, 0.0, -0.05),
            1.2 * 0.8 * 1.0,
        ),
    ];
    for (name, node, truth) in cases {
        let (lo, hi) = around(&node, 64, 3.0);
        let m = sdf_to_mesh(&node, lo, hi, &cfg(64));
        let v = signed_volume(&m);
        let rel = (v - truth).abs() / truth;
        println!("{name}: volume {v:.6} closed form {truth:.6} rel {rel:.2e}");
        assert!(rel < 0.01, "{name}: volume rel {rel}");
    }
}

#[cfg(feature = "gpu-mesh")]
mod gpu {
    use super::*;
    use alice_sdf::mesh::{gpu_marching_cubes, GpuMarchingCubesConfig};

    fn gpu_mesh(s: &Scene) -> Option<Mesh> {
        let c = GpuMarchingCubesConfig {
            resolution: s.res as u32,
            ..Default::default()
        };
        match gpu_marching_cubes(&s.node, s.lo, s.hi, &c) {
            Ok(m) => Some(m),
            Err(e) => {
                assert!(
                    std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                    "ALICE_SDF_REQUIRE_GPU is set but GPU marching cubes failed: {e}"
                );
                eprintln!("no GPU adapter: skipped ({e})");
                None
            }
        }
    }

    #[test]
    fn gpu_mesh_is_closed_and_matches_the_cpu_topology() {
        let mut checked = 0;
        for s in scenes() {
            let Some(g) = gpu_mesh(&s) else { return };
            let cpu = marching_cubes(&s.node, s.lo, s.hi, &cfg(s.res));
            assert_closed(&format!("{}/gpu", s.name), &g, s.chi);
            println!(
                "{}: GPU V {} F {} / CPU V {} F {}",
                s.name,
                g.vertices.len(),
                g.indices.len() / 3,
                cpu.vertices.len(),
                cpu.indices.len() / 3
            );
            assert_eq!(
                g.vertices.len(),
                cpu.vertices.len(),
                "{}: vertex count",
                s.name
            );
            assert_eq!(
                g.indices.len(),
                cpu.indices.len(),
                "{}: triangle count",
                s.name
            );
            checked += 1;
        }
        assert!(checked > 0, "no scene checked");
    }

    #[test]
    fn gpu_normals_are_the_gradient_at_the_vertex() {
        let node = SdfNode::sphere(0.8);
        let (lo, hi) = around(&node, 64, 3.0);
        let s = Scene {
            name: "sphere",
            node,
            lo,
            hi,
            res: 64,
            chi: 2,
        };
        let Some(g) = gpu_mesh(&s) else { return };
        assert!(!g.vertices.is_empty());
        let worst = max_normal_angle_on_sphere(&g);
        println!("gpu: worst normal angle {worst:.2e} rad");
        assert!(worst < 1e-3, "gpu: normal {worst} rad from p/|p|");
    }
}
