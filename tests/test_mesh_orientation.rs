//! Marching-cubes output invariants: outward winding, positive signed volume,
//! watertightness, no duplicate vertices (external review round 2, 2026-09-15:
//! every triangle of every generated mesh was wound inward, so STL facets
//! contradicted their stored normals and slicers saw an inside-out solid).
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::{sdf_to_mesh, MarchingCubesConfig, Mesh};
use alice_sdf::prelude::*;
use std::collections::HashMap;

fn mesh_of(node: &SdfNode, res: usize) -> Mesh {
    let cfg = MarchingCubesConfig {
        resolution: res,
        ..Default::default()
    };
    sdf_to_mesh(node, Vec3::splat(-1.6), Vec3::splat(1.6), &cfg)
}

/// Divergence theorem: Σ (a · (b × c)) / 6 over CCW-outward triangles.
fn signed_volume(m: &Mesh) -> f32 {
    m.indices
        .chunks_exact(3)
        .map(|t| {
            let a = m.vertices[t[0] as usize].position;
            let b = m.vertices[t[1] as usize].position;
            let c = m.vertices[t[2] as usize].position;
            a.dot(b.cross(c)) / 6.0
        })
        .sum()
}

/// (outward, inward) counts of geometric face normals against the SDF gradient
/// at the triangle centroid. Slivers below 1e-3 of a cell face (a corner
/// value sitting on the iso-level) have no reliable orientation and are
/// skipped.
fn winding(node: &SdfNode, m: &Mesh, cell: f32) -> (usize, usize) {
    let (mut out, mut inw) = (0, 0);
    let min_area = 1e-3 * cell * cell;
    for t in m.indices.chunks_exact(3) {
        let a = m.vertices[t[0] as usize].position;
        let b = m.vertices[t[1] as usize].position;
        let c = m.vertices[t[2] as usize].position;
        let geo = (b - a).cross(c - a);
        if 0.5 * geo.length() < min_area {
            continue;
        }
        let grad = normal(node, (a + b + c) / 3.0, 1e-3);
        let cos = geo.dot(grad) / geo.length();
        if cos > 0.0 {
            out += 1;
        } else {
            eprintln!(
                "inward: area {:.2e} cos {cos:.3} at {:?}",
                0.5 * geo.length(),
                (a + b + c) / 3.0
            );
            inw += 1;
        }
    }
    (out, inw)
}

/// Edges referenced by exactly one triangle (directed-edge pairing).
fn open_edges(m: &Mesh) -> usize {
    let mut count: HashMap<(u32, u32), i32> = HashMap::new();
    for t in m.indices.chunks_exact(3) {
        for (u, v) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let key = if u < v { (u, v) } else { (v, u) };
            *count.entry(key).or_insert(0) += if u < v { 1 } else { -1 };
        }
    }
    count.values().filter(|&&c| c != 0).count()
}

fn distinct_positions(m: &Mesh) -> usize {
    let mut set = std::collections::HashSet::new();
    for v in &m.vertices {
        set.insert(v.position.to_array().map(f32::to_bits));
    }
    set.len()
}

#[test]
fn marching_cubes_triangles_face_outward() {
    for (name, node) in [
        ("sphere", SdfNode::sphere(1.0)),
        ("torus", SdfNode::torus(0.8, 0.3)),
        ("box", SdfNode::box3d(0.7, 0.5, 0.9)),
        (
            "subtract",
            SdfNode::box3d(1.8, 1.8, 1.8).subtract(SdfNode::sphere(1.1)),
        ),
    ] {
        let m = mesh_of(&node, 32);
        let (out, inw) = winding(&node, &m, 3.2 / 32.0);
        assert!(
            inw == 0 && out > 0,
            "{name}: {out} outward / {inw} inward triangles (all must face outward)"
        );
    }
}

#[test]
fn signed_volume_is_positive_and_converges_to_analytic() {
    let node = SdfNode::sphere(1.0);
    let truth = 4.0 / 3.0 * std::f32::consts::PI;
    for res in [16usize, 32, 64] {
        let v = signed_volume(&mesh_of(&node, res));
        assert!(v > 0.0, "res {res}: signed volume {v} must be positive");
        let rel = (v - truth).abs() / truth;
        // inscribed-polyhedron error shrinks with resolution; 5 % at res 16
        assert!(
            rel < 0.05,
            "res {res}: volume {v} vs {truth} ({rel:.3} rel)"
        );
    }
    let torus = SdfNode::torus(0.8, 0.3);
    let v = signed_volume(&mesh_of(&torus, 64));
    let truth = 2.0 * std::f32::consts::PI * std::f32::consts::PI * 0.8 * 0.3 * 0.3;
    assert!(
        v > 0.0 && (v - truth).abs() / truth < 0.05,
        "torus volume {v} vs {truth}"
    );
}

#[test]
fn sdf_to_mesh_is_watertight_with_shared_vertices() {
    let node = SdfNode::sphere(1.0);
    for res in [16usize, 32, 64] {
        let m = mesh_of(&node, res);
        let degenerate = m
            .indices
            .chunks_exact(3)
            .filter(|t| t[0] == t[1] || t[1] == t[2] || t[0] == t[2])
            .count();
        assert_eq!(
            degenerate, 0,
            "res {res}: {degenerate} degenerate triangles"
        );
        let oe = open_edges(&m);
        assert_eq!(
            oe,
            0,
            "res {res}: {oe} open edges of {} vertices ({} distinct)",
            m.vertices.len(),
            distinct_positions(&m)
        );
        let distinct = distinct_positions(&m);
        assert_eq!(
            m.vertices.len(),
            distinct,
            "res {res}: {} vertices for {distinct} distinct positions",
            m.vertices.len()
        );
    }
}

/// The compiled marching-cubes path shares the corner numbering fix.
#[test]
fn compiled_marching_cubes_faces_outward() {
    use alice_sdf::mesh::sdf_to_mesh_compiled;
    let node = SdfNode::torus(0.8, 0.3);
    let compiled = CompiledSdf::compile(&node);
    let cfg = MarchingCubesConfig {
        resolution: 32,
        ..Default::default()
    };
    let m = sdf_to_mesh_compiled(&compiled, Vec3::splat(-1.6), Vec3::splat(1.6), &cfg);
    let (out, inw) = winding(&node, &m, 3.2 / 32.0);
    assert!(inw == 0 && out > 0, "{out} outward / {inw} inward");
    assert!(signed_volume(&m) > 0.0);
}

/// STL: the stored facet normal and the right-hand rule over the three
/// vertices must agree (slicers ignore the stored normal and use the
/// winding; until 1.10.3 every facet contradicted its normal).
#[test]
fn stl_facets_agree_with_their_normals() {
    use alice_sdf::io::stl::{export_stl, import_stl};
    let node = SdfNode::sphere(1.0);
    let m = mesh_of(&node, 32);
    let dir = std::env::temp_dir().join(format!("alice_sdf_stl_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sphere.stl");
    export_stl(&m, &path).unwrap();
    let back = import_stl(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let (mut agree, mut contradict) = (0, 0);
    for t in back.indices.chunks_exact(3) {
        let a = back.vertices[t[0] as usize];
        let b = back.vertices[t[1] as usize].position;
        let c = back.vertices[t[2] as usize].position;
        let geo = (b - a.position).cross(c - a.position);
        if geo.length() < 1e-9 {
            continue;
        }
        if geo.dot(a.normal) > 0.0 {
            agree += 1;
        } else {
            contradict += 1;
        }
    }
    assert!(
        contradict == 0 && agree > 0,
        "{agree} facets agree, {contradict} contradict"
    );
    assert!(signed_volume(&back) > 0.0);
}

/// GPU marching cubes shares the corner numbering with the CPU tables
/// (Port Parity Oracle: run locally on Metal / Vulkan, skipped without an
/// adapter).
#[cfg(feature = "gpu-mesh")]
#[test]
fn gpu_marching_cubes_faces_outward() {
    use alice_sdf::mesh::{gpu_marching_cubes, GpuMarchingCubesConfig};
    let node = SdfNode::sphere(1.0);
    let cfg = GpuMarchingCubesConfig {
        resolution: 32,
        ..Default::default()
    };
    let Ok(m) = gpu_marching_cubes(&node, Vec3::splat(-1.6), Vec3::splat(1.6), &cfg) else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let (out, inw) = winding(&node, &m, 3.2 / 32.0);
    assert!(inw == 0 && out > 0, "GPU MC: {out} outward / {inw} inward");
    let v = signed_volume(&m);
    let truth = 4.0 / 3.0 * std::f32::consts::PI;
    assert!(
        v > 0.0 && (v - truth).abs() / truth < 0.05,
        "GPU MC volume {v} vs {truth}"
    );
}
