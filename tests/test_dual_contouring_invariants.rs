//! Dual contouring output invariants — the same oracle as marching cubes
//! (`tests/test_mesh_orientation.rs`) plus the property DC exists for:
//! sharp features. A box's vertices must sit on its faces / edges and its
//! volume error must beat marching cubes at the same resolution.
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::{
    dual_contouring, sdf_to_mesh, DualContouringConfig, MarchingCubesConfig, Mesh,
};
use alice_sdf::prelude::*;
use std::collections::HashMap;

const BOUND: f32 = 1.6;

fn dc(node: &SdfNode, res: usize) -> Mesh {
    let cfg = DualContouringConfig {
        resolution: res,
        ..Default::default()
    };
    dual_contouring(node, Vec3::splat(-BOUND), Vec3::splat(BOUND), &cfg)
}

fn mc(node: &SdfNode, res: usize) -> Mesh {
    let cfg = MarchingCubesConfig {
        resolution: res,
        ..Default::default()
    };
    sdf_to_mesh(node, Vec3::splat(-BOUND), Vec3::splat(BOUND), &cfg)
}

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

/// Every triangle, whatever its area: where the surface is tangent to a grid
/// plane (torus inner equator at radius R − r = 0.5 on the 0.1 grid) DC used
/// to emit folded fins of ~0.5 % cell area between the two rows of dual
/// vertices on either side of the plane; 1.14.0 collapses them, so nothing
/// is skipped here any more.
fn winding(node: &SdfNode, m: &Mesh, _cell: f32) -> (usize, usize) {
    let (mut out, mut inw) = (0, 0);
    for t in m.indices.chunks_exact(3) {
        let a = m.vertices[t[0] as usize].position;
        let b = m.vertices[t[1] as usize].position;
        let c = m.vertices[t[2] as usize].position;
        let geo = (b - a).cross(c - a);
        assert!(
            geo.length() > 0.0,
            "zero-area triangle {:?} {:?} {:?}",
            a,
            b,
            c
        );
        let grad = normal(node, (a + b + c) / 3.0, 1e-3);
        if geo.dot(grad) > 0.0 {
            out += 1;
        } else {
            eprintln!(
                "inward: area {:.2e} cos {:.3} centroid {:?} |f|={:.3e}",
                0.5 * geo.length(),
                geo.dot(grad) / geo.length(),
                (a + b + c) / 3.0,
                eval(node, (a + b + c) / 3.0).abs()
            );
            inw += 1;
        }
    }
    (out, inw)
}

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

fn max_abs_f_at_vertices(node: &SdfNode, m: &Mesh) -> f32 {
    m.vertices
        .iter()
        .map(|v| eval(node, v.position).abs())
        .fold(0.0, f32::max)
}

#[test]
fn dual_contouring_faces_outward_and_is_closed() {
    let res = 32;
    let cell = 2.0 * BOUND / res as f32;
    for (name, node) in [
        ("sphere", SdfNode::sphere(1.0)),
        ("torus", SdfNode::torus(0.8, 0.3)),
        ("box", SdfNode::box3d(1.4, 1.0, 1.8)),
        (
            "subtract",
            SdfNode::box3d(1.8, 1.8, 1.8).subtract(SdfNode::sphere(1.1)),
        ),
    ] {
        let m = dc(&node, res);
        assert!(!m.indices.is_empty(), "{name}: empty mesh");
        let (out, inw) = winding(&node, &m, cell);
        assert!(inw == 0 && out > 0, "{name}: {out} outward / {inw} inward");
        assert_eq!(open_edges(&m), 0, "{name}: open edges");
        let worst = max_abs_f_at_vertices(&node, &m);
        assert!(
            worst < cell,
            "{name}: a vertex is {worst} from the surface (cell {cell})"
        );
    }
}

#[test]
fn dual_contouring_volume_matches_analytic() {
    let sphere = SdfNode::sphere(1.0);
    let truth = 4.0 / 3.0 * std::f32::consts::PI;
    for res in [16usize, 32, 64] {
        let v = signed_volume(&dc(&sphere, res));
        let rel = (v - truth).abs() / truth;
        assert!(
            v > 0.0 && rel < 0.05,
            "res {res}: volume {v} vs {truth} ({rel:.3})"
        );
    }
    let torus = SdfNode::torus(0.8, 0.3);
    let truth = 2.0 * std::f32::consts::PI * std::f32::consts::PI * 0.8 * 0.3 * 0.3;
    let v = signed_volume(&dc(&torus, 64));
    assert!(
        v > 0.0 && (v - truth).abs() / truth < 0.05,
        "torus {v} vs {truth}"
    );
}

/// The reason DC exists: a box keeps its edges and corners. Every vertex
/// sits on the box surface (|f| ≈ 0, not just within a cell), and the
/// volume error is below marching cubes' at the same resolution.
#[test]
fn dual_contouring_preserves_sharp_features() {
    let node = SdfNode::box3d(1.4, 1.0, 1.8); // half extents 0.7 / 0.5 / 0.9
    let truth = 1.4 * 1.0 * 1.8;
    for res in [16usize, 32] {
        let cell = 2.0 * BOUND / res as f32;
        let m = dc(&node, res);
        let worst = max_abs_f_at_vertices(&node, &m);
        assert!(
            worst < 0.05 * cell,
            "res {res}: DC vertex {worst} off the box surface (cell {cell})"
        );
        let dc_err = (signed_volume(&m) - truth).abs() / truth;
        let mc_err = (signed_volume(&mc(&node, res)) - truth).abs() / truth;
        assert!(
            dc_err < 0.02 && dc_err <= mc_err,
            "res {res}: DC volume error {dc_err:.4} vs MC {mc_err:.4}"
        );
        // corners: the 8 box corners are reproduced within a small fraction of a cell
        for sx in [-0.7f32, 0.7] {
            for sy in [-0.5f32, 0.5] {
                for sz in [-0.9f32, 0.9] {
                    let corner = Vec3::new(sx, sy, sz);
                    let nearest = m
                        .vertices
                        .iter()
                        .map(|v| (v.position - corner).length())
                        .fold(f32::MAX, f32::min);
                    assert!(
                        nearest < 0.25 * cell,
                        "res {res}: corner {corner:?} nearest vertex {nearest} (cell {cell})"
                    );
                }
            }
        }
    }
}

/// Tangent configurations depend on where the grid planes fall: the torus
/// inner equator (radius 0.5) and the sphere (radius 1) sit exactly on grid
/// planes at some resolutions and between them at others. All of them must
/// come out closed, outward and free of fins.
#[test]
fn dual_contouring_is_fin_free_across_resolutions() {
    for res in [16usize, 24, 32, 40, 64] {
        let cell = 2.0 * BOUND / res as f32;
        for (name, node) in [
            ("torus", SdfNode::torus(0.8, 0.3)),
            ("sphere", SdfNode::sphere(1.0)),
            ("cylinder", SdfNode::cylinder(0.5, 1.0)),
        ] {
            let m = dc(&node, res);
            let (out, inw) = winding(&node, &m, cell);
            assert!(
                inw == 0 && out > 0,
                "{name} res {res}: {out} outward / {inw} inward"
            );
            assert_eq!(open_edges(&m), 0, "{name} res {res}: open edges");
            let worst = max_abs_f_at_vertices(&node, &m);
            assert!(
                worst < cell,
                "{name} res {res}: vertex {worst} off the surface"
            );
        }
    }
}
