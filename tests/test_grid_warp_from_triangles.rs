//! `grid_warp_from_triangles`: a `GridWarpData` from a surface given twice, at rest and posed
//! (a cloth, a skinned mesh): each grid node holds the affine map from the posed space back
//! to the rest space of the posed triangle nearest to it, so a rest-space SDF warped by the
//! grid follows the surface.
//!
//! Oracles (closed forms, not the builder's own output):
//! - a triangle's map sends its three posed vertices to the rest vertices, and the posed unit
//!   normal from its first vertex to the rest unit normal (the definition of the map);
//! - posed = rest: every map is the identity, and the warped SDF equals the child;
//! - posed = rest + t: the warped SDF at p is the child at p − t;
//! - posed = rest turned 90° about z: the warped SDF at p is the child at the point turned back;
//! - a flat sheet stretched ×2 along x in its plane: the warped SDF at p is the child at
//!   (p.x / 2, p.y, p.z) (the normal keeps its length, so thickness is kept);
//! - bad input is an error, never a grid.
//!
//! Every evaluation goes through `SdfNode::grid_warp` and `eval`, the production path.
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::transforms::{
    affine_posed_to_rest, grid_warp_from_triangles, GridWarp, GridWarpBuildError,
};
use alice_sdf::SdfNode;
use glam::Vec3;

/// A flat n × n sheet in the plane z = 0 over [0, 1]², two triangles per cell.
fn sheet(n: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut v = Vec::new();
    for j in 0..n {
        for i in 0..n {
            v.push(Vec3::new(
                i as f32 / (n - 1) as f32,
                j as f32 / (n - 1) as f32,
                0.0,
            ));
        }
    }
    let mut t = Vec::new();
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let a = (j * n + i) as u32;
            let (b, c, d) = (a + 1, a + n as u32, a + n as u32 + 1);
            t.push([a, b, d]);
            t.push([a, d, c]);
        }
    }
    (v, t)
}

/// A small rest-space shape near the sheet: a sphere centred on it.
fn child() -> SdfNode {
    SdfNode::sphere(0.3).translate(0.5, 0.5, 0.0)
}

fn warped(rest: &[Vec3], posed: &[Vec3], tris: &[[u32; 3]], lo: Vec3, hi: Vec3) -> SdfNode {
    let data = grid_warp_from_triangles(rest, posed, tris, lo, hi, 0.05, 0.5).expect("builds");
    child().grid_warp(GridWarp::new(data).expect("a valid grid"))
}

/// Points within 0.2 of the posed surface. The boxes leave a margin of 0.6 around the posed
/// sheet, more than the shape's radius 0.3, so the box (which clips a grid warp's shape)
/// never decides a value here.
fn probes(map: impl Fn(Vec3) -> Vec3) -> Vec<Vec3> {
    let mut v = Vec::new();
    for j in 1..8 {
        for i in 1..8 {
            for z in [-0.15f32, -0.05, 0.0, 0.05, 0.15] {
                v.push(map(Vec3::new(i as f32 / 8.0, j as f32 / 8.0, z)));
            }
        }
    }
    v
}

fn assert_close(got: f32, want: f32, tol: f32, what: &str) {
    assert!((got - want).abs() <= tol, "{what}: got {got}, want {want}");
}

#[test]
fn a_triangle_map_sends_posed_vertices_and_normal_to_rest() {
    let rest = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let posed = [
        Vec3::new(0.2, -0.1, 0.4),
        Vec3::new(1.7, 0.3, 0.9),
        Vec3::new(-0.1, 1.2, 0.6),
    ];
    let m = affine_posed_to_rest(rest, posed).expect("not degenerate");
    let apply = |p: Vec3| {
        Vec3::new(
            m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
            m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
            m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11],
        )
    };
    for k in 0..3 {
        let q = apply(posed[k]);
        assert!(
            (q - rest[k]).length() <= 1e-5,
            "vertex {k}: {q:?} vs {:?}",
            rest[k]
        );
    }
    let np = (posed[1] - posed[0]).cross(posed[2] - posed[0]).normalize();
    let nr = (rest[1] - rest[0]).cross(rest[2] - rest[0]).normalize();
    let q = apply(posed[0] + np);
    assert!((q - (rest[0] + nr)).length() <= 1e-5, "normal: {q:?}");
    // a triangle without area has no map
    assert!(affine_posed_to_rest(rest, [posed[0], posed[0], posed[2]]).is_none());
}

#[test]
fn posed_equal_rest_is_the_identity() {
    let (rest, tris) = sheet(5);
    let lo = Vec3::new(-0.6, -0.6, -0.6);
    let hi = Vec3::new(1.6, 1.6, 0.6);
    let data = grid_warp_from_triangles(&rest, &rest, &tris, lo, hi, 0.05, 0.5).expect("builds");
    let id = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    for (k, m) in data.maps.chunks(12).enumerate() {
        for (a, b) in m.iter().zip(id) {
            assert!((a - b).abs() <= 1e-6, "node {k}: {m:?}");
        }
    }
    let node = warped(&rest, &rest, &tris, lo, hi);
    for p in probes(|p| p) {
        assert_close(eval(&node, p), eval(&child(), p), 1e-5, "identity");
    }
}

#[test]
fn a_translated_surface_carries_the_shape() {
    let (rest, tris) = sheet(5);
    let t = Vec3::new(0.3, -0.2, 0.1);
    let posed: Vec<Vec3> = rest.iter().map(|p| *p + t).collect();
    let node = warped(
        &rest,
        &posed,
        &tris,
        Vec3::new(-0.3, -0.8, -0.5),
        Vec3::new(1.9, 1.4, 0.7),
    );
    for p in probes(|p| p + t) {
        assert_close(eval(&node, p), eval(&child(), p - t), 1e-4, "translation");
    }
}

#[test]
fn a_turned_surface_carries_the_shape() {
    // 90° about z through the sheet's centre: (x, y) -> (c.x − (y − c.y), c.y + (x − c.x))
    let c = Vec3::new(0.5, 0.5, 0.0);
    let turn = |p: Vec3| Vec3::new(c.x - (p.y - c.y), c.y + (p.x - c.x), p.z);
    let back = |p: Vec3| Vec3::new(c.x + (p.y - c.y), c.y - (p.x - c.x), p.z);
    let (rest, tris) = sheet(5);
    let posed: Vec<Vec3> = rest.iter().map(|p| turn(*p)).collect();
    let node = warped(
        &rest,
        &posed,
        &tris,
        Vec3::new(-0.6, -0.6, -0.6),
        Vec3::new(1.6, 1.6, 0.6),
    );
    for p in probes(turn) {
        assert_close(eval(&node, p), eval(&child(), back(p)), 1e-4, "rotation");
    }
}

#[test]
fn a_sheet_stretched_in_its_plane_keeps_its_thickness() {
    let (rest, tris) = sheet(5);
    let posed: Vec<Vec3> = rest
        .iter()
        .map(|p| Vec3::new(2.0 * p.x, p.y, p.z))
        .collect();
    let node = warped(
        &rest,
        &posed,
        &tris,
        Vec3::new(-0.6, -0.6, -0.6),
        Vec3::new(2.6, 1.6, 0.6),
    );
    for p in probes(|p| Vec3::new(2.0 * p.x, p.y, p.z)) {
        let want = eval(&child(), Vec3::new(p.x / 2.0, p.y, p.z));
        let got = eval(&node, p);
        // the warp stretches distances along x by up to 2: the node returns a safe lower
        // bound of the posed distance, so compare the sign and the zero set: inside / outside
        // agree, and the value never exceeds the rest-space distance
        assert_eq!(
            got < 0.0,
            want < 0.0,
            "inside/outside at {p:?}: got {got}, want {want}"
        );
        assert!(
            got.abs() <= want.abs() + 1e-4,
            "not a lower bound at {p:?}: {got} vs {want}"
        );
    }
}

#[test]
fn bad_input_is_an_error() {
    let (rest, tris) = sheet(3);
    let lo = Vec3::splat(-0.5);
    let hi = Vec3::splat(1.5);
    let b = |r: &[Vec3], p: &[Vec3], t: &[[u32; 3]], lo, hi, cell| {
        grid_warp_from_triangles(r, p, t, lo, hi, cell, 0.5)
    };
    assert!(matches!(
        b(&rest, &rest[1..], &tris, lo, hi, 0.1),
        Err(GridWarpBuildError::LengthMismatch { .. })
    ));
    assert!(matches!(
        b(&rest, &rest, &[], lo, hi, 0.1),
        Err(GridWarpBuildError::NoTriangles)
    ));
    assert!(matches!(
        b(&rest, &rest, &[[0, 1, 99]], lo, hi, 0.1),
        Err(GridWarpBuildError::IndexOutOfRange { .. })
    ));
    assert!(matches!(
        b(&rest, &rest, &[[0, 0, 1]], lo, hi, 0.1),
        Err(GridWarpBuildError::Degenerate { triangle: 0 })
    ));
    assert!(matches!(
        b(&rest, &rest, &tris, hi, lo, 0.1),
        Err(GridWarpBuildError::BadBox)
    ));
    assert!(matches!(
        b(&rest, &rest, &tris, lo, hi, 0.0),
        Err(GridWarpBuildError::BadCell)
    ));
    assert!(matches!(
        b(&rest, &rest, &tris, lo, hi, f32::NAN),
        Err(GridWarpBuildError::BadCell)
    ));
    let mut nan = rest.clone();
    nan[2].x = f32::NAN;
    assert!(matches!(
        b(&rest, &nan, &tris, lo, hi, 0.1),
        Err(GridWarpBuildError::NotFinite)
    ));
}

#[test]
fn each_node_takes_the_map_of_the_nearest_triangle() {
    // two separate sheets moved apart in opposite directions: a node near one must carry its
    // map, not the other's. The sheets are 3 apart so the jump between the two maps (half way)
    // is farther than the reach (plus a cell) from every probe: there the warp is rigid and
    // the node returns the distance itself, not GridWarp's lower bound
    let (a, ta) = sheet(3);
    let b: Vec<Vec3> = a.iter().map(|p| *p + Vec3::new(4.0, 0.0, 0.0)).collect();
    let rest: Vec<Vec3> = a.iter().chain(&b).copied().collect();
    let off = a.len() as u32;
    let tris: Vec<[u32; 3]> = ta
        .iter()
        .copied()
        .chain(ta.iter().map(|t| [t[0] + off, t[1] + off, t[2] + off]))
        .collect();
    let (ua, ub) = (Vec3::new(0.0, 0.0, 0.2), Vec3::new(0.0, 0.0, -0.2));
    let posed: Vec<Vec3> = a
        .iter()
        .map(|p| *p + ua)
        .chain(b.iter().map(|p| *p + ub))
        .collect();
    let shape = SdfNode::sphere(0.3)
        .translate(0.5, 0.5, 0.0)
        .union(SdfNode::sphere(0.3).translate(4.5, 0.5, 0.0));
    let data = grid_warp_from_triangles(
        &rest,
        &posed,
        &tris,
        Vec3::new(-0.6, -0.6, -0.8),
        Vec3::new(5.6, 1.6, 0.8),
        0.05,
        0.5,
    )
    .expect("builds");
    let node = shape.clone().grid_warp(GridWarp::new(data).expect("valid"));
    let mut n = 0;
    for (u, x0) in [(ua, 0.0f32), (ub, 4.0)] {
        for p in probes(|p| p + Vec3::new(x0, 0.0, 0.0) + u) {
            assert_close(eval(&node, p), eval(&shape, p - u), 1e-4, "two patches");
            n += 1;
        }
    }
    assert!(n > 0);
}
