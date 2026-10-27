//! The generalized winding number of a triangle mesh, and the sign it gives a mesh-derived SDF
//! that is not watertight.
//!
//! oracle (independent of the implementation):
//! - the solid angle of the octant triangle `(1,0,0) (0,1,0) (0,0,1)` seen from the origin is
//!   `4π / 8` (it covers one octant of the unit sphere), so its winding contribution is `1/8`;
//! - for a closed, consistently outward-wound polyhedron the winding number is exactly 1 inside
//!   and 0 outside (Gauss), so a sphere mesh gives 1 / 0 up to rounding;
//! - with a hole or a slit, a point deep inside still sees almost the whole surface, so its
//!   winding stays near 1 (well above ½), while a flood fill leaks through the opening;
//! - two closed boxes that touch, or that leave a 1 mm gap between them: inside each box the
//!   winding is 1, in the gap and outside it is 0;
//! - a flipped triangle makes the directed edges it shares appear twice in the same direction:
//!   the winding report counts them, and a mesh SDF with the winding sign refuses that mesh.
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::winding::{winding_number, winding_report};
use alice_sdf::mesh::{BvhTriangle, MeshSdf, MeshSignMode, MeshToSdfConfig};
use glam::Vec3;

/// A UV sphere of radius 1, outward wound, keeping the quads `keep(row, column)` accepts
fn uv_sphere(lat: u32, lon: u32, keep: impl Fn(u32, u32) -> bool) -> (Vec<Vec3>, Vec<u32>) {
    let mut v = Vec::new();
    // poles as single vertices so the closed sphere is a closed (manifold) mesh
    v.extend([Vec3::Y, -Vec3::Y]);
    for i in 1..lat {
        let th = std::f32::consts::PI * i as f32 / lat as f32;
        for j in 0..lon {
            let ph = std::f32::consts::TAU * j as f32 / lon as f32;
            v.extend([Vec3::new(
                alice_det_math::sin(th) * alice_det_math::cos(ph),
                alice_det_math::cos(th),
                alice_det_math::sin(th) * alice_det_math::sin(ph),
            )]);
        }
    }
    let ring = |i: u32, j: u32| 2 + (i - 1) * lon + (j % lon);
    let mut idx: Vec<u32> = Vec::new();
    for i in 0..lat {
        for j in 0..lon {
            if !keep(i, j) {
                continue;
            }
            if i == 0 {
                idx.extend([0, ring(1, j + 1), ring(1, j)]);
            } else if i == lat - 1 {
                idx.extend([1, ring(lat - 1, j), ring(lat - 1, j + 1)]);
            } else {
                let (a, b) = (ring(i, j), ring(i, j + 1));
                let (c, d) = (ring(i + 1, j), ring(i + 1, j + 1));
                idx.extend([a, b, c, b, d, c]);
            }
        }
    }
    (v, idx)
}

fn triangles(v: &[Vec3], idx: &[u32]) -> Vec<BvhTriangle> {
    idx.chunks_exact(3)
        .map(|t| BvhTriangle::new(v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]))
        .collect()
}

/// An axis box `[lo, hi]`, outward wound, appended to `(v, idx)`
fn add_box(v: &mut Vec<Vec3>, idx: &mut Vec<u32>, lo: Vec3, hi: Vec3) {
    let b = u32::try_from(v.len()).unwrap();
    for k in 0..8u32 {
        v.extend([Vec3::new(
            if k & 1 == 0 { lo.x } else { hi.x },
            if k & 2 == 0 { lo.y } else { hi.y },
            if k & 4 == 0 { lo.z } else { hi.z },
        )]);
    }
    // each face as two triangles, counter-clockwise seen from outside
    for f in [
        [0, 2, 3, 1], // -z
        [4, 5, 7, 6], // +z
        [0, 1, 5, 4], // -y
        [2, 6, 7, 3], // +y
        [0, 4, 6, 2], // -x
        [1, 3, 7, 5], // +x
    ] {
        idx.extend([b + f[0], b + f[1], b + f[2], b + f[0], b + f[2], b + f[3]]);
    }
}

/// Points strictly inside the unit sphere, at least 0.2 from its surface
fn deep_inside() -> Vec<Vec3> {
    let mut p = Vec::new();
    for x in -6..=6 {
        for y in -6..=6 {
            for z in -6..=6 {
                let q = Vec3::new(x as f32, y as f32, z as f32) * 0.12;
                if q.length() < 0.8 {
                    p.extend([q]);
                }
            }
        }
    }
    p
}

fn winding_config() -> MeshToSdfConfig {
    let mut c = MeshToSdfConfig::accurate();
    c.sign_mode = MeshSignMode::WindingNumber;
    c
}

#[test]
fn the_octant_triangle_has_one_eighth_winding() {
    let t = [BvhTriangle::new(Vec3::X, Vec3::Y, Vec3::Z)];
    let w = winding_number(&t, Vec3::ZERO);
    assert!((w - 0.125).abs() < 1e-12, "{w}");
    // the same triangle wound the other way is −1/8
    let t = [BvhTriangle::new(Vec3::X, Vec3::Z, Vec3::Y)];
    let w = winding_number(&t, Vec3::ZERO);
    assert!((w + 0.125).abs() < 1e-12, "{w}");
}

#[test]
fn a_closed_sphere_has_winding_one_inside_and_zero_outside() {
    let (v, idx) = uv_sphere(18, 36, |_, _| true);
    let tris = triangles(&v, &idx);
    assert_eq!(winding_report(&idx).inconsistent_edges, 0);
    assert_eq!(winding_report(&idx).boundary_edges, 0);
    let inside = deep_inside();
    assert!(inside.len() > 1000);
    for p in &inside {
        let w = winding_number(&tris, *p);
        assert!((w - 1.0).abs() < 1e-9, "{p}: {w}");
    }
    for p in [
        Vec3::splat(2.0),
        Vec3::new(0.0, 1.5, 0.0),
        Vec3::new(-3.0, 0.1, 0.2),
    ] {
        let w = winding_number(&tris, p);
        assert!(w.abs() < 1e-9, "{p}: {w}");
    }
}

#[test]
fn a_sphere_with_a_hole_or_a_slit_keeps_its_inside() {
    let inside = deep_inside();
    let mut compared = 0;
    for (name, (v, idx)) in [
        ("polar hole", uv_sphere(18, 36, |i, _| i != 0)),
        (
            "slit",
            uv_sphere(18, 36, |i, j| !(j == 0 && i > 2 && i < 15)),
        ),
    ] {
        let report = winding_report(&idx);
        assert!(report.boundary_edges > 0, "{name}: the mesh must be open");
        assert_eq!(report.inconsistent_edges, 0, "{name}");
        let tris = triangles(&v, &idx);
        for p in &inside {
            let w = winding_number(&tris, *p);
            assert!(w > 0.75, "{name} {p}: {w}");
        }
        let sdf = MeshSdf::try_new(&v, &idx, &winding_config()).unwrap();
        for p in &inside {
            assert!(sdf.eval(*p) < 0.0, "{name} {p}: {}", sdf.eval(*p));
            compared += 1;
        }
        for p in [
            Vec3::splat(2.0),
            Vec3::new(0.0, 1.5, 0.0),
            Vec3::new(0.0, -1.5, 0.0),
        ] {
            assert!(sdf.eval(p) > 0.0, "{name} {p}");
        }
        // the flood fill leaks through the same opening (why this mode exists)
        let fill = MeshSdf::try_new(&v, &idx, &MeshToSdfConfig::accurate()).unwrap();
        assert!(
            inside.iter().all(|p| fill.eval(*p) > 0.0),
            "{name}: the flood fill no longer leaks"
        );
    }
    assert!(compared > 2000);
}

#[test]
fn touching_boxes_and_a_one_millimetre_gap_are_signed_by_geometry() {
    // two boxes that touch on x = 0, and two boxes 1 mm apart on x = 0 / 0.001
    for (name, gap) in [("touching", 0.0f32), ("1 mm gap", 0.001)] {
        let (mut v, mut idx) = (Vec::new(), Vec::new());
        add_box(
            &mut v,
            &mut idx,
            Vec3::new(-1.0, -0.5, -0.5),
            Vec3::new(0.0, 0.5, 0.5),
        );
        add_box(
            &mut v,
            &mut idx,
            Vec3::new(gap, -0.5, -0.5),
            Vec3::new(1.0 + gap, 0.5, 0.5),
        );
        assert_eq!(winding_report(&idx).inconsistent_edges, 0, "{name}");
        let tris = triangles(&v, &idx);
        for (p, want) in [
            (Vec3::new(-0.5, 0.0, 0.0), 1.0),
            (Vec3::new(0.5 + gap, 0.0, 0.0), 1.0),
            (Vec3::new(-0.0005, 0.1, 0.1), 1.0),
            (Vec3::new(0.0, 0.0, 2.0), 0.0),
            (Vec3::new(3.0, 3.0, 3.0), 0.0),
        ] {
            let w = winding_number(&tris, p);
            assert!((w - want).abs() < 1e-6, "{name} {p}: {w}");
        }
        if gap > 0.0 {
            let w = winding_number(&tris, Vec3::new(gap * 0.5, 0.0, 0.0));
            assert!(w.abs() < 1e-6, "{name}: the gap reads {w}");
        }
    }
}

/// Two closed boxes that interpenetrate (as a posed body self-intersects at creases and
/// modelled overlaps): the winding is 2 in the overlap and 1 in each box alone, the sign is
/// "inside" in both, and the flip check does not fire (the winding is consistent, the surfaces
/// only cross). At (0.4, 0, 0) — inside box A, outside box B, nearest to B's face — the face
/// normal sign reads "outside", so this case also tells the winding sign from that fallback.
#[test]
fn interpenetrating_solids_read_two_in_the_overlap_and_stay_inside() {
    let (mut v, mut idx) = (Vec::new(), Vec::new());
    add_box(&mut v, &mut idx, Vec3::splat(-1.0), Vec3::splat(1.0));
    add_box(
        &mut v,
        &mut idx,
        Vec3::new(0.5, -1.0, -1.0),
        Vec3::new(2.5, 1.0, 1.0),
    );
    let report = winding_report(&idx);
    assert_eq!(report.inconsistent_edges, 0, "{report:?}");
    let tris = triangles(&v, &idx);
    for (p, want) in [
        (Vec3::new(0.75, 0.0, 0.0), 2.0),
        (Vec3::new(0.4, 0.0, 0.0), 1.0),
        (Vec3::new(2.0, 0.0, 0.0), 1.0),
        (Vec3::new(-2.0, 0.0, 0.0), 0.0),
    ] {
        let w = winding_number(&tris, p);
        assert!((w - want).abs() < 1e-9, "{p}: {w}");
    }
    let sdf = MeshSdf::try_new(&v, &idx, &winding_config()).unwrap();
    for p in [
        Vec3::new(0.75, 0.0, 0.0),
        Vec3::new(0.4, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
    ] {
        assert!(sdf.eval(p) < 0.0, "{p}: {}", sdf.eval(p));
        assert!(sdf.eval_batch(&[p])[0] < 0.0, "{p} (batch)");
    }
    assert!(sdf.eval(Vec3::new(-2.0, 0.0, 0.0)) > 0.0);
    // the face-normal sign gets (0.4, 0, 0) wrong: the case has teeth against that fallback
    let mut face = MeshToSdfConfig::accurate();
    face.sign_mode = MeshSignMode::NearestFaceNormal;
    let face = MeshSdf::try_new(&v, &idx, &face).unwrap();
    assert!(face.eval(Vec3::new(0.4, 0.0, 0.0)) > 0.0);
}

/// A degenerate triangle (a repeated index) has no edges in the report: counting them would
/// make an edge it shares with a real triangle look non-manifold or inconsistently wound, and
/// `try_new` would refuse a valid mesh
#[test]
fn degenerate_triangles_do_not_add_edges_to_the_report() {
    let r = winding_report(&[0, 1, 2, 0, 1, 1]);
    assert_eq!(
        (
            r.degenerate_triangles,
            r.boundary_edges,
            r.inconsistent_edges,
            r.non_manifold_edges
        ),
        (1, 3, 0, 0),
        "{r:?}"
    );
    // a closed sphere with degenerate triangles on existing edges, in both directions
    let (v, mut idx) = uv_sphere(12, 24, |_, _| true);
    let (a, b) = (idx[3 * 40], idx[3 * 40 + 1]);
    idx.extend([a, b, b, b, a, a, a, a, a]);
    let r = winding_report(&idx);
    assert_eq!(
        (
            r.degenerate_triangles,
            r.boundary_edges,
            r.inconsistent_edges,
            r.non_manifold_edges
        ),
        (3, 0, 0, 0),
        "{r:?}"
    );
    MeshSdf::try_new(&v, &idx, &winding_config()).unwrap();
}

#[test]
fn a_flipped_triangle_is_detected_and_refused() {
    let (v, mut idx) = uv_sphere(12, 24, |_, _| true);
    assert_eq!(winding_report(&idx).inconsistent_edges, 0);
    MeshSdf::try_new(&v, &idx, &winding_config()).unwrap();
    // flip triangle 40: its 3 edges now run the same way as its neighbours'
    idx.swap(40 * 3 + 1, 40 * 3 + 2);
    let report = winding_report(&idx);
    assert_eq!(report.inconsistent_edges, 3, "{report:?}");
    let err = MeshSdf::try_new(&v, &idx, &winding_config())
        .map(|_| ())
        .unwrap_err();
    assert!(err.reason.contains("winding"), "{}", err.reason);
    // the other modes do not read the winding report
    MeshSdf::try_new(&v, &idx, &MeshToSdfConfig::accurate()).unwrap();
}

#[test]
fn degenerate_and_on_surface_queries_are_finite() {
    // a zero-area triangle contributes nothing; a point on a vertex or an edge is finite
    let t = [
        BvhTriangle::new(Vec3::ZERO, Vec3::ZERO, Vec3::X),
        BvhTriangle::new(Vec3::X, Vec3::Y, Vec3::Z),
    ];
    assert_eq!(winding_number(&t[..1], Vec3::new(0.3, 0.2, 0.1)), 0.0);
    for p in [Vec3::X, Vec3::new(0.5, 0.5, 0.0), Vec3::splat(1.0 / 3.0)] {
        assert!(winding_number(&t, p).is_finite(), "{p}");
    }
    // a point in a triangle's plane (on it, or off it in the plane) contributes exactly 0 (the
    // rule of `solid_angle`): without it the in-plane point inside the triangle would read ±½,
    // and a vertex query could read ½ from `atan2(0, -0)`
    let octant = [BvhTriangle::new(Vec3::X, Vec3::Y, Vec3::Z)];
    // exactly in the plane x + y + z = 1 (dyadic coordinates, so `det` is exactly 0): inside
    // the triangle, on a vertex, on an edge, and outside it
    for p in [
        Vec3::new(0.25, 0.25, 0.5),
        Vec3::X,
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::new(2.0, -0.5, -0.5),
    ] {
        assert_eq!(winding_number(&octant, p), 0.0, "{p}");
    }
    // a hair off the plane (1/3 is not representable) is not on it: the one-sided limit ±½
    let w = winding_number(&octant, Vec3::splat(1.0 / 3.0));
    assert!((w.abs() - 0.5).abs() < 1e-6, "{w}");
    assert_eq!(winding_number(&[], Vec3::ONE), 0.0);
}

/// The winding number is the law's bits on every target: a few values pinned (computed on
/// macOS arm64, checked by CI on Linux and Windows x86_64 and Linux arm64)
#[test]
fn winding_values_are_pinned_to_the_bit() {
    let (v, idx) = uv_sphere(18, 36, |i, _| i != 0);
    let tris = triangles(&v, &idx);
    let got: Vec<u64> = [
        Vec3::ZERO,
        Vec3::new(0.3, -0.2, 0.1),
        Vec3::new(0.0, 0.9, 0.0),
        Vec3::new(1.2, 0.4, -0.7),
    ]
    .iter()
    .map(|p| winding_number(&tris, *p).to_bits())
    .collect();
    assert_eq!(
        got,
        PINNED,
        "{:?}",
        got.iter().map(|b| format!("{b:#018x}")).collect::<Vec<_>>()
    );
}

const PINNED: [u64; 4] = [
    0x3fef_c214_b45a_f31c,
    0x3fef_d902_a786_415c,
    0x3fe7_0935_b8de_20b6,
    0xbf55_382f_d0fd_4e2c,
];
