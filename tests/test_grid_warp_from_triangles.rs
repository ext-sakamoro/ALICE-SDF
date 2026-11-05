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
//! - bad input is an error, never a grid;
//! - with a shell bound (the child lies within t of the rest sheet), a folded sheet shows no
//!   surface farther than σ·t from the posed sheet (without it, the fold makes such ghosts),
//!   and the per-node floor stays at or below the true distance to the posed shape, on a
//!   stretched and on a folded sheet, near and far (true distance: dense samples of the
//!   child's rest surface carried forward through each sample's own triangle).
//!
//! Every evaluation goes through `SdfNode::grid_warp` and `eval`, the production path.
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::transforms::{
    affine_posed_to_rest, grid_warp_from_triangles, GridWarp, GridWarpBuildError, WarpGrid,
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
    let data = grid_warp_from_triangles(
        rest,
        posed,
        tris,
        &WarpGrid {
            box_min: lo,
            box_max: hi,
            cell: 0.05,
            reach: 0.5,
            shell: None,
        },
    )
    .expect("builds");
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
    let data = grid_warp_from_triangles(
        &rest,
        &rest,
        &tris,
        &WarpGrid {
            box_min: lo,
            box_max: hi,
            cell: 0.05,
            reach: 0.5,
            shell: None,
        },
    )
    .expect("builds");
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
        grid_warp_from_triangles(
            r,
            p,
            t,
            &WarpGrid {
                box_min: lo,
                box_max: hi,
                cell,
                reach: 0.5,
                shell: None,
            },
        )
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
        &WarpGrid {
            box_min: Vec3::new(-0.6, -0.6, -0.8),
            box_max: Vec3::new(5.6, 1.6, 0.8),
            cell: 0.05,
            reach: 0.5,
            shell: None,
        },
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

/// A sheet folded over itself: the right half is turned back over the left and lifted 0.15
/// (posed x = 1 − x, z = 0.15 for rest x > 0.5).
fn folded(rest: &[Vec3]) -> Vec<Vec3> {
    rest.iter()
        .map(|p| {
            if p.x > 0.5 {
                Vec3::new(1.0 - p.x, p.y, 0.15)
            } else {
                *p
            }
        })
        .collect()
}

/// A thin slab on the rest sheet: within 0.02 of it (the shell bound used below is 0.03).
fn slab() -> SdfNode {
    SdfNode::box3d(0.8, 0.8, 0.04).translate(0.5, 0.5, 0.0)
}

fn sheet_distance(p: Vec3, posed: &[Vec3], tris: &[[u32; 3]]) -> f32 {
    // dense barycentric samples of every posed triangle (classification only)
    let mut d = f32::INFINITY;
    for t in tris {
        let (a, b, c) = (
            posed[t[0] as usize],
            posed[t[1] as usize],
            posed[t[2] as usize],
        );
        for i in 0..=16 {
            for j in 0..=(16 - i) {
                let (u, v) = (i as f32 / 16.0, j as f32 / 16.0);
                d = d.min((a + (b - a) * u + (c - a) * v - p).length());
            }
        }
    }
    d
}

#[test]
fn a_folded_sheet_has_no_ghost_with_a_shell() {
    let (rest, tris) = sheet(9);
    let posed = folded(&rest);
    let (lo, hi) = (Vec3::new(-0.4, -0.4, -0.4), Vec3::new(1.4, 1.4, 0.6));
    let cell = 0.05;
    let shell = 0.03;
    let band = shell + cell * 3.0f32.sqrt();
    let ghosts = |data| {
        let node = slab().grid_warp(GridWarp::new(data).expect("valid"));
        let mut n = 0;
        for k in 0..24 {
            for j in 0..24 {
                for i in 0..24 {
                    let p = lo
                        + (hi - lo) * Vec3::new(i as f32 + 0.5, j as f32 + 0.5, k as f32 + 0.5)
                            / 24.0;
                    // inside, yet farther from the posed sheet than the floor guarantees to
                    // clear: σ·t plus the cell diagonal (σ = 1 here: each half moves rigidly);
                    // nearer than that the floor is resolution-limited by construction
                    if eval(&node, p) < 0.0 && sheet_distance(p, &posed, &tris) > band {
                        n += 1;
                    }
                }
            }
        }
        n
    };
    let without = grid_warp_from_triangles(
        &rest,
        &posed,
        &tris,
        &WarpGrid {
            box_min: lo,
            box_max: hi,
            cell,
            reach: 0.3,
            shell: None,
        },
    )
    .expect("builds");
    let with = grid_warp_from_triangles(
        &rest,
        &posed,
        &tris,
        &WarpGrid {
            box_min: lo,
            box_max: hi,
            cell,
            reach: 0.3,
            shell: Some(shell),
        },
    )
    .expect("builds");
    // the scene folds: without the shell the fold makes ghost surfaces
    assert!(
        ghosts(without) > 0,
        "the folded scene shows no ghost without a shell"
    );
    assert_eq!(ghosts(with), 0, "ghost surfaces remain with the shell");
}

#[test]
fn the_floor_never_exceeds_the_distance_to_the_posed_shape() {
    // The child overhangs the sheet's edge along x by 0.02 and is 0.02 thick on each side, so
    // every point of it is within sqrt(0.02² + 0.02²) < t = 0.03 of the rest sheet, and the
    // overhang is where the stretch factor σ matters (an in-plane offset beyond the edge grows
    // by σ). Scenes: the sheet stretched ×4 along x (σ = 4; a cell of 0.02 makes the dropped
    // σ term, 0.09, larger than the cell diagonal 0.035) and the sheet folded over itself.
    let (rest, tris) = sheet(9);
    let shell = 0.03;
    let stretched: Vec<Vec3> = rest
        .iter()
        .map(|p| Vec3::new(4.0 * p.x, p.y, p.z))
        .collect();
    let mut checked = 0;
    for (posed, cell) in [(stretched, 0.02f32), (folded(&rest), 0.05)] {
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for p in &posed {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let (lo, hi) = (lo - Vec3::splat(0.4), hi + Vec3::splat(0.4));
        let data = grid_warp_from_triangles(
            &rest,
            &posed,
            &tris,
            &WarpGrid {
                box_min: lo,
                box_max: hi,
                cell,
                reach: 0.3,
                shell: Some(shell),
            },
        )
        .expect("builds");
        let warp = GridWarp::new(data).expect("valid");
        // the posed shape: surface samples of the overhanging slab carried forward through the
        // rest triangle under each sample (its own map, rest -> posed)
        let forward = |x: Vec3| {
            let (ci, cj) = (
                (x.x * 8.0).clamp(0.0, 7.0) as usize,
                (x.y * 8.0).clamp(0.0, 7.0) as usize,
            );
            let fx = x.x * 8.0 - ci as f32;
            let fy = x.y * 8.0 - cj as f32;
            let t = tris[(cj * 8 + ci) * 2 + usize::from(fy > fx)];
            let r = [
                rest[t[0] as usize],
                rest[t[1] as usize],
                rest[t[2] as usize],
            ];
            let q = [
                posed[t[0] as usize],
                posed[t[1] as usize],
                posed[t[2] as usize],
            ];
            let m = affine_posed_to_rest(q, r).expect("map").map(f64::from);
            let (a, b, c) = (f64::from(x.x), f64::from(x.y), f64::from(x.z));
            Vec3::new(
                (m[0] * a + m[1] * b + m[2] * c + m[3]) as f32,
                (m[4] * a + m[5] * b + m[6] * c + m[7]) as f32,
                (m[8] * a + m[9] * b + m[10] * c + m[11]) as f32,
            )
        };
        let mut shape = Vec::new();
        let (nx, ny, nz) = (260, 200, 8);
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    if !(k == 0 || k == nz || i == 0 || i == nx || j == 0 || j == ny) {
                        continue;
                    }
                    let x = Vec3::new(
                        -0.02 + 1.04 * i as f32 / nx as f32,
                        0.1 + 0.8 * j as f32 / ny as f32,
                        -0.02 + 0.04 * k as f32 / nz as f32,
                    );
                    shape.push(forward(x));
                }
            }
        }
        // largest gap between neighbouring posed samples, halved: the sampled distance exceeds
        // the true one by at most this (x spacing 0.004 × 4 stretch, y 0.004, z 0.005)
        let gap = 0.5 * (0.016f32 * 0.016 + 0.004 * 0.004 + 0.005 * 0.005).sqrt();
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 40) as f32 / (1u64 << 24) as f32
        };
        // the posed sheet's boundary and crease: where the distance to the sheet kinks, the
        // trilinear blend of corner floors is not exact and the cell-diagonal term is what keeps
        // it below the distance (away from them the distance is nearly linear)
        let edge: Vec<Vec3> = posed
            .iter()
            .zip(&rest)
            .filter(|(_, r)| {
                r.x < 0.01 || r.x > 0.99 || r.y < 0.01 || r.y > 0.99 || (r.x - 0.5).abs() < 0.07
            })
            .map(|(p, _)| *p)
            .collect();
        for n in 0..6000 {
            // near the boundary / crease, near the shape, and anywhere in the box
            let p = if n % 3 == 0 {
                let e = edge[(next() * (edge.len() - 1) as f32) as usize];
                e + Vec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 0.12
            } else if n % 3 == 1 {
                lo + (hi - lo) * Vec3::new(next(), next(), next())
            } else {
                let y = shape[(next() * (shape.len() - 1) as f32) as usize];
                y + Vec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 0.2
            };
            let (_, _, _, floor) = warp.warp(p);
            let d = shape
                .iter()
                .map(|y| (p - *y).length())
                .fold(f32::INFINITY, f32::min);
            assert!(
                floor <= d + gap,
                "floor {floor} above the distance {d} at {p:?} (cell {cell})"
            );
            checked += 1;
        }
    }
    assert!(checked > 0);
}

#[test]
fn the_cell_diagonal_keeps_the_floor_below_the_distance_for_a_thin_shell() {
    // A thin child (0.004 thick on each side of a flat sheet, shell t = 0.005) on a coarse grid
    // (cell 0.08) whose nodes straddle the sheet's plane at z = ±0.04: between them the
    // distance to the sheet is |z|, whose trilinear blend overshoots by up to 0.04 > t. Only
    // the cell-diagonal term keeps the blended floor below the distance there.
    let (rest, tris) = sheet(9);
    let shell = 0.005;
    let child_half = 0.004;
    let (lo, hi) = (Vec3::new(-0.4, -0.4, -0.36), Vec3::new(1.4, 1.4, 0.36));
    let data = grid_warp_from_triangles(
        &rest,
        &rest,
        &tris,
        &WarpGrid {
            box_min: lo,
            box_max: hi,
            cell: 0.08,
            reach: 0.3,
            shell: Some(shell),
        },
    )
    .expect("builds");
    // the grid's z nodes: -0.36 + 0.08 k, so ±0.04 are nodes
    assert!(data.dims[2] >= 10);
    let warp = GridWarp::new(data).expect("valid");
    let mut checked = 0;
    for j in 0..40 {
        for i in 0..40 {
            for k in 0..20 {
                let p = Vec3::new(
                    0.12 + 0.76 * i as f32 / 39.0,
                    0.12 + 0.76 * j as f32 / 39.0,
                    child_half + 0.035 * k as f32 / 19.0,
                );
                // the posed shape is the slab itself (posed = rest): the distance from above is
                // the height over its top face
                let d = p.z - child_half;
                let (_, _, _, floor) = warp.warp(p);
                assert!(
                    floor <= d + 1e-5,
                    "floor {floor} above the distance {d} at {p:?}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
}
