//! Mesh -> SDF sign determination must not depend on triangle winding.
//!
//! The legacy sign rule (`MeshBvh::signed_distance` -> `Triangle::signed_distance`)
//! takes the sign from the *face normal* of the `|d|`-minimal triangle, so it is a
//! function of the input winding, not of the geometry. That breaks on the three
//! shapes that externally generated meshes actually produce:
//!
//! - an **open surface** has no interior at all, so every signed distance must be
//!   `+UDF`; the legacy rule reports negative on one side of the sheet,
//! - **flipping the winding** of a closed mesh leaves the solid unchanged, so the
//!   field must be invariant; the legacy rule negates it,
//! - **separately wound components** (each part closed, but wound independently)
//!   must each report their own interior as negative; the legacy rule inverts the
//!   component whose winding disagrees.
//!
//! Every expected value below comes from a closed form (exact box SDF / exact
//! distance to a rectangle), never from calling the implementation.
//!
//! `sdf_under_test` is the single switch that selects which sign rule the oracle
//! exercises. It is deliberately a one-liner so the same assertions can be run
//! against the legacy rule (red) and against the topology-independent rule
//! (green) without editing a single expected value.
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::{mesh_to_sdf_exact, MeshSdf, MeshSignMode, MeshToSdfConfig};
use alice_sdf::prelude::*;

// ---------------------------------------------------------------------------
// switch under test
// ---------------------------------------------------------------------------

/// Resolution (cells along the longest bbox axis) used by every flood-fill case
/// here. 64 resolves the 0.1-thick plate of `thin_plate_interior_is_negative`
/// (half-thickness 0.05 > half cell diagonal 0.027).
const FLOOD_RES: u32 = 64;

// The single switch. Point it at `legacy_sdf` to re-run every assertion below
// against the winding-dependent rule; no expected value changes either way.
fn sdf_under_test(vertices: &[Vec3], indices: &[u32]) -> MeshSdf {
    let mut cfg = MeshToSdfConfig::topology_robust();
    cfg.sign_flood_fill_resolution = FLOOD_RES;
    mesh_to_sdf_exact(vertices, indices, &cfg).expect("mesh is non-empty")
}

fn legacy_sdf(vertices: &[Vec3], indices: &[u32]) -> MeshSdf {
    let mut cfg = MeshToSdfConfig::accurate();
    cfg.sign_mode = MeshSignMode::NearestFaceNormal;
    mesh_to_sdf_exact(vertices, indices, &cfg).expect("mesh is non-empty")
}

// ---------------------------------------------------------------------------
// analytic oracles (closed form, no implementation calls)
// ---------------------------------------------------------------------------

/// Exact SDF of an axis-aligned box centred at the origin (Inigo Quilez).
/// The tessellated box mesh below *is* this box, so the match is exact, not
/// an approximation.
fn analytic_box(p: Vec3, half: Vec3) -> f32 {
    let q = p.abs() - half;
    q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
}

/// Exact unsigned distance from `p` to the rectangle `[-hx, hx] x [-hy, hy]`
/// lying in the plane `z = 0`. An open sheet has no interior, so this is also
/// the exact *signed* distance.
fn analytic_open_rect(p: Vec3, hx: f32, hy: f32) -> f32 {
    let c = Vec3::new(p.x.clamp(-hx, hx), p.y.clamp(-hy, hy), 0.0);
    (p - c).length()
}

// ---------------------------------------------------------------------------
// mesh builders (deterministic, outward CCW winding)
// ---------------------------------------------------------------------------

/// Closed axis-aligned box, 8 vertices / 12 triangles, every face wound CCW
/// when seen from outside.
fn box_mesh(center: Vec3, half: Vec3) -> (Vec<Vec3>, Vec<u32>) {
    let mut verts = Vec::with_capacity(8);
    for &sz in &[-1.0f32, 1.0] {
        for &sy in &[-1.0f32, 1.0] {
            for &sx in &[-1.0f32, 1.0] {
                verts.push(center + Vec3::new(sx * half.x, sy * half.y, sz * half.z));
            }
        }
    }
    // vertex index = x + 2*y + 4*z, with x/y/z in {0,1}
    let mut indices = Vec::with_capacity(36);
    let mut quad = |a: u32, b: u32, c: u32, d: u32| {
        indices.extend_from_slice(&[a, b, c, a, c, d]);
    };
    quad(0, 2, 3, 1); // -Z
    quad(4, 5, 7, 6); // +Z
    quad(0, 4, 6, 2); // -X
    quad(1, 3, 7, 5); // +X
    quad(0, 1, 5, 4); // -Y
    quad(2, 6, 7, 3); // +Y
    (verts, indices)
}

/// Open sheet: the rectangle `[-1,1] x [-1,1]` at `z = 0`, 2 triangles.
fn open_quad() -> (Vec<Vec3>, Vec<u32>) {
    let verts = vec![
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(-1.0, 1.0, 0.0),
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];
    (verts, indices)
}

/// Reverse the winding of every triangle. The solid is unchanged.
fn flip_winding(indices: &[u32]) -> Vec<u32> {
    indices
        .chunks_exact(3)
        .flat_map(|t| [t[0], t[2], t[1]])
        .collect()
}

fn append_mesh(dst_v: &mut Vec<Vec3>, dst_i: &mut Vec<u32>, src_v: &[Vec3], src_i: &[u32]) {
    let base = u32::try_from(dst_v.len()).expect("vertex count fits u32");
    dst_v.extend_from_slice(src_v);
    dst_i.extend(src_i.iter().map(|&i| i + base));
}

/// Deterministic lattice over `[-e, e]^3`, `n` samples per axis.
fn lattice(extent: f32, n: u32) -> Vec<Vec3> {
    let mut pts = Vec::with_capacity((n * n * n) as usize);
    let step = 2.0 * extent / f32::from(u16::try_from(n - 1).expect("n <= 65536"));
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                pts.push(Vec3::new(
                    (ix as f32).mul_add(step, -extent),
                    (iy as f32).mul_add(step, -extent),
                    (iz as f32).mul_add(step, -extent),
                ));
            }
        }
    }
    pts
}

/// Sign mismatch rate (%) and max absolute error against an analytic oracle,
/// counted only where the oracle is far enough from the surface that the sign
/// is unambiguous (`|expected| > band`).
struct Report {
    checked: usize,
    sign_mismatch: usize,
    max_err: f32,
    worst_at: Vec3,
}

impl Report {
    fn rate(&self) -> f32 {
        if self.checked == 0 {
            return 0.0;
        }
        100.0 * self.sign_mismatch as f32 / self.checked as f32
    }
}

fn measure(sdf: &MeshSdf, pts: &[Vec3], band: f32, oracle: impl Fn(Vec3) -> f32) -> Report {
    let mut r = Report {
        checked: 0,
        sign_mismatch: 0,
        max_err: 0.0,
        worst_at: Vec3::ZERO,
    };
    for &p in pts {
        let want = oracle(p);
        if want.abs() <= band {
            continue;
        }
        let got = sdf.eval(p);
        r.checked += 1;
        if (got < 0.0) != (want < 0.0) {
            r.sign_mismatch += 1;
        }
        let err = (got - want).abs();
        if err > r.max_err {
            r.max_err = err;
            r.worst_at = p;
        }
    }
    r
}

// ---------------------------------------------------------------------------
// oracle 1 — an open surface has no interior
// ---------------------------------------------------------------------------

#[test]
fn open_sheet_has_no_interior() {
    let (v, i) = open_quad();
    let sdf = sdf_under_test(&v, &i);
    let pts = lattice(1.5, 13);
    // band: sign is meaningless within one cell of the sheet
    let band = 2.0 * 3.0 / FLOOD_RES as f32;
    let r = measure(&sdf, &pts, band, |p| analytic_open_rect(p, 1.0, 1.0));

    eprintln!(
        "[open_sheet] checked {} sign-mismatch {} ({:.1}%) max_err {:.3e} at {:?}",
        r.checked,
        r.sign_mismatch,
        r.rate(),
        r.max_err,
        r.worst_at
    );
    assert_eq!(
        r.sign_mismatch, 0,
        "an open sheet encloses nothing, so every signed distance must be >= 0; \
         {} of {} samples came back negative",
        r.sign_mismatch, r.checked
    );
    assert!(
        r.max_err < 1e-5,
        "open sheet distance must equal the exact rectangle distance, max err {:.3e} at {:?}",
        r.max_err,
        r.worst_at
    );
}

// ---------------------------------------------------------------------------
// oracle 2 — flipping the winding leaves the solid, hence the field, unchanged
// ---------------------------------------------------------------------------

#[test]
fn winding_flip_leaves_field_invariant() {
    let half = Vec3::new(0.8, 0.6, 0.5);
    let (v, i) = box_mesh(Vec3::ZERO, half);
    let flipped = flip_winding(&i);

    let a = sdf_under_test(&v, &i);
    let b = sdf_under_test(&v, &flipped);

    let pts = lattice(1.4, 13);
    let mut differing = 0usize;
    let mut max_diff = 0.0f32;
    let mut worst = Vec3::ZERO;
    for &p in &pts {
        let (x, y) = (a.eval(p), b.eval(p));
        if x.to_bits() != y.to_bits() {
            differing += 1;
            let d = (x - y).abs();
            if d > max_diff {
                max_diff = d;
                worst = p;
            }
        }
    }
    eprintln!(
        "[winding_flip] differing {}/{} max_diff {:.3e} at {:?}",
        differing,
        pts.len(),
        max_diff,
        worst
    );
    assert_eq!(
        differing,
        0,
        "reversing every triangle leaves the same solid, so the field must be \
         bit-identical; {} of {} samples differ (max {:.3e} at {:?})",
        differing,
        pts.len(),
        max_diff,
        worst
    );
}

// ---------------------------------------------------------------------------
// oracle 3 — separate components wound independently
// ---------------------------------------------------------------------------

#[test]
fn separated_components_with_mixed_winding() {
    let half = Vec3::splat(0.5);
    let ca = Vec3::new(-2.0, 0.0, 0.0);
    let cb = Vec3::new(2.0, 0.0, 0.0);

    let (va, ia) = box_mesh(ca, half);
    let (vb, ib) = box_mesh(cb, half);
    let ib = flip_winding(&ib); // second component wound the other way

    let mut v = Vec::new();
    let mut i = Vec::new();
    append_mesh(&mut v, &mut i, &va, &ia);
    append_mesh(&mut v, &mut i, &vb, &ib);

    let sdf = sdf_under_test(&v, &i);
    let pts = lattice(3.0, 15);
    let band = 2.0 * 6.0 / FLOOD_RES as f32;
    let r = measure(&sdf, &pts, band, |p| {
        analytic_box(p - ca, half).min(analytic_box(p - cb, half))
    });

    eprintln!(
        "[mixed_winding] checked {} sign-mismatch {} ({:.1}%) max_err {:.3e} at {:?}",
        r.checked,
        r.sign_mismatch,
        r.rate(),
        r.max_err,
        r.worst_at
    );
    assert_eq!(
        r.sign_mismatch, 0,
        "each component is closed, so its interior is negative regardless of how \
         that component happens to be wound; {} of {} samples have the wrong sign",
        r.sign_mismatch, r.checked
    );
    assert!(
        r.max_err < 1e-5,
        "distance to a disjoint union is the min of the exact box distances, \
         max err {:.3e} at {:?}",
        r.max_err,
        r.worst_at
    );
}

// ---------------------------------------------------------------------------
// oracle 4 — the ordinary closed, consistently wound case must stay exact
// ---------------------------------------------------------------------------

#[test]
fn closed_box_matches_analytic() {
    let half = Vec3::new(0.7, 0.5, 0.9);
    let (v, i) = box_mesh(Vec3::ZERO, half);
    let sdf = sdf_under_test(&v, &i);

    let pts = lattice(1.6, 15);
    let band = 2.0 * 1.8 / FLOOD_RES as f32;
    let r = measure(&sdf, &pts, band, |p| analytic_box(p, half));

    eprintln!(
        "[closed_box] checked {} sign-mismatch {} ({:.1}%) max_err {:.3e} at {:?}",
        r.checked,
        r.sign_mismatch,
        r.rate(),
        r.max_err,
        r.worst_at
    );
    assert_eq!(r.sign_mismatch, 0, "closed box sign must match the box SDF");
    assert!(
        r.max_err < 1e-5,
        "closed box must match the exact box SDF, max err {:.3e} at {:?}",
        r.max_err,
        r.worst_at
    );
}

// ---------------------------------------------------------------------------
// oracle 5 — a thin plate must still have an interior at this resolution
// ---------------------------------------------------------------------------

#[test]
fn thin_plate_interior_is_negative() {
    let half = Vec3::new(1.0, 1.0, 0.05);
    let (v, i) = box_mesh(Vec3::ZERO, half);
    let sdf = sdf_under_test(&v, &i);

    // exact expectations: the deepest interior point is -half.z
    for (p, want) in [
        (Vec3::ZERO, -0.05_f32),
        (Vec3::new(0.5, 0.3, 0.0), -0.05),
        (Vec3::new(0.5, 0.3, 0.02), -0.03),
        (Vec3::new(0.0, 0.0, 0.1), 0.05),
        (Vec3::new(0.0, 0.0, -0.1), 0.05),
    ] {
        let got = sdf.eval(p);
        eprintln!("[thin_plate] {p:?} got {got:+.6} want {want:+.6}");
        assert!(
            (got - want).abs() < 1e-5,
            "thin plate at {p:?}: got {got:+.6}, analytic box SDF says {want:+.6}"
        );
    }
}

// ---------------------------------------------------------------------------
// the legacy rule is kept on purpose — pin what it actually does so the reason
// the second mode exists stays visible
// ---------------------------------------------------------------------------

#[test]
fn legacy_rule_is_winding_dependent() {
    let half = Vec3::new(0.8, 0.6, 0.5);
    let (v, i) = box_mesh(Vec3::ZERO, half);
    let flipped = flip_winding(&i);

    let a = legacy_sdf(&v, &i);
    let b = legacy_sdf(&v, &flipped);

    let p = Vec3::new(0.1, 0.2, 0.05); // interior
    let (x, y) = (a.eval(p), b.eval(p));
    eprintln!("[legacy] interior: as-wound {x:+.6}, flipped {y:+.6}");
    assert!(
        x < 0.0 && y > 0.0,
        "legacy sign comes from the face normal, so flipping the winding must \
         invert it (as-wound {x:+.6}, flipped {y:+.6}) — if this ever stops \
         holding, the legacy mode has silently changed meaning"
    );

    // and it reports a phantom interior for an open sheet
    let (qv, qi) = open_quad();
    let q = legacy_sdf(&qv, &qi);
    let below = q.eval(Vec3::new(0.0, 0.0, -0.5));
    eprintln!("[legacy] open sheet at z=-0.5: {below:+.6}");
    assert!(
        below < 0.0,
        "legacy rule puts a phantom interior on one side of an open sheet"
    );
}
