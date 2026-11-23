//! Kink cells end to end (sheet_field::kink, kink_build): a thin tapered strip (a hair tip) whose interpolated `d_b`
//! reads "beyond the rim" on its own surface near the tip, so rays onto the strip miss without the sidecar; with the
//! shared cover (exact `d_b` in the rim cells, exact `s` in the s cells) every ray hits within 0.05 of the strip.
//!
//! Oracle: the strip is the plane z = 0 inside its outline, so a ray straight down from z = 0.5 at a point at least
//! `INSET` inside the outline hits at t = 0.5 exactly. The node values come from the shared point evaluator
//! (`KinkSheet::from_geom` + `eval_all`), the same definition the builders use.
//! - Without the sidecar the defect must show (at least one miss): otherwise the scene does not exercise the cells.
//! - With it: 0 misses, CPU march (cell walk) and the fixed-step reference agree on every ray, the GPU (when a device
//!   is present; `ALICE_SDF_REQUIRE_GPU` makes its absence a failure) equals the CPU.

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::kink::{KinkError, KinkFile, KinkSheet, ASKC_VERSION};
use alice_sdf::sheet_field::kink_build::{
    kink_sheet, write_askc, write_askc_with_provenance, SheetGeom,
};
use alice_sdf::sheet_field::{MarchResult, RootMode, SheetField, ValidRule};
use glam::{DVec3, Vec3};

/// sin and cos through the crate's deterministic f32 functions (the same bits on every platform)
fn dsin(x: f64) -> f64 {
    f64::from(alice_det_math::sin_cos(x as f32).0)
}

fn dcos(x: f64) -> f64 {
    f64::from(alice_det_math::sin_cos(x as f32).1)
}

const H: f32 = 0.25;
const LEN: f64 = 3.0;
const W0: f64 = 0.24;
const INSET: f64 = 0.01;

/// The strip in z = 0: x from 0 to LEN, half width from W0/2 at x = 0 to 0 at the tip, in segments of 0.5.
fn strip() -> (Vec<DVec3>, Vec<[u32; 3]>, Vec<[u32; 2]>) {
    let n = 6usize;
    let mut v = Vec::new();
    for i in 0..=n {
        let x = LEN * i as f64 / n as f64;
        let hw = 0.5 * W0 * (1.0 - x / LEN);
        v.push(DVec3::new(x, -hw, 0.0));
        v.push(DVec3::new(x, hw, 0.0));
    }
    let mut f = Vec::new();
    for i in 0..n as u32 {
        let (a, b, c, d) = (2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3);
        f.push([a, c, b]);
        if i + 1 < n as u32 {
            f.push([b, c, d]);
        }
    }
    // the outline (every edge with one face); the last two tip vertices coincide in position but are separate indices
    let mut rim = Vec::new();
    for i in 0..n as u32 {
        rim.push([2 * i, 2 * i + 2]);
        if i + 1 < n as u32 {
            rim.push([2 * i + 1, 2 * i + 3]);
        }
    }
    rim.push([0, 1]);
    rim.push([2 * n as u32 - 1, 2 * n as u32]);
    (v, f, rim)
}

fn field(with_kink: bool) -> SheetField {
    let (v, f, rim) = strip();
    let geom = SheetGeom {
        verts: &v,
        faces: &f,
        rim: &rim,
        slab: 2.0 * f64::from(H),
        uv: None,
        nrm: None,
    };
    let ks = KinkSheet::from_geom(&geom);
    let tris: Vec<u32> = (0..f.len() as u32).collect();
    let segs: Vec<u32> = (0..rim.len() as u32).collect();
    let lv = level_from_fn(
        Vec3::new(-1.013, -1.021, -1.007),
        H,
        [3, 1, 1],
        |_c, _e| true,
        |p| {
            let (s, db, _) = ks.eval_all(p.as_dvec3(), &tris, &segs).expect("faces");
            vec![s as f32, db as f32]
        },
    )
    .unwrap();
    let mut sh = vec![sheet(vec![lv]).unwrap()];
    if with_kink {
        let (rec, st) = kink_sheet(&geom, &sh[0], 0.01);
        assert!(st.db_cells > 0 && st.surface > 0, "{st:?}");
        let mut bytes = Vec::new();
        write_askc(&mut bytes, &[(rec, geom.slab, 1)]).unwrap();
        KinkFile::read(&mut bytes.as_slice())
            .unwrap()
            .attach(&mut sh, 0)
            .unwrap();
    }
    let mut fld = SheetField::new(sh, ValidRule::CellV1);
    fld.set_root_mode(RootMode::CellWalk);
    fld
}

/// Points of the strip at least INSET inside its outline.
fn probes() -> Vec<Vec3> {
    let mut out = Vec::new();
    for i in 1..300 {
        let x = 0.01 * f64::from(i);
        let hw = 0.5 * W0 * (1.0 - x / LEN) - INSET;
        if hw <= 0.0 {
            continue;
        }
        for j in -4..=4 {
            out.push(Vec3::new(x as f32, (hw * f64::from(j) / 4.0) as f32, 0.5));
        }
    }
    out
}

const fn hit_t(m: &MarchResult) -> Option<f32> {
    match m {
        MarchResult::Hit(h) => Some(h.t),
        _ => None,
    }
}

#[test]
fn rays_onto_a_thin_tip_hit_with_the_kink_cells_and_miss_without() {
    let dir = Vec3::new(0.0, 0.0, -1.0);
    let ps = probes();
    assert!(ps.len() > 1000, "{}", ps.len());
    let base = field(false);
    let miss_base = ps
        .iter()
        .filter(|o| {
            hit_t(&base.march_decided(**o, dir, 1.0)).is_none_or(|t| (t - 0.5).abs() > 0.05)
        })
        .count();
    assert!(
        miss_base > 0,
        "the scene does not show the defect without the sidecar (0 of {} rays miss)",
        ps.len()
    );
    let fk = field(true);
    let mut miss = 0;
    for o in &ps {
        let m = hit_t(&fk.march_decided(*o, dir, 1.0));
        let r = hit_t(&fk.march_fixed_step(*o, dir, 1.0, 0.0005));
        if m.is_none_or(|t| (t - 0.5).abs() > 0.05) {
            miss += 1;
        }
        match (m, r) {
            (Some(a), Some(b)) => assert!((a - b).abs() <= 1e-3, "{o}: march {a} fixed {b}"),
            (a, b) => assert_eq!(a.is_some(), b.is_some(), "{o}: march {a:?} fixed {b:?}"),
        }
    }
    assert_eq!(
        miss,
        0,
        "with the sidecar {miss} of {} rays miss (without: {miss_base})",
        ps.len()
    );
    eprintln!(
        "kink e2e: {} rays, misses without the sidecar {miss_base}, with it 0",
        ps.len()
    );
    gpu_equals_cpu(&fk, &ps, dir);
}

#[cfg(feature = "gpu")]
fn gpu_equals_cpu(fk: &SheetField, ps: &[Vec3], dir: Vec3) {
    use alice_sdf::sheet_field::gpu::GpuSheetField;
    let g = match GpuSheetField::new_with_mode(fk, RootMode::CellWalk) {
        Ok(g) => g,
        Err(e) => {
            assert!(
                std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                "ALICE_SDF_REQUIRE_GPU set but no GPU: {e}"
            );
            eprintln!("gpu-test: skipped (no GPU: {e})");
            return;
        }
    };
    let gm = g.march_batch_decided(ps, dir, 1.0).unwrap();
    for (o, gr) in ps.iter().zip(&gm) {
        let c = hit_t(&fk.march_decided(*o, dir, 1.0));
        let gt = hit_t(gr);
        match (c, gt) {
            (Some(a), Some(b)) => assert!((a - b).abs() <= 1e-3, "{o}: cpu {a} gpu {b}"),
            (a, b) => assert_eq!(a.is_some(), b.is_some(), "{o}: cpu {a:?} gpu {b:?}"),
        }
    }
    eprintln!("gpu-test: {} rays, GPU = CPU", ps.len());
}

#[cfg(not(feature = "gpu"))]
const fn gpu_equals_cpu(_: &SheetField, _: &[Vec3], _: Vec3) {}

/// A sheet folded along the x axis: face A in z = 0 (y from 0 to 1), face B from the fold edge at dihedral `phi` to A
/// (the inside angle of the fold). Rays come down onto A at a shallow angle `beta` above A, heading towards the fold.
fn fold(phi: f64) -> (Vec<DVec3>, Vec<[u32; 3]>, Vec<[u32; 2]>) {
    let (c, s) = (dcos(phi), dsin(phi));
    let v = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(3.0, 0.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(3.0, 1.0, 0.0),
        DVec3::new(0.0, c, s),
        DVec3::new(3.0, c, s),
    ];
    // A: 0 1 2 / 1 3 2 (normal +z); B: the fold edge 0-1 and the far edge 4-5, normal pointing away from A's inside
    let f = vec![[0, 1, 2], [1, 3, 2], [0, 4, 1], [1, 4, 5]];
    let rim = vec![[0, 2], [2, 3], [3, 1], [0, 4], [4, 5], [5, 1]];
    (v, f, rim)
}

fn fold_field(phi: f64, with_kink: bool, s_cells: bool) -> SheetField {
    let (v, f, rim) = fold(phi);
    let geom = SheetGeom {
        verts: &v,
        faces: &f,
        rim: &rim,
        slab: 2.0 * f64::from(H),
        uv: None,
        nrm: None,
    };
    let ks = KinkSheet::from_geom(&geom);
    let tris: Vec<u32> = (0..f.len() as u32).collect();
    let segs: Vec<u32> = (0..rim.len() as u32).collect();
    let lv = level_from_fn(
        Vec3::new(-1.013, -1.021, -1.007),
        H,
        [3, 2, 2],
        |_c, _e| true,
        |p| {
            let (s, db, _) = ks.eval_all(p.as_dvec3(), &tris, &segs).expect("faces");
            vec![s as f32, db as f32]
        },
    )
    .unwrap();
    let mut sh = vec![sheet(vec![lv]).unwrap()];
    if with_kink {
        let (rec, _) = kink_sheet(&geom, &sh[0], if s_cells { 0.01 } else { 1e9 });
        let mut bytes = Vec::new();
        write_askc(&mut bytes, &[(rec, geom.slab, 1)]).unwrap();
        KinkFile::read(&mut bytes.as_slice())
            .unwrap()
            .attach(&mut sh, 0)
            .unwrap();
    }
    let mut fld = SheetField::new(sh, ValidRule::CellV1);
    fld.set_root_mode(RootMode::CellWalk);
    fld
}

/// The s cells' tooth: a sharp fold (inside angle 20 degrees), rays 5 degrees above face A heading to the fold. The
/// interpolated s cannot follow the fold, so the zero lies off the surface by more than the cover's eps (0.01) along
/// the normal; the s cells (the same sidecar with the s cover) put every hit within eps. Measured once over a sweep of
/// fold angles 20 / 35 / 60 / 90 and ray angles 1 / 2 / 5 degrees: with s cells 0 everywhere, without them up to 1652 of
/// 2301 (the sharp folds); this is one of the sharpest.
#[test]
fn rays_near_a_sharp_fold_stay_within_eps_with_the_s_cells() {
    let (phi, b) = (20f64.to_radians(), 5f64.to_radians());
    let dir = Vec3::new(0.0, -dcos(b) as f32, -dsin(b) as f32);
    let (no_s, with_s) = (fold_field(phi, true, false), fold_field(phi, true, true));
    let (mut m0, mut m1, mut n) = (0, 0, 0);
    for i in 1..60 {
        for j in 1..40 {
            let q = DVec3::new(0.05 * f64::from(i), 0.05 + 0.015 * f64::from(j), 0.0);
            let o = (q - dir.as_dvec3() * 0.5).as_vec3();
            let off = |m: &MarchResult| {
                hit_t(m).map_or(f32::INFINITY, |t| (t - 0.5).abs() * dsin(b) as f32)
            };
            n += 1;
            if off(&no_s.march_decided(o, dir, 1.0)) > 0.01 {
                m0 += 1;
            }
            let m = with_s.march_decided(o, dir, 1.0);
            if off(&m) > 0.01 {
                m1 += 1;
            }
            let r = with_s.march_fixed_step(o, dir, 1.0, 0.0005);
            match (hit_t(&m), hit_t(&r)) {
                (Some(a), Some(c)) => assert!((a - c).abs() <= 1e-3, "{o}: march {a} fixed {c}"),
                (a, c) => assert_eq!(a.is_some(), c.is_some(), "{o}: march {a:?} fixed {c:?}"),
            }
        }
    }
    assert!(
        m0 > 0,
        "the fold does not show the defect without the s cells (0 of {n})"
    );
    assert_eq!(
        m1, 0,
        "with the s cells {m1} of {n} hits lie beyond eps (without: {m0})"
    );
    eprintln!("fold e2e: {n} rays, beyond eps without s cells {m0}, with 0");
}

/// A square sheet in the plane `z = z0` (normal +z), or with `vertical` the plane `y = y0` (normal +y), on one level whose
/// grid starts at `lo`; channels from the exact distance (no kink cells: the cell walk alone is under test).
#[cfg(feature = "gpu")]
fn plane_field(lo: Vec3, nb: [u32; 3], vertical: bool, at: f32) -> SheetField {
    let lv = level_from_fn(
        lo,
        H,
        nb,
        |_c, _e| true,
        |p| {
            let (s, u, v) = if vertical {
                (p.y - at, p.x, p.z)
            } else {
                (p.z - at, p.x, p.y)
            };
            let (cu, cv) = if vertical {
                (
                    lo.x + 8.0 * H * nb[0] as f32 / 2.0,
                    lo.z + 8.0 * H * nb[2] as f32 / 2.0,
                )
            } else {
                (
                    lo.x + 8.0 * H * nb[0] as f32 / 2.0,
                    lo.y + 8.0 * H * nb[1] as f32 / 2.0,
                )
            };
            let half = 1.5f32;
            let (du, dv) = ((u - cu).abs() - half, (v - cv).abs() - half);
            vec![
                s,
                Vec3::new(du.max(0.0), dv.max(0.0), 0.0).length() + du.max(dv).min(0.0),
            ]
        },
    )
    .unwrap();
    let mut f = SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1);
    f.set_root_mode(RootMode::CellWalk);
    f
}

/// Two ways the GPU cell walk once spun in one cell to its cap (the ray then came back undecided while the CPU, in f64,
/// decided it): (a) a ray along a cell face plane whose direction carries a 1e-16 residue in that axis (cos 90 degrees),
/// (b) a field far from the origin of the coordinates (z near 1360 mm, an f32 ulp of 1.2e-4 mm) with rays starting a
/// few mm away (t small), where a hair of 4 ulp of t past a face does not move the f32 point. Every ray must be decided
/// and equal the CPU.
#[cfg(feature = "gpu")]
#[test]
fn the_gpu_cell_walk_decides_rays_on_face_planes_and_far_from_the_origin() {
    use alice_sdf::sheet_field::gpu::GpuSheetField;
    let cases: Vec<(&str, SheetField, Vec<Vec3>, Vec3)> = vec![
        {
            // (a) vertical sheet y = 1.0; rays head -y with a 1e-16 x residue, origins exactly on x cell faces
            let lo = Vec3::new(-2.0, -1.0, -2.0);
            let f = plane_field(lo, [2, 2, 2], true, 1.0);
            let o: Vec<Vec3> = (0..16)
                .flat_map(|i| {
                    (0..16).map(move |j| {
                        Vec3::new(lo.x + H * (i + 2) as f32, 2.9, lo.z + 0.5 + 0.17 * j as f32)
                    })
                })
                .collect();
            ("face plane", f, o, Vec3::new(-1.224_646_8e-16, -1.0, 0.0))
        },
        {
            // (b) horizontal sheet at z = 1360.0, rays from 3 mm above at 30 degrees below the horizontal
            let lo = Vec3::new(-2.013, -2.021, 1358.0);
            let f = plane_field(lo, [2, 2, 2], false, 1360.0);
            let d = Vec3::new(0.0, 0.866_025_4, -0.5);
            let o: Vec<Vec3> = (0..16)
                .flat_map(|i| {
                    (0..16).map(move |j| {
                        Vec3::new(-1.0 + 0.13 * i as f32, -2.0 + 0.09 * j as f32, 1361.5)
                    })
                })
                .collect();
            ("far from the origin", f, o, d)
        },
        {
            // (c) horizontal sheet at z = 468.3 seen from 470 mm away (t ~ 470, an ulp of t ~ 3e-5 mm against a 0.01
            // mm floor step): over 4096 rays at irregular offsets some floor steps end an ulp after the surface, where
            // the f32 walk places the root past the step's end; the step's end signs must still decide the crossing
            let lo = Vec3::new(-2.0, -2.0, 466.3);
            let f = plane_field(lo, [2, 2, 2], false, 468.3);
            let d = Vec3::new(0.0, 0.866_025_4, -0.5);
            let o: Vec<Vec3> = (0..64)
                .flat_map(|i| {
                    (0..64).map(move |j| {
                        Vec3::new(
                            -1.2 + 0.0371 * i as f32,
                            -1.2 + 0.0293 * j as f32 + 0.000_37 * i as f32,
                            468.3,
                        ) - d * 470.0
                    })
                })
                .collect();
            ("span end an ulp past the surface", f, o, d)
        },
    ];
    for (name, f, o, d) in cases {
        let g = match GpuSheetField::new_with_mode(&f, RootMode::CellWalk) {
            Ok(g) => g,
            Err(e) => {
                assert!(
                    std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                    "ALICE_SDF_REQUIRE_GPU set but no GPU: {e}"
                );
                eprintln!("gpu-test: skipped (no GPU: {e})");
                return;
            }
        };
        let tmax = if name.starts_with("span end") {
            480.0
        } else {
            6.0
        };
        let gm = g.march_batch_decided(&o, d, tmax).unwrap();
        let undecided = gm
            .iter()
            .filter(|m| matches!(m, MarchResult::Undecided))
            .count();
        assert_eq!(
            undecided,
            0,
            "{name}: {undecided} of {} rays undecided on the GPU",
            o.len()
        );
        let mut hits = 0;
        for (oo, gr) in o.iter().zip(&gm) {
            let c = f.march_decided(*oo, d, tmax);
            match (hit_t(&c), hit_t(gr)) {
                (Some(a), Some(b)) => {
                    hits += 1;
                    assert!((a - b).abs() <= 1e-3, "{name} {oo}: cpu {a} gpu {b}");
                }
                (a, b) => assert_eq!(a.is_some(), b.is_some(), "{name} {oo}: cpu {a:?} gpu {b:?}"),
            }
        }
        assert!(
            hits > 0,
            "{name}: no ray hit (the scene does not reach the walk)"
        );
        eprintln!(
            "gpu-test {name}: {} rays, {hits} hits, 0 undecided, GPU = CPU",
            o.len()
        );
    }
}

/// A flat strip (z = 0, x from 0 to 3, y from -0.3 to 0.3) cut at x = 1.5 into two halves whose UVs disagree there (a
/// seam: u = x / 3 on the left, x / 3 + 10 on the right, v = y): the stored UV channels interpolate across the seam.
fn seam_field(with_kink: bool) -> SheetField {
    let v = vec![
        DVec3::new(0.0, -0.3, 0.0),
        DVec3::new(1.5, -0.3, 0.0),
        DVec3::new(3.0, -0.3, 0.0),
        DVec3::new(0.0, 0.3, 0.0),
        DVec3::new(1.5, 0.3, 0.0),
        DVec3::new(3.0, 0.3, 0.0),
    ];
    let f = vec![[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]];
    let rim = vec![[0, 1], [1, 2], [2, 5], [5, 4], [4, 3], [3, 0]];
    let uv_of = |vi: u32, right: bool| {
        let p = v[vi as usize];
        [p.x / 3.0 + if right { 10.0 } else { 0.0 }, p.y]
    };
    let uv: Vec<[[f64; 2]; 3]> = f
        .iter()
        .enumerate()
        .map(|(i, t)| t.map(|c| uv_of(c, i >= 2)))
        .collect();
    let geom = SheetGeom {
        verts: &v,
        faces: &f,
        rim: &rim,
        slab: 2.0 * f64::from(H),
        uv: Some(&uv),
        nrm: None,
    };
    let ks = KinkSheet::from_geom(&geom);
    let tris: Vec<u32> = (0..f.len() as u32).collect();
    let segs: Vec<u32> = (0..rim.len() as u32).collect();
    let lv = level_from_fn(
        Vec3::new(-1.013, -1.021, -1.007),
        H,
        [3, 1, 1],
        |_c, _e| true,
        |p| {
            let (s, db, _) = ks.eval_all(p.as_dvec3(), &tris, &segs).expect("faces");
            let (uvp, _) = ks.eval_attributes(p.as_dvec3(), &tris).expect("faces");
            let [u, w] = uvp.expect("uv");
            vec![s as f32, db as f32, u, w]
        },
    )
    .unwrap();
    let mut sh = vec![sheet(vec![lv]).unwrap()];
    if with_kink {
        let (rec, st) = kink_sheet(&geom, &sh[0], 0.01);
        assert!(st.attr_cells > 0, "{st:?}");
        let mut bytes = Vec::new();
        write_askc(&mut bytes, &[(rec, geom.slab, 1)]).unwrap();
        KinkFile::read(&mut bytes.as_slice())
            .unwrap()
            .attach(&mut sh, 0)
            .unwrap();
    }
    let mut fld = SheetField::new(sh, ValidRule::CellV1);
    fld.set_root_mode(RootMode::CellWalk);
    fld
}

/// The attribute cells' tooth: rays straight down near the seam. The hit's UV must be the analytic UV of the half it
/// lies on (to 1e-3); the interpolated channels blend the two halves near the seam, the attribute cells do not.
#[test]
fn uv_at_a_seam_is_the_hit_face_s_with_the_attribute_cells() {
    let dir = Vec3::new(0.0, 0.0, -1.0);
    let (plain, kinked) = (seam_field(false), seam_field(true));
    let (mut bad0, mut bad1, mut n) = (0, 0, 0);
    for i in 0..60 {
        for j in -3..=3 {
            let x = 1.2 + 0.01 * f64::from(i) + 0.000_37;
            let o = Vec3::new(x as f32, (0.08 * f64::from(j)) as f32, 0.5);
            let truth = x / 3.0 + if x > 1.5 { 10.0 } else { 0.0 };
            n += 1;
            for (fld, bad) in [(&plain, &mut bad0), (&kinked, &mut bad1)] {
                match fld.march_decided(o, dir, 1.0) {
                    MarchResult::Hit(h) => {
                        let u = f64::from(h.uv.expect("uv")[0]);
                        if (u - truth).abs() > 1e-3 {
                            *bad += 1;
                        }
                    }
                    _ => *bad += 1,
                }
            }
        }
    }
    assert!(
        bad0 > 0,
        "the interpolated UV does not show the seam (0 of {n})"
    );
    assert_eq!(
        bad1, 0,
        "with the attribute cells {bad1} of {n} hits read a wrong UV (without: {bad0})"
    );
    eprintln!("seam e2e: {n} rays, wrong UV without the attribute cells {bad0}, with 0");
}

/// The step margin's tooth: a fold of 60 degrees with rays climbing into it at 15 degrees from 400 mm away (t large, as
/// in a camera view). Without the margin a full step taken from an interpolated value can pass a crossing the field
/// shows (2 of 1560 rays in this scene, measured once); with it the march equals the fixed-step reference on every ray.
#[test]
fn rays_from_far_into_a_fold_equal_the_fixed_step_reference() {
    let back = 400.0f32;
    let b = 15f64.to_radians();
    let dir = Vec3::new(0.0, -dcos(b) as f32, dsin(b) as f32);
    let f = fold_field(60f64.to_radians(), true, true);
    let mut bad = 0;
    let mut n = 0;
    for i in 1..40 {
        for j in 0..40 {
            let o = Vec3::new(0.075 * i as f32, 1.2, -0.3 + 0.015 * j as f32) - dir * back;
            let m = hit_t(&f.march_decided(o, dir, back + 3.0));
            let r = hit_t(&f.march_fixed_step(o + dir * (back - 0.5), dir, 3.5, 0.0005))
                .map(|t| t + back - 0.5);
            n += 1;
            let same = match (m, r) {
                (Some(x), Some(y)) => (x - y).abs() <= 2e-3,
                (None, None) => true,
                _ => false,
            };
            if !same {
                bad += 1;
            }
        }
    }
    assert_eq!(
        bad, 0,
        "the march differs from the fixed-step reference on {bad} of {n} rays"
    );
}

/// The sidecar carries this crate's semantics identifier after the version: read back it is accepted; written under
/// another identifier, or relabelled as a version without one (1 to 3), or as an unknown version, it is refused
#[test]
fn a_sidecar_carries_the_semantics_identifier_and_another_is_refused() {
    let records = [(None, 0.01, 0)];
    let mut own = Vec::new();
    write_askc(&mut own, &records).unwrap();
    assert!(KinkFile::read(&mut own.as_slice()).is_ok());
    let other = alice_sdf::provenance::Provenance::new([7u8; 32]);
    let mut foreign = Vec::new();
    write_askc_with_provenance(&mut foreign, &records, &other).unwrap();
    assert_eq!(
        foreign[8..40],
        [7u8; 32],
        "the identifier sits after the version"
    );
    assert_eq!(foreign[40..], own[40..], "only the identifier differs");
    match KinkFile::read(&mut foreign.as_slice()) {
        Err(KinkError::Semantics(e)) => assert!(e.to_string().contains(&other.hex()), "{e}"),
        other => panic!("another identifier was not refused: {other:?}"),
    }
    for v in [1u32, 3, ASKC_VERSION + 1] {
        let mut bad = own.clone();
        bad[4..8].copy_from_slice(&v.to_le_bytes());
        assert!(
            matches!(
                KinkFile::read(&mut bad.as_slice()),
                Err(KinkError::Format(_))
            ),
            "version {v} was not refused"
        );
    }
}
