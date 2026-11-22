//! Attribute channels stored as a band: per brick, only the nodes that are corners of a cell the surface may
//! cross (the 8 corner `s` values straddle 0 within `eps_steps` quantization steps) keep their `u`, `v` and
//! author-normal bytes; `s` and `d_b` stay whole. A hit interpolates attributes inside the cell it lies in, so a
//! band that holds those cells returns the same attributes, and a read outside the band is refused, not shaded.
//!
//! Oracles (none is the sampler's own output):
//! - at every hit the band gives the attributes of the full field, bit for bit, on the CPU and the GPU (the same
//!   bytes are read; only the address differs);
//! - the band stores fewer attribute bytes than the full field, and the geometry is the full field's;
//! - ASHF: a banded field is written as v3 and read back with the same bytes, masks and row starts; a field
//!   without a band is still v1 (or v2 with a coarse channel), byte for byte;
//! - an empty band (every cell excluded) makes the GPU march refuse the hits and the CPU return NaN attributes;
//! - refusals: a band of a level without attributes or of a level already banded, a mask or byte count that
//!   does not match, and the band kind in a v2 file.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use alice_sdf::sheet_field::build::{band, coarsen, level_from_fn, sheet};
use alice_sdf::sheet_field::stream::{write_ashf, AshfIndex};
use alice_sdf::sheet_field::{MarchResult, SheetField, ValidRule, BAND};
use glam::Vec3;
use std::io::Cursor;

fn square_db(x: f32, y: f32) -> f32 {
    let dx = x.abs() - 1.0;
    let dy = y.abs() - 1.0;
    Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0)
}

/// a gently curved patch z = 0.05 (x^2 - y) with attributes varying across it
fn patch() -> SheetField {
    let lo = Vec3::new(-1.5 + 0.013, -1.5 + 0.021, -0.5 + 0.007);
    let lv = level_from_fn(
        lo,
        0.0625,
        [6, 6, 2],
        |_c, _e| true,
        |p| {
            vec![
                p.z - 0.05 * (p.x * p.x - p.y),
                square_db(p.x, p.y),
                0.5 + 0.01 * p.x + 0.003 * p.y * p.y,
                0.25 + 0.02 * p.y,
                0.2 * p.x,
                -0.1 * p.y + 0.02 * p.x * p.y,
                1.0,
            ]
        },
    )
    .unwrap();
    SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1)
}

fn banded(f: &SheetField, eps_steps: f32) -> SheetField {
    let sheets = f
        .sheets
        .iter()
        .map(|s| {
            sheet(
                s.levels
                    .iter()
                    .map(|lv| band(lv, eps_steps).unwrap())
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    SheetField::new(sheets, f.rule)
}

fn origins() -> Vec<Vec3> {
    let mut v = Vec::new();
    for i in 0..29 {
        for j in 0..23 {
            v.push(Vec3::new(
                -0.9 + 0.063 * i as f32,
                -0.9 + 0.079 * j as f32,
                0.45,
            ));
        }
    }
    v
}

const fn bits(m: &MarchResult) -> Option<(u32, usize, [u32; 2], [u32; 3])> {
    match m {
        MarchResult::Hit(h) => {
            let uv = h.uv.unwrap();
            let n = h.attr_normal.unwrap();
            Some((
                h.t.to_bits(),
                h.sheet,
                [uv[0].to_bits(), uv[1].to_bits()],
                [n.x.to_bits(), n.y.to_bits(), n.z.to_bits()],
            ))
        }
        _ => None,
    }
}

#[test]
fn at_every_hit_the_band_gives_the_full_fields_attributes_bit_for_bit() {
    let full = patch();
    let b = banded(&full, 2.0);
    let (mut hits, mut full_bytes, mut band_bytes) = (0, 0usize, 0usize);
    for (a, c) in full.sheets[0].levels.iter().zip(&b.sheets[0].levels) {
        for (ca, cb) in a.channels.iter().zip(&c.channels).skip(2) {
            assert_eq!(cb.nodes, BAND);
            full_bytes += ca.q.len();
            band_bytes += cb.q.len();
        }
        assert_eq!(a.channels[0].q, c.channels[0].q, "s changed");
        assert_eq!(a.channels[1].q, c.channels[1].q, "d_b changed");
    }
    assert!(
        band_bytes * 2 < full_bytes,
        "band {band_bytes} of {full_bytes} attribute bytes"
    );
    for o in origins() {
        let (x, y) = (
            full.march_decided(o, -Vec3::Z, 1.0),
            b.march_decided(o, -Vec3::Z, 1.0),
        );
        let (bx, by) = (bits(&x), bits(&y));
        assert!(bx.is_some(), "{o}: the full field missed ({x:?})");
        assert_eq!(bx, by, "{o}: full {x:?} band {y:?}");
        hits += 1;
    }
    assert_eq!(hits, origins().len());

    #[cfg(feature = "gpu")]
    {
        use alice_sdf::sheet_field::gpu::GpuSheetField;
        let (gf, gb) = match (GpuSheetField::new(&full), GpuSheetField::new(&b)) {
            (Ok(a), Ok(c)) => (a, c),
            (Err(e), _) | (_, Err(e)) => {
                assert!(
                    std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                    "ALICE_SDF_REQUIRE_GPU set but no GPU: {e}"
                );
                eprintln!("gpu-test: skipped (no GPU: {e})");
                return;
            }
        };
        let os = origins();
        let ra = gf.march_batch_decided(&os, -Vec3::Z, 1.0).unwrap();
        let rb = gb.march_batch_decided(&os, -Vec3::Z, 1.0).unwrap();
        for ((o, a), c) in os.iter().zip(&ra).zip(&rb) {
            assert!(bits(a).is_some(), "{o}: gpu full {a:?}");
            assert_eq!(bits(a), bits(c), "{o}: gpu full {a:?} band {c:?}");
        }
    }
}

#[test]
fn an_empty_band_is_refused_on_the_gpu_and_nan_on_the_cpu() {
    let full = patch();
    let empty = banded(&full, -1.0e9);
    let lv = &empty.sheets[0].levels[0];
    assert!(lv.band.iter().all(|&w| w == 0), "the empty band kept nodes");
    assert!(lv.channels[2].q.is_empty());
    let o = origins()[100];
    match empty.march_decided(o, -Vec3::Z, 1.0) {
        MarchResult::Hit(h) => {
            let uv = h.uv.unwrap();
            assert!(
                uv[0].is_nan() && uv[1].is_nan(),
                "{o}: uv {uv:?} read outside the band"
            );
        }
        m => panic!("{o}: {m:?}"),
    }
    #[cfg(feature = "gpu")]
    {
        use alice_sdf::sheet_field::gpu::GpuSheetField;
        let g = match GpuSheetField::new(&empty) {
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
        let err = g
            .march_batch_decided(&[o], -Vec3::Z, 1.0)
            .expect_err("a hit outside the band was shaded");
        assert!(err.to_string().contains("outside its band"), "{err}");
    }
}

#[test]
fn a_banded_field_is_ashf_v3_and_others_keep_their_version() {
    let full = patch();
    let version = |f: &SheetField| {
        let mut b = Vec::new();
        write_ashf(&mut b, f).unwrap();
        (u32::from_le_bytes(b[4..8].try_into().unwrap()), b)
    };
    assert_eq!(version(&full).0, 1);
    let coarse = SheetField::new(
        vec![sheet(vec![{
            let mut lv = full.sheets[0].levels[0].clone();
            for ch in lv.channels.iter_mut().skip(2) {
                *ch = coarsen(ch, 5).unwrap();
            }
            lv
        }])
        .unwrap()],
        full.rule,
    );
    assert_eq!(version(&coarse).0, 2);
    let b = banded(&full, 2.0);
    let (v, bytes) = version(&b);
    assert_eq!(v, 3);
    assert!(bytes.len() < version(&full).1.len());
    let mut cur = Cursor::new(bytes);
    let ix = AshfIndex::scan(&mut cur).unwrap();
    let back = ix.read_sheets(&mut cur, 0..ix.spans.len()).unwrap();
    for (a, c) in b.sheets[0].levels.iter().zip(&back[0].levels) {
        assert_eq!(a.band, c.band);
        for (ca, cc) in a.channels.iter().zip(&c.channels) {
            assert_eq!(
                (ca.nodes, &ca.q, &ca.rows, &ca.starts),
                (cc.nodes, &cc.q, &cc.rows, &cc.starts)
            );
            assert_eq!((&ca.scale, &ca.offset), (&cc.scale, &cc.offset));
        }
    }
    let rf = SheetField::new(back, ix.rule);
    for o in origins().into_iter().step_by(37) {
        assert_eq!(
            bits(&b.march_decided(o, -Vec3::Z, 1.0)),
            bits(&rf.march_decided(o, -Vec3::Z, 1.0))
        );
    }
}

#[test]
fn malformed_bands_and_the_band_kind_in_a_v2_file_are_refused() {
    let full = patch();
    let lv = &full.sheets[0].levels[0];
    // a level without attributes, and a level already banded
    let mut two = lv.clone();
    two.channels.truncate(2);
    assert!(band(&two, 2.0).is_err());
    let b = band(lv, 2.0).unwrap();
    assert!(band(&b, 2.0).is_err());
    // a mask that no longer matches the bytes
    let mut bad = b.clone();
    let k = bad.band.iter().position(|&w| w != 0).unwrap();
    bad.band[k] &= bad.band[k] - 1;
    assert!(
        sheet(vec![bad]).is_err(),
        "a mask with one bit less was accepted"
    );
    // a band channel declared in a v2 file: write v3, relabel it v2
    let f = SheetField::new(vec![sheet(vec![b]).unwrap()], full.rule);
    let mut bytes = Vec::new();
    write_ashf(&mut bytes, &f).unwrap();
    bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
    // refused for the band kind itself (the v3 mask words would misalign a v2 read and refuse it anyway)
    let err = AshfIndex::scan(&mut Cursor::new(bytes))
        .expect_err("a band channel in a v2 file was accepted");
    assert!(
        err.to_string().contains("0 nodes per brick axis"),
        "refused for another reason: {err}"
    );
}
