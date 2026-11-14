//! A field marched in batches read from disk gives the same hits, bit for bit, as the same field held in memory.
//!
//! Oracles (none of them is the streamed march's own output):
//! - layout: the byte count of a written field follows from the counts alone (module docs' layout), and the index
//!   found by seeking past the arrays matches the payload of the sheets as read;
//! - geometry: stacked flat squares `z = z_k`, rays straight down; the first hit is the highest square that covers
//!   the ray's (x, y), at `t = z0 - z_k` within one `s` quantization step (the analytic plane);
//! - invariance: the batch split (memory budget) changes nothing: budgets {one chunk per batch, two, everything}
//!   give identical results, and the peak host bytes stay within the budget;
//! - merge rule (module docs): crafted per-chunk results check smaller-t wins, equal t keeps the earlier chunk,
//!   undecided anywhere is undecided, miss needs a miss everywhere, sheets renumbered by the chunk start.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::stream::{
    chunk_split, host_batches, march_streamed, sheet_payload_bytes, write_ashf, AshfIndex,
    StreamError,
};
use alice_sdf::sheet_field::{Hit, MarchResult, Sheet, SheetField, ValidRule, BRICK_NODES};
use glam::Vec3;
use std::io::Cursor;

const H: f32 = 0.25;
const Z0: f32 = 15.0;

/// One flat square `z = z`, centre (cx, cy), half size `half`, in a 16 x 16 x 2 mm box (8 x 8 x 1 bricks).
fn square(z: f32, cx: f32, cy: f32, half: f32, uv: bool) -> Sheet {
    let lo = Vec3::new(cx - 8.0 + 0.013, cy - 8.0 + 0.021, z - 1.0 + 0.007);
    let lv = level_from_fn(
        lo,
        H,
        [8, 8, 1],
        |_c, _e| true,
        |p| {
            let (dx, dy) = ((p.x - cx).abs() - half, (p.y - cy).abs() - half);
            let db = Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0);
            let mut v = vec![p.z - z, db];
            if uv {
                v.extend([(p.x - cx) / 16.0 + 0.5, (p.y - cy) / 16.0 + 0.5]);
            }
            v
        },
    )
    .unwrap();
    sheet(vec![lv]).unwrap()
}

/// 12 squares stacked downward, shifted so lower ones show around the upper ones; every third carries UV.
fn stack() -> Vec<(Sheet, f32, f32, f32, f32)> {
    (0..12)
        .map(|k| {
            let z = 10.0 - 0.7 * k as f32;
            let (cx, cy) = (((k * 5) % 7) as f32 - 3.0, ((k * 3) % 5) as f32 - 2.0);
            let half = 2.0 + (k % 4) as f32 * 0.9;
            (square(z, cx, cy, half, k % 3 == 0), z, cx, cy, half)
        })
        .collect()
}

fn field_bytes(sheets: Vec<Sheet>) -> Vec<u8> {
    let mut b = Vec::new();
    write_ashf(&mut b, &SheetField::new(sheets, ValidRule::CellV1)).unwrap();
    b
}

fn rays() -> Vec<Vec3> {
    (0..24 * 24)
        .map(|i| {
            Vec3::new(
                -9.0 + 18.0 * ((i % 24) as f32 + 0.37) / 24.0,
                -9.0 + 18.0 * ((i / 24) as f32 + 0.61) / 24.0,
                Z0,
            )
        })
        .collect()
}

fn cpu(origins: &[Vec3]) -> impl FnMut(&SheetField) -> Result<Vec<MarchResult>, String> + '_ {
    move |f: &SheetField| {
        Ok(origins
            .iter()
            .map(|o| f.march_decided(*o, -Vec3::Z, 2.0 * Z0))
            .collect())
    }
}

/// results as bits (t, normal, uv, attr normal as raw f32 bits; sheet as is)
fn bits(r: &[MarchResult]) -> Vec<(u8, Vec<u32>)> {
    r.iter()
        .map(|m| match m {
            MarchResult::Hit(h) => {
                let mut v = vec![h.t.to_bits(), h.sheet as u32];
                v.extend(h.normal.to_array().map(f32::to_bits));
                v.extend(h.uv.map_or([u32::MAX; 2], |u| u.map(f32::to_bits)));
                v.extend(
                    h.attr_normal
                        .map_or([u32::MAX; 3], |a| a.to_array().map(f32::to_bits)),
                );
                (0, v)
            }
            MarchResult::Miss => (1, vec![]),
            MarchResult::Undecided => (2, vec![]),
        })
        .collect()
}

#[test]
fn the_layout_size_follows_from_the_counts_and_the_index_matches_what_is_read() {
    let st = stack();
    let sheets: Vec<Sheet> = st.iter().map(|s| s.0.clone()).collect();
    let b = field_bytes(sheets.clone());
    // expected size from the layout alone: 16 header bytes; per sheet 4; per level 32 + 4 index + 4 nch; per
    // channel 4 + 4 rows + 4 + 8 kept + kept * BRICK_NODES
    let mut want = 16usize;
    for sh in &sheets {
        want += 4;
        for lv in &sh.levels {
            want += 32 + 4 * lv.index.len() + 4;
            for c in &lv.channels {
                want += 4 + 4 * c.rows.len() + 4 + 8 * c.scale.len() + c.scale.len() * BRICK_NODES;
            }
        }
    }
    assert_eq!(b.len(), want);
    let ix = AshfIndex::scan(&mut Cursor::new(&b)).unwrap();
    assert_eq!(ix.spans.len(), 12);
    assert_eq!(ix.spans[0].offset, 16);
    for w in ix.spans.windows(2) {
        assert_eq!(w[0].offset + w[0].len, w[1].offset, "spans tile the file");
    }
    let back = ix.read_sheets(&mut Cursor::new(&b), 0..12).unwrap();
    for (k, (a, s)) in back.iter().zip(&ix.spans).enumerate() {
        assert_eq!(sheet_payload_bytes(a), s.host_bytes, "sheet {k}");
        assert_eq!(
            a.levels
                .iter()
                .flat_map(|l| &l.channels)
                .map(|c| c.q.len() as u64)
                .sum::<u64>(),
            s.channel_bytes,
            "sheet {k}"
        );
    }
    // read back and written again: the same bytes; a middle range reads the same sheets as the full read
    assert_eq!(field_bytes(back.clone()), b);
    let mid = ix.read_sheets(&mut Cursor::new(&b), 4..9).unwrap();
    assert_eq!(field_bytes(mid), field_bytes(back[4..9].to_vec()));
}

#[test]
fn hits_are_the_analytic_first_square_and_do_not_depend_on_the_budget() {
    let st = stack();
    let b = field_bytes(st.iter().map(|s| s.0.clone()).collect());
    let ix = AshfIndex::scan(&mut Cursor::new(&b)).unwrap();
    // a cap of ~2 sheets of channel bytes: several chunks, split by the field only
    let cap = ix.spans.iter().map(|s| s.channel_bytes).max().unwrap() * 2;
    let chunks = chunk_split(&ix, cap).unwrap();
    assert!(chunks.len() >= 4, "{chunks:?}");
    let cost = |c: &std::ops::Range<usize>| {
        ix.spans[c.clone()]
            .iter()
            .map(|s| s.host_bytes)
            .sum::<u64>()
    };
    let one = chunks.iter().map(cost).max().unwrap();
    let all: u64 = ix.spans.iter().map(|s| s.host_bytes).sum();
    let origins = rays();
    let mut runs = Vec::new();
    for budget in [one, 2 * one, all] {
        let batches = host_batches(&ix, &chunks, budget).unwrap();
        let (r, stats) = march_streamed(
            &ix,
            &mut Cursor::new(&b),
            &chunks,
            &batches,
            origins.len(),
            cpu(&origins),
        )
        .unwrap();
        assert!(
            stats.peak_host_bytes <= budget,
            "peak {} over budget {budget}",
            stats.peak_host_bytes
        );
        assert!(stats.peak_host_bytes > 0);
        assert_eq!(stats.chunks, chunks.len());
        assert_eq!(stats.batches, batches.len());
        runs.push((budget, stats, r));
    }
    assert_eq!(
        runs[0].1.batches,
        chunks.len(),
        "the smallest budget holds one chunk at a time"
    );
    assert_eq!(runs[2].1.batches, 1, "everything fits in one batch");
    let base = bits(&runs[2].2);
    for (budget, _, r) in &runs[..2] {
        assert_eq!(bits(r), base, "budget {budget} changed the result");
    }
    // analytic: the highest square covering (x, y), at t = Z0 - z; s spans the 2 mm brick in 255 steps
    let (mut hits, mut misses) = (0, 0);
    for (o, m) in origins.iter().zip(&runs[2].2) {
        let want = st.iter().enumerate().find(|(_, s)| {
            let (dx, dy) = ((o.x - s.2).abs() - s.4, (o.y - s.3).abs() - s.4);
            dx < -0.05 && dy < -0.05
        });
        let near_rim = st.iter().any(|s| {
            let (dx, dy) = ((o.x - s.2).abs() - s.4, (o.y - s.3).abs() - s.4);
            dx.max(dy).abs() <= 0.05
        });
        if near_rim {
            continue;
        }
        match (want, m) {
            (Some((k, s)), MarchResult::Hit(h)) => {
                assert_eq!(h.sheet, k, "ray {o}");
                assert!(
                    (h.t - (Z0 - s.1)).abs() <= 2.0 / 255.0,
                    "ray {o}: t {} want {}",
                    h.t,
                    Z0 - s.1
                );
                hits += 1;
            }
            (None, MarchResult::Miss) => misses += 1,
            (w, m) => panic!("ray {o}: want {:?}, got {m:?}", w.map(|x| x.0)),
        }
    }
    assert!(hits > 200 && misses > 20, "hits {hits}, misses {misses}");
}

fn hit(t: f32, sheet: usize) -> MarchResult {
    MarchResult::Hit(Hit {
        t,
        sheet,
        normal: Vec3::Z,
        uv: None,
        attr_normal: None,
    })
}

#[test]
fn the_merge_keeps_the_nearest_the_earlier_chunk_on_ties_and_undecided_anywhere() {
    // three one-sheet chunks; the callback answers per chunk from a table (rays x chunks)
    let b = field_bytes(vec![
        square(5.0, 0.0, 0.0, 2.0, false),
        square(3.0, 0.0, 0.0, 2.0, false),
        square(1.0, 0.0, 0.0, 2.0, false),
    ]);
    let ix = AshfIndex::scan(&mut Cursor::new(&b)).unwrap();
    let chunks = chunk_split(&ix, ix.spans[0].channel_bytes).unwrap();
    assert_eq!(chunks, vec![0..1, 1..2, 2..3]);
    let m = MarchResult::Miss;
    let u = MarchResult::Undecided;
    let table = [
        [hit(4.0, 0), hit(2.0, 0), hit(3.0, 0)], // nearest: chunk 1 -> sheet 1
        [hit(2.0, 0), hit(2.0, 0), hit(2.0, 0)], // tie: chunk 0 -> sheet 0
        [m, hit(7.0, 0), m],                     // one hit among misses -> sheet 1
        [m, m, m],                               // miss
        [hit(1.0, 0), u, m],                     // undecided anywhere -> undecided
        [m, m, u],                               // undecided last
    ];
    let want = [hit(2.0, 1), hit(2.0, 0), hit(7.0, 1), m, u, u];
    for budget in [ix.spans[0].host_bytes, u64::MAX] {
        let batches = host_batches(&ix, &chunks, budget).unwrap();
        let mut call = 0usize;
        let (r, _) = march_streamed(
            &ix,
            &mut Cursor::new(&b),
            &chunks,
            &batches,
            table.len(),
            |_f| {
                let c = call;
                call += 1;
                Ok(table.iter().map(|row| row[c]).collect())
            },
        )
        .unwrap();
        assert_eq!(r, want, "budget {budget}");
    }
}

#[test]
fn bad_input_and_over_budget_are_refused() {
    let b = field_bytes(stack().into_iter().map(|s| s.0).take(4).collect());
    let ix = AshfIndex::scan(&mut Cursor::new(&b)).unwrap();
    let mut bad = b.clone();
    bad[0] = b'X';
    assert!(matches!(
        AshfIndex::scan(&mut Cursor::new(&bad)),
        Err(StreamError::Format(_))
    ));
    let mut bad = b.clone();
    bad[4] = 2;
    assert!(matches!(
        AshfIndex::scan(&mut Cursor::new(&bad)),
        Err(StreamError::Format(_))
    ));
    let mut bad = b.clone();
    bad.push(0);
    assert!(
        matches!(
            AshfIndex::scan(&mut Cursor::new(&bad)),
            Err(StreamError::Format(_))
        ),
        "trailing byte"
    );
    let short = &b[..b.len() - 1];
    assert!(
        AshfIndex::scan(&mut Cursor::new(short)).is_err(),
        "truncated"
    );
    assert!(
        AshfIndex::scan(&mut Cursor::new(&b[..10])).is_err(),
        "header only"
    );
    assert!(
        ix.read_sheets(&mut Cursor::new(&b), 2..9).is_err(),
        "range past the sheets"
    );
    let small = ix.spans.iter().map(|s| s.channel_bytes).min().unwrap() - 1;
    assert!(
        matches!(chunk_split(&ix, small), Err(StreamError::Budget(_))),
        "a sheet over the cap"
    );
    let chunks = chunk_split(&ix, u64::MAX).unwrap();
    assert_eq!(chunks.len(), 1);
    assert!(
        matches!(host_batches(&ix, &chunks, 1), Err(StreamError::Budget(_))),
        "a chunk over the budget"
    );
    let origins = rays();
    let r = march_streamed(
        &ix,
        &mut Cursor::new(&b),
        &chunks,
        &[],
        origins.len(),
        cpu(&origins),
    );
    assert!(
        matches!(r, Err(StreamError::Format(_))),
        "batches that do not tile the chunks"
    );
    let r = march_streamed(
        &ix,
        &mut Cursor::new(&b),
        &chunks,
        &[0..1],
        origins.len(),
        |_f| Ok(vec![MarchResult::Miss]),
    );
    assert!(
        matches!(r, Err(StreamError::March(_))),
        "one result for many rays"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn the_gpu_march_streamed_equals_the_cpu_march_streamed_per_chunk_split() {
    use alice_sdf::sheet_field::gpu::GpuSheetField;
    let b = field_bytes(stack().into_iter().map(|s| s.0).collect());
    let ix = AshfIndex::scan(&mut Cursor::new(&b)).unwrap();
    let cap = ix.spans.iter().map(|s| s.channel_bytes).max().unwrap() * 3;
    let chunks = chunk_split(&ix, cap).unwrap();
    let origins = rays();
    let one = chunks
        .iter()
        .map(|c| {
            ix.spans[c.clone()]
                .iter()
                .map(|s| s.host_bytes)
                .sum::<u64>()
        })
        .max()
        .unwrap();
    let batches = host_batches(&ix, &chunks, one).unwrap();
    let gpu = march_streamed(
        &ix,
        &mut Cursor::new(&b),
        &chunks,
        &batches,
        origins.len(),
        |f| {
            let g = GpuSheetField::new(f).map_err(|e| e.to_string())?;
            g.march_batch_decided(&origins, -Vec3::Z, 2.0 * Z0)
                .map_err(|e| e.to_string())
        },
    );
    match gpu {
        Ok((g, _)) => {
            let all = host_batches(&ix, &chunks, u64::MAX).unwrap();
            let g2 = march_streamed(
                &ix,
                &mut Cursor::new(&b),
                &chunks,
                &all,
                origins.len(),
                |f| {
                    let g = GpuSheetField::new(f).map_err(|e| e.to_string())?;
                    g.march_batch_decided(&origins, -Vec3::Z, 2.0 * Z0)
                        .map_err(|e| e.to_string())
                },
            )
            .unwrap()
            .0;
            assert_eq!(bits(&g), bits(&g2), "gpu: the budget changed the result");
            let c = march_streamed(
                &ix,
                &mut Cursor::new(&b),
                &chunks,
                &batches,
                origins.len(),
                cpu(&origins),
            )
            .unwrap()
            .0;
            let same = g.iter().zip(&c).filter(|(a, b)| matches!((a, b), (MarchResult::Hit(x), MarchResult::Hit(y)) if x.sheet == y.sheet) || matches!((a, b), (MarchResult::Miss, MarchResult::Miss))).count();
            assert_eq!(
                same,
                origins.len(),
                "gpu and cpu disagree on hit owners or misses"
            );
        }
        Err(StreamError::March(e)) => {
            assert!(
                std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                "ALICE_SDF_REQUIRE_GPU set but no GPU: {e}"
            );
            eprintln!("gpu-test: skipped (no GPU: {e})");
        }
        Err(e) => panic!("{e}"),
    }
}
