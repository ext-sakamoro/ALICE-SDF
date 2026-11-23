//! Attribute channels on a coarse grid: `u`, `v` and the author normal stored on 5, 3 or 2 nodes per brick axis
//! (every 2nd, 4th or 8th corner) instead of 9, read by trilinear interpolation on their own grid, while `s` and
//! `d_b` stay on 9 nodes. The geometry and the march are unchanged; only the attributes are read from fewer nodes.
//!
//! Oracles (none is the sampler's own output):
//! - at a coarse node the sample is that node's stored value (q * scale + offset, read from the bytes);
//! - the attributes are linear in the position, which trilinear interpolation reproduces on any grid, so the
//!   fine and the coarse samples agree within one quantization step of the row (each is within half a step);
//! - the GPU marches a coarse field like the CPU: same sheet, `t` and UV within the parity tolerances;
//! - ASHF: a field with a coarse channel is written as v2 and read back with the same bytes and node counts, a
//!   field without one is still v1, byte for byte as before;
//! - refusals: `s` or `d_b` on a coarse grid, and a v2 file naming a node count that is not a grid.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use alice_sdf::sheet_field::build::{coarsen, level_from_fn, sheet};
use alice_sdf::sheet_field::stream::{write_ashf, AshfIndex};
use alice_sdf::sheet_field::{SheetField, ValidRule, BRICK, COARSE_NODES, NODES};
use glam::Vec3;
use std::io::Cursor;

const H: f32 = 0.0625;

fn square_db(x: f32, y: f32) -> f32 {
    let dx = x.abs() - 1.0;
    let dy = y.abs() - 1.0;
    Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0)
}

/// the plane patch z = 0 with linear attributes (u, v, and an author normal tilted across the patch)
fn patch() -> SheetField {
    let lo = Vec3::new(-1.5 + 0.013, -1.5 + 0.021, -0.5 + 0.007);
    let lv = level_from_fn(
        lo,
        H,
        [6, 6, 2],
        |_c, _e| true,
        |p| {
            vec![
                p.z,
                square_db(p.x, p.y),
                0.5 + 0.01 * p.x,
                0.25 + 0.02 * p.y,
                0.2 * p.x,
                -0.1 * p.y,
                1.0 + 0.05 * p.x,
            ]
        },
    )
    .unwrap();
    SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1)
}

/// the same field with every attribute channel (2..) on `nodes` per axis
fn coarse(f: &SheetField, nodes: u32) -> SheetField {
    let sheets = f
        .sheets
        .iter()
        .map(|s| {
            let levels = s
                .levels
                .iter()
                .map(|lv| {
                    let mut lv = lv.clone();
                    for ch in lv.channels.iter_mut().skip(2) {
                        *ch = coarsen(ch, nodes).unwrap();
                    }
                    lv
                })
                .collect();
            sheet(levels).unwrap()
        })
        .collect();
    SheetField::new(sheets, f.rule)
}

/// points inside the patch at non-node positions
fn points() -> Vec<Vec3> {
    let mut v = Vec::new();
    for i in 0..23 {
        for j in 0..19 {
            v.push(Vec3::new(
                -0.9 + 0.081 * i as f32,
                -0.9 + 0.097 * j as f32,
                0.003 * ((i * 7 + j * 3) % 11) as f32 - 0.015,
            ));
        }
    }
    v
}

#[test]
fn at_a_coarse_node_the_sample_is_that_nodes_stored_value() {
    let f = patch();
    let mut compared = 0;
    for &n in &COARSE_NODES {
        let c = coarse(&f, n);
        let lv = &c.sheets[0].levels[0];
        let stride = BRICK / (n as usize - 1);
        let ch = &lv.channels[2];
        assert_eq!(ch.nodes, n);
        // interior bricks (1..5) so that no other brick answers at the shared faces
        for (bx, by, bz) in [(1usize, 1usize, 0usize), (2, 3, 1), (4, 2, 0), (3, 4, 1)] {
            let k = lv.index[(bx * lv.nb[1] as usize + by) * lv.nb[2] as usize + bz];
            assert!(k >= 0, "brick {bx} {by} {bz} is active");
            let r = ch.rows[k as usize] as usize;
            let nn = n as usize;
            // nodes the brick owns: index <= nn - 2 (its far face belongs to the next brick, which stores the
            // shared corner with its own scale and offset)
            let (a, b) = (1.min(nn - 2), nn - 2);
            for (x, y, z) in [(0, 0, 0), (a, 0, a), (b, a, 0), (a, b, b)] {
                let cell = [
                    bx * BRICK + x * stride,
                    by * BRICK + y * stride,
                    bz * BRICK + z * stride,
                ];
                let p = lv.lo + Vec3::new(cell[0] as f32, cell[1] as f32, cell[2] as f32) * lv.h;
                let want = f32::from(ch.q[r * nn * nn * nn + (x * nn + y) * nn + z]) * ch.scale[r]
                    + ch.offset[r];
                let smp = c.sheets[0].sample(p, ValidRule::None);
                let got = smp.uv.unwrap_or_else(|| panic!("{n} nodes, brick {bx},{by},{bz} node {x},{y},{z}: no level answered at {p} ({smp:?})"))[0];
                assert!(
                    (got - want).abs() <= 1e-3 * ch.scale[r] + 1e-6,
                    "{n} nodes, brick {bx},{by},{bz} node {x},{y},{z}: sampled {got}, stored {want}"
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, COARSE_NODES.len() * 16);
}

#[test]
fn linear_attributes_agree_between_the_fine_and_the_coarse_grid_within_one_step() {
    let f = patch();
    let fine = &f.sheets[0];
    let mut compared = 0;
    for &n in &COARSE_NODES {
        let c = coarse(&f, n);
        let lv = &c.sheets[0].levels[0];
        // one quantization step of each attribute channel: the largest row scale
        let step: Vec<f32> = (2..7)
            .map(|k| lv.channels[k].scale.iter().copied().fold(0.0, f32::max))
            .collect();
        for p in points() {
            let a = fine.sample(p, ValidRule::None);
            let b = c.sheets[0].sample(p, ValidRule::None);
            assert_eq!(
                (a.f, a.g, a.s),
                (b.f, b.g, b.s),
                "{n} nodes: the geometry moved at {p}"
            );
            let (ua, ub) = (a.uv.unwrap(), b.uv.unwrap());
            let (na, nb) = (a.attr_normal.unwrap(), b.attr_normal.unwrap());
            let fa = [ua[0], ua[1], na[0], na[1], na[2]];
            let fb = [ub[0], ub[1], nb[0], nb[1], nb[2]];
            for k in 0..5 {
                assert!(
                    (fa[k] - fb[k]).abs() <= 1.001 * step[k],
                    "{n} nodes, channel {} at {p}: fine {} coarse {} (step {})",
                    k + 2,
                    fa[k],
                    fb[k],
                    step[k]
                );
            }
            compared += 1;
        }
    }
    assert!(compared > 1000, "compared {compared}");
}

#[cfg(feature = "gpu")]
#[test]
fn the_gpu_marches_a_coarse_field_like_the_cpu() {
    use alice_sdf::sheet_field::gpu::GpuSheetField;
    let f = coarse(&patch(), 5);
    let origins: Vec<Vec3> = points().iter().map(|p| Vec3::new(p.x, p.y, 0.4)).collect();
    let g = match GpuSheetField::new(&f) {
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
    let gpu = g.march_batch_decided(&origins, -Vec3::Z, 1.0).unwrap();
    let mut hits = 0;
    for (o, m) in origins.iter().zip(&gpu) {
        match (m, f.march_decided(*o, -Vec3::Z, 1.0)) {
            (
                alice_sdf::sheet_field::MarchResult::Hit(a),
                alice_sdf::sheet_field::MarchResult::Hit(b),
            ) => {
                assert_eq!(a.sheet, b.sheet, "{o}");
                assert!((a.t - b.t).abs() <= 1e-3, "{o}: t {} vs {}", a.t, b.t);
                let (ua, ub) = (a.uv.unwrap(), b.uv.unwrap());
                assert!(
                    (ua[0] - ub[0]).abs() <= 2e-4 && (ua[1] - ub[1]).abs() <= 2e-4,
                    "{o}: uv {ua:?} vs {ub:?}"
                );
                let (na, nb) = (a.attr_normal.unwrap(), b.attr_normal.unwrap());
                assert!(na.dot(nb) >= 0.99999, "{o}: author normal {na} vs {nb}");
                hits += 1;
            }
            (m, c) => panic!("{o}: gpu {m:?} cpu {c:?}"),
        }
    }
    assert_eq!(hits, origins.len());
}

#[test]
fn a_coarse_field_is_ashf_v2_and_a_fine_field_stays_v1() {
    let f = patch();
    let mut fine_bytes = Vec::new();
    write_ashf(&mut fine_bytes, &f).unwrap();
    assert_eq!(u32::from_le_bytes(fine_bytes[4..8].try_into().unwrap()), 1);
    for &n in &COARSE_NODES {
        let c = coarse(&f, n);
        let mut bytes = Vec::new();
        write_ashf(&mut bytes, &c).unwrap();
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            2,
            "{n} nodes"
        );
        assert!(
            bytes.len() < fine_bytes.len(),
            "{n} nodes: {} >= {}",
            bytes.len(),
            fine_bytes.len()
        );
        let mut cur = Cursor::new(bytes);
        let ix = AshfIndex::scan(&mut cur).unwrap();
        let back = ix.read_sheets(&mut cur, 0..ix.spans.len()).unwrap();
        for (a, b) in c.sheets[0].levels.iter().zip(&back[0].levels) {
            for (ca, cb) in a.channels.iter().zip(&b.channels) {
                assert_eq!((ca.nodes, &ca.q, &ca.rows), (cb.nodes, &cb.q, &cb.rows));
                assert_eq!((&ca.scale, &ca.offset), (&cb.scale, &cb.offset));
            }
        }
    }
    // a fine field reads back as 9 nodes everywhere
    let mut cur = Cursor::new(fine_bytes);
    let ix = AshfIndex::scan(&mut cur).unwrap();
    let back = ix.read_sheets(&mut cur, 0..1).unwrap();
    assert!(back[0].levels[0]
        .channels
        .iter()
        .all(|c| c.nodes == NODES as u32));
}

#[test]
fn s_or_d_b_on_a_coarse_grid_and_a_v2_file_with_a_non_grid_node_count_are_refused() {
    let f = patch();
    let lv = f.sheets[0].levels[0].clone();
    for c in 0..2 {
        let mut bad = lv.clone();
        let ch = &lv.channels[c];
        bad.channels[c] = alice_sdf::sheet_field::Channel {
            rows: ch.rows.clone(),
            q: vec![0u8; ch.scale.len() * 125],
            scale: ch.scale.clone(),
            offset: ch.offset.clone(),
            nodes: 5,
            starts: Vec::new(),
        };
        assert!(
            sheet(vec![bad]).is_err(),
            "channel {c} on 5 nodes was accepted"
        );
    }
    // coarsen refuses an already coarse channel and a non-grid count
    assert!(coarsen(&coarsen(&lv.channels[2], 5).unwrap(), 3).is_err());
    assert!(coarsen(&lv.channels[2], 4).is_err());
    // a v2 file whose first channel names 4 nodes per axis
    let c = coarse(&f, 5);
    let mut bytes = Vec::new();
    write_ashf(&mut bytes, &c).unwrap();
    // header 16, sheet: levels 4, level: h 4 + lo 12 + nb 12 + n_index 4 + index, n_channels 4, channel 0: n_rows 4 + rows, n_kept 4, nodes
    let ni = u32::from_le_bytes(bytes[48..52].try_into().unwrap()) as usize;
    let o = 52 + 4 * ni + 4;
    let nr = u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
    let nodes_at = o + 4 + 4 * nr + 4;
    assert_eq!(
        u32::from_le_bytes(bytes[nodes_at..nodes_at + 4].try_into().unwrap()),
        9
    );
    bytes[nodes_at..nodes_at + 4].copy_from_slice(&4u32.to_le_bytes());
    // refused for the count itself (a size mismatch further on would refuse it too, for the wrong reason)
    let err = AshfIndex::scan(&mut Cursor::new(bytes)).expect_err("4 nodes per axis was accepted");
    assert!(
        err.to_string().contains("4 nodes per brick axis"),
        "refused for another reason: {err}"
    );
}
