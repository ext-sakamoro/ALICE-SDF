//! The GPU march goes out in dispatches sized by their measured time, in an order spread over the batch, with a
//! wait bound, a per-ray work budget and a check that the dispatch ran.
//!
//! Oracles (none is the march's own output):
//! - grouping: the results with the adaptive sizes equal, bit for bit, the results of a field forced to the
//!   smallest dispatches (each ray's march does not depend on its neighbours); the sizes sum to the batch, a size
//!   is at most twice the one before, and after a dispatch over the target the next is at most half of it;
//! - the spread order is a permutation of the batch for many sizes;
//! - a wait bound of zero (the dispatch cannot have returned) is a typed `DispatchDidNotFinish` naming the rays,
//!   and the field refuses further work;
//! - a dispatch over the device's workgroups per dimension is refused before submitting, never read unwritten;
//! - a constructed worst case (100 overlapping sheets) with a small work budget ends every ray as undecided and
//!   counts it, and with the default budget the same rays hit.
#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use alice_sdf::compiled::GpuError;
use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::gpu::{
    dispatch_cap, spread_blocks, spread_stride, GpuSheetField, MARCH_OUT_BYTES_PER_RAY,
    MIN_DISPATCH,
};
use alice_sdf::sheet_field::{MarchResult, Sheet, SheetField, ValidRule};
use glam::Vec3;
use std::time::Duration;

fn square(z: f32, half: f32) -> Sheet {
    let lo = Vec3::new(-8.0 + 0.013, -8.0 + 0.021, z - 1.0 + 0.007);
    let lv = level_from_fn(
        lo,
        0.25,
        [8, 8, 1],
        |_c, _e| true,
        |p| {
            let (dx, dy) = (p.x.abs() - half, p.y.abs() - half);
            vec![
                p.z - z,
                Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0),
            ]
        },
    )
    .unwrap();
    sheet(vec![lv]).unwrap()
}

/// 100 flat squares 0.01 apart, every box overlapping: the deepest live list the tests build
fn deep() -> SheetField {
    SheetField::new(
        (0..100).map(|k| square(0.01 * k as f32, 5.0)).collect(),
        ValidRule::CellV1,
    )
}

fn origins(n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|i| {
            let (a, b) = (
                (i % 211) as f32 / 211.0,
                (i / 211) as f32 / (n / 211 + 1) as f32,
            );
            Vec3::new(-6.0 + 12.0 * a, -6.0 + 12.0 * b, 5.0)
        })
        .collect()
}

fn gpu(f: &SheetField) -> Option<GpuSheetField> {
    match GpuSheetField::new(f) {
        Ok(g) => Some(g),
        Err(e) => {
            assert!(
                std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                "ALICE_SDF_REQUIRE_GPU set but no GPU: {e}"
            );
            eprintln!("gpu-test: skipped (no GPU: {e})");
            None
        }
    }
}

#[test]
fn the_spread_order_is_a_permutation() {
    for n in [
        1usize, 2, 3, 10, 64, 97, 1000, 1024, 65_536, 262_144, 10_240_000,
    ] {
        let s = spread_stride(n);
        if n <= 2_000_000 {
            let mut seen = vec![false; n];
            for j in 0..n {
                let i = ((j as u128 * s as u128) % n as u128) as usize;
                assert!(!seen[i], "n {n} stride {s}: ray {i} twice");
                seen[i] = true;
            }
        } else {
            let gcd = |mut a: usize, mut b: usize| {
                while b != 0 {
                    (a, b) = (b, a % b);
                }
                a
            };
            assert_eq!(gcd(s, n), 1, "n {n} stride {s}");
        }
    }
}

/// The block order is a permutation of the batch for many sizes and blocks, keeps each block contiguous, and is
/// the contiguous order for block 0 or a block at least the batch.
#[test]
fn the_block_order_is_a_permutation_of_contiguous_blocks() {
    for n in [1usize, 2, 3, 63, 64, 65, 1000, 4096, 4097, 65_536, 100_003] {
        for block in [0usize, 1, 7, 64, 1024, 4096, 16_384, 1 << 20] {
            let o = spread_blocks(n, block);
            assert_eq!(o.len(), n, "n {n} block {block}");
            let mut seen = vec![false; n];
            for &i in &o {
                assert!(!seen[i], "n {n} block {block}: ray {i} twice");
                seen[i] = true;
            }
            if block == 0 || block >= n {
                assert!(
                    o.iter().enumerate().all(|(j, &i)| i == j),
                    "n {n} block {block}: not contiguous"
                );
            } else {
                // the order is whole blocks one after another: each starts at a multiple of `block`, runs `block`
                // rays (the last block of the batch, n % block, runs fewer), and blocks are not interleaved
                let mut at = 0;
                let mut short = 0;
                while at < n {
                    let start = o[at];
                    assert_eq!(
                        start % block,
                        0,
                        "n {n} block {block}: a block starts at {start}"
                    );
                    let len = block.min(n - start);
                    assert!(
                        (0..len).all(|r| o[at + r] == start + r),
                        "n {n} block {block}: the block at {start} is not contiguous"
                    );
                    short += usize::from(len < block);
                    at += len;
                }
                assert!(short <= 1, "n {n} block {block}: {short} short blocks");
            }
        }
    }
}

#[test]
fn the_results_do_not_depend_on_the_dispatch_sizes_and_the_sizes_follow_the_rule() {
    let f = deep();
    let (Some(a), Some(b)) = (gpu(&f), gpu(&f)) else {
        return;
    };
    let os = origins(60_000);
    let ra = a.march_batch_decided(&os, -Vec3::Z, 10.0).unwrap();
    b.set_dispatch_target(Duration::from_nanos(1));
    let rb = b.march_batch_decided(&os, -Vec3::Z, 10.0).unwrap();
    assert!(
        ra.iter().any(|m| matches!(m, MarchResult::Hit(_))),
        "no hits: the scene is wrong"
    );
    let diff: Vec<usize> = (0..ra.len()).filter(|&i| ra[i] != rb[i]).collect();
    if !diff.is_empty() {
        for &i in diff.iter().take(4) {
            eprintln!(
                "DIFF ray {i} at {}: adaptive {:?} smallest {:?}",
                os[i], ra[i], rb[i]
            );
        }
    }
    assert!(
        diff.is_empty(),
        "the results depend on the dispatch sizes: {} of {} rays differ",
        diff.len(),
        ra.len()
    );
    for (g, name) in [(&a, "adaptive"), (&b, "smallest")] {
        let st = g.last_march_stats();
        // a dispatch the device cut short is issued again (smaller): those rays are counted twice in the sizes
        eprintln!(
            "{name}: cut dispatches {}, re-issued rays {}",
            st.cut_dispatches, st.reissued_rays
        );
        assert_eq!(
            st.dispatches.iter().map(|d| d.0).sum::<usize>(),
            os.len() + st.reissued_rays as usize,
            "{name}"
        );
        assert_eq!(
            st.dispatch_work.len(),
            st.dispatches.len(),
            "{name}: a work total per dispatch"
        );
        assert_eq!(
            st.dispatch_round.len(),
            st.dispatches.len(),
            "{name}: a round per dispatch"
        );
        assert!(
            st.dispatch_work.iter().all(|&w| w > 0),
            "{name}: a dispatch with no work"
        );
        // the size rule holds between first issues; re-issue rounds halve the cut size instead
        let first: Vec<(usize, f64)> = st
            .dispatches
            .iter()
            .zip(&st.dispatch_round)
            .filter(|(_, &k)| k == 0)
            .map(|(d, _)| *d)
            .collect();
        for w in first.windows(2) {
            let ((r0, t0), (r1, _)) = (w[0], w[1]);
            let last = w[1] == *first.last().unwrap();
            assert!(
                r1 <= (2 * r0).max(MIN_DISPATCH) || last,
                "{name}: size {r0} -> {r1}"
            );
            if name == "smallest" && !last {
                assert!(
                    t0 <= 0.0 || r1 <= (r0 / 2).max(MIN_DISPATCH),
                    "{name}: over target, {r0} -> {r1}"
                );
            }
        }
    }
    assert!(
        b.last_march_stats().dispatches.len() > a.last_march_stats().dispatches.len(),
        "the tiny target did not make smaller dispatches"
    );
}

#[test]
fn a_wait_bound_of_zero_is_a_typed_error_and_the_field_refuses_more() {
    let f = deep();
    let Some(g) = gpu(&f) else { return };
    g.set_dispatch_wait(Duration::ZERO);
    let os = origins(20_000);
    match g.march_batch_decided(&os, -Vec3::Z, 10.0) {
        Err(GpuError::DispatchDidNotFinish { rays, limit_ms }) => {
            assert!(rays > 0 && rays <= os.len(), "rays {rays}");
            assert_eq!(limit_ms, 0);
        }
        other => panic!("not the typed error: {other:?}"),
    }
    assert!(
        g.march_batch_decided(&os[..10], -Vec3::Z, 10.0).is_err(),
        "the field took more work"
    );
}

#[test]
fn a_dispatch_over_the_workgroup_limit_is_refused_not_read_unwritten() {
    let f = deep();
    let Some(g) = gpu(&f) else { return };
    let n = 65_535 * 64 + 1;
    let pts = vec![Vec3::new(0.0, 0.0, 0.5); n];
    match g.eval_batch(&pts) {
        Err(GpuError::DispatchTooLarge { points, .. }) => assert_eq!(points, n),
        Ok(v) => panic!("an oversized dispatch returned {} values", v.len()),
        Err(e) => panic!("another error: {e}"),
    }
}

#[test]
fn a_ray_over_the_work_budget_ends_undecided_and_is_counted() {
    let f = deep();
    let Some(g) = gpu(&f) else { return };
    let mut os = origins(4_096);
    // rays far outside every sheet's box: misses at any budget, so the "a miss never becomes a hit" rule is checked
    os.extend((0..8).map(|i| Vec3::new(60.0 + i as f32, -60.0, 5.0)));
    let hits = g.march_batch_decided(&os, -Vec3::Z, 10.0).unwrap();
    let n_hit = hits
        .iter()
        .filter(|m| matches!(m, MarchResult::Hit(_)))
        .count();
    assert!(n_hit > 0, "the default budget refused the scene");
    let worst = g.last_march_stats().max_work;
    assert!(worst > 1_000, "the deep stack did little work: {worst}");
    g.set_work_budget(worst / 4)
        .expect("a budget under the maximum");
    let cut = g.march_batch_decided(&os, -Vec3::Z, 10.0).unwrap();
    let st = g.last_march_stats();
    assert!(st.over_budget > 0, "no ray ran out of the budget");
    // a smaller budget can only turn a result into Undecided: a hit stays the same hit, a miss stays a miss (never
    // a hit), and nothing undecided at the full budget becomes decided
    let n_miss = hits.iter().filter(|m| **m == MarchResult::Miss).count();
    assert!(
        n_miss > 0,
        "no miss at the full budget: the miss rule is untested"
    );
    for (m, c) in hits.iter().zip(&cut) {
        assert!(
            *c == *m || *c == MarchResult::Undecided,
            "a ray over the budget became something else: {c:?} (was {m:?})"
        );
    }
    let undecided = cut.iter().filter(|m| **m == MarchResult::Undecided).count() as u64;
    assert_eq!(
        undecided, st.over_budget,
        "undecided rays and the counted ones differ"
    );
}

/// One hitting ray swept over every budget below its work: each budget from `W - min(4096, W - 1)` to `W - 1` (the cell walk of
/// the last floor step and the hit's normal and attribute samples are the final ticks of `W`) and a stride of 97 below
/// that ends undecided and counted over the budget; at `W` the same hit returns. `W` is the ray's work at the full
/// budget. The full-budget hit is returned for the caller's own checks.
fn sweep_budgets(f: &SheetField, o: Vec3, d: Vec3) -> Option<MarchResult> {
    let g = gpu(f)?;
    let o = [o];
    let full = g.march_batch_decided(&o, d, 10.0).unwrap();
    assert!(
        matches!(full[0], MarchResult::Hit(_)),
        "the ray does not hit: {:?}",
        full[0]
    );
    let w = g.last_march_stats().max_work;
    assert!(w > 64, "the ray did little work: {w}");
    // every budget in the last `dense` ticks, a stride below
    let dense = 4_096.min(w - 1);
    let budgets: Vec<u32> = (1..w - dense).step_by(97).chain(w - dense..w).collect();
    let mut compared = 0u32;
    for &b in &budgets {
        g.set_work_budget(b).expect("a budget under the maximum");
        let r = g.march_batch_decided(&o, d, 10.0).unwrap();
        assert_eq!(
            r[0],
            MarchResult::Undecided,
            "budget {b} of {w}: a ray over the budget returned {:?}",
            r[0]
        );
        assert_eq!(
            g.last_march_stats().over_budget,
            1,
            "budget {b}: the ray was not counted"
        );
        compared += 1;
    }
    assert_eq!(compared as usize, budgets.len());
    assert!(
        compared >= dense,
        "{compared} budgets for a dense window of {dense}"
    );
    g.set_work_budget(w).expect("the full work");
    let again = g.march_batch_decided(&o, d, 10.0).unwrap();
    assert_eq!(
        again, full,
        "at its own work the ray no longer returns the same hit"
    );
    assert_eq!(g.last_march_stats().over_budget, 0);
    Some(full[0])
}

/// A ray stopped by the budget at any tick, also inside the cell walk of its last floor step or inside the hit's
/// normal and attribute samples, ends undecided, never a hit: the kernel decides the status after the last tick, and
/// the counter only grows. On the flat stack (see [`sweep_budgets`]).
#[test]
fn a_ray_stopped_anywhere_by_the_budget_is_undecided_never_a_hit() {
    let _ = sweep_budgets(&deep(), Vec3::new(0.31, 0.27, 2.0), -Vec3::Z);
}

/// The same sweep on a curved sheet (a sphere cap of radius 3 about (0, 0, -2)), hit off its axis so the normal is
/// tilted: a hit whose normal samples were cut short would come back with a wrong normal, which the flat stack's
/// constant normal could hide. Oracle for the full-budget normal: the sphere's radial direction at the hit point.
#[test]
fn a_ray_stopped_anywhere_by_the_budget_on_a_curved_sheet_is_undecided_never_a_hit() {
    let c = Vec3::new(0.0, 0.0, -2.0);
    let lv = level_from_fn(
        Vec3::new(-8.0 + 0.013, -8.0 + 0.021, -0.5 + 0.007),
        0.25,
        [8, 8, 1],
        |_c, _e| true,
        |p| {
            let (dx, dy) = (p.x.abs() - 3.0, p.y.abs() - 3.0);
            vec![
                (p - c).length() - 3.0,
                Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0),
            ]
        },
    )
    .unwrap();
    let f = SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1);
    let o = Vec3::new(1.2, 0.4, 2.0);
    let Some(MarchResult::Hit(h)) = sweep_budgets(&f, o, -Vec3::Z) else {
        return;
    };
    let radial = (o - Vec3::Z * h.t - c).normalize();
    assert!(
        radial.x > 0.3,
        "the hit is on the axis, the normal is not tilted: {radial}"
    );
    assert!(
        h.normal.dot(radial) > 0.999,
        "full-budget normal {} vs the sphere's {radial}",
        h.normal
    );
}

/// A point or origin that is not finite reads as on the CPU: a NaN is in no brick on both sides (it used to index
/// brick 0 on the GPU), so the sheet's value is its outside bound. Oracle: the CPU's `eval` and `march_decided` on the
/// same inputs, bit for bit on `f` and `g`.
#[test]
fn a_point_that_is_not_finite_reads_as_on_the_cpu() {
    let f = deep();
    let Some(g) = gpu(&f) else { return };
    let bad = [
        Vec3::new(f32::NAN, 0.0, 0.5),
        Vec3::new(0.0, f32::NAN, 0.5),
        Vec3::new(0.0, 0.0, f32::NAN),
        Vec3::new(f32::INFINITY, 0.0, 0.5),
        Vec3::new(0.0, f32::NEG_INFINITY, 0.5),
        Vec3::splat(f32::NAN),
    ];
    let got = g.eval_batch(&bad).unwrap();
    let mut compared = 0;
    for (p, s) in bad.iter().zip(&got) {
        let (cf, cg, _) = f.eval(*p);
        assert_eq!(
            (s.f.to_bits(), s.g.to_bits()),
            (cf.to_bits(), cg.to_bits()),
            "{p}: GPU ({}, {}) vs CPU ({cf}, {cg})",
            s.f,
            s.g
        );
        compared += 1;
    }
    assert_eq!(compared, bad.len());
    let origins: Vec<Vec3> = bad.iter().map(|p| *p + Vec3::new(0.0, 0.0, 2.0)).collect();
    let marched = g.march_batch_decided(&origins, -Vec3::Z, 10.0).unwrap();
    for (o, m) in origins.iter().zip(&marched) {
        let c = f.march_decided(*o, -Vec3::Z, 10.0).unwrap();
        assert_eq!(*m, c, "origin {o}: GPU {m:?} vs CPU {c:?}");
    }
}

/// The upload checks every sheet's layout before any buffer or device work: a sheet assembled from its public fields
/// with a dropped `s` row (which `sheet` would refuse) is refused as `InvalidFormat`, with or without an adapter.
#[test]
fn the_upload_refuses_a_sheet_whose_layout_does_not_fit() {
    let mut f = deep();
    f.sheets[0].levels[0].channels[0].rows[0] = -1;
    match GpuSheetField::new(&f) {
        Err(GpuError::InvalidFormat(m)) => {
            assert!(m.contains("sheet 0") && m.contains("drops a brick"), "{m}");
        }
        Err(e) => panic!("refused as another error: {e}"),
        Ok(_) => panic!("a sheet with a dropped s row was uploaded"),
    }
}

/// A dispatch's results must fit one storage binding: on a device whose largest binding is 128 MiB the cap is
/// 128 MiB / 64 B = 2,097,152 rays, under the 65,535 x 64 = 4,194,240 the workgroup limit alone allows (a dispatch
/// sized by time alone would grow past the binding and fail the whole batch). Without a binding limit the workgroup
/// limit decides; a binding under one ray still allows one workgroup.
#[test]
fn the_dispatch_cap_is_the_smaller_of_the_workgroup_and_binding_limits() {
    assert_eq!(MARCH_OUT_BYTES_PER_RAY, 64);
    assert_eq!(dispatch_cap(65_535, 128 << 20), (128 << 20) / 64);
    assert_eq!(dispatch_cap(65_535, u64::MAX), 65_535 * 64);
    assert_eq!(dispatch_cap(65_535, 2 << 30), 65_535 * 64);
    assert_eq!(dispatch_cap(65_535, 10), 64);
    assert!(dispatch_cap(65_535, 128 << 20) >= MIN_DISPATCH);
}

/// An empty field is refused at upload, typed and before any device work, rather than failing at dispatch on a
/// binding smaller than one record; the CPU marches it (every ray a miss).
#[test]
fn an_empty_field_is_refused_at_upload_and_misses_on_the_cpu() {
    let f = SheetField::new(Vec::new(), ValidRule::CellV1);
    match GpuSheetField::new(&f) {
        Err(GpuError::InvalidFormat(m)) => assert!(m.contains("no sheets"), "{m}"),
        Err(e) => panic!("refused as another error: {e}"),
        Ok(_) => panic!("an empty field was uploaded"),
    }
    assert_eq!(
        f.march_decided(Vec3::new(0.0, 0.0, 5.0), -Vec3::Z, 10.0)
            .unwrap(),
        MarchResult::Miss
    );
}
