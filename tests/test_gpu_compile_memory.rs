//! The memory a driver takes to build a tree's pipelines and run the first dispatch does not grow with
//! the number of nodes of one kind: one call site per kind, in a loop over the node data buffer.
//!
//! Why: a driver that compiles in the process (llvmpipe JIT-compiles at the first dispatch) takes
//! memory that follows the emitted call sites and data, not the tree's values. One copy of a loft's
//! functions per loft took more than 6 GB on llvmpipe for two curve lofts, more than a hosted runner
//! can spare, and a module-scope data array read with run-time indices took more than 12 GB for one.
//!
//! The quantity: the resident memory of a fresh process after it built the evaluator and ran the
//! first dispatch, less the same for a sphere (the driver's own share). Each case runs in a child
//! process of this test binary, so no case sees what another left allocated. It is measured where the
//! compile is in the process (`/proc`, Linux); elsewhere the driver may compile in another process,
//! and the test says so and measures nothing (`ALICE_SDF_REQUIRE_COMPILE_MEMORY` makes that a
//! failure, as the lavapipe job sets it).
//!
//! The teeth: the per-node form (Dynamic mode, every polygon vertex its own emitted read) grows with
//! the polygons; the test asserts it does, so a bound that the looped form meets is not met by
//! accident.
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gpu")]

use alice_sdf::compiled::{GpuEvaluator, TranspileMode, WgslShader};
use alice_sdf::prelude::*;
use glam::Vec2;
use std::process::Command;

/// The growth bound: the excess of 16 nodes over 4 nodes' (a placeholder, as the pipeline-time bound
/// in tests/test_one_call_site.rs, until every kind is looped)
const GROWTH: f64 = 2.0;

/// Resident memory below this is the allocator's noise, not a compile's (the excesses are compared
/// with at least this floor, so 1 MiB against 3 MiB is not a growth)
const FLOOR_MIB: f64 = 16.0;

const CASE_ENV: &str = "ALICE_SDF_COMPILE_MEMORY_CASE";

fn star(n: usize, r: f32, dr: f32) -> Vec<Vec2> {
    (0..n)
        .map(|j| {
            let (s, c) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / n as f32);
            let rr = if j % 2 == 0 { r } else { r - dr };
            Vec2::new(rr * c, rr * s)
        })
        .collect()
}

fn ring(m: usize, r: f32) -> Vec<Vec2> {
    (0..m)
        .map(|j| {
            let (s, c) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / m as f32);
            Vec2::new(r * c, r * s)
        })
        .collect()
}

/// A balanced union, as a body's parts are
fn union_of(mut level: Vec<SdfNode>) -> SdfNode {
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut it = level.into_iter();
        while let Some(a) = it.next() {
            next.push(match it.next() {
                Some(b) => a.union(b),
                None => a,
            });
        }
        level = next;
    }
    level.pop().expect("a part")
}

fn polygons(n: usize) -> SdfNode {
    union_of(
        (0..n as u32)
            .map(|k| {
                let (i, j) = ((k % 8) as f32, (k / 8) as f32);
                SdfNode::polygon_2d(star(46, 0.04 + 0.002 * (k % 7) as f32, 0.02), 0.03).translate(
                    i * 0.1,
                    j * 0.1,
                    0.0,
                )
            })
            .collect(),
    )
}

fn curve_lofts(n: usize) -> SdfNode {
    union_of(
        (0..n)
            .map(|i| {
                let x = 0.3 * i as f32;
                let axis: Vec<Vec3> = (0..4).map(|r| Vec3::new(x, 0.3 * r as f32, 0.0)).collect();
                let rows = axis.iter().map(|_| ring(12, 0.2)).collect();
                SdfNode::loft_curves(axis, Vec3::X, rows).expect("a curve loft")
            })
            .collect(),
    )
}

fn tree(kind: &str, n: usize) -> SdfNode {
    match kind {
        "sphere" => SdfNode::sphere(0.2),
        "polygons" => polygons(n),
        "curve-lofts" => curve_lofts(n),
        _ => panic!("unknown kind {kind}"),
    }
}

/// The evaluator of `node` in `form`: `looped` = the node data buffer form (one call site per looped
/// kind), `per-node` = Dynamic mode (one emitted copy per node)
fn build(form: &str, node: &SdfNode) -> GpuEvaluator {
    match form {
        "looped" => {
            let s = WgslShader::try_transpile_with_node_data(node, TranspileMode::Hardcoded)
                .expect("transpile");
            GpuEvaluator::from_shader(&s).expect("evaluator")
        }
        "per-node" => GpuEvaluator::new_dynamic(node).expect("evaluator"),
        _ => panic!("unknown form {form}"),
    }
}

/// The child's side: build the case named in the environment, run the first dispatch, print the peak
/// resident memory (`VmHWM`) and the number of values it read back
#[test]
fn compile_memory_child() {
    let Ok(case) = std::env::var(CASE_ENV) else {
        return;
    };
    let mut it = case.split(':');
    let (form, kind, n) = (
        it.next().expect("form"),
        it.next().expect("kind"),
        it.next().expect("n").parse::<usize>().expect("n"),
    );
    let g = build(form, &tree(kind, n));
    let v = g
        .eval_batch(&[Vec3::ZERO, Vec3::splat(0.1)])
        .expect("first dispatch");
    let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    let kib: f64 = status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        .expect("VmHWM");
    println!(
        "compile-memory {case} {:.1} MiB {} values",
        kib / 1024.0,
        v.len()
    );
}

/// The peak resident memory of a fresh process that built `form` for `n` nodes of `kind`, in MiB
fn measure(form: &str, kind: &str, n: usize) -> f64 {
    let case = format!("{form}:{kind}:{n}");
    // named before the child starts, so a job killed over its memory budget names the case
    eprintln!("compile-memory case {case}: starting");
    let out = Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "compile_memory_child",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CASE_ENV, &case)
        .output()
        .expect("child");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{case}: the child failed\n{text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // libtest prints the child's line after its own `test compile_memory_child ... ` on one line
    let tag = format!("compile-memory {case} ");
    let line = text
        .lines()
        .find_map(|l| l.find(&tag).map(|at| &l[at..]))
        .unwrap_or_else(|| panic!("{case}: no measurement in\n{text}"));
    let mut w = line.split_whitespace().skip(2);
    let mib: f64 = w.next().and_then(|x| x.parse().ok()).expect("MiB");
    assert_eq!(
        w.nth(1),
        Some("2"),
        "{case}: the first dispatch read no values"
    );
    mib
}

/// Whether this machine can measure: a GPU (or an error when one is required) and a driver whose
/// compile is in the process
fn measurable() -> bool {
    if let Err(e) = GpuEvaluator::new(&SdfNode::sphere(0.2)) {
        assert!(
            std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
            "ALICE_SDF_REQUIRE_GPU is set but the GPU evaluator failed: {e}"
        );
        eprintln!("skipping: {e}");
        return false;
    }
    if !std::path::Path::new("/proc/self/status").exists() {
        assert!(
            std::env::var_os("ALICE_SDF_REQUIRE_COMPILE_MEMORY").is_none(),
            "ALICE_SDF_REQUIRE_COMPILE_MEMORY is set but there is no /proc to measure with"
        );
        eprintln!("skipping: no /proc here, the driver may compile in another process");
        return false;
    }
    true
}

/// The excesses over a sphere's for `small` and `large` nodes of `kind` in `form`
fn excesses(form: &str, kind: &str, small: usize, large: usize) -> (f64, f64) {
    let base = measure(form, "sphere", 1);
    let a = measure(form, kind, small) - base;
    let b = measure(form, kind, large) - base;
    eprintln!(
        "{form} {kind}: sphere {base:.0} MiB, {small} nodes +{a:.0} MiB, {large} nodes +{b:.0} MiB"
    );
    assert!(base > 0.0, "{form} {kind}: measured nothing");
    (a, b)
}

fn grows(a: f64, b: f64) -> bool {
    b.max(FLOOR_MIB) > GROWTH * a.max(FLOOR_MIB)
}

#[test]
fn the_compile_memory_does_not_grow_with_the_polygons() {
    if !measurable() {
        return;
    }
    let (a, b) = excesses("looped", "polygons", 4, 16);
    assert!(
        !grows(a, b),
        "looped polygons: +{a:.0} MiB -> +{b:.0} MiB from 4 to 16"
    );
    // the teeth: the same polygons, one emitted copy each, grow (2 to 8, smaller than the looped
    // sizes: 16 per-node polygons alone took about 3 GB on a real driver)
    let (pa, pb) = excesses("per-node", "polygons", 2, 8);
    assert!(
        grows(pa, pb),
        "per-node polygons: +{pa:.0} MiB -> +{pb:.0} MiB from 2 to 8, the measure does not see a \
         growing compile"
    );
}

#[test]
#[ignore = "known defect: B1 lofts/stacks one-call-site pending (sdf b1 one call site)"]
fn the_compile_memory_does_not_grow_with_the_lofts() {
    if !measurable() {
        return;
    }
    // two curve lofts in one copy each took more than 6 GB on llvmpipe (one took 317 MiB)
    let (a, b) = excesses("looped", "curve-lofts", 1, 2);
    assert!(
        !grows(a, b),
        "looped curve lofts: +{a:.0} MiB -> +{b:.0} MiB from 1 to 2"
    );
}

/// The memory budget of the job that runs this test (scripts/peak_rss.py --max-mib in the lavapipe
/// job and in preflight)
const JOB_BUDGET_MIB: f64 = 3072.0;

#[test]
fn a_dynamic_loft_compiles_within_the_job_budget() {
    if !measurable() {
        return;
    }
    // Dynamic mode with the loft's rows in a module-scope array read at run-time indices took more
    // than 12 GB on llvmpipe for one curve loft; the structural cause is pinned by
    // `the_dynamic_evaluator_shader_declares_no_constant_data_array`, this is the memory it leaves
    let peak = measure("per-node", "curve-lofts", 1);
    let looped = measure("looped", "curve-lofts", 1);
    eprintln!(
        "dynamic curve loft: {peak:.0} MiB, node data form {looped:.0} MiB (peak, not excess)"
    );
    assert!(
        peak < JOB_BUDGET_MIB,
        "a Dynamic curve loft peaks at {peak:.0} MiB, over the job budget {JOB_BUDGET_MIB:.0} MiB"
    );
}
