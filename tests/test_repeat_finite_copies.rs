//! `repeat_finite` places whole copies of its child on the lattice of its spacing, on every path.
//!
//! A small sphere repeated with spacing 10 is sampled along each axis (the other two coordinates
//! 0) in steps of 1/40 of the sphere's diameter; each run of samples inside the shape is a copy
//! seen along that axis. Every copy must be a whole child (its inside run as wide as the sphere's
//! diameter) centred on a multiple of the spacing, and the copies must be symmetric about the
//! origin. The tree evaluator, the compiled bytecode (scalar and SIMD batch) and the JIT (feature
//! `jit`) are each checked against this, not against each other.
//!
//! Even counts are the control (they place whole copies today). Odd counts are the case under
//! test: clamping the cell index to ±count/2 there makes the outermost cells half-integers, so the
//! child appears cut in half at ±1.5 spacing (count 3) or ±0.5 spacing (count 1).
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::{eval_compiled, eval_compiled_batch_simd, CompiledSdf};
use alice_sdf::eval::eval;
use alice_sdf::types::SdfNode;
use glam::Vec3;

const SPACING: f32 = 10.0;
const RADIUS: f32 = 0.5;
const STEP: f32 = 2.0 * RADIUS / 40.0;

fn repeated(count: u32) -> SdfNode {
    SdfNode::sphere(RADIUS).repeat_finite([count; 3], Vec3::splat(SPACING))
}

/// the sample points along `axis` (0 = x) over the whole extent the count could reach, and beyond
fn line(axis: usize, count: u32) -> Vec<Vec3> {
    let reach = (count as f32 + 2.0) * SPACING;
    let n = (2.0 * reach / STEP) as i32;
    (0..=n)
        .map(|i| {
            let mut p = Vec3::ZERO;
            p[axis] = -reach + i as f32 * STEP;
            p
        })
        .collect()
}

/// the inside runs of `values` along `points` as (start, end) coordinates on `axis`
fn runs(points: &[Vec3], values: &[f32], axis: usize) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    let mut start: Option<f32> = None;
    for (p, &d) in points.iter().zip(values) {
        match (d < 0.0, start) {
            (true, None) => start = Some(p[axis]),
            (false, Some(s)) => {
                out.push((s, p[axis] - STEP));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, points.last().map_or(s, |p| p[axis])));
    }
    out
}

/// every run is a whole sphere centred on a multiple of the spacing, and the runs are symmetric
fn check(path: &str, count: u32, axis: usize, runs: &[(f32, f32)]) -> Result<usize, String> {
    if runs.is_empty() {
        return Err(format!("{path} count {count} axis {axis}: no copy at all"));
    }
    for &(a, b) in runs {
        let width = b - a + STEP;
        let centre = (a + b) / 2.0;
        let lattice = (centre / SPACING).round() * SPACING;
        if (width - 2.0 * RADIUS).abs() > 2.0 * STEP || (centre - lattice).abs() > STEP {
            return Err(format!(
                "{path} count {count} axis {axis}: a copy over [{a:.3}, {b:.3}] (width {width:.3}, \
                 centre {centre:.3}) is not a whole child on the lattice of {SPACING}; copies: {runs:?}"
            ));
        }
    }
    let centres: Vec<f32> = runs.iter().map(|&(a, b)| (a + b) / 2.0).collect();
    let mirrored: Vec<f32> = centres.iter().rev().map(|c| -c).collect();
    if centres
        .iter()
        .zip(&mirrored)
        .any(|(c, m)| (c - m).abs() > STEP)
    {
        return Err(format!(
            "{path} count {count} axis {axis}: copies are not symmetric about the origin: {centres:?}"
        ));
    }
    Ok(runs.len())
}

fn check_all(path: &str, counts: &[u32], eval_line: impl Fn(&SdfNode, &[Vec3]) -> Vec<f32>) {
    let mut failures = Vec::new();
    let mut compared = 0;
    for &count in counts {
        let node = repeated(count);
        for axis in 0..3 {
            let points = line(axis, count);
            let values = eval_line(&node, &points);
            match check(path, count, axis, &runs(&points, &values, axis)) {
                Ok(_) => compared += 1,
                Err(e) => failures.push(e),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(compared, counts.len() * 3, "{path}: every count and axis compared");
}

const EVEN: [u32; 3] = [2, 4, 6];
const ODD: [u32; 4] = [1, 3, 5, 7];

fn tree(node: &SdfNode, points: &[Vec3]) -> Vec<f32> {
    points.iter().map(|&p| eval(node, p)).collect()
}

fn compiled(node: &SdfNode, points: &[Vec3]) -> Vec<f32> {
    let c = CompiledSdf::compile(node);
    points.iter().map(|&p| eval_compiled(&c, p)).collect()
}

fn simd(node: &SdfNode, points: &[Vec3]) -> Vec<f32> {
    eval_compiled_batch_simd(&CompiledSdf::compile(node), points)
}

#[test]
fn even_counts_place_whole_copies_on_the_lattice_on_every_path() {
    check_all("tree", &EVEN, tree);
    check_all("compiled", &EVEN, compiled);
    check_all("simd", &EVEN, simd);
}

#[test]
fn odd_counts_place_whole_copies_on_the_lattice_in_the_tree_evaluator() {
    check_all("tree", &ODD, tree);
}

#[test]
fn odd_counts_place_whole_copies_on_the_lattice_in_the_compiled_bytecode() {
    check_all("compiled", &ODD, compiled);
    check_all("simd", &ODD, simd);
}

#[cfg(feature = "jit")]
#[test]
fn odd_and_even_counts_place_whole_copies_on_the_lattice_in_the_jit() {
    use alice_sdf::compiled::jit::JitCompiledSdf;
    let jit = |node: &SdfNode, points: &[Vec3]| {
        JitCompiledSdf::compile(node)
            .expect("repeat_finite of a sphere is JIT-supported")
            .eval_batch(points)
    };
    check_all("jit", &EVEN, jit);
    check_all("jit", &ODD, jit);
}
