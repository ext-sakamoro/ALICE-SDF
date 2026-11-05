//! The WGSL emit with the node data buffer (`WgslShader::transpile_with_node_data`, what the GPU
//! evaluator runs) for the loft, the smooth section stack and the section stack:
//!
//! - their data is in the buffer: no module-scope array, so a tree whose data exceeds the WGSL array
//!   constructor's limit (32767 elements) still validates in naga, and the GPU gives the CPU's value;
//! - every loop the emit writes has a constant trip count (`for (…; i < <literal>; …)`), so data with
//!   a bad count cannot run a loop without end (a hung GPU);
//! - every read of the buffer is `clamp(base + (i), base, end)` with literal bounds inside the
//!   buffer: an index from bad data reads the node's own values, never past the buffer.
//!
//! The checks read the emitted text; each asserts it found something to check (an empty match would
//! pass vacuously). The GPU comparison skips without an adapter unless `ALICE_SDF_REQUIRE_GPU` is set.
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gpu")]

use alice_sdf::compiled::{TranspileMode, WgslShader};
use alice_sdf::prelude::*;
use glam::Vec2;

/// The WGSL array constructor's element limit
const WGSL_ARRAY_LIMIT: usize = 32767;

fn circle(r: f32, c: Vec2, m: usize) -> Vec<Vec2> {
    (0..m)
        .map(|j| {
            let (s, co) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / m as f32);
            c + Vec2::new(co, s) * r
        })
        .collect()
}

/// A smooth stack whose data block exceeds the array limit: 70 levels of two loops of 120 control
/// points
fn big_smooth_stack() -> SdfNode {
    let levels: Vec<f32> = (0..70).map(|i| -0.35 + 0.01 * i as f32).collect();
    let loops = levels
        .iter()
        .map(|&y| vec![circle(0.12 + 0.05 * y, Vec2::new(-0.15, 0.0), 120), circle(0.1, Vec2::new(0.15, 0.02 * y), 120)])
        .collect();
    SdfNode::smooth_section_stack(levels, loops, 0.01, 0.1).expect("a valid stack")
}

fn section_stack() -> SdfNode {
    let levels = vec![0.0, 0.2, 0.4, 0.6];
    let loops = levels.iter().map(|&y| vec![circle(0.2 + 0.1 * y, Vec2::ZERO, 24), circle(0.05, Vec2::new(0.05, 0.0), 8)]).collect();
    SdfNode::section_stack(levels, loops).expect("a valid section stack")
}

fn lofts() -> SdfNode {
    let ellipse = |a: f32, b: f32| -> Vec<Vec2> { circle(1.0, Vec2::ZERO, 12).iter().map(|p| Vec2::new(a * p.x, b * p.y)).collect() };
    let curves = SdfNode::loft_curves_capped(
        vec![Vec3::ZERO, Vec3::new(0.03, 0.3, 0.0), Vec3::new(0.05, 0.6, 0.02)],
        Vec3::X,
        vec![ellipse(0.2, 0.12), ellipse(0.18, 0.13), ellipse(0.15, 0.1)],
        [[0.1, 0.0, 0.0], [0.15, 0.06, -0.03]],
    )
    .expect("capped curve loft");
    let polar = SdfNode::loft_capped(
        vec![Vec3::ZERO, Vec3::new(0.02, 0.3, 0.0), Vec3::new(0.0, 0.6, 0.03)],
        Vec3::Z,
        1,
        vec![vec![0.15, 0.01, 0.0], vec![0.17, 0.0, 0.01], vec![0.12, 0.0, 0.0]],
        [0.08, 0.12],
    )
    .expect("capped polar loft");
    curves.union(polar.translate(0.5, 0.0, 0.0))
}

fn trees() -> Vec<(&'static str, SdfNode)> {
    vec![
        ("big_smooth_stack", big_smooth_stack()),
        ("section_stack", section_stack()),
        ("lofts", lofts()),
        ("all", big_smooth_stack().union(section_stack().translate(0.8, 0.0, 0.0)).union(lofts().translate(0.0, 0.0, 0.8))),
    ]
}

/// Every module-scope array's declared length
fn const_array_lengths(src: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for line in src.lines() {
        let l = line.trim_start();
        if !(l.starts_with("const ") || l.starts_with("var<private>")) {
            continue;
        }
        if let Some(at) = l.find("array<f32, ").or_else(|| l.find("array<i32, ")) {
            let rest = &l[at + 11..];
            let n: String = rest.chars().take_while(char::is_ascii_digit).collect();
            out.push(n.parse().expect("an array length"));
        }
    }
    out
}

/// The condition of every `for (init; cond; step)` header
fn loop_conditions(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = src;
    while let Some(at) = rest.find("for (") {
        let tail = &rest[at + 5..];
        let close = tail.find(')').expect("a loop header");
        let header = &tail[..close];
        let parts: Vec<&str> = header.split(';').collect();
        assert_eq!(parts.len(), 3, "a loop header with three parts: {header}");
        out.push(parts[1].trim().to_string());
        rest = &tail[close..];
    }
    out
}

/// A condition `<var> < <integer literal>` or `<var> <= <integer literal>` (optional `i` / `u`)
fn literal_bound(cond: &str) -> bool {
    let rhs = if let Some((_, r)) = cond.split_once("<=") { r } else if let Some((_, r)) = cond.split_once('<') { r } else { return false };
    let r = rhs.trim().trim_end_matches(['i', 'u']);
    !r.is_empty() && r.chars().all(|c| c.is_ascii_digit())
}

/// Every buffer read as `(lo, hi)` of its `clamp(…, lo, hi)`, or `None` for a read that is not clamped
fn buffer_reads(src: &str) -> Vec<Option<(usize, usize)>> {
    let mut out = Vec::new();
    let mut rest = src;
    while let Some(at) = rest.find("alice_node_data[") {
        let tail = &rest[at + 16..];
        if !tail.starts_with("clamp(") {
            out.push(None);
            rest = tail;
            continue;
        }
        // the clamp's last two arguments, at its own nesting level
        let body = &tail[6..];
        let (mut depth, mut end) = (1usize, 0usize);
        for (i, c) in body.char_indices() {
            match c {
                '(' | '[' => depth += 1,
                ')' | ']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let args = &body[..end];
        let mut top = Vec::new();
        let (mut d, mut start) = (0usize, 0usize);
        for (i, c) in args.char_indices() {
            match c {
                '(' | '[' => d += 1,
                ')' | ']' => d -= 1,
                ',' if d == 0 => {
                    top.push(args[start..i].trim().to_string());
                    start = i + 1;
                }
                _ => {}
            }
        }
        top.push(args[start..].trim().to_string());
        let num = |s: &str| s.trim_end_matches(['i', 'u']).parse::<usize>().ok();
        out.push(match top.as_slice() {
            [_, lo, hi] => num(lo).zip(num(hi)),
            _ => None,
        });
        rest = &body[end..];
    }
    out
}

#[test]
fn the_big_stack_needs_more_than_the_array_limit() {
    // precondition: the constant path would declare an array above the limit
    let plain = WgslShader::transpile(&big_smooth_stack(), TranspileMode::Hardcoded);
    let longest = const_array_lengths(&plain.source).into_iter().max().unwrap_or(0);
    assert!(longest > WGSL_ARRAY_LIMIT, "the big stack's array has {longest} elements");
}

#[test]
fn loft_and_stack_data_is_in_the_buffer_and_validates() {
    for (what, node) in trees() {
        let sh = WgslShader::transpile_with_node_data(&node, TranspileMode::Hardcoded);
        assert!(!sh.node_data.is_empty(), "{what}: no node data");
        let longest = const_array_lengths(&sh.source).into_iter().max().unwrap_or(0);
        assert!(longest <= WGSL_ARRAY_LIMIT, "{what}: a module-scope array of {longest} elements");
        for prefix in ["alice_loft_d_", "alice_sss_d_", "alice_stack_"] {
            assert!(
                !sh.source.lines().any(|l| l.trim_start().starts_with(&format!("const {prefix}"))),
                "{what}: a module-scope {prefix} array with the node data buffer"
            );
        }
        let module = naga::front::wgsl::parse_str(&sh.source).unwrap_or_else(|e| panic!("{what}: {}", e.emit_to_string(&sh.source)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
    }
}

#[test]
fn every_emitted_loop_has_a_constant_trip_count() {
    let mut checked = 0usize;
    for (what, node) in trees() {
        let sh = WgslShader::transpile_with_node_data(&node, TranspileMode::Hardcoded);
        for cond in loop_conditions(&sh.source) {
            assert!(literal_bound(&cond), "{what}: a loop bounded by `{cond}`, not a literal");
            checked += 1;
        }
    }
    assert!(checked >= 8, "only {checked} loops checked");
}

#[test]
fn every_buffer_read_is_clamped_inside_the_buffer() {
    let mut checked = 0usize;
    for (what, node) in trees() {
        let sh = WgslShader::transpile_with_node_data(&node, TranspileMode::Hardcoded);
        for r in buffer_reads(&sh.source) {
            let (lo, hi) = r.unwrap_or_else(|| panic!("{what}: a read of alice_node_data that is not clamped by literals"));
            assert!(lo <= hi && hi < sh.node_data.len(), "{what}: a read clamped to {lo}..={hi} of {}", sh.node_data.len());
            checked += 1;
        }
    }
    assert!(checked >= 20, "only {checked} reads checked");
}

#[test]
fn the_gpu_evaluates_the_buffer_trees_as_the_cpu() {
    use alice_sdf::compiled::GpuEvaluator;
    let mut s: u64 = 0xDA7A_B0B0;
    let mut r = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    };
    for (what, node) in trees() {
        let gpu = match GpuEvaluator::new(&node) {
            Ok(g) => g,
            Err(e) => {
                assert!(std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(), "{what}: no GPU: {e}");
                eprintln!("skipping {what}: no GPU ({e})");
                return;
            }
        };
        let pts: Vec<Vec3> = (0..2048).map(|_| Vec3::new(r() * 1.6 - 0.5, r() * 1.2 - 0.4, r() * 1.6 - 0.5)).collect();
        let got = gpu.eval_batch(&pts).expect("gpu eval");
        assert_eq!(got.len(), pts.len());
        let mut worst = (0.0f32, Vec3::ZERO);
        for (p, g) in pts.iter().zip(&got) {
            let c = eval(&node, *p);
            let e = (c - g).abs() / c.abs().max(1.0);
            assert!(e.is_finite(), "{what}: non-finite at {p:?}: cpu {c}, gpu {g}");
            if e > worst.0 {
                worst = (e, *p);
            }
        }
        assert!(worst.0 <= 2e-5, "{what}: GPU differs from the CPU by {} at {:?}", worst.0, worst.1);
    }
}

#[test]
fn no_checked_read_leaves_its_proven_range_on_the_gpu() {
    use alice_sdf::compiled::GpuEvaluator;
    let mut s: u64 = 0x5EAD_C0DE;
    let mut r = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    };
    for (what, node) in trees() {
        let sh = WgslShader::try_transpile_with_node_data_checked(&node, TranspileMode::Hardcoded).expect("the emit proves its reads");
        let checked = sh.source.matches("alice_nd(").count();
        assert!(checked > 20, "{what}: only {checked} checked reads in the text");
        let gpu = match GpuEvaluator::new_checked(&node) {
            Ok(g) => g,
            Err(e) => {
                assert!(std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(), "{what}: no GPU: {e}");
                eprintln!("skipping {what}: no GPU ({e})");
                return;
            }
        };
        let pts: Vec<Vec3> = (0..2048).map(|_| Vec3::new(r() * 3.0 - 1.0, r() * 2.4 - 0.8, r() * 3.0 - 1.0)).collect();
        let got = gpu.eval_batch(&pts).expect("gpu eval");
        let plain = GpuEvaluator::new(&node).expect("gpu").eval_batch(&pts).expect("gpu eval");
        assert!(got.iter().zip(&plain).all(|(a, b)| a.to_bits() == b.to_bits()), "{what}: the checked shader's values differ");
        assert_eq!(gpu.read_hits().expect("the counter"), Some(0), "{what}: reads left their proven range on the GPU");
    }
}
