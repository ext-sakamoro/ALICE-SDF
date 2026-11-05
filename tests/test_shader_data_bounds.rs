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

mod common;

use alice_sdf::compiled::{TranspileMode, WgslShader};
use common::loft_stack_scenes::{big_smooth_stack, contraction_budget, trees, within_contraction_budget};
use alice_sdf::prelude::*;

/// The read sites the emit writes per scene (the smooth stack's, the section stack's, the two lofts',
/// and all of them in one tree)
const SITES: &[(&str, usize)] = &[("big_smooth_stack", 13), ("section_stack", 14), ("lofts", 89), ("all", 116)];

/// The WGSL array constructor's element limit
const WGSL_ARRAY_LIMIT: usize = 32767;

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
        // the compute shader declares the node data binding the module's functions read
        let text = sh.to_compute_shader();
        let module = naga::front::wgsl::parse_str(&text).unwrap_or_else(|e| panic!("{what}: {}", e.emit_to_string(&text)));
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
    let mut gaps = Vec::new();
    let mut s: u64 = 0x5EAD_C0DE;
    let mut r = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    };
    for (what, node) in trees() {
        let sh = WgslShader::try_transpile_with_node_data_checked(&node, TranspileMode::Hardcoded).expect("the emit proves its reads");
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
        assert_eq!(gpu.read_hits().expect("the counter"), Some(0), "{what}: reads left their proven range on the GPU");
        // the values: the CPU's within the parity tolerance; bit for bit with the plain shader on the
        // Metal engine (fast math off). Through wgpu the compiler may contract the two texts
        // differently, so there the plain shader is compared within the tolerance too
        let plain = GpuEvaluator::new(&node).expect("gpu").eval_batch(&pts).expect("gpu eval");
        // KNOWN-DEFECT-ARM: known defect: g3-vulkan-nocontraction: a fusing wgpu driver (Vulkan, Metal through wgpu) is tolerated within the MEASURED contraction budget (C++ -ffp-contract=fast vs off, x4 chosen, not derived) until no wgpu backend contracts
        // the arm, only while the driver fuses: two shader texts (checked and plain) may then be
        // contracted differently; on a driver that does not fuse (the Metal engine, fast math off)
        // the two are bit for bit equal
        let fuses = gpu_fuses_a_multiply_add() == Some(true);
        let budget = contraction_budget(what);
        let (mut differed, mut gap) = (0usize, 0.0f32);
        let mut over = None;
        for ((p, g), q) in pts.iter().zip(&got).zip(&plain) {
            let c = eval(&node, *p);
            assert!((c - g).abs() <= 2e-5 * c.abs().max(1.0), "{what}: checked GPU {g} vs CPU {c} at {p:?}");
            if fuses && g.to_bits() != q.to_bits() {
                differed += 1;
                gap = gap.max((g - q).abs() / c.abs().max(1.0));
                if !within_contraction_budget(*g, *q, c, budget) && over.is_none() {
                    over = Some((*g, *q, *p));
                }
            } else {
                assert_eq!(g.to_bits(), q.to_bits(), "{what}: the checked shader's value differs at {p:?}");
            }
        }
        // the real GPU's gap beside the budget measured on the C++ proxy
        println!("{what}: {differed} of {} values differ, largest relative gap {gap:e}, budget {budget:e} (fusing driver: {fuses})", pts.len());
        gaps.push((what, over, budget));
    }
    for (what, over, budget) in gaps {
        if let Some((g, q, p)) = over {
            panic!("{what}: checked {g} vs plain {q} at {p:?}, past the budget {budget:e}");
        }
    }
}

/// Whether the GPU behind `GpuEvaluator` rounds `a * b + c` once (fused): with a = b = 1 + 2⁻¹²
/// and c = −(1 + 2⁻¹¹) two roundings give 0 and one gives 2⁻²⁴ (the discriminator of
/// `tests/test_gpu_fma_contraction.rs`). `None` without a GPU
fn gpu_fuses_a_multiply_add() -> Option<bool> {
    use alice_sdf::compiled::GpuEvaluator;
    let wgsl = "struct InputPoint { x: f32, y: f32, z: f32, _pad: f32, }
struct OutputDistance { distance: f32, _pad1: f32, _pad2: f32, _pad3: f32, }
@group(0) @binding(0) var<storage, read> input_points: array<InputPoint>;
@group(0) @binding(1) var<storage, read_write> output_distances: array<OutputDistance>;
@group(0) @binding(2) var<uniform> point_count: u32;
fn sdf_eval(p: vec3<f32>) -> f32 { return p.x * p.y + p.z; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= point_count) { return; }
    let point = input_points[idx];
    output_distances[idx].distance = sdf_eval(vec3<f32>(point.x, point.y, point.z));
}
";
    let e = GpuEvaluator::from_wgsl(wgsl).ok()?.into_parts().0;
    let a = 1.0 + 1.0 / 4096.0;
    let got = e.eval_batch(&[Vec3::new(a, a, -(1.0 + 1.0 / 2048.0))]).ok()?[0];
    Some(got.to_bits() != 0.0f32.to_bits())
}

/// Every read site, counted in both forms and pinned per scene: the checked reads (`alice_nd(`, the
/// helper's own definition not counted) equal the clamped reads of the plain buffer emit, and both
/// equal the pinned count, so a site the emit drops or writes unchecked changes a count
#[test]
fn every_read_site_is_checked_and_clamped() {
    let mut counts = Vec::new();
    for (what, node) in trees() {
        let checked_src = WgslShader::try_transpile_with_node_data_checked(&node, TranspileMode::Hardcoded).expect("the emit proves its reads").source;
        let checked = checked_src.matches("alice_nd(").count() - checked_src.matches("fn alice_nd(").count();
        let clamped = WgslShader::try_transpile_with_node_data(&node, TranspileMode::Hardcoded).expect("the emit proves its reads").source.matches("alice_node_data[clamp(").count();
        println!("{what}: {checked} checked, {clamped} clamped read sites");
        counts.push((what, checked, clamped));
    }
    for (what, checked, clamped) in counts {
        assert_eq!(checked, clamped, "{what}: {checked} checked reads but {clamped} clamped reads");
        let expected = SITES.iter().find(|(n, _)| *n == what).map(|(_, c)| *c);
        assert_eq!(Some(checked), expected, "{what}: {checked} read sites (pinned {expected:?})");
    }
}

/// The arm's check has teeth: a gap of ten times the budget is refused, a gap at the budget passes
#[test]
fn the_contraction_arm_refuses_ten_times_its_budget() {
    let mut checked = 0;
    for (what, _) in trees() {
        let b = contraction_budget(what);
        for cpu in [0.25f32, 3.0] {
            let x = 0.5f32;
            let scale = cpu.abs().max(1.0);
            assert!(within_contraction_budget(x, x + 0.99 * b * scale, cpu, b), "{what}: a gap inside the budget is refused");
            assert!(!within_contraction_budget(x, x + 10.0 * b * scale + f32::MIN_POSITIVE, cpu, b), "{what}: ten times the budget passes");
            checked += 1;
        }
    }
    assert_eq!(checked, 8);
}
