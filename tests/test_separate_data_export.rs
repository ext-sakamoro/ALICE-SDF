//! The separate-data shader export (`compiled::data_export`): the text reads the per-node data
//! arrays from one R32F texture instead of holding them as constants.
//!
//! Oracles: (1) a tree without data arrays exports the single-file text unchanged; (2) the texture
//! holds the very values the single-file text holds as constants, in order; (3) for a large shape
//! the text is a small fraction of the single-file text; (4) on the GPU the separate-data shader
//! with its texture bound gives the same bits as the single-file shader at every point (the same
//! arithmetic, only the place the data is read from differs), for WGSL and for GLSL (its Vulkan
//! compute form); (5) the GLSL and HLSL exports declare the texture and read it with exact texel
//! fetches (HLSL is checked as text here and by DXC in its own workflow, not executed); (6) a tree
//! whose data does not fit one texture is refused with a typed error.

#![cfg(feature = "gpu")]

use alice_sdf::compiled::data_export::{
    DataExportError, DATA_TEXTURE_NAME, DATA_TEXTURE_WIDTH, MAX_DATA_TEXTURE_ROWS,
};
use alice_sdf::compiled::{TranspileMode, WgslShader};

mod common;
#[path = "common/gpu_batches.rs"]
mod gpu_batches;
use alice_sdf::prelude::*;
use glam::Vec2;

/// A closed outline of `n` vertices on a wobbled circle
fn outline(n: usize) -> Vec<Vec2> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32 * std::f32::consts::TAU;
            let r = 0.6 + 0.1 * alice_det_math::sin(5.0 * t);
            let (s, c) = alice_det_math::sin_cos(t);
            Vec2::new(r * c, r * s)
        })
        .collect()
}

/// A shape whose data (24,000 values) is past the single-file text's per-array limit (the text
/// splits it into several constant arrays)
fn large_shape() -> SdfNode {
    SdfNode::polygon_2d(outline(12_000), 0.25)
}

fn gpu_or_skip<T>(what: &str, r: Result<T, alice_sdf::compiled::GpuError>) -> Option<T> {
    match r {
        Ok(v) => Some(v),
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "{what}: ALICE_SDF_REQUIRE_GPU is set but the GPU path failed: {e}"
            );
            eprintln!("skipping {what}: {e}");
            None
        }
    }
}

#[test]
fn a_shape_without_data_exports_the_single_file_text() {
    let node = SdfNode::sphere(0.5).union(SdfNode::box3d(0.3, 0.2, 0.4).translate(0.6, 0.0, 0.0));
    let export = WgslShader::export_with_data_texture(&node).expect("export");
    assert!(export.data.is_none());
    assert_eq!(
        export.source,
        WgslShader::transpile(&node, TranspileMode::Hardcoded).source
    );
}

#[test]
fn the_texture_holds_the_constants_of_the_single_file_text_in_order() {
    let pts = outline(50);
    let node = SdfNode::polygon_2d(pts.clone(), 0.25);
    let export = WgslShader::export_with_data_texture(&node).expect("export");
    let data = export.data.expect("a polygon outline is a data array");
    assert_eq!(data.width, DATA_TEXTURE_WIDTH);
    let flat: Vec<f32> = pts.iter().flat_map(|p| [p.x, p.y]).collect();
    assert_eq!(data.len, flat.len());
    for (i, v) in flat.iter().enumerate() {
        assert_eq!(
            data.value(i).map(f32::to_bits),
            Some(v.to_bits()),
            "value {i}"
        );
    }
    assert!(
        export.source.contains(DATA_TEXTURE_NAME),
        "the text binds the texture"
    );
    assert!(
        !export.source.contains("alice_poly_v_"),
        "no outline constants in the text"
    );
}

#[test]
fn a_large_shapes_text_is_small() {
    let node = large_shape();
    let single = WgslShader::transpile(&node, TranspileMode::Hardcoded).source;
    let export = WgslShader::export_with_data_texture(&node).expect("export");
    assert_eq!(export.data.as_ref().map(|d| d.len), Some(24_000));
    assert!(
        export.source.len() * 20 < single.len(),
        "separate-data text {} bytes vs single-file {} bytes",
        export.source.len(),
        single.len()
    );
}

#[test]
fn separate_data_and_single_file_agree_bit_for_bit_on_the_gpu() {
    use alice_sdf::compiled::GpuEvaluator;
    let node = large_shape();
    let single = WgslShader::transpile(&node, TranspileMode::Hardcoded);
    let export = WgslShader::export_with_data_texture(&node).expect("export");
    let Some(a) = gpu_or_skip("single-file", GpuEvaluator::from_shader(&single)) else {
        return;
    };
    let Some(b) = gpu_or_skip("separate-data", GpuEvaluator::from_data_export(&export)) else {
        return;
    };
    // a grid across the outline (inside, on, outside) at three heights
    let mut pts = Vec::new();
    for j in 0..48 {
        for i in 0..48 {
            for z in [-0.3f32, 0.0, 0.2] {
                pts.push(glam::Vec3::new(
                    i as f32 / 47.0 * 1.8 - 0.9,
                    j as f32 / 47.0 * 1.8 - 0.9,
                    z,
                ));
            }
        }
    }
    let da = gpu_batches::eval_in_batches(&a, &pts).expect("dispatch");
    let db = gpu_batches::eval_in_batches(&b, &pts).expect("dispatch");
    assert_eq!(da.len(), pts.len());
    let differ: Vec<usize> = (0..pts.len())
        .filter(|&k| da[k].to_bits() != db[k].to_bits())
        .collect();
    assert!(
        differ.is_empty(),
        "{} of {} points differ, first {:?}",
        differ.len(),
        pts.len(),
        differ.first()
    );
}

#[cfg(feature = "glsl")]
#[test]
fn the_glsl_export_reads_the_texture_with_texel_fetches() {
    use alice_sdf::compiled::glsl::GlslShader;
    let export = GlslShader::export_with_data_texture(&large_shape()).expect("export");
    assert!(export.data.is_some());
    assert!(export
        .source
        .contains(&format!("uniform highp sampler2D {DATA_TEXTURE_NAME}")));
    assert!(
        export.source.contains("texelFetch("),
        "exact fetch, no filtering"
    );
    assert!(!export.source.contains("alice_poly_v_"));
}

#[cfg(feature = "hlsl")]
#[test]
fn the_hlsl_export_reads_the_texture_with_loads() {
    use alice_sdf::compiled::HlslShader;
    let export = HlslShader::export_with_data_texture(&large_shape()).expect("export");
    assert!(export.data.is_some());
    assert!(export
        .source
        .contains(&format!("Texture2D<float> {DATA_TEXTURE_NAME}")));
    assert!(
        export.source.contains(".Load("),
        "exact fetch, no filtering"
    );
    assert!(!export.source.contains("alice_poly_v_"));
}

/// The points the GPU checks use: a grid across the outline (inside, on, outside) at three heights
fn grid() -> Vec<glam::Vec3> {
    let mut pts = Vec::new();
    for j in 0..48 {
        for i in 0..48 {
            for z in [-0.3f32, 0.0, 0.2] {
                pts.push(glam::Vec3::new(
                    i as f32 / 47.0 * 1.8 - 0.9,
                    j as f32 / 47.0 * 1.8 - 0.9,
                    z,
                ));
            }
        }
    }
    pts
}

#[cfg(feature = "glsl")]
#[test]
fn the_glsl_separate_data_and_single_file_agree_bit_for_bit_on_the_gpu() {
    use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
    use alice_sdf::compiled::GpuEvaluator;
    // 4,000 vertices: the single-file Vulkan form holds its data as a private array, and at the
    // 12,000 of `large_shape` that array is past the stack Metal gives a compute function (the
    // pipeline is refused); the separate-data form has no array and runs there too (below)
    let node = SdfNode::polygon_2d(outline(4_000), 0.25);
    let single =
        GlslShader::transpile(&node, GlslTranspileMode::Hardcoded).to_vulkan_compute_shader();
    let export = GlslShader::export_with_data_texture(&node).expect("export");
    let Some(a) = gpu_or_skip(
        "GLSL single-file",
        GpuEvaluator::from_glsl_compute(&single).map(|c| c.into_parts().0),
    ) else {
        return;
    };
    let Some(b) = gpu_or_skip(
        "GLSL separate-data",
        GpuEvaluator::from_glsl_data_export(&export),
    ) else {
        return;
    };
    let pts = grid();
    let da = gpu_batches::eval_in_batches(&a, &pts).expect("dispatch");
    let db = gpu_batches::eval_in_batches(&b, &pts).expect("dispatch");
    let differ = (0..pts.len())
        .filter(|&k| da[k].to_bits() != db[k].to_bits())
        .count();
    assert_eq!(differ, 0, "{differ} of {} points differ", pts.len());
    // the large shape: the separate-data form builds and gives the same values as the WGSL
    // separate-data form (the same text but for the language)
    let big = large_shape();
    let Some(g) = gpu_or_skip(
        "GLSL separate-data, large",
        GpuEvaluator::from_glsl_data_export(
            &GlslShader::export_with_data_texture(&big).expect("export"),
        ),
    ) else {
        return;
    };
    let Some(w) = gpu_or_skip(
        "WGSL separate-data, large",
        GpuEvaluator::from_data_export(
            &WgslShader::export_with_data_texture(&big).expect("export"),
        ),
    ) else {
        return;
    };
    let dg = gpu_batches::eval_in_batches(&g, &pts).expect("dispatch");
    let dw = gpu_batches::eval_in_batches(&w, &pts).expect("dispatch");
    let far = (0..pts.len())
        .filter(|&k| (dg[k] - dw[k]).abs() > 1e-5)
        .count();
    assert_eq!(
        far,
        0,
        "{far} of {} points differ by more than 1e-5",
        pts.len()
    );
}

#[test]
fn a_tree_whose_data_does_not_fit_one_texture_is_refused() {
    // one value more than a texture holds, as a lattice-free polygon outline (two values a vertex)
    let max = (DATA_TEXTURE_WIDTH * MAX_DATA_TEXTURE_ROWS) as usize;
    let node = SdfNode::polygon_2d(outline(max / 2 + 1), 0.25);
    assert_eq!(
        WgslShader::export_with_data_texture(&node).map(|_| ()),
        Err(DataExportError::TooLarge {
            values: max + 2,
            max_values: max
        })
    );
}

/// The module-scope array declarations of a shader text (a data array left in the text)
fn array_decls(text: &str) -> Vec<&str> {
    text.lines()
        .map(str::trim_start)
        .filter(|l| {
            (l.starts_with("const ") && l.contains(": array<"))
                || ((l.starts_with("const float ")
                    || l.starts_with("const int ")
                    || l.starts_with("static const float ")
                    || l.starts_with("static const int "))
                    && l.contains('['))
        })
        .collect()
}

#[test]
fn every_corpus_tree_keeps_no_data_array_in_the_text() {
    // the WGSL, GLSL and HLSL exports of every corpus tree: every data array went to the texture
    // (the loft's pieces and directions and the stacks' integer arrays included), and a tree
    // whose single-file text has data arrays has a texture
    let mut with_data = 0;
    for (name, node) in common::corpus::gate_corpus() {
        let mut texts = vec![(
            "wgsl",
            WgslShader::export_with_data_texture(&node).expect("export"),
        )];
        #[cfg(feature = "glsl")]
        texts.push((
            "glsl",
            alice_sdf::compiled::glsl::GlslShader::export_with_data_texture(&node).expect("export"),
        ));
        #[cfg(feature = "hlsl")]
        texts.push((
            "hlsl",
            alice_sdf::compiled::HlslShader::export_with_data_texture(&node).expect("export"),
        ));
        let single_has_arrays =
            !array_decls(&WgslShader::transpile(&node, TranspileMode::Hardcoded).source).is_empty();
        for (lang, e) in &texts {
            let left = array_decls(&e.source);
            assert!(
                left.is_empty(),
                "{name} ({lang}): {:?}",
                &left[..left.len().min(3)]
            );
            assert_eq!(e.data.is_some(), single_has_arrays, "{name} ({lang})");
        }
        with_data += usize::from(single_has_arrays);
    }
    assert!(
        with_data >= 10,
        "only {with_data} corpus trees have data arrays"
    );
}

/// The corpus trees of every data-array kind (polygon, loft with and without pieces, the stacks,
/// a height map, a lattice)
const KINDS: [&str; 10] = [
    "polygon_41",
    "loft_curves",
    "loft_nofade",
    "loft_faded",
    "loft_curves_41",
    "section_stack",
    "smooth_section_stack",
    "smooth_stack_41",
    "heightmap_displacement",
    "lattice_deform",
];

fn kind_trees() -> Vec<(&'static str, SdfNode)> {
    let all = common::corpus::gate_corpus();
    let picked: Vec<_> = all.into_iter().filter(|(n, _)| KINDS.contains(n)).collect();
    assert_eq!(picked.len(), KINDS.len(), "a kind tree left the corpus");
    picked
}

fn small_grid() -> Vec<glam::Vec3> {
    let mut pts = Vec::new();
    for j in 0..16 {
        for i in 0..16 {
            for z in [-0.4f32, 0.0, 0.3] {
                pts.push(glam::Vec3::new(
                    i as f32 / 15.0 * 2.4 - 1.2,
                    j as f32 / 15.0 * 2.4 - 1.2,
                    z,
                ));
            }
        }
    }
    pts
}

/// The kinds whose data the shader reads with run-time indices (a loop over the outline, the
/// stack's rows, the height map's texels): no compiler folds these reads, so the separate-data
/// shader gives the single file's bits on every backend
const RUNTIME_INDEXED: [&str; 4] = [
    "polygon_41",
    "loft_nofade",
    "section_stack",
    "heightmap_displacement",
];

/// The wgpu lane's bound on the other kinds (lofts with pieces, smooth stacks), in units of
/// [`unit`]: wgpu compiles Metal with fast math on (wgpu-hal 24.0.4 metal/device.rs:194, a default
/// `CompileOptions`), so a constant read at a fixed index can be folded and contracted differently
/// from a texture read; measured at most 1.2e-7 apart (1 unit) on Apple silicon
const UNIT_BOUND: f32 = 2.0;

/// The rounding unit of a distance value: the ulp at max(|v|, 1). A distance near the surface is
/// a small number computed from coordinates near 1, so its rounding is that of 1, not of itself
/// (counted in its own ulps, a 1.5e-8 difference at a value near 0 is a million ulps)
fn unit(v: f32) -> f32 {
    let x = v.abs().max(1.0);
    f32::from_bits(x.to_bits() + 1) - x
}

/// The distance between two values in rounding units
fn units(a: f32, b: f32) -> f32 {
    (a - b).abs() / unit(a.abs().max(b.abs()))
}

fn differing(a: &[f32], b: &[f32]) -> usize {
    (0..a.len())
        .filter(|&k| a[k].to_bits() != b[k].to_bits())
        .count()
}

fn max_units(a: &[f32], b: &[f32]) -> f32 {
    (0..a.len()).map(|k| units(a[k], b[k])).fold(0f32, f32::max)
}

/// The largest distance from the CPU's values
fn max_from_cpu(gpu: &[f32], cpu: &[f32]) -> f32 {
    (0..gpu.len())
        .map(|k| (gpu[k] - cpu[k]).abs())
        .fold(0f32, f32::max)
}

/// `values` each moved by `n` rounding units (a deliberate perturbation the bounds must catch)
fn nudged(values: &[f32], n: f32) -> Vec<f32> {
    values.iter().map(|v| v + n * unit(*v)).collect()
}

/// The wgpu lane's rule for one kind: `single` and `separate` the single-file and separate-data
/// values at `pts`. Returns the violation, if any
fn wgpu_rule(
    name: &str,
    node: &SdfNode,
    pts: &[glam::Vec3],
    single: &[f32],
    separate: &[f32],
) -> Option<String> {
    if RUNTIME_INDEXED.contains(&name) {
        let n = differing(single, separate);
        return (n > 0).then(|| format!("{name}: {n} of {} points differ in bits", pts.len()));
    }
    if name == "lattice_deform" {
        // the lattice inverts its map by iteration on the GPU: two texts that round 1 ulp apart
        // drift further (measured 4.2e-5 apart, both 1.1e-4 from the CPU), so its rule is the
        // distance from the CPU: no further than the single file's, plus 1 %
        let cpu: Vec<f32> = pts.iter().map(|p| alice_sdf::eval(node, *p)).collect();
        let (a, b) = (max_from_cpu(single, &cpu), max_from_cpu(separate, &cpu));
        return (b > a * 1.01)
            .then(|| format!("{name}: {b:e} from the CPU, the single file {a:e}"));
    }
    let u = max_units(single, separate);
    (u > UNIT_BOUND).then(|| format!("{name}: {u} units from the single file (bound {UNIT_BOUND})"))
}

#[test]
fn wgpu_rule_has_teeth() {
    // a +4 unit perturbation of every value fails the unit bound and the bit rule; the lattice's
    // CPU-distance rule fails when every value moves by 2 % of the single file's distance
    let (_, lattice) = kind_trees()
        .into_iter()
        .find(|(n, _)| *n == "lattice_deform")
        .unwrap();
    let pts = small_grid();
    let cpu: Vec<f32> = pts.iter().map(|p| alice_sdf::eval(&lattice, *p)).collect();
    // a stand-in single file 1e-4 from the CPU, as the GPU lattice is
    let single: Vec<f32> = cpu.iter().map(|v| v + 1e-4).collect();
    assert!(wgpu_rule("lattice_deform", &lattice, &pts, &single, &single).is_none());
    let moved: Vec<f32> = single.iter().map(|v| v + 2e-6).collect();
    assert!(wgpu_rule("lattice_deform", &lattice, &pts, &single, &moved).is_some());
    for name in ["loft_faded", "polygon_41"] {
        assert!(wgpu_rule(name, &lattice, &pts, &cpu, &cpu).is_none());
        assert!(
            wgpu_rule(name, &lattice, &pts, &cpu, &nudged(&cpu, 4.0)).is_some(),
            "{name}"
        );
    }
    // the unit is the ulp of 1 below 1 and the value's own above
    assert_eq!(unit(0.25), f32::EPSILON);
    assert_eq!(unit(-3.0), 2.0 * f32::EPSILON);
    assert_eq!(units(1e-8, 0.0), 1e-8 / f32::EPSILON);
}

#[test]
fn every_kind_follows_the_wgpu_rule_against_the_single_file() {
    // WGSL; on macOS with msl the evaluators are the Metal engine and the strict test below holds
    use alice_sdf::compiled::GpuEvaluator;
    let pts = small_grid();
    let mut broken = Vec::new();
    for (name, node) in kind_trees() {
        let single = WgslShader::transpile(&node, TranspileMode::Hardcoded);
        let export = WgslShader::export_with_data_texture(&node).expect("export");
        assert!(export.data.is_some(), "{name} has data");
        let Some(a) = gpu_or_skip(name, GpuEvaluator::from_shader(&single)) else {
            return;
        };
        let Some(b) = gpu_or_skip(name, GpuEvaluator::from_data_export(&export)) else {
            return;
        };
        let da = gpu_batches::eval_in_batches(&a, &pts).expect("dispatch");
        let db = gpu_batches::eval_in_batches(&b, &pts).expect("dispatch");
        broken.extend(wgpu_rule(name, &node, &pts, &da, &db));
        // the rule is not met by chance: the same values moved 4 units break it (not the lattice's)
        if name != "lattice_deform" {
            assert!(
                wgpu_rule(name, &node, &pts, &da, &nudged(&db, 4.0)).is_some(),
                "{name}"
            );
        }
    }
    assert!(broken.is_empty(), "{broken:#?}");
}

#[cfg(feature = "glsl")]
#[test]
fn every_kind_follows_the_wgpu_rule_against_the_single_file_glsl() {
    // the GLSL forms always run through wgpu (naga's GLSL front end)
    use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
    use alice_sdf::compiled::GpuEvaluator;
    let pts = small_grid();
    let mut broken = Vec::new();
    for (name, node) in kind_trees() {
        let single =
            GlslShader::transpile(&node, GlslTranspileMode::Hardcoded).to_vulkan_compute_shader();
        let export = GlslShader::export_with_data_texture(&node).expect("export");
        assert!(export.data.is_some(), "{name} has data");
        let Some(a) = gpu_or_skip(
            name,
            GpuEvaluator::from_glsl_compute(&single).map(|c| c.into_parts().0),
        ) else {
            return;
        };
        let Some(b) = gpu_or_skip(name, GpuEvaluator::from_glsl_data_export(&export)) else {
            return;
        };
        let da = gpu_batches::eval_in_batches(&a, &pts).expect("dispatch");
        let db = gpu_batches::eval_in_batches(&b, &pts).expect("dispatch");
        broken.extend(wgpu_rule(name, &node, &pts, &da, &db));
        if name != "lattice_deform" {
            assert!(
                wgpu_rule(name, &node, &pts, &da, &nudged(&db, 4.0)).is_some(),
                "{name}"
            );
        }
    }
    assert!(broken.is_empty(), "{broken:#?}");
}

/// The strict lane: the Metal engine compiles with fast math off (`MslShader`), so neither text is
/// folded or contracted and the separate-data shader gives the bits of the single file and of the
/// node data buffer form for every kind. Runs on macOS with `msl` (CI's macOS job and the local
/// full preflight, `ALICE_SDF_REQUIRE_METAL=1`); elsewhere it is ignored and the wgpu rule above
/// is what holds
#[test]
#[cfg_attr(
    not(all(target_os = "macos", feature = "msl")),
    ignore = "needs the Metal engine (macOS with msl); the wgpu lane is every_kind_follows_the_wgpu_rule_against_the_single_file"
)]
fn every_kind_agrees_bit_for_bit_on_the_metal_engine() {
    use alice_sdf::compiled::GpuEvaluator;
    let pts = small_grid();
    let mut compared = 0;
    for (name, node) in kind_trees() {
        let single = WgslShader::transpile(&node, TranspileMode::Hardcoded);
        let buffer = WgslShader::try_transpile_plain_with_node_data(&node).expect("transpile");
        let export = WgslShader::export_with_data_texture(&node).expect("export");
        let build = |r: Result<GpuEvaluator, alice_sdf::compiled::GpuError>| match r {
            Ok(e) => Some(e),
            Err(e) => {
                assert!(
                    std::env::var_os("ALICE_SDF_REQUIRE_METAL").is_none(),
                    "ALICE_SDF_REQUIRE_METAL is set but no Metal device: {e}"
                );
                eprintln!("skipping the Metal lane: {e}");
                None
            }
        };
        let (Some(a), Some(b), Some(c)) = (
            build(GpuEvaluator::from_shader(&single)),
            build(GpuEvaluator::from_shader(&buffer)),
            build(GpuEvaluator::from_data_export(&export)),
        ) else {
            return;
        };
        // the device the strict lane ran on, in the log (a hosted macOS runner is a VM)
        eprintln!("{name}: {}", c.device_info());
        assert!(
            c.device_info().contains("fast math off"),
            "{}",
            c.device_info()
        );
        let da = gpu_batches::eval_in_batches(&a, &pts).expect("dispatch");
        let db = gpu_batches::eval_in_batches(&b, &pts).expect("dispatch");
        let dc = gpu_batches::eval_in_batches(&c, &pts).expect("dispatch");
        assert_eq!(
            differing(&da, &dc),
            0,
            "{name}: separate-data vs single file"
        );
        assert_eq!(
            differing(&db, &dc),
            0,
            "{name}: separate-data vs node data buffer"
        );
        compared += 1;
    }
    assert_eq!(compared, KINDS.len());
}
