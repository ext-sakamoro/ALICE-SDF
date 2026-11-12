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
        .contains(&format!("uniform sampler2D {DATA_TEXTURE_NAME}")));
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
    let node = large_shape();
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
