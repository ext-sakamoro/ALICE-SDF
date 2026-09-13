//! Validate the shader source emitted by `SceneShaderBuilder`
//!
//! Parses WGSL output via `naga::front::wgsl` and GLSL output via
//! `naga::front::glsl` to verify the generated code is at least
//! syntactically valid before any GPU driver sees it.
//!
//! Only compiled when the shader-transpiler features are enabled;
//! `naga` is otherwise unused.
//!
//! Author: Moroya Sakamoto

#![cfg(all(feature = "glsl", feature = "gpu"))]

use alice_sdf::prelude::*;

fn build_scene() -> SdfNode {
    SdfNode::sphere(1.0)
        .subtract(SdfNode::box3d(0.7, 0.7, 0.7))
        .smooth_union(SdfNode::torus(1.2, 0.15), 0.15)
}

fn build_pipeline() -> NprColorNode {
    // Two-tone core + Fresnel edge highlight + outline overlay + scale
    NprColorNode::TwoTone {
        shadow: glam::Vec3::new(0.15, 0.13, 0.30),
        light: glam::Vec3::new(0.92, 0.85, 0.70),
        threshold: 0.5,
    }
    .with_fresnel(glam::Vec3::new(0.9, 0.8, 0.6), 2.0)
    .with_outline(glam::Vec3::new(0.03, 0.03, 0.06), 0.85)
    .scale(1.05)
}

fn build_pipeline_full() -> NprColorNode {
    // Exercise all Phase 8 variants: TwoTone -> Saturate -> Bloom -> Posterize
    NprColorNode::TwoTone {
        shadow: glam::Vec3::new(0.15, 0.13, 0.30),
        light: glam::Vec3::new(0.92, 0.85, 0.70),
        threshold: 0.5,
    }
    .saturate(1.2)
    .bloom(0.4, 1.05)
    .posterize(4)
}

fn build_pipeline_uv() -> NprColorNode {
    // Exercise Phase 9 UV-dependent variants: Palette3(UvY) then vignette
    use alice_sdf::npr::dsl::PaletteSource;
    NprColorNode::Palette3 {
        source: PaletteSource::UvY,
        c0: glam::Vec3::new(0.9, 0.5, 0.3),
        c1: glam::Vec3::new(0.6, 0.6, 0.85),
        c2: glam::Vec3::new(0.15, 0.2, 0.55),
    }
    .vignetted(0.4, 0.3)
}

#[test]
fn wgsl_default_pipeline_parses() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Wgsl).build();
    let result = naga::front::wgsl::parse_str(&source);
    assert!(
        result.is_ok(),
        "WGSL parse failed:\n{}\n---source---\n{source}",
        result
            .err()
            .map(|e| e.emit_to_string(&source))
            .unwrap_or_default()
    );
}

#[test]
fn wgsl_pipeline_driven_parses() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Wgsl)
        .with_pipeline(build_pipeline())
        .build();
    let result = naga::front::wgsl::parse_str(&source);
    assert!(
        result.is_ok(),
        "WGSL parse failed:\n{}\n---source---\n{source}",
        result
            .err()
            .map(|e| e.emit_to_string(&source))
            .unwrap_or_default()
    );
}

#[test]
fn glsl_default_pipeline_parses() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Glsl).build();
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options {
        stage: naga::ShaderStage::Fragment,
        defines: Default::default(),
    };
    let result = frontend.parse(&options, &source);
    assert!(
        result.is_ok(),
        "GLSL parse failed: {:?}\n---source---\n{source}",
        result.err()
    );
}

#[test]
fn glsl_pipeline_driven_parses() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Glsl)
        .with_pipeline(build_pipeline())
        .build();
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options {
        stage: naga::ShaderStage::Fragment,
        defines: Default::default(),
    };
    let result = frontend.parse(&options, &source);
    assert!(
        result.is_ok(),
        "GLSL parse failed: {:?}\n---source---\n{source}",
        result.err()
    );
}

#[test]
fn wgsl_full_variant_pipeline_parses() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Wgsl)
        .with_pipeline(build_pipeline_full())
        .build();
    let result = naga::front::wgsl::parse_str(&source);
    assert!(
        result.is_ok(),
        "WGSL parse failed:\n{}\n---source---\n{source}",
        result
            .err()
            .map(|e| e.emit_to_string(&source))
            .unwrap_or_default()
    );
}

#[test]
fn glsl_full_variant_pipeline_parses() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Glsl)
        .with_pipeline(build_pipeline_full())
        .build();
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options {
        stage: naga::ShaderStage::Fragment,
        defines: Default::default(),
    };
    let result = frontend.parse(&options, &source);
    assert!(
        result.is_ok(),
        "GLSL parse failed: {:?}\n---source---\n{source}",
        result.err()
    );
}

#[test]
fn wgsl_uv_pipeline_parses() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Wgsl)
        .with_pipeline(build_pipeline_uv())
        .build();
    let result = naga::front::wgsl::parse_str(&source);
    assert!(
        result.is_ok(),
        "WGSL parse failed:\n{}\n---source---\n{source}",
        result
            .err()
            .map(|e| e.emit_to_string(&source))
            .unwrap_or_default()
    );
}

#[test]
fn glsl_uv_pipeline_parses() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Glsl)
        .with_pipeline(build_pipeline_uv())
        .build();
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options {
        stage: naga::ShaderStage::Fragment,
        defines: Default::default(),
    };
    let result = frontend.parse(&options, &source);
    assert!(
        result.is_ok(),
        "GLSL parse failed: {:?}\n---source---\n{source}",
        result.err()
    );
}

#[test]
fn wgsl_uv_pipeline_semantic_validates() {
    // UV pipeline through the semantic validator to catch binding / type errors
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Wgsl)
        .with_pipeline(build_pipeline_uv())
        .build();
    let module =
        naga::front::wgsl::parse_str(&source).expect("WGSL parse must succeed before validation");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    );
    let result = validator.validate(&module);
    assert!(
        result.is_ok(),
        "WGSL validation failed: {:?}\n---source---\n{source}",
        result.err()
    );
}

#[test]
fn wgsl_full_pipeline_semantic_validates() {
    // Beyond parse: run naga's Validator to catch resource / type / layout errors
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Wgsl)
        .with_pipeline(build_pipeline_full())
        .build();
    let module =
        naga::front::wgsl::parse_str(&source).expect("WGSL parse must succeed before validation");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    );
    let result = validator.validate(&module);
    assert!(
        result.is_ok(),
        "WGSL validation failed: {:?}\n---source---\n{source}",
        result.err()
    );
}
