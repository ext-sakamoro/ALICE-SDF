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
