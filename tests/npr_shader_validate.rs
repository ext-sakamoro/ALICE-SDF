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

fn build_pipeline_hatch() -> NprColorNode {
    // Exercise Phase 10 Hatch variant on top of two-tone base
    NprColorNode::TwoTone {
        shadow: glam::Vec3::new(0.15, 0.13, 0.30),
        light: glam::Vec3::new(0.92, 0.85, 0.70),
        threshold: 0.5,
    }
    .with_hatch(0.4, 40.0, 0.15, glam::Vec3::new(0.05, 0.05, 0.1))
}

fn build_pipeline_phase11() -> NprColorNode {
    // Exercise Phase 11: Palette5 base + Tonemap + SpeedLine
    use alice_sdf::npr::dsl::PaletteSource;
    NprColorNode::Palette5 {
        source: PaletteSource::NDotL,
        c0: glam::Vec3::new(0.10, 0.05, 0.20),
        c1: glam::Vec3::new(0.50, 0.10, 0.30),
        c2: glam::Vec3::new(0.90, 0.40, 0.20),
        c3: glam::Vec3::new(0.95, 0.85, 0.50),
        c4: glam::Vec3::new(0.80, 0.95, 0.95),
    }
    .tonemap_reinhard(1.2)
    .with_speed_lines(
        glam::Vec2::new(0.5, 0.5),
        24,
        0.03,
        glam::Vec3::new(0.02, 0.02, 0.05),
    )
}

const fn build_pipeline_time_cycle() -> NprColorNode {
    // Exercise Phase 12: PaletteSource::TimeCycle animated palette
    use alice_sdf::npr::dsl::PaletteSource;
    NprColorNode::Palette3 {
        source: PaletteSource::TimeCycle,
        c0: glam::Vec3::new(0.1, 0.15, 0.35),
        c1: glam::Vec3::new(0.85, 0.65, 0.35),
        c2: glam::Vec3::new(0.35, 0.75, 0.60),
    }
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
        defines: naga::FastHashMap::default(),
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
        defines: naga::FastHashMap::default(),
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
        defines: naga::FastHashMap::default(),
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
        defines: naga::FastHashMap::default(),
    };
    let result = frontend.parse(&options, &source);
    assert!(
        result.is_ok(),
        "GLSL parse failed: {:?}\n---source---\n{source}",
        result.err()
    );
}

#[test]
fn wgsl_hatch_pipeline_parses() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Wgsl)
        .with_pipeline(build_pipeline_hatch())
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
fn glsl_hatch_pipeline_parses() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Glsl)
        .with_pipeline(build_pipeline_hatch())
        .build();
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options {
        stage: naga::ShaderStage::Fragment,
        defines: naga::FastHashMap::default(),
    };
    let result = frontend.parse(&options, &source);
    assert!(
        result.is_ok(),
        "GLSL parse failed: {:?}\n---source---\n{source}",
        result.err()
    );
}

#[test]
fn wgsl_phase11_pipeline_parses() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Wgsl)
        .with_pipeline(build_pipeline_phase11())
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
fn glsl_phase11_pipeline_parses() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Glsl)
        .with_pipeline(build_pipeline_phase11())
        .build();
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options {
        stage: naga::ShaderStage::Fragment,
        defines: naga::FastHashMap::default(),
    };
    let result = frontend.parse(&options, &source);
    assert!(
        result.is_ok(),
        "GLSL parse failed: {:?}\n---source---\n{source}",
        result.err()
    );
}

#[test]
fn wgsl_time_cycle_pipeline_parses() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Wgsl)
        .with_pipeline(build_pipeline_time_cycle())
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
fn glsl_time_cycle_pipeline_parses() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Glsl)
        .with_pipeline(build_pipeline_time_cycle())
        .build();
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options {
        stage: naga::ShaderStage::Fragment,
        defines: naga::FastHashMap::default(),
    };
    let result = frontend.parse(&options, &source);
    assert!(
        result.is_ok(),
        "GLSL parse failed: {:?}\n---source---\n{source}",
        result.err()
    );
}

#[test]
fn wgsl_time_cycle_pipeline_semantic_validates() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Wgsl)
        .with_pipeline(build_pipeline_time_cycle())
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
fn wgsl_phase11_pipeline_semantic_validates() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Wgsl)
        .with_pipeline(build_pipeline_phase11())
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
fn wgsl_hatch_pipeline_semantic_validates() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Wgsl)
        .with_pipeline(build_pipeline_hatch())
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

/// The built-in hit block (no `with_pipeline`) is driven by
/// `with_shading` / `with_outline`, and `with_camera` moves the ray origin:
/// each configured value must reach the statement that uses it, with the
/// documented clamps (bands >= 1, inner >= 0, outer >= inner + 1e-4), and
/// the result must still validate. With a pipeline set the shading and
/// outline parameters are documented as ignored on the hit branch.
#[test]
fn wgsl_builtin_shading_outline_and_camera_reach_the_source() {
    let scene = build_scene();
    let shadow = glam::Vec3::new(0.125, 0.25, 0.5);
    let light = glam::Vec3::new(0.75, 0.625, 0.375);
    let outline = glam::Vec3::new(0.0625, 0.03125, 0.015625);
    let cam = glam::Vec3::new(0.5, 1.25, -4.5);
    let builder = SceneShaderBuilder::new(&scene, ShaderLanguage::Wgsl)
        .with_shading(shadow, light, 5)
        .with_outline(outline, 0.01, 0.04)
        .with_camera(cam);
    let source = builder.build();
    let mut compared = 0;
    for needle in [
        "alice_soft_toon_ramp(ndl, f32(5),",
        "mix(vec3<f32>(0.125000, 0.250000, 0.500000), vec3<f32>(0.750000, 0.625000, 0.375000), vec3<f32>(brightness))",
        "alice_distance_field_outline_soft(d, 0.010000, 0.040000)",
        "vec3<f32>(0.062500, 0.031250, 0.015625), outline_mask)",
        "let ray_origin = vec3<f32>(0.500000, 1.250000, -4.500000);",
    ] {
        assert!(source.contains(needle), "missing `{needle}`\n---source---\n{source}");
        compared += 1;
    }
    assert_eq!(compared, 5);
    let module = naga::front::wgsl::parse_str(&source).expect("WGSL parses");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("WGSL validates");

    // Clamps: 0 bands becomes 1, a negative inner width 0, an outer width
    // below inner + 1e-4 is raised to it.
    let clamped = SceneShaderBuilder::new(&scene, ShaderLanguage::Wgsl)
        .with_shading(shadow, light, 0)
        .with_outline(outline, -0.5, -1.0)
        .build();
    assert!(
        clamped.contains("alice_soft_toon_ramp(ndl, f32(1),"),
        "{clamped}"
    );
    assert!(
        clamped.contains("alice_distance_field_outline_soft(d, 0.0, "),
        "{clamped}"
    );

    // A pipeline replaces the built-in block: the shading colours are gone.
    let piped = builder.with_pipeline(build_pipeline()).build();
    assert!(!piped.contains("alice_soft_toon_ramp(ndl, f32(5),"));
    assert!(piped.contains("let ray_origin = vec3<f32>(0.500000, 1.250000, -4.500000);"));
}

#[test]
fn glsl_builtin_shading_outline_and_camera_reach_the_source() {
    let scene = build_scene();
    let source = SceneShaderBuilder::new(&scene, ShaderLanguage::Glsl)
        .with_shading(
            glam::Vec3::new(0.125, 0.25, 0.5),
            glam::Vec3::new(0.75, 0.625, 0.375),
            4,
        )
        .with_outline(glam::Vec3::new(0.0625, 0.0, 0.0), 0.02, 0.05)
        .with_camera(glam::Vec3::new(0.0, 2.0, -6.0))
        .build();
    for needle in [
        "alice_soft_toon_ramp(ndl, 4.0,",
        "mix(vec3(0.125000, 0.250000, 0.500000), vec3(0.750000, 0.625000, 0.375000), brightness)",
        "alice_distance_field_outline_soft(d, 0.020000, 0.050000)",
        "vec3 ray_origin = vec3(0.0, 2.0, -6.0);",
    ] {
        assert!(
            source.contains(needle),
            "missing `{needle}`\n---source---\n{source}"
        );
    }
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options {
        stage: naga::ShaderStage::Fragment,
        defines: naga::FastHashMap::default(),
    };
    frontend.parse(&options, &source).expect("GLSL parses");
}
