//! Emit a fully composed NPR shader source for a small SDF scene
//!
//! Uses [`SceneShaderBuilder`] to compose the NPR helper library, the
//! transpiled SDF evaluator, and a raymarching `main()` into a single
//! shader source string. Prints the GLSL output (and, optionally, WGSL
//! and HLSL if their features are enabled) so callers can pipe the
//! result into a shader compiler or paste it into a live shader
//! playground.
//!
//! # Running
//! ```bash
//! cargo run --features glsl --example npr_scene_shader
//! cargo run --features glsl,hlsl,gpu --example npr_scene_shader
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(any(feature = "glsl", feature = "hlsl", feature = "gpu"))]
use alice_sdf::prelude::*;

#[cfg(any(feature = "glsl", feature = "hlsl", feature = "gpu"))]
fn main() {
    // Compose a small scene: sphere with a subtracted box, smooth-unioned
    // with a torus. The transpiler will flatten this into a single
    // `sdf_eval(vec3 p)` function inside the emitted shader.
    let scene = SdfNode::sphere(1.0)
        .subtract(SdfNode::box3d(0.7, 0.7, 0.7))
        .smooth_union(SdfNode::torus(1.2, 0.15), 0.15);

    #[cfg(feature = "glsl")]
    emit_shader("GLSL", &scene, ShaderLanguage::Glsl);
    #[cfg(feature = "gpu")]
    emit_shader("WGSL", &scene, ShaderLanguage::Wgsl);
    #[cfg(feature = "hlsl")]
    emit_shader("HLSL", &scene, ShaderLanguage::Hlsl);
}

#[cfg(any(feature = "glsl", feature = "hlsl", feature = "gpu"))]
fn emit_shader(label: &str, scene: &SdfNode, language: ShaderLanguage) {
    // Custom NPR pipeline via NprColorNode + with_pipeline
    // Two-tone base color + hard outline over the top
    let pipeline = NprColorNode::TwoTone {
        shadow: Vec3::new(0.15, 0.13, 0.30),
        light: Vec3::new(0.92, 0.85, 0.70),
        threshold: 0.5,
    }
    .with_outline(Vec3::new(0.03, 0.03, 0.06), 0.85);

    let source = SceneShaderBuilder::new(scene, language)
        .with_pipeline(pipeline)
        .with_sky(
            Vec3::new(0.90, 0.60, 0.40),
            Vec3::new(0.65, 0.70, 0.85),
            Vec3::new(0.20, 0.30, 0.60),
        )
        .with_sun(Vec3::new(0.4, 0.7, -0.6), Vec3::new(1.0, 0.95, 0.8))
        .with_raymarch(96, 20.0, 1e-3)
        .build();

    let separator = "=".repeat(60);
    println!("{separator}");
    println!("{label} — {} bytes", source.len());
    println!("{separator}");
    println!("{source}");
    println!();
}

#[cfg(not(any(feature = "glsl", feature = "hlsl", feature = "gpu")))]
fn main() {
    println!(
        "npr_scene_shader: enable at least one shader feature (glsl / hlsl / gpu) to emit output"
    );
}
