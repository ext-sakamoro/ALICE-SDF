//! Shader export for OpenGL / WebGL, Unreal Engine and Nuke
//!
//! One scene is written out in the forms the GLSL, HLSL and BlinkScript
//! transpilers provide beyond the bare `sdf_eval` function:
//!
//! - GLSL: an OpenGL 4.5 compute shader, a Shadertoy-style fragment shader and
//!   the full WebGL2 rendering pipeline (`RenderConfig`)
//! - HLSL: a compute shader and an Unreal material function (`.ush`)
//! - BlinkScript: the `sdf_eval` body for a hand-written kernel
//! - the Dynamic parameter layouts that drive the uniform / constant buffers
//!
//! The values these produce are checked in `tests/test_glsl_export_oracle.rs`
//! and `tests/test_hlsl_export_oracle.rs`.
//!
//! ```bash
//! cargo run --example shader_export --features "glsl,hlsl,blinkscript"
//! cargo run --example shader_export --features "glsl,hlsl,blinkscript" -- out_dir
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::glsl::{render_pipeline::build_full_shader, RenderConfig};
use alice_sdf::compiled::{
    BlinkScriptShader, BlinkScriptTranspileMode, GlslShader, GlslTranspileMode, HlslShader,
    HlslTranspileMode,
};
use alice_sdf::prelude::*;

fn main() {
    let shape = SdfNode::sphere(1.0)
        .translate(0.25, -0.5, 0.125)
        .smooth_union(SdfNode::box3d(0.5, 0.4, 0.3), 0.2);
    let out_dir = std::env::args().nth(1).map(std::path::PathBuf::from);
    let mut written = Vec::new();
    let mut emit = |name: &str, src: &str| {
        println!("{name:<32} {:>6} bytes", src.len());
        if let Some(dir) = &out_dir {
            std::fs::create_dir_all(dir).expect("create output directory");
            let path = dir.join(name);
            std::fs::write(&path, src).expect("write shader");
            written.push(path);
        }
    };

    // ---- GLSL ----
    let glsl = GlslShader::transpile(&shape, GlslTranspileMode::Hardcoded);
    let compute = glsl.to_compute_shader();
    assert!(compute.starts_with("#version 450") && compute.contains("void main()"));
    emit("sdf_compute.comp", &compute);

    let fragment = glsl.to_fragment_shader();
    assert!(fragment.contains("uniform vec2 iResolution;") && fragment.contains("sdf_eval(p)"));
    emit("sdf_shadertoy.frag", &fragment);

    let config = RenderConfig {
        max_steps: 160,
        max_distance: 120.0,
        spectral_rendering: true,
        vfx_effects: true,
        ..RenderConfig::default()
    };
    let full = glsl.to_fragment_shader_full(&config);
    assert!(full.starts_with("#version 300 es"));
    assert!(full.contains("for(int i=0;i<160;i++)"));
    assert!(full.contains("out vec4 fragColor;"));
    emit("sdf_full_pipeline.frag", &full);

    // `build_full_shader` takes any `sdf_eval` source; `dual_sdf` adds a
    // cheaper `sdf_eval_lite` for AO / shadows / rain.
    let lite = format!(
        "{}\nfloat sdf_eval_lite(vec3 p) {{ return length(p - vec3(0.25, -0.5, 0.125)) - 1.0; }}\n",
        glsl.get_eval_function()
    );
    let dual = build_full_shader(
        &lite,
        &RenderConfig {
            dual_sdf: true,
            ..RenderConfig::default()
        },
    );
    assert!(dual.contains("return sdf_eval_lite(p);"));
    emit("sdf_full_pipeline_dual.frag", &dual);

    // ---- HLSL ----
    let hlsl = HlslShader::transpile(&shape, HlslTranspileMode::Hardcoded);
    let hlsl_compute = hlsl.to_compute_shader();
    assert!(hlsl_compute.contains("[numthreads("));
    emit("sdf_compute.hlsl", &hlsl_compute);

    let material_function = hlsl.export_ue5_material_function();
    assert!(material_function.contains("float AliceSdf_Eval(float3 WorldPosition)"));
    emit("MF_AliceSdf.ush", &material_function);

    // ---- BlinkScript ----
    let blink = BlinkScriptShader::transpile(&shape, BlinkScriptTranspileMode::Hardcoded);
    let body = blink.get_eval_function();
    assert!(body.contains("float sdf_eval(float3 p)"));
    emit("sdf_eval.blink", body);

    // ---- Dynamic parameters ----
    // Dynamic mode reads every constant from a buffer; the layouts below are
    // what an engine uploads when the scene's values change (same tree
    // shape, no recompilation).
    let moved = SdfNode::sphere(1.5)
        .translate(-0.5, 0.25, 0.0)
        .smooth_union(SdfNode::box3d(0.6, 0.4, 0.3), 0.1);
    let glsl_params = GlslShader::extract_params(&moved);
    let hlsl_params = HlslShader::extract_params(&moved);
    let blink_params = BlinkScriptShader::extract_params(&moved);
    assert_eq!(
        glsl_params.len(),
        GlslShader::transpile(&shape, GlslTranspileMode::Dynamic)
            .param_layout
            .len()
    );
    assert_eq!(
        hlsl_params, blink_params,
        "BlinkScript reuses the HLSL body"
    );
    println!("GLSL params  ({:>2}): {glsl_params:?}", glsl_params.len());
    println!("HLSL params  ({:>2}): {hlsl_params:?}", hlsl_params.len());
    emit(
        "MF_AliceSdf_dynamic.ush",
        &HlslShader::transpile(&shape, HlslTranspileMode::Dynamic).export_ue5_material_function(),
    );

    for path in &written {
        println!("wrote {}", path.display());
    }
}
