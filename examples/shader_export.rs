//! Shader export for OpenGL / WebGL, Unreal Engine and Nuke
//!
//! One scene is written out in the forms the GLSL, HLSL and BlinkScript
//! transpilers provide beyond the bare `sdf_eval` function:
//!
//! - GLSL: an OpenGL 4.5 compute shader, its Vulkan variant, a Shadertoy-style
//!   fragment shader and the full WebGL2 rendering pipeline (`RenderConfig`)
//! - HLSL: a compute shader, an Unreal material function (`.ush`) and Custom
//!   node body, and the Unity Shader Graph Custom Function files
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

#[path = "support/stored_text.rs"]
mod stored_text;

use alice_sdf::compiled::glsl::{render_pipeline::build_full_shader, RenderConfig};
use alice_sdf::compiled::{
    BlinkScriptShader, BlinkScriptTranspileMode, GlslShader, GlslTranspileMode, HlslShader,
    HlslTranspileMode,
};
use alice_sdf::prelude::*;

/// Whether an emitted text is a whole shader or a fragment
#[derive(Clone, Copy)]
enum Kind {
    Whole,
    Fragment,
}

fn main() {
    let out_dir = std::env::args().nth(1).map(std::path::PathBuf::from);
    for path in run(out_dir.as_deref()) {
        println!("wrote {}", path.display());
    }
}

/// Emits every form and returns the files written
///
/// Each form is written into `out_dir` when one is given; tests/test_g8_example_text.rs runs
/// this and checks each file against the shader rule
pub fn run(out_dir: Option<&std::path::Path>) -> Vec<std::path::PathBuf> {
    let shape = SdfNode::sphere(1.0)
        .translate(0.25, -0.5, 0.125)
        .smooth_union(SdfNode::box3d(0.5, 0.4, 0.3), 0.2);
    let mut written = Vec::new();
    // a whole shader carries the identifier its transpiler wrote; a fragment, pasted into a host
    // file or graph, carries none (ALICE-Law G8, examples/support/stored_text.rs)
    let mut emit = |name: &str, src: &str, kind: Kind| {
        println!("{name:<32} {:>6} bytes", src.len());
        if let Some(dir) = out_dir {
            let path = dir.join(name);
            match kind {
                Kind::Whole => stored_text::write_whole_text(&path, src),
                Kind::Fragment => stored_text::write_fragment_text(&path, src),
            }
            .expect("write shader");
            written.push(path);
        }
    };

    // ---- GLSL ----
    let glsl = GlslShader::transpile(&shape, GlslTranspileMode::Hardcoded);
    let compute = glsl.to_compute_shader();
    assert!(compute.starts_with("#version 450") && compute.contains("void main()"));
    emit("sdf_compute.comp", &compute, Kind::Whole);

    let vulkan = glsl.to_vulkan_compute_shader();
    assert!(vulkan.contains("set = 0, binding = 2) uniform Constants"));
    emit("sdf_compute_vulkan.comp", &vulkan, Kind::Whole);

    let fragment = glsl.to_fragment_shader();
    assert!(fragment.contains("uniform vec2 iResolution;") && fragment.contains("sdf_eval(p)"));
    emit("sdf_shadertoy.frag", &fragment, Kind::Whole);

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
    emit("sdf_full_pipeline.frag", &full, Kind::Whole);

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
    emit("sdf_full_pipeline_dual.frag", &dual, Kind::Whole);

    // ---- HLSL ----
    let hlsl = HlslShader::transpile(&shape, HlslTranspileMode::Hardcoded);
    let hlsl_compute = hlsl.to_compute_shader();
    assert!(hlsl_compute.contains("[numthreads("));
    emit("sdf_compute.hlsl", &hlsl_compute, Kind::Whole);

    let material_function = hlsl.export_ue5_material_function();
    assert!(material_function.contains("float AliceSdf_Eval(float3 WorldPosition)"));
    emit("MF_AliceSdf.ush", &material_function, Kind::Fragment);

    // Unreal Custom node: a function body (helpers are member functions of a
    // local struct, as HLSL has no nested functions).
    let custom_node = hlsl.to_ue5_custom_node();
    assert!(custom_node.contains("return alice_sdf_custom_node.sdf_eval(p);"));
    emit("ue5_custom_node.hlsl", &custom_node, Kind::Fragment);

    // Unity Shader Graph Custom Function files (Type: File).
    let unity_cf = hlsl.to_unity_custom_function();
    assert!(unity_cf.contains("void SdfEval_float(float3 p, out float distance)"));
    emit("SdfEval.hlsl", &unity_cf, Kind::Fragment);
    let unity_sg = hlsl.export_unity_shader_graph();
    assert!(unity_sg.contains("void AliceSdf_float(float3 Position"));
    assert!(!unity_sg.contains("vec3"));
    emit("AliceSdf.hlsl", &unity_sg, Kind::Fragment);

    // ---- BlinkScript ----
    let blink = BlinkScriptShader::transpile(&shape, BlinkScriptTranspileMode::Hardcoded);
    let body = blink.get_eval_function();
    assert!(body.contains("float sdf_eval(float3 p)"));
    emit("sdf_eval.blink", body, Kind::Fragment);

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
        Kind::Fragment,
    );

    written
}
