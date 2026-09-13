//! Validate the GPU bytecode WGSL evaluator (Phase 14)
//!
//! Wraps the string returned by `emit_wgsl_bytecode_evaluator` in a
//! minimal fragment-shader entry point that binds a program buffer and
//! calls the evaluator, then parses and semantically validates the
//! result with `naga`.
//!
//! `naga` is a dev-dependency; this test does not require any Cargo
//! feature to be enabled.
//!
//! Author: Moroya Sakamoto

use alice_sdf::npr::compiled_color::emit_wgsl_bytecode_evaluator;

fn wrap_evaluator_in_fragment_shader() -> String {
    let evaluator = emit_wgsl_bytecode_evaluator();
    format!(
        r#"{evaluator}

@group(0) @binding(0) var<storage, read> alice_npr_program: array<u32>;

struct AliceNprUniforms {{
    program_len: u32,
    n_dot_l: f32,
    n_dot_v: f32,
    sdf: f32,
    uv_x: f32,
    uv_y: f32,
    time: f32,
    _pad: u32,
}};

@group(0) @binding(1) var<uniform> alice_npr_uniforms: AliceNprUniforms;

fn alice_npr_load(index: u32) -> u32 {{
    return alice_npr_program[index];
}}

struct FragInput {{
    @builtin(position) frag_pos: vec4<f32>,
}};

@fragment
fn fs_main(_in: FragInput) -> @location(0) vec4<f32> {{
    let ctx = AliceNprBytecodeCtx(
        alice_npr_uniforms.n_dot_l,
        alice_npr_uniforms.n_dot_v,
        alice_npr_uniforms.sdf,
        vec2<f32>(alice_npr_uniforms.uv_x, alice_npr_uniforms.uv_y),
        alice_npr_uniforms.time,
    );
    let color = alice_npr_eval_bytecode(alice_npr_uniforms.program_len, ctx);
    return vec4<f32>(color, 1.0);
}}
"#
    )
}

#[test]
fn emitted_evaluator_parses_with_naga_wgsl_frontend() {
    let source = wrap_evaluator_in_fragment_shader();
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
fn emitted_evaluator_passes_naga_semantic_validation() {
    let source = wrap_evaluator_in_fragment_shader();
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
fn emitted_evaluator_contains_all_expected_opcodes() {
    // Sanity: without the constants, a caller-provided wrapper cannot
    // route to the right opcode. Guards against a future refactor that
    // drops one accidentally.
    let source = emit_wgsl_bytecode_evaluator();
    for name in [
        "ALICE_OP_PUSH_CONSTANT",
        "ALICE_OP_TOON",
        "ALICE_OP_SOFT_TOON",
        "ALICE_OP_TWO_TONE",
        "ALICE_OP_MULTIPLY",
        "ALICE_OP_ADD",
        "ALICE_OP_SCALE",
        "ALICE_OP_OUTLINE_OVER",
        "ALICE_OP_FRESNEL",
        "ALICE_OP_SATURATE",
        "ALICE_OP_BLOOM",
        "ALICE_OP_POSTERIZE_COLOR",
        "ALICE_OP_VIGNETTE",
        "ALICE_OP_PALETTE3",
        "ALICE_OP_PALETTE5",
        "ALICE_OP_HATCH",
        "ALICE_OP_TONEMAP",
        "ALICE_OP_SPEED_LINE",
    ] {
        assert!(source.contains(name), "missing {name}");
    }
}
