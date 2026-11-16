//! The full NPR bytecode evaluator module the GPU parity test dispatches (the evaluator of
//! `emit_wgsl_bytecode_evaluator` with its bindings and entry point), shared with the shader loop census
#![allow(dead_code)]

use alice_sdf::npr::compiled_color::emit_wgsl_bytecode_evaluator;

/// Threads per workgroup of the entry point
pub const WORKGROUP: u32 = 64;

/// The module: program words at binding 0, contexts at 1, colors at 2, sizes (program_len, count) at 3
pub fn shader_source() -> String {
    format!(
        r"{evaluator}

@group(0) @binding(0) var<storage, read> program: array<u32>;
@group(0) @binding(1) var<storage, read> contexts: array<f32>;
@group(0) @binding(2) var<storage, read_write> colors: array<f32>;
@group(0) @binding(3) var<uniform> sizes: vec4<u32>;

fn alice_npr_load(index: u32) -> u32 {{
    return program[index];
}}

@compute @workgroup_size({WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    if (i >= sizes.y) {{ return; }}
    let b = i * 6u;
    let ctx = AliceNprBytecodeCtx(
        contexts[b],
        contexts[b + 1u],
        contexts[b + 2u],
        vec2<f32>(contexts[b + 3u], contexts[b + 4u]),
        contexts[b + 5u],
    );
    let color = alice_npr_eval_bytecode(sizes.x, ctx);
    colors[i * 3u] = color.x;
    colors[i * 3u + 1u] = color.y;
    colors[i * 3u + 2u] = color.z;
}}
",
        evaluator = emit_wgsl_bytecode_evaluator()
    )
}
