//! Compile an NPR colour tree to GPU bytecode and back
//!
//! Builds a stylised-shading tree with the `NprColorNode` builder helpers,
//! compiles it to the stack-machine form, checks it, serialises it to the
//! `u32` stream the WGSL evaluator reads, decodes it again, and shows that
//! all three forms shade a sweep of points identically. Also prints the
//! opcode table (tag, payload words, stack effect) and the size of the WGSL
//! evaluator, and uses the scalar primitives (`NprInput`, outlines, noise)
//! that feed such a tree.
//!
//! # Running
//! ```bash
//! cargo run --example npr_bytecode_program
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::npr::compiled_color::{
    emit_wgsl_bytecode_evaluator, gpu_opcode_tag, gpu_palette_source_tag, opcode_word_count,
    ColorOp, CompiledColorPipeline, DeserializeError, GpuColorProgram, SerializeError, StackError,
};
use alice_sdf::npr::dsl::{NprColorContext, NprColorNode, PaletteSource};
use alice_sdf::npr::noise::{NoiseField, PerlinNoise, SimplexNoise, WorleyNoise};
use alice_sdf::npr::outline::{depth_step_outline, distance_field_outline};
use alice_sdf::npr::NprInput;
use alice_sdf::types::SdfNode;
use glam::{Vec2, Vec3};

fn main() {
    // --- A tree built with the builder helpers -----------------------------
    let toon = NprColorNode::Toon {
        shadow: Vec3::new(0.15, 0.1, 0.3),
        light: Vec3::new(0.95, 0.85, 0.7),
        bands: 4,
    };
    let rim = NprColorNode::Palette3 {
        source: PaletteSource::NDotV,
        c0: Vec3::new(1.0, 0.8, 0.6),
        c1: Vec3::new(0.3, 0.3, 0.4),
        c2: Vec3::ZERO,
    };
    let tree = toon
        .multiply(NprColorNode::Constant(Vec3::new(1.0, 0.95, 0.9)))
        .plus(rim.scale(0.25))
        .bloom(0.1, 1.05)
        .posterize(6)
        .with_hatch(0.785, 12.0, 0.08, Vec3::new(0.05, 0.05, 0.1))
        .with_speed_lines(Vec2::new(0.5, 0.5), 48, 0.05, Vec3::ZERO);

    // --- Compile, check, serialise, decode ---------------------------------
    let compiled = CompiledColorPipeline::compile(&tree);
    compiled
        .validate()
        .expect("compile() emits a balanced program");
    let program: GpuColorProgram = compiled.serialize().expect("no Fallback opcode");
    let decoded = program.deserialize().expect("the stream decodes");
    println!(
        "{} opcodes ({} native), {} words = {} bytes",
        compiled.ops.len(),
        compiled.native_op_count(),
        program.as_words().len(),
        program.byte_len()
    );
    assert_eq!(compiled.native_op_count(), compiled.ops.len());
    assert_eq!(program.byte_len(), 4 * program.as_words().len());

    // Walk the stream with the per-tag payload length.
    let words = program.as_words();
    let mut pc = 0;
    let mut depth = 0usize;
    println!("\n  pc  tag  payload  stack");
    for op in &decoded.ops {
        let tag = words[pc];
        let payload = opcode_word_count(tag).expect("known tag");
        let (pops, pushes) = op.stack_effect();
        depth = depth - pops + pushes;
        println!("{pc:>4} {tag:>4} {payload:>8}  -{pops} +{pushes} -> {depth}");
        pc += 1 + payload;
    }
    assert_eq!(pc, words.len(), "the walk ends on the last word");
    assert_eq!(depth, 1, "one colour remains");

    // All three forms shade alike.
    let mut worst = 0.0f32;
    for k in 0..64 {
        let s = k as f32 / 63.0;
        let ctx = NprColorContext {
            sdf: 0.0,
            normal: Vec3::Z,
            view: Vec3::new(0.0, 0.0, 1.0 - s),
            light: Vec3::new(0.0, 0.0, s),
            uv: Vec2::new(s, 1.0 - s * 0.5),
            time: 0.0,
        };
        let want = tree.eval(&ctx);
        for got in [compiled.eval(&ctx), decoded.eval(&ctx)] {
            worst = worst.max((got - want).abs().max_element());
        }
    }
    println!("\nmax |bytecode - tree| over 64 points: {worst:e}");
    assert_eq!(worst, 0.0);

    // --- What the checks reject --------------------------------------------
    let unbalanced = CompiledColorPipeline {
        ops: vec![ColorOp::Add],
    };
    let e: StackError = unbalanced.validate().unwrap_err();
    println!("validate([Add]): {e}");
    let truncated = GpuColorProgram {
        words: vec![gpu_opcode_tag::PUSH_CONSTANT, 0],
    };
    let e: DeserializeError = truncated.deserialize().unwrap_err();
    println!("deserialize(truncated): {e}");
    let bad_source = GpuColorProgram {
        words: [
            gpu_opcode_tag::PALETTE3,
            gpu_palette_source_tag::TIME_CYCLE + 1,
        ]
        .into_iter()
        .chain([0; 9])
        .collect(),
    };
    println!(
        "deserialize(unknown palette source): {}",
        bad_source.deserialize().unwrap_err()
    );
    let fallback = CompiledColorPipeline {
        ops: vec![ColorOp::Fallback(Box::new(NprColorNode::Constant(
            Vec3::ONE,
        )))],
    };
    let e: SerializeError = fallback.serialize().unwrap_err();
    println!("serialize(Fallback): {e}");

    // --- Opcode table and the GPU evaluator ---------------------------------
    use gpu_opcode_tag as t;
    let table = [
        ("PUSH_CONSTANT", t::PUSH_CONSTANT),
        ("TOON", t::TOON),
        ("SOFT_TOON", t::SOFT_TOON),
        ("TWO_TONE", t::TWO_TONE),
        ("MULTIPLY", t::MULTIPLY),
        ("ADD", t::ADD),
        ("SCALE", t::SCALE),
        ("OUTLINE_OVER", t::OUTLINE_OVER),
        ("FRESNEL", t::FRESNEL),
        ("SATURATE", t::SATURATE),
        ("BLOOM", t::BLOOM),
        ("POSTERIZE_COLOR", t::POSTERIZE_COLOR),
        ("VIGNETTE", t::VIGNETTE),
        ("PALETTE3", t::PALETTE3),
        ("PALETTE5", t::PALETTE5),
        ("HATCH", t::HATCH),
        ("TONEMAP", t::TONEMAP),
        ("SPEED_LINE", t::SPEED_LINE),
    ];
    let wgsl = emit_wgsl_bytecode_evaluator();
    println!("\nWGSL evaluator: {} bytes", wgsl.len());
    for (name, tag) in table {
        let decl = format!("const ALICE_OP_{name}: u32 = {tag}u;");
        assert!(wgsl.contains(&decl), "{decl}");
        println!(
            "  {name:<16} tag {tag:>2}  payload {} words",
            opcode_word_count(tag).unwrap()
        );
    }
    use gpu_palette_source_tag as p;
    for (name, tag) in [
        ("N_DOT_L", p::N_DOT_L),
        ("N_DOT_V", p::N_DOT_V),
        ("SDF", p::SDF),
        ("UV_Y", p::UV_Y),
        ("TIME_CYCLE", p::TIME_CYCLE),
    ] {
        assert!(wgsl.contains(&format!("const ALICE_PS_{name}: u32 = {tag}u;")));
    }

    // --- Scalar inputs of a tree ---------------------------------------------
    let sphere = SdfNode::sphere(1.0);
    let p = Vec3::new(0.0, 0.0, 1.02);
    let input = NprInput::new(
        eval(&sphere, p),
        Vec3::Z,
        Vec3::new(0.0, 0.6, 0.8),
        Vec3::new(0.0, 0.0, 1.0),
        p,
    );
    let outline = distance_field_outline(input.sdf, 0.05);
    let crease = depth_step_outline(0.4, 0.25);
    println!(
        "\nn.l = {:.2}, n.v = {:.2}, outline {outline}, depth step {crease}",
        input.n_dot_l(),
        input.n_dot_v()
    );
    assert_eq!(outline, 1.0, "|sdf| = 0.02 is inside the 0.05 outline");
    assert_eq!(crease, 1.0);

    let q = Vec3::new(0.3, 1.7, -2.2);
    let perlin = PerlinNoise::new(3).with_frequency(4.0).sample_scalar(q);
    let simplex = SimplexNoise::new(3).with_frequency(4.0).sample_scalar(q);
    let worley = WorleyNoise::new(3).with_frequency(4.0).sample_scalar(q);
    println!("noise at {q}: perlin {perlin:.4}, simplex {simplex:.4}, worley {worley:.4}");
    assert_eq!(perlin, PerlinNoise::new(3).sample_scalar(q * 4.0));
    assert_eq!(simplex, SimplexNoise::new(3).sample_scalar(q * 4.0));
}
