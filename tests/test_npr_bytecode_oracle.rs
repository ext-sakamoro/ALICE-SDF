//! NPR colour bytecode vs its format definition and the tree walker
//! (`npr::compiled_color`, `npr::dsl` builders).
//!
//! - Wire format: each instruction is `[tag, ..payload]`, `f32` as its bits,
//!   `u32` as is, `PaletteSource` as a small tag. The reference encoder below
//!   is written from the per-tag layout in `gpu_opcode_tag` (and the order the
//!   WGSL evaluator reads the payload in), not from `serialize`.
//! - The tag numbers are the contract between the CPU encoder and the WGSL
//!   evaluator: the constants the emitted WGSL declares must equal the Rust
//!   constants, and the tags must be the contiguous range `0..18`.
//! - `opcode_word_count` must equal the payload length of the reference
//!   encoder for every tag and be `None` outside the range.
//! - Stack discipline: `stack_effect` must be the arity table of the
//!   `ColorOp` doc, and `validate` / `deserialize` must report the first
//!   underflow, an unbalanced end, an unknown tag, a truncated payload and an
//!   unknown palette source.
//! - Round trip: `deserialize(serialize(compile(n)))` evaluates bit for bit
//!   like the recursive tree walker `n.eval`, which shares no code path with
//!   the stack machine.
//! - The DSL builders build the documented node and evaluate to the closed
//!   form of their law.
//!
//! Author: Moroya Sakamoto

use alice_sdf::npr::compiled_color::{
    emit_wgsl_bytecode_evaluator, gpu_opcode_tag as t, gpu_palette_source_tag as ps,
    opcode_word_count, ColorOp, CompiledColorPipeline, DeserializeError, GpuColorProgram,
    SerializeError, StackError,
};
use alice_sdf::npr::dsl::{NprColorContext, NprColorNode, PaletteSource};
use glam::{Vec2, Vec3};

const fn bits3(v: Vec3) -> [u32; 3] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]
}

/// Reference encoder: one instruction from the per-tag layout.
fn reference_words(op: &ColorOp) -> Vec<u32> {
    let mut w = Vec::new();
    match op {
        ColorOp::PushConstant(c) => {
            w.push(0);
            w.extend(bits3(*c));
        }
        ColorOp::Toon { bands } => w.extend([1, *bands]),
        ColorOp::SoftToon { bands, smoothness } => w.extend([2, *bands, smoothness.to_bits()]),
        ColorOp::TwoTone { threshold } => w.extend([3, threshold.to_bits()]),
        ColorOp::Multiply => w.push(4),
        ColorOp::Add => w.push(5),
        ColorOp::Scale { factor } => w.extend([6, factor.to_bits()]),
        ColorOp::OutlineOver { outline, alpha } => {
            w.push(7);
            w.extend(bits3(*outline));
            w.push(alpha.to_bits());
        }
        ColorOp::Fresnel { edge, power } => {
            w.push(8);
            w.extend(bits3(*edge));
            w.push(power.to_bits());
        }
        ColorOp::Saturate { factor } => w.extend([9, factor.to_bits()]),
        ColorOp::Bloom {
            threshold,
            intensity,
        } => w.extend([10, threshold.to_bits(), intensity.to_bits()]),
        ColorOp::PosterizeColor { levels } => w.extend([11, *levels]),
        ColorOp::Vignette { radius, softness } => {
            w.extend([12, radius.to_bits(), softness.to_bits()]);
        }
        ColorOp::Palette3 { source, c0, c1, c2 } => {
            w.extend([13, source_tag(*source)]);
            for c in [c0, c1, c2] {
                w.extend(bits3(*c));
            }
        }
        ColorOp::Palette5 {
            source,
            c0,
            c1,
            c2,
            c3,
            c4,
        } => {
            w.extend([14, source_tag(*source)]);
            for c in [c0, c1, c2, c3, c4] {
                w.extend(bits3(*c));
            }
        }
        ColorOp::Hatch {
            angle_rad,
            density,
            thickness,
            ink,
        } => {
            w.extend([
                15,
                angle_rad.to_bits(),
                density.to_bits(),
                thickness.to_bits(),
            ]);
            w.extend(bits3(*ink));
        }
        ColorOp::Tonemap { exposure } => w.extend([16, exposure.to_bits()]),
        ColorOp::SpeedLine {
            focus,
            count,
            thickness,
            ink,
        } => {
            w.extend([
                17,
                focus.x.to_bits(),
                focus.y.to_bits(),
                *count,
                thickness.to_bits(),
            ]);
            w.extend(bits3(*ink));
        }
        ColorOp::Fallback(_) => panic!("Fallback has no wire form"),
    }
    w
}

const fn source_tag(s: PaletteSource) -> u32 {
    match s {
        PaletteSource::NDotL => 0,
        PaletteSource::NDotV => 1,
        PaletteSource::Sdf => 2,
        PaletteSource::UvY => 3,
        PaletteSource::TimeCycle => 4,
    }
}

const RED: Vec3 = Vec3::new(0.9, 0.1, 0.05);

/// One instance of every opcode with distinct payload values.
fn every_op() -> Vec<ColorOp> {
    vec![
        ColorOp::PushConstant(Vec3::new(0.25, -1.5, 3.0)),
        ColorOp::Toon { bands: 5 },
        ColorOp::SoftToon {
            bands: 3,
            smoothness: 0.07,
        },
        ColorOp::TwoTone { threshold: 0.4 },
        ColorOp::Multiply,
        ColorOp::Add,
        ColorOp::Scale { factor: -2.5 },
        ColorOp::OutlineOver {
            outline: Vec3::new(0.01, 0.02, 0.03),
            alpha: 0.75,
        },
        ColorOp::Fresnel {
            edge: Vec3::new(1.0, 0.5, 0.25),
            power: 3.5,
        },
        ColorOp::Saturate { factor: 1.25 },
        ColorOp::Bloom {
            threshold: 0.6,
            intensity: 1.75,
        },
        ColorOp::PosterizeColor { levels: 6 },
        ColorOp::Vignette {
            radius: 0.3,
            softness: 0.2,
        },
        ColorOp::Palette3 {
            source: PaletteSource::Sdf,
            c0: RED,
            c1: Vec3::Y,
            c2: Vec3::Z,
        },
        ColorOp::Palette5 {
            source: PaletteSource::TimeCycle,
            c0: Vec3::X,
            c1: Vec3::Y,
            c2: Vec3::Z,
            c3: RED,
            c4: Vec3::ONE,
        },
        ColorOp::Hatch {
            angle_rad: 0.7,
            density: 9.0,
            thickness: 0.12,
            ink: Vec3::new(0.1, 0.1, 0.2),
        },
        ColorOp::Tonemap { exposure: 1.5 },
        ColorOp::SpeedLine {
            focus: Vec2::new(0.5, 0.4),
            count: 32,
            thickness: 0.2,
            ink: Vec3::ZERO,
        },
    ]
}

#[test]
fn every_opcode_encodes_to_its_layout() {
    let ops = every_op();
    let mut tags = Vec::new();
    for op in &ops {
        let want = reference_words(op);
        let got = CompiledColorPipeline {
            ops: vec![op.clone()],
        }
        .serialize()
        .expect("no Fallback");
        assert_eq!(got.as_words(), want.as_slice(), "{op:?}");
        assert_eq!(got.words, want);
        assert_eq!(got.byte_len(), 4 * want.len());
        assert_eq!(
            opcode_word_count(want[0]),
            Some(want.len() - 1),
            "payload length of tag {}",
            want[0]
        );
        tags.push(want[0]);
    }
    tags.sort_unstable();
    assert_eq!(tags, (0..18).collect::<Vec<u32>>(), "18 contiguous tags");
    for tag in [18, 19, 255, u32::MAX] {
        assert_eq!(opcode_word_count(tag), None, "tag {tag}");
    }
}

#[test]
fn rust_tags_equal_the_constants_of_the_emitted_wgsl() {
    let wgsl = emit_wgsl_bytecode_evaluator();
    let consts: Vec<(&str, u32)> = vec![
        ("ALICE_OP_PUSH_CONSTANT", t::PUSH_CONSTANT),
        ("ALICE_OP_TOON", t::TOON),
        ("ALICE_OP_SOFT_TOON", t::SOFT_TOON),
        ("ALICE_OP_TWO_TONE", t::TWO_TONE),
        ("ALICE_OP_MULTIPLY", t::MULTIPLY),
        ("ALICE_OP_ADD", t::ADD),
        ("ALICE_OP_SCALE", t::SCALE),
        ("ALICE_OP_OUTLINE_OVER", t::OUTLINE_OVER),
        ("ALICE_OP_FRESNEL", t::FRESNEL),
        ("ALICE_OP_SATURATE", t::SATURATE),
        ("ALICE_OP_BLOOM", t::BLOOM),
        ("ALICE_OP_POSTERIZE_COLOR", t::POSTERIZE_COLOR),
        ("ALICE_OP_VIGNETTE", t::VIGNETTE),
        ("ALICE_OP_PALETTE3", t::PALETTE3),
        ("ALICE_OP_PALETTE5", t::PALETTE5),
        ("ALICE_OP_HATCH", t::HATCH),
        ("ALICE_OP_TONEMAP", t::TONEMAP),
        ("ALICE_OP_SPEED_LINE", t::SPEED_LINE),
        ("ALICE_PS_N_DOT_L", ps::N_DOT_L),
        ("ALICE_PS_N_DOT_V", ps::N_DOT_V),
        ("ALICE_PS_SDF", ps::SDF),
        ("ALICE_PS_UV_Y", ps::UV_Y),
        ("ALICE_PS_TIME_CYCLE", ps::TIME_CYCLE),
    ];
    let mut compared = 0;
    for (name, value) in &consts {
        let decl = format!("const {name}: u32 = ");
        let start = wgsl
            .find(&decl)
            .unwrap_or_else(|| panic!("{name} not declared"));
        let rest = &wgsl[start + decl.len()..];
        let lit = &rest[..rest.find('u').expect("u32 literal suffix")];
        assert_eq!(lit.parse::<u32>().unwrap(), *value, "{name}");
        compared += 1;
    }
    assert_eq!(compared, consts.len());
    // The opcode tags are also the reference encoder's numbers.
    for (op, (_, tag)) in every_op().iter().zip(&consts) {
        assert_eq!(reference_words(op)[0], *tag);
    }
    let sources = [
        PaletteSource::NDotL,
        PaletteSource::NDotV,
        PaletteSource::Sdf,
        PaletteSource::UvY,
        PaletteSource::TimeCycle,
    ];
    for (s, (_, tag)) in sources.iter().zip(&consts[18..]) {
        assert_eq!(source_tag(*s), *tag);
    }
}

#[test]
fn stack_effect_is_the_documented_arity_table() {
    let want = [
        (0, 1), // PushConstant
        (2, 1), // Toon
        (2, 1), // SoftToon
        (2, 1), // TwoTone
        (2, 1), // Multiply
        (2, 1), // Add
        (1, 1), // Scale
        (1, 1), // OutlineOver
        (1, 1), // Fresnel
        (1, 1), // Saturate
        (1, 1), // Bloom
        (1, 1), // PosterizeColor
        (1, 1), // Vignette
        (0, 1), // Palette3
        (0, 1), // Palette5
        (1, 1), // Hatch
        (1, 1), // Tonemap
        (1, 1), // SpeedLine
    ];
    let ops = every_op();
    assert_eq!(ops.len(), want.len());
    for (op, w) in ops.iter().zip(want) {
        assert_eq!(op.stack_effect(), w, "{op:?}");
    }
    assert_eq!(
        ColorOp::Fallback(Box::new(NprColorNode::Constant(Vec3::ONE))).stack_effect(),
        (0, 1)
    );
}

#[test]
fn validate_reports_the_first_stack_error() {
    let push = || ColorOp::PushConstant(Vec3::ONE);
    let p = |ops: Vec<ColorOp>| CompiledColorPipeline { ops };
    assert_eq!(
        p(vec![ColorOp::Add]).validate(),
        Err(StackError::Underflow {
            index: 0,
            needs: 2,
            depth: 0
        })
    );
    assert_eq!(
        p(vec![
            push(),
            ColorOp::Multiply,
            ColorOp::Scale { factor: 1.0 }
        ])
        .validate(),
        Err(StackError::Underflow {
            index: 1,
            needs: 2,
            depth: 1
        })
    );
    assert_eq!(
        p(vec![push(), push()]).validate(),
        Err(StackError::Unbalanced { depth: 2 })
    );
    assert_eq!(
        p(vec![]).validate(),
        Err(StackError::Unbalanced { depth: 0 })
    );
    assert_eq!(p(vec![push(), push(), ColorOp::Add]).validate(), Ok(()));
}

#[test]
fn deserialize_rejects_malformed_streams() {
    let prog = |words: Vec<u32>| GpuColorProgram { words };
    let one = 1.0f32.to_bits();
    assert_eq!(
        prog(vec![0, one, one, one, 99]).deserialize().unwrap_err(),
        DeserializeError::UnknownOpcode {
            word_offset: 4,
            tag: 99
        }
    );
    assert_eq!(
        prog(vec![0, one, one]).deserialize().unwrap_err(),
        DeserializeError::Truncated {
            word_offset: 0,
            expected_payload: 3
        }
    );
    let mut pal = vec![13, 7];
    pal.extend([one; 9]);
    assert_eq!(
        prog(pal).deserialize().unwrap_err(),
        DeserializeError::UnknownPaletteSource {
            word_offset: 1,
            tag: 7
        }
    );
    assert_eq!(
        prog(vec![5]).deserialize().unwrap_err(),
        DeserializeError::Stack(StackError::Underflow {
            index: 0,
            needs: 2,
            depth: 0
        })
    );
    let fallback = CompiledColorPipeline {
        ops: vec![
            ColorOp::PushConstant(Vec3::ONE),
            ColorOp::Fallback(Box::new(NprColorNode::Constant(Vec3::ONE))),
            ColorOp::Add,
        ],
    };
    assert_eq!(fallback.native_op_count(), 2);
    assert_eq!(
        fallback.serialize().unwrap_err(),
        SerializeError::UnsupportedFallback {
            instruction_index: 1
        }
    );
}

const fn ctx(n_dot_l: f32, n_dot_v: f32, sdf: f32, uv: Vec2, time: f32) -> NprColorContext {
    NprColorContext {
        sdf,
        normal: Vec3::Z,
        view: Vec3::new(0.0, 0.0, n_dot_v),
        light: Vec3::new(0.0, 0.0, n_dot_l),
        uv,
        time,
    }
}

const fn c(r: f32, g: f32, b: f32) -> NprColorNode {
    NprColorNode::Constant(Vec3::new(r, g, b))
}

/// A tree that uses every node kind, composed with the builder helpers.
fn every_node() -> NprColorNode {
    let toon = NprColorNode::Toon {
        shadow: Vec3::new(0.1, 0.1, 0.3),
        light: Vec3::new(0.9, 0.8, 0.6),
        bands: 4,
    };
    let soft = NprColorNode::SoftToon {
        shadow: Vec3::ZERO,
        light: Vec3::ONE,
        bands: 3,
        smoothness: 0.1,
    };
    let two = NprColorNode::TwoTone {
        shadow: Vec3::new(0.2, 0.0, 0.0),
        light: Vec3::new(0.0, 0.2, 0.0),
        threshold: 0.5,
    };
    let p3 = NprColorNode::Palette3 {
        source: PaletteSource::UvY,
        c0: Vec3::X,
        c1: Vec3::Y,
        c2: Vec3::Z,
    };
    let p5 = NprColorNode::Palette5 {
        source: PaletteSource::NDotV,
        c0: Vec3::ZERO,
        c1: Vec3::X,
        c2: Vec3::Y,
        c3: Vec3::Z,
        c4: Vec3::ONE,
    };
    toon.multiply(soft)
        .plus(two)
        .scale(0.8)
        .with_outline(Vec3::splat(0.02), 0.3)
        .with_fresnel(Vec3::ONE, 2.0)
        .plus(NprColorNode::Saturate {
            child: Box::new(p3),
            factor: 0.7,
        })
        .bloom(0.2, 1.1)
        .posterize(8)
        .vignetted(0.4, 0.2)
        .multiply(p5.plus(c(0.5, 0.5, 0.5)))
        .with_hatch(0.3, 6.0, 0.1, Vec3::splat(0.05))
        .tonemap_reinhard(1.2)
        .with_speed_lines(Vec2::new(0.5, 0.5), 16, 0.15, Vec3::ZERO)
}

#[test]
fn round_trip_evaluates_like_the_tree_walker_bit_for_bit() {
    let node = every_node();
    let compiled = CompiledColorPipeline::compile(&node);
    assert_eq!(compiled.validate(), Ok(()));
    assert_eq!(
        compiled.native_op_count(),
        compiled.ops.len(),
        "no Fallback"
    );
    let program = compiled.serialize().expect("serialize");
    let decoded = program.deserialize().expect("deserialize");
    assert_eq!(decoded.serialize().expect("re-serialize"), program);
    let mut compared = 0;
    for i in 0..24 {
        for j in 0..24 {
            let s = i as f32 / 23.0;
            let u = j as f32 / 23.0;
            let cx = ctx(
                s.mul_add(2.0, -1.0),
                1.0 - s,
                u - 0.5,
                Vec2::new(u, s),
                3.0 * s,
            );
            let want = node.eval(&cx);
            for got in [compiled.eval(&cx), decoded.eval(&cx)] {
                assert_eq!(
                    [got.x.to_bits(), got.y.to_bits(), got.z.to_bits()],
                    [want.x.to_bits(), want.y.to_bits(), want.z.to_bits()],
                    "at {cx:?}"
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 2 * 24 * 24);
}

#[test]
fn builders_build_their_node_and_law() {
    let cx = ctx(0.5, 0.5, 0.0, Vec2::new(0.25, 0.75), 0.0);
    let a = Vec3::new(0.2, 0.4, 0.8);
    let b = Vec3::new(0.5, 0.25, 2.0);

    // Arithmetic: componentwise product, sum, scalar multiple.
    assert_eq!(c(a.x, a.y, a.z).multiply(c(b.x, b.y, b.z)).eval(&cx), a * b);
    assert_eq!(c(a.x, a.y, a.z).plus(c(b.x, b.y, b.z)).eval(&cx), a + b);
    assert_eq!(c(a.x, a.y, a.z).scale(3.0).eval(&cx), a * 3.0);

    // Posterize: floor(clamp(c) * L) / (L - 1), clamped; levels below 2 become 2.
    let NprColorNode::PosterizeColor { levels, .. } = c(0.0, 0.0, 0.0).posterize(1) else {
        panic!("posterize builds PosterizeColor");
    };
    assert_eq!(levels, 2);
    let q = |x: f32, l: f32| ((x.clamp(0.0, 1.0) * l).floor() / (l - 1.0)).min(1.0);
    assert_eq!(
        c(0.3, 0.55, 0.99).posterize(3).eval(&cx),
        Vec3::new(q(0.3, 3.0), q(0.55, 3.0), q(0.99, 3.0))
    );
    assert_eq!(
        c(0.3, 0.55, 0.99).posterize(3).eval(&cx),
        Vec3::new(0.0, 0.5, 1.0)
    );

    // Bloom: the colour times intensity when its largest channel passes the
    // threshold, black otherwise.
    assert_eq!(
        c(0.7, 0.1, 0.1).bloom(0.6, 2.0).eval(&cx),
        Vec3::new(0.7, 0.1, 0.1) * 2.0
    );
    assert_eq!(c(0.5, 0.1, 0.1).bloom(0.6, 2.0).eval(&cx), Vec3::ZERO);

    // Hatch at angle 0: lines run along u, the phase is v * density; a line
    // covers the phases within `thickness` of an integer.
    let ink = Vec3::new(0.0, 0.0, 1.0);
    let hatch = c(1.0, 1.0, 1.0).with_hatch(0.0, 4.0, 0.1, ink);
    let at_v = |v: f32| hatch.eval(&ctx(0.0, 0.0, 0.0, Vec2::new(0.37, v), 0.0));
    assert_eq!(at_v(0.25 + 0.01), ink, "phase 1.04: on a line");
    assert_eq!(at_v(0.125), Vec3::ONE, "phase 0.5: between lines");
    assert_eq!(at_v(0.5 - 0.01), ink, "phase 1.96: on a line");

    // Speed lines: the phase is (atan2 + pi) * count / 2 pi around the focus;
    // at count 4 the line centres sit on the axes.
    let speed = c(1.0, 1.0, 1.0).with_speed_lines(Vec2::new(0.5, 0.5), 4, 0.1, ink);
    let at = |x: f32, y: f32| speed.eval(&ctx(0.0, 0.0, 0.0, Vec2::new(x, y), 0.0));
    assert_eq!(at(0.9, 0.5), ink, "angle 0: phase 2, on a line");
    assert_eq!(at(0.5, 0.9), ink, "angle pi/2: phase 3, on a line");
    let diag = std::f32::consts::FRAC_1_SQRT_2 * 0.4;
    assert_eq!(
        at(0.5 + diag, 0.5 + diag),
        Vec3::ONE,
        "angle pi/4: phase 2.5"
    );
    assert_eq!(at(0.5, 0.5), Vec3::ONE, "no line at the focus");
}
