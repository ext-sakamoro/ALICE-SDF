//! Colour by surface normal family and by region (`npr::dsl::NprColorNode::{FamilyPalette, RegionNoise}`,
//! `npr::compiled_color::ColorOp::{FamilyPalette, RegionNoise}`): the closed forms of the two laws, their
//! degenerate inputs, and the bytecode round trip, all checked against values computed here, never against
//! the implementation's own output.
//!
//! - `FamilyPalette`: the family is the index of the largest `n · family_normal` (strict, so a tie goes to the
//!   lowest index); the colour is that family's palette entry; no families = black.
//! - `RegionNoise`: `base + region_amp * (2 h - 1) + noise_amp * v` with `h` the crate's lattice-corner hash of
//!   the cell `floor(position / cell)` (constant inside a cell) and `v` the crate's value noise in `[-1, 1]` at
//!   `position / cell`, so the region term is piecewise constant and the whole offset is bounded by
//!   `|region_amp| + |noise_amp|` per channel.
//! - `validate` rejects a family count of 0 or above 8 and a cell that is not positive, before evaluation.
//! - The bytecode of the two opcodes (tags 18 and 19) round-trips and evaluates bit for bit like the tree walker.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm serves as an independent reference"
)]

use alice_sdf::modifiers::{cell_hash_3d, hash_noise_3d};
use alice_sdf::npr::compiled_color::{
    gpu_opcode_tag as t, opcode_word_count, ColorOp, CompiledColorPipeline, StackError,
};
use alice_sdf::npr::dsl::{NprColorContext, NprColorNode};
use glam::Vec3;

const fn ctx(normal: Vec3, position: Vec3) -> NprColorContext {
    NprColorContext::new(normal, Vec3::Z, Vec3::Z).with_position(position)
}

const fn bits(v: Vec3) -> [u32; 3] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]
}

const FAMILIES: [(Vec3, Vec3); 4] = [
    (Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.9, 0.8, 0.7)), // up: light
    (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.5, 0.3, 0.2)), // +x: brown
    (Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.2, 0.3, 0.5)), // +z: blue
    (Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.1, 0.1, 0.1)), // down: dark
];

fn family_node() -> NprColorNode {
    NprColorNode::FamilyPalette {
        families: FAMILIES.to_vec(),
    }
}

#[test]
fn the_family_is_the_largest_normal_dot_and_a_tie_goes_to_the_lowest_index() {
    let node = family_node();
    // each family normal picks its own colour
    for (n, c) in FAMILIES {
        assert_eq!(bits(node.eval(&ctx(n, Vec3::ZERO))), bits(c), "normal {n}");
    }
    // a normal between +x and +z, nearer +x, is brown; nearer +z, blue (cos 20 deg vs cos 70 deg)
    let near_x = Vec3::new(20f32.to_radians().cos(), 0.0, 20f32.to_radians().sin());
    assert_eq!(
        bits(node.eval(&ctx(near_x, Vec3::ZERO))),
        bits(FAMILIES[1].1)
    );
    let near_z = Vec3::new(70f32.to_radians().cos(), 0.0, 70f32.to_radians().sin());
    assert_eq!(
        bits(node.eval(&ctx(near_z, Vec3::ZERO))),
        bits(FAMILIES[2].1)
    );
    // the exact diagonal between +x and +z has the same dot with both: the lowest index (brown) wins
    let d = Vec3::new(
        core::f32::consts::FRAC_1_SQRT_2,
        0.0,
        core::f32::consts::FRAC_1_SQRT_2,
    );
    assert_eq!(
        d.dot(FAMILIES[1].0).to_bits(),
        d.dot(FAMILIES[2].0).to_bits()
    );
    assert_eq!(bits(node.eval(&ctx(d, Vec3::ZERO))), bits(FAMILIES[1].1));
    // a normal pointing away from every family still picks the least-negative dot (-x is nearest to up/down/+z at 0)
    let away = Vec3::new(-1.0, 0.0, 0.0);
    assert_eq!(
        bits(node.eval(&ctx(away, Vec3::ZERO))),
        bits(FAMILIES[0].1),
        "dots 0, -1, 0, 0: index 0"
    );
}

#[test]
fn no_families_is_black_and_more_than_eight_families_fall_back_to_the_tree_walker() {
    let empty = NprColorNode::FamilyPalette {
        families: Vec::new(),
    };
    assert_eq!(
        bits(empty.eval(&ctx(Vec3::Y, Vec3::ZERO))),
        bits(Vec3::ZERO)
    );
    let compiled = CompiledColorPipeline::compile(&empty);
    assert_eq!(
        compiled.validate(),
        Err(StackError::Payload {
            index: 0,
            what: "family count"
        })
    );
    let nine = NprColorNode::FamilyPalette {
        families: (0..9)
            .map(|i| {
                (
                    Vec3::new(i as f32, 1.0, 0.0).normalize(),
                    Vec3::splat(i as f32 / 9.0),
                )
            })
            .collect(),
    };
    let compiled = CompiledColorPipeline::compile(&nine);
    assert_eq!(
        compiled.fallback_op_count(),
        1,
        "nine families have no 8-slot opcode"
    );
    assert!(compiled.serialize().is_err(), "a Fallback has no wire form");
    // and the tree walker still picks family 8 for its own normal
    let n8 = Vec3::new(8.0, 1.0, 0.0).normalize();
    assert_eq!(
        bits(nine.eval(&ctx(n8, Vec3::ZERO))),
        bits(Vec3::splat(8.0 / 9.0))
    );
    assert_eq!(
        bits(compiled.eval(&ctx(n8, Vec3::ZERO))),
        bits(Vec3::splat(8.0 / 9.0))
    );
}

fn region_node(base: Vec3) -> NprColorNode {
    NprColorNode::RegionNoise {
        base: Box::new(NprColorNode::Constant(base)),
        cell: 0.04,
        region_amp: Vec3::new(0.10, 0.05, 0.02),
        noise_amp: Vec3::new(0.04, 0.04, 0.04),
        seed: 11,
    }
}

#[test]
fn the_region_term_is_the_cell_hash_and_constant_inside_a_cell_and_the_noise_is_the_crate_law() {
    let base = Vec3::new(0.5, 0.4, 0.3);
    let node = region_node(base);
    let (cell, region_amp, noise_amp, seed) = (
        0.04f32,
        Vec3::new(0.10, 0.05, 0.02),
        Vec3::splat(0.04),
        11u32,
    );
    let mut checked = 0;
    for (i, p) in [
        Vec3::new(0.013, 0.021, -0.007),
        Vec3::new(0.039, 0.0, 0.0),
        Vec3::new(-0.081, 0.123, 0.456),
        Vec3::new(1.2345, -0.5, 0.0404),
    ]
    .into_iter()
    .enumerate()
    {
        let q = p / cell;
        let h = cell_hash_3d(q.floor(), seed);
        let v = hash_noise_3d(q, seed.wrapping_add(1));
        let want = base + region_amp * (2.0 * h - 1.0) + noise_amp * v;
        let got = node.eval(&ctx(Vec3::Y, p));
        assert_eq!(bits(got), bits(want), "point {i}: {p}");
        assert!((0.0..1.0).contains(&h) && (-1.0..=1.0).contains(&v));
        checked += 1;
    }
    assert_eq!(checked, 4);
    // inside one cell the region term does not move: two points of cell (0, 0, 0) differ only by the noise term
    let a = node.eval(&ctx(Vec3::Y, Vec3::new(0.001, 0.002, 0.003)));
    let b = node.eval(&ctx(Vec3::Y, Vec3::new(0.030, 0.035, 0.010)));
    let h0 = cell_hash_3d(Vec3::ZERO, seed);
    for (x, p) in [
        (a, Vec3::new(0.001, 0.002, 0.003)),
        (b, Vec3::new(0.030, 0.035, 0.010)),
    ] {
        // the same cell hash h0 for both points, only the smooth term moves (compared as the full expression:
        // subtracting the terms back out would round)
        let v = hash_noise_3d(p / cell, seed.wrapping_add(1));
        assert_eq!(
            bits(x),
            bits(base + region_amp * (2.0 * h0 - 1.0) + noise_amp * v)
        );
    }
    // across the cell boundary the region term takes the neighbouring cell's hash, which differs for this seed
    let h1 = cell_hash_3d(Vec3::new(1.0, 0.0, 0.0), seed);
    assert_ne!(h0.to_bits(), h1.to_bits());
    // the offset is bounded per channel by |region_amp| + |noise_amp|
    for k in 0..200 {
        let p = Vec3::new(
            k as f32 * 0.0137,
            (k as f32 * 0.0071).sin(),
            -0.2 + k as f32 * 0.003,
        );
        let off = node.eval(&ctx(Vec3::Y, p)) - base;
        for c in 0..3 {
            assert!(
                off[c].abs() <= region_amp[c] + noise_amp[c] + 1e-6,
                "k {k} channel {c}: {off}"
            );
        }
    }
}

#[test]
fn a_non_positive_cell_is_rejected_before_evaluation() {
    for cell in [0.0f32, -0.04, f32::NEG_INFINITY, f32::NAN] {
        let node = NprColorNode::RegionNoise {
            base: Box::new(NprColorNode::Constant(Vec3::ONE)),
            cell,
            region_amp: Vec3::ONE,
            noise_amp: Vec3::ONE,
            seed: 1,
        };
        let compiled = CompiledColorPipeline::compile(&node);
        assert_eq!(
            compiled.validate(),
            Err(StackError::Payload {
                index: 1,
                what: "region cell"
            }),
            "cell {cell}"
        );
    }
}

#[test]
fn the_two_opcodes_have_their_layout_and_round_trip_bit_for_bit() {
    let fam = ColorOp::FamilyPalette {
        count: 4,
        normals: [
            FAMILIES[0].0,
            FAMILIES[1].0,
            FAMILIES[2].0,
            FAMILIES[3].0,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
        ],
        colors: [
            FAMILIES[0].1,
            FAMILIES[1].1,
            FAMILIES[2].1,
            FAMILIES[3].1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
        ],
    };
    let reg = ColorOp::RegionNoise {
        cell: 0.04,
        region_amp: Vec3::new(0.10, 0.05, 0.02),
        noise_amp: Vec3::splat(0.04),
        seed: 11,
    };
    // reference layout, written from the tag table: [18, count, 8 normals, 8 colours] and [19, cell, region, noise, seed]
    let mut want_fam = vec![t::FAMILY_PALETTE, 4];
    if let ColorOp::FamilyPalette {
        normals, colors, ..
    } = &fam
    {
        for n in normals {
            want_fam.extend(bits(*n));
        }
        for c in colors {
            want_fam.extend(bits(*c));
        }
    }
    let want_reg = {
        let mut w = vec![t::REGION_NOISE, 0.04f32.to_bits()];
        w.extend(bits(Vec3::new(0.10, 0.05, 0.02)));
        w.extend(bits(Vec3::splat(0.04)));
        w.push(11);
        w
    };
    assert_eq!((t::FAMILY_PALETTE, t::REGION_NOISE), (18, 19));
    assert_eq!(opcode_word_count(18), Some(49));
    assert_eq!(opcode_word_count(19), Some(8));
    assert_eq!(fam.stack_effect(), (0, 1));
    assert_eq!(reg.stack_effect(), (1, 1));
    let pipeline = CompiledColorPipeline {
        ops: vec![fam, reg],
    };
    assert_eq!(pipeline.validate(), Ok(()));
    let program = pipeline.serialize().expect("serialize");
    let mut want = want_fam.clone();
    want.extend(want_reg);
    assert_eq!(program.as_words(), want.as_slice());
    let decoded = program.deserialize().expect("deserialize");
    assert_eq!(decoded.serialize().expect("re-serialize"), program);
    // the same program written as a DSL tree evaluates identically through the walker, the stack machine and the decoded stream
    let node = NprColorNode::RegionNoise {
        base: Box::new(family_node()),
        cell: 0.04,
        region_amp: Vec3::new(0.10, 0.05, 0.02),
        noise_amp: Vec3::splat(0.04),
        seed: 11,
    };
    let compiled = CompiledColorPipeline::compile(&node);
    assert_eq!(compiled.serialize().expect("compile serialize"), program);
    let mut compared = 0;
    for i in 0..16 {
        for j in 0..16 {
            let a = i as f32 / 15.0 * core::f32::consts::TAU;
            let b = (j as f32 / 15.0 - 0.5) * core::f32::consts::PI;
            let n = Vec3::new(b.cos() * a.cos(), b.sin(), b.cos() * a.sin());
            let p = Vec3::new(
                i as f32 * 0.0173 - 0.1,
                j as f32 * 0.0111,
                (i * j) as f32 * 0.0007,
            );
            let c = ctx(n, p);
            let walker = node.eval(&c);
            for got in [compiled.eval(&c), pipeline.eval(&c), decoded.eval(&c)] {
                assert_eq!(bits(got), bits(walker), "normal {n} position {p}");
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 3 * 16 * 16);
}
