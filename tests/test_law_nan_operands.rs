//! The Law's NaN rule for union / intersection across evaluators (ALICE-Law G2 / G11)
//!
//! The Law max / min (alice-det-math `max` / `min`) give the other operand when one is
//! NaN, and the canonical NaN when both are. A sphere of NaN radius makes one operand
//! NaN at every point; two make both NaN. The tree evaluator, the compiled (bytecode)
//! evaluator, the f32x8 evaluator and both JITs must give the same bits.
#![cfg(feature = "jit")]
use alice_sdf::compiled::jit::{JitCompiledSdf, JitSimdSdf};
use alice_sdf::compiled::{eval_compiled, eval_compiled_batch_simd, CompiledSdf};
use alice_sdf::prelude::*;

fn scenes() -> Vec<(&'static str, SdfNode)> {
    let nan = || SdfNode::sphere(f32::NAN);
    let one = || SdfNode::sphere(1.0);
    let odd = |bits: u32| SdfNode::sphere(f32::from_bits(bits));
    vec![
        ("union nan left", nan().union(one())),
        ("union nan right", one().union(nan())),
        ("intersection nan left", nan().intersection(one())),
        ("intersection nan right", one().intersection(nan())),
        ("subtraction nan", one().subtract(nan())),
        ("union both nan", nan().union(nan())),
        ("intersection both nan", nan().intersection(nan())),
        // NaNs with payloads other than the canonical one: the result is still canonical
        (
            "union both payload nan",
            odd(0x7fc0_0123).union(odd(0xffc0_0456)),
        ),
        (
            "intersection both payload nan",
            odd(0xffc0_0456).intersection(odd(0x7fc0_0123)),
        ),
    ]
}

const POINTS: [Vec3; 3] = [
    Vec3::ZERO,
    Vec3::new(0.3, -2.0, 0.7),
    Vec3::new(5.0, 5.0, -5.0),
];

#[test]
fn every_evaluator_gives_the_tree_bits_on_nan_operands() {
    let mut differ: Vec<String> = Vec::new();
    for (name, node) in scenes() {
        let compiled = CompiledSdf::compile(&node);
        let jit = JitCompiledSdf::compile(&node).expect("jit compile");
        let simd_jit = JitSimdSdf::compile(&compiled).expect("simd jit compile");
        let xs: Vec<f32> = POINTS.iter().map(|p| p.x).collect();
        let ys: Vec<f32> = POINTS.iter().map(|p| p.y).collect();
        let zs: Vec<f32> = POINTS.iter().map(|p| p.z).collect();
        let simd_jit_out = simd_jit.eval_batch(&xs, &ys, &zs);
        let simd_out = eval_compiled_batch_simd(&compiled, &POINTS);
        for (i, &p) in POINTS.iter().enumerate() {
            let t = eval(&node, p);
            let got = [
                ("compiled", eval_compiled(&compiled, p)),
                ("jit", jit.eval(p)),
                ("simd jit", simd_jit_out[i]),
                ("simd", simd_out[i]),
            ];
            for (what, v) in got {
                if v.to_bits() != t.to_bits() {
                    differ.extend([format!(
                        "{name} at {p}: tree {t} ({:#x}), {what} {v} ({:#x})",
                        t.to_bits(),
                        v.to_bits()
                    )]);
                }
            }
        }
    }
    assert!(
        differ.is_empty(),
        "{} disagreements:\n{}",
        differ.len(),
        differ.join("\n")
    );
}

#[test]
fn one_nan_operand_gives_the_other_and_two_give_the_canonical_nan() {
    for &p in &POINTS {
        let finite = eval(&SdfNode::sphere(1.0), p);
        for (name, node) in scenes() {
            let d = eval(&node, p);
            let want = if name.contains("both") {
                f32::NAN
            } else {
                finite
            };
            assert_eq!(d.to_bits(), want.to_bits(), "{name} at {p}: {d}");
        }
    }
}

/// The SIMD gradient of a field that is NaN leaves every lane as the canonical quiet NaN
/// (ALICE-Law G4), whatever sign the NaN took through the differences
#[test]
fn a_nan_gradient_is_the_canonical_nan() {
    use alice_sdf::compiled::{eval_distance_and_gradient_simd, eval_gradient_simd, Vec3x8};
    use wide::f32x8;
    let node = SdfNode::sphere(f32::from_bits(0xffc0_0456))
        .intersection(SdfNode::sphere(f32::from_bits(0x7fc0_0123)));
    let compiled = CompiledSdf::compile(&node);
    let p = Vec3x8 {
        x: f32x8::new([0.0, 0.3, 5.0, -1.0, 2.0, 0.1, -3.0, 0.7]),
        y: f32x8::new([0.0, -2.0, 5.0, 1.0, -2.0, 0.2, 3.0, 0.8]),
        z: f32x8::new([0.0, 0.7, -5.0, 0.5, 0.0, 0.3, 1.0, 0.9]),
    };
    let (gx, gy, gz) = eval_gradient_simd(&compiled, p, 1e-3);
    let (_, hx, hy, hz) = eval_distance_and_gradient_simd(&compiled, p, 1e-3);
    for (what, v) in [
        ("gx", gx),
        ("gy", gy),
        ("gz", gz),
        ("hx", hx),
        ("hy", hy),
        ("hz", hz),
    ] {
        for (l, x) in v.to_array().iter().enumerate() {
            assert_eq!(
                x.to_bits(),
                0x7fc0_0000,
                "{what} lane {l}: {x} ({:#x})",
                x.to_bits()
            );
        }
    }
}
