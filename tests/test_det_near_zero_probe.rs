//! Dense near-zero probe: det-math 0.5's sin/cos change is at ±0, tiny, and the
//! neighbourhoods of k·π/2. `twist`/`bend` call `alice_det_math::sin_cos` at
//! eval time (angle = `p.y * strength`), so this drives every compiled evaluator
//! (scalar, f32x8 SIMD, BVH, and the cranelift JIT hand-port) through exactly
//! those angles and asserts they are bit-identical — the test that would FAIL if
//! the 0.5 arithmetic reached SDF eval and any evaluator disagreed. It also pins
//! det-math directly (sin_cos at the exact near-zero inputs, scalar vs SIMD) and
//! shows teeth: det-math and the platform libm disagree at these angles, so a
//! single evaluator silently using libm would break the bit-identity above.
#![cfg(feature = "jit")]

use alice_sdf::compiled::{
    eval_compiled, eval_compiled_batch_simd, eval_compiled_bvh, CompiledSdf, CompiledSdfBvh,
};
use alice_sdf::prelude::*;

fn bits(v: f32) -> String {
    format!("{:08x}", v.to_bits())
}

/// f32 angles where det-math 0.5's near-zero / quadrant behaviour lives.
fn probe_angles() -> Vec<f32> {
    use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    // sign-aware neighbour: toward +inf for n>0, toward -inf for n<0, crossing 0 correctly.
    // Bit-stepping, not f32::next_up/next_down (those need Rust 1.86 > MSRV 1.85).
    fn step(mut a: f32, n: i32) -> f32 {
        let up = n > 0;
        for _ in 0..n.unsigned_abs() {
            if a.is_nan() {
                break;
            }
            let bits = a.to_bits();
            a = if a == 0.0 {
                f32::from_bits(if up { 1 } else { 0x8000_0001 })
            } else if up == (bits >> 31 == 0) {
                f32::from_bits(bits + 1)
            } else {
                f32::from_bits(bits - 1)
            };
        }
        a
    }
    let mut v = vec![0.0_f32, -0.0, f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 1e-7, -1e-7, 1e-4];
    for k in -4..=4 {
        let base = (k as f32) * FRAC_PI_2;
        for d in -2..=2 {
            v.push(step(base, d));
        }
    }
    for a in [PI, -PI, FRAC_PI_4, 2.0 * PI, -2.0 * PI] {
        for d in -1..=1 {
            v.push(step(a, d));
        }
    }
    v.retain(|x| x.is_finite());
    v
}

#[test]
fn near_zero_twist_is_bit_identical_across_evaluators() {
    // strength 1.0 → internal twist angle == p.y, so p.y walks the probe angles.
    let node = SdfNode::box3d(0.8, 0.8, 0.8).twist(1.0);
    let compiled = CompiledSdf::try_compile(&node).expect("compile");
    let bvh = CompiledSdfBvh::try_compile(&node).expect("compile bvh");
    let jit = alice_sdf::compiled::jit::JitSimdSdf::compile(&compiled).expect("jit compile");

    let pts: Vec<Vec3> = probe_angles()
        .iter()
        .flat_map(|&a| [Vec3::new(0.3, a, 0.4), Vec3::new(-0.6, a, 0.2)])
        .collect();
    let simd = eval_compiled_batch_simd(&compiled, &pts);
    let jit_out = {
        let xs: Vec<f32> = pts.iter().map(|p| p.x).collect();
        let ys: Vec<f32> = pts.iter().map(|p| p.y).collect();
        let zs: Vec<f32> = pts.iter().map(|p| p.z).collect();
        jit.eval_batch(&xs, &ys, &zs)
    };

    let mut compared = 0usize;
    let mut fail: Vec<String> = Vec::new();
    for (i, &p) in pts.iter().enumerate() {
        let s = eval_compiled(&compiled, p);
        let paths = [("simd", simd[i]), ("bvh", eval_compiled_bvh(&bvh, p)), ("jit", jit_out[i])];
        for (name, v) in paths {
            compared += 1;
            if v.to_bits() != s.to_bits() {
                fail.push(format!("{name} @ ({},{},{}): scalar={} {name}={}", p.x, p.y, p.z, bits(s), bits(v)));
            }
        }
    }
    assert!(compared > 0, "0 compared — probe built no points");
    assert!(fail.is_empty(), "evaluators diverged on near-zero twist ({} of {} compared):\n{}", fail.len(), compared, fail.join("\n"));
    eprintln!("near-zero twist: {compared} comparisons bit-identical across scalar/simd/bvh/jit over {} points", pts.len());
}

#[test]
fn twist_by_zero_is_identity_through_det_math() {
    // At p.y = 0 the twist angle is 0; det-math sin_cos(0) = (0, 1), so twist is a
    // no-op. This ties the eval to det-math's exact near-zero value: if sin(0)≠0 or
    // cos(0)≠1 in any evaluator the twisted and plain distances would differ.
    let twisted = CompiledSdf::try_compile(&SdfNode::box3d(0.8, 0.8, 0.8).twist(1.0)).unwrap();
    let plain = CompiledSdf::try_compile(&SdfNode::box3d(0.8, 0.8, 0.8)).unwrap();
    let mut compared = 0usize;
    for &p in &[Vec3::new(0.3, 0.0, 0.4), Vec3::new(-0.6, -0.0, 0.2), Vec3::new(1.2, 0.0, -0.9)] {
        compared += 1;
        assert_eq!(eval_compiled(&twisted, p).to_bits(), eval_compiled(&plain, p).to_bits(),
            "twist-by-0 not identity at {p:?}: det-math sin_cos(0) must be (0,1)");
    }
    assert!(compared > 0, "0 compared");
    // det-math 0.5 directly at the exact near-zero inputs: pin the values the eval relies on.
    assert_eq!(alice_det_math::sin_cos(0.0_f32), (0.0, 1.0), "det-math sin_cos(0) must be (0,1)");
    let (s_neg0, c_neg0) = alice_det_math::sin_cos(-0.0_f32);
    assert_eq!((c_neg0.to_bits()), (1.0_f32).to_bits(), "cos(-0) must be +1");
    let _ = s_neg0;
}

#[test]
fn probe_angles_are_det_math_sensitive_teeth() {
    // Teeth: det-math and the platform libm must DISAGREE at some probe angle, else
    // the bit-identity test above is vacuous — a libm swap in one evaluator would be
    // caught precisely because det-math's near-zero/quadrant arithmetic differs here.
    let mut disagree = 0usize;
    let mut compared = 0usize;
    for a in probe_angles() {
        let (sd, cd) = alice_det_math::sin_cos(a);
        // This teeth check exists to prove the platform libm and det-math DISAGREE at
        // these angles, so a stray libm call in any evaluator would break the bit-identity
        // test above. Calling libm here is the whole point, hence the scoped allow.
        #[allow(clippy::disallowed_methods)]
        let (sl, cl) = (a.sin(), a.cos()); // platform libm
        compared += 1;
        if sd.to_bits() != sl.to_bits() || cd.to_bits() != cl.to_bits() {
            disagree += 1;
        }
    }
    assert!(compared > 0, "0 compared");
    assert!(disagree > 0, "det-math and libm agreed on every probe angle — the probe would not catch a libm swap (no teeth)");
    eprintln!("teeth: det-math ≠ libm on {disagree} of {compared} probe angles");
}
