//! Parameter-buffer JIT: compile once, change shape parameters without
//! recompiling.
//!
//! Compiles an animated scene with `JitCompiledSdfDynamic` (scalar) and
//! `JitSimdSdfDynamic` (8-wide), then for a few frames rebuilds the tree with
//! new radii / offsets, pushes only the parameters (`update_params`) and
//! checks the result against a fresh compile and the bytecode interpreter.
//!
//! # Running
//! ```bash
//! cargo run --release --example jit_dynamic --features jit
//! ```
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "example scene and animation values; the crate output these examples show is computed by the library"
)]

use alice_sdf::compiled::eval_compiled;
use alice_sdf::compiled::jit::{
    extract_jit_params, extract_simd_params, JitCompiledSdf, JitCompiledSdfDynamic,
    JitSimdSdfDynamic,
};
use alice_sdf::prelude::*;
use alice_sdf::soa::SoAPoints;

/// The scene at animation time `t`: the structure is fixed, the numbers move.
fn frame(t: f32) -> SdfNode {
    SdfNode::sphere(0.5 + 0.2 * t.sin())
        .translate(t.cos(), 0.0, 0.0)
        .union(SdfNode::box3d(0.6, 0.3 + 0.1 * t, 0.4).twist(0.3 * t))
}

fn main() {
    println!("ALICE-SDF — dynamic-parameter JIT");
    println!("=================================");

    let mut scalar = JitCompiledSdfDynamic::compile(&frame(0.0)).expect("JIT compile");
    let mut simd =
        JitSimdSdfDynamic::compile(&CompiledSdf::compile(&frame(0.0))).expect("JIT SIMD compile");
    println!(
        "scalar: {} params, Lipschitz {}; SIMD: {} params",
        scalar.params().len(),
        scalar.lipschitz(),
        simd.params().len()
    );
    assert_eq!(scalar.params(), extract_jit_params(&frame(0.0)).as_slice());

    let pts: Vec<Vec3> = (0..64)
        .map(|i| {
            let a = i as f32 * 0.2;
            Vec3::new(a.cos() * 1.5, (a * 0.5).sin(), a.sin() * 1.2)
        })
        .collect();
    let soa = SoAPoints::from_vec3_slice(&pts);
    let (xs, ys, zs) = (
        pts.iter().map(|p| p.x).collect::<Vec<_>>(),
        pts.iter().map(|p| p.y).collect::<Vec<_>>(),
        pts.iter().map(|p| p.z).collect::<Vec<_>>(),
    );

    for step in 1..=4 {
        let t = step as f32 * 0.4;
        let tree = frame(t);
        let compiled = CompiledSdf::compile(&tree);
        scalar.update_params(&tree);
        simd.update_params(&compiled);
        assert_eq!(simd.params(), extract_simd_params(&compiled).as_slice());

        let fresh = JitCompiledSdf::compile(&tree).expect("JIT compile");
        let d_batch = scalar.eval_batch(&pts);
        let d_par = scalar.eval_batch_parallel(&pts);
        let f_batch = fresh.eval_batch(&pts);
        let f_par = fresh.eval_batch_parallel(&pts);
        let s_batch = simd.eval_batch(&xs, &ys, &zs);
        let s_soa = simd.eval_soa(&soa);
        // SAFETY: the evaluator is alive and each array holds 8 lanes.
        let first8 = unsafe {
            simd.eval_8(
                &std::array::from_fn(|i| xs[i]),
                &std::array::from_fn(|i| ys[i]),
                &std::array::from_fn(|i| zs[i]),
            )
        };
        let mut worst = 0.0f32;
        for (k, p) in pts.iter().enumerate() {
            let interp = eval_compiled(&compiled, *p);
            assert_eq!(d_batch[k], scalar.eval(*p));
            assert_eq!(d_par[k], d_batch[k]);
            assert_eq!(f_par[k], f_batch[k]);
            assert_eq!(s_soa[k], s_batch[k]);
            worst = worst
                .max((d_batch[k] - f_batch[k]).abs())
                .max((d_batch[k] - interp).abs())
                .max((s_batch[k] - interp).abs());
        }
        assert_eq!(&first8[..], &s_batch[..8]);
        println!("t = {t:.1}: max deviation from fresh compile / interpreter {worst:e}");
        assert!(worst < 1e-4);
    }

    println!("\nall checks passed");
}
