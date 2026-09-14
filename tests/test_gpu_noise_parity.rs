//! GPU ↔ CPU parity for the procedural-noise law (`SurfaceRoughness`).
//!
//! The noise law is defined once in `modifiers::surface_roughness`
//! (`hash_noise_3d` / `fbm`, PCG lattice hash) and the WGSL transpiler emits
//! the same function, so a GPU evaluation of a `SurfaceRoughness` node must
//! agree with `eval` to floating-point precision. This test runs the WGSL
//! path through `GpuEvaluator` when a GPU adapter is available and skips
//! otherwise (CI runners have no GPU; run it locally after touching the noise
//! law or any `HELPER_HASH_NOISE`).

#![cfg(feature = "gpu")]

use alice_sdf::compiled::GpuEvaluator;
use alice_sdf::prelude::*;

/// Deterministic LCG points in a ±3 box.
fn points(n: usize) -> Vec<Vec3> {
    let mut state: u64 = 0x6e01_5e00_0000_0001;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((state >> 40) as f32) / ((1u64 << 24) as f32)).mul_add(6.0, -3.0)
    };
    (0..n).map(|_| Vec3::new(next(), next(), next())).collect()
}

fn gpu_or_skip(node: &SdfNode) -> Option<GpuEvaluator> {
    match GpuEvaluator::new(node) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping GPU noise parity: {e}");
            None
        }
    }
}

fn assert_gpu_matches_cpu(name: &str, node: &SdfNode, abs_tol: f32) {
    let Some(gpu) = gpu_or_skip(node) else {
        return;
    };
    let pts = points(2048);
    let got = gpu.eval_batch(&pts).expect("gpu eval");
    let mut worst = (0.0_f32, Vec3::ZERO, 0.0_f32, 0.0_f32);
    for (p, g) in pts.iter().zip(&got) {
        let c = eval(node, *p);
        let diff = (g - c).abs();
        if diff > worst.0 {
            worst = (diff, *p, c, *g);
        }
    }
    assert!(
        worst.0 <= abs_tol,
        "{name}: GPU/CPU drift {:.3e} at {:?} (cpu={} gpu={})",
        worst.0,
        worst.1,
        worst.2,
        worst.3
    );
    eprintln!(
        "{name}: max |gpu - cpu| = {:.3e} over {} points",
        worst.0,
        pts.len()
    );
}

#[test]
fn surface_roughness_gpu_matches_cpu() {
    for (name, node, tol) in [
        (
            "sphere roughness f3 a0.05 o2",
            SdfNode::sphere(1.0).surface_roughness(3.0, 0.05, 2),
            1e-4,
        ),
        (
            "box roughness f1.7 a0.2 o4",
            SdfNode::box3d(1.6, 1.0, 0.8).surface_roughness(1.7, 0.2, 4),
            2e-4,
        ),
        (
            "torus roughness f5 a0.02 o1",
            SdfNode::torus(0.9, 0.25).surface_roughness(5.0, 0.02, 1),
            1e-4,
        ),
        (
            "translated roughness f2 a0.1 o3",
            SdfNode::sphere(0.7)
                .translate(0.3, -0.2, 0.5)
                .surface_roughness(2.0, 0.1, 3),
            1e-4,
        ),
    ] {
        assert_gpu_matches_cpu(name, &node, tol);
    }
}

/// Control: a node without procedural noise must match too (sanity of the
/// GPU path itself, independent of the noise law).
#[test]
fn plain_csg_gpu_matches_cpu() {
    let node = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(1.0, 1.0, 1.0).translate(0.8, 0.0, 0.0), 0.3);
    assert_gpu_matches_cpu("smooth union control", &node, 1e-4);
}
