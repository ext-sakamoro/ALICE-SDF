//! GPU vs CPU for nodes whose large tables go to the node data buffer (lattice deform control
//! points, heightmap samples): the shader must read the buffer and give the CPU value
//! oracle: the CPU evaluator `eval` (an independent code path from the WGSL transpiler)

#![cfg(feature = "gpu")]

use alice_sdf::compiled::{GpuEvaluator, TranspileMode, WgslShader};
use alice_sdf::prelude::*;

fn points() -> Vec<Vec3> {
    let mut pts = Vec::with_capacity(24 * 24 * 24);
    for a in 0..24 {
        for b in 0..24 {
            for c in 0..24 {
                pts.push(
                    Vec3::new(a as f32 + 0.3, b as f32 + 0.7, c as f32 + 0.1) / 24.0 * 0.5
                        - Vec3::splat(0.25),
                );
            }
        }
    }
    pts
}

fn check(name: &str, node: &SdfNode) {
    let shader = WgslShader::transpile_with_node_data(node, TranspileMode::Hardcoded);
    assert!(
        !shader.node_data.is_empty(),
        "{name}: the table must go to the node data buffer"
    );
    let pts = points();
    let got = GpuEvaluator::new(node)
        .expect("gpu evaluator")
        .eval_batch(&pts)
        .expect("gpu eval");
    assert_eq!(got.len(), pts.len());
    let mut worst = 0.0f32;
    for (p, v) in pts.iter().zip(&got) {
        worst = worst.max((eval(node, *p) - v).abs());
    }
    assert!(
        worst < 2e-5,
        "{name}: max |gpu - cpu| {worst} over {} points",
        pts.len()
    );
}

/// Lattice deform: 4 x 5 x 4 control points, each moved off the regular grid
#[test]
fn lattice_deform_reads_the_node_data_buffer() {
    let (nx, ny, nz) = (3u32, 4u32, 3u32);
    let (lo, hi) = (Vec3::splat(-0.2), Vec3::splat(0.2));
    let mut cps = Vec::new();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                let t = Vec3::new(
                    i as f32 / nx as f32,
                    j as f32 / ny as f32,
                    k as f32 / nz as f32,
                );
                let p = lo + (hi - lo) * t;
                let (s, _) = alice_det_math::sin_cos(7.0 * t.y);
                let (_, c) = alice_det_math::sin_cos(5.0 * t.x);
                cps.push(p + Vec3::new(0.03 * s, 0.02 * t.x * t.z, -0.025 * c));
            }
        }
    }
    check(
        "lattice",
        &SdfNode::sphere(0.12).lattice_deform(cps, nx, ny, nz, lo, hi),
    );
}

/// Heightmap displacement: 17 x 17 samples from an integer hash (no two neighbours equal)
#[test]
fn heightmap_displacement_reads_the_node_data_buffer() {
    let w = 17u32;
    let hm: Vec<f32> = (0..w * w)
        .map(|i| ((i.wrapping_mul(2_654_435_761) >> 13) % 1000) as f32 / 1000.0)
        .collect();
    check(
        "heightmap",
        &SdfNode::box3d(0.3, 0.1, 0.3).heightmap_displacement(hm, w, w, 0.02, 3.0),
    );
}
