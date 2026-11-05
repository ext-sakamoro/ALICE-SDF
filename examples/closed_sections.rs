//! Shapes from closed sections: a spline outline extruded, stacks of section loops along an axis
//! (piecewise linear and smooth), and a loft whose sections are closed curves; each is written
//! as WGSL, GLSL and HLSL, with and without the union bound skip, and the source sizes printed.
//!
//! Run with `cargo run --example closed_sections --features gpu,glsl,hlsl`
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "example: the platform libm builds the outlines"
)]

use alice_sdf::compiled::{
    GlslShader, GlslTranspileMode, GpuEvaluator, HlslShader, HlslTranspileMode, TranspileMode,
    WgslShader,
};
use alice_sdf::prelude::*;
use glam::Vec2;

fn ring(n: usize, a: f32, b: f32) -> Vec<Vec2> {
    (0..n)
        .map(|k| {
            let t = std::f32::consts::TAU * k as f32 / n as f32;
            Vec2::new(a * t.cos(), b * t.sin())
        })
        .collect()
}

fn main() -> Result<(), String> {
    let outline = SdfNode::polygon_2d_closed_spline(&ring(6, 0.3, 0.2), 8, 0.1)?;
    let stack = SdfNode::section_stack(
        vec![0.0, 0.4, 0.8],
        vec![
            vec![ring(24, 0.2, 0.15)],
            vec![ring(24, 0.25, 0.2)],
            vec![ring(24, 0.15, 0.1)],
        ],
    )?;
    let smooth_stack = SdfNode::smooth_section_stack(
        vec![0.0, 0.3, 0.6, 0.9],
        vec![
            vec![ring(12, 0.2, 0.15)],
            vec![ring(12, 0.25, 0.2)],
            vec![ring(12, 0.22, 0.18)],
            vec![ring(12, 0.15, 0.1)],
        ],
        0.01,
        0.1,
    )?;
    let axis = vec![
        Vec3::ZERO,
        Vec3::new(0.05, 0.4, 0.0),
        Vec3::new(0.0, 0.8, 0.05),
    ];
    let rows = vec![
        ring(12, 0.2, 0.15),
        ring(12, 0.18, 0.16),
        ring(12, 0.12, 0.1),
    ];
    let (axis_capped, rows_capped) = (axis.clone(), rows.clone());
    let loft =
        SdfNode::loft_curves(axis.clone(), Vec3::X, rows.clone()).map_err(|e| e.to_string())?;
    let knots: Vec<f32> = (0..=12)
        .map(|j| j as f32 + if j % 4 == 1 { 0.25 } else { 0.0 })
        .collect();
    let loft_knots =
        SdfNode::loft_curves_knots(axis, Vec3::X, knots, rows).map_err(|e| e.to_string())?;
    // a polar loft (Fourier radii per row), capped, and a capped curve loft
    let polar = SdfNode::loft(
        vec![Vec3::ZERO, Vec3::new(0.02, 0.4, 0.0), Vec3::new(0.0, 0.8, 0.02)],
        Vec3::Z,
        1,
        vec![vec![0.2, 0.01, 0.0], vec![0.22, 0.0, 0.01], vec![0.15, 0.0, 0.0]],
    )
    .map_err(|e| e.to_string())?;
    let polar_capped = SdfNode::loft_capped(
        vec![Vec3::ZERO, Vec3::new(0.02, 0.4, 0.0), Vec3::new(0.0, 0.8, 0.02)],
        Vec3::Z,
        1,
        vec![vec![0.2, 0.01, 0.0], vec![0.22, 0.0, 0.01], vec![0.15, 0.0, 0.0]],
        [0.08, 0.1],
    )
    .map_err(|e| e.to_string())?;
    let curve_capped = SdfNode::loft_curves_capped(
        axis_capped,
        Vec3::X,
        rows_capped,
        [[0.1, 0.0, 0.0], [0.08, 0.02, 0.0]],
    )
    .map_err(|e| e.to_string())?;
    if let (SdfNode::SectionStack(a), SdfNode::SmoothSectionStack(b)) = (&stack, &smooth_stack) {
        println!("stack bounds: {} and {}", a.lipschitz(), b.lipschitz());
    }
    let scene = outline
        .translate(-0.8, 0.0, 0.0)
        .union(stack.translate(-0.3, 0.0, 0.0))
        .union(smooth_stack.translate(0.2, 0.0, 0.0))
        .union(loft.translate(0.7, 0.0, 0.0))
        .union(loft_knots.translate(1.2, 0.0, 0.0))
        .union(polar.translate(1.7, 0.0, 0.0))
        .union(polar_capped.translate(2.2, 0.0, 0.0))
        .union(curve_capped.translate(2.7, 0.0, 0.0));
    println!("field at the origin: {}", eval(&scene, Vec3::ZERO));
    let sizes = [
        (
            "WGSL",
            WgslShader::transpile(&scene, TranspileMode::Hardcoded)
                .source
                .len(),
            WgslShader::transpile_without_bound_skip(&scene, TranspileMode::Hardcoded)
                .source
                .len(),
        ),
        (
            "GLSL",
            GlslShader::transpile(&scene, GlslTranspileMode::Hardcoded)
                .source
                .len(),
            GlslShader::transpile_without_bound_skip(&scene, GlslTranspileMode::Hardcoded)
                .source
                .len(),
        ),
        (
            "HLSL",
            HlslShader::transpile(&scene, HlslTranspileMode::Hardcoded)
                .source
                .len(),
            HlslShader::transpile_without_bound_skip(&scene, HlslTranspileMode::Hardcoded)
                .source
                .len(),
        ),
    ];
    for (lang, skip, plain) in sizes {
        println!("{lang}: {skip} bytes with the bound skip, {plain} without");
    }
    // the fallible emits report a read they cannot prove in range; the checked ones count, at run
    // time, each read outside its proven range
    let wgsl = WgslShader::try_transpile_with_node_data_checked(&scene, TranspileMode::Hardcoded)
        .map_err(|e| e.to_string())?;
    let glsl = GlslShader::try_transpile(&scene, GlslTranspileMode::Hardcoded)
        .map_err(|e| e.to_string())?;
    let hlsl = HlslShader::try_transpile_checked(&scene, HlslTranspileMode::Hardcoded)
        .map_err(|e| e.to_string())?;
    let buffered = WgslShader::transpile_with_node_data(&scene, TranspileMode::Hardcoded);
    println!(
        "checked WGSL {} bytes, GLSL {} bytes, checked HLSL {} bytes, {} values in the node data buffer",
        wgsl.source.len(),
        glsl.source.len(),
        hlsl.source.len(),
        buffered.node_data.len()
    );
    // needs a GPU: `-- --gpu` evaluates a small batch and reads the out-of-range count
    if std::env::args().any(|a| a == "--gpu") {
        let gpu = GpuEvaluator::new_checked(&scene).map_err(|e| e.to_string())?;
        let pts: Vec<Vec3> = (0..64).map(|i| Vec3::new(0.05 * i as f32, 0.2, 0.0)).collect();
        let values = gpu.eval_batch(&pts).map_err(|e| e.to_string())?;
        let hits = gpu.read_hits().map_err(|e| e.to_string())?;
        println!("{} values, reads outside their range: {hits:?}", values.len());
    }
    Ok(())
}
