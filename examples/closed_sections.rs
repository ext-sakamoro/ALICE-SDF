//! Shapes from closed sections: a spline outline extruded, stacks of section loops along an axis
//! (piecewise linear and smooth), and a loft whose sections are closed curves; each is written
//! as WGSL, GLSL and HLSL, with and without the union bound skip, and the source sizes printed.
//!
//! Run with `cargo run --example closed_sections --features gpu,glsl,hlsl`
//!
//! Author: Moroya Sakamoto
#![allow(clippy::disallowed_methods, reason = "example: the platform libm builds the outlines")]

use alice_sdf::compiled::{GlslShader, GlslTranspileMode, HlslShader, HlslTranspileMode, TranspileMode, WgslShader};
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
    let stack = SdfNode::section_stack(vec![0.0, 0.4, 0.8], vec![vec![ring(24, 0.2, 0.15)], vec![ring(24, 0.25, 0.2)], vec![ring(24, 0.15, 0.1)]])?;
    let smooth_stack = SdfNode::smooth_section_stack(
        vec![0.0, 0.3, 0.6, 0.9],
        vec![vec![ring(12, 0.2, 0.15)], vec![ring(12, 0.25, 0.2)], vec![ring(12, 0.22, 0.18)], vec![ring(12, 0.15, 0.1)]],
        0.01,
        0.1,
    )?;
    let axis = vec![Vec3::ZERO, Vec3::new(0.05, 0.4, 0.0), Vec3::new(0.0, 0.8, 0.05)];
    let rows = vec![ring(12, 0.2, 0.15), ring(12, 0.18, 0.16), ring(12, 0.12, 0.1)];
    let loft = SdfNode::loft_curves(axis.clone(), Vec3::X, rows.clone()).map_err(|e| e.to_string())?;
    let knots: Vec<f32> = (0..=12).map(|j| j as f32 + if j % 4 == 1 { 0.25 } else { 0.0 }).collect();
    let loft_knots = SdfNode::loft_curves_knots(axis, Vec3::X, knots, rows).map_err(|e| e.to_string())?;
    let scene = outline
        .translate(-0.8, 0.0, 0.0)
        .union(stack.translate(-0.3, 0.0, 0.0))
        .union(smooth_stack.translate(0.2, 0.0, 0.0))
        .union(loft.translate(0.7, 0.0, 0.0))
        .union(loft_knots.translate(1.2, 0.0, 0.0));
    println!("field at the origin: {}", eval(&scene, Vec3::ZERO));
    let sizes = [
        ("WGSL", WgslShader::transpile(&scene, TranspileMode::Hardcoded).source.len(), WgslShader::transpile_without_bound_skip(&scene, TranspileMode::Hardcoded).source.len()),
        ("GLSL", GlslShader::transpile(&scene, GlslTranspileMode::Hardcoded).source.len(), GlslShader::transpile_without_bound_skip(&scene, GlslTranspileMode::Hardcoded).source.len()),
        ("HLSL", HlslShader::transpile(&scene, HlslTranspileMode::Hardcoded).source.len(), HlslShader::transpile_without_bound_skip(&scene, HlslTranspileMode::Hardcoded).source.len()),
    ];
    for (lang, skip, plain) in sizes {
        println!("{lang}: {skip} bytes with the bound skip, {plain} without");
    }
    Ok(())
}
