//! Measurement only (diagnostic branch): the const GLSL form's build and first dispatch on one driver,
//! one case per process (`DIAG_CASE`)
#![cfg(all(feature = "gpu", feature = "glsl"))]
#![allow(clippy::disallowed_methods, reason = "measurement of wall times")]

use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
use alice_sdf::compiled::GpuEvaluator;
use alice_sdf::prelude::*;
use glam::Vec2;
use std::time::{Duration, Instant};

fn ring(a: f32, b: f32, m: usize) -> Vec<Vec2> {
    (0..m)
        .map(|j| {
            let (s, c) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / m as f32);
            Vec2::new(a * c, b * s)
        })
        .collect()
}

fn loft(rows: usize, points: usize) -> SdfNode {
    let axis: Vec<Vec3> = (0..rows)
        .map(|r| Vec3::new(0.0, 0.1 * r as f32, 0.0))
        .collect();
    let sections: Vec<Vec<Vec2>> = (0..rows)
        .map(|r| ring(0.1 + 0.001 * r as f32, 0.08, points))
        .collect();
    SdfNode::loft_curves_capped(
        axis,
        Vec3::X,
        sections,
        [[0.01, 0.0, 0.0], [0.01, 0.0, 0.0]],
    )
    .expect("loft")
}

fn with_const_arrays(text: &str) -> String {
    text.lines()
        .map(|l| {
            if l.starts_with("float ") && l.contains("] = float[") {
                format!("const {l}\n")
            } else {
                format!("{l}\n")
            }
        })
        .collect()
}

fn scene(name: &str) -> SdfNode {
    match name {
        "loft-2-4" => loft(2, 4),
        "section-stack-3" => SdfNode::section_stack(
            vec![0.0, 0.2, 0.4],
            (0..3).map(|_| vec![ring(0.2, 0.2, 12)]).collect(),
        )
        .expect("stack"),
        "polygon-1" => SdfNode::polygon_2d(ring(0.2, 0.15, 12), 0.03),
        _ => panic!("unknown case {name}"),
    }
}

#[test]
#[ignore = "measurement"]
fn one_case() {
    let case = std::env::var("DIAG_CASE").expect("DIAG_CASE");
    let (name, form) = case.split_once(':').expect("name:form");
    let node = scene(name);
    let variable =
        GlslShader::transpile(&node, GlslTranspileMode::Hardcoded).to_vulkan_compute_shader();
    let text = if form == "const" {
        with_const_arrays(&variable)
    } else {
        variable
    };
    let arrays = text.lines().filter(|l| l.contains("] = float[")).count();
    let t = Instant::now();
    let g = GpuEvaluator::from_glsl_compute(&text)
        .expect("build")
        .into_parts()
        .0;
    let build = t.elapsed().as_secs_f64();
    g.set_wait_limit(Duration::from_secs(890));
    let t = Instant::now();
    let v = g
        .eval_batch(&[Vec3::new(0.0, 0.05, 0.0)])
        .expect("first dispatch");
    let first = t.elapsed().as_secs_f64();
    let pts: Vec<Vec3> = (0..1024)
        .map(|i| Vec3::new(0.0004 * i as f32 - 0.2, 0.05, 0.1))
        .collect();
    let t = Instant::now();
    let w = g.eval_batch(&pts).expect("dispatch");
    let disp = t.elapsed().as_secs_f64();
    let cpu = alice_sdf::eval::eval(&node, Vec3::new(0.0, 0.05, 0.0));
    println!(
        "CASE done {case}: {} bytes, {arrays} arrays, build {build:.2} s, first dispatch {first:.2} s, 1024 points {disp:.2} s, values {} {}, |gpu-cpu| {:e}",
        text.len(), v.len(), w.len(), (v[0] - cpu).abs()
    );
}
