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
    with_first_const(text, usize::MAX)
}

/// `text` with the first `k` module-scope float arrays (in declaration order) `const`, the rest
/// variables
fn with_first_const(text: &str, k: usize) -> String {
    let mut seen = 0usize;
    let mut out = String::with_capacity(text.len() + 64);
    for l in text.lines() {
        if l.starts_with("float ") && l.contains("] = float[") {
            seen += 1;
            if seen <= k {
                out.push_str("const ");
            }
        }
        out.push_str(l);
        out.push('\n');
    }
    out
}

/// `text` with every module-scope `const float NAME[N]` as a variable
fn with_variable_arrays(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for l in text.lines() {
        match l.strip_prefix("const ") {
            Some(rest) if rest.starts_with("float ") && rest.contains("] = float[") => {
                out.push_str(rest);
            }
            _ => out.push_str(l),
        }
        out.push('\n');
    }
    out
}

/// Each module-scope float array of `text`: name, length, and the number of reads `NAME[`
fn array_table(text: &str) -> Vec<(String, usize, usize)> {
    text.lines()
        .filter(|l| l.contains("] = float["))
        .filter_map(|l| {
            let rest = l.trim_start_matches("const ").strip_prefix("float ")?;
            let (name, tail) = rest.split_once('[')?;
            let n: usize = tail.split(']').next()?.parse().ok()?;
            let reads = text.matches(&format!("{name}[")).count() - 1;
            Some((name.to_owned(), n, reads))
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
    } else if let Some(k) = form.strip_prefix("const-first-") {
        let k: usize = k.parse().expect("k");
        let t = with_first_const(&variable, k);
        let table = array_table(&t);
        let consts: Vec<String> = table
            .iter()
            .take(k)
            .map(|(n, len, r)| format!("{n}[{len}] x{r}"))
            .collect();
        eprintln!(
            "CASE arrays {case}: {} total, const: {}",
            table.len(),
            consts.join(", ")
        );
        t
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
    eprintln!("CASE built {case}: build {build:.2} s");
    let limit: u64 = std::env::var("DIAG_WAIT_S").map_or(890, |v| v.parse().expect("DIAG_WAIT_S"));
    g.set_wait_limit(Duration::from_secs(limit));
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

/// Writes the WebGL2 (GLSL ES 3.00) full fragment shader of `DIAG_CASE` (`name:const` or
/// `name:variable`) to `DIAG_OUT`, for the browser measurement
#[test]
#[ignore = "measurement"]
fn emit_webgl2() {
    let case = std::env::var("DIAG_CASE").expect("DIAG_CASE");
    let out = std::env::var("DIAG_OUT").expect("DIAG_OUT");
    let (name, form) = case.split_once(':').expect("name:form");
    let node = scene(name);
    let full = GlslShader::transpile_with_version(&node, GlslTranspileMode::Hardcoded, 300)
        .to_fragment_shader_full(&alice_sdf::compiled::glsl::RenderConfig::default());
    let text = match form {
        "const" => full,
        "variable" => with_variable_arrays(&full),
        _ => panic!("form {form}"),
    };
    let table = array_table(&text);
    let consts = text
        .lines()
        .filter(|l| l.starts_with("const float ") && l.contains("] = float["))
        .count();
    std::fs::write(&out, &text).expect("write");
    println!(
        "EMIT {case}: {} bytes, {} arrays ({consts} const), reads {}",
        text.len(),
        table.len(),
        table.iter().map(|t| t.2).sum::<usize>()
    );
}
