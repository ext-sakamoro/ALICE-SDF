//! Measurement only: resident memory per evaluator built in Dynamic mode, per case and per loft count
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gpu")]
#![allow(
    clippy::disallowed_methods,
    reason = "the elapsed time is printed for the measurement"
)]

use alice_sdf::compiled::{GpuEvaluator, TranspileMode, WgslShader};
use alice_sdf::prelude::*;
use glam::Vec2;
use std::time::Instant;

fn rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        })
        .map_or(-1.0, |k| k / 1024.0)
}

fn ring(m: usize, r: f32) -> Vec<Vec2> {
    (0..m)
        .map(|j| {
            let (s, c) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / m as f32);
            Vec2::new(r * c, r * s)
        })
        .collect()
}
fn levels() -> Vec<f32> {
    (0..6).map(|i| 0.15 * i as f32).collect()
}
fn axis(x: f32) -> Vec<Vec3> {
    (0..4).map(|i| Vec3::new(x, 0.3 * i as f32, 0.0)).collect()
}
fn curves(r: f32, x: f32) -> SdfNode {
    SdfNode::loft_curves(
        axis(x),
        Vec3::X,
        axis(x).iter().map(|_| ring(12, r)).collect(),
    )
    .unwrap()
}
fn polar(r: f32) -> SdfNode {
    SdfNode::loft(
        axis(0.0),
        Vec3::Z,
        1,
        axis(0.0).iter().map(|_| vec![r, 0.0, 0.0]).collect(),
    )
    .unwrap()
}
fn section(r: f32) -> SdfNode {
    SdfNode::section_stack(
        levels(),
        levels().iter().map(|_| vec![ring(12, r)]).collect(),
    )
    .unwrap()
}
fn smooth(r: f32) -> SdfNode {
    SdfNode::smooth_section_stack(
        levels(),
        levels().iter().map(|_| vec![ring(12, r)]).collect(),
        0.01,
        0.1,
    )
    .unwrap()
}
fn poly(m: usize) -> SdfNode {
    SdfNode::Polygon2D {
        vertices: ring(m, 0.2),
        half_height: 0.2,
    }
}
fn union_of(v: Vec<SdfNode>) -> SdfNode {
    v.into_iter().reduce(|a, b| a.union(b)).unwrap()
}

fn step(what: &str, build: impl FnOnce() -> GpuEvaluator) -> GpuEvaluator {
    let before = rss_mib();
    let t = Instant::now();
    let g = build();
    let built = rss_mib();
    let _ = g
        .eval_batch(&[Vec3::ZERO, Vec3::splat(0.1)])
        .expect("dispatch");
    eprintln!(
        "rss-probe {what}: before {before:.0} MiB, after build {built:.0} MiB, after first dispatch {:.0} MiB, {:.2} s",
        rss_mib(),
        t.elapsed().as_secs_f64()
    );
    g
}

#[test]
fn rss_per_case_dropped() {
    let cases: Vec<(&str, SdfNode)> = vec![
        ("sphere", SdfNode::sphere(0.2)),
        ("polygon 5", poly(5)),
        ("polygon 9", poly(9)),
        ("polygon 12", poly(12)),
        ("polygon 41", poly(41)),
        ("curve loft", curves(0.2, 0.0).translate(0.1, 0.0, 0.0)),
        ("polar loft", polar(0.2).translate(0.1, 0.0, 0.0)),
        ("section stack", section(0.2).translate(0.1, 0.0, 0.0)),
        ("smooth stack", smooth(0.2).translate(0.1, 0.0, 0.0)),
    ];
    for (n, t) in &cases {
        drop(step(&format!("dynamic {n}"), || {
            GpuEvaluator::new_dynamic(t).expect("dynamic")
        }));
        eprintln!("rss-probe dynamic {n}: after drop {:.0} MiB", rss_mib());
    }
    for (n, t) in &cases {
        let s = WgslShader::try_transpile_with_node_data(t, TranspileMode::Hardcoded)
            .expect("node data");
        drop(step(&format!("node-data {n}"), || {
            GpuEvaluator::from_shader(&s).expect("node data")
        }));
    }
}

#[test]
fn rss_per_loft_count() {
    for k in [1usize, 2, 4] {
        let t = union_of((0..k).map(|i| curves(0.2, 0.3 * i as f32)).collect());
        let s = WgslShader::try_transpile_with_node_data(&t, TranspileMode::Hardcoded)
            .expect("node data");
        drop(step(&format!("node-data {k} curve lofts"), || {
            GpuEvaluator::from_shader(&s).expect("node data")
        }));
        drop(step(&format!("dynamic {k} curve lofts"), || {
            GpuEvaluator::new_dynamic(&t).expect("dynamic")
        }));
    }
}
