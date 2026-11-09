//! The Vulkan compute form of GLSL writes its module-scope float arrays as variables and every other
//! form as `const`. A long `const` array took over 12 minutes to compile at the first dispatch on a
//! driver that compiles there, against about a second as a variable; ANGLE on NVIDIA refuses the
//! variable form on one loft and links the `const` form (3.0 s, and 7.1 s with curves, checked by
//! hand on an arm64 board, Chrome WebGL2).
//!
//! The two forms are compared on two lanes. On llvmpipe (Linux, the GPU lane) the const form is
//! compiled at the first dispatch and its cost jumps with the number of arrays: a section stack with 2
//! arrays takes 0.15 s, a loft with 12 does not finish in 890 s. So that lane compares them on the
//! section stack. The loft is compared on the macOS lane (`ci.yml`, Metal through wgpu), where the
//! shader is compiled when the pipeline is created; that test is ignored elsewhere and the macOS step
//! runs it with `--ignored` and fails on 0 tests run.
//!
//! What this does not cover: the lanes run the compute form, not WebGL2's text, and D3D11 is
//! unmeasured. This does not make a whole body work in GLSL: that many arrays neither compile in 10
//! minutes as variables nor pass ANGLE's limit.
//!
//! Author: Moroya Sakamoto
#![cfg(all(feature = "gpu", feature = "glsl"))]

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
    .expect("a curve loft")
}

fn section_stack() -> SdfNode {
    SdfNode::section_stack(
        vec![0.0, 0.2, 0.4],
        (0..3).map(|_| vec![ring(0.2, 0.2, 12)]).collect(),
    )
    .expect("a section stack")
}

/// (`const` array declarations, variable array declarations) in the text
fn array_forms(text: &str) -> (usize, usize) {
    let mut forms = (0, 0);
    for line in text.lines() {
        forms.0 += usize::from(line.starts_with("const float ") && line.contains("] = float["));
        forms.1 += usize::from(line.starts_with("float ") && line.contains("] = float["));
    }
    forms
}

#[test]
fn only_the_vulkan_compute_form_has_variable_arrays() {
    for node in [loft(6, 16), section_stack()] {
        let shader = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded);
        let vulkan = array_forms(&shader.to_vulkan_compute_shader());
        assert_eq!(vulkan.0, 0, "a const array in the Vulkan compute form");
        assert!(vulkan.1 > 0, "no array in the Vulkan compute form");
        for (name, text) in [
            ("source", shader.source.clone()),
            ("fragment", shader.to_fragment_shader()),
            ("OpenGL compute", shader.to_compute_shader()),
        ] {
            let forms = array_forms(&text);
            assert_eq!(forms.1, 0, "a variable array in the {name} form");
            assert_eq!(forms.0, vulkan.1, "{name}: not the same arrays");
        }
    }
}

fn evaluator(text: &str) -> Option<GpuEvaluator> {
    match GpuEvaluator::from_glsl_compute(text) {
        Ok(checked) => Some(checked.into_parts().0),
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "no GPU: {e:?}"
            );
            eprintln!("skipping: {e:?}");
            None
        }
    }
}

/// The limit counts the wait's pauses, so a slow compile fails here as a read over its limit
#[test]
fn a_glsl_loft_compiles_within_a_minute() {
    let node = loft(6, 16);
    let text =
        GlslShader::transpile(&node, GlslTranspileMode::Hardcoded).to_vulkan_compute_shader();
    let Some(gpu) = evaluator(&text) else { return };
    gpu.set_wait_limit(Duration::from_secs(60));
    let points = [Vec3::new(0.0, 0.2, 0.0), Vec3::new(1.0, 0.2, 0.0)];
    let got = gpu.eval_batch(&points).expect("a read within the limit");
    assert_eq!(got.len(), points.len());
    assert!(got[0] < 0.0 && got[1] > 0.0, "{got:?}");
    let cpu = alice_sdf::eval::eval(&node, points[1]);
    assert!((got[1] - cpu).abs() < 1e-4, "{} against {cpu}", got[1]);
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

/// The const form's pipeline build, as the evaluator creates it (a driver that compiles at the first
/// dispatch does little here)
const CONST_BUILD_LIMIT: Duration = Duration::from_secs(60);

/// The const form's first dispatch of one point, which holds the compile on a driver that compiles
/// there: the read of a constant array at a run-time index copies the whole array, so the const form
/// costs far more than the variable form (loft(3, 6) took over 300 s on llvmpipe)
const CONST_FIRST_DISPATCH_LIMIT: Duration = Duration::from_secs(180);

/// The const form's dispatch of the 1024 points once compiled
const CONST_DISPATCH_LIMIT: Duration = Duration::from_secs(60);

/// The const and variable forms of `node`'s compute shader on one driver: the const form's build,
/// first dispatch and dispatch are each timed and bounded, the count of points whose bits differ is
/// printed and the difference bounded, because a driver may fold the const reads
#[allow(
    clippy::disallowed_methods,
    reason = "the build and dispatch times are measured and bounded, not part of a computation"
)]
fn const_and_variable_forms_agree(node: &SdfNode) {
    let variable =
        GlslShader::transpile(node, GlslTranspileMode::Hardcoded).to_vulkan_compute_shader();
    let constant = with_const_arrays(&variable);
    assert_ne!(variable, constant, "the two forms are the same text");
    let t = Instant::now();
    let Some(a) = evaluator(&constant) else {
        return;
    };
    let build = t.elapsed();
    let b = evaluator(&variable).expect("the variable form on the same driver");
    let points: Vec<Vec3> = (0..1024)
        .map(|i| {
            let t = i as f32 * 0.002;
            let (s, _) = alice_det_math::sin_cos(t);
            let (_, c) = alice_det_math::sin_cos(3.0 * t);
            Vec3::new(0.2 * s, 0.5 * t - 0.2, 0.2 * c)
        })
        .collect();
    a.set_wait_limit(CONST_FIRST_DISPATCH_LIMIT);
    let t = Instant::now();
    a.eval_batch(&points[..1])
        .expect("the const form's first dispatch");
    let first = t.elapsed();
    a.set_wait_limit(CONST_DISPATCH_LIMIT);
    let t = Instant::now();
    let x = a.eval_batch(&points).expect("const");
    let dispatch = t.elapsed();
    eprintln!(
        "const form: build {:.2} s, first dispatch {:.2} s, {} points {:.2} s",
        build.as_secs_f64(),
        first.as_secs_f64(),
        points.len(),
        dispatch.as_secs_f64()
    );
    assert!(
        build < CONST_BUILD_LIMIT,
        "the const form's build took {build:?}"
    );
    let y = b.eval_batch(&points).expect("variable");
    assert_eq!((x.len(), y.len()), (points.len(), points.len()));
    let differ = x
        .iter()
        .zip(&y)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count();
    let max = x
        .iter()
        .zip(&y)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "const against variable: {differ} of {} differ, largest {max}",
        x.len()
    );
    assert!(max < 2e-6, "largest difference {max}");
}

/// The two forms on one driver, on a section stack (2 arrays) that llvmpipe compiles in well under a
/// second in the const form
#[test]
fn const_and_variable_arrays_agree_on_one_driver() {
    const_and_variable_forms_agree(&section_stack());
}

/// The two forms on a loft (12 arrays), on a driver that compiles at pipeline creation: the macOS
/// lane runs this with `--ignored` (llvmpipe does not finish the const form's first dispatch in 890 s)
#[test]
#[ignore = "runs on the macOS lane, where Metal compiles the shader at pipeline creation"]
fn const_and_variable_arrays_agree_on_a_loft() {
    const_and_variable_forms_agree(&loft(2, 4));
}
