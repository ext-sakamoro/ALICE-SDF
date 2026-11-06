//! No emitted shader wraps an index through a float divide.
//!
//! `int(floor(float(j) / float(m)))` is not the integer quotient once a compiler replaces the divide by a
//! multiply with the rounded reciprocal, which a compiler may do with a constant divisor: 41 · (1 / 41)
//! is below 1 in `f32`, its floor is 0, and the index lands one past the outline. dxc does this (it
//! marks the multiply `fast`). The text is templated, so the check reads the emitted shaders of every
//! tree in the gate corpus in every dialect and mode, not the sources.
//!
//! Author: Moroya Sakamoto
#![cfg(all(feature = "gpu", feature = "glsl", feature = "hlsl"))]

mod common;
use alice_sdf::compiled::{GlslShader, GlslTranspileMode, HlslShader, HlslTranspileMode, TranspileMode, WgslShader};
use common::corpus::gate_corpus;

/// The lines that turn a rounded quotient into an integer
fn float_wraps(src: &str) -> Vec<String> {
    const ROUNDS: [&str; 4] = ["floor(", "trunc(", "round(", "ceil("];
    src.lines()
        .filter(|l| {
            let t = l.replace(' ', "");
            l.contains('/')
                && ["int(", "i32(", "(int)"]
                    .iter()
                    .any(|cast| ROUNDS.iter().any(|r| t.contains(&format!("{cast}{r}"))))
        })
        .map(|l| l.trim().to_string())
        .collect()
}

#[test]
fn the_check_sees_the_old_wrap() {
    assert_eq!(float_wraps("    int k = j - m * int(floor(float(j) / float(m)));").len(), 1);
    assert_eq!(float_wraps("    var k: i32 = j - m * i32(floor(f32(j) / f32(m)));").len(), 1);
    assert!(float_wraps("    int k = (j + 4 * m) % m;").is_empty());
}

#[test]
fn no_shader_wraps_an_index_through_a_float_divide() {
    let mut compared = 0;
    let mut found = Vec::new();
    for (name, node) in gate_corpus() {
        for dynamic in [false, true] {
            let w = if dynamic { TranspileMode::Dynamic } else { TranspileMode::Hardcoded };
            let g = if dynamic { GlslTranspileMode::Dynamic } else { GlslTranspileMode::Hardcoded };
            let h = if dynamic { HlslTranspileMode::Dynamic } else { HlslTranspileMode::Hardcoded };
            for (dialect, src) in [
                ("wgsl", WgslShader::transpile(&node, w).source),
                ("glsl", GlslShader::transpile(&node, g).source),
                ("hlsl", HlslShader::transpile(&node, h).source),
            ] {
                compared += 1;
                for l in float_wraps(&src) {
                    found.push(format!("{name} {dialect}: {l}"));
                }
            }
        }
    }
    assert!(compared > 900, "only {compared} shaders read");
    assert!(found.is_empty(), "{} float wraps, first: {:?}", found.len(), &found[..found.len().min(5)]);
}
