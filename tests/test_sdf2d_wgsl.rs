//! The 2D SDF (`Sdf2dNode`) as WGSL: `Wgsl2dShader`
//!
//! - every node variant emits WGSL that naga parses and validates, as a fragment
//!   (`sdf2d_eval(p: vec2<f32>) -> f32` and its helpers) and as a whole compute shader;
//! - the compute shader is whole (it has an entry point) and carries this crate's semantics
//!   identifier; the fragment carries none (ALICE-Law G8);
//! - on a GPU the compute shader gives the distances of `eval_2d` within a tolerance (the GPU's
//!   `length` / `atan2` / `sin` / `cos` are not this crate's deterministic math, so the bits are
//!   not specified, only the distance);
//! - a parameter that is not a finite number is refused, not emitted.
//!
//! Oracle: `sdf2d::eval_2d`, the CPU law; the tolerance is the one of tests/test_gpu_law_parity.rs
//! (1e-4 relative to max(|d|, 1)). The corpus names every variant through an exhaustive match,
//! so a new variant fails to compile here until it has a case.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "gpu")]

#[path = "common/shader_text.rs"]
mod shader_text;

use alice_sdf::compiled::wgsl::{Wgsl2dError, Wgsl2dShader};
use alice_sdf::compiled::GpuEvaluator;
use alice_sdf::prelude::Vec3;
use alice_sdf::provenance::{self, Provenance, KEY};
use alice_sdf::sdf2d::{eval_2d, Sdf2dNode};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use shader_text::has_entry_point;

/// The variant a node is (exhaustive: a new variant fails to compile until it has a case below)
const fn variant(n: &Sdf2dNode) -> &'static str {
    match n {
        Sdf2dNode::Circle { .. } => "circle",
        Sdf2dNode::Rect { .. } => "rect",
        Sdf2dNode::RoundedRect { .. } => "rounded_rect",
        Sdf2dNode::Line { .. } => "line",
        Sdf2dNode::Bezier { .. } => "bezier",
        Sdf2dNode::FontGlyph { .. } => "font_glyph",
        Sdf2dNode::Union(..) => "union",
        Sdf2dNode::Subtract(..) => "subtract",
        Sdf2dNode::Intersect(..) => "intersect",
        Sdf2dNode::SmoothUnion { .. } => "smooth_union",
        Sdf2dNode::Ring { .. } => "ring",
        Sdf2dNode::RegularPolygon { .. } => "regular_polygon",
        Sdf2dNode::Star { .. } => "star",
        Sdf2dNode::Ellipse { .. } => "ellipse",
        Sdf2dNode::Onion { .. } => "onion",
        Sdf2dNode::Translate { .. } => "translate",
        Sdf2dNode::Rotate { .. } => "rotate",
        Sdf2dNode::Scale { .. } => "scale",
    }
}

/// A 64x64 glyph grid: the signed distance to a circle of radius 0.3 around the centre, in the
/// grid's unit square
fn glyph() -> Sdf2dNode {
    let mut data = Box::new([0.0f32; 4096]);
    for y in 0..64 {
        for x in 0..64 {
            let u = x as f32 / 63.0 - 0.5;
            let v = y as f32 / 63.0 - 0.5;
            data[y * 64 + x] = (u * u + v * v).sqrt() - 0.3;
        }
    }
    Sdf2dNode::FontGlyph {
        data,
        advance: 0.6,
        bbox_min: [-0.5, -0.5],
        bbox_max: [0.5, 0.5],
    }
}

const fn circle() -> Sdf2dNode {
    Sdf2dNode::Circle {
        center: [0.1, -0.2],
        radius: 0.7,
    }
}

const fn rect() -> Sdf2dNode {
    Sdf2dNode::Rect {
        center: [-0.25, 0.125],
        half_extents: [0.6, 0.35],
    }
}

/// One node per variant, then a few compositions
fn corpus() -> Vec<(&'static str, Sdf2dNode)> {
    let b = |n: Sdf2dNode| Box::new(n);
    let nodes = vec![
        circle(),
        rect(),
        Sdf2dNode::RoundedRect {
            center: [0.0, 0.25],
            half_extents: [0.8, 0.4],
            corner_radius: 0.15,
        },
        Sdf2dNode::Line {
            a: [-0.7, -0.3],
            b: [0.6, 0.5],
            thickness: 0.05,
        },
        Sdf2dNode::Bezier {
            p0: [-0.8, 0.0],
            p1: [-0.3, 0.9],
            p2: [0.3, -0.9],
            p3: [0.8, 0.0],
            thickness: 0.04,
        },
        glyph(),
        Sdf2dNode::Union(b(circle()), b(rect())),
        Sdf2dNode::Subtract(b(rect()), b(circle())),
        Sdf2dNode::Intersect(b(circle()), b(rect())),
        Sdf2dNode::SmoothUnion {
            a: b(circle()),
            b: b(rect()),
            k: 0.2,
        },
        Sdf2dNode::Ring {
            center: [0.0, 0.0],
            outer_radius: 0.6,
            thickness: 0.08,
        },
        Sdf2dNode::RegularPolygon {
            center: [0.05, 0.0],
            radius: 0.7,
            sides: 6,
        },
        Sdf2dNode::Star {
            center: [0.0, 0.1],
            outer_radius: 0.8,
            inner_radius: 0.35,
            points: 5,
        },
        Sdf2dNode::Ellipse {
            center: [0.0, 0.0],
            semi_axes: [0.9, 0.4],
        },
        Sdf2dNode::Onion {
            child: b(circle()),
            thickness: 0.05,
        },
        Sdf2dNode::Translate {
            child: b(rect()),
            offset: [0.3, -0.4],
        },
        Sdf2dNode::Rotate {
            child: b(rect()),
            angle: 0.6,
        },
        Sdf2dNode::Scale {
            child: b(glyph()),
            factor: 1.5,
        },
    ];
    let mut out: Vec<(&'static str, Sdf2dNode)> =
        nodes.into_iter().map(|n| (variant(&n), n)).collect();
    // compositions: a transformed glyph inside a union, a smooth union of transformed shapes
    out.push((
        "composite_glyph",
        Sdf2dNode::Union(
            b(Sdf2dNode::Translate {
                child: b(glyph()),
                offset: [0.4, 0.0],
            }),
            b(Sdf2dNode::Rotate {
                child: b(circle()),
                angle: -1.1,
            }),
        ),
    ));
    out.push((
        "composite_smooth",
        Sdf2dNode::SmoothUnion {
            a: b(Sdf2dNode::Scale {
                child: b(rect()),
                factor: 0.7,
            }),
            b: b(Sdf2dNode::Onion {
                child: b(Sdf2dNode::Ellipse {
                    center: [0.2, 0.2],
                    semi_axes: [0.5, 0.3],
                }),
                thickness: 0.03,
            }),
            k: 0.1,
        },
    ));
    out
}

fn validate(name: &str, what: &str, src: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(src)
        .map_err(|e| format!("{name} ({what}): parse: {}", e.emit_to_string(src)))?;
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map(|_| ())
        .map_err(|e| format!("{name} ({what}): validate: {e:?}"))
}

#[test]
fn the_corpus_names_every_variant() {
    let names: std::collections::BTreeSet<&str> = corpus().iter().map(|(n, _)| *n).collect();
    for v in [
        "circle",
        "rect",
        "rounded_rect",
        "line",
        "bezier",
        "font_glyph",
        "union",
        "subtract",
        "intersect",
        "smooth_union",
        "ring",
        "regular_polygon",
        "star",
        "ellipse",
        "onion",
        "translate",
        "rotate",
        "scale",
    ] {
        assert!(names.contains(v), "{v} has no case");
    }
}

#[test]
fn every_variant_emits_valid_wgsl() {
    let mut problems = Vec::new();
    let mut checked = 0;
    for (name, node) in corpus() {
        let s = Wgsl2dShader::transpile(&node).unwrap();
        for (what, src) in [
            ("fragment", s.source.clone()),
            ("compute", s.to_compute_shader()),
        ] {
            match validate(name, what, &src) {
                Ok(()) => checked += 1,
                Err(e) => problems.push(e),
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
    assert_eq!(checked, 2 * corpus().len());
}

#[test]
fn the_compute_shader_is_whole_and_carries_the_identifier_and_the_fragment_none() {
    for (name, node) in corpus() {
        let s = Wgsl2dShader::transpile(&node).unwrap();
        let whole = s.to_compute_shader();
        assert!(has_entry_point(&whole), "{name}");
        assert_eq!(
            provenance::from_bytes(whole.as_bytes()).unwrap(),
            Provenance::sdf(),
            "{name}"
        );
        assert!(
            !has_entry_point(&s.source),
            "{name}: the fragment has no entry point"
        );
        assert!(
            !s.source.contains(KEY),
            "{name}: the fragment carries no identifier"
        );
    }
}

#[test]
fn a_parameter_that_is_not_finite_is_refused() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let n = Sdf2dNode::Circle {
            center: [0.0, bad],
            radius: 1.0,
        };
        assert!(matches!(
            Wgsl2dShader::transpile(&n),
            Err(Wgsl2dError::NonFinite(_))
        ));
        let deep = Sdf2dNode::Union(
            Box::new(circle()),
            Box::new(Sdf2dNode::Rotate {
                child: Box::new(rect()),
                angle: bad,
            }),
        );
        assert!(Wgsl2dShader::transpile(&deep).is_err());
    }
}

/// Points of the plane around the shapes: a grid over [-1.5, 1.5]² (z = 0)
fn points() -> Vec<Vec3> {
    let n = 48;
    let mut out = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let x = -1.5 + 3.0 * (i as f32 + 0.5) / n as f32;
            let y = -1.5 + 3.0 * (j as f32 + 0.5) / n as f32;
            out.push(Vec3::new(x, y, 0.0));
        }
    }
    out
}

#[test]
fn the_gpu_gives_the_cpu_distances_within_the_tolerance() {
    let pts = points();
    let mut ran = 0;
    let mut problems = Vec::new();
    for (name, node) in corpus() {
        let s = Wgsl2dShader::transpile(&node).unwrap();
        let gpu = match GpuEvaluator::from_wgsl_pinned(&s.to_compute_shader()) {
            Ok(g) => g,
            Err(e) => {
                assert!(
                    std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                    "ALICE_SDF_REQUIRE_GPU is set but the shader did not build: {e}"
                );
                eprintln!("gpu-test: skipped ({e})");
                return;
            }
        };
        let got = gpu.eval_batch(&pts).expect("gpu eval");
        let mut worst = (0.0f32, 0usize);
        for (k, (p, g)) in pts.iter().zip(&got).enumerate() {
            let c = eval_2d(&node, [p.x, p.y]);
            let diff = (g - c).abs() / c.abs().max(1.0);
            if diff > worst.0 {
                worst = (diff, k);
            }
        }
        if worst.0 > 1e-4 {
            let p = pts[worst.1];
            problems.push(format!(
                "{name}: drift {:.3e} at ({}, {}) cpu={} gpu={}",
                worst.0,
                p.x,
                p.y,
                eval_2d(&node, [p.x, p.y]),
                got[worst.1]
            ));
        }
        eprintln!("{name}: max relative |gpu - cpu| = {:.3e}", worst.0);
        ran += 1;
    }
    eprintln!("gpu-test: ran");
    assert!(problems.is_empty(), "{}", problems.join("\n"));
    assert_eq!(ran, corpus().len());
}
