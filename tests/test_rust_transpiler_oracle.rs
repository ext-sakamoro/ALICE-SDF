//! Oracle for the Rust source emit (`compiled::rust`, feature `rust`).
//!
//! The emitted `fn sdf` / `fn normal` are compiled with `rustc` (linked only
//! against `alice_det_math`, never `alice_sdf`) and run on a grid; every
//! distance and normal is compared **by bits** with the independent reference,
//! the bytecode interpreter `eval_compiled` / `eval_compiled_normal`. A run
//! that compares nothing fails, and so does a supported opcode that no scene
//! exercises.
//!
//! Scenes: every corpus entry the emitter accepts (the rest must be rejected
//! with `UnsupportedOpcode` for an opcode `is_supported` says has no emitter),
//! plus nested CSG / transform / modifier compositions. The generated program
//! is built at `opt-level=0` and `opt-level=3`; Rust never contracts
//! `a * b + c`, so both must match.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "rust")]

mod common;

use alice_sdf::compiled::rust::{
    is_supported, RustSource, RustTranspileError, RustTranspileOptions, DEFAULT_NORMAL_EPSILON,
};
use alice_sdf::compiled::{eval_compiled, eval_compiled_normal, CompiledSdf, OpCode};
use alice_sdf::prelude::*;
use glam::Vec2;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Grid at 0.5 spacing (cell / axis ties) + fixed points + an LCG spray.
fn sample_points() -> Vec<Vec3> {
    let mut pts = Vec::new();
    for i in 0..11 {
        for j in 0..11 {
            for k in 0..11 {
                pts.push(Vec3::new(
                    -2.5 + 0.5 * i as f32,
                    -2.5 + 0.5 * j as f32,
                    -2.5 + 0.5 * k as f32,
                ));
            }
        }
    }
    pts.extend([
        Vec3::new(-0.0, -0.0, -0.0),
        Vec3::new(0.25, 0.0, 0.0),
        Vec3::new(0.3, -0.7, 1.1),
        Vec3::new(-0.5, 0.3, 0.0),
        Vec3::new(0.0, 0.3, 0.7),
        Vec3::new(1e-7, -1e-7, 1e-7),
        Vec3::new(40.0, -35.0, 60.0),
    ]);
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32) * 6.0 - 3.0
    };
    for _ in 0..200 {
        pts.push(Vec3::new(next(), next(), next()));
    }
    pts
}

/// Compositions that nest CSG, transforms and modifiers (the corpus is mostly
/// one node over a primitive).
fn compositions() -> Vec<(&'static str, SdfNode)> {
    let s = SdfNode::sphere(0.6);
    let b = SdfNode::box3d(0.5, 0.4, 0.3);
    let c = SdfNode::cylinder(0.3, 0.8);
    vec![
        (
            "smooth_union_rotated_box",
            s.clone().smooth_union(
                b.clone()
                    .rotate_euler(0.3, -0.7, 1.1)
                    .translate(0.5, 0.1, -0.2),
                0.3,
            ),
        ),
        (
            "smooth_subtract_scaled",
            b.clone()
                .scale(1.7)
                .smooth_subtract(s.clone().translate(0.4, 0.4, 0.0), 0.25),
        ),
        (
            "smooth_intersect_nonuniform",
            s.clone()
                .scale_xyz(1.5, 0.8, 1.2)
                .smooth_intersection(c.clone().rotate_euler(1.2, 0.0, 0.4), 0.2),
        ),
        (
            "subtract_intersect_union_tree",
            b.clone()
                .intersection(s.clone().scale(1.4))
                .subtract(c.clone().union(c.clone().rotate_euler(1.5707964, 0.0, 0.0)))
                .union(SdfNode::torus(0.9, 0.1).translate(0.0, -0.9, 0.0)),
        ),
        (
            "chamfer_and_exp_chain",
            s.clone()
                .chamfer_union(b.clone().translate(0.6, 0.0, 0.0), 0.15)
                .chamfer_subtract(c, 0.1)
                .exp_smooth_union(SdfNode::sphere(0.3).translate(-0.8, 0.5, 0.0), 0.2)
                .exp_smooth_subtract(SdfNode::sphere(0.25).translate(0.0, 0.6, 0.0), 0.1)
                .exp_smooth_intersection(SdfNode::box3d(1.5, 1.5, 1.5), 0.3)
                .chamfer_intersection(SdfNode::sphere(1.6), 0.2),
        ),
        (
            "xor_morph",
            s.clone()
                .xor(b.clone().translate(0.3, 0.0, 0.0))
                .morph(SdfNode::torus(0.7, 0.2), 0.35),
        ),
        (
            "twist_bend_round_onion",
            b.clone()
                .twist(1.3)
                .bend(0.4)
                .round(0.05)
                .onion(0.04)
                .rotate_euler(-0.2, 0.9, 0.0),
        ),
        (
            "repeat_in_rotated_frame",
            SdfNode::sphere(0.2)
                .repeat_infinite(0.7, 0.9, 1.1)
                .intersection(SdfNode::box3d(2.0, 2.0, 2.0))
                .rotate_euler(0.4, 0.4, 0.4),
        ),
        (
            "repeat_finite_scaled",
            SdfNode::box3d(0.15, 0.15, 0.15)
                .repeat_finite([3, 2, 1], Vec3::new(0.6, 0.7, 0.8))
                .scale(1.3),
        ),
        (
            "mirror_elongate_shear",
            SdfNode::capsule(Vec3::new(0.2, 0.0, 0.0), Vec3::new(0.7, 0.4, 0.1), 0.15)
                .mirror(true, false, true)
                .elongate(0.2, 0.0, 0.1)
                .shear(0.3, -0.2, 0.1),
        ),
        (
            "octant_polar_revolution",
            SdfNode::link(0.2, 0.3, 0.08)
                .polar_repeat(5)
                .octant_mirror()
                .union(SdfNode::hex_prism(0.3, 0.2).revolution(0.6)),
        ),
        (
            "extrude_taper_displace",
            SdfNode::cone(0.5, 0.7)
                .taper(0.3)
                .displacement(0.05)
                .smooth_union(SdfNode::sphere(0.4).extrude(0.3), 0.1),
        ),
        (
            "metric_blend_nested",
            SdfNode::metric_blend(
                s.twist(0.8),
                b.translate(0.2, 0.0, 0.0),
                Vec3::new(0.1, 0.2, -0.1),
                0.5,
                0.4,
            ),
        ),
        (
            "plane_gyroid_infinite_cylinder",
            SdfNode::gyroid(3.0, 0.1)
                .intersection(SdfNode::sphere(1.5))
                .subtract(SdfNode::infinite_cylinder(0.3))
                .intersection(SdfNode::plane(Vec3::new(0.0, 0.6, 0.8), 0.3)),
        ),
        (
            "rounded_box_deep_transforms",
            SdfNode::rounded_box(0.4, 0.3, 0.2, 0.05)
                .translate(0.1, 0.2, 0.3)
                .rotate_euler(0.5, 0.0, 0.0)
                .scale(0.9)
                .rotate_euler(0.0, 0.7, 0.0)
                .translate(-0.3, 0.0, 0.2)
                .scale_xyz(1.1, 0.9, 1.3),
        ),
    ]
}

/// One scene that goes into the generated program.
struct Scene {
    name: String,
    compiled: CompiledSdf,
    source: String,
    eps: f32,
    /// emitted with a custom det-math path: the module aliases it
    alias: Option<&'static str>,
}

/// `libalice_det_math-*.rlib` of the locked version, next to this test binary.
fn det_math_rlib(deps: &Path) -> PathBuf {
    let lock = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"))
        .expect("Cargo.lock (the oracle links the locked alice-det-math)");
    // the version alice-sdf itself depends on (the lock may hold several:
    // `"alice-det-math 0.3.1"` in the dependency list when it does)
    let blocks: Vec<&str> = lock.split("[[package]]").collect();
    let sdf = blocks
        .iter()
        .find(|b| b.contains("name = \"alice-sdf\""))
        .expect("alice-sdf in Cargo.lock");
    let dep = sdf
        .lines()
        .map(|l| l.trim().trim_end_matches(',').trim_matches('"'))
        .find(|l| *l == "alice-det-math" || l.starts_with("alice-det-math "))
        .expect("alice-sdf depends on alice-det-math");
    let version = match dep.split_once(' ') {
        Some((_, v)) => v.to_string(),
        None => blocks
            .iter()
            .find(|b| b.contains("name = \"alice-det-math\""))
            .and_then(|b| b.lines().find(|l| l.starts_with("version = ")))
            .map(|l| {
                l.trim_start_matches("version = ")
                    .trim_matches('"')
                    .to_string()
            })
            .expect("alice-det-math in Cargo.lock"),
    };
    let needle = format!("alice-det-math-{version}");
    let mut found: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(deps)
        .expect("deps dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy().to_string();
            n.starts_with("libalice_det_math-") && n.ends_with(".rlib")
        })
        .filter(|p| {
            // the dep-info file names the source directory, hence the version
            let stem = p.file_stem().unwrap().to_string_lossy().to_string();
            let d = deps.join(format!("{}.d", stem.trim_start_matches("lib")));
            std::fs::read_to_string(d).is_ok_and(|s| s.contains(&needle))
        })
        .map(|p| (std::fs::metadata(&p).unwrap().modified().unwrap(), p))
        .collect();
    found.sort();
    found
        .pop()
        .unwrap_or_else(|| {
            panic!(
                "no libalice_det_math rlib for {version} in {}",
                deps.display()
            )
        })
        .1
}

fn program(scenes: &[Scene], pts: &[Vec3]) -> String {
    let mut s = String::new();
    for (i, sc) in scenes.iter().enumerate() {
        s.push_str(&format!("mod s{i} {{\n"));
        if let Some(a) = sc.alias {
            s.push_str(&format!("use ::alice_det_math as {a};\n"));
        }
        s.push_str(&sc.source);
        s.push_str("}\n");
    }
    s.push_str("type Sdf = fn(f32, f32, f32) -> f32;\n");
    s.push_str("type Nrm = fn(f32, f32, f32) -> (f32, f32, f32);\n");
    s.push_str("const SCENES: &[(Sdf, Nrm)] = &[\n");
    for i in 0..scenes.len() {
        s.push_str(&format!("    (s{i}::sdf, s{i}::normal),\n"));
    }
    s.push_str("];\nconst PTS: &[[u32; 3]] = &[\n");
    for p in pts {
        s.push_str(&format!(
            "    [{:#010x}, {:#010x}, {:#010x}],\n",
            p.x.to_bits(),
            p.y.to_bits(),
            p.z.to_bits()
        ));
    }
    s.push_str(
        "];\nfn main() {\n\
         \x20   use std::fmt::Write;\n\
         \x20   let mut out = String::new();\n\
         \x20   for (f, n) in SCENES {\n\
         \x20       for p in PTS {\n\
         \x20           let (x, y, z) = (f32::from_bits(p[0]), f32::from_bits(p[1]), f32::from_bits(p[2]));\n\
         \x20           let d = f(x, y, z);\n\
         \x20           let (a, b, c) = n(x, y, z);\n\
         \x20           let _ = writeln!(out, \"{} {} {} {}\", d.to_bits(), a.to_bits(), b.to_bits(), c.to_bits());\n\
         \x20       }\n\
         \x20   }\n\
         \x20   print!(\"{out}\");\n\
         }\n",
    );
    s
}

/// Build the generated program with `rustc` and return its stdout.
fn build_and_run(src: &str, opt: u8) -> String {
    let exe_self = std::env::current_exe().unwrap();
    let deps = exe_self.parent().unwrap().to_path_buf();
    let rlib = det_math_rlib(&deps);
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("rust_transpiler_oracle");
    std::fs::create_dir_all(&dir).unwrap();
    let main_rs = dir.join("main.rs");
    std::fs::write(&main_rs, src).unwrap();
    let exe = dir.join(format!("oracle_o{opt}{}", std::env::consts::EXE_SUFFIX));
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let out = Command::new(rustc)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["--edition", "2021", "--crate-type", "bin", "-C"])
        .arg(format!("opt-level={opt}"))
        .arg("--extern")
        .arg(format!("alice_det_math={}", rlib.display()))
        .arg("-L")
        .arg(format!("dependency={}", deps.display()))
        .arg("-D")
        .arg("warnings")
        .arg(&main_rs)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("spawn rustc");
    assert!(
        out.status.success(),
        "rustc failed on the emitted source:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Command::new(&exe).output().expect("run generated program");
    assert!(run.status.success(), "generated program failed");
    String::from_utf8(run.stdout).unwrap()
}

const fn same(a: u32, b: f32) -> bool {
    a == b.to_bits() || (f32::from_bits(a).is_nan() && b.is_nan())
}

#[test]
fn emitted_rust_is_bit_identical_to_eval_compiled() {
    let pts = sample_points();
    let mut scenes = Vec::new();
    let mut rejected = Vec::new();

    for (name, node) in common::corpus::corpus() {
        let Ok(compiled) = CompiledSdf::try_compile(&node) else {
            continue; // not a bytecode scene, nothing to mirror
        };
        match RustSource::transpile(&node) {
            Ok(src) => scenes.push(Scene {
                name: name.to_string(),
                compiled,
                source: src.source().to_string(),
                eps: DEFAULT_NORMAL_EPSILON,
                alias: None,
            }),
            Err(RustTranspileError::UnsupportedOpcode { opcode, index }) => {
                assert!(
                    !is_supported(opcode),
                    "{name}: {opcode:?} rejected but listed"
                );
                assert_eq!(compiled.instructions()[index].opcode, opcode, "{name}");
                rejected.push(format!("{name}:{opcode:?}"));
            }
            Err(e) => panic!("{name}: unexpected error {e}"),
        }
    }
    for (name, node) in compositions() {
        let src = RustSource::transpile(&node)
            .unwrap_or_else(|e| panic!("composition {name} must be supported: {e}"));
        scenes.push(Scene {
            name: name.to_string(),
            compiled: CompiledSdf::compile(&node),
            source: src.source().to_string(),
            eps: DEFAULT_NORMAL_EPSILON,
            alias: None,
        });
    }
    // non-default options: another normal step and a renamed det-math path
    // (a scene that calls det-math, so the alias is used)
    let node = compositions()
        .into_iter()
        .find(|(n, _)| *n == "twist_bend_round_onion")
        .unwrap()
        .1;
    let opts = RustTranspileOptions::default()
        .with_normal_epsilon(0.0137)
        .with_det_math_path("dm");
    scenes.push(Scene {
        name: "custom_options".into(),
        compiled: CompiledSdf::compile(&node),
        source: RustSource::transpile_with(&node, &opts)
            .unwrap()
            .source()
            .to_string(),
        eps: 0.0137,
        alias: Some("dm"),
    });

    // every emitter must be exercised by at least one compared scene
    let used: BTreeSet<String> = scenes
        .iter()
        .flat_map(|s| {
            s.compiled
                .instructions()
                .iter()
                .map(|i| format!("{:?}", i.opcode))
        })
        .collect();
    let all_supported_seen = [
        OpCode::Sphere,
        OpCode::Box3d,
        OpCode::RoundedBox,
        OpCode::Cylinder,
        OpCode::Torus,
        OpCode::Plane,
        OpCode::Capsule,
        OpCode::Cone,
        OpCode::HexPrism,
        OpCode::Link,
        OpCode::InfiniteCylinder,
        OpCode::Gyroid,
        OpCode::Union,
        OpCode::Intersection,
        OpCode::Subtraction,
        OpCode::SmoothUnion,
        OpCode::SmoothIntersection,
        OpCode::SmoothSubtraction,
        OpCode::ChamferUnion,
        OpCode::ChamferIntersection,
        OpCode::ChamferSubtraction,
        OpCode::ExpSmoothUnion,
        OpCode::ExpSmoothIntersection,
        OpCode::ExpSmoothSubtraction,
        OpCode::XOR,
        OpCode::Morph,
        OpCode::MetricBlend,
        OpCode::Translate,
        OpCode::Rotate,
        OpCode::Scale,
        OpCode::ScaleNonUniform,
        OpCode::Twist,
        OpCode::Bend,
        OpCode::RepeatInfinite,
        OpCode::RepeatFinite,
        OpCode::Round,
        OpCode::Onion,
        OpCode::Elongate,
        OpCode::Mirror,
        OpCode::OctantMirror,
        OpCode::Revolution,
        OpCode::Extrude,
        OpCode::Taper,
        OpCode::Displacement,
        OpCode::PolarRepeat,
        OpCode::Shear,
    ];
    let missing: Vec<_> = all_supported_seen
        .iter()
        .filter(|op| {
            assert!(is_supported(**op));
            !used.contains(&format!("{op:?}"))
        })
        .collect();
    assert!(
        missing.is_empty(),
        "supported opcodes no scene exercises: {missing:?}"
    );

    let src = program(&scenes, &pts);
    let mut compared = 0usize;
    let mut failures = Vec::new();
    let mut mismatches = 0usize;
    for opt in [0u8, 3] {
        let out = build_and_run(&src, opt);
        let mut lines = out.lines();
        for sc in &scenes {
            for p in &pts {
                let line = lines.next().expect("one output line per scene and point");
                let v: Vec<u32> = line.split(' ').map(|t| t.parse().unwrap()).collect();
                let d = eval_compiled(&sc.compiled, *p);
                let n = eval_compiled_normal(&sc.compiled, *p, sc.eps);
                for (got, want, what) in [
                    (v[0], d, "d"),
                    (v[1], n.x, "nx"),
                    (v[2], n.y, "ny"),
                    (v[3], n.z, "nz"),
                ] {
                    compared += 1;
                    if same(got, want) {
                        continue;
                    }
                    mismatches += 1;
                    if failures.len() < 20 {
                        failures.push(format!(
                            "O{opt} {} {what} @ {p:?}: emitted {:e} ({got:#010x}) vs eval_compiled {want:e} ({:#010x})",
                            sc.name,
                            f32::from_bits(got),
                            want.to_bits()
                        ));
                    }
                }
            }
        }
        assert!(lines.next().is_none(), "extra output lines");
    }
    let expected = 2 * scenes.len() * pts.len() * 4;
    eprintln!(
        "rust transpiler oracle: {} scenes x {} points x 4 values x 2 opt levels = {compared} bit comparisons; {} corpus entries rejected: {rejected:?}",
        scenes.len(),
        pts.len(),
        rejected.len()
    );
    assert!(compared > 0, "compared nothing");
    assert_eq!(compared, expected);
    assert_eq!(
        mismatches,
        0,
        "{mismatches} mismatches, first:\n{}",
        failures.join("\n")
    );
}

#[test]
fn unsupported_opcodes_are_errors_not_values() {
    let cases: Vec<(SdfNode, OpCode)> = vec![
        (SdfNode::ellipsoid(1.0, 0.5, 0.7), OpCode::Ellipsoid),
        (SdfNode::sphere(1.0).noise(0.1, 2.0, 7), OpCode::Noise),
        (
            SdfNode::sphere(1.0).union(SdfNode::octahedron(0.5).translate(1.0, 0.0, 0.0)),
            OpCode::Octahedron,
        ),
        (
            SdfNode::polygon_2d(vec![Vec2::ZERO, Vec2::X, Vec2::Y], 0.2),
            OpCode::Polygon2D,
        ),
    ];
    for (node, want) in cases {
        match RustSource::transpile(&node) {
            Err(RustTranspileError::UnsupportedOpcode { opcode, index }) => {
                assert_eq!(opcode, want);
                assert_eq!(
                    CompiledSdf::compile(&node).instructions()[index].opcode,
                    want
                );
            }
            other => panic!("{want:?}: expected UnsupportedOpcode, got {other:?}"),
        }
    }
}

#[test]
fn degenerate_normal_epsilon_is_rejected() {
    let node = SdfNode::sphere(1.0);
    for eps in [
        0.0f32,
        -0.0,
        -1e-3,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        let opts = RustTranspileOptions::default().with_normal_epsilon(eps);
        match RustSource::transpile_with(&node, &opts) {
            Err(RustTranspileError::InvalidNormalEpsilon(e)) => {
                assert_eq!(e.to_bits(), eps.to_bits());
            }
            other => panic!("eps {eps}: expected InvalidNormalEpsilon, got {other:?}"),
        }
    }
    // smallest positive normal value is accepted
    let opts = RustTranspileOptions::default().with_normal_epsilon(f32::MIN_POSITIVE);
    assert!(RustSource::transpile_with(&node, &opts).is_ok());
}
