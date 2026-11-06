//! Local diagnosis only, not for main.
#![cfg(feature = "gpu")]
#![allow(missing_docs, clippy::all, dead_code)]
use alice_sdf::compiled::{GpuEvaluator, TranspileMode, WgslShader};
use alice_sdf::prelude::*;
use glam::Vec3;

#[path = "../src/compiled/wgsl/spirv_no_contraction.rs"]
mod spirv_no_contraction;

fn shapes() -> (String, String) {
    let tw = SdfNode::box3d(0.1, 0.4, 0.05)
        .translate(0.25, 0.0, 0.1)
        .twist(2.0)
        .translate(0.0, 2.0, 0.0);
    let (c, r) = alice_sdf::bound::bounding_sphere(&tw).unwrap();
    let other = SdfNode::torus(r + 0.02, 0.01).translate(c.x, c.y, c.z);
    let node = other.union(tw);
    let strip = |s: String| {
        s.lines()
            .filter(|l| !l.starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    (
        strip(WgslShader::transpile(&node, TranspileMode::Hardcoded).to_compute_shader()),
        strip(
            WgslShader::transpile_with_union_bounds(&node, TranspileMode::Hardcoded)
                .to_compute_shader(),
        ),
    )
}

fn spv(text: &str) -> Vec<u8> {
    use naga::back::spv;
    use naga::proc::{BoundsCheckPolicies, BoundsCheckPolicy};
    let m = naga::front::wgsl::parse_str(text).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&m)
    .unwrap();
    let o = spv::Options {
        lang_version: (1, 0),
        flags: spv::WriterFlags::empty(),
        bounds_check_policies: BoundsCheckPolicies {
            index: BoundsCheckPolicy::Restrict,
            buffer: BoundsCheckPolicy::Restrict,
            image_load: BoundsCheckPolicy::Restrict,
            binding_array: BoundsCheckPolicy::Unchecked,
        },
        zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::Polyfill,
        ..spv::Options::default()
    };
    let w = spv::write_vec(&m, &info, &o, None).unwrap();
    spirv_no_contraction::mark_no_contraction(&w)
        .unwrap()
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect()
}

#[test]
fn dump() {
    let out = std::env::var("RA_OUT").unwrap();
    let (p, b) = shapes();
    std::fs::write(format!("{out}/plain.wgsl"), &p).unwrap();
    std::fs::write(format!("{out}/bounded.wgsl"), &b).unwrap();
    std::fs::write(format!("{out}/plain.spv"), spv(&p)).unwrap();
    std::fs::write(format!("{out}/bounded.spv"), spv(&b)).unwrap();
}

const SENTINEL: f32 = -1.0e30;

fn instrument(text: &str, after: &str, expr: &str) -> String {
    let at = text
        .find(&format!("let {after} = "))
        .unwrap_or_else(|| panic!("{after}"));
    let end = at + text[at..].find(";\n").unwrap() + 2;
    let mut t = format!("{}    dbg = {expr};\n{}", &text[..end], &text[end..]);
    let f = t.find("fn sdf_eval").unwrap();
    t.insert_str(f, "var<private> dbg: f32 = -1.0e30;\n");
    let a = "output_distances[idx].distance = distance;";
    assert!(t.contains(a));
    t.replace(
        a,
        "output_distances[idx].distance = dbg;\n    output_distances[idx]._pad1 = distance;",
    )
}

fn run(text: &str, pts: &[Vec3]) -> Vec<f32> {
    GpuEvaluator::from_wgsl(text)
        .unwrap_or_else(|e| panic!("{e}\n{text}"))
        .into_parts()
        .0
        .eval_batch(pts)
        .unwrap()
}

fn points() -> Vec<Vec3> {
    let mut v = vec![Vec3::new(0.171_363_55, 2.020_705_2, -0.531_694_5)];
    let mut s = 0x7015_u64;
    let mut f = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    for _ in 0..1023 {
        v.push(Vec3::new(
            0.6 * f() - 0.3,
            1.6 + 0.8 * f(),
            0.6 * f() - 0.3 - 0.3,
        ));
    }
    v
}

#[test]
fn repro_sweep() {
    // the final values only, how often and where the two pipelines differ, and which operand won
    let pts = points();
    let (p, b) = shapes();
    let (x, y) = (run(&p, &pts), run(&b, &pts));
    let mut shown = 0;
    for (i, (a, c)) in x.iter().zip(&y).enumerate() {
        if a.to_bits() != c.to_bits() && shown < 6 {
            println!(
                "\nRA diff point {:?} plain {:#x} bounded {:#x}",
                pts[i],
                a.to_bits(),
                c.to_bits()
            );
            shown += 1;
        }
    }
    // the plain shader with its final min as the bounded shader's select, and the bounded shader
    // with the guard conditions constant true, one change at a time
    let select_form = p.replace(
        "let d18 = min(d2, d17);\n    return d18;",
        "var u: f32 = d2;\n    u = min(u, d17);\n    return u;",
    );
    let open_guard = regex_free_open(&b);
    for (name, text) in [
        ("plain with var min", select_form),
        ("bounded guards true", open_guard),
    ] {
        let z = run(&text, &pts);
        let vs_p = z
            .iter()
            .zip(&x)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let vs_b = z
            .iter()
            .zip(&y)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "\nRA variant {name}: repro {:#x}, vs plain {vs_p}, vs bounded {vs_b} of {}",
            z[0].to_bits(),
            pts.len()
        );
    }
}

fn regex_free_open(b: &str) -> String {
    b.lines()
        .map(|l| {
            let t = l.trim_start();
            if t.starts_with("if (!d_ub_set || ") {
                "    if (true) {".to_string()
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every `length(X)` rewritten by `f(X)`.
fn relen(t: &str, f: &dyn Fn(&str) -> String) -> String {
    let mut out = String::new();
    let mut rest = t;
    while let Some(i) = rest.find("length(") {
        out.push_str(&rest[..i]);
        let body = &rest[i + 7..];
        let mut depth = 1;
        let mut end = 0;
        for (j, c) in body.char_indices() {
            if c == '(' {
                depth += 1;
            }
            if c == ')' {
                depth -= 1;
                if depth == 0 {
                    end = j;
                    break;
                }
            }
        }
        out.push_str(&f(&relen(&body[..end], f)));
        rest = &body[end + 1..];
    }
    out.push_str(rest);
    out
}

#[test]
fn lowered_length() {
    let pts = points();
    let (p, b) = shapes();
    let helpers = "fn l2(v: vec2<f32>) -> f32 { return sqrt(v.x * v.x + v.y * v.y); }\nfn l3(v: vec3<f32>) -> f32 { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }\n";
    let by_dot = |x: &str| format!("sqrt(dot(({x}), ({x})))");
    let by_mul = |x: &str| {
        if x == "d0.xz" || x == "d1" {
            format!("l2({x})")
        } else {
            format!("l3({x})")
        }
    };
    let add = |t: String| {
        let f = t.find("fn sdf_eval").unwrap();
        format!("{}{helpers}{}", &t[..f], &t[f..])
    };
    let (x0, y0) = (run(&p, &pts), run(&b, &pts));
    let n = |a: &[f32], c: &[f32]| {
        a.iter()
            .zip(c)
            .filter(|(a, c)| a.to_bits() != c.to_bits())
            .count()
    };
    println!(
        "\nRA lowered builtin length: plain vs bounded differ {}/{}",
        n(&x0, &y0),
        pts.len()
    );
    let (x, y) = (
        run(&relen(&p, &by_dot), &pts),
        run(&relen(&b, &by_dot), &pts),
    );
    println!("\nRA lowered sqrt(dot): plain vs bounded differ {}, vs builtin plain {}, vs builtin bounded {}", n(&x, &y), n(&x, &x0), n(&y, &y0));
    let (x, y) = (
        run(&add(relen(&p, &by_mul)), &pts),
        run(&add(relen(&b, &by_mul)), &pts),
    );
    println!("\nRA lowered sqrt(x*x+..): plain vs bounded differ {}, vs builtin plain {}, vs builtin bounded {}", n(&x, &y), n(&x, &x0), n(&y, &y0));
    // the claim itself: 2-D length next to a 3-D length of the same vector, with no guard at all
    let single = run(&KERNEL.replace("BODY", "length(p.xz)"), &pts);
    let shared = run(
        &KERNEL.replace("BODY", "select(length(p.xz), -1.0, length(p) > 1.0e30)"),
        &pts,
    );
    println!(
        "\nRA length(p.xz) alone vs next to length(p): differ {}/{}",
        n(&single, &shared),
        pts.len()
    );
    let single = run(&KERNEL.replace("BODY", "sqrt(p.x * p.x + p.z * p.z)"), &pts);
    let shared = run(&KERNEL.replace("BODY", "select(sqrt(p.x * p.x + p.z * p.z), -1.0, sqrt(p.x * p.x + p.y * p.y + p.z * p.z) > 1.0e30)"), &pts);
    println!(
        "\nRA sqrt(x*x+z*z) alone vs next to the 3-D form: differ {}/{}",
        n(&single, &shared),
        pts.len()
    );
}

#[test]
fn compare_values() {
    let pts = points();
    let (p, b) = shapes();
    let (x, y) = (run(&p, &pts), run(&b, &pts));
    let diffs = x
        .iter()
        .zip(&y)
        .filter(|(a, c)| a.to_bits() != c.to_bits())
        .count();
    println!(
        "\nRA final unmodified: repro plain {:#x} bounded {:#x}; differ {diffs}/{}",
        x[0].to_bits(),
        y[0].to_bits(),
        pts.len()
    );
    let targets = [
        ("d0", "d0.x"),
        ("d0", "d0.y"),
        ("d0", "d0.z"),
        ("d1", "length(d0.xz)"),
        ("d1", "d1.x"),
        ("d1", "d1.y"),
        ("d1", "length(d1)"),
        ("d2", "d2"),
        ("d3", "d3.x"),
        ("d3", "d3.y"),
        ("d3", "d3.z"),
        ("d4", "d4"),
        ("d5", "d5"),
        ("d6", "d6"),
        ("d7", "d7"),
        ("d8", "d8"),
        ("d9", "d9"),
        ("d12", "d12"),
        ("d13", "d13"),
        ("d14", "d14.x"),
        ("d14", "d14.y"),
        ("d14", "d14.z"),
        ("d15", "d15.x"),
        ("d15", "d15.y"),
        ("d15", "d15.z"),
        ("d16", "d16.x"),
        ("d16", "d16.y"),
        ("d16", "d16.z"),
        ("d16", "length(max(d16, vec3<f32>(0.0)))"),
        ("d16", "min(max(d16.x, max(d16.y, d16.z)), 0.0)"),
        ("d17", "d17"),
    ];
    for (line, expr) in targets {
        let (x, y) = (
            run(&instrument(&p, line, expr), &pts),
            run(&instrument(&b, line, expr), &pts),
        );
        let mut compared = 0;
        let mut d = 0;
        for (a, c) in x.iter().zip(&y) {
            if *c == SENTINEL {
                continue;
            }
            compared += 1;
            if a.to_bits() != c.to_bits() {
                d += 1;
            }
        }
        println!(
            "RA {expr:45} repro plain {:#010x} bounded {:#010x} differ {d}/{compared}",
            x[0].to_bits(),
            y[0].to_bits()
        );
    }
}

const KERNEL: &str = "struct InputPoint { x: f32, y: f32, z: f32, _pad: f32, }
struct OutputDistance { distance: f32, _pad1: f32, _pad2: f32, _pad3: f32, }
@group(0) @binding(0) var<storage, read> input_points: array<InputPoint>;
@group(0) @binding(1) var<storage, read_write> output_distances: array<OutputDistance>;
@group(0) @binding(2) var<uniform> point_count: u32;
fn sdf_eval(p: vec3<f32>) -> f32 { return BODY; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= point_count) { return; }
    let point = input_points[idx];
    output_distances[idx].distance = sdf_eval(vec3<f32>(point.x, point.y, point.z));
}
";

#[test]
fn builtins() {
    let pts = points();
    let two = |p: Vec3| ((p.x * p.x + p.y * p.y) + p.z * p.z).sqrt();
    let fused = |p: Vec3| (p.z.mul_add(p.z, p.x.mul_add(p.x, p.y * p.y))).sqrt();
    let fused2 = |p: Vec3| (p.z.mul_add(p.z, p.y.mul_add(p.y, p.x * p.x))).sqrt();
    for body in [
        "length(p)",
        "sqrt(p.x * p.x + p.y * p.y + p.z * p.z)",
        "sqrt(dot(p, p))",
        "length(p + vec3<f32>(0.0))",
        "length(max(p, vec3<f32>(-9.0)))",
    ] {
        let got = run(&KERNEL.replace("BODY", body), &pts);
        let n = |f: &dyn Fn(Vec3) -> f32| {
            got.iter()
                .zip(&pts)
                .filter(|(g, p)| g.to_bits() != f(**p).to_bits())
                .count()
        };
        println!(
            "\nRA builtin {body:42} vs two-roundings {} vs fused-a {} vs fused-b {} of {}",
            n(&two),
            n(&fused),
            n(&fused2),
            pts.len()
        );
    }
    let refl = run(&KERNEL.replace("BODY", "length(p)"), &pts);
    let dup = run(
        &KERNEL.replace("BODY", "length(p) + 0.0 * length(p.xy)"),
        &pts,
    );
    println!(
        "RA length in two pipelines differ {}/{}",
        refl.iter()
            .zip(&dup)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count(),
        pts.len()
    );
    for body in ["sin(p.x)", "cos(p.x)"] {
        let a = run(&KERNEL.replace("BODY", body), &pts);
        let wrapped = format!("{body} + 0.0 * length(p)");
        let c = run(&KERNEL.replace("BODY", &wrapped), &pts);
        println!(
            "RA {body} two pipelines differ {}/{}",
            a.iter()
                .zip(&c)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count(),
            pts.len()
        );
    }
}
