//! HLSL **compile** gate: every HLSL output of `HlslShader`, for every corpus
//! node in both transpile modes, is compiled by the DirectX Shader Compiler
//! (`dxc`) inside the smallest wrapper its consumer puts around it.
//!
//! The value oracles (`test_hlsl_blinkscript_parity.rs`,
//! `test_hlsl_export_oracle.rs`) build the emit as C++ against a shim, which
//! accepts spellings HLSL rejects (and the other way round). This test asks a
//! real HLSL front end whether the text is HLSL at all.
//!
//! | output                                | wrapper (what the consumer does)                         | profile  |
//! |---------------------------------------|----------------------------------------------------------|----------|
//! | `source`                              | compute entry calling `sdf_eval`                         | `cs_6_0` |
//! | `to_compute_shader`                   | none (complete shader, entry `main`)                     | `cs_6_0` |
//! | `export_ue5_material_function`        | `#include`d (as the `.ush`) by a compute entry calling `AliceSdf_Eval` / `AliceSdf_Normal` | `cs_6_0` |
//! | `to_ue5_custom_node`                  | `float CustomExpression0(float3 p) { <body> }` as UE does, called from a compute entry | `cs_6_0` |
//! | `to_unity_custom_function`            | `#include`d by a pixel shader calling `SdfEval_float`    | `ps_6_0` |
//! | `export_unity_shader_graph`           | `#include`d by a pixel shader calling `AliceSdf_float` and `AliceSdf_half` | `ps_6_0` |
//!
//! In Dynamic mode the bare `source`, the Custom node body and the compute
//! entry of the Custom node read the global `params`, which the consumer
//! declares; the wrapper declares it as `cbuffer SdfParams { float4 params[1024]; }`.
//! The other outputs declare their own parameters and get no extra declaration.
//!
//! # Teeth
//!
//! `dxc_rejects_known_bad_outputs` compiles three deliberately broken outputs
//! (the old Custom node body with nested function definitions, `float3`
//! spelled `vec3`, and a data array as an initialised struct field) and
//! requires every one of them to fail. Without it a `dxc` that accepted
//! anything, or a wrapper that never reached the emit, would pass.
//!
//! # Skips
//!
//! `dxc` is taken from `ALICE_SDF_DXC` or `PATH`. Without one the tests print
//! that they skipped and pass; `ALICE_SDF_REQUIRE_DXC=1` turns that into a
//! failure (CI sets it). With `dxc` present, 0 compiled units is a failure.
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "hlsl")]

mod common;

use alice_sdf::compiled::hlsl::{HlslShader, HlslTranspileMode};
use common::corpus::corpus;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The `dxc` command, or `None` (skip) when there is none.
fn dxc_or_skip(test: &str) -> Option<String> {
    let cand = std::env::var("ALICE_SDF_DXC").unwrap_or_else(|_| "dxc".to_string());
    let ok = Command::new(&cand)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if ok {
        return Some(cand);
    }
    assert!(
        std::env::var_os("ALICE_SDF_REQUIRE_DXC").is_none(),
        "ALICE_SDF_REQUIRE_DXC is set but `{cand}` did not run (set ALICE_SDF_DXC or put dxc on PATH)"
    );
    eprintln!("SKIPPED {test}: no DirectX Shader Compiler (`{cand} --version` failed); nothing was compiled");
    None
}

/// Scratch directory for the generated `.hlsl` files.
fn scratch(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(tag);
    std::fs::create_dir_all(&dir).expect("cannot create scratch dir");
    dir
}

/// One translation unit for `dxc`.
struct Unit {
    name: String,
    profile: &'static str,
    entry: &'static str,
    /// A file the unit `#include`s first, as its consumer does with an
    /// include file (UE `.ush`, Unity Custom Function `Type: File`).
    include: Option<String>,
    src: String,
}

const DYNAMIC_PARAMS: &str = "cbuffer SdfParams : register(b1) {\n    float4 params[1024];\n};\n";

/// The `params` declaration the consumer of a bare output adds in Dynamic mode.
const fn consumer_params(mode: HlslTranspileMode) -> &'static str {
    match mode {
        HlslTranspileMode::Dynamic => DYNAMIC_PARAMS,
        HlslTranspileMode::Hardcoded => "",
    }
}

/// A compute entry that writes `expr` (a `float`, may use `p`) per thread.
fn compute_entry(expr: &str) -> String {
    format!(
        "\nRWStructuredBuffer<float> alice_out : register(u0);\n\
         [numthreads(64, 1, 1)]\n\
         void cs_main(uint3 id : SV_DispatchThreadID) {{\n\
         \x20   float3 p = float3((float)id.x, (float)id.y, (float)id.z) * 0.01 - 1.0;\n\
         \x20   alice_out[id.x] = {expr};\n\
         }}\n"
    )
}

/// A pixel entry that writes `body` (statements ending in `return <float4>;`)
/// with the world position `p` as input.
fn pixel_entry(body: &str) -> String {
    format!(
        "\nfloat4 ps_main(float4 pos : SV_Position, float3 p : TEXCOORD0) : SV_Target {{\n\
         {body}\n\
         }}\n"
    )
}

/// How UE wraps a Custom node body: a function whose parameters are the
/// node's inputs (here the one input `p`), returning the node's output.
fn ue_custom_expression(body: &str) -> String {
    format!("float CustomExpression0(float3 p) {{\n{body}\n}}\n")
}

/// Every HLSL output of one shader, wrapped for its consumer.
fn units_for(name: &str, tag: &str, sh: &HlslShader) -> Vec<Unit> {
    let params = consumer_params(sh.mode);
    vec![
        Unit {
            name: format!("{name}.{tag}.source"),
            profile: "cs_6_0",
            entry: "cs_main",
            include: None,
            src: format!("{params}{}{}", sh.source, compute_entry("sdf_eval(p)")),
        },
        Unit {
            name: format!("{name}.{tag}.compute_shader"),
            profile: "cs_6_0",
            entry: "main",
            include: None,
            src: sh.to_compute_shader(),
        },
        Unit {
            name: format!("{name}.{tag}.ue5_material_function"),
            profile: "cs_6_0",
            entry: "cs_main",
            include: Some(sh.export_ue5_material_function()),
            src: compute_entry("AliceSdf_Eval(p) + AliceSdf_Normal(p).x"),
        },
        Unit {
            name: format!("{name}.{tag}.ue5_custom_node"),
            profile: "cs_6_0",
            entry: "cs_main",
            include: None,
            src: format!(
                "{params}{}{}",
                ue_custom_expression(&sh.to_ue5_custom_node()),
                compute_entry("CustomExpression0(p)")
            ),
        },
        Unit {
            name: format!("{name}.{tag}.unity_custom_function"),
            profile: "ps_6_0",
            entry: "ps_main",
            include: Some(sh.to_unity_custom_function()),
            src: pixel_entry(
                "    float d;\n    SdfEval_float(p, d);\n    return float4(d, 0.0, 0.0, 1.0);",
            ),
        },
        Unit {
            name: format!("{name}.{tag}.unity_shader_graph"),
            profile: "ps_6_0",
            entry: "ps_main",
            include: Some(sh.export_unity_shader_graph()),
            src: pixel_entry(
                "    float d;\n    float3 n;\n    AliceSdf_float(p, d, n);\n\
                     \x20   half dh;\n    half3 nh;\n    AliceSdf_half(p, dh, nh);\n\
                     \x20   return float4(d + (float)dh, n.x + (float)nh.y, 0.0, 1.0);",
            ),
        },
    ]
}

/// Compile one unit. `Ok` carries the warning lines, `Err` the error lines.
fn compile(dxc: &str, dir: &Path, unit: &Unit) -> Result<Vec<String>, String> {
    let file = dir.join(format!("{}.hlsl", unit.name));
    let obj = dir.join(format!("{}.dxil", unit.name));
    let src = match &unit.include {
        Some(text) => {
            let inc = format!("{}.inc.hlsl", unit.name);
            let path = dir.join(&inc);
            std::fs::write(&path, text).map_err(|e| format!("write {}: {e}", path.display()))?;
            format!("#include \"{inc}\"\n{}", unit.src)
        }
        None => unit.src.clone(),
    };
    std::fs::write(&file, src).map_err(|e| format!("write {}: {e}", file.display()))?;
    // A container left by an earlier run must not count as this run's output.
    let _ = std::fs::remove_file(&obj);
    let out = Command::new(dxc)
        .args(["-T", unit.profile, "-E", unit.entry])
        .arg(&file)
        .arg("-Fo")
        .arg(&obj)
        .output()
        .map_err(|e| format!("cannot run {dxc}: {e}"))?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    if out.status.success() && obj.is_file() {
        return Ok(stderr
            .lines()
            .filter(|l| l.contains("warning:"))
            .map(str::to_string)
            .collect());
    }
    let errors: Vec<&str> = stderr.lines().filter(|l| l.contains("error")).collect();
    if errors.is_empty() {
        return Err(format!("dxc failed without an error line: {}", out.status));
    }
    Err(errors.join("\n"))
}

/// Compile every unit over the available cores; results in `units` order.
fn compile_all(dxc: &str, dir: &Path, units: &[Unit]) -> Vec<Result<Vec<String>, String>> {
    let lanes = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
    let mut results: Vec<(usize, Result<Vec<String>, String>)> = std::thread::scope(|s| {
        // Every lane is spawned before the first `join`, or the lanes run
        // one after another.
        let mut handles = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            handles.push(s.spawn(move || {
                units
                    .iter()
                    .enumerate()
                    .skip(lane)
                    .step_by(lanes)
                    .map(|(i, u)| (i, compile(dxc, dir, u)))
                    .collect::<Vec<_>>()
            }));
        }
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("compile lane panicked"))
            .collect()
    });
    results.sort_by_key(|(i, _)| *i);
    results.into_iter().map(|(_, r)| r).collect()
}

/// Number of HLSL outputs `units_for` produces per shader.
const OUTPUTS: usize = 6;

/// Every corpus node x {Hardcoded, Dynamic} x every output compiles with `dxc`.
#[test]
fn dxc_compiles_every_hlsl_output_for_every_corpus_node() {
    let Some(dxc) = dxc_or_skip("dxc_compiles_every_hlsl_output_for_every_corpus_node") else {
        return;
    };
    let dir = scratch("dxc_corpus");
    let mut units = Vec::new();
    for (name, node) in corpus() {
        for (tag, mode) in [
            ("hardcoded", HlslTranspileMode::Hardcoded),
            ("dynamic", HlslTranspileMode::Dynamic),
        ] {
            units.extend(units_for(name, tag, &HlslShader::transpile(&node, mode)));
        }
    }
    let results = compile_all(&dxc, &dir, &units);
    let failed: Vec<String> = units
        .iter()
        .zip(&results)
        .filter_map(|(u, r)| r.as_ref().err().map(|e| format!("{}:\n{e}", u.name)))
        .collect();
    let compiled = results.iter().filter(|r| r.is_ok()).count();
    // Warnings do not fail the gate; list each distinct one (without the
    // file and line) with the number of units that raised it.
    let mut warnings: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for w in results.iter().filter_map(|r| r.as_ref().ok()).flatten() {
        let msg = w
            .split_once("warning:")
            .map_or(w.as_str(), |(_, m)| m.trim());
        *warnings.entry(msg.to_string()).or_default() += 1;
    }
    for (msg, n) in &warnings {
        eprintln!("dxc warning x{n}: {msg}");
    }
    eprintln!(
        "dxc: compiled {compiled} of {} units ({} corpus nodes x 2 modes x {OUTPUTS} outputs), {} failed",
        units.len(),
        corpus().len(),
        failed.len()
    );
    assert!(
        failed.is_empty(),
        "{} unit(s) did not compile with dxc (sources in {}):\n{}",
        failed.len(),
        dir.display(),
        failed.join("\n")
    );
    assert!(compiled > 0, "dxc compiled 0 units");
    assert_eq!(
        compiled,
        corpus().len() * 2 * OUTPUTS,
        "not every corpus node x mode x output was compiled"
    );
}

/// Rewrite the Custom node body so every module-scope data array of the
/// source is an initialised field of the struct instead of a `static const`
/// local of the member function that reads it.
fn data_array_as_struct_field(sh: &HlslShader) -> Option<String> {
    let arrays: Vec<&str> = sh
        .source
        .lines()
        .filter(|l| l.starts_with("static const float "))
        .collect();
    if arrays.is_empty() {
        return None;
    }
    let mut body = String::new();
    for line in sh.to_ue5_custom_node().lines() {
        if !line.trim_start().starts_with("static const float ") {
            body.push_str(line);
            body.push('\n');
        }
    }
    let mut fields = String::new();
    for line in arrays {
        fields.push_str("    ");
        fields.push_str(line.trim_start_matches("static const "));
        fields.push('\n');
    }
    let open = "struct AliceSdfCustomNode {\n";
    let at = body.find(open).expect("custom node defines its struct") + open.len();
    body.insert_str(at, &fields);
    Some(body)
}

/// The gate fails on outputs that are not HLSL. Each mutation reproduces a
/// form the emit had or could regress to.
#[test]
fn dxc_rejects_known_bad_outputs() {
    let Some(dxc) = dxc_or_skip("dxc_rejects_known_bad_outputs") else {
        return;
    };
    let dir = scratch("dxc_teeth");
    let mut units: Vec<Unit> = Vec::new();
    let mut with_arrays = 0;
    for (name, node) in corpus() {
        let sh = HlslShader::transpile(&node, HlslTranspileMode::Hardcoded);
        // (1) The old Custom node body: helper and `sdf_eval` definitions
        // pasted inside the function UE generates.
        units.push(Unit {
            name: format!("{name}.nested_functions"),
            profile: "cs_6_0",
            entry: "cs_main",
            include: None,
            src: format!(
                "{}{}",
                ue_custom_expression(&format!("{}\nreturn sdf_eval(p);", sh.source)),
                compute_entry("CustomExpression0(p)")
            ),
        });
        // (2) GLSL type spelling.
        units.push(Unit {
            name: format!("{name}.vec3"),
            profile: "cs_6_0",
            entry: "cs_main",
            include: None,
            src: format!(
                "{}{}",
                sh.source.replace("float3", "vec3"),
                compute_entry("sdf_eval(p)")
            ),
        });
        // (3) Data array as an initialised struct field of the Custom node.
        if let Some(body) = data_array_as_struct_field(&sh) {
            with_arrays += 1;
            units.push(Unit {
                name: format!("{name}.array_field"),
                profile: "cs_6_0",
                entry: "cs_main",
                include: None,
                src: format!(
                    "{}{}",
                    ue_custom_expression(&body),
                    compute_entry("CustomExpression0(p)")
                ),
            });
        }
    }
    assert!(
        with_arrays > 0,
        "no corpus node has a module-scope data array: the array mutation measured nothing"
    );
    let results = compile_all(&dxc, &dir, &units);
    let accepted: Vec<&str> = units
        .iter()
        .zip(&results)
        .filter(|(_, r)| r.is_ok())
        .map(|(u, _)| u.name.as_str())
        .collect();
    let rejected = results.iter().filter(|r| r.is_err()).count();
    eprintln!(
        "dxc rejected {rejected} of {} broken units ({with_arrays} with data arrays)",
        units.len()
    );
    // Each unit must be rejected for the defect it carries, not for something
    // the wrapper broke.
    let mut wrong_reason = Vec::new();
    let mut sampled = std::collections::BTreeSet::new();
    for (u, r) in units.iter().zip(&results) {
        let Err(e) = r else { continue };
        let kind = u.name.rsplit('.').next().unwrap_or_default();
        if sampled.insert(kind.to_string()) {
            eprintln!(
                "dxc rejects {kind}, e.g. {}: {}",
                u.name,
                e.lines().next().unwrap_or_default()
            );
        }
        let expected = match kind {
            "nested_functions" => "function definition is not allowed here",
            "vec3" => "unknown type name 'vec3'",
            "array_field" => "struct/class members cannot have default values",
            other => panic!("no expected diagnostic for mutation `{other}`"),
        };
        if !e.contains(expected) {
            wrong_reason.push(format!("{} (expected `{expected}`):\n{e}", u.name));
        }
    }
    assert!(
        accepted.is_empty(),
        "dxc accepted {} deliberately broken unit(s) (sources in {}):\n{}",
        accepted.len(),
        dir.display(),
        accepted.join("\n")
    );
    assert!(
        wrong_reason.is_empty(),
        "dxc rejected {} broken unit(s) for another reason than the defect:\n{}",
        wrong_reason.len(),
        wrong_reason.join("\n")
    );
    assert!(rejected > 0, "0 broken units were compiled");
}
