//! What a GLSL compiler does with `ALICE_PRECISE` (ALICE-Law G3): every corpus
//! node's GLSL, in both transpile modes and every public output a compiler takes
//! as it is (`source` in a compute entry, `to_compute_shader`,
//! `to_vulkan_compute_shader`, `to_fragment_shader`), is compiled to SPIR-V by
//! glslang, and every multiply (`FMul`, `VectorTimesScalar`) whose result an add
//! or a subtract reads must be decorated `NoContraction`, the add too: the SPIR-V
//! form of `precise` (no fusing of a multiply and an add). The text oracle
//! (`test_shader_precise_temporaries.rs`) checks that the qualifier is written;
//! this checks that a compiler reads it as `precise`, on each fusable pair.
//!
//! # Teeth
//!
//! - The same units compiled with `-DALICE_PRECISE=` (the qualifier defined
//!   empty) must carry no `NoContraction` at all: the decorations come from the
//!   qualifier, not from the compiler's defaults.
//! - The version gate of the qualifier: `HASH_NOISE_GLSL` carries the
//!   definition and precise locals. At `#version 450` and `#version 320 es` its
//!   SPIR-V has `NoContraction`; at `#version 330` and `#version 300 es`, which
//!   have no `precise`, it must still compile (the qualifier is empty there; a
//!   `precise` would be a syntax error).
//!
//! # Skips
//!
//! glslang is taken from `ALICE_SDF_GLSLANG` or `PATH`. Without one the tests
//! print that they skipped and pass; `ALICE_SDF_REQUIRE_GLSLANG=1` turns that
//! into a failure (CI sets it). With glslang present, 0 compiled units is a
//! failure.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "glsl")]

mod common;

use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
use alice_sdf::modifiers::HASH_NOISE_GLSL;
use common::corpus::corpus;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The glslang command, or `None` (skip) when there is none.
fn glslang_or_skip(test: &str) -> Option<String> {
    let cand = std::env::var("ALICE_SDF_GLSLANG").unwrap_or_else(|_| "glslang".to_string());
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
        std::env::var_os("ALICE_SDF_REQUIRE_GLSLANG").is_none(),
        "ALICE_SDF_REQUIRE_GLSLANG is set but `{cand}` did not run (set ALICE_SDF_GLSLANG or put glslang on PATH)"
    );
    eprintln!("SKIPPED {test}: no glslang (`{cand} --version` failed); nothing was compiled");
    None
}

fn scratch(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(tag);
    std::fs::create_dir_all(&dir).expect("cannot create scratch dir");
    dir
}

/// Compile `src` (stage taken from the extension of `name`) with the extra
/// arguments; `Ok` is glslang's standard output (the SPIR-V listing with `-H`).
fn compile(
    glslang: &str,
    dir: &Path,
    name: &str,
    src: &str,
    args: &[&str],
) -> Result<String, String> {
    let file = dir.join(name);
    std::fs::write(&file, src).map_err(|e| format!("write {}: {e}", file.display()))?;
    let mut cmd = Command::new(glslang);
    cmd.args(args).arg(&file);
    // `-o` with SPIR-V output (`-V` Vulkan, `-G` OpenGL), into the scratch directory;
    // without either glslang only validates
    if args.contains(&"-V") || args.contains(&"-G") {
        cmd.arg("-o").arg(dir.join(format!("{name}.spv")));
    }
    let out = cmd
        .output()
        .map_err(|e| format!("cannot run {glslang}: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    if out.status.success() {
        Ok(stdout)
    } else {
        Err(format!(
            "{name}: {}{}",
            stdout,
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

fn no_contraction(listing: &str) -> usize {
    listing.matches("NoContraction").count()
}

/// The consumer of `GlslShader::source`: a 450 compute entry that writes
/// `sdf_eval(p)`, with the `params` block a Dynamic source reads.
fn compute_unit(source: &str, mode: GlslTranspileMode) -> String {
    let params = match mode {
        GlslTranspileMode::Dynamic => {
            "layout(std140, binding = 1) uniform SdfParams {\n    vec4 params[1024];\n};\n"
        }
        GlslTranspileMode::Hardcoded => "",
    };
    format!(
        "#version 450\nlayout(local_size_x = 64) in;\nlayout(std430, binding = 0) buffer AliceOut {{ float alice_out[]; }};\n\
         {params}{source}\n\
         void main() {{\n    ALICE_PRECISE vec3 p = vec3(gl_GlobalInvocationID) * 0.01 - 1.0;\n    alice_out[gl_GlobalInvocationID.x] = sdf_eval(p);\n}}\n"
    )
}

/// What the SPIR-V listing (`glslang -H`) says about contraction: the ids decorated
/// `NoContraction`, and every multiply whose result an add or a subtract reads.
#[derive(Default)]
struct Contraction {
    /// (multiply id, add id) pairs where either is not decorated
    undecorated: Vec<(String, String)>,
    /// multiply-to-add pairs read
    pairs: usize,
    decorations: usize,
}

fn read_contraction(listing: &str) -> Contraction {
    let mut nc = std::collections::HashSet::new();
    let mut muls = std::collections::HashSet::new();
    let mut adds: Vec<(String, Vec<String>)> = Vec::new();
    for line in listing.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("Decorate ") {
            if rest.ends_with("NoContraction") {
                nc.insert(rest.split_whitespace().next().unwrap_or("").to_string());
            }
            continue;
        }
        let Some((id, rhs)) = t.split_once(':') else {
            continue;
        };
        if id.is_empty() || !id.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let words: Vec<&str> = rhs.split_whitespace().collect();
        if words.len() < 3 {
            continue;
        }
        match words[1] {
            "FMul" | "VectorTimesScalar" => {
                muls.insert(id.to_string());
            }
            "FAdd" | "FSub" => adds.push((
                id.to_string(),
                words[2..].iter().map(|w| w.to_string()).collect(),
            )),
            _ => {}
        }
    }
    let mut out = Contraction {
        decorations: nc.len(),
        ..Contraction::default()
    };
    for (add, operands) in adds {
        for m in operands.iter().filter(|o| muls.contains(*o)) {
            out.pairs += 1;
            if !nc.contains(m) || !nc.contains(&add) {
                out.undecorated.push((m.clone(), add.clone()));
            }
        }
    }
    out
}

#[test]
fn every_multiply_feeding_an_add_is_no_contraction() {
    let Some(glslang) = glslang_or_skip("every_multiply_feeding_an_add_is_no_contraction") else {
        return;
    };
    let dir = scratch("glslang_precise");
    let (mut units, mut pairs, mut control) = (0, 0, 0);
    let mut per_emitter: Vec<(&str, usize)> = Vec::new();
    let mut problems = Vec::new();
    for (name, node) in corpus() {
        for (tag, mode) in [
            ("hardcoded", GlslTranspileMode::Hardcoded),
            ("dynamic", GlslTranspileMode::Dynamic),
        ] {
            let sh = GlslShader::transpile(&node, mode);
            // every public output that a GLSL compiler takes as it is; `source` in the
            // compute entry its consumer adds. -G: OpenGL rules (std140 uniforms of the
            // OpenGL outputs), -V: Vulkan
            let source_unit = compute_unit(&sh.source, mode);
            let outputs: [(&str, String, &str, &[&str]); 4] = [
                ("source", source_unit.clone(), "comp", &["-V", "-H"]),
                (
                    "to_compute_shader",
                    sh.to_compute_shader(),
                    "comp",
                    &["-G", "--auto-map-locations", "-H"],
                ),
                (
                    "to_vulkan_compute_shader",
                    sh.to_vulkan_compute_shader(),
                    "comp",
                    &["-V", "-H"],
                ),
                (
                    "to_fragment_shader",
                    sh.to_fragment_shader(),
                    "frag",
                    &["-G", "--auto-map-locations", "-H"],
                ),
            ];
            for (emitter, src, stage, args) in outputs {
                let file = format!("{name}.{tag}.{emitter}.{stage}");
                match compile(&glslang, &dir, &file, &src, args) {
                    Ok(listing) => {
                        units += 1;
                        let c = read_contraction(&listing);
                        pairs += c.pairs;
                        match per_emitter.iter_mut().find(|e| e.0 == emitter) {
                            Some(e) => e.1 += c.pairs,
                            None => per_emitter.push((emitter, c.pairs)),
                        }
                        for (m, a) in c.undecorated.iter().take(3) {
                            problems.push(format!(
                                "{name} ({tag}) {emitter}: multiply {m} feeds {a} without NoContraction"
                            ));
                        }
                    }
                    Err(e) => problems.push(e),
                }
            }
            // control: the `source` unit and a wrapper (the fragment shader) with the
            // qualifier defined empty
            for (control_name, src, stage, args) in [
                (
                    "plain",
                    source_unit.clone(),
                    "comp",
                    &["-V", "-H", "-DALICE_PRECISE="][..],
                ),
                (
                    "plain_fragment",
                    sh.to_fragment_shader(),
                    "frag",
                    &["-G", "--auto-map-locations", "-H", "-DALICE_PRECISE="][..],
                ),
            ] {
                let file = format!("{name}.{tag}.{control_name}.{stage}");
                match compile(&glslang, &dir, &file, &src, args) {
                    Ok(listing) => control += read_contraction(&listing).decorations,
                    Err(e) => problems.push(e),
                }
            }
        }
    }
    eprintln!(
        "glslang: {units} units, {pairs} multiply-to-add pairs read, per emitter {per_emitter:?}; NoContraction with ALICE_PRECISE empty: {control}"
    );
    assert!(
        units > 0 && pairs > 0,
        "compared nothing: {units} units, {pairs} pairs"
    );
    for (emitter, n) in &per_emitter {
        assert!(*n > 0, "{emitter}: 0 multiply-to-add pairs read");
    }
    assert_eq!(
        control, 0,
        "NoContraction without the qualifier: the decorations do not come from ALICE_PRECISE"
    );
    assert!(
        problems.is_empty(),
        "{} problems, first 20:\n{}",
        problems.len(),
        problems[..problems.len().min(20)].join("\n")
    );
}

#[test]
fn the_qualifier_is_precise_from_450_and_es_320_and_empty_before() {
    let Some(glslang) =
        glslang_or_skip("the_qualifier_is_precise_from_450_and_es_320_and_empty_before")
    else {
        return;
    };
    let dir = scratch("glslang_precise_versions");
    let body = "layout(location = 0) out vec4 alice_color;\nvoid main() {\n    alice_color = vec4(alice_hash3(gl_FragCoord.xyz, 7u));\n}\n";
    let es = "precision highp float;\nprecision highp int;\n";
    // (version line, extra header, glslang arguments, whether `precise` is expected)
    let cases = [
        ("#version 450", "", &["-V", "-H"][..], true),
        ("#version 320 es", es, &["-V", "-H"][..], true),
        ("#version 330", "", &[][..], false),
        ("#version 300 es", es, &[][..], false),
    ];
    for (version, header, args, precise) in cases {
        let src = format!("{version}\n{header}{HASH_NOISE_GLSL}\n{body}");
        let name = format!("{}.frag", version.replace(['#', ' '], "_"));
        let listing =
            compile(&glslang, &dir, &name, &src, args).unwrap_or_else(|e| panic!("{version}: {e}"));
        if precise {
            assert!(
                no_contraction(&listing) > 0,
                "{version}: no NoContraction, ALICE_PRECISE was not `precise`"
            );
        }
    }
}
