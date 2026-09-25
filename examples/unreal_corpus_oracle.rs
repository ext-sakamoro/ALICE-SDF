//! Unreal corpus oracle — generates what the UE5 plugin's automation tests
//! compare the engine against.
//!
//! The plugin (`unreal-plugin/`) reaches alice-sdf two ways: the C FFI
//! (`alice_sdf.dll`) and the HLSL transpiler. Neither had an oracle inside
//! Unreal until 3.2.0 — the DLL was last rebuilt for 1.7.2 and the C++ never
//! compiled on UE 5.7. `AliceSdfCorpusOracleTest.cpp` now runs, inside a
//! real editor process on a real DX12 device, the same corpus every other
//! parity test uses (`tests/common/corpus.rs`):
//!
//! 1. **FFI parity** — every corpus node saved as `.asdf`, loaded back through
//!    `alice_sdf_load`, evaluated with `alice_sdf_eval` / `alice_sdf_eval_compiled`
//!    on the `test_det_golden` grid, compared *bit for bit* with the tree
//!    evaluator's answer recorded here (`<name>.golden`, one hex `f32` per
//!    point, `points.txt` = the grid).
//! 2. **HLSL GPU oracle** — every shader-capable node's HLSL is written to
//!    `Shaders/CorpusOracle/Corpus/<name>.ush`; `SdfCorpusOracle.usf` selects one
//!    per `ALICE_CORPUS_NODE` permutation of a global compute shader and the
//!    test dispatches it, reads back, and compares with the DLL at the same
//!    `1e-4 · max(|d|, 1)` tolerance as `test_gpu_law_parity`. This is the
//!    HLSL execution oracle the crate did not have (naga has no HLSL front
//!    end, so the wgpu lanes only cover WGSL and GLSL).
//!
//! `Shaders/CorpusOracle/` (not `Shaders/Generated/`: UE reserves
//! `<shader virtual dir>/Generated` for C++-generated shader files and
//! asserts when a mapped directory contains one).
//!
//! The generated shader files and the C++ manifest header are committed so
//! the plugin is self-contained; CI regenerates them and fails on drift
//! (`git diff --exit-code`). The `.asdf` + golden files are *not* committed —
//! CI writes them to `ALICE_SDF_GOLDEN_DIR` before launching the editor.
//!
//! # Running
//! ```bash
//! cargo run --example unreal_corpus_oracle --features "ffi,hlsl" -- \
//!     --plugin unreal-plugin --golden target/unreal-golden
//! ```
//!
//! Author: Moroya Sakamoto

#[path = "../tests/common/corpus.rs"]
mod corpus;

use alice_sdf::compiled::{shader_unsupported_nodes, CompiledSdf, HlslShader, HlslTranspileMode};
use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

/// Same grid as `tests/test_det_golden.rs::grid` — 512 LCG points in a ±3
/// box plus the axis / cell / sector ties. Keep the two in sync (the golden
/// hashes there and the golden bits here must describe the same points).
fn grid() -> Vec<Vec3> {
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32) * 6.0 - 3.0
    };
    let mut pts: Vec<Vec3> = (0..512)
        .map(|_| Vec3::new(next(), next(), next()))
        .collect();
    for i in -6..=6 {
        let v = i as f32 * 0.5;
        pts.push(Vec3::new(v, 0.0, 0.0));
        pts.push(Vec3::new(0.0, v, 0.0));
        pts.push(Vec3::new(0.0, 0.0, v));
        pts.push(Vec3::new(v, v, -v));
        pts.push(Vec3::new(-v, 0.25, 0.0));
    }
    pts
}

/// Number of leading grid points that are LCG samples (the rest are ties).
/// The GPU oracle uses only these — exact ties (`atan2(0, -1)`) are
/// platform-dependent on GPUs, as `test_gpu_law_parity` documents.
const RANDOM_POINTS: usize = 512;

struct Args {
    plugin: PathBuf,
    golden: PathBuf,
}

fn parse_args() -> Args {
    let mut plugin = PathBuf::from("unreal-plugin");
    let mut golden = PathBuf::from("target/unreal-golden");
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--plugin" => plugin = PathBuf::from(it.next().expect("--plugin <dir>")),
            "--golden" => golden = PathBuf::from(it.next().expect("--golden <dir>")),
            other => panic!("unknown argument {other} (expected --plugin / --golden)"),
        }
    }
    Args { plugin, golden }
}

/// A corpus name is used as a file stem and as a C identifier fragment.
fn check_name(name: &str) {
    assert!(
        !name.is_empty()
            && name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
        "corpus name `{name}` is not [a-z0-9_]+"
    );
}

fn write_if_changed(path: &Path, contents: &str) -> bool {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create dir");
    }
    let changed = fs::read_to_string(path).ok().as_deref() != Some(contents);
    if changed {
        fs::write(path, contents).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }
    changed
}

fn main() {
    let args = parse_args();
    let pts = grid();
    let corpus = corpus::corpus();

    // ---- golden (not committed) ---------------------------------------
    fs::create_dir_all(&args.golden).expect("golden dir");
    let mut points_txt = String::new();
    for p in &pts {
        writeln!(
            points_txt,
            "{:08x} {:08x} {:08x}",
            p.x.to_bits(),
            p.y.to_bits(),
            p.z.to_bits()
        )
        .unwrap();
    }
    fs::write(args.golden.join("points.txt"), points_txt).expect("points.txt");

    let mut all_names: Vec<&str> = Vec::new();
    // Not every node has a bytecode form (`terrain` carries a heightmap the
    // VM has no opcode for). The manifest records which, so the automation
    // test can require `alice_sdf_compile` to succeed exactly where it should.
    let mut compilable: Vec<bool> = Vec::new();
    let mut shader_names: Vec<&str> = Vec::new();
    let corpus_dir = args.plugin.join("Shaders/CorpusOracle/Corpus");
    let mut shader_files_changed = 0usize;

    for (name, node) in &corpus {
        check_name(name);
        all_names.push(name);
        compilable.push(CompiledSdf::try_compile(node).is_ok());

        // .asdf round trip is part of what the FFI test exercises.
        let tree = SdfTree::new(node.clone());
        alice_sdf::io::save(&tree, args.golden.join(format!("{name}.asdf")))
            .unwrap_or_else(|e| panic!("{name}: save .asdf: {e}"));

        let mut golden = String::new();
        for &p in &pts {
            writeln!(golden, "{:08x}", eval(node, p).to_bits()).unwrap();
        }
        fs::write(args.golden.join(format!("{name}.golden")), golden).expect("golden");

        if shader_unsupported_nodes(node).is_empty() {
            shader_names.push(name);
            let shader = HlslShader::transpile(node, HlslTranspileMode::Hardcoded);
            let src = format!(
                "// GENERATED by examples/unreal_corpus_oracle.rs from tests/common/corpus.rs — do not edit\n\
                 // corpus node `{name}`: HlslShader::transpile(.., Hardcoded)\n\
                 {}",
                shader.source
            );
            if write_if_changed(&corpus_dir.join(format!("{name}.ush")), &src) {
                shader_files_changed += 1;
            }
        }
    }

    // Stale .ush from a renamed / removed corpus node would still compile
    // into nothing; remove anything the corpus no longer names.
    if let Ok(rd) = fs::read_dir(&corpus_dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            if path.extension().and_then(|e| e.to_str()) == Some("ush")
                && !shader_names.contains(&stem.as_str())
            {
                fs::remove_file(&path).expect("remove stale .ush");
                shader_files_changed += 1;
                eprintln!("removed stale {}", path.display());
            }
        }
    }

    // ---- SdfCorpusOracle.usf (committed) -------------------------------
    let mut usf = String::from(
        "// GENERATED by examples/unreal_corpus_oracle.rs — do not edit\n\
         //\n\
         // One permutation per shader-capable corpus node (ALICE_CORPUS_NODE);\n\
         // AliceSdfCorpusOracleTest.cpp dispatches each and compares with the\n\
         // DLL's alice_sdf_eval. Included only when ALICE_SDF_CORPUS_ORACLE=1\n\
         // is set in the editor's environment (FAliceSdfCorpusOracleCS).\n\
         #include \"/Engine/Public/Platform.ush\"\n\n",
    );
    for (i, name) in shader_names.iter().enumerate() {
        let kw = if i == 0 { "#if" } else { "#elif" };
        writeln!(usf, "{kw} ALICE_CORPUS_NODE == {i}").unwrap();
        writeln!(
            usf,
            "#include \"/Plugin/AliceSDF/CorpusOracle/Corpus/{name}.ush\""
        )
        .unwrap();
    }
    usf.push_str(
        "#else\n#error \"ALICE_CORPUS_NODE out of range — regenerate with examples/unreal_corpus_oracle.rs\"\n#endif\n\n\
         StructuredBuffer<float4> Points;\n\
         RWStructuredBuffer<float> Distances;\n\
         uint PointCount;\n\n\
         [numthreads(64, 1, 1)]\n\
         void MainCS(uint3 DispatchThreadId : SV_DispatchThreadID)\n\
         {\n\
         \tif (DispatchThreadId.x >= PointCount)\n\
         \t{\n\
         \t\treturn;\n\
         \t}\n\
         \tDistances[DispatchThreadId.x] = sdf_eval(Points[DispatchThreadId.x].xyz);\n\
         }\n",
    );
    if write_if_changed(
        &args.plugin.join("Shaders/CorpusOracle/SdfCorpusOracle.usf"),
        &usf,
    ) {
        shader_files_changed += 1;
    }

    // ---- C++ manifest (committed) --------------------------------------
    let mut h = String::from(
        "// GENERATED by examples/unreal_corpus_oracle.rs from tests/common/corpus.rs — do not edit\n\
         //\n\
         // The corpus every alice-sdf parity test runs, as seen by the UE5 plugin's\n\
         // automation tests. ALICE_SDF_CORPUS_SHADER_COUNT is the permutation count\n\
         // of FAliceSdfCorpusOracleCS (SdfCorpusOracle.usf); the shader list is the\n\
         // corpus minus shader_unsupported_nodes().\n\
         #pragma once\n\n\
         #include \"CoreMinimal.h\"\n\n",
    );
    writeln!(h, "#define ALICE_SDF_CORPUS_COUNT {}", all_names.len()).unwrap();
    writeln!(
        h,
        "#define ALICE_SDF_CORPUS_SHADER_COUNT {}",
        shader_names.len()
    )
    .unwrap();
    writeln!(h, "#define ALICE_SDF_CORPUS_GRID_POINTS {}", pts.len()).unwrap();
    writeln!(
        h,
        "#define ALICE_SDF_CORPUS_RANDOM_POINTS {RANDOM_POINTS}\n"
    )
    .unwrap();
    h.push_str("static const TCHAR* const GAliceSdfCorpusNames[ALICE_SDF_CORPUS_COUNT] = {\n");
    for name in &all_names {
        writeln!(h, "\tTEXT(\"{name}\"),").unwrap();
    }
    h.push_str("};\n\n");
    h.push_str("// Whether CompiledSdf::try_compile accepts the node (alice_sdf_compile).\n");
    h.push_str("static const bool GAliceSdfCorpusCompilable[ALICE_SDF_CORPUS_COUNT] = {\n");
    for (name, ok) in all_names.iter().zip(&compilable) {
        writeln!(h, "\t{}, // {name}", if *ok { "true" } else { "false" }).unwrap();
    }
    h.push_str("};\n\n");
    h.push_str(
        "static const TCHAR* const GAliceSdfCorpusShaderNames[ALICE_SDF_CORPUS_SHADER_COUNT] = {\n",
    );
    for name in &shader_names {
        writeln!(h, "\tTEXT(\"{name}\"),").unwrap();
    }
    h.push_str("};\n");
    let manifest = args
        .plugin
        .join("Source/AliceSDF/Private/Generated/AliceSdfCorpusManifest.h");
    let manifest_changed = write_if_changed(&manifest, &h);

    println!(
        "unreal corpus oracle: {} nodes ({} shader-capable), {} grid points → {}",
        all_names.len(),
        shader_names.len(),
        pts.len(),
        args.golden.display()
    );
    println!(
        "generated files changed: {} shader, manifest {}",
        shader_files_changed,
        if manifest_changed {
            "changed"
        } else {
            "unchanged"
        }
    );
}
