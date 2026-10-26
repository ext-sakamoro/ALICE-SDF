//! ALICE-Law G8: every file the examples write is checked against the shader rule: a text with a
//! shader entry point (a whole shader) carries the semantics identifier, a text without one (a
//! fragment pasted into a host file, a list a host build reads) carries none, and the generated
//! Rust module, whole by rule, carries it
//!
//! The examples themselves run here, into temporary directories: `shader_export::run`,
//! `msl_emit::write_kernel`, `rust_transpile::write_source` and `unreal_corpus_oracle::write_plugin`
//! (the plugin files it commits). Whether a file is whole is decided by the entry point
//! (`tests/common/shader_text.rs`, the rule `tests/test_shader_text_provenance.rs` applies to the
//! emitters), not by the kind the example chose, so a whole kernel written as a fragment without
//! its tag fails. Every whole text is then written under another identifier and without one, and
//! `examples/support/stored_text.rs` must refuse both and refuse it as a fragment.
//!
//! Oracle: the identifier is the constant `SDF_SEMANTICS_ID` (`Provenance::sdf`); the generated
//! Rust must name it, so a semantics move without re-emission fails here.
//!
//! Author: Moroya Sakamoto

#![cfg(all(
    feature = "msl",
    feature = "glsl",
    feature = "hlsl",
    feature = "blinkscript",
    feature = "rust"
))]
#![allow(
    clippy::duplicate_mod,
    reason = "each example is its own crate and includes examples/support/stored_text.rs; this test includes the examples together"
)]

#[path = "../examples/support/stored_text.rs"]
mod stored_text;

#[path = "common/shader_text.rs"]
mod shader_text;

// ALLOW-DEAD: each example's main and its own helpers are not called from this test
#[allow(dead_code)]
#[path = "../examples/msl_emit.rs"]
mod msl_emit;

// ALLOW-DEAD: each example's main and its own helpers are not called from this test
#[allow(dead_code)]
#[path = "../examples/rust_transpile.rs"]
mod rust_transpile;

// ALLOW-DEAD: each example's main and its own helpers are not called from this test
#[allow(dead_code)]
#[path = "../examples/shader_export.rs"]
mod shader_export;

// ALLOW-DEAD: each example's main and its own helpers are not called from this test
#[allow(dead_code)]
#[path = "../examples/unreal_corpus_oracle.rs"]
mod unreal_corpus_oracle;

#[path = "common/temp.rs"]
mod temp;

use alice_sdf::compiled::rust::RustSource;
use alice_sdf::prelude::*;
use alice_sdf::provenance::{self, Provenance, StoredSemantics, KEY};
use alice_sdf::SDF_SEMANTICS_ID;
use shader_text::has_entry_point;
use std::path::{Path, PathBuf};

/// The example writer of whole text, as `provenance_writers.py` names it
const WRITERS: &[(&str, &str)] = &[(
    "examples/support/stored_text.rs::write_whole_text",
    "whole shader text and generated Rust",
)];

fn dir(tag: &str) -> temp::TempDir {
    temp::TempDir::new(&format!("alice_sdf_g8_example_text_{tag}"))
}

fn files_under(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut todo = vec![root.to_path_buf()];
    while let Some(d) = todo.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                todo.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn shape() -> SdfNode {
    SdfNode::sphere(0.8)
        .smooth_union(SdfNode::torus(1.0, 0.2), 0.15)
        .translate(0.0, 0.25, 0.0)
}

/// Runs every example's writer, each into its own directory: (example, files it wrote)
/// What each example wrote, with the directories that hold it (dropping them removes the files, so the
/// caller keeps them while it reads)
fn written_by_the_examples(tag: &str) -> (Vec<(&'static str, Vec<PathBuf>)>, Vec<temp::TempDir>) {
    let shaders = dir(&format!("{tag}_shader_export"));
    let mut listed = shader_export::run(Some(&shaders));
    let msl = dir(&format!("{tag}_msl_emit"));
    msl_emit::write_kernel(&msl.join("kernel.metal"), &msl_emit::kernel(&shape())).unwrap();
    let rust = dir(&format!("{tag}_rust_transpile"));
    rust_transpile::write_source(
        &rust.join("scene.rs"),
        &RustSource::transpile(&shape()).unwrap(),
    )
    .unwrap();
    let plugin = dir(&format!("{tag}_unreal_corpus_oracle"));
    let files = unreal_corpus_oracle::write_plugin(&plugin, 1);
    assert!(files.shader_nodes > 0 && files.manifest_changed);
    let out = vec![
        ("shader_export", files_under(&shaders)),
        ("msl_emit", files_under(&msl)),
        ("rust_transpile", files_under(&rust)),
        ("unreal_corpus_oracle", files_under(&plugin)),
    ];
    // what shader_export says it wrote is what is on disk
    listed.sort();
    assert_eq!(listed, out[0].1);
    // one .ush per shader-capable node, the .usf and the manifest header
    assert_eq!(out[3].1.len(), files.shader_nodes + 2);
    (out, vec![shaders, msl, rust, plugin])
}

/// Whether the file at `path` is whole by the rule: generated Rust is, shader text is when it has
/// an entry point
fn is_whole(path: &Path, text: &str) -> bool {
    path.extension().is_some_and(|e| e == "rs") || has_entry_point(text)
}

/// The whole texts the examples wrote: (example and file name, text)
fn whole_texts(tag: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let (written, _dirs) = written_by_the_examples(tag);
    for (example, files) in written {
        for path in files {
            let text = std::fs::read_to_string(&path).unwrap();
            if is_whole(&path, &text) {
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                out.push((format!("{example}_{name}"), text));
            }
        }
    }
    assert!(out.len() >= 9, "{} whole texts", out.len());
    out
}

#[test]
fn the_writer_table_names_the_example_writer() {
    assert_eq!(WRITERS.len(), 1);
    assert!(WRITERS[0]
        .0
        .starts_with("examples/support/stored_text.rs::"));
}

#[test]
fn every_file_the_examples_write_follows_the_shader_rule() {
    let (mut whole, mut fragments) = (0, 0);
    let (written, _dirs) = written_by_the_examples("rule");
    for (example, files) in written {
        assert!(!files.is_empty(), "{example} wrote nothing");
        for path in files {
            let text = std::fs::read_to_string(&path).unwrap();
            if is_whole(&path, &text) {
                whole += 1;
                assert_eq!(
                    provenance::read(&path).map(|p| p.semantics_id).ok(),
                    Some(SDF_SEMANTICS_ID),
                    "{example}: {} is whole and must carry the identifier",
                    path.display()
                );
                assert_eq!(
                    provenance::check_sdf(&path).unwrap(),
                    StoredSemantics::Current,
                    "{}",
                    path.display()
                );
            } else {
                fragments += 1;
                assert!(
                    !text.contains(KEY),
                    "{example}: {} has no entry point and must carry no identifier",
                    path.display()
                );
            }
        }
    }
    // the examples write both kinds: 6 whole kernels of shader_export, the MSL, the Rust and the
    // .usf; 6 fragments of shader_export, a .ush per node and the manifest header
    assert!(whole >= 9, "{whole} whole files");
    assert!(fragments >= 8, "{fragments} fragments");
}

#[test]
fn a_whole_text_under_another_identifier_is_refused_naming_both() {
    let other = Provenance::new([0xAB; 32]);
    let d = dir("other");
    for (name, text) in whole_texts("other_src") {
        let moved = text.replace(&Provenance::sdf().hex(), &other.hex());
        assert_ne!(moved, text, "{name}: the identifier is in the text");
        // the writer refuses it, naming both identifiers
        let path = d.join(&name);
        let e = stored_text::write_whole_text(&path, &moved)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains(&other.hex()) && e.contains(&Provenance::sdf().hex()),
            "{name}: {e}"
        );
        assert!(!path.exists(), "{name}: nothing is written");
        // and the crate's reader refuses such a file, naming both
        std::fs::write(&path, &moved).unwrap();
        let e = provenance::check_sdf(&path).unwrap_err().to_string();
        assert!(
            e.contains(&other.hex()) && e.contains(&Provenance::sdf().hex()),
            "{name}: {e}"
        );
    }
}

#[test]
fn a_whole_text_without_the_identifier_is_written_neither_way() {
    let d = dir("bare");
    for (name, text) in whole_texts("bare_src") {
        let mut bare = String::new();
        for line in text.lines().filter(|l| !l.contains(KEY)) {
            bare.push_str(line);
            bare.push('\n');
        }
        let path = d.join(&name);
        assert!(
            stored_text::write_whole_text(&path, &bare).is_err(),
            "{name}"
        );
        assert!(!path.exists(), "{name}");
        // and a whole text is never written as a fragment
        assert!(
            stored_text::write_fragment_text(&path, &text).is_err(),
            "{name}"
        );
        assert!(!path.exists(), "{name}");
    }
}

#[test]
fn the_generated_rust_names_the_current_semantics_identifier() {
    let src = RustSource::transpile(&shape()).unwrap();
    assert_eq!(
        provenance::from_bytes(src.source().as_bytes())
            .unwrap()
            .semantics_id,
        SDF_SEMANTICS_ID
    );
    // in the leading block, before the first item
    let first_item = src.source().find("pub fn sdf").unwrap();
    assert!(src.source()[..first_item].contains(&Provenance::sdf().tagged()));
}
