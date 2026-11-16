//! Every unmangled export in `src/` is left out of this crate's unit-test build.
//!
//! The crate depends on itself (Cargo.toml, the dev-dependency that turns the GPU slot guard on in test builds),
//! so `cargo test --lib` links two copies of it: the unit-test build and the dev-dependency's. Mangled items do
//! not collide; an unmangled export does (the linker sees the same symbol twice). So every export carries
//! `#[cfg_attr(not(test), no_mangle)]` (the function stays, mangled, for the unit tests that call it), and every
//! macro known to emit unmangled symbols is applied through `#[cfg_attr(not(test), <macro>)]` (the item stays, so
//! what it uses is still used in the unit-test build).
//!
//! What this scan cannot see: a derive or function-like macro that emits an unmangled symbol, and an attribute
//! macro from a binding crate that is not in [`NOT_EXPORTING`] or [`EXPORTING`] (that one fails here until it is
//! classified). The proof is the link itself: CI's `cargo test --lib` with the ffi, godot and python features.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

/// Attribute macros that emit unmangled symbols: each use must be `#[cfg_attr(not(test), <macro>)]`
const EXPORTING: &[&str] = &["gdextension", "pymodule"];
/// Attribute macros of the binding crates that emit none (registration and helper attributes)
const NOT_EXPORTING: &[&str] = &[
    "pyclass",
    "pymethods",
    "pyfunction",
    "pyo3",
    "staticmethod",
    "new",
    "godot_api",
    "func",
    "var",
    "class",
];
/// Attributes of the language itself
const BUILTIN: &[&str] = &[
    "cfg",
    "cfg_attr",
    "allow",
    "expect",
    "warn",
    "deny",
    "derive",
    "doc",
    "inline",
    "must_use",
    "test",
    "repr",
    "deprecated",
    "non_exhaustive",
    "track_caller",
    "cold",
    "rustfmt",
    "clippy",
];
/// Files compiled only for a target the unit-test build never uses (wasm32: `#[wasm_bindgen]` exports there)
const OTHER_TARGET_ONLY: &[&str] = &["src/wasm.rs"];
/// The binding crates whose attribute macros are classified above
const BINDING_CRATES: &[&str] = &["use godot", "use pyo3", "use wasm_bindgen", "use uniffi"];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).expect("read src") {
        let p = e.expect("entry").path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

fn attribute_name(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("#[")?;
    let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == ':'))?;
    Some(rest[..end].rsplit("::").next().unwrap_or(&rest[..end]))
}

#[test]
fn every_unmangled_export_is_left_out_of_the_unit_test_build() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    assert!(!files.is_empty(), "no source file scanned");
    let (mut exports, mut macros, mut problems) = (0usize, 0usize, Vec::new());
    for path in &files {
        let rel = path
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let text = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        let binding = BINDING_CRATES.iter().any(|c| text.contains(c));
        for (i, line) in lines.iter().enumerate() {
            let t = line.trim_start();
            if t.starts_with("//") {
                continue;
            }
            if t.contains("no_mangle") || t.contains("export_name") {
                exports += 1;
                if !t.starts_with("#[cfg_attr(not(test), no_mangle)]") {
                    problems.push(format!(
                        "{rel}:{}: `{}` (use #[cfg_attr(not(test), no_mangle)])",
                        i + 1,
                        t
                    ));
                }
                continue;
            }
            if !binding || OTHER_TARGET_ONLY.contains(&rel.as_str()) {
                continue;
            }
            let Some(name) = attribute_name(t) else {
                continue;
            };
            // `#[cfg_attr(not(test), <macro>)]`: the macro is applied only outside the unit-test build
            if let Some(inner) = t.strip_prefix("#[cfg_attr(not(test), ") {
                let gated = inner
                    .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or("");
                if EXPORTING.contains(&gated) {
                    macros += 1;
                }
                continue;
            }
            if EXPORTING.contains(&name) {
                macros += 1;
                let gated = lines[..i]
                    .iter()
                    .rev()
                    .take_while(|l| {
                        l.trim_start().starts_with("#[") || l.trim_start().starts_with("//")
                    })
                    .any(|l| l.trim_start().starts_with("#[cfg(not(test))]"));
                if !gated {
                    problems.push(format!(
                        "{rel}:{}: #[{name}] emits unmangled symbols: write #[cfg_attr(not(test), {name})]",
                        i + 1
                    ));
                }
            } else if !NOT_EXPORTING.contains(&name) && !BUILTIN.contains(&name) {
                problems.push(format!(
                    "{rel}:{}: attribute macro #[{name}] of a binding crate is not classified (EXPORTING or NOT_EXPORTING)",
                    i + 1
                ));
            }
        }
    }
    println!(
        "[unmangled-exports] files {} exports {exports} export macros {macros}",
        files.len()
    );
    assert!(
        exports > 0 && macros > 0,
        "compared nothing: exports {exports}, export macros {macros}"
    );
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn the_wasm_exports_are_compiled_for_wasm32_only() {
    let lib =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs")).unwrap();
    let at = lib
        .find("pub mod wasm;")
        .expect("src/lib.rs declares the wasm module");
    let before = &lib[..at];
    let attr = before.lines().rev().find(|l| !l.trim().is_empty()).unwrap();
    assert!(
        attr.contains("target_arch = \"wasm32\""),
        "the wasm module is not wasm32-only: {attr}"
    );
}
