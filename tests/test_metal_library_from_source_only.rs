//! The `metal` dependency is classified as no parser in `scripts/api_boundary.toml`
//! because the Metal backend (`src/compiled/wgsl/gpu_metal.rs`) builds its library
//! only from MSL text this crate generates through naga (`new_library_with_source`).
//! A library or dynamic library loaded from bytes, a file or a URL, the default
//! library (a precompiled `default.metallib` of the bundle) or a binary archive
//! (precompiled pipelines) would read data from outside the process, and the
//! classification would be wrong. This test fails when any file under `src/` names
//! such a call, and when the source call is gone (the check would then compare
//! nothing). On macOS it also reads the locked `metal` crate and fails when its
//! `Device` has a library or archive constructor this list does not name.
//!
//! Author: Moroya Sakamoto

use std::path::{Path, PathBuf};

/// Metal calls that build a library from data that does not come from this crate:
/// the `metal` crate's methods and the Objective-C selectors behind them (a name
/// matches every call it is a prefix of: `new_dynamic_library_with_url`,
/// `new_binary_archive_with_descriptor`, ...)
const FROM_DATA: &[&str] = &[
    "new_library_with_data",
    "new_library_with_file",
    "new_library_with_url",
    "new_dynamic_library",
    "new_default_library",
    "new_binary_archive",
    "newLibraryWithData",
    "newLibraryWithFile",
    "newLibraryWithURL",
    "newDynamicLibrary",
    "newDefaultLibrary",
    "newBinaryArchive",
    "metallib",
];

/// The call the backend uses: a library compiled from MSL text, and its selector
const FROM_SOURCE: &str = "new_library_with_source";
#[cfg(target_os = "macos")]
const FROM_SOURCE_SELECTOR: &str = "newLibraryWithSource";

/// `(line number, name)` of every library-from-data call `text` names
fn from_data_calls(text: &str) -> Vec<(usize, &'static str)> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        for name in FROM_DATA {
            if line.contains(name) {
                out.push((n + 1, *name));
            }
        }
    }
    out
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let p = entry.unwrap().path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn a_planted_library_from_data_call_is_found() {
    let text = "let lib = device.new_library_with_source(&src, &options)?;\n\
                let lib = device.new_library_with_data(&bytes)?;\n";
    assert_eq!(from_data_calls(text), vec![(2, "new_library_with_data")]);
    for name in FROM_DATA {
        assert_eq!(
            from_data_calls(&format!("x.{name}(y)")).len(),
            1,
            "{name} is not found"
        );
    }
}

#[test]
fn the_source_call_alone_is_not_a_finding() {
    assert!(from_data_calls("device.new_library_with_source(&src, &options)").is_empty());
}

#[test]
fn src_builds_metal_libraries_from_source_only() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    assert!(!files.is_empty(), "no file under src/ was read");
    let mut found = Vec::new();
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap();
        for (line, name) in from_data_calls(&text) {
            found.push(format!(
                "{}:{line}: {name}",
                f.strip_prefix(root).unwrap().display()
            ));
        }
    }
    assert!(
        found.is_empty(),
        "a Metal library built from outside data makes the `metal` entry of \
         scripts/api_boundary.toml wrong: {found:?}"
    );
    let backend = std::fs::read_to_string(root.join("src/compiled/wgsl/gpu_metal.rs")).unwrap();
    assert!(
        backend.contains(FROM_SOURCE),
        "the Metal backend no longer calls {FROM_SOURCE}: this check compares nothing"
    );
}

/// The version of `name` in Cargo.lock
#[cfg(target_os = "macos")]
fn locked_version(root: &Path, name: &str) -> String {
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).unwrap();
    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        if line == format!("name = \"{name}\"") {
            let v = lines.next().unwrap();
            return v
                .trim_start_matches("version = \"")
                .trim_end_matches('"')
                .to_string();
        }
    }
    panic!("{name} is not in Cargo.lock");
}

/// `src/device.rs` of the locked `metal` crate, found through `cargo metadata` (the
/// crate is a dependency on macOS, so its source is in the cargo home)
#[cfg(target_os = "macos")]
fn locked_metal_device_rs(root: &Path) -> PathBuf {
    let version = locked_version(root, "metal");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let out = std::process::Command::new(cargo)
        .args(["metadata", "--format-version", "1", "--locked", "--offline"])
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "cargo metadata: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json = String::from_utf8(out.stdout).unwrap();
    let tail = format!("metal-{version}/Cargo.toml\"");
    let end = json
        .find(&tail)
        .unwrap_or_else(|| panic!("metal {version} has no manifest in cargo metadata"));
    let start = json[..end].rfind('"').unwrap() + 1;
    let manifest = PathBuf::from(&json[start..end + tail.len() - 1]);
    manifest.parent().unwrap().join("src/device.rs")
}

/// (the `Device` methods that build a library or a binary archive, the selectors those
/// methods send) in the `metal` crate's `src/device.rs`
#[cfg(target_os = "macos")]
fn library_constructors(text: &str) -> (Vec<String>, Vec<String>) {
    let ident = |s: &str| -> String {
        s.chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect()
    };
    let mut methods = Vec::new();
    let mut selectors = Vec::new();
    for line in text.lines() {
        if let Some(i) = line.find("pub fn new_") {
            let name = ident(&line[i + "pub fn ".len()..]);
            if name.contains("library") || name.contains("binary_archive") {
                methods.push(name);
            }
        }
        if let Some(i) = line.find("msg_send![self, new") {
            let name = ident(&line[i + "msg_send![self, ".len()..]);
            if name.contains("Library") || name.contains("Archive") {
                selectors.push(name);
            }
        }
    }
    methods.sort();
    methods.dedup();
    selectors.sort();
    selectors.dedup();
    (methods, selectors)
}

#[cfg(target_os = "macos")]
#[test]
fn every_library_constructor_of_the_locked_metal_crate_is_guarded() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let device_rs = locked_metal_device_rs(root);
    let (methods, selectors) = library_constructors(&std::fs::read_to_string(&device_rs).unwrap());
    // the source call is among them: the reading of device.rs found what it looks for
    assert!(
        methods.iter().any(|m| m == FROM_SOURCE)
            && selectors.iter().any(|s| s == FROM_SOURCE_SELECTOR),
        "{} enumerated {methods:?} / {selectors:?}: the source call is not among them",
        device_rs.display()
    );
    // the enumeration covers every kind of constructor: a filter narrowed to fewer kinds
    // (libraries only, archives only) would leave the others out of the comparison
    for want in [
        "new_default_library",
        "new_library_with_data",
        "new_dynamic_library",
        "new_binary_archive_with_descriptor",
    ] {
        assert!(
            methods.iter().any(|m| m == want),
            "{want} was not enumerated from {}: {methods:?}",
            device_rs.display()
        );
    }
    for want in [
        "newDefaultLibrary",
        "newLibraryWithData",
        "newDynamicLibrary",
        "newBinaryArchiveWithDescriptor",
    ] {
        assert!(
            selectors.iter().any(|s| s == want),
            "{want} was not enumerated from {}: {selectors:?}",
            device_rs.display()
        );
    }
    // metal 0.31.0 has 7 of each; a newer lock may add more, never fewer without this
    // check being looked at again
    assert!(
        methods.len() >= 7 && selectors.len() >= 7,
        "{} methods / {} selectors enumerated, metal 0.31.0 has 7 / 7: {methods:?} / {selectors:?}",
        methods.len(),
        selectors.len()
    );
    let unguarded: Vec<&String> = methods
        .iter()
        .filter(|m| *m != FROM_SOURCE)
        .chain(selectors.iter().filter(|s| *s != FROM_SOURCE_SELECTOR))
        .filter(|name| from_data_calls(&format!("device.{name}(x)")).is_empty())
        .collect();
    assert!(
        unguarded.is_empty(),
        "constructors of the locked metal crate that FROM_DATA does not name: {unguarded:?} \
         (of {methods:?} / {selectors:?})"
    );
}
