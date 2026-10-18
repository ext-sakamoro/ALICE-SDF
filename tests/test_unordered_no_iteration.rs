//! The library's hash collections cannot be iterated (ALICE-Law G13)
//!
//! `src/unordered.rs` wraps `HashMap` / `HashSet` without `iter`, `IntoIterator` or
//! `Deref`, so code that would observe their unspecified order does not compile.
//! Each case compiles that module with one function added, with `rustc`, and checks the
//! error says why: an iteration that is not there, not an unrelated type error (the
//! `compile_fail` doctests of the public module check the error code only). The control
//! uses the allowed methods and must compile.

use std::path::{Path, PathBuf};
use std::process::Command;

fn compile(case: &str) -> (bool, String) {
    let module = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/unordered.rs");
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("unordered-case-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let lib: PathBuf = dir.join("lib.rs");
    // a Rust string literal of the path (quoted and escaped, also on Windows)
    let module = format!("{:?}", module.to_string_lossy());
    let src = format!(
        "#![allow(dead_code)]\n#[path = {module}]\nmod unordered;\nuse unordered::{{UnorderedMap, UnorderedSet}};\n{case}\n"
    );
    std::fs::write(&lib, src).unwrap();
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let out = Command::new(rustc)
        .args([
            "--edition",
            "2021",
            "--crate-type",
            "lib",
            "--emit",
            "metadata",
            "-o",
        ])
        .arg(dir.join("out.rmeta"))
        .arg(&lib)
        .output()
        .expect("run rustc");
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn assert_refused(case: &str, why: &[&str]) {
    let (ok, err) = compile(case);
    assert!(!ok, "compiled, but must not:\n{case}");
    assert!(
        why.iter().any(|w| err.contains(w)),
        "refused for another reason:\n{err}"
    );
}

#[test]
fn the_allowed_methods_compile() {
    let (ok, err) = compile(
        "pub fn control(m: &mut UnorderedMap<u32, u32>, s: &mut UnorderedSet<u32>) -> usize {
            m.insert(1, 2);
            *m.entry(3).or_insert(0) += 1;
            s.insert(4);
            let _ = (m.get(&1), m[&1], m.contains_key(&3), s.contains(&4), m.sorted_entries(), m.sorted_keys());
            m.count_values_where(|&v| v > 0) + m.sum_values_by(|&v| v as usize) + m.max_value().unwrap_or(0) as usize + s.len()
        }",
    );
    assert!(ok, "the control must compile:\n{err}");
}

#[test]
fn into_iter_on_a_map_does_not_compile() {
    assert_refused(
        "pub fn a(m: &UnorderedMap<u32, u32>) -> Vec<u32> { m.clone().into_iter().map(|(k, _)| k).collect() }",
        &["is not an iterator", "into_iter"],
    );
}

#[test]
fn a_for_loop_over_a_map_does_not_compile() {
    assert_refused(
        "pub fn b(m: &UnorderedMap<u32, u32>) -> Vec<u32> { let mut v = Vec::new(); for (k, _) in m { v.extend([*k]); } v }",
        &["is not an iterator"],
    );
}

#[test]
fn iter_on_a_map_or_a_set_does_not_compile() {
    assert_refused(
        "pub fn c(m: &UnorderedMap<u32, u32>) -> usize { m.iter().count() }",
        &["no method named `iter`"],
    );
    assert_refused(
        "pub fn d(s: &UnorderedSet<u32>) -> usize { s.iter().count() }",
        &["no method named `iter`"],
    );
    assert_refused(
        "pub fn e(s: &UnorderedSet<u32>) -> usize { let mut n = 0; for _ in s { n += 1; } n }",
        &["is not an iterator"],
    );
}

#[test]
fn the_inner_collection_is_not_reachable() {
    assert_refused(
        "pub fn f(m: &UnorderedMap<u32, u32>) -> usize { m.0.iter().count() }",
        &["is private"],
    );
    assert_refused(
        "pub fn g(m: &UnorderedMap<u32, u32>) -> usize { (*m).iter().count() }",
        &["no method named `iter`", "cannot be dereferenced"],
    );
}
