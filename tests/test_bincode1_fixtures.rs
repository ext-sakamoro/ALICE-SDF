//! Fixtures of the binary `.asdf` bytes, captured from the bincode 2 legacy encoder
//!
//! `tests/fixtures/bincode1/` holds the `.asdf` files `save_asdf` wrote while this crate encoded
//! the body with bincode 2's legacy config: one per corpus scene (every `SdfNode` variant the
//! corpus builds, in `corpus.bin` as a little-endian u32 length then the file, names in
//! `corpus.names`) and a left-deep union chain of 500 spheres (`chain500.asdf`). `SHA256SUMS`
//! names each file. The tests pin that the files written now are those bytes, that they load to
//! the same tree, that the ASP I-packet carries the same body bytes, and that writing again gives
//! the same bytes. Each tree's version string is fixed ("fixture") so the bytes do not follow the
//! crate version.
//!
//! Author: Moroya Sakamoto

mod common;
#[path = "common/temp.rs"]
mod temp;

use std::path::{Path, PathBuf};

#[cfg(feature = "asp")]
use alice_sdf::io::ASDF_HEADER_LEN;
use alice_sdf::io::{load_asdf, save_asdf};
use alice_sdf::prelude::*;
use common::corpus::corpus;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bincode1")
}

fn read(name: &str) -> Vec<u8> {
    std::fs::read(dir().join(name)).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

fn sha256_hex(b: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;
    Sha256::digest(b).iter().fold(String::new(), |mut s, x| {
        let _ = write!(s, "{x:02x}");
        s
    })
}

fn tree(root: SdfNode) -> SdfTree {
    SdfTree {
        version: "fixture".to_string(),
        root,
        metadata: None,
    }
}

/// `n` unit spheres joined left-deep: union(union(union(s, s), s), s) ...
fn chain(n: usize) -> SdfNode {
    let mut node = SdfNode::sphere(1.0);
    for i in 1..n {
        node = node.union(SdfNode::sphere(1.0).translate(i as f32, 0.0, 0.0));
    }
    node
}

/// Runs `f` on a thread with a 64 MiB stack (the deep chain recurses once per level in serde's
/// derive, in a debug build)
fn deep<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(f)
        .expect("spawns")
        .join()
        .expect("joins")
}

fn written(t: &SdfTree) -> Vec<u8> {
    let path = temp::TempPath::new("alice_sdf_bincode1", "t.asdf");
    save_asdf(t, &path).expect("save_asdf");
    std::fs::read(&path).expect("reads back")
}

fn loaded(bytes: &[u8]) -> SdfTree {
    let path = temp::TempPath::new("alice_sdf_bincode1", "t.asdf");
    std::fs::write(&path, bytes).expect("writes");
    load_asdf(&path).expect("load_asdf")
}

fn json(t: &SdfTree) -> serde_json::Value {
    serde_json::to_value(t).expect("json")
}

/// The corpus files in `corpus.bin`, in order
fn corpus_files() -> Vec<(String, Vec<u8>)> {
    let names = String::from_utf8(read("corpus.names")).expect("utf-8");
    let bin = read("corpus.bin");
    let mut at = 0;
    let mut out = Vec::new();
    for name in names.lines() {
        let len = u32::from_le_bytes(bin[at..at + 4].try_into().expect("a length")) as usize;
        at += 4;
        out.push((name.to_string(), bin[at..at + len].to_vec()));
        at += len;
    }
    assert_eq!(at, bin.len(), "corpus.bin holds exactly the named files");
    out
}

/// Writes the fixtures (run once with the bincode 2 encoder:
/// `ALICE_SDF_WRITE_BINCODE1_FIXTURES=1 cargo test --test test_bincode1_fixtures write -- --ignored`).
#[test]
#[ignore = "writes tests/fixtures/bincode1; run only to capture"]
fn write_bincode1_fixtures() {
    if std::env::var_os("ALICE_SDF_WRITE_BINCODE1_FIXTURES").is_none() {
        return;
    }
    std::fs::create_dir_all(dir()).unwrap();
    let mut names = String::new();
    let mut bin = Vec::new();
    for (name, node) in corpus() {
        let b = written(&tree(node));
        names.push_str(name);
        names.push('\n');
        bin.extend_from_slice(&u32::try_from(b.len()).unwrap().to_le_bytes());
        bin.extend_from_slice(&b);
    }
    let chain500 = deep(|| written(&tree(chain(500))));
    let files = [
        ("corpus.names", names.into_bytes()),
        ("corpus.bin", bin),
        ("chain500.asdf", chain500),
    ];
    let mut sums = String::new();
    for (name, b) in &files {
        std::fs::write(dir().join(name), b).unwrap();
        sums.push_str(&format!("{}  {name}\n", sha256_hex(b)));
    }
    std::fs::write(dir().join("SHA256SUMS"), sums).unwrap();
}

#[test]
fn every_fixture_matches_its_recorded_sha256() {
    let sums = String::from_utf8(read("SHA256SUMS")).expect("utf-8");
    let mut checked = 0;
    for line in sums.lines() {
        let (hash, name) = line.split_once("  ").expect("`<sha256>  <name>`");
        assert_eq!(sha256_hex(&read(name)), hash, "{name}");
        checked += 1;
    }
    assert_eq!(checked, 3, "SHA256SUMS lists every fixture");
}

#[test]
fn the_files_written_now_are_the_files_bincode_wrote() {
    let files = corpus_files();
    let scenes = corpus();
    assert_eq!(files.len(), scenes.len(), "one file per corpus scene");
    assert!(files.len() > 100, "{}", files.len());
    for ((name, want), (scene, node)) in files.iter().zip(scenes) {
        assert_eq!(name, scene, "corpus order");
        let got = written(&tree(node));
        assert!(got == *want, "{name}: the written file differs");
        assert!(written(&loaded(want)) == *want, "{name}: re-written");
    }
}

#[test]
fn files_bincode_wrote_load_to_the_same_tree() {
    for ((name, bytes), (_, node)) in corpus_files().iter().zip(corpus()) {
        assert_eq!(json(&loaded(bytes)), json(&tree(node)), "{name}");
    }
}

#[test]
fn the_deep_chain_round_trips_through_the_fixture() {
    let want = read("chain500.asdf");
    let (got, back_same) = deep(move || {
        let t = tree(chain(500));
        let got = written(&t);
        let back = loaded(&want);
        (got == want, json(&back) == json(&t))
    });
    assert!(got, "chain500: the written file differs");
    assert!(back_same, "chain500: loads to the same tree");
}

#[test]
fn writing_the_same_tree_again_gives_the_same_bytes() {
    for (name, node) in corpus().into_iter().take(20) {
        let t = tree(node);
        let first = written(&t);
        for round in 0..3 {
            assert!(written(&t) == first, "{name}: round {round}");
        }
    }
}

/// The ASP I-packet carries the `.asdf` body (the file without its header) after the 32-byte
/// semantics identifier, one byte per coefficient entry
#[cfg(feature = "asp")]
#[test]
fn the_asp_i_packet_carries_the_body_bincode_wrote() {
    use alice_sdf::asp_bridge::{create_sdf_i_packet, decode_sdf_i_packet};
    for ((name, file), (_, node)) in corpus_files().iter().zip(corpus()) {
        let t = tree(node);
        let packet = create_sdf_i_packet(&t, 1).expect("packs");
        let i = packet.as_i_packet().expect("an I-packet");
        let coeffs = i.regions[0]
            .dct_coefficients
            .as_ref()
            .expect("coefficients");
        let body: Vec<u8> = coeffs.iter().skip(32).map(|&(_, _, v)| v as u8).collect();
        assert!(
            body == file[ASDF_HEADER_LEN..],
            "{name}: packet body differs"
        );
        assert_eq!(
            json(&decode_sdf_i_packet(&packet).expect("decodes")),
            json(&t),
            "{name}"
        );
    }
}

/// A body nested deeper than the reader's limit (1024 levels; a left-deep chain of `n` unions
/// needs about `2n`) is refused before the recursion can exhaust the stack: 500 loads, 520 does
/// not. The writer has no limit, so the refused file is written by this crate.
#[test]
fn a_body_nested_past_the_limit_is_refused() {
    let (ok500, err520) = deep(|| {
        let ok = loaded(&written(&tree(chain(500))));
        let path = temp::TempPath::new("alice_sdf_bincode1", "deep.asdf");
        save_asdf(&tree(chain(520)), &path).expect("the writer has no depth limit");
        (
            ok.root.node_count(),
            load_asdf(&path).err().map(|e| e.to_string()),
        )
    });
    assert_eq!(ok500, 1498, "500 spheres, 499 translations and 499 unions");
    let e = err520.expect("a 520-chain is refused");
    assert!(e.contains("deep") || e.contains("depth"), "{e}");
}

/// The ASP I-packet reads with the same nesting limit
#[cfg(feature = "asp")]
#[test]
fn an_asp_packet_nested_past_the_limit_is_refused() {
    use alice_sdf::asp_bridge::{create_sdf_i_packet, decode_sdf_i_packet};
    let (ok500, err520) = deep(|| {
        let ok = create_sdf_i_packet(&tree(chain(500)), 1).expect("packs");
        let deep = create_sdf_i_packet(&tree(chain(520)), 2).expect("packs");
        (
            decode_sdf_i_packet(&ok).is_ok(),
            decode_sdf_i_packet(&deep).err(),
        )
    });
    assert!(ok500);
    let e = err520.expect("a 520-chain packet is refused");
    assert!(e.contains("deep") || e.contains("depth"), "{e}");
}

/// Bytes after the tree in the body are refused (the writer never leaves any; the bincode 2 reader
/// ignored them); the CRC is recomputed so the refusal is the decoder's
#[test]
fn a_body_with_bytes_after_the_tree_is_refused() {
    let (_, file) = corpus_files().swap_remove(0);
    let mut body = file[alice_sdf::io::ASDF_HEADER_LEN..].to_vec();
    body.push(0);
    let mut longer = file[..alice_sdf::io::ASDF_HEADER_LEN].to_vec();
    longer[12..16].copy_from_slice(&crc32fast::hash(&body).to_le_bytes());
    longer.extend_from_slice(&body);
    let path = temp::TempPath::new("alice_sdf_bincode1", "longer.asdf");
    std::fs::write(&path, &longer).expect("writes");
    let e = load_asdf(&path).expect_err("refused").to_string();
    assert!(e.contains("used 24 of the 25 input bytes"), "{e}");
    assert!(
        loaded(&file).root.node_count() > 0,
        "the file as written loads"
    );
}
