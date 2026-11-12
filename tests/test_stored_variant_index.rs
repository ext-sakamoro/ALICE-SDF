//! The stored forms (`.asdf`, the ASP bridge: bincode with the legacy config) name a node's variant by
//! its position in `SdfNode`, as a little-endian u32 at the start of the node, at every depth. A
//! variant inserted in the middle of the enum moves every later variant's number, and a file written
//! before reads as a different node (or not at all). New variants go last.
//!
//! `tests/fixtures/sdf_node_variants.tsv` records every variant's number. The order of the variants in
//! `src/types/mod.rs` must keep every recorded name at its number; a variant the table does not have
//! must come after all recorded ones and fails until it is added (append with
//! ALICE_SDF_RECORD_VARIANT_INDEX=1, which never rewrites a recorded number). The order read from the
//! source is checked against the encoder: every corpus scene's root encodes to the number of its
//! variant name (the name from its JSON tag). Nothing compared fails.
//!
//! Author: Moroya Sakamoto

mod common;

use common::corpus::corpus;

const TABLE: &str = "tests/fixtures/sdf_node_variants.tsv";
const SOURCE: &str = include_str!("../src/types/mod.rs");

/// The variant names of `SdfNode`, in declaration order, read from its source
fn declared() -> Vec<String> {
    let start = SOURCE.find("pub enum SdfNode {").expect("the enum");
    let body = &SOURCE[start..];
    let end = body.find("\n}\n").expect("the enum's end");
    body[..end]
        .lines()
        .skip(1)
        .filter_map(|l| {
            let rest = l.strip_prefix("    ")?;
            if rest.starts_with(' ') || !rest.starts_with(|c: char| c.is_ascii_uppercase()) {
                return None;
            }
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            let after = &rest[name.len()..];
            (after.starts_with(" {") || after.starts_with('(') || after.starts_with(','))
                .then_some(name)
        })
        .collect()
}

fn recorded() -> Vec<(u32, String)> {
    std::fs::read_to_string(TABLE)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (n, name) = l.split_once('\t').expect("number<TAB>name");
            (n.parse().expect("a number"), name.to_owned())
        })
        .collect()
}

#[test]
fn every_variant_keeps_its_stored_number() {
    let names = declared();
    assert!(
        names.len() > 100,
        "read {} variants from the source",
        names.len()
    );
    let table = recorded();
    let record = std::env::var_os("ALICE_SDF_RECORD_VARIANT_INDEX").is_some();
    let mut moved = Vec::new();
    for (n, name) in &table {
        match names.iter().position(|x| x == name) {
            Some(i) if i as u32 == *n => {}
            Some(i) => moved.push(format!("{name}: stored as {n}, now {i}")),
            None => moved.push(format!("{name}: stored as {n}, no longer declared")),
        }
    }
    assert!(
        moved.is_empty(),
        "variants moved (one inserted mid-enum?):\n{}",
        moved.join("\n")
    );
    let new: Vec<(usize, &String)> = names.iter().enumerate().skip(table.len()).collect();
    if record && !new.is_empty() {
        let mut text = std::fs::read_to_string(TABLE).unwrap_or_else(|_| {
            "# number<TAB>SdfNode variant: its number in the stored forms (bincode, legacy); append only\n".into()
        });
        for (i, name) in &new {
            text.push_str(&format!("{i}\t{name}\n"));
        }
        std::fs::write(TABLE, text).expect("writes the table");
    } else {
        assert!(
            new.is_empty(),
            "variants not in {TABLE} (append with ALICE_SDF_RECORD_VARIANT_INDEX=1): {new:?}"
        );
    }
    assert!(record || !table.is_empty(), "nothing compared");
}

/// The order read from the source is the encoder's: each corpus scene's root encodes to the number of
/// its variant name
#[test]
fn the_declared_order_is_the_encoders() {
    let names = declared();
    let mut compared = 0;
    for (scene, node) in corpus() {
        let bytes =
            bincode::serde::encode_to_vec(&node, bincode::config::legacy()).expect("encodes");
        let got = u32::from_le_bytes(bytes[..4].try_into().expect("a variant number"));
        let json = serde_json::to_value(&node).expect("json");
        let name = json
            .as_object()
            .and_then(|o| o.keys().next().cloned())
            .or_else(|| json.as_str().map(str::to_owned))
            .expect("an externally tagged variant");
        let want = names
            .iter()
            .position(|x| *x == name)
            .unwrap_or_else(|| panic!("{scene}: {name} not declared"));
        assert_eq!(got as usize, want, "{scene}: {name}");
        compared += 1;
    }
    assert!(compared > 100, "{compared}");
}
