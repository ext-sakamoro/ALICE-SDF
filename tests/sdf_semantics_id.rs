//! `alice_sdf::SDF_SEMANTICS_ID`: what it folds, and that it moves with each entry
//! (ALICE-Law G6)
//!
//! The expected values come from the sources the entries pin: the golden table is read
//! from `tests/test_det_golden.rs`, the rule entries are the SHA-256 of their text, and
//! the det-math entry is the resolved crate's identifier. The fold is written out here.

use alice_det_math::SemanticsCheck;
use alice_sdf::semantics::{MAX_MIN_RULE, MUL_ADD_RULE, SDF_SEMANTICS_PINS};
use alice_sdf::{
    check_sdf_semantics, PREVIOUS_SDF_SEMANTICS_IDS, SDF_SEMANTICS_BY_DET_MATH, SDF_SEMANTICS_ID,
};
use sha2::{Digest, Sha256};

const GOLDEN_SOURCE: &str = include_str!("test_det_golden.rs");
const RECORDED_HEX: &str = "151079595c1e8d8fc661909bac7ea0dde0ba5a8b3ef500019d01c6af64defb69";

fn sha(b: &[u8]) -> [u8; 32] {
    Sha256::digest(b).into()
}

fn hex(b: &[u8; 32]) -> String {
    b.iter().fold(String::new(), |mut s, x| {
        use std::fmt::Write;
        write!(s, "{x:02x}").expect("writing to a String");
        s
    })
}

fn fold(pins: &[(&str, [u8; 32])]) -> [u8; 32] {
    let mut sorted: Vec<_> = pins.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut h = Sha256::new();
    for (name, digest) in sorted {
        h.update((name.len() as u32).to_be_bytes());
        h.update(name.as_bytes());
        h.update(digest);
    }
    h.finalize().into()
}

/// (name, hash) rows of `GOLDEN` in tests/test_det_golden.rs
fn golden_rows() -> Vec<(String, String)> {
    let start = GOLDEN_SOURCE
        .find("const GOLDEN: &[(&str, &str)] = &[")
        .expect("the GOLDEN table");
    let body = &GOLDEN_SOURCE[start..];
    let body = &body[..body.find("];").expect("the end of the table")];
    let strings: Vec<&str> = body.split('"').skip(1).step_by(2).collect();
    assert_eq!(strings.len() % 2, 0, "rows of two strings");
    strings
        .chunks(2)
        .map(|c| (c[0].to_string(), c[1].to_string()))
        .collect()
}

fn pin(name: &str) -> [u8; 32] {
    SDF_SEMANTICS_PINS
        .iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no pin {name}"))
        .1
}

#[test]
fn every_entry_matches_its_source() {
    let mut rows = golden_rows();
    assert!(rows.len() >= 100, "read {} golden rows", rows.len());
    assert!(rows.iter().all(|(_, h)| h.len() == 64));
    rows.sort();
    let canon = rows.iter().fold(String::new(), |mut s, (n, h)| {
        use std::fmt::Write;
        writeln!(s, "{n}={h}").expect("writing to a String");
        s
    });
    assert_eq!(
        pin("golden tests/test_det_golden.rs GOLDEN"),
        sha(canon.as_bytes())
    );
    assert_eq!(pin("tag max / min rule"), sha(MAX_MIN_RULE.as_bytes()));
    assert_eq!(
        pin("tag Real::mul_add rounds twice"),
        sha(MUL_ADD_RULE.as_bytes())
    );
    assert_eq!(pin("alice-det-math"), alice_det_math::SEMANTICS_ID);
    assert_eq!(SDF_SEMANTICS_PINS.len(), 4);
}

#[test]
fn every_row_is_the_fold_of_the_pins_and_this_build_uses_its_row() {
    assert!(!SDF_SEMANTICS_BY_DET_MATH.is_empty());
    for (det_math, id) in SDF_SEMANTICS_BY_DET_MATH {
        let pins: Vec<(&str, [u8; 32])> = SDF_SEMANTICS_PINS
            .iter()
            .map(|&(n, d)| {
                if n == "alice-det-math" {
                    (n, *det_math)
                } else {
                    (n, d)
                }
            })
            .collect();
        assert_eq!(fold(&pins), *id, "row for {}", hex(det_math));
    }
    assert_eq!(SDF_SEMANTICS_ID, fold(SDF_SEMANTICS_PINS));
    assert_eq!(hex(&SDF_SEMANTICS_ID), RECORDED_HEX);
    assert_ne!(SDF_SEMANTICS_ID, alice_det_math::SEMANTICS_ID);
}

#[test]
fn the_identifier_moves_with_each_entry() {
    for i in 0..SDF_SEMANTICS_PINS.len() {
        let mut pins = SDF_SEMANTICS_PINS.to_vec();
        pins[i].1[0] ^= 1;
        assert_ne!(fold(&pins), SDF_SEMANTICS_ID, "{}", pins[i].0);
    }
}

#[test]
fn check_reads_this_previous_and_unknown_identifiers() {
    assert_eq!(
        check_sdf_semantics(&SDF_SEMANTICS_ID),
        SemanticsCheck::Verified
    );
    assert!(!PREVIOUS_SDF_SEMANTICS_IDS.is_empty());
    for p in PREVIOUS_SDF_SEMANTICS_IDS {
        assert_eq!(check_sdf_semantics(p), SemanticsCheck::VerifiedPrevious(*p));
    }
    // the identifier 6.1 exposed as its own is a previous one
    assert!(PREVIOUS_SDF_SEMANTICS_IDS.contains(&alice_det_math::SEMANTICS_ID));
    assert_eq!(check_sdf_semantics(&[0u8; 32]), SemanticsCheck::Mismatch);
}
