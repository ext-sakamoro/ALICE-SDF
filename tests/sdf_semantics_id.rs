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
// with alice-det-math 0.5.2 at 39507d1b; 151079595… (with 0.5.1) is a previous identifier
const RECORDED_HEX: &str = "6e98f0bed3f8691ca470ae2fe2c404c893856a3e3f6a9cda9d67528ae8074b37";

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
    // the identifier 7.0.0 stamped (SDF_SEMANTICS_ID with alice-det-math 0.5.1) and the one 6.1
    // exposed as its own (alice-det-math 0.5.1's) are previous ones
    for previous in [
        "151079595c1e8d8fc661909bac7ea0dde0ba5a8b3ef500019d01c6af64defb69",
        "85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315",
    ] {
        assert!(
            PREVIOUS_SDF_SEMANTICS_IDS
                .iter()
                .any(|p| hex(p) == previous),
            "{previous} is not a previous identifier"
        );
    }
    assert_eq!(check_sdf_semantics(&[0u8; 32]), SemanticsCheck::Mismatch);
}

/// The oracle that shows this build reproduces what each earlier identifier stamped:
/// `(identifier, test file, test function)`. The earlier generations differ from this build's
/// alice-det-math (0.5.2 at 39507d1b) only in `metric::clamp`; the oracle shows the one path
/// to it from this crate (`metric::smoothstep`) gives the same bits under both clamps, and the
/// golden table entry (`every_row_is_the_fold_of_the_pins_and_this_build_uses_its_row`) shows no
/// golden moved
const PREVIOUS_ORACLES: &[(&str, &str, &str)] = &[
    (
        // SDF_SEMANTICS_ID with alice-det-math 0.5.1 (7.0.0)
        "151079595c1e8d8fc661909bac7ea0dde0ba5a8b3ef500019d01c6af64defb69",
        "tests/test_det_math_clamp_generation.rs",
        "smoothstep_gives_the_same_bits_under_both_clamps",
    ),
    (
        // alice-det-math 0.5.1's identifier, exposed as alice-sdf's own by 6.0 / 6.1
        "85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315",
        "tests/test_det_math_clamp_generation.rs",
        "smoothstep_gives_the_same_bits_under_both_clamps",
    ),
    (
        // alice-det-math 0.5.0's
        "135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000",
        "tests/test_det_math_clamp_generation.rs",
        "smoothstep_gives_the_same_bits_under_both_clamps",
    ),
];

/// Every previous identifier has an oracle that exists as a test (the file holds `#[test]` and
/// that function), and every oracle names a previous identifier: an identifier without one, or an
/// oracle removed from its file, fails here
#[test]
fn every_previous_identifier_has_an_oracle_that_exists() {
    let previous: Vec<String> = PREVIOUS_SDF_SEMANTICS_IDS.iter().map(hex).collect();
    let listed: Vec<&str> = PREVIOUS_ORACLES.iter().map(|(id, _, _)| *id).collect();
    let mut a = previous;
    a.sort();
    let mut b: Vec<String> = listed.iter().map(|s| (*s).to_owned()).collect();
    b.sort();
    assert_eq!(
        a, b,
        "PREVIOUS_SDF_SEMANTICS_IDS and the identifiers with an oracle"
    );
    let mut compared = 0;
    for (id, file, test) in PREVIOUS_ORACLES {
        let src =
            std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(file))
                .unwrap_or_else(|e| panic!("{id}: {file}: {e}"));
        let at = src
            .find(&format!("fn {test}()"))
            .unwrap_or_else(|| panic!("{id}: {file} has no fn {test}"));
        assert!(
            src[..at].trim_end().ends_with("#[test]"),
            "{id}: {file}::{test} is not a #[test]"
        );
        compared += 1;
    }
    assert!(compared > 0, "compared nothing");
    assert!(!PREVIOUS_SDF_SEMANTICS_IDS.contains(&SDF_SEMANTICS_ID));
}
