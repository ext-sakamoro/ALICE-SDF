//! `alice_sdf::SDF_SEMANTICS_ID`: what it folds, and that it moves with each entry
//! (ALICE-Law G6)
//!
//! The expected values come from the sources the entries pin: the golden table is read
//! from `tests/test_det_golden.rs`, the rule entries are the SHA-256 of their text, and
//! the det-math entry is the resolved crate's identifier. The fold is written out here.

use alice_det_math::SemanticsCheck;
use alice_sdf::semantics::{MAX_MIN_RULE, MUL_ADD_RULE, SDF_SEMANTICS_PINS};
use alice_sdf::{
    check_sdf_semantics, PREVIOUS_SDF_SEMANTICS_IDS, SDF_MOVED_LAWS_OVERSTATE_FIX,
    SDF_SEMANTICS_BY_DET_MATH, SDF_SEMANTICS_ID, SDF_SEMANTICS_MOVED_LAWS,
};
use sha2::{Digest, Sha256};

const GOLDEN_SOURCE: &str = include_str!("test_det_golden.rs");
// with alice-det-math 0.5.2 at 2d01c63f and the distance laws that over-stated corrected; no earlier
// identifier is one this build reproduces whole (a stored tree may be, see SDF_SEMANTICS_MOVED_LAWS)
const RECORDED_HEX: &str = "c75bbbc61844839669d5d53d71c4774f9efac764e366b5e775978e3825e23ded";

/// The golden rows added after `150a80c5…`: a tree law and a 2D law that moved with the others and had
/// no row to show it
const ROWS_ADDED_AFTER_150A80C5: &[&str] = &["bezier", "sdf2d_bezier"];

/// The rows that moved after `150a80c5…`: (row, the node's tag in a stored tree, its hash in that build)
const ROWS_MOVED_AFTER_150A80C5: &[(&str, &str, &str)] = &[
    (
        "blobby_cross",
        "BlobbyCross",
        "55160ecb2dad3122acdb48ccc6797374002b8cbafbb524621de3501cf30d9077",
    ),
    (
        "helix",
        "Helix",
        "1ffa278889d2fbe0af505056a94e047f2d1c62fb6e063c349db967696bdf182f",
    ),
    (
        "parabola_segment",
        "ParabolaSegment",
        "d50caf335ea24d510b72e2d239299e00ccdae2992f3f4cecaa6aefc1be5ff55c",
    ),
    (
        "sweep_bezier",
        "SweepBezier",
        "dcc34097d512b7bcf759c3f983da09bf896f30cb5257e491dec04010dc0b7d98",
    ),
];

/// The golden rows added after `0f6c58d0…`: new nodes, no row of that build changed
const ROWS_ADDED_AFTER_0F6C58D0: &[&str] = &[
    "loft",
    "loft_curves",
    "loft_nofade",
    "loft_faded",
    "section_stack",
    "smooth_section_stack",
];

/// The golden rows added after `ad04549c…`, up to `0f6c58d0…`: new nodes, no row of that build changed
const ROWS_ADDED_AFTER_AD04549C: &[&str] = &["grid_warp", "pose"];

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
fn check_reads_this_identifier_and_refuses_the_earlier_ones() {
    assert_eq!(
        check_sdf_semantics(&SDF_SEMANTICS_ID),
        SemanticsCheck::Verified
    );
    assert!(PREVIOUS_SDF_SEMANTICS_IDS.is_empty());
    // the identifiers earlier releases stamped (150a80c5, 0f6c58d0 and ad04549c before the distance
    // laws moved, 6e98f0be with alice-det-math 0.5.2 at 39507d1b, 151079595 with 0.5.1 by 7.0.0, the
    // det-math identifiers 6.0 / 6.1 exposed) are refused
    let mut refused = 0;
    for earlier in [
        ID_150A80C5,
        ID_0F6C58D0,
        AD04549C,
        "6e98f0bed3f8691ca470ae2fe2c404c893856a3e3f6a9cda9d67528ae8074b37",
        "151079595c1e8d8fc661909bac7ea0dde0ba5a8b3ef500019d01c6af64defb69",
        "85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315",
        "135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000",
    ] {
        let id: [u8; 32] =
            core::array::from_fn(|i| u8::from_str_radix(&earlier[2 * i..2 * i + 2], 16).unwrap());
        assert_eq!(
            check_sdf_semantics(&id),
            SemanticsCheck::Mismatch,
            "{earlier}"
        );
        refused += 1;
    }
    assert_eq!(refused, 7);
    assert_eq!(check_sdf_semantics(&[0u8; 32]), SemanticsCheck::Mismatch);
}

/// The pins of `6e98f0be…` (alice-det-math 0.5.2 at 39507d1b), frozen as they were: the golden
/// table digest of that build, the two rule tags and its det-math entry
const PINS_6E98F0BE: &[(&str, &str)] = &[
    (
        "golden tests/test_det_golden.rs GOLDEN",
        "f13e0794beaee9943112db526bfe522eca30e77272647ebe20bea589d2484cf3",
    ),
    (
        "tag max / min rule",
        "f1afc99dfe0c89acaeb21672974f99532b90735aae6ade6c6f5e70cbf032a329",
    ),
    (
        "tag Real::mul_add rounds twice",
        "24aab83311c38b3d2de5aacaa4f3bb4e0ff5db6497ac3c4c9e74d993c29f1553",
    ),
    (
        "alice-det-math",
        "a2e46794a49b62730d8eb75d10e6dbfdd59d6562a985ae94c716fc637a57e3cf",
    ),
];

/// The golden windows added with alice-det-math 2d01c63f, as the 39507d1b build computes them
/// (recorded from that build with this file's `far_scenes`)
const WINDOWS_AT_39507D1B: &[(&str, &str)] = &[
    (
        "twist_plane_below_2_13",
        "7fd43fd5396e0bf562926b6705a47427baa60913aa30025f9b5d0dc5a9d832e0",
    ),
    (
        "twist_plane_from_2_13",
        "f100134c107aa2be3c924643c986cdca4799e8894c7674754491823577838350",
    ),
    (
        "infinite_cylinder_r0_normal",
        "a46ef0bdb3b688dfdf62392950f9a1b421e7f38bacd4f7c7db94bf8d9a9bb760",
    ),
    (
        "infinite_cylinder_r0_subnormal",
        "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1",
    ),
];

/// `6e98f0be…` is no longer an identifier of this build, and the reason is exactly this
/// generation's changes: every golden row of that build is unchanged (the rows of this table
/// without the added windows hash to its golden pin), and of the windows added with it the ones
/// below the change have the bits they have under 39507d1b while the `sin_cos` window from `2^13`
/// and the `hypot` window of two subnormals moved. A further change to any law moves this and
/// fails here.
#[test]
fn the_6e98f0be_identifier_is_rejected_because_exactly_these_windows_moved() {
    let unhex = |s: &str| -> [u8; 32] {
        core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
    };
    let frozen: Vec<(&str, [u8; 32])> = PINS_6E98F0BE.iter().map(|&(n, d)| (n, unhex(d))).collect();
    assert_eq!(
        hex(&fold(&frozen)),
        "6e98f0bed3f8691ca470ae2fe2c404c893856a3e3f6a9cda9d67528ae8074b37"
    );
    let rows = rows_at_150a80c5();
    let windows: Vec<&str> = WINDOWS_AT_39507D1B.iter().map(|(n, _)| *n).collect();
    let mut older: Vec<(String, String)> = rows
        .iter()
        .filter(|(n, _)| {
            !windows.contains(&n.as_str())
                && !ROWS_ADDED_AFTER_AD04549C.contains(&n.as_str())
                && !ROWS_ADDED_AFTER_0F6C58D0.contains(&n.as_str())
        })
        .cloned()
        .collect();
    assert_eq!(
        older.len()
            + windows.len()
            + ROWS_ADDED_AFTER_AD04549C.len()
            + ROWS_ADDED_AFTER_0F6C58D0.len(),
        rows.len(),
        "every window is a row"
    );
    older.sort();
    let canon = older.iter().fold(String::new(), |mut s, (n, h)| {
        use std::fmt::Write;
        writeln!(s, "{n}={h}").expect("writing to a String");
        s
    });
    assert_eq!(
        hex(&sha(canon.as_bytes())),
        PINS_6E98F0BE[0].1,
        "an older golden row moved"
    );
    let mut moved = Vec::new();
    for (name, then) in WINDOWS_AT_39507D1B {
        let now = &rows
            .iter()
            .find(|(n, _)| n == name)
            .expect("a window row")
            .1;
        if now != then {
            moved.push(*name);
        }
    }
    assert_eq!(
        moved,
        ["twist_plane_from_2_13", "infinite_cylinder_r0_subnormal"]
    );
}

/// The oracle that shows this build reproduces what an earlier identifier stamped:
/// `(identifier, test file, test function)`. Empty with `PREVIOUS_SDF_SEMANTICS_IDS`; an
/// identifier put back into the previous list needs its oracle here
const PREVIOUS_ORACLES: &[(&str, &str, &str)] = &[];

const ID_150A80C5: &str = "150a80c505111a736fe705e0fa4e58aa521de2136cf0abc2762d84fbbb614f79";
const ID_0F6C58D0: &str = "0f6c58d0244a39c92712977c7cdbd2282e5c13d3cbd435f57189f79d2f6b69a4";
const AD04549C: &str = "ad04549c331ec33377ef1d4419c0c6756cef6a852520e470a53f642257da4938";

/// The golden rows as they were at `150a80c5…`: without the added rows, the moved rows put back
fn rows_at_150a80c5() -> Vec<(String, String)> {
    let mut rows = golden_rows();
    let all = rows.len();
    rows.retain(|(n, _)| !ROWS_ADDED_AFTER_150A80C5.contains(&n.as_str()));
    assert_eq!(rows.len() + ROWS_ADDED_AFTER_150A80C5.len(), all);
    for (row, _, then) in ROWS_MOVED_AFTER_150A80C5 {
        let at = rows
            .iter()
            .position(|(n, _)| n == row)
            .unwrap_or_else(|| panic!("no row {row}"));
        assert_ne!(rows[at].1, *then, "{row} did not move");
        rows[at].1 = (*then).to_owned();
    }
    rows
}

/// The canonical text of golden rows, hashed as the golden entry is
fn golden_digest(rows: &[(String, String)]) -> [u8; 32] {
    let mut rows = rows.to_vec();
    rows.sort();
    let canon = rows.iter().fold(String::new(), |mut s, (n, h)| {
        use std::fmt::Write;
        writeln!(s, "{n}={h}").expect("writing to a String");
        s
    });
    sha(canon.as_bytes())
}

/// This build's pins with another golden entry
fn fold_with_golden(golden: [u8; 32]) -> [u8; 32] {
    let pins: Vec<(&str, [u8; 32])> = SDF_SEMANTICS_PINS
        .iter()
        .map(|&(n, d)| {
            if n.starts_with("golden ") {
                (n, golden)
            } else {
                (n, d)
            }
        })
        .collect();
    fold(&pins)
}

/// What the stored tree rule rests on: the three earlier identifiers are refused because exactly the
/// listed rows moved. The table with those rows put back folds to `150a80c5…`, without the lofts to
/// `0f6c58d0…`, and without the grid warp and pose rows as well to `ad04549c…`; so every other row has
/// the bits those builds had. A row that moves, or is added, without being listed fails here
#[test]
fn the_earlier_identifiers_moved_by_exactly_these_rows() {
    let mut rows = rows_at_150a80c5();
    let golden = golden_digest(&rows);
    assert_eq!(
        hex(&golden),
        "9014e45fd46635353576228b3e8cd1eedc54f3274895249b7718bfb88caf551c"
    );
    assert_eq!(hex(&fold_with_golden(golden)), ID_150A80C5);

    let all = rows.len();
    rows.retain(|(n, _)| !ROWS_ADDED_AFTER_0F6C58D0.contains(&n.as_str()));
    assert_eq!(rows.len() + ROWS_ADDED_AFTER_0F6C58D0.len(), all);
    let golden = golden_digest(&rows);
    assert_eq!(
        hex(&golden),
        "ecfb9c749cac3ec06f9fec60bf1735ee9aff187f0d2873825895e744682750a5"
    );
    assert_eq!(hex(&fold_with_golden(golden)), ID_0F6C58D0);

    let all = rows.len();
    rows.retain(|(n, _)| !ROWS_ADDED_AFTER_AD04549C.contains(&n.as_str()));
    assert_eq!(rows.len() + ROWS_ADDED_AFTER_AD04549C.len(), all);
    let golden = golden_digest(&rows);
    assert_eq!(
        hex(&golden),
        "84831086d072608aa773445b0a838e5aa530c32d936fc1d6bf62ff02325b709d"
    );
    assert_eq!(hex(&fold_with_golden(golden)), AD04549C);
}

/// The stored tree rule covers the three identifiers above, and its node list is the nodes of the moved
/// rows plus `Bezier` (which moved with no row of those builds to show it). `sdf2d_bezier` is no node of
/// a tree
#[test]
fn the_stored_tree_rule_lists_the_moved_nodes() {
    let mut ids: Vec<String> = SDF_SEMANTICS_MOVED_LAWS
        .iter()
        .map(|(id, _)| hex(id))
        .collect();
    ids.sort();
    let mut want = [ID_0F6C58D0, ID_150A80C5, AD04549C].map(str::to_owned);
    want.sort();
    assert_eq!(ids, want);
    let mut nodes: Vec<&str> = ROWS_MOVED_AFTER_150A80C5.iter().map(|r| r.1).collect();
    nodes.push("Bezier");
    nodes.sort_unstable();
    assert_eq!(nodes, SDF_MOVED_LAWS_OVERSTATE_FIX);
    for (_, laws) in SDF_SEMANTICS_MOVED_LAWS {
        assert_eq!(*laws, SDF_MOVED_LAWS_OVERSTATE_FIX);
    }
}

/// Every previous identifier has an oracle that exists as a test (the file holds `#[test]` and
/// that function), and every oracle names a previous identifier: an identifier without one, or an
/// oracle removed from its file, fails here
#[test]
fn every_previous_identifier_has_an_oracle_that_exists() {
    let previous: Vec<String> = PREVIOUS_SDF_SEMANTICS_IDS.iter().map(hex).collect();
    let mut a = previous;
    a.sort();
    let mut b: Vec<String> = PREVIOUS_ORACLES
        .iter()
        .map(|(id, _, _)| (*id).to_owned())
        .collect();
    b.sort();
    assert_eq!(
        a, b,
        "PREVIOUS_SDF_SEMANTICS_IDS and the identifiers with an oracle"
    );
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
    }
    assert!(!PREVIOUS_SDF_SEMANTICS_IDS.contains(&SDF_SEMANTICS_ID));
}
