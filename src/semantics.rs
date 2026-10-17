//! Content hash of what this crate computes: [`SDF_SEMANTICS_ID`].
//!
//! `alice-det-math`'s identifier ([`crate::SEMANTICS_ID`], re-exported) names the
//! arithmetic of the kernels this crate evaluates with; [`SDF_SEMANTICS_ID`] names what
//! alice-sdf itself computes with them. It is the fold of [`SDF_SEMANTICS_PINS`]: the
//! tree-evaluator goldens of `tests/test_det_golden.rs` (one digest over the whole table),
//! the two rules every evaluator shares (max / min and the two-rounding `mul_add`), and
//! `alice_det_math::SEMANTICS_ID`. Two builds with the same identifier compute every
//! pinned distance to the same bits; a build that differs in any of them has a different
//! value. It does not depend on the enabled features.
//!
//! The identifier moves only when one of those moves (a golden re-recorded, a rule
//! changed, a det-math generation), not with the crate's version.
//!
//! # Hex form (with `alice-det-math` 0.5.1; other generations in [`SDF_SEMANTICS_BY_DET_MATH`])
//!
//! ```text
//! 151079595c1e8d8fc661909bac7ea0dde0ba5a8b3ef500019d01c6af64defb69
//! ```
//!
//! # Fold
//!
//! The procedure of `alice-det-math`'s `SEMANTICS_ID` fold: the entries sorted by name
//! (ascending byte order), then SHA-256 over, for each entry in that order, the name length
//! as 4 bytes big-endian (`u32`), the name bytes, and the 32 digest bytes. The golden entry
//! is the SHA-256 of the lines `name=hash\n` of every `GOLDEN` row, sorted by name.
//!
//! # Checked by
//!
//! `tests/sdf_semantics_id.rs` reads the golden table from `tests/test_det_golden.rs` (the
//! source, not a copy), checks the rule entries against their text and the det-math entry
//! against `alice_det_math::SEMANTICS_ID`, and recomputes every row of
//! [`SDF_SEMANTICS_BY_DET_MATH`].

/// Decodes 64 hex digits into 32 bytes at compile time; anything else is a compile error.
const fn hex32(s: &str) -> [u8; 32] {
    const fn nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("not a lower-case hex digit"),
        }
    }
    let b = s.as_bytes();
    assert!(b.len() == 64, "a digest is 64 hex digits");
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = nibble(b[2 * i]) * 16 + nibble(b[2 * i + 1]);
        i += 1;
    }
    out
}

/// The text the max / min rule entry hashes
pub const MAX_MIN_RULE: &str = "alice-sdf: max / min are IEEE 754 maximumNumber / minimumNumber with -0 < +0, NaN gives the other operand and two NaNs the canonical quiet NaN 0x7fc00000, in every evaluator and the JIT";

/// The text the `mul_add` rule entry hashes
pub const MUL_ADD_RULE: &str = "alice-sdf: Real::mul_add is a * b + c with two roundings, the same as the JIT and the emitted shaders";

/// The `(name, digest)` entries [`SDF_SEMANTICS_ID`] folds
///
/// The golden table digest, the SHA-256 of each rule text, and the resolved `alice_det_math::SEMANTICS_ID`.
pub const SDF_SEMANTICS_PINS: &[(&str, [u8; 32])] = &[
    (
        "golden tests/test_det_golden.rs GOLDEN",
        hex32("f13e0794beaee9943112db526bfe522eca30e77272647ebe20bea589d2484cf3"),
    ),
    (
        "tag max / min rule",
        hex32("f1afc99dfe0c89acaeb21672974f99532b90735aae6ade6c6f5e70cbf032a329"),
    ),
    (
        "tag Real::mul_add rounds twice",
        hex32("24aab83311c38b3d2de5aacaa4f3bb4e0ff5db6497ac3c4c9e74d993c29f1553"),
    ),
    // the resolved alice-det-math's own identifier, not a copy of it: the identifier is
    // looked up by it in SDF_SEMANTICS_BY_DET_MATH
    ("alice-det-math", alice_det_math::SEMANTICS_ID),
];

/// [`SDF_SEMANTICS_ID`] for each `alice-det-math` generation this crate knows
///
/// `(alice_det_math::SEMANTICS_ID, fold of SDF_SEMANTICS_PINS with that det-math entry)`.
/// A build against a generation with no row does not compile ([`sdf_semantics_id_for`]
/// panics in a const context), rather than reporting an identifier of another generation.
pub const SDF_SEMANTICS_BY_DET_MATH: &[([u8; 32], [u8; 32])] = &[
    // alice-det-math 0.5.0
    (
        hex32("135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000"),
        hex32("d44898f9c186c9811b8ca760d0f53da66a25580012218047dd897c8e013cc3a2"),
    ),
    // alice-det-math 0.5.1
    (
        hex32("85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"),
        hex32("151079595c1e8d8fc661909bac7ea0dde0ba5a8b3ef500019d01c6af64defb69"),
    ),
];

/// The identifier of a build that resolved the `alice-det-math` with identifier `det_math`
///
/// # Panics
///
/// When [`SDF_SEMANTICS_BY_DET_MATH`] has no row for `det_math` (in a const context this
/// stops the compile). A new `alice-det-math` generation needs its row, recomputed from the
/// pins.
#[must_use]
pub const fn sdf_semantics_id_for(det_math: &[u8; 32]) -> [u8; 32] {
    let mut i = 0;
    while i < SDF_SEMANTICS_BY_DET_MATH.len() {
        if same32(&SDF_SEMANTICS_BY_DET_MATH[i].0, det_math) {
            return SDF_SEMANTICS_BY_DET_MATH[i].1;
        }
        i += 1;
    }
    panic!("alice-det-math SEMANTICS_ID has no row in SDF_SEMANTICS_BY_DET_MATH: add its fold");
}

const fn same32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut i = 0;
    while i < 32 {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// The identifier of what alice-sdf computes, in this build
pub const SDF_SEMANTICS_ID: [u8; 32] = sdf_semantics_id_for(&alice_det_math::SEMANTICS_ID);

/// Identifiers this crate published as its own before [`SDF_SEMANTICS_ID`] existed
///
/// 6.1.0 and earlier exposed `alice-det-math`'s identifier as their own; this build still
/// computes their distances to the same bits (the goldens did not move between them).
pub const PREVIOUS_SDF_SEMANTICS_IDS: &[[u8; 32]] = &[
    // alice-det-math 0.5.1's identifier, exposed as alice_sdf::SEMANTICS_ID by 6.0 / 6.1
    hex32("85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"),
    // alice-det-math 0.5.0's
    hex32("135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000"),
];

/// Whether a stored identifier names what this build computes
///
/// [`SDF_SEMANTICS_ID`] is `Verified`, one of [`PREVIOUS_SDF_SEMANTICS_IDS`] is
/// `VerifiedPrevious`, anything else `Mismatch`.
pub fn check_sdf_semantics(stored: &[u8; 32]) -> alice_det_math::SemanticsCheck {
    if *stored == SDF_SEMANTICS_ID {
        alice_det_math::SemanticsCheck::Verified
    } else if PREVIOUS_SDF_SEMANTICS_IDS.contains(stored) {
        alice_det_math::SemanticsCheck::VerifiedPrevious(*stored)
    } else {
        alice_det_math::SemanticsCheck::Mismatch
    }
}
