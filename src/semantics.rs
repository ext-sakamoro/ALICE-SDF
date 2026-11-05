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
//! # Hex form (with `alice-det-math` 0.5.2 at 2d01c63f; other generations in [`SDF_SEMANTICS_BY_DET_MATH`])
//!
//! ```text
//! 0f6c58d0244a39c92712977c7cdbd2282e5c13d3cbd435f57189f79d2f6b69a4
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
        hex32("ecfb9c749cac3ec06f9fec60bf1735ee9aff187f0d2873825895e744682750a5"),
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
    // alice-det-math 0.5.2 at 2d01c63f, with the grid warp and the pose in the golden table (sin /
    // cos at |x| >= 2^13 through sin64 / cos64, powf of a
    // negative base, powf64 as fdlibm e_pow, hypot of two subnormals; erf added). The rows of the
    // earlier generations are not folds of these pins any more (the golden table gained the
    // windows that show what moved); their identifiers are in the frozen table of
    // `tests/sdf_semantics_id.rs`
    (
        hex32("a2a11b9b826227fa7a9ed8ab2c1ce63dbb07ad4c39aa2c46a363ddea6b47dbe9"),
        hex32("0f6c58d0244a39c92712977c7cdbd2282e5c13d3cbd435f57189f79d2f6b69a4"),
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

/// Earlier identifiers whose distances this build still computes to the same bits, newest first
///
/// `ad04549c…` (alice-det-math 0.5.2 at 2d01c63f, before the grid warp and the pose): those two
/// nodes only added golden rows, and every row of that build has the same bits here
/// (`tests/sdf_semantics_id.rs` folds the table without the added rows back to it). Older ones
/// stay refused: alice-det-math at 2d01c63f reduces `sin` / `cos` / `sin_cos` arguments from
/// `2^13` on in double precision and evaluates `hypot` of two subnormals in `f64`, and this crate
/// calls them on the caller's coordinates (twist, bend, noise, the cylinder and cone families), so
/// a distance at such a point has other bits than under any earlier generation. `6e98f0be…` (0.5.2
/// at 39507d1b), `151079595…` (0.5.1, 7.0.0) and the det-math identifiers 6.0 / 6.1 exposed
/// (`85e08846…`, `135c407d…`) therefore read as a mismatch; `tests/sdf_semantics_id.rs` shows
/// which golden windows moved.
pub const PREVIOUS_SDF_SEMANTICS_IDS: &[[u8; 32]] = &[hex32(
    "ad04549c331ec33377ef1d4419c0c6756cef6a852520e470a53f642257da4938",
)];

/// The tree nodes (their tags in the serialized tree) whose distance moved after the identifiers in
/// [`SDF_SEMANTICS_MOVED_LAWS`]: each returned a distance larger than the true one, and a tracer could
/// step through the surface. A frozen list: a later move of a law gets a new entry, not an edit.
pub const SDF_MOVED_LAWS_OVERSTATE_FIX: &[&str] = &[
    "Bezier",
    "BlobbyCross",
    "Helix",
    "ParabolaSegment",
    "SweepBezier",
];

/// Earlier identifiers a stored tree may still be read under, with the nodes whose distance moved
///
/// Every other golden row of those builds has the same bits here, so a tree that uses none of the listed
/// nodes computes what it computed when it was written, and its identifier says so for that tree. A tree
/// that uses one is refused (`IoError::MovedLaw`): the stored shape was tuned against a distance that no
/// longer exists. A form without a tree (a mesh, a shader text, a state snapshot) has no nodes to look at
/// and stays refused: those identifiers are not in [`PREVIOUS_SDF_SEMANTICS_IDS`].
pub const SDF_SEMANTICS_MOVED_LAWS: &[([u8; 32], &[&str])] = &[
    (
        hex32("0f6c58d0244a39c92712977c7cdbd2282e5c13d3cbd435f57189f79d2f6b69a4"),
        SDF_MOVED_LAWS_OVERSTATE_FIX,
    ),
    (
        hex32("ad04549c331ec33377ef1d4419c0c6756cef6a852520e470a53f642257da4938"),
        SDF_MOVED_LAWS_OVERSTATE_FIX,
    ),
];

/// The entries of `laws` the tree uses, found by the node tags of its serialized form (so a node nested in
/// any transform, modifier or container counts, and no list of variants has to follow new nodes)
pub fn moved_laws_in(tree: &crate::types::SdfTree, laws: &[&'static str]) -> Vec<&'static str> {
    fn walk(value: &serde_json::Value, laws: &[&'static str], found: &mut Vec<&'static str>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, inner) in map {
                    if let Some(law) = laws.iter().find(|l| **l == key) {
                        if !found.contains(law) {
                            found.push(law);
                        }
                    }
                    walk(inner, laws, found);
                }
            }
            serde_json::Value::Array(items) => {
                for inner in items {
                    walk(inner, laws, found);
                }
            }
            _ => {}
        }
    }
    let mut found = Vec::new();
    match serde_json::to_value(&tree.root) {
        Ok(value) => walk(&value, laws, &mut found),
        // a tree that cannot be looked at is not known to be free of them
        Err(_) => found.extend_from_slice(laws),
    }
    found
}

/// Whether a stored identifier names what this build computes
///
/// [`SDF_SEMANTICS_ID`] is `Verified`, one of [`PREVIOUS_SDF_SEMANTICS_IDS`] is
/// `VerifiedPrevious`, anything else `Mismatch`.
pub fn check_sdf_semantics(stored: &[u8; 32]) -> alice_det_math::SemanticsCheck {
    check_sdf_semantics_against(stored, PREVIOUS_SDF_SEMANTICS_IDS)
}

/// [`check_sdf_semantics`] against the earlier identifiers `previous` instead of
/// [`PREVIOUS_SDF_SEMANTICS_IDS`]: the one copy of the rule, which the crate's loaders map, so the
/// `VerifiedPrevious` branch stays tested while the list is empty
pub(crate) fn check_sdf_semantics_against(
    stored: &[u8; 32],
    previous: &[[u8; 32]],
) -> alice_det_math::SemanticsCheck {
    if *stored == SDF_SEMANTICS_ID {
        alice_det_math::SemanticsCheck::Verified
    } else if previous.contains(stored) {
        alice_det_math::SemanticsCheck::VerifiedPrevious(*stored)
    } else {
        alice_det_math::SemanticsCheck::Mismatch
    }
}
