//! Content hash of what this crate computes: [`SDF_SEMANTICS_ID`].
//!
//! `alice-det-math`'s identifier ([`crate::SEMANTICS_ID`], re-exported) names the
//! arithmetic of the kernels this crate evaluates with; [`SDF_SEMANTICS_ID`] names what
//! alice-sdf itself computes with them. It is the fold of [`SDF_SEMANTICS_PINS`]: the
//! tree-evaluator goldens of `tests/test_det_golden.rs` (one digest over the whole table),
//! the two rules every evaluator shares (max / min and the two-rounding `mul_add`), and
//! the identifier of each `alice-det-math` function the crate calls ([`DET_MATH_PINS`], named
//! `det-math <function>`; a function of `alice-det-math` the crate does not call moves
//! nothing). Two builds with the same identifier compute every
//! pinned distance to the same bits; a build that differs in any of them has a different
//! value. It does not depend on the enabled features.
//!
//! The identifier moves only when one of those moves (a golden re-recorded, a rule
//! changed, a called det-math function), not with the crate's version.
//!
//! # Hex form (earlier `alice-det-math` generations in [`SDF_SEMANTICS_BY_DET_MATH`])
//!
//! ```text
//! db7ff4a4b7daf3f242f353fc7af2c448832e8625c19f2522561d3b83a496f775
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
//! source, not a copy), checks the rule entries against their text and each det-math entry
//! against `alice_det_math::function_pin` (the `const _` below also stops the compile on a
//! mismatch), and checks that [`SDF_SEMANTICS_ID`] is the fold.

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
/// The golden table digest and the SHA-256 of each rule text; [`SDF_SEMANTICS_ID`] folds them with
/// [`DET_MATH_PINS`].
pub const SDF_SEMANTICS_PINS: &[(&str, [u8; 32])] = &[
    (
        "golden tests/test_det_golden.rs GOLDEN",
        hex32("08621647ba1fa80888533c53e4b012c40a52180962e9d2d98cf08801c1106344"),
    ),
    (
        "tag max / min rule",
        hex32("f1afc99dfe0c89acaeb21672974f99532b90735aae6ade6c6f5e70cbf032a329"),
    ),
    (
        "tag Real::mul_add rounds twice",
        hex32("24aab83311c38b3d2de5aacaa4f3bb4e0ff5db6497ac3c4c9e74d993c29f1553"),
    ),
];

/// The `alice-det-math` functions this crate calls
///
/// `ALICE-Law`'s `scripts/fold_set_check.py` compares the list with the calls in `src/`.
pub const DET_MATH_FUNCTIONS: &[&str] = &[
    "acos",
    "asin",
    "atan2",
    "atan2_64",
    "cbrt",
    "cos",
    "cos64",
    "exp",
    "hypot",
    "lerp",
    "ln",
    "log10_64",
    "log2",
    "max",
    "max64",
    "metric_axis_extent",
    "metric_euclidean_radius",
    "metric_lipschitz",
    "metric_minimum",
    "metric_new",
    "metric_norm",
    "metric_normalised",
    "metric_weights",
    "min",
    "min64",
    "norm_l1",
    "norm_l2",
    "norm_linf",
    "powf",
    "powi",
    "round",
    "sin",
    "sin64",
    "sin_cos",
    "smoothstep",
    "sqrt64",
    "tan",
];

/// The identifier of each function of [`DET_MATH_FUNCTIONS`], in that order
///
/// As recorded when [`SDF_SEMANTICS_ID`] was folded. The `const _` below compares each with
/// `alice_det_math::function_pin` at compile time, so a resolved `alice-det-math` whose
/// function computes other bits stops the compile instead of reporting this identifier.
pub const DET_MATH_PINS: &[(&str, [u8; 32])] = &[
    (
        "acos",
        hex32("98f2aab18db196ed4a9de02005318ccf98f929c52895ada6218e99219114ccbf"),
    ),
    (
        "asin",
        hex32("64dd727592b08cedcdcb58584e400921e35a954d00f7b6c7884900528a8ab081"),
    ),
    (
        "atan2",
        hex32("f1227770b42015176d620dba415653be1a14b9708624f4f70594011c54fef376"),
    ),
    (
        "atan2_64",
        hex32("ccdc96ba060eea7bb72fb909e5b64608812abb708f77cc59b820d8db8bd2e416"),
    ),
    (
        "cbrt",
        hex32("511090d6109210070d5d4d7bdd4628c90c6aec61d30c6df616ccafcc244df6ae"),
    ),
    (
        "cos",
        hex32("f7627941400acabe93c8dca4090eee12f8910fe4c14a8735ad456cef3c48ed7b"),
    ),
    (
        "cos64",
        hex32("d9cf0bbeda9dbf12a0c98e913ce75b6f5a404c8ff6bac44aee88269f8f05bb75"),
    ),
    (
        "exp",
        hex32("606918f2d637f46f9a792a29b3ee31a991ef3196064456caa7b3a7ca81700a63"),
    ),
    (
        "hypot",
        hex32("c97aba4deb475308ea0c63c46ad778172c0cbd8fe0844fb23546d5df854f99b6"),
    ),
    (
        "lerp",
        hex32("23496377b28f21efcb22e52f3b5f30df4da73c9f18817c1b6c42aa0cfbc601f6"),
    ),
    (
        "ln",
        hex32("6fe18bc0fba8b7e3e871c4cbe555ddb40a460be7b3c7690303ee30accdcb7291"),
    ),
    (
        "log10_64",
        hex32("a287811831194c21c4d97c1db3b12392c5e439cb47f2c8d7f986fa5885589ea1"),
    ),
    (
        "log2",
        hex32("ce762397a3dcf55dd2ef35dfe1984aa7a173d9a2c944895914b0697b19864056"),
    ),
    (
        "max",
        hex32("8bd32fcb313f7fb3d7a699fc3075273de7e5a4e083912d8dc7f4cc4016fc2286"),
    ),
    (
        "max64",
        hex32("aac586eacc9acd69cf5988ffe736283c71fad8146027e86034d3e9b881db28b6"),
    ),
    (
        "metric_axis_extent",
        hex32("736ce0da2df008aa08538ba08c62d11e6dc2f4e49af50c5d3f58ad4afc8f40cc"),
    ),
    (
        "metric_euclidean_radius",
        hex32("273786c2ce33f332f5cbb823668b376ac8977b7a6c32eedd137852af9ecd29ee"),
    ),
    (
        "metric_lipschitz",
        hex32("03b351c6af738ede2017c836668fb9bfdbb4d6d170f669bce50417e4d09a6c70"),
    ),
    (
        "metric_minimum",
        hex32("31854083c9dfa4f16c986d0fbbe4efa3d2abd2e777d35de2f35d0f32af2b2681"),
    ),
    (
        "metric_new",
        hex32("d63b376cba0b866696b9fb644370352db768b7eb0d07ada6659f4286f4c20e54"),
    ),
    (
        "metric_norm",
        hex32("03c785f764f0f0d0c6988a0138203e60b1a694f51d4c5a8336830c143922bcea"),
    ),
    (
        "metric_normalised",
        hex32("6a4d5c3373df97f483e666469f334fcc5cebba3003990f1b8486e31468de864c"),
    ),
    (
        "metric_weights",
        hex32("bab65ee2072ccba53d53b09ceaa2bf405a33f2f8266e3c37290f3b1808c6f2ec"),
    ),
    (
        "min",
        hex32("2939e83813b26c868ec3113fb1c25102f16f28d8f24c343a727d4b1bec5904fe"),
    ),
    (
        "min64",
        hex32("1bef418073152f5e1e76f41718aa29b2517de8966a8c8ea8c1c2b5ba92fab3cb"),
    ),
    (
        "norm_l1",
        hex32("320797cd031904eae33421e96b0ea8740d8b718c236a9e4c9476654d47770ac8"),
    ),
    (
        "norm_l2",
        hex32("7fac5510ba5b5b7f62f3725be174031f53e1581f3d0057e720c2bc77808f32a5"),
    ),
    (
        "norm_linf",
        hex32("c65f5da66f91af2228b075ff2eba5fdefc1d512a1f235d26c080fc3f4df9cf69"),
    ),
    (
        "powf",
        hex32("0f1f85adcfa801b4afbbf858cf4c30f4ed0abdd3b7163d71aa9aa2b6031d9117"),
    ),
    (
        "powi",
        hex32("d3d76ba59208f7b0151ccde9b63043f538bc6b7737d1f16ff02d918f5806be45"),
    ),
    (
        "round",
        hex32("664695a5ceabc94e9fd1f6fb956055e1e3c1e2eab35c6af53ab97881e495bb0a"),
    ),
    (
        "sin",
        hex32("ee1b80d12df18ec0de5bce5a6bf05432484dabfc1cee1bb9b3f3291dd862cfec"),
    ),
    (
        "sin64",
        hex32("9d31aa70c38253a2ca1d1c7849f0a650c7ac1a15f8a594d10ad2c9b69c62b2ea"),
    ),
    (
        "sin_cos",
        hex32("025acd2cdc900eff683e39ef1e71de115dafe8da8a1a4b02847b41d75cbd130f"),
    ),
    (
        "smoothstep",
        hex32("91f832bd9ec528b403a4a3651f3b9d71ecb3218a70016910f9e5d83b0047247a"),
    ),
    (
        "sqrt64",
        hex32("051b24b99798891af42895e889ecb5add515496fc771ba5637a830f8f645988d"),
    ),
    (
        "tan",
        hex32("6b715b71c6fee0f708eee391597b2626737555f32610d75539b23fe5eb35e22b"),
    ),
];

const _: () = alice_det_math::assert_function_pins(DET_MATH_PINS);

/// Whether `pins` names exactly `functions`, in that order
const fn pins_name_the_functions(pins: &[(&str, [u8; 32])], functions: &[&str]) -> bool {
    if pins.len() != functions.len() {
        return false;
    }
    let mut i = 0;
    while i < pins.len() {
        let (a, b) = (pins[i].0.as_bytes(), functions[i].as_bytes());
        if a.len() != b.len() {
            return false;
        }
        let mut j = 0;
        while j < a.len() {
            if a[j] != b[j] {
                return false;
            }
            j += 1;
        }
        i += 1;
    }
    true
}

// a pin for a function the crate does not call, or a called function with no pin, stops the
// compile: the fold takes exactly the functions DET_MATH_FUNCTIONS lists
const _: () = assert!(
    pins_name_the_functions(DET_MATH_PINS, DET_MATH_FUNCTIONS),
    "DET_MATH_PINS does not name DET_MATH_FUNCTIONS in order"
);

/// The identifiers of the generations that folded `alice-det-math` whole
///
/// `(alice_det_math::SEMANTICS_ID, fold of SDF_SEMANTICS_PINS with an `alice-det-math` entry
/// holding it)`. This build folds the functions it calls instead ([`DET_MATH_PINS`]); the table
/// stays as the record of those identifiers.
///
/// It records what earlier builds stamped and is never recomputed when a later change moves a
/// golden row: `tests/sdf_semantics_id.rs` folds each row from today's table with the rows added
/// since removed and the moved rows back at their old hashes (`golden_before`).
pub const SDF_SEMANTICS_BY_DET_MATH: &[([u8; 32], [u8; 32])] = &[
    // alice-det-math 0.5.2 at 2d01c63f, with the grid warp and the pose in the golden table and
    // without the rows added after it (the lofts and the section stacks) (sin /
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

/// The identifier of what alice-sdf computes, in this build
///
/// Hex: `db7ff4a4b7daf3f242f353fc7af2c448832e8625c19f2522561d3b83a496f775`, the fold of
/// [`SDF_SEMANTICS_PINS`] and [`DET_MATH_PINS`]. The same value in every build, whatever cargo
/// features are enabled.
pub const SDF_SEMANTICS_ID: [u8; 32] =
    hex32("db7ff4a4b7daf3f242f353fc7af2c448832e8625c19f2522561d3b83a496f775");

/// Earlier identifiers whose distances this build still computes to the same bits, newest first
///
/// `dca9b092…` (after the heightmap face blend, before the bell pull): the bell pull only added a
/// golden row, every row of that build has the same bits here (`tests/sdf_semantics_id.rs`
/// `the_dca9b092_rows_are_unchanged`). Earlier ones read as a mismatch: the heightmap displacement
/// blends the faces within 1/16 of the dominant component (continuous at the face ties), so its
/// distances near a tie have other bits than under `d2d2f0df…` (after the smooth blends),
/// `72277bd6…`, `0f6ffc20…`, `0f6c58d0…` and `ad04549c…`; `tests/sdf_semantics_id.rs` folds each
/// from this build's table with the added rows removed and the heightmap row back at its old hash,
/// so the heightmap row is the only row that moved. Older ones were already refused:
/// alice-det-math at 2d01c63f reduces `sin` / `cos` / `sin_cos` arguments from `2^13` on in
/// double precision and evaluates `hypot` of two subnormals in `f64`, and this crate calls them on
/// the caller's coordinates (twist, bend, noise, the cylinder and cone families), so a distance at
/// such a point has other bits than under any earlier generation. `6e98f0be…` (0.5.2 at
/// 39507d1b), `151079595…` (0.5.1, 7.0.0) and the det-math identifiers 6.0 / 6.1 exposed
/// (`85e08846…`, `135c407d…`) read as a mismatch; `tests/sdf_semantics_id.rs` shows which golden
/// windows moved.
pub const PREVIOUS_SDF_SEMANTICS_IDS: &[[u8; 32]] = &[hex32(
    "dca9b09275b952e2e45ab7cd9426cd99f86e3cc65a16ffc2ffd76b188b01780f",
)];

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
