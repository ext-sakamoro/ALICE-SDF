//! `SdfNode::RepeatCells` (batch 1: mirrored repetition) against a brute-force
//! reference built independently of the node's law.
//!
//! Law: on each axis with spacing `s > 0` the cell is `c = round_half_up(p/s)`
//! (infinite) or `c = clamp(round_half_up(p/s + (n−1)/2), 0, n−1)` for `n`
//! cells centred on the origin; the child is evaluated at `p − centre(c)`, and
//! with `mirror` that axis is negated in odd cells, so neighbouring copies are
//! reflections of each other across the cell boundary. An axis with spacing 0
//! is not repeated.
//!
//! Oracles:
//! * the reference is the minimum of the child's exact distance over every
//!   copy (centre and flip computed here, not by the node);
//! * a child inside its cell: the node equals the reference (mirrored
//!   repetition is exact: for `p` and a child on the same side of a mirror
//!   plane, a reflected copy is never closer than the unreflected one);
//! * a child wider than its cell: the node over-estimates (documented as
//!   approximate); the test measures the error and checks the direction;
//! * `mirror = false` equals `RepeatInfinite` to the bit (one cell law);
//! * the field is continuous across cell boundaries and 1-Lipschitz for a
//!   1-Lipschitz child (sampled difference quotients);
//! * tree and compiled bytecode agree to the bit;
//! * with an odd count the centre cell (the one holding the origin) is not
//!   flipped, as the origin cell of the infinite repetition is not;
//! * `neighbours` / `variation` (batch 2) cannot be set: their types hold only
//!   the off state, which is stored as a plain `false` / `None`, and every
//!   reader (JSON, the `.asdf` body, the I-packet) refuses a stored one with
//!   the field's name, wherever the node sits (under a union, under a metric
//!   blend); an invalid spacing or count is refused at construction, in the
//!   readers and by the compiler.
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the reference is built with the platform libm"
)]
#![allow(clippy::float_cmp)]

use alice_sdf::modifiers::{CellNeighbours, RepeatCellsError, MAX_CELL_COUNT};
use alice_sdf::prelude::*;

/// A child off-centre in its cell (so that mirroring changes the shape) and
/// inside the cell of spacing (2, 1.5) on x / y.
fn contained_child() -> SdfNode {
    SdfNode::box3d_half_extents(0.3, 0.2, 0.25).translate(0.45, 0.2, 0.0)
}

/// A child that reaches past its cell's boundary on x (x in [0.75, 1.35] with
/// the boundary at 1): for a point at x = 0.5 the own copy is 0.25 away but the
/// mirrored neighbour's copy (x in [0.65, 1.25]) is 0.15 away.
fn wide_child() -> SdfNode {
    SdfNode::box3d_half_extents(0.3, 0.2, 0.25).translate(1.05, 0.0, 0.0)
}

const SPACING: Vec3 = Vec3::new(2.0, 1.5, 0.0);

/// (mirror parity index, centre in units of spacing) for every copy of one
/// axis. The parity index counts cells from the one holding the origin
/// (`c − ⌊(n − 1)/2⌋`, or `c` when infinite), so that cell is never flipped.
fn copies(n: Option<u32>) -> Vec<(i32, f32)> {
    match n {
        None => (-8..=8).map(|c| (c, c as f32)).collect(),
        Some(n) => (0..n as i32)
            .map(|c| (c - (n as i32 - 1) / 2, c as f32 - (n as f32 - 1.0) * 0.5))
            .collect(),
    }
}

/// The minimum over all copies of the child's distance.
fn brute_force(child: &SdfNode, p: Vec3, count: Option<[u32; 3]>, mirror: bool) -> f32 {
    let axis = |i: usize| -> Vec<(i32, f32)> {
        if SPACING[i] > 0.0 {
            copies(count.map(|c| c[i]))
        } else {
            vec![(0, 0.0)]
        }
    };
    let mut best = f32::INFINITY;
    for &(cx, ox) in &axis(0) {
        for &(cy, oy) in &axis(1) {
            for &(cz, oz) in &axis(2) {
                let mut q = p - Vec3::new(ox * SPACING.x, oy * SPACING.y, oz * SPACING.z);
                if mirror {
                    if cx.rem_euclid(2) == 1 {
                        q.x = -q.x;
                    }
                    if cy.rem_euclid(2) == 1 {
                        q.y = -q.y;
                    }
                    if cz.rem_euclid(2) == 1 {
                        q.z = -q.z;
                    }
                }
                best = best.min(eval(child, q));
            }
        }
    }
    best
}

fn points() -> Vec<Vec3> {
    let mut s = 0x2545_f491u32;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s as f32 / u32::MAX as f32
    };
    (0..3000)
        .map(|_| Vec3::new(r() * 12.0 - 6.0, r() * 9.0 - 4.5, r() * 2.0 - 1.0))
        .collect()
}

#[test]
fn mirrored_repetition_of_a_contained_child_is_exact() {
    for count in [None, Some([3, 4, 1]), Some([1, 2, 1])] {
        let child = contained_child();
        let node = child.clone().repeat_cells(SPACING, count, true).unwrap();
        let mut worst = 0.0f32;
        let pts = points();
        for &p in &pts {
            let got = eval(&node, p);
            let want = brute_force(&child, p, count, true);
            worst = worst.max((got - want).abs());
        }
        assert!(pts.len() >= 3000);
        assert!(
            worst <= 2e-6,
            "count {count:?}: worst |node − brute force| {worst}"
        );
    }
}

/// A child wider than its cell: the single-cell value is an over-estimate
/// (never below the true distance), and it is wrong by a measurable amount.
#[test]
fn a_child_wider_than_its_cell_is_approximate_and_over_estimates() {
    let child = wide_child();
    let node = child.clone().repeat_cells(SPACING, None, true).unwrap();
    let (mut worst, mut below) = (0.0f32, 0usize);
    for &p in &points() {
        let got = eval(&node, p);
        let want = brute_force(&child, p, None, true);
        worst = worst.max(got - want);
        if got < want - 2e-6 {
            below += 1;
        }
    }
    assert_eq!(
        below, 0,
        "the single-cell value went below the true distance"
    );
    assert!(
        worst > 0.05,
        "expected a measurable over-estimate, got {worst}"
    );
}

#[test]
fn without_mirror_it_is_repeat_infinite_to_the_bit() {
    let child = contained_child();
    let spacing = Vec3::new(2.0, 1.5, 3.0);
    let a = child.clone().repeat_cells(spacing, None, false).unwrap();
    let b = child.repeat_infinite(spacing.x, spacing.y, spacing.z);
    for &p in &points() {
        assert_eq!(eval(&a, p).to_bits(), eval(&b, p).to_bits(), "{p:?}");
    }
}

#[test]
fn mirrored_field_is_one_lipschitz_across_cell_boundaries() {
    for child in [contained_child(), wide_child()] {
        let node = child.repeat_cells(SPACING, None, true).unwrap();
        let mut pairs = 0;
        let mut worst = 0.0f32;
        for &p in &points() {
            for d in [
                Vec3::X * 0.013,
                Vec3::Y * 0.011,
                Vec3::new(0.007, -0.009, 0.004),
            ] {
                let q = p + d;
                worst = worst.max((eval(&node, p) - eval(&node, q)).abs() / d.length());
                pairs += 1;
            }
        }
        assert!(pairs > 8000);
        assert!(worst <= 1.0 + 1e-3, "difference quotient {worst}");
    }
}

#[test]
fn tree_and_compiled_agree() {
    for (count, mirror) in [
        (None, true),
        (Some([3, 4, 1]), true),
        (Some([2, 2, 1]), false),
    ] {
        let node = contained_child()
            .repeat_cells(SPACING, count, mirror)
            .unwrap();
        let compiled = CompiledSdf::compile(&node);
        for &p in &points() {
            assert_eq!(
                eval(&node, p).to_bits(),
                eval_compiled(&compiled, p).to_bits(),
                "{count:?} {mirror} {p:?}"
            );
        }
    }
}

#[test]
fn invalid_parameters_are_refused() {
    let c = contained_child;
    assert!(matches!(
        c().repeat_cells(Vec3::new(-1.0, 1.0, 0.0), None, true),
        Err(RepeatCellsError::Spacing(_))
    ));
    assert!(matches!(
        c().repeat_cells(Vec3::new(f32::NAN, 1.0, 0.0), None, true),
        Err(RepeatCellsError::Spacing(_))
    ));
    assert!(matches!(
        c().repeat_cells(Vec3::ZERO, None, true),
        Err(RepeatCellsError::Spacing(_))
    ));
    assert!(matches!(
        c().repeat_cells(SPACING, Some([0, 2, 1]), true),
        Err(RepeatCellsError::Count(_))
    ));
}

fn supported() -> SdfNode {
    contained_child().repeat_cells(SPACING, None, true).unwrap()
}

/// A count above `MAX_CELL_COUNT` is refused (past it the cell boundaries are no
/// longer exact in `f32`), the bound itself is accepted.
#[test]
fn the_cell_count_is_bounded() {
    let c = contained_child;
    assert!(c()
        .repeat_cells(SPACING, Some([MAX_CELL_COUNT, 1, 1]), true)
        .is_ok());
    assert!(matches!(
        c().repeat_cells(SPACING, Some([MAX_CELL_COUNT + 1, 1, 1]), true),
        Err(RepeatCellsError::Count(_))
    ));
}

/// With an odd count the centre cell holds the child unflipped: there the node
/// equals the child itself (and the infinite repetition's origin cell does too).
#[test]
fn the_centre_cell_is_not_flipped() {
    for count in [Some([3, 3, 1]), Some([5, 1, 1]), Some([7, 3, 1]), None] {
        let node = contained_child()
            .repeat_cells(SPACING, count, true)
            .unwrap();
        for p in [
            Vec3::new(0.45, 0.2, 0.0),
            Vec3::new(0.7, 0.3, 0.1),
            Vec3::new(-0.6, -0.5, 0.2),
        ] {
            assert_eq!(eval(&node, p), eval(&contained_child(), p), "{count:?} {p}");
        }
    }
}

/// `neighbours` and `variation` hold only the off state, and the off state is
/// stored as a plain `false` and `None` (the stored form does not change when
/// the law lands)
#[test]
fn the_off_state_is_stored_as_false_and_none() {
    let node = supported();
    let SdfNode::RepeatCells {
        neighbours,
        variation,
        ..
    } = &node
    else {
        unreachable!()
    };
    assert_eq!(*neighbours, CellNeighbours::OFF);
    assert!(variation.is_none());
    let json = alice_sdf::io::to_json_string(&SdfTree::new(node.clone())).unwrap();
    assert!(
        json.contains("\"neighbours\": false") || json.contains("\"neighbours\":false"),
        "{json}"
    );
    assert!(
        json.contains("\"variation\": null") || json.contains("\"variation\":null"),
        "{json}"
    );
    // the binary body: ... spacing, count None (0), mirror (1), neighbours (0), variation None (0)
    let body = asdf_body(&SdfTree::new(node));
    assert_eq!(tail_positions(&body).len(), 1);
}

/// The bytes after the root `RepeatCells`'s spacing: count `None`, mirror,
/// neighbours, variation `None`
fn tail_pattern() -> Vec<u8> {
    let mut v = Vec::new();
    for x in SPACING.to_array() {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v.extend_from_slice(&[0, 1, 0, 0]);
    v
}

/// Where `tail_pattern` starts in `bytes`
fn tail_positions(bytes: &[u8]) -> Vec<usize> {
    let pat = tail_pattern();
    bytes
        .windows(pat.len())
        .enumerate()
        .filter(|(_, w)| *w == pat.as_slice())
        .map(|(i, _)| i)
        .collect()
}

/// The offset of the `neighbours` byte from the start of `tail_pattern`
const NEIGHBOURS_AT: usize = 12 + 2;

fn asdf_file(tree: &SdfTree, name: &str) -> (std::path::PathBuf, Vec<u8>) {
    let path = std::env::temp_dir().join(format!(
        "alice_sdf_repeat_cells_{}_{name}.asdf",
        std::process::id()
    ));
    alice_sdf::io::save_asdf(tree, &path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    (path, bytes)
}

fn asdf_body(tree: &SdfTree) -> Vec<u8> {
    let (path, bytes) = asdf_file(tree, "body");
    let _ = std::fs::remove_file(&path);
    let header = alice_sdf::io::AsdfHeader::from_bytes(&bytes).unwrap();
    bytes[header.byte_len()..].to_vec()
}

/// The trees the readers are given: the node alone, under a union, and under a
/// metric blend (whose children an earlier tree walk skipped)
fn placements() -> Vec<(&'static str, SdfTree)> {
    vec![
        ("alone", SdfTree::new(supported())),
        (
            "union",
            SdfTree::new(SdfNode::sphere(0.5).union(supported())),
        ),
        (
            "metric_blend",
            SdfTree::new(SdfNode::metric_blend(
                SdfNode::sphere(0.5),
                supported(),
                Vec3::ZERO,
                1.0,
                0.25,
            )),
        ),
    ]
}

/// The JSON reader refuses a stored `neighbours: true` and a stored variation,
/// naming the field, wherever the node sits; the same text with the off state
/// loads
#[test]
fn the_json_reader_refuses_a_stored_neighbours_or_variation() {
    use alice_sdf::io::{from_json_string, to_json_string};
    for (place, tree) in placements() {
        let json = to_json_string(&tree).unwrap();
        assert!(from_json_string(&json).is_ok(), "{place}");
        let on = json.replacen("\"neighbours\": false", "\"neighbours\": true", 1);
        assert_ne!(on, json, "{place}: no neighbours field in {json}");
        let err = from_json_string(&on).unwrap_err().to_string();
        assert!(err.contains("neighbours"), "{place}: {err}");
        let varied = json.replacen(
            "\"variation\": null",
            "\"variation\": {\"seed\": 7, \"scale\": 0.1}",
            1,
        );
        assert_ne!(varied, json, "{place}");
        let err = from_json_string(&varied).unwrap_err().to_string();
        assert!(err.contains("variation"), "{place}: {err}");
    }
}

/// The `.asdf` reader refuses a stored `neighbours: true` (the byte flipped, the
/// checksum recomputed so only the field differs), naming the field, wherever
/// the node sits
#[test]
fn the_asdf_reader_refuses_a_stored_neighbours() {
    for (place, tree) in placements() {
        let (path, mut bytes) = asdf_file(&tree, place);
        assert!(alice_sdf::io::load_asdf(&path).is_ok(), "{place}");
        let header = alice_sdf::io::AsdfHeader::from_bytes(&bytes).unwrap();
        let start = header.byte_len();
        let at = tail_positions(&bytes[start..]);
        assert_eq!(at.len(), 1, "{place}");
        bytes[start + at[0] + NEIGHBOURS_AT] = 1;
        let crc = crc32fast::hash(&bytes[start..]);
        bytes[12..16].copy_from_slice(&crc.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let err = alice_sdf::io::load_asdf(&path).unwrap_err().to_string();
        let _ = std::fs::remove_file(&path);
        assert!(err.contains("neighbours"), "{place}: {err}");
    }
}

/// The I-packet reader refuses a stored `neighbours: true` (one coefficient
/// flipped in an otherwise valid packet), naming the field, wherever the node
/// sits
#[cfg(feature = "asp")]
#[test]
fn the_i_packet_reader_refuses_a_stored_neighbours() {
    use alice_sdf::asp_bridge::{create_sdf_i_packet, decode_sdf_i_packet};
    for (place, tree) in placements() {
        let packet = create_sdf_i_packet(&tree, 1).unwrap();
        assert!(decode_sdf_i_packet(&packet).is_ok(), "{place}");
        let mut payload = packet.as_i_packet().unwrap().clone();
        let coeffs = payload.regions[0].dct_coefficients.as_mut().unwrap();
        let bytes: Vec<u8> = coeffs.iter().map(|c| c.2 as u8).collect();
        let at = tail_positions(&bytes);
        assert_eq!(at.len(), 1, "{place}");
        coeffs[at[0] + NEIGHBOURS_AT].2 = 1.0;
        let patched = libasp::AspPacket::create_i_packet(1, payload).unwrap();
        let err = decode_sdf_i_packet(&patched).unwrap_err();
        assert!(err.contains("neighbours"), "{place}: {err}");
    }
}
