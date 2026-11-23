//! The air-side proof of gaps narrower than a limit (`alice_sdf::validity::prove_gaps`).
//!
//! oracle: closed forms of the field's direction next to facing walls and corners.
//! - Two parallel pins of radius 0.3 whose axes are 1.6 apart leave 1.0 of air between walls whose
//!   directions are opposite (dot −1) across the plane between them: against `L = 2.0` that is a gap
//!   (the nearest wall points are 1.0 apart, below 2.0); against `L = 0.9` the plane is at 0.5 from
//!   both walls, at least `L/2`, so no gap.
//! - A box standing on a floor meets it in a 90° corner of air: the two wall directions are
//!   perpendicular (dot 0, above the facing bound −0.5), so no gap.
//! - A V groove of 60° has wall directions 120° apart (dot −0.5, on the bound): its apex is a witness
//!   or undecided, and the grid's crevice reading decides it (a mouth at or above the limit is a
//!   crevice, below it a failure).
//! - A proof whose work budget runs out cannot pass: the open-air box on a floor passes with the default
//!   budget and is `Undetermined(ProofIncomplete)` with a 64th of the work it needs.

use alice_sdf::validity::{
    measure_gaps, prove_gaps, GapFailure, GapUndetermined, GapVerdict, FACING_DOT, GAP_PROOF_WORK,
    GAP_TOLERANCE,
};
use alice_sdf::SdfNode;
use glam::Vec3;

/// Grid / bound offsets: the proof must not depend on where the bound starts.
const OFFSETS: [f32; 10] = [0.0, 0.03, 0.07, 0.11, 0.13, 0.17, 0.21, 0.29, 0.37, 0.43];

const DEPTH: u32 = 9;

fn pins(axis_gap: f32) -> SdfNode {
    let floor = SdfNode::box3d(20.0, 2.0, 20.0).translate(0.0, -1.0, 0.0);
    let pin = |x: f32| SdfNode::cylinder(0.3, 6.0).translate(x, 3.0, 0.0);
    floor.union(pin(-axis_gap * 0.5)).union(pin(axis_gap * 0.5))
}

#[test]
fn the_facing_rule_is_stated() {
    // walls more than 120° apart face each other; a right angle (dot 0) and 90° corners never do
    assert!((FACING_DOT + 0.5).abs() < 1e-6);
}

#[test]
fn two_pins_one_apart_hold_a_gap_below_two_on_every_offset() {
    let s = pins(1.6);
    let mut compared = 0;
    for off in OFFSETS {
        let lo = Vec3::new(-3.0 + off, 0.5 + off, -2.0 + off);
        let p = prove_gaps(&s, 2.0, lo, Vec3::new(3.0, 5.0, 2.0), DEPTH, GAP_PROOF_WORK)
            .expect("proof");
        assert!(!p.witnesses.is_empty(), "off {off}: no gap proven ({p:?})");
        for w in p.witnesses.iter().map(|w| w.at) {
            assert!(
                w.x.abs() < 0.5 && w.z.abs() < 0.6,
                "off {off}: witness {w} outside the gap"
            );
        }
        compared += 1;
    }
    assert_eq!(compared, OFFSETS.len());
}

#[test]
fn two_pins_one_apart_hold_no_gap_below_point_nine() {
    let s = pins(1.6);
    for off in OFFSETS {
        let lo = Vec3::new(-3.0 + off, 0.5 + off, -2.0 + off);
        let p = prove_gaps(&s, 0.9, lo, Vec3::new(3.0, 5.0, 2.0), DEPTH, GAP_PROOF_WORK)
            .expect("proof");
        assert!(
            p.witnesses.is_empty(),
            "off {off}: gap below 0.9 claimed at {:?}",
            p.witnesses
        );
        assert_eq!(p.undecided, 0, "off {off}: {} boxes undecided", p.undecided);
    }
}

#[test]
fn a_box_standing_on_a_floor_has_no_gap() {
    let floor = SdfNode::box3d(20.0, 2.0, 20.0).translate(0.0, -1.0, 0.0);
    let block = SdfNode::box3d(4.0, 4.0, 4.0).translate(0.0, 2.0, 0.0);
    let s = floor.union(block);
    for off in [0.0, 0.07, 0.13, 0.21] {
        let lo = Vec3::new(-5.0 + off, 0.2 + off, -5.0 + off);
        let p = prove_gaps(&s, 2.0, lo, Vec3::new(5.0, 6.0, 5.0), DEPTH, GAP_PROOF_WORK)
            .expect("proof");
        assert!(
            p.witnesses.is_empty(),
            "off {off}: 90° corner read as a gap at {:?}",
            p.witnesses
        );
        assert_eq!(p.undecided, 0, "off {off}: {} boxes undecided", p.undecided);
    }
}

/// A V groove cut into a block: air angle `alpha`, depth chosen for the mouth width `mouth`
fn v_groove(alpha_deg: f32, mouth: f32) -> SdfNode {
    let half = (alpha_deg * 0.5).to_radians();
    let depth = mouth * 0.5 / alice_det_math::tan(half);
    let block = SdfNode::box3d(30.0, 20.0, 10.0).translate(0.0, -10.0, 0.0);
    // the groove: two half-spaces meeting at the apex (0, -depth), opening upward
    let wall = |sign: f32| {
        SdfNode::plane(
            Vec3::new(
                sign * alice_det_math::cos(half),
                -alice_det_math::sin(half),
                0.0,
            ),
            0.0,
        )
        .translate(0.0, -depth, 0.0)
    };
    block.subtract(wall(1.0).intersection(wall(-1.0)))
}

#[test]
fn a_sixty_degree_groove_is_a_crevice_by_its_mouth() {
    for (mouth, fails) in [(5.0_f32, false), (0.5, true)] {
        let s = v_groove(60.0, mouth);
        let lo = Vec3::new(-6.0, -6.0, -3.0);
        let hi = Vec3::new(6.0, 1.0, 3.0);
        let p = prove_gaps(&s, 0.8, lo, hi, DEPTH, GAP_PROOF_WORK).expect("proof");
        assert!(
            !p.witnesses.is_empty() || !p.undecided_at.is_empty(),
            "mouth {mouth}: the 60° apex is on the facing bound and must not be cleared ({p:?})"
        );
        let g = measure_gaps(&s, lo, hi, 0.2).expect("samples");
        match (g.verdict(0.8, &p), fails) {
            (GapVerdict::Fail(GapFailure::CreviceMouth { mouth: m, .. }), true) => {
                assert!(m < 0.8, "mouth read {m}");
            }
            (GapVerdict::Pass(_), false) => {}
            (v, _) => panic!("60° V mouth {mouth} against 0.8: {v:?}"),
        }
    }
}

#[test]
fn a_proof_for_another_limit_is_not_used() {
    let s = pins(1.6);
    let lo = Vec3::new(-3.0, 0.5, -2.0);
    let hi = Vec3::new(3.0, 5.0, 2.0);
    let p = prove_gaps(&s, 0.9, lo, hi, DEPTH, GAP_PROOF_WORK).expect("proof");
    let g = measure_gaps(&s, lo, hi, 0.25).expect("samples");
    assert!(matches!(g.verdict(2.0, &p), GapVerdict::Undetermined(_)));
}

#[test]
fn a_proof_out_of_boxes_does_not_pass() {
    let floor = SdfNode::box3d(20.0, 2.0, 20.0).translate(0.0, -1.0, 0.0);
    let block = SdfNode::box3d(4.0, 4.0, 4.0).translate(0.0, 2.0, 0.0);
    let s = floor.union(block);
    let lo = Vec3::new(-5.0, 0.2, -5.0);
    let hi = Vec3::new(5.0, 6.0, 5.0);
    let g = measure_gaps(&s, lo, hi, 0.25).expect("samples");
    let full = prove_gaps(&s, 2.0, lo, hi, DEPTH, GAP_PROOF_WORK).expect("proof");
    assert!(!full.exhausted && full.boxes > 8, "{} boxes", full.boxes);
    assert!(matches!(g.verdict(2.0, &full), GapVerdict::Pass(_)));
    // the budget counts evaluations weighted by node count; a small one stops the proof early, at the
    // same box on every run
    let budget = full.work / 64;
    let short = prove_gaps(&s, 2.0, lo, hi, DEPTH, budget).expect("proof");
    assert!(
        short.exhausted && short.boxes < full.boxes,
        "{} of {} boxes",
        short.boxes,
        full.boxes
    );
    assert!(
        short.work >= budget,
        "stopped at {} below the budget {budget}",
        short.work
    );
    let again = prove_gaps(&s, 2.0, lo, hi, DEPTH, budget).expect("proof");
    assert_eq!((again.boxes, again.work), (short.boxes, short.work));
    match g.verdict(2.0, &short) {
        GapVerdict::Undetermined(GapUndetermined::ProofIncomplete {
            exhausted: true, ..
        }) => {}
        v => panic!("a proof out of boxes read as {v:?}"),
    }
}

/// A plate 3 thick with a round hole of radius `r` along y.
fn plate_with_hole(r: f32) -> SdfNode {
    SdfNode::box3d_half_extents(4.0, 1.5, 4.0).subtract(SdfNode::cylinder(r, 10.0))
}

// oracle: a round hole of diameter 2R is a gap below L exactly when 2R < L; every facing pair of its
// wall points (directions more than 120° apart) spans a chord 2R·sin(φ/2), as short as 0.87·2R, so a
// chord reading calls a 2.2 hole a gap below 2.0
#[test]
fn a_round_hole_wider_than_the_limit_is_no_gap_on_every_offset() {
    let s = plate_with_hole(1.1);
    let mut compared = 0;
    for off in OFFSETS {
        let lo = Vec3::new(-1.3 + off * 0.1, -1.4 + off, -1.3 + off * 0.1);
        let hi = Vec3::new(1.3, 1.4, 1.3);
        let p = prove_gaps(&s, 2.0, lo, hi, DEPTH, GAP_PROOF_WORK).expect("proof");
        assert!(
            p.witnesses.is_empty(),
            "off {off}: a 2.2 hole read as a gap below 2.0: {:?}",
            p.witnesses
        );
        compared += 1;
    }
    assert_eq!(compared, OFFSETS.len());
}

#[test]
fn a_round_hole_narrower_than_the_limit_is_a_gap_of_its_diameter() {
    let s = plate_with_hole(0.9);
    let mut widths = 0;
    for off in OFFSETS {
        let lo = Vec3::new(-1.1 + off * 0.1, -1.4 + off, -1.1 + off * 0.1);
        let hi = Vec3::new(1.1, 1.4, 1.1);
        let p = prove_gaps(&s, 2.0, lo, hi, DEPTH, GAP_PROOF_WORK).expect("proof");
        assert!(!p.witnesses.is_empty(), "off {off}: no gap proven ({p:?})");
        for w in &p.witnesses {
            let width = w.width.expect("a witness carries its width");
            assert!(
                (width - 1.8).abs() <= 0.5 * GAP_TOLERANCE + 1e-3,
                "off {off}: width {width} at {} for a 1.8 hole",
                w.at
            );
            widths += 1;
        }
    }
    assert!(widths >= OFFSETS.len());
}

/// Two blocks with a slot of `width` between them, centred on the origin (walls at x = ±width/2)
fn slot_at_origin(width: f32) -> SdfNode {
    let half = width * 0.5;
    let block = |x: f32| SdfNode::box3d(10.0, 6.0, 10.0).translate(x, 0.0, 0.0);
    block(-half - 5.0).union(block(half + 5.0))
}

// oracle (no false pass): air narrower than the limit is a gap whatever the facing pairs read wider.
// A 1.0 slot (axis-aligned and turned about two axes), a 1.8 round hole and two pins 1.0 apart are each
// below the limit 2.0; each is proven a gap with a width no shorter than its narrowest air less half
// the tolerance, on every offset. A width read wider at every pair (the larger reading taken from a
// reading that is never short at the bottleneck) proves nothing here and turns this red.
#[test]
fn air_narrower_than_the_limit_is_a_gap_of_at_least_its_width() {
    let turned = slot_at_origin(1.0).rotate_euler(0.0, 0.31, 0.47);
    let cases: [(&str, SdfNode, Vec3, Vec3, f32); 4] = [
        (
            "slot",
            slot_at_origin(1.0),
            Vec3::splat(-0.9),
            Vec3::splat(0.9),
            1.0,
        ),
        (
            "turned slot",
            turned,
            Vec3::splat(-0.9),
            Vec3::splat(0.9),
            1.0,
        ),
        (
            "round hole",
            plate_with_hole(0.9),
            Vec3::new(-1.1, -1.4, -1.1),
            Vec3::new(1.1, 1.4, 1.1),
            1.8,
        ),
        (
            "pins",
            pins(1.6),
            Vec3::new(-3.0, 0.5, -2.0),
            Vec3::new(3.0, 5.0, 2.0),
            1.0,
        ),
    ];
    let mut compared = 0;
    for (what, node, lo, hi, narrowest) in &cases {
        for off in OFFSETS {
            let lo = *lo + Vec3::new(off * 0.1, off, off * 0.1);
            let p = prove_gaps(node, 2.0, lo, *hi, DEPTH, GAP_PROOF_WORK).expect("proof");
            assert!(
                !p.witnesses.is_empty(),
                "{what} off {off}: no gap proven ({p:?})"
            );
            for w in &p.witnesses {
                let width = w.width.expect("a witness carries its width");
                assert!(
                    width >= narrowest - 0.5 * GAP_TOLERANCE - 1e-3 && width + GAP_TOLERANCE < 2.0,
                    "{what} off {off}: width {width} at {} for air {narrowest} wide",
                    w.at
                );
                compared += 1;
            }
        }
    }
    assert!(compared >= cases.len() * OFFSETS.len());
}
