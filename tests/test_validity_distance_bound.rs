//! The air proof must read the distance to the material, not the field value, where a cutter leaves
//! virtual walls.
//!
//! oracle: closed forms of a slot cut through a slab.
//! - A slab `|y| ≤ 1` with a slot `|x| < 0.7` cut by a cutter that reaches `|y| ≤ 1.1`: the cutter's
//!   top and bottom faces lie 0.1 outside the slab, so inside the slot near the slab's faces the field
//!   `max(a, −b)` is about 0.1, while the nearest material (the slot walls) is up to 0.7 away. The
//!   only gap is the slot, 1.4 wide: against `L = 0.5` there is no gap; against `L = 2.0` there is one.
//! - The same values of `a` and `b` arise when a block of material sits on the cutter's top face. A
//!   block over `1.1 ≤ y ≤ 2` leaves a layer of air 0.1 thick over the slab, a real gap below 0.5.
//!   No rule on the node values alone can tell the two scenes apart, so this pins that the proof does
//!   not clear the first by trusting the field, nor the second by ignoring it.

use alice_sdf::validity::{prove_gaps, GAP_PROOF_WORK};
use alice_sdf::SdfNode;
use glam::Vec3;

const DEPTH: u32 = 9;

fn slotted_slab() -> SdfNode {
    let slab = SdfNode::box3d(8.0, 2.0, 8.0);
    let cutter = SdfNode::box3d(1.4, 2.2, 10.0);
    slab.subtract(cutter)
}

fn lo() -> Vec3 {
    Vec3::new(-3.0, -1.6, -2.0)
}

fn hi() -> Vec3 {
    Vec3::new(3.0, 1.6, 2.0)
}

#[test]
fn a_slot_wider_than_the_limit_is_not_a_gap_though_the_cutter_overhangs() {
    let p = prove_gaps(&slotted_slab(), 0.5, lo(), hi(), DEPTH, GAP_PROOF_WORK).expect("proof");
    assert!(p.witnesses.is_empty(), "gap below 0.5 claimed at {:?}", p.witnesses);
    assert!(!p.exhausted, "budget exhausted ({} boxes)", p.boxes);
    assert_eq!(p.undecided, 0, "{} boxes undecided", p.undecided);
}

#[test]
fn the_same_slot_is_a_gap_below_two() {
    let p = prove_gaps(&slotted_slab(), 2.0, lo(), hi(), DEPTH, GAP_PROOF_WORK).expect("proof");
    assert!(!p.witnesses.is_empty(), "no gap proven ({p:?})");
    for w in p.witnesses.iter().map(|w| w.at) {
        assert!(w.x.abs() < 0.7 && w.y.abs() < 1.0, "witness {w} outside the slot");
    }
}

#[test]
fn material_on_the_cutter_face_makes_a_thin_layer_of_air_a_gap() {
    let block = SdfNode::box3d(8.0, 0.9, 8.0).translate(0.0, 1.55, 0.0);
    let s = slotted_slab().union(block);
    let p = prove_gaps(&s, 0.5, lo(), Vec3::new(3.0, 2.4, 2.0), DEPTH, GAP_PROOF_WORK)
        .expect("proof");
    assert!(!p.witnesses.is_empty(), "the 0.1 layer was cleared ({p:?})");
    for w in p.witnesses.iter().map(|w| w.at) {
        assert!(w.y > 0.95 && w.y < 1.15, "witness {w} outside the layer");
    }
}
