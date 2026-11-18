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

const fn lo() -> Vec3 {
    Vec3::new(-3.0, -1.6, -2.0)
}

const fn hi() -> Vec3 {
    Vec3::new(3.0, 1.6, 2.0)
}

#[test]
fn a_slot_wider_than_the_limit_is_not_a_gap_though_the_cutter_overhangs() {
    let p = prove_gaps(&slotted_slab(), 0.5, lo(), hi(), DEPTH, GAP_PROOF_WORK).expect("proof");
    assert!(
        p.witnesses.is_empty(),
        "gap below 0.5 claimed at {:?}",
        p.witnesses
    );
    assert!(!p.exhausted, "budget exhausted ({} boxes)", p.boxes);
    assert_eq!(p.undecided, 0, "{} boxes undecided", p.undecided);
}

#[test]
fn the_same_slot_is_a_gap_below_two() {
    let p = prove_gaps(&slotted_slab(), 2.0, lo(), hi(), DEPTH, GAP_PROOF_WORK).expect("proof");
    assert!(!p.witnesses.is_empty(), "no gap proven ({p:?})");
    for w in p.witnesses.iter().map(|w| w.at) {
        assert!(
            w.x.abs() < 0.7 && w.y.abs() < 1.0,
            "witness {w} outside the slot"
        );
    }
}

#[test]
fn material_on_the_cutter_face_makes_a_thin_layer_of_air_a_gap() {
    let block = SdfNode::box3d(8.0, 0.9, 8.0).translate(0.0, 1.55, 0.0);
    let s = slotted_slab().union(block);
    let p = prove_gaps(
        &s,
        0.5,
        lo(),
        Vec3::new(3.0, 2.4, 2.0),
        DEPTH,
        GAP_PROOF_WORK,
    )
    .expect("proof");
    assert!(!p.witnesses.is_empty(), "the 0.1 layer was cleared ({p:?})");
    for w in p.witnesses.iter().map(|w| w.at) {
        assert!(w.y > 0.95 && w.y < 1.15, "witness {w} outside the layer");
    }
}

/// A V groove of 40° and depth 3 cut into a slab (its air is a real crevice, the field exact there).
fn v_groove() -> SdfNode {
    let slab = SdfNode::box3d(30.0, 10.0, 20.0).translate(0.0, -5.0, 0.0);
    let half = 20.0_f32.to_radians();
    let half_space = |t: f32| {
        SdfNode::box3d(60.0, 60.0, 40.0)
            .translate(30.0, 0.0, 0.0)
            .rotate_euler(0.0, 0.0, t)
    };
    slab.subtract(
        half_space(half)
            .intersection(half_space(std::f32::consts::PI - half))
            .translate(0.0, -3.0, 0.0),
    )
}

/// Nanoseconds per unit of work of one proof, the fastest of three runs.
// the work counts stay far below 2^53, so the cast to f64 is exact
#[allow(clippy::cast_precision_loss)]
#[allow(
    clippy::disallowed_methods,
    reason = "the test compares wall-clock time per unit of work; the proof itself never reads the clock"
)]
fn ns_per_unit(s: &SdfNode, limit: f32, lo: Vec3, hi: Vec3) -> f64 {
    (0..3)
        .map(|_| {
            let t = std::time::Instant::now();
            let p = prove_gaps(s, limit, lo, hi, DEPTH, GAP_PROOF_WORK).expect("proof");
            assert!(!p.exhausted, "budget exhausted ({} boxes)", p.boxes);
            t.elapsed().as_secs_f64() * 1e9 / p.work as f64
        })
        .fold(f64::INFINITY, f64::min)
}

/// oracle: the budget bounds time only if work tracks it. The proof on the slot leans on the material
/// cover (cached reads and heap work), the V groove on the field alone; their time per unit of work
/// stays within a factor of 4 of each other.
#[test]
fn work_tracks_time_with_and_without_the_material_cover() {
    let slot = ns_per_unit(&slotted_slab(), 0.5, lo(), hi());
    let v = ns_per_unit(
        &v_groove(),
        1.9,
        Vec3::new(-5.0, -4.0, -3.0),
        Vec3::new(5.0, 1.0, 3.0),
    );
    let ratio = slot.max(v) / slot.min(v);
    assert!(
        ratio <= 4.0,
        "slot {slot:.1} ns/unit, V {v:.1} ns/unit (ratio {ratio:.2})"
    );
}

// Scenes where the witness cannot clear the gap and only the material cover decides: under a cutter
// that overhangs the part by a little, the field reads the overhang (a few hundredths) while the walls
// are several times further, so a march to a wall is no evidence of a wall (it runs more than twice
// the field value). The proof's region is the band under the overhang alone, so no witness from the
// depth of the cut can stand in for the cover.

/// Whether the proof left the region without a gap claim or an undecided box (it passed it).
fn passed(s: &SdfNode, limit: f32, lo: Vec3, hi: Vec3) -> bool {
    let p = prove_gaps(s, limit, lo, hi, DEPTH, GAP_PROOF_WORK).expect("proof");
    assert!(!p.exhausted, "budget exhausted ({} boxes)", p.boxes);
    p.witnesses.is_empty() && p.undecided == 0
}

/// A slab `|y| <= 1` cut by a cutter `width` wide (centred on x = 0) whose top lies `overhang` above
/// the slab.
fn slot_under_overhang(slab_width: f32, width: f32, overhang: f32) -> SdfNode {
    let slab = SdfNode::box3d(slab_width, 2.0, 8.0);
    slab.subtract(SdfNode::box3d(width, 2.0 + 2.0 * overhang, 10.0))
}

#[test]
fn a_slot_narrower_than_the_limit_under_an_overhang_does_not_pass() {
    // width 0.4 against L = 0.5; under the top face the field reads at most 0.02 + depth below it
    let s = slot_under_overhang(8.0, 0.4, 0.02);
    let (lo, hi) = (Vec3::new(-0.15, 0.9, -0.5), Vec3::new(0.15, 0.98, 0.5));
    assert!(
        !passed(&s, 0.5, lo, hi),
        "the 0.4 slot under the overhang passed"
    );
}

#[test]
fn a_slot_between_thin_fins_under_an_overhang_does_not_pass() {
    // the walls are fins 0.1 thick: the field inside them never goes below −0.05
    let s = slot_under_overhang(0.6, 0.4, 0.02);
    let (lo, hi) = (Vec3::new(-0.15, 0.9, -0.5), Vec3::new(0.15, 0.98, 0.5));
    assert!(!passed(&s, 0.5, lo, hi), "the 0.4 slot between fins passed");
}

#[test]
fn a_forty_degree_v_under_an_overhang_does_not_pass() {
    // a V 40° wide with its apex 0.5 below the top face, cut by a cutter 0.01 over the face: on the
    // bisector the walls face at 140° (dot −0.77) and lie about 0.15 away
    let slab = SdfNode::box3d(30.0, 10.0, 20.0).translate(0.0, -5.0, 0.0);
    let half = 20.0_f32.to_radians();
    let half_space = |t: f32| {
        SdfNode::box3d(60.0, 60.0, 40.0)
            .translate(30.0, 0.0, 0.0)
            .rotate_euler(0.0, 0.0, t)
    };
    let wedge = half_space(half)
        .intersection(half_space(std::f32::consts::PI - half))
        .translate(0.0, -0.5, 0.0);
    let cap = SdfNode::box3d(60.0, 60.0, 40.0).translate(0.0, 0.01 - 30.0, 0.0);
    let s = slab.subtract(wedge.intersection(cap));
    let (lo, hi) = (Vec3::new(-0.05, -0.1, -0.5), Vec3::new(0.05, -0.03, 0.5));
    assert!(
        !passed(&s, 1.0, lo, hi),
        "the 40° V under the overhang passed"
    );
}

#[test]
fn one_wall_under_an_overhang_passes() {
    // the cutter takes the slab's whole x > 0 half: one wall, no gap, though the wall's points within
    // the reach spread along it in directions almost opposite
    let slab = SdfNode::box3d(8.0, 2.0, 8.0);
    let s = slab.subtract(SdfNode::box3d(10.0, 2.04, 10.0).translate(5.0, 0.0, 0.0));
    let (lo, hi) = (Vec3::new(0.05, 0.9, -0.5), Vec3::new(0.3, 0.98, 0.5));
    let p = prove_gaps(&s, 0.5, lo, hi, DEPTH, GAP_PROOF_WORK).expect("proof");
    assert!(p.witnesses.is_empty(), "gap claimed at {:?}", p.witnesses);
    assert!(!p.exhausted, "budget exhausted ({} boxes)", p.boxes);
    assert_eq!(p.undecided, 0, "{} boxes undecided", p.undecided);
}
