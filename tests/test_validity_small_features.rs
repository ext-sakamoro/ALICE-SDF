//! Gaps bounded by features smaller than the grid, and gaps bounded by an edge, against closed forms
//!
//! `measure_gaps` promises (its `resolution`) that every empty slab wider than `√3 · cell` between
//! walls holds a grid point within that distance of a surface. Between parts narrower than a cell (two
//! thin pins, two fin tips) the grid may hold no sample whose wall point sees the other part, so the grid
//! alone does not promise to find such a gap. The verdict does: with the air-side proof over the same
//! bound, a gap of 1.0 between such parts against a limit of 2.0 fails, either as a width measured within
//! [`GAP_TOLERANCE`] or as a gap the proof found (its width an upper bound below the limit). It is never
//! lost.
//!
//! Width bound for an edge facing a wall: the wall point of a sample is the foot of the gradient from a
//! grid point, the nearest point of the surface; on an edge the foot is the edge line itself. From a wall
//! at distance `w` from the edge, the measured width must be `w` within [`GAP_TOLERANCE`], not up to half
//! a cell more.
//!
//! oracle: closed-form distances between the primitives (cylinder axes 1.6 apart with radius 0.3 leave
//! 1.0 of air; the box tips are planes 1.0 apart; the turned block's edge line is 1.0 from the wall).

use alice_sdf::validity::{
    measure_gaps, prove_gaps, GapFailure, GapVerdict, GAP_PROOF_WORK, GAP_TOLERANCE,
};
use alice_sdf::SdfNode;
use glam::Vec3;

/// Grid offsets used to move the scene against the grid, so no case relies on one lucky alignment
const OFFSETS: [f32; 4] = [0.0, 0.07, 0.13, 0.21];

/// The limit the sub-cell gaps (1.0 wide) are failed against
const LIMIT: f32 = 2.0;

/// The verdict over `lo..hi` must fail on the 1.0 gap: a measured width within the tolerance, or a gap
/// the proof found with an upper-bound width below the limit
fn assert_gap_fails(s: &SdfNode, lo: Vec3, hi: Vec3, cell: f32, what: &str) {
    let r = measure_gaps(s, lo, hi, cell).expect("samples");
    let p = prove_gaps(s, LIMIT, lo, hi, 9, GAP_PROOF_WORK).expect("proof");
    match r.verdict(LIMIT, &p) {
        GapVerdict::Fail(GapFailure::Width { width, .. }) => assert!(
            (width - 1.0).abs() <= GAP_TOLERANCE,
            "{what}: measured {width}, true 1.0"
        ),
        GapVerdict::Fail(GapFailure::ProvenGap { width, .. }) => assert!(
            width + GAP_TOLERANCE >= 1.0 && width < LIMIT,
            "{what}: proven width {width}, true 1.0, limit {LIMIT}"
        ),
        v => panic!("{what}: the 1.0 gap was not failed ({v:?})"),
    }
}

/// Two upright pins of radius 0.3 standing on a floor (top at y = 0), surfaces 1.0 apart along x
fn two_pins() -> SdfNode {
    let floor = SdfNode::box3d(20.0, 2.0, 20.0).translate(0.0, -1.0, 0.0);
    let pin = |x: f32| SdfNode::cylinder(0.3, 6.0).translate(x, 3.0, 0.0);
    floor.union(pin(-0.8)).union(pin(0.8))
}

#[test]
fn the_gap_between_two_pins_thinner_than_a_cell_fails() {
    let s = two_pins();
    for cell in [0.5_f32, 0.4] {
        for off in OFFSETS {
            let lo = Vec3::new(-3.0 + off, 0.5, -2.0 + off);
            let what = format!("pins, cell {cell} off {off}");
            assert_gap_fails(&s, lo, Vec3::new(3.0, 5.0, 2.0), cell, &what);
        }
    }
}

/// Two fins 0.2 thick lying in y = 0, their flat tips facing each other 1.0 apart (x = ±0.5)
fn two_fins() -> SdfNode {
    let fin = |x: f32| SdfNode::box3d(6.0, 0.2, 8.0).translate(x, 0.0, 0.0);
    fin(-3.5).union(fin(3.5))
}

#[test]
fn the_gap_between_two_fin_tips_thinner_than_a_cell_fails() {
    let s = two_fins();
    for cell in [0.5_f32, 0.4] {
        for off in OFFSETS {
            let lo = Vec3::new(-3.0 + off, -2.0 + off, -2.0 + off);
            let what = format!("fins, cell {cell} off {off}");
            assert_gap_fails(&s, lo, Vec3::new(3.0, 2.0, 2.0), cell, &what);
        }
    }
}

/// A wall whose face is x = 0 and a square block turned 45 degrees whose edge line is x = -1
fn edge_facing_wall() -> SdfNode {
    let wall = SdfNode::box3d(4.0, 20.0, 20.0).translate(2.0, 0.0, 0.0);
    let half_diagonal = 2.0 * 2.0_f32.sqrt();
    let block = SdfNode::box3d(4.0, 4.0, 10.0)
        .rotate_euler(0.0, 0.0, std::f32::consts::FRAC_PI_4)
        .translate(-1.0 - half_diagonal, 0.0, 0.0);
    wall.union(block)
}

#[test]
fn an_edge_facing_a_wall_is_measured_within_the_stated_tolerance() {
    let s = edge_facing_wall();
    for cell in [0.25_f32, 0.33, 0.4, 0.5] {
        for off in OFFSETS {
            let lo = Vec3::new(-8.0 + off, -5.07 + off, -3.0);
            let r = measure_gaps(&s, lo, Vec3::new(0.5, 5.0, 3.0), cell).expect("samples");
            let w = r
                .min_gap
                .unwrap_or_else(|| panic!("cell {cell} off {off}: edge gap lost ({r:?})"));
            assert!(
                (w - 1.0).abs() <= GAP_TOLERANCE,
                "cell {cell} off {off}: edge to face measured {w}, true 1.0"
            );
        }
    }
}
