//! Through cuts and the grid's sheet test: a cutter that passes through the body leaves its own faces in
//! the air beyond the body (a cylinder bore's end caps, a slot box's far face). Inside the cut the field
//! of `body − cutter` is the cutter's term, a lower bound of the distance whose gradient points at those
//! faces, not at the nearest wall: `p − d n` from a grid point there lands on a cutter face in the air,
//! where the field is not zero. Such a point is on no surface, so it is neither a wall to read a width
//! from nor a sheet; the grid finds the wall by marching and counts the samples that reach none.
//!
//! The four shapes are the reductions of LOL presets that read "gap not measured" (one through cut each,
//! the trees as the presets build them, bound and pitch as a 2 mm limit reads them).

use alice_sdf::tight_aabb::{compute_tight_aabb_with_config, TightAabbConfig};
use alice_sdf::validity::{measure_gaps, GapError, GapReport};
use alice_sdf::SdfNode;
use core::f32::consts::FRAC_PI_2;
use glam::Vec3;

/// The grid pitch a 2 mm limit is read at (`L / (2·√3)`)
fn cell() -> f32 {
    2.0 / (2.0 * 3.0_f32.sqrt())
}

fn report(node: &SdfNode) -> GapReport {
    let b = compute_tight_aabb_with_config(node, &TightAabbConfig::preset_large());
    measure_gaps(node, b.min, b.max, cell()).expect("samples")
}

/// No sample is read as a sheet: there is no zero-thickness surface in a single through cut. The samples
/// inside the cut whose `p − d n` lands on the cutter's face in the air are counted as having no wall
fn assert_no_sheet(what: &str, r: &GapReport) {
    assert!(
        r.no_wall_samples > 0 && r.no_wall_samples < r.samples,
        "{what}: samples with no wall {} of {}",
        r.no_wall_samples,
        r.samples
    );
    assert_eq!(
        (r.sheet_samples, r.unconfirmed_sheet_samples),
        (0, 0),
        "{what}: sheet samples (confirmed, unconfirmed) at {:?} / {:?}",
        r.sheet_location,
        r.unconfirmed_sheet_location
    );
}

/// LOL `cable_clip(7, 28)`: a block with a cylinder bore through its length, the bore 2 longer than the block
#[test]
fn a_bore_through_a_block_reads_no_sheet() {
    let node = SdfNode::box3d_half_extents(5.5, 14.0, 5.5).subtract(SdfNode::cylinder(3.6, 30.0));
    assert_no_sheet("cable clip", &report(&node));
}

/// LOL `esp32_enclosure(45, 18, 12)`: a rounded case with a port box cut through one end wall
#[test]
fn a_port_through_a_rounded_wall_reads_no_sheet() {
    let node = SdfNode::rounded_box(23.0, 7.0, 9.5, 1.6)
        .subtract(SdfNode::box3d_half_extents(1.8, 3.25, 4.5).translate(23.3, 4.75, 0.0))
        .rotate_euler(FRAC_PI_2, 0.0, 0.0);
    assert_no_sheet("esp32 enclosure", &report(&node));
}

/// LOL `card_tray(44, 68, 25)`: a block with a cylinder cut through its thickness
#[test]
fn a_round_cut_through_a_tray_reads_no_sheet() {
    let node = SdfNode::box3d_half_extents(25.0, 13.25, 37.0)
        .subtract(SdfNode::cylinder(9.0, 28.5))
        .rotate_euler(FRAC_PI_2, 0.0, 0.0);
    assert_no_sheet("card tray", &report(&node));
}

/// LOL `can_rack(2, 66, 10)`: a long plate with a box notch cut through its end
#[test]
fn a_notch_through_a_plate_end_reads_no_sheet() {
    let node = SdfNode::box3d_half_extents(1.5, 141.47, 207.0)
        .subtract(SdfNode::box3d_half_extents(37.0, 35.5, 3.0).translate(0.0, -31.03, -205.5))
        .rotate_euler(FRAC_PI_2, 0.0, 0.0);
    assert_no_sheet("can rack", &report(&node));
}

/// A real zero-thickness sheet beside a through bore: the T-slot cutter (a neck box and an outward
/// punch box unioned on the block's face) leaves the field exactly 0 over the opening, air on both
/// sides; a bore through the block's height leaves its caps in the air above and below. The sheet is
/// still found and confirmed on the exact field, and the bore's caps add no sheet of either kind.
#[test]
fn a_real_sheet_beside_a_through_bore_is_still_a_sheet() {
    let block = SdfNode::box3d(20.0, 20.0, 20.0); // faces at ±10
    let neck = SdfNode::box3d(2.0, 6.0, 30.0).translate(9.0, 0.0, 0.0);
    let punch = SdfNode::box3d(5.0, 6.0, 30.0).translate(12.5, 0.0, 0.0);
    let bore = SdfNode::cylinder(2.0, 30.0).translate(-5.0, 0.0, 0.0);
    let node = block.subtract(neck.union(punch)).subtract(bore);
    let r = measure_gaps(
        &node,
        Vec3::new(-10.5, -10.5, -3.0),
        Vec3::new(12.0, 10.5, 3.0),
        0.25,
    )
    .expect("samples");
    assert!(r.sheet_samples > 0, "{r:?}");
    let s = r.sheet_location.expect("a sheet point");
    assert!(
        (s.x - 10.0).abs() < 0.01 && s.y.abs() <= 3.01,
        "sheet at {s}"
    );
    assert_eq!(
        r.unconfirmed_sheet_samples, 0,
        "unconfirmed sheet at {:?}",
        r.unconfirmed_sheet_location
    );
}

/// A bound inside the cable clip's bore, at the block's top end: every grid point near a surface there
/// takes its direction from the bore's end cap out in the air (the bore wall is beyond the sample reach),
/// so no sample has a wall. Such a grid compared nothing, and says so rather than reporting no gap
#[test]
fn a_grid_whose_samples_all_find_no_wall_measured_nothing() {
    let node = SdfNode::box3d_half_extents(5.5, 14.0, 5.5).subtract(SdfNode::cylinder(3.6, 30.0));
    match measure_gaps(
        &node,
        Vec3::new(-1.0, 13.8, -1.0),
        Vec3::new(1.0, 14.6, 1.0),
        cell(),
    ) {
        Err(GapError::NoSamples) => {}
        r => panic!("bore end: {r:?}"),
    }
}

/// Two plates 1 apart in x, the field squashed along z by 4 (the field is a quarter of the distance, so
/// `p − d n` stops a quarter of the way to the wall): the grid marches to the wall, and the width it
/// reads is the walls' distance, 1, not `d` plus the way back (about 0.8 here)
#[test]
fn a_gap_in_a_field_slower_than_the_distance_reads_the_walls_distance() {
    let plates = SdfNode::box3d_half_extents(2.0, 3.0, 3.0)
        .translate(-2.5, 0.0, 0.0)
        .union(SdfNode::box3d_half_extents(2.0, 3.0, 3.0).translate(2.5, 0.0, 0.0))
        .scale_xyz(1.0, 1.0, 0.25);
    let r = measure_gaps(
        &plates,
        Vec3::new(-1.0, -2.0, -0.6),
        Vec3::new(1.0, 2.0, 0.6),
        0.25,
    )
    .expect("samples");
    let w = r.min_gap.expect("a width");
    assert!((w - 1.0).abs() < 0.01, "width {w} ({r:?})");
}
