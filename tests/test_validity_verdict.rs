//! One printability verdict for gaps against a limit: `GapReport::verdict(limit)`
//!
//! The report lists widths, crevices, sheets and the resolutions separately; a caller comparing against
//! a printer's limit needs one answer, and that answer has to carry the sizes it holds at. Rules (each
//! pinned below against a closed-form scene):
//!
//! * `Fail` when a measured width `w` has `w + GAP_TOLERANCE < limit`, when a crevice's mouth (its widest
//!   perpendicular width before the perpendicular leaves the groove) is below the limit (the groove cannot
//!   be printed open), or when the field has a zero-thickness sheet (no material, a defect of the field).
//! * otherwise `Undetermined` when the limit is not a finite positive number, or when the any-shape
//!   resolution (`2·√3·cell`, the narrowest gap between arbitrary parts the grid is sure to find) is above
//!   the limit: a gap narrower than the limit could lie between the grid points.
//! * otherwise `Pass(GapCertificate)`: no gap in `[any_shape_resolution, limit)` was found, and nothing
//!   narrower than `any_shape_resolution` and no feature smaller than `min_certified_feature` (one cell)
//!   is claimed. There is no bare pass.
//!
//! A `Fail` outranks `Undetermined`: a gap found below the limit fails whatever else is unknown.

use alice_sdf::validity::{
    measure_gaps, GapCertificate, GapFailure, GapUndetermined, GapVerdict, GAP_TOLERANCE,
};
use alice_sdf::SdfNode;
use glam::Vec3;

const OFFSETS: [f32; 4] = [0.0, 0.07, 0.13, 0.21];

fn any_shape(cell: f32) -> f32 {
    2.0 * 3.0_f32.sqrt() * cell
}

fn assert_certificate(v: GapVerdict, cell: f32, what: &str) {
    match v {
        GapVerdict::Pass(GapCertificate {
            cell: c,
            any_shape_resolution,
            min_certified_feature,
        }) => {
            assert!((c - cell).abs() < 1e-6, "{what}: cell {c}");
            assert!(
                (any_shape_resolution - any_shape(cell)).abs() < 1e-5,
                "{what}: any-shape resolution {any_shape_resolution}"
            );
            assert!(
                min_certified_feature >= cell,
                "{what}: certifies features of {min_certified_feature} below one cell {cell}"
            );
        }
        v => panic!("{what}: expected a pass, got {v:?}"),
    }
}

/// Two blocks with a slot of `width` between them centred at x0
fn slot(width: f32, x0: f32) -> SdfNode {
    let half = width * 0.5;
    let left = SdfNode::box3d(10.0, 6.0, 10.0).translate(x0 - half - 5.0, 3.0, 0.0);
    let right = SdfNode::box3d(10.0, 6.0, 10.0).translate(x0 + half + 5.0, 3.0, 0.0);
    left.union(right)
}

fn slot_bounds() -> (Vec3, Vec3) {
    (Vec3::new(-12.0, -1.0, -6.0), Vec3::new(12.0, 7.0, 6.0))
}

#[test]
fn a_slot_below_the_limit_fails_with_its_width_and_passes_a_lower_limit() {
    let (lo, hi) = slot_bounds();
    let cell = 0.25;
    let r = measure_gaps(&slot(1.0, 0.37), lo, hi, cell).expect("samples");
    match r.verdict(2.0) {
        GapVerdict::Fail(GapFailure::Width { width, at }) => {
            assert!((width - 1.0).abs() <= GAP_TOLERANCE, "width {width}");
            assert!(at.x > -0.2 && at.x < 0.9, "at {at}");
        }
        v => panic!("slot 1.0 against 2.0: {v:?}"),
    }
    // 0.9 is above the any-shape resolution 0.866 and below the slot: a pass with its sizes
    assert_certificate(r.verdict(0.9), cell, "slot 1.0 against 0.9");
    // exactly at the limit is not failed by the measurement's own error
    assert_certificate(r.verdict(1.0), cell, "slot 1.0 against 1.0");
}

#[test]
fn nothing_found_passes_only_when_the_grid_resolves_the_limit() {
    let block = SdfNode::box3d(10.0, 6.0, 10.0);
    let (lo, hi) = (Vec3::splat(-8.0), Vec3::splat(8.0));
    let r = measure_gaps(&block, lo, hi, 0.1).expect("samples");
    assert_certificate(r.verdict(0.5), 0.1, "open air at cell 0.1 against 0.5");
    // at cell 0.25 the any-shape resolution 0.866 is above 0.5: a 0.5 gap could hide between the points
    let r = measure_gaps(&block, lo, hi, 0.25).expect("samples");
    match r.verdict(0.5) {
        GapVerdict::Undetermined(GapUndetermined::ResolutionAboveLimit {
            any_shape_resolution,
        }) => {
            assert!((any_shape_resolution - any_shape(0.25)).abs() < 1e-5);
        }
        v => panic!("open air at a limit below the any-shape resolution: {v:?}"),
    }
}

#[test]
fn a_limit_that_is_not_a_finite_positive_number_is_undetermined() {
    let (lo, hi) = slot_bounds();
    let r = measure_gaps(&slot(1.0, 0.37), lo, hi, 0.25).expect("samples");
    for limit in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            r.verdict(limit),
            GapVerdict::Undetermined(GapUndetermined::BadLimit),
            "limit {limit}"
        );
    }
}

#[test]
fn a_zero_field_sheet_fails_whatever_the_limit() {
    let block = SdfNode::box3d(20.0, 20.0, 20.0);
    let neck = SdfNode::box3d(2.0, 6.0, 30.0).translate(9.0, 0.0, 0.0);
    let punch = SdfNode::box3d(5.0, 6.0, 30.0).translate(12.5, 0.0, 0.0);
    let seamed = block.subtract(neck.union(punch));
    let r = measure_gaps(
        &seamed,
        Vec3::new(5.0, -5.0, -3.0),
        Vec3::new(12.0, 5.0, 3.0),
        0.25,
    )
    .expect("samples");
    for limit in [0.9_f32, 2.0] {
        match r.verdict(limit) {
            GapVerdict::Fail(GapFailure::Sheet { at }) => {
                assert!((at.x - 10.0).abs() < 0.01 && at.y.abs() <= 3.01, "at {at}");
            }
            v => panic!("sheet against {limit}: {v:?}"),
        }
    }
}

/// A slab with top y = 0 and a V groove of air angle `alpha` degrees and depth `depth` along z
fn v_groove(alpha: f32, depth: f32) -> SdfNode {
    let slab = SdfNode::box3d(30.0, 10.0, 20.0).translate(0.0, -5.0, 0.0);
    let half = alpha.to_radians() * 0.5;
    // half-space { p · n >= 0 } for n at angle θ in the xy plane
    let half_space = |theta: f32| {
        SdfNode::box3d(60.0, 60.0, 40.0)
            .translate(30.0, 0.0, 0.0)
            .rotate_euler(0.0, 0.0, theta)
    };
    let cutter = half_space(half)
        .intersection(half_space(std::f32::consts::PI - half))
        .translate(0.0, -depth, 0.0);
    slab.subtract(cutter)
}

/// oracle: the widest perpendicular width of a symmetric V of air angle α and depth D is reached where
/// the perpendicular from one wall meets the other wall's rim: 2·D·sin(α/2)
#[test]
fn a_crevice_whose_mouth_is_below_the_limit_fails_with_its_mouth() {
    let (alpha, depth) = (40.0_f32, 3.0);
    let mouth_true = 2.0 * depth * (alpha.to_radians() * 0.5).sin(); // 2.052
    for cell in [0.15_f32, 0.2, 0.25] {
        let r = measure_gaps(
            &v_groove(alpha, depth),
            Vec3::new(-5.0, -4.0, -3.0),
            Vec3::new(5.0, 1.0, 3.0),
            cell,
        )
        .expect("samples");
        assert!(
            !r.crevices.is_empty(),
            "cell {cell}: the V is a crevice ({r:?})"
        );
        for c in &r.crevices {
            assert!(
                (c.mouth - mouth_true).abs() <= 0.05,
                "cell {cell}: mouth {} vs {mouth_true}",
                c.mouth
            );
        }
        match r.verdict(3.0) {
            GapVerdict::Fail(GapFailure::CreviceMouth { mouth, .. }) => {
                assert!(
                    (mouth - mouth_true).abs() <= 0.05,
                    "cell {cell}: mouth {mouth}"
                );
            }
            v => panic!("cell {cell}: V mouth {mouth_true} against 3.0: {v:?}"),
        }
        // a mouth wider than the limit prints open (its bottom rounds off): no failure from the crevice
        assert_certificate(r.verdict(1.9), cell, "V mouth 2.05 against 1.9");
    }
}

#[test]
fn a_gap_narrower_than_the_any_shape_resolution_never_passes_bare() {
    // two pins of radius 0.3 with 0.3 of air between them, cell 0.5 (any-shape resolution 1.73):
    // against 0.5 nothing can be certified
    let floor = SdfNode::box3d(20.0, 2.0, 20.0).translate(0.0, -1.0, 0.0);
    let pin = |x: f32| SdfNode::cylinder(0.3, 6.0).translate(x, 3.0, 0.0);
    let s = floor.union(pin(-0.45)).union(pin(0.45));
    for off in OFFSETS {
        let lo = Vec3::new(-3.0 + off, 0.5, -2.0 + off);
        let r = measure_gaps(&s, lo, Vec3::new(3.0, 5.0, 2.0), 0.5).expect("samples");
        assert!(
            !matches!(r.verdict(0.5), GapVerdict::Pass(_)),
            "off {off}: {:?} ({r:?})",
            r.verdict(0.5)
        );
    }
}

/// fine cell: found; coarse cell: Undetermined, or a pass that certifies nothing below one cell
fn fine_and_coarse(
    name: &str,
    s: &SdfNode,
    lo0: Vec3,
    hi: Vec3,
    truth: f32,
    fine: f32,
    coarse: f32,
    limit: f32,
) {
    for off in OFFSETS {
        let lo = lo0 + Vec3::splat(off);
        let r = measure_gaps(s, lo, hi, fine).expect("samples");
        match r.verdict(limit) {
            GapVerdict::Fail(GapFailure::Width { width, .. }) => assert!(
                (width - truth).abs() <= GAP_TOLERANCE,
                "{name} off {off} cell {fine}: width {width} vs {truth}"
            ),
            v => panic!("{name} off {off} cell {fine}: {v:?} ({r:?})"),
        }
        let r = measure_gaps(s, lo, hi, coarse).expect("samples");
        match r.verdict(limit) {
            GapVerdict::Fail(GapFailure::Width { width, .. }) => assert!(
                (width - truth).abs() <= GAP_TOLERANCE,
                "{name} off {off} cell {coarse}: width {width} vs {truth}"
            ),
            GapVerdict::Undetermined(_) => {}
            v @ GapVerdict::Pass(_) => assert_certificate(v, coarse, name),
            v => panic!("{name} off {off} cell {coarse}: {v:?}"),
        }
    }
}

#[test]
fn a_shallow_groove_is_measured_at_a_fine_cell_and_not_certified_below_one_cell() {
    // groove 1.0 wide and 0.2 deep in a slab with top y = 0
    let slab = SdfNode::box3d(20.0, 4.0, 10.0).translate(0.0, -2.0, 0.0);
    let groove = SdfNode::box3d(1.0, 1.0, 20.0).translate(0.13, 0.3, 0.0);
    let s = slab.subtract(groove);
    fine_and_coarse(
        "shallow groove",
        &s,
        Vec3::new(-4.0, -1.0, -2.0),
        Vec3::new(4.0, 2.0, 2.0),
        1.0,
        0.05,
        0.5,
        2.0,
    );
}

#[test]
fn two_small_spheres_are_measured_at_a_fine_cell_and_not_certified_below_one_cell() {
    // r 0.3 (below 0.69 · 0.5), surfaces 1.0 apart
    let s = SdfNode::sphere(0.3)
        .translate(-0.8, 0.0, 0.0)
        .union(SdfNode::sphere(0.3).translate(0.8, 0.0, 0.0));
    fine_and_coarse(
        "spheres",
        &s,
        Vec3::new(-2.0, -1.5, -1.5),
        Vec3::new(2.0, 1.5, 1.5),
        1.0,
        0.25,
        0.5,
        2.0,
    );
}

#[test]
fn a_thin_fin_is_measured_at_a_fine_cell_and_not_certified_below_one_cell() {
    // fin 0.2 thick (below one cell) whose tip faces a wall 1.0 away
    let wall = SdfNode::box3d(4.0, 20.0, 20.0).translate(2.0, 0.0, 0.0);
    let fin = SdfNode::box3d(6.0, 0.2, 10.0).translate(-4.0, 0.0, 0.0);
    let s = wall.union(fin);
    fine_and_coarse(
        "thin fin",
        &s,
        Vec3::new(-8.0, -5.07, -3.0),
        Vec3::new(0.5, 5.0, 3.0),
        1.0,
        0.25,
        0.5,
        2.0,
    );
}

#[test]
fn a_gap_between_small_parts_just_above_the_any_shape_bound_fails_at_the_coarse_cell() {
    // r 0.3 spheres with 1.8 of air: above 2·√3·0.5 = 1.732, so the medial pass must find it at cell 0.5
    let s = SdfNode::sphere(0.3)
        .translate(-1.2, 0.0, 0.0)
        .union(SdfNode::sphere(0.3).translate(1.2, 0.0, 0.0));
    assert!(1.8 > any_shape(0.5));
    for off in OFFSETS {
        let lo = Vec3::new(-2.5, -1.5, -1.5) + Vec3::splat(off);
        let r = measure_gaps(&s, lo, Vec3::new(2.5, 1.5, 1.5), 0.5).expect("samples");
        match r.verdict(2.0) {
            GapVerdict::Fail(GapFailure::Width { width, .. }) => assert!(
                (width - 1.8).abs() <= GAP_TOLERANCE,
                "off {off}: width {width}"
            ),
            v => panic!("off {off}: 1.8 between spheres against 2.0 at cell 0.5: {v:?} ({r:?})"),
        }
    }
}

/// The thin slice: two pins of radius 0.3 with 1.0 of air between them, measured at cell 0.25, give
/// one verdict per limit.
///
/// oracle: the closed form of two parallel cylinders of radius r whose axes are `c` apart: the
/// narrowest air between them is `c − 2r` = 1.6 − 0.6 = 1.0, on the line through both axes; against a
/// limit of 2.0 that is a `Fail` with width 1.0 (± `GAP_TOLERANCE`), against 0.9 (above the any-shape
/// resolution `2·√3·0.25 ≈ 0.866`) a `Pass` whose certificate states cell 0.25, 0.866 and one cell
#[test]
fn two_pins_one_apart_fail_two_and_pass_point_nine_at_cell_quarter() {
    let floor = SdfNode::box3d(20.0, 2.0, 20.0).translate(0.0, -1.0, 0.0);
    let pin = |x: f32| SdfNode::cylinder(0.3, 6.0).translate(x, 3.0, 0.0);
    let s = floor.union(pin(-0.8)).union(pin(0.8));
    let cell = 0.25;
    let mut compared = 0;
    for off in OFFSETS {
        let lo = Vec3::new(-3.0 + off, 0.5, -2.0 + off);
        let r = measure_gaps(&s, lo, Vec3::new(3.0, 5.0, 2.0), cell).expect("samples");
        match r.verdict(2.0) {
            GapVerdict::Fail(GapFailure::Width { width, .. }) => assert!(
                (width - 1.0).abs() <= GAP_TOLERANCE,
                "off {off}: width {width} vs the closed form 1.0"
            ),
            v => panic!("off {off}: pins 1.0 apart against 2.0: {v:?} ({r:?})"),
        }
        assert_certificate(
            r.verdict(0.9),
            cell,
            &format!("off {off}: pins 1.0 apart against 0.9"),
        );
        compared += 1;
    }
    assert!(compared >= 2, "only {compared} offsets compared");
}

/// The same pins at cell 0.5 (any-shape resolution ≈ 1.73): the grid may not resolve a pin of radius
/// 0.3, but it must never answer with nothing. A width, or an `Undetermined` verdict for a limit it
/// cannot speak for, is allowed; a missing gap with a `Pass` against 2.0 is not.
///
/// oracle: the same closed form (1.0 of air); the rule (ruling A, point 1) that a gap between parts
/// smaller than a cell is a width or undetermined, not silence
#[test]
fn pins_smaller_than_a_cell_are_a_width_or_undetermined_never_nothing() {
    let floor = SdfNode::box3d(20.0, 2.0, 20.0).translate(0.0, -1.0, 0.0);
    let pin = |x: f32| SdfNode::cylinder(0.3, 6.0).translate(x, 3.0, 0.0);
    let s = floor.union(pin(-0.8)).union(pin(0.8));
    let mut compared = 0;
    for off in OFFSETS {
        let lo = Vec3::new(-3.0 + off, 0.5, -2.0 + off);
        let r = measure_gaps(&s, lo, Vec3::new(3.0, 5.0, 2.0), 0.5).expect("samples");
        match r.verdict(2.0) {
            GapVerdict::Fail(GapFailure::Width { width, .. }) => assert!(
                (width - 1.0).abs() <= GAP_TOLERANCE + 0.5,
                "off {off}: width {width} vs the closed form 1.0"
            ),
            GapVerdict::Undetermined(_) => {}
            v => panic!("off {off}: pins 1.0 apart at cell 0.5 against 2.0 read as {v:?} ({r:?})"),
        }
        compared += 1;
    }
    assert!(compared >= 2, "only {compared} offsets compared");
}
