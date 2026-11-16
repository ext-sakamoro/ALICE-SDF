//! Gap width (hole / slot) along the SDF gradient, against closed-form scenes
//!
//! `measure_gaps` samples every grid point of pitch `cell` in empty space within the stated
//! resolution of a surface, moves it onto the surface and marches along `+∇d` through the empty
//! space until it re-enters material: that distance is the gap width there. A ray that leaves the
//! bound without re-entering is open air, not a gap. Any empty slab wider than the report's
//! `resolution` (`√3 · cell`) holds a grid point in every orientation, so it is found; narrower
//! gaps may be missed and the report says so through `resolution`.
//!
//! Scenes (Y up): a floor, a wall and a post (cylinder) at a known gap from the wall; two blocks
//! with a slot between them, placed off a 1 mm grid; open air only; and a bound with no surface.
//! Every case asserts it measured at least one sample.

use alice_sdf::eval::eval;
use alice_sdf::validity::{local_gap, measure_gaps, GapError, GAP_TOLERANCE};
use alice_sdf::SdfNode;
use glam::Vec3;

/// Floor (top at y = 0), a wall whose inner face is x = 10, and a post of radius 3 standing on the
/// floor with its surface `gap` from the wall
fn post_and_wall(gap: f32) -> SdfNode {
    let floor = SdfNode::box3d(40.0, 2.0, 20.0).translate(0.0, -1.0, 0.0);
    let wall = SdfNode::box3d(4.0, 10.0, 20.0).translate(12.0, 5.0, 0.0);
    let post = SdfNode::cylinder(3.0, 10.0).translate(10.0 - gap - 3.0, 5.0, 0.0);
    floor.union(wall).union(post)
}

const fn bounds() -> (Vec3, Vec3) {
    (Vec3::new(-21.0, -3.0, -11.0), Vec3::new(21.0, 11.0, 11.0))
}

#[test]
fn a_post_next_to_a_wall_reports_the_gap_between_them() {
    let (lo, hi) = bounds();
    for gap in [1.0_f32, 2.0, 3.0] {
        let r = measure_gaps(&post_and_wall(gap), lo, hi, 0.25).expect("samples");
        assert!(r.samples > 0, "gap {gap}: no sample");
        let min = r
            .min_gap
            .unwrap_or_else(|| panic!("gap {gap}: not found ({r:?})"));
        assert!(
            (min - gap).abs() <= 0.05,
            "gap {gap}: measured {min} ({r:?})"
        );
        let at = r.location.expect("location");
        // the narrowest place is between the post and the wall, at x in (10 - gap, 10)
        assert!(at.x > 10.0 - gap - 0.1 && at.x < 10.1, "gap {gap}: at {at}");
        assert!(
            r.resolution < gap,
            "resolution {} must resolve {gap}",
            r.resolution
        );
    }
}

/// Two blocks with a slot of `width` between them; the slot's centre is at `x0`
fn slot(width: f32, x0: f32) -> SdfNode {
    let half = width * 0.5;
    let left = SdfNode::box3d(10.0, 6.0, 10.0).translate(x0 - half - 5.0, 3.0, 0.0);
    let right = SdfNode::box3d(10.0, 6.0, 10.0).translate(x0 + half + 5.0, 3.0, 0.0);
    left.union(right)
}

#[test]
fn a_slot_narrower_than_a_1_mm_grid_and_off_its_points_is_found() {
    // 0.6 mm wide, centred at x = 0.37: no point of a 1 mm grid lies inside it
    let r = measure_gaps(
        &slot(0.6, 0.37),
        Vec3::new(-12.0, -1.0, -6.0),
        Vec3::new(12.0, 7.0, 6.0),
        0.25,
    )
    .expect("samples");
    let min = r
        .min_gap
        .unwrap_or_else(|| panic!("slot not found ({r:?})"));
    assert!((min - 0.6).abs() <= 0.03, "measured {min}");
    assert!(r.resolution < 0.6);
}

#[test]
fn a_gap_below_the_resolution_is_not_reported_as_passed() {
    // 0.3 mm slot at cell 0.4: the resolution (√3 · 0.4 ≈ 0.69) is above the slot, so the report
    // must say so; when the slot happens to be found its width is right
    let r = measure_gaps(
        &slot(0.3, 0.13),
        Vec3::new(-12.0, -1.0, -6.0),
        Vec3::new(12.0, 7.0, 6.0),
        0.4,
    )
    .expect("samples");
    assert!((r.resolution - 3.0_f32.sqrt() * 0.4).abs() < 1e-5, "{r:?}");
    assert!(
        r.resolution > 0.3,
        "the report must not claim to resolve 0.3 mm"
    );
    if let Some(min) = r.min_gap {
        assert!((min - 0.3).abs() <= 0.03, "measured {min}");
    }
}

#[test]
fn open_air_is_not_a_gap() {
    let block = SdfNode::box3d(10.0, 6.0, 10.0);
    let r = measure_gaps(&block, Vec3::splat(-8.0), Vec3::splat(8.0), 0.25).expect("samples");
    assert!(r.samples > 0);
    assert_eq!(r.min_gap, None, "{r:?}");
}

#[test]
fn a_bound_with_no_surface_compares_nothing_and_is_an_error() {
    let block = SdfNode::box3d(1.0, 1.0, 1.0);
    let r = measure_gaps(&block, Vec3::splat(50.0), Vec3::splat(60.0), 0.5);
    assert!(matches!(r, Err(GapError::NoSamples)), "{r:?}");
}

#[test]
fn a_bad_cell_is_an_error() {
    let block = SdfNode::box3d(1.0, 1.0, 1.0);
    for cell in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
        let r = measure_gaps(&block, Vec3::splat(-2.0), Vec3::splat(2.0), cell);
        assert!(matches!(r, Err(GapError::BadCell)), "{cell}: {r:?}");
    }
}

#[test]
fn local_gap_crosses_a_slab_of_air_exactly() {
    // two half-spaces x < 0 and x > 1.5 (as boxes): from x = 0 along +x the gap is 1.5
    let a = SdfNode::box3d(10.0, 10.0, 10.0).translate(-5.0, 0.0, 0.0);
    let b = SdfNode::box3d(10.0, 10.0, 10.0).translate(6.5, 0.0, 0.0);
    let s = a.union(b);
    let g = local_gap(&s, Vec3::ZERO, Vec3::X, 20.0).expect("re-enters");
    assert!((g - 1.5).abs() < 1e-3, "{g}");
    // toward open air (+y) it never re-enters
    assert_eq!(
        local_gap(&s, Vec3::new(0.75, 5.0, 0.0), Vec3::Y, 20.0),
        None
    );
}

#[test]
fn grid_points_a_hair_outside_a_flat_face_are_not_a_gap() {
    // the block's faces sit 1e-5 off grid points (cell 0.25 from -8): those points are in empty space
    // within the march tolerance of the face; marching away from the face must reach open air
    let block = SdfNode::box3d(10.0, 6.0, 10.0).translate(0.0, 1e-5, 0.0);
    let r = measure_gaps(&block, Vec3::splat(-8.0), Vec3::splat(8.0), 0.25).expect("samples");
    assert!(r.samples > 0);
    assert_eq!(r.min_gap, None, "{r:?}");
    // and a slot whose walls sit a hair off the grid is still its width
    let s = measure_gaps(
        &slot(1.0, 0.5).translate(0.0, 1e-5, 0.0),
        Vec3::new(-12.0, -1.0, -6.0),
        Vec3::new(12.0, 7.0, 6.0),
        0.25,
    )
    .expect("samples");
    let min = s
        .min_gap
        .unwrap_or_else(|| panic!("slot not found ({s:?})"));
    assert!((min - 1.0).abs() <= 0.03, "measured {min}");
}

#[test]
fn local_gap_from_a_point_on_or_a_hair_off_the_face_crosses_the_slab() {
    // two half-spaces x < 0 and x > 1.5 (as boxes); start on the face, a hair outside it and a hair inside
    let a = SdfNode::box3d(10.0, 10.0, 10.0).translate(-5.0, 0.0, 0.0);
    let b = SdfNode::box3d(10.0, 10.0, 10.0).translate(6.5, 0.0, 0.0);
    let s = a.union(b);
    for x in [0.0_f32, 1e-5, -1e-5, 5e-5] {
        let g = local_gap(&s, Vec3::new(x, 1.0, 1.0), Vec3::X, 20.0)
            .unwrap_or_else(|| panic!("from x = {x}: no gap"));
        assert!((g - 1.5).abs() < 1e-3, "from x = {x}: {g}");
    }
}

#[test]
fn exact_slots_and_holes_report_within_the_stated_tolerance() {
    use alice_sdf::validity::GAP_TOLERANCE;
    // slots of exactly g, centred on and off the grid points, and a through hole of diameter g
    for g in [0.8_f32, 1.0, 2.0, 2.5] {
        for x0 in [0.0_f32, 0.37, 0.125] {
            let r = measure_gaps(
                &slot(g, x0),
                Vec3::new(-12.0, -1.0, -6.0),
                Vec3::new(12.0, 7.0, 6.0),
                0.25,
            )
            .expect("samples");
            let w = r
                .min_gap
                .unwrap_or_else(|| panic!("slot {g} at {x0}: not found"));
            assert!(
                (w - g).abs() <= GAP_TOLERANCE,
                "slot {g} at {x0}: {w} (±{GAP_TOLERANCE})"
            );
        }
        let plate = SdfNode::box3d(20.0, 4.0, 20.0).subtract(SdfNode::cylinder(g * 0.5, 10.0));
        let r = measure_gaps(
            &plate,
            Vec3::new(-11.0, -3.0, -11.0),
            Vec3::new(11.0, 3.0, 11.0),
            0.2,
        )
        .expect("samples");
        let w = r.min_gap.unwrap_or_else(|| panic!("hole {g}: not found"));
        assert!(
            (w - g).abs() <= GAP_TOLERANCE,
            "hole {g}: {w} (±{GAP_TOLERANCE})"
        );
    }
    // and local_gap from on the face reads the slab within the same bound
    let a = SdfNode::box3d(10.0, 10.0, 10.0).translate(-5.0, 0.0, 0.0);
    let b = SdfNode::box3d(10.0, 10.0, 10.0).translate(7.0, 0.0, 0.0);
    let w = local_gap(&a.union(b), Vec3::new(0.0, 1.0, 1.0), Vec3::X, 20.0).expect("gap");
    assert!((w - 2.0).abs() <= GAP_TOLERANCE, "slab 2.0: {w}");
}

// ── wedges (walls that meet) vs widths (walls that stay apart) ──
//
// A gap whose two walls converge and meet (an acute joint: a tilted plate on a floor, a cylinder lying on
// a plate) has no width: it closes to a line, and every grid sample near the line would report a
// grid-dependent small value. Those samples are reported as crevices (angle + location), never as a
// width. Walls that approach but stay apart (a sphere over a plate) or that would meet only beyond a
// real floor face (a tapered slot) are widths.

/// Floor (top at y = 0) and a plate hinged on the line x = y = 0, leaving an air wedge of `alpha` degrees
/// on the +x side between the floor's top and the plate's underside
fn hinged(alpha: f32) -> SdfNode {
    let floor = SdfNode::box3d(40.0, 4.0, 20.0).translate(0.0, -2.0, 0.0);
    let plate = SdfNode::box3d(40.0, 4.0, 20.0)
        .translate(20.0, 2.0, 0.0)
        .rotate_euler(0.0, 0.0, alpha.to_radians());
    floor.union(plate)
}

#[test]
fn acute_wedges_are_crevices_with_their_angle_not_widths() {
    for alpha in [20.0_f32, 45.0, 60.0, 80.0] {
        // only x >= 0.5 is sampled: the plate's end face meets the floor on the -x side at 180 - 90 -
        // alpha degrees, a second (real) crevice this test does not look at
        let r = measure_gaps(
            &hinged(alpha),
            Vec3::new(0.5, -1.0, -8.0),
            Vec3::new(25.0, 25.0, 8.0),
            0.25,
        )
        .expect("samples");
        assert_eq!(
            r.min_gap, None,
            "alpha {alpha}: a wedge reported as a width ({r:?})"
        );
        assert!(!r.crevices.is_empty(), "alpha {alpha}: no crevice ({r:?})");
        for c in &r.crevices {
            assert!(
                (c.angle_deg - alpha).abs() <= 2.0,
                "alpha {alpha}: crevice angle {}",
                c.angle_deg
            );
        }
    }
}

#[test]
fn a_right_or_obtuse_corner_is_neither_a_width_nor_a_crevice() {
    for alpha in [90.0_f32, 120.0] {
        // only the +x side (the 120 degree side) is sampled: no acute air there
        let r = measure_gaps(
            &hinged(alpha),
            Vec3::new(1.0, -1.0, -8.0),
            Vec3::new(25.0, 25.0, 8.0),
            0.25,
        )
        .expect("samples");
        assert_eq!(r.min_gap, None, "alpha {alpha}: {r:?}");
        assert!(r.crevices.is_empty(), "alpha {alpha}: {:?}", r.crevices);
    }
}

#[test]
fn a_wedge_whose_walls_meet_outside_the_sampled_region_is_still_a_crevice() {
    // the apex line is at x = 0; only x in [3, 25] is sampled, the walls still meet (inside the part)
    let r = measure_gaps(
        &hinged(30.0),
        Vec3::new(3.0, -1.0, -8.0),
        Vec3::new(25.0, 25.0, 8.0),
        0.25,
    )
    .expect("samples");
    assert_eq!(r.min_gap, None, "{r:?}");
    assert!(!r.crevices.is_empty(), "{r:?}");
}

#[test]
fn a_cylinder_lying_on_a_plate_is_a_crevice_and_a_standing_one_is_no_hole() {
    let plate = SdfNode::box3d(40.0, 4.0, 20.0).translate(0.0, -2.0, 0.0);
    // lying along z, radius 5, touching the plate along the line x = 0, y = 0
    let lying = plate.clone().union(
        SdfNode::cylinder(5.0, 8.0)
            .rotate_euler(std::f32::consts::FRAC_PI_2, 0.0, 0.0)
            .translate(0.0, 5.0, 0.0),
    );
    let r = measure_gaps(
        &lying,
        Vec3::new(-12.0, -1.0, -8.0),
        Vec3::new(12.0, 11.0, 8.0),
        0.25,
    )
    .expect("samples");
    assert_eq!(r.min_gap, None, "{r:?}");
    assert!(!r.crevices.is_empty(), "{r:?}");
    // standing on its end: the walls meet at 90 degrees
    let standing = plate.union(SdfNode::cylinder(5.0, 8.0).translate(0.0, 4.0, 0.0));
    let r = measure_gaps(
        &standing,
        Vec3::new(-12.0, -1.0, -10.0),
        Vec3::new(12.0, 9.0, 10.0),
        0.25,
    )
    .expect("samples");
    assert_eq!(r.min_gap, None, "{r:?}");
}

#[test]
fn a_sphere_held_over_a_plate_is_a_width_not_a_crevice() {
    let plate = SdfNode::box3d(40.0, 4.0, 20.0).translate(0.0, -2.0, 0.0);
    let s = plate.union(SdfNode::sphere(5.0).translate(0.0, 6.0, 0.0)); // 1.0 above the plate
    let r = measure_gaps(
        &s,
        Vec3::new(-12.0, -1.0, -8.0),
        Vec3::new(12.0, 12.0, 8.0),
        0.25,
    )
    .expect("samples");
    let w = r.min_gap.unwrap_or_else(|| panic!("no width ({r:?})"));
    assert!((w - 1.0).abs() <= 0.02, "sphere 1.0 above the plate: {w}");
    assert!(r.crevices.is_empty(), "{:?}", r.crevices);
}

#[test]
fn a_tapered_slot_with_a_floor_is_a_width_not_a_crevice() {
    // a groove 6 mm deep cut into a block (top at y = 0): flanks leaning out 20 degrees from vertical, a
    // flat 0.3 mm floor at y = -6; the flanks would meet only below the floor, inside the material
    let t = 20.0_f32.to_radians();
    // half-spaces as large boxes whose inner face passes through the floor corner (±0.15, -6)
    let right_of_left_flank = SdfNode::box3d(40.0, 40.0, 12.0)
        .translate(20.0, 0.0, 0.0)
        .rotate_euler(0.0, 0.0, t)
        .translate(-0.15, -6.0, 0.0);
    let left_of_right_flank = SdfNode::box3d(40.0, 40.0, 12.0)
        .translate(-20.0, 0.0, 0.0)
        .rotate_euler(0.0, 0.0, -t)
        .translate(0.15, -6.0, 0.0);
    let above_floor = SdfNode::box3d(40.0, 40.0, 12.0).translate(0.0, 14.0, 0.0);
    let air = right_of_left_flank
        .intersection(left_of_right_flank)
        .intersection(above_floor);
    let block = SdfNode::box3d(30.0, 10.0, 10.0).translate(0.0, -5.0, 0.0);
    let s = block.subtract(air);
    // the scene is what it says: empty in the groove and on its floor centre, solid beside the floor
    // corner and below the floor
    assert!(
        eval(&s, Vec3::new(0.0, -5.9, 0.0)) > 0.0,
        "groove floor centre is empty"
    );
    assert!(
        eval(&s, Vec3::new(0.0, -1.0, 0.0)) > 0.0,
        "groove is open near the top"
    );
    assert!(
        eval(&s, Vec3::new(-0.6, -5.9, 0.0)) < 0.0,
        "beside the floor is material"
    );
    assert!(
        eval(&s, Vec3::new(0.0, -6.2, 0.0)) < 0.0,
        "below the floor is material"
    );
    let r = measure_gaps(
        &s,
        Vec3::new(-8.0, -9.0, -6.0),
        Vec3::new(8.0, 1.0, 6.0),
        0.1,
    )
    .expect("samples");
    let w = r
        .min_gap
        .unwrap_or_else(|| panic!("the groove has a width ({r:?})"));
    assert!(
        w >= 0.25,
        "narrowest width {w} (floor 0.3, flanks widen upward)"
    );
    assert!(
        r.crevices.is_empty(),
        "a tapered slot with a floor classified as a crevice: {:?}",
        r.crevices
    );
}

/// Samples on an edge of their own wall are skipped (`edge_samples`); a real gap bounded by an edge on
/// one side must still be reported through the samples on the other side. Closed form: a wall whose face
/// is x = 0, and (a) a square block turned 45 degrees with its edge on the line x = -1, (b) a 0.4 thick fin
/// whose flat tip is the plane x = -1. Both gaps are 1.0 wide. From the wall, a sample at height `δ` off
/// the block's edge meets the block's face at `1 + |δ|` (the faces lean 45 degrees), and `|δ| ≤ cell / 2`;
/// the fin's tip faces the wall, so its gap is exact.
#[test]
fn a_gap_bounded_by_an_edge_on_one_side_is_still_reported() {
    let wall = SdfNode::box3d(4.0, 20.0, 20.0).translate(2.0, 0.0, 0.0);
    let half_diagonal = 2.0 * 2.0_f32.sqrt();
    let block = SdfNode::box3d(4.0, 4.0, 10.0)
        .rotate_euler(0.0, 0.0, std::f32::consts::FRAC_PI_4)
        .translate(-1.0 - half_diagonal, 0.0, 0.0);
    let fin = SdfNode::box3d(6.0, 0.4, 10.0).translate(-4.0, 0.0, 0.0);
    // bounds off the edge line, so no grid row lies exactly on it
    let (lo, hi) = (Vec3::new(-8.0, -5.07, -3.0), Vec3::new(0.5, 5.0, 3.0));
    for cell in [0.25_f32, 0.33, 0.4] {
        let r = measure_gaps(&wall.clone().union(block.clone()), lo, hi, cell).expect("samples");
        let w = r
            .min_gap
            .unwrap_or_else(|| panic!("cell {cell}: edge gap lost ({r:?})"));
        assert!(
            w >= 1.0 - GAP_TOLERANCE && w <= 1.0 + 0.5 * cell + GAP_TOLERANCE,
            "cell {cell}: edge to face measured {w}"
        );
        assert!(
            r.edge_samples > 0,
            "cell {cell}: the block's edge is counted ({r:?})"
        );
        assert!(r.crevices.is_empty(), "cell {cell}: {:?}", r.crevices);
        let r = measure_gaps(&wall.clone().union(fin.clone()), lo, hi, cell).expect("samples");
        let w = r
            .min_gap
            .unwrap_or_else(|| panic!("cell {cell}: fin gap lost ({r:?})"));
        assert!(
            (w - 1.0).abs() <= GAP_TOLERANCE,
            "cell {cell}: fin tip to wall measured {w}"
        );
    }
}

/// Two boxes unioned along a shared face, then subtracted from a block so that the shared face lies on
/// the block's outer face (the shape a T-slot cutter built from a neck box and an outward punch box
/// makes): the union's field is 0 on the seam instead of negative, so `block − cutter` is exactly 0 over
/// the whole opening, a zero-thickness sheet with air on both sides. It is not material: the report
/// counts it as a sheet and reads no width from it (here the distance from the sheet to the slot floor,
/// 2.0). A real wall 0.2 thick has material behind its faces and is not a sheet.
#[test]
fn a_zero_field_sheet_is_reported_and_no_width_is_read_from_it() {
    let block = SdfNode::box3d(20.0, 20.0, 20.0); // faces at ±10
                                                  // neck x in [8, 10] and punch x in [10, 15], both 6 wide in y: the seam x = 10 is the block's face
    let neck = SdfNode::box3d(2.0, 6.0, 30.0).translate(9.0, 0.0, 0.0);
    let punch = SdfNode::box3d(5.0, 6.0, 30.0).translate(12.5, 0.0, 0.0);
    let seamed = block.clone().subtract(neck.union(punch));
    // field across the mouth: zero on the sheet, positive either side
    let on = eval(&seamed, Vec3::new(10.0, 0.0, 0.0));
    assert!(
        on.abs() < 1e-4 && eval(&seamed, Vec3::new(9.9, 0.0, 0.0)) > 0.05,
        "sheet {on}"
    );
    let (lo, hi) = (Vec3::new(5.0, -5.0, -3.0), Vec3::new(12.0, 5.0, 3.0));
    let r = measure_gaps(&seamed, lo, hi, 0.25).expect("samples");
    assert!(r.sheet_samples > 0, "{r:?}");
    let s = r.sheet_location.expect("a sheet point");
    assert!(
        (s.x - 10.0).abs() < 0.01 && s.y.abs() <= 3.01,
        "sheet at {s}"
    );
    if let Some(w) = r.min_gap {
        assert!(w > 2.5, "a width read off the sheet: {w} ({r:?})");
    }
    // the same slot cut by one box (no seam): no sheet
    let one = block.subtract(SdfNode::box3d(7.0, 6.0, 30.0).translate(11.5, 0.0, 0.0));
    let r = measure_gaps(&one, lo, hi, 0.25).expect("samples");
    assert_eq!(r.sheet_samples, 0, "{r:?}");
    // a real thin wall 0.2 thick between two 1.0 slots: material behind, not a sheet; the slots are widths
    let base = SdfNode::box3d(20.0, 2.0, 10.0).translate(0.0, -1.0, 0.0);
    let wall = |x: f32, t: f32| SdfNode::box3d(t, 6.0, 10.0).translate(x, 3.0, 0.0);
    let thin = base
        .union(wall(-1.6, 1.0))
        .union(wall(0.0, 0.2))
        .union(wall(1.6, 1.0));
    let r = measure_gaps(
        &thin,
        Vec3::new(-3.0, -1.0, -3.0),
        Vec3::new(3.0, 7.0, 3.0),
        0.2,
    )
    .expect("samples");
    assert_eq!(r.sheet_samples, 0, "{r:?}");
    let w = r.min_gap.expect("the slots");
    assert!(
        (w - 1.0).abs() <= GAP_TOLERANCE,
        "slot beside the thin wall: {w}"
    );
}
