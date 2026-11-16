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

use alice_sdf::validity::{local_gap, measure_gaps, GapError};
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
    let min = s.min_gap.unwrap_or_else(|| panic!("slot not found ({s:?})"));
    assert!((min - 1.0).abs() <= 0.03, "measured {min}");
}
