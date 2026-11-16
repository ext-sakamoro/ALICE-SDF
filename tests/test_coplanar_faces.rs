//! Coplanar, overlapping faces between the operands of a CSG node, found analytically (no sampling)
//!
//! Two solids that touch along a shared face give a field that is exactly 0 on that face instead of
//! negative (a union seam), and a cutter whose face lies exactly on the material's face leaves the
//! field exactly 0 over the cut's mouth (a flush cut): both are zero-thickness sheets with air on both
//! sides once the result is cut or used. `coplanar_faces` collects the planar faces of each operand
//! (box faces and cylinder caps, through translate, quaternion rotate and uniform scale) and reports
//! every pair across an operator that lies in the same plane (within `COPLANAR_REL_TOL` relative) with
//! overlapping extent:
//!
//! * union: faces with opposite normals (the two solids touch back to back): `UnionSeam`
//! * subtraction `a − b`: a face of `b` with the same normal as a face of `a` (the cutter stops
//!   exactly at the surface): `FlushCut`
//! * intersection: faces with opposite normals (the two solids only touch): `IntersectionContact`
//!
//! Same-normal coplanar faces in a union (two blocks flush side by side) are an ordinary flat exterior
//! and are not reported. Nodes the analysis cannot follow (smooth blends, non-uniform scale, noise,
//! repetition, ...) are counted as not analysed, never silently treated as clean.
//!
//! (`SdfNode::box3d` and `SdfNode::cylinder` take full sizes; the comments give the resulting spans)
//!
//! oracle: closed-form face positions of the scenes (box faces at centre ± half extent, cylinder caps
//! at ± half height) and their overlap areas (rectangle intersection, disk containment).

use alice_sdf::coplanar::{coplanar_faces, CoplanarKind, COPLANAR_REL_TOL};
use alice_sdf::SdfNode;
use glam::{Quat, Vec3};

/// A block with a T-slot cutter built from a neck box (x in [8, 10]) and an outward punch box
/// (x in [10, 15]) unioned along x = 10, the block's +x face
fn t_slot(seamed: bool) -> SdfNode {
    let block = SdfNode::box3d(20.0, 20.0, 20.0); // faces at ±10
    if seamed {
        let neck = SdfNode::box3d(2.0, 6.0, 30.0).translate(9.0, 0.0, 0.0);
        let punch = SdfNode::box3d(5.0, 6.0, 30.0).translate(12.5, 0.0, 0.0);
        block.subtract(neck.union(punch))
    } else {
        // one cutter reaching 5 past the face: no seam, no flush face
        block.subtract(SdfNode::box3d(7.0, 6.0, 30.0).translate(11.5, 0.0, 0.0))
    }
}

#[test]
fn a_cutter_built_from_two_boxes_touching_at_the_face_is_a_union_seam() {
    let r = coplanar_faces(&t_slot(true));
    let seam = r
        .pairs
        .iter()
        .find(|p| p.kind == CoplanarKind::UnionSeam)
        .unwrap_or_else(|| panic!("no seam found: {r:?}"));
    // the plane x = 10, overlap 6 (y) × 30 (z)
    assert!(seam.normal.x.abs() > 0.999, "normal {}", seam.normal);
    assert!((seam.point.x - 10.0).abs() < 1e-4, "point {}", seam.point);
    assert!(
        (seam.overlap_area - 180.0).abs() < 1e-2,
        "area {}",
        seam.overlap_area
    );
    assert!(r.ops_analysed >= 2, "{r:?}");
    assert_eq!(r.ops_not_analysed, 0, "{r:?}");
}

#[test]
fn one_cutter_reaching_past_the_face_reports_nothing() {
    let r = coplanar_faces(&t_slot(false));
    assert!(r.pairs.is_empty(), "{:?}", r.pairs);
    assert!(r.ops_analysed >= 1);
}

#[test]
fn an_open_top_cavity_flush_with_the_rim_is_a_flush_cut() {
    // cup: outer cylinder r 10 y in [-25, 25], cavity r 8 y in [-20, 25] (top flush with the rim)
    let outer = SdfNode::cylinder(10.0, 50.0);
    let cavity = SdfNode::cylinder(8.0, 45.0).translate(0.0, 2.5, 0.0);
    let r = coplanar_faces(&outer.clone().subtract(cavity));
    let cut = r
        .pairs
        .iter()
        .find(|p| p.kind == CoplanarKind::FlushCut)
        .unwrap_or_else(|| panic!("no flush cut: {r:?}"));
    assert!((cut.point.y - 25.0).abs() < 1e-4, "{}", cut.point);
    // the cavity's cap (r 8) inside the rim's cap (r 10): area π · 64, the caps are polygons within 1 %
    let disk = std::f32::consts::PI * 64.0;
    assert!(
        (cut.overlap_area - disk).abs() < 0.01 * disk,
        "area {}",
        cut.overlap_area
    );
    // the cavity reaching 5 past the rim: nothing
    let deep = SdfNode::cylinder(8.0, 50.0).translate(0.0, 5.0, 0.0);
    assert!(coplanar_faces(&outer.subtract(deep)).pairs.is_empty());
}

#[test]
fn a_counterbore_pocket_ending_at_the_plate_face_is_a_flush_cut() {
    // plate 5 thick (y in [-2.5, 2.5]); through hole r 2.1 reaching 5 past both faces; head pocket
    // r 3.5, 3 deep, its top at y = 2.5 (the plate's face)
    let plate = SdfNode::box3d(30.0, 5.0, 30.0);
    let through = SdfNode::cylinder(2.1, 15.0);
    let pocket = SdfNode::cylinder(3.5, 3.0).translate(0.0, 1.0, 0.0);
    let r = coplanar_faces(&plate.clone().subtract(through.clone().union(pocket)));
    let cut = r
        .pairs
        .iter()
        .find(|p| p.kind == CoplanarKind::FlushCut)
        .unwrap_or_else(|| panic!("no flush cut: {r:?}"));
    assert!((cut.point.y - 2.5).abs() < 1e-4, "{}", cut.point);
    // the pocket reaching 5 above the face: nothing
    let pocket_out = SdfNode::cylinder(3.5, 8.0).translate(0.0, 3.5, 0.0);
    assert!(coplanar_faces(&plate.subtract(through.union(pocket_out)))
        .pairs
        .is_empty());
}

#[test]
fn the_coplanarity_tolerance_is_pinned_from_both_sides() {
    // two unit boxes touching at x = 10; the second moved along x by just under / over the tolerance
    let scale = 10.5_f32; // the largest plane offset in the scene
    let pair = |dx: f32| {
        SdfNode::box3d(1.0, 1.0, 1.0)
            .translate(9.5, 0.0, 0.0)
            .union(SdfNode::box3d(1.0, 1.0, 1.0).translate(10.5 + dx, 0.0, 0.0))
    };
    let under = 0.5 * COPLANAR_REL_TOL * scale;
    let over = 4.0 * COPLANAR_REL_TOL * scale;
    assert!(
        !coplanar_faces(&pair(under)).pairs.is_empty(),
        "an offset of {under} (under the tolerance) is a seam"
    );
    assert!(
        coplanar_faces(&pair(over)).pairs.is_empty(),
        "an offset of {over} (over the tolerance) is a gap, not a seam"
    );
    // the same edge in absolute numbers, independent of the constant: at plane offsets near 10,
    // 2e-6 (0.2 ppm) is float rounding of one coincident face and must stay a seam; 1e-4 (10 ppm)
    // is a real gap a printer cannot see but the field can, and must not be called a seam
    assert!(!coplanar_faces(&pair(2e-6)).pairs.is_empty(), "2e-6 is a seam");
    assert!(coplanar_faces(&pair(1e-4)).pairs.is_empty(), "1e-4 is a gap");
}

#[test]
fn a_rotated_touching_pair_is_found_and_a_non_parallel_pair_is_not() {
    let q = Quat::from_rotation_z(37.0_f32.to_radians());
    let a = SdfNode::box3d(2.0, 2.0, 2.0).translate(-1.0, 0.0, 0.0);
    let b = SdfNode::box3d(2.0, 2.0, 2.0).translate(1.0, 0.0, 0.0);
    // both rotated together: they still touch along a (rotated) shared face
    let r = coplanar_faces(&a.clone().union(b).rotate(q));
    let seam = r
        .pairs
        .iter()
        .find(|p| p.kind == CoplanarKind::UnionSeam)
        .unwrap_or_else(|| panic!("rotated seam lost: {r:?}"));
    let n = q * Vec3::X;
    assert!(seam.normal.dot(n).abs() > 0.9999, "normal {}", seam.normal);
    assert!(
        (seam.overlap_area - 4.0).abs() < 1e-3,
        "area {}",
        seam.overlap_area
    );
    // only the second box turned (by 5 degrees about its own centre): the faces are no longer parallel
    let tilted = SdfNode::box3d(2.0, 2.0, 2.0)
        .rotate(Quat::from_rotation_z(5.0_f32.to_radians()))
        .translate(1.0, 0.0, 0.0);
    assert!(coplanar_faces(&a.union(tilted)).pairs.is_empty());
}

#[test]
fn flush_exteriors_and_disjoint_coplanar_faces_are_not_reported() {
    // two blocks side by side with flush tops (same normal, a union): an ordinary flat exterior; the
    // only report is the shared face between them
    let side_by_side = SdfNode::box3d(2.0, 2.0, 2.0)
        .translate(-1.0, 0.0, 0.0)
        .union(SdfNode::box3d(2.0, 2.0, 2.0).translate(1.0, 0.0, 0.0));
    let r = coplanar_faces(&side_by_side);
    assert_eq!(r.pairs.len(), 1, "{:?}", r.pairs);
    assert_eq!(r.pairs[0].kind, CoplanarKind::UnionSeam);
    // facing planes at x = 10 but apart in y (no overlap): nothing
    let apart = SdfNode::box3d(1.0, 1.0, 1.0)
        .translate(9.5, 0.0, 0.0)
        .union(SdfNode::box3d(1.0, 1.0, 1.0).translate(10.5, 5.0, 0.0));
    assert!(coplanar_faces(&apart).pairs.is_empty());
}

#[test]
fn nodes_the_analysis_cannot_follow_are_counted_not_treated_as_clean() {
    let blended = SdfNode::box3d(1.0, 1.0, 1.0).smooth_union(SdfNode::sphere(1.0), 0.2);
    let r = coplanar_faces(&blended.union(SdfNode::box3d(1.0, 1.0, 1.0).translate(5.0, 0.0, 0.0)));
    assert!(r.ops_not_analysed >= 1, "{r:?}");
    assert!(!r.is_fully_analysed());
    // the node the analysis stopped at is named, once (the union above it is not a kind of its own)
    assert_eq!(
        r.not_analysed_kinds.get("SmoothUnion"),
        Some(&1),
        "{:?}",
        r.not_analysed_kinds
    );
    assert_eq!(r.not_analysed_kinds.len(), 1, "{:?}", r.not_analysed_kinds);
    // a shape of primitives and rigid transforms only is fully analysed
    assert!(coplanar_faces(&t_slot(true)).is_fully_analysed());
}

#[test]
fn a_pair_turned_about_a_general_axis_is_found_exactly() {
    // the touching pair of the rotated test, turned about the axis (1, 2, 3) by 71 degrees
    let q = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 71.0_f32.to_radians());
    let a = SdfNode::box3d(2.0, 2.0, 2.0).translate(-1.0, 0.0, 0.0);
    let b = SdfNode::box3d(2.0, 2.0, 2.0).translate(1.0, 0.0, 0.0);
    let r = coplanar_faces(&a.union(b).rotate(q));
    assert!(r.is_fully_analysed(), "{r:?}");
    let seam = r
        .pairs
        .iter()
        .find(|p| p.kind == CoplanarKind::UnionSeam)
        .unwrap_or_else(|| panic!("seam lost under a general rotation: {r:?}"));
    assert!(seam.normal.dot(q * Vec3::X).abs() > 0.9999, "normal {}", seam.normal);
    // the seam's centroid is the rotated origin (the shared face is centred on it)
    assert!(seam.point.length() < 1e-4, "point {}", seam.point);
    assert!((seam.overlap_area - 4.0).abs() < 1e-3, "area {}", seam.overlap_area);
}

#[test]
fn a_non_uniform_scale_is_counted_as_not_analysed() {
    let a = SdfNode::box3d(2.0, 2.0, 2.0).translate(-1.0, 0.0, 0.0);
    let b = SdfNode::box3d(2.0, 2.0, 2.0).translate(1.0, 0.0, 0.0);
    let stretched = a.union(b).scale_xyz(1.0, 2.0, 3.0);
    let r = coplanar_faces(&stretched);
    assert!(r.ops_not_analysed >= 1, "{r:?}");
    assert!(!r.is_fully_analysed());
    // and the same pair under a uniform scale is followed and found
    let a = SdfNode::box3d(2.0, 2.0, 2.0).translate(-1.0, 0.0, 0.0);
    let b = SdfNode::box3d(2.0, 2.0, 2.0).translate(1.0, 0.0, 0.0);
    let r = coplanar_faces(&a.union(b).scale(2.5));
    assert!(r.is_fully_analysed(), "{r:?}");
    let seam = r.pairs.first().expect("seam under a uniform scale");
    assert!((seam.overlap_area - 25.0).abs() < 1e-2, "area {}", seam.overlap_area);
}

#[test]
fn an_empty_primitive_has_no_faces() {
    // a cavity whose height is not positive is an empty shape (its field is positive everywhere): a cup
    // "cut" by it is a solid block with no mouth, and the analysis must not report the empty cutter's caps
    let outer = SdfNode::cylinder(10.0, 50.0);
    for h in [0.0_f32, -5.0] {
        let empty = SdfNode::cylinder(8.0, h).translate(0.0, 25.0, 0.0);
        let r = coplanar_faces(&outer.clone().subtract(empty));
        assert!(r.pairs.is_empty(), "height {h}: {:?}", r.pairs);
        assert!(r.is_fully_analysed(), "height {h}: {r:?}");
    }
    let flat_box = SdfNode::box3d(2.0, 0.0, 2.0).translate(1.0, 0.0, 0.0);
    let r = coplanar_faces(&SdfNode::box3d(2.0, 2.0, 2.0).translate(-1.0, 0.0, 0.0).union(flat_box));
    assert!(r.pairs.is_empty(), "{:?}", r.pairs);
}

#[test]
fn a_solid_resting_on_a_cut_floor_is_a_union_seam() {
    // plate y in [-5, 5]; pocket y in [2, 6] (it reaches past the top, so no flush cut); the pocket's
    // floor y = 2 is the cutter's bottom face, which faces up only once the cutter's faces are reversed;
    // an insert y in [2, 4] standing on that floor touches it back to back
    let plate = SdfNode::box3d(30.0, 10.0, 30.0);
    let pocket = SdfNode::box3d(10.0, 4.0, 10.0).translate(0.0, 4.0, 0.0);
    let insert = SdfNode::box3d(4.0, 2.0, 4.0).translate(0.0, 3.0, 0.0);
    let r = coplanar_faces(&plate.subtract(pocket).union(insert));
    let seam = r
        .pairs
        .iter()
        .find(|p| p.kind == CoplanarKind::UnionSeam)
        .unwrap_or_else(|| panic!("insert on the pocket floor not found: {r:?}"));
    assert!((seam.point.y - 2.0).abs() < 1e-4, "{}", seam.point);
    assert!((seam.overlap_area - 16.0).abs() < 1e-3, "area {}", seam.overlap_area);
}

#[test]
fn rounded_boxes_touching_on_their_flat_faces_are_a_seam_of_the_flat_area() {
    // rounded_box(h, r) has its flat faces at ±(h + r), flat over ±h in the other two axes
    let a = SdfNode::rounded_box(1.0, 1.0, 1.0, 0.5).translate(-1.5, 0.0, 0.0);
    let b = SdfNode::rounded_box(1.0, 1.0, 1.0, 0.5).translate(1.5, 0.0, 0.0);
    let r = coplanar_faces(&a.union(b));
    assert!(r.is_fully_analysed(), "{r:?}");
    let seam = r.pairs.first().unwrap_or_else(|| panic!("{r:?}"));
    assert_eq!(seam.kind, CoplanarKind::UnionSeam);
    assert!(seam.point.x.abs() < 1e-4, "{}", seam.point);
    assert!((seam.overlap_area - 4.0).abs() < 1e-3, "area {}", seam.overlap_area);
}

#[test]
fn a_box_against_a_half_space_is_a_seam_and_a_flush_intersection_is_not() {
    // box x in [-1, 1]; the half-space x > 1 (material where -x + 1 < 0) touches its +x face back to back
    let block = SdfNode::box3d(2.0, 2.0, 2.0);
    let beyond = SdfNode::plane(Vec3::new(-1.0, 0.0, 0.0), -1.0);
    let r = coplanar_faces(&block.clone().union(beyond));
    assert!(r.is_fully_analysed(), "{r:?}");
    let seam = r.pairs.first().unwrap_or_else(|| panic!("{r:?}"));
    assert_eq!(seam.kind, CoplanarKind::UnionSeam);
    assert!((seam.overlap_area - 4.0).abs() < 1e-3, "area {}", seam.overlap_area);
    // the half-space x < 1 cut to the box: same plane, same normal, an ordinary flat face
    let within = SdfNode::plane(Vec3::X, 1.0);
    assert!(coplanar_faces(&block.intersection(within)).pairs.is_empty());
}

#[test]
fn an_elongated_box_is_followed_and_other_elongated_children_are_counted() {
    // box3d(2,2,2) elongated by 1 along x is a box of half extents (2, 1, 1): x in [-4, 0] after the move
    let long = SdfNode::box3d(2.0, 2.0, 2.0).elongate(1.0, 0.0, 0.0).translate(-2.0, 0.0, 0.0);
    let other = SdfNode::box3d(2.0, 2.0, 2.0).translate(1.0, 0.0, 0.0);
    let r = coplanar_faces(&long.union(other));
    assert!(r.is_fully_analysed(), "{r:?}");
    let seam = r.pairs.first().unwrap_or_else(|| panic!("{r:?}"));
    assert!(seam.point.x.abs() < 1e-4, "{}", seam.point);
    assert!((seam.overlap_area - 4.0).abs() < 1e-3, "area {}", seam.overlap_area);
    // an elongated sphere has no planar face to follow exactly through the stretch: counted
    let r = coplanar_faces(&SdfNode::sphere(1.0).elongate(1.0, 0.0, 0.0));
    assert_eq!(r.not_analysed_kinds.get("Elongate"), Some(&1), "{r:?}");
}
