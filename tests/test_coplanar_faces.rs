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

use alice_sdf::coplanar::{coplanar_faces, CoplanarKind, COPLANAR_REL_TOL, SMOOTH_OPENS_K};
use alice_sdf::SdfNode;
use glam::{Quat, Vec3};

/// No coincidence at all: neither a defect nor a candidate the field showed to be a surface (the
/// analysis-level claims: the tolerance, parallelism, overlap, empty shapes)
const fn no_candidate(r: &alice_sdf::coplanar::CoplanarReport) -> bool {
    r.pairs.is_empty() && r.surfaces.is_empty()
}

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
    // the overlap polygon is the 6 x 30 rectangle on x = 10: its own area and plane agree with the pair
    let poly = &seam.overlap_polygon;
    assert!(poly.len() >= 4, "{poly:?}");
    assert!(poly.iter().all(|v| (v.x - 10.0).abs() < 1e-4), "{poly:?}");
    let ys = poly
        .iter()
        .map(|v| v.y)
        .fold((f32::MAX, f32::MIN), |(a, b), y| (a.min(y), b.max(y)));
    let zs = poly
        .iter()
        .map(|v| v.z)
        .fold((f32::MAX, f32::MIN), |(a, b), z| (a.min(z), b.max(z)));
    assert!(
        (ys.0 + 3.0).abs() < 1e-4 && (ys.1 - 3.0).abs() < 1e-4,
        "y span {ys:?}"
    );
    assert!(
        (zs.0 + 15.0).abs() < 1e-4 && (zs.1 - 15.0).abs() < 1e-4,
        "z span {zs:?}"
    );
    assert!(r.ops_analysed >= 2, "{r:?}");
    assert_eq!(r.ops_not_analysed, 0, "{r:?}");
}

#[test]
fn one_cutter_reaching_past_the_face_reports_nothing() {
    let r = coplanar_faces(&t_slot(false));
    assert!(r.pairs.is_empty() && r.surfaces.is_empty(), "{:?}", r.pairs);
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
    assert!(no_candidate(&coplanar_faces(&outer.subtract(deep))));
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
        !no_candidate(&coplanar_faces(&pair(under))),
        "an offset of {under} (under the tolerance) is a seam"
    );
    assert!(
        no_candidate(&coplanar_faces(&pair(over))),
        "an offset of {over} (over the tolerance) is a gap, not a seam"
    );
    // the same edge in absolute numbers, independent of the constant: at plane offsets near 10,
    // 2e-6 (0.2 ppm) is float rounding of one coincident face and must stay a seam; 1e-4 (10 ppm)
    // is a real gap a printer cannot see but the field can, and must not be called a seam
    assert!(
        !no_candidate(&coplanar_faces(&pair(2e-6))),
        "2e-6 is a seam"
    );
    assert!(no_candidate(&coplanar_faces(&pair(1e-4))), "1e-4 is a gap");
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
    assert!(no_candidate(&coplanar_faces(&a.union(tilted))));
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
    assert!(no_candidate(&coplanar_faces(&apart)));
}

#[test]
fn nodes_the_analysis_cannot_follow_are_counted_not_treated_as_clean() {
    // a twist reshapes its child's faces (no plane stays a plane): not followed
    let blended = SdfNode::box3d(1.0, 1.0, 1.0).twist(0.5);
    let r = coplanar_faces(&blended.union(SdfNode::box3d(1.0, 1.0, 1.0).translate(5.0, 0.0, 0.0)));
    assert!(r.ops_not_analysed >= 1, "{r:?}");
    assert!(!r.is_fully_analysed());
    // exactly: the twist itself, and the union above it (one operand not followed); nothing is
    // counted as analysed (an operator with an operand that was not followed is not analysed either)
    assert_eq!((r.ops_analysed, r.ops_not_analysed), (0, 2), "{r:?}");
    // the node the analysis stopped at is named, once (the union above it is not a kind of its own)
    assert_eq!(
        r.not_analysed_kinds.get("Twist"),
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
    assert!(
        seam.normal.dot(q * Vec3::X).abs() > 0.9999,
        "normal {}",
        seam.normal
    );
    // the seam's centroid is the rotated origin (the shared face is centred on it)
    assert!(seam.point.length() < 1e-4, "point {}", seam.point);
    assert!(
        (seam.overlap_area - 4.0).abs() < 1e-3,
        "area {}",
        seam.overlap_area
    );
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
    assert!(
        (seam.overlap_area - 25.0).abs() < 1e-2,
        "area {}",
        seam.overlap_area
    );
}

#[test]
fn an_empty_primitive_has_no_faces() {
    // a cavity whose height is not positive is an empty shape (its field is positive everywhere): a cup
    // "cut" by it is a solid block with no mouth, and the analysis must not report the empty cutter's caps
    let outer = SdfNode::cylinder(10.0, 50.0);
    for h in [0.0_f32, -5.0] {
        let empty = SdfNode::cylinder(8.0, h).translate(0.0, 25.0, 0.0);
        let r = coplanar_faces(&outer.clone().subtract(empty));
        assert!(
            r.pairs.is_empty() && r.surfaces.is_empty(),
            "height {h}: no candidate either: {:?} {:?}",
            r.pairs,
            r.surfaces
        );
        assert!(r.is_fully_analysed(), "height {h}: {r:?}");
    }
    let flat_box = SdfNode::box3d(2.0, 0.0, 2.0).translate(1.0, 0.0, 0.0);
    let r = coplanar_faces(
        &SdfNode::box3d(2.0, 2.0, 2.0)
            .translate(-1.0, 0.0, 0.0)
            .union(flat_box),
    );
    assert!(
        r.pairs.is_empty() && r.surfaces.is_empty(),
        "no candidate either: {:?} {:?}",
        r.pairs,
        r.surfaces
    );
    // a box of zero height lying exactly on another box's top face (y = 1): empty, so no seam with it
    let on_top = SdfNode::box3d(2.0, 0.0, 2.0).translate(0.0, 1.0, 0.0);
    let r = coplanar_faces(&SdfNode::box3d(2.0, 2.0, 2.0).union(on_top));
    assert!(
        r.pairs.is_empty() && r.surfaces.is_empty(),
        "an empty box on a face: {:?}",
        r.pairs
    );
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
    assert!(
        (seam.overlap_area - 16.0).abs() < 1e-3,
        "area {}",
        seam.overlap_area
    );
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
    assert!(
        (seam.overlap_area - 4.0).abs() < 1e-3,
        "area {}",
        seam.overlap_area
    );
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
    assert!(
        (seam.overlap_area - 4.0).abs() < 1e-3,
        "area {}",
        seam.overlap_area
    );
    // on the box's +x face, not its -x face: the half-space's face is where its field is zero
    assert!((seam.point.x - 1.0).abs() < 1e-4, "seam at {}", seam.point);
    assert_eq!(r.pairs.len(), 1, "{:?}", r.pairs);
    // the half-space x < 1 cut to the box: same plane, same normal, an ordinary flat face
    let within = SdfNode::plane(Vec3::X, 1.0);
    assert!(no_candidate(&coplanar_faces(&block.intersection(within))));
}

#[test]
fn an_elongated_box_is_followed_and_other_elongated_children_are_counted() {
    // box3d(2,2,2) elongated by 1 along x is a box of half extents (2, 1, 1): x in [-4, 0] after the move
    let long = SdfNode::box3d(2.0, 2.0, 2.0)
        .elongate(1.0, 0.0, 0.0)
        .translate(-2.0, 0.0, 0.0);
    let other = SdfNode::box3d(2.0, 2.0, 2.0).translate(1.0, 0.0, 0.0);
    let r = coplanar_faces(&long.union(other));
    assert!(r.is_fully_analysed(), "{r:?}");
    let seam = r.pairs.first().unwrap_or_else(|| panic!("{r:?}"));
    assert!(seam.point.x.abs() < 1e-4, "{}", seam.point);
    assert!(
        (seam.overlap_area - 4.0).abs() < 1e-3,
        "area {}",
        seam.overlap_area
    );
    // an elongated sphere has no planar face to follow exactly through the stretch: counted
    let r = coplanar_faces(&SdfNode::sphere(1.0).elongate(1.0, 0.0, 0.0));
    assert_eq!(r.not_analysed_kinds.get("Elongate"), Some(&1), "{r:?}");
}

#[test]
fn faces_meeting_only_along_an_edge_do_not_overlap() {
    // box A x in [-1, 0], y in [0, 1]; box B x in [0, 1], y in [1, 2]: their faces on x = 0 face each
    // other in the same plane but share only the edge y = 1 (no area), so they do not touch along a face
    let a = SdfNode::box3d(1.0, 1.0, 1.0).translate(-0.5, 0.5, 0.0);
    let b = SdfNode::box3d(1.0, 1.0, 1.0).translate(0.5, 1.5, 0.0);
    let r = coplanar_faces(&a.union(b));
    assert!(r.is_fully_analysed(), "{r:?}");
    assert!(
        r.pairs.is_empty() && r.surfaces.is_empty(),
        "no candidate either: {:?} {:?}",
        r.pairs,
        r.surfaces
    );
}

#[test]
fn a_smooth_operator_opens_a_coincidence_only_above_the_surface_band() {
    // oracle: the polynomial blend moves the field at the coincident face by k / 4 (measured), which
    // clears the widest default surface band (SURFACE_HIT_EPS = 1e-3) only for k > 4e-3
    assert!((SMOOTH_OPENS_K - 4e-3).abs() < 1e-9, "{SMOOTH_OPENS_K}");
    let a = || SdfNode::box3d(2.0, 2.0, 2.0).translate(-1.0, 0.0, 0.0);
    let b = || SdfNode::box3d(2.0, 2.0, 2.0).translate(1.0, 0.0, 0.0);
    let block = || SdfNode::box3d(4.0, 4.0, 4.0);
    let cutter = || SdfNode::box3d(1.0, 1.0, 6.0).translate(1.5, 0.0, 0.0); // +x face on the block's
    let at = SMOOTH_OPENS_K;
    let above = SMOOTH_OPENS_K.next_up();
    for (name, k, want_pair) in [
        ("at", at, true),
        ("above", above, false),
        ("zero", 0.0, true),
    ] {
        let u = coplanar_faces(&a().smooth_union(b(), k));
        let s = coplanar_faces(&block().smooth_subtract(cutter(), k));
        let i = coplanar_faces(&a().smooth_intersection(b(), k));
        for (op, r, kind) in [
            ("union", &u, CoplanarKind::UnionSeam),
            ("subtract", &s, CoplanarKind::FlushCut),
            ("intersect", &i, CoplanarKind::IntersectionContact),
        ] {
            assert!(r.is_fully_analysed(), "{op} k {name}: {r:?}");
            let found = r.pairs.iter().any(|p| p.kind == kind);
            assert_eq!(found, want_pair, "{op} k {name} ({k:e}): {:?}", r.pairs);
        }
    }
    // a smooth union's faces still pass up: a box resting on the blended pair's top is a seam
    let top = SdfNode::box3d(1.0, 1.0, 1.0).translate(-1.0, 1.5, 0.0);
    let r = coplanar_faces(&a().smooth_union(b(), 0.5).union(top));
    assert!(
        r.pairs.iter().any(|p| p.kind == CoplanarKind::UnionSeam),
        "{:?}",
        r.pairs
    );
}

#[test]
fn every_overlap_polygon_is_strictly_convex_without_repeated_or_collinear_vertices() {
    // scenes whose clipping meets vertices exactly on clip lines (equal widths, shared corners)
    let scenes = [
        t_slot(true),
        SdfNode::box3d(2.0, 2.0, 2.0)
            .translate(-1.0, 0.0, 0.0)
            .union(SdfNode::box3d(2.0, 2.0, 2.0).translate(1.0, 0.0, 0.0)),
        SdfNode::box3d(2.0, 2.0, 2.0)
            .translate(-1.0, 0.0, 0.0)
            .union(SdfNode::box3d(2.0, 1.0, 2.0).translate(1.0, 0.5, 0.0)),
        SdfNode::cylinder(10.0, 50.0)
            .subtract(SdfNode::cylinder(8.0, 45.0).translate(0.0, 2.5, 0.0)),
    ];
    let mut checked = 0;
    for (si, node) in scenes.iter().enumerate() {
        for p in coplanar_faces(node).pairs {
            let v = &p.overlap_polygon;
            assert!(v.len() >= 3, "scene {si}: {v:?}");
            for i in 0..v.len() {
                let (a, b, c) = (v[i], v[(i + 1) % v.len()], v[(i + 2) % v.len()]);
                assert!(
                    a.distance(b) > 1e-6,
                    "scene {si}: repeated vertex {a} in {v:?}"
                );
                // strictly convex: every consecutive triple turns the same way, never straight
                let turn = (b - a).cross(c - b).dot(p.normal);
                assert!(
                    turn > 1e-9,
                    "scene {si}: vertex {b} is not a strict left turn ({turn:e}) in {v:?}"
                );
            }
            checked += 1;
        }
    }
    assert!(checked >= 4, "only {checked} polygons checked");
}

// ── field classification of the candidate pairs (the analysis finds candidates from the tree; whether
// a candidate is a zero-thickness sheet is read from the final field over its overlap) ──

use alice_sdf::coplanar::SheetSide;

/// A base slab (x in [-20, 20], y in [-2, 2], z in [-10, 10]) and a rib standing on it (x in [-5, 5],
/// y in [2, 8], full depth), optionally with the base cut away under x in [`cut_from`, 6]
fn rib_on_base(cut_from: Option<f32>) -> SdfNode {
    let base = SdfNode::box3d(40.0, 4.0, 20.0);
    let base = match cut_from {
        Some(x0) => base.subtract(SdfNode::box3d(6.0 - x0, 10.0, 30.0).translate(
            f32::midpoint(6.0, x0),
            0.0,
            0.0,
        )),
        None => base,
    };
    base.union(SdfNode::box3d(10.0, 6.0, 20.0).translate(0.0, 5.0, 0.0))
}

#[test]
fn a_rib_on_a_base_is_a_material_sheet_over_the_rib_footprint() {
    let r = coplanar_faces(&rib_on_base(None));
    assert_eq!(r.pairs.len(), 1, "{:?}", r.pairs);
    assert!(r.surfaces.is_empty(), "{:?}", r.surfaces);
    let p = &r.pairs[0];
    assert_eq!(p.kind, CoplanarKind::UnionSeam);
    assert_eq!(p.sheets.len(), 1, "{:?}", p.sheets);
    let s = &p.sheets[0];
    assert_eq!(s.side, SheetSide::Material);
    // the footprint x in [-5, 5], z in [-10, 10], less the edge band (2δ, δ = 1e-3 × 40) and a grid step
    for (got, want) in [
        (s.min.x, -5.0),
        (s.max.x, 5.0),
        (s.min.z, -10.0),
        (s.max.z, 10.0),
    ] {
        assert!((got - want).abs() < 0.6, "sheet extent {s:?}");
    }
    assert!(
        (s.min.y - 2.0).abs() < 1e-5 && (s.max.y - 2.0).abs() < 1e-5,
        "on the plane y = 2: {s:?}"
    );
    assert!(r.field_evaluations > 0, "the classification read the field");
}

#[test]
fn a_rib_half_over_a_cut_is_a_sheet_over_the_remaining_half_only() {
    // the base is cut away under x in [0, 6]: over x > 0 the rib's underside is a real surface (air
    // below), over x < 0 the seam is still a sheet; the pair is reported with the sheet's extent only
    let r = coplanar_faces(&rib_on_base(Some(0.0)));
    let seams: Vec<_> = r
        .pairs
        .iter()
        .filter(|p| p.kind == CoplanarKind::UnionSeam)
        .collect();
    assert_eq!(seams.len(), 1, "{:?}", r.pairs);
    let s = &seams[0].sheets;
    assert_eq!(s.len(), 1, "{s:?}");
    assert_eq!(s[0].side, SheetSide::Material);
    assert!((s[0].min.x + 5.0).abs() < 0.6, "{s:?}");
    assert!(
        s[0].max.x <= 0.0 && s[0].max.x > -0.6,
        "the sheet stops where the base does: {s:?}"
    );
}

#[test]
fn a_rib_wholly_over_a_cut_is_a_surface_not_a_sheet() {
    // the base is cut away under the whole rib: the coincidence the tree shows is a real surface
    let r = coplanar_faces(&rib_on_base(Some(-6.0)));
    assert!(
        r.pairs.iter().all(|p| p.kind != CoplanarKind::UnionSeam),
        "{:?}",
        r.pairs
    );
    assert_eq!(
        r.surfaces
            .iter()
            .filter(|p| p.kind == CoplanarKind::UnionSeam)
            .count(),
        1,
        "{:?}",
        r.surfaces
    );
    assert!(r.surfaces.iter().all(|p| p.sheets.is_empty()));
}

#[test]
fn the_classification_reads_the_final_field_through_a_node_the_analysis_cannot_follow() {
    // a block filling around half the seam (x in [0, 8], y in [0, 4]) under a twist of strength 0: the
    // analysis does not follow the twist (and does not visit below it), so the block's faces are unknown
    // and the seam is still a candidate; the classification reads the final field, twist included, and
    // finds the sheet only over the half the block does not fill
    let filler = SdfNode::box3d(8.0, 4.0, 30.0)
        .translate(4.0, 2.0, 0.0)
        .twist(0.0);
    let r = coplanar_faces(&rib_on_base(None).union(filler));
    assert!(!r.is_fully_analysed());
    let seams: Vec<_> = r
        .pairs
        .iter()
        .filter(|p| p.kind == CoplanarKind::UnionSeam)
        .collect();
    assert_eq!(seams.len(), 1, "{:?}", r.pairs);
    let s = &seams[0].sheets;
    assert_eq!(s.len(), 1, "{s:?}");
    assert_eq!(s[0].side, SheetSide::Material);
    // samples within δ (1e-3 × 40) of the block's face read 0 within δ and count as sheet: the edge
    // is at x = 0 to within δ and one step
    assert!(
        (s[0].min.x + 5.0).abs() < 0.6 && s[0].max.x < 0.15,
        "only the uncovered half: {s:?}"
    );
}

#[test]
fn a_cutter_mouth_flush_with_the_rim_is_an_air_sheet() {
    // an open-top cavity whose cutter stops exactly at the rim: a membrane of air over the mouth
    let cup = SdfNode::box3d(20.0, 20.0, 20.0)
        .subtract(SdfNode::box3d(16.0, 18.0, 16.0).translate(0.0, 1.0, 0.0));
    let r = coplanar_faces(&cup);
    assert_eq!(r.pairs.len(), 1, "{:?}", r.pairs);
    assert_eq!(r.pairs[0].kind, CoplanarKind::FlushCut);
    assert_eq!(
        r.pairs[0].sheets.iter().map(|s| s.side).collect::<Vec<_>>(),
        vec![SheetSide::Air]
    );
}

#[test]
fn a_filled_seam_beside_a_bore_is_not_a_sheet() {
    // a rib on a base whose seam is filled by a block reaching into both, with a bore (r = 1) through
    // the rib and the base: the rib's underside and the base's top still coincide (a candidate), but
    // the field at the plane is negative everywhere except next to the bore's wall, where it is near 0
    // because the bore's wall crosses the plane, not because of a sheet. A sample there reads 0 within
    // δ with material on both sides; a sheet is told apart by the field moving away from its value on
    // the plane by about δ on both sides (a V across the plane), which a wall crossing the plane does not
    let filled = rib_on_base(None).union(SdfNode::box3d(10.0, 4.0, 20.0).translate(0.0, 2.0, 0.0));
    let bored = filled.subtract(SdfNode::cylinder(1.0, 30.0));
    let r = coplanar_faces(&bored);
    assert!(
        r.pairs.iter().all(|p| p.kind != CoplanarKind::UnionSeam),
        "no sheet: {:?}",
        r.pairs
    );
    assert!(
        r.surfaces.iter().any(|p| p.kind == CoplanarKind::UnionSeam),
        "the candidate is a surface: {:?}",
        r.surfaces
    );
}

#[test]
fn one_pair_can_hold_an_air_sheet_and_a_material_sheet_side_by_side() {
    // a cutter built from two boxes touching at x = 0 (A: x in [-5, 0], B: x in [0, 5], y and z in
    // +-10) cuts a block (x in +-8, y in [0, 8], z in +-8): inside the cut, its seam is a membrane of air;
    // below the block, two other boxes touching at the same plane (E, F: y in [-9, 0]) make the same
    // plane a seam inside material. The cutter's pair covers both: two regions, one per side, adjacent
    // along y = 0 (they must not be merged into one)
    let cutter = SdfNode::box3d(5.0, 20.0, 20.0)
        .translate(-2.5, 0.0, 0.0)
        .union(SdfNode::box3d(5.0, 20.0, 20.0).translate(2.5, 0.0, 0.0));
    let block = SdfNode::box3d(16.0, 8.0, 16.0).translate(0.0, 4.0, 0.0);
    let below = SdfNode::box3d(4.0, 9.0, 16.0)
        .translate(-2.0, -4.5, 0.0)
        .union(SdfNode::box3d(4.0, 9.0, 16.0).translate(2.0, -4.5, 0.0));
    let part = block.subtract(cutter).union(below);
    let r = coplanar_faces(&part);
    let big = r
        .pairs
        .iter()
        .find(|p| p.normal.x.abs() > 0.99 && p.overlap_area > 399.0)
        .unwrap_or_else(|| panic!("the cutter's seam: {:?}", r.pairs));
    let mut sides: Vec<SheetSide> = big.sheets.iter().map(|s| s.side).collect();
    sides.sort_by_key(|s| matches!(s, SheetSide::Air));
    assert_eq!(
        sides,
        vec![SheetSide::Material, SheetSide::Air],
        "{:?}",
        big.sheets
    );
}

#[test]
fn a_pair_too_thin_to_classify_stays_a_defect_unclassified() {
    // a rib standing on a base's edge over a strip 1e-4 wide: δ would be a quarter of the strip's half
    // width, far below the floor (1000 ulp at the part's extent), so the sheet cannot be told from
    // rounding; the pair stays in `pairs` with no sheet regions and no samples taken
    let base = SdfNode::box3d(40.0, 4.0, 20.0);
    let rib = SdfNode::box3d(10.0, 6.0, 20.0).translate(20.0 + 5.0 - 1e-4, 5.0, 0.0);
    let r = coplanar_faces(&base.union(rib));
    let seam = r
        .pairs
        .iter()
        .find(|p| p.kind == CoplanarKind::UnionSeam)
        .unwrap_or_else(|| {
            panic!(
                "kept as a defect: {:?} / surfaces {:?}",
                r.pairs, r.surfaces
            )
        });
    assert!(
        seam.sheets.is_empty() && seam.sample_step == 0.0,
        "{seam:?}"
    );
    assert!(seam.overlap_area < 1e-2, "{seam:?}");
}

#[test]
fn an_overlap_counts_only_above_its_relative_area_tolerance() {
    // two 10 x 10 x 10 boxes touching at x = 0, the second shifted along y so their faces overlap in a
    // strip of width w (area 10 w against the faces' 100): the overlap counts when its area is above
    // COPLANAR_REL_TOL times the smaller face's area, i.e. w > 1e-5 (a strip at a tenth of that is not
    // an overlap, one at ten times it is)
    let pair = |w: f32| {
        SdfNode::box3d(10.0, 10.0, 10.0)
            .translate(-5.0, 0.0, 0.0)
            .union(SdfNode::box3d(10.0, 10.0, 10.0).translate(5.0, 10.0 - w, 0.0))
    };
    assert!(
        no_candidate(&coplanar_faces(&pair(5e-6))),
        "a strip 5e-6 wide (area 5e-5, below 1e-4)"
    );
    assert!(
        !no_candidate(&coplanar_faces(&pair(1e-3))),
        "a strip 1e-3 wide (area 1e-2, above 1e-4)"
    );
}

#[test]
fn sheets_of_opposite_sides_stay_apart_when_the_step_is_coarser_than_the_surface_between_them() {
    // a cutter with huge faces (A | B touching at x = 0, y and z in +-2000) makes the overlap's grid step
    // about 15.6 mm, while the part is about 100 mm across (δ about 0.1): the real surface at y = 0 that
    // separates the air sheet (the cutter's seam inside a block, y in [0, 40]) from the material sheet
    // (two boxes touching at x = 0, y in [-40, 0]) is far thinner than a step, so samples of the two sides
    // are grid neighbours. They are still two regions, one per side
    let cutter = SdfNode::box3d(5.0, 4000.0, 4000.0)
        .translate(-2.5, 0.0, 0.0)
        .union(SdfNode::box3d(5.0, 4000.0, 4000.0).translate(2.5, 0.0, 0.0));
    // the boundary between the two sheets at y = 7.3, between two grid rows (the grid starts at the
    // polygon's corner y = -2000 with a step of 15.625, so it has a row at y = 0 itself)
    let lift = 7.3;
    let block = SdfNode::box3d(100.0, 40.0, 80.0).translate(0.0, 20.0 + lift, 0.0);
    let below = SdfNode::box3d(50.0, 40.0, 80.0)
        .translate(-25.0, -20.0 + lift, 0.0)
        .union(SdfNode::box3d(50.0, 40.0, 80.0).translate(25.0, -20.0 + lift, 0.0));
    let r = coplanar_faces(&block.subtract(cutter).union(below));
    let big = r
        .pairs
        .iter()
        .find(|p| p.normal.x.abs() > 0.99 && p.overlap_area > 1.0e6)
        .unwrap_or_else(|| panic!("the cutter's seam: {:?}", r.pairs));
    assert!(
        big.sample_step > 10.0,
        "the step must be coarse here: {}",
        big.sample_step
    );
    let mut sides: Vec<SheetSide> = big.sheets.iter().map(|s| s.side).collect();
    sides.sort_by_key(|s| matches!(s, SheetSide::Air));
    assert_eq!(
        sides,
        vec![SheetSide::Material, SheetSide::Air],
        "{:?}",
        big.sheets
    );
}

// ── following repeated instances ──

#[test]
fn bolt_holes_repeated_around_an_axis_that_end_on_the_plate_faces_are_flush_cuts() {
    // a flange disc (r 30, y in [-3, 3]) less 4 bolt holes (r 2, x = 20) repeated around Y, the holes
    // exactly as long as the plate is thick: each hole's ends lie on the plate's faces, a membrane of
    // air over both ends of every hole
    let holes = SdfNode::cylinder(2.0, 6.0)
        .translate(20.0, 0.0, 0.0)
        .polar_repeat(4);
    let r = coplanar_faces(&SdfNode::cylinder(30.0, 6.0).subtract(holes));
    let flush: Vec<_> = r
        .pairs
        .iter()
        .filter(|p| p.kind == CoplanarKind::FlushCut)
        .collect();
    // 4 instances x 2 ends
    assert_eq!(flush.len(), 8, "{:?}", r.pairs);
    assert!(flush
        .iter()
        .all(|p| p.sheets.iter().all(|s| s.side == SheetSide::Air) && !p.sheets.is_empty()));
    // the instances sit at 0, 90, 180, 270 degrees around Y: one hole's end centred at x = -20
    assert!(
        flush
            .iter()
            .any(|p| (p.point - Vec3::new(-20.0, 3.0, 0.0)).length() < 0.1),
        "{flush:?}"
    );
    assert!(r.is_fully_analysed(), "{r:?}");
}

#[test]
fn two_instances_meeting_on_their_sector_boundary_are_a_seam() {
    // count 2: each sector is a half space around +x / -x, the boundary the plane x = 0; a box with its
    // face on that plane meets its own instance turned half a turn face to face
    let half = SdfNode::box3d(10.0, 4.0, 4.0)
        .translate(5.0, 0.0, 0.0)
        .polar_repeat(2);
    let r = coplanar_faces(&half);
    assert!(
        r.pairs
            .iter()
            .any(|p| p.kind == CoplanarKind::UnionSeam && p.normal.x.abs() > 0.99),
        "{:?} / surfaces {:?}",
        r.pairs,
        r.surfaces
    );
}

#[test]
fn bolt_holes_reaching_past_the_plate_are_not_reported() {
    let holes = SdfNode::cylinder(2.0, 16.0)
        .translate(20.0, 0.0, 0.0)
        .polar_repeat(4);
    let r = coplanar_faces(&SdfNode::cylinder(30.0, 6.0).subtract(holes));
    assert!(no_candidate(&r), "{:?} / {:?}", r.pairs, r.surfaces);
    assert!(r.is_fully_analysed(), "{r:?}");
}

#[test]
fn copies_of_a_finite_repeat_that_touch_are_seams() {
    // three 10 x 10 x 10 boxes 10 apart along X (count 2 or 3 gives copies at -1, 0, 1): each pair of
    // neighbours touches face to face, two seams
    let row = SdfNode::box3d(10.0, 10.0, 10.0).repeat_finite([3, 0, 0], Vec3::new(10.0, 1.0, 1.0));
    let r = coplanar_faces(&row);
    let seams: Vec<_> = r
        .pairs
        .iter()
        .filter(|p| p.kind == CoplanarKind::UnionSeam)
        .collect();
    assert_eq!(seams.len(), 2, "{:?} / surfaces {:?}", r.pairs, r.surfaces);
    let mut xs: Vec<f32> = seams.iter().map(|p| p.point.x).collect();
    xs.sort_by(f32::total_cmp);
    assert!(
        (xs[0] + 5.0).abs() < 1e-3 && (xs[1] - 5.0).abs() < 1e-3,
        "{xs:?}"
    );
    assert!(r.is_fully_analysed(), "{r:?}");
}

#[test]
fn slots_repeated_along_a_plate_that_end_on_its_faces_are_flush_cuts() {
    // a plate (y in [-2, 2]) less 3 slots 4 deep (exactly the plate's thickness) repeated 20 apart
    let slots = SdfNode::box3d(4.0, 4.0, 8.0).repeat_finite([3, 0, 0], Vec3::new(20.0, 1.0, 1.0));
    let r = coplanar_faces(&SdfNode::box3d(80.0, 4.0, 30.0).subtract(slots));
    let flush = r
        .pairs
        .iter()
        .filter(|p| p.kind == CoplanarKind::FlushCut)
        .count();
    assert_eq!(flush, 6, "3 slots x 2 ends: {:?}", r.pairs);
    assert!(r.is_fully_analysed(), "{r:?}");
}

#[test]
fn spaced_copies_of_a_finite_repeat_are_not_reported() {
    let row = SdfNode::box3d(8.0, 10.0, 10.0).repeat_finite([3, 0, 0], Vec3::new(10.0, 1.0, 1.0));
    let r = coplanar_faces(&row);
    assert!(no_candidate(&r) && r.is_fully_analysed(), "{r:?}");
}

// ── HexPrism: hexagon in XY whose flat sides lie at distance `hex_radius` from the axis (the field's
// apothem: `sdf_hex_prism` compares |y| with hex_radius), prism along Z, caps at z = ±half_height ──

#[test]
fn a_hex_prism_standing_on_a_block_is_a_seam_over_its_whole_cap() {
    let (a, h) = (5.0_f32, 4.0_f32);
    // the field's own geometry: the flat side at y = a and the vertex at x = 2a/√3 read 0
    let hex = SdfNode::hex_prism(a, 2.0 * h); // the constructor takes the full height
    assert!(alice_sdf::eval::eval(&hex, Vec3::new(0.0, a, 0.0)).abs() < 1e-5);
    assert!(
        alice_sdf::eval::eval(&hex, Vec3::new(2.0 * a / 3.0_f32.sqrt(), 0.0, 0.0)).abs() < 1e-4
    );
    let part = SdfNode::box3d(30.0, 30.0, 10.0)
        .translate(0.0, 0.0, -5.0)
        .union(hex.translate(0.0, 0.0, h));
    let r = coplanar_faces(&part);
    let seam = r
        .pairs
        .iter()
        .find(|p| p.kind == CoplanarKind::UnionSeam)
        .unwrap_or_else(|| panic!("{r:?}"));
    // the hexagon's area: 2√3 a²
    let area = 2.0 * 3.0_f32.sqrt() * a * a;
    assert!(
        (seam.overlap_area - area).abs() < 1e-3 * area,
        "{} vs {area}",
        seam.overlap_area
    );
    assert!(r.is_fully_analysed(), "{r:?}");
}

#[test]
fn a_block_against_a_hex_prism_side_is_a_seam_over_that_side() {
    let (a, h) = (5.0_f32, 4.0_f32);
    // a block on the flat side y = a, wider than the side (2a/√3) and taller than the prism
    let part = SdfNode::hex_prism(a, 2.0 * h).union(SdfNode::box3d(20.0, 6.0, 20.0).translate(
        0.0,
        a + 3.0,
        0.0,
    ));
    let r = coplanar_faces(&part);
    let seam = r
        .pairs
        .iter()
        .find(|p| p.kind == CoplanarKind::UnionSeam)
        .unwrap_or_else(|| panic!("{r:?}"));
    let side = 2.0 * a / 3.0_f32.sqrt() * 2.0 * h;
    assert!(
        (seam.overlap_area - side).abs() < 1e-3 * side,
        "{} vs {side}",
        seam.overlap_area
    );
    assert!(r.is_fully_analysed(), "{r:?}");
}

// ── Cone: base disk of `radius` at y = −half_height (normal −Y), apex at +half_height ──

#[test]
fn a_cone_standing_on_a_block_is_a_seam_over_its_base() {
    let (r0, h) = (4.0_f32, 6.0_f32);
    let cone = SdfNode::cone(r0, 2.0 * h); // the constructor takes the full height
                                           // the field's own geometry: the base's rim at (r0, -h, 0) reads 0
    assert!(alice_sdf::eval::eval(&cone, Vec3::new(r0, -h, 0.0)).abs() < 1e-5);
    let part = SdfNode::box3d(20.0, 4.0, 20.0)
        .translate(0.0, -2.0, 0.0)
        .union(cone.translate(0.0, h, 0.0));
    let r = coplanar_faces(&part);
    let seam = r
        .pairs
        .iter()
        .find(|p| p.kind == CoplanarKind::UnionSeam)
        .unwrap_or_else(|| panic!("{r:?}"));
    let disk = std::f32::consts::PI * r0 * r0;
    assert!(
        (seam.overlap_area - disk).abs() < 3e-3 * disk,
        "{} vs {disk}",
        seam.overlap_area
    );
    assert!(r.is_fully_analysed(), "{r:?}");
}

#[test]
fn a_countersink_cone_whose_base_lies_on_the_plate_face_is_a_flush_cut() {
    // a cone cutter turned apex-down with its base exactly on the plate's top face (y = 3): a membrane
    // of air over the countersink's mouth
    let (r0, h) = (4.0_f32, 3.0_f32);
    let sink = SdfNode::cone(r0, 2.0 * h)
        .rotate(Quat::from_rotation_x(std::f32::consts::PI))
        .translate(0.0, 3.0 - h, 0.0);
    let r = coplanar_faces(&SdfNode::box3d(20.0, 6.0, 20.0).subtract(sink));
    assert!(
        r.pairs.iter().any(|p| p.kind == CoplanarKind::FlushCut),
        "{r:?}"
    );
    assert!(r.is_fully_analysed(), "{r:?}");
}
