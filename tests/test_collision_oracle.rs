//! SDF–SDF collision oracle, against the closed form of two spheres.
//!
//! Spheres `A` (radius ra, centre ca) and `B` (rb, cb) with centre distance D:
//! - they overlap iff `D < ra + rb`; separation gap is `D − ra − rb`
//! - the deepest point of the lens has depth `(ra + rb − D) / 2` (equal radii)
//! - `B`'s outward normal at `p` is `(p − cb) / |p − cb|`
//! - a ball of radius r moving at speed v along the centre line from
//!   distance s first touches `A` at `t = (s − ra − r) / v`
//! - the closest surface point of a sphere to `q` is `ca + ra·(q − ca)/|q − ca|`
//!
//! The grid functions are approximations, so they are checked against
//! bounds that follow from the closed form (e.g. `sdf_distance` can never be
//! below the true gap, by the triangle inequality), not against themselves.
//!
//! Author: Moroya Sakamoto

use alice_sdf::prelude::*;

const fn aabb() -> Aabb {
    Aabb {
        min: Vec3::splat(-3.0),
        max: Vec3::splat(3.0),
    }
}

#[test]
fn overlap_matches_centre_distance() {
    let a = SdfNode::sphere(1.0);
    let mut compared = 0;
    for &(d, expected) in &[
        (0.5_f32, true),
        (1.0, true),
        (1.6, true),
        (2.4, false),
        (3.0, false),
    ] {
        let b = SdfNode::sphere(1.0).translate(d, 0.0, 0.0);
        assert_eq!(sdf_overlap(&a, &b, &aabb(), 32), expected, "D={d}");
        compared += 1;
    }
    assert!(compared > 0);
}

#[test]
fn contacts_lie_in_the_lens_with_closed_form_depth_and_normal() {
    let d = 1.5_f32;
    let cb = Vec3::new(d, 0.0, 0.0);
    let a = SdfNode::sphere(1.0);
    let b = SdfNode::sphere(1.0).translate(d, 0.0, 0.0);
    let res = 32;
    let contacts = sdf_collide(&a, &b, &aabb(), res);
    assert!(!contacts.is_empty());

    let cell_half_diag = 0.5 * (6.0 / res as f32) * 3f32.sqrt();
    let deepest = 1.0 - d / 2.0;
    for c in &contacts {
        // inside both spheres
        assert!(
            c.point.length() < 1.0 && (c.point - cb).length() < 1.0,
            "{c:?}"
        );
        // depth = −max(dA, dB) from the closed-form distances
        let expected = -((c.point.length() - 1.0).max((c.point - cb).length() - 1.0));
        assert!((c.depth - expected).abs() < 1e-5, "{c:?} vs {expected}");
        // normal = B's outward normal
        let n = (c.point - cb).normalize();
        assert!(c.normal.dot(n) > 0.999, "{c:?} vs {n}");
    }
    // sorted deepest first, deepest close to the lens midpoint depth
    assert!(contacts.windows(2).all(|w| w[0].depth >= w[1].depth));
    assert!(contacts[0].depth <= deepest + 1e-6);
    assert!(contacts[0].depth >= deepest - cell_half_diag);

    // manifold = mean point, normalised mean normal, max depth
    let m = compute_manifold(&contacts).unwrap();
    let n = contacts.len() as f32;
    let mean = contacts.iter().fold(Vec3::ZERO, |s, c| s + c.point) / n;
    let mean_n = contacts
        .iter()
        .fold(Vec3::ZERO, |s, c| s + c.normal)
        .normalize();
    assert_eq!(m.count, contacts.len());
    assert!((m.center - mean).length() < 1e-5);
    assert!((m.normal - mean_n).length() < 1e-5);
    assert_eq!(m.max_depth, contacts[0].depth);
    // the lens is symmetric about x = D/2 and the normal points from B to A (−X)
    assert!((m.center.x - d / 2.0).abs() < 0.1);
    assert!(m.normal.x < -0.99);
    assert!(compute_manifold(&[]).is_none());
}

#[test]
fn separation_is_an_upper_bound_on_the_gap() {
    let a = SdfNode::sphere(1.0);
    let res = 32;
    let cell_diag = (6.0 / res as f32) * 3f32.sqrt();
    let mut compared = 0;
    for &d in &[2.5_f32, 3.0, 3.5] {
        let b = SdfNode::sphere(1.0).translate(d, 0.0, 0.0);
        let gap = d - 2.0;
        let got = sdf_distance(&a, &b, &aabb(), res);
        assert!(got >= gap - 1e-5, "D={d}: {got} < gap {gap}");
        assert!(got <= gap + cell_diag, "D={d}: {got} vs gap {gap}");
        compared += 1;
    }
    let b = SdfNode::sphere(1.0).translate(1.0, 0.0, 0.0);
    assert_eq!(sdf_distance(&a, &b, &aabb(), res), 0.0);
    assert!(compared > 0);
}

#[test]
fn ccd_time_of_impact_matches_closed_form() {
    let a = SdfNode::sphere(1.0);
    let mut compared = 0;
    for &(s, v, r) in &[
        (5.0_f32, 10.0_f32, 0.25_f32),
        (4.0, 3.0, 0.5),
        (2.0, 1.0, 0.0),
    ] {
        let start = Vec3::new(-s, 0.0, 0.0);
        let (toi, at) = sdf_ccd(&a, start, Vec3::new(v, 0.0, 0.0), 2.0, r).expect("hit");
        let expected = (s - 1.0 - r) / v;
        assert!(
            (toi - expected).abs() < 1e-4,
            "s={s} v={v} r={r}: {toi} vs {expected}"
        );
        assert!((at.x - (-1.0 - r)).abs() < 1e-4);
        compared += 1;
    }
    // passes beside the sphere, or stops short within dt
    assert!(sdf_ccd(&a, Vec3::new(-5.0, 2.0, 0.0), Vec3::X * 10.0, 1.0, 0.5).is_none());
    assert!(sdf_ccd(&a, Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 1.0, 0.25).is_none());
    assert!(compared > 0);
}

#[test]
fn closest_point_on_a_sphere() {
    let c = Vec3::new(0.5, -1.0, 2.0);
    let node = SdfNode::sphere(2.0).translate(c.x, c.y, c.z);
    let mut compared = 0;
    for &q in &[
        Vec3::new(3.5, 3.0, 2.0),
        Vec3::new(0.5, -1.0, 6.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(-4.0, 2.0, -1.0),
    ] {
        let (p, residual) = sdf_closest_point(&node, q, 32);
        let expected = c + 2.0 * (q - c).normalize();
        assert!((p - expected).length() < 1e-3, "q={q}: {p} vs {expected}");
        assert!(residual < 1e-3);
        compared += 1;
    }
    assert!(compared > 0);
}
