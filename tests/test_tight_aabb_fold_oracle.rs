//! Oracle for the level-set AABB bound on `PolarRepeat`, `Twist`, `Taper`,
//! `IcosahedralSymmetry`, `OctantMirror` and the primitives `Diamond`, `HexPrism`,
//! `Ellipsoid`
//!
//! Before: `analytic_aabb` returned `Unsupported` for each of these, so a gear
//! (`polar_repeat` of a tooth), a vase (`onion` of a `twist` of a `taper`) and an
//! `icosahedral_symmetry` solid fell back to the interval search.
//!
//! Independent references (none of them calls the bound under test):
//! (a) closed form of the box for each node, written from the geometry
//! (b) soundness: every grid / seeded random point with `f <= 0` lies inside the box
//! (c) tightness: the box against the extent of the solid points found by dense sampling
//!
//! Every test counts the solid points it checked and fails when none were found
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]

use alice_sdf::eval;
use alice_sdf::prelude::*;
use alice_sdf::tight_aabb::{analytic_aabb, AnalyticAabb};
use glam::Vec3;
use std::sync::Arc;

fn bounded(node: &SdfNode) -> Aabb {
    match analytic_aabb(node) {
        AnalyticAabb::Bounded(b) => b,
        other => panic!("expected a bound, got {other:?} for {node:?}"),
    }
}

fn unsupported(node: &SdfNode) -> bool {
    matches!(analytic_aabb(node), AnalyticAabb::Unsupported)
}

fn near(a: Vec3, b: Vec3, tol: f32) -> bool {
    (a - b).abs().max_element() <= tol
}

/// Seeded xorshift64*, uniform in `[0, 1)`
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// Extent of the solid points found in `[lo, hi]`: (count, min, max). Samples are a
/// fixed grid plus seeded random points.
fn solid_extent(node: &SdfNode, lo: Vec3, hi: Vec3, grid: u32, random: u32) -> (usize, Vec3, Vec3) {
    let mut count = 0;
    let mut mn = Vec3::splat(f32::INFINITY);
    let mut mx = Vec3::splat(f32::NEG_INFINITY);
    let mut visit = |p: Vec3| {
        if eval(node, p) <= 0.0 {
            count += 1;
            mn = mn.min(p);
            mx = mx.max(p);
        }
    };
    let step = (hi - lo) / grid as f32;
    for i in 0..=grid {
        for j in 0..=grid {
            for k in 0..=grid {
                visit(lo + step * Vec3::new(i as f32, j as f32, k as f32));
            }
        }
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for _ in 0..random {
        let u = Vec3::new(rng.next(), rng.next(), rng.next());
        visit(lo + (hi - lo) * u);
    }
    (count, mn, mx)
}

/// Soundness (b) over a window much larger than the box, and the ratio (c) of the box to
/// the sampled extent. Returns (solid points, worst per-axis ratio of box size to sampled size).
fn check(node: &SdfNode, b: &Aabb, window: f32, grid: u32, random: u32) -> (usize, f32) {
    let w = Vec3::splat(window);
    let (count, mn, mx) = solid_extent(node, -w, w, grid, random);
    assert!(count > 0, "no solid point sampled for {node:?}");
    assert!(
        mn.cmpge(b.min).all() && mx.cmple(b.max).all(),
        "solid extent [{mn:?}, {mx:?}] leaves the box {b:?}"
    );
    let ratio = ((b.max - b.min) / (mx - mn)).max_element();
    (count, ratio)
}

/// Ratio (c) of the box to the extent of the solid points in a window around the solid
/// (random points only, so the window should hug the expected extent).
fn tight_ratio(node: &SdfNode, b: &Aabb, window: Vec3, random: u32) -> f32 {
    let (count, mn, mx) = solid_extent(node, -window, window, 2, random);
    assert!(count > 0, "no solid point sampled for {node:?}");
    assert!(
        mn.cmpge(b.min).all() && mx.cmple(b.max).all(),
        "solid extent [{mn:?}, {mx:?}] leaves the box {b:?}"
    );
    ((b.max - b.min) / (mx - mn)).max_element()
}

#[test]
fn polar_repeat_box_is_the_radius_of_the_farthest_corner() {
    // tooth: cylinder r = 3 (Y axis), height 4, moved to x = 31: child box x in [28, 34],
    // z in [-3, 3], y in [-2, 2] -> farthest corner (34, 3)
    let node = SdfNode::cylinder(3.0, 4.0)
        .translate(31.0, 0.0, 0.0)
        .polar_repeat(12);
    let b = bounded(&node);
    let r = (34.0f32 * 34.0 + 9.0).sqrt();
    assert!(near(b.max, Vec3::new(r, 2.0, r), 5e-3), "{b:?}");
    assert!(near(b.min, Vec3::new(-r, -2.0, -r), 5e-3), "{b:?}");
    let (count, ratio) = check(&node, &b, 40.0, 60, 400_000);
    eprintln!("polar_repeat tooth: solid {count}, box/extent {ratio:.4}");
    assert!(ratio < 1.02, "ratio {ratio}");
}

#[test]
fn polar_repeat_of_an_off_axis_box_covers_every_copy() {
    // 5 copies of a box at (10, 0, 4) (+-2, +-1, +-1): the copies leave the child's own box
    let node = SdfNode::box3d(4.0, 2.0, 2.0)
        .translate(10.0, 0.0, 4.0)
        .polar_repeat(5);
    let b = bounded(&node);
    let r = (12.0f32 * 12.0 + 5.0 * 5.0).sqrt();
    assert!(near(b.max, Vec3::new(r, 1.0, r), 5e-3), "{b:?}");
    let (count, ratio) = check(&node, &b, 16.0, 80, 400_000);
    eprintln!("polar_repeat box: solid {count}, box/extent {ratio:.4}");
}

#[test]
fn twist_box_is_the_radius_of_the_farthest_corner() {
    // box (+-2, +-5, +-1) moved to (6, 0, 3): x in [4, 8], z in [2, 4] -> corner (8, 4)
    let node = SdfNode::box3d(4.0, 10.0, 2.0)
        .translate(6.0, 0.0, 3.0)
        .twist(0.35);
    let b = bounded(&node);
    let r = (64.0f32 + 16.0).sqrt();
    assert!(near(b.max, Vec3::new(r, 5.0, r), 5e-3), "{b:?}");
    assert!(near(b.min, Vec3::new(-r, -5.0, -r), 5e-3), "{b:?}");
    let (count, ratio) = check(&node, &b, 10.0, 80, 400_000);
    eprintln!("twist box: solid {count}, box/extent {ratio:.4}");
}

#[test]
fn twist_of_a_solid_of_revolution_uses_its_radius_not_the_box_corner() {
    // cylinder r = 20: a box corner would be 28.3; the disc is 20
    let node = SdfNode::cylinder(20.0, 40.0).twist(0.3);
    let b = bounded(&node);
    assert!(near(b.max, Vec3::new(20.0, 20.0, 20.0), 5e-2), "{b:?}");
    check(&node, &b, 24.0, 80, 400_000);
    // taper widens the disc to 20 * (1 + 0.015 * 20) = 26 at y = -20
    let node = SdfNode::cylinder(20.0, 40.0).taper(0.015).twist(0.025);
    let b = bounded(&node);
    assert!(near(b.max, Vec3::new(26.0, 20.0, 26.0), 5e-2), "{b:?}");
    assert!(near(b.min, Vec3::new(-26.0, -20.0, -26.0), 5e-2), "{b:?}");
    check(&node, &b, 30.0, 80, 600_000);
    let ratio = tight_ratio(&node, &b, Vec3::new(27.0, 21.0, 27.0), 1_500_000);
    eprintln!("twist(taper(cylinder)): box/extent {ratio:.4}");
    assert!(ratio < 1.05, "ratio {ratio}");
}

#[test]
fn taper_box_is_the_child_extent_times_the_widest_scale() {
    // cylinder r = 2, y in [-3, 3], f = 0.1: 1 - y f in [0.7, 1.3] -> radius 2.6 at y = -3
    let node = SdfNode::cylinder(2.0, 6.0).taper(0.1);
    let b = bounded(&node);
    assert!(near(b.max, Vec3::new(2.6, 3.0, 2.6), 5e-3), "{b:?}");
    assert!(near(b.min, Vec3::new(-2.6, -3.0, -2.6), 5e-3), "{b:?}");
    let (count, ratio) = check(&node, &b, 4.0, 80, 400_000);
    eprintln!("taper cylinder: solid {count}, box/extent {ratio:.4}");
    assert!(ratio < 1.03, "ratio {ratio}");
    // negative factor widens the top instead
    let node = SdfNode::cylinder(2.0, 6.0).taper(-0.1);
    let b = bounded(&node);
    assert!(near(b.max, Vec3::new(2.6, 3.0, 2.6), 5e-3), "{b:?}");
    check(&node, &b, 4.0, 80, 400_000);
}

#[test]
fn taper_of_an_off_center_box_keeps_the_sign_of_each_side() {
    // box x in [1, 3], y in [0, 2], z in [-2, -1] at f = 0.2: scale 1 - 0.2 y in [0.6, 1]
    // x in [0.6, 3], z in [-2, -0.6]
    let node = SdfNode::box3d(2.0, 2.0, 1.0)
        .translate(2.0, 1.0, -1.5)
        .taper(0.2);
    let b = bounded(&node);
    assert!(near(b.min, Vec3::new(0.6, 0.0, -2.0), 5e-3), "{b:?}");
    assert!(near(b.max, Vec3::new(3.0, 2.0, -0.6), 5e-3), "{b:?}");
    check(&node, &b, 4.0, 100, 400_000);
}

#[test]
fn taper_across_the_singular_plane_stays_unsupported() {
    // 1 - 0.06 y reaches 0 at y = 16.7 < 20
    assert!(unsupported(&SdfNode::cylinder(20.0, 40.0).taper(0.06)));
    assert!(unsupported(&SdfNode::cylinder(20.0, 40.0).taper(-0.06)));
    // exactly on the plane
    assert!(unsupported(&SdfNode::cylinder(20.0, 40.0).taper(0.05)));
    // an unsupported child stays unsupported
    assert!(unsupported(&SdfNode::sphere(2.0).bend(0.1).taper(0.1)));
}

#[test]
fn degenerate_folds_stay_unsupported() {
    // zero copies: the sector angle is not finite
    assert!(unsupported(&SdfNode::sphere(2.0).polar_repeat(0)));
    assert!(unsupported(&SdfNode::sphere(2.0).twist(f32::NAN)));
    assert!(unsupported(&SdfNode::sphere(2.0).twist(f32::INFINITY)));
    // a child that is not bounded
    assert!(unsupported(&SdfNode::sphere(2.0).bend(0.1).polar_repeat(4)));
    assert!(unsupported(&SdfNode::sphere(2.0).bend(0.1).twist(0.1)));
    assert!(unsupported(
        &SdfNode::sphere(2.0).bend(0.1).icosahedral_symmetry()
    ));
}

#[test]
fn taper_with_a_level_above_zero_covers_the_cone() {
    // round(r) asks the taper for level r: the field there is the Jacobian-bounded
    // distance, which exceeds r far out; the bound must hold for the whole level set
    let node = SdfNode::cylinder(5.0, 10.0).taper(0.04).round(1.5);
    let b = bounded(&node);
    let (count, ratio) = check(&node, &b, 12.0, 90, 600_000);
    eprintln!("taper round: solid {count}, box/extent {ratio:.4}");
    // a taper without a usable reach cannot bound a positive level
    let blind = SdfNode::Taper {
        child: Arc::new(SdfNode::cylinder(5.0, 10.0)),
        factor: 0.04,
        reach: [f32::INFINITY; 2],
    };
    assert!(
        matches!(analytic_aabb(&blind), AnalyticAabb::Bounded(_)),
        "level 0 needs no reach"
    );
    assert!(unsupported(&blind.round(1.5)));
}

#[test]
fn icosahedral_symmetry_box_is_the_cube_of_the_farthest_corner() {
    let node = SdfNode::sphere(6.0)
        .translate(20.0, 0.0, 0.0)
        .icosahedral_symmetry();
    let b = bounded(&node);
    // child box x in [14, 26], y, z in [-6, 6]: corner norm sqrt(26^2 + 6^2 + 6^2)
    let r = (26.0f32 * 26.0 + 72.0).sqrt();
    assert!(near(b.max, Vec3::splat(r), 5e-2), "{b:?}");
    assert!(near(b.min, Vec3::splat(-r), 5e-2), "{b:?}");
    let (count, ratio) = check(&node, &b, 30.0, 80, 600_000);
    eprintln!("icosahedral sphere: solid {count}, box/extent {ratio:.4}");
}

#[test]
fn octant_mirror_box_is_the_largest_coordinate() {
    // box x in [3, 5], y in [-1, 1], z in [0, 2]: images use |x| <= 5
    let node = SdfNode::box3d(2.0, 2.0, 2.0)
        .translate(4.0, 0.0, 1.0)
        .octant_mirror();
    let b = bounded(&node);
    assert!(near(b.max, Vec3::splat(5.0), 5e-3), "{b:?}");
    assert!(near(b.min, Vec3::splat(-5.0), 5e-3), "{b:?}");
    check(&node, &b, 8.0, 80, 400_000);
}

#[test]
fn diamond_box_is_the_cylinder_of_the_equator_and_apexes() {
    let node = SdfNode::diamond(3.0, 8.0);
    let b = bounded(&node);
    assert!(near(b.max, Vec3::new(3.0, 4.0, 3.0), 5e-3), "{b:?}");
    assert!(near(b.min, Vec3::new(-3.0, -4.0, -3.0), 5e-3), "{b:?}");
    let (count, ratio) = check(&node, &b, 6.0, 80, 400_000);
    eprintln!("diamond: solid {count}, box/extent {ratio:.4}");
    assert!(ratio < 1.03, "ratio {ratio}");
    // a rounded diamond asks for level 0.5: the dilated set
    let node = SdfNode::diamond(3.0, 8.0).round(0.5);
    let b = bounded(&node);
    assert!(near(b.max, Vec3::new(3.5, 4.5, 3.5), 5e-3), "{b:?}");
    check(&node, &b, 6.0, 80, 400_000);
    // degenerate dimensions are not covered
    assert!(unsupported(&SdfNode::diamond(0.0, 8.0)));
    assert!(unsupported(&SdfNode::diamond(3.0, -1.0)));
}

#[test]
fn hex_prism_box_has_the_corners_on_x_and_the_flats_on_y() {
    let node = SdfNode::hex_prism(3.0, 4.0);
    let b = bounded(&node);
    let corner = 3.0 * 2.0 / 3.0f32.sqrt();
    assert!(near(b.max, Vec3::new(corner, 3.0, 2.0), 5e-3), "{b:?}");
    assert!(near(b.min, Vec3::new(-corner, -3.0, -2.0), 5e-3), "{b:?}");
    let (count, ratio) = check(&node, &b, 6.0, 90, 400_000);
    eprintln!("hex prism: solid {count}, box/extent {ratio:.4}");
    assert!(ratio < 1.03, "ratio {ratio}");
}

#[test]
fn ellipsoid_box_is_the_radii() {
    let node = SdfNode::Ellipsoid {
        radii: Vec3::new(4.0, 1.5, 2.5),
    };
    let b = bounded(&node);
    assert!(near(b.max, Vec3::new(4.0, 1.5, 2.5), 5e-3), "{b:?}");
    assert!(near(b.min, Vec3::new(-4.0, -1.5, -2.5), 5e-3), "{b:?}");
    let (count, ratio) = check(&node, &b, 6.0, 80, 400_000);
    eprintln!("ellipsoid: solid {count}, box/extent {ratio:.4}");
    assert!(ratio < 1.03, "ratio {ratio}");
    assert!(unsupported(&SdfNode::Ellipsoid {
        radii: Vec3::new(1.0, 0.0, 1.0)
    }));
}

#[test]
fn nested_folds_compose() {
    // twist of a polar repeat of a taper: each stage is bounded by the stage below
    let node = SdfNode::cylinder(2.0, 6.0)
        .taper(0.1)
        .translate(10.0, 0.0, 0.0)
        .polar_repeat(7)
        .twist(0.2);
    let b = bounded(&node);
    // taper box x in [-2.6, 2.6] -> moved [7.4, 12.6], z +-2.6: corner radius sqrt(12.6^2 + 2.6^2)
    let r = (12.6f32 * 12.6 + 2.6 * 2.6).sqrt();
    assert!(near(b.max, Vec3::new(r, 3.0, r), 5e-2), "{b:?}");
    check(&node, &b, 16.0, 90, 600_000);
    let ratio = tight_ratio(&node, &b, Vec3::new(14.0, 3.5, 14.0), 1_500_000);
    eprintln!("nested: box/extent {ratio:.4}");
}

/// Fence 1: gear
fn gear() -> SdfNode {
    SdfNode::cylinder(30.0, 8.0)
        .subtract(SdfNode::cylinder(8.0, 10.0))
        .union(
            SdfNode::cylinder(3.0, 4.0)
                .translate(31.0, 0.0, 0.0)
                .polar_repeat(12),
        )
}

/// Fence 2: vase
fn vase() -> SdfNode {
    SdfNode::cylinder(20.0, 40.0)
        .taper(0.015)
        .twist(0.025)
        .onion(1.5)
}

/// Fence 3: icosahedral solid
fn icosa() -> SdfNode {
    SdfNode::sphere(6.0)
        .translate(20.0, 0.0, 0.0)
        .icosahedral_symmetry()
}

#[test]
fn fence_gear_is_bounded_and_tight() {
    let node = gear();
    let b = bounded(&node);
    let (count, ratio) = check(&node, &b, 40.0, 100, 800_000);
    eprintln!("FENCE gear: box {b:?} solid {count} box/extent {ratio:.4}");
    // the farthest tooth reaches x = 34: the box stops at the corner radius of one tooth
    assert!(b.max.x < 34.2, "{b:?}");
    assert!(ratio < 1.02, "ratio {ratio}");
}

#[test]
fn fence_vase_is_bounded_and_tight() {
    let node = vase();
    let b = bounded(&node);
    let (count, ratio) = check(&node, &b, 40.0, 110, 800_000);
    eprintln!("FENCE vase: box {b:?} solid {count} box/extent {ratio:.4}");
    // child: radius 20 at y = 0, 26 at y = -20; onion level 1.5 adds the cone slack
    let near_ratio = tight_ratio(&node, &b, Vec3::new(45.0, 23.0, 45.0), 4_000_000);
    eprintln!("FENCE vase near-surface: box/extent {near_ratio:.4}");
}

#[test]
fn fence_icosahedral_is_bounded() {
    let node = icosa();
    let b = bounded(&node);
    let (count, ratio) = check(&node, &b, 30.0, 100, 800_000);
    eprintln!("FENCE icosahedral: box {b:?} solid {count} box/extent {ratio:.4}");
    let near_ratio = tight_ratio(&node, &b, Vec3::splat(27.5), 3_000_000);
    eprintln!("FENCE icosahedral near-surface: box/extent {near_ratio:.4}");
    assert!(near_ratio < 1.25, "ratio {near_ratio}");
}

#[test]
fn ellipsoid_at_a_positive_level_grows_by_the_level() {
    // round(0.5) asks the ellipsoid for level 0.5: the exact field grows the shape by 0.5
    let node = SdfNode::Ellipsoid {
        radii: Vec3::new(4.0, 1.5, 2.5),
    }
    .round(0.5);
    let b = bounded(&node);
    assert!(near(b.max, Vec3::new(4.5, 2.0, 3.0), 5e-3), "{b:?}");
    assert!(near(b.min, Vec3::new(-4.5, -2.0, -3.0), 5e-3), "{b:?}");
    let (count, ratio) = check(&node, &b, 6.0, 80, 400_000);
    eprintln!("ellipsoid round: solid {count}, box/extent {ratio:.4}");
    assert!(ratio < 1.03, "ratio {ratio}");
}

#[test]
fn taper_cone_radius_includes_the_level_times_the_cone_slope_term() {
    // reach of the child is [20 sqrt(2), 20]; level 1.5 from round: the evaluator's cone is
    // rho <= k |y - 1/f| + level sqrt(1 + k^2) with k = reach_xz |f|, |y| <= reach_y + level
    let (f, level) = (0.015f32, 1.5f32);
    let node = SdfNode::cylinder(20.0, 40.0).taper(f).round(level);
    let k = 20.0 * 2.0f32.sqrt() * f;
    let expected = k * (20.0 + level + 1.0 / f) + level * (k * k + 1.0).sqrt();
    let b = bounded(&node);
    assert!(
        b.max.x >= expected - 1e-3 && b.max.x <= expected + 1e-2,
        "{} vs {expected}",
        b.max.x
    );
    assert!(
        b.min.z <= -expected + 1e-3 && b.min.z >= -expected - 1e-2,
        "{b:?}"
    );
    check(&node, &b, 45.0, 90, 600_000);
}

#[test]
fn twist_of_a_centred_torus_uses_major_plus_minor() {
    // torus (1, 0.3): the disc about the Y axis has radius 1.3; the box corner would be 1.84
    let node = SdfNode::torus(1.0, 0.3).twist(1.0);
    let b = bounded(&node);
    assert!(b.max.x >= 1.3 && b.max.x <= 1.31, "{b:?}");
    assert!(b.min.z <= -1.3 && b.min.z >= -1.31, "{b:?}");
    let (count, ratio) = check(&node, &b, 2.0, 80, 400_000);
    eprintln!("twist torus: solid {count}, box/extent {ratio:.4}");
    assert!(ratio < 1.05, "ratio {ratio}");
}
