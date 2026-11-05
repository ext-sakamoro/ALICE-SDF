//! A loft whose sections are wider than the axis's radius of curvature is refused at build time:
//! there the section planes of neighbouring axis points cross inside the sections, the swept solid
//! folds over itself and no distance field describes it. The axis here is a circular arc of radius
//! `RC` (its points on the circle, so the Catmull-Rom curve has curvature close to `1/RC`); the law
//! accepts sections with `1.1 κ R < 1` (see `primitives::loft`), so sections of radius `0.3 RC` pass
//! and `0.95 RC` are refused, for polar and curve sections.

use alice_sdf::prelude::*;
use glam::Vec2;

const RC: f32 = 0.5;

fn arc() -> Vec<Vec3> {
    (0..9)
        .map(|i| {
            let (s, c) = alice_det_math::sin_cos(0.15 * i as f32);
            Vec3::new(RC * (1.0 - c), RC * s, 0.0)
        })
        .collect()
}

fn circle(r: f32) -> Vec<Vec2> {
    (0..12)
        .map(|j| {
            let (s, c) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / 12.0);
            Vec2::new(c, s) * r
        })
        .collect()
}

fn polar(r: f32) -> Result<SdfNode, String> {
    SdfNode::loft(arc(), Vec3::Z, 1, vec![vec![r, 0.0, 0.0]; 4]).map_err(|e| e.to_string())
}

fn curves(r: f32) -> Result<SdfNode, String> {
    SdfNode::loft_curves(arc(), Vec3::Z, vec![circle(r); 4]).map_err(|e| e.to_string())
}

#[test]
fn sections_within_the_radius_of_curvature_are_accepted() {
    assert!(polar(0.3 * RC).is_ok(), "{:?}", polar(0.3 * RC).err());
    assert!(curves(0.3 * RC).is_ok(), "{:?}", curves(0.3 * RC).err());
}

#[test]
fn sections_wider_than_the_radius_of_curvature_are_refused() {
    for (what, r) in [("polar", polar(0.95 * RC)), ("curves", curves(0.95 * RC))] {
        let e = r.err().unwrap_or_else(|| panic!("{what}: accepted"));
        assert!(e.contains("radius of curvature"), "{what}: {e}");
    }
}

/// A four-point axis whose middle segment has a near-cusp halfway between two of the law's curvature
/// samples (64 per segment): its derivative is `(A (t − t₀)² + ε, B (t − t₀))` with `t₀ = 1/2 + 1/128`
/// and `A = ε/(1/128)²`, so `C′ × C″` vanishes at the neighbouring samples while the curvature at
/// `t₀` is `B/ε² = 2.5·10⁵`. Sampled, the largest curvature reads 547 (computed independently with
/// the same sampling), so sections of radius 10⁻³ (`1.1 κ R`: 0.60 sampled, 275 true) pass a sampled
/// check but fold over at the cusp; a bound that holds between samples refuses them
fn cusp_axis() -> Vec<Vec3> {
    vec![
        Vec3::new(-14.169_333, 1.007_812, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.734_667, -0.007_812, 0.0),
        Vec3::new(15.88, 0.984_375, 0.0),
    ]
}

#[test]
fn a_bend_between_the_curvature_samples_is_refused() {
    let r = 1e-3;
    let polar = SdfNode::loft(cusp_axis(), Vec3::Z, 1, vec![vec![r, 0.0, 0.0]; 4]).map_err(|e| e.to_string());
    let curve = SdfNode::loft_curves(cusp_axis(), Vec3::Z, vec![circle(r); 4]).map_err(|e| e.to_string());
    for (what, res) in [("polar", polar), ("curves", curve)] {
        let e = res.err().unwrap_or_else(|| panic!("{what}: a section wider than the cusp's radius of curvature was accepted"));
        assert!(e.contains("radius of curvature"), "{what}: {e}");
    }
}

/// Close to the limit, as the closest real body part measured (1.1 κ R ≈ 0.85): sections of radius
/// 0.386 RC on the arc (curvature ≈ 1/RC) are still accepted, so a bound that holds between samples
/// stays tight enough for real axes
#[test]
fn sections_near_the_limit_are_accepted() {
    let r = 0.386 * RC;
    assert!(polar(r).is_ok(), "{:?}", polar(r).err());
    assert!(curves(r).is_ok(), "{:?}", curves(r).err());
}
