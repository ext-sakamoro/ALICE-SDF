//! The number of the loft bound law moves whenever the law's values move.
//!
//! A stored loft is accepted only with the number of this build's law, and its bounds must equal the
//! bounds recomputed on load. The number therefore has to change with the law: two builds that share a
//! number but not the values would refuse each other's trees with the wrong message, and a tree could
//! not say which law it came from.
//!
//! `LAW_VALUES` holds one row per law number, the digest of every bound of fixed lofts under that law.
//! A change of the values with the same number finds its number already holding another digest and
//! fails. To change the law: raise `LOFT_LAW_VERSION` and add a row; never edit an existing row.
//!
//! Author: Moroya Sakamoto
use alice_sdf::prelude::*;
use alice_sdf::primitives::loft::{LoftBounds, LOFT_LAW_VERSION};
use glam::Vec2;

/// (law number, digest of `all_bounds`)
const LAW_VALUES: &[(u32, u64)] = &[(2, 0x9f5d_7a61_6aa0_f09e)];

fn ellipse(a: f32, b: f32) -> Vec<Vec2> {
    (0..12)
        .map(|j| {
            let (s, c) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / 12.0);
            Vec2::new(a * c, b * s)
        })
        .collect()
}

/// Lofts that reach every bound: straight and bent axes, flat ends and domes
fn lofts() -> Vec<SdfNode> {
    let rows = || vec![ellipse(0.2, 0.12), ellipse(0.18, 0.13), ellipse(0.15, 0.1), ellipse(0.1, 0.08)];
    let bent = vec![Vec3::ZERO, Vec3::new(0.03, 0.3, 0.0), Vec3::new(0.05, 0.6, 0.02), Vec3::new(0.04, 0.9, 0.0)];
    let straight = vec![Vec3::ZERO, Vec3::new(0.0, 0.3, 0.0), Vec3::new(0.0, 0.6, 0.0), Vec3::new(0.0, 0.9, 0.0)];
    let polar = vec![vec![0.15, 0.01, 0.0, 0.01, 0.0], vec![0.17, 0.0, 0.01, 0.0, 0.0], vec![0.12, 0.0, 0.0, 0.0, 0.01]];
    let polar_axis = vec![Vec3::ZERO, Vec3::new(0.02, 0.3, 0.0), Vec3::new(0.0, 0.6, 0.03)];
    vec![
        SdfNode::loft_curves(bent.clone(), Vec3::X, rows()).expect("bent curve loft"),
        SdfNode::loft_curves(straight, Vec3::X, rows()).expect("straight curve loft"),
        SdfNode::loft_curves_capped(bent, Vec3::X, rows(), [[0.1, 0.0, 0.0], [0.15, 0.06, -0.03]]).expect("capped curve loft"),
        SdfNode::loft(polar_axis.clone(), Vec3::Z, 2, polar.clone()).expect("polar loft"),
        SdfNode::loft_capped(polar_axis, Vec3::Z, 2, polar, [0.08, 0.12]).expect("capped polar loft"),
    ]
}

fn bounds_of(n: &SdfNode) -> &LoftBounds {
    match n {
        SdfNode::Loft { bounds, .. } | SdfNode::LoftCurves { bounds, .. } => bounds,
        _ => panic!("not a loft"),
    }
}

/// FNV-1a over the bound's stored text (every field, with the number taken out so the digest holds
/// values only)
fn digest(all: &[&LoftBounds]) -> u64 {
    let mut d = 0xcbf2_9ce4_8422_2325_u64;
    for b in all {
        let mut v = serde_json::to_value(b).expect("bounds to JSON");
        v.as_object_mut().expect("an object").remove("law");
        for byte in v.to_string().bytes() {
            d = (d ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    d
}

#[test]
fn the_law_number_moves_with_the_bound_values() {
    let trees = lofts();
    let all: Vec<&LoftBounds> = trees.iter().map(bounds_of).collect();
    assert!(all.iter().all(|b| b.law == LOFT_LAW_VERSION), "a loft carries another law number");
    assert!(all.iter().any(|b| b.fade.is_some()), "no loft reaches the fade");
    assert!(all.iter().any(|b| b.lq != [1.0, 1.0]), "no loft reaches the domes");
    assert!(all.iter().any(|b| b.m > 0.0), "no loft reaches the local correction");
    let now = digest(&all);
    let row = LAW_VALUES.iter().find(|(n, _)| *n == LOFT_LAW_VERSION);
    let Some((n, pinned)) = row else {
        panic!("law {LOFT_LAW_VERSION} has no row in LAW_VALUES: add ({LOFT_LAW_VERSION}, {now:#x})");
    };
    assert_eq!(*pinned, now, "the bound values moved under law {n}: raise LOFT_LAW_VERSION and add a row, never edit one (digest now {now:#x})");
}

#[test]
fn every_row_names_its_own_number_and_values() {
    assert!(!LAW_VALUES.is_empty());
    for (i, a) in LAW_VALUES.iter().enumerate() {
        for b in &LAW_VALUES[i + 1..] {
            assert!(a.0 < b.0, "rows are in rising law order");
            assert_ne!(a.1, b.1, "laws {} and {} hold the same values, so the raise was not needed", a.0, b.0);
        }
    }
}
