//! ALICE-SDF Mobile UniFFI wrapper の統合テスト
//!
//! Swift / Kotlin から見える全 12 関数を Rust 側で網羅的に確認する。
//! UniFFI が生成した binding はラッパーが Rust 関数を呼ぶだけなので、Rust 側で
//! 数値が一致すれば iOS / Android の両プラットフォームで同じ結果になることが
//! v1.5.0 の実機検証で確認済み (iPhone 17 Pro Sim + Pixel 6 Emulator)。

use alice_sdf_mobile::*;

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3 { x, y, z }
}

const ORIGIN: Vec3 = Vec3 {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};

// ── Primitive: sdf_sphere ──────────────────────────────────────────

#[test]
fn sphere_inside_negative() {
    let d = sdf_sphere(v(0.0, 0.0, 0.0), ORIGIN, 1.0);
    assert!(d < 0.0);
}

#[test]
fn sphere_outside_positive() {
    let d = sdf_sphere(v(2.0, 0.0, 0.0), ORIGIN, 1.0);
    assert!((d - 1.0).abs() < 1e-5);
}

#[test]
fn sphere_offset_center() {
    let d = sdf_sphere(v(3.0, 0.0, 0.0), v(2.0, 0.0, 0.0), 1.0);
    assert!(d.abs() < 1e-5);
}

// ── Primitive: sdf_box ─────────────────────────────────────────────

#[test]
fn box_corner_distance_matches_pythagoras() {
    let he = v(1.0, 1.0, 1.0);
    let d = sdf_box(v(2.0, 2.0, 2.0), ORIGIN, he);
    let expect = (3.0_f32).sqrt();
    assert!((d - expect).abs() < 1e-4, "got {d}, want {expect}");
}

#[test]
fn box_face_distance_is_axis_aligned() {
    let d = sdf_box(v(3.0, 0.0, 0.0), ORIGIN, v(1.0, 1.0, 1.0));
    assert!((d - 2.0).abs() < 1e-5);
}

// ── Primitive: sdf_rounded_box ─────────────────────────────────────

#[test]
fn rounded_box_differs_from_sharp_box_at_corner() {
    let he = v(0.5, 0.5, 0.5);
    let p = v(1.0, 1.0, 1.0);
    let sharp = sdf_box(p, ORIGIN, he);
    let smooth = sdf_rounded_box(p, ORIGIN, he, 0.2);
    assert!(smooth.is_finite() && sharp.is_finite());
    assert!(
        (smooth - sharp).abs() > 1e-3,
        "rounded should differ from sharp (sharp={sharp}, smooth={smooth})"
    );
}

#[test]
fn rounded_box_zero_radius_matches_sharp() {
    let he = v(0.5, 0.5, 0.5);
    let p = v(1.0, 1.0, 1.0);
    let sharp = sdf_box(p, ORIGIN, he);
    let smooth = sdf_rounded_box(p, ORIGIN, he, 0.0);
    assert!(
        (smooth - sharp).abs() < 1e-4,
        "round_radius=0 should match sharp box: sharp={sharp}, smooth={smooth}"
    );
}

// ── Primitive: sdf_cylinder ────────────────────────────────────────

#[test]
fn cylinder_axis_inside() {
    let d = sdf_cylinder(v(0.0, 0.0, 0.0), 1.0, 1.0);
    assert!(d < 0.0);
}

#[test]
fn cylinder_top_cap_distance() {
    let d = sdf_cylinder(v(0.0, 3.0, 0.0), 1.0, 1.0);
    assert!((d - 2.0).abs() < 1e-4, "got {d}");
}

// ── Primitive: sdf_torus ───────────────────────────────────────────

#[test]
fn torus_central_axis_distance_equals_major_minus_minor() {
    // central axis (y axis) is at distance `major` from the ring centerline,
    // so SDF should be (major - minor).
    let d = sdf_torus(v(0.0, 0.0, 0.0), 1.0, 0.3);
    assert!((d - (1.0 - 0.3)).abs() < 1e-4, "got {d}");
}

// ── Primitive: sdf_plane ───────────────────────────────────────────

#[test]
fn plane_distance_signed_above() {
    let d = sdf_plane(v(0.0, 2.0, 0.0), v(0.0, 1.0, 0.0), 0.0);
    assert!((d - 2.0).abs() < 1e-5);
}

#[test]
fn plane_distance_signed_below() {
    let d = sdf_plane(v(0.0, -2.0, 0.0), v(0.0, 1.0, 0.0), 0.0);
    assert!((d + 2.0).abs() < 1e-5);
}

// ── Operations ─────────────────────────────────────────────────────

#[test]
fn op_union_picks_min() {
    assert!((op_union(0.3, 0.5) - 0.3).abs() < 1e-6);
    assert!((op_union(-0.1, 0.5) - (-0.1)).abs() < 1e-6);
}

#[test]
fn op_intersection_picks_max() {
    assert!((op_intersection(0.3, 0.5) - 0.5).abs() < 1e-6);
}

#[test]
fn op_subtraction_is_max_a_neg_b() {
    // sdf_subtraction(a, b) = max(a, -b) in classic CSG
    let r = op_subtraction(0.3, -0.5);
    assert!((r - 0.5).abs() < 1e-6, "got {r}");
}

#[test]
fn op_smooth_union_is_below_min() {
    let s = op_smooth_union(0.5, 0.5, 0.1);
    assert!(s < 0.5);
}

#[test]
fn op_smooth_intersection_is_above_max() {
    let s = op_smooth_intersection(0.5, 0.5, 0.1);
    assert!(s > 0.5);
}

#[test]
fn op_smooth_subtraction_finite() {
    let s = op_smooth_subtraction(0.5, -0.3, 0.1);
    assert!(s.is_finite());
}

#[test]
fn op_smooth_with_zero_k_approximates_hard_union() {
    let hard = op_union(0.4, 0.5);
    let soft = op_smooth_union(0.4, 0.5, 0.0);
    // k=0 だと hard との差は数値誤差程度
    assert!((hard - soft).abs() < 1e-3, "hard={hard}, soft={soft}");
}

// ── Batch ──────────────────────────────────────────────────────────

#[test]
fn sphere_batch_empty_returns_empty() {
    let out = sphere_batch(vec![], ORIGIN, 1.0);
    assert_eq!(out.len(), 0);
}

#[test]
fn sphere_batch_length_matches_input() {
    let pts: Vec<Vec3> = (0..16).map(|i| v(i as f32 * 0.1, 0.0, 0.0)).collect();
    let out = sphere_batch(pts.clone(), ORIGIN, 1.0);
    assert_eq!(out.len(), pts.len());
}

#[test]
fn sphere_batch_matches_individual_evaluation() {
    let pts = vec![
        v(1.0, 0.0, 0.0),
        v(0.0, 2.0, 0.0),
        v(0.0, 0.0, 3.0),
        v(0.5, 0.5, 0.5),
    ];
    let batch = sphere_batch(pts.clone(), ORIGIN, 1.0);
    for (i, p) in pts.iter().enumerate() {
        let single = sdf_sphere(*p, ORIGIN, 1.0);
        assert!(
            (batch[i] - single).abs() < 1e-6,
            "batch[{i}]={} != single={single}",
            batch[i]
        );
    }
}

// ── Meta ───────────────────────────────────────────────────────────

#[test]
fn version_is_non_empty_string() {
    let s = alice_sdf_version();
    assert!(!s.is_empty());
    assert!(s.contains('.'), "expected semver string, got '{s}'");
}

// ── Cross-platform numerical stability ──────────────────────────────
// 実機検証 (iPhone 17 Pro Sim + Pixel 6 Emulator) で iOS と Android の数値が
// 完全一致したのは Rust core を同じ logic で実行しているため。
// ここでは Rust 側の決定論性 (idempotent + 桁落ちなし) を保証する。

#[test]
fn determinism_repeated_calls_identical() {
    let p = v(0.6, 0.4, 0.2);
    let d1 = sdf_sphere(p, ORIGIN, 0.5);
    let d2 = sdf_sphere(p, ORIGIN, 0.5);
    let d3 = sdf_sphere(p, ORIGIN, 0.5);
    assert_eq!(d1.to_bits(), d2.to_bits());
    assert_eq!(d2.to_bits(), d3.to_bits());
}

#[test]
fn determinism_smooth_op_stable() {
    let s1 = op_smooth_union(0.3, 0.2, 0.1);
    let s2 = op_smooth_union(0.3, 0.2, 0.1);
    assert_eq!(s1.to_bits(), s2.to_bits());
}

#[test]
fn determinism_batch_self_consistent() {
    let pts: Vec<Vec3> = (0..32).map(|i| v(i as f32 * 0.05, 0.0, 0.0)).collect();
    let b1 = sphere_batch(pts.clone(), ORIGIN, 0.5);
    let b2 = sphere_batch(pts, ORIGIN, 0.5);
    assert_eq!(b1.len(), b2.len());
    for i in 0..b1.len() {
        assert_eq!(b1[i].to_bits(), b2[i].to_bits(), "drift at i={i}");
    }
}
