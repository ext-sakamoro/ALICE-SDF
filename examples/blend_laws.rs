//! Blend laws — n-ary CSG and the smooth-minimum family
//!
//! Folds a list of distances with `sdf_union_multi` / `sdf_intersection_multi`,
//! and blends two distances with every smooth minimum the crate exposes
//! (polynomial with a precomputed reciprocal `*_rk`, cubic, exponential,
//! square root). Each value is printed and checked against its closed form
//! (iq's formulas); the full oracle is `tests/test_smooth_ops_oracle.rs` and
//! `tests/test_csg_multi_oracle.rs`.
//!
//! # Running
//! ```bash
//! cargo run --example blend_laws
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::operations::{
    sdf_intersection_multi, sdf_smooth_intersection_rk, sdf_smooth_subtraction_rk,
    sdf_smooth_union_rk, sdf_union_multi, smooth_min_cubic, smooth_min_exp, smooth_min_root,
};

fn check(name: &str, got: f32, want: f64) {
    println!("  {name:<34} {got:>10.6}  (closed form {want:>10.6})");
    assert!(
        (got as f64 - want).abs() <= 1e-5 * (1.0 + want.abs()),
        "{name}: {got} vs {want}"
    );
}

fn main() {
    println!("ALICE-SDF — blend laws");
    println!("======================");

    let ds = [0.8_f32, -0.25, 1.5, 0.1];
    println!("distances {ds:?}");
    check("sdf_union_multi (min)", sdf_union_multi(&ds), -0.25);
    check(
        "sdf_intersection_multi (max)",
        sdf_intersection_multi(&ds),
        1.5,
    );

    let (a, b, k) = (0.3_f32, 0.5_f32, 0.4_f32);
    let (a6, b6, k6) = (a as f64, b as f64, k as f64);
    println!();
    println!("a = {a}, b = {b}, k = {k}");
    // polynomial: min - h^2 k / 4, h = max(1 - |a - b| / k, 0)
    let h = (1.0 - (a6 - b6).abs() / k6).max(0.0);
    let poly_min = a6.min(b6) - h * h * k6 * 0.25;
    let poly_max = a6.max(b6) + h * h * k6 * 0.25;
    let rk = 1.0 / k;
    check(
        "sdf_smooth_union_rk",
        sdf_smooth_union_rk(a, b, k, rk),
        poly_min,
    );
    check(
        "sdf_smooth_intersection_rk",
        sdf_smooth_intersection_rk(a, b, k, rk),
        poly_max,
    );
    // subtraction of B from A is the smooth max of (a, -b)
    let hs = (1.0 - (a6 + b6).abs() / k6).max(0.0);
    check(
        "sdf_smooth_subtraction_rk",
        sdf_smooth_subtraction_rk(a, b, k, rk),
        a6.max(-b6) + hs * hs * k6 * 0.25,
    );
    // cubic: min - h^3 k / 6
    check(
        "smooth_min_cubic",
        smooth_min_cubic(a, b, k),
        a6.min(b6) - h.powi(3) * k6 / 6.0,
    );
    // exponential, rate convention: -ln(e^{-k a} + e^{-k b}) / k
    let kr = 8.0_f32;
    check(
        "smooth_min_exp (rate 8)",
        smooth_min_exp(a, b, kr),
        -((-kr as f64 * a6).exp() + (-kr as f64 * b6).exp()).ln() / kr as f64,
    );
    // square root: (a + b - sqrt((b - a)^2 + k^2)) / 2
    check(
        "smooth_min_root",
        smooth_min_root(a, b, k),
        0.5 * (a6 + b6 - ((b6 - a6).powi(2) + k6 * k6).sqrt()),
    );
    println!();
    println!("all checks passed");
}
