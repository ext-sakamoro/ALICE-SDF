//! Forward-mode derivatives with dual numbers, and surface curvature.
//!
//! - `Dual` differentiates a 1D expression in one pass
//! - `Dual3` carries a value with its gradient; the `dual3_*` primitives give
//!   the distance and its exact gradient together
//! - `eval_with_gradient` / `eval_dual3` do the same for any `SdfNode`
//! - `principal_curvatures` / `gaussian_curvature` read the shape of the
//!   surface at a point
//!
//! Every printed number is checked against the closed form.
//!
//! Run: `cargo run --example autodiff_curvature`
//!
//! Author: Moroya Sakamoto

use alice_sdf::autodiff::{
    dual3_box, dual3_plane, dual3_point, dual3_sphere, dual3_torus, eval_dual3, eval_with_gradient,
    gaussian_curvature, principal_curvatures, Dual, Dual3,
};
use alice_sdf::types::SdfNode;
use glam::Vec3;

fn main() {
    // ── Dual: f(x) = sqrt(x² + 1) · sin(x), f'(x) by the chain rule ──
    let x = 0.8_f32;
    let v = Dual::variable(x);
    let f = (v * v + Dual::constant(1.0)).sqrt() * v.sin();
    let s = (x * x + 1.0).sqrt();
    let df = x / s * x.sin() + s * x.cos();
    println!(
        "f({x}) = {:.6}, f' = {:.6} (closed form {df:.6})",
        f.val, f.dot
    );
    assert!((f.dot - df).abs() < 1e-5);
    let g = (v.cos() / (v * v + Dual::constant(2.0)))
        .abs()
        .max(Dual::constant(0.0));
    let gx = x.cos() / (x * x + 2.0);
    let dg = (-x.sin() * (x * x + 2.0) - x.cos() * 2.0 * x) / (x * x + 2.0).powi(2);
    println!(
        "g({x}) = {:.6}, g' = {:.6} (closed form {dg:.6})",
        g.val, g.dot
    );
    assert!((g.val - gx).abs() < 1e-6 && (g.dot - dg).abs() < 1e-5);
    let h = v.min(Dual::constant(0.5)).clamp(-1.0, 1.0);
    assert_eq!((h.val, h.dot), (0.5, 0.0));

    // ── Dual3: value and gradient of primitives in one evaluation ──
    let p = Vec3::new(1.2, 0.9, -0.4);
    let (px, py, pz) = dual3_point(p);
    let sphere = dual3_sphere(px, py, pz, 1.0);
    println!(
        "sphere   d = {:.5}, ∇ = {:?}, |∇| = {:.5}",
        sphere.val,
        sphere.gradient(),
        sphere.gradient_magnitude()
    );
    assert!((sphere.gradient() - p.normalize()).length() < 1e-6);

    let boxed = dual3_box(px, py, pz, Vec3::new(1.0, 0.5, 0.5));
    let q = Vec3::new(0.2, 0.4, 0.0);
    println!("box      d = {:.5}, ∇ = {:?}", boxed.val, boxed.gradient());
    assert!((boxed.val - q.length()).abs() < 1e-6);
    assert!((boxed.gradient() - q.normalize()).length() < 1e-6);

    let torus = dual3_torus(px, py, pz, 1.0, 0.25);
    let plane = dual3_plane(px, py, pz, Vec3::Y, 0.5);
    println!("torus    d = {:.5}, ∇ = {:?}", torus.val, torus.gradient());
    println!("plane    d = {:.5}, ∇ = {:?}", plane.val, plane.gradient());
    assert!((torus.gradient_magnitude() - 1.0).abs() < 1e-5);
    assert_eq!(plane.gradient(), Vec3::Y);

    // combine Dual3 values by hand: a capsule-like |p.xz| - r with a clamp
    let radial = Dual3::length2(px, pz) - Dual3::constant(0.3);
    let shell = radial
        .abs()
        .min(Dual3::length3(px, py, pz).sqrt())
        .max(Dual3::constant(0.0));
    let wrapped = Dual3::from_val_grad(shell.val, shell.gradient()).clamp(0.0, 10.0);
    println!(
        "custom   d = {:.5}, ∇ = {:?}",
        wrapped.val,
        wrapped.gradient()
    );
    assert_eq!(wrapped.gradient(), shell.gradient());

    // ── any SdfNode: distance + analytic gradient ──
    let node = SdfNode::sphere(1.0);
    let (d, grad) = eval_with_gradient(&node, p);
    let dual = eval_dual3(&node, p);
    println!("node     d = {d:.5}, ∇ = {grad:?}");
    assert!((d - (p.length() - 1.0)).abs() < 1e-6);
    assert_eq!(dual.gradient(), grad);

    // ── curvature ──
    let (big_r, small_r) = (2.0_f32, 0.5_f32);
    let ring = SdfNode::torus(big_r, small_r);
    let outer = Vec3::new(big_r + small_r, 0.0, 0.0);
    let inner = Vec3::new(big_r - small_r, 0.0, 0.0);
    let (k1, k2) = principal_curvatures(&ring, outer, 1e-2);
    let k_in = gaussian_curvature(&ring, inner, 1e-2);
    println!(
        "torus outer k = ({k1:.4}, {k2:.4}) (closed form ({:.4}, {:.4}))",
        1.0 / small_r,
        1.0 / (big_r + small_r)
    );
    println!(
        "torus inner K = {k_in:.4} (closed form {:.4}, a saddle)",
        -1.0 / (small_r * (big_r - small_r))
    );
    assert!((k1 - 1.0 / small_r).abs() < 0.04 && (k2 - 1.0 / (big_r + small_r)).abs() < 0.01);
    assert!((k_in + 1.0 / (small_r * (big_r - small_r))).abs() < 0.05);
}
