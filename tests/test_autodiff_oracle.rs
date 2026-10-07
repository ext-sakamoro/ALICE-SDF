//! Forward-mode dual numbers and SDF curvature against closed forms.
//!
//! Every expected value is written from calculus, in f64, without calling the
//! code under test:
//! - `Dual`: product / quotient / chain rule for sin, cos, sqrt, abs, and the
//!   selection rules of min / max / clamp
//! - `Dual3` primitives: ∇|p| = p/|p| for the sphere, the face normal (or the
//!   normalised positive part of q) for the box, the torus tube direction,
//!   and the plane normal
//! - curvature: sphere k1 = k2 = 1/r, torus outer equator (1/r, 1/(R+r)) and
//!   inner equator (1/r, -1/(R-r)), Gaussian curvature = k1·k2
//!
//! Author: Moroya Sakamoto

use alice_sdf::autodiff::{
    dual3_box, dual3_plane, dual3_point, dual3_sphere, dual3_torus, eval_dual3, eval_with_gradient,
    gaussian_curvature, principal_curvatures, Dual, Dual3,
};
use alice_sdf::types::SdfNode;
use glam::Vec3;

fn close(a: f32, b: f64, tol: f64) -> bool {
    (f64::from(a) - b).abs() <= tol * (1.0 + b.abs())
}

#[test]
fn dual_number_rules_match_calculus() {
    let xs = [-2.5_f64, -1.0, -0.3, 0.2, 0.7, 1.5, 3.0];
    let mut compared = 0;
    for &x in &xs {
        let v = Dual::variable(x as f32);
        let c = Dual::constant(1.75);
        assert_eq!(v.dot, 1.0);
        assert_eq!(c.dot, 0.0);

        // f = x · sin(x): f' = sin x + x cos x
        let f = v * v.sin();
        assert!(close(f.val, x * x.sin(), 1e-6));
        assert!(close(f.dot, x.sin() + x * x.cos(), 1e-5), "x={x}");

        // f = cos(x) / (x² + 1.75): quotient rule
        let den = v * v + c;
        let g = v.cos() / den;
        let d = x * x + 1.75;
        let g_dot = (-x.sin() * d - x.cos() * 2.0 * x) / (d * d);
        assert!(close(g.val, x.cos() / d, 1e-6));
        assert!(close(g.dot, g_dot, 1e-5), "x={x}");

        // f = sqrt(x² + 1.75) · 3: chain rule, f' = 3x / sqrt(x² + 1.75)
        let h = den.sqrt() * 3.0;
        assert!(close(h.val, 3.0 * d.sqrt(), 1e-6));
        assert!(close(h.dot, 3.0 * x / d.sqrt(), 1e-5));

        // |-x| = |x|: derivative sign(x) (all lattice x are non-zero)
        let a = (-v).abs();
        assert_eq!(a.val, (x as f32).abs());
        assert_eq!(a.dot, if x > 0.0 { 1.0 } else { -1.0 });

        // min / max / clamp select one branch with its derivative
        let m = v.min(c);
        let mx = v.max(c);
        assert_eq!(m.dot, if x <= 1.75 { 1.0 } else { 0.0 });
        assert_eq!(mx.dot, if x >= 1.75 { 1.0 } else { 0.0 });
        let cl = v.clamp(-1.0, 1.0);
        let inside = (-1.0..=1.0).contains(&x);
        assert_eq!(cl.val, (x as f32).clamp(-1.0, 1.0));
        assert_eq!(cl.dot, if inside { 1.0 } else { 0.0 });
        compared += 1;
    }
    assert_eq!(compared, xs.len());
}

#[test]
fn dual3_primitive_gradients_match_closed_forms() {
    let pts = [
        Vec3::new(2.0, 0.5, -1.0),
        Vec3::new(-0.3, 1.7, 0.4),
        Vec3::new(0.25, -0.5, 2.5),
        Vec3::new(-1.5, -1.5, 0.75),
    ];
    let mut compared = 0;
    for &p in &pts {
        let (px, py, pz) = dual3_point(p);
        assert_eq!(px.gradient(), Vec3::X);
        assert_eq!(py.gradient(), Vec3::Y);
        assert_eq!(pz.gradient(), Vec3::Z);
        let (x, y, z) = (f64::from(p.x), f64::from(p.y), f64::from(p.z));
        let len = (x * x + y * y + z * z).sqrt();

        // sphere: f = |p| - r, ∇f = p / |p|
        let s = dual3_sphere(px, py, pz, 0.8);
        assert!(close(s.val, len - 0.8, 1e-6));
        assert!(close(s.dx, x / len, 1e-6));
        assert!(close(s.dy, y / len, 1e-6));
        assert!(close(s.dz, z / len, 1e-6));
        assert!(close(s.gradient_magnitude(), 1.0, 1e-6));

        // plane: f = n·p - d, ∇f = n
        let n = Vec3::new(0.0, 0.6, 0.8);
        let pl = dual3_plane(px, py, pz, n, 0.3);
        assert!(close(pl.val, 0.6 * y + 0.8 * z - 0.3, 1e-6));
        assert_eq!(pl.gradient(), n);

        // torus: q = (|p.xz| - R, p.y), ∇f = (q.x·p.x/|p.xz|, q.y, q.x·p.z/|p.xz|)/|q|
        let (big_r, small_r) = (1.2_f64, 0.3_f64);
        let t = dual3_torus(px, py, pz, big_r as f32, small_r as f32);
        let rxz = (x * x + z * z).sqrt();
        let qx = rxz - big_r;
        let ql = (qx * qx + y * y).sqrt();
        assert!(close(t.val, ql - small_r, 1e-6));
        assert!(close(t.dx, qx * x / rxz / ql, 1e-5));
        assert!(close(t.dy, y / ql, 1e-5));
        assert!(close(t.dz, qx * z / rxz / ql, 1e-5));
        compared += 1;
    }
    assert_eq!(compared, pts.len());
}

#[test]
fn dual3_box_gradient_is_face_normal_or_corner_direction() {
    let half = Vec3::new(1.0, 0.5, 0.25);
    // (point, expected gradient) — face regions give the face normal, the
    // exterior corner region gives normalize(max(q, 0)), and inside the
    // gradient is the axis of the largest q
    let q_corner = Vec3::new(0.5, 0.25, 0.75);
    let cases = [
        (Vec3::new(1.5, 0.1, 0.0), Vec3::X),
        (Vec3::new(-0.2, -0.9, 0.1), -Vec3::Y),
        (Vec3::new(0.3, 0.2, -0.6), -Vec3::Z),
        (Vec3::new(1.5, 0.75, 1.0), q_corner / q_corner.length()),
        (Vec3::new(0.9, 0.0, 0.0), Vec3::X),
        (Vec3::new(0.0, -0.45, 0.0), -Vec3::Y),
    ];
    for (p, g) in cases {
        let (px, py, pz) = dual3_point(p);
        let d = dual3_box(px, py, pz, half);
        let got = d.gradient();
        assert!((got - g).length() < 1e-6, "p={p:?} got={got:?} want={g:?}");
        // value: exterior distance |max(q,0)| + interior min(max q, 0)
        let q = p.abs() - half;
        let want = f64::from(q.max(Vec3::ZERO).length()) + f64::from(q.max_element().min(0.0));
        assert!(close(d.val, want, 1e-6));
    }
}

#[test]
fn dual3_helpers_follow_their_definitions() {
    let g = Vec3::new(3.0, -4.0, 12.0);
    let d = Dual3::from_val_grad(2.0, g);
    assert_eq!(d.gradient(), g);
    assert!(close(d.gradient_magnitude(), 13.0, 1e-7));
    let c = Dual3::constant(5.0);
    assert_eq!(c.gradient(), Vec3::ZERO);

    let neg = Dual3::from_val_grad(-2.0, g);
    assert_eq!(neg.abs().val, 2.0);
    assert_eq!(neg.abs().gradient(), -g);
    assert_eq!(d.abs().gradient(), g);

    assert_eq!(d.min(neg).val, -2.0);
    assert_eq!(d.max(neg).gradient(), g);

    // clamp outside the range freezes the derivative
    assert_eq!(d.clamp(-1.0, 1.0).val, 1.0);
    assert_eq!(d.clamp(-1.0, 1.0).gradient(), Vec3::ZERO);
    assert_eq!(d.clamp(0.0, 3.0).gradient(), g);

    // sqrt(f): ∇ = ∇f / (2 sqrt f)
    let four = Dual3::from_val_grad(4.0, g);
    let r = four.sqrt();
    assert_eq!(r.val, 2.0);
    assert_eq!(r.gradient(), g / 4.0);

    // length2 / length3 over the seed triple equal |(x,z)| and |p|
    let (px, py, pz) = dual3_point(Vec3::new(3.0, 12.0, 4.0));
    let l2 = Dual3::length2(px, pz);
    assert_eq!(l2.val, 5.0);
    assert!((l2.gradient() - Vec3::new(0.6, 0.0, 0.8)).length() < 1e-7);
    let l3 = Dual3::length3(px, py, pz);
    assert_eq!(l3.val, 13.0);
    assert!((l3.gradient() - Vec3::new(3.0, 12.0, 4.0) / 13.0).length() < 1e-7);
}

#[test]
fn eval_with_gradient_on_a_sphere_is_distance_and_radial_direction() {
    let sphere = SdfNode::sphere(1.5);
    let mut compared = 0;
    for p in [
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.3, -2.2, 1.1),
        Vec3::new(-0.4, 0.2, 0.5),
    ] {
        let (d, g) = eval_with_gradient(&sphere, p);
        let len = f64::from(p.length());
        assert!(close(d, len - 1.5, 1e-6));
        assert!((g - p / p.length()).length() < 1e-5, "p={p:?} g={g:?}");
        let dd = eval_dual3(&sphere, p);
        assert_eq!(dd.val, d);
        assert_eq!(dd.gradient(), g);
        compared += 1;
    }
    assert_eq!(compared, 3);
}

/// oracle: differential geometry of the sphere and the torus.
/// The Hessian is a central difference of the analytic gradient, so the
/// tolerance is the O(ε²·k³) truncation plus f32 cancellation (ε = 1e-2 keeps
/// the cancellation term ~1e-5 / 1e-2 = 1e-3 below the 2 % used here).
#[test]
fn principal_curvatures_match_sphere_and_torus() {
    let eps = 1e-2;
    let tol = 0.02;
    let mut compared = 0;

    for r in [0.5_f32, 1.0, 2.0] {
        let sphere = SdfNode::sphere(r);
        for p in [
            Vec3::new(r, 0.0, 0.0),
            Vec3::new(0.0, -r, 0.0),
            Vec3::new(1.0, 1.0, 1.0).normalize() * r,
        ] {
            let (k1, k2) = principal_curvatures(&sphere, p, eps);
            let k = 1.0 / f64::from(r);
            assert!(
                close(k1, k, tol) && close(k2, k, tol),
                "r={r} p={p:?} k=({k1},{k2})"
            );
            let kg = gaussian_curvature(&sphere, p, eps);
            assert!(close(kg, k * k, 2.0 * tol), "r={r} K={kg}");
            compared += 1;
        }
    }

    let (big_r, small_r) = (2.0_f32, 0.5_f32);
    let torus = SdfNode::torus(big_r, small_r);
    // outer equator: tube curvature 1/r and 1/(R + r) around the axis
    let (k1, k2) = principal_curvatures(&torus, Vec3::new(big_r + small_r, 0.0, 0.0), eps);
    assert!(close(k1, 1.0 / f64::from(small_r), tol), "k1={k1}");
    assert!(close(k2, 1.0 / f64::from(big_r + small_r), tol), "k2={k2}");
    compared += 1;
    // inner equator: saddle, the second curvature is -1/(R - r)
    let p_in = Vec3::new(0.0, 0.0, big_r - small_r);
    let (k1, k2) = principal_curvatures(&torus, p_in, eps);
    assert!(close(k1, 1.0 / f64::from(small_r), tol), "k1={k1}");
    assert!(close(k2, -1.0 / f64::from(big_r - small_r), tol), "k2={k2}");
    let kg = gaussian_curvature(&torus, p_in, eps);
    assert!(kg < 0.0);
    assert!(close(
        kg,
        -1.0 / f64::from(small_r * (big_r - small_r)),
        2.0 * tol
    ));
    compared += 1;

    assert_eq!(compared, 11);
}
