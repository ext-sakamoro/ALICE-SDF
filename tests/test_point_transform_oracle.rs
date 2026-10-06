//! Point transforms and the container types vs closed forms.
//!
//! References:
//!
//! * translation: `p - o` / `p + o` (exact in `f32`),
//! * uniform scale: `(p / s, s)` and `p s`; a sphere of radius `r` scaled by
//!   `s` is at distance `|p| - r s`,
//! * rotation: the matrix of the quaternion / of intrinsic X-Y-Z Euler angles
//!   (`Rx(a) Ry(b) Rz(c)`) built in `f64`; the SDF transform applies the
//!   transpose, the forward transform the matrix itself,
//! * tree evaluator: a translated / scaled / rotated sphere has the closed
//!   form distance `|R^T (p - o)| / s * s - r` and the gradient
//!   `(p - c) / |p - c|` (rotation leaves a sphere invariant, so the gradient
//!   must come back in world space),
//! * `Aabb`: centre `(min + max) / 2`, size `max - min`, half extents, the
//!   box test, the union of two boxes (componentwise min / max),
//! * `Ray::at`: `o + t d / |d|`,
//! * sine displacement: `d + a sin(fx x) sin(fy y) sin(fz z)`.
//!
//! Author: Moroya Sakamoto
#![allow(clippy::float_cmp)]

use alice_sdf::eval::eval_gradient;
use alice_sdf::transforms::{
    transform_rotate, transform_rotate_euler, transform_rotate_inverse, transform_scale,
    transform_scale_inverse, transform_translate, transform_translate_inverse,
};
use alice_sdf::types::{Aabb, Ray, SdfMetadata, SdfTree};
use alice_sdf::{eval, SdfNode};
use glam::{Quat, Vec3};

type M = [[f64; 3]; 3];

fn points() -> Vec<Vec3> {
    let mut out = Vec::new();
    for ix in -4..=4 {
        for iy in -4..=4 {
            for iz in -4..=4 {
                out.push(Vec3::new(
                    ix as f32 * 0.53 + 0.019,
                    iy as f32 * 0.47 - 0.031,
                    iz as f32 * 0.41 + 0.007,
                ));
            }
        }
    }
    out
}

fn mul(a: M, b: M) -> M {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}
fn apply(m: M, p: Vec3) -> [f64; 3] {
    let v = [p.x as f64, p.y as f64, p.z as f64];
    [0, 1, 2].map(|i| (0..3).map(|k| m[i][k] * v[k]).sum())
}
fn transpose(m: M) -> M {
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| m[j][i]))
}
fn rx(a: f64) -> M {
    let (s, c) = a.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}
fn ry(a: f64) -> M {
    let (s, c) = a.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}
fn rz(a: f64) -> M {
    let (s, c) = a.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}
/// Rotation matrix of the unit quaternion `(x, y, z, w)`.
fn quat_matrix(q: Quat) -> M {
    let (x, y, z, w) = (q.x as f64, q.y as f64, q.z as f64, q.w as f64);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn close3(what: &str, p: Vec3, got: Vec3, want: [f64; 3], tol: f64) {
    for (g, w) in [got.x, got.y, got.z].into_iter().zip(want) {
        assert!(
            (g as f64 - w).abs() <= tol * (1.0 + w.abs()),
            "{what} at {p:?}: got {got:?}, closed form {want:?}"
        );
    }
}

#[test]
fn translation_and_its_inverse() {
    let mut n = 0;
    for &o in &[Vec3::new(0.5, -1.0, 2.0), Vec3::new(-0.3, 0.25, 0.0)] {
        for p in points() {
            assert_eq!(
                transform_translate(p, o),
                Vec3::new(p.x - o.x, p.y - o.y, p.z - o.z)
            );
            assert_eq!(
                transform_translate_inverse(p, o),
                Vec3::new(p.x + o.x, p.y + o.y, p.z + o.z)
            );
            n += 2;
        }
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn uniform_scale_and_its_inverse() {
    let mut n = 0;
    for &s in &[0.5_f32, 2.0, 3.7] {
        for p in points() {
            let (q, mult) = transform_scale(p, s);
            let ss = s as f64;
            close3(
                "transform_scale",
                p,
                q,
                [p.x as f64 / ss, p.y as f64 / ss, p.z as f64 / ss],
                1e-6,
            );
            assert_eq!(mult, s);
            close3(
                "transform_scale_inverse",
                p,
                transform_scale_inverse(p, s),
                [p.x as f64 * ss, p.y as f64 * ss, p.z as f64 * ss],
                1e-6,
            );
            n += 2;
        }
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn quaternion_and_euler_rotations_match_their_matrices() {
    let quats = [
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        Quat::from_rotation_y(0.7),
        Quat::from_axis_angle(Vec3::new(1.0, 2.0, -0.5).normalize(), 1.3),
    ];
    let mut n = 0;
    for q in quats {
        let m = quat_matrix(q);
        for p in points() {
            close3(
                "transform_rotate",
                p,
                transform_rotate(p, q),
                apply(transpose(m), p),
                1e-5,
            );
            close3(
                "transform_rotate_inverse",
                p,
                transform_rotate_inverse(p, q),
                apply(m, p),
                1e-5,
            );
            n += 2;
        }
    }
    for &(a, b, c) in &[(0.3_f32, -0.8_f32, 1.1_f32), (1.5, 0.2, -0.4)] {
        let m = mul(mul(rx(a as f64), ry(b as f64)), rz(c as f64));
        for p in points() {
            close3(
                "transform_rotate_euler",
                p,
                transform_rotate_euler(p, a, b, c),
                apply(transpose(m), p),
                1e-5,
            );
            n += 1;
        }
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn transformed_sphere_nodes_have_the_closed_form_distance_and_gradient() {
    let r = 0.6_f32;
    let c = Vec3::new(0.4, -0.3, 0.2);
    let s = 1.7_f32;
    let q = Quat::from_axis_angle(Vec3::new(0.3, 1.0, 0.2).normalize(), 0.9);
    let translated = SdfNode::sphere(r).translate_vec(c);
    let scaled = SdfNode::sphere(r).scale(s);
    let rotated = SdfNode::sphere(r).rotate(q).translate_vec(c);
    let mut n = 0;
    for p in points() {
        let dc = ((p.x - c.x) as f64)
            .hypot((p.y - c.y) as f64)
            .hypot((p.z - c.z) as f64);
        let d0 = (p.x as f64).hypot(p.y as f64).hypot(p.z as f64);
        let tol = 1e-5;
        assert!((eval(&translated, p) as f64 - (dc - r as f64)).abs() < tol * (1.0 + dc));
        assert!((eval(&scaled, p) as f64 - (d0 - (r * s) as f64)).abs() < tol * (1.0 + d0));
        assert!((eval(&rotated, p) as f64 - (dc - r as f64)).abs() < tol * (1.0 + dc));
        if dc > 0.2 {
            let g = eval_gradient(&rotated, p);
            let want = [
                (p.x - c.x) as f64 / dc,
                (p.y - c.y) as f64 / dc,
                (p.z - c.z) as f64 / dc,
            ];
            close3("gradient of rotated sphere", p, g, want, 2e-3);
            let gt = eval_gradient(&translated, p);
            close3("gradient of translated sphere", p, gt, want, 2e-3);
            n += 2;
        }
        n += 3;
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn aabb_accessors_and_union() {
    let a = Aabb::from_center_extents(Vec3::new(1.0, -2.0, 0.5), Vec3::new(0.5, 1.5, 2.0));
    assert_eq!(a.min, Vec3::new(0.5, -3.5, -1.5));
    assert_eq!(a.max, Vec3::new(1.5, -0.5, 2.5));
    assert_eq!(a.center(), Vec3::new(1.0, -2.0, 0.5));
    assert_eq!(a.size(), Vec3::new(1.0, 3.0, 4.0));
    assert_eq!(a.half_extents(), Vec3::new(0.5, 1.5, 2.0));
    let b = Aabb::new(Vec3::new(-1.0, -1.0, 0.0), Vec3::new(0.0, 4.0, 1.0));
    let u = a.union(&b);
    assert_eq!(u.min, Vec3::new(-1.0, -3.5, -1.5));
    assert_eq!(u.max, Vec3::new(1.5, 4.0, 2.5));
    let mut n = 0;
    for p in points() {
        let inside = (a.min.x..=a.max.x).contains(&p.x)
            && (a.min.y..=a.max.y).contains(&p.y)
            && (a.min.z..=a.max.z).contains(&p.z);
        assert_eq!(a.contains(p), inside, "{p:?}");
        // the union contains everything either box contains
        if a.contains(p) || b.contains(p) {
            assert!(u.contains(p));
        }
        n += 1;
    }
    // corners are inside (closed box)
    assert!(a.contains(a.min) && a.contains(a.max));
    assert!(n > 0, "no comparisons made");
}

#[test]
fn ray_at_walks_along_the_normalised_direction() {
    let o = Vec3::new(1.0, 2.0, -3.0);
    let ray = Ray::new(o, Vec3::new(0.0, 3.0, 4.0));
    let mut n = 0;
    for i in -4..=8 {
        let t = i as f32 * 0.75;
        close3(
            "Ray::at",
            o,
            ray.at(t),
            [1.0, 2.0 + 0.6 * t as f64, -3.0 + 0.8 * t as f64],
            1e-6,
        );
        n += 1;
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn tree_with_metadata_keeps_root_metadata_and_version() {
    let root = SdfNode::sphere(1.0).union(SdfNode::box3d(1.0, 1.0, 1.0));
    let meta = SdfMetadata {
        name: Some("pair".into()),
        author: Some("test".into()),
        ..SdfMetadata::default()
    };
    let tree = SdfTree::with_metadata(root.clone(), meta);
    assert_eq!(tree.version, env!("CARGO_PKG_VERSION"));
    assert_eq!(tree.node_count(), 3);
    let m = tree.metadata.as_ref().expect("metadata is kept");
    assert_eq!(m.name.as_deref(), Some("pair"));
    assert_eq!(m.author.as_deref(), Some("test"));
    assert_eq!(
        eval(&tree.root, Vec3::new(2.0, 0.0, 0.0)),
        eval(&root, Vec3::new(2.0, 0.0, 0.0))
    );
}

#[test]
fn sine_displacement_is_the_product_of_axis_sines() {
    let (amp, f) = (0.1_f32, Vec3::new(3.0, 7.0, 5.0));
    let aniso = SdfNode::sphere(1.0).sine_displacement_aniso(amp, f);
    let iso = SdfNode::sphere(1.0).sine_displacement(amp, 4.0);
    let iso_as_aniso = SdfNode::sphere(1.0).sine_displacement_aniso(amp, Vec3::splat(4.0));
    let mut n = 0;
    for p in points() {
        let (x, y, z) = (p.x as f64, p.y as f64, p.z as f64);
        let base = x.hypot(y).hypot(z) - 1.0;
        let want = base
            + amp as f64 * (f.x as f64 * x).sin() * (f.y as f64 * y).sin() * (f.z as f64 * z).sin();
        let got = eval(&aniso, p) as f64;
        assert!(
            (got - want).abs() < 1e-5 * (1.0 + want.abs()),
            "{p:?}: {got} vs {want}"
        );
        assert_eq!(eval(&iso, p), eval(&iso_as_aniso, p));
        n += 2;
    }
    assert!(n > 0, "no comparisons made");
}
