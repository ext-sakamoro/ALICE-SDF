//! `Polygon2D` as the general closed 2D curve: any vertex count, any concavity, the exact distance and
//! sign, extruded along z.
//!
//! Oracles independent of the implementation:
//! - the 2D distance by brute force over every edge segment, the sign by the winding number (an
//!   angle sum, not the crossing parity the implementation uses);
//! - the square by the box's closed form, the circle (a 256-gon) by `|p| − r` within the sagitta
//!   `r (1 − cos π/n)`;
//! - `|∇d| = 1` off the surface and off the medial axis (central differences);
//! - the extrusion by its closed form over z;
//! - the compiled bytecode equals the tree bit for bit;
//! - a closed spline sampled to a polygon passes through its control points.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm serves as an independent reference"
)]

use alice_sdf::prelude::*;
use glam::Vec2;

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// Distance to the nearest edge segment, and to the nearest edge that does not share a vertex with
/// it (the medial-axis margin: where these two are close, the nearest feature can jump)
fn edge_distances(q: Vec2, v: &[Vec2]) -> (f64, f64) {
    let n = v.len();
    let d: Vec<f64> = (0..n)
        .map(|i| {
            let a = v[i].as_dvec2();
            let b = v[(i + 1) % n].as_dvec2();
            let p = q.as_dvec2();
            let e = b - a;
            let t = ((p - a).dot(e) / e.dot(e)).clamp(0.0, 1.0);
            (p - (a + e * t)).length()
        })
        .collect();
    let best = (0..n).min_by(|&a, &b| d[a].total_cmp(&d[b])).expect("edges");
    let other = (0..n)
        .filter(|&j| j != best && j != (best + 1) % n && j != (best + n - 1) % n)
        .map(|j| d[j])
        .fold(f64::INFINITY, f64::min);
    (d[best], other)
}

/// The nearest feature: (edge, 0 interior / 1 its start vertex / 2 its end vertex), vertices named by
/// the lower edge index so a vertex is one feature
fn nearest_feature(q: Vec2, v: &[Vec2]) -> (usize, u8) {
    let n = v.len();
    let mut best = (f64::INFINITY, (0usize, 0u8));
    for i in 0..n {
        let a = v[i].as_dvec2();
        let b = v[(i + 1) % n].as_dvec2();
        let p = q.as_dvec2();
        let e = b - a;
        let t = (p - a).dot(e) / e.dot(e);
        let (pt, f) = if t <= 0.0 {
            (a, (i, 1))
        } else if t >= 1.0 {
            (b, ((i + 1) % n, 1))
        } else {
            (a + e * t, (i, 0))
        };
        let d = (p - pt).length();
        if d < best.0 {
            best = (d, f);
        }
    }
    best.1
}

/// The winding number by the angle sum
fn winding(q: Vec2, v: &[Vec2]) -> i32 {
    let mut sum = 0.0f64;
    for i in 0..v.len() {
        let a = (v[i] - q).as_dvec2();
        let b = (v[(i + 1) % v.len()] - q).as_dvec2();
        sum += (a.x * b.y - a.y * b.x).atan2(a.dot(b));
    }
    (sum / std::f64::consts::TAU).round() as i32
}

fn reference(q: Vec2, v: &[Vec2]) -> f64 {
    let d = edge_distances(q, v).0;
    if winding(q, v) != 0 {
        -d
    } else {
        d
    }
}

fn l_shape() -> Vec<Vec2> {
    [(0.0, 0.0), (2.0, 0.0), (2.0, 0.6), (0.6, 0.6), (0.6, 1.6), (0.0, 1.6)]
        .iter()
        .map(|&(x, y)| Vec2::new(x - 0.8, y - 0.8))
        .collect()
}

fn star(points: usize, r_out: f32, r_in: f32) -> Vec<Vec2> {
    (0..2 * points)
        .map(|k| {
            let a = std::f32::consts::PI * k as f32 / points as f32;
            let r = if k % 2 == 0 { r_out } else { r_in };
            Vec2::new(r * a.cos(), r * a.sin())
        })
        .collect()
}

fn ngon(n: usize, r: f32) -> Vec<Vec2> {
    (0..n)
        .map(|k| {
            let a = std::f32::consts::TAU * k as f32 / n as f32;
            Vec2::new(r * a.cos(), r * a.sin())
        })
        .collect()
}

/// The 2D field at z = 0 of a tall extrusion (|d₂| < half height, so the slice is the 2D distance)
fn slice(node: &SdfNode, q: Vec2) -> f32 {
    eval(node, Vec3::new(q.x, q.y, 0.0))
}

fn check_against_reference(v: &[Vec2], span: f32, seed: u64) -> f64 {
    let node = SdfNode::polygon_2d(v.to_vec(), 50.0);
    let mut r = lcg(seed);
    let mut worst = 0.0f64;
    for _ in 0..6000 {
        let q = Vec2::new(r() * 2.0 * span - span, r() * 2.0 * span - span);
        worst = worst.max((f64::from(slice(&node, q)) - reference(q, v)).abs());
    }
    worst
}

#[test]
fn the_square_is_the_box_closed_form() {
    let a = 0.5f32;
    let sq = vec![Vec2::new(-a, -a), Vec2::new(a, -a), Vec2::new(a, a), Vec2::new(-a, a)];
    let node = SdfNode::polygon_2d(sq, 50.0);
    let mut r = lcg(11);
    let mut worst = 0.0f32;
    for _ in 0..6000 {
        let q = Vec2::new(r() * 3.0 - 1.5, r() * 3.0 - 1.5);
        let d = q.abs() - Vec2::splat(a);
        let want = d.max(Vec2::ZERO).length() + d.x.max(d.y).min(0.0);
        worst = worst.max((slice(&node, q) - want).abs());
    }
    assert!(worst < 2e-6, "largest difference {worst}");
}

#[test]
fn a_concave_l_shape_matches_the_brute_force_reference() {
    let worst = check_against_reference(&l_shape(), 1.6, 12);
    assert!(worst < 2e-6, "largest difference {worst}");
}

#[test]
fn a_star_matches_the_brute_force_reference() {
    let worst = check_against_reference(&star(7, 1.0, 0.4), 1.4, 13);
    assert!(worst < 2e-6, "largest difference {worst}");
}

#[test]
fn a_256_gon_reads_the_circle_within_its_sagitta() {
    let (n, rad) = (256usize, 0.8f32);
    let node = SdfNode::polygon_2d(ngon(n, rad), 50.0);
    let sag = rad * (1.0 - (std::f32::consts::PI / n as f32).cos());
    let mut r = lcg(14);
    let mut worst = 0.0f32;
    for _ in 0..6000 {
        let q = Vec2::new(r() * 3.0 - 1.5, r() * 3.0 - 1.5);
        worst = worst.max((slice(&node, q) - (q.length() - rad)).abs());
    }
    assert!(worst <= sag + 2e-6, "largest difference {worst}, sagitta {sag}");
    assert!(check_against_reference(&ngon(n, rad), 1.5, 15) < 2e-6);
}

#[test]
fn the_gradient_is_unit_off_the_surface_and_off_the_medial_axis() {
    for (v, span, seed) in [(l_shape(), 1.6f32, 16u64), (star(7, 1.0, 0.4), 1.4, 17), (ngon(64, 0.8), 1.4, 18)] {
        let node = SdfNode::polygon_2d(v.clone(), 50.0);
        let mut r = lcg(seed);
        let h = 1e-3f32;
        let mut checked = 0;
        let mut worst = 0.0f32;
        for _ in 0..8000 {
            let q = Vec2::new(r() * 2.0 * span - span, r() * 2.0 * span - span);
            let (d1, d2) = edge_distances(q, &v);
            // off the surface, off the medial axis between distant edges, and the same nearest feature
            // at all four difference points (a reflex corner's bisector is a kink between adjacent edges)
            let same = [Vec2::new(h, 0.0), Vec2::new(-h, 0.0), Vec2::new(0.0, h), Vec2::new(0.0, -h)]
                .iter()
                .all(|o| nearest_feature(q + *o, &v) == nearest_feature(q, &v));
            if d1 < 0.02 || d2 - d1 < 0.02 || !same {
                continue;
            }
            let gx = (slice(&node, q + Vec2::new(h, 0.0)) - slice(&node, q - Vec2::new(h, 0.0))) / (2.0 * h);
            let gy = (slice(&node, q + Vec2::new(0.0, h)) - slice(&node, q - Vec2::new(0.0, h))) / (2.0 * h);
            worst = worst.max((Vec2::new(gx, gy).length() - 1.0).abs());
            checked += 1;
        }
        assert!(checked > 1000, "only {checked} points off the surface and the medial axis");
        assert!(worst < 2e-3, "largest | |grad| - 1 | {worst}");
    }
}

#[test]
fn the_extrusion_follows_its_closed_form_in_z() {
    let v = star(5, 1.0, 0.45);
    let hh = 0.3f32;
    let node = SdfNode::polygon_2d(v.clone(), hh);
    let mut r = lcg(19);
    let mut worst = 0.0f64;
    for _ in 0..6000 {
        let p = Vec3::new(r() * 3.0 - 1.5, r() * 3.0 - 1.5, r() * 1.6 - 0.8);
        let d2 = reference(Vec2::new(p.x, p.y), &v);
        let dz = f64::from(p.z.abs() - hh);
        let want = d2.max(dz).min(0.0) + (d2.max(0.0).powi(2) + dz.max(0.0).powi(2)).sqrt();
        worst = worst.max((f64::from(eval(&node, p)) - want).abs());
    }
    assert!(worst < 2e-6, "largest difference {worst}");
}

#[test]
fn the_compiled_bytecode_equals_the_tree() {
    for v in [l_shape(), star(7, 1.0, 0.4), ngon(256, 0.8)] {
        let node = SdfNode::polygon_2d(v, 0.4);
        let compiled = alice_sdf::compiled::CompiledSdf::compile(&node);
        let mut r = lcg(20);
        for _ in 0..2000 {
            let p = Vec3::new(r() * 3.0 - 1.5, r() * 3.0 - 1.5, r() * 1.6 - 0.8);
            let a = eval(&node, p);
            let b = alice_sdf::compiled::eval_compiled(&compiled, p);
            assert_eq!(a.to_bits(), b.to_bits(), "at {p:?}: tree {a}, bytecode {b}");
        }
    }
}

#[test]
fn the_analytic_gradient_is_unit_and_agrees_with_differences() {
    for (v, span, seed) in [(l_shape(), 1.6f32, 21u64), (star(7, 1.0, 0.4), 1.4, 22), (ngon(64, 0.8), 1.4, 23)] {
        let node = SdfNode::polygon_2d(v.clone(), 0.5);
        let mut r = lcg(seed);
        let h = 1e-3f32;
        let mut checked = 0;
        let (mut worst_unit, mut worst_fd) = (0.0f32, 0.0f32);
        for _ in 0..8000 {
            let p = Vec3::new(r() * 2.0 * span - span, r() * 2.0 * span - span, r() * 1.6 - 0.8);
            let q = Vec2::new(p.x, p.y);
            let (d1, d2) = edge_distances(q, &v);
            let same = [Vec2::new(h, 0.0), Vec2::new(-h, 0.0), Vec2::new(0.0, h), Vec2::new(0.0, -h)]
                .iter()
                .all(|o| nearest_feature(q + *o, &v) == nearest_feature(q, &v));
            let dz = p.z.abs() - 0.5;
            // off the surface, off both medial sets (in the plane and between side and cap)
            if d1 < 0.02 || d2 - d1 < 0.02 || !same || dz.abs() < 0.02 || (d1 as f32 - dz.abs()).abs() < 0.02 {
                continue;
            }
            let g = alice_sdf::eval::eval_gradient(&node, p);
            let fd = Vec3::new(
                eval(&node, p + Vec3::X * h) - eval(&node, p - Vec3::X * h),
                eval(&node, p + Vec3::Y * h) - eval(&node, p - Vec3::Y * h),
                eval(&node, p + Vec3::Z * h) - eval(&node, p - Vec3::Z * h),
            ) / (2.0 * h);
            worst_unit = worst_unit.max((g.length() - 1.0).abs());
            worst_fd = worst_fd.max((g - fd).length());
            checked += 1;
        }
        assert!(checked > 1000, "only {checked} points");
        assert!(worst_unit < 1e-5, "largest | |grad| - 1 | {worst_unit}");
        assert!(worst_fd < 2e-3, "largest |grad - differences| {worst_fd}");
    }
}

#[test]
fn a_closed_spline_passes_through_its_points_and_keeps_its_sign() {
    // a concave closed outline of 6 points, 8 samples per span: the polygon has 48 vertices, every
    // 8th one a control point, and its inside matches the control polygon's away from the curve
    let pts = l_shape();
    let node = SdfNode::polygon_2d_closed_spline(&pts, 8, 50.0).expect("a closed spline");
    let SdfNode::Polygon2D { vertices, .. } = &node else { panic!("a polygon node") };
    assert_eq!(vertices.len(), pts.len() * 8);
    for (k, p) in pts.iter().enumerate() {
        assert_eq!(vertices[k * 8], *p, "control point {k}");
    }
    assert!(slice(&node, Vec2::new(-0.5, -0.5)) < 0.0);
    assert!(slice(&node, Vec2::new(0.8, 0.8)) > 0.0);
    assert!(SdfNode::polygon_2d_closed_spline(&pts[..2], 8, 1.0).is_err());
    assert!(SdfNode::polygon_2d_closed_spline(&pts, 0, 1.0).is_err());
}

/// The material shader is appended to the main shader: with two polygons above the unroll
/// limit each under its own material, the combined module must carry every module-scope array
/// once, under distinct names, and pass naga's validator
#[cfg(feature = "gpu")]
#[test]
fn the_material_shader_carries_its_polygon_arrays_without_name_clashes() {
    use alice_sdf::compiled::{TranspileMode, WgslShader};
    let a = SdfNode::polygon_2d(ngon(12, 0.3), 0.2).with_material(3);
    let b = SdfNode::polygon_2d(star(7, 0.4, 0.2), 0.1)
        .translate(0.5, 0.0, 0.0)
        .with_material(7);
    let node = a.union(b);
    let main = WgslShader::transpile(&node, TranspileMode::Hardcoded);
    let material = WgslShader::transpile_material(&node, TranspileMode::Hardcoded);
    let src = format!(
        "{}\n{}\n@compute @workgroup_size(1)\nfn main() {{ let m = sdf_eval_material(vec3<f32>(0.0)); let d = sdf_eval(vec3<f32>(0.0)); }}\n",
        main.source, material
    );
    // every array the material functions read is declared exactly once
    let mut names: Vec<&str> = src
        .match_indices("alice_poly_v_")
        .map(|(i, _)| {
            let rest = &src[i..];
            let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap();
            &rest[..end]
        })
        .collect();
    names.sort_unstable();
    names.dedup();
    assert!(names.len() >= 4, "main and material arrays: {names:?}");
    for n in &names {
        let decls = src.matches(&format!("var<private> {n}:")).count();
        assert_eq!(decls, 1, "{n} declared {decls} times");
    }
    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .expect("valid");
}
