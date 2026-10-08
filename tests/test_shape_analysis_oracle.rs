//! Closed-form oracles for the analysis modules: `measure` (volume, surface
//! area, center of mass), `heatmap` (cross-section slices and colour maps),
//! `shell` (variable-thickness offset surface), `morphology` (signed offset and
//! tolerance fit) and the Lipschitz rule `RaymarchConfig::relaxed` takes from
//! `fidelity`.
//!
//! Every expected value comes from geometry (a ball's volume, an annulus's
//! distance, a circle's pixel count), never from calling the implementation.
//! Monte Carlo estimates are checked against the closed form with a tolerance
//! of 4 standard errors, where the standard error is derived here from the
//! closed-form fill ratio, not read back from the estimate.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]

use alice_sdf::compiled::CompiledSdf;
use alice_sdf::eval::eval;
use alice_sdf::heatmap::{generate_heatmap, heatmap_to_rgba, ColorMap, HeatmapConfig, SlicePlane};
use alice_sdf::measure::{estimate_center_of_mass, estimate_surface_area, estimate_volume};
use alice_sdf::morphology::{
    eval_offset, eval_offset_batch, eval_offset_batch_parallel, tolerance_fits,
    tolerance_max_violation,
};
use alice_sdf::raycast::RaymarchConfig;
use alice_sdf::shell::{
    eval_shell, eval_shell_batch, eval_shell_batch_parallel, eval_shell_compiled,
    eval_shell_compiled_batch_parallel, eval_shell_gradient, shell_node, ShellConfig,
};
use alice_sdf::types::{Aabb, SdfNode};
use glam::Vec3;
use std::f64::consts::PI;
use std::sync::Arc;

fn cube(half: f32) -> Aabb {
    Aabb {
        min: Vec3::splat(-half),
        max: Vec3::splat(half),
    }
}

/// 4σ band of a binomial-proportion estimate of `exact`, where `exact` is the
/// closed-form quantity and `scale` converts the proportion to it.
fn within_4_sigma(estimate: f64, exact: f64, scale: f64, samples: u64) -> bool {
    let p = exact / scale;
    let sigma = (p * (1.0 - p) / samples as f64).sqrt() * scale;
    (estimate - exact).abs() <= 4.0 * sigma
}

// ── measure ─────────────────────────────────────────────────

#[test]
fn ball_volume_matches_four_thirds_pi_r_cubed() {
    let mut compared = 0;
    for &r in &[0.5_f32, 1.0, 1.7] {
        let half = r * 1.2;
        let samples = 200_000;
        let est = estimate_volume(&SdfNode::sphere(r), cube(half), samples, 7);
        let exact = 4.0 / 3.0 * PI * f64::from(r).powi(3);
        let box_volume = f64::from(2.0 * half).powi(3);
        assert!(
            within_4_sigma(est.volume, exact, box_volume, samples),
            "r={r}: estimate {} vs 4/3·π·r³ = {exact}",
            est.volume
        );
        assert_eq!(est.sample_count, samples);
        assert!((est.fill_ratio - exact / box_volume).abs() < 0.01);
        compared += 1;
    }
    assert_eq!(compared, 3);
}

#[test]
fn box_volume_is_the_product_of_its_sides() {
    // full sizes 1 × 2 × 3 → volume 6; sampled in a 4 × 4 × 4 cube
    let b = SdfNode::box3d(1.0, 2.0, 3.0);
    let samples = 200_000;
    let est = estimate_volume(&b, cube(2.0), samples, 11);
    assert!(
        within_4_sigma(est.volume, 6.0, 64.0, samples),
        "estimate {} vs 6",
        est.volume
    );
}

#[test]
fn sphere_surface_area_matches_four_pi_r_squared() {
    // the ε-layer around a sphere of radius r has volume 4π/3·((r+ε)³ − (r−ε)³)
    // = 4π(2r²ε + 2ε³/3), so the method's exact expectation is 4π(r² + ε²/3)
    let (r, eps) = (1.0_f32, 0.02_f32);
    let samples = 400_000;
    let half = 1.2_f32;
    let est = estimate_surface_area(&SdfNode::sphere(r), cube(half), samples, eps, 3);
    let (r, e) = (f64::from(r), f64::from(eps));
    let exact = 4.0 * PI * (r * r + e * e / 3.0);
    let box_volume = f64::from(2.0 * half).powi(3);
    // proportion → area factor is box_volume / 2ε
    assert!(
        within_4_sigma(est.area, exact, box_volume / (2.0 * e), samples),
        "estimate {} vs 4π(r² + ε²/3) = {exact}",
        est.area
    );
    assert!((est.area - 4.0 * PI).abs() / (4.0 * PI) < 0.04);
}

#[test]
fn center_of_mass_of_a_translated_ball_is_its_center() {
    let c = Vec3::new(0.3, -0.2, 0.5);
    let ball = SdfNode::sphere(1.0).translate(c.x, c.y, c.z);
    let aabb = Aabb {
        min: c - Vec3::splat(1.1),
        max: c + Vec3::splat(1.1),
    };
    let com = estimate_center_of_mass(&ball, aabb, 200_000, 5);
    // a coordinate of a uniform unit ball has variance r²/5
    let sigma = (0.2 / com.interior_count as f64).sqrt() as f32;
    assert!(com.interior_count > 50_000);
    for axis in 0..3 {
        assert!(
            (com.center[axis] - c[axis]).abs() <= 4.0 * sigma,
            "axis {axis}: {} vs {}",
            com.center[axis],
            c[axis]
        );
    }
}

// ── heatmap ─────────────────────────────────────────────────

/// Pixel-centre coordinates of a slice, computed independently of the crate.
fn pixel_uv(res: u32, range: f32) -> Vec<(u32, u32, f32, f32)> {
    let step = 2.0 * range / res as f32;
    let mut out = Vec::new();
    for iy in 0..res {
        for ix in 0..res {
            let u = (ix as f32 + 0.5) * step - range;
            let v = (iy as f32 + 0.5) * step - range;
            out.push((ix, iy, u, v));
        }
    }
    out
}

#[test]
fn xy_slice_of_a_sphere_is_the_circle_distance() {
    let (res, range) = (64_u32, 1.5_f32);
    let config = HeatmapConfig {
        resolution: res,
        range,
        plane: SlicePlane::XY(0.0),
    };
    let hm = generate_heatmap(&SdfNode::sphere(1.0), &config);
    assert_eq!((hm.width, hm.height), (res, res));

    let (mut inside, mut compared) = (0_u32, 0_usize);
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for (ix, iy, u, v) in pixel_uv(res, range) {
        let exact = (u * u + v * v).sqrt() - 1.0;
        let got = hm.sample(ix, iy);
        assert!((got - exact).abs() < 1e-5, "({ix},{iy}): {got} vs {exact}");
        if exact < 0.0 {
            inside += 1;
        }
        lo = lo.min(exact);
        hi = hi.max(exact);
        compared += 1;
    }
    assert_eq!(compared, (res * res) as usize);
    assert_eq!(hm.inside_pixel_count(), inside);
    assert!((hm.min_val - lo).abs() < 1e-5 && (hm.max_val - hi).abs() < 1e-5);
    // outside the image the sample is "infinitely far"
    assert_eq!(hm.sample(res, 0), f32::MAX);
    assert_eq!(hm.sample(0, res), f32::MAX);
}

#[test]
fn off_centre_slices_cut_circles_of_radius_sqrt_one_minus_h_squared() {
    // a plane at height h cuts the unit sphere in a circle of radius √(1 − h²);
    // the field on the plane is √(u² + v² + h²) − 1
    let (res, range, h) = (48_u32, 1.2_f32, 0.6_f32);
    let planes = [
        SlicePlane::XY(h),
        SlicePlane::XZ(h),
        SlicePlane::YZ(h),
        // normal +Z: the basis is right = Y × Z = X, forward = Z × X = Y
        SlicePlane::Custom {
            origin: Vec3::new(0.0, 0.0, h),
            normal: Vec3::new(0.0, 0.0, 2.0),
        },
    ];
    let mut compared = 0;
    for plane in planes {
        let hm = generate_heatmap(
            &SdfNode::sphere(1.0),
            &HeatmapConfig {
                resolution: res,
                range,
                plane,
            },
        );
        let mut inside = 0;
        let mut surface = 0;
        for (ix, iy, u, v) in pixel_uv(res, range) {
            let exact = (u * u + v * v + h * h).sqrt() - 1.0;
            assert!((hm.sample(ix, iy) - exact).abs() < 1e-5, "{plane:?}");
            if u * u + v * v < 1.0 - h * h {
                inside += 1;
            }
            if exact.abs() < 0.05 {
                surface += 1;
            }
            compared += 1;
        }
        assert_eq!(hm.inside_pixel_count(), inside, "{plane:?}");
        assert_eq!(hm.surface_pixel_count(0.05), surface, "{plane:?}");
    }
    assert_eq!(compared, 4 * (res * res) as usize);
}

/// Where a slice puts pixel `(u, v)` for a plane at offset `h`.
type PixelToWorld = fn(f32, f32, f32) -> Vec3;

#[test]
fn each_plane_maps_pixels_to_its_own_axes() {
    // an off-centre ball tells the axes apart: the field at world point q is
    // |q − c| − 0.8, and each plane puts pixel (u, v) at a stated q
    let c = Vec3::new(0.3, -0.2, 0.1);
    let ball = SdfNode::sphere(0.8).translate(c.x, c.y, c.z);
    let (res, range, h) = (40_u32, 1.2_f32, 0.25_f32);
    let cases: [(SlicePlane, PixelToWorld); 5] = [
        (SlicePlane::XY(h), |u, v, h| Vec3::new(u, v, h)),
        (SlicePlane::XZ(h), |u, v, h| Vec3::new(u, h, v)),
        (SlicePlane::YZ(h), |u, v, h| Vec3::new(h, u, v)),
        // normal +Z: up = Y, right = Y × Z = X, forward = Z × X = Y
        (
            SlicePlane::Custom {
                origin: Vec3::new(0.0, 0.0, h),
                normal: Vec3::new(0.0, 0.0, 3.0),
            },
            |u, v, h| Vec3::new(u, v, h),
        ),
        // normal +X: up = Y, right = Y × X = −Z, forward = X × (−Z) = Y
        (
            SlicePlane::Custom {
                origin: Vec3::new(h, 0.0, 0.0),
                normal: Vec3::X,
            },
            |u, v, h| Vec3::new(h, v, -u),
        ),
    ];
    let mut compared = 0;
    for (plane, at) in cases {
        let hm = generate_heatmap(
            &ball,
            &HeatmapConfig {
                resolution: res,
                range,
                plane,
            },
        );
        for (ix, iy, u, v) in pixel_uv(res, range) {
            let exact = (at(u, v, h) - c).length() - 0.8;
            let got = hm.sample(ix, iy);
            assert!(
                (got - exact).abs() < 1e-5,
                "{plane:?} ({ix},{iy}): {got} vs {exact}"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 5 * (res * res) as usize);
}

#[test]
fn colour_maps_follow_the_sign_and_the_order_of_the_distance() {
    let hm = generate_heatmap(
        &SdfNode::sphere(1.0),
        &HeatmapConfig {
            resolution: 32,
            range: 1.5,
            plane: SlicePlane::XY(0.0),
        },
    );
    let cool = heatmap_to_rgba(&hm, ColorMap::Coolwarm);
    let binary = heatmap_to_rgba(&hm, ColorMap::Binary);
    let viridis = heatmap_to_rgba(&hm, ColorMap::Viridis);
    let magma = heatmap_to_rgba(&hm, ColorMap::Magma);
    assert_eq!(cool.len(), hm.pixels.len());

    // the corner is the farthest pixel: t = +1 → pure red, binary black
    let far = hm
        .pixels
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(cool[far], [255, 0, 0, 255]);
    assert_eq!(binary[far], [0, 0, 0, 255]);

    let mut order: Vec<usize> = (0..hm.pixels.len()).collect();
    order.sort_by(|&a, &b| hm.pixels[a].total_cmp(&hm.pixels[b]));
    for (i, &d) in hm.pixels.iter().enumerate() {
        let [r, g, b, a] = cool[i];
        assert_eq!(a, 255);
        if d < 0.0 {
            assert!(
                b == 255 && r == g,
                "inside must be blue-white, got {:?}",
                cool[i]
            );
        } else {
            assert!(
                r == 255 && g == b,
                "outside must be red-white, got {:?}",
                cool[i]
            );
        }
        let [br, bg, bb, _] = binary[i];
        assert!(br == bg && bg == bb);
    }
    // perceptual maps are monotone in the distance (red channel rises with t)
    for w in order.windows(2) {
        assert!(viridis[w[0]][0] <= viridis[w[1]][0]);
        assert!(magma[w[0]][0] <= magma[w[1]][0]);
    }
}

// ── shell ───────────────────────────────────────────────────

/// Exact SDF of the spherical annulus `1 − inner ≤ |p| ≤ 1 + outer`.
fn annulus(p: Vec3, inner: f32, outer: f32) -> f32 {
    let r = p.length();
    (r - (1.0 + outer)).max((1.0 - inner) - r)
}

fn radial_points() -> Vec<Vec3> {
    let dirs = [
        Vec3::X,
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(1.0, 1.0, 1.0).normalize(),
        Vec3::new(-0.3, 0.8, 0.52).normalize(),
    ];
    let mut pts = Vec::new();
    for d in dirs {
        for i in 0..=40 {
            pts.push(d * (0.05 * i as f32));
        }
    }
    pts
}

#[test]
fn shell_config_constructors_state_their_thickness() {
    let s = ShellConfig::new(0.1, 0.3);
    assert_eq!((s.inner_offset, s.outer_offset), (0.1, 0.3));
    assert!((s.wall_thickness() - 0.4).abs() < 1e-7);
    let u = ShellConfig::uniform(0.25);
    assert_eq!((u.inner_offset, u.outer_offset), (0.25, 0.25));
    assert!((u.wall_thickness() - 0.5).abs() < 1e-7);
}

#[test]
fn shell_of_a_sphere_is_the_annulus_on_every_path() {
    let sphere = SdfNode::sphere(1.0);
    let compiled = CompiledSdf::compile(&sphere);
    let pts = radial_points();
    let mut compared = 0;
    for &(inner, outer) in &[(0.1_f32, 0.3_f32), (0.2, 0.2), (0.05, 0.0)] {
        let cfg = ShellConfig::new(inner, outer);
        let batch = eval_shell_batch(&sphere, &pts, &cfg);
        let par = eval_shell_batch_parallel(&sphere, &pts, &cfg);
        let cpar = eval_shell_compiled_batch_parallel(&compiled, &pts, &cfg);
        for (i, &p) in pts.iter().enumerate() {
            let exact = annulus(p, inner, outer);
            for (name, got) in [
                ("eval_shell", eval_shell(&sphere, p, &cfg)),
                ("compiled", eval_shell_compiled(&compiled, p, &cfg)),
                ("batch", batch[i]),
                ("batch_parallel", par[i]),
                ("compiled_batch_parallel", cpar[i]),
            ] {
                assert!(
                    (got - exact).abs() < 1e-5,
                    "{name} at {p:?} (inner {inner}, outer {outer}): {got} vs {exact}"
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 3 * 5 * radial_points().len());
}

#[test]
fn shell_gradient_points_away_from_the_wall() {
    // outside the annulus the outward normal is radial, inside its hole it is
    // the inward radial direction
    let sphere = SdfNode::sphere(1.0);
    let cfg = ShellConfig::new(0.1, 0.3);
    let mut compared = 0;
    for d in [Vec3::X, Vec3::new(1.0, -2.0, 0.5).normalize()] {
        let out = eval_shell_gradient(&sphere, d * 1.6, &cfg).normalize();
        assert!((out - d).length() < 1e-3, "outer side: {out:?} vs {d:?}");
        let hole = eval_shell_gradient(&sphere, d * 0.5, &cfg).normalize();
        assert!(
            (hole + d).length() < 1e-3,
            "inner side: {hole:?} vs {:?}",
            -d
        );
        compared += 2;
    }
    assert_eq!(compared, 4);
}

#[test]
fn asymmetric_shell_node_is_the_band_from_r_minus_inner_to_r_plus_outer() {
    // oracle: the annulus 0.9 ≤ |p| ≤ 1.3 for inner 0.1 / outer 0.3 — negative
    // inside the band, zero on 0.9 and 1.3, the radial distance outside
    let sphere = SdfNode::sphere(1.0);
    let cfg = ShellConfig::new(0.1, 0.3);
    let node = shell_node(Arc::new(sphere.clone()), cfg);
    let table = [
        (0.8_f32, 0.1_f32),
        (0.9, 0.0),
        (1.0, -0.1),
        (1.1, -0.2),
        (1.2, -0.1),
        (1.3, 0.0),
        (1.4, 0.1),
    ];
    let mut compared = 0;
    for (r, exact) in table {
        for d in [Vec3::X, Vec3::new(-0.6, 0.0, 0.8)] {
            let p = d * r;
            let got = eval(&node, p);
            assert!(
                (got - exact).abs() < 1e-5,
                "r={r}: shell_node {got} vs {exact}"
            );
            assert!((got - eval_shell(&sphere, p, &cfg)).abs() < 1e-5, "r={r}");
            compared += 1;
        }
    }
    assert_eq!(compared, 2 * table.len());
    // and the whole radial sweep agrees with the annulus
    for p in radial_points() {
        assert!(
            (eval(&node, p) - annulus(p, 0.1, 0.3)).abs() < 1e-5,
            "{p:?}"
        );
    }
}

#[test]
fn asymmetric_shell_node_of_a_box_moves_each_face_band() {
    // in front of a face of a box with half-extent 1, d = x − 1, so the band
    // is 1 − inner ≤ x ≤ 1 + outer and the field is max(x − 1 − outer, 1 − inner − x)
    let b = SdfNode::box3d(2.0, 2.0, 2.0);
    for &(inner, outer) in &[(0.05_f32, 0.4_f32), (0.3, 0.1)] {
        let cfg = ShellConfig::new(inner, outer);
        let node = shell_node(Arc::new(b.clone()), cfg);
        let mut compared = 0;
        for i in 0..=30 {
            let x = 0.75 + 0.03 * i as f32;
            let p = Vec3::new(x, 0.1, -0.2);
            let exact = (x - 1.0 - outer).max(1.0 - inner - x);
            let got = eval(&node, p);
            assert!(
                (got - exact).abs() < 1e-5,
                "x={x} ({inner},{outer}): {got} vs {exact}"
            );
            assert!((got - eval_shell(&b, p, &cfg)).abs() < 1e-5);
            compared += 1;
        }
        assert_eq!(compared, 31);
    }
}

#[test]
fn symmetric_shell_node_is_the_onion_band() {
    // inner = outer = t: |d| − t, the value the symmetric case always had
    let sphere = SdfNode::sphere(1.0);
    for &t in &[0.1_f32, 0.25] {
        let node = shell_node(Arc::new(sphere.clone()), ShellConfig::uniform(t));
        assert!(matches!(node, SdfNode::Onion { .. }));
        for p in radial_points() {
            let exact = (p.length() - 1.0).abs() - t;
            assert!((eval(&node, p) - exact).abs() < 1e-5, "t={t} {p:?}");
        }
    }
}

// ── morphology ──────────────────────────────────────────────

#[test]
fn offset_of_a_sphere_is_the_sphere_of_radius_r_plus_offset() {
    let sphere = SdfNode::sphere(1.0);
    let pts = radial_points();
    let mut compared = 0;
    for &radius in &[0.25_f32, -0.4, 0.0] {
        let batch = eval_offset_batch(&sphere, &pts, radius);
        let par = eval_offset_batch_parallel(&sphere, &pts, radius);
        for (i, &p) in pts.iter().enumerate() {
            let exact = p.length() - (1.0 + radius);
            for got in [eval_offset(&sphere, p, radius), batch[i], par[i]] {
                assert!(
                    (got - exact).abs() < 1e-5,
                    "{p:?} r={radius}: {got} vs {exact}"
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 3 * 3 * radial_points().len());
}

#[test]
fn offset_of_a_box_moves_each_face_by_the_offset() {
    // in front of a face (inside the face's slab) the box distance is the
    // distance to the face plane, so the offset field is that minus r
    let b = SdfNode::box3d(2.0, 2.0, 2.0); // half-extent 1
    for &x in &[1.5_f32, 2.0, 3.25] {
        let p = Vec3::new(x, 0.2, -0.4);
        let got = eval_offset(&b, p, 0.3);
        assert!((got - (x - 1.0 - 0.3)).abs() < 1e-5, "x={x}: {got}");
    }
}

#[test]
fn tolerance_fit_matches_the_radius_gap() {
    let outer = SdfNode::sphere(1.0);
    // a smaller ball fits with no tolerance
    assert!(tolerance_fits(
        &SdfNode::sphere(0.9),
        &outer,
        0.0,
        4096,
        1.3
    ));
    // a ball of 1.2 fits once the outer one is inflated to 1.25
    assert!(tolerance_fits(
        &SdfNode::sphere(1.2),
        &outer,
        0.25,
        4096,
        1.3
    ));
    // with 0.1 the gap is 1.2 − 1.1 = 0.1, attained at the grid point (1.2, 0, 0):
    // 25 points per axis over [−1.2, 1.2] give a step of 0.1
    let v = tolerance_max_violation(&SdfNode::sphere(1.2), &outer, 0.1, 25 * 25 * 25, 1.2)
        .expect("a ball of 1.2 does not fit inside 1.1");
    assert!((v - 0.1).abs() < 1e-5, "violation {v} vs 0.1");
    assert!(!tolerance_fits(
        &SdfNode::sphere(1.2),
        &outer,
        0.1,
        25 * 25 * 25,
        1.2
    ));
}

// ── fidelity → RaymarchConfig::relaxed ──────────────────────

#[test]
fn relaxed_tracing_divides_by_the_proved_bound() {
    // non-uniform scale (1, 2, 1): the bound is max/min = 2
    let stretched = SdfNode::sphere(1.0).scale_xyz(1.0, 2.0, 1.0);
    let c = RaymarchConfig::relaxed(&stretched);
    assert_eq!((c.omega, c.lipschitz, c.max_steps), (1.6, 2.0, 512));

    // an exact field: L = 1, over-relaxed, the default budget
    let c = RaymarchConfig::relaxed(&SdfNode::sphere(1.0));
    assert_eq!((c.omega, c.lipschitz, c.max_steps), (1.6, 1.0, 256));

    // a uniform scale-down keeps L = 1 (a bound below one never shrinks the step)
    let c = RaymarchConfig::relaxed(&SdfNode::sphere(1.0).scale(0.5));
    assert_eq!(c.lipschitz, 1.0);

    // polar repetition of an arbitrary child has no finite bound: plain tracing
    let c = RaymarchConfig::relaxed(&SdfNode::sphere(0.4).polar_repeat(7));
    assert_eq!((c.omega, c.lipschitz, c.max_steps), (1.0, 1.0, 256));
}
