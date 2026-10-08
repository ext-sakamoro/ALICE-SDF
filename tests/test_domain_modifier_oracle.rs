//! Single-axis domain modifiers vs closed-form point maps computed in `f64`.
//!
//! References (none of them calls the crate's modifiers):
//!
//! * mirror: `|x|` on the chosen axis, the other two untouched (exact),
//! * twist / bend: a plane rotation by an angle that is linear in one
//!   coordinate, written out as `(c u - s v, s u + c v)` in `f64`;
//!   twist keeps the axis coordinate and the radius around the axis,
//! * cheap bend: `x + k y^2`,
//! * single-axis repetition: `x - s floor(x / s + 1/2)` (ties round up, the
//!   crate's documented convention; the sample points stay clear of ties),
//! * polar repetition: the angle `atan2(z, x)` folded into the sector
//!   `[-pi/n, pi/n)` around +X, the radius and `y` unchanged,
//! * IFS fold: a second, independent greedy orbit fold in `f64` (pick the
//!   affine map that brings the point closest to the origin, keep the point
//!   if none does better); points whose choice is within `1e-3` of a tie are
//!   skipped and counted,
//! * fBm: the definition `sum a^i n(l^i p, seed + i) / sum a^i` rebuilt from
//!   `perlin_noise_3d`; gradient noise vanishes on the integer lattice, so fBm
//!   with an integer lacunarity is exactly 0 there,
//! * simplex-noise displacement: documented as the Perlin displacement, so it
//!   must equal `modifier_noise_perlin`, and is the identity on the lattice,
//! * Bezier sweep: brute-force distance to the quadratic Bezier in `f64`
//!   (200 001 samples) and `y` passed through.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
#![allow(clippy::float_cmp)]

use alice_sdf::modifiers::{
    fbm_noise_3d, ifs_fold, ifs_fold_with_scale, modifier_bend_cheap, modifier_bend_x,
    modifier_bend_z, modifier_mirror_x, modifier_mirror_y, modifier_mirror_z,
    modifier_noise_perlin, modifier_noise_simplex, modifier_repeat_polar, modifier_repeat_x,
    modifier_repeat_y, modifier_repeat_z, modifier_twist_x, modifier_twist_z, perlin_noise_3d,
    sweep_bezier_dist_y,
};
use glam::Vec3;

fn points() -> Vec<Vec3> {
    let mut out = Vec::new();
    for ix in -5..=5 {
        for iy in -5..=5 {
            for iz in -5..=5 {
                out.push(Vec3::new(
                    ix as f32 * 0.43 + 0.011,
                    iy as f32 * 0.37 - 0.017,
                    iz as f32 * 0.31 + 0.023,
                ));
            }
        }
    }
    out
}

fn close3(what: &str, p: Vec3, got: Vec3, want: [f64; 3], tol: f64) {
    for (g, w) in [got.x, got.y, got.z].into_iter().zip(want) {
        assert!(
            (g as f64 - w).abs() <= tol * (1.0 + w.abs()),
            "{what} at {p:?}: got {got:?}, closed form {want:?}"
        );
    }
}

/// `(c u - s v, s u + c v)` for the angle `a`.
fn rot(u: f64, v: f64, a: f64) -> (f64, f64) {
    let (s, c) = a.sin_cos();
    (c * u - s * v, s * u + c * v)
}

#[test]
fn single_axis_mirrors_take_the_absolute_value_of_one_coordinate() {
    let mut n = 0;
    for p in points() {
        assert_eq!(modifier_mirror_x(p), Vec3::new(p.x.abs(), p.y, p.z));
        assert_eq!(modifier_mirror_y(p), Vec3::new(p.x, p.y.abs(), p.z));
        assert_eq!(modifier_mirror_z(p), Vec3::new(p.x, p.y, p.z.abs()));
        n += 3;
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn axis_twists_rotate_the_cross_section_by_an_angle_linear_in_the_axis() {
    let mut n = 0;
    for &k in &[0.0_f32, 0.7, -1.9] {
        for p in points() {
            let (x, y, z) = (p.x as f64, p.y as f64, p.z as f64);
            let (y1, z1) = rot(y, z, x * k as f64);
            close3(
                "modifier_twist_x",
                p,
                modifier_twist_x(p, k),
                [x, y1, z1],
                1e-5,
            );
            let (x2, y2) = rot(x, y, z * k as f64);
            close3(
                "modifier_twist_z",
                p,
                modifier_twist_z(p, k),
                [x2, y2, z],
                1e-5,
            );
            // the axis coordinate is untouched exactly; the radius is kept
            let t = modifier_twist_x(p, k);
            assert_eq!(t.x, p.x);
            assert!(((t.y as f64).hypot(t.z as f64) - y.hypot(z)).abs() < 1e-5);
            n += 3;
        }
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn axis_bends_and_the_cheap_bend_match_their_point_maps() {
    let mut n = 0;
    for &k in &[0.0_f32, 0.45, -1.1] {
        let kk = k as f64;
        for p in points() {
            let (x, y, z) = (p.x as f64, p.y as f64, p.z as f64);
            // `modifier_bend_x`: rotation of (x, z) by `k y`
            let (x1, z1) = rot(x, z, kk * y);
            close3(
                "modifier_bend_x",
                p,
                modifier_bend_x(p, k),
                [x1, y, z1],
                1e-5,
            );
            // `modifier_bend_z`: rotation of (y, z) by `k y`
            let (y2, z2) = rot(y, z, kk * y);
            close3(
                "modifier_bend_z",
                p,
                modifier_bend_z(p, k),
                [x, y2, z2],
                1e-5,
            );
            close3(
                "modifier_bend_cheap",
                p,
                modifier_bend_cheap(p, k),
                [x + kk * y * y, y, z],
                1e-6,
            );
            n += 3;
        }
    }
    assert!(n > 0, "no comparisons made");
}

fn fold(x: f64, s: f64) -> Option<f64> {
    let q = x / s + 0.5;
    // stay clear of the tie, where an ulp decides the cell
    if (q - q.round()).abs() < 1e-4 {
        return None;
    }
    Some(x - s * q.floor())
}

#[test]
fn single_axis_repetition_folds_one_coordinate_into_the_centred_cell() {
    let mut n = 0;
    let mut skipped = 0;
    for &s in &[0.5_f32, 1.3, 2.0] {
        for p in points() {
            let (x, y, z) = (p.x as f64, p.y as f64, p.z as f64);
            let ss = s as f64;
            match (fold(x, ss), fold(y, ss), fold(z, ss)) {
                (Some(fx), Some(fy), Some(fz)) => {
                    close3(
                        "modifier_repeat_x",
                        p,
                        modifier_repeat_x(p, s),
                        [fx, y, z],
                        1e-5,
                    );
                    close3(
                        "modifier_repeat_y",
                        p,
                        modifier_repeat_y(p, s),
                        [x, fy, z],
                        1e-5,
                    );
                    close3(
                        "modifier_repeat_z",
                        p,
                        modifier_repeat_z(p, s),
                        [x, y, fz],
                        1e-5,
                    );
                    for c in [modifier_repeat_x(p, s).x, modifier_repeat_y(p, s).y] {
                        assert!(c.abs() <= s * 0.5 + 1e-6);
                    }
                    n += 3;
                }
                _ => skipped += 1,
            }
        }
    }
    assert!(n > 0, "no comparisons made ({skipped} skipped)");
    assert!(skipped * 10 < n, "too many tie skips: {skipped} of {n}");
}

#[test]
fn polar_repetition_folds_the_angle_into_the_sector_around_plus_x() {
    let mut n = 0;
    for &count in &[3_u32, 5, 8] {
        let sector = std::f64::consts::TAU / count as f64;
        for p in points() {
            let (x, y, z) = (p.x as f64, p.y as f64, p.z as f64);
            let r = x.hypot(z);
            if r < 1e-3 {
                continue;
            }
            let a = z.atan2(x);
            let q = a / sector + 0.5;
            if (q - q.round()).abs() < 1e-4 {
                continue;
            }
            let b = a - sector * q.floor();
            let want = [r * b.cos(), y, r * b.sin()];
            close3(
                "modifier_repeat_polar",
                p,
                modifier_repeat_polar(p, count),
                want,
                2e-5,
            );
            n += 1;
        }
    }
    assert!(n > 0, "no comparisons made");
}

/// Column-major 4x4 affine map applied to a point.
fn apply(m: &[f32; 16], p: [f64; 3]) -> [f64; 3] {
    let mut q = [0.0; 3];
    for (i, qi) in q.iter_mut().enumerate() {
        *qi =
            m[i] as f64 * p[0] + m[4 + i] as f64 * p[1] + m[8 + i] as f64 * p[2] + m[12 + i] as f64;
    }
    q
}

#[test]
fn ifs_fold_is_the_greedy_orbit_fold() {
    // translate by -1 along X / Y, and a half-scale about the origin
    let t: [[f32; 16]; 3] = [
        [
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., -1., 0., 0., 1.,
        ],
        [
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., -1., 0., 1.,
        ],
        [
            0.5, 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 1.,
        ],
    ];
    let mut n = 0;
    let mut skipped = 0;
    'pts: for p in points() {
        for iters in [0_u32, 1, 2, 4] {
            let mut q = [p.x as f64, p.y as f64, p.z as f64];
            for _ in 0..iters {
                let mut best = q;
                let mut best_d = q.iter().map(|c| c * c).sum::<f64>();
                let mut ds = vec![best_d];
                for m in &t {
                    let c = apply(m, q);
                    let d = c.iter().map(|c| c * c).sum::<f64>();
                    ds.push(d);
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                ds.sort_by(f64::total_cmp);
                if ds[1] - ds[0] < 1e-3 {
                    skipped += 1;
                    continue 'pts;
                }
                q = best;
            }
            close3("ifs_fold", p, ifs_fold(p, &t, iters), q, 1e-5);
            // the scaled variant folds the same point
            assert_eq!(ifs_fold(p, &t, iters), ifs_fold_with_scale(p, &t, iters).0);
            n += 1;
        }
    }
    assert!(n > 0, "no comparisons made ({skipped} skipped)");
}

#[test]
fn fbm_is_the_normalised_octave_sum_and_vanishes_on_the_lattice() {
    let mut n = 0;
    for p in points() {
        let (x, y, z) = (p.x, p.y, p.z);
        // one octave is the base noise itself
        assert_eq!(
            fbm_noise_3d(x, y, z, 7, 1, 2.0, 0.5),
            perlin_noise_3d(x, y, z, 7)
        );
        // three octaves, lacunarity 2, persistence 0.5
        let want = (perlin_noise_3d(x, y, z, 7) as f64
            + 0.5 * perlin_noise_3d(2.0 * x, 2.0 * y, 2.0 * z, 8) as f64
            + 0.25 * perlin_noise_3d(4.0 * x, 4.0 * y, 4.0 * z, 9) as f64)
            / 1.75;
        let got = fbm_noise_3d(x, y, z, 7, 3, 2.0, 0.5) as f64;
        assert!((got - want).abs() < 1e-6, "fbm at {p:?}: {got} vs {want}");
        n += 2;
    }
    for ix in -3..=3 {
        for iy in -3..=3 {
            for iz in -3..=3 {
                let (x, y, z) = (ix as f32, iy as f32, iz as f32);
                assert_eq!(fbm_noise_3d(x, y, z, 3, 4, 2.0, 0.6), 0.0);
                n += 1;
            }
        }
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn fbm_octave_seeds_wrap_instead_of_overflowing() {
    // `seed + octave` wraps (the release-build value); a plain `+` would panic in debug
    let (x, y, z) = (0.3_f32, -0.7, 1.1);
    let got = fbm_noise_3d(x, y, z, u32::MAX, 2, 2.0, 0.5) as f64;
    let want = (perlin_noise_3d(x, y, z, u32::MAX) as f64
        + 0.5 * perlin_noise_3d(2.0 * x, 2.0 * y, 2.0 * z, 0) as f64)
        / 1.5;
    assert!((got - want).abs() < 1e-6, "{got} vs {want}");
}

#[test]
fn simplex_displacement_is_the_perlin_displacement() {
    let mut n = 0;
    for p in points() {
        for &(amp, freq, seed) in &[(0.2_f32, 1.5_f32, 3_u32), (0.05, 4.0, 11)] {
            let d = 0.25;
            assert_eq!(
                modifier_noise_simplex(d, p, amp, freq, seed),
                modifier_noise_perlin(d, p, amp, freq, seed)
            );
            n += 1;
        }
    }
    // on the lattice the displacement is zero: the distance comes back unchanged
    for ix in -3..=3 {
        let p = Vec3::new(ix as f32, 1.0, -2.0);
        assert_eq!(modifier_noise_simplex(0.5, p, 0.3, 1.0, 5), 0.5);
        n += 1;
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn bezier_sweep_distance_is_the_brute_force_curve_distance() {
    const SAMPLES: usize = 200_000;
    let curves = [
        ([0.0_f32, 0.0], [0.5, 1.0], [1.0, 0.0]),
        ([-1.0, 0.3], [0.2, -1.4], [1.2, 0.8]),
    ];
    let mut n = 0;
    for (a, b, c) in curves {
        let pts: Vec<[f64; 2]> = (0..=SAMPLES)
            .map(|i| {
                let t = i as f64 / SAMPLES as f64;
                let u = 1.0 - t;
                let f = |k: usize| {
                    u * u * a[k] as f64 + 2.0 * u * t * b[k] as f64 + t * t * c[k] as f64
                };
                [f(0), f(1)]
            })
            .collect();
        for p in points().into_iter().step_by(5) {
            let want = pts
                .iter()
                .map(|q| (p.x as f64 - q[0]).hypot(p.z as f64 - q[1]))
                .fold(f64::INFINITY, f64::min);
            let (d, y) = sweep_bezier_dist_y(p.x, p.y, p.z, a[0], a[1], b[0], b[1], c[0], c[1]);
            assert!(
                (d as f64 - want).abs() < 1e-4,
                "sweep at {p:?}: {d} vs brute force {want}"
            );
            assert_eq!(y, p.y);
            n += 1;
        }
    }
    assert!(n > 0, "no comparisons made");
}
