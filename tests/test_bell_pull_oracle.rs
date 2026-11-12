//! Bell pull: the warp `x(p) = p − Σ v·exp(−|p.xy − g − v|² / 2σ²)` and the value `child(x(p)) / Λ`.
//!
//! Oracles, each against something the implementation does not compute:
//! - the profile: one pull along x, a plane child `x = 0`, evaluated through `SdfNode::bell_pull` and
//!   `eval`, equals the closed form `(p.x − v·exp(−((p.x − g − v)² + p.y²) / 2σ²)) / Λ` in f64
//! - the carried point: the rest point `g` is where `g + v` maps back to (exactly the bell's top)
//! - one to one: the Jacobian of `x` measured by central differences in f64 has determinant at least
//!   `(1 − S)²` and norm at most `Λ = 1 + S` (`S = Σ|v|·e^(−1/2)/σ`)
//! - Lipschitz: over a 1-Lipschitz child the value is 1-Lipschitz (sampled pairs), and it never
//!   exceeds the distance to the warped surface (bisection along rays from the sphere's centre)
//! - refusals: a width that is not positive, a value that is not finite, a slope sum above
//!   `MOST_SLOPE` (and the sum just at it is accepted); the stored form is checked again on load
//!
//! Author: Moroya Sakamoto

use alice_det_math::{exp64, hypot64};
use alice_sdf::eval::eval;
use alice_sdf::prelude::Vec3;
use alice_sdf::transforms::bell_pull::{BELL_SLOPE, MOST_SLOPE};
use alice_sdf::transforms::{BellPull, BellPullError};
use alice_sdf::SdfNode;

/// `e^(−1/2)` in f64, the reference for `BELL_SLOPE`
const SLOPE64: f64 = 0.606_530_659_712_633_4;

/// The map in f64, written from the formula (not from the implementation)
fn back64(pulls: [[f64; 4]; 2], sigma: f64, p: [f64; 3]) -> [f64; 3] {
    let mut x = p;
    for u in pulls {
        let (rx, ry) = (p[0] - u[0] - u[2], p[1] - u[1] - u[3]);
        let w = exp64(-(rx * rx + ry * ry) / (2.0 * sigma * sigma));
        x[0] -= u[2] * w;
        x[1] -= u[3] * w;
    }
    x
}

fn slope_sum64(pulls: [[f64; 4]; 2], sigma: f64) -> f64 {
    pulls.iter().map(|u| hypot64(u[2], u[3])).sum::<f64>() * SLOPE64 / sigma
}

fn to64(p: [[f32; 4]; 2]) -> [[f64; 4]; 2] {
    p.map(|u| u.map(f64::from))
}

fn grid(lo: f32, hi: f32, n: usize) -> Vec<Vec3> {
    let step = (hi - lo) / (n - 1) as f32;
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                out.push(Vec3::new(
                    lo + step * i as f32,
                    lo + step * j as f32,
                    lo + step * k as f32,
                ));
            }
        }
    }
    out
}

#[test]
fn the_bell_slope_is_e_to_the_minus_half() {
    assert!((f64::from(BELL_SLOPE) - SLOPE64).abs() < 1e-7);
}

/// One pull along x over the plane `x = 0`: the value is the closed form, through the production entry
#[test]
fn one_pull_over_a_plane_is_the_closed_form_profile() {
    let pulls = [[0.2, 0.1, 0.3, 0.0], [0.0; 4]];
    let sigma = 0.4;
    let pull = BellPull::new(pulls, sigma).expect("one to one");
    // the plane x = 0 as a half space: the distance is p.x
    let plane = SdfNode::plane(Vec3::X, 0.0);
    let node = plane.bell_pull(pull);
    let lambda = 1.0 + slope_sum64(to64(pulls), f64::from(sigma));
    let mut compared = 0;
    let mut worst = 0.0f64;
    for p in grid(-1.0, 1.0, 9) {
        let want = back64(to64(pulls), f64::from(sigma), p.to_array().map(f64::from))[0] / lambda;
        let got = f64::from(eval(&node, p));
        worst = worst.max((got - want).abs());
        compared += 1;
    }
    assert_eq!(compared, 729);
    assert!(worst < 2e-6, "worst {worst}");
}

/// The rest point `g` is where `g + v` maps back to: the bell's top carries it by the whole `v`
#[test]
fn the_point_pressed_is_carried_by_the_whole_pull() {
    let pull = BellPull::new([[0.2, -0.1, 0.15, 0.1], [0.0; 4]], 0.5).expect("one to one");
    let x = pull.back(Vec3::new(0.35, 0.0, 0.7));
    assert!((x - Vec3::new(0.2, -0.1, 0.7)).length() < 1e-6, "{x}");
    // far from both bells the map is the identity (to the f32 of the bell's tail)
    let far = Vec3::new(9.0, -9.0, 0.3);
    assert!((pull.back(far) - far).length() < 1e-6);
}

/// The map's Jacobian (central differences of the f64 formula at the f32 parameters) has
/// determinant at least `(1 − S)²` and norm at most `Λ`, and `lambda()` is `1 + S`
#[test]
fn the_map_is_one_to_one_with_its_slope_bound() {
    let cases: [([[f32; 4]; 2], f32); 3] = [
        ([[0.0, 0.0, 0.3, 0.0], [0.0; 4]], 0.34),
        ([[0.3, 0.0, 0.25, 0.0], [-0.2, 0.4, -0.12, 0.18]], 0.5),
        ([[0.0, 0.0, 0.0, 0.5], [0.1, 0.0, 0.0, -0.1]], 0.45),
    ];
    let mut compared = 0;
    for (pulls, sigma) in cases {
        let pull = BellPull::new(pulls, sigma).expect("one to one");
        let (p64, s64) = (to64(pulls), f64::from(sigma));
        let s = slope_sum64(p64, s64);
        assert!((f64::from(pull.lambda()) - (1.0 + s)).abs() < 1e-6);
        let h = 1e-5;
        for p in grid(-1.2, 1.2, 13) {
            let p = p.to_array().map(f64::from);
            let d = |i: usize| {
                let (mut a, mut b) = (p, p);
                a[i] += h;
                b[i] -= h;
                let (fa, fb) = (back64(p64, s64, a), back64(p64, s64, b));
                [(fa[0] - fb[0]) / (2.0 * h), (fa[1] - fb[1]) / (2.0 * h)]
            };
            let (jx, jy) = (d(0), d(1));
            // the xy block (z is the identity)
            let det = jx[0] * jy[1] - jy[0] * jx[1];
            assert!(
                det >= (1.0 - s) * (1.0 - s) - 1e-6,
                "det {det} < (1 − {s})²"
            );
            // the spectral norm of the 2×2 block
            let (a, b, c, dd) = (jx[0], jy[0], jx[1], jy[1]);
            let t = a * a + b * b + c * c + dd * dd;
            let norm = f64::midpoint(t, ((t * t) - 4.0 * det * det).max(0.0).sqrt()).sqrt();
            assert!(norm <= 1.0 + s + 1e-6, "norm {norm} > 1 + {s}");
            compared += 1;
        }
    }
    assert_eq!(compared, 3 * 13 * 13 * 13);
}

/// Over a 1-Lipschitz child the value is 1-Lipschitz, and it never exceeds the distance to the
/// warped surface (found by bisection on the field along rays from inside)
#[test]
fn the_value_is_one_lipschitz_and_a_distance_lower_bound() {
    let pull = BellPull::new([[0.3, 0.0, 0.25, 0.0], [-0.2, 0.4, -0.12, 0.18]], 0.5).expect("ok");
    let node = SdfNode::sphere(0.6).bell_pull(pull);
    let pts = grid(-1.1, 1.1, 11);
    let mut pairs = 0;
    for (i, a) in pts.iter().enumerate() {
        for b in pts.iter().skip(i + 1).step_by(37) {
            let (fa, fb) = (eval(&node, *a), eval(&node, *b));
            assert!(
                (fa - fb).abs() <= (*a - *b).length() * (1.0 + 1e-5) + 1e-6,
                "{a} {b}"
            );
            pairs += 1;
        }
    }
    assert!(pairs > 10_000, "{pairs}");
    // the surface along a ray from the centre: the field is negative at the centre and positive far
    let mut checked = 0;
    for dir in grid(-1.0, 1.0, 5).into_iter().filter(|d| d.length() > 0.5) {
        let dir = dir.normalize();
        let (mut lo, mut hi) = (0.0f32, 2.0f32);
        assert!(eval(&node, Vec3::ZERO) < 0.0 && eval(&node, dir * hi) > 0.0);
        for _ in 0..60 {
            let mid = f32::midpoint(lo, hi);
            if eval(&node, dir * mid) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let surface = dir * hi;
        // a point outside at distance t along the ray is at most t from the surface point
        for t in [0.05f32, 0.2, 0.5] {
            let q = surface + dir * t;
            assert!(eval(&node, q) <= t + 1e-5, "{q}: {} > {t}", eval(&node, q));
            checked += 1;
        }
    }
    assert!(checked > 50, "{checked}");
}

#[test]
fn widths_values_and_folding_pulls_are_refused() {
    assert_eq!(BellPull::new([[0.0; 4]; 2], 0.0), Err(BellPullError::Bad));
    assert_eq!(BellPull::new([[0.0; 4]; 2], -1.0), Err(BellPullError::Bad));
    assert_eq!(
        BellPull::new([[0.0; 4]; 2], f32::NAN),
        Err(BellPullError::Bad)
    );
    assert_eq!(
        BellPull::new([[f32::INFINITY, 0.0, 0.0, 0.0], [0.0; 4]], 0.5),
        Err(BellPullError::Bad)
    );
    assert!(
        matches!(BellPull::new([[0.0, 0.0, 0.6, 0.0], [0.0; 4]], 0.34), Err(BellPullError::Folds(s)) if s > MOST_SLOPE)
    );
    // a slope sum just at the limit is accepted, and a hair above it refused
    let at = MOST_SLOPE * 0.5 / BELL_SLOPE;
    assert!(BellPull::new([[0.0, 0.0, at * 0.999_99, 0.0], [0.0; 4]], 0.5).is_ok());
    assert!(BellPull::new([[0.0, 0.0, at * 1.001, 0.0], [0.0; 4]], 0.5).is_err());
    // no pull at all is the identity, and accepted
    let still = BellPull::new([[0.0; 4]; 2], 0.34).expect("no pull folds nothing");
    let p = Vec3::new(0.3, -0.2, 0.1);
    assert_eq!(still.back(p), p);
    assert_eq!(still.lambda(), 1.0);
}

/// The stored form is checked again when it is read back: a folding pull in the data is refused
#[test]
fn a_stored_pull_is_checked_again_on_load() {
    let ok = BellPull::new([[0.3, 0.0, 0.25, 0.0], [0.0; 4]], 0.5).expect("ok");
    let text = serde_json::to_string(&ok).expect("writes");
    assert_eq!(serde_json::from_str::<BellPull>(&text).expect("reads"), ok);
    let bad = text.replace("0.25", "0.9");
    assert_ne!(bad, text);
    assert!(serde_json::from_str::<BellPull>(&bad).is_err(), "{bad}");
}
