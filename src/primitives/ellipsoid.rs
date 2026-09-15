//! Ellipsoid SDF (Deep Fried Edition)
//!
//! Approximate but high-quality SDF for an ellipsoid.
//! Based on Inigo Quilez's formula: k0*(k0-1)/k1
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::{Real, Vec3R};
use glam::Vec3;

/// Bisection budget for the nearest-point root (`GetRoot`, Eberly): the
/// loop also stops as soon as the interval no longer shrinks in f32.
const ELLIPSOID_BISECT_STEPS: u32 = 64;

/// Bisector for `F(s) = Σ (rᵢ zᵢ / (s + rᵢ))² − 1 = 0` on `(s₀, s₁)`
/// (Eberly, "Distance from a Point to an Ellipse, an Ellipsoid, or a
/// Hyperellipsoid", §2.6, in the normalised coordinates `r = (e/e_min)²`,
/// `z = y/e`).
#[inline(always)]
fn ellipsoid_bisector(n: usize, r: [f32; 3], z: [f32; 3], g: f32) -> f32 {
    let mut nn = [0.0f32; 3];
    let mut norm2 = 0.0f32;
    for i in 0..n {
        nn[i] = r[i] * z[i];
        norm2 += nn[i] * nn[i];
    }
    let mut s0 = z[n - 1] - 1.0;
    let mut s1 = if g < 0.0 { 0.0 } else { norm2.sqrt() - 1.0 };
    let mut s = 0.0f32;
    for _ in 0..ELLIPSOID_BISECT_STEPS {
        s = f32::midpoint(s0, s1);
        if s == s0 || s == s1 {
            break;
        }
        let mut gs = -1.0f32;
        for i in 0..n {
            let ratio = nn[i] / (s + r[i]);
            gs += ratio * ratio;
        }
        if gs > 0.0 {
            s0 = s;
        } else if gs < 0.0 {
            s1 = s;
        } else {
            break;
        }
    }
    s
}

/// Squared distance from `y` (first orthant) to the ellipsoid with semi-axes
/// `e` sorted decreasing, on the first `n` axes (Eberly's
/// `SqrDistanceSpecial`, recursive in the dimension when the last
/// coordinate is 0).
fn ellipsoid_sqr_distance(n: usize, e: [f32; 3], y: [f32; 3]) -> f32 {
    if n == 1 {
        let d = y[0] - e[0];
        return d * d;
    }
    let last = n - 1;
    if y[last] > 0.0 {
        let mut z = [0.0f32; 3];
        let mut g = -1.0f32;
        for i in 0..n {
            z[i] = y[i] / e[i];
            g += z[i] * z[i];
        }
        if g == 0.0 {
            return 0.0;
        }
        let mut r = [0.0f32; 3];
        for i in 0..n {
            let q = e[i] / e[last];
            r[i] = q * q;
        }
        let sbar = ellipsoid_bisector(n, r, z, g);
        let mut d2 = 0.0f32;
        for i in 0..n {
            let x = r[i] * y[i] / (sbar + r[i]);
            let d = x - y[i];
            d2 += d * d;
        }
        return d2;
    }
    // y[last] == 0: the nearest point may lie off the y[last] = 0 plane
    let mut numer = [0.0f32; 3];
    let mut denom = [0.0f32; 3];
    let mut inside = true;
    for i in 0..last {
        numer[i] = e[i] * y[i];
        denom[i] = e[i] * e[i] - e[last] * e[last];
        if numer[i] >= denom[i] {
            inside = false;
        }
    }
    if inside {
        let mut discr = 1.0f32;
        let mut xde = [0.0f32; 3];
        for i in 0..last {
            xde[i] = numer[i] / denom[i];
            discr -= xde[i] * xde[i];
        }
        if discr > 0.0 {
            let mut d2 = 0.0f32;
            for i in 0..last {
                let d = e[i] * xde[i] - y[i];
                d2 += d * d;
            }
            let xl = e[last] * discr.sqrt();
            return d2 + xl * xl;
        }
    }
    ellipsoid_sqr_distance(last, e, y)
}

/// Exact signed distance to an axis-aligned ellipsoid with semi-axes
/// `radii` (Eberly's robust nearest-point algorithm: sort the axes, fold
/// the point into the first orthant, bisect for the Lagrange parameter).
/// Exact SDF inside and out, Lipschitz 1.
///
/// Until 1.10.3 this was Inigo Quilez's `k0·(k0 − 1)/k1` approximation,
/// whose gradient grows like `(max r / min r)⁴` far from the surface and
/// which is discontinuous at the centre — sphere tracing could skip an
/// anisotropic ellipsoid and `eval_lipschitz` had no finite bound for it.
#[inline]
pub fn sdf_ellipsoid_exact(p: Vec3, radii: Vec3) -> f32 {
    let safe = [radii.x.max(1e-10), radii.y.max(1e-10), radii.z.max(1e-10)];
    let mut y = [p.x.abs(), p.y.abs(), p.z.abs()];
    let mut e = safe;
    // sort axes decreasing, carrying the point with them (3-element network)
    if e[0] < e[1] {
        e.swap(0, 1);
        y.swap(0, 1);
    }
    if e[1] < e[2] {
        e.swap(1, 2);
        y.swap(1, 2);
    }
    if e[0] < e[1] {
        e.swap(0, 1);
        y.swap(0, 1);
    }
    let d = ellipsoid_sqr_distance(3, e, y).sqrt();
    let inside = (y[0] / e[0]).powi(2) + (y[1] / e[1]).powi(2) + (y[2] / e[2]).powi(2) < 1.0;
    if inside {
        -d
    } else {
        d
    }
}

/// Ellipsoid with `radii` (generic over [`Real`]): the exact law applied
/// per lane (the bisection is branchy; SIMD lanes evaluate it one by one).
#[inline(always)]
pub fn sdf_ellipsoid_r<R: Real>(p: Vec3R<R>, radii: Vec3) -> R {
    R::map3(p.x, p.y, p.z, |q| sdf_ellipsoid_exact(q, radii))
}

/// Ellipsoid with `radii` (exact signed distance).
#[inline(always)]
pub fn sdf_ellipsoid(p: Vec3, radii: Vec3) -> f32 {
    sdf_ellipsoid_exact(p, radii)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ellipsoid_origin() {
        // At origin, should be inside (negative distance)
        let d = sdf_ellipsoid(Vec3::ZERO, Vec3::new(1.0, 2.0, 1.0));
        assert!(d < 0.0, "Origin should be inside ellipsoid, got {}", d);
    }

    #[test]
    fn test_ellipsoid_on_surface_x() {
        // On the x-axis surface
        let d = sdf_ellipsoid(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 2.0, 1.0));
        assert!(d.abs() < 0.01, "Should be on surface at x=rx, got {}", d);
    }

    #[test]
    fn test_ellipsoid_on_surface_y() {
        // On the y-axis surface
        let d = sdf_ellipsoid(Vec3::new(0.0, 2.0, 0.0), Vec3::new(1.0, 2.0, 1.0));
        assert!(d.abs() < 0.01, "Should be on surface at y=ry, got {}", d);
    }

    #[test]
    fn test_ellipsoid_sphere_equivalence() {
        // With equal radii, should approximate a sphere
        let r = 1.5;
        let p = Vec3::new(r, 0.0, 0.0);
        let d = sdf_ellipsoid(p, Vec3::splat(r));
        assert!(
            d.abs() < 0.01,
            "Equal radii should be like a sphere, got {}",
            d
        );
    }

    /// Brute force oracle: densest sampling of the surface parametrisation.
    fn brute(p: Vec3, r: Vec3) -> f32 {
        let mut best = f32::MAX;
        let n = 600;
        for i in 0..=n {
            let th = std::f32::consts::PI * i as f32 / n as f32;
            for j in 0..2 * n {
                let ph = std::f32::consts::TAU * j as f32 / (2 * n) as f32;
                let s = Vec3::new(
                    r.x * th.sin() * ph.cos(),
                    r.y * th.sin() * ph.sin(),
                    r.z * th.cos(),
                );
                best = best.min((p - s).length());
            }
        }
        let inside = (p.x / r.x).powi(2) + (p.y / r.y).powi(2) + (p.z / r.z).powi(2) < 1.0;
        if inside {
            -best
        } else {
            best
        }
    }

    #[test]
    fn test_ellipsoid_exact_matches_brute_force() {
        for r in [
            Vec3::new(1.2, 0.6, 0.9),
            Vec3::new(3.0, 1.0, 0.5),
            Vec3::new(1.0, 1.0, 0.1),
        ] {
            for p in [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.3, 0.2, 0.1),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
                Vec3::new(0.0, 0.0, 2.0),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(-0.7, 0.4, -0.2),
                Vec3::new(19.0, 0.0, -0.38),
                Vec3::new(0.5, 0.0, 0.0),
                Vec3::new(0.0, 0.45, 0.0),
            ] {
                let d = sdf_ellipsoid(p, r);
                let b = brute(p, r);
                assert!((d - b).abs() < 4e-3, "{r:?} {p:?}: exact {d} vs brute {b}");
            }
        }
    }

    #[test]
    fn test_ellipsoid_outside() {
        let d = sdf_ellipsoid(Vec3::new(5.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }
}
