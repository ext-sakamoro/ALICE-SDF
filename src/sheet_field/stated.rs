//! The stated light model a decoded sheet field is judged under, and the reference renderer's rules around it.
//!
//! One source for every implementation (the CPU gate renderer here, the WebGL decode as a copy held by a parity test):
//! Lambert x 0.96 + single-scatter GGX with constant reflectance 0.04 (no Fresnel), alpha = roughness^2, direct light
//! only. The rules were measured on the reference renderer itself (Cycles, Blender 5.2.3) with probe scenes:
//!
//! - **smooth-normal cut**: light behind the interpolated smooth normal reaches no reflection closure (diffuse and
//!   glossy are exactly 0 there).
//! - **diffuse ramp**: above the cut the diffuse closure is scaled by the bump-shadowing term of Conty Estevez et al.
//!   2019 with the smooth normal as the geometric one: a strip with swept custom normals matched it to a mean
//!   |error| 0.007 over 161 columns. The glossy closure has the cut only.
//! - **glossy normal**: the glossy closure evaluates on the normal corrected by `ensure_valid_reflection(Ng, view, N)`;
//!   the diffuse keeps the mapped normal (157 columns at 15 / 30 deg, mean |error| 0.0005).
//! - **alpha-zero write**: where an 8-bit alpha rounds to 0, the colour is written premultiplied (value x coverage).
//!
//! The functions keep the order of floating-point operations of the renderer they were moved from, so that renderer
//! reproduces its earlier images bit for bit.
use glam::Vec3;

/// Which of the reference renderer's rules apply (all off = the plain stated model).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rules {
    /// Light behind the interpolated smooth normal reaches no closure.
    pub smooth_cut: bool,
    /// The diffuse closure carries the bump-shadowing term on the smooth normal (requires `smooth_cut`).
    pub diffuse_ramp: bool,
    /// The glossy closure evaluates on the view-corrected normal.
    pub glossy_view_correction: bool,
}

/// `ensure_valid_reflection(Ng, I, N)`: turn `n` toward `ng` until the reflection of the view direction `i` about it
/// stays above the geometric surface (threshold `min(0.9 * i.ng, 0.01)`).
pub fn ensure_valid_reflection(ng: Vec3, i: Vec3, n: Vec3) -> Vec3 {
    let r = 2.0 * n.dot(i) * n - i;
    let iz = i.dot(ng);
    let threshold = (0.9 * iz).min(0.01);
    if ng.dot(r) >= threshold {
        return n;
    }
    let x = (n - n.dot(ng) * ng).normalize();
    let ix = i.dot(x);
    let a = ix * ix + iz * iz;
    let b = 2.0 * (a + iz * threshold);
    let c = (threshold + iz) * (threshold + iz);
    let disc = (b * b - 4.0 * a * c).max(0.0).sqrt();
    let nz2 = if ix < 0.0 {
        0.25 * (b + disc) / a
    } else {
        0.25 * (b - disc) / a
    };
    let nx = (1.0 - nz2).max(0.0).sqrt();
    let nz = nz2.max(0.0).sqrt();
    x * nx + ng * nz
}

/// The directions at one hit: `ns` the mapped shading normal, `nsm` the interpolated smooth normal (on the viewer's
/// side), `ng` the geometric normal, `vd` the direction to the viewer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HitFrame {
    /// Mapped shading normal.
    pub ns: Vec3,
    /// Interpolated smooth normal, on the viewer's side.
    pub nsm: Vec3,
    /// Geometric normal.
    pub ng: Vec3,
    /// Direction to the viewer.
    pub vd: Vec3,
}

/// Whether light from direction `l` reaches the closures at all: `ns` the mapped shading normal, `nsm` the smooth
/// normal (on the viewer's side), `ng` the geometric normal.
pub fn light_admitted(rules: Rules, ns: Vec3, nsm: Vec3, ng: Vec3, l: Vec3) -> bool {
    let nl = ns.dot(l);
    !((nl <= 0.0 && !rules.glossy_view_correction)
        || ng.dot(l) <= 0.0
        || (rules.smooth_cut && nsm.dot(l) <= 0.0))
}

/// The diffuse bump-shadowing factor (1 when the ramp is off).
pub fn diffuse_ramp(rules: Rules, nsm: Vec3, ns: Vec3, l: Vec3) -> f32 {
    if !rules.diffuse_ramp {
        return 1.0;
    }
    let ci = nsm.dot(l);
    let cd = nsm.dot(ns).clamp(1e-4, 1.0);
    let a2 = (0.125 * (1.0 - cd * cd) / (cd * cd)).min(1.0);
    let ti = (1.0 - ci * ci) / (ci * ci).max(1e-12);
    2.0 / (1.0 + (1.0 + a2 * ti).sqrt())
}

/// Single-scatter GGX parts for normal `n` (`nl` / `nv` its cosines, `hv` the half vector, `a2` = alpha^2): the
/// distribution `D` and `1 / (2 (lv + ll))`, the height-correlated visibility; the glossy term is `D * vis * 0.04`.
fn ggx_parts(n: Vec3, hv: Vec3, nl: f32, nv: f32, a2: f32) -> (f32, f32) {
    let nh = n.dot(hv).max(0.0);
    let q = nh * nh * (a2 - 1.0) + 1.0;
    let dg = a2 / (std::f32::consts::PI * (q * q));
    let lv = nv * (nl * nl * (1.0 - a2) + a2).sqrt();
    let ll = nl * (nv * nv * (1.0 - a2) + a2).sqrt();
    (dg, 0.5 / (lv + ll).max(1e-6))
}

/// One sun's contribution at a hit, as `(scale, per-channel term)`.
///
/// The caller adds `w * irr * scale * term[q]` (in that order of multiplication) for a light already admitted by
/// [`light_admitted`] and not in shadow. `alb` is the linear albedo, `rough` the roughness (alpha = rough^2).
pub fn sun_term(rules: Rules, f: HitFrame, l: Vec3, alb: [f32; 3], rough: f32) -> (f32, [f32; 3]) {
    let HitFrame { ns, nsm, ng, vd } = f;
    let rr = rough.max(0.02);
    let a2 = (rr * rr) * (rr * rr);
    let nv = ns.dot(vd).max(1e-4);
    let nl = ns.dot(l);
    let hv = (l + vd).normalize();
    let gd = diffuse_ramp(rules, nsm, ns, l);
    if !rules.glossy_view_correction {
        let (dg, vis) = ggx_parts(ns, hv, nl, nv, a2);
        let spec = dg * vis * 0.04;
        return (nl, alb.map(|a| a / std::f32::consts::PI * 0.96 * gd + spec));
    }
    let n_s = ensure_valid_reflection(ng, vd, ns);
    let nls = n_s.dot(l);
    let nvs = n_s.dot(vd).max(1e-4);
    let spec_s = if nls > 0.0 {
        let (dg, vis) = ggx_parts(n_s, hv, nls, nvs, a2);
        nls * dg * vis * 0.04
    } else {
        0.0
    };
    (
        1.0,
        alb.map(|a| nl.max(0.0) * a / std::f32::consts::PI * 0.96 * gd + spec_s),
    )
}

/// The divisor that turns a pixel's coverage-weighted sums into the written colour.
///
/// The coverage itself, or (with the alpha-zero rule, where the 8-bit alpha rounds to 0) the total filter weight,
/// i.e. the premultiplied colour.
pub const fn write_divisor(
    alpha_zero_premultiplied: bool,
    a8: u8,
    cov: f32,
    total_weight: f32,
) -> f32 {
    if alpha_zero_premultiplied && a8 == 0 {
        total_weight
    } else {
        cov
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // oracle: the probe columns (Blender 5.2.3 Cycles): diffuse radiance / ungated Lambert on a strip whose mapped normal
    // is tilted 62 deg from the smooth one; values read from probe_term/log_diffuse_n.txt
    #[test]
    fn diffuse_ramp_matches_reference_columns() {
        let cols = [
            (0.5149f32, 0.4815f32, 0.8030f32),
            (0.1415, 0.4597, 0.3545),
            (0.0206, 0.4607, 0.0615),
            (0.7301, 0.4665, 0.9160),
        ];
        let r = Rules {
            smooth_cut: true,
            diffuse_ramp: true,
            glossy_view_correction: false,
        };
        for (sl, sm, measured) in cols {
            // a smooth normal at +Z, light at cos sl, mapped normal at cos sm from the smooth one (in the light's plane)
            let nsm = Vec3::Z;
            let l = Vec3::new((1.0 - sl * sl).sqrt(), 0.0, sl);
            let ns = Vec3::new(-(1.0 - sm * sm).sqrt(), 0.0, sm);
            let g = diffuse_ramp(r, nsm, ns, l);
            assert!(
                (g - measured).abs() < 0.02,
                "smooth.l {sl}: ramp {g} vs reference {measured}"
            );
        }
        assert_eq!(
            diffuse_ramp(Rules::default(), Vec3::Z, Vec3::X, Vec3::Z),
            1.0
        );
    }

    #[test]
    fn smooth_cut_admits_no_light_behind_the_smooth_normal() {
        let r = Rules {
            smooth_cut: true,
            ..Rules::default()
        };
        let (ns, nsm, ng) = (Vec3::new(-0.8, 0.0, 0.6).normalize(), Vec3::Z, Vec3::Z);
        let behind = Vec3::new(-1.0, 0.0, -0.05).normalize();
        assert!(!light_admitted(r, ns, nsm, ng, behind) || ng.dot(behind) <= 0.0);
        let l = Vec3::new(-0.7, 0.0, -0.02).normalize();
        let ng_up = Vec3::new(-0.3, 0.0, 1.0).normalize();
        assert!(ns.dot(l) > 0.0 && ng_up.dot(l) > 0.0 && nsm.dot(l) < 0.0);
        assert!(
            !light_admitted(r, ns, nsm, ng_up, l),
            "cut on: light behind the smooth normal is refused"
        );
        assert!(
            light_admitted(Rules::default(), ns, nsm, ng_up, l),
            "cut off: the same light is admitted"
        );
    }

    // oracle: probe_view GGX columns at 15 deg elevation, sun from L = (-sin 60, 0, cos 60), roughness 0.5, energy 3:
    // where the reflection would go below the surface the reference gives a constant 0.078 (raw normal: 0)
    #[test]
    fn glossy_uses_the_view_corrected_normal() {
        let e = 15f32.to_radians();
        let vd = Vec3::new(-e.cos(), 0.0, e.sin());
        let l = Vec3::new(-(60f32.to_radians().sin()), 0.0, 60f32.to_radians().cos());
        let r = Rules {
            glossy_view_correction: true,
            ..Rules::default()
        };
        for tilt in [21f32, 38.0, 54.0, 70.0] {
            let ns = Vec3::new(tilt.to_radians().sin(), 0.0, tilt.to_radians().cos());
            let (s, t) = sun_term(
                r,
                HitFrame {
                    ns,
                    nsm: ns,
                    ng: Vec3::Z,
                    vd,
                },
                l,
                [0.0; 3],
                0.5,
            );
            let radiance = 3.0 * s * t[0] / 0.04; // white glossy (the probe's colour 1) instead of reflectance 0.04
            assert!(
                (radiance - 0.078).abs() < 0.003,
                "tilt {tilt}: {radiance} vs reference 0.078"
            );
        }
    }

    #[test]
    fn alpha_zero_writes_the_premultiplied_colour() {
        assert_eq!(write_divisor(true, 0, 0.001, 64.0), 64.0);
        assert_eq!(write_divisor(true, 3, 0.7, 64.0), 0.7);
        assert_eq!(write_divisor(false, 0, 0.001, 64.0), 0.001);
    }
}
