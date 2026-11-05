//! Loft: polar sections swept along a spline axis, the section a smooth function of the axis
//! parameter, each end flat or closed by a dome.
//!
//! # The shape
//!
//! - **Axis**: a uniform Catmull-Rom curve `C(s)` through the points `A₀ … Aₙ₋₁` (n ≥ 2),
//!   `s ∈ [0, n − 1]`, the end tangents continued by reflected phantom points
//!   (`A₋₁ = 2A₀ − A₁`, `Aₙ = 2Aₙ₋₁ − Aₙ₋₂`), so two points give a straight segment at uniform speed.
//! - **Frame**: `T = C'(s)/|C'(s)|`, `N = normalize(up − (up·T)T)`, `B = T × N`. `up` must not be
//!   parallel to the axis anywhere.
//! - **Section**: in the frame's plane, with `(u, v) = ((p − C)·N, (p − C)·B)`, `ρ = |(u, v)|` and
//!   `θ` the angle of `(u, v)` from `N` toward `B`, the section is `ρ ≤ r(θ)` with
//!   `r(θ) = a₀ + Σₖ₌₁..K (aₖ cos kθ + bₖ sin kθ)`, `K ≤` [`MAX_HARMONICS`]. `cos kθ, sin kθ` come from
//!   the rotation recurrence started at `(u/ρ, v/ρ)` (no trigonometric call).
//! - **Along the axis**: the coefficient rows `c₀ … c_{k−1}` (k ≥ 2, each `[a₀, a₁, b₁, …, a_K, b_K]`)
//!   are the control rows of a uniform cubic B-spline in `w = s (k − 1)/(n − 1)`, the end rows reflected
//!   (`c₋₁ = 2c₀ − c₁`, `c_k = 2c_{k−1} − c_{k−2}`) so the first and last rows are met exactly. The
//!   radius at `s` is the B-spline of the rows' radii at `θ` (the same as the radius of the
//!   B-spline of the coefficients).
//! - **Ends**: the planes through `C(0)` and `C(n − 1)` normal to the tangent there. An end with cap
//!   height `H > 0` is closed by the dome `(ρ/r_end(θ))² + (h/H)² ≤ 1` beyond the plane (`h` the height
//!   past it, `r_end` the end row): a hemisphere for a circle of radius `H`, an end that closes to a
//!   point for any star-shaped end section.
//!
//! # The field and its bound
//!
//! - In the section: `g = max(ρ − r, c)`, where `c = ρ₀ − r_min` holds the field constant in the core
//!   `ρ < ρ₀ = r_min / 2` (`r_min` a lower bound of every row's radius). Outside the core,
//!   `|∇(ρ − r)| ≤ √(1 + (r'_max/ρ₀)²) = L₂` (`r'_max` an upper bound of `|∂r/∂θ|`); with no angular
//!   term (`r'_max = 0`) there is no core and `L₂ = 1`. `d₂ = g / L₂`.
//! - Along the axis: `d₂` and the signed distance past the end planes `d_w` combine as an extrusion,
//!   `min(max(d₂, d_w), 0) + |max((d₂, d_w), 0)|`, divided by `L = (1 + 1.1·G)/(1 − 1.1·κR)`, from `R`
//!   the largest radius, `κ` the largest axis curvature (sampled, [`BOUND_SAMPLES`] per segment) and
//!   `G` the largest rate the radius changes per unit axis length (the rows' largest step, times
//!   `(k − 1)/(n − 1)` over the slowest axis speed) plus `R` times the frame's turn rate.
//! - A dome: `q = √((ρ/r_end)² + (h/H)²)`; `|∇q| ≤ max(√(1/r_min² + r'_max²/r_min⁴), 1/H) = L_q` (the two
//!   terms are orthogonal), so `(q − 1)/L_q` is 1-Lipschitz, cut by the half-space `h ≥ 0`, and joined
//!   to the body by `min`.
//! - The field is 1-Lipschitz and exact for a straight axis with a constant circle (and a
//!   hemispherical dome on it). Elsewhere it is a bound: never past the surface, but the march
//!   slows where `L` is large.
//!
//! Every bound is taken over the rows (the B-spline weights are convex, so the radius and its
//! angular derivative at any `s` lie within the rows' extremes), each row sampled at
//! `64(K + 1)` angles and widened by the sampling step times the next derivative's bound.

use crate::compiled::real::LawMinMax;
use glam::{Vec2, Vec3};

/// Axis samples per segment in the closest-point scan
pub const SCAN_PER_SEGMENT: u32 = 16;
/// Newton steps on the axis around the best scan sample
pub const AXIS_NEWTON: u32 = 4;
/// Samples per segment for the axis curvature and frame turn bounds
pub const BOUND_SAMPLES: u32 = 64;
/// The most harmonics a section may have
pub const MAX_HARMONICS: u32 = 64;
/// The most control points a curve section may have
pub const MAX_CONTROL_POINTS: u32 = 512;
/// Curve sections: scan samples per span
pub const CURVE_SCAN: u32 = 8;
/// Curve sections: Newton steps after the scan
pub const CURVE_NEWTON: u32 = 4;
/// The core value used when a loft has no angular term (no core)
const NO_CORE: f32 = -3.0e38;

/// Why a loft was refused
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoftError(pub String);

impl std::fmt::Display for LoftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "loft: {}", self.0)
    }
}

impl std::error::Error for LoftError {}

/// The version of the loft's bound law. Bounds are derived data: a stored tree carries them, and
/// loading accepts them only from this version of the law (an older tree's bounds mean something else,
/// so it is refused with a request to rebuild from its axis and rows, never read with a changed meaning)
pub const LOFT_LAW_VERSION: u32 = 2;

/// The bounds a loft's field is divided by, computed by [`validate`]; checked on load
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "LoftBoundsStored")]
pub struct LoftBounds {
    /// The bound law's version ([`LOFT_LAW_VERSION`])
    pub law: u32,
    /// The far bound `L_g` (the whole field divided by it is 1-Lipschitz)
    pub l: f32,
    /// The section bound `L₂`
    pub l2: f32,
    /// The core value `c` (≤ 0; a large negative number when there is no core)
    pub core: f32,
    /// The domes' bounds `L_q` (start, end); 1 for a flat end
    pub lq: [f32; 2],
    /// The clamp `m` of the local correction (0: the bound is the same on every span)
    pub m: f32,
    /// The local correction fades out between these distances from the axis curve (`None`: no fade,
    /// a straight axis); written as `null` when `None` (every field is always written, so binary forms
    /// that are not self-describing load it too)
    pub fade: Option<[f32; 2]>,
    /// The local bound `L(s)` at the B-spline knots (one per row), linear between them
    pub knots: Vec<f32>,
    /// The saturation width `B(s)` at the same knots (linear between them); empty with no correction
    #[serde(default)]
    pub bknots: Vec<f32>,
    /// A radius holding the whole surface around the axis curve (the largest radius and dome height)
    pub reach: f32,
    /// The far bound's pieces, [`PIECE_FLOATS`] each: the piece frame `T, N, B` and the offsets of its
    /// [`PIECE_PLANES`] planes (see [`far_bound`])
    pub pieces: Vec<f32>,
    /// Curve-mode caps per end: `[o_u, o_v, w, σ]` (apex offset in the end frame, the end section's
    /// erosion depth, the shear's Lipschitz factor); zeros for a flat end and in polar mode
    pub cap_curve: [[f32; 4]; 2],
}

/// The stored form of [`LoftBounds`], checked on load
#[derive(serde::Deserialize)]
struct LoftBoundsStored {
    #[serde(default)]
    law: u32,
    l: f32,
    l2: f32,
    core: f32,
    lq: [f32; 2],
    m: f32,
    #[serde(default)]
    fade: Option<[f32; 2]>,
    knots: Vec<f32>,
    #[serde(default)]
    bknots: Vec<f32>,
    reach: f32,
    #[serde(default)]
    pieces: Vec<f32>,
    #[serde(default)]
    cap_curve: [[f32; 4]; 2],
}

impl TryFrom<LoftBoundsStored> for LoftBounds {
    type Error = String;
    fn try_from(b: LoftBoundsStored) -> Result<Self, String> {
        if b.law != LOFT_LAW_VERSION {
            return Err(format!(
                "loft: stored bounds are from bound law {} (this build: {LOFT_LAW_VERSION}); rebuild the loft from its axis and rows",
                b.law
            ));
        }
        let shape_ok = !b.knots.is_empty()
            && (if b.m > 0.0 {
                b.bknots.len() == b.knots.len()
            } else {
                b.bknots.is_empty()
            })
            && b.pieces.len() % PIECE_FLOATS == 0
            && !b.pieces.is_empty()
            && [b.l, b.l2, b.core, b.lq[0], b.lq[1], b.m, b.reach]
                .iter()
                .all(|x| x.is_finite())
            && b.knots
                .iter()
                .chain(&b.bknots)
                .chain(&b.pieces)
                .all(|x| x.is_finite());
        if !shape_ok {
            return Err("loft: stored bounds are inconsistent (knots, saturation widths or far-bound pieces); rebuild the loft from its axis and rows".into());
        }
        Ok(Self {
            law: b.law,
            l: b.l,
            l2: b.l2,
            core: b.core,
            lq: b.lq,
            m: b.m,
            fade: b.fade,
            knots: b.knots,
            bknots: b.bknots,
            reach: b.reach,
            pieces: b.pieces,
            cap_curve: b.cap_curve,
        })
    }
}

impl LoftBounds {
    /// Every field equal bit for bit (the load check: `PartialEq` on `f32` would take `-0 == +0`)
    #[must_use]
    pub fn same_bits(&self, o: &Self) -> bool {
        fn v(a: &[f32], b: &[f32]) -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        }
        self.law == o.law
            && v(
                &[
                    self.l, self.l2, self.core, self.lq[0], self.lq[1], self.m, self.reach,
                ],
                &[o.l, o.l2, o.core, o.lq[0], o.lq[1], o.m, o.reach],
            )
            && match (self.fade, o.fade) {
                (None, None) => true,
                (Some(a), Some(b)) => v(&a, &b),
                _ => false,
            }
            && v(&self.knots, &o.knots)
            && v(&self.bknots, &o.bknots)
            && v(&self.pieces, &o.pieces)
            && v(self.cap_curve.as_flattened(), o.cap_curve.as_flattened())
    }
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

const REBUILD: &str = "rebuild the loft from its axis and rows";

/// The stored polar loft (`SdfNode::Loft`), rebuilt on load
#[derive(serde::Deserialize)]
struct PolarStored {
    axis: Vec<Vec3>,
    up: Vec3,
    harmonics: u32,
    rows: Vec<f32>,
    caps: [f32; 2],
    bounds: LoftBounds,
}

/// The fields of a polar loft read back from a stored tree: the bounds are recomputed from the
/// axis, rows and caps and must equal the stored ones bit for bit, as must the rows
///
/// # Errors
/// A row length that is not `2K + 1`, any input [`validate`] refuses, or stored bounds or rows that
/// differ from the recomputed ones
#[allow(
    clippy::type_complexity,
    reason = "the variant's fields, as serde's deserialize_with takes them"
)]
// ALLOW-UNWIRED: called by the derived Deserialize of SdfNode through #[serde(deserialize_with = "crate::primitives::loft::deserialize_polar")]
pub fn deserialize_polar<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<(Vec<Vec3>, Vec3, u32, Vec<f32>, [f32; 2], LoftBounds), D::Error> {
    use serde::de::Error;
    let s = <PolarStored as serde::Deserialize>::deserialize(d)?;
    let w = 2 * s.harmonics as usize + 1;
    if s.rows.len() % w != 0 {
        return Err(D::Error::custom(format!(
            "loft: {} stored row values are not rows of 2K + 1 = {w}; {REBUILD}",
            s.rows.len()
        )));
    }
    let rows: Vec<Vec<f32>> = s.rows.chunks(w).map(<[f32]>::to_vec).collect();
    let (flat, b) = validate(&s.axis, s.up, s.harmonics, &rows, s.caps)
        .map_err(|e| D::Error::custom(e.to_string()))?;
    if !same_bits(&flat, &s.rows) || !b.same_bits(&s.bounds) {
        return Err(D::Error::custom(format!("loft: stored bounds differ from the bounds recomputed from its axis, rows and caps; {REBUILD}")));
    }
    Ok((s.axis, s.up, s.harmonics, flat, s.caps, b))
}

/// The stored curve loft (`SdfNode::LoftCurves`), rebuilt on load
#[derive(serde::Deserialize)]
struct CurvesStored {
    axis: Vec<Vec3>,
    up: Vec3,
    section_points: u32,
    points: Vec<f32>,
    knots: Vec<f32>,
    #[serde(default)]
    caps: [[f32; 3]; 2],
    bounds: LoftBounds,
}

/// The fields of a curve loft read back from a stored tree, rebuilt as [`deserialize_polar`]
///
/// # Errors
/// A point count that is not rows of `2M`, any input [`validate_curves`] refuses, or stored bounds
/// or points that differ from the recomputed ones
#[allow(
    clippy::type_complexity,
    reason = "the variant's fields, as serde's deserialize_with takes them"
)]
// ALLOW-UNWIRED: called by the derived Deserialize of SdfNode through #[serde(deserialize_with = "crate::primitives::loft::deserialize_curves")]
pub fn deserialize_curves<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<
    (
        Vec<Vec3>,
        Vec3,
        u32,
        Vec<f32>,
        Vec<f32>,
        [[f32; 3]; 2],
        LoftBounds,
    ),
    D::Error,
> {
    use serde::de::Error;
    let s = <CurvesStored as serde::Deserialize>::deserialize(d)?;
    let w = 2 * s.section_points as usize;
    if w == 0 || s.points.len() % w != 0 {
        return Err(D::Error::custom(format!(
            "loft: {} stored point values are not rows of 2M = {w}; {REBUILD}",
            s.points.len()
        )));
    }
    let rows: Vec<Vec<glam::Vec2>> = s
        .points
        .chunks(w)
        .map(|r| r.chunks(2).map(|p| glam::Vec2::new(p[0], p[1])).collect())
        .collect();
    let (flat, b) = validate_curves(&s.axis, s.up, &s.knots, &rows, s.caps)
        .map_err(|e| D::Error::custom(e.to_string()))?;
    if !same_bits(&flat, &s.points) || !b.same_bits(&s.bounds) {
        return Err(D::Error::custom(format!("loft: stored bounds differ from the bounds recomputed from its axis, rows and caps; {REBUILD}")));
    }
    Ok((s.axis, s.up, s.section_points, flat, s.knots, s.caps, b))
}

/// Far bound: pieces per axis segment
pub const PIECES_PER_SEGMENT: usize = 8;
/// Far bound: planes per piece (16 in the piece's section plane, 16 tilted 45° toward +T, 16 toward
/// −T, then +T and −T)
pub const PIECE_PLANES: usize = 50;
/// Far bound: floats per piece (frame 9 + offsets)
pub const PIECE_FLOATS: usize = 9 + PIECE_PLANES;
/// cos and sin of 2πj/16
const DIR16: [(f32, f32); 16] = [
    (1.0, 0.0),
    (0.923_879_5, 0.382_683_43),
    (0.707_106_77, 0.707_106_77),
    (0.382_683_43, 0.923_879_5),
    (0.0, 1.0),
    (-0.382_683_43, 0.923_879_5),
    (-0.707_106_77, 0.707_106_77),
    (-0.923_879_5, 0.382_683_43),
    (-1.0, 0.0),
    (-0.923_879_5, -0.382_683_43),
    (-0.707_106_77, -0.707_106_77),
    (-0.382_683_43, -0.923_879_5),
    (0.0, -1.0),
    (0.382_683_43, -0.923_879_5),
    (0.707_106_77, -0.707_106_77),
    (0.923_879_5, -0.382_683_43),
];
/// 1/√2
const INV_SQRT2: f32 = 0.707_106_77;

/// Plane `k`'s unit normal in a piece frame
#[inline]
fn plane_normal(k: usize, t: Vec3, n: Vec3, b: Vec3) -> Vec3 {
    match k {
        48 => t,
        49 => -t,
        _ => {
            let (c, s) = DIR16[k % 16];
            let d = n * c + b * s;
            match k / 16 {
                0 => d,
                1 => (d + t) * INV_SQRT2,
                _ => (d - t) * INV_SQRT2,
            }
        }
    }
}

/// The far bound over the pieces of a flat block: min over pieces of max over each piece's planes of
/// `n·p − h`. Each piece's planes hold that piece's part of the surface, so the max is ≤ the distance to
/// that part and 1-Lipschitz; the min over pieces is ≤ the distance to the surface and 1-Lipschitz.
/// A piece is skipped when a subset of its planes (in-plane 0, 4, 8, 12 and ±T, itself a lower bound
/// of its max) already reaches the least so far: the min is the full one.
pub fn far_bound(p: Vec3, pieces: &[f32]) -> f32 {
    let mut best = f32::INFINITY;
    for pc in pieces.chunks_exact(PIECE_FLOATS) {
        let t = Vec3::new(pc[0], pc[1], pc[2]);
        let n = Vec3::new(pc[3], pc[4], pc[5]);
        let b = Vec3::new(pc[6], pc[7], pc[8]);
        let h = &pc[9..];
        let mut lb = f32::NEG_INFINITY;
        for k in [0usize, 4, 8, 12, 48, 49] {
            lb = lb.law_max(plane_normal(k, t, n, b).dot(p) - h[k]);
        }
        if lb >= best {
            continue;
        }
        let mut v = lb;
        for (k, hk) in h.iter().enumerate() {
            v = v.law_max(plane_normal(k, t, n, b).dot(p) - hk);
        }
        best = best.law_min(v);
    }
    best
}

/// The flat block of a loft (the bytecode's aux data and the shaders' module-scope array):
/// `[n, K or M, k, up.x, up.y, up.z, H₀, H₁, L_g, L₂, c, L_q0, L_q1, B, R_b, mode, pieces, fade₀, fade₁,
/// axis (3n), rows (k w), knots (k)]`, fade₀ < 0 meaning no fade (both −1)
/// with mode 0 (polar: `K` harmonics, rows of `w = 2K + 1` coefficients) or 1 (curve: `M` control points,
/// rows of `w = 2M` coordinates `u₀, v₀, u₁, v₁, …`, then the section knots `K₀ … K_M`)
pub const HEAD: usize = 27;

/// What a section is: the polar curve of `K` harmonics, or the closed periodic uniform cubic
/// B-spline of `M` control points
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoftSection {
    /// `r(θ) = a₀ + Σ (aₖ cos kθ + bₖ sin kθ)`, `K` harmonics
    Polar(u32),
    /// a closed counter-clockwise periodic uniform cubic B-spline of `M` control points
    Curve(u32),
}

impl LoftSection {
    fn count(self) -> u32 {
        match self {
            Self::Polar(k) | Self::Curve(k) => k,
        }
    }
    fn width(self) -> usize {
        match self {
            Self::Polar(k) => 2 * k as usize + 1,
            Self::Curve(m) => 2 * m as usize,
        }
    }
    fn mode(self) -> f32 {
        match self {
            Self::Polar(_) => 0.0,
            Self::Curve(_) => 1.0,
        }
    }
}

/// The flat block of a loft
pub fn flatten(
    axis: &[Vec3],
    up: Vec3,
    section: LoftSection,
    rows: &[f32],
    section_knots: &[f32],
    caps: [f32; 2],
    b: &LoftBounds,
) -> Vec<f32> {
    let k = rows.len() / section.width();
    let mut v = vec![
        axis.len() as f32,
        section.count() as f32,
        k as f32,
        up.x,
        up.y,
        up.z,
        caps[0],
        caps[1],
        b.l,
        b.l2,
        b.core,
        b.lq[0],
        b.lq[1],
        b.m,
        b.reach,
        section.mode(),
        (b.pieces.len() / PIECE_FLOATS) as f32,
        b.fade.map_or(-1.0, |f| f[0]),
        b.fade.map_or(-1.0, |f| f[1]),
        b.cap_curve[0][0],
        b.cap_curve[0][1],
        b.cap_curve[0][2],
        b.cap_curve[0][3],
        b.cap_curve[1][0],
        b.cap_curve[1][1],
        b.cap_curve[1][2],
        b.cap_curve[1][3],
    ];
    v.extend(axis.iter().flat_map(|a| [a.x, a.y, a.z]));
    v.extend_from_slice(rows);
    v.extend_from_slice(&b.knots);
    v.extend_from_slice(&b.bknots);
    v.extend_from_slice(section_knots);
    v.extend_from_slice(&b.pieces);
    v
}

/// The loft's data, read by the tree (slices) and the bytecode (the flat block) alike, so the two
/// run the same arithmetic
struct Loft<'a> {
    n: usize,
    /// 0 polar, 1 curve
    curve: bool,
    /// harmonics (polar) or control points (curve)
    kh: usize,
    /// row width
    w: usize,
    kr: usize,
    up: Vec3,
    caps: [f32; 2],
    lg: f32,
    l2: f32,
    core: f32,
    lq: [f32; 2],
    m: f32,
    fade: Option<[f32; 2]>,
    cap_curve: [[f32; 4]; 2],
    reach: f32,
    axis: &'a [f32],
    rows: &'a [f32],
    knots: &'a [f32],
    bknots: &'a [f32],
    /// section knots `K₀ … K_M` (curve mode)
    sknots: &'a [f32],
    /// the far bound's pieces
    pieces: &'a [f32],
}

impl<'a> Loft<'a> {
    fn from_flat(f: &'a [f32]) -> Self {
        let n = f[0] as usize;
        let kh = f[1] as usize;
        let kr = f[2] as usize;
        let curve = f[15] != 0.0;
        let w = if curve { 2 * kh } else { 2 * kh + 1 };
        Self {
            n,
            curve,
            kh,
            w,
            kr,
            up: Vec3::new(f[3], f[4], f[5]),
            caps: [f[6], f[7]],
            lg: f[8],
            l2: f[9],
            core: f[10],
            lq: [f[11], f[12]],
            m: f[13],
            fade: if f[17] < 0.0 {
                None
            } else {
                Some([f[17], f[18]])
            },
            cap_curve: [[f[19], f[20], f[21], f[22]], [f[23], f[24], f[25], f[26]]],
            reach: f[14],
            axis: &f[HEAD..HEAD + 3 * n],
            rows: &f[HEAD + 3 * n..HEAD + 3 * n + kr * w],
            knots: &f[HEAD + 3 * n + kr * w..HEAD + 3 * n + kr * w + kr],
            bknots: if f[13] > 0.0 {
                &f[HEAD + 3 * n + kr * w + kr..HEAD + 3 * n + kr * w + 2 * kr]
            } else {
                &[]
            },
            sknots: {
                let at = HEAD + 3 * n + kr * w + kr + if f[13] > 0.0 { kr } else { 0 };
                if curve {
                    &f[at..at + kh + 1]
                } else {
                    &[]
                }
            },
            pieces: {
                let at = HEAD
                    + 3 * n
                    + kr * w
                    + kr
                    + if f[13] > 0.0 { kr } else { 0 }
                    + if curve { kh + 1 } else { 0 };
                &f[at..at + f[16] as usize * PIECE_FLOATS]
            },
        }
    }

    #[inline]
    fn axis_pt(&self, i: isize) -> Vec3 {
        let at = |j: usize| Vec3::new(self.axis[3 * j], self.axis[3 * j + 1], self.axis[3 * j + 2]);
        let n = self.n as isize;
        if i < 0 {
            2.0 * at(0) - at(1)
        } else if i >= n {
            2.0 * at(self.n - 1) - at(self.n - 2)
        } else {
            at(i as usize)
        }
    }

    /// Axis point and derivative at `s ∈ [0, n − 1]`
    #[inline]
    fn axis_at(&self, s: f32) -> (Vec3, Vec3) {
        let last = (self.n - 1) as f32;
        let s = s.law_max(0.0).law_min(last);
        let i = (s as usize).min(self.n - 2);
        let t = s - i as f32;
        let i = i as isize;
        let (a, b, c, d) = (
            self.axis_pt(i - 1),
            self.axis_pt(i),
            self.axis_pt(i + 1),
            self.axis_pt(i + 2),
        );
        let t2 = t * t;
        let t3 = t2 * t;
        let k1 = c - a;
        let k2 = 2.0 * a - 5.0 * b + 4.0 * c - d;
        let k3 = -a + 3.0 * b - 3.0 * c + d;
        let pos = 0.5 * (2.0 * b + k1 * t + k2 * t2 + k3 * t3);
        let der = 0.5 * (k1 + 2.0 * k2 * t + 3.0 * k3 * t2);
        (pos, der)
    }

    /// The axis's second derivative at `s` (the segment's `k₂ + 3 k₃ t`)
    #[inline]
    fn axis_dd(&self, s: f32) -> Vec3 {
        let last = (self.n - 1) as f32;
        let s = s.law_max(0.0).law_min(last);
        let i = (s as usize).min(self.n - 2);
        let t = s - i as f32;
        let i = i as isize;
        let (a, b, c, d) = (
            self.axis_pt(i - 1),
            self.axis_pt(i),
            self.axis_pt(i + 1),
            self.axis_pt(i + 2),
        );
        let k2 = 2.0 * a - 5.0 * b + 4.0 * c - d;
        let k3 = -a + 3.0 * b - 3.0 * c + d;
        k2 + 3.0 * k3 * t
    }

    /// The fixed scan of the axis: every sample `s = k / SCAN_PER_SEGMENT`, `k = 0 … (n − 1) SCAN_PER_SEGMENT`;
    /// the least squared distance, ties to the smaller `s` (the first in `k` order)
    #[cfg(test)]
    fn scan_full(&self, p: Vec3) -> (f32, f32) {
        let steps = (self.n - 1) as u32 * SCAN_PER_SEGMENT;
        let h = 1.0 / SCAN_PER_SEGMENT as f32;
        let (mut best_s, mut best_d) = (0.0f32, f32::INFINITY);
        for k in 0..=steps {
            let s = k as f32 * h;
            let e = (self.axis_at(s).0 - p).length_squared();
            if e < best_d {
                best_d = e;
                best_s = s;
            }
        }
        (best_s, best_d)
    }

    /// The same scan, by segments: segment `i` (samples `k = 16 i … 16 i + 15`, the last segment also
    /// `k = (n − 1)·16`) lies in the capsule around the chord `Aᵢ Aᵢ₊₁` whose radius is the larger
    /// distance of the segment's two inner Bézier points to the chord (the segment is inside the
    /// hull of its four Bézier points); segments are visited from the least lower bound, and a
    /// segment whose bound (less a rounding margin) exceeds the best so far holds no sample that could
    /// win. Ties go to the smaller `s`, so the result is the full scan's, bit for bit.
    fn scan_culled(&self, p: Vec3) -> (f32, f32) {
        const MAX_SEG: usize = 256;
        let nseg = self.n - 1;
        let h = 1.0 / SCAN_PER_SEGMENT as f32;
        let mut lbs = [0.0f32; MAX_SEG];
        let mut order = [0usize; MAX_SEG];
        let segs = nseg.min(MAX_SEG);
        for i in 0..segs {
            let ii = i as isize;
            let (a, b, c, d) = (
                self.axis_pt(ii - 1),
                self.axis_pt(ii),
                self.axis_pt(ii + 1),
                self.axis_pt(ii + 2),
            );
            let seg_dist = |x: Vec3| -> f32 {
                let e = c - b;
                let t = ((x - b).dot(e) / e.dot(e)).law_max(0.0).law_min(1.0);
                (x - (b + e * t)).length()
            };
            let r = seg_dist(b + (c - a) / 6.0).law_max(seg_dist(c - (d - b) / 6.0));
            let margin = 1e-5 * ((p - b).length() + (c - b).length() + r);
            lbs[i] = seg_dist(p) - r - margin;
            order[i] = i;
        }
        // insertion sort by the bound (segments are few)
        for i in 1..segs {
            let mut j = i;
            while j > 0 && lbs[order[j - 1]] > lbs[order[j]] {
                order.swap(j - 1, j);
                j -= 1;
            }
        }
        let (mut best_s, mut best_d) = (0.0f32, f32::INFINITY);
        for &i in &order[..segs] {
            let lb = lbs[i];
            if lb > 0.0 && lb * lb > best_d {
                break;
            }
            let k0 = i as u32 * SCAN_PER_SEGMENT;
            let k1 = if i + 1 == nseg {
                k0 + SCAN_PER_SEGMENT
            } else {
                k0 + SCAN_PER_SEGMENT - 1
            };
            for k in k0..=k1 {
                let s = k as f32 * h;
                let e = (self.axis_at(s).0 - p).length_squared();
                if e < best_d || (e == best_d && s < best_s) {
                    best_d = e;
                    best_s = s;
                }
            }
        }
        (best_s, best_d)
    }

    /// The closest axis parameter to `p`: the scan, then fixed ternary steps
    fn closest_param(&self, p: Vec3) -> f32 {
        let last = (self.n - 1) as f32;
        let h = 1.0 / SCAN_PER_SEGMENT as f32;
        let (best_s, _) = self.scan_culled(p);
        let lo = (best_s - h).law_max(0.0);
        let hi = (best_s + h).law_min(last);
        // Newton on g(s) = (C(s) − p)·C'(s), g' = C'·C' + (C − p)·C'', kept within the bracket; a step
        // is taken only where g' > 0 (a minimum, not a maximum, of the distance)
        let mut sn = best_s;
        for _ in 0..AXIS_NEWTON {
            let (c, d) = self.axis_at(sn);
            let r = c - p;
            let g = r.dot(d);
            let gp = d.dot(d) + r.dot(self.axis_dd(sn));
            if gp > 0.0 {
                sn = (sn - g / gp).law_max(lo).law_min(hi);
            }
        }
        // a bracket touching an end: the end itself when it is closer (a minimum on the boundary,
        // where g ≠ 0)
        let mut best = sn;
        let mut be = (self.axis_at(best).0 - p).length_squared();
        for end in [0.0, last] {
            if lo == end || hi == end {
                let e = (self.axis_at(end).0 - p).length_squared();
                if e < be {
                    be = e;
                    best = end;
                }
            }
        }
        best
    }

    /// Row `i`'s radius at the angle with cosine and sine `(c1, s1)`, reflected rows past the ends
    #[inline]
    fn row_radius(&self, i: isize, c1: f32, s1: f32) -> f32 {
        let kr = self.kr as isize;
        if i < 0 {
            2.0 * self.row_radius(0, c1, s1) - self.row_radius(1, c1, s1)
        } else if i >= kr {
            2.0 * self.row_radius(kr - 1, c1, s1) - self.row_radius(kr - 2, c1, s1)
        } else {
            let w = self.w;
            let row = &self.rows[i as usize * w..(i as usize + 1) * w];
            polar_radius(row, c1, s1)
        }
    }

    /// The B-spline parameter at axis parameter `s`: its span and the fraction in it
    #[inline]
    fn span(&self, s: f32) -> (usize, f32) {
        let w = s * (self.kr - 1) as f32 / (self.n - 1) as f32;
        let w = w.law_max(0.0).law_min((self.kr - 1) as f32);
        let i = (w as usize).min(self.kr - 2);
        (i, w - i as f32)
    }

    /// The local bound `L(s)`: linear between the knots
    #[inline]
    fn local_b(&self, s: f32) -> f32 {
        let (i, t) = self.span(s);
        self.bknots[i] + (self.bknots[i + 1] - self.bknots[i]) * t
    }

    #[inline]
    fn local_l(&self, s: f32) -> f32 {
        let (i, t) = self.span(s);
        self.knots[i] + (self.knots[i + 1] - self.knots[i]) * t
    }

    /// The section radius at axis parameter `s` and the angle `(c1, s1)`
    #[inline]
    fn radius(&self, s: f32, c1: f32, s1: f32) -> f32 {
        let (i, t) = self.span(s);
        let i = i as isize;
        let t2 = t * t;
        let t3 = t2 * t;
        let u = 1.0 - t;
        let b0 = u * u * u / 6.0;
        let b1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;
        let b2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;
        let b3 = t3 / 6.0;
        b0 * self.row_radius(i - 1, c1, s1)
            + b1 * self.row_radius(i, c1, s1)
            + b2 * self.row_radius(i + 1, c1, s1)
            + b3 * self.row_radius(i + 2, c1, s1)
    }

    /// Control point `j` (periodic) of row `i`, reflected rows past the ends
    #[inline]
    fn row_point(&self, i: isize, j: usize) -> Vec2 {
        let kr = self.kr as isize;
        let at = |r: usize| {
            Vec2::new(
                self.rows[r * self.w + 2 * j],
                self.rows[r * self.w + 2 * j + 1],
            )
        };
        if i < 0 {
            2.0 * at(0) - at(1)
        } else if i >= kr {
            2.0 * at(self.kr - 1) - at(self.kr - 2)
        } else {
            at(i as usize)
        }
    }

    /// The B-spline weights along the axis at `s`: (first row, b₀ … b₃)
    #[inline]
    fn row_weights(&self, s: f32) -> (isize, [f32; 4]) {
        let (i, t) = self.span(s);
        let t2 = t * t;
        let t3 = t2 * t;
        let u = 1.0 - t;
        (
            i as isize - 1,
            [
                u * u * u / 6.0,
                (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
                (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
                t3 / 6.0,
            ],
        )
    }

    /// The section's control point `j` (any integer, periodic) at the weights
    #[inline]
    fn section_point(&self, i0: isize, b: &[f32; 4], j: isize) -> Vec2 {
        let jj = j.rem_euclid(self.kh as isize) as usize;
        b[0] * self.row_point(i0, jj)
            + b[1] * self.row_point(i0 + 1, jj)
            + b[2] * self.row_point(i0 + 2, jj)
            + b[3] * self.row_point(i0 + 3, jj)
    }

    /// The section knot `i` (any integer): the periodic extension of `K₀ … K_M`
    #[inline]
    fn sknot(&self, i: isize) -> f32 {
        let m = self.kh as isize;
        let period = self.sknots[self.kh] - self.sknots[0];
        let r = i.rem_euclid(m);
        let wraps = (i - r) / m;
        self.sknots[r as usize] + wraps as f32 * period
    }

    /// The section curve on span `j` at the weights (see [`closed_curve_on_span`]); the oracles of
    /// this module read the derivatives through it, the field calls the crate function directly
    #[cfg(test)]
    #[inline]
    fn curve_on_span(&self, i0: isize, b: &[f32; 4], j: isize, t: f32) -> (Vec2, Vec2, Vec2) {
        closed_curve_on_span(&|x| self.section_point(i0, b, x), &|x| self.sknot(x), j, t)
    }

    /// The signed distance in the section plane at `s` from `q` to the section curve (see [`closed_curve_distance`])
    fn curve_distance(&self, s: f32, q: Vec2) -> f32 {
        let (i0, b) = self.row_weights(s);
        closed_curve_distance(
            self.kh as isize,
            &|x| self.section_point(i0, &b, x),
            &|x| self.sknot(x),
            q,
        )
    }

    /// The curve dome at end `e` (0 at `s = 0`, 1 at `s = n − 1`): the end section at height `h`
    /// moved toward the apex by `o h/H` and eroded by `w (1 − √(1 − (h/H)²))`, cut by the end's
    /// half-space. 1-Lipschitz: in the end frame's orthonormal coordinates `(u, v, h)` the map
    /// `p ↦ (q, h)`, `q = (u, v) − o h/H`, is linear with norm `σ = c/2 + √(1 + c²/4)` (`c = |o|/H`,
    /// the largest singular value of `[[1, −c], [0, 1]]`); `(q, h) ↦ (d_S(q), h)` is 1-Lipschitz (the
    /// section's signed distance is); `(d, h) ↦ |((d + w)⁺/w, h/H)|` is `1/min(w, H)`-Lipschitz (the
    /// gradient of the norm is a unit vector scaled by `diag(1/w, 1/H)`). Scaled by `min(w, H)/σ`
    /// the composition is 1-Lipschitz, and `max(·, −h)` keeps it so
    fn curve_dome(&self, p: Vec3, e: usize) -> f32 {
        // the end section eroded by w (1 − √(1 − (h/H)²)) and sheared by o h/H: in (x, y) =
        // (d_S(q − o h/H) + w, h) the solid is (x⁺/w)² + (y/H)² ≤ 1; the map's Jacobian has largest
        // singular value σ = c/2 + √(1 + c²/4), c = |o|/H, and the norm's gradient is ≤ 1/min(w, H)
        let s = if e == 0 { 0.0 } else { (self.n - 1) as f32 };
        let (c, der) = self.axis_at(s);
        let (t, nn, bb) = frame(der, self.up);
        let out = if e == 0 { -t } else { t };
        let [ou, ov, w, sigma] = self.cap_curve[e];
        let hh = self.caps[e];
        let r = p - c;
        let h = r.dot(out);
        // the section at height h is the end section moved toward the apex by o h/H
        let q = Vec2::new(r.dot(nn), r.dot(bb)) - Vec2::new(ou, ov) * (h / hh);
        let d = self.curve_distance(s, q);
        let x = (d + w).law_max(0.0) / w;
        let y = h / hh;
        let v = ((x * x + y * y).sqrt() - 1.0) * w.law_min(hh) / sigma;
        v.law_max(-h)
    }

    fn dome(&self, p: Vec3, e: usize) -> f32 {
        let s = if e == 0 { 0.0 } else { (self.n - 1) as f32 };
        let (c, der) = self.axis_at(s);
        let (t, nn, bb) = frame(der, self.up);
        let out = if e == 0 { -t } else { t };
        let r = p - c;
        let h = r.dot(out);
        let (u, v) = (r.dot(nn), r.dot(bb));
        let rho = (u * u + v * v).sqrt();
        let (c1, s1) = if rho > 0.0 {
            (u / rho, v / rho)
        } else {
            (1.0, 0.0)
        };
        let i = if e == 0 { 0 } else { self.kr as isize - 1 };
        let re = self.row_radius(i, c1, s1);
        let a = rho / re;
        let bq = h / self.caps[e];
        let q = (a * a + bq * bq).sqrt();
        ((q - 1.0) / self.lq[e]).law_max(-h)
    }

    fn eval(&self, p: Vec3) -> f32 {
        let s = self.closest_param(p);
        let (c, der) = self.axis_at(s);
        let (_, nn, bb) = frame(der, self.up);
        let r = p - c;
        let (u, v) = (r.dot(nn), r.dot(bb));
        let rho = (u * u + v * v).sqrt();
        let (c1, s1) = if rho > 0.0 {
            (u / rho, v / rho)
        } else {
            (1.0, 0.0)
        };
        let d2 = if self.curve {
            self.curve_distance(s, Vec2::new(u, v))
        } else {
            let g = (rho - self.radius(s, c1, s1)).law_max(self.core);
            g / self.l2
        };
        let (c0, d0) = self.axis_at(0.0);
        let (ce, de) = self.axis_at((self.n - 1) as f32);
        let dw = (c0 - p)
            .dot(d0.normalize())
            .law_max((p - ce).dot(de.normalize()));
        let o = Vec2::new(d2.law_max(0.0), dw.law_max(0.0));
        // F, then f = F/L_g + clamp(F (1/L(s) − 1/L_g), −m, m): F/L(s) near the surface, F/L_g ± m far
        let big_f = d2.law_max(dw).law_min(0.0) + o.length();
        // f = F/L_g + (1/L(s) − 1/L_g) sat(F) ψ: sat(F) = F/√(1 + (F/B)²) (smooth, |sat| ≤ B,
        // 0 < sat' ≤ 1), ψ a smoothstep from 1 at |p − C(s)| ≤ ρ* to 0 at ≥ ρ** (∞: no fade)
        let mut f = big_f / self.lg;
        if self.m > 0.0 {
            let x = big_f / self.local_b(s);
            let sat = big_f / (1.0 + x * x).sqrt();
            let dist = r.length();
            let psi = match self.fade {
                None => 1.0,
                Some([fa, _]) if dist <= fa => 1.0,
                Some([_, fb]) if dist >= fb => 0.0,
                Some([fa, fb]) => {
                    let u = (fb - dist) / (fb - fa);
                    u * u * (3.0 - 2.0 * u)
                }
            };
            f += (1.0 / self.local_l(s) - 1.0 / self.lg) * sat * psi;
        }
        for e in 0..2 {
            if self.caps[e] > 0.0 {
                f = f.law_min(if self.curve {
                    self.curve_dome(p, e)
                } else {
                    self.dome(p, e)
                });
            }
        }
        // the far bound (pieces of planes holding the surface), the larger of the two
        // outside only: the pieces hold the surface, so the far bound is ≤ 0 on it and the switch at
        // f = 0 is continuous; inside the loft's own field stands
        // lowered by τ = max(B/4, R_b/80), so that near the surface (within about τ L/(L − 1)) the loft's own
        // smooth field stands and the planes' directions never set the normal; still ≤ the distance
        if self.pieces.is_empty() || f <= 0.0 {
            f
        } else {
            // joined C1: f + φ(b − f) with φ(x) = 0 (x ≤ 0), x²/(2k) (x ≤ k), x − k/2 (beyond):
            // 0 ≤ φ(x) ≤ max(0, x) keeps the result ≤ max(f, b) ≤ the distance, and φ' ∈ [0, 1]
            // makes its gradient a convex mix of the two (1-Lipschitz)
            let tau = self.m.law_max(0.0125 * self.reach);
            let k = self.m.law_max(0.0125 * self.reach);
            let x = far_bound(p, self.pieces) - tau - f;
            let phi = if x <= 0.0 {
                0.0
            } else if x <= k {
                x * x / (2.0 * k)
            } else {
                x - 0.5 * k
            };
            f + phi
        }
    }
}

/// The section curve on span `j` (`K_j ≤ t ≤ K_{j+1}`) by de Boor's algorithm with the span's
/// control points `P_{j−1} … P_{j+2}` and knots `K_{j−2} … K_{j+3}`: point, first and second
/// derivative (the derivative curves' control points `3ΔP/ΔK` and `2ΔQ/ΔK`)
#[inline]
pub(crate) fn closed_curve_on_span(
    point: &impl Fn(isize) -> Vec2,
    knot: &impl Fn(isize) -> f32,
    j: isize,
    t: f32,
) -> (Vec2, Vec2, Vec2) {
    let mut d = [Vec2::ZERO; 4];
    for (k, dk) in d.iter_mut().enumerate() {
        *dk = point(j - 1 + k as isize);
    }
    let mut u = [0.0f32; 6];
    for (k, uk) in u.iter_mut().enumerate() {
        *uk = knot(j - 2 + k as isize);
    }
    // first derivative control points (degree 2) and second (degree 1)
    let mut q = [Vec2::ZERO; 3];
    for k in 0..3 {
        q[k] = (d[k + 1] - d[k]) * (3.0 / (u[k + 3] - u[k]));
    }
    let r0 = (q[1] - q[0]) * (2.0 / (u[3] - u[1]));
    let r1 = (q[2] - q[1]) * (2.0 / (u[4] - u[2]));
    // de Boor, degree 3
    let mut p = d;
    for r in 1..=3usize {
        for i in (r..=3).rev() {
            let a = (t - u[i - 1]) / (u[i + 3 - r] - u[i - 1]);
            p[i] = p[i - 1] * (1.0 - a) + p[i] * a;
        }
    }
    // degree 2
    for r in 1..=2usize {
        for i in (r..=2).rev() {
            let a = (t - u[i]) / (u[i + 3 - r] - u[i]);
            q[i] = q[i - 1] * (1.0 - a) + q[i] * a;
        }
    }
    let a = (t - u[2]) / (u[3] - u[2]);
    (p[3], q[2], r0 * (1.0 - a) + r1 * a)
}

/// The signed distance from `q` to a closed cubic B-spline curve of `m` spans (control points `point(j)`, any
/// integer `j`, periodic; knots `knot(i)`, the periodic extension), shared by the loft's curve sections and the
/// smooth section stack (negative inside). Per span, a lower bound of the distance (the circle around the span's four control
/// points, which hold the span) is compared with the best so far, and only a span that may come
/// closer is scanned at [`CURVE_SCAN`] even steps: the best sample is the one a full scan finds.
/// Then [`CURVE_NEWTON`] Newton steps on `(c − q)·c' = 0`, kept within the span and one scan step
/// of the best sample; the closer of the two is kept. The sign from the side of the tangent
/// there (counter-clockwise: inside on the left).
pub(crate) fn closed_curve_distance(
    m: isize,
    point: &impl Fn(isize) -> Vec2,
    knot: &impl Fn(isize) -> f32,
    q: Vec2,
) -> f32 {
    let (mut bj, mut bt, mut be) = (0isize, knot(0), f32::INFINITY);
    for j in 0..m {
        let (p0, p1, p2, p3) = (point(j - 1), point(j), point(j + 1), point(j + 2));
        let c = (p0 + p1 + p2 + p3) * 0.25;
        let rad = (p0 - c)
            .length()
            .law_max((p1 - c).length())
            .law_max((p2 - c).length())
            .law_max((p3 - c).length());
        let lb = (q - c).length() - rad;
        if lb > 0.0 && lb * lb >= be {
            continue;
        }
        let (k0, k1) = (knot(j), knot(j + 1));
        for k in 0..CURVE_SCAN {
            let t = k0 + (k1 - k0) * (k as f32 / CURVE_SCAN as f32);
            let e = (closed_curve_on_span(point, knot, j, t).0 - q).length_squared();
            if e < be {
                be = e;
                bt = t;
                bj = j;
            }
        }
    }
    let (k0, k1) = (knot(bj), knot(bj + 1));
    // one scan step either side, into the neighbouring span when the best sample is a span's end
    let (lo, hi) = (
        bt - (k1 - k0) / CURVE_SCAN as f32,
        bt + (k1 - k0) / CURVE_SCAN as f32,
    );
    let lo = lo.law_max(knot(bj - 1));
    let hi = hi.law_min(knot(bj + 2));
    let on = |t: f32| -> isize {
        if t < k0 {
            bj - 1
        } else if t > k1 {
            bj + 1
        } else {
            bj
        }
    };
    let mut t = bt;
    for _ in 0..CURVE_NEWTON {
        let (c, d, dd) = closed_curve_on_span(point, knot, on(t), t);
        let r = c - q;
        let g = r.dot(d);
        let gp = d.dot(d) + r.dot(dd);
        if gp > 0.0 {
            t = (t - g / gp).law_max(lo).law_min(hi);
        }
    }
    let (c, d, _) = closed_curve_on_span(point, knot, on(t), t);
    let (mut cc, mut dc) = (c, d);
    if (c - q).length_squared() > be {
        let (c0, d0, _) = closed_curve_on_span(point, knot, bj, bt);
        cc = c0;
        dc = d0;
    }
    let r = q - cc;
    let dist = r.length();
    if dc.x * r.y - dc.y * r.x > 0.0 {
        -dist
    } else {
        dist
    }
}

/// A row's polar radius at the angle with cosine and sine `(c1, s1)`: the rotation recurrence
#[inline]
fn polar_radius(row: &[f32], c1: f32, s1: f32) -> f32 {
    let mut r = row[0];
    let (mut ck, mut sk) = (c1, s1);
    let kh = (row.len() - 1) / 2;
    for k in 0..kh {
        r += row[1 + 2 * k] * ck + row[2 + 2 * k] * sk;
        let next_c = ck * c1 - sk * s1;
        sk = sk * c1 + ck * s1;
        ck = next_c;
    }
    r
}

/// The frame at a derivative: (T, N, B)
#[inline]
pub fn frame(der: Vec3, up: Vec3) -> (Vec3, Vec3, Vec3) {
    let t = der.normalize();
    let n = (up - t * up.dot(t)).normalize();
    let b = t.cross(n);
    (t, n, b)
}

/// The loft's distance at `p` from the flat block of [`flatten`]
pub fn sdf_loft_flat(p: Vec3, flat: &[f32]) -> f32 {
    Loft::from_flat(flat).eval(p)
}

/// The tree's loft distance
#[allow(clippy::too_many_arguments)]
pub fn sdf_loft(
    p: Vec3,
    axis: &[Vec3],
    up: Vec3,
    section: LoftSection,
    rows: &[f32],
    section_knots: &[f32],
    caps: [f32; 2],
    b: &LoftBounds,
) -> f32 {
    sdf_loft_flat(p, &flatten(axis, up, section, rows, section_knots, caps, b))
}

/// A box holding the loft's surface, from the flat block: each axis segment is the cubic Bézier
/// `b, b + (c − a)/6, c − (d − b)/6, c`, inside the hull of those four points; the box of the axis's
/// Bézier points grown by `R_b`
pub fn aabb_flat(flat: &[f32]) -> (Vec3, Vec3) {
    let d = Loft::from_flat(flat);
    let n = d.n as isize;
    let mut mn = Vec3::splat(f32::INFINITY);
    let mut mx = Vec3::splat(f32::NEG_INFINITY);
    for i in 0..n - 1 {
        let (a, b, c, e) = (
            d.axis_pt(i - 1),
            d.axis_pt(i),
            d.axis_pt(i + 1),
            d.axis_pt(i + 2),
        );
        for q in [b, b + (c - a) / 6.0, c - (e - b) / 6.0, c] {
            mn = mn.law_min(q);
            mx = mx.law_max(q);
        }
    }
    // R_b holds every section and dome around the axis (the B-spline weights along the axis are convex)
    let r = d.reach;
    (mn - Vec3::splat(r), mx + Vec3::splat(r))
}

/// Per row: (the least radius, the largest radius, the largest |∂r/∂θ|), each widened by the sampling
fn row_bounds(row: &[f32]) -> (f32, f32, f32) {
    let kh = (row.len() - 1) / 2;
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for k in 0..kh {
        let a = row[1 + 2 * k].abs() + row[2 + 2 * k].abs();
        s1 += (k + 1) as f32 * a;
        s2 += ((k + 1) * (k + 1)) as f32 * a;
    }
    let m = 64 * (kh + 1);
    let step = std::f64::consts::TAU / m as f64;
    let (mut lo, mut hi, mut dmax) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f32);
    for j in 0..m {
        let th = j as f64 * step;
        let (c1, s1_) = (
            alice_det_math::cos(th as f32),
            alice_det_math::sin(th as f32),
        );
        let r = polar_radius(row, c1, s1_);
        lo = lo.law_min(r);
        hi = hi.law_max(r);
        // ∂r/∂θ = Σ k (−aₖ sin kθ + bₖ cos kθ)
        let (mut ck, mut sk, mut dr) = (c1, s1_, 0.0f32);
        for k in 0..kh {
            dr += (k + 1) as f32 * (-row[1 + 2 * k] * sk + row[2 + 2 * k] * ck);
            let nc = ck * c1 - sk * s1_;
            sk = sk * c1 + ck * s1_;
            ck = nc;
        }
        dmax = dmax.law_max(dr.abs());
    }
    let half = (step / 2.0) as f32;
    (lo - half * s1, hi + half * s1, dmax + half * s2)
}

/// Checks a loft's input and computes its bounds
pub fn validate(
    axis: &[Vec3],
    up: Vec3,
    harmonics: u32,
    rows: &[Vec<f32>],
    caps: [f32; 2],
) -> Result<(Vec<f32>, LoftBounds), LoftError> {
    let err = |s: String| Err(LoftError(s));
    if axis.len() < 2 || axis.len() > 257 {
        return err(format!(
            "the axis needs 2 to 257 points, it has {}",
            axis.len()
        ));
    }
    if harmonics > MAX_HARMONICS {
        return err(format!("{harmonics} harmonics, at most {MAX_HARMONICS}"));
    }
    if !axis.iter().all(|a| a.is_finite())
        || !up.is_finite()
        || !caps.iter().all(|c| c.is_finite())
        || rows.iter().flatten().any(|v| !v.is_finite())
    {
        return err("the axis, up, caps and rows must be finite".into());
    }
    let w = 2 * harmonics as usize + 1;
    if rows.len() < 2 || rows.iter().any(|r| r.len() != w) {
        return err(format!("2 rows or more, each of 2K + 1 = {w} coefficients"));
    }
    if caps.iter().any(|c| *c < 0.0) {
        return err("a cap height must be 0 (flat) or positive".into());
    }
    for pair in axis.windows(2) {
        if (pair[1] - pair[0]).length_squared() == 0.0 {
            return err("two consecutive axis points coincide".into());
        }
    }
    let (mut rmin, mut rmax, mut dmax) = (f32::INFINITY, 0.0f32, 0.0f32);
    let mut end = [(0.0f32, 0.0f32); 2];
    for (i, row) in rows.iter().enumerate() {
        let (lo, hi, d) = row_bounds(row);
        rmin = rmin.law_min(lo);
        rmax = rmax.law_max(hi);
        dmax = dmax.law_max(d);
        if i == 0 {
            end[0] = (lo, d);
        }
        if i == rows.len() - 1 {
            end[1] = (lo, d);
        }
    }
    if rmin <= 0.0 {
        return err(format!("a section's radius reaches 0 or below (least radius bound {rmin}): the centre must be inside every section"));
    }
    // the step between consecutive rows j, j + 1, as a bound of |r_{j+1}(θ) − r_j(θ)| (Σ |Δ coefficient|)
    let steps: Vec<f32> = rows
        .windows(2)
        .map(|pair| {
            pair[0]
                .iter()
                .zip(&pair[1])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, |x, y| x + y)
        })
        .collect();
    let flat_rows: Vec<f32> = rows.iter().flatten().copied().collect();
    let (l2, core) = if dmax == 0.0 {
        (1.0, NO_CORE)
    } else {
        let rho0 = 0.5 * rmin;
        ((1.0 + (dmax / rho0) * (dmax / rho0)).sqrt(), rho0 - rmin)
    };
    let mut lq = [1.0f32; 2];
    for e in 0..2 {
        if caps[e] > 0.0 {
            let (lo, d) = end[e];
            let sec = (1.0 / (lo * lo) + d * d / (lo * lo * lo * lo)).sqrt();
            lq[e] = sec.law_max(1.0 / caps[e]);
        }
    }
    let reach = rmax.law_max(caps[0]).law_max(caps[1]);
    let b = finish(
        axis,
        up,
        LoftSection::Polar(harmonics),
        &flat_rows,
        caps,
        &steps,
        rmax,
        (l2, core, lq, reach),
        (
            &[],
            rmax + dmax,
            [(end[0].0, end[0].1), (end[1].0, end[1].1)],
            [[0.0; 4]; 2],
        ),
    )?;
    Ok((flat_rows, b))
}

/// Checks a curve-section loft's input and computes its bounds
pub fn validate_curves(
    axis: &[Vec3],
    up: Vec3,
    section_knots: &[f32],
    rows: &[Vec<Vec2>],
    caps: [[f32; 3]; 2],
) -> Result<(Vec<f32>, LoftBounds), LoftError> {
    let err = |s: String| Err(LoftError(s));
    if axis.len() < 2 || axis.len() > 257 {
        return err(format!(
            "the axis needs 2 to 257 points, it has {}",
            axis.len()
        ));
    }
    if !axis.iter().all(|a| a.is_finite())
        || !up.is_finite()
        || rows.iter().flatten().any(|v| !v.is_finite())
    {
        return err("the axis, up and control points must be finite".into());
    }
    if rows.len() < 2 {
        return err("2 rows or more".into());
    }
    let m = rows[0].len();
    if m < 4 || m > MAX_CONTROL_POINTS as usize || rows.iter().any(|r| r.len() != m) {
        return err(format!(
            "every row needs the same number of control points, 4 to {MAX_CONTROL_POINTS}"
        ));
    }
    for pair in axis.windows(2) {
        if (pair[1] - pair[0]).length_squared() == 0.0 {
            return err("two consecutive axis points coincide".into());
        }
    }
    if section_knots.len() != m + 1
        || !section_knots.iter().all(|k| k.is_finite())
        || section_knots.windows(2).any(|w| w[1] <= w[0])
    {
        return err(format!(
            "the section knots must be M + 1 = {} finite values rising strictly",
            m + 1
        ));
    }
    for row in rows {
        // twice the signed area of the control polygon
        let area: f32 = (0..m).map(|j| row[j].perp_dot(row[(j + 1) % m])).sum();
        if area <= 0.0 {
            return err("every section must run counter-clockwise (positive area)".into());
        }
    }
    // every row, and the blended sections between rows, must be simple: a crossing section has no
    // consistent inside and the field jumps across the crossing
    if let Some(at) = first_crossing_section(rows, section_knots) {
        return err(format!("the section {at} crosses itself (every section between the rows must be a simple loop)"));
    }
    // the B-spline weights are convex (along the section and along the axis), so every section point
    // lies within the largest control point length, and the section moves along the axis at most at
    // the largest control point step
    let rmax = rows
        .iter()
        .flatten()
        .map(|p| p.length())
        .fold(0.0f32, f32::law_max);
    let steps: Vec<f32> = rows
        .windows(2)
        .map(|pair| {
            pair[0]
                .iter()
                .zip(&pair[1])
                .map(|(a, b)| (*a - *b).length())
                .fold(0.0f32, f32::law_max)
        })
        .collect();
    let flat_rows: Vec<f32> = rows.iter().flatten().flat_map(|p| [p.x, p.y]).collect();
    // |c'| on span j is at most 3 max |ΔP| over the smallest three-span knot interval
    let mut dp = 0.0f32;
    for row in rows {
        for j in 0..m {
            dp = dp.law_max((row[j] - row[(j + m - 1) % m]).length());
        }
    }
    let period = section_knots[m] - section_knots[0];
    let kn = |i: isize| -> f32 {
        let r = i.rem_euclid(m as isize);
        section_knots[r as usize] + ((i - r) / m as isize) as f32 * period
    };
    let min3 = (0..m as isize)
        .map(|j| kn(j + 2) - kn(j - 1))
        .fold(f32::INFINITY, f32::law_min);
    let cder = 3.0 * dp / min3;
    // caps: height H ≥ 0 (0 = flat), apex offset o in the end frame; the erosion depth w = the end
    // section's inradius (the deepest interior point: a 64 × 64 grid of its control box, then a fixed
    // pattern search; any positive w gives a valid field, the inradius closes the cap to a point), the shear
    // factor σ = c/2 + √(1 + c²/4), c = |o|/H
    if caps.iter().flatten().any(|v| !v.is_finite()) || caps.iter().any(|c| c[0] < 0.0) {
        return err(
            "a cap's height must be 0 (flat) or positive, and its apex offset finite".into(),
        );
    }
    let mut cap_curve = [[0.0f32; 4]; 2];
    let mut heights = [0.0f32; 2];
    for e in 0..2 {
        let [hh, ou, ov] = caps[e];
        if hh <= 0.0 {
            continue;
        }
        let row = if e == 0 {
            &rows[0]
        } else {
            &rows[rows.len() - 1]
        };
        let point = |j: isize| row[j.rem_euclid(m as isize) as usize];
        let knot = |i: isize| {
            let r = i.rem_euclid(m as isize);
            section_knots[r as usize] + ((i - r) / m as isize) as f32 * period
        };
        let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
        for p in row {
            lo = lo.law_min(*p);
            hi = hi.law_max(*p);
        }
        let depth = |q: Vec2| -closed_curve_distance(m as isize, &point, &knot, q);
        let (mut w, mut at) = (f32::NEG_INFINITY, lo);
        for i in 0..64 {
            for j in 0..64 {
                let q =
                    lo + (hi - lo) * Vec2::new((i as f32 + 0.5) / 64.0, (j as f32 + 0.5) / 64.0);
                let dq = depth(q);
                if dq > w {
                    w = dq;
                    at = q;
                }
            }
        }
        // then a fixed pattern search from the best cell: 8 neighbours at a step halved on no gain
        let mut step = (hi - lo).max_element() / 64.0;
        for _ in 0..48 {
            let mut moved = false;
            for (dx, dy) in [
                (1.0f32, 0.0f32),
                (-1.0, 0.0),
                (0.0, 1.0),
                (0.0, -1.0),
                (1.0, 1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
                (-1.0, -1.0),
            ] {
                let q = at + Vec2::new(dx, dy) * step;
                let dq = depth(q);
                if dq > w {
                    w = dq;
                    at = q;
                    moved = true;
                }
            }
            if !moved {
                step *= 0.5;
            }
        }
        if w <= 0.0 {
            return err("a capped end's section has no interior".into());
        }
        let c = Vec2::new(ou, ov).length() / hh;
        cap_curve[e] = [ou, ov, w, 0.5 * c + (1.0 + 0.25 * c * c).sqrt()];
        heights[e] = hh;
    }
    let reach = rmax
        + (0..2)
            .map(|e| heights[e] + Vec2::new(caps[e][1], caps[e][2]).length())
            .fold(0.0f32, f32::law_max);
    let b = finish(
        axis,
        up,
        LoftSection::Curve(m as u32),
        &flat_rows,
        heights,
        &steps,
        rmax,
        (1.0, NO_CORE, [1.0, 1.0], reach),
        (section_knots, cder, [(0.0, 0.0); 2], cap_curve),
    )?;
    Ok((flat_rows, b))
}

/// The loft field's lower bound in terms of `d = dist(p, B)`, `B` any set holding the axis curve
/// (the box of its Bézier control points): `field(p) ≥ scale · d − margin`, as `(scale, margin)`;
/// `None` when a factor is not positive and finite
///
/// Derivation (`C(s*)` the axis point the field projects `p` on, `r = p − C(s*)`, `ρ` its part
/// across the axis, `h` along it; `|r| ≥ dist(p, curve) ≥ d`):
/// - near field `F/L_g + (1/L(s) − 1/L_g) sat(F) ψ`: outside, `sat(F) ≥ 0`, `ψ ≥ 0` and
///   `L(s) ≤ L_g`, so it is `≥ F/L_g`. `F ≥ |(d₂⁺, d_w⁺)|` with the section distance
///   `d₂ ≥ (ρ − R)/ℓ₂` (polar: `(max(ρ − r(θ), core))/ℓ₂`, `r(θ) ≤ R`; curve: the curve lies in the
///   disc of its control points, so `d₂ ≥ |q| − R` with `ℓ₂ = 1`) and the end half-spaces
///   `d_w ≥ |h|` (`h = 0` at an interior `s*`, the stationary point of `|p − C(s)|`; at an end `h` is
///   the distance past the end plane). So `F ≥ |((ρ − R)⁺, h)| / max(ℓ₂, 1) ≥ (|r| − R)/max(ℓ₂, 1)`:
///   `κ = 1/(L_g max(ℓ₂, 1))`, radius `R` = the largest section radius over the rows (every section
///   is a convex mix of rows, the reflected rows of the end spans included)
/// - polar dome at end `e`: `(|(ρ/r_e, h/H)| − 1)/λ_e ≥ (|r| − R_e)/(R_e λ_e)`, `R_e = max(r_e, H)`
/// - curve dome at end `e`: in `(x, y) = ((d_S(q′) + w)⁺/w, h/H)`, `q′ = q + o h/H`, the shear
///   `(a, b) ↦ (a − c b, b)` (`c = |o|/H`) has smallest singular value `1/σ` and `σ² ≥ 1 + c²`, so
///   `|((a − c b − R′)⁺, b)| ≥ |r|/σ − R′` (`R′ = (R − w)⁺`) and the dome is
///   `≥ (m/(σ² M)) (|r| − σ (R′ + M))`, `m = min(w, H)`, `M = max(w, H)`
/// - the far bound's C1 join only adds `φ ≥ 0`, and `max(·, −h)` only raises a dome
///
/// Every term is `κ_i (|r_i| − R_i) ≥ κ_i d − κ_i R_i ≥ (min κ) d − max(κ_i R_i)`.
#[must_use]
pub fn lower_bound_factors(
    section: LoftSection,
    rows: &[f32],
    caps: [f32; 2],
    b: &LoftBounds,
) -> Option<(f32, f32)> {
    let w = section.width();
    if w == 0 || rows.len() < 2 * w || rows.len() % w != 0 {
        return None;
    }
    let kr = rows.len() / w;
    let row = |i: usize| &rows[i * w..(i + 1) * w];
    // the end spans read a reflected row 2R₀ − R₁, but folded into the real rows their weights
    // (2b₀ + b₁, b₂ − b₀, b₃) are ≥ 0 with sum 1 (b₂ ≥ b₀ on [0, 1]): every section is a convex mix
    // of real rows, so the real rows bound the radius
    let all = (0..kr).map(|i| row(i).to_vec());
    let polar_r = |r: &[f32]| -> f32 {
        r[0].abs()
            + r[1..]
                .chunks(2)
                .map(|c| (c[0] * c[0] + c[1] * c[1]).sqrt())
                .sum::<f32>()
    };
    let curve_r = |r: &[f32]| -> f32 {
        r.chunks(2)
            .map(|c| (c[0] * c[0] + c[1] * c[1]).sqrt())
            .fold(0.0f32, f32::law_max)
    };
    let curve = matches!(section, LoftSection::Curve(_));
    let radius = all
        .map(|r| if curve { curve_r(&r) } else { polar_r(&r) })
        .fold(0.0f32, f32::law_max);
    let mut terms: Vec<(f32, f32)> = vec![(1.0 / (b.l * b.l2.law_max(1.0)), radius)];
    for e in 0..2 {
        let hh = caps[e];
        if hh <= 0.0 {
            continue;
        }
        if curve {
            let [_, _, cw, sigma] = b.cap_curve[e];
            let (m, mm) = (cw.law_min(hh), cw.law_max(hh));
            terms.push((
                m / (sigma * sigma * mm),
                sigma * ((radius - cw).law_max(0.0) + mm),
            ));
        } else {
            let re = polar_r(row(if e == 0 { 0 } else { kr - 1 })).law_max(hh);
            terms.push((1.0 / (re * b.lq[e]), re));
        }
    }
    let scale = terms.iter().map(|t| t.0).fold(f32::INFINITY, f32::law_min);
    let margin = terms.iter().map(|t| t.0 * t.1).fold(0.0f32, f32::law_max);
    (scale.is_finite() && scale > 0.0 && margin.is_finite()).then_some((scale, margin))
}

/// Bernstein coefficients on `[0, 1]` of the polynomial `Σ a_j t^j` restricted to `[u0, u1]`
fn bernstein_on(a: &[f64], u0: f64, u1: f64) -> Vec<f64> {
    let n = a.len() - 1;
    let binom = |n: usize, k: usize| -> f64 {
        (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
    };
    // power basis of u for p(u0 + (u1 − u0) u)
    let h = u1 - u0;
    let mut c = vec![0.0f64; n + 1];
    for (k, &ak) in a.iter().enumerate() {
        for j in 0..=k {
            let pw = |x: f64, e: usize| (0..e).fold(1.0f64, |acc, _| acc * x);
            c[j] += ak * binom(k, j) * pw(u0, k - j) * pw(h, j);
        }
    }
    (0..=n)
        .map(|i| (0..=i).map(|j| binom(i, j) / binom(n, j) * c[j]).sum())
        .collect()
}

/// An upper bound of the axis curve's curvature `|C′ × C″| / |C′|³` that holds everywhere, not only
/// at samples: per segment the uniform Catmull-Rom `C′ = (K₁ + 2K₂ t + 3K₃ t²)/2` gives
/// `C′ × C″ = (2 K₁×K₂ + 6 K₁×K₃ t + 6 K₂×K₃ t²)/4` (quadratic) and `|C′|²` (quartic); on each piece
/// of the segment the largest Bernstein coefficient norm bounds `|C′ × C″|` from above and the
/// smallest Bernstein coefficient bounds `|C′|²` from below (a polynomial lies in the hull of its
/// coefficients). A piece whose lower bound is not positive is halved, to a depth limit, after which
/// the bound is infinite (the axis may stop there)
fn axis_curvature_bound(d: &Loft) -> f32 {
    const PIECES: usize = 8;
    const DEPTH: u32 = 12;
    let mut worst = 0.0f64;
    for seg in 0..d.n - 1 {
        let i = seg as isize;
        let (a, b, c, e) = (
            d.axis_pt(i - 1),
            d.axis_pt(i),
            d.axis_pt(i + 1),
            d.axis_pt(i + 2),
        );
        let f = |v: Vec3| [f64::from(v.x), f64::from(v.y), f64::from(v.z)];
        let (k1, k2, k3) = (
            f(c - a),
            f(2.0 * a - 5.0 * b + 4.0 * c - e),
            f(-a + 3.0 * b - 3.0 * c + e),
        );
        let cr = |u: [f64; 3], v: [f64; 3]| {
            [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ]
        };
        let dt = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        let (c12, c13, c23) = (cr(k1, k2), cr(k1, k3), cr(k2, k3));
        let cross: [Vec<f64>; 3] =
            core::array::from_fn(|x| vec![0.5 * c12[x], 1.5 * c13[x], 1.5 * c23[x]]);
        let speed2 = vec![
            0.25 * dt(k1, k1),
            dt(k1, k2),
            dt(k2, k2) + 1.5 * dt(k1, k3),
            3.0 * dt(k2, k3),
            2.25 * dt(k3, k3),
        ];
        let mut stack: Vec<(f64, f64, u32)> = (0..PIECES)
            .map(|p| (p as f64 / PIECES as f64, (p + 1) as f64 / PIECES as f64, 0))
            .collect();
        while let Some((u0, u1, depth)) = stack.pop() {
            let den = bernstein_on(&speed2, u0, u1)
                .into_iter()
                .fold(f64::INFINITY, f64::min);
            if den <= 0.0 {
                if depth >= DEPTH {
                    return f32::INFINITY;
                }
                let m = 0.5 * (u0 + u1);
                stack.push((u0, m, depth + 1));
                stack.push((m, u1, depth + 1));
                continue;
            }
            let bc: [Vec<f64>; 3] = core::array::from_fn(|x| bernstein_on(&cross[x], u0, u1));
            let num = (0..3)
                .map(|i| (bc[0][i] * bc[0][i] + bc[1][i] * bc[1][i] + bc[2][i] * bc[2][i]).sqrt())
                .fold(0.0f64, f64::max);
            worst = worst.max(num / (den * den.sqrt()));
        }
    }
    // rounded up to f32 so the bound never drops below the f64 value
    let w = worst as f32;
    if f64::from(w) < worst {
        f32::from_bits(w.to_bits() + 1)
    } else {
        w
    }
}

/// Samples per section span in the simplicity check
const SIMPLE_SAMPLES_PER_SPAN: usize = 16;
/// Blended sections checked per row span (the rows' B-spline along the axis)
const SIMPLE_BLENDS_PER_ROW_SPAN: usize = 8;

/// The first section that crosses itself, sampled at the axis weights the field uses (rows reflected
/// past the ends as in the field); its description, or `None` when all are simple
fn first_crossing_section(rows: &[Vec<Vec2>], section_knots: &[f32]) -> Option<String> {
    let m = rows[0].len();
    let kr = rows.len();
    let period = section_knots[m] - section_knots[0];
    let knot = |i: isize| {
        let r = i.rem_euclid(m as isize);
        section_knots[r as usize] + ((i - r) / m as isize) as f32 * period
    };
    let sample = |cp: &[Vec2]| -> Vec<Vec2> {
        let point = |j: isize| cp[j.rem_euclid(m as isize) as usize];
        let mut out = Vec::with_capacity(m * SIMPLE_SAMPLES_PER_SPAN);
        for j in 0..m as isize {
            for k in 0..SIMPLE_SAMPLES_PER_SPAN {
                let t =
                    knot(j) + (knot(j + 1) - knot(j)) * (k as f32 / SIMPLE_SAMPLES_PER_SPAN as f32);
                out.push(closed_curve_on_span(&point, &knot, j, t).0);
            }
        }
        out
    };
    let row = |i: isize, j: usize| -> Vec2 {
        if i < 0 {
            2.0 * rows[0][j] - rows[1][j]
        } else if i >= kr as isize {
            2.0 * rows[kr - 1][j] - rows[kr - 2][j]
        } else {
            rows[i as usize][j]
        }
    };
    // the sections are the blends (the rows are control rows, not sections): every span at
    // t = 0, 1/8, …, and the last span's end
    for span in 0..kr - 1 {
        let last_k = if span + 2 == kr {
            SIMPLE_BLENDS_PER_ROW_SPAN
        } else {
            SIMPLE_BLENDS_PER_ROW_SPAN - 1
        };
        for k in 0..=last_k {
            let t = k as f32 / SIMPLE_BLENDS_PER_ROW_SPAN as f32;
            let (t2, t3, u) = (t * t, t * t * t, 1.0 - t);
            let b = [
                u * u * u / 6.0,
                (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
                (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
                t3 / 6.0,
            ];
            let i0 = span as isize - 1;
            let cp: Vec<Vec2> = (0..m)
                .map(|j| {
                    b[0] * row(i0, j)
                        + b[1] * row(i0 + 1, j)
                        + b[2] * row(i0 + 2, j)
                        + b[3] * row(i0 + 3, j)
                })
                .collect();
            if super::simple_loop::closed_polyline_crosses(&sample(&cp)) {
                return Some(format!("between rows {span} and {} (at {t})", span + 1));
            }
        }
    }
    None
}

/// The axis part of the bounds, common to both section kinds: the curvature check, the per-span
/// local bounds, the knots, the clamp
#[allow(clippy::too_many_arguments)]
fn finish(
    axis: &[Vec3],
    up: Vec3,
    section: LoftSection,
    flat_rows: &[f32],
    caps: [f32; 2],
    steps: &[f32],
    rmax: f32,
    (l2, core, lq, reach): (f32, f32, [f32; 2], f32),
    (sknots, ang, ends, cap_curve): (&[f32], f32, [(f32, f32); 2], [[f32; 4]; 2]),
) -> Result<LoftBounds, LoftError> {
    let err = |s: String| Err(LoftError(s));
    let kr = flat_rows.len() / section.width();
    let provisional = LoftBounds {
        law: LOFT_LAW_VERSION,
        l: 1.0,
        l2: 1.0,
        core: NO_CORE,
        lq: [1.0, 1.0],
        m: 0.0,
        fade: None,
        knots: vec![1.0; kr],
        bknots: Vec::new(),
        reach: 0.0,
        pieces: Vec::new(),
        cap_curve: [[0.0; 4]; 2],
    };
    let provisional = LoftBounds {
        cap_curve,
        ..provisional
    };
    let block = flatten(axis, up, section, flat_rows, sknots, caps, &provisional);
    let data = Loft::from_flat(&block);
    let last = (axis.len() - 1) as f32;
    let per = BOUND_SAMPLES as usize * (axis.len() - 1);
    let ds = last / per as f32;
    let (mut kappa, mut slow) = (0.0f32, f32::INFINITY);
    let mut prev: Option<Vec3> = None;
    // per sample: (s, speed, frame turn rate)
    let mut samples: Vec<(f32, f32, f32)> = Vec::with_capacity(per + 1);
    for i in 0..=per {
        let s = i as f32 * ds;
        let (_, d1) = data.axis_at(s);
        let speed = d1.length();
        if speed == 0.0 {
            return err("the axis stops (zero speed)".into());
        }
        slow = slow.law_min(speed);
        let t = d1 / speed;
        if (up - t * up.dot(t)).length() < 1e-3 * up.length() {
            return err("up is parallel to the axis".into());
        }
        let h = 1e-3;
        let (_, da) = data.axis_at((s - h).law_max(0.0));
        let (_, db) = data.axis_at((s + h).law_min(last));
        let span = (s + h).law_min(last) - (s - h).law_max(0.0);
        let dd = (db - da) / span;
        kappa = kappa.law_max(d1.cross(dd).length() / (speed * speed * speed));
        let (_, nn, _) = frame(d1, up);
        let mut turn = 0.0f32;
        if let Some(pn) = prev {
            let len = (data.axis_at(s).0 - data.axis_at(s - ds).0)
                .length()
                .law_max(1e-12);
            turn = (nn - pn).length() / len;
            // the turn between two samples belongs to both
            if let Some(last_sample) = samples.last_mut() {
                last_sample.2 = last_sample.2.law_max(turn);
            }
        }
        prev = Some(nn);
        samples.push((s, speed, turn));
    }
    // the sampled curvature can miss a bend between samples; the bound holds between them
    kappa = kappa.law_max(axis_curvature_bound(&data));
    let bend = 1.1 * kappa * rmax;
    if bend >= 1.0 {
        return err(format!("the sections (radius {rmax}) are wider than the axis's radius of curvature (curvature {kappa})"));
    }
    // per span i of the B-spline (rows i − 1 … i + 2): the local bound Λᵢ = (1 + 1.1 Gᵢ)/(1 − 1.1 κR),
    // Gᵢ = the rows' largest step there × (k − 1)/(n − 1) over the span's slowest speed + R × its turn
    let w_per_s = (kr - 1) as f32 / last;
    let span_s = last / (kr - 1) as f32;
    let lam: Vec<f32> = (0..kr - 1)
        .map(|i| {
            let lo_j = i.saturating_sub(1);
            let hi_j = (i + 1).min(kr - 2);
            let stp = steps[lo_j..=hi_j]
                .iter()
                .copied()
                .fold(0.0f32, f32::law_max);
            let (a, b) = (i as f32 * span_s - ds, (i + 1) as f32 * span_s + ds);
            let (mut sl, mut tr) = (f32::INFINITY, 0.0f32);
            for &(s, sp, t) in samples.iter().filter(|x| x.0 >= a && x.0 <= b) {
                let _ = s;
                sl = sl.law_min(sp);
                tr = tr.law_max(t);
            }
            (1.0 + 1.1 * (stp * w_per_s / sl + rmax * tr)) / (1.0 - bend)
        })
        .collect();
    // the knots: the larger neighbouring span bound, so L(s) ≥ Λᵢ on span i and is continuous
    let base: Vec<f32> = (0..kr)
        .map(|j| lam[j.saturating_sub(1).min(kr - 2)].law_max(lam[j.min(kr - 2)]))
        .collect();
    let top = base.iter().copied().fold(0.0f32, f32::law_max);
    let (l, m, knots, fade, bknots) = if base.iter().all(|&b| b == top) {
        // one bound everywhere: f = F / L, no correction
        (top, 0.0, base, None, Vec::new())
    } else {
        // L(s) = (1 + ε) × the knots, L_g = (1 + δ) × max L(s); f = F/L_g + κ(s) sat(F, B(s)) ψ with
        // κ(s) = 1/L(s) − 1/L_g ≥ 0 and sat(F, B) = F/√(1 + (F/B)²). ∇f = ∇F (1/L_g + κ ∂sat/∂F ψ)
        // + (sat κ' + κ ∂sat/∂B B') ∇s ψ + sat κ ∇ψ, with 0 < ∂sat/∂F ≤ 1, |sat| ≤ B, |∂sat/∂B| ≤ 1:
        // |∇f| ≤ 1/(1 + ε) + 3 · ε/(3(1 + ε)) = 1 given, on every span, B ≤ ε/(3 (1 + ε) |κ'| |∇s|),
        // |B'| |∇s| ≤ ε/(3 (1 + ε) κ_max), and B ≤ ε (ρ** − ρ*)/(3 (1 + ε) 1.5 κ_max) (the fade).
        // B is per knot, so the band is narrow only where L(s) changes fast, not over the whole loft.
        const EPS: f32 = 0.25;
        const DELTA: f32 = 0.25;
        let knots: Vec<f32> = base.iter().map(|b| (1.0 + EPS) * b).collect();
        let kmax = (1.0 + EPS) * top;
        let kmin_l = knots.iter().copied().fold(f32::INFINITY, f32::law_min);
        let lg = (1.0 + DELTA) * kmax;
        let kap_max = 1.0 / kmin_l - 1.0 / lg;
        let (fade, grad_s, b_fade) = if kappa > 0.0 {
            let span = 1.0 / kappa - rmax;
            let (ra, rb) = (rmax + 0.5 * span, rmax + 0.75 * span);
            let gs = 1.0 / (slow * (1.0 - kappa * rb));
            (
                Some([ra, rb]),
                gs,
                EPS * (rb - ra) / (3.0 * (1.0 + EPS) * 1.5 * kap_max),
            )
        } else {
            (None, 1.0 / slow, f32::INFINITY)
        };
        let budget = EPS / (3.0 * (1.0 + EPS));
        // per span: |κ'| = |d(1/L)/ds| ≤ |ΔL| w_per_s / min(L)² on that span
        let span_req: Vec<f32> = knots
            .windows(2)
            .map(|p| {
                let dk = (p[1] - p[0]).abs() * w_per_s / (p[0].law_min(p[1]) * p[0].law_min(p[1]));
                if dk > 0.0 {
                    budget / (dk * grad_s)
                } else {
                    f32::INFINITY
                }
            })
            .collect();
        let cap = rmax.law_max(1e-30);
        let req: Vec<f32> = (0..kr)
            .map(|j| {
                let a = if j > 0 {
                    span_req[j - 1]
                } else {
                    f32::INFINITY
                };
                let b = if j + 1 < kr {
                    span_req[j]
                } else {
                    f32::INFINITY
                };
                a.law_min(b).law_min(b_fade).law_min(cap)
            })
            .collect();
        // the lower envelope with slope |ΔB| w_per_s ≤ budget / (κ_max |∇s|) per knot step
        let step_b = budget / (kap_max * grad_s * w_per_s);
        let bknots: Vec<f32> = (0..kr)
            .map(|j| {
                (0..kr)
                    .map(|i| req[i] + step_b * (i as f32 - j as f32).abs())
                    .fold(f32::INFINITY, f32::law_min)
            })
            .collect();
        if !bknots.iter().all(|b| *b > 0.0 && b.is_finite()) {
            return err(format!("internal: a saturation width came out non-positive or infinite (κ {kappa}, R {rmax})"));
        }
        // the scalar B: the least width (where the band is everywhere valid; it sizes the far join)
        let band = bknots.iter().copied().fold(f32::INFINITY, f32::law_min);
        (lg, band, knots, fade, bknots)
    };
    let step_max = steps.iter().copied().fold(0.0f32, f32::law_max);
    let pieces = piece_planes(
        &data,
        rmax,
        ang,
        step_max * w_per_s,
        caps,
        ends,
        kappa,
        slow,
    );
    Ok(LoftBounds {
        law: LOFT_LAW_VERSION,
        l,
        l2,
        core,
        lq,
        m,
        fade,
        knots,
        bknots,
        reach,
        pieces,
        cap_curve,
    })
}

/// The cubic Bézier coefficients of the part `[ta, tb]` of the cubic with coefficients `c` on `[0, 1]`
fn bezier_sub(c: [f32; 4], ta: f32, tb: f32) -> [f32; 4] {
    let split = |c: [f32; 4], t: f32| -> ([f32; 4], [f32; 4]) {
        let l = |a: f32, b: f32| a + (b - a) * t;
        let (a1, b1, c1) = (l(c[0], c[1]), l(c[1], c[2]), l(c[2], c[3]));
        let (a2, b2) = (l(a1, b1), l(b1, c1));
        let a3 = l(a2, b2);
        ([c[0], a1, a2, a3], [a3, b2, c1, c[3]])
    };
    let (left, _) = split(c, tb);
    if tb <= 0.0 {
        return [left[0]; 4];
    }
    let (_, right) = split(left, ta / tb);
    right
}

/// The far bound's pieces: each axis segment in [`PIECES_PER_SEGMENT`] parts, and each dome. Plane `k`'s
/// offset bounds `n_k·x` over the piece's surface `x = C(s) + u N(s) + v B(s)`:
/// - `n·C(s)` ≤ the largest `n·X` over the segment's four Bézier points (the segment lies in their hull);
/// - with `a = (n·N_mid, n·B_mid)` at the piece's middle frame, `u n·N + v n·B ≤ (u, v)·a + |(u, v)| δ`,
///   `δ` = the frame's largest change from the middle over the piece (its rate `|N'| + |B'|` ≤
///   `3 |C''|max / (|C'|min |up⊥|min)` times the half length);
/// - `(u, v)·a` over the section: curve mode, the largest over the control points of the rows that
///   reach the piece (the section lies in their hull, the rows mix with convex weights); polar mode, the
///   largest over those rows and `64 (K + 1)` angles of `r(θ) (cos θ, sin θ)·a`, plus the angle step's
///   half times `|a| (R + max |∂r/∂θ|)`, which bounds that function's rate.
/// A dome adds `√((H n·out)² + S²)` to `n·C_end`, `S` the end row's support in `a` (exact frame).
#[allow(clippy::too_many_arguments)]
fn piece_planes(
    d: &Loft,
    rmax: f32,
    ang: f32,
    rs_per_s: f32,
    caps: [f32; 2],
    ends: [(f32, f32); 2],
    kappa: f32,
    slow: f32,
) -> Vec<f32> {
    let _ = (rs_per_s, kappa);
    let mut out = Vec::new();
    let last = (d.n - 1) as f32;
    let nseg = d.n - 1;
    let w_per_s = (d.kr - 1) as f32 / last;
    // the support of the rows `lo..=hi` in direction a (2D), and the largest control radius among them
    let support = |lo: usize, hi: usize, a: Vec2| -> (f32, f32) {
        let mut best = f32::NEG_INFINITY;
        let mut rad = 0.0f32;
        if d.curve {
            for i in lo..=hi {
                for j in 0..d.kh {
                    let q = d.row_point(i as isize, j);
                    best = best.law_max(q.dot(a));
                    rad = rad.law_max(q.length());
                }
            }
        } else {
            let m = 64 * (d.kh + 1);
            for i in lo..=hi {
                for j in 0..m {
                    let th = std::f32::consts::TAU * j as f32 / m as f32;
                    let (c1, s1) = (alice_det_math::cos(th), alice_det_math::sin(th));
                    let r = d.row_radius(i as isize, c1, s1);
                    best = best.law_max(r * (c1 * a.x + s1 * a.y));
                    rad = rad.law_max(r);
                }
            }
            best += a.length() * ang * 0.5 * std::f32::consts::TAU / m as f32;
            rad = rad.law_max(rmax);
        }
        (best, rad)
    };
    // the frame's rate bound over an axis segment: 3 |C''|max / (|C'|min |up⊥|min)
    let frame_rate = |seg: usize| -> f32 {
        let i = seg as isize;
        let (a, b, c, e) = (
            d.axis_pt(i - 1),
            d.axis_pt(i),
            d.axis_pt(i + 1),
            d.axis_pt(i + 2),
        );
        let k2 = 2.0 * a - 5.0 * b + 4.0 * c - e;
        let k3 = -a + 3.0 * b - 3.0 * c + e;
        let dd = k2.length().law_max((k2 + 3.0 * k3).length());
        let mut upp = f32::INFINITY;
        for q in 0..=32 {
            let (_, der) = d.axis_at(seg as f32 + q as f32 / 32.0);
            let t = der.normalize();
            upp = upp.law_min((d.up - t * d.up.dot(t)).length() / d.up.length());
        }
        3.0 * dd / (slow * upp)
    };
    for pc in 0..nseg * PIECES_PER_SEGMENT {
        let seg = pc / PIECES_PER_SEGMENT;
        let (sa, sb) = (
            pc as f32 / PIECES_PER_SEGMENT as f32,
            (pc + 1) as f32 / PIECES_PER_SEGMENT as f32,
        );
        let (sa, sb) = (sa.law_min(last), sb.law_min(last));
        let (_, dmid) = d.axis_at(0.5 * (sa + sb));
        let (t, n, b) = frame(dmid, d.up);
        let delta = frame_rate(seg) * 0.5 * (sb - sa);
        let ii = seg as isize;
        let bez = {
            let (a, bb, c, e) = (
                d.axis_pt(ii - 1),
                d.axis_pt(ii),
                d.axis_pt(ii + 1),
                d.axis_pt(ii + 2),
            );
            [bb, bb + (c - a) / 6.0, c - (e - bb) / 6.0, c]
        };
        let (wa, wb) = (sa * w_per_s, sb * w_per_s);
        let s0 = (wa as usize).min(d.kr - 2);
        let s1 = (wb.ceil() as usize).saturating_sub(1).max(s0).min(d.kr - 2);
        out.extend_from_slice(&[t.x, t.y, t.z, n.x, n.y, n.z, b.x, b.y, b.z]);
        for k in 0..PIECE_PLANES {
            let nk = plane_normal(k, t, n, b);
            let ax = bez
                .iter()
                .map(|x| nk.dot(*x))
                .fold(f32::NEG_INFINITY, f32::law_max);
            let a = Vec2::new(nk.dot(n), nk.dot(b));
            // per span: the section is Σ βⱼ(t) rowⱼ with convex weights, and the support function is
            // sublinear, so its support is ≤ Σ βⱼ(t) hⱼ — a cubic in t, bounded by its Bézier
            // coefficients. The end spans (reflected rows, folded weights) take the largest row.
            let mut sup = f32::NEG_INFINITY;
            let mut rad = 0.0f32;
            for sp in s0..=s1 {
                if sp == 0 || sp + 2 >= d.kr {
                    let (lo, hi) = (sp.saturating_sub(1), (sp + 2).min(d.kr - 1));
                    let (h, r) = support(lo, hi, a);
                    sup = sup.law_max(h);
                    rad = rad.law_max(r);
                } else {
                    let mut hh = [0.0f32; 4];
                    for (q, hq) in hh.iter_mut().enumerate() {
                        let (h, r) = support(sp - 1 + q, sp - 1 + q, a);
                        *hq = h;
                        rad = rad.law_max(r);
                    }
                    let bz = [
                        (hh[0] + 4.0 * hh[1] + hh[2]) / 6.0,
                        (4.0 * hh[1] + 2.0 * hh[2]) / 6.0,
                        (2.0 * hh[1] + 4.0 * hh[2]) / 6.0,
                        (hh[1] + 4.0 * hh[2] + hh[3]) / 6.0,
                    ];
                    // the part of the span inside the piece, by de Casteljau (its Bézier points hold it)
                    let ta = (wa - sp as f32).law_max(0.0).law_min(1.0);
                    let tb = (wb - sp as f32).law_max(0.0).law_min(1.0);
                    let sub = bezier_sub(bz, ta, tb);
                    sup = sup.law_max(sub.iter().copied().fold(f32::NEG_INFINITY, f32::law_max));
                }
            }
            out.push(ax + sup + rad * 2.0 * delta);
        }
    }
    for (e, &cap) in caps.iter().enumerate() {
        if cap <= 0.0 {
            continue;
        }
        let s = if e == 0 { 0.0 } else { last };
        let (c, der) = d.axis_at(s);
        let (tt, n, b) = frame(der, d.up);
        let outv = if e == 0 { -tt } else { tt };
        let row = if e == 0 { 0 } else { d.kr - 1 };
        let _ = ends;
        out.extend_from_slice(&[tt.x, tt.y, tt.z, n.x, n.y, n.z, b.x, b.y, b.z]);
        for k in 0..PIECE_PLANES {
            let nk = plane_normal(k, tt, n, b);
            let (sup, _) = support(row, row, Vec2::new(nk.dot(n), nk.dot(b)));
            if d.curve {
                // the eroded, sheared section at height h lies in the end section shifted by o h/H, so
                // n·x ≤ n·C + h (n·out) + (h/H) n·(N o_u + B o_v) + S, linear in h: its larger end
                let [ou, ov, _, _] = d.cap_curve[e];
                let lift = cap * nk.dot(outv) + nk.dot(n * ou + b * ov);
                out.push(nk.dot(c) + lift.law_max(0.0) + sup);
            } else {
                let a_out = cap * nk.dot(outv);
                out.push(nk.dot(c) + (a_out * a_out + sup.law_max(0.0) * sup.law_max(0.0)).sqrt());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The closest axis parameter (scan + bracketed Newton) against a dense brute-force argmin of the
    /// axis curve: the distance to the found point exceeds the true least distance by at most 1e-5
    /// (relative to the scale), on a bent 9-point axis and a straight one, near and far
    #[test]
    fn the_closest_axis_point_matches_a_dense_search() {
        let bent: Vec<Vec3> = (0..9)
            .map(|i| {
                Vec3::new(
                    0.3 * alice_det_math::sin(0.7 * i as f32),
                    0.25 * i as f32,
                    0.1 * alice_det_math::cos(1.3 * i as f32),
                )
            })
            .collect();
        let rows = vec![vec![0.1f32, 0.0, 0.0], vec![0.1, 0.0, 0.0]];
        let mut st: u64 = 9;
        let mut rnd = || {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((st >> 40) as f32) / ((1u64 << 24) as f32)
        };
        for axis in [bent, vec![Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)]] {
            let (fr, b) = validate(&axis, Vec3::Z, 1, &rows, [0.0, 0.0]).expect("valid");
            let block = flatten(
                &axis,
                Vec3::Z,
                LoftSection::Polar(1),
                &fr,
                &[],
                [0.0, 0.0],
                &b,
            );
            let d = Loft::from_flat(&block);
            let last = (axis.len() - 1) as f32;
            let dense: Vec<Vec3> = (0..=200_000)
                .map(|i| d.axis_at(last * i as f32 / 200_000.0).0)
                .collect();
            let mut worst = (0.0f32, Vec3::ZERO);
            for _ in 0..2000 {
                let scale = if rnd() < 0.5 { 0.3 } else { 3.0 };
                let p = Vec3::new(
                    (rnd() - 0.5) * scale,
                    rnd() * 2.4 - 0.2,
                    (rnd() - 0.5) * scale,
                );
                let truth = dense
                    .iter()
                    .map(|q| (*q - p).length())
                    .fold(f32::INFINITY, f32::min);
                let got = (d.axis_at(d.closest_param(p)).0 - p).length();
                let err = (got - truth) / scale;
                if err > worst.0 {
                    worst = (err, p);
                }
            }
            assert!(worst.0 < 1e-5, "closest axis point off by {:?}", worst);
        }
    }

    /// Every far-bound piece's planes hold its part of the surface: on a strongly bent axis with an
    /// elongated section, dense surface samples of each piece satisfy `n·x ≤ h` for all 50 planes
    #[test]
    fn far_bound_pieces_hold_their_surface() {
        let axis: Vec<Vec3> = (0..4)
            .map(|i| {
                let a = 0.5 * i as f32;
                Vec3::new(
                    200.0 * (1.0 - alice_det_math::cos(a)),
                    200.0 * alice_det_math::sin(a),
                    0.0,
                )
            })
            .collect();
        let rows = vec![vec![40.0f32, 0.0, 0.0, 18.0, 6.0]; 4];
        let (fr, b) = validate(&axis, Vec3::Z, 2, &rows, [0.0, 0.0]).expect("valid");
        let block = flatten(
            &axis,
            Vec3::Z,
            LoftSection::Polar(2),
            &fr,
            &[],
            [0.0, 0.0],
            &b,
        );
        let d = Loft::from_flat(&block);
        let mut worst = f32::NEG_INFINITY;
        let mut checked = 0usize;
        for (pc, piece) in b.pieces.chunks_exact(PIECE_FLOATS).enumerate() {
            let (t, n, bb) = (
                Vec3::new(piece[0], piece[1], piece[2]),
                Vec3::new(piece[3], piece[4], piece[5]),
                Vec3::new(piece[6], piece[7], piece[8]),
            );
            let (sa, sb) = (
                pc as f32 / PIECES_PER_SEGMENT as f32,
                (pc + 1) as f32 / PIECES_PER_SEGMENT as f32,
            );
            for i in 0..=64 {
                let s = sa + (sb - sa) * i as f32 / 64.0;
                let (c, der) = d.axis_at(s);
                let (_, fnn, fb) = frame(der, d.up);
                for j in 0..720 {
                    let th = std::f32::consts::TAU * j as f32 / 720.0;
                    let (c1, s1) = (alice_det_math::cos(th), alice_det_math::sin(th));
                    let x = c + (fnn * c1 + fb * s1) * d.radius(s, c1, s1);
                    for k in 0..PIECE_PLANES {
                        worst = worst.max(plane_normal(k, t, n, bb).dot(x) - piece[9 + k]);
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0);
        assert!(
            worst <= 0.0,
            "a surface sample lies {worst} past its piece's plane"
        );
    }

    /// The culled axis scan against the full scan, bit for bit
    #[test]
    fn the_culled_axis_scan_is_the_full_scan_bit_for_bit() {
        // a bent 9-point axis and a straight 2-point one, a million points each spread from the axis
        // out to well beyond it
        let bent: Vec<Vec3> = (0..9)
            .map(|i| {
                Vec3::new(
                    0.3 * alice_det_math::sin(0.7 * i as f32),
                    0.25 * i as f32,
                    0.1 * alice_det_math::cos(1.3 * i as f32),
                )
            })
            .collect();
        let rows = vec![vec![0.1f32, 0.0, 0.0], vec![0.1, 0.0, 0.0]];
        let mut st: u64 = 5;
        let mut rnd = || {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((st >> 40) as f32) / ((1u64 << 24) as f32)
        };
        for axis in [bent, vec![Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)]] {
            let (fr, b) = validate(&axis, Vec3::Z, 1, &rows, [0.0, 0.0]).expect("valid");
            let block = flatten(
                &axis,
                Vec3::Z,
                LoftSection::Polar(1),
                &fr,
                &[],
                [0.0, 0.0],
                &b,
            );
            let d = Loft::from_flat(&block);
            let mut compared = 0u32;
            for _ in 0..1_000_000 {
                let scale = if rnd() < 0.5 { 0.3 } else { 3.0 };
                let p = Vec3::new(
                    (rnd() - 0.5) * scale,
                    rnd() * 2.4 - 0.2,
                    (rnd() - 0.5) * scale,
                );
                let (a, ad) = d.scan_full(p);
                let (c, cd) = d.scan_culled(p);
                assert_eq!(
                    (a.to_bits(), ad.to_bits()),
                    (c.to_bits(), cd.to_bits()),
                    "at {p:?}"
                );
                compared += 1;
            }
            assert_eq!(compared, 1_000_000);
        }
    }

    /// The section curve's first derivative (de Boor on the derivative control points) against
    /// central differences, and the section distance against a brute-force search, on non-uniform knots
    #[test]
    fn curve_derivatives_match_differences() {
        for m in [12usize, 300] {
            check_curve(m);
        }
    }

    fn check_curve(m: usize) {
        let row: Vec<Vec2> = (0..m)
            .map(|k| {
                let a = std::f32::consts::TAU * k as f32 / m as f32;
                Vec2::new(0.3 * alice_det_math::cos(a), 0.2 * alice_det_math::sin(a))
            })
            .collect();
        let sk: Vec<f32> = (0..=m)
            .map(|j| j as f32 + if j % 3 == 1 { 0.3 } else { 0.0 })
            .collect();
        let (pts, b) = validate_curves(
            &[Vec3::ZERO, Vec3::Y],
            Vec3::X,
            &sk,
            &[row.clone(), row],
            [[0.0; 3]; 2],
        )
        .expect("valid");
        let block = flatten(
            &[Vec3::ZERO, Vec3::Y],
            Vec3::X,
            LoftSection::Curve(m as u32),
            &pts,
            &sk,
            [0.0, 0.0],
            &b,
        );
        let d = Loft::from_flat(&block);
        let (i0, w) = d.row_weights(0.5);
        let mut worst = (0.0f32, 0.0f32);
        for j in 0..m as isize {
            let (k0, k1) = (d.sknot(j), d.sknot(j + 1));
            for k in 1..8 {
                let t = k0 + (k1 - k0) * k as f32 / 8.0;
                let h = 1e-3 * (k1 - k0);
                let (_, d1, d2) = d.curve_on_span(i0, &w, j, t);
                let (a, ..) = d.curve_on_span(i0, &w, j, t - h);
                let (c, ..) = d.curve_on_span(i0, &w, j, t);
                let (e, ..) = d.curve_on_span(i0, &w, j, t + h);
                worst.0 = worst.0.max((d1 - (e - a) / (2.0 * h)).length());
                worst.1 = worst.1.max((d2 - (e - 2.0 * c + a) / (h * h)).length());
            }
        }
        assert!(worst.0 < 1e-3, "first derivative error {worst:?}");
        // the distance against a brute-force search of the same curve
        let mut st: u64 = 3;
        let mut rnd = || {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((st >> 40) as f32) / ((1u64 << 24) as f32)
        };
        let mut wd = (0.0f32, Vec2::ZERO, 0.0f32, 0.0f32);
        for _ in 0..400 {
            let q = Vec2::new(rnd() * 0.8 - 0.4, rnd() * 0.6 - 0.3);
            let mut best = f32::INFINITY;
            for j in 0..m as isize {
                let (k0, k1) = (d.sknot(j), d.sknot(j + 1));
                for k in 0..(24000 / m) {
                    let t = k0 + (k1 - k0) * k as f32 / (24000 / m) as f32;
                    best = best.min((d.curve_on_span(i0, &w, j, t).0 - q).length());
                }
            }
            let got = d.curve_distance(0.5, q).abs();
            if (got - best).abs() > wd.0 {
                wd = ((got - best).abs(), q, got, best);
            }
        }
        assert!(wd.0 < 2e-5, "M {m}: distance error {wd:?}");
    }
}
