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
/// Ternary refinement steps around the best scan sample
pub const REFINE_STEPS: u32 = 16;
/// Samples per segment for the axis curvature and frame turn bounds
pub const BOUND_SAMPLES: u32 = 64;
/// The most harmonics a section may have
pub const MAX_HARMONICS: u32 = 64;
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

/// The bounds a loft's field is divided by, computed by [`validate`]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LoftBounds {
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
    /// The local bound `L(s)` at the B-spline knots (one per row), linear between them
    pub knots: Vec<f32>,
    /// A radius holding the whole surface around the axis curve (the largest radius and dome height)
    pub reach: f32,
}

/// The flat block of a loft (the bytecode's aux data and the shaders' module-scope array):
/// `[n, K, k, up.x, up.y, up.z, H₀, H₁, L_g, L₂, c, L_q0, L_q1, m, R_b, axis (3n), rows (k(2K + 1)), knots (k)]`
pub const HEAD: usize = 15;

/// The flat block of a loft
pub fn flatten(axis: &[Vec3], up: Vec3, harmonics: u32, rows: &[f32], caps: [f32; 2], b: &LoftBounds) -> Vec<f32> {
    let k = rows.len() / (2 * harmonics as usize + 1);
    let mut v = vec![
        axis.len() as f32,
        harmonics as f32,
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
    ];
    v.extend(axis.iter().flat_map(|a| [a.x, a.y, a.z]));
    v.extend_from_slice(rows);
    v.extend_from_slice(&b.knots);
    v
}

/// The loft's data, read by the tree (slices) and the bytecode (the flat block) alike, so the two
/// run the same arithmetic
struct Loft<'a> {
    n: usize,
    kh: usize,
    kr: usize,
    up: Vec3,
    caps: [f32; 2],
    lg: f32,
    l2: f32,
    core: f32,
    lq: [f32; 2],
    m: f32,
    reach: f32,
    axis: &'a [f32],
    rows: &'a [f32],
    knots: &'a [f32],
}

impl<'a> Loft<'a> {
    fn from_flat(f: &'a [f32]) -> Self {
        let n = f[0] as usize;
        let kh = f[1] as usize;
        let kr = f[2] as usize;
        Self {
            n,
            kh,
            kr,
            up: Vec3::new(f[3], f[4], f[5]),
            caps: [f[6], f[7]],
            lg: f[8],
            l2: f[9],
            core: f[10],
            lq: [f[11], f[12]],
            m: f[13],
            reach: f[14],
            axis: &f[HEAD..HEAD + 3 * n],
            rows: &f[HEAD + 3 * n..HEAD + 3 * n + kr * (2 * kh + 1)],
            knots: &f[HEAD + 3 * n + kr * (2 * kh + 1)..HEAD + 3 * n + kr * (2 * kh + 1) + kr],
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
        let (a, b, c, d) = (self.axis_pt(i - 1), self.axis_pt(i), self.axis_pt(i + 1), self.axis_pt(i + 2));
        let t2 = t * t;
        let t3 = t2 * t;
        let k1 = c - a;
        let k2 = 2.0 * a - 5.0 * b + 4.0 * c - d;
        let k3 = -a + 3.0 * b - 3.0 * c + d;
        let pos = 0.5 * (2.0 * b + k1 * t + k2 * t2 + k3 * t3);
        let der = 0.5 * (k1 + 2.0 * k2 * t + 3.0 * k3 * t2);
        (pos, der)
    }

    /// The closest axis parameter to `p`: a fixed scan, then fixed ternary steps
    fn closest_param(&self, p: Vec3) -> f32 {
        let last = (self.n - 1) as f32;
        let steps = (self.n - 1) as u32 * SCAN_PER_SEGMENT;
        let h = 1.0 / SCAN_PER_SEGMENT as f32;
        let mut best_s = 0.0f32;
        let mut best_d = f32::INFINITY;
        for k in 0..=steps {
            let s = k as f32 * h;
            let e = (self.axis_at(s).0 - p).length_squared();
            if e < best_d {
                best_d = e;
                best_s = s;
            }
        }
        let mut lo = (best_s - h).law_max(0.0);
        let mut hi = (best_s + h).law_min(last);
        for _ in 0..REFINE_STEPS {
            let m1 = lo + (hi - lo) / 3.0;
            let m2 = hi - (hi - lo) / 3.0;
            let e1 = (self.axis_at(m1).0 - p).length_squared();
            let e2 = (self.axis_at(m2).0 - p).length_squared();
            if e1 <= e2 {
                hi = m2;
            } else {
                lo = m1;
            }
        }
        // a bracket still touching an end: the end itself when it is closer (the minimum on the
        // boundary is where the ternary steps converge only linearly)
        let mut best = 0.5 * (lo + hi);
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
            let w = 2 * self.kh + 1;
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

    /// The dome at end `e` (0 at `s = 0`, 1 at `s = n − 1`): its field cut by the end's half-space
    fn dome(&self, p: Vec3, e: usize) -> f32 {
        let s = if e == 0 { 0.0 } else { (self.n - 1) as f32 };
        let (c, der) = self.axis_at(s);
        let (t, nn, bb) = frame(der, self.up);
        let out = if e == 0 { -t } else { t };
        let r = p - c;
        let h = r.dot(out);
        let (u, v) = (r.dot(nn), r.dot(bb));
        let rho = (u * u + v * v).sqrt();
        let (c1, s1) = if rho > 0.0 { (u / rho, v / rho) } else { (1.0, 0.0) };
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
        let (c1, s1) = if rho > 0.0 { (u / rho, v / rho) } else { (1.0, 0.0) };
        let g = (rho - self.radius(s, c1, s1)).law_max(self.core);
        let d2 = g / self.l2;
        let (c0, d0) = self.axis_at(0.0);
        let (ce, de) = self.axis_at((self.n - 1) as f32);
        let dw = (c0 - p).dot(d0.normalize()).law_max((p - ce).dot(de.normalize()));
        let o = Vec2::new(d2.law_max(0.0), dw.law_max(0.0));
        // F, then f = F/L_g + clamp(F (1/L(s) − 1/L_g), −m, m): F/L(s) near the surface, F/L_g ± m far
        let big_f = d2.law_max(dw).law_min(0.0) + o.length();
        let corr = (big_f * (1.0 / self.local_l(s) - 1.0 / self.lg)).law_max(-self.m).law_min(self.m);
        let mut f = big_f / self.lg + corr;
        for e in 0..2 {
            if self.caps[e] > 0.0 {
                f = f.law_min(self.dome(p, e));
            }
        }
        // the far bound: the distance to the tube of radius R_b around the axis curve holds the whole
        // surface, so it never exceeds the distance to it, and it is 1-Lipschitz; the larger of the two
        f.law_max(r.length() - self.reach)
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
pub fn sdf_loft(p: Vec3, axis: &[Vec3], up: Vec3, harmonics: u32, rows: &[f32], caps: [f32; 2], b: &LoftBounds) -> f32 {
    sdf_loft_flat(p, &flatten(axis, up, harmonics, rows, caps, b))
}

/// A box holding the loft's surface, from the flat block: each axis segment is the cubic Bézier
/// `b, b + (c − a)/6, c − (d − b)/6, c`, inside the hull of those four points; the box of the axis's
/// Bézier points, grown by a radius bound (|a₀| + Σ(|aₖ| + |bₖ|) over the rows, times 3 for the
/// reflected rows) plus the larger dome height
pub fn aabb_flat(flat: &[f32]) -> (Vec3, Vec3) {
    let d = Loft::from_flat(flat);
    let n = d.n as isize;
    let mut mn = Vec3::splat(f32::INFINITY);
    let mut mx = Vec3::splat(f32::NEG_INFINITY);
    for i in 0..n - 1 {
        let (a, b, c, e) = (d.axis_pt(i - 1), d.axis_pt(i), d.axis_pt(i + 1), d.axis_pt(i + 2));
        for q in [b, b + (c - a) / 6.0, c - (e - b) / 6.0, c] {
            mn = mn.law_min(q);
            mx = mx.law_max(q);
        }
    }
    let w = 2 * d.kh + 1;
    let bound = |row: &[f32]| row.iter().map(|x| x.abs()).fold(0.0f32, |a, b| a + b);
    let mut r = 0.0f32;
    for i in 0..d.kr {
        r = r.law_max(bound(&d.rows[i * w..(i + 1) * w]));
    }
    // a reflected row 2c₀ − c₁ is bounded by 3 times the largest row
    let r = 3.0 * r + d.caps[0].law_max(d.caps[1]);
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
        let (c1, s1_) = (alice_det_math::cos(th as f32), alice_det_math::sin(th as f32));
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
pub fn validate(axis: &[Vec3], up: Vec3, harmonics: u32, rows: &[Vec<f32>], caps: [f32; 2]) -> Result<(Vec<f32>, LoftBounds), LoftError> {
    let err = |s: String| Err(LoftError(s));
    if axis.len() < 2 {
        return err(format!("the axis needs 2 points or more, it has {}", axis.len()));
    }
    if harmonics > MAX_HARMONICS {
        return err(format!("{harmonics} harmonics, at most {MAX_HARMONICS}"));
    }
    if !axis.iter().all(|a| a.is_finite()) || !up.is_finite() || !caps.iter().all(|c| c.is_finite()) || rows.iter().flatten().any(|v| !v.is_finite()) {
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
        .map(|pair| pair[0].iter().zip(&pair[1]).map(|(a, b)| (a - b).abs()).fold(0.0f32, |x, y| x + y))
        .collect();
    let flat_rows: Vec<f32> = rows.iter().flatten().copied().collect();
    let kr = rows.len();
    let provisional = LoftBounds { l: 1.0, l2: 1.0, core: NO_CORE, lq: [1.0, 1.0], m: 0.0, knots: vec![1.0; kr], reach: 0.0 };
    let block = flatten(axis, up, harmonics, &flat_rows, caps, &provisional);
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
            let len = (data.axis_at(s).0 - data.axis_at(s - ds).0).length().law_max(1e-12);
            turn = (nn - pn).length() / len;
            // the turn between two samples belongs to both
            if let Some(last_sample) = samples.last_mut() {
                last_sample.2 = last_sample.2.law_max(turn);
            }
        }
        prev = Some(nn);
        samples.push((s, speed, turn));
    }
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
            let stp = steps[lo_j..=hi_j].iter().copied().fold(0.0f32, f32::law_max);
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
    let base: Vec<f32> = (0..kr).map(|j| lam[j.saturating_sub(1).min(kr - 2)].law_max(lam[j.min(kr - 2)])).collect();
    let top = base.iter().copied().fold(0.0f32, f32::law_max);
    let (l, m, knots) = if base.iter().all(|&b| b == top) {
        // one bound everywhere: f = F / L, no correction
        (top, 0.0, base)
    } else {
        // L(s) = (1 + ε) × the knots, L_g = (1 + δ) × max L(s); in the band |F| ≤ B the term
        // |F| |L'| |∇s| / L² ≤ ε/(1 + ε), so |∇(F/L(s))| ≤ 1/(1 + ε) + ε/(1 + ε) = 1; the clamp m
        // keeps the correction active only where |F| < m / min(1/L(s) − 1/L_g) ≤ B
        const EPS: f32 = 0.25;
        const DELTA: f32 = 0.25;
        let knots: Vec<f32> = base.iter().map(|b| (1.0 + EPS) * b).collect();
        let kmax = (1.0 + EPS) * top;
        let kmin_l = knots.iter().copied().fold(f32::INFINITY, f32::law_min);
        let lg = (1.0 + DELTA) * kmax;
        let dlds = knots.windows(2).map(|p| (p[1] - p[0]).abs()).fold(0.0f32, f32::law_max) * w_per_s;
        // |∇s| ≤ 1/(slow (1 − κρ)) ≤ 2/slow while ρ ≤ 1/(2κ); a point with |F| ≤ B has ρ ≤ R + L₂ B
        let grad_s = 2.0 / slow;
        let l2_here = if dmax == 0.0 { 1.0 } else { (1.0 + (dmax / (0.5 * rmin)) * (dmax / (0.5 * rmin))).sqrt() };
        let band_curv = if kappa > 0.0 { (0.5 / kappa - rmax) / l2_here } else { f32::INFINITY };
        let band = (EPS * kmin_l * kmin_l / ((1.0 + EPS) * dlds * grad_s)).law_min(band_curv);
        let kappa_min = (1.0 / kmax) * (1.0 - 1.0 / (1.0 + DELTA));
        (lg, kappa_min * band, knots)
    };
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
            if lo <= 0.0 {
                return err("a capped end's section must keep a positive radius".into());
            }
            let sec = (1.0 / (lo * lo) + d * d / (lo * lo * lo * lo)).sqrt();
            lq[e] = sec.law_max(1.0 / caps[e]);
        }
    }
    let reach = rmax.law_max(caps[0]).law_max(caps[1]);
    Ok((flat_rows, LoftBounds { l, l2, core, lq, m, knots, reach }))
}
