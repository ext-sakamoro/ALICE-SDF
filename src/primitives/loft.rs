//! Loft: closed planar sections swept along a spline axis, the section interpolated between stations.
//!
//! # The field
//!
//! - **Axis**: a uniform Catmull-Rom curve `C(s)` through the points `A₀ … Aₙ₋₁` (n ≥ 2),
//!   `s ∈ [0, n − 1]`, the end tangents continued by reflected phantom points
//!   (`A₋₁ = 2A₀ − A₁`, `Aₙ = 2Aₙ₋₁ − Aₙ₋₂`), so two points give a straight segment at uniform speed.
//! - **Frame**: `T = C'(s)/|C'(s)|`, `N = normalize(up − (up·T)T)`, `B = T × N`. `up` must not be
//!   parallel to the axis anywhere.
//! - **Sections**: one closed polygon per station `sₖ` (strictly increasing, the first 0 and the last
//!   n − 1), all with the same vertex count, vertex `j` corresponding across sections. Between two
//!   stations each vertex moves along a uniform Catmull-Rom curve in the local parameter, with
//!   reflected phantom sections at the ends.
//! - **Distance**: for a point `p`, `s*` is the axis parameter of the closest axis point (found by a
//!   fixed scan of [`SCAN_PER_SEGMENT`] samples per segment and [`REFINE_STEPS`] ternary steps, the
//!   same steps on every backend); `(u, v) = ((p − C(s*))·N, (p − C(s*))·B)`; `d₂` is the 2D signed
//!   distance to the section at `s*`; `d_w` is the signed distance past the end planes (the planes
//!   through `C(0)` and `C(n − 1)` normal to the tangent there); the two combine as an extrusion,
//!   `min(max(d₂, d_w), 0) + |max((d₂, d_w), 0)|`, and the result is divided by the bound `L`.
//!
//! # The bound
//!
//! `L = (1 + 1.1·G) / (1 − 1.1·κR)`, from samples of the axis and the sections ([`BOUND_SAMPLES`] per
//! segment): `R` the largest section radius, `κ` the largest axis curvature, and `G` the largest
//! rate at which the section changes per unit axis length (vertex speed plus `R` times the frame's
//! turn rate). It is 1 for a straight axis with a constant section, where the field is the exact
//! distance; elsewhere dividing by it keeps the field from growing faster than 1 per unit length,
//! so sphere tracing never steps past the surface. A loft whose sections are wider than the axis's
//! radius of curvature (`1.1·κR ≥ 1`) is refused.

use crate::primitives::sdf_polygon_2d_xy;
use glam::{Vec2, Vec3};

/// Axis samples per segment in the closest-point scan
pub const SCAN_PER_SEGMENT: u32 = 16;
/// Ternary refinement steps around the best scan sample
pub const REFINE_STEPS: u32 = 16;
/// Samples per segment for the Lipschitz bound
pub const BOUND_SAMPLES: u32 = 64;

/// Why a loft was refused
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoftError(pub String);

impl std::fmt::Display for LoftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "loft: {}", self.0)
    }
}

impl std::error::Error for LoftError {}

/// The loft's data, read by the tree (slices) and the bytecode (one flat `f32` block) alike, so the
/// two run the same arithmetic
pub trait LoftData {
    /// Axis point count
    fn n(&self) -> usize;
    /// Axis point `i`
    fn axis(&self, i: usize) -> Vec3;
    /// Station count
    fn k(&self) -> usize;
    /// Station `i`'s axis parameter
    fn station(&self, i: usize) -> f32;
    /// Vertices per section
    fn m(&self) -> usize;
    /// Section `i`'s vertex `j`
    fn vert(&self, i: usize, j: usize) -> Vec2;
    /// The frame's up direction
    fn up(&self) -> Vec3;
    /// The bound the distance is divided by
    fn lipschitz(&self) -> f32;
}

/// The tree's form: slices
pub struct LoftSlices<'a> {
    /// Axis points
    pub axis: &'a [Vec3],
    /// Up direction
    pub up: Vec3,
    /// Stations
    pub stations: &'a [f32],
    /// Sections, `m` per station
    pub sections: &'a [Vec2],
    /// Vertices per section
    pub m: usize,
    /// The bound
    pub lipschitz: f32,
}

impl LoftData for LoftSlices<'_> {
    fn n(&self) -> usize { self.axis.len() }
    fn axis(&self, i: usize) -> Vec3 { self.axis[i] }
    fn k(&self) -> usize { self.stations.len() }
    fn station(&self, i: usize) -> f32 { self.stations[i] }
    fn m(&self) -> usize { self.m }
    fn vert(&self, i: usize, j: usize) -> Vec2 { self.sections[i * self.m + j] }
    fn up(&self) -> Vec3 { self.up }
    fn lipschitz(&self) -> f32 { self.lipschitz }
}

/// The bytecode's form: `[n, k, m, up.x, up.y, up.z, L, axis (3n), stations (k), sections (2km)]`
pub struct LoftFlat<'a>(pub &'a [f32]);

impl LoftFlat<'_> {
    const HEAD: usize = 7;
    fn nn(&self) -> usize { self.0[0] as usize }
    fn kk(&self) -> usize { self.0[1] as usize }
}

impl LoftData for LoftFlat<'_> {
    fn n(&self) -> usize { self.nn() }
    fn axis(&self, i: usize) -> Vec3 {
        let o = Self::HEAD + 3 * i;
        Vec3::new(self.0[o], self.0[o + 1], self.0[o + 2])
    }
    fn k(&self) -> usize { self.kk() }
    fn station(&self, i: usize) -> f32 { self.0[Self::HEAD + 3 * self.nn() + i] }
    fn m(&self) -> usize { self.0[2] as usize }
    fn vert(&self, i: usize, j: usize) -> Vec2 {
        let o = Self::HEAD + 3 * self.nn() + self.kk() + 2 * (i * self.m() + j);
        Vec2::new(self.0[o], self.0[o + 1])
    }
    fn up(&self) -> Vec3 { Vec3::new(self.0[3], self.0[4], self.0[5]) }
    fn lipschitz(&self) -> f32 { self.0[6] }
}

/// The flat block of a loft (the bytecode's aux data and the shaders' module-scope array)
pub fn flatten(axis: &[Vec3], up: Vec3, stations: &[f32], sections: &[Vec2], m: usize, lipschitz: f32) -> Vec<f32> {
    let mut v = vec![axis.len() as f32, stations.len() as f32, m as f32, up.x, up.y, up.z, lipschitz];
    v.extend(axis.iter().flat_map(|a| [a.x, a.y, a.z]));
    v.extend_from_slice(stations);
    v.extend(sections.iter().flat_map(|s| [s.x, s.y]));
    v
}

/// Catmull-Rom point and derivative at `t ∈ [0, 1]` between `b` and `c`
#[inline]
fn cr(a: Vec3, b: Vec3, c: Vec3, d: Vec3, t: f32) -> (Vec3, Vec3) {
    let t2 = t * t;
    let t3 = t2 * t;
    let k1 = c - a;
    let k2 = 2.0 * a - 5.0 * b + 4.0 * c - d;
    let k3 = -a + 3.0 * b - 3.0 * c + d;
    let pos = 0.5 * (2.0 * b + k1 * t + k2 * t2 + k3 * t3);
    let der = 0.5 * (k1 + 2.0 * k2 * t + 3.0 * k3 * t2);
    (pos, der)
}

/// The axis control point `i`, with reflected phantom points one step past either end
#[inline]
fn axis_pt<D: LoftData + ?Sized>(d: &D, i: isize) -> Vec3 {
    let n = d.n() as isize;
    if i < 0 {
        2.0 * d.axis(0) - d.axis(1)
    } else if i >= n {
        2.0 * d.axis((n - 1) as usize) - d.axis((n - 2) as usize)
    } else {
        d.axis(i as usize)
    }
}

/// Axis point and derivative at `s ∈ [0, n − 1]`
#[inline]
pub fn axis_at<D: LoftData + ?Sized>(d: &D, s: f32) -> (Vec3, Vec3) {
    let n = d.n();
    let last = (n - 1) as f32;
    let s = s.clamp(0.0, last);
    let i = (s as usize).min(n - 2);
    let t = s - i as f32;
    let i = i as isize;
    cr(axis_pt(d, i - 1), axis_pt(d, i), axis_pt(d, i + 1), axis_pt(d, i + 2), t)
}

/// The closest axis parameter to `p`: a fixed scan, then fixed ternary steps around the best sample
pub fn closest_param<D: LoftData + ?Sized>(d: &D, p: Vec3) -> f32 {
    let n = d.n();
    let last = (n - 1) as f32;
    let steps = (n - 1) as u32 * SCAN_PER_SEGMENT;
    let h = 1.0 / SCAN_PER_SEGMENT as f32;
    let mut best_s = 0.0f32;
    let mut best_d = f32::INFINITY;
    for k in 0..=steps {
        let s = k as f32 * h;
        let e = (axis_at(d, s).0 - p).length_squared();
        if e < best_d {
            best_d = e;
            best_s = s;
        }
    }
    let mut lo = (best_s - h).max(0.0);
    let mut hi = (best_s + h).min(last);
    for _ in 0..REFINE_STEPS {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        let e1 = (axis_at(d, m1).0 - p).length_squared();
        let e2 = (axis_at(d, m2).0 - p).length_squared();
        if e1 <= e2 {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    0.5 * (lo + hi)
}

/// The frame at a derivative: (T, N, B)
#[inline]
pub fn frame(der: Vec3, up: Vec3) -> (Vec3, Vec3, Vec3) {
    let t = der.normalize();
    let n = (up - t * up.dot(t)).normalize();
    let b = t.cross(n);
    (t, n, b)
}

/// The section's vertex `j` at axis parameter `s`
#[inline]
pub fn section_vertex<D: LoftData + ?Sized>(d: &D, j: usize, s: f32) -> Vec2 {
    let kc = d.k();
    let mut k = 0usize;
    while k + 2 < kc && s > d.station(k + 1) {
        k += 1;
    }
    let s0 = d.station(k);
    let s1 = d.station(k + 1);
    let t = ((s - s0) / (s1 - s0)).clamp(0.0, 1.0);
    let get = |i: isize| -> Vec2 {
        if i < 0 {
            2.0 * d.vert(0, j) - d.vert(1, j)
        } else if i >= kc as isize {
            2.0 * d.vert(kc - 1, j) - d.vert(kc - 2, j)
        } else {
            d.vert(i as usize, j)
        }
    };
    let ki = k as isize;
    let (a, b, c, e) = (get(ki - 1), get(ki), get(ki + 1), get(ki + 2));
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * (2.0 * b + (c - a) * t + (2.0 * a - 5.0 * b + 4.0 * c - e) * t2 + (-a + 3.0 * b - 3.0 * c + e) * t3)
}

/// The loft's distance at `p` (already divided by the bound)
pub fn sdf_loft<D: LoftData + ?Sized>(d: &D, p: Vec3) -> f32 {
    let n = d.n();
    let m = d.m();
    let s = closest_param(d, p);
    let (c, der) = axis_at(d, s);
    let (_, nn, bb) = frame(der, d.up());
    let r = p - c;
    let q = Vec2::new(r.dot(nn), r.dot(bb));
    let mut sec = [Vec2::ZERO; MAX_SECTION_VERTICES];
    for (j, v) in sec.iter_mut().enumerate().take(m) {
        *v = section_vertex(d, j, s);
    }
    let d2 = sdf_polygon_2d_xy(q, &sec[..m]);
    let (c0, d0) = axis_at(d, 0.0);
    let (c1, d1) = axis_at(d, (n - 1) as f32);
    let dw = (c0 - p).dot(d0.normalize()).max((p - c1).dot(d1.normalize()));
    let o = Vec2::new(d2.max(0.0), dw.max(0.0));
    (d2.max(dw).min(0.0) + o.length()) / d.lipschitz()
}

/// The tree's loft distance
pub fn sdf_loft_slices(p: Vec3, axis: &[Vec3], up: Vec3, stations: &[f32], sections: &[Vec2], m: usize, lipschitz: f32) -> f32 {
    sdf_loft(&LoftSlices { axis, up, stations, sections, m, lipschitz }, p)
}

/// The bytecode's loft distance (the flat block of [`flatten`])
pub fn sdf_loft_flat(p: Vec3, flat: &[f32]) -> f32 {
    sdf_loft(&LoftFlat(flat), p)
}

/// The most vertices a section may have
pub const MAX_SECTION_VERTICES: usize = 256;

/// Checks a loft's input and computes its bound; returns the flat sections and `L`
pub fn validate(axis: &[Vec3], up: Vec3, stations: &[f32], sections: &[Vec<Vec2>]) -> Result<(Vec<Vec2>, f32), LoftError> {
    let err = |s: String| Err(LoftError(s));
    if axis.len() < 2 {
        return err(format!("the axis needs 2 points or more, it has {}", axis.len()));
    }
    if !axis.iter().all(|a| a.is_finite()) || !up.is_finite() || !stations.iter().all(|s| s.is_finite()) {
        return err("the axis, up and stations must be finite".into());
    }
    if sections.iter().flatten().any(|v| !v.is_finite()) {
        return err("every section vertex must be finite".into());
    }
    let last = (axis.len() - 1) as f32;
    if stations.len() < 2 || stations.len() != sections.len() {
        return err(format!("{} stations for {} sections: 2 or more, one per section", stations.len(), sections.len()));
    }
    if stations[0] != 0.0 || stations[stations.len() - 1] != last || stations.windows(2).any(|w| w[1] <= w[0]) {
        return err(format!("the stations must rise strictly from 0 to {last}"));
    }
    let m = sections[0].len();
    if m < 3 || m > MAX_SECTION_VERTICES || sections.iter().any(|s| s.len() != m) {
        return err(format!("every section needs the same vertex count, 3 to {MAX_SECTION_VERTICES}"));
    }
    for w in axis.windows(2) {
        if (w[1] - w[0]).length_squared() == 0.0 {
            return err("two consecutive axis points coincide".into());
        }
    }
    let flat: Vec<Vec2> = sections.iter().flatten().copied().collect();
    let data = LoftSlices { axis, up, stations, sections: &flat, m, lipschitz: 1.0 };
    // samples for the bound, and the up check
    let per = BOUND_SAMPLES as usize * (axis.len() - 1);
    let mut kappa = 0.0f32;
    let mut grow = 0.0f32;
    let mut radius = 0.0f32;
    let ds = last / per as f32;
    let mut prev: Option<(Vec3, Vec<Vec2>)> = None;
    for i in 0..=per {
        let s = i as f32 * ds;
        let (_, d1) = axis_at(&data, s);
        let speed = d1.length();
        if speed == 0.0 {
            return err("the axis stops (zero speed)".into());
        }
        let t = d1 / speed;
        if (up - t * up.dot(t)).length() < 1e-3 * up.length() {
            return err("up is parallel to the axis".into());
        }
        let h = 1e-3;
        let (_, da) = axis_at(&data, (s - h).max(0.0));
        let (_, db) = axis_at(&data, (s + h).min(last));
        let span = (s + h).min(last) - (s - h).max(0.0);
        let dd = (db - da) / span;
        kappa = kappa.max(d1.cross(dd).length() / (speed * speed * speed));
        let (_, nn, _) = frame(d1, up);
        let verts: Vec<Vec2> = (0..m).map(|j| section_vertex(&data, j, s)).collect();
        radius = radius.max(verts.iter().map(|v| v.length()).fold(0.0, f32::max));
        if let Some((pn, pv)) = &prev {
            let len = (axis_at(&data, s).0 - axis_at(&data, s - ds).0).length().max(1e-12);
            let vmove = verts.iter().zip(pv).map(|(a, b)| (*a - *b).length()).fold(0.0, f32::max);
            let turn = (nn - *pn).length();
            grow = grow.max((vmove + radius * turn) / len);
        }
        prev = Some((nn, verts));
    }
    let bend = 1.1 * kappa * radius;
    if bend >= 1.0 {
        return err(format!("the sections (radius {radius}) are wider than the axis's radius of curvature (curvature {kappa})"));
    }
    let l = (1.0 + 1.1 * grow) / (1.0 - bend);
    Ok((flat, l))
}
