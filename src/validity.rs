//! Printability validation — `ValidityReport`.
//!
//! The question this module answers is *"can this shape be printed"*, and it
//! answers the two failure modes that actually lose geometry:
//!
//! * a wall thinner than the nozzle is not printed at all — it silently
//!   disappears from the part,
//! * an unsupported overhang droops, so the printed surface is not the exported
//!   surface.
//!
//! Each answer states **how** it was obtained, because "sampled and found
//! nothing" and "proved there is nothing" are different claims and only the
//! second one is a guarantee.
//!
//! ## Two independent thickness checks (and why one is not enough)
//!
//! **A — erosion, proved by interval arithmetic** ([`ErosionVerdict`])
//!
//! Erosion is exact for an SDF: `Round { radius: -t/2 }` evaluates to
//! `d(p) + t/2`, so only material deeper than `t/2` from the surface survives.
//! If the eroded shape is *empty*, every part of the shape is thinner than `t`.
//! The octree either finds a witness point inside the eroded shape
//! ([`ErosionVerdict::HasThickEnoughRegion`]), proves emptiness over the whole
//! bound ([`ErosionVerdict::EntirelyTooThin`]), or runs out of depth
//! ([`ErosionVerdict::Undecided`] — **not a pass**).
//!
//! This is a *global* statement. It catches "the whole part is a 0.3 mm sheet"
//! with a proof, and it says nothing about a thin fin on an otherwise solid
//! block.
//!
//! **A rejected design, recorded so it is not retried**: the first attempt was
//! "a box proven inside the shape *and* proven outside the eroded shape is a
//! wall thinner than `t`". That is wrong — it holds for **every** shape, because
//! the material within `t/2` of any surface always erodes away, however thick
//! the body behind it is. Local thickness is a medial-axis quantity and deciding
//! it needs connectivity analysis (flood fill), which is exactly what
//! `alice_lol::law` declined to make sound for `Continuity` because it is
//! grid-resolution dependent.
//!
//! **B — local thickness, measured exactly per triangle** ([`local_thickness`])
//!
//! From each triangle centroid, march inward by sphere tracing: while inside,
//! `-d(p)` is the exact distance to the nearest surface, so stepping by it never
//! overshoots, and the distance at which the ray leaves the solid is the local
//! thickness at that point. The direction is the negative SDF gradient read just
//! below the surface, not the mesh face normal: a remeshed tessellation can carry
//! rim slivers and flipped faces whose normal points out through a neighbouring
//! face, which measured micrometre "walls" on millimetre parts. Every individual measurement is exact
//! (it reads the SDF, it does not estimate); what is sampled is *where* the
//! measurements are taken, and that is the mesh tessellation — a density the
//! caller chose and can state. This is the approach mesh-repair tools take.
//!
//! Together: A proves the global case, B localises the thin regions with exact
//! per-point numbers, and [`ValidityReport::is_printable`] requires A to be a
//! proof (not `Undecided`) *and* B to find nothing below the requirement.
//!
//! ## Overhang is a closed form
//!
//! The overhang angle of a triangle is `asin(-n · b)` for unit face normal `n`
//! and unit build direction `b`: a vertical wall gives `0`, a downward-facing
//! horizontal ceiling gives `π/2`. Upward-facing triangles report `0` rather
//! than a negative angle. No sampling is involved.
//!
//! **What this does not decide**: whether an overhang is *supported*. A box on
//! the build plate has a horizontal downward face (overhang `π/2`) needing no
//! support. That requires the plate position and the slicer's support policy,
//! so [`overhang_stats`] reports the geometric angle and leaves the policy out.

use crate::compiled::real::LawMinMax;
use crate::eval::{eval, eval_normal};
use crate::interval::{eval_interval, Vec3Interval};
use crate::mesh::{validate_mesh, Mesh, MeshValidation};
use crate::types::SdfNode;
use glam::Vec3;

/// Smallest march step, so a ray starting exactly on a surface still advances.
const MARCH_EPS: f32 = 1e-4;

/// What a part must satisfy to be printable by a given process.
#[derive(Debug, Clone, Copy)]
pub struct PrintRequirements {
    /// Minimum wall thickness, in the same unit as the SDF.
    ///
    /// For FDM this is usually two extrusion widths (a 0.4 mm nozzle → 0.8 mm).
    pub min_wall: f32,
    /// Largest overhang angle that prints without support \[rad\].
    ///
    /// The common FDM figure is 45° (`std::f32::consts::FRAC_PI_4`).
    pub max_overhang: f32,
    /// Unit build direction (the direction layers stack towards).
    pub build_direction: Vec3,
    /// Octree depth used by the erosion proof.
    ///
    /// Each level halves the box edge; 6 levels resolve a bound to 1/64 of its
    /// size. Raising it makes [`ErosionVerdict::Undecided`] rarer at a cost of
    /// up to `8^n` boxes.
    pub erosion_depth: u32,
}

impl PrintRequirements {
    /// FDM defaults: 0.8 mm walls, 45° overhang, +Z build direction, depth 6.
    ///
    /// The unit is millimetres, matching the 3 MF / STEP exporters.
    #[must_use]
    pub const fn fdm_0_4_nozzle() -> Self {
        Self {
            min_wall: 0.8,
            max_overhang: std::f32::consts::FRAC_PI_4,
            build_direction: Vec3::Z,
            erosion_depth: 6,
        }
    }
}

/// Global thickness verdict from the erosion proof (three-valued).
///
/// `Undecided` must never be treated as a pass — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErosionVerdict {
    /// A point was **proven** to lie inside the shape eroded by `min_wall / 2`,
    /// so some region is at least `min_wall` thick.
    HasThickEnoughRegion {
        /// Centre of the box that was proven to be inside the eroded shape.
        witness: Vec3,
    },
    /// The octree closed over the whole bound without finding surviving
    /// material: **every** part of the shape is thinner than `min_wall`.
    EntirelyTooThin,
    /// The depth limit was reached with boxes still straddling the eroded
    /// surface.
    Undecided,
}

impl ErosionVerdict {
    /// `true` only when a thick-enough region was proven to exist.
    #[must_use]
    pub const fn has_thick_region(&self) -> bool {
        matches!(self, Self::HasThickEnoughRegion { .. })
    }
}

/// Outcome of validating a shape for printing.
#[derive(Debug, Clone)]
pub struct ValidityReport {
    /// Mesh topology, from [`validate_mesh`].
    pub mesh: MeshValidation,
    /// A — global erosion proof (three-valued).
    pub erosion: ErosionVerdict,
    /// B — smallest local thickness measured over the triangles \[same unit as
    /// the SDF\], or `None` when no triangle could be measured.
    pub min_local_thickness: Option<f32>,
    /// B — number of triangles whose local thickness is below `min_wall`.
    pub thin_triangles: usize,
    /// Largest overhang angle over all triangles \[rad\].
    pub max_overhang: f32,
    /// Number of triangles whose overhang exceeds `max_overhang`.
    pub overhang_triangles: usize,
    /// The requirements this report was produced against.
    pub requirements: PrintRequirements,
}

impl ValidityReport {
    /// `true` only when every check passed **and** the global check is a proof.
    ///
    /// An [`ErosionVerdict::Undecided`] fails this deliberately: "the octree
    /// could not decide" is not "the part is fine".
    #[must_use]
    pub const fn is_printable(&self) -> bool {
        self.mesh.is_clean()
            && self.erosion.has_thick_region()
            && self.thin_triangles == 0
            && self.overhang_triangles == 0
    }
}

/// Largest overhang angle and the number of triangles over the threshold.
///
/// Returns `(max_angle_rad, over_threshold_count)`. Degenerate triangles (zero
/// area) are skipped because they have no meaningful normal. Upward-facing
/// triangles contribute `0`, not a negative angle.
#[must_use]
pub fn overhang_stats(mesh: &Mesh, build_direction: Vec3, threshold: f32) -> (f32, usize) {
    let dir = build_direction.normalize_or_zero();
    let mut max_angle = 0.0_f32;
    let mut over = 0_usize;
    for idx in mesh.indices.as_chunks::<3>().0.iter() {
        let Some(normal) = face_normal(mesh, idx) else {
            continue; // 退化三角形は法線を持たない
        };
        // 下向き成分 −n·b を [0, 1] に落として asin: 垂直壁 0、水平天井 π/2
        let downward = (-normal.dot(dir)).clamp(-1.0, 1.0);
        let angle = if downward <= 0.0 {
            0.0
        } else {
            alice_det_math::asin(downward)
        };
        if angle > max_angle {
            max_angle = angle;
        }
        if angle > threshold {
            over += 1;
        }
    }
    (max_angle, over)
}

/// Unit geometric normal of the triangle, or `None` when it is degenerate.
fn face_normal(mesh: &Mesh, idx: &[u32]) -> Option<Vec3> {
    let p = |i: u32| mesh.vertices.get(i as usize).map(|v| v.position);
    let (a, b, c) = (p(idx[0])?, p(idx[1])?, p(idx[2])?);
    let n = (b - a).cross(c - a);
    let len = n.length();
    if len <= f32::EPSILON {
        return None;
    }
    Some(n / len)
}

/// Exact local thickness at `origin` when marching along `inward`.
///
/// `origin` is expected to sit on (or just outside) the surface and `inward`
/// to be a unit vector pointing into the solid. Returns the distance at which
/// the ray leaves the solid, or `None` when the ray does not enter the solid
/// or does not leave it within `max_march`.
///
/// Each step advances by `-d(p)`, the exact distance to the nearest surface
/// while inside, so the march never overshoots the far wall.
#[must_use]
pub fn local_thickness(node: &SdfNode, origin: Vec3, inward: Vec3, max_march: f32) -> Option<f32> {
    let dir = inward.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }

    // 1. 固体に入るまで進む (origin が表面ちょうど or 少し外にある前提)
    //    外では d(p) が最近表面までの距離なので、それだけ進んでも通り越さない
    let mut t = MARCH_EPS;
    loop {
        let d = eval(node, origin + dir * t);
        if d < 0.0 {
            break; // 固体に入った
        }
        t += d.law_max(MARCH_EPS);
        if t > max_march {
            return None; // 固体に入らなかった
        }
    }

    // 2. 固体を抜けるまで進む 内側では −d(p) が最近表面までの距離
    loop {
        let d = eval(node, origin + dir * t);
        if d >= 0.0 {
            return Some(t);
        }
        t += (-d).law_max(MARCH_EPS);
        if t > max_march {
            return None; // 抜けなかった (bound が足りない)
        }
    }
}

/// Depth below the surface at which the inward direction is read (see [`surface_sample`]).
const SAMPLE_DEPTH: f32 = 1e-3;

/// Where and in which direction to measure the wall under a tessellation point `p`.
///
/// The mesh face normal is not used: a remeshed or repaired tessellation can carry faces whose
/// normal is tilted or opposed to the surface (a rim sliver, a flipped face), and a ray along such
/// a normal leaves the solid through a neighbouring face after a few micrometres. Instead `p` is
/// moved along the SDF gradient to the depth [`SAMPLE_DEPTH`] below the surface, and the inward
/// direction is the negative gradient **there**. Inside a solid the gradient of an exact SDF is the
/// normal of the nearest surface, so at a convex edge it is one face's normal (never the bisector
/// a rounded tessellation suggests), and the march crosses that face's wall.
///
/// Returns the point just outside the surface on that line and the inward unit direction, the
/// arguments [`local_thickness`] takes.
fn surface_sample(node: &SdfNode, p: Vec3) -> (Vec3, Vec3) {
    let d = eval(node, p);
    let inside = p - eval_normal(node, p) * (d + SAMPLE_DEPTH);
    let n = eval_normal(node, inside);
    let to_surface = -eval(node, inside);
    (inside + n * (to_surface + MARCH_EPS), -n)
}

/// Decides whether **any** region of the shape survives erosion by
/// `min_wall / 2` (see [`ErosionVerdict`]).
///
/// `bmin` / `bmax` must enclose the shape; material outside the bound is not
/// examined.
#[must_use]
pub fn prove_erosion(
    node: &SdfNode,
    min_wall: f32,
    bmin: Vec3,
    bmax: Vec3,
    max_depth: u32,
) -> ErosionVerdict {
    // erode: `Round { radius: -t/2 }` は d(p) + t/2 を返すので、表面から t/2 より
    // 深い材料だけが負に残る (SDF の offset なので厳密)
    let eroded = node.clone().round(-min_wall * 0.5);

    let mut undecided = false;
    let mut stack: Vec<(Vec3, Vec3, u32)> = vec![(bmin, bmax, 0)];
    while let Some((lo, hi, depth)) = stack.pop() {
        let iv = eval_interval(&eroded, Vec3Interval::from_bounds(lo, hi));
        if iv.is_positive() {
            continue; // 箱は erode 後の外と証明された = 残った材料はない
        }
        if iv.is_negative() {
            // 箱が丸ごと erode 後の内部 = min_wall 以上の肉厚を持つ領域の証明
            return ErosionVerdict::HasThickEnoughRegion {
                witness: (lo + hi) * 0.5,
            };
        }
        if depth >= max_depth {
            undecided = true; // 表面を跨いだまま深さが尽きた
            continue;
        }
        let mid = (lo + hi) * 0.5;
        for i in 0..8_usize {
            let (x0, x1) = if i & 1 == 0 {
                (lo.x, mid.x)
            } else {
                (mid.x, hi.x)
            };
            let (y0, y1) = if i & 2 == 0 {
                (lo.y, mid.y)
            } else {
                (mid.y, hi.y)
            };
            let (z0, z1) = if i & 4 == 0 {
                (lo.z, mid.z)
            } else {
                (mid.z, hi.z)
            };
            stack.push((Vec3::new(x0, y0, z0), Vec3::new(x1, y1, z1), depth + 1));
        }
    }

    if undecided {
        ErosionVerdict::Undecided
    } else {
        // 全 box が「erode 後の外」と証明された = どこも min_wall より薄い
        ErosionVerdict::EntirelyTooThin
    }
}

/// Validates a shape and its tessellation against `req`.
#[must_use]
pub fn validate_for_printing(
    node: &SdfNode,
    mesh: &Mesh,
    bmin: Vec3,
    bmax: Vec3,
    req: PrintRequirements,
) -> ValidityReport {
    let erosion = prove_erosion(node, req.min_wall, bmin, bmax, req.erosion_depth);
    let (max_overhang, overhang_triangles) =
        overhang_stats(mesh, req.build_direction, req.max_overhang);

    // B — 三角形ごとに、重心の下の壁を SDF の勾配の向きで測る (面の法線は向きに使わない)
    // march 上限は bound の対角長 (それを超えるなら bound の外)
    let span = (bmax - bmin).length();
    let mut min_local: Option<f32> = None;
    let mut thin_triangles = 0_usize;
    for idx in mesh.indices.as_chunks::<3>().0.iter() {
        // 面積 0 の三角形は標本にしない (重心が表面の点として意味を持たない)
        if face_normal(mesh, idx).is_none() {
            continue;
        }
        let p = |i: u32| mesh.vertices[i as usize].position;
        let centroid = (p(idx[0]) + p(idx[1]) + p(idx[2])) / 3.0;
        let (origin, inward) = surface_sample(node, centroid);
        if let Some(thickness) = local_thickness(node, origin, inward, span) {
            min_local = Some(min_local.map_or(thickness, |m| m.law_min(thickness)));
            if thickness < req.min_wall {
                thin_triangles += 1;
            }
        }
    }

    ValidityReport {
        mesh: validate_mesh(mesh),
        erosion,
        min_local_thickness: min_local,
        thin_triangles,
        max_overhang,
        overhang_triangles,
        requirements: req,
    }
}

/// Gaps (holes / slots / pockets) found by [`measure_gaps`].
#[derive(Debug, Clone, PartialEq)]
pub struct GapReport {
    /// Narrowest gap width measured \[same unit as the SDF\], `None` when no sample re-entered
    /// material (only open air around the shape).
    pub min_gap: Option<f32>,
    /// Surface point where `min_gap` was measured.
    pub location: Option<Vec3>,
    /// Surface samples marched: grid points in empty space within `resolution` of a surface.
    pub samples: usize,
    /// Samples whose march re-entered material (gap samples); the rest reached open air.
    pub gaps: usize,
    /// `√3 · cell`: every empty slab wider than this holds a grid point within this distance of a
    /// surface in any orientation, so it is found; narrower gaps may be missed (unverified, not
    /// passed). A caller comparing against a limit at or below this value has not checked it.
    pub resolution: f32,
    /// Acute joints where two walls meet (a tilted plate on a floor, a cylinder lying on a plate):
    /// they close to a line and have no width, so their samples are listed here and never counted in
    /// `min_gap` / `gaps`. One entry per joint location (samples within `2 · cell` of a listed one are
    /// merged), at most [`MAX_CREVICES`].
    pub crevices: Vec<Crevice>,
    /// Gap samples classified as crevices (before merging into `crevices`).
    pub crevice_samples: usize,
    /// Samples whose wall point sits on an edge of its own wall (the wall's direction turns by more
    /// than [`EDGE_COS`] within a quarter cell): the wall has no single direction there, so neither a
    /// width nor a crevice is read from them. The faces either side are sampled by their own grid
    /// points.
    pub edge_samples: usize,
}

/// An acute joint where two walls meet, found by [`measure_gaps`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crevice {
    /// Surface point of the sample that found it.
    pub location: Vec3,
    /// Angle of the air between the two walls at the sample, in degrees (below 90).
    pub angle_deg: f32,
}

/// Most crevices listed in a [`GapReport`] (further ones are still counted in `crevice_samples`).
pub const MAX_CREVICES: usize = 1024;

/// `sin` of the largest angle between facing walls still read as parallel (about 0.06°): below it the
/// walls have no direction in which they meet. Near the line where curved walls touch (a cylinder lying
/// on a plate) the normals also turn almost antiparallel, so parallelism is not a stopping rule during
/// the walk; a width is decided by the width that stops closing.
const PARALLEL_SIN: f32 = 1e-3;

/// `cos` of the turn of the wall's direction (within a quarter cell) above which a sample's wall point
/// is on an edge (about 20°).
const EDGE_COS: f32 = 0.94;

/// Whether the wall point `a` (direction `n`) is on an edge of its wall: a quarter cell away along
/// either way of two tangents, the field's direction turns by more than [`EDGE_COS`]. The direction is
/// read at the stepped point itself, not at its foot on the surface: past an edge the stepped point lies
/// over the other face, while its foot falls back onto the edge line, where the direction is a tie
/// between the two faces.
fn on_wall_edge(node: &SdfNode, a: Vec3, n: Vec3, cell: f32) -> bool {
    let t1 = n.any_orthonormal_vector();
    let t2 = n.cross(t1);
    let step = 0.25 * cell;
    [t1, -t1, t2, -t2]
        .iter()
        .any(|&t| eval_normal(node, a + t * step).normalize_or_zero().dot(n) < EDGE_COS)
}

/// Steps of the walk toward where two converging walls would meet.
const WEDGE_MAX_STEPS: usize = 48;

/// Whether the gap measured from surface point `a` (normal `n`, width `width`) is a width between
/// walls that stay apart, or a crevice where the walls meet (`Some(angle in degrees)`).
///
/// Facing walls that are parallel (within [`PARALLEL_SIN`]) are a width. Otherwise the walk moves
/// along the start wall toward the line where the two walls' tangent planes meet, half the remaining
/// distance per step, re-measuring the width from the wall each time: the walls meet when the width
/// closes to within a few [`GAP_TOLERANCE`]; they stay apart when the width stops closing (a sphere
/// over a plate bottoms out) or the march finds open air (the walk ran off the wall, e.g. onto the
/// floor of a tapered slot). The angle is the air angle between the walls at the first sample.
fn crevice_angle(node: &SdfNode, a: Vec3, n: Vec3, width: f32, reach: f32) -> Option<f32> {
    let wall_normal = |at: Vec3| eval_normal(node, at).normalize_or_zero();
    let mut a = a;
    let mut n = n;
    let mut w = width;
    let mut n_h = wall_normal(a + n * w);
    let c0 = n.dot(n_h);
    if n_h == Vec3::ZERO {
        return None;
    }
    let angle =
        alice_det_math::acos((-c0).law_max(-1.0).law_min(1.0)) * (180.0 / core::f32::consts::PI);
    for _ in 0..WEDGE_MAX_STEPS {
        let h = a + n * w;
        // along the start wall, toward the other wall's tangent plane; facing walls that are parallel
        // (within PARALLEL_SIN) have no such direction: a width
        let along = n * n.dot(n_h) - n_h;
        if along.length() <= PARALLEL_SIN {
            return None;
        }
        let along = along.normalize();
        let approach = along.dot(n_h);
        if approach >= 0.0 {
            return None;
        }
        let s = (h - a).dot(n_h) / approach;
        if !(s.is_finite() && s > 0.0) {
            return None;
        }
        // half way to the meeting line, back onto the wall
        let q = a + along * (0.5 * s);
        let nq = wall_normal(q);
        let a2 = q - nq * eval(node, q);
        let n2 = wall_normal(a2);
        if n2 == Vec3::ZERO {
            return None;
        }
        let w2 = local_gap(node, a2, n2, reach)?;
        if w2 <= 4.0 * GAP_TOLERANCE {
            return Some(angle);
        }
        if w2 >= 0.95 * w {
            return None;
        }
        a = a2;
        n = n2;
        w = w2;
        n_h = wall_normal(a + n * w);
        if n_h == Vec3::ZERO {
            return None;
        }
    }
    None
}

/// Why [`measure_gaps`] measured nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapError {
    /// `cell` is not a finite positive number, or the bound is not finite and ordered.
    BadCell,
    /// The grid would exceed [`MAX_GAP_GRID_POINTS`] points.
    TooManyPoints,
    /// No grid point lay in empty space near a surface: nothing was compared.
    NoSamples,
}

/// Largest grid [`measure_gaps`] evaluates (2²⁶ points).
pub const MAX_GAP_GRID_POINTS: usize = 1 << 26;

/// Error bound of a reported gap width against the true width between facing walls \[same unit as the
/// SDF\].
///
/// For distance-exact fields: the re-entry is taken where the field drops below the march
/// tolerance and the distance field there is added back, so the hit is within one tolerance of the wall,
/// and the start is the sample's exact distance to its own wall. A comparison against a limit should
/// fail only when `width + GAP_TOLERANCE < limit` (a gap designed at exactly the limit is not failed by
/// the measurement's own error).
pub const GAP_TOLERANCE: f32 = 2.0 * MARCH_EPS;

/// Steps after which a march through empty space is treated as open air (a ray grazing a
/// surface advances by the tiny distance to it).
const GAP_MAX_STEPS: usize = 10_000;

/// Gap width at `origin` when marching through empty space along `outward`.
///
/// The dual of [`local_thickness`]: `origin` is expected on (or just inside) the surface with
/// `outward` pointing into empty space. The march first leaves material (if `origin` is inside),
/// then advances by `d(p)`, the exact distance to the nearest surface, until it re-enters
/// material. Returns the distance from where it left material to where it re-entered, or `None`
/// when it does not re-enter within `max_march` (open air) or grazes a surface for longer than
/// the step budget.
#[must_use]
pub fn local_gap(node: &SdfNode, origin: Vec3, outward: Vec3, max_march: f32) -> Option<f32> {
    let dir = outward.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }
    let (crossing, clear) = leave_surface(node, origin, dir, 0.0, max_march)?;
    // the gap starts where the ray leaves material (crossing), not where it is clear of the tolerance
    march_empty(node, origin, dir, clear, max_march).map(|enter| enter - crossing)
}

/// `origin + dir · t0` から `dir` に進み、`(材料を抜けた t, 面から MARCH_EPS 以上離れた t)` を返す
/// (`limit` を越えたら `None`) 抜けた所は `d > 0` になった最初の `t` (出発点が空間側ならその `t0`)
///
/// 再進入の判定 (`d < MARCH_EPS`) は出発点で発火しうる: 面から許容以内の点 (面の上、浮動小数の丸めで
/// 面のすぐ外) から march すると 1 歩目で「材料に戻った」と読み、幅 ~0 の隙間になる ⇒ 判定の前に面から
/// 離れる 内側では −d が最近表面までの距離なので、その分ずつ進む
fn leave_surface(
    node: &SdfNode,
    origin: Vec3,
    dir: Vec3,
    t0: f32,
    limit: f32,
) -> Option<(f32, f32)> {
    let mut t = t0;
    let mut crossing: Option<f32> = None;
    loop {
        let d = eval(node, origin + dir * t);
        if d > 0.0 && crossing.is_none() {
            crossing = Some(t);
        }
        if d >= MARCH_EPS {
            return Some((crossing.unwrap_or(t), t));
        }
        t += (-d).law_max(MARCH_EPS);
        if t > limit {
            return None;
        }
    }
}

/// From `origin + dir * t0` (in empty space) march along `dir` until material: the parameter `t`
/// where `d` first falls below [`MARCH_EPS`], or `None` (open air / step budget)
fn march_empty(node: &SdfNode, origin: Vec3, dir: Vec3, t0: f32, max_march: f32) -> Option<f32> {
    let mut t = t0;
    for _ in 0..GAP_MAX_STEPS {
        let d = eval(node, origin + dir * t);
        if d < MARCH_EPS {
            return Some(t + d.law_max(0.0));
        }
        t += d;
        if t > max_march {
            return None;
        }
    }
    None
}

/// Narrowest gap (hole / slot / pocket width) of `node` inside `bmin`..`bmax`.
///
/// Samples every point of a grid of pitch `cell` that lies in empty space within `√3 · cell` of a
/// surface (`0 < d ≤ √3 · cell`). Such a point lies on the ray from its nearest surface point
/// along `+∇d`, so the gap there is `d` plus the march from the point to where the ray re-enters
/// material ([`local_gap`]'s march). A ray that leaves the bound diagonal without re-entering is
/// open air. Any empty slab wider than [`GapReport::resolution`] (`√3 · cell`) holds such a grid
/// point in any orientation and is found; narrower gaps may be missed, and the report states
/// that resolution so a caller does not read "not found" as "not there".
///
/// # Errors
///
/// [`GapError::BadCell`] (`cell` not finite positive, or a bad bound),
/// [`GapError::TooManyPoints`] (the grid exceeds [`MAX_GAP_GRID_POINTS`]),
/// [`GapError::NoSamples`] (no surface inside the bound: nothing compared).
pub fn measure_gaps(
    node: &SdfNode,
    bmin: Vec3,
    bmax: Vec3,
    cell: f32,
) -> Result<GapReport, GapError> {
    let valid = cell.is_finite()
        && cell > 0.0
        && bmin.is_finite()
        && bmax.is_finite()
        && !bmax.cmplt(bmin).any();
    if !valid {
        return Err(GapError::BadCell);
    }
    let span = bmax - bmin;
    let count = |s: f32| -> Option<usize> {
        let n = (s / cell).ceil();
        // ceil of a finite non-negative ratio; the cast is exact below 2^24 and guarded above
        if n.is_finite() && n < 16_777_216.0 {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Some(n as usize + 1)
        } else {
            None
        }
    };
    let (nx, ny, nz) = match (count(span.x), count(span.y), count(span.z)) {
        (Some(x), Some(y), Some(z)) => (x, y, z),
        _ => return Err(GapError::TooManyPoints),
    };
    if nx
        .checked_mul(ny)
        .and_then(|v| v.checked_mul(nz))
        .is_none_or(|v| v > MAX_GAP_GRID_POINTS)
    {
        return Err(GapError::TooManyPoints);
    }
    let resolution = 3.0_f32.sqrt() * cell;
    let reach = span.length();
    let mut samples = 0_usize;
    let mut gaps = 0_usize;
    let mut best: Option<(f32, Vec3)> = None;
    let mut crevices: Vec<Crevice> = Vec::new();
    let mut crevice_samples = 0_usize;
    let mut edge_samples = 0_usize;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                #[allow(clippy::cast_precision_loss)] // grid indices are below 2^24
                let p = bmin + Vec3::new(i as f32, j as f32, k as f32) * cell;
                let d = eval(node, p);
                // 空間側で表面から resolution 以内 (NaN は比較が偽なので標本にしない)
                let near = d > 0.0 && d <= resolution;
                if !near {
                    continue;
                }
                let n = eval_normal(node, p).normalize_or_zero();
                if n == Vec3::ZERO {
                    continue;
                }
                samples += 1;
                // 標本が自分の面から march の許容以内にある時は、まず面から離れる (leave_surface): 離れる前に
                // 「材料に戻った」と読むと、面のすぐ外の格子点が幅 2d の隙間になる (格子点が面と
                // 浮動小数の丸め分だけずれた平らな面で、何も無い所に 1e-5 mm の隙間を報告した)
                let Some((_, t0)) = leave_surface(node, p, n, 0.0, resolution) else {
                    continue;
                };
                let Some(enter) = march_empty(node, p, n, t0, reach) else {
                    continue;
                };
                let width = d + enter;
                let at = p - n * d;
                if on_wall_edge(node, at, n, cell) {
                    edge_samples += 1;
                    continue;
                }
                if let Some(angle_deg) = crevice_angle(node, at, n, width, reach) {
                    crevice_samples += 1;
                    let merge = 2.0 * cell;
                    if crevices.len() < MAX_CREVICES
                        && !crevices
                            .iter()
                            .any(|c: &Crevice| c.location.distance(at) <= merge)
                    {
                        crevices.push(Crevice {
                            location: at,
                            angle_deg,
                        });
                    }
                    continue;
                }
                gaps += 1;
                if best.is_none_or(|(b, _)| width < b) {
                    best = Some((width, at));
                }
            }
        }
    }
    if samples == 0 {
        return Err(GapError::NoSamples);
    }
    Ok(GapReport {
        min_gap: best.map(|(w, _)| w),
        location: best.map(|(_, at)| at),
        samples,
        gaps,
        resolution,
        crevices,
        crevice_samples,
        edge_samples,
    })
}

/// Failure from [`export_step_validated`].
#[derive(Debug)]
pub enum ValidatedExportError {
    /// Writing the file failed.
    Io(std::io::Error),
    /// The shape did not satisfy the requirements, so **nothing was written**.
    NotPrintable(Box<ValidityReport>),
}

impl std::fmt::Display for ValidatedExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "STEP export failed: {e}"),
            Self::NotPrintable(r) => write!(
                f,
                "not printable, nothing written (erosion {:?}, thin triangles {}, \
                 overhang triangles {}, min local thickness {:?})",
                r.erosion, r.thin_triangles, r.overhang_triangles, r.min_local_thickness
            ),
        }
    }
}

impl std::error::Error for ValidatedExportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::NotPrintable(_) => None,
        }
    }
}

impl From<std::io::Error> for ValidatedExportError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Validates the shape and writes STEP **only when it passes**.
///
/// [`crate::io::step::export_step`] does not validate anything, so a part headed
/// for a printer should go through here: when `req` is not met the file is *not*
/// written and the report comes back inside
/// [`ValidatedExportError::NotPrintable`], which is the difference between
/// "exported a shape that cannot be printed" and "refused with a reason".
///
/// The mesh is tessellated once for validation and `export_step` tessellates
/// again internally. Both use the same `bounds` / `resolution`, so the triangles
/// are identical (the evaluator is deterministic); the cost is one extra
/// marching-cubes pass. Call [`validate_for_printing`] and `export_step`
/// separately if that matters.
///
/// # Errors
///
/// - [`ValidatedExportError::NotPrintable`] when the report fails
///   [`ValidityReport::is_printable`] (no file is written)
/// - [`ValidatedExportError::Io`] when writing fails
pub fn export_step_validated(
    path: impl AsRef<std::path::Path>,
    node: &SdfNode,
    cfg: &crate::io::step::StepConfig,
    req: PrintRequirements,
) -> Result<ValidityReport, ValidatedExportError> {
    export_step_validated_with_provenance(
        path,
        node,
        cfg,
        req,
        &crate::provenance::Provenance::sdf(),
    )
}

/// [`export_step_validated`] with the caller's semantics identifier in the STEP header (see
/// [`crate::io::step::export_step_with_provenance`])
///
/// # Errors
///
/// As [`export_step_validated`].
pub fn export_step_validated_with_provenance(
    path: impl AsRef<std::path::Path>,
    node: &SdfNode,
    cfg: &crate::io::step::StepConfig,
    req: PrintRequirements,
    provenance: &crate::provenance::Provenance,
) -> Result<ValidityReport, ValidatedExportError> {
    let (lo, hi) = cfg.bounds;
    let (bmin, bmax) = (Vec3::splat(lo), Vec3::splat(hi));
    let mc = crate::mesh::MarchingCubesConfig {
        resolution: usize::try_from(cfg.resolution).unwrap_or(usize::MAX),
        ..Default::default()
    };
    let mesh = crate::mesh::sdf_to_mesh(node, bmin, bmax, &mc);
    let report = validate_for_printing(node, &mesh, bmin, bmax, req);
    if !report.is_printable() {
        return Err(ValidatedExportError::NotPrintable(Box::new(report)));
    }
    crate::io::step::export_step_with_provenance(path, node, cfg, provenance)?;
    Ok(report)
}
