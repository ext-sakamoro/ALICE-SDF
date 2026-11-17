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
//! ## Gaps: grid widths, air-side proof of facing walls
//!
//! [`measure_gaps`] reads widths and crevices on a grid (one ray per sample, so it can miss a gap whose
//! middle falls between samples). A pass rests on [`prove_gaps`] instead: the gradient of the distance
//! points away from the nearest wall, so on either side of the ridge between two walls that face each
//! other across the air the directions are opposite. Over each box the gradient is enclosed by
//! intervals ([`crate::interval_gradient`]); a box whose enclosure cannot hold two directions with dot
//! below [`FACING_DOT`] has no such ridge, and a box with two air points seeing facing walls whose
//! `d₁ + d₂ + |p₁ − p₂|` is below the limit proves a gap narrower than it. A right-angle corner (dot 0)
//! is never a gap; a groove of 60° or less is decided by its mouth. Limit: node kinds without a
//! gradient enclosure leave their boxes undecided, so such a shape is Undetermined, never Pass.
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
use crate::interval_gradient::{eval_interval_gradient, GradientInterval};
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
        for (a, b) in octants(lo, hi) {
            stack.push((a, b, depth + 1));
        }
    }

    if undecided {
        ErosionVerdict::Undecided
    } else {
        // 全 box が「erode 後の外」と証明された = どこも min_wall より薄い
        ErosionVerdict::EntirelyTooThin
    }
}

/// The eight octants of the box `lo..hi`, split at its centre (the octree step shared by
/// [`prove_erosion`] and [`prove_gaps`]).
fn octants(lo: Vec3, hi: Vec3) -> [(Vec3, Vec3); 8] {
    let mid = (lo + hi) * 0.5;
    core::array::from_fn(|i| {
        let pick = |bit: usize, l: f32, m: f32, h: f32| if i & bit == 0 { (l, m) } else { (m, h) };
        let (x0, x1) = pick(1, lo.x, mid.x, hi.x);
        let (y0, y1) = pick(2, lo.y, mid.y, hi.y);
        let (z0, z1) = pick(4, lo.z, mid.z, hi.z);
        (Vec3::new(x0, y0, z0), Vec3::new(x1, y1, z1))
    })
}

/// The dot product of two wall directions below which the walls face each other across the air.
///
/// Their normals are then more than 120° apart (a groove sharper than 60°): such walls nearer than the
/// limit leave a gap narrower than it. A right-angle corner (dot 0) is not a gap. The 60° groove sits on the line itself
/// and is decided by its mouth (see [`GapReport::verdict`]).
pub const FACING_DOT: f32 = -0.5;

/// Most separate regions [`prove_gaps`] lists per kind (one per region: points within `L` of a listed one
/// are merged); further ones set [`GapProof::truncated`].
pub const MAX_GAP_WITNESSES: usize = 256;

/// A work budget for [`prove_gaps`]: every evaluation of the tree the proof makes, weighted by the
/// tree's node count (a point value 1, a point direction 2, an interval value 2, an interval
/// direction 3).
///
/// A box's cost is not fixed: next to walls its air points are marched to the surface (up to
/// 64 steps each), so counting boxes alone, or boxes times nodes, leaves the time unbounded on some
/// trees. Counting the evaluations bounds it and stays the same on every machine.
///
/// The value: on 42 LOL presets whose proof took a second or more (limit 0.5, depth 10, arm64), one
/// unit of work took 6.7 to 25.4 ns (median 13.3); `2³⁰` units is about 27 s at the slowest rate, under
/// a 30 s cap per proof.
pub const GAP_PROOF_WORK: u64 = 1 << 30;

/// One region the proof found narrow (a witness) or could not decide (an undecided box).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GapRegion {
    /// The centre of the box.
    pub at: Vec3,
    /// The most opposed pair of wall directions seen from the box's air points (centre and corners),
    /// `None` when fewer than two of them are air.
    pub walls: Option<(Vec3, Vec3)>,
    /// For a witness, the distance between the two nearest wall points (below the limit).
    pub width: Option<f32>,
}

/// The result of [`prove_gaps`] for one limit.
#[derive(Debug, Clone, PartialEq)]
pub struct GapProof {
    /// The limit `L` the proof was made for.
    pub limit: f32,
    /// One point per proven narrow region: two air points of the box around it see walls facing each
    /// other (direction dot below [`FACING_DOT`]) that are less than `L` apart.
    pub witnesses: Vec<GapRegion>,
    /// One per region of boxes neither cleared nor proven at `max_depth` (or whose field has no gradient
    /// enclosure); regions within `L` of a witness are dropped (already proven narrow).
    pub undecided_at: Vec<GapRegion>,
    /// More regions were found than [`MAX_GAP_WITNESSES`].
    pub truncated: bool,
    /// The work budget ran out before every box was cleared, proven or left at `max_depth`: the boxes
    /// still open are not counted in `undecided`, and the proof cannot pass.
    pub exhausted: bool,
    /// Work spent: every evaluation of the tree, weighted by the node count (see `GAP_PROOF_WORK`).
    pub work: u64,
    /// Boxes counted undecided (each listed region may hold many).
    pub undecided: usize,
    /// Boxes visited.
    pub boxes: usize,
}

/// Lower bound of `a · b` over two vectors whose components lie in `g` (each axis independently).
fn min_dot(g: &GradientInterval) -> f32 {
    let lo_sq = |i: crate::interval::Interval| {
        if i.lo < 0.0 && i.hi > 0.0 {
            i.lo * i.hi
        } else {
            let m = i.lo.abs().law_min(i.hi.abs());
            m * m
        }
    };
    lo_sq(g.x) + lo_sq(g.y) + lo_sq(g.z)
}

fn push_region(list: &mut Vec<GapRegion>, r: GapRegion, merge: f32, truncated: &mut bool) {
    if list.iter().any(|w| w.at.distance(r.at) <= merge) {
        return;
    }
    if list.len() < MAX_GAP_WITNESSES {
        list.push(r);
    } else {
        *truncated = true;
    }
}

/// Proves, box by box over `bmin..bmax`, where the air holds walls that face each other less than
/// `limit` apart (see the module's "Gaps" section).
///
/// The field's gradient at an air point points away from its nearest wall. Where two walls face each
/// other across the air, the points on either side of the ridge between them see opposite directions;
/// a corner's two walls meet at an angle and do not. A box is
/// - skipped when it is all material (`d ≤ 0`), or all air at least `L/2` from every wall (a ridge there
///   is at least `L` wide);
/// - cleared when the enclosure of `∇d` over the box ([`eval_interval_gradient`]) proves every pair of
///   directions has dot above [`FACING_DOT`]: no ridge between facing walls crosses it (a dot exactly at
///   the bound, the 60° groove, is not cleared);
/// - a witness when two of its points (centre and corners) are air and see walls facing each other less
///   than `L` apart (see `facing_pair`);
/// - split otherwise, or undecided at `max_depth` (also where the enclosure is unknown for a node kind).
///
/// Soundness of clearing: a gap is a ridge in the air between facing walls, so a box with such a ridge
/// holds two air points whose directions have dot below [`FACING_DOT`]. The enclosure over the whole box
/// contains every point's direction, so it contains those two and is not cleared. Enclosing a larger set
/// (material included) only makes clearing rarer, so the proof needs no value band (the evaluator's band,
/// [`crate::interval_gradient::eval_interval_gradient_in_band`], narrows it but changes no decision in the
/// measured scenes once convex edges are enclosed exactly).
///
/// Width equal to the limit: where walls are exactly `L` apart along a surface (a slot as wide as the
/// limit), every box on the ridge holds pairs on both sides of `L` within its own rounding, so it is
/// neither cleared nor a witness and stays undecided: the proof cannot decide equality, and the verdict
/// is `Undetermined` there.
///
/// Cost: boxes are refined only next to ridges; between walls `w` apart the ridge clears once boxes are
/// about `(w − L)/2` across, so widths just above the limit cost the most. `max_work` bounds the boxes
/// work, every evaluation weighted by the tree's node count ([`GAP_PROOF_WORK`] is a default); when it
/// runs out the proof is marked
/// [`GapProof::exhausted`] and its verdict is `Undetermined`, never a pass. Witnesses found before that
/// stand: each is proven on its own.
///
/// Exact for distance-exact fields. Where the field underestimates the distance (smooth blends), a
/// witness may be a field defect rather than real narrow air; clearing stays sound because it rests on
/// the gradient enclosure, which holds both children of every blend.
///
/// # Errors
///
/// [`GapError::BadCell`]: `limit` is not finite positive, or the bound is not finite and ordered.
pub fn prove_gaps(
    node: &SdfNode,
    limit: f32,
    bmin: Vec3,
    bmax: Vec3,
    max_depth: u32,
    max_work: u64,
) -> Result<GapProof, GapError> {
    let valid = limit.is_finite()
        && limit > 0.0
        && bmin.is_finite()
        && bmax.is_finite()
        && !bmax.cmplt(bmin).any();
    if !valid {
        return Err(GapError::BadCell);
    }
    let half = 0.5 * limit;
    let mut proof = GapProof {
        limit,
        witnesses: Vec::new(),
        undecided_at: Vec::new(),
        truncated: false,
        exhausted: false,
        work: 0,
        undecided: 0,
        boxes: 0,
    };
    let mut m = Meter::new(node);
    let mut stack: Vec<(Vec3, Vec3, u32)> = vec![(bmin, bmax, 0)];
    while let Some((lo, hi, depth)) = stack.pop() {
        if m.work >= max_work {
            // the box just taken and the rest of the stack stay open: no pass can be read from them
            proof.exhausted = true;
            break;
        }
        proof.boxes += 1;
        let b = Vec3Interval::from_bounds(lo, hi);
        let iv = m.interval(b);
        if iv.hi <= 0.0 || iv.lo >= half {
            continue;
        }
        // the enclosure is over every point of the box, a superset of its air points: clearing on it is
        // sound (a larger set only clears less); see the "Soundness" paragraph of `prove_gaps`
        let g = m.gradient(b);
        if g.known && min_dot(&g) > FACING_DOT {
            continue;
        }
        let c = (lo + hi) * 0.5;
        let seen = air_walls(&mut m, lo, hi);
        if let Some((walls, width)) = facing_pair(&mut m, &seen, limit) {
            let r = GapRegion {
                at: c,
                walls: Some(walls),
                width: Some(width),
            };
            push_region(&mut proof.witnesses, r, limit, &mut proof.truncated);
            continue;
        }
        if depth >= max_depth {
            proof.undecided += 1;
            // the walls of an undecided box are read around it, grown by twice its field value, so a box
            // next to a groove's bisector sees both walls (its own corners may all see the same one)
            let reach = Vec3::splat(2.0 * m.value(c).law_max(0.0));
            let around = air_walls(&mut m, lo - reach, hi + reach);
            let r = GapRegion {
                at: c,
                walls: most_opposed(&around),
                width: None,
            };
            push_region(&mut proof.undecided_at, r, limit, &mut proof.truncated);
            continue;
        }
        for (a, b) in octants(lo, hi) {
            stack.push((a, b, depth + 1));
        }
    }
    // a region already proven narrow does not wait on its undecided boxes (they sit where the width
    // crosses the limit, next to the witness)
    let witnesses = proof.witnesses.clone();
    proof
        .undecided_at
        .retain(|u| !witnesses.iter().any(|w| w.at.distance(u.at) <= limit));
    proof.work = m.work;
    Ok(proof)
}

/// The proof's work count: every evaluation of the tree it makes, weighted by the tree's node count
/// (a point value 1, a point direction 2, an interval value 2, an interval direction 3), so the budget
/// bounds the time of a proof on any tree and stays the same on every machine.
struct Meter<'a> {
    node: &'a SdfNode,
    nodes: u64,
    work: u64,
}

impl<'a> Meter<'a> {
    fn new(node: &'a SdfNode) -> Self {
        Self {
            node,
            nodes: u64::from(node.node_count().max(1)),
            work: 0,
        }
    }

    const fn charge(&mut self, units: u64) {
        self.work = self.work.saturating_add(units.saturating_mul(self.nodes));
    }

    fn value(&mut self, p: Vec3) -> f32 {
        self.charge(1);
        eval(self.node, p)
    }

    fn direction(&mut self, p: Vec3) -> Vec3 {
        self.charge(2);
        eval_normal(self.node, p).normalize_or_zero()
    }

    fn interval(&mut self, b: Vec3Interval) -> crate::interval::Interval {
        self.charge(2);
        eval_interval(self.node, b)
    }

    fn gradient(&mut self, b: Vec3Interval) -> GradientInterval {
        self.charge(3);
        eval_interval_gradient(self.node, b)
    }
}

/// The box's air points (centre and corners) with their field value and wall direction.
fn air_walls(m: &mut Meter<'_>, lo: Vec3, hi: Vec3) -> Vec<(Vec3, f32, Vec3)> {
    let mut pts: Vec<(Vec3, f32, Vec3)> = Vec::with_capacity(9);
    let mut add = |p: Vec3| {
        let d = m.value(p);
        if d > 0.0 {
            let n = m.direction(p);
            if n != Vec3::ZERO {
                pts.push((p, d, n));
            }
        }
    };
    add((lo + hi) * 0.5);
    for i in 0..8_usize {
        add(Vec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        ));
    }
    pts
}

/// Steps of the march from an air point to its wall in [`wall_point`].
const WALL_MARCH_STEPS: usize = 64;

/// The wall point reached from the air point `p` by marching against its wall direction `n`, stepping by
/// the field value (which never overshoots: the field is at most the distance), until the field is within
/// `GAP_TOLERANCE / 4` of zero. `None` when it does not get there. The point is on the surface whether or
/// not the field is distance-exact (a subtraction underestimates the distance, so `p − d n` stops short).
fn wall_point(node: &SdfNode, p: Vec3, n: Vec3) -> Option<Vec3> {
    wall_point_with(&mut |q| eval(node, q), p, n)
}

/// [`wall_point`] over any field reader (the proof's reader counts its work).
fn wall_point_with(field: &mut impl FnMut(Vec3) -> f32, p: Vec3, n: Vec3) -> Option<Vec3> {
    let mut q = p;
    for _ in 0..WALL_MARCH_STEPS {
        let d = field(q);
        if d <= 0.25 * GAP_TOLERANCE {
            return Some(q);
        }
        q -= n * d;
    }
    None
}

/// A pair of air points that see walls facing each other less than `limit` apart: wall directions with
/// dot below [`FACING_DOT`], the wall points `wᵢ` (reached by [`wall_point`], on the surface) each in
/// front of the other's wall (`(w₂ − w₁)·n₁ > 0`, `(w₁ − w₂)·n₂ > 0`, so the two sides of a thin fin,
/// whose walls face away, are not a gap), and `|w₁ − w₂| < limit`: two surface points across the air
/// nearer than the limit. Returns the two directions and `|w₁ − w₂|`.
fn facing_pair(
    m: &mut Meter<'_>,
    pts: &[(Vec3, f32, Vec3)],
    limit: f32,
) -> Option<((Vec3, Vec3), f32)> {
    // each air point's wall point is marched at most once (lazily: only points in a facing pair)
    let mut walls: Vec<Option<Option<Vec3>>> = vec![None; pts.len()];
    let mut wall = |m: &mut Meter<'_>, k: usize| {
        *walls[k].get_or_insert_with(|| wall_point_with(&mut |q| m.value(q), pts[k].0, pts[k].2))
    };
    for (i, a) in pts.iter().enumerate() {
        for (j, b) in pts.iter().enumerate().skip(i + 1) {
            if a.2.dot(b.2) >= FACING_DOT {
                continue;
            }
            let (Some(w1), Some(w2)) = (wall(m, i), wall(m, j)) else {
                continue;
            };
            let width = w1.distance(w2);
            if a.2.dot(b.2) < FACING_DOT
                && (w2 - w1).dot(a.2) > 0.0
                && (w1 - w2).dot(b.2) > 0.0
                && width < limit
            {
                return Some(((a.2, b.2), width));
            }
        }
    }
    None
}

/// The pair of wall directions with the lowest dot among the air points.
fn most_opposed(pts: &[(Vec3, f32, Vec3)]) -> Option<(Vec3, Vec3)> {
    let mut best: Option<((Vec3, Vec3), f32)> = None;
    for (i, a) in pts.iter().enumerate() {
        for b in &pts[i + 1..] {
            let d = a.2.dot(b.2);
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some(((a.2, b.2), d));
            }
        }
    }
    best.map(|(w, _)| w)
}

/// Largest difference, in degrees, between a region's air angle and a crevice's `angle_deg`.
///
/// Within it the region is read as part of that groove (the grid reads the angle at one sample,
/// the proof at a box near the joint).
pub const CREVICE_ANGLE_MATCH_DEG: f32 = 5.0;

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
    /// `2·√3 · cell`: a gap of any shape (between pins, fin tips, an edge and a face, not only two
    /// walls) wider than this is found by the medial pass (see the module's MODEL section); narrower
    /// gaps between such parts may be missed. [`GapReport::verdict`] certifies nothing below it.
    pub any_shape_resolution: f32,
    /// The grid pitch the report was measured at; features (groove depths, fin and pin sizes) below one
    /// cell are not certified.
    pub cell: f32,
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
    /// Samples whose wall point, or the point where the march came back, has no material behind it:
    /// the field touches zero there without going negative (a zero-thickness sheet, e.g. two boxes
    /// unioned along a shared face that is then subtracted from a solid). Such a surface does not
    /// exist as material, so no width is read from it; the field itself is defective there (a mesher
    /// or slicer may turn that zero level set into a membrane).
    pub sheet_samples: usize,
    /// One of the sheet points (the first found), `None` when `sheet_samples` is 0. Each is confirmed
    /// on the exact field (see [`GapReport::unconfirmed_sheet_samples`]).
    pub sheet_location: Option<Vec3>,
    /// Samples that looked like a sheet on the grid (no material behind the wall point read from the
    /// sample's own value) but whose point is not confirmed on the exact field: marched along its
    /// direction it does not reach the surface (the field is not a distance there, e.g. a plateau with
    /// no gradient), or at the surface the field is not positive on both sides (a region where the
    /// field is zero rather than negative). No width is read from them, and they keep the verdict from
    /// passing; they are not a failure, since nothing on the exact field shows a sheet.
    pub unconfirmed_sheet_samples: usize,
    /// One unconfirmed sheet point (the first found), `None` when `unconfirmed_sheet_samples` is 0.
    pub unconfirmed_sheet_location: Option<Vec3>,
}

/// An acute joint where two walls meet, found by [`measure_gaps`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crevice {
    /// Point on the joint line (within `4 · GAP_TOLERANCE` of where the two walls meet) that the
    /// sample's walk along its wall reached.
    pub location: Vec3,
    /// Angle of the air between the two walls at the sample, in degrees (below 90).
    pub angle_deg: f32,
    /// Widest perpendicular width of the groove: walking from the sample away from the joint along its
    /// wall, the width measured perpendicular to that wall grows until the perpendicular passes the
    /// other wall's rim; the width there (refined by bisection) is the mouth. For a symmetric V of air
    /// angle `α` and depth `D` it is `2·D·sin(α/2)`. A groove whose walls run out of the bound before
    /// they end reports the last width measured inside it.
    pub mouth: f32,
}

/// Most crevices listed in a [`GapReport`] (further ones are still counted in `crevice_samples`).
pub const MAX_CREVICES: usize = 1024;

/// What a [`GapVerdict::Pass`] holds at.
///
/// The air-side proof ([`prove_gaps`]) ruled out walls facing each other nearer than the limit over the
/// whole bound, and every groove it met was read by the grid with a
/// mouth at or above the limit. The pass comes from the proof; the grid's cell and any-shape resolution
/// are reported for the widths and locations it measured, not as a condition of the pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GapCertificate {
    /// The grid pitch measured at.
    pub cell: f32,
    /// Narrowest gap between arbitrary parts that the grid is sure to find (`2·√3 · cell`).
    pub any_shape_resolution: f32,
    /// Smallest feature (groove depth, fin or pin size) the pass speaks for: one cell.
    pub min_certified_feature: f32,
    /// Narrowest gap between facing walls the air-side proof rules out: 0 (every width below the limit;
    /// a gap narrower than the proof's boxes leaves them undecided, which does not pass).
    pub proven_from: f32,
}

/// Why [`GapReport::verdict`] failed the shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GapFailure {
    /// A measured width below the limit (`width + GAP_TOLERANCE < limit`).
    Width {
        /// The narrowest width measured.
        width: f32,
        /// Surface point it was measured from.
        at: Vec3,
    },
    /// An acute groove whose mouth is below the limit: it cannot be printed open.
    CreviceMouth {
        /// The narrowest mouth among the crevices.
        mouth: f32,
        /// Where its walls meet.
        at: Vec3,
    },
    /// The air-side proof found walls facing each other nearer than the limit, outside any groove the
    /// grid read (see [`prove_gaps`]).
    ProvenGap {
        /// The centre of the box where the facing walls were seen.
        at: Vec3,
        /// The distance between the two nearest wall points there (an upper bound of the gap).
        width: f32,
    },
    /// The field touches zero without material (a zero-thickness sheet): a defect of the field.
    Sheet {
        /// One sheet point.
        at: Vec3,
    },
}

/// Why [`GapReport::verdict`] could not decide.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GapUndetermined {
    /// The limit is not a finite positive number.
    BadLimit,
    /// The proof was made for another limit than the one asked.
    ProofForAnotherLimit {
        /// The proof's limit.
        proof_limit: f32,
    },
    /// The grid read sheets that the exact field does not confirm (see
    /// [`GapReport::unconfirmed_sheet_samples`]) and no failure was found.
    UnconfirmedSheet {
        /// One unconfirmed sheet point.
        at: Vec3,
        /// Samples read as unconfirmed sheets.
        samples: usize,
    },
    /// The proof left boxes undecided at its depth, listed more narrow regions than it keeps, or ran
    /// out of its work budget.
    ProofIncomplete {
        /// Boxes left undecided.
        undecided: usize,
        /// Narrow regions beyond the listed ones.
        truncated: bool,
        /// The proof's work budget ran out (see [`GapProof::exhausted`]).
        exhausted: bool,
    },
}

/// One answer for the gaps of a shape against a printer's limit, from [`GapReport::verdict`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GapVerdict {
    /// Nothing below the limit was found, at the sizes the certificate states.
    Pass(GapCertificate),
    /// A gap, crevice mouth or sheet below the limit was found.
    Fail(GapFailure),
    /// The grid cannot speak for this limit.
    Undetermined(GapUndetermined),
}

impl GapReport {
    /// The verdict for a printer that cannot keep gaps narrower than `limit` open, with the air-side
    /// proof `proof` made by [`prove_gaps`] for the same limit and bound.
    ///
    /// `Fail` (any zero-thickness sheet, a width with `width + GAP_TOLERANCE < limit`, or a crevice
    /// mouth below `limit`) outranks `Undetermined`, which outranks `Pass`. `Undetermined`: a limit
    /// that is not finite positive, a proof for another limit, narrow air the proof found that no
    /// crevice reading explains, a proof left incomplete, or (when the proof found narrow air) a grid
    /// too coarse to read it. `Pass` comes only from a proof that found no narrow air and left nothing
    /// undecided; its certificate states the band it holds for (`proven_from` up to the limit).
    #[must_use]
    pub fn verdict(&self, limit: f32, proof: &GapProof) -> GapVerdict {
        if !(limit.is_finite() && limit > 0.0) {
            return GapVerdict::Undetermined(GapUndetermined::BadLimit);
        }
        if proof.limit.to_bits() != limit.to_bits() {
            return GapVerdict::Undetermined(GapUndetermined::ProofForAnotherLimit {
                proof_limit: proof.limit,
            });
        }
        if let Some(at) = self.sheet_location {
            return GapVerdict::Fail(GapFailure::Sheet { at });
        }
        if let (Some(width), Some(at)) = (self.min_gap, self.location) {
            if width + GAP_TOLERANCE < limit {
                return GapVerdict::Fail(GapFailure::Width { width, at });
            }
        }
        let narrowest_mouth = self
            .crevices
            .iter()
            .filter(|c| c.mouth < limit)
            .min_by(|a, b| a.mouth.total_cmp(&b.mouth));
        if let Some(c) = narrowest_mouth {
            return GapVerdict::Fail(GapFailure::CreviceMouth {
                mouth: c.mouth,
                at: c.location,
            });
        }
        // a region the proof found (facing walls, or a box it could not decide) is part of a groove
        // when its walls meet at an angle (not parallel) that matches a crevice whose mouth is at or
        // above the limit (the ones below already failed), and it lies along that groove: within the
        // groove's wall length `mouth / (2 sin(α/2))` (plus L) of the crevice's joint line. The grid
        // reads such a groove by its mouth
        let in_groove = |r: &GapRegion| {
            let Some((n1, n2)) = r.walls else {
                return false;
            };
            let joint = n1.cross(n2);
            if joint.length() <= PARALLEL_SIN {
                return false;
            }
            let dir = joint.normalize();
            let air_angle = alice_det_math::acos((-n1.dot(n2)).law_max(-1.0).law_min(1.0))
                * (180.0 / core::f32::consts::PI);
            self.crevices.iter().any(|c| {
                let off = r.at - c.location;
                let from_line = (off - dir * off.dot(dir)).length();
                let half = (0.5 * c.angle_deg).to_radians();
                let wall_length = c.mouth / (2.0 * alice_det_math::sin(half).law_max(1e-3));
                c.mouth >= limit
                    && (c.angle_deg - air_angle).abs() <= CREVICE_ANGLE_MATCH_DEG
                    && from_line <= wall_length + limit
            })
        };
        // the proof found walls facing each other nearer than the limit, outside any groove: a gap
        if let Some(w) = proof.witnesses.iter().find(|w| !in_groove(w)) {
            return GapVerdict::Fail(GapFailure::ProvenGap {
                at: w.at,
                width: w.width.unwrap_or(limit),
            });
        }
        if let Some(at) = self.unconfirmed_sheet_location {
            return GapVerdict::Undetermined(GapUndetermined::UnconfirmedSheet {
                at,
                samples: self.unconfirmed_sheet_samples,
            });
        }
        let open_boxes = proof.undecided_at.iter().filter(|w| !in_groove(w)).count();
        if open_boxes > 0 || proof.truncated || proof.exhausted {
            return GapVerdict::Undetermined(GapUndetermined::ProofIncomplete {
                undecided: proof.undecided,
                truncated: proof.truncated,
                exhausted: proof.exhausted,
            });
        }
        GapVerdict::Pass(GapCertificate {
            cell: self.cell,
            any_shape_resolution: self.any_shape_resolution,
            min_certified_feature: self.cell,
            proven_from: 0.0,
        })
    }
}

/// `sin` of the largest angle between facing walls still read as parallel (about 0.06°): below it the
/// walls have no direction in which they meet. Near the line where curved walls touch (a cylinder lying
/// on a plate) the normals also turn almost antiparallel, so parallelism is not a stopping rule during
/// the walk; a width is decided by the width that stops closing.
const PARALLEL_SIN: f32 = 1e-3;

/// How far behind a surface point the field is read to tell material from a zero-thickness sheet
/// \[same unit as the SDF\]: a few times the march tolerance, far below any printable wall.
const SHEET_PROBE: f32 = 4.0 * GAP_TOLERANCE;

/// Whether the surface point `at`, whose material lies on the side `-out`, has material behind it.
///
/// A zero-thickness sheet has air on both sides: a short step behind it the field is positive again
/// and grows away from the sheet (its direction is `-out`). A step behind a point on a convex edge can
/// land exactly on the neighbouring face, where the field is 0 but its direction is that face's
/// normal, not `-out`: that is material, not a sheet.
fn material_behind(node: &SdfNode, at: Vec3, out: Vec3) -> bool {
    let behind = at - out * SHEET_PROBE;
    !(eval(node, behind) >= 0.0
        && eval_normal(node, behind).normalize_or_zero().dot(out) < -EDGE_COS)
}

/// Whether a sheet candidate `c` (a wall point read from a grid sample, air on the side `out`) is a
/// sheet on the exact field: marched along `-out` it reaches the surface (within `GAP_TOLERANCE`), and
/// a probe behind the surface ([`SHEET_PROBE`]) finds the field positive again, at least half the
/// probe (air on the far side of a zero level set). The near side needs no probe: the march came to
/// the surface from the candidate's air in steps of the field's own value, and on a field that bounds
/// the distance from below no step crosses a surface, so the path it took is air. A candidate read
/// where the field is not a distance (no progress toward the surface) or where the field is zero
/// behind the surface rather than positive is not confirmed.
fn sheet_confirmed(node: &SdfNode, c: Vec3, out: Vec3) -> bool {
    let Some(s) = wall_point(node, c, out) else {
        return false;
    };
    let tol = GAP_TOLERANCE;
    eval(node, s) >= -tol && eval(node, s - out * SHEET_PROBE) > 0.5 * SHEET_PROBE
}

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
fn crevice_angle(
    node: &SdfNode,
    a: Vec3,
    n: Vec3,
    width: f32,
    reach: f32,
    cell: f32,
) -> Option<(f32, Vec3, f32)> {
    let wall_normal = |at: Vec3| eval_normal(node, at).normalize_or_zero();
    let (a0, n0, w0) = (a, n, width);
    let mut away: Option<Vec3> = None;
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
        away.get_or_insert(-along);
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
            let dir = away?;
            return Some((angle, a2, crevice_mouth(node, a0, n0, w0, dir, reach, cell)));
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

/// Steps of the walk out of a crevice (each one cell along the wall) before the mouth is taken as the
/// last width measured.
const MOUTH_MAX_STEPS: usize = 4096;

/// Bisection steps that place the mouth between the last perpendicular that crossed the groove and the
/// first that left it.
const MOUTH_BISECT: usize = 24;

/// The mouth of a crevice: from wall point `a0` (normal `n0`, width `w0`), walk along the wall in `dir`
/// (away from the joint) one `cell` at a time, measuring the width perpendicular to the wall; the width
/// grows until the perpendicular passes the other wall's rim and finds open air. The step where that
/// happens is bisected, and the last width that still crossed the groove is the mouth.
fn crevice_mouth(
    node: &SdfNode,
    a0: Vec3,
    n0: Vec3,
    w0: f32,
    dir: Vec3,
    reach: f32,
    cell: f32,
) -> f32 {
    // a point on the wall `s` along `dir` from a0, with its wall normal (read off the wall along the
    // previous normal, so an edge does not tie the direction)
    let on_wall = |s: f32| -> Option<(Vec3, Vec3)> {
        let q = a0 + dir * s;
        let off = q + n0 * SHEET_PROBE;
        let nq = eval_normal(node, off).normalize_or_zero();
        if nq == Vec3::ZERO || nq.dot(n0) < EDGE_COS {
            return None;
        }
        Some((q - nq * eval(node, q), nq))
    };
    let width_at = |s: f32| on_wall(s).and_then(|(a, n)| local_gap(node, a, n, reach));
    let mut best = w0;
    let (mut inside, mut outside) = (0.0_f32, None);
    for k in 1..=MOUTH_MAX_STEPS {
        #[allow(clippy::cast_precision_loss)] // k is at most 4096
        let s = cell * k as f32;
        if s > reach {
            break;
        }
        match width_at(s) {
            Some(w) if w >= best => {
                best = w;
                inside = s;
            }
            // the width stopped growing: past the widest point
            Some(_) => break,
            None => {
                outside = Some(s);
                break;
            }
        }
    }
    if let Some(mut out) = outside {
        for _ in 0..MOUTH_BISECT {
            let mid = inside + 0.5 * (out - inside);
            match width_at(mid) {
                Some(w) => {
                    best = best.law_max(w);
                    inside = mid;
                }
                None => out = mid,
            }
        }
    }
    best
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

/// Most gap candidates refined on the surface after the grid pass (the narrowest first).
const MAX_REFINED_CANDIDATES: usize = 64;

/// Steps of the local search in [`refine_gap`].
const REFINE_MAX_STEPS: usize = 64;

/// The narrowest width near the gap sample at wall point `a0` (normal `n0`, width `w0`): a pattern
/// search on the wall that moves the start point along the two tangents, puts it back on the wall and
/// re-measures the width along the wall normal there, keeping a move only when the width shrinks and
/// halving the step otherwise. The start point never travels more than `cell` from `a0` (so the search
/// stays on the gap it started from and cannot slide into a joint where the width closes), and a width
/// at or below `4 · GAP_TOLERANCE` (a joint) is not taken. Returns `(w0, a0)` when nothing narrower is
/// found.
fn refine_gap(node: &SdfNode, a0: Vec3, n0: Vec3, w0: f32, reach: f32, cell: f32) -> (f32, Vec3) {
    let (mut a, mut n, mut w) = (a0, n0, w0);
    let mut step = 0.5 * cell;
    for _ in 0..REFINE_MAX_STEPS {
        if step < 0.5 * GAP_TOLERANCE {
            break;
        }
        let t1 = n.any_orthonormal_vector();
        let t2 = n.cross(t1);
        let mut moved = false;
        for t in [t1, -t1, t2, -t2] {
            let q = a + t * step;
            let nq = eval_normal(node, q).normalize_or_zero();
            if nq == Vec3::ZERO {
                continue;
            }
            let aq = q - nq * eval(node, q);
            if aq.distance(a0) > cell {
                continue;
            }
            let n2 = eval_normal(node, aq + nq * SHEET_PROBE).normalize_or_zero();
            if n2 == Vec3::ZERO || n2.dot(n) < EDGE_COS {
                continue;
            }
            if let Some(w2) = local_gap(node, aq, n2, reach) {
                if w2 > 4.0 * GAP_TOLERANCE && w2 < w {
                    (a, n, w) = (aq, n2, w2);
                    moved = true;
                    break;
                }
            }
        }
        if !moved {
            step *= 0.5;
        }
    }
    (w, a)
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
    // every gap sample (width, wall point, wall normal): the ones within a cell of the raw minimum are
    // refined on the surface after the pass
    let mut candidates: Vec<(f32, Vec3, Vec3)> = Vec::new();
    let mut crevices: Vec<Crevice> = Vec::new();
    let mut crevice_samples = 0_usize;
    let mut edge_samples = 0_usize;
    let mut sheet_samples = 0_usize;
    let mut sheet_location = None;
    let mut unconfirmed_sheet_samples = 0_usize;
    let mut unconfirmed_sheet_location = None;
    let mut sheet = |c: Vec3, out: Vec3| {
        if sheet_confirmed(node, c, out) {
            sheet_samples += 1;
            sheet_location.get_or_insert(c);
        } else {
            unconfirmed_sheet_samples += 1;
            unconfirmed_sheet_location.get_or_insert(c);
        }
    };
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
                // the wall point needs material behind it; a zero-thickness sheet has none (the field
                // only touches zero there), so no width is read from it, whether or not the ray comes back
                let at = p - n * d;
                if !material_behind(node, at, n) {
                    sheet(at, n);
                    continue;
                }
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
                // the surface the ray came back to needs material behind it too
                let hit = p + n * enter;
                if !material_behind(node, hit, -n) {
                    sheet(hit, -n);
                    continue;
                }
                if on_wall_edge(node, at, n, cell) {
                    edge_samples += 1;
                    continue;
                }
                if let Some((angle_deg, joint, mouth)) =
                    crevice_angle(node, at, n, width, reach, cell)
                {
                    crevice_samples += 1;
                    let merge = 2.0 * cell;
                    if crevices.len() < MAX_CREVICES
                        && !crevices
                            .iter()
                            .any(|c: &Crevice| c.location.distance(joint) <= merge)
                    {
                        crevices.push(Crevice {
                            location: joint,
                            angle_deg,
                            mouth,
                        });
                    }
                    continue;
                }
                gaps += 1;
                candidates.push((width, at, n));
                if best.is_none_or(|(b, _)| width < b) {
                    best = Some((width, at));
                }
            }
        }
    }
    if samples == 0 {
        return Err(GapError::NoSamples);
    }
    // a grid point measures along its own wall normal, which crosses the gap off its narrowest line by
    // up to the sample's offset: refine every candidate within a cell of the raw minimum by a local search
    // on the wall (bounded to one cell of travel, so it cannot slide into a joint)
    if let Some((raw, _)) = best {
        candidates.sort_by(|x, y| x.0.total_cmp(&y.0));
        for &(width, at, n) in candidates
            .iter()
            .take_while(|c| c.0 <= raw + cell)
            .take(MAX_REFINED_CANDIDATES)
        {
            let (w, a) = refine_gap(node, at, n, width, reach, cell);
            if best.is_none_or(|(b, _)| w < b) {
                best = Some((w, a));
            }
        }
    }
    Ok(GapReport {
        min_gap: best.map(|(w, _)| w),
        location: best.map(|(_, at)| at),
        samples,
        gaps,
        resolution,
        any_shape_resolution: 2.0 * resolution,
        cell,
        crevices,
        crevice_samples,
        edge_samples,
        sheet_samples,
        sheet_location,
        unconfirmed_sheet_samples,
        unconfirmed_sheet_location,
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
