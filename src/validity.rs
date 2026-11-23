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
//! The gap the proof and the verdict decide: an air point `p` narrower than `L` has its distance to
//! the material below `L/2` and two nearest material points `w₁`, `w₂` seen from it in directions with
//! dot below [`FACING_DOT`] (so `|w₁ − w₂| < L`, and the walls' normals there face each other). Every
//! rule reads this one statement: a witness is two air points whose wall points satisfy it; a box is
//! cleared when no point of it can, either because the field is at least `L/2` there, or because the
//! gradient enclosure holds no two directions with dot below [`FACING_DOT`], or because the material
//! near it, read from the sign of the field alone, cannot hold two nearest points in such directions.
//! The last clears what a field value cannot: a subtraction whose cutter overhangs the part leaves faces
//! where the field reads the overhang while the material is much further (see `clear_of_material`).
//!
//! Claim under test (not proven): no box that holds a gap point reaches the material cover, so the cover
//! only clears boxes without one and never decides a gap. The proof reads the cover on a box only when
//! the box is all air, no sampled pair of wall points is real (each wall a sample's march reached lies
//! more than twice the field value away, or the two are not in front of each other), and the march from
//! the box's centre misses the surface or lands more than twice the centre's value away. A gap point in
//! the box has both nearest walls within `L/2`; where the field is within a factor two of the distance to
//! the material, the samples' marches reach those walls and make a real pair, so the box goes to the
//! witness or is split instead. A box past the cover's gate that holds a gap would need both nearest
//! walls hidden behind faces nearer than them, where the field is less than half the distance. No such
//! scene is known (the slot, fins and V under an overhang in `tests/test_validity_distance_bound.rs`
//! all decide the gap before the cover, with the cover forced to clear), and a search over random
//! combined shapes is still to run. Until it has, the cover keeps soundness tests of its own (the rule
//! on two cones, material thinner than its field reads, a 40° V), so it stays sound if the claim fails.
//! Its cones are bounds under rounding as well: each half-angle and each angle between two axes is an
//! upper bound of the exact one, and a pair clears only below `2π/3` less the rounding left over (the
//! derivation term by term at `cover_cone`, `cone_pair` and `CLEAR_BELOW`).
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

use crate::compiled::real::{LawMinMax, LawVec};
use crate::diff::tree_hash;
use crate::eval::{eval, eval_normal};
use crate::interval::{eval_interval, Vec3Interval};
use crate::interval_gradient::{eval_interval_gradient, GradientInterval};
use crate::mesh::{validate_mesh, Mesh, MeshValidation};
use crate::types::SdfNode;
use glam::{DVec3, Vec3};

mod gap_open;
use gap_open::box_is_open;
pub use gap_open::{narrow_air_beside_is_open, narrow_air_is_open, OPEN_REACH_LIMITS};

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
    /// For a witness, the distance between the two wall points across the gap: within
    /// [`GAP_TOLERANCE`] of the distance between the walls (each march stops just short of its wall),
    /// and below the limit by more than that.
    pub width: Option<f32>,
}

/// The result of [`prove_gaps`] for one limit.
#[derive(Debug, Clone, PartialEq)]
pub struct GapProof {
    /// The limit `L` the proof was made for.
    pub limit: f32,
    /// One point per proven narrow region: two air points of the box around it see walls facing each
    /// other (direction dot below [`FACING_DOT`]) that are less than `L` apart, by more than
    /// [`GAP_TOLERANCE`] (the wall points' own tolerance).
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
    /// Lower corner of the bound proven over.
    pub bmin: Vec3,
    /// Upper corner of the bound proven over.
    pub bmax: Vec3,
    /// The subdivision depth the proof stopped at.
    pub max_depth: u32,
    /// [`crate::diff::tree_hash`] of the shape proven.
    ///
    /// A guard within one build only: the hash is not stable across versions of this crate, so do not
    /// store it to match a proof to a shape later. A stored proof is bound to the shape by the shape
    /// itself and to the arithmetic by [`crate::SDF_SEMANTICS_ID`].
    pub tree: u64,
    /// The unit build axis of a proof made by [`prove_gaps_in_layers`], `None` for [`prove_gaps`]. A
    /// layered proof does not subdivide narrow air that is open in its layer, so it speaks only for
    /// [`GapReport::verdict_in_layers`] with the same axis.
    pub build_axis: Option<Vec3>,
    /// Boxes a layered proof certified open in their layer and did not subdivide (0 for [`prove_gaps`]).
    pub open_boxes: usize,
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
/// - cleared when it is all air and the material within `L/2` of it, read from the field's sign alone,
///   cannot hold two nearest points of one of its points in directions with dot below [`FACING_DOT`]
///   (`clear_of_material`; at `max_depth` this is also tried on its octants, two levels down);
/// - a witness when two of its points (centre and corners) are air and see walls facing each other less
///   than `L` apart by more than [`GAP_TOLERANCE`] (see `facing_pair`; a pair within the tolerance of
///   `L` is neither narrower nor wider, so its box is split like any other and left undecided);
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
    prove(node, limit, bmin, bmax, max_depth, max_work, None)
}

/// [`prove_gaps`] for a shape printed in layers stacked along `build_axis`: a box whose every point is
/// open narrow air in its own layer (certified as in the "Boxes" paragraph of the `gap_open` MODEL) is
/// counted in [`GapProof::open_boxes`] and not subdivided, so a neck between separate walls ends
/// instead of being split until the budget runs out. Enclosed narrow air is proven or left undecided as
/// in [`prove_gaps`]. Read the result with [`GapReport::verdict_in_layers`] and the same axis.
///
/// # Errors
///
/// [`GapError::BadCell`]: as [`prove_gaps`], or `build_axis` zero or not finite.
pub fn prove_gaps_in_layers(
    node: &SdfNode,
    limit: f32,
    bmin: Vec3,
    bmax: Vec3,
    max_depth: u32,
    max_work: u64,
    build_axis: Vec3,
) -> Result<GapProof, GapError> {
    let len = build_axis.length();
    if !(len.is_finite() && len > 0.0) {
        return Err(GapError::BadCell);
    }
    prove(
        node,
        limit,
        bmin,
        bmax,
        max_depth,
        max_work,
        Some(build_axis / len),
    )
}

fn prove(
    node: &SdfNode,
    limit: f32,
    bmin: Vec3,
    bmax: Vec3,
    max_depth: u32,
    max_work: u64,
    build_axis: Option<Vec3>,
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
        bmin,
        bmax,
        max_depth,
        tree: tree_hash(node),
        build_axis,
        open_boxes: 0,
    };
    let mut m = Meter::new(node);
    let mut cover = CoverTree::new(bmin, bmax, half);
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
        // a box of air that is open narrow air in every point's own layer is a neck, not a gap
        if let Some(axis) = build_axis.filter(|_| iv.lo > 0.0) {
            let (open, evals) = box_is_open(node, lo, hi, limit, axis);
            m.charge(evals);
            if open {
                proof.open_boxes += 1;
                continue;
            }
        }
        let c = (lo + hi) * 0.5;
        let seen = air_walls(&mut m, lo, hi);
        let mut unreached = false;
        if let Some((walls, width)) = facing_pair(&mut m, &seen, limit, &mut unreached) {
            let r = GapRegion {
                at: c,
                walls: Some(walls),
                width: Some(width),
            };
            push_region(&mut proof.witnesses, r, limit, &mut proof.truncated);
            continue;
        }
        // a facing pair whose march does not reach the surface is a face the field has where no
        // material is (a subtraction's cutter overhanging the part): the box may still be L/2 from
        // every wall, which only the material cover can read
        // the field reads a face where no material is when the march from the box's centre does not
        // reach the surface, or reaches it much further than the field said: only then can the cover
        // clear what the field cannot
        if unreached && iv.lo > 0.0 {
            let d = m.value(c);
            let n = m.direction(c);
            unreached = d > 0.0
                && wall_point_with(&mut |q| m.value(q), c, n)
                    .is_none_or(|w| w.distance(c) > 2.0 * d);
        }
        if unreached
            && iv.lo > 0.0
            && clear_of_material(&mut m, &mut cover, lo, hi, half, max_work) == Cover::Clear
        {
            continue;
        }
        if depth >= max_depth {
            // the cover's bound tightens with the box: its parts are read, two levels down
            if unreached
                && iv.lo > 0.0
                && clear_in_parts(&mut m, lo, hi, half, max_work, COVER_EXTRA_DEPTH)
            {
                continue;
            }
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

/// Whether every point of the box `lo..hi` is at least `reach` from the material, read from the sign of
/// the field alone: a box whose value enclosure has a lower bound above zero holds no material, whatever
/// the field's magnitude there. Boxes of the region around `lo..hi` grown by `reach` are split while they
/// may hold material and lie nearer than `reach`; one no larger than `lo..hi` that may still hold material
/// that near leaves the box uncleared. Stops uncleared when the work reaches `max_work`.
///
/// The value of a field is only a lower bound on the distance, so a value test cannot clear a box next
/// to a cutter's face: inside a slot cut by a cutter that overhangs the slab, the field reads the
/// overhang (0.1) while the slot's walls are much further; the same values arise where material does
/// sit on that face. The sign separates the two.
fn clear_of_material(
    m: &mut Meter<'_>,
    tree: &mut CoverTree,
    lo: Vec3,
    hi: Vec3,
    reach: f32,
    max_work: u64,
) -> Cover {
    let Some((held, upper)) = material_cones(m, tree, lo, hi, reach, max_work) else {
        return Cover::Facing;
    };
    // a point of the box on a gap has two nearest material points, each at its distance (at most
    // `upper`), seen in directions with dot below `FACING_DOT`: only boxes within `upper` can hold
    // them. The search already holds only those: it prunes a box farther than `upper` when it pops,
    // pops boxes nearest first (a child is never nearer than its parent), and `upper` drops only by a
    // material point of a later box, which is at least that box's distance away, so no box it holds is
    // left farther than the final `upper` (`cover_tests::held_cones_stay_within_the_material_bound`).
    debug_assert!(held.iter().all(|h| h.0 <= upper));
    let mut out = Cover::Clear;
    for (i, &(_, d1, h1)) in held.iter().enumerate() {
        for &(_, d2, h2) in &held[i..] {
            match cone_pair(d1, h1, d2, h2) {
                Cover::Facing => return Cover::Facing,
                Cover::Wide => out = Cover::Wide,
                Cover::Clear => {}
            }
        }
    }
    out
}

/// A held cone: its least distance from the box, its axis (not normalised), and an upper bound on its
/// half-angle.
type HeldCone = (f32, DVec3, f32);

/// The cones of directions from the box `lo..hi` to the boxes of the cover near enough to hold its
/// nearest material, with the upper bound on the distance to the material found on the way; `None`
/// when the work, the reads or the cones ran out (the box then stays).
fn material_cones(
    m: &mut Meter<'_>,
    tree: &mut CoverTree,
    lo: Vec3,
    hi: Vec3,
    reach: f32,
    max_work: u64,
) -> Option<(Vec<HeldCone>, f32)> {
    let leaf = (hi - lo).law_max_element();
    // the directions from the box to a box `a..b`: a cone around the centre of their difference that
    // holds every `y − x`, with the least distance between the two boxes
    let cone = |a: Vec3, b: Vec3| {
        let (dlo, dhi) = (a - hi, b - lo);
        let gap = dlo.law_max(-dhi).law_max(Vec3::ZERO).length();
        let (axis, half_angle) = cover_cone(lo, hi, a, b);
        (gap, axis, half_angle)
    };
    // an upper bound on the distance from any point of the box to the material, from material points
    // found on the way (a point whose field is at most zero is material)
    let mut upper = f32::INFINITY;
    let mut held: Vec<HeldCone> = Vec::new();
    // nearest boxes first: the cones that decide the box are found before the far ones
    let mut stack: std::collections::BinaryHeap<Near> = std::collections::BinaryHeap::new();
    for key in tree.start(lo - Vec3::splat(reach), hi + Vec3::splat(reach)) {
        m.bookkeeping(1);
        stack.push(Near(u32::MAX, key));
    }
    let mut reads = 0_usize;
    while let Some(Near(_, key)) = stack.pop() {
        m.bookkeeping(1);
        if m.work >= max_work || reads >= MAX_COVER_READS {
            return None;
        }
        let (a, b) = tree.bounds(key);
        let (gap, dir, half_angle) = cone(a, b);
        if gap >= reach || gap > upper {
            continue;
        }
        reads += 1;
        if !tree.may_hold_material(m, key) {
            continue;
        }
        if half_angle > CONE_LEAF
            && (b - a).law_max_element() > leaf
            && key.depth < CoverKey::MAX_DEPTH
        {
            for child in key.children() {
                let (ca, cb) = tree.bounds(child);
                m.bookkeeping(1);
                stack.push(Near(near_key(cone(ca, cb).0), child));
            }
            continue;
        }
        // a material point bounds the distance from every point of the box: the centre of a cover box
        // (cached) and the point of it nearest the box's centre
        let q = (a + b) * 0.5;
        if tree.material_at_centre(m, key) {
            upper = upper.law_min((q - lo).abs().law_max((q - hi).abs()).length());
        }
        let near = ((lo + hi) * 0.5).law_clamp(a, b);
        if m.value(near) <= 0.0 {
            upper = upper.law_min((near - lo).abs().law_max((near - hi).abs()).length());
        }
        if held.len() >= MAX_CONES {
            return None;
        }
        held.push((gap, dir, half_angle));
    }
    Some((held, upper))
}

/// The cone of directions from the box `lo..hi` to the box `a..b`: its axis `C` (the centre of the
/// difference box `a − hi ..= b − lo`) and an upper bound on its half-angle `asin(R/|C|)`, where `R` is
/// the difference box's half-diagonal; the ball of radius `R` around `C` holds every `y − x`, so the
/// cone holds every direction from the box to the other box. `π` when the ball holds the origin.
///
/// Rounding (`u = 2⁻⁵³`, one `f64` rounding): the corners are `f32`, so each difference is one
/// rounding in `f64`, and with `D = C ∓ R` per component the computed centre and half-diagonal are
/// within `u(2|C| + R)(1 + u)` and `u(|C| + 2R)(1 + u)` of the exact ones; each length adds at most
/// `2.5u` (three squares, two sums, a square root) and the ratio `u`. For `q = R/|C|` that leaves the
/// computed ratio at least `q − u(1 + 10q + q²)`, which is `q − 12u` for `q ≤ 1`, and at least
/// `1 − 12u` whenever `q ≥ 1` (the bound grows with `q` until `|C|` is lost in the rounding, where the
/// computed ratio is near `1/u`). Adding [`Q_SLACK`] (`12u`, `u` for second-order terms, `u` for the
/// addition) leaves a ratio at least the exact one, or at least 1. It is cast to the `f32` at or above
/// it, `asin` is increasing, and `alice_det_math::asin` is within 1 ulp of the correctly rounded value,
/// so two steps up hold the exact half-angle (two, so the bound holds when the exact value lies past a
/// power of two above the computed one).
///
/// The bias goes on the argument, not on the result: `asin` is only ½-Hölder at 1, so an `f32` ratio
/// a few ulp short of 1 moves the half-angle by up to `sqrt(2δ)` (`7·10⁻⁴` rad at an exact `q = 1`,
/// `cover_tests::a_cone_through_the_box_holds_a_half_space`), more than any margin on the angles
/// would cover.
///
/// The `f32` bound rests on the accuracy table of `alice-det-math` (`asin` / `acos` within 1 ulp
/// of the correctly rounded value, so within two steps of the exact one), which that crate measures
/// on every `f32` of `[-1, 1]` (its `tests/exhaustive_ulp_f32.rs`).
fn cover_cone(lo: Vec3, hi: Vec3, a: Vec3, b: Vec3) -> (DVec3, f32) {
    let (dlo, dhi) = (a.as_dvec3() - hi.as_dvec3(), b.as_dvec3() - lo.as_dvec3());
    let c = (dlo + dhi) * 0.5;
    let r = ((dhi - dlo) * 0.5).length();
    let cl = c.length();
    let q = if cl > 0.0 {
        r / cl + Q_SLACK
    } else {
        f64::INFINITY
    };
    let half_angle = if q < 1.0 {
        two_ulps_up(alice_det_math::asin(f32_at_or_above(q)))
    } else {
        core::f32::consts::PI
    };
    (c, half_angle)
}

/// One `f64` rounding, `2⁻⁵³`.
const U64: f64 = f64::EPSILON / 2.0;

/// Added to a cone's computed ratio `R/|C|` so it is at least the exact one ([`cover_cone`]): `12u`
/// for the differences, the lengths and the division, `u` for second-order terms, `u` for the addition.
const Q_SLACK: f64 = 14.0 * U64;

/// Taken off the computed cosine of the angle between two axes so it is at most the exact cosine
/// ([`cone_pair`]): `3u` for the dot of three products, `5u` for the two lengths, `u` for their
/// product, `u` for the division, `u` for second-order terms, `u` for the subtraction.
const DOT_SLACK: f64 = 12.0 * U64;

/// A pair of cones whose angles sum below this clears ([`cone_pair`]). The sum is in `f64`, of the
/// angle between the computed axes (an upper bound), and the two half-angles (upper bounds around the
/// exact centres). The widest angle between the cones is at most that sum plus two axis terms: each
/// computed axis is within `u(2 + q)(1 + u)|C| ≤ 3.01u|C|` of the exact centre where a pair can clear
/// (`q < 1`), so at most `asin(3.01u) < 5u` off it. The `f64` sum of three angles near `2π/3` rounds
/// twice by at most half an ulp of `[2, 4)`, `2u` each. So a pair clears soundly when the sum is below
/// `2π/3 − 14u`. `2·FRAC_PI_3` is within `2u` of `2π/3` (the constant is correctly rounded), and the
/// grid of `f64` in `[2, 4)` is `4u`, so `2·FRAC_PI_3 − 20u` is exact and at most `2π/3 − 18u`
/// (`cover_tests::the_clearing_bound_is_its_derivation` recomputes it).
const CLEAR_BELOW: f64 = 2.094_395_102_393_193_5;

/// `x` rounded to the `f32` at or above it.
fn f32_at_or_above(x: f64) -> f32 {
    #[allow(clippy::cast_possible_truncation)] // the rounding is the point; it is corrected below
    let y = x as f32;
    if f64::from(y) < x {
        y.next_up()
    } else {
        y
    }
}

/// `x` rounded to the `f32` at or below it.
fn f32_at_or_below(x: f64) -> f32 {
    #[allow(clippy::cast_possible_truncation)] // the rounding is the point; it is corrected below
    let y = x as f32;
    if f64::from(y) > x {
        y.next_down()
    } else {
        y
    }
}

/// Two steps up: past a result within 1 ulp of the exact one on either side of a power of two.
const fn two_ulps_up(x: f32) -> f32 {
    x.next_up().next_up()
}

/// Whether two cones of directions (axes `c1`, `c2`, half-angle bounds `h1`, `h2`) can hold directions
/// with dot below [`FACING_DOT`]: the widest angle between them is the angle between the axes plus both
/// half-angles. `Facing` when the axes alone are (nearly) that far apart, so smaller boxes will not clear
/// it; `Wide` when only the widths reach it.
///
/// The angle between the axes is an upper bound: the cosine computed in `f64` less [`DOT_SLACK`] is at
/// most the exact one, cast to the `f32` at or below it, and `acos` is decreasing and within 1 ulp, so
/// two steps up hold the exact angle. The sum clears only below [`CLEAR_BELOW`].
fn cone_pair(c1: DVec3, h1: f32, c2: DVec3, h2: f32) -> Cover {
    let den = c1.length() * c2.length();
    // a zero, overflowing or NaN length (a cone whose ball holds the origin) never clears
    if !den.is_finite() || den <= 0.0 {
        return Cover::Wide;
    }
    let cos_lo = f32_at_or_below(c1.dot(c2) / den - DOT_SLACK)
        .law_max(-1.0)
        .law_min(1.0);
    let between = two_ulps_up(alice_det_math::acos(cos_lo));
    if between >= FACING_ANGLE - FACING_MARGIN {
        Cover::Facing
    } else if f64::from(between) + f64::from(h1) + f64::from(h2) < CLEAR_BELOW {
        Cover::Clear
    } else {
        Cover::Wide
    }
}

/// A box of [`clear_of_material`]'s search, ordered nearest first by its distance's bits.
struct Near(u32, CoverKey);

/// The heap key of a distance: smaller distances come out first (non-negative floats order as bits).
const fn near_key(gap: f32) -> u32 {
    u32::MAX - gap.to_bits()
}

impl PartialEq for Near {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl Eq for Near {}
impl PartialOrd for Near {
    fn partial_cmp(&self, o: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Near {
    fn cmp(&self, o: &Self) -> core::cmp::Ordering {
        self.0.cmp(&o.0)
    }
}

/// A box of the material cover: its depth below the root and its index on each axis at that depth.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CoverKey {
    depth: u8,
    at: [u32; 3],
}

impl CoverKey {
    /// Deepest box of the cover (indices fit `u32`).
    const MAX_DEPTH: u8 = 24;

    fn children(self) -> [Self; 8] {
        core::array::from_fn(|i| Self {
            depth: self.depth + 1,
            at: core::array::from_fn(|k| 2 * self.at[k] + u32::from((i >> k) & 1 == 1)),
        })
    }
}

/// The material cover shared by every box of one proof: one octree over the proof's bound grown by the
/// reach, whose boxes are read once (whether they may hold material, whether their centre is material)
/// whichever box of the proof asks.
struct CoverTree {
    origin: Vec3,
    size: Vec3,
    seen: std::collections::BTreeMap<CoverKey, (bool, Option<bool>)>,
}

impl CoverTree {
    fn new(bmin: Vec3, bmax: Vec3, reach: f32) -> Self {
        // a cube, so every box of the cover is a cube (the cones stay round)
        let origin = bmin - Vec3::splat(reach);
        Self {
            origin,
            size: Vec3::splat((bmax + Vec3::splat(reach) - origin).law_max_element()),
            seen: std::collections::BTreeMap::new(),
        }
    }

    /// A cover of its own around one box (the box grown by the reach, split through the box's centre):
    /// its boxes line up with the box, which reads tighter cones next to it than the shared cover.
    fn around(lo: Vec3, hi: Vec3, reach: f32) -> Self {
        let a = lo - Vec3::splat(reach);
        Self {
            origin: a,
            size: hi + Vec3::splat(reach) - a,
            seen: std::collections::BTreeMap::new(),
        }
    }

    /// The boxes, at the deepest level whose boxes are no smaller than `a..b` on any axis, that `a..b`
    /// overlaps (at most two per axis): the search around a box starts there, not at the root.
    fn start(&self, a: Vec3, b: Vec3) -> Vec<CoverKey> {
        let ext = (b - a).law_max(Vec3::splat(f32::MIN_POSITIVE));
        let ratio = (self.size / ext).law_min_element().law_max(1.0);
        // floor(log2(ratio)) read from the exponent bits: exact, and ratio >= 1 is normal or infinite
        let exponent = ((ratio.to_bits() >> 23) & 0xff) as u8;
        let depth = exponent.saturating_sub(127).min(CoverKey::MAX_DEPTH);
        let n = (1_u64 << depth) as f32;
        let cell = |p: Vec3| {
            let t = ((p - self.origin) / self.size * n).floor();
            t.law_clamp(Vec3::ZERO, Vec3::splat(n - 1.0))
        };
        let (c0, c1) = (cell(a), cell(b));
        let mut out = Vec::with_capacity(8);
        for i in c0.x as u32..=c1.x as u32 {
            for j in c0.y as u32..=c1.y as u32 {
                for k in c0.z as u32..=c1.z as u32 {
                    out.push(CoverKey {
                        depth,
                        at: [i, j, k],
                    });
                }
            }
        }
        out
    }

    fn bounds(&self, k: CoverKey) -> (Vec3, Vec3) {
        let n = (1_u64 << k.depth) as f32;
        let at = Vec3::new(k.at[0] as f32, k.at[1] as f32, k.at[2] as f32);
        (
            self.origin + self.size * (at / n),
            self.origin + self.size * ((at + Vec3::ONE) / n),
        )
    }

    /// The value enclosure's lower bound over the box is at most zero.
    fn may_hold_material(&mut self, m: &mut Meter<'_>, k: CoverKey) -> bool {
        m.bookkeeping(1);
        if let Some(e) = self.seen.get(&k) {
            return e.0;
        }
        let (a, b) = self.bounds(k);
        let may = m.interval(Vec3Interval::from_bounds(a, b)).lo <= 0.0;
        self.seen.insert(k, (may, None));
        may
    }

    /// The field at the box's centre is at most zero (the centre is material).
    fn material_at_centre(&mut self, m: &mut Meter<'_>, k: CoverKey) -> bool {
        m.bookkeeping(1);
        if let Some(&(_, Some(c))) = self.seen.get(&k) {
            return c;
        }
        let (a, b) = self.bounds(k);
        let c = m.value((a + b) * 0.5) <= 0.0;
        self.seen.entry(k).or_insert((true, None)).1 = Some(c);
        c
    }
}

/// What [`clear_of_material`] read around a box.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cover {
    /// No point of the box can have two nearest material points facing each other.
    Clear,
    /// Only the cones' widths keep the box from clearing: its parts may clear.
    Wide,
    /// Two cones face each other along their axes (or the work or the cones ran out): the box stays.
    Facing,
}

/// The cones' axes within this angle of facing make [`clear_of_material`] give up on a box rather
/// than read its parts.
const FACING_MARGIN: f32 = 0.05;

/// [`clear_of_material`] on the box, or else on each of its octants down to `extra` more levels while
/// only the cones' widths keep it from clearing.
fn clear_in_parts(
    m: &mut Meter<'_>,
    lo: Vec3,
    hi: Vec3,
    reach: f32,
    max_work: u64,
    extra: u32,
) -> bool {
    let mut own = CoverTree::around(lo, hi, reach);
    match clear_of_material(m, &mut own, lo, hi, reach, max_work) {
        Cover::Clear => true,
        Cover::Facing => false,
        Cover::Wide => {
            extra > 0
                && octants(lo, hi)
                    .into_iter()
                    .all(|(a, b)| clear_in_parts(m, a, b, reach, max_work, extra - 1))
        }
    }
}

/// Work units of one bookkeeping operation of the material cover (a heap push or pop, a cache lookup).
const BOOKKEEPING_UNITS: u64 = 3;

/// Levels below the proof's last depth at which [`clear_in_parts`] reads the material cover.
const COVER_EXTRA_DEPTH: u32 = 2;

/// The angle between two directions whose dot is [`FACING_DOT`] (120°).
const FACING_ANGLE: f32 = 2.0 * core::f32::consts::FRAC_PI_3;
/// Half-angle under which a box that may hold material is kept whole in [`clear_of_material`].
const CONE_LEAF: f32 = 0.26;
/// Boxes read by one [`clear_of_material`] before it gives up on the box.
const MAX_COVER_READS: usize = 4096;
/// Boxes that may hold material kept by [`clear_of_material`] before it gives up on the box.
const MAX_CONES: usize = 512;

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

    /// Work of the proof's own bookkeeping (a heap or cache operation of the material cover), counted so
    /// the work keeps tracking the time where the tree's evaluations are cached (see `BOOKKEEPING_UNITS`).
    const fn bookkeeping(&mut self, ops: u64) {
        self.work = self
            .work
            .saturating_add(ops.saturating_mul(BOOKKEEPING_UNITS));
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
/// whose walls face away, are not a gap), and `|w₁ − w₂| + GAP_TOLERANCE < limit`: two surface points
/// across the air nearer than the limit by more than the wall points' tolerance, the same margin the
/// grid's widths fail by. Returns the two directions and `|w₁ − w₂|`.
fn facing_pair(
    m: &mut Meter<'_>,
    pts: &[(Vec3, f32, Vec3)],
    limit: f32,
    unreached: &mut bool,
) -> Option<((Vec3, Vec3), f32)> {
    // each air point's wall point is marched at most once (lazily: only points in a facing pair)
    let mut walls: Vec<Option<Option<Vec3>>> = vec![None; pts.len()];
    let mut wall = |m: &mut Meter<'_>, k: usize| {
        *walls[k].get_or_insert_with(|| wall_point_with(&mut |q| m.value(q), pts[k].0, pts[k].2))
    };
    let mut real_pair = false;
    for (i, a) in pts.iter().enumerate() {
        for (j, b) in pts.iter().enumerate().skip(i + 1) {
            if a.2.dot(b.2) >= FACING_DOT {
                continue;
            }
            let (Some(w1), Some(w2)) = (wall(m, i), wall(m, j)) else {
                continue;
            };
            let width = w1.distance(w2);
            let fronts = (w2 - w1).dot(a.2) > 0.0 && (w1 - w2).dot(b.2) > 0.0;
            let real = fronts && a.0.distance(w1) <= 2.0 * a.1 && b.0.distance(w2) <= 2.0 * b.1;
            // each wall point stops in the air where the field falls to `GAP_TOLERANCE / 4` (the march
            // never overshoots a field that bounds the distance from below): on a distance-exact field
            // that is within a quarter tolerance of its wall, so `width` is short of the walls'
            // distance by at most half the tolerance, and a width within `GAP_TOLERANCE` of the limit
            // proves nothing either way
            if a.2.dot(b.2) < FACING_DOT && fronts && width + GAP_TOLERANCE < limit {
                return Some(((a.2, b.2), width));
            }
            // a facing pair of the material's walls (in front of each other, reached about where the
            // field said) decides the box by width; one of the field's own faces does not
            if real {
                real_pair = true;
            }
        }
    }
    *unreached = !real_pair;
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
    /// than about 20° within a quarter cell): the wall has no single direction there, so neither a
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
    /// Samples that looked like a sheet on the grid (a wall point on the surface with no material
    /// behind it) but whose point is not confirmed on the exact field: a probe behind the surface does
    /// not find the field positive by at least half the probe (the field rises slower than the distance
    /// there, or is zero rather than positive). No width is read from them, and they keep the verdict
    /// from passing; they are not a failure, since nothing on the exact field shows a sheet.
    pub unconfirmed_sheet_samples: usize,
    /// One unconfirmed sheet point (the first found), `None` when `unconfirmed_sheet_samples` is 0.
    pub unconfirmed_sheet_location: Option<Vec3>,
    /// Samples with no wall in their direction: `p − d n` is not on the surface (the field is a lower
    /// bound whose gradient does not point at the nearest wall, as inside a through cut, where the
    /// cutter's term points at the cutter's own face out in the air) and a march along `−n` reaches no
    /// surface either. Neither a width nor a sheet is read from them; the walls near them are read by
    /// their own samples. Counted among `samples`; a grid where every sample is one of these measured
    /// nothing ([`GapError::NoSamples`]). A count above 0 is never a pass on its own: a sample with no
    /// wall measured nothing, and a pass comes only from the proof over the whole bound.
    pub no_wall_samples: usize,
    /// One sample point with no wall (the first found), `None` when `no_wall_samples` is 0.
    pub no_wall_location: Option<Vec3>,
    /// Lower corner of the bound the grid sampled.
    pub bmin: Vec3,
    /// Upper corner of the bound the grid sampled.
    pub bmax: Vec3,
    /// [`crate::diff::tree_hash`] of the shape the grid sampled.
    ///
    /// A guard within one build only: the hash is not stable across versions of this crate, so do not
    /// store it to match a proof to a shape later. A stored proof is bound to the shape by the shape
    /// itself and to the arithmetic by [`crate::SDF_SEMANTICS_ID`].
    pub tree: u64,
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
    /// Lower corner of the bound the pass holds for (the bound the grid sampled; the proof covers it).
    pub bmin: Vec3,
    /// Upper corner of the bound the pass holds for.
    pub bmax: Vec3,
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
        /// The distance between the two wall points there: within [`GAP_TOLERANCE`] of the gap, and
        /// below the limit by more than that.
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
    /// The build axis given to [`GapReport::verdict_in_layers`] is zero or not finite.
    BadBuildAxis,
    /// The proof was made for another build axis than the verdict reads with (`None`: a plain
    /// [`prove_gaps`] proof read in layers, or a layered proof read by the plain verdict).
    ProofForAnotherBuildAxis {
        /// The proof's build axis.
        proof_axis: Option<Vec3>,
    },
    /// The proof was made for another limit than the one asked.
    ProofForAnotherLimit {
        /// The proof's limit.
        proof_limit: f32,
    },
    /// The proof was made for another shape than the one the grid sampled (their tree hashes differ).
    ProofForAnotherShape,
    /// The proof's bound does not cover the bound the grid sampled, so part of the measured region has
    /// no proof.
    ProofForAnotherRegion {
        /// Lower corner of the proof's bound.
        proof_bmin: Vec3,
        /// Upper corner of the proof's bound.
        proof_bmax: Vec3,
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
    /// that is not finite positive, a proof for another limit, another shape or a bound that does not cover the grid's, narrow air the proof found that no
    /// crevice reading explains, a proof left incomplete, or (when the proof found narrow air) a grid
    /// too coarse to read it. `Pass` comes only from a proof that found no narrow air and left nothing
    /// undecided; its certificate states the band it holds for (`proven_from` up to the limit).
    #[must_use]
    pub fn verdict(&self, limit: f32, proof: &GapProof) -> GapVerdict {
        if proof.build_axis.is_some() {
            return GapVerdict::Undetermined(GapUndetermined::ProofForAnotherBuildAxis {
                proof_axis: proof.build_axis,
            });
        }
        self.verdict_with(limit, proof, &|_| false, &|_| false)
    }

    /// [`GapReport::verdict`] for a shape printed in layers stacked along `build_axis`: narrow air that
    /// is open in its layer (it reaches air of the limit's width within [`OPEN_REACH_LIMITS`] limits in
    /// the same layer, see [`narrow_air_is_open`]) is a neck between separate walls, not a hole, and
    /// does not fail the shape. A measured width below the limit is skipped when the air beside its wall
    /// point is open; a proof witness or undecided box is skipped when its centre is open air. Enclosed
    /// narrow air (a hole, a slot closed in the layer, a hole through a plate along the build axis) fails
    /// as in [`GapReport::verdict`]. A skipped width leaves the answer to the proof: `Pass` still needs a
    /// complete proof.
    ///
    /// `node` must be the shape the report and the proof were made for (its tree hash is checked).
    /// `Undetermined(BadBuildAxis)` for a zero or non-finite axis.
    #[must_use]
    pub fn verdict_in_layers(
        &self,
        node: &SdfNode,
        limit: f32,
        proof: &GapProof,
        build_axis: Vec3,
    ) -> GapVerdict {
        if !(build_axis.is_finite() && build_axis.length() > 0.0) {
            return GapVerdict::Undetermined(GapUndetermined::BadBuildAxis);
        }
        if tree_hash(node) != self.tree {
            return GapVerdict::Undetermined(GapUndetermined::ProofForAnotherShape);
        }
        let axis = build_axis / build_axis.length();
        if proof.build_axis.map(f32x3_bits) != Some(f32x3_bits(axis)) {
            return GapVerdict::Undetermined(GapUndetermined::ProofForAnotherBuildAxis {
                proof_axis: proof.build_axis,
            });
        }
        self.verdict_with(
            limit,
            proof,
            &|at| narrow_air_beside_is_open(node, at, limit, build_axis),
            &|at| narrow_air_is_open(node, at, limit, build_axis),
        )
    }

    /// The verdict, skipping a measured width whose wall point has open air beside it
    /// (`open_beside`) and proof regions whose centre is open air (`open_at`).
    fn verdict_with(
        &self,
        limit: f32,
        proof: &GapProof,
        open_beside: &dyn Fn(Vec3) -> bool,
        open_at: &dyn Fn(Vec3) -> bool,
    ) -> GapVerdict {
        if !(limit.is_finite() && limit > 0.0) {
            return GapVerdict::Undetermined(GapUndetermined::BadLimit);
        }
        if proof.limit.to_bits() != limit.to_bits() {
            return GapVerdict::Undetermined(GapUndetermined::ProofForAnotherLimit {
                proof_limit: proof.limit,
            });
        }
        if proof.tree != self.tree {
            return GapVerdict::Undetermined(GapUndetermined::ProofForAnotherShape);
        }
        if proof.bmin.cmpgt(self.bmin).any() || proof.bmax.cmplt(self.bmax).any() {
            return GapVerdict::Undetermined(GapUndetermined::ProofForAnotherRegion {
                proof_bmin: proof.bmin,
                proof_bmax: proof.bmax,
            });
        }
        if let Some(at) = self.sheet_location {
            return GapVerdict::Fail(GapFailure::Sheet { at });
        }
        if let (Some(width), Some(at)) = (self.min_gap, self.location) {
            if width + GAP_TOLERANCE < limit && !open_beside(at) {
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
        if let Some(w) = proof
            .witnesses
            .iter()
            .find(|w| !in_groove(w) && !open_at(w.at))
        {
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
        let open_boxes = proof
            .undecided_at
            .iter()
            .filter(|w| !in_groove(w) && !open_at(w.at))
            .count();
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
            bmin: self.bmin,
            bmax: self.bmax,
        })
    }
}

/// The bits of a vector, for comparing build axes exactly.
fn f32x3_bits(v: Vec3) -> [u32; 3] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]
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
    /// No grid point lay in empty space near a surface, or none of those that did reached a wall:
    /// nothing was compared.
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
/// [`GapError::NoSamples`] (no surface inside the bound, or no sample reached a wall: nothing
/// compared).
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
    let mut no_wall_samples = 0_usize;
    let mut no_wall_location = None;
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
                // the wall point is `p − d n` only where that lands on the surface (a distance-exact
                // field). Where the field is a lower bound whose gradient points elsewhere (inside a
                // through cut the cutter's term points at the cutter's own face out in the air), that
                // point is on no surface: march to the wall instead, and a sample whose march reaches
                // none has no wall in its direction (counted, no width and no sheet read from it)
                let guess = p - n * d;
                let (at, lead) = if eval(node, guess).abs() <= GAP_TOLERANCE {
                    (guess, d)
                } else if let Some(w) = wall_point(node, p, n) {
                    (w, p.distance(w))
                } else {
                    no_wall_samples += 1;
                    no_wall_location.get_or_insert(p);
                    continue;
                };
                // the wall point needs material behind it; a zero-thickness sheet has none (the field
                // only touches zero there), so no width is read from it, whether or not the ray comes back
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
                // from the sample back to its wall (`d` where `p − d n` is the wall), then out to the hit
                let width = lead + enter;
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
    // a sample whose march reached no wall compared nothing: a grid of only such samples measured
    // nothing, whatever its count
    if samples == no_wall_samples {
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
        no_wall_samples,
        no_wall_location,
        bmin,
        bmax,
        tree: tree_hash(node),
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

#[cfg(test)]
mod cover_tests {
    //! oracle: the rule on two cones (closed form: the widest angle between two cones is the angle
    //! between their axes plus both half-angles) and three scenes where only the material cover decides
    //! a box, so no other rule of the proof can hide a wrong cover.
    use super::*;

    fn cover(s: &SdfNode, lo: Vec3, hi: Vec3, reach: f32) -> Cover {
        let mut m = Meter::new(s);
        let mut tree = CoverTree::new(lo, hi, reach);
        clear_of_material(&mut m, &mut tree, lo, hi, reach, GAP_PROOF_WORK)
    }

    fn at(degrees: f32) -> DVec3 {
        let (s, c) = alice_det_math::sin_cos(degrees.to_radians());
        DVec3::new(f64::from(c), f64::from(s), 0.0)
    }

    /// A small seeded generator (xorshift64*), so the random scenes are the same on every run.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            // the top 24 bits, exact in f32: a value in [0, 1)
            #[allow(clippy::cast_precision_loss)]
            let u = (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32;
            u / 16_777_216.0
        }
        fn range(&mut self, a: f32, b: f32) -> f32 {
            a + (b - a) * self.next()
        }
        fn point(&mut self, r: f32) -> Vec3 {
            Vec3::new(self.range(-r, r), self.range(-r, r), self.range(-r, r))
        }
    }

    /// Scenes of the random check of the held cones (seed and count stated so a failure reproduces).
    const HELD_SEED: u64 = 0x5DF0_C0FE_2026_1118;
    const HELD_SCENES: usize = 400;

    #[test]
    fn held_cones_stay_within_the_material_bound() {
        // every cone the search holds is no farther than the final upper bound on the distance to the
        // material, so dropping the farther ones would drop nothing (the argument at `clear_of_material`)
        let mut rng = Rng(HELD_SEED);
        let (mut scenes_with_cones, mut cones) = (0_usize, 0_usize);
        for _ in 0..HELD_SCENES {
            let mut s = SdfNode::sphere(rng.range(0.1, 1.0)).translate_vec(rng.point(1.0));
            for _ in 0..3 {
                let size = Vec3::new(
                    rng.range(0.05, 1.5),
                    rng.range(0.05, 1.5),
                    rng.range(0.05, 1.5),
                );
                let part = SdfNode::box3d(size.x, size.y, size.z).translate_vec(rng.point(1.0));
                s = if rng.next() < 0.3 {
                    s.subtract(part)
                } else {
                    s.union(part)
                };
            }
            let lo = rng.point(1.2);
            let hi = lo + Vec3::splat(rng.range(0.02, 0.3));
            let reach = rng.range(0.1, 1.0);
            let mut m = Meter::new(&s);
            let mut tree = CoverTree::new(lo, hi, reach);
            if let Some((held, upper)) =
                material_cones(&mut m, &mut tree, lo, hi, reach, GAP_PROOF_WORK)
            {
                for h in &held {
                    assert!(
                        h.0 <= upper,
                        "held cone at {} beyond the bound {upper} (box {lo}..{hi})",
                        h.0
                    );
                }
                cones += held.len();
                scenes_with_cones += usize::from(!held.is_empty());
            }
        }
        // the check saw cones in a fair share of the scenes (not an empty pass)
        assert!(
            scenes_with_cones >= HELD_SCENES / 4,
            "{scenes_with_cones} scenes held cones"
        );
        assert!(cones >= HELD_SCENES, "{cones} cones checked");
    }

    #[test]
    fn two_cones_compare_by_their_bounds_not_their_axes() {
        // axes 110° apart, 6° each side: the widest pair is 122°, past 120°
        let h = 6.0_f32.to_radians();
        assert_eq!(cone_pair(at(0.0), h, at(110.0), h), Cover::Wide);
        // axes 100° apart, 6° each side: at most 112°
        assert_eq!(cone_pair(at(0.0), h, at(100.0), h), Cover::Clear);
        // axes 140° apart: facing on the axes alone
        assert_eq!(cone_pair(at(0.0), 0.0, at(140.0), 0.0), Cover::Facing);
    }

    #[test]
    fn a_box_over_one_wall_clears_in_parts() {
        // a slab |y| <= 0.5 and a box 0.2 above it: one wall has no facing pair, though the wall's
        // points within the reach are seen in directions almost opposite along the wall
        let s = SdfNode::box3d(20.0, 1.0, 20.0);
        let lo = Vec3::new(-0.05, 0.65, -0.05);
        let hi = lo + Vec3::splat(0.1);
        // the box itself may read `Wide` (its own width); it is never `Facing`, and its parts clear
        assert_ne!(cover(&s, lo, hi, 1.0), Cover::Facing);
        let mut m = Meter::new(&s);
        assert!(clear_in_parts(
            &mut m,
            lo,
            hi,
            1.0,
            GAP_PROOF_WORK,
            COVER_EXTRA_DEPTH
        ));
    }

    #[test]
    fn the_clearing_bound_is_its_derivation() {
        // 2π/3 less the axis terms (2 × 5u) and the sum's two roundings (2 × 2u) is 14u below; the
        // constant is the nearest grid point of [2, 4) (spacing 4u) under 2·FRAC_PI_3 − 18u
        assert_eq!(CLEAR_BELOW, 2.0 * core::f64::consts::FRAC_PI_3 - 20.0 * U64);
        assert_eq!(2.0 * core::f64::consts::FRAC_PI_3 - CLEAR_BELOW, 20.0 * U64);
        assert_eq!(Q_SLACK, (12.0 + 1.0 + 1.0) * U64);
        assert_eq!(DOT_SLACK, (3.0 + 5.0 + 1.0 + 1.0 + 1.0 + 1.0) * U64);
    }

    #[test]
    fn a_pair_that_reaches_one_hundred_twenty_degrees_does_not_clear() {
        // axes θ apart and half-angles at or above (2π/3 − θ)/2 each: the cones reach 120°, so the
        // pair must not clear; with the half-angles 1e-6 rad short of that it clears (the bounds give
        // up less than that)
        let third = 2.0 * core::f64::consts::FRAC_PI_3;
        let mut rng = Rng(HELD_SEED);
        let mut checked = 0;
        for _ in 0..4000 {
            let theta = 60.0_f64.to_radians() + f64::from(rng.next()) * 55.0_f64.to_radians();
            let (s, c) = alice_det_math::sin_cos64(theta);
            let (a, b) = (DVec3::X, DVec3::new(c, s, 0.0));
            // the angle between the two axes as built (atan2 within 1 ulp; the pad covers it)
            let exact = alice_det_math::atan2_64(s, c);
            let h = f32_at_or_above((third - exact) / 2.0 + 1e-12);
            assert_ne!(cone_pair(a, h, b, h), Cover::Clear, "θ {theta}");
            let short = f32_at_or_below((third - exact) / 2.0 - 1e-6);
            assert_eq!(cone_pair(a, short, b, short), Cover::Clear, "θ {theta}");
            checked += 1;
        }
        assert_eq!(checked, 4000);
    }

    #[test]
    fn the_casts_round_towards_the_safe_side() {
        // each cast gives the nearest f32 on its side: at or past x, and one step back crosses x
        let mut rng = Rng(HELD_SEED ^ 0xCA57);
        let mut below_nearest = 0;
        for _ in 0..20_000 {
            let x = f64::from(rng.next()) * 4.0 - 2.0 + f64::from(rng.next()) * 1e-9;
            let up = f32_at_or_above(x);
            assert!(
                f64::from(up) >= x && f64::from(up.next_down()) < x,
                "up {x}"
            );
            let down = f32_at_or_below(x);
            assert!(
                f64::from(down) <= x && f64::from(down.next_up()) > x,
                "down {x}"
            );
            #[allow(clippy::cast_possible_truncation)]
            let nearest = x as f32;
            if f64::from(nearest) < x {
                below_nearest += 1;
            }
        }
        // a nearest cast lands on the wrong side for some of these inputs, so they test the direction
        assert!(
            below_nearest > 1000,
            "{below_nearest} inputs below their nearest f32"
        );
    }

    #[test]
    fn a_cone_through_the_box_holds_a_half_space() {
        // the difference box (−p..w, 2p..w/2, 0..e) has dlo·dhi = −pw + 2p·w/2 = 0 exactly: its ball
        // passes through the origin, so the cone is exactly a half-space (half-angle π/2). In f32 these
        // corners round the centre longer than the half-diagonal (ratio 1 − 2.4e-7) and asin then
        // falls 7e-4 rad short.
        let p = f32::from_bits(0x3eb9_b428);
        let w = f32::from_bits(0x3fdf_28ff);
        let e = f32::from_bits(0x3dde_b5c8);
        let (a, b) = (Vec3::new(-p, 2.0 * p, 0.0), Vec3::new(w, w / 2.0, e));
        let (_, half) = cover_cone(Vec3::ZERO, Vec3::ZERO, a, b);
        assert!(half >= core::f32::consts::FRAC_PI_2, "half-angle {half}");
    }

    #[test]
    fn a_box_in_a_forty_degree_v_does_not_clear() {
        // the walls of a 40° V, seen from its bisector, face at 140° (dot −0.77): a gap below 1
        let slab = SdfNode::box3d(30.0, 10.0, 20.0).translate(0.0, -5.0, 0.0);
        let half = 20.0_f32.to_radians();
        let half_space = |t: f32| {
            SdfNode::box3d(60.0, 60.0, 40.0)
                .translate(30.0, 0.0, 0.0)
                .rotate_euler(0.0, 0.0, t)
        };
        let s = slab.subtract(
            half_space(half)
                .intersection(half_space(core::f32::consts::PI - half))
                .translate(0.0, -3.0, 0.0),
        );
        // on the bisector 1.0 above the apex: 0.34 from both walls
        let lo = Vec3::new(-0.025, -2.025, -0.025);
        let c = cover(&s, lo, lo + Vec3::splat(0.05), 0.5);
        assert_eq!(c, Cover::Facing);
    }

    #[test]
    fn a_box_between_thin_plates_does_not_clear() {
        // plates 0.1 thick: the field inside them never goes below −0.05, yet they are material
        let plate = |y: f32| SdfNode::box3d(20.0, 0.1, 20.0).translate(0.0, y, 0.0);
        let s = plate(-0.3).union(plate(0.3));
        let lo = Vec3::splat(-0.05);
        let c = cover(&s, lo, lo + Vec3::splat(0.1), 0.5);
        assert_eq!(c, Cover::Facing);
    }
}
