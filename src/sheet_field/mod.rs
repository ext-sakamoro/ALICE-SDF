//! Sheet field: a union of open, zero-thickness sheets, each stored as sparse bricks of quantized channels.
//!
//! A sheet is an open surface (no inside). Its distance is carried by two channels sampled on a grid:
//! `s`, the distance along the side normal of the nearest feature (signed by that feature's normal), and
//! `d_b`, the boundary channel (positive beyond the sheet's rim, negative inside it). The unsigned distance is
//! `f = sqrt(s^2 + max(d_b, 0)^2)` and its signed companion is `g = sign(s) * f`. Optional channels carry the
//! surface UV of the nearest point, so a hit can be textured without the source mesh.
//!
//! Storage: each level of a sheet is a grid of bricks of `BRICK^3` cells (`(BRICK+1)^3` nodes). Active bricks
//! store, per channel, one byte per node with a per-brick scale and offset (`value = q * scale + offset`).
//! A channel may drop a brick (`row = -1`, value 0): the builder does this for `d_b` where every node is <= 0.
//!
//! Validity rule `CellV1`: a cell whose 8 corners all have `|s| > sqrt(3)/2 * h + scale` cannot contain the
//! surface (`|s|` never exceeds the true distance, and a surface point in the cell lies within half a cell
//! diagonal of its nearest corner). Such a cell is answered with the lower bound
//! `max(min|s| - scale - sqrt(3)/2 * h, 0.05)`, tightened per corner (below), positive, so a sign jump of `s` there (two parts of one open
//! sheet seen from opposite sides) never reads as a crossing. A masked cell never holds a true hit. A sign
//! jump in an unmasked cell can still cross, at most `(sqrt(3) + sqrt(3)/2) * h + scale` from the sheet.
//!
//! The masked cell's bound, per corner: for a corner `c` with stored `s_c`, every surface point `x` has
//! `|x - c| >= |s_c| - scale` (the same fact: `|s|` never exceeds the true distance, up to the storage step), so
//! `|x - p| >= |x - c| - |p - c| >= |s_c| - scale - |p - c|` for the point `p` in the cell. Every corner gives a
//! bound, so the distance from `p` to the sheet is at least `max_c (|s_c| - |p - c|) - scale`. The cell's value
//! is the larger of that and the bound above (which pairs the smallest `|s_c|` with the largest corner distance),
//! floored at 0.05 as before. Measured: on the heaviest ray of a 15-part body view, 259 of 272 steps of at most
//! 0.05 were set by a masked cell while the sheet's own `|s|` there had a median of 0.67.
//!
//! Union: `f = min` over sheets of (`f_k` inside sheet k's box, else `max(distance to the box, 3 h_k)`), a lower
//! bound on the distance to sheet k. A ray hit ([`SheetField::march`]) is a sign change of one sheet's own
//! `g_k` between two steps (the union's `g` jumps where the nearest sheet switches, so it is not used for
//! crossings), bisected, and accepted only where that sheet answers unmasked with `|s| <= 0.02` (a sign jump
//! of `s` is not a crossing) and `d_b <= step/2`: inside its boundary up to the rounding of the stored `d_b`
//! (half a quantization step). A crossing of the sheet's plane outside its rim is no hit, and nothing stops a
//! ray by proximity alone (a distance threshold would widen every rim and every sheet by that threshold). The
//! hit's owner is the sheet whose crossing it is. A ray that runs out of steps before the far distance is
//! reported as undecided, never as a hit.
//!
//! Step floor and budget: each step advances `max(f, MIN_STEP)`. Where `f >= MIN_STEP` the step stays inside the
//! ball free of every sheet (`f` is a lower bound on the distance), so nothing is skipped. Where `f < MIN_STEP` the
//! step is the floor and may overrun the free ball: a single crossing of an open sheet is still caught (the
//! sheet's own `s` has opposite signs at the two ends of the step, whatever its length); what the floor can
//! skip is two crossings of the same sheet within one step, i.e. a fold of one sheet whose two layers lie closer
//! than `MIN_STEP` along the ray (at normal incidence, the fold's thickness), entered and left without a net sign
//! change. A fold thicker than `MIN_STEP` cannot be skipped. [`decided_steps`] gives the budget that covers any
//! `tmax` at the floor step, so [`SheetField::march_decided`] always ends in a hit or a miss.

pub mod build;
pub mod cellroot;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod kink;
pub mod kink_build;
pub mod nearest;
pub mod stream;

use glam::Vec3;

use crate::mesh::bvh::BvhNode;
use crate::mesh::gpu_bvh::{flatten_nodes, FlatNodes, LEAF};
use crate::types::Aabb;

/// Cells per brick edge.
pub const BRICK: usize = 8;
/// Nodes per brick edge.
pub const NODES: usize = BRICK + 1;
/// Nodes per brick.
pub const BRICK_NODES: usize = NODES * NODES * NODES;
pub(crate) const HALF_DIAG: f32 = 0.866_025_4;
const SQRT3: f32 = 1.732_050_8;
/// The smallest step of the march (mm): a fold of one sheet thinner than this along the ray can be skipped.
pub const MIN_STEP: f32 = 0.01;

/// The step budget that covers `tmax` at the floor step.
///
/// Every step advances at least [`MIN_STEP`] less the f32 rounding of `t` (at most `tmax * f32::EPSILON`), so a
/// march with this many steps reaches `tmax` and is never undecided. Non-finite or non-positive `tmax` gives 1
/// (the march then ends at once).
#[must_use]
pub fn decided_steps(tmax: f32) -> u32 {
    if !tmax.is_finite() || tmax <= 0.0 {
        return 1;
    }
    let least = MIN_STEP - tmax * f32::EPSILON;
    assert!(
        least > 0.0,
        "tmax {tmax}: the f32 rounding of t reaches the step floor"
    );
    let n = (f64::from(tmax) / f64::from(least)).ceil() + 1.0;
    assert!(n < f64::from(u32::MAX), "tmax {tmax}: budget {n} steps");
    n as u32
}
/// A bisected sign change with `|s|` above this is a sign jump of `s`, not a crossing of the sheet.
pub(crate) const JUMP_S: f32 = 0.02;

/// The floor of a masked cell's lower bound (the field's length unit, mm).
///
/// It is a valid lower bound only where a masked cell's true distance is at least it: masking needs `min |s_c| >
/// HALF_DIAG h + q0`, so the true distance is over `HALF_DIAG h`, and the floor holds for `h >= MASKED_FLOOR /
/// HALF_DIAG` (about 0.0577). A level with a smaller cell is refused where fields are read
/// ([`stream::AshfIndex::read_sheets`]).
pub const MASKED_FLOOR: f32 = 0.05;

/// The most midpoints one dip search samples (see [`Sheet::dip`]); past it the ray is undecided. MEASUREMENT BUILD:
/// set from the deepest dip measured on the real fields before commit.
pub const DIP_NODE_CAP: u32 = 64;

/// The answer of one dip search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Dip {
    /// A sign change inside `[t_a, t_b]`, with `g` at `t_a`.
    Crossing(f32, f32, f32),
    /// No zero can lie inside the step (every open interval was closed by the bound).
    Clear,
    /// The search stopped at its cap with an interval still open: nothing is known about the step.
    Exhausted,
}

/// One step of a profiled march ([`SheetField::march_profile`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepRecord {
    /// Where the step started.
    pub t: f32,
    /// The step taken (`dd` floored at [`MIN_STEP`]).
    pub step: f32,
    /// The union value the step came from.
    pub dd: f32,
    /// The live sheets sampled at `t`.
    pub n_live: u32,
    /// The sheet whose value set `dd` (`None` outside every box).
    pub dd_sheet: Option<usize>,
    /// Whether that sheet's cell at the point was masked (its value was the cell lower bound, not `|s|`).
    pub dd_masked: bool,
    /// That sheet's interpolated `|s|` at the point (the offset to its layer; NaN outside its levels).
    pub dd_s: f32,
}

/// Which cells' sign is trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidRule {
    /// The provable cell rule (see the module docs).
    CellV1,
    /// No mask (kept for tests: it must let sign-jump crossings through).
    None,
}

impl ValidRule {
    /// The rule a stored form names; unknown names are refused (there is no default rule).
    pub fn from_name(name: &str) -> Result<Self, SheetFieldError> {
        match name {
            "cell-v1" => Ok(Self::CellV1),
            other => Err(SheetFieldError::UnknownRule(other.to_owned())),
        }
    }
}

/// Errors from building or loading a sheet field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SheetFieldError {
    /// The form names a validity rule this crate does not know.
    UnknownRule(String),
    /// A level or channel has inconsistent sizes.
    Shape(String),
}

impl std::fmt::Display for SheetFieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownRule(r) => write!(f, "unknown validity rule `{r}`"),
            Self::Shape(s) => write!(f, "inconsistent sheet field data: {s}"),
        }
    }
}

impl std::error::Error for SheetFieldError {}

/// One quantized channel of a level.
#[derive(Debug, Clone)]
pub struct Channel {
    /// Per active brick: its row in `q`, or -1 when the channel dropped the brick (value 0 there).
    pub rows: Vec<i32>,
    /// `rows_kept * nodes^3` bytes, nodes in x-major order (`x * nodes^2 + y * nodes + z`).
    pub q: Vec<u8>,
    /// Nodes per brick axis: [`NODES`] (one node per cell corner), or a coarser grid of an attribute channel
    /// (5, 3 or 2: every `BRICK / (nodes - 1)`-th corner) that is read by trilinear interpolation on its own
    /// grid; or [`BAND`]: only the nodes in the level's band mask ([`Level::band`]), packed row after row. The
    /// channels `s` and `d_b` (0 and 1) are always on [`NODES`].
    pub nodes: u32,
    /// For a [`BAND`] channel: the byte where each kept row starts in `q` (rows hold different node counts);
    /// empty otherwise.
    pub starts: Vec<u32>,
    /// Per kept row.
    pub scale: Vec<f32>,
    /// Per kept row.
    pub offset: Vec<f32>,
}

/// One grid level of a sheet.
#[derive(Debug, Clone)]
pub struct Level {
    /// Cell size.
    pub h: f32,
    /// Grid origin.
    pub lo: Vec3,
    /// Bricks per axis.
    pub nb: [u32; 3],
    /// Dense brick index (`x * nb1 * nb2 + y * nb2 + z`): active brick number, or -1.
    pub index: Vec<i32>,
    /// `s`, `d_b`, then optionally `u`, `v`.
    pub channels: Vec<Channel>,
    /// The band mask of the attribute channels stored as [`BAND`]: [`BAND_WORDS`] words per active brick (in
    /// active-brick order), bit `i` of the node index `i = (x * NODES + y) * NODES + z`; empty when no channel is
    /// a band.
    pub band: Vec<u32>,
}

/// One open sheet.
#[derive(Debug, Clone)]
pub struct Sheet {
    /// Levels in lookup order: the first level with an active brick at a point answers there.
    pub levels: Vec<Level>,
    /// Box of all levels (lower corner).
    pub lo: Vec3,
    /// Box of all levels (upper corner).
    pub hi: Vec3,
    /// The coarsest cell size (the outside lower bound uses `3 * hc`).
    pub hc: f32,
    /// The kink cells, where `d_b` is computed exactly from the sheet's triangles and rim segments ([`kink`]).
    pub kink: Option<std::sync::Arc<kink::KinkSheet>>,
}

/// What one sheet says at a point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// Unsigned distance `f` (after the validity rule).
    pub f: f32,
    /// Signed companion `g`.
    pub g: f32,
    /// Whether a level of the sheet answered (false: the outside lower bound).
    pub inside: bool,
    /// Whether the validity rule replaced the value.
    pub masked: bool,
    /// UV of the nearest surface point, when the sheet carries UV channels and a level answered.
    pub uv: Option<[f32; 2]>,
    /// The author normal of the nearest surface point (interpolated, not normalized), when the sheet carries the
    /// author-normal channels 4-6 and a level answered.
    pub attr_normal: Option<[f32; 3]>,
    /// The interpolated side channel `s` (0 when no level answered).
    pub s: f32,
    /// The interpolated boundary channel `d_b` (0 when no level answered, or in a brick that stores no `d_b`:
    /// every node there is `<= 0`, so the point is inside the boundary).
    pub db: f32,
    /// The quantization step of `d_b` in the answering brick (0 when the brick stores no `d_b`).
    pub db_step: f32,
    /// The bound on `|d s / d t|` and `|d d_b / d t|` along any unit direction in the answering cell,
    /// `sqrt(3) (1 + 2 max(scale_s, scale_db) / h)` (module docs, the dip search); 0 when no level answered.
    pub lip: f32,
}

impl Sample {
    /// How far the point is from an accepted crossing, as a quantity with slope at most [`Sample::lip`]: a crossing
    /// is accepted only where `s = 0` and `d_b <= d_b step / 2`, so `max(|s|, d_b - d_b step / 2)` (outside every
    /// level, the outside lower bound).
    #[must_use]
    pub fn reach(&self) -> f32 {
        if self.inside {
            self.s.abs().max(self.db - 0.5 * self.db_step)
        } else {
            self.f
        }
    }
}

/// The union of sheets.
#[derive(Debug, Clone)]
pub struct SheetField {
    /// The sheets.
    pub sheets: Vec<Sheet>,
    /// Validity rule.
    pub rule: ValidRule,
    /// A BVH over the sheet boxes (leaf entries are sheet indices), flattened ([`crate::mesh::gpu_bvh`]).
    index: FlatNodes,
    /// How a floor step looks for zeros its two ends cannot show (CPU march; see [`RootMode`]).
    root_mode: RootMode,
}

/// How the CPU march finds a zero inside a floor step whose ends have the same sign.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RootMode {
    /// M1: the bounded dip search ([`Sheet::dip`]).
    #[default]
    Dip,
    /// M2: the cell walk ([`cellroot`]): every floor step is walked cell by cell for the first root of each live sheet.
    CellWalk,
    /// The oracle's reference (not for rendering): every floor step is re-walked in 20 equal sub-steps (0.0005), each
    /// compared by sign and bisected. Elsewhere the march's own steps stand (they stay inside the free ball, so they
    /// cannot skip a zero when the bound is sound); so this is the fixed-step truth wherever the bound is sound.
    FixedFloor,
}

/// A ray hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// Distance along the (unit) ray.
    pub t: f32,
    /// The sheet that owns the hit (smallest `f_k` there).
    pub sheet: usize,
    /// Unit normal from the owning sheet's `g_k` (central differences, step 0.05).
    pub normal: Vec3,
    /// UV at the hit, when the owning sheet carries UV channels.
    pub uv: Option<[f32; 2]>,
    /// The author normal at the hit (normalized), when the owning sheet carries the 3 author-normal channels
    /// (channels 4-6). Its orientation is the author's; a renderer that shades like the source mesh flips it by
    /// the side of [`Hit::normal`] it sees, as it would flip a mesh normal by the triangle's face.
    pub attr_normal: Option<Vec3>,
}

/// The result of marching one ray.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarchResult {
    /// A hit.
    Hit(Hit),
    /// The ray reached the far distance.
    Miss,
    /// The ray ran out of steps before the far distance (neither a hit nor a miss).
    Undecided,
}

fn box_distance(p: Vec3, lo: Vec3, hi: Vec3) -> f32 {
    (lo - p).max(p - hi).max(Vec3::ZERO).length()
}

/// The one lower bound on the distance to a sheet for points with no active brick.
pub fn outside_lower_bound(p: Vec3, lo: Vec3, hi: Vec3, hc: f32) -> f32 {
    box_distance(p, lo, hi).max(3.0 * hc)
}

/// The node counts per brick axis an attribute channel may use besides [`NODES`]: 2h, 4h and 8h grids.
pub const COARSE_NODES: [u32; 3] = [5, 3, 2];

/// [`Channel::nodes`] of an attribute channel that keeps only the nodes in its level's band mask.
pub const BAND: u32 = 0;

/// Words of one brick's band mask (one bit per node of the 9^3).
pub const BAND_WORDS: usize = BRICK_NODES.div_ceil(32);

impl Level {
    fn node(ch: &Channel, brick: usize, x: usize, y: usize, z: usize) -> Option<f32> {
        let row = ch.rows[brick];
        if row < 0 {
            return None;
        }
        let r = row as usize;
        let n = ch.nodes as usize;
        let q = ch.q[r * n * n * n + (x * n + y) * n + z];
        Some(f32::from(q) * ch.scale[r] + ch.offset[r])
    }

    /// A band channel's value at node `i` of active brick `brick`: `NaN` when the node is not in the band (the
    /// attribute is not stored there; a reader must not shade from it).
    fn band_node(&self, ch: &Channel, brick: usize, i: usize) -> f32 {
        let row = ch.rows[brick];
        if row < 0 {
            return 0.0;
        }
        let m = &self.band[brick * BAND_WORDS..(brick + 1) * BAND_WORDS];
        let (w, b) = (i / 32, i % 32);
        if m[w] >> b & 1 == 0 {
            return f32::NAN;
        }
        let rank = m[..w].iter().map(|x| x.count_ones()).sum::<u32>()
            + (m[w] & ((1u32 << b) - 1)).count_ones();
        let r = row as usize;
        let q = ch.q[(ch.starts[r] + rank) as usize];
        f32::from(q) * ch.scale[r] + ch.offset[r]
    }

    /// A coarse channel at the brick-local point `w` (cell units, `0..=BRICK`): trilinear on its own grid.
    fn coarse_at(ch: &Channel, brick: usize, w: [f32; 3]) -> f32 {
        let n = ch.nodes as usize;
        let stride = (BRICK / (n - 1)) as f32;
        let mut c = [0usize; 3];
        let mut t = [0.0f32; 3];
        for a in 0..3 {
            let wc = w[a] / stride;
            c[a] = (wc.floor().max(0.0) as usize).min(n - 2);
            t[a] = wc - c[a] as f32;
        }
        let mut acc = 0.0f32;
        for dx in 0..2 {
            for dy in 0..2 {
                for dz in 0..2 {
                    let ww = (if dx == 1 { t[0] } else { 1.0 - t[0] })
                        * (if dy == 1 { t[1] } else { 1.0 - t[1] })
                        * (if dz == 1 { t[2] } else { 1.0 - t[2] });
                    acc +=
                        ww * Self::node(ch, brick, c[0] + dx, c[1] + dy, c[2] + dz).unwrap_or(0.0);
                }
            }
        }
        acc
    }
}

struct CellLookup {
    values: [f32; 8],
    nch: usize,
    umin: f32,
    /// `max` over the cell's corners of `|s_c| - |p - c|` (the per-corner part of the masked cell's bound).
    ucorner: f32,
    q0: f32,
    q1: f32,
    h: f32,
    /// Which cell answered: level, active brick, cell `x * 64 + y * 8 + z` (for the kink cells).
    at: (usize, usize, usize),
}

impl Sheet {
    /// Whether a bisected sign change of this sheet's `g` at `p` is a crossing of the sheet: a level answers,
    /// unmasked, `|s| <= JUMP_S`, and `d_b <= step/2` (inside the boundary up to the rounding of the stored
    /// `d_b`; a brick that stores no `d_b` has every node `<= 0`, so its points are inside).
    /// The earliest sign change of this sheet's `g` inside a march step whose two ends have the same sign (module
    /// docs, the dip search). `a` and `b` are `(t, g, |s|)` at the ends and `lip` the slope bound. An interval
    /// `[a, b]` can hold a zero only if `|s_a| + |s_b| <= lip (t_b - t_a)`; intervals that pass are halved, earliest
    /// half first, down to the march's bisection resolution (the step over 2^30). The search is bounded: it samples
    /// at most [`DIP_NODE_CAP`] midpoints and keeps at most 32 pending intervals; when either bound stops it with an
    /// interval still open it answers [`Dip::Exhausted`] (the ray is then undecided, never a miss), so no step's
    /// search is unbounded and no open interval is dropped silently. The second value is the number of midpoints
    /// sampled (counted in the ray's work).
    pub(crate) fn dip(
        &self,
        origin: Vec3,
        dir: Vec3,
        a: (f32, f32, f32),
        b: (f32, f32, f32),
        lip: f32,
        rule: ValidRule,
    ) -> (Dip, u32) {
        if lip <= 0.0 {
            return (Dip::Clear, 0);
        }
        let side = a.1 < 0.0;
        let floor = (b.0 - a.0) * (1.0 / 1_073_741_824.0);
        // depth-first, earliest first: at most one pending right half per level
        type End = (f32, f32, f32);
        let mut stack: [(End, End); 32] = [(a, b); 32];
        let mut n = 1usize;
        let mut nodes = 0u32;
        stack[0] = (a, b);
        while n > 0 {
            n -= 1;
            let (l, r) = stack[n];
            if l.2 + r.2 > lip * (r.0 - l.0) || r.0 - l.0 <= floor {
                continue;
            }
            if nodes >= DIP_NODE_CAP {
                return (Dip::Exhausted, nodes);
            }
            let tm = f32::midpoint(l.0, r.0);
            let c = self.sample_geom(origin + dir * tm, rule);
            nodes += 1;
            // where no level answers, the outside lower bound stands in for |s| (no crossing lies nearer than it)
            let m = (tm, c.g, c.reach());
            if (c.g < 0.0) != side {
                return (Dip::Crossing(l.0, tm, l.1), nodes);
            }
            if n + 2 > stack.len() {
                return (Dip::Exhausted, nodes);
            }
            stack[n] = (m, r);
            stack[n + 1] = (l, m);
            n += 2;
        }
        (Dip::Clear, nodes)
    }

    /// The exact `s` and unsigned distance at `p` when `p`'s answering cell is an s cell ([`kink::CELL_S`]).
    pub(crate) fn exact_s_here(&self, p: Vec3) -> Option<(f32, f32)> {
        let ks = self.kink.as_ref()?;
        let c = self.lookup(p, 0)?;
        let e = ks.exact_at(c.at.0, c.at.1, c.at.2, p)?;
        e.s.map(|s| (s, e.ud))
    }

    /// [`Self::crossing_inside`] for the cell walk (M2).
    pub(crate) fn crossing_accepted(&self, p: Vec3, rule: ValidRule) -> bool {
        self.crossing_inside(p, rule)
    }

    fn crossing_inside(&self, p: Vec3, rule: ValidRule) -> bool {
        let c = self.sample_geom(p, rule);
        c.inside && !c.masked && c.s.abs() <= JUMP_S && c.db <= 0.5 * c.db_step
    }

    /// The cell value at `p` from the first level that answers, interpolating the first `nmax` channels only
    /// (the march needs `s` and `d_b`; attributes are read once, at the hit).
    fn lookup(&self, p: Vec3, nmax: usize) -> Option<CellLookup> {
        for (li, lv) in self.levels.iter().enumerate() {
            let u = (p - lv.lo) / lv.h;
            let b = BRICK as f32;
            let bi = [(u.x / b).floor(), (u.y / b).floor(), (u.z / b).floor()];
            if bi
                .iter()
                .zip(lv.nb.iter())
                .any(|(&v, &n)| v < 0.0 || v >= n as f32)
            {
                continue;
            }
            let (bx, by, bz) = (bi[0] as usize, bi[1] as usize, bi[2] as usize);
            let k = lv.index[(bx * lv.nb[1] as usize + by) * lv.nb[2] as usize + bz];
            if k < 0 {
                continue;
            }
            let k = k as usize;
            let w = [u.x - bi[0] * b, u.y - bi[1] * b, u.z - bi[2] * b];
            let c = [
                (w[0].floor() as usize).min(BRICK - 1),
                (w[1].floor() as usize).min(BRICK - 1),
                (w[2].floor() as usize).min(BRICK - 1),
            ];
            let t = [w[0] - c[0] as f32, w[1] - c[1] as f32, w[2] - c[2] as f32];
            let mut out = CellLookup {
                values: [0.0; 8],
                nch: lv.channels.len().min(8).min(nmax),
                umin: f32::INFINITY,
                ucorner: f32::NEG_INFINITY,
                q0: 0.0,
                q1: 0.0,
                h: lv.h,
                at: (li, k, (c[0] * BRICK + c[1]) * BRICK + c[2]),
            };
            for (ci, ch) in lv.channels.iter().take(8.min(nmax)).enumerate() {
                let mut acc = 0.0f32;
                let row = ch.rows[k];
                if row >= 0 && ch.nodes == BAND {
                    for dx in 0..2 {
                        for dy in 0..2 {
                            for dz in 0..2 {
                                let ww = (if dx == 1 { t[0] } else { 1.0 - t[0] })
                                    * (if dy == 1 { t[1] } else { 1.0 - t[1] })
                                    * (if dz == 1 { t[2] } else { 1.0 - t[2] });
                                let i = ((c[0] + dx) * NODES + c[1] + dy) * NODES + c[2] + dz;
                                acc += ww * lv.band_node(ch, k, i);
                            }
                        }
                    }
                } else if row >= 0 && ch.nodes != NODES as u32 {
                    acc = Level::coarse_at(ch, k, w);
                } else if row >= 0 {
                    for dx in 0..2 {
                        for dy in 0..2 {
                            for dz in 0..2 {
                                let ww = (if dx == 1 { t[0] } else { 1.0 - t[0] })
                                    * (if dy == 1 { t[1] } else { 1.0 - t[1] })
                                    * (if dz == 1 { t[2] } else { 1.0 - t[2] });
                                let v = Level::node(ch, k, c[0] + dx, c[1] + dy, c[2] + dz)
                                    .unwrap_or(0.0);
                                acc += ww * v;
                                if ci == 0 {
                                    out.umin = out.umin.min(v.abs());
                                    let (ex, ey, ez) =
                                        (t[0] - dx as f32, t[1] - dy as f32, t[2] - dz as f32);
                                    let dist = lv.h * (ex * ex + ey * ey + ez * ez).sqrt();
                                    out.ucorner = out.ucorner.max(v.abs() - dist);
                                }
                            }
                        }
                    }
                    if ci == 0 {
                        out.q0 = ch.scale[row as usize];
                    } else if ci == 1 {
                        out.q1 = ch.scale[row as usize];
                    }
                }
                out.values[ci] = acc;
            }
            return Some(out);
        }
        None
    }

    /// The sheet's value at `p` under `rule`, with every stored channel (UV and author normal included).
    pub fn sample(&self, p: Vec3, rule: ValidRule) -> Sample {
        self.sample_n(p, rule, 8)
    }

    /// The geometric part of [`Sheet::sample`] (`f`, `g`, `s`, `d_b`; no attributes): what marching needs.
    pub(crate) fn sample_geom(&self, p: Vec3, rule: ValidRule) -> Sample {
        self.sample_n(p, rule, 2)
    }

    fn sample_n(&self, p: Vec3, rule: ValidRule, nmax: usize) -> Sample {
        let Some(c) = self.lookup(p, nmax) else {
            let v = outside_lower_bound(p, self.lo, self.hi, self.hc);
            return Sample {
                f: v,
                g: v,
                inside: false,
                masked: false,
                uv: None,
                attr_normal: None,
                s: 0.0,
                db: 0.0,
                db_step: 0.0,
                lip: 0.0,
            };
        };
        // in a flagged cell s and / or d_b are the exact values from the cell's triangles and rim segments, not the
        // interpolation ([`kink`])
        let (mut s, mut db) = (c.values[0], c.values[1]);
        let (mut ex_uv, mut ex_nrm) = (None, None);
        if let Some(e) = self
            .kink
            .as_ref()
            .filter(|_| nmax >= 2)
            .and_then(|ks| ks.exact_at(c.at.0, c.at.1, c.at.2, p))
        {
            s = e.s.unwrap_or(s);
            db = e.db.unwrap_or(db);
            // and the attributes of the exact closest point (its face's corners by the barycentric weights), so a hit
            // that the exact s moved reads the UV / author normal of where it is, not a blend across a seam
            (ex_uv, ex_nrm) = (e.uv, e.nrm);
        }
        let (f, g, masked) = if rule == ValidRule::CellV1 && c.umin > HALF_DIAG * c.h + c.q0 {
            let v = (c.umin - c.q0 - HALF_DIAG * c.h)
                .max(c.ucorner - c.q0)
                .max(MASKED_FLOOR);
            (v, v, true)
        } else {
            let f = (s * s + db.max(0.0) * db.max(0.0)).sqrt();
            (f, if s < 0.0 { -f } else { f }, false)
        };
        let uv = if c.nch >= 4 {
            Some(ex_uv.unwrap_or([c.values[2], c.values[3]]))
        } else {
            None
        };
        let attr_normal =
            (c.nch >= 7).then(|| ex_nrm.unwrap_or([c.values[4], c.values[5], c.values[6]]));
        Sample {
            f,
            g,
            inside: true,
            masked,
            uv,
            attr_normal,
            s,
            db,
            db_step: c.q1,
            lip: SQRT3 * (1.0 + 2.0 * c.q0.max(c.q1) / c.h),
        }
    }
}

impl SheetField {
    /// A field over `sheets` under `rule`; builds the BVH over the sheet boxes.
    pub fn new(sheets: Vec<Sheet>, rule: ValidRule) -> Self {
        let boxes: Vec<Aabb> = sheets.iter().map(|s| Aabb::new(s.lo, s.hi)).collect();
        let tree = BvhNode::build_over_boxes(&boxes, 4);
        Self {
            sheets,
            rule,
            index: flatten_nodes(tree.as_ref()),
            root_mode: RootMode::Dip,
        }
    }

    /// The step margin: [`kink::KINK_STEP_MARGIN`] when a sheet carries kink cells, else 0 (the march is then as before).
    #[must_use]
    pub fn step_margin(&self) -> f32 {
        if self.sheets.iter().any(|s| s.kink.is_some()) {
            kink::KINK_STEP_MARGIN
        } else {
            0.0
        }
    }

    /// How floor steps are searched (the GPU program is compiled for it).
    #[must_use]
    pub const fn root_mode(&self) -> RootMode {
        self.root_mode
    }

    /// Choose how floor steps are searched (CPU march, and the GPU program compiled from this field).
    pub const fn set_root_mode(&mut self, mode: RootMode) {
        self.root_mode = mode;
    }

    /// The flattened BVH over the sheet boxes.
    pub const fn index(&self) -> &FlatNodes {
        &self.index
    }

    fn node(&self, i: u32) -> (Vec3, Vec3, u32, u32) {
        let n = &self.index.nodes[i as usize * 8..i as usize * 8 + 8];
        let f = |w: u32| f32::from_bits(w);
        (
            Vec3::new(f(n[0]), f(n[1]), f(n[2])),
            Vec3::new(f(n[4]), f(n[5]), f(n[6])),
            n[3],
            n[7],
        )
    }

    /// The sheets whose box holds `p`, ascending (a parent-link walk of the BVH, no stack).
    pub fn sheets_containing(&self, p: Vec3) -> Vec<usize> {
        let mut out = Vec::new();
        if self.index.nodes.is_empty() {
            return out;
        }
        let inside = |lo: Vec3, hi: Vec3| p.cmpge(lo).all() && p.cmple(hi).all();
        let (rlo, rhi, _, _) = self.node(0);
        if !inside(rlo, rhi) {
            return out;
        }
        const NONE: u32 = u32::MAX;
        let (mut at, mut prev) = (0u32, NONE);
        loop {
            let (_, _, a, b) = self.node(at);
            if a & LEAF != 0 {
                let first = (a & !LEAF) as usize;
                for &k in &self.index.leaf_entries[first..first + b as usize] {
                    if self.sheet_in_box(k as usize, p) {
                        out.push(k as usize);
                    }
                }
            } else {
                let (llo, lhi, _, _) = self.node(a);
                let (blo, bhi, _, _) = self.node(b);
                let (in_l, in_r) = (inside(llo, lhi), inside(blo, bhi));
                if prev == NONE || prev == self.index.parents[at as usize] {
                    if in_l {
                        prev = at;
                        at = a;
                        continue;
                    }
                    if in_r {
                        prev = at;
                        at = b;
                        continue;
                    }
                } else if prev == a && in_r {
                    prev = at;
                    at = b;
                    continue;
                }
            }
            if at == 0 {
                break;
            }
            prev = at;
            at = self.index.parents[at as usize];
        }
        out.sort_unstable();
        out
    }

    /// The union value `(f, g, owning sheet)` at `p`: min over sheets of (`f_k` inside sheet k's box, else the
    /// outside lower bound), lowest index on ties. A BVH walk, nearer child first, that skips a node whose box
    /// is farther than the best value so far (every sheet in it is at least that far); equals
    /// [`Self::eval_linear`] bit for bit.
    pub fn eval(&self, p: Vec3) -> (f32, f32, Option<usize>) {
        let (mut fb, mut gb, mut kb) = (f32::INFINITY, f32::INFINITY, None::<usize>);
        if self.index.nodes.is_empty() {
            return (fb, gb, kb);
        }
        const NONE: u32 = u32::MAX;
        let (mut at, mut prev) = (0u32, NONE);
        loop {
            let (_, _, a, b) = self.node(at);
            if a & LEAF != 0 {
                let first = (a & !LEAF) as usize;
                for &k in &self.index.leaf_entries[first..first + b as usize] {
                    let k = k as usize;
                    let sh = &self.sheets[k];
                    let (v, w) = if box_distance(p, sh.lo, sh.hi) == 0.0 {
                        let s = sh.sample_geom(p, self.rule);
                        (s.f, s.g)
                    } else {
                        let v = outside_lower_bound(p, sh.lo, sh.hi, sh.hc);
                        (v, v)
                    };
                    if v < fb || (v == fb && kb.is_some_and(|kk| k < kk)) {
                        fb = v;
                        gb = w;
                        kb = Some(k);
                    }
                }
            } else {
                let (llo, lhi, _, _) = self.node(a);
                let (rlo, rhi, _, _) = self.node(b);
                let (dl, dr) = (box_distance(p, llo, lhi), box_distance(p, rlo, rhi));
                let (first, df, second, ds) = if dr < dl {
                    (b, dr, a, dl)
                } else {
                    (a, dl, b, dr)
                };
                if prev == NONE || prev == self.index.parents[at as usize] {
                    if df <= fb {
                        prev = at;
                        at = first;
                        continue;
                    }
                    if ds <= fb {
                        prev = at;
                        at = second;
                        continue;
                    }
                } else if prev == first && ds <= fb {
                    prev = at;
                    at = second;
                    continue;
                }
            }
            if at == 0 {
                break;
            }
            prev = at;
            at = self.index.parents[at as usize];
        }
        (fb, gb, kb)
    }

    /// March one ray (`dir` unit length) up to `tmax` with at most `steps` steps (BVH-backed; equals
    /// [`Self::march_linear`]).
    pub fn march(&self, origin: Vec3, dir: Vec3, tmax: f32, steps: u32) -> MarchResult {
        self.march_with(origin, dir, tmax, steps, true, None)
    }

    /// [`Self::march_decided`] recording each step: the march itself (same code path), plus per step the position
    /// `t`, the step taken and the live sheets sampled. For profiling heavy rays.
    #[must_use]
    pub fn march_profile(
        &self,
        origin: Vec3,
        dir: Vec3,
        tmax: f32,
    ) -> (MarchResult, Vec<StepRecord>) {
        let mut rec = Vec::new();
        let m = self.march_with(origin, dir, tmax, decided_steps(tmax), true, Some(&mut rec));
        (m, rec)
    }

    /// [`Self::march`] with the budget from [`decided_steps`]: the result is a hit or a miss, never undecided.
    #[must_use]
    pub fn march_decided(&self, origin: Vec3, dir: Vec3, tmax: f32) -> MarchResult {
        self.march(origin, dir, tmax, decided_steps(tmax))
    }
}

impl SheetField {
    /// The union value by a linear scan over all sheets (the reference the BVH-backed [`Self::eval`] equals).
    pub fn eval_linear(&self, p: Vec3) -> (f32, f32, Option<usize>) {
        let (mut fb, mut gb, mut kb) = (f32::INFINITY, f32::INFINITY, None);
        for (k, sh) in self.sheets.iter().enumerate() {
            let (v, w) = if box_distance(p, sh.lo, sh.hi) == 0.0 {
                let s = sh.sample_geom(p, self.rule);
                (s.f, s.g)
            } else {
                let v = outside_lower_bound(p, sh.lo, sh.hi, sh.hc);
                (v, v)
            };
            if v < fb {
                fb = v;
                gb = w;
                kb = Some(k);
            }
        }
        (fb, gb, kb)
    }

    fn sheet_in_box(&self, k: usize, p: Vec3) -> bool {
        let sh = &self.sheets[k];
        p.cmpge(sh.lo).all() && p.cmple(sh.hi).all()
    }

    /// The reference march for the oracles (not for rendering): a fixed step `h` from `t = 0`, each step's sheets
    /// compared by the sign of `g` at its two ends and bisected as [`Self::march`]; no bound and no floor. It is the
    /// truth wherever no sheet has two zeros closer than `h` along the ray. Rays that reach `tmax` are a miss.
    #[must_use]
    pub fn march_fixed_step(&self, origin: Vec3, dir: Vec3, tmax: f32, h: f32) -> MarchResult {
        let k_n = self.sheets.len();
        let mut gprev: Vec<Option<f32>> = (0..k_n)
            .map(|k| {
                self.sheet_in_box(k, origin)
                    .then(|| self.sheets[k].sample_geom(origin, self.rule).g)
            })
            .collect();
        let mut i = 1u64;
        let mut tprev = 0.0f64;
        loop {
            let t = (i as f64 * f64::from(h)).min(f64::from(tmax));
            let p = origin + dir * t as f32;
            let mut thit = f32::INFINITY;
            let mut khit = 0usize;
            let mut gcur: Vec<Option<f32>> = vec![None; k_n];
            for k in (0..k_n).filter(|&k| self.sheet_in_box(k, p)) {
                let sh = &self.sheets[k];
                let gc = sh.sample_geom(p, self.rule).g;
                gcur[k] = Some(gc);
                let Some(gp) = gprev[k] else { continue };
                if (gc < 0.0) == (gp < 0.0) {
                    continue;
                }
                let (mut a, mut b, mut ga) = (tprev as f32, t as f32, gp);
                for _ in 0..30 {
                    let mid = f32::midpoint(a, b);
                    let gm = sh.sample_geom(origin + dir * mid, self.rule).g;
                    if (gm < 0.0) == (ga < 0.0) {
                        a = mid;
                        ga = gm;
                    } else {
                        b = mid;
                    }
                }
                let tm = f32::midpoint(a, b);
                if sh.crossing_inside(origin + dir * tm, self.rule) && tm < thit {
                    thit = tm;
                    khit = k;
                }
            }
            if thit.is_finite() {
                return MarchResult::Hit(self.hit_on(origin + dir * thit, thit, khit));
            }
            if t >= f64::from(tmax) {
                return MarchResult::Miss;
            }
            tprev = t;
            gprev = gcur;
            i += 1;
        }
    }

    /// [`Self::march`] with linear scans over all sheets (the reference the BVH-backed march equals).
    pub fn march_linear(&self, origin: Vec3, dir: Vec3, tmax: f32, steps: u32) -> MarchResult {
        self.march_with(origin, dir, tmax, steps, false, None)
    }

    fn march_with(
        &self,
        origin: Vec3,
        dir: Vec3,
        tmax: f32,
        steps: u32,
        bvh: bool,
        mut rec: Option<&mut Vec<StepRecord>>,
    ) -> MarchResult {
        let k_n = self.sheets.len();
        // per live sheet: (g, |s|, the slope bound) at the previous step
        let mut gprev: Vec<Option<(f32, f32, f32)>> = (0..k_n)
            .map(|k| {
                if self.sheet_in_box(k, origin) {
                    let c = self.sheets[k].sample_geom(origin, self.rule);
                    Some((c.g, c.reach(), c.lip))
                } else {
                    None
                }
            })
            .collect();
        let (mut t, mut tprev) = (0.0f32, 0.0f32);
        let margin = self.step_margin();
        // whether the step that ended at `t` was a floor step (its start's bound was below MIN_STEP)
        let mut floor_step = false;
        for _ in 0..steps {
            let p = origin + dir * t;
            let (dd, _, dd_k) = if bvh {
                self.eval(p)
            } else {
                self.eval_linear(p)
            };
            let mut thit = f32::INFINITY;
            let mut khit = 0usize;
            let mut gcur: Vec<Option<(f32, f32, f32)>> = vec![None; k_n];
            let live: Vec<usize> = if bvh {
                self.sheets_containing(p)
            } else {
                (0..k_n).filter(|&k| self.sheet_in_box(k, p)).collect()
            };
            let n_live = live.len();
            for k in live {
                let sh = &self.sheets[k];
                let cs = sh.sample_geom(p, self.rule);
                let gc = cs.g;
                gcur[k] = Some((gc, cs.reach(), cs.lip));
                if t <= 0.0 {
                    continue;
                }
                let Some((gp, sp, lp)) = gprev[k] else {
                    continue;
                };
                if self.root_mode == RootMode::FixedFloor && floor_step {
                    let n = 20u32;
                    let mut gq = gp;
                    let mut tq = tprev;
                    for j in 1..=n {
                        let tj = if j == n {
                            t
                        } else {
                            tprev + (t - tprev) * j as f32 / n as f32
                        };
                        let gj = sh.sample_geom(origin + dir * tj, self.rule).g;
                        if (gj < 0.0) != (gq < 0.0) {
                            let (mut a, mut b, mut ga) = (tq, tj, gq);
                            for _ in 0..30 {
                                let mid = f32::midpoint(a, b);
                                let gm = sh.sample_geom(origin + dir * mid, self.rule).g;
                                if (gm < 0.0) == (ga < 0.0) {
                                    a = mid;
                                    ga = gm;
                                } else {
                                    b = mid;
                                }
                            }
                            let tm = f32::midpoint(a, b);
                            if sh.crossing_inside(origin + dir * tm, self.rule) {
                                if tm < thit {
                                    thit = tm;
                                    khit = k;
                                }
                                break;
                            }
                        }
                        gq = gj;
                        tq = tj;
                    }
                    continue;
                }
                if self.root_mode == RootMode::CellWalk && floor_step {
                    // M2: a floor step may hold zeros its ends cannot show; walk it cell by cell
                    match sh.walk_first_root(origin, dir, tprev, t, self.rule) {
                        cellroot::Walk::Root(tr) => {
                            if tr < thit {
                                thit = tr;
                                khit = k;
                            }
                        }
                        cellroot::Walk::Clear(_) => {
                            // a crossing exists when the step's ends have opposite signs, whatever the walk
                            // located: the walk only places roots, and a root it rounds past the step's end (the
                            // f32 walk on the GPU when the end lands an ulp after the surface) must not turn the
                            // sign change into a clear step
                            if (gc < 0.0) != (gp < 0.0) {
                                let (mut a, mut b, mut ga) = (tprev, t, gp);
                                for _ in 0..30 {
                                    let mid = f32::midpoint(a, b);
                                    let gm = sh.sample_geom(origin + dir * mid, self.rule).g;
                                    if (gm < 0.0) == (ga < 0.0) {
                                        a = mid;
                                        ga = gm;
                                    } else {
                                        b = mid;
                                    }
                                }
                                let tm = f32::midpoint(a, b);
                                if tm < thit && sh.crossing_inside(origin + dir * tm, self.rule) {
                                    thit = tm;
                                    khit = k;
                                }
                            }
                        }
                        cellroot::Walk::Exhausted => return MarchResult::Undecided,
                    }
                    continue;
                }
                let _ = (sp, lp);
                let (mut a, mut b, mut ga) = (tprev, t, gp);
                if (gc < 0.0) == (gp < 0.0) && self.root_mode != RootMode::Dip {
                    // not a floor step: the step stayed inside the free ball, so ends of one sign hold no zero
                    continue;
                }
                if (gc < 0.0) == (gp < 0.0) {
                    // the same side at both ends: a crossing inside the step needs |s| to go to 0 and back
                    // (both ends must be answered by a level: only there is |s| a measure of the way to a crossing)
                    let lip = if lp > 0.0 && cs.lip > 0.0 {
                        lp.max(cs.lip)
                    } else {
                        0.0
                    };
                    match sh.dip(
                        origin,
                        dir,
                        (tprev, gp, sp),
                        (t, gc, cs.reach()),
                        lip,
                        self.rule,
                    ) {
                        (Dip::Crossing(da, db, dga), _) => (a, b, ga) = (da, db, dga),
                        (Dip::Clear, _) => continue,
                        (Dip::Exhausted, _) => return MarchResult::Undecided,
                    }
                }
                for _ in 0..30 {
                    let mid = f32::midpoint(a, b);
                    let gm = sh.sample_geom(origin + dir * mid, self.rule).g;
                    if (gm < 0.0) == (ga < 0.0) {
                        a = mid;
                        ga = gm;
                    } else {
                        b = mid;
                    }
                }
                let tm = f32::midpoint(a, b);
                if sh.crossing_inside(origin + dir * tm, self.rule) && tm < thit {
                    thit = tm;
                    khit = k;
                }
            }
            if thit.is_finite() {
                return MarchResult::Hit(self.hit_on(origin + dir * thit, thit, khit));
            }
            tprev = t;
            gprev = gcur;
            if let Some(r) = rec.as_deref_mut() {
                let (dd_masked, dd_s) = dd_k.map_or((false, f32::NAN), |k| {
                    let c = self.sheets[k].sample_geom(p, self.rule);
                    (c.masked, if c.inside { c.s.abs() } else { f32::NAN })
                });
                r.push(StepRecord {
                    t,
                    step: (dd - margin).max(MIN_STEP),
                    dd,
                    n_live: n_live as u32,
                    dd_sheet: dd_k,
                    dd_masked,
                    dd_s,
                });
            }
            let dd_step = dd - margin;
            floor_step = dd_step < MIN_STEP;
            t += dd_step.max(MIN_STEP);
            if t >= tmax {
                return MarchResult::Miss;
            }
        }
        MarchResult::Undecided
    }

    /// The hit at `p` on sheet `k` (the sheet whose crossing it is): normal = the gradient of the sheet's own
    /// side channel `s` (central differences, 0.05 mm); the distance `g` would bend it toward the rim where
    /// `d_b > 0` reaches the stencil. UV from its channels.
    fn hit_on(&self, p: Vec3, t: f32, k: usize) -> Hit {
        let e = 0.05f32;
        let sh = &self.sheets[k];
        let s = |q: Vec3| sh.sample_geom(q, self.rule).s;
        let n = Vec3::new(
            s(p + Vec3::X * e) - s(p - Vec3::X * e),
            s(p + Vec3::Y * e) - s(p - Vec3::Y * e),
            s(p + Vec3::Z * e) - s(p - Vec3::Z * e),
        );
        let at = sh.sample(p, self.rule);
        let normal = if n.length() > 1e-12 {
            n.normalize()
        } else {
            Vec3::Z
        };
        Hit {
            t,
            sheet: k,
            normal,
            uv: at.uv,
            attr_normal: at
                .attr_normal
                .map(Vec3::from_array)
                .filter(|n| n.length() > 1e-12)
                .map(Vec3::normalize),
        }
    }
}

#[cfg(test)]
mod dip_tests {
    use super::build::{level_from_fn, sheet};
    use super::{Dip, Sheet, ValidRule, DIP_NODE_CAP};
    use glam::Vec3;

    /// A slab |z| < 0.002 (s = |z| - 0.002, slope 1 along z) on cells of 0.0005 (the stored kink is sharp), tall
    /// enough (z in about -0.032..0.032) for steps long enough to sit between the bound and half of it.
    fn slab() -> Sheet {
        let lv = level_from_fn(
            Vec3::new(-0.004 + 0.00013, -0.004 + 0.00021, -0.032 + 0.00007),
            0.0005,
            [2, 2, 16],
            |_c, _e| true,
            |p| vec![p.z.abs() - 0.002, -1.0],
        )
        .unwrap();
        sheet(vec![lv]).unwrap()
    }

    /// The dip search's threshold, both sides (module docs): along a vertical ray through the slab the stored
    /// `|s|` has slope 1, so a step whose ends both sit `a` above the slab holds two zeros exactly when its length
    /// exceeds `2 a + 0.004`. Oracle: that geometry, not the search.
    /// - ends just close enough (`|s_a| + |s_b|` between `L dt / 2` and `L dt`, zeros inside): found with `L`, and
    ///   not with `L / 2` (the threshold is needed at its full value);
    /// - ends just too far for a zero (`|s_a| + |s_b| > dt - 0.004` but `<= L dt`, so searched): nothing found;
    /// - ends over `L dt`: nothing searched, nothing found.
    #[test]
    fn the_dip_search_finds_zeros_exactly_inside_its_bound() {
        let sh = slab();
        let rule = ValidRule::CellV1;
        let o = Vec3::new(0.0003, 0.0002, 0.0);
        let d = -Vec3::Z;
        let at = |t: f32| {
            let c = sh.sample_geom(o + d * t, rule);
            (t, c.g, c.s.abs(), c.lip)
        };
        // a step from z = +z0 to z = -z0 (t from -z0 to +z0 along -z): ends at |s| = z0 - 0.002
        let step = |z0: f32| {
            let (a, b) = (at(-z0), at(z0));
            (a, b, a.3.max(b.3))
        };
        // zeros inside, |s_a| + |s_b| = 2 (z0 - 0.002) = dt - 0.004, between L dt / 2 and L dt (L ~ 1.84): the
        // slope along the ray is 1, so only a long step puts the sum over half the bound
        let z0 = 0.03;
        let (a, b, lip) = step(z0);
        let sum = a.2 + b.2;
        assert!(
            sum < lip * (b.0 - a.0) && sum > 0.5 * lip * (b.0 - a.0),
            "sum {sum} lip {lip} dt {}",
            b.0 - a.0
        );
        assert!((a.1 < 0.0) == (b.1 < 0.0), "the ends are on opposite sides");
        let found = sh.dip(o, d, (a.0, a.1, a.2), (b.0, b.1, b.2), lip, rule);
        assert!(
            matches!(found.0, Dip::Crossing(..)),
            "a slab {} wide inside a step of {} was not found",
            0.004,
            b.0 - a.0
        );
        assert!(
            sh.dip(o, d, (a.0, a.1, a.2), (b.0, b.1, b.2), 0.5 * lip, rule)
                .0
                == Dip::Clear,
            "half the bound still found it: this case does not test the threshold"
        );
        // no zero can be inside: ends farther than the slab allows, yet within L dt, so the search runs and must
        // return nothing (it never reports a sign change that is not there)
        for &(za, zb) in &[(0.0035f32, 0.0035f32), (0.003, 0.0052), (0.0041, 0.0029)] {
            // a step from z = +za down to z = +zb on the same side never meets the slab
            let (a, b) = (at(-za), at(-zb));
            assert!(
                sh.dip(o, d, (a.0, a.1, a.2), (b.0, b.1, b.2), a.3.max(b.3), rule)
                    .0
                    == Dip::Clear,
                "{za} {zb}"
            );
        }
        // over the bound: not searched
        let (a, b) = (at(-0.0039), at(-0.0021));
        let lip = a.3.max(b.3);
        assert!(
            a.2 + b.2 > lip * (b.0 - a.0)
                || sh.dip(o, d, (a.0, a.1, a.2), (b.0, b.1, b.2), lip, rule).0 == Dip::Clear
        );
    }
    /// The bound on the search (M1): a step the bound cannot close (a slope bound far above the true slope keeps
    /// every half open) ends in `Exhausted` after exactly `DIP_NODE_CAP` midpoints, never in `Clear` (a cap that
    /// answered "no zero" would turn an unfinished search into a miss), and a step it can close in fewer nodes
    /// still answers as before. Oracle: the slab's geometry (no zero between two points both above it).
    #[test]
    fn the_dip_search_is_bounded_and_never_reports_an_unfinished_search_as_clear() {
        let sh = slab();
        let rule = ValidRule::CellV1;
        let o = Vec3::new(0.0003, 0.0002, 0.0);
        let d = -Vec3::Z;
        let at = |t: f32| {
            let c = sh.sample_geom(o + d * t, rule);
            (t, c.g, c.s.abs())
        };
        // both ends above the slab (no zero inside), with a slope bound 1e6 times the true one: nothing closes
        let (a, b) = (at(-0.03), at(-0.0035));
        let (r, nodes) = sh.dip(o, d, a, b, 1e6, rule);
        assert_eq!(
            r,
            Dip::Exhausted,
            "an unfinished search must not answer {r:?}"
        );
        assert_eq!(
            nodes, DIP_NODE_CAP,
            "the cap was not the bound that stopped it"
        );
        // the same ends with the true slope bound: closed (Clear) within the cap
        let (r, nodes) = sh.dip(o, d, a, b, 1.84, rule);
        assert_eq!(r, Dip::Clear);
        assert!(nodes < DIP_NODE_CAP, "{nodes}");
    }
}
