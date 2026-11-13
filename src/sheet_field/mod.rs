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
//! `max(min|s| - scale - sqrt(3)/2 * h, 0.05)`, positive, so a sign jump of `s` there (two parts of one open
//! sheet seen from opposite sides) never reads as a crossing. A masked cell never holds a true hit. A sign
//! jump in an unmasked cell can still cross, at most `(sqrt(3) + sqrt(3)/2) * h + scale` from the sheet.
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

pub mod build;
#[cfg(feature = "gpu")]
pub mod gpu;

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
const HALF_DIAG: f32 = 0.866_025_4;
/// A bisected sign change with `|s|` above this is a sign jump of `s`, not a crossing of the sheet.
const JUMP_S: f32 = 0.02;

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
#[derive(Debug, Clone, PartialEq)]
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
    /// `rows_kept * BRICK_NODES` bytes, nodes in x-major order (`x * NODES^2 + y * NODES + z`).
    pub q: Vec<u8>,
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
    /// The interpolated side channel `s` (0 when no level answered).
    pub s: f32,
    /// The interpolated boundary channel `d_b` (0 when no level answered, or in a brick that stores no `d_b`:
    /// every node there is `<= 0`, so the point is inside the boundary).
    pub db: f32,
    /// The quantization step of `d_b` in the answering brick (0 when the brick stores no `d_b`).
    pub db_step: f32,
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

impl Level {
    fn node(&self, ch: &Channel, brick: usize, x: usize, y: usize, z: usize) -> Option<f32> {
        let row = ch.rows[brick];
        if row < 0 {
            return None;
        }
        let r = row as usize;
        let q = ch.q[r * BRICK_NODES + (x * NODES + y) * NODES + z];
        Some(f32::from(q) * ch.scale[r] + ch.offset[r])
    }
}

struct CellLookup {
    values: [f32; 4],
    nch: usize,
    umin: f32,
    q0: f32,
    q1: f32,
    h: f32,
}

impl Sheet {
    /// Whether a bisected sign change of this sheet's `g` at `p` is a crossing of the sheet: a level answers,
    /// unmasked, `|s| <= JUMP_S`, and `d_b <= step/2` (inside the boundary up to the rounding of the stored
    /// `d_b`; a brick that stores no `d_b` has every node `<= 0`, so its points are inside).
    fn crossing_inside(&self, p: Vec3, rule: ValidRule) -> bool {
        let c = self.sample(p, rule);
        c.inside && !c.masked && c.s.abs() <= JUMP_S && c.db <= 0.5 * c.db_step
    }

    fn lookup(&self, p: Vec3) -> Option<CellLookup> {
        for lv in &self.levels {
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
                values: [0.0; 4],
                nch: lv.channels.len().min(4),
                umin: f32::INFINITY,
                q0: 0.0,
                q1: 0.0,
                h: lv.h,
            };
            for (ci, ch) in lv.channels.iter().take(4).enumerate() {
                let mut acc = 0.0f32;
                let row = ch.rows[k];
                if row >= 0 {
                    for dx in 0..2 {
                        for dy in 0..2 {
                            for dz in 0..2 {
                                let ww = (if dx == 1 { t[0] } else { 1.0 - t[0] })
                                    * (if dy == 1 { t[1] } else { 1.0 - t[1] })
                                    * (if dz == 1 { t[2] } else { 1.0 - t[2] });
                                let v = lv
                                    .node(ch, k, c[0] + dx, c[1] + dy, c[2] + dz)
                                    .unwrap_or(0.0);
                                acc += ww * v;
                                if ci == 0 {
                                    out.umin = out.umin.min(v.abs());
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

    /// The sheet's value at `p` under `rule`.
    pub fn sample(&self, p: Vec3, rule: ValidRule) -> Sample {
        let Some(c) = self.lookup(p) else {
            let v = outside_lower_bound(p, self.lo, self.hi, self.hc);
            return Sample {
                f: v,
                g: v,
                inside: false,
                masked: false,
                uv: None,
                s: 0.0,
                db: 0.0,
                db_step: 0.0,
            };
        };
        let (s, db) = (c.values[0], c.values[1]);
        let mut f = (s * s + db.max(0.0) * db.max(0.0)).sqrt();
        let mut g = if s < 0.0 { -f } else { f };
        let mut masked = false;
        if rule == ValidRule::CellV1 && c.umin > HALF_DIAG * c.h + c.q0 {
            f = (c.umin - c.q0 - HALF_DIAG * c.h).max(0.05);
            g = f;
            masked = true;
        }
        let uv = if c.nch >= 4 {
            Some([c.values[2], c.values[3]])
        } else {
            None
        };
        Sample {
            f,
            g,
            inside: true,
            masked,
            uv,
            s,
            db,
            db_step: c.q1,
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
        }
    }

    /// The flattened BVH over the sheet boxes.
    pub fn index(&self) -> &FlatNodes {
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
                        let s = sh.sample(p, self.rule);
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
        self.march_with(origin, dir, tmax, steps, true)
    }
}

impl SheetField {
    /// The union value by a linear scan over all sheets (the reference the BVH-backed [`Self::eval`] equals).
    pub fn eval_linear(&self, p: Vec3) -> (f32, f32, Option<usize>) {
        let (mut fb, mut gb, mut kb) = (f32::INFINITY, f32::INFINITY, None);
        for (k, sh) in self.sheets.iter().enumerate() {
            let (v, w) = if box_distance(p, sh.lo, sh.hi) == 0.0 {
                let s = sh.sample(p, self.rule);
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

    /// [`Self::march`] with linear scans over all sheets (the reference the BVH-backed march equals).
    pub fn march_linear(&self, origin: Vec3, dir: Vec3, tmax: f32, steps: u32) -> MarchResult {
        self.march_with(origin, dir, tmax, steps, false)
    }

    fn march_with(&self, origin: Vec3, dir: Vec3, tmax: f32, steps: u32, bvh: bool) -> MarchResult {
        let k_n = self.sheets.len();
        let mut gprev: Vec<Option<f32>> = (0..k_n)
            .map(|k| {
                if self.sheet_in_box(k, origin) {
                    Some(self.sheets[k].sample(origin, self.rule).g)
                } else {
                    None
                }
            })
            .collect();
        let (mut t, mut tprev) = (0.0f32, 0.0f32);
        for _ in 0..steps {
            let p = origin + dir * t;
            let (dd, _, _) = if bvh {
                self.eval(p)
            } else {
                self.eval_linear(p)
            };
            let mut thit = f32::INFINITY;
            let mut khit = 0usize;
            let mut gcur: Vec<Option<f32>> = vec![None; k_n];
            let live: Vec<usize> = if bvh {
                self.sheets_containing(p)
            } else {
                (0..k_n).filter(|&k| self.sheet_in_box(k, p)).collect()
            };
            for k in live {
                let sh = &self.sheets[k];
                let gc = sh.sample(p, self.rule).g;
                gcur[k] = Some(gc);
                if t <= 0.0 {
                    continue;
                }
                let Some(gp) = gprev[k] else { continue };
                if (gc < 0.0) == (gp < 0.0) {
                    continue;
                }
                let (mut a, mut b, mut ga) = (tprev, t, gp);
                for _ in 0..30 {
                    let mid = 0.5 * (a + b);
                    let gm = sh.sample(origin + dir * mid, self.rule).g;
                    if (gm < 0.0) == (ga < 0.0) {
                        a = mid;
                        ga = gm;
                    } else {
                        b = mid;
                    }
                }
                let tm = 0.5 * (a + b);
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
            t += dd.max(0.01);
            if t >= tmax {
                return MarchResult::Miss;
            }
        }
        MarchResult::Undecided
    }

    /// The hit at `p` on sheet `k` (the sheet whose crossing it is): normal from the sheet's own signed field,
    /// UV from its channels.
    fn hit_on(&self, p: Vec3, t: f32, k: usize) -> Hit {
        let e = 0.05f32;
        let sh = &self.sheets[k];
        let g = |q: Vec3| sh.sample(q, self.rule).g;
        let n = Vec3::new(
            g(p + Vec3::X * e) - g(p - Vec3::X * e),
            g(p + Vec3::Y * e) - g(p - Vec3::Y * e),
            g(p + Vec3::Z * e) - g(p - Vec3::Z * e),
        );
        let normal = if n.length() > 1e-12 {
            n.normalize()
        } else {
            Vec3::Z
        };
        Hit {
            t,
            sheet: k,
            normal,
            uv: sh.sample(p, self.rule).uv,
        }
    }
}
