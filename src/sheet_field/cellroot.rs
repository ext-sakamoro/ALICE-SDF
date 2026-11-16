//! M2: the first root of a sheet's side channel `s` inside a floor step, found cell by cell (no step length can hide a
//! zero). Where the march's lower bound is below [`super::MIN_STEP`], the floor step may hold two zeros of one sheet
//! (a fold or a graze valley thinner than the step); a sign test at the two ends cannot see them. Here the step is
//! walked through the cells of the level that answers at each point; inside one cell `s` along the ray is the
//! trilinear interpolant of the 8 stored corners, a cubic in the ray parameter, and its first zero with a sign
//! change is taken exactly (up to the bisection width below).
//!
//! Rules (fixed before the code):
//! - Coordinates: inside a cell everything is in cell units (`q = (p - cell_lo) / h`, `q` in `[0, 1]^3`; the ray
//!   parameter `u` advances 1 per `h` of `t`), so the magnitudes are near 1 whatever `t` is.
//! - Root: the cubic's derivative roots split `[0, u_end]` into monotone pieces; the first piece whose ends have
//!   opposite signs is bisected [`BISECT_STEPS`] times (width `<= sqrt(3) 2^-24` cell units).
//! - Multiplicity: a root counts only when `s` has opposite signs at `u* - DELTA` and `u* + DELTA` (`DELTA` =
//!   2^-16 cell units; a tangent touch is no hit). When `|s|` at either probe is below the cubic's own rounding
//!   (`AMBIG_EPS` times the sum of `|coefficients|`) the root is ambiguous: it is not a hit and is counted.
//! - Masked cell (the validity rule refuses the surface there): no root is taken inside.
//! - Level / brick switch: each cell is looked up again at its entry, so a change of the answering level restarts
//!   the walk in the new level's grid; a sign change across a level boundary is not a root of either cubic and is
//!   never a hit (as the march's sign-jump rule).
//! - Bound: at most [`CELL_WALK_CAP`] cells per call; past it the answer is [`Walk::Exhausted`] (undecided).

use glam::Vec3;

use super::{outside_lower_bound, Level, Sheet, ValidRule, BRICK, HALF_DIAG};

/// Bisection steps on one monotone piece.
pub const BISECT_STEPS: u32 = 24;
/// The multiplicity probe offset, in cell units.
pub const DELTA: f64 = 1.0 / 65_536.0;
/// The relative rounding below which a probe value is ambiguous.
pub const AMBIG_EPS: f64 = 8.0 / 8_388_608.0;
/// Cells one call may visit.
pub const CELL_WALK_CAP: u32 = 64;

/// The answer of one cell walk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Walk {
    /// The first accepted root, at this `t`.
    Root(f32),
    /// No accepted root inside the interval; the second value counts the ambiguous roots met.
    Clear(u32),
    /// The walk stopped at [`CELL_WALK_CAP`] before the interval's end.
    Exhausted,
}

/// One cell of the answering level at a point.
struct Cell {
    lo: [f64; 3],
    h: f64,
    corners: [f64; 8],
    masked: bool,
}

/// `p(u) = c[0] + c[1] u + c[2] u^2 + c[3] u^3`.
fn eval(c: &[f64; 4], u: f64) -> f64 {
    ((c[3] * u + c[2]) * u + c[1]) * u + c[0]
}

fn mul(a: &[f64; 4], b: [f64; 2]) -> [f64; 4] {
    [a[0] * b[0], a[1] * b[0] + a[0] * b[1], a[2] * b[0] + a[1] * b[1], a[3] * b[0] + a[2] * b[1]]
}

/// The cubic of the trilinear interpolant along `q0 + e u`.
fn cubic(corners: &[f64; 8], q0: [f64; 3], e: [f64; 3]) -> [f64; 4] {
    let mut out = [0.0; 4];
    for (i, &v) in corners.iter().enumerate() {
        let (dx, dy, dz) = (i >> 2 & 1, i >> 1 & 1, i & 1);
        let w = |bit: usize, a: usize| if bit == 1 { [q0[a], e[a]] } else { [1.0 - q0[a], -e[a]] };
        let p = mul(&mul(&[1.0, 0.0, 0.0, 0.0], w(dx, 0)), w(dy, 1));
        let p = mul(&p, w(dz, 2));
        for k in 0..4 {
            out[k] += v * p[k];
        }
    }
    out
}

/// The first zero with a sign change of `c` in `(0, u_end]`; `Err(())` when the first one found is ambiguous.
fn first_root(c: &[f64; 4], u_end: f64) -> Option<Result<f64, ()>> {
    // derivative 3 c3 u^2 + 2 c2 u + c1: its roots in (0, u_end) split the interval into monotone pieces
    let (a, b, cc) = (3.0 * c[3], 2.0 * c[2], c[1]);
    let mut cuts: Vec<f64> = Vec::with_capacity(4);
    cuts.push(0.0);
    if a.abs() > 0.0 {
        let disc = b * b - 4.0 * a * cc;
        if disc >= 0.0 {
            let r = disc.sqrt();
            let mut xs = [(-b - r) / (2.0 * a), (-b + r) / (2.0 * a)];
            xs.sort_by(f64::total_cmp);
            for x in xs {
                if x > 0.0 && x < u_end {
                    cuts.push(x);
                }
            }
        }
    } else if b.abs() > 0.0 {
        let x = -cc / b;
        if x > 0.0 && x < u_end {
            cuts.push(x);
        }
    }
    cuts.push(u_end);
    let scale: f64 = c.iter().map(|x| x.abs()).sum();
    for w in cuts.windows(2) {
        let (mut l, mut r) = (w[0], w[1]);
        let (fl, fr) = (eval(c, l), eval(c, r));
        if (fl < 0.0) == (fr < 0.0) {
            continue;
        }
        for _ in 0..BISECT_STEPS {
            let m = 0.5 * (l + r);
            if (eval(c, m) < 0.0) == (fl < 0.0) {
                l = m;
            } else {
                r = m;
            }
        }
        let u = 0.5 * (l + r);
        let (sm, sp) = (eval(c, u - DELTA), eval(c, u + DELTA));
        if sm.abs() <= AMBIG_EPS * scale || sp.abs() <= AMBIG_EPS * scale {
            return Some(Err(()));
        }
        if (sm < 0.0) != (sp < 0.0) {
            return Some(Ok(u));
        }
    }
    None
}

impl Sheet {
    fn cell_at(&self, p: Vec3, rule: ValidRule) -> Option<Cell> {
        for lv in &self.levels {
            if let Some(c) = cell_in(lv, p, rule) {
                return Some(c);
            }
        }
        None
    }

    /// M2 (module docs): the first accepted root of this sheet in `[ta, tb]` along `o + d t`.
    pub(crate) fn walk_first_root(&self, o: Vec3, d: Vec3, ta: f32, tb: f32, rule: ValidRule) -> Walk {
        let (o64, d64) = (o.as_dvec3(), d.as_dvec3());
        let mut t = f64::from(ta);
        let tend = f64::from(tb);
        let mut ambiguous = 0u32;
        for _ in 0..CELL_WALK_CAP {
            if t >= tend {
                return Walk::Clear(ambiguous);
            }
            let p = o + d * t as f32;
            let Some(cell) = self.cell_at(p, rule) else {
                // no level answers: no surface nearer than the outside bound
                t += f64::from(outside_lower_bound(p, self.lo, self.hi, self.hc));
                continue;
            };
            let pw = o64 + d64 * t;
            let q0 = [(pw.x - cell.lo[0]) / cell.h, (pw.y - cell.lo[1]) / cell.h, (pw.z - cell.lo[2]) / cell.h];
            let e = [d64.x, d64.y, d64.z];
            // exit of the cell along the ray, in cell units
            let mut u_exit = f64::INFINITY;
            for a in 0..3 {
                if e[a] > 0.0 {
                    u_exit = u_exit.min((1.0 - q0[a]) / e[a]);
                } else if e[a] < 0.0 {
                    u_exit = u_exit.min(-q0[a] / e[a]);
                }
            }
            let u_exit = u_exit.max(0.0);
            let u_end = u_exit.min((tend - t) / cell.h);
            if !cell.masked {
                match first_root(&cubic(&cell.corners, q0, e), u_end) {
                    Some(Ok(u)) => {
                        let tr = t + u * cell.h;
                        if self.crossing_accepted(o + d * tr as f32, rule) {
                            return Walk::Root(tr as f32);
                        }
                    }
                    Some(Err(())) => ambiguous += 1,
                    None => {}
                }
            }
            // into the next cell: past the exit by a hair (2^-20 of a cell), so the next lookup is the next cell
            t += (u_exit + 1.0 / 1_048_576.0) * cell.h;
        }
        if t >= tend {
            Walk::Clear(ambiguous)
        } else {
            Walk::Exhausted
        }
    }
}

fn cell_in(lv: &Level, p: Vec3, rule: ValidRule) -> Option<Cell> {
    let u = (p - lv.lo) / lv.h;
    let b = BRICK as f32;
    let bi = [(u.x / b).floor(), (u.y / b).floor(), (u.z / b).floor()];
    if bi.iter().zip(lv.nb.iter()).any(|(&v, &n)| v < 0.0 || v >= n as f32) {
        return None;
    }
    let (bx, by, bz) = (bi[0] as usize, bi[1] as usize, bi[2] as usize);
    let k = lv.index[(bx * lv.nb[1] as usize + by) * lv.nb[2] as usize + bz];
    if k < 0 {
        return None;
    }
    let k = k as usize;
    let w = [u.x - bi[0] * b, u.y - bi[1] * b, u.z - bi[2] * b];
    let c = [
        (w[0].floor() as usize).min(BRICK - 1),
        (w[1].floor() as usize).min(BRICK - 1),
        (w[2].floor() as usize).min(BRICK - 1),
    ];
    let ch = &lv.channels[0];
    let row = ch.rows[k];
    let mut corners = [0.0f64; 8];
    let mut umin = f32::INFINITY;
    for (i, slot) in corners.iter_mut().enumerate() {
        let (dx, dy, dz) = (i >> 2 & 1, i >> 1 & 1, i & 1);
        let v = Level::node(ch, k, c[0] + dx, c[1] + dy, c[2] + dz).unwrap_or(0.0);
        umin = umin.min(v.abs());
        *slot = f64::from(v);
    }
    let q0 = if row >= 0 { ch.scale[row as usize] } else { 0.0 };
    let masked = rule == ValidRule::CellV1 && umin > HALF_DIAG * lv.h + q0;
    let h = f64::from(lv.h);
    let lo = [
        f64::from(lv.lo.x) + (bi[0] as f64 * f64::from(b) + c[0] as f64) * h,
        f64::from(lv.lo.y) + (bi[1] as f64 * f64::from(b) + c[1] as f64) * h,
        f64::from(lv.lo.z) + (bi[2] as f64 * f64::from(b) + c[2] as f64) * h,
    ];
    Some(Cell { lo, h, corners, masked })
}

#[cfg(test)]
mod tests {
    use super::{cubic, eval, first_root};

    /// The cubic equals the trilinear interpolant at sampled points (oracle: the trilinear formula itself).
    #[test]
    fn the_cubic_is_the_trilinear_interpolant_along_the_ray() {
        let corners = [0.3, -0.7, 1.1, 0.2, -0.4, 0.9, -1.3, 0.05];
        let (q0, e) = ([0.1, 0.25, 0.6], [0.48, -0.6, 0.64]);
        let c = cubic(&corners, q0, e);
        for k in 0..=10 {
            let u = 0.05 * f64::from(k);
            let q = [q0[0] + e[0] * u, q0[1] + e[1] * u, q0[2] + e[2] * u];
            let mut want = 0.0;
            for (i, &v) in corners.iter().enumerate() {
                let (dx, dy, dz) = (i >> 2 & 1, i >> 1 & 1, i & 1);
                let w = |bit: usize, x: f64| if bit == 1 { x } else { 1.0 - x };
                want += v * w(dx, q[0]) * w(dy, q[1]) * w(dz, q[2]);
            }
            assert!((eval(&c, u) - want).abs() < 1e-12, "u {u}: {} vs {want}", eval(&c, u));
        }
    }

    /// Two simple roots inside one interval (a valley crossing zero twice): the first is taken; a tangent touch
    /// (double root, no sign change) is not a root. Oracle: the roots of the polynomial written from them.
    #[test]
    fn the_first_simple_root_is_taken_and_a_tangent_touch_is_not() {
        // (u - 0.3)(u - 0.5) = u^2 - 0.8 u + 0.15
        let two = [0.15, -0.8, 1.0, 0.0];
        let Some(Ok(u)) = first_root(&two, 1.0) else { panic!("no root") };
        assert!((u - 0.3).abs() < 1e-6, "{u}");
        // (u - 0.4)^2: touches zero, no sign change
        let touch = [0.16, -0.8, 1.0, 0.0];
        assert!(!matches!(first_root(&touch, 1.0), Some(Ok(_))), "a tangent touch was taken as a root");
        // the valley's two roots both lie past the interval's end: none
        assert!(first_root(&two, 0.25).is_none());
    }
    /// A fold thinner than the floor step (M2's reason to exist), at the scale of the real fields (h 0.25 mm): a
    /// node plane at z = 0 stores s = 0.9 (-0.004) and its neighbours at z = +-0.25 store 0.9 (0.246), so the
    /// trilinear s along z is linear in each cell and its zeros are at z = +-0.004: two zeros 0.008 apart, under
    /// MIN_STEP 0.01. The factor 0.9 makes the march's bound conservative (as on real fields): its free-ball steps
    /// then stop short of the near face by a geometric remainder, and for start heights whose remainder leaves the
    /// first floor step starting within 0.002 of the face, the step's end is past the far face (both zeros inside one
    /// step, ends of one sign): with s = 0.9 d each free-ball step leaves a tenth of the remaining distance, so start
    /// heights z0 - 0.004 in [0.111, 0.2) put the first floor step inside (0.004, 0.006); the sweep 0.12..0.197
    /// covers them (checked by the wiring mutant: without the walk these rays miss).
    /// Vertical rays from above at 60 start heights (every phase of the floor steps against the fold). Oracle: the
    /// field's own definition, the zero of the line through the two stored (u8-decoded) node values at z = 0 and
    /// z = 0.25 (near +0.004; the quantisation moves it by up to half a step), read at the nodes themselves where
    /// the interpolation weight is 1; the cell walk hits it on every ray within 4 ulp(t) + 2^-20 h, and 0 compared
    /// is a failure.
    #[test]
    fn the_cell_walk_finds_the_near_face_of_a_fold_thinner_than_the_floor_step() {
        use crate::sheet_field::build::{level_from_fn, sheet};
        use crate::sheet_field::{MarchResult, RootMode, SheetField, ValidRule};
        use glam::Vec3;
        let lv = level_from_fn(
            Vec3::new(-1.0 + 0.013, -1.0 + 0.021, -2.0),
            0.25,
            [1, 1, 2],
            |_c, _e| true,
            |p| vec![0.9 * if p.z.abs() < 0.125 { -0.004 } else { p.z.abs() - 0.004 }, -1.0],
        )
        .unwrap();
        let sh = sheet(vec![lv]).unwrap();
        let (x, y) = (0.1f32, 0.2f32);
        // the stored values at the two nodes (x, y inside one cell column: the bilinear weights in x, y sum to 1 and
        // the field does not vary in x, y, so the value is the node plane's)
        let s0 = f64::from(sh.sample(Vec3::new(x, y, 0.0), ValidRule::CellV1).s);
        let s1 = f64::from(sh.sample(Vec3::new(x, y, 0.25), ValidRule::CellV1).s);
        assert!(s0 < 0.0 && s1 > 0.0, "{s0} {s1}");
        let z_zero = (0.25 * -s0 / (s1 - s0)) as f32;
        assert!((z_zero - 0.004).abs() < 0.005, "{z_zero}");
        let mut f = SheetField::new(vec![sh], ValidRule::CellV1);
        f.set_root_mode(RootMode::CellWalk);
        let mut compared = 0;
        for k in 0..60 {
            let z0 = 0.12 + 0.0013 * k as f32;
            let o = Vec3::new(x, y, z0);
            let want = z0 - z_zero;
            match f.march(o, -Vec3::Z, 1.0, 4096) {
                MarchResult::Hit(h) => {
                    let tol = 4.0 * f32::EPSILON * want + 0.25 / 1_048_576.0 + 2.0 * f32::EPSILON * z0;
                    assert!((h.t - want).abs() <= tol, "z0 {z0}: t {} want {want} (tol {tol})", h.t);
                    compared += 1;
                }
                other => panic!("z0 {z0}: {other:?} (the near face at t {want} was missed)"),
            }
        }
        assert_eq!(compared, 60, "0 compared would be a pass without a test");
    }
    /// The walk crosses cells: a ray starting 1e-4 before an x face of its cell, going +x and slightly down, meets the
    /// fold's near face only in the next cell. Oracle: the same stored-node zero as above, t = (z0 - z_zero) / |d_z|.
    /// A walk that skipped a cell (or stopped at the first) would answer Clear.
    #[test]
    fn the_cell_walk_finds_a_root_in_the_next_cell() {
        use crate::sheet_field::build::{level_from_fn, sheet};
        use crate::sheet_field::ValidRule;
        use glam::Vec3;
        let lo = Vec3::new(-1.0 + 0.013, -1.0 + 0.021, -2.0);
        let lv = level_from_fn(lo, 0.25, [1, 1, 2], |_c, _e| true, |p| {
            vec![0.9 * if p.z.abs() < 0.125 { -0.004 } else { p.z.abs() - 0.004 }, -1.0]
        })
        .unwrap();
        let sh = sheet(vec![lv]).unwrap();
        let (x, y) = (lo.x + 4.0 * 0.25 - 1e-4, 0.2f32);
        let s0 = f64::from(sh.sample(Vec3::new(x + 0.01, y, 0.0), ValidRule::CellV1).s);
        let s1 = f64::from(sh.sample(Vec3::new(x + 0.01, y, 0.25), ValidRule::CellV1).s);
        let z_zero = (0.25 * -s0 / (s1 - s0)) as f32;
        let z0 = z_zero + 0.0002;
        let d = Vec3::new(1.0, 0.0, -0.05).normalize();
        let want = (z0 - z_zero) / -d.z;
        assert!(want * d.x > 1e-4 * 1.5, "the root is not past the face: test geometry");
        match sh.walk_first_root(Vec3::new(x, y, z0), d, 0.0, 0.01, ValidRule::CellV1) {
            super::Walk::Root(t) => assert!((t - want).abs() < 1e-5, "t {t} want {want}"),
            other => panic!("{other:?}: the root in the next cell was missed"),
        }
    }
}
