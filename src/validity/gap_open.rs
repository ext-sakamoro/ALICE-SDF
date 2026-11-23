//! Open narrow air versus enclosed narrow air, read in the print layer.
//!
//! A gap below a printer's limit matters when the air is *enclosed* in the layer being printed: a hole,
//! or a slot closed at both ends. There the perimeters laid around the air close it. A neck between
//! two separate walls (a post standing next to a case wall) is different: within a short distance in
//! the same layer the air widens to the limit, and the two walls are printed as separate islands. This
//! module tells the two apart for one point of narrow air.
//!
//! ## MODEL
//!
//! * Input: the field `f` (an SDF whose distance is exact or underestimated: `|∇f| ≤ 1`), a point of
//!   narrow air, the printer's limit `L` and the build axis `a` (the direction layers stack in).
//! * The layer is the plane through the point perpendicular to `a`. It is sampled on a square grid of
//!   pitch `h = L / OPEN_PITCH_PER_LIMIT`, centred on the point, inside the disc of radius
//!   `R = OPEN_REACH_LIMITS · L`.
//! * Two neighbouring samples `p`, `q` (distance `h`) are joined when the segment between them is air,
//!   shown by either of two sufficient conditions: (i) `f(p) + f(q) > h` (with `|∇f| ≤ 1` every point
//!   `x` of the segment has `f(x) ≥ max(f(p) − |x−p|, f(q) − |x−q|) > 0`), or (ii) the interval bound
//!   of `f` over the segment's bounding box is positive. Either way a join never crosses material,
//!   however thin the wall.
//! * The point is **open** when a sample `e` joined to it has a disc of diameter `L` of air in the
//!   layer around it, shown by either (i) `f(e) ≥ L / 2` (a ball, so also a disc) or (ii) the interval
//!   bound of `f` over the bounding box of the in-layer square of side `L` centred on `e` is positive
//!   (the square holds the disc). Otherwise it is **enclosed**.
//! * Why (ii): `f` is a 3D distance, so near a floor or a ceiling it stays below `L/2` however wide the
//!   air is in the layer, and (i) alone reads every neck within `L/2` of a floor as enclosed. For a
//!   build axis along a coordinate axis the bounding boxes in (ii) are flat (zero thickness along the
//!   axis), so material above or below the layer does not count. For any other axis they are thicker
//!   than the layer, which can only read more material (enclosed), never less.
//! * Approximations: the reach is measured on the grid (a sample is in the disc when its grid offset
//!   is within `R`), so the boundary is resolved to one pitch `h`. A field that underestimates the
//!   distance, and the outward-rounded interval bounds, can only read open air as enclosed, which keeps
//!   a failure.
//! * Not handled (read as enclosed, so the failure stands): a point whose wall faces mostly along the
//!   build axis (a gap between a floor and the overhang above it is not a gap within one layer), and a
//!   seed that is not air.
//! * **Boxes** (used by [`prove_gaps_in_layers`] to stop subdividing open narrow air): a box of centre
//!   `c` and half-diagonal `hd` is certified open when the search from `c` succeeds with every
//!   condition taken `hd` stronger: `f(c) > hd`; a step by (i) needs `f(p) + f(q) > h + 2·hd`, by (ii)
//!   a positive bound over the segment's bounding box grown by `hd` on every side; the end by (i) needs
//!   `f(e) ≥ L/2 + hd`, by (ii) a positive bound over the square's bounding box grown by `hd`. Proof:
//!   take any point `x = c + δ` of the box, `|δ| ≤ hd`, and translate the centre's path by `δ`. It
//!   lies in `x`'s own layer (the plane through `c` moved by `δ`) and stays within `R` of `x`. Since
//!   `f` is 1-Lipschitz, the start has `f(x) > 0`, a translated step by (i) has
//!   `f(p+δ) + f(q+δ) > h`, and an end by (i) has `f(e+δ) ≥ L/2`. A translated segment or square lies
//!   inside the original bounding box grown by `|δ| ≤ hd` on every side, so a positive bound over the
//!   grown box shows it is air. So every translated step is air and the translated end has a disc of
//!   diameter `L` of air in `x`'s layer: `x` is open, for every `x` in the box. No assumption on the
//!   box's height along the axis is needed.
//! * Verified by: `gap_open_tests` (open slot vs closed pocket of the same width, the vertical
//!   through-hole, the reach boundary just inside and just outside, a non-Z build axis, the box
//!   margin against a case that passes only without it, a neck just above a floor (open) and a pocket
//!   standing on a floor (enclosed), the layered proof end to end).

use super::*;
use crate::compiled::real::LawMinMax;
use crate::interval::{next_down, next_up};

/// How far, in limits, the air next to a narrow point may run before it reaches a disc of the limit's
/// width for the point to count as open: `R = 2 · L`.
///
/// Reason: a neck between two separate walls widens on both sides of its narrowest point; for walls
/// whose curvature radius is at least the limit (a post of diameter ≥ L beside a flat wall) the width
/// grows from the neck to `L` within about `L` either side, so `2 · L` covers those with a margin of
/// one limit. A slot that stays narrow for longer than `2 · L` from the point in every in-layer
/// direction is a channel, and the hole rule applies to it.
pub const OPEN_REACH_LIMITS: f32 = 2.0;

/// The layer grid's pitch, in fractions of the limit: `h = L / 8`.
const OPEN_PITCH_PER_LIMIT: f32 = 8.0;

/// The least in-layer part of a wall's outward direction for a surface point to be seeded in the
/// layer: walls facing within 60° of the build axis are not read (enclosed).
const IN_LAYER_MIN: f32 = 0.5;

/// Whether the narrow air at the air point `seed` is open in the layer perpendicular to `build_axis`
/// (see the module's MODEL). `false` for a seed that is not air, a limit that is not finite positive,
/// or a build axis that is zero or not finite.
#[must_use]
pub fn narrow_air_is_open(node: &SdfNode, seed: Vec3, limit: f32, build_axis: Vec3) -> bool {
    open_with_margin(node, seed, limit, build_axis, 0.0).0
}

/// Whether every point of the box `[lo, hi]` is open narrow air in its own layer (see "Boxes" in the
/// module's MODEL), and the number of field evaluations spent.
pub(super) fn box_is_open(
    node: &SdfNode,
    lo: Vec3,
    hi: Vec3,
    limit: f32,
    axis: Vec3,
) -> (bool, u64) {
    let hd = 0.5 * (hi - lo).length();
    open_with_margin(node, 0.5 * (lo + hi), limit, axis, hd)
}

/// The layer search from `seed`, with the box margin `margin` (0 for the point test, the box's
/// half-diagonal to certify every point of a box). A step `p → q` is taken when either
/// `f(p) + f(q) > h + 2·margin` or the interval bound of `f` over the segment's bounding box grown by
/// `margin` is positive; the search ends open at a sample `e` where either `f(e) ≥ L/2 + margin` or
/// the interval bound over the bounding box of the in-layer square of side `L` centred on `e`, grown
/// by `margin`, is positive. The seed needs `f > margin`.
fn open_with_margin(
    node: &SdfNode,
    seed: Vec3,
    limit: f32,
    build_axis: Vec3,
    margin: f32,
) -> (bool, u64) {
    if !limit.is_finite() || limit <= 0.0 || !seed.is_finite() || margin.is_nan() || margin < 0.0 {
        return (false, 0);
    }
    let Some(axis) = unit(build_axis) else {
        return (false, 0);
    };
    let mut evals = 1u64;
    if eval(node, seed) <= margin {
        return (false, evals);
    }
    let (u, v) = layer_basis(axis);
    let h = limit / OPEN_PITCH_PER_LIMIT;
    let wide = 0.5 * limit + margin;
    let join = h + 2.0 * margin;
    // the disc radius in grid steps (OPEN_REACH_LIMITS · OPEN_PITCH_PER_LIMIT = 16)
    let n = (OPEN_REACH_LIMITS * OPEN_PITCH_PER_LIMIT) as i32;
    let side = (2 * n + 1) as usize;
    let index = |i: i32, j: i32| ((i + n) as usize) * side + (j + n) as usize;
    let at = |i: i32, j: i32| seed + u * (i as f32 * h) + v * (j as f32 * h);
    let mut value: Vec<Option<f32>> = vec![None; side * side];
    let mut seen = vec![false; side * side];
    let field = |i: i32, j: i32, value: &mut [Option<f32>], evals: &mut u64| -> f32 {
        let k = index(i, j);
        *value[k].get_or_insert_with(|| {
            *evals += 1;
            eval(node, at(i, j))
        })
    };
    // air over the bounding box of `points`, grown by `grown` on every side (outward-rounded)
    let air_over = |points: &[Vec3], grown: f32, evals: &mut u64| -> bool {
        *evals += 1;
        let (mut lo, mut hi) = (points[0], points[0]);
        for p in &points[1..] {
            lo = lo.law_min(*p);
            hi = hi.law_max(*p);
        }
        let grow = |x: f32, s: f32| {
            if s < 0.0 {
                next_down(x - grown)
            } else {
                next_up(x + grown)
            }
        };
        let lo = Vec3::new(grow(lo.x, -1.0), grow(lo.y, -1.0), grow(lo.z, -1.0));
        let hi = Vec3::new(grow(hi.x, 1.0), grow(hi.y, 1.0), grow(hi.z, 1.0));
        eval_interval(node, Vec3Interval::from_bounds(lo, hi)).lo > 0.0
    };
    let half = 0.5 * limit;
    let mut stack = vec![(0i32, 0i32)];
    seen[index(0, 0)] = true;
    let mut open = false;
    while let Some((i, j)) = stack.pop() {
        let f = field(i, j, &mut value, &mut evals);
        let e = at(i, j);
        if f >= wide
            || air_over(
                &[
                    e + u * half + v * half,
                    e + u * half - v * half,
                    e - u * half + v * half,
                    e - u * half - v * half,
                ],
                margin,
                &mut evals,
            )
        {
            open = true;
            break;
        }
        for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (ni, nj) = (i + di, j + dj);
            if ni * ni + nj * nj > n * n || seen[index(ni, nj)] {
                continue;
            }
            if f + field(ni, nj, &mut value, &mut evals) > join
                || air_over(&[e, at(ni, nj)], margin, &mut evals)
            {
                seen[index(ni, nj)] = true;
                stack.push((ni, nj));
            }
        }
    }
    (open, evals)
}

/// Whether the narrow air beside the surface point `at` (a wall point a gap width was measured from)
/// is open: the seed is one layer pitch off the wall along the wall's in-layer outward direction.
/// `false` when the wall faces mostly along the build axis (see [`IN_LAYER_MIN`]).
#[must_use]
pub fn narrow_air_beside_is_open(node: &SdfNode, at: Vec3, limit: f32, build_axis: Vec3) -> bool {
    let (Some(axis), Some(out)) = (unit(build_axis), unit(eval_normal(node, at))) else {
        return false;
    };
    let in_layer = out - axis * out.dot(axis);
    if in_layer.length() < IN_LAYER_MIN {
        return false;
    }
    let seed = at + in_layer.normalize() * (limit / OPEN_PITCH_PER_LIMIT);
    narrow_air_is_open(node, seed, limit, build_axis)
}

fn unit(v: Vec3) -> Option<Vec3> {
    let len = v.length();
    (len.is_finite() && len > 0.0).then(|| v / len)
}

/// Two unit vectors spanning the plane perpendicular to the unit `axis`.
fn layer_basis(axis: Vec3) -> (Vec3, Vec3) {
    let helper = if axis.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = axis.cross(helper).normalize();
    (u, axis.cross(u))
}

#[cfg(test)]
mod gap_open_tests {
    use super::*;

    const L: f32 = 2.0;

    /// Two separate blocks 1.5 apart (x in [-0.75, 0.75] is air), `2a` long in y, tall in z.
    fn open_slot(a: f32) -> SdfNode {
        let block = |x: f32| SdfNode::box3d_half_extents(2.0, a, 20.0).translate(x, 0.0, 0.0);
        block(-2.75).union(block(2.75))
    }

    /// A pocket 1.5 wide and 6 long cut 10 deep from the top of a block: closed in every layer.
    fn closed_pocket() -> SdfNode {
        SdfNode::box3d_half_extents(10.0, 10.0, 10.0)
            .subtract(SdfNode::box3d_half_extents(0.75, 3.0, 10.0).translate(0.0, 0.0, 5.0))
    }

    #[test]
    fn open_slot_is_open_and_closed_pocket_of_the_same_width_is_not() {
        let wall = Vec3::new(-0.75, 0.0, 0.0);
        assert!(narrow_air_beside_is_open(&open_slot(2.0), wall, L, Vec3::Z));
        assert!(!narrow_air_beside_is_open(
            &closed_pocket(),
            Vec3::new(-0.75, 0.0, 2.0),
            L,
            Vec3::Z
        ));
    }

    /// A 1 mm hole through a 3 mm plate is open at both ends along the build axis, and still enclosed in
    /// its layer: the hole rule applies to it.
    #[test]
    fn vertical_through_hole_stays_enclosed() {
        let plate = SdfNode::box3d_half_extents(10.0, 10.0, 1.5).subtract(
            SdfNode::cylinder(0.5, 10.0).rotate_euler(core::f32::consts::FRAC_PI_2, 0.0, 0.0),
        );
        let at = Vec3::new(0.5, 0.0, 0.0);
        assert!(eval(&plate, Vec3::ZERO) > 0.0, "the hole is air");
        assert!(!narrow_air_beside_is_open(&plate, at, L, Vec3::Z));
    }

    /// The reach `R = 2 · L = 4` from the seed (wall point + h): the wide disc beyond the end of the slot
    /// starts at `a + √(1 − 0.75²) ≈ a + 0.66` along y at mid-gap (0.5 = 2h off the seed). Just inside:
    /// a = 2.8 (wide sample at y = 3.5); just outside: a = 3.4 (first wide sample at y = 4.25 > R).
    #[test]
    fn reach_boundary_just_inside_and_just_outside() {
        let wall = Vec3::new(-0.75, 0.0, 0.0);
        assert!(narrow_air_beside_is_open(&open_slot(2.8), wall, L, Vec3::Z));
        assert!(!narrow_air_beside_is_open(
            &open_slot(3.4),
            wall,
            L,
            Vec3::Z
        ));
    }

    /// The build axis is the caller's: a slot 1.5 high (air z in [-0.75, 0.75]) that is 4 long in y and
    /// 40 long in x. Layers across y (axis x) see it open at y = ±2, within the reach; layers across x
    /// (axis y) see it run 20 either side, past the reach (enclosed); with axis z the wall faces the
    /// build axis and is not read in the layer.
    #[test]
    fn build_axis_is_a_parameter() {
        let block = |z: f32| SdfNode::box3d_half_extents(20.0, 2.0, 2.0).translate(0.0, 0.0, z);
        let slot = block(-2.75).union(block(2.75));
        let wall = Vec3::new(0.0, 0.0, -0.75);
        assert!(narrow_air_beside_is_open(&slot, wall, L, Vec3::X));
        assert!(!narrow_air_beside_is_open(&slot, wall, L, Vec3::Y));
        assert!(!narrow_air_beside_is_open(&slot, wall, L, Vec3::Z));
    }

    #[test]
    fn bad_inputs_are_not_open() {
        let s = open_slot(2.0);
        let seed = Vec3::new(0.0, 0.0, 0.0);
        assert!(narrow_air_is_open(&s, seed, L, Vec3::Z));
        assert!(!narrow_air_is_open(&s, seed, L, Vec3::ZERO));
        assert!(!narrow_air_is_open(&s, seed, f32::NAN, Vec3::Z));
        assert!(
            !narrow_air_is_open(&s, Vec3::new(-2.0, 0.0, 0.0), L, Vec3::Z),
            "seed in material"
        );
    }

    /// The box margin: a slot open at its y ends below z = 0 and capped at both ends above it. A box
    /// reaching above z = 0 has its centre in the open half, so the centre's point test reads open, while
    /// its points above z = 0 are enclosed. Read with the half-diagonal margin the box is not open;
    /// without the margin it would be
    #[test]
    fn box_margin_keeps_a_box_with_enclosed_points_out() {
        let block = |x: f32| SdfNode::box3d_half_extents(2.0, 2.0, 5.0).translate(x, 0.0, 0.0);
        let cap = |y: f32| SdfNode::box3d_half_extents(1.0, 0.25, 2.5).translate(0.0, y, 2.5);
        let node = block(-2.75)
            .union(block(2.75))
            .union(cap(2.25))
            .union(cap(-2.25));
        // the field is the 3D distance, so the air just under a cap reads thin: the open point sits 1.2
        // below the caps, the enclosed one 1 above their underside
        let centre = Vec3::new(0.0, 0.0, -1.2);
        assert!(narrow_air_is_open(&node, centre, L, Vec3::Z));
        assert!(!narrow_air_is_open(
            &node,
            Vec3::new(0.0, 0.0, 1.0),
            L,
            Vec3::Z
        ));
        // a box around that centre reaching above z = 0 (half-diagonal 1.31): its top points are enclosed
        let (lo, hi) = (Vec3::new(-0.2, -0.2, -2.48), Vec3::new(0.2, 0.2, 0.08));
        assert_eq!(0.5 * (lo + hi), centre);
        assert!(!box_is_open(&node, lo, hi, L, Vec3::Z).0);
        // a box wholly in the open half is certified
        let (lo, hi) = (Vec3::new(-0.1, -0.1, -2.2), Vec3::new(0.1, 0.1, -2.0));
        assert!(box_is_open(&node, lo, hi, L, Vec3::Z).0);
    }

    /// A block of material with the air boxes `(centre, half extents)` cut out.
    fn carved(air: &[(Vec3, Vec3)]) -> SdfNode {
        air.iter().fold(
            SdfNode::box3d_half_extents(10.0, 10.0, 10.0),
            |m, (c, h)| {
                m.subtract(SdfNode::box3d_half_extents(h.x, h.y, h.z).translate(c.x, c.y, c.z))
            },
        )
    }

    /// The union of the solid boxes `(centre, half extents)`.
    fn solid(parts: &[(Vec3, Vec3)]) -> SdfNode {
        parts
            .iter()
            .map(|(c, h)| SdfNode::box3d_half_extents(h.x, h.y, h.z).translate(c.x, c.y, c.z))
            .reduce(SdfNode::union)
            .expect("parts")
    }

    /// Each part of the box margin is needed. The box `[-0.2, 0.2]² × [-0.5, 0.5]` (half-diagonal 0.57)
    /// sits in an enclosed slot; its centre's layer reaches wide air through a way that only exists near
    /// z = 0, so its points at z = ±0.5 are enclosed and the box must not be certified
    #[test]
    fn every_part_of_the_box_margin_is_needed() {
        let (lo, hi) = (Vec3::new(-0.2, -0.2, -0.5), Vec3::new(0.2, 0.2, 0.5));
        let top = Vec3::new(0.0, 0.0, 0.5);
        // steps: a neck 0.5 wide and 0.4 high (f ≤ 0.2) joins the slot to a chamber; steps through it sum
        // to about 0.4 > h but not > h + 2·hd
        // (built from solid pieces: air cut out of a block box by box reads 0 where two cut boxes meet,
        // since the difference is a bound there, which would close the neck)
        let neck = solid(&[
            (Vec3::new(-2.875, 0.0, 0.0), Vec3::new(2.125, 2.0, 5.0)),
            (Vec3::new(2.875, 0.0, 0.0), Vec3::new(2.125, 2.0, 5.0)),
            (Vec3::new(0.0, -1.5, 0.0), Vec3::new(0.75, 0.5, 5.0)),
            (Vec3::new(-0.5, 1.5, 0.0), Vec3::new(0.25, 0.5, 5.0)),
            (Vec3::new(0.5, 1.5, 0.0), Vec3::new(0.25, 0.5, 5.0)),
            (Vec3::new(0.0, 1.5, 2.6), Vec3::new(0.25, 0.5, 2.4)),
            (Vec3::new(0.0, 1.5, -2.6), Vec3::new(0.25, 0.5, 2.4)),
        ]);
        assert!(narrow_air_is_open(&neck, Vec3::ZERO, L, Vec3::Z));
        assert!(!narrow_air_is_open(&neck, top, L, Vec3::Z));
        assert!(!box_is_open(&neck, lo, hi, L, Vec3::Z).0);
        // wide end: the slot (1.8 wide) ends in a sphere of radius 1.4 (f = 1.4 at its centre, ≥ L/2 but
        // below L/2 + hd); a layer 0.5 off its centre holds no air L wide
        let (lo2, hi2) = (lo + Vec3::new(0.0, 2.5, 0.0), hi + Vec3::new(0.0, 2.5, 0.0));
        let ball = carved(&[(Vec3::ZERO, Vec3::new(0.9, 3.0, 2.0))])
            .subtract(SdfNode::sphere(1.4).translate(0.0, 3.3, 0.0));
        assert!(narrow_air_is_open(
            &ball,
            Vec3::new(0.0, 2.5, 0.0),
            L,
            Vec3::Z
        ));
        assert!(!narrow_air_is_open(
            &ball,
            Vec3::new(0.0, 2.5, 0.5),
            L,
            Vec3::Z
        ));
        assert!(!box_is_open(&ball, lo2, hi2, L, Vec3::Z).0);
    }

    /// A step never crosses material, however thin: a pocket closed by a wall 0.1 thick (thinner than
    /// the pitch h = 0.25) with open air behind it is still enclosed
    #[test]
    fn a_step_never_crosses_a_thin_wall() {
        // a slot closed at y = -1 by a back wall and at y in [1.0, 1.1] by a thin wall, open air beyond
        let node = solid(&[
            (Vec3::new(-2.875, 0.0, 0.0), Vec3::new(2.125, 2.0, 5.0)),
            (Vec3::new(2.875, 0.0, 0.0), Vec3::new(2.125, 2.0, 5.0)),
            (Vec3::new(0.0, -1.5, 0.0), Vec3::new(0.75, 0.5, 5.0)),
            (Vec3::new(0.0, 1.05, 0.0), Vec3::new(0.75, 0.05, 5.0)),
        ]);
        assert!(eval(&node, Vec3::new(0.0, 1.05, 0.0)) < 0.0);
        assert!(!narrow_air_is_open(&node, Vec3::ZERO, L, Vec3::Z));
    }

    /// The verdict end to end (grid + layered proof) for the open slot, the closed pocket of the same
    /// width 1.5 and the 1 mm through-hole under the limit 2, printed along z: the open slot passes
    /// within the budget (the plain verdict fails it), the pocket and the hole fail
    #[test]
    fn a_neck_just_above_a_floor_is_open_and_a_pocket_on_a_floor_is_not() {
        // floor top at z = 1; a post (radius 3) 1.5 from a wall: 0.3 above the floor the 3D distance is
        // at most 0.3, so only the in-layer reading sees the wide air beside the neck
        let floor = SdfNode::box3d_half_extents(6.0, 6.0, 1.0);
        let post = floor
            .clone()
            .union(SdfNode::box3d_half_extents(1.0, 6.0, 4.0).translate(5.5, 0.0, 3.0))
            .union(
                SdfNode::cylinder(3.0, 6.0)
                    .rotate_euler(core::f32::consts::FRAC_PI_2, 0.0, 0.0)
                    .translate(0.0, 0.0, 3.0),
            );
        let neck = Vec3::new(3.75, 0.0, 1.3);
        assert!(eval(&post, neck) < 0.5 * L);
        assert!(narrow_air_is_open(&post, neck, L, Vec3::Z));
        assert!(narrow_air_beside_is_open(
            &post,
            Vec3::new(3.0, 0.0, 1.3),
            L,
            Vec3::Z
        ));
        // a pocket 1.5 × 3 walled on all four sides, standing on the same floor
        let pocket = solid(&[
            (Vec3::ZERO, Vec3::new(6.0, 6.0, 1.0)),
            (Vec3::new(-3.75, 0.0, 3.0), Vec3::new(3.0, 6.0, 2.0)),
            (Vec3::new(3.75, 0.0, 3.0), Vec3::new(3.0, 6.0, 2.0)),
            (Vec3::new(0.0, -3.75, 3.0), Vec3::new(0.75, 2.25, 2.0)),
            (Vec3::new(0.0, 3.75, 3.0), Vec3::new(0.75, 2.25, 2.0)),
        ]);
        for z in [1.3, 3.0] {
            assert!(
                !narrow_air_is_open(&pocket, Vec3::new(0.0, 0.0, z), L, Vec3::Z),
                "{z}"
            );
        }
    }

    #[test]
    fn the_interval_wide_end_needs_the_box_margin() {
        // a slot 1.4 wide (tall) runs into a chamber 2.2 × 2.2 but only 0.6 high: at z = 0 the chamber
        // holds an L-square, 0.5 higher it does not exist; f ≤ 0.3 in the chamber, so only the interval
        // reading finds it
        let scene = carved(&[
            (Vec3::new(0.0, 0.5, 0.0), Vec3::new(0.7, 2.5, 5.0)),
            (Vec3::new(0.0, 3.0, 0.0), Vec3::new(1.1, 1.1, 0.3)),
        ]);
        let c = Vec3::new(0.0, 1.5, 0.0);
        assert!(narrow_air_is_open(&scene, c, L, Vec3::Z));
        assert!(!narrow_air_is_open(
            &scene,
            c + Vec3::new(0.0, 0.0, 0.5),
            L,
            Vec3::Z
        ));
        let half = Vec3::new(0.2, 0.2, 0.5);
        assert!(!box_is_open(&scene, c - half, c + half, L, Vec3::Z).0);
        // a box held within the chamber's height is certified
        let thin = Vec3::new(0.05, 0.05, 0.05);
        assert!(box_is_open(&scene, c - thin, c + thin, L, Vec3::Z).0);
    }

    #[test]
    fn the_wide_end_square_is_grown_by_the_box_margin() {
        // a closed flat chamber 2.5 × 2.5 × 1.6: at z = 0 it holds an L-square (±1 inside ±1.25), so
        // the point is open; a box of half-diagonal 0.52 needs the square grown to ±1.52, which meets
        // the walls, and f ≤ 0.8 never reaches L/2 + 0.52, so the box is not certified
        let scene = carved(&[(Vec3::ZERO, Vec3::new(1.25, 1.25, 0.8))]);
        assert!(narrow_air_is_open(&scene, Vec3::ZERO, L, Vec3::Z));
        let half = Vec3::splat(0.3);
        assert!(!box_is_open(&scene, -half, half, L, Vec3::Z).0);
    }

    #[test]
    fn a_step_box_is_grown_below_the_layer_too() {
        // a corridor 1.0 wide and 1.2 high (z in ±0.6) leads to a wide chamber; across it a bump
        // rises from the floor to z = −0.15. A point at z = 0 passes over the bump (f ≈ 0.15 on both
        // sides of a step, sum above h = 0.25) and is open; a box of half-diagonal 0.35 is held 0.35
        // below its layer, which meets the bump, so the box is not certified
        let scene = carved(&[
            (Vec3::new(1.25, 0.0, 0.0), Vec3::new(1.75, 0.5, 0.6)),
            (Vec3::new(4.0, 0.0, 0.0), Vec3::new(2.0, 3.0, 3.0)),
        ])
        .union(SdfNode::box3d_half_extents(0.2, 0.6, 0.275).translate(1.0, 0.0, -0.425));
        assert!(narrow_air_is_open(&scene, Vec3::ZERO, L, Vec3::Z));
        let half = Vec3::splat(0.2);
        assert!(!box_is_open(&scene, -half, half, L, Vec3::Z).0);
    }

    fn verdicts(node: &SdfNode, half: Vec3) -> (GapVerdict, GapVerdict, GapProof) {
        let (bmin, bmax) = (-half, half);
        let cell = L / (2.0 * 3.0_f32.sqrt());
        let report = measure_gaps(node, bmin, bmax, cell).expect("grid");
        let plain = prove_gaps(node, L, bmin, bmax, 12, GAP_PROOF_WORK / 16).expect("proof");
        let layered = prove_gaps_in_layers(node, L, bmin, bmax, 12, GAP_PROOF_WORK / 16, Vec3::Z)
            .expect("proof");
        (
            report.verdict(L, &plain),
            report.verdict_in_layers(node, L, &layered, Vec3::Z),
            layered,
        )
    }

    #[test]
    fn layered_verdict_passes_the_open_slot_and_fails_the_pocket_and_the_hole() {
        let (plain, layered, proof) = verdicts(&open_slot(2.0), Vec3::new(5.0, 3.0, 2.0));
        assert!(!matches!(plain, GapVerdict::Pass(_)), "{plain:?}");
        assert!(
            matches!(layered, GapVerdict::Pass(_)),
            "{layered:?} {proof:?}"
        );
        assert!(proof.open_boxes > 0 && !proof.exhausted, "{proof:?}");
        // a post (radius 3, x in [-3, 3]) 1.5 from a wall (x from 4.5), both on a floor: the neck is read from boxes that straddle
        // the wall, which are split until their all-air parts are certified
        let post = SdfNode::box3d_half_extents(6.0, 6.0, 1.0)
            .union(SdfNode::box3d_half_extents(1.0, 6.0, 4.0).translate(5.5, 0.0, 3.0))
            .union(
                SdfNode::cylinder(3.0, 6.0)
                    .rotate_euler(core::f32::consts::FRAC_PI_2, 0.0, 0.0)
                    .translate(0.0, 0.0, 3.0),
            );
        let (plain, layered, proof) = verdicts(&post, Vec3::new(6.5, 6.5, 7.5));
        assert!(!matches!(plain, GapVerdict::Pass(_)), "{plain:?}");
        assert!(
            matches!(layered, GapVerdict::Pass(_)),
            "{layered:?} {proof:?}"
        );
        let (_, layered, _) = verdicts(&closed_pocket(), Vec3::new(6.0, 6.0, 6.0));
        assert!(matches!(layered, GapVerdict::Fail(_)), "{layered:?}");
        let hole = SdfNode::box3d_half_extents(4.0, 4.0, 1.5).subtract(
            SdfNode::cylinder(0.5, 10.0).rotate_euler(core::f32::consts::FRAC_PI_2, 0.0, 0.0),
        );
        let (_, layered, _) = verdicts(&hole, Vec3::new(5.0, 5.0, 2.0));
        assert!(matches!(layered, GapVerdict::Fail(_)), "{layered:?}");
    }

    #[test]
    fn layered_verdict_reads_its_inputs() {
        let node = open_slot(2.0);
        let half = Vec3::new(5.0, 3.0, 2.0);
        let report = measure_gaps(&node, -half, half, 0.5).expect("grid");
        let layered =
            prove_gaps_in_layers(&node, L, -half, half, 6, 1 << 20, Vec3::Z).expect("proof");
        let plain = prove_gaps(&node, L, -half, half, 6, 1 << 20).expect("proof");
        assert_eq!(
            report.verdict_in_layers(&node, L, &layered, Vec3::ZERO),
            GapVerdict::Undetermined(GapUndetermined::BadBuildAxis)
        );
        assert_eq!(
            report.verdict_in_layers(&closed_pocket(), L, &layered, Vec3::Z),
            GapVerdict::Undetermined(GapUndetermined::ProofForAnotherShape)
        );
        // a layered proof is read only in layers with its own axis, a plain proof only by `verdict`
        assert!(matches!(
            report.verdict_in_layers(&node, L, &layered, Vec3::X),
            GapVerdict::Undetermined(GapUndetermined::ProofForAnotherBuildAxis { .. })
        ));
        assert!(matches!(
            report.verdict_in_layers(&node, L, &plain, Vec3::Z),
            GapVerdict::Undetermined(GapUndetermined::ProofForAnotherBuildAxis {
                proof_axis: None
            })
        ));
        assert!(matches!(
            report.verdict(L, &layered),
            GapVerdict::Undetermined(GapUndetermined::ProofForAnotherBuildAxis { .. })
        ));
        assert_eq!(
            prove_gaps_in_layers(
                &node,
                L,
                -half,
                half,
                6,
                1 << 20,
                Vec3::new(f32::NAN, 0.0, 0.0)
            ),
            Err(GapError::BadCell)
        );
    }
}
