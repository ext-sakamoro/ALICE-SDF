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
//! * Two neighbouring samples `p`, `q` (distance `h`) are joined when `f(p) + f(q) > h`. With
//!   `|∇f| ≤ 1`, every point `x` of the segment has `f(x) ≥ max(f(p) − |x−p|, f(q) − |x−q|) > 0`, so a
//!   join never crosses material, however thin the wall.
//! * The point is **open** when a sample joined to it has `f ≥ L / 2`: a disc of diameter `L` of air,
//!   in the layer, within `R` of the point. Otherwise it is **enclosed**.
//! * Approximations: the reach is measured on the grid (a sample is in the disc when its grid offset
//!   is within `R`), so the boundary is resolved to one pitch `h`. A field that underestimates the
//!   distance can only read open air as enclosed (it never reads `f ≥ L/2` where the true distance is
//!   smaller), which keeps a failure.
//! * Not handled (read as enclosed, so the failure stands): a point whose wall faces mostly along the
//!   build axis (a gap between a floor and the overhang above it is not a gap within one layer), and a
//!   seed that is not air.
//! * Verified by: `gap_open_tests` (open slot vs closed pocket of the same width, the vertical
//!   through-hole, the reach boundary just inside and just outside, a non-Z build axis).

use super::*;

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
    if !(limit.is_finite() && limit > 0.0) || !seed.is_finite() {
        return false;
    }
    let Some(axis) = unit(build_axis) else {
        return false;
    };
    if eval(node, seed) <= 0.0 {
        return false;
    }
    let (u, v) = layer_basis(axis);
    let h = limit / OPEN_PITCH_PER_LIMIT;
    let wide = 0.5 * limit;
    // the disc radius in grid steps (OPEN_REACH_LIMITS · OPEN_PITCH_PER_LIMIT = 16)
    let n = (OPEN_REACH_LIMITS * OPEN_PITCH_PER_LIMIT) as i32;
    let side = (2 * n + 1) as usize;
    let index = |i: i32, j: i32| ((i + n) as usize) * side + (j + n) as usize;
    let mut value: Vec<Option<f32>> = vec![None; side * side];
    let mut seen = vec![false; side * side];
    let field = |i: i32, j: i32, value: &mut [Option<f32>]| -> f32 {
        let k = index(i, j);
        *value[k].get_or_insert_with(|| eval(node, seed + u * (i as f32 * h) + v * (j as f32 * h)))
    };
    let mut stack = vec![(0i32, 0i32)];
    seen[index(0, 0)] = true;
    while let Some((i, j)) = stack.pop() {
        let f = field(i, j, &mut value);
        if f >= wide {
            return true;
        }
        for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (ni, nj) = (i + di, j + dj);
            if ni * ni + nj * nj > n * n || seen[index(ni, nj)] {
                continue;
            }
            if f + field(ni, nj, &mut value) > h {
                seen[index(ni, nj)] = true;
                stack.push((ni, nj));
            }
        }
    }
    false
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
        assert!(!narrow_air_beside_is_open(&closed_pocket(), Vec3::new(-0.75, 0.0, 2.0), L, Vec3::Z));
    }

    /// A 1 mm hole through a 3 mm plate is open at both ends along the build axis, and still enclosed in
    /// its layer: the hole rule applies to it.
    #[test]
    fn vertical_through_hole_stays_enclosed() {
        let plate = SdfNode::box3d_half_extents(10.0, 10.0, 1.5).subtract(SdfNode::cylinder(0.5, 10.0).rotate_euler(
            core::f32::consts::FRAC_PI_2,
            0.0,
            0.0,
        ));
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
        assert!(!narrow_air_beside_is_open(&open_slot(3.4), wall, L, Vec3::Z));
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
        assert!(!narrow_air_is_open(&s, Vec3::new(-2.0, 0.0, 0.0), L, Vec3::Z), "seed in material");
    }

    /// The verdict end to end (grid + air-side proof) for the open slot and the closed pocket of the same
    /// width 1.5 under the limit 2: printed in layers along z the open slot is not a failure (the plain
    /// verdict fails it), the closed pocket fails in both
    fn verdicts(node: &SdfNode, half: Vec3) -> (GapVerdict, GapVerdict) {
        let (bmin, bmax) = (-half, half);
        let cell = L / (2.0 * 3.0_f32.sqrt());
        let report = measure_gaps(node, bmin, bmax, cell).expect("grid");
        let proof = prove_gaps(node, L, bmin, bmax, 12, GAP_PROOF_WORK).expect("proof");
        (
            report.verdict(L, &proof),
            report.verdict_in_layers(node, L, &proof, Vec3::Z),
        )
    }

    #[test]
    fn verdict_in_layers_passes_the_open_slot_and_fails_the_closed_pocket() {
        let (plain, layered) = verdicts(&open_slot(2.0), Vec3::new(6.0, 4.0, 11.0));
        assert!(matches!(plain, GapVerdict::Fail(_)), "{plain:?}");
        assert!(matches!(layered, GapVerdict::Pass(_)), "{layered:?}");
        let (plain, layered) = verdicts(&closed_pocket(), Vec3::new(11.0, 11.0, 11.0));
        assert!(matches!(plain, GapVerdict::Fail(_)), "{plain:?}");
        assert!(matches!(layered, GapVerdict::Fail(_)), "{layered:?}");
    }

    #[test]
    fn verdict_in_layers_reads_its_inputs() {
        let node = open_slot(2.0);
        let half = Vec3::new(6.0, 4.0, 11.0);
        let report = measure_gaps(&node, -half, half, 0.5).expect("grid");
        let proof = prove_gaps(&node, L, -half, half, 12, GAP_PROOF_WORK).expect("proof");
        assert_eq!(
            report.verdict_in_layers(&node, L, &proof, Vec3::ZERO),
            GapVerdict::Undetermined(GapUndetermined::BadBuildAxis)
        );
        assert_eq!(
            report.verdict_in_layers(&closed_pocket(), L, &proof, Vec3::Z),
            GapVerdict::Undetermined(GapUndetermined::ProofForAnotherShape)
        );
    }
}
