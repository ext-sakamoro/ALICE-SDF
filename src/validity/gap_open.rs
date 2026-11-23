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
//! * **Boxes** (used by [`prove_gaps_in_layers`] to stop subdividing open narrow air): a box of
//!   half-diagonal `hd` is certified open when the search from its centre succeeds with every field
//!   value read `hd` lower: the centre has `f > hd`, every step `f(p) + f(q) > h + 2·hd`, and the end
//!   `f ≥ L/2 + hd`. Proof: take any point `x = c + δ` of the box, `|δ| ≤ hd`, and translate the
//!   centre's path by `δ`. It lies in `x`'s own layer (the plane through `c` moved by `δ`), stays within
//!   `R` of `x`, and since `f` is 1-Lipschitz each translated value is at least the original minus
//!   `hd`: the start has `f(x) > 0`, every translated step has `f(p+δ) + f(q+δ) > h` (so the segment
//!   is air, as above), and the end has `f ≥ L/2`. So `x` is open, for every `x` in the box. No
//!   assumption on the box's height along the axis is needed.
//! * Verified by: `gap_open_tests` (open slot vs closed pocket of the same width, the vertical
//!   through-hole, the reach boundary just inside and just outside, a non-Z build axis, the box
//!   margin against a case that passes only without it, the layered proof end to end).

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

/// The layer search from `seed` with every field value read `margin` lower: a step needs
/// `f(p) + f(q) > h + 2·margin`, the seed `f > margin`, the wide end `f ≥ L/2 + margin`. With
/// `margin = 0` it is the point test; with the box's half-diagonal it certifies every point of the box.
fn open_with_margin(
    node: &SdfNode,
    seed: Vec3,
    limit: f32,
    build_axis: Vec3,
    margin: f32,
) -> (bool, u64) {
    if !(limit.is_finite() && limit > 0.0) || !seed.is_finite() || !(margin >= 0.0) {
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
    let mut value: Vec<Option<f32>> = vec![None; side * side];
    let mut seen = vec![false; side * side];
    let mut field = |i: i32, j: i32, value: &mut [Option<f32>]| -> f32 {
        let k = index(i, j);
        *value[k].get_or_insert_with(|| {
            evals += 1;
            eval(node, seed + u * (i as f32 * h) + v * (j as f32 * h))
        })
    };
    let mut stack = vec![(0i32, 0i32)];
    seen[index(0, 0)] = true;
    let mut open = false;
    while let Some((i, j)) = stack.pop() {
        let f = field(i, j, &mut value);
        if f >= wide {
            open = true;
            break;
        }
        for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (ni, nj) = (i + di, j + dj);
            if ni * ni + nj * nj > n * n || seen[index(ni, nj)] {
                continue;
            }
            if f + field(ni, nj, &mut value) > join {
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

    /// The verdict end to end (grid + layered proof) for the open slot, the closed pocket of the same
    /// width 1.5 and the 1 mm through-hole under the limit 2, printed along z: the open slot passes
    /// within the budget (the plain verdict fails it), the pocket and the hole fail
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
