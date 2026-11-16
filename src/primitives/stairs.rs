//! Stairs SDF (Deep Fried Edition)
//!
//! Staircase profile in XY plane, extruded along Z-axis.
//! The staircase is a union of progressively taller rectangular steps.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::LawMinMax;
use glam::{Vec2, Vec3};

/// SDF of a single step box in local coordinates
#[inline(always)]
fn step_box(lx: f32, ly: f32, s: f32, sw: f32, sh: f32) -> f32 {
    let cx = s * sw + (sw * 0.5);
    let hy = (s + 1.0) * sh * 0.5;
    let dx = sw * -0.5 + (lx - cx).abs();
    let dy = (ly - hy).abs() - hy;
    Vec2::new(dx.law_max(0.0), dy.law_max(0.0)).length() + dx.law_max(dy).law_min(0.0)
}

/// The most steps a staircase may have.
///
/// 4096 is a stated ceiling, measured max is 5, 0 stored files affected (2026-11-16: the corpus, goldens and tests
/// of this crate use at most 5 steps; no stored tree here or in the dependent repositories holds a staircase). It
/// keeps one evaluation cheap (4096 boxes) and the shaders' step index exact in f32 (far below 2^24, where a float
/// counter stops counting).
///
/// The contract: a staircase over it is refused where a tree enters this crate ([`SdfNode::try_stairs`](
/// crate::SdfNode::try_stairs), deserialisation of a stored tree, compile, and every GPU evaluator construction),
/// with [`StairsError`]. Every evaluator (this function and the GLSL / HLSL / WGSL helpers) also stops its loop over
/// the steps here; inside the contract that bound is never reached. A tree built by hand outside the contract (a
/// `Stairs` literal over the maximum) is out of contract: evaluated directly on the CPU it is bounded by this
/// loop cap (its first `MAX_STAIR_STEPS` steps), and every entry above refuses it.
pub const MAX_STAIR_STEPS: u32 = 4096;

/// Why a staircase was refused ([`MAX_STAIR_STEPS`])
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StairsError {
    /// More steps than [`MAX_STAIR_STEPS`]
    TooManySteps {
        /// The step count given
        n_steps: f32,
    },
    /// The step count is NaN or infinite
    NotFinite,
}

impl std::fmt::Display for StairsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManySteps { n_steps } => write!(
                f,
                "stairs: {n_steps} steps is over the most a staircase may have ({MAX_STAIR_STEPS}, primitives::MAX_STAIR_STEPS)"
            ),
            Self::NotFinite => write!(f, "stairs: the step count is not finite"),
        }
    }
}

impl std::error::Error for StairsError {}

/// Whether `n_steps` is inside the staircase contract ([`MAX_STAIR_STEPS`])
///
/// # Errors
///
/// [`StairsError`] for a non-finite count or one over the maximum.
pub fn check_stair_steps(n_steps: f32) -> Result<(), StairsError> {
    if !n_steps.is_finite() {
        Err(StairsError::NotFinite)
    } else if n_steps > MAX_STAIR_STEPS as f32 {
        Err(StairsError::TooManySteps { n_steps })
    } else {
        Ok(())
    }
}

/// Read a stored `n_steps` of a `Stairs` node: a count outside the contract is refused ([`check_stair_steps`])
///
/// # Errors
///
/// The data is not an `f32`, or it is outside the contract.
#[allow(
    clippy::disallowed_methods,
    reason = "stored form other-type: the field is read as part of an SdfNode, whose stored tree goes through the tree loader's identifier check; this reads an f32 and refuses a value outside the staircase contract"
)]
// ALLOW-UNWIRED: called by the derived Deserialize of SdfNode through #[serde(deserialize_with = "crate::primitives::stairs::deserialize_stair_steps")] on the Stairs variant's n_steps field (src/types/mod.rs), a string the guard does not read
pub fn deserialize_stair_steps<'de, D: serde::Deserializer<'de>>(de: D) -> Result<f32, D::Error> {
    use serde::Deserialize as _;
    let n = f32::deserialize(de)?;
    check_stair_steps(n).map_err(serde::de::Error::custom)?;
    Ok(n)
}

/// SDF for a staircase shape, extruded along Z-axis
///
/// The staircase has n_steps, each step_width wide and step_height tall.
/// Step i has height (i+1)*step_height. Centered at origin.
/// - `step_width`: width of each step
/// - `step_height`: height of each step
/// - `n_steps`: number of steps (as f32)
/// - `half_depth`: half the extrusion depth along Z
#[inline(always)]
pub fn sdf_stairs(
    p: Vec3,
    step_width: f32,
    step_height: f32,
    n_steps: f32,
    half_depth: f32,
) -> f32 {
    let sw = step_width;
    let sh = step_height;
    let n = n_steps.law_max(1.0);
    let tw = n * sw;
    let th = n * sh;

    // Center coordinates: local x in [0, tw], y in [0, th]
    let lx = p.x + tw * 0.5;
    let ly = p.y + th * 0.5;

    // Union of the n step boxes: the exact exterior distance is the minimum
    // over every box. The pre-1.11.0 candidate set {si − 1, si, si + 1, sj}
    // missed the nearest box for points above or beside the staircase and
    // jumped where the candidates changed (Lipschitz property test).
    let mut d_2d = f32::MAX;
    for si in 0..(n as u32).min(MAX_STAIR_STEPS) {
        d_2d = d_2d.law_min(step_box(lx, ly, si as f32, sw, sh));
    }

    // Extrude along Z
    let d_z = p.z.abs() - half_depth;
    let w = Vec2::new(d_2d.law_max(0.0), d_z.law_max(0.0));
    d_2d.law_max(d_z).law_min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_past_the_maximum_are_not_evaluated() {
        // oracle: a staircase evaluates its first MAX_STAIR_STEPS steps; the boxes are placed by their index and
        // the centring by n, so with n over the maximum the value equals the minimum over boxes 0..MAX at that
        // centring (the loop stops there instead of running for every step)
        let n = MAX_STAIR_STEPS as f32 + 1000.0;
        let (sw, sh) = (0.01, 0.01);
        let p = Vec3::new(0.0, 0.0, 0.0);
        let (tw, th) = (n * sw, n * sh);
        let (lx, ly) = (p.x + tw * 0.5, p.y + th * 0.5);
        let mut d = f32::MAX;
        for si in 0..MAX_STAIR_STEPS {
            d = d.law_min(step_box(lx, ly, si as f32, sw, sh));
        }
        let dz = p.z.abs() - 1.0;
        let w = Vec2::new(d.law_max(0.0), dz.law_max(0.0));
        let expected = d.law_max(dz).law_min(0.0) + w.length();
        assert_eq!(sdf_stairs(p, sw, sh, n, 1.0).to_bits(), expected.to_bits());
    }

    #[test]
    fn test_stairs_center_inside() {
        // Point clearly inside a step (offset from step edges)
        let d = sdf_stairs(Vec3::new(0.1, -0.3, 0.0), 0.5, 0.5, 4.0, 0.5);
        assert!(d < 0.0, "Point inside step should be negative, got {}", d);
    }

    #[test]
    fn test_stairs_far_outside() {
        let d = sdf_stairs(Vec3::new(5.0, 0.0, 0.0), 0.5, 0.5, 4.0, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_stairs_symmetry_z() {
        let d1 = sdf_stairs(Vec3::new(0.1, 0.1, 0.2), 0.5, 0.5, 4.0, 0.5);
        let d2 = sdf_stairs(Vec3::new(0.1, 0.1, -0.2), 0.5, 0.5, 4.0, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Z");
    }

    #[test]
    fn test_stairs_step_shape() {
        // Point on the first step (low-left) should be inside
        let d = sdf_stairs(Vec3::new(-0.8, -0.8, 0.0), 0.5, 0.5, 4.0, 0.5);
        assert!(d < 0.0, "Point on first step should be inside, got {}", d);
    }
}
