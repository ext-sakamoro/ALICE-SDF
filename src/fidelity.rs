//! What a field's distance claim is worth.
//!
//! [`eval_lipschitz`] already computes the
//! bound `L` with `|f(p) − f(q)| ≤ L·|p − q|` on the exterior. What it does
//! not say — except in prose — is what a caller should *do* with a bound
//! above one, and the two failure modes are opposite:
//!
//! * `L ≤ 1`. Because `f` vanishes on the surface, `|f(p)| ≤ L·dist(p) ≤
//!   dist(p)`: the field never claims more distance than there is, so a
//!   sphere-tracing step of `f(p)` cannot cross the surface. The field may
//!   still be slack (report far less than the true distance), which costs
//!   steps, never correctness.
//! * `1 < L < ∞`. The same inequality now allows `f(p)` up to `L·dist(p)`:
//!   the field can over-report, a step of `f(p)` can land past the surface,
//!   and thin geometry is pierced. Stepping `f(p) / L` is safe again — that
//!   is exactly what [`RaymarchConfig`](crate::raycast::RaymarchConfig)
//!   divides by.
//! * `L = ∞`. Domain repetition with an arbitrary child, the taper singular
//!   plane, laws with a candidate-selection jump. No step size is safe from
//!   the bound alone.
//!
//! [`distance_fidelity`] turns the float into that three-way answer so a
//! consumer can branch on it instead of re-deriving the rule, and
//! [`Fidelity::safe_step_scale`] gives the divisor directly.
//!
//! This is a *static* claim about the law. The measured counterpart — what
//! the field actually does over a region, including the slack `L` cannot
//! see — is [`measure_tension`](crate::measure::measure_tension).

use crate::interval::eval_lipschitz;
use crate::types::SdfNode;

/// What a node's distance values can be trusted to mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fidelity {
    /// `L ≤ 1`: the field never reports more than the true distance. A
    /// sphere-tracing step of `f(p)` is safe, and `lipschitz` is the bound
    /// that was proved (it can be below one, which only costs steps).
    NeverOverReports {
        /// The proved bound, `≤ 1`.
        lipschitz: f32,
    },
    /// `1 < L < ∞`: the field can report up to `lipschitz` times the true
    /// distance. Divide the step by `lipschitz` to march it safely.
    OverReportsBy {
        /// The proved bound, `> 1`.
        lipschitz: f32,
    },
    /// No finite bound on the exterior: no step size follows from the law.
    Unbounded,
}

impl Fidelity {
    /// The divisor that makes a step of `f(p)` safe — `1.0` when the field
    /// never over-reports, `lipschitz` when it can, and `None` when nothing
    /// can be concluded.
    #[must_use]
    pub const fn safe_step_scale(self) -> Option<f32> {
        match self {
            Self::NeverOverReports { .. } => Some(1.0),
            Self::OverReportsBy { lipschitz } => Some(lipschitz),
            Self::Unbounded => None,
        }
    }

    /// True when a sphere-tracing step of the reported distance can cross
    /// the surface — the mode that pierces thin geometry.
    #[must_use]
    pub const fn can_overshoot(self) -> bool {
        !matches!(self, Self::NeverOverReports { .. })
    }
}

/// Classifies a tree by what its distance values can be trusted to mean.
///
/// Derived from [`eval_lipschitz`], so it inherits that function's guarantee
/// and its corpus-wide property test.
#[must_use]
pub fn distance_fidelity(node: &SdfNode) -> Fidelity {
    let l = eval_lipschitz(node);
    if !l.is_finite() {
        return Fidelity::Unbounded;
    }
    if l <= 1.0 {
        Fidelity::NeverOverReports { lipschitz: l }
    } else {
        Fidelity::OverReportsBy { lipschitz: l }
    }
}
