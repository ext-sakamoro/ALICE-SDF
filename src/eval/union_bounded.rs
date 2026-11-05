//! Union operand bounds (R0): a union evaluates an operand only when its bounding sphere could
//! hold a value below the nearest one found so far
//!
//! The operands of the top union chain (nested [`SdfNode::Union`] nodes) are visited nearest
//! bound first; an operand whose [`crate::bound::Bound::lower`] is greater than the minimum so far
//! is skipped, since its value is at least that bound. The union's law ([`sdf_union`],
//! alice-det-math's `min`: a total order with −0 < +0 that ignores NaN) is commutative and
//! associative, and a skipped value is strictly greater than the result, so the bounded value is
//! bit-equal to [`crate::eval::eval`] of the same node: the SDF value is never approximated
//! An operand without a bound ([`crate::bound::bound`] `None`) is always evaluated

use super::eval;
use crate::bound::{bound, BoundTarget};
use crate::operations::sdf_union;
use crate::types::SdfNode;
use glam::Vec3;

/// How many operands of the union chain a bounded evaluation evaluated and skipped
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UnionBoundStats {
    /// Operands evaluated
    pub evaluated: usize,
    /// Operands skipped by their bound
    pub skipped: usize,
}

/// The operands of the union chain at the top of `node` (`node` itself when it is not a union)
pub(crate) fn union_operands<'a>(node: &'a SdfNode, out: &mut Vec<&'a SdfNode>) {
    match node {
        SdfNode::Union { a, b } => {
            union_operands(a, out);
            union_operands(b, out);
        }
        n => out.push(n),
    }
}

/// [`crate::eval::eval`] of `node` with the union operand bounds: bit-equal, fewer operands
/// evaluated away from most of them
#[must_use]
pub fn eval_union_bounded(node: &SdfNode, point: Vec3) -> f32 {
    eval_union_bounded_stats(node, point).0
}

/// [`eval_union_bounded`] and the counts of evaluated and skipped operands
#[must_use]
pub fn eval_union_bounded_stats(node: &SdfNode, point: Vec3) -> (f32, UnionBoundStats) {
    let mut ops = Vec::new();
    union_operands(node, &mut ops);
    if ops.len() < 2 {
        let st = UnionBoundStats {
            evaluated: 1,
            skipped: 0,
        };
        return (eval(node, point), st);
    }
    // (lower bound, index): unbounded operands first (−∞), then nearest bound first; the index
    // breaks ties so the order is a function of the input only
    let mut order: Vec<(f32, usize)> = ops
        .iter()
        .enumerate()
        .map(|(i, op)| {
            let lb = bound(op, BoundTarget::Cpu).map_or(f32::NEG_INFINITY, |b| {
                let lb = b.lower(point);
                // a NaN bound decides nothing: evaluate
                if lb.is_nan() {
                    f32::NEG_INFINITY
                } else {
                    lb
                }
            });
            (lb, i)
        })
        .collect();
    order.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));

    let mut st = UnionBoundStats::default();
    // the first evaluated value starts the fold (not +∞), so an all-NaN union stays NaN as in
    // the plain fold
    let mut d: Option<f32> = None;
    for (lb, i) in order {
        // skip only when the bound is strictly greater (a NaN minimum skips nothing)
        if d.is_some_and(|d| lb > d) {
            st.skipped += 1;
            continue;
        }
        let v = eval(ops[i], point);
        d = Some(d.map_or(v, |d| sdf_union(d, v)));
        st.evaluated += 1;
    }
    (d.expect("a union chain has at least two operands"), st)
}
