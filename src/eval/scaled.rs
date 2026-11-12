//! The field with its local scale: `(d, s)` where `d` is exactly [`super::eval`] and `d · s` is the
//! value in the units the child laws were written in, before any node divided it by a local slope
//! bound.
//!
//! `d` stays the tracer's step (it is the sound one). `s` is for the hit test only: a node such as
//! a grid warp divides its child's value by Λ(p) so the step never passes the surface, but then a
//! point h away from a surface where the warp compresses shows a value near h / Λ, and a hit test
//! on `|d| < ε` takes it for the surface. Testing `|d| · s < ε` instead keeps every real surface
//! (d is 0 there whatever s is) and rejects those near-misses.
//!
//! The rules (each combines the children's `(d, s)` with the same law [`super::eval`] uses for d):
//!
//! - a grid warp: `Λ(p) · s_child` where its child term decides the value; `1` where the floor, the
//!   reach clamp or the box decides it (those values are already distances in the posed space);
//! - non-uniform scale (`d = k_min · child`): `s_child · k_max / k_min`; a lattice deform
//!   (`d = child / c`): `s_child · c`; a projective transform (`d = child · min(c, L)`):
//!   `s_child · c / min(c, L)`;
//! - two operands (union, intersection, subtraction, xor and every blend: smooth, chamfer,
//!   stairs, columns, exponential, morph, pipe, engrave, groove, tongue): where the value is one
//!   operand's value or its negation, that operand's `s` (a blend outside its blend region, and
//!   always for the plain operations); elsewhere `min(s_a, s_b)`, which rejects no more than
//!   either operand would;
//! - translate, rotate, uniform scale, pose, material, skinning: the child's `s` (the value is the
//!   child's, or a fixed multiple of it);
//! - every other node: `s = 1`, the hit test as before this function existed (not tighter). A
//!   modifier over a warp (rounding, shelling, repetition, twist, ...) therefore keeps the old
//!   behaviour for the warp beneath it.
//!
//! No finite positive `s` removes a surface: d is 0 on it. A larger `s` only costs steps near it.
//!
//! Author: Moroya Sakamoto

use super::eval;
use crate::compiled::real::LawMinMax;
use crate::operations::*;
use crate::transforms::*;
use crate::types::SdfNode;
use glam::Vec3;

/// The field and its local scale at `point`: `(eval(node, point), s)` (see the module docs)
///
/// `d` is bit-identical to [`eval`] for every tree. `s = 1` for a tree with no grid warp,
/// non-uniform scale, lattice deform or projective transform.
#[must_use]
pub fn eval_scaled(node: &SdfNode, point: Vec3) -> (f32, f32) {
    let pair = |a: &SdfNode, b: &SdfNode| (eval_scaled(a, point), eval_scaled(b, point));
    match node {
        SdfNode::Union { a, b } => blend(pair(a, b), sdf_union),
        SdfNode::Intersection { a, b } => blend(pair(a, b), sdf_intersection),
        SdfNode::Subtraction { a, b } => blend(pair(a, b), sdf_subtraction),
        SdfNode::SmoothUnion { a, b, k } => blend(pair(a, b), |x, y| sdf_smooth_union(x, y, *k)),
        SdfNode::SmoothIntersection { a, b, k } => {
            blend(pair(a, b), |x, y| sdf_smooth_intersection(x, y, *k))
        }
        SdfNode::SmoothSubtraction { a, b, k } => {
            blend(pair(a, b), |x, y| sdf_smooth_subtraction(x, y, *k))
        }
        SdfNode::ChamferUnion { a, b, r } => blend(pair(a, b), |x, y| sdf_chamfer_union(x, y, *r)),
        SdfNode::ChamferIntersection { a, b, r } => {
            blend(pair(a, b), |x, y| sdf_chamfer_intersection(x, y, *r))
        }
        SdfNode::ChamferSubtraction { a, b, r } => {
            blend(pair(a, b), |x, y| sdf_chamfer_subtraction(x, y, *r))
        }
        SdfNode::StairsUnion { a, b, r, n } => {
            blend(pair(a, b), |x, y| sdf_stairs_union(x, y, *r, *n))
        }
        SdfNode::StairsIntersection { a, b, r, n } => {
            blend(pair(a, b), |x, y| sdf_stairs_intersection(x, y, *r, *n))
        }
        SdfNode::StairsSubtraction { a, b, r, n } => {
            blend(pair(a, b), |x, y| sdf_stairs_subtraction(x, y, *r, *n))
        }
        SdfNode::XOR { a, b } => blend(pair(a, b), sdf_xor),
        SdfNode::Morph { a, b, t } => blend(pair(a, b), |x, y| sdf_morph(x, y, *t)),
        SdfNode::ColumnsUnion { a, b, r, n } => {
            blend(pair(a, b), |x, y| sdf_columns_union(x, y, *r, *n))
        }
        SdfNode::ColumnsIntersection { a, b, r, n } => {
            blend(pair(a, b), |x, y| sdf_columns_intersection(x, y, *r, *n))
        }
        SdfNode::ColumnsSubtraction { a, b, r, n } => {
            blend(pair(a, b), |x, y| sdf_columns_subtraction(x, y, *r, *n))
        }
        SdfNode::Pipe { a, b, r } => blend(pair(a, b), |x, y| sdf_pipe(x, y, *r)),
        SdfNode::Engrave { a, b, r } => blend(pair(a, b), |x, y| sdf_engrave(x, y, *r)),
        SdfNode::Groove { a, b, ra, rb } => blend(pair(a, b), |x, y| sdf_groove(x, y, *ra, *rb)),
        SdfNode::Tongue { a, b, ra, rb } => blend(pair(a, b), |x, y| sdf_tongue(x, y, *ra, *rb)),
        SdfNode::ExpSmoothUnion { a, b, k } => {
            blend(pair(a, b), |x, y| sdf_exp_smooth_union(x, y, *k))
        }
        SdfNode::ExpSmoothIntersection { a, b, k } => {
            blend(pair(a, b), |x, y| sdf_exp_smooth_intersection(x, y, *k))
        }
        SdfNode::ExpSmoothSubtraction { a, b, k } => {
            blend(pair(a, b), |x, y| sdf_exp_smooth_subtraction(x, y, *k))
        }
        SdfNode::Translate { child, offset } => {
            eval_scaled(child, transform_translate(point, *offset))
        }
        SdfNode::Rotate { child, rotation } => {
            // the point exactly as `eval` moves it
            let p = crate::compiled::real::rotate_inverse::<f32>(*rotation, point.into()).into();
            eval_scaled(child, p)
        }
        SdfNode::Scale { child, factor } => {
            let (p, mult) = transform_scale(point, *factor);
            let (d, s) = eval_scaled(child, p);
            (d * mult, s)
        }
        SdfNode::ScaleNonUniform { child, factors } => {
            let (p, mult) = transform_scale_nonuniform(point, *factors);
            let (d, s) = eval_scaled(child, p);
            let k_max = factors.x.law_max(factors.y.law_max(factors.z));
            (d * mult, s * (k_max / mult))
        }
        SdfNode::ProjectiveTransform {
            child,
            inv_matrix,
            lipschitz_bound,
        } => {
            let (q, correction) =
                crate::transforms::projective::projective_transform(point, inv_matrix);
            let used = correction.law_min(*lipschitz_bound);
            let (d, s) = eval_scaled(child, q);
            (d * used, s * (correction / used))
        }
        SdfNode::LatticeDeform {
            child,
            control_points,
            nx,
            ny,
            nz,
            bbox_min,
            bbox_max,
        } => {
            let (q, correction) = crate::transforms::lattice::lattice_deform(
                point,
                control_points,
                *nx,
                *ny,
                *nz,
                *bbox_min,
                *bbox_max,
            );
            let (d, s) = eval_scaled(child, q);
            (d / correction, s * correction)
        }
        SdfNode::SdfSkinning { child, bones } => {
            let (q, _) = crate::transforms::skinning::sdf_skinning(point, bones);
            eval_scaled(child, q)
        }
        SdfNode::GridWarp { child, warp } => {
            let (x, e, lam, fl) = warp.warp(point);
            let (c, s) = eval_scaled(child, x);
            let (d, child_term) = warp.finish_with_term(c, e, lam, fl);
            // where the child's term gives the value it is c / Λ; elsewhere a posed distance
            (d, if child_term { lam * s } else { 1.0 })
        }
        SdfNode::BellPull { child, pull } => {
            let (c, s) = eval_scaled(child, pull.back(point));
            let lam = pull.lambda();
            (c / lam, lam * s)
        }
        SdfNode::Pose { child } | SdfNode::WithMaterial { child, .. } => eval_scaled(child, point),
        _ => (eval(node, point), 1.0),
    }
}

/// Two operands combined by `law`: where the value is one operand's (or its negation, as a
/// subtraction or an xor gives) that operand's scale, else (inside a blend) the smaller scale
#[inline]
fn blend(
    ((d1, s1), (d2, s2)): ((f32, f32), (f32, f32)),
    law: impl Fn(f32, f32) -> f32,
) -> (f32, f32) {
    let v = law(d1, d2);
    let s = if v.abs() == d1.abs() {
        s1
    } else if v.abs() == d2.abs() {
        s2
    } else {
        s1.law_min(s2)
    };
    (v, s)
}
