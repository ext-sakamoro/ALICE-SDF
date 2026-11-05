//! Bounding spheres of SDF subtrees: a lower bound of the evaluated value
//!
//! [`bounding_sphere`] returns `(c, r)` such that, for every point `p`, the value
//! [`crate::eval::eval`] computes for the node is at least `|p − c| − r` (up to the rounding of
//! the evaluator, which [`lower_bound`] absorbs with a relative slack) The bound is a property of
//! the evaluated value, not only of the true distance, so it holds for operands whose formula is
//! exact outside and inside (sphere, box, capsule, torus) and for the transforms that keep that
//! (translate, unit rotate, uniform positive scale, twist, round)
//!
//! A node whose value is not provably bounded this way (an underestimating SDF, a non-uniform
//! scale, a repetition, a noise, a smooth blend, ...) returns `None`: a union never skips it
//!
//! Used by [`crate::eval::eval_union_bounded`] and
//! [`crate::compiled::WgslShader::transpile_with_union_bounds`]: a union operand whose lower bound
//! is greater than the nearest value found so far cannot be the minimum and is not evaluated

use crate::types::SdfNode;
use glam::Vec3;

/// The relative slack of [`lower_bound`]: covers the rounding of the evaluator against the exact
/// arithmetic the bound is proven in (rotations, twist's sine / cosine, the reciprocal of a scale)
pub const BOUND_SLACK_REL: f32 = 1e-5;

/// A bounding sphere `(centre, radius)` of `node`'s evaluated value, or `None` when the node is not
/// provably bounded (see the module documentation)
#[must_use]
pub fn bounding_sphere(node: &SdfNode) -> Option<(Vec3, f32)> {
    let s = match node {
        SdfNode::Sphere { radius } => (*radius >= 0.0).then_some((Vec3::ZERO, *radius))?,
        SdfNode::Box3d { half_extents } => {
            (half_extents.min_element() >= 0.0).then_some((Vec3::ZERO, half_extents.length()))?
        }
        SdfNode::Capsule {
            point_a,
            point_b,
            radius,
        } => (*radius >= 0.0).then_some((
            (*point_a + *point_b) * 0.5,
            (*point_b - *point_a).length() * 0.5 + *radius,
        ))?,
        SdfNode::Torus {
            major_radius,
            minor_radius,
        } => (*major_radius >= 0.0 && *minor_radius >= 0.0)
            .then_some((Vec3::ZERO, *major_radius + *minor_radius))?,

        SdfNode::Translate { child, offset } => {
            let (c, r) = bounding_sphere(child)?;
            (c + *offset, r)
        }
        SdfNode::Rotate { child, rotation } => {
            // the evaluator's rotation is an isometry only for a unit quaternion
            if (rotation.length_squared() - 1.0).abs() > 1e-6 {
                return None;
            }
            let (c, r) = bounding_sphere(child)?;
            (*rotation * c, r)
        }
        SdfNode::Scale { child, factor } => {
            if !(*factor > 0.0 && factor.is_finite()) {
                return None;
            }
            let (c, r) = bounding_sphere(child)?;
            (c * *factor, r * *factor)
        }
        // a rotation of the xz plane about the y axis: the sphere about the axis point at the
        // child centre's height, grown by the centre's distance from the axis, is invariant
        SdfNode::Twist { child, .. } => {
            let (c, r) = bounding_sphere(child)?;
            (
                Vec3::new(0.0, c.y, 0.0),
                r + Vec3::new(c.x, 0.0, c.z).length(),
            )
        }
        SdfNode::Round { child, radius } => {
            if *radius < 0.0 {
                return None;
            }
            let (c, r) = bounding_sphere(child)?;
            (c, r + *radius)
        }
        SdfNode::Pose { child } | SdfNode::WithMaterial { child, .. } => bounding_sphere(child)?,

        // min(a, b) ≥ min of the two bounds ≥ the bound of a sphere enclosing both
        SdfNode::Union { a, b } => enclose(bounding_sphere(a)?, bounding_sphere(b)?),
        // max(a, b) ≥ a's value (and ≥ b's): either bound holds, the smaller sphere is tighter
        SdfNode::Intersection { a, b } => match (bounding_sphere(a), bounding_sphere(b)) {
            (Some(x), Some(y)) => {
                if y.1 < x.1 {
                    y
                } else {
                    x
                }
            }
            (Some(x), None) | (None, Some(x)) => x,
            (None, None) => return None,
        },
        // max(a, −b) ≥ a's value
        SdfNode::Subtraction { a, .. } => bounding_sphere(a)?,

        _ => return None,
    };
    (s.0.is_finite() && s.1.is_finite()).then_some(s)
}

/// The smallest sphere enclosing two spheres
fn enclose((ca, ra): (Vec3, f32), (cb, rb): (Vec3, f32)) -> (Vec3, f32) {
    let d = (cb - ca).length();
    if d + rb <= ra {
        return (ca, ra);
    }
    if d + ra <= rb {
        return (cb, rb);
    }
    let r = (d + ra + rb) * 0.5;
    let c = ca + (cb - ca) * ((r - ra) / d);
    // the rounding of the centre: grow by the error it can carry so both spheres stay inside
    (c, r * (1.0 + BOUND_SLACK_REL))
}

/// The lower bound of a node's value at `p` from its bounding sphere `(c, r)`
///
/// `|p − c| − r` minus the relative slack [`BOUND_SLACK_REL`]: an operand whose `lower_bound` is
/// greater than a value already found cannot be smaller than it
#[inline]
#[must_use]
pub fn lower_bound(c: Vec3, r: f32, p: Vec3) -> f32 {
    let slack = BOUND_SLACK_REL * (1.0 + p.length() + c.length() + r);
    (p - c).length() - r - slack
}
