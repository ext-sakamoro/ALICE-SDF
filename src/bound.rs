//! Lower bounds of SDF subtrees: `value(p) ≥ k·|p − c| − r − slack`
//!
//! [`bound`] returns a [`Bound`] such that, for every point `p`, the value an evaluator computes
//! for the node is at least [`Bound::lower`]`(p)`. The bound is a property of the evaluated value
//! (not only of the true distance), so it holds for operands whose formula is exact outside and
//! inside (sphere, box, capsule, torus) and for the transforms that keep that (translate, unit
//! rotate, uniform positive scale, twist, round), through unions, intersections and subtractions
//!
//! A node whose value is not provably bounded this way (an underestimating SDF, a non-uniform or
//! non-positive scale, a repetition, a noise, a smooth blend, ...) has no bound: a union never
//! skips it
//!
//! # The two evaluators ([`BoundTarget`])
//!
//! - [`BoundTarget::Cpu`]: [`crate::eval::eval`]; twist's sine and cosine are alice-det-math's
//!   (within a few ulp), so the rotation keeps the length up to rounding and `k = 1`
//! - [`BoundTarget::Shader`]: the transpiled shaders; twist reduces its angle to about
//!   `[−π/4, π/4]` and calls the native `sin` / `cos` there, whose error WGSL bounds by 2⁻¹¹
//!   absolute on `[−π, π]`. The rotation then scales the xz length by
//!   `√(c² + s²) ∈ [1 − √2·2⁻¹¹, 1 + √2·2⁻¹¹]`, so each twist on the path multiplies `k` by
//!   `1 − δ`, `δ = 2⁻¹⁰` ([`SHADER_TWIST_DELTA`], ≥ √2·2⁻¹¹ with room for the rotation's
//!   rounding). The reduction keeps the argument in `[−π, π]` while
//!   `|strength · y| ≤ 2¹⁶ · π/2` (the derivation is next to the twist's emission in
//!   `compiled::transpiler_common`); beyond that the shader's twist is outside the contract
//!
//! # The slack ([`Bound::lower`])
//!
//! The proofs are in exact arithmetic; the evaluators round. Every node on a path from the
//! operand's root to a leaf evaluates a handful of operations (at most about a dozen: the
//! quaternion rotation's two cross products, a box's abs / sub / max / length / min, a twist's
//! two products and a sum per axis) on quantities whose size is at most `M = 1 + |p| + |c| + r`
//! in the operand's frame (a scale rescales the quantities and the result together, so `M` in
//! the parent's frame covers it). WGSL's `+ − ×` are correctly rounded (½ ulp), `sqrt` /
//! `length` are within a few ulp, and the CPU laws are the same or tighter: a node adds at most
//! `16 · 2⁻²⁴ · M = 2⁻²⁰ · M`. Over a path of `depth` nodes plus the guard's own `length`, the
//! slack is `(depth + 1) · 2⁻²⁰ · M` ([`SLACK_PER_NODE`])
//!
//! Used by [`crate::eval::eval_union_bounded`] and
//! [`crate::compiled::WgslShader::transpile_with_union_bounds`]: a union operand whose lower bound
//! is greater than the nearest value found so far cannot be the minimum and is not evaluated

use crate::types::SdfNode;
use glam::Vec3;

/// The rounding budget of one node, relative to `M = 1 + |p| + |c| + r` (see the module
/// documentation): `16 · 2⁻²⁴ = 2⁻²⁰`
pub const SLACK_PER_NODE: f32 = 1.0 / 1_048_576.0;

/// The shader twist's bound on `|√(c² + s²) − 1|` for the native sine and cosine of the reduced
/// angle, with the rotation's rounding: `2⁻¹⁰` (≥ √2 · 2⁻¹¹, see the module documentation)
pub const SHADER_TWIST_DELTA: f32 = 1.0 / 1024.0;

/// Which evaluator a bound is for
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundTarget {
    /// [`crate::eval::eval`] (alice-det-math's sine and cosine)
    Cpu,
    /// The transpiled shaders (native sine and cosine on the reduced angle)
    Shader,
}

/// A lower bound of a node's evaluated value, read through [`Bound::lower`] (the only entry point,
/// so the shape of the bound can grow beyond a sphere): today `value(p) ≥ k·|p − c| − r`, up to
/// the slack of `depth` nodes
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bound {
    /// Centre
    pub(crate) c: Vec3,
    /// Radius: the node's surface and interior lie in the sphere `|p − c| ≤ r / k`
    pub(crate) r: f32,
    /// The slope, `0 < k ≤ 1` (1 on the CPU; each shader twist multiplies it by `1 − δ`)
    pub(crate) k: f32,
    /// The number of nodes on the longest path from the node to a leaf (a leaf is 1)
    pub(crate) depth: u32,
}

impl Bound {
    /// The lower bound of the node's value at `p`, minus the rounding slack of its depth: an
    /// operand whose `lower` is greater than a value already found cannot be smaller than it
    #[inline]
    #[must_use]
    pub fn lower(&self, p: Vec3) -> f32 {
        self.k * (p - self.c).length() - self.r - self.slack(p)
    }

    /// The number of nodes on the longest path from the node to a leaf (a leaf is 1), which the
    /// slack grows with
    #[must_use]
    pub const fn depth(&self) -> u32 {
        self.depth
    }

    /// The rounding slack at `p`: `(depth + 1) · 2⁻²⁰ · (1 + |p| + |c| + r)` (the derivation is in
    /// the module documentation; a box bound takes the same per-node budget with its own `M`)
    #[inline]
    #[must_use]
    pub fn slack(&self, p: Vec3) -> f32 {
        let m = 1.0 + p.length() + self.c.length() + self.r;
        (self.depth as f32 + 1.0) * SLACK_PER_NODE * m
    }
}

/// The bound of `node`'s value for the evaluator `target`, or `None` when the node is not
/// provably bounded (see the module documentation)
#[must_use]
pub fn bound(node: &SdfNode, target: BoundTarget) -> Option<Bound> {
    let leaf = |c: Vec3, r: f32| Bound {
        c,
        r,
        k: 1.0,
        depth: 1,
    };
    let up = |b: Bound| Bound {
        depth: b.depth + 1,
        ..b
    };
    let b = match node {
        SdfNode::Sphere { radius } => (*radius >= 0.0).then(|| leaf(Vec3::ZERO, *radius))?,
        SdfNode::Box3d { half_extents } => {
            (half_extents.min_element() >= 0.0).then(|| leaf(Vec3::ZERO, half_extents.length()))?
        }
        SdfNode::Capsule {
            point_a,
            point_b,
            radius,
        } => (*radius >= 0.0).then(|| {
            leaf(
                (*point_a + *point_b) * 0.5,
                (*point_b - *point_a).length() * 0.5 + *radius,
            )
        })?,
        SdfNode::Torus {
            major_radius,
            minor_radius,
        } => (*major_radius >= 0.0 && *minor_radius >= 0.0)
            .then(|| leaf(Vec3::ZERO, *major_radius + *minor_radius))?,

        SdfNode::Translate { child, offset } => {
            let b = bound(child, target)?;
            up(Bound {
                c: b.c + *offset,
                ..b
            })
        }
        SdfNode::Rotate { child, rotation } => {
            // the evaluator's rotation is an isometry only for a unit quaternion
            if (rotation.length_squared() - 1.0).abs() > 1e-6 {
                return None;
            }
            let b = bound(child, target)?;
            up(Bound {
                c: *rotation * b.c,
                ..b
            })
        }
        SdfNode::Scale { child, factor } => {
            if !(*factor > 0.0 && factor.is_finite()) {
                return None;
            }
            // value = s · child(p / s) ≥ s · (k |p/s − c| − r) = k |p − s c| − s r
            let b = bound(child, target)?;
            up(Bound {
                c: b.c * *factor,
                r: b.r * *factor,
                ..b
            })
        }
        // a rotation of the xz plane about the y axis: with a = (0, c.y, 0) and ρ = |c_xz|,
        // value ≥ k (|q − a| − ρ) − r and |q − a| ≥ (1 − δ) |p − a| (δ = 0 on the CPU), so
        // value ≥ k (1 − δ) |p − a| − (r + k ρ) ≥ k (1 − δ) |p − a| − (r + ρ)
        SdfNode::Twist { child, .. } => {
            let b = bound(child, target)?;
            let delta = match target {
                BoundTarget::Cpu => 0.0,
                BoundTarget::Shader => SHADER_TWIST_DELTA,
            };
            up(Bound {
                c: Vec3::new(0.0, b.c.y, 0.0),
                r: b.r + Vec3::new(b.c.x, 0.0, b.c.z).length(),
                k: b.k * (1.0 - delta),
                depth: b.depth + 1,
            })
        }
        SdfNode::Round { child, radius } => {
            if *radius < 0.0 {
                return None;
            }
            let b = bound(child, target)?;
            up(Bound {
                r: b.r + *radius,
                ..b
            })
        }
        SdfNode::Pose { child } | SdfNode::WithMaterial { child, .. } => up(bound(child, target)?),

        // min(a, b) ≥ min of the two bounds ≥ the bound of a sphere enclosing both, with the
        // smaller slope
        SdfNode::Union { a, b } => {
            let (x, y) = (bound(a, target)?, bound(b, target)?);
            let (c, r) = enclose((x.c, x.r), (y.c, y.r));
            Bound {
                c,
                r,
                k: x.k.min(y.k),
                depth: x.depth.max(y.depth) + 1,
            }
        }
        // max(a, b) ≥ a's value (and ≥ b's): either bound holds, the smaller sphere is tighter
        SdfNode::Intersection { a, b } => match (bound(a, target), bound(b, target)) {
            (Some(x), Some(y)) => up(if y.r < x.r { y } else { x }),
            (Some(x), None) | (None, Some(x)) => up(x),
            (None, None) => return None,
        },
        // max(a, −b) ≥ a's value
        SdfNode::Subtraction { a, .. } => up(bound(a, target)?),

        _ => return None,
    };
    (b.c.is_finite() && b.r.is_finite()).then_some(b)
}

/// The CPU bound's sphere `(c, r)` ([`bound`] with [`BoundTarget::Cpu`]): the surface and the
/// interior of `node` lie in it
#[must_use]
pub fn bounding_sphere(node: &SdfNode) -> Option<(Vec3, f32)> {
    bound(node, BoundTarget::Cpu).map(|b| (b.c, b.r))
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
    let grow = 4.0 * f32::EPSILON * (r + ca.length() + cb.length());
    (c, r + grow)
}
