//! Interval Arithmetic for SDF evaluation
//!
//! Evaluates SDF trees over spatial regions (AABBs) instead of single points.
//! Returns conservative distance bounds [lo, hi] guaranteed to contain
//! all possible distance values within the region.
//!
//! - If `result.lo > 0`: entire region is outside the surface (safe to skip)
//! - If `result.hi < 0`: entire region is inside the surface
//! - Otherwise: surface may cross through this region
//!
//! Author: Moroya Sakamoto

use crate::types::SdfNode;
use glam::{Quat, Vec3};
use std::ops::{Add, Mul, Neg, Sub};

/// A closed interval [lo, hi] representing a range of possible values
#[derive(Clone, Copy, Debug)]
pub struct Interval {
    /// Lower bound
    pub lo: f32,
    /// Upper bound
    pub hi: f32,
}

impl Interval {
    /// Create a new interval
    #[inline(always)]
    pub fn new(lo: f32, hi: f32) -> Self {
        debug_assert!(lo <= hi + 1e-6, "lo ({}) > hi ({})", lo, hi);
        Self { lo, hi }
    }

    /// Create a point interval [v, v]
    #[inline(always)]
    pub fn point(v: f32) -> Self {
        Self { lo: v, hi: v }
    }

    /// The entire real line
    pub const EVERYTHING: Self = Self {
        lo: f32::NEG_INFINITY,
        hi: f32::INFINITY,
    };

    /// Zero interval
    pub const ZERO: Self = Self { lo: 0.0, hi: 0.0 };

    /// Check if the interval is entirely positive
    #[inline(always)]
    pub fn is_positive(self) -> bool {
        self.lo > 0.0
    }

    /// Check if the interval is entirely negative
    #[inline(always)]
    pub fn is_negative(self) -> bool {
        self.hi < 0.0
    }

    /// Absolute value of an interval
    #[inline(always)]
    pub fn abs(self) -> Self {
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            Self {
                lo: -self.hi,
                hi: -self.lo,
            }
        } else {
            Self {
                lo: 0.0,
                hi: self.hi.max(-self.lo),
            }
        }
    }

    /// Square root (clamped to non-negative)
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self {
            lo: self.lo.max(0.0).sqrt(),
            hi: self.hi.max(0.0).sqrt(),
        }
    }

    /// Square of an interval
    #[inline(always)]
    pub fn sqr(self) -> Self {
        if self.lo >= 0.0 {
            Self {
                lo: self.lo * self.lo,
                hi: self.hi * self.hi,
            }
        } else if self.hi <= 0.0 {
            Self {
                lo: self.hi * self.hi,
                hi: self.lo * self.lo,
            }
        } else {
            Self {
                lo: 0.0,
                hi: (self.lo * self.lo).max(self.hi * self.hi),
            }
        }
    }

    /// Minimum of two intervals
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.min(other.hi),
        }
    }

    /// Maximum of two intervals
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self {
            lo: self.lo.max(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Clamp to scalar range
    #[inline(always)]
    pub fn clamp(self, lo: f32, hi: f32) -> Self {
        Self {
            lo: self.lo.clamp(lo, hi),
            hi: self.hi.clamp(lo, hi),
        }
    }

    /// Expand interval by a constant in both directions
    #[inline(always)]
    pub fn expand(self, amount: f32) -> Self {
        Self {
            lo: self.lo - amount,
            hi: self.hi + amount,
        }
    }
}

impl Add for Interval {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            lo: self.lo + rhs.lo,
            hi: self.hi + rhs.hi,
        }
    }
}

impl Sub for Interval {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            lo: self.lo - rhs.hi,
            hi: self.hi - rhs.lo,
        }
    }
}

impl Mul for Interval {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let a = self.lo * rhs.lo;
        let b = self.lo * rhs.hi;
        let c = self.hi * rhs.lo;
        let d = self.hi * rhs.hi;
        Self {
            lo: a.min(b).min(c).min(d),
            hi: a.max(b).max(c).max(d),
        }
    }
}

impl Neg for Interval {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
}

impl Mul<f32> for Interval {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        if rhs >= 0.0 {
            Self {
                lo: self.lo * rhs,
                hi: self.hi * rhs,
            }
        } else {
            Self {
                lo: self.hi * rhs,
                hi: self.lo * rhs,
            }
        }
    }
}

impl Sub<f32> for Interval {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        Self {
            lo: self.lo - rhs,
            hi: self.hi - rhs,
        }
    }
}

// ============================================================
// Vec3Interval: 3D interval (axis-aligned box)
// ============================================================

/// 3D interval representing an axis-aligned box region
#[derive(Clone, Copy, Debug)]
pub struct Vec3Interval {
    /// X interval
    pub x: Interval,
    /// Y interval
    pub y: Interval,
    /// Z interval
    pub z: Interval,
}

impl Vec3Interval {
    /// Create from min/max corners
    #[inline]
    pub fn from_bounds(min: Vec3, max: Vec3) -> Self {
        Self {
            x: Interval::new(min.x, max.x),
            y: Interval::new(min.y, max.y),
            z: Interval::new(min.z, max.z),
        }
    }

    /// Translate by subtracting offset
    #[inline(always)]
    pub fn translate(self, offset: Vec3) -> Self {
        Self {
            x: self.x - Interval::point(offset.x),
            y: self.y - Interval::point(offset.y),
            z: self.z - Interval::point(offset.z),
        }
    }

    /// Rotate by quaternion (conservative: bounding box of rotated corners)
    #[inline]
    pub fn rotate(self, rotation: Quat) -> Self {
        let inv_rot = rotation.conjugate();
        let corners = [
            Vec3::new(self.x.lo, self.y.lo, self.z.lo),
            Vec3::new(self.x.lo, self.y.lo, self.z.hi),
            Vec3::new(self.x.lo, self.y.hi, self.z.lo),
            Vec3::new(self.x.lo, self.y.hi, self.z.hi),
            Vec3::new(self.x.hi, self.y.lo, self.z.lo),
            Vec3::new(self.x.hi, self.y.lo, self.z.hi),
            Vec3::new(self.x.hi, self.y.hi, self.z.lo),
            Vec3::new(self.x.hi, self.y.hi, self.z.hi),
        ];
        let first = inv_rot * corners[0];
        let mut min = first;
        let mut max = first;
        for &c in &corners[1..] {
            let r = inv_rot * c;
            min = min.min(r);
            max = max.max(r);
        }
        Self::from_bounds(min, max)
    }

    /// 3D interval length: sqrt(x² + y² + z²)
    #[inline]
    pub fn length(self) -> Interval {
        (self.x.sqr() + self.y.sqr() + self.z.sqr()).sqrt()
    }

    /// 2D length in XZ: sqrt(x² + z²)
    #[inline]
    pub fn length_xz(self) -> Interval {
        (self.x.sqr() + self.z.sqr()).sqrt()
    }

    /// Component-wise mirror (abs specified axes)
    #[inline]
    pub fn mirror(self, axes: Vec3) -> Self {
        Self {
            x: if axes.x != 0.0 { self.x.abs() } else { self.x },
            y: if axes.y != 0.0 { self.y.abs() } else { self.y },
            z: if axes.z != 0.0 { self.z.abs() } else { self.z },
        }
    }
}

// ============================================================
// Interval SDF evaluation
// ============================================================

/// Evaluate an SDF tree over a spatial region, returning distance bounds.
///
/// For all points p in the region: `result.lo <= sdf(p) <= result.hi`
pub fn eval_interval(node: &SdfNode, bounds: Vec3Interval) -> Interval {
    match node {
        // ============ Exact-IA Primitives ============
        SdfNode::Sphere { radius } => bounds.length() - Interval::point(*radius),

        SdfNode::Box3d { half_extents } => {
            let h = *half_extents;
            let qx = bounds.x.abs() - Interval::point(h.x);
            let qy = bounds.y.abs() - Interval::point(h.y);
            let qz = bounds.z.abs() - Interval::point(h.z);
            let outer = Vec3Interval {
                x: qx.max(Interval::ZERO),
                y: qy.max(Interval::ZERO),
                z: qz.max(Interval::ZERO),
            }
            .length();
            let inner = qx.max(qy).max(qz).min(Interval::ZERO);
            outer + inner
        }

        SdfNode::Cylinder {
            radius,
            half_height,
        } => {
            let dx = bounds.length_xz() - Interval::point(*radius);
            let dy = bounds.y.abs() - Interval::point(*half_height);
            let ox = dx.max(Interval::ZERO);
            let oy = dy.max(Interval::ZERO);
            (ox.sqr() + oy.sqr()).sqrt() + dx.max(dy).min(Interval::ZERO)
        }

        SdfNode::Torus {
            major_radius,
            minor_radius,
        } => {
            let qx = bounds.length_xz() - Interval::point(*major_radius);
            (qx.sqr() + bounds.y.sqr()).sqrt() - Interval::point(*minor_radius)
        }

        SdfNode::Plane { normal, distance } => {
            bounds.x * Interval::point(normal.x)
                + bounds.y * Interval::point(normal.y)
                + bounds.z * Interval::point(normal.z)
                - Interval::point(*distance)
        }

        SdfNode::Capsule {
            point_a,
            point_b,
            radius,
        } => {
            let ab = *point_b - *point_a;
            let ab_sq = ab.length_squared();
            if ab_sq < 1e-10 {
                let p = bounds.translate(*point_a);
                return p.length() - Interval::point(*radius);
            }
            let pa = Vec3Interval {
                x: bounds.x - Interval::point(point_a.x),
                y: bounds.y - Interval::point(point_a.y),
                z: bounds.z - Interval::point(point_a.z),
            };
            let dot = pa.x * Interval::point(ab.x)
                + pa.y * Interval::point(ab.y)
                + pa.z * Interval::point(ab.z);
            let t = (dot * (1.0 / ab_sq)).clamp(0.0, 1.0);
            let cx = pa.x - Interval::point(ab.x) * t;
            let cy = pa.y - Interval::point(ab.y) * t;
            let cz = pa.z - Interval::point(ab.z) * t;
            Vec3Interval {
                x: cx,
                y: cy,
                z: cz,
            }
            .length()
                - Interval::point(*radius)
        }

        SdfNode::InfiniteCylinder { radius } => bounds.length_xz() - Interval::point(*radius),

        SdfNode::RoundedBox {
            half_extents,
            round_radius,
        } => {
            let h = *half_extents;
            let qx = bounds.x.abs() - Interval::point(h.x);
            let qy = bounds.y.abs() - Interval::point(h.y);
            let qz = bounds.z.abs() - Interval::point(h.z);
            let outer = Vec3Interval {
                x: qx.max(Interval::ZERO),
                y: qy.max(Interval::ZERO),
                z: qz.max(Interval::ZERO),
            }
            .length();
            let inner = qx.max(qy).max(qz).min(Interval::ZERO);
            outer + inner - Interval::point(*round_radius)
        }

        // ============ Conservative-IA Primitives (bounding sphere) ============
        SdfNode::Cone {
            radius,
            half_height,
        } => ia_bsphere(
            bounds,
            ((2.0 * half_height).powi(2) + radius.powi(2)).sqrt(),
        ),
        SdfNode::Ellipsoid { radii } => ia_bsphere(bounds, radii.x.max(radii.y).max(radii.z)),
        SdfNode::RoundedCone {
            r1,
            r2,
            half_height,
        } => ia_bsphere(
            bounds,
            ((2.0 * half_height).powi(2) + r1.max(*r2).powi(2)).sqrt(),
        ),
        SdfNode::Pyramid { half_height } => {
            ia_bsphere(bounds, (0.25 + (2.0 * half_height).powi(2)).sqrt())
        }
        SdfNode::Octahedron { size } => ia_bsphere(bounds, *size),
        SdfNode::HexPrism {
            hex_radius,
            half_height,
        } => ia_bsphere(bounds, (hex_radius.powi(2) + half_height.powi(2)).sqrt()),
        SdfNode::Link {
            half_length,
            r1,
            r2,
        } => ia_bsphere(bounds, half_length + r1 + r2),
        SdfNode::Triangle { .. } | SdfNode::Bezier { .. } => Interval::EVERYTHING,
        SdfNode::CappedCone {
            half_height,
            r1,
            r2,
        } => ia_bsphere(
            bounds,
            ((2.0 * half_height).powi(2) + r1.max(*r2).powi(2)).sqrt(),
        ),
        SdfNode::CappedTorus {
            major_radius,
            minor_radius,
            ..
        } => ia_bsphere(bounds, major_radius + minor_radius),
        SdfNode::RoundedCylinder {
            radius,
            round_radius,
            half_height,
        } => ia_bsphere(
            bounds,
            ((radius + round_radius).powi(2) + half_height.powi(2)).sqrt(),
        ),
        SdfNode::TriangularPrism { width, half_depth } => {
            ia_bsphere(bounds, (width.powi(2) + half_depth.powi(2)).sqrt())
        }
        SdfNode::CutSphere { radius, .. } => ia_bsphere(bounds, *radius),
        SdfNode::CutHollowSphere { radius, .. } => ia_bsphere(bounds, *radius),
        SdfNode::DeathStar { ra, rb, .. } => ia_bsphere(bounds, ra.max(*rb)),
        SdfNode::SolidAngle { radius, .. } => ia_bsphere(bounds, *radius),
        SdfNode::Rhombus {
            la,
            lb,
            half_height,
            ..
        } => ia_bsphere(bounds, (la.max(*lb).powi(2) + half_height.powi(2)).sqrt()),
        SdfNode::Horseshoe {
            radius,
            half_length,
            width,
            thickness,
            ..
        } => ia_bsphere(bounds, radius + half_length + width + thickness),
        SdfNode::Vesica { radius, .. } => ia_bsphere(bounds, *radius),
        SdfNode::InfiniteCone { .. } | SdfNode::Gyroid { .. } | SdfNode::SchwarzP { .. } => {
            Interval::EVERYTHING
        }
        SdfNode::Heart { size } => ia_bsphere(bounds, *size * 2.0),
        SdfNode::Tube {
            outer_radius,
            half_height,
            ..
        } => ia_bsphere(bounds, (outer_radius.powi(2) + half_height.powi(2)).sqrt()),
        SdfNode::Barrel {
            radius,
            half_height,
            bulge,
        } => ia_bsphere(
            bounds,
            ((radius + bulge.abs()).powi(2) + half_height.powi(2)).sqrt(),
        ),
        SdfNode::Diamond {
            radius,
            half_height,
        } => ia_bsphere(bounds, (radius.powi(2) + half_height.powi(2)).sqrt()),
        SdfNode::ChamferedCube { half_extents, .. } => ia_bsphere(bounds, half_extents.length()),
        SdfNode::Superellipsoid { half_extents, .. } => ia_bsphere(bounds, half_extents.length()),
        SdfNode::RoundedX {
            width,
            round_radius,
            half_height,
        } => ia_bsphere(bounds, (width + round_radius).max(*half_height)),
        SdfNode::Pie {
            radius,
            half_height,
            ..
        } => ia_bsphere(bounds, (radius.powi(2) + half_height.powi(2)).sqrt()),
        SdfNode::Trapezoid {
            r1,
            r2,
            trap_height,
            half_depth,
        } => ia_bsphere(
            bounds,
            (r1.max(*r2).powi(2) + trap_height.powi(2) + half_depth.powi(2)).sqrt(),
        ),
        SdfNode::Parallelogram {
            width,
            para_height,
            skew,
            half_depth,
        } => ia_bsphere(
            bounds,
            ((width + skew.abs()).powi(2) + para_height.powi(2) + half_depth.powi(2)).sqrt(),
        ),
        SdfNode::Tunnel {
            width,
            height_2d,
            half_depth,
        } => ia_bsphere(
            bounds,
            (width.powi(2) + height_2d.powi(2) + half_depth.powi(2)).sqrt(),
        ),
        SdfNode::UnevenCapsule {
            r1,
            r2,
            cap_height,
            half_depth,
        } => ia_bsphere(bounds, r1.max(*r2) + cap_height + half_depth),
        SdfNode::Egg { ra, rb } => ia_bsphere(bounds, *ra + *rb),
        SdfNode::ArcShape {
            radius,
            thickness,
            half_height,
            ..
        } => ia_bsphere(
            bounds,
            ((radius + thickness).powi(2) + half_height.powi(2)).sqrt(),
        ),
        SdfNode::Moon {
            ra,
            rb,
            half_height,
            ..
        } => ia_bsphere(bounds, (ra.max(*rb).powi(2) + half_height.powi(2)).sqrt()),
        SdfNode::CrossShape {
            length,
            thickness,
            half_height,
            ..
        } => ia_bsphere(
            bounds,
            (length.max(*thickness).powi(2) + half_height.powi(2)).sqrt(),
        ),
        SdfNode::BlobbyCross { size, half_height } => {
            ia_bsphere(bounds, (size.powi(2) + half_height.powi(2)).sqrt())
        }
        SdfNode::ParabolaSegment {
            width,
            para_height,
            half_depth,
        } => ia_bsphere(
            bounds,
            (width.powi(2) + para_height.powi(2) + half_depth.powi(2)).sqrt(),
        ),
        SdfNode::RegularPolygon {
            radius,
            half_height,
            ..
        } => ia_bsphere(bounds, (radius.powi(2) + half_height.powi(2)).sqrt()),
        SdfNode::StarPolygon {
            radius,
            half_height,
            ..
        } => ia_bsphere(bounds, (radius.powi(2) + half_height.powi(2)).sqrt()),
        SdfNode::Stairs {
            step_width,
            step_height,
            n_steps,
            half_depth,
        } => {
            let extent = ((step_width * n_steps).powi(2)
                + (step_height * n_steps).powi(2)
                + half_depth.powi(2))
            .sqrt();
            ia_bsphere(bounds, extent)
        }
        SdfNode::Helix {
            major_r,
            minor_r,
            half_height,
            ..
        } => ia_bsphere(
            bounds,
            ((major_r + minor_r).powi(2) + half_height.powi(2)).sqrt(),
        ),
        SdfNode::Tetrahedron { size } => ia_bsphere(bounds, *size),
        SdfNode::Dodecahedron { radius } => ia_bsphere(bounds, *radius),
        SdfNode::Icosahedron { radius } => ia_bsphere(bounds, *radius),
        SdfNode::TruncatedOctahedron { radius } => ia_bsphere(bounds, *radius),
        SdfNode::TruncatedIcosahedron { radius } => ia_bsphere(bounds, *radius),
        SdfNode::BoxFrame { half_extents, .. } => ia_bsphere(bounds, half_extents.max_element()),
        SdfNode::DiamondSurface { .. } => Interval::EVERYTHING,
        SdfNode::Neovius { .. } => Interval::EVERYTHING,
        SdfNode::Lidinoid { .. } => Interval::EVERYTHING,
        SdfNode::IWP { .. } => Interval::EVERYTHING,
        SdfNode::FRD { .. } => Interval::EVERYTHING,
        SdfNode::FischerKochS { .. } => Interval::EVERYTHING,
        SdfNode::PMY { .. } => Interval::EVERYTHING,

        // ============ 2D Primitives ============
        SdfNode::Circle2D {
            radius,
            half_height,
        } => {
            let xy_len = (bounds.x.sqr() + bounds.y.sqr()).sqrt();
            let d2d = xy_len - Interval::point(*radius);
            let dz = bounds.z.abs() - Interval::point(*half_height);
            d2d.max(dz)
        }
        SdfNode::Rect2D {
            half_extents,
            half_height,
        } => {
            let dx = bounds.x.abs() - Interval::point(half_extents.x);
            let dy = bounds.y.abs() - Interval::point(half_extents.y);
            let d2d = Interval::new(
                dx.lo.max(dy.lo).min(0.0),
                0.0_f32.max(dx.hi.max(0.0).hypot(dy.hi.max(0.0))),
            );
            let dz = bounds.z.abs() - Interval::point(*half_height);
            d2d.max(dz)
        }
        SdfNode::Segment2D { .. } | SdfNode::Polygon2D { .. } => Interval::EVERYTHING,
        SdfNode::RoundedRect2D { half_extents, .. } => {
            ia_bsphere(bounds, half_extents.max_element())
        }
        SdfNode::Annular2D { outer_radius, .. } => ia_bsphere(bounds, *outer_radius),

        // ============ Operations ============
        SdfNode::Union { a, b } => eval_interval(a, bounds).min(eval_interval(b, bounds)),
        SdfNode::Intersection { a, b } => eval_interval(a, bounds).max(eval_interval(b, bounds)),
        SdfNode::Subtraction { a, b } => eval_interval(a, bounds).max(-eval_interval(b, bounds)),
        SdfNode::SmoothUnion { a, b, k } => {
            let sharp = eval_interval(a, bounds).min(eval_interval(b, bounds));
            Interval::new(sharp.lo - k * 0.25, sharp.hi)
        }
        SdfNode::SmoothIntersection { a, b, k } => {
            let sharp = eval_interval(a, bounds).max(eval_interval(b, bounds));
            Interval::new(sharp.lo, sharp.hi + k * 0.25)
        }
        SdfNode::SmoothSubtraction { a, b, k } => {
            let sharp = eval_interval(a, bounds).max(-eval_interval(b, bounds));
            Interval::new(sharp.lo, sharp.hi + k * 0.25)
        }
        SdfNode::ChamferUnion { a, b, r } => {
            let sharp = eval_interval(a, bounds).min(eval_interval(b, bounds));
            Interval::new(sharp.lo - r, sharp.hi)
        }
        SdfNode::ChamferIntersection { a, b, r } => {
            let sharp = eval_interval(a, bounds).max(eval_interval(b, bounds));
            Interval::new(sharp.lo, sharp.hi + r)
        }
        SdfNode::ChamferSubtraction { a, b, r } => {
            let sharp = eval_interval(a, bounds).max(-eval_interval(b, bounds));
            Interval::new(sharp.lo, sharp.hi + r)
        }
        SdfNode::StairsUnion { a, b, r, .. } => {
            let sharp = eval_interval(a, bounds).min(eval_interval(b, bounds));
            Interval::new(sharp.lo - r, sharp.hi)
        }
        SdfNode::StairsIntersection { a, b, r, .. } => {
            let sharp = eval_interval(a, bounds).max(eval_interval(b, bounds));
            Interval::new(sharp.lo, sharp.hi + r)
        }
        SdfNode::StairsSubtraction { a, b, r, .. } => {
            let sharp = eval_interval(a, bounds).max(-eval_interval(b, bounds));
            Interval::new(sharp.lo, sharp.hi + r)
        }
        SdfNode::XOR { a, b } => {
            let ia = eval_interval(a, bounds);
            let ib = eval_interval(b, bounds);
            // XOR = min(a,b).max(-max(a,b)); conservative: union of both bounds
            let min_ab = ia.min(ib);
            let neg_max = -(ia.max(ib));
            min_ab.max(neg_max)
        }
        SdfNode::Morph { a, b, t } => {
            let ia = eval_interval(a, bounds);
            let ib = eval_interval(b, bounds);
            ia * (1.0 - *t) + ib * *t
        }
        SdfNode::ColumnsUnion { a, b, r, .. } => {
            let sharp = eval_interval(a, bounds).min(eval_interval(b, bounds));
            Interval::new(sharp.lo - r, sharp.hi)
        }
        SdfNode::ColumnsIntersection { a, b, r, .. } => {
            let sharp = eval_interval(a, bounds).max(eval_interval(b, bounds));
            Interval::new(sharp.lo, sharp.hi + r)
        }
        SdfNode::ColumnsSubtraction { a, b, r, .. } => {
            let sharp = eval_interval(a, bounds).max(-eval_interval(b, bounds));
            Interval::new(sharp.lo, sharp.hi + r)
        }
        SdfNode::Pipe { a, b, r } => {
            let ia = eval_interval(a, bounds);
            let ib = eval_interval(b, bounds);
            // pipe = sqrt(a²+b²) - r; conservative bound
            let sq = ia * ia + ib * ib;
            Interval::new(sq.lo.max(0.0).sqrt() - r, sq.hi.max(0.0).sqrt() + r.abs())
        }
        SdfNode::Engrave { a, b, r } => {
            let ia = eval_interval(a, bounds);
            let ib = eval_interval(b, bounds);
            // engrave = max(a, (a+r-|b|)*FRAC_1_SQRT_2)
            let abs_b = ib.abs();
            let inner = Interval::new(
                (ia.lo + r - abs_b.hi) * std::f32::consts::FRAC_1_SQRT_2,
                (ia.hi + r - abs_b.lo).max(0.0) * std::f32::consts::FRAC_1_SQRT_2,
            );
            ia.max(inner)
        }
        SdfNode::Groove { a, b, ra, rb } => {
            let ia = eval_interval(a, bounds);
            let ib = eval_interval(b, bounds);
            // groove = max(a, min(a+ra, rb-|b|))
            let abs_b = ib.abs();
            let arm1 = Interval::new(ia.lo + ra, ia.hi + ra);
            let arm2 = Interval::new(rb - abs_b.hi, rb - abs_b.lo);
            ia.max(arm1.min(arm2))
        }
        SdfNode::Tongue { a, b, ra, rb } => {
            let ia = eval_interval(a, bounds);
            let ib = eval_interval(b, bounds);
            // tongue = min(a, max(a-ra, |b|-rb))
            let abs_b = ib.abs();
            let arm1 = Interval::new(ia.lo - ra, ia.hi - ra);
            let arm2 = Interval::new(abs_b.lo - rb, abs_b.hi - rb);
            ia.min(arm1.max(arm2))
        }
        SdfNode::ExpSmoothUnion { a, b, k } => {
            let sharp = eval_interval(a, bounds).min(eval_interval(b, bounds));
            Interval::new(sharp.lo - k * 0.5, sharp.hi)
        }
        SdfNode::ExpSmoothIntersection { a, b, k } => {
            let sharp = eval_interval(a, bounds).max(eval_interval(b, bounds));
            Interval::new(sharp.lo, sharp.hi + k * 0.5)
        }
        SdfNode::ExpSmoothSubtraction { a, b, k } => {
            let sharp = eval_interval(a, bounds).max(-eval_interval(b, bounds));
            Interval::new(sharp.lo, sharp.hi + k * 0.5)
        }
        SdfNode::Shear { child, shear } => {
            let max_shear = shear.x.abs().max(shear.y.abs()).max(shear.z.abs());
            let max_extent = (bounds.x.hi - bounds.x.lo)
                .max(bounds.y.hi - bounds.y.lo)
                .max(bounds.z.hi - bounds.z.lo);
            let child_interval = eval_interval(child, bounds);
            Interval::new(
                child_interval.lo - max_shear * max_extent,
                child_interval.hi + max_shear * max_extent,
            )
        }
        SdfNode::Animated { child, .. } => eval_interval(child, bounds),

        // ============ Transforms ============
        SdfNode::Translate { child, offset } => eval_interval(child, bounds.translate(*offset)),
        SdfNode::Rotate { child, rotation } => eval_interval(child, bounds.rotate(*rotation)),
        SdfNode::Scale { child, factor } => {
            let inv = 1.0 / *factor;
            let scaled = Vec3Interval {
                x: bounds.x * inv,
                y: bounds.y * inv,
                z: bounds.z * inv,
            };
            eval_interval(child, scaled) * *factor
        }
        SdfNode::ScaleNonUniform { child, factors } => {
            let min_f = factors.x.min(factors.y).min(factors.z);
            let max_f = factors.x.max(factors.y).max(factors.z);
            let scaled = Vec3Interval {
                x: bounds.x * (1.0 / factors.x),
                y: bounds.y * (1.0 / factors.y),
                z: bounds.z * (1.0 / factors.z),
            };
            let d = eval_interval(child, scaled);
            Interval::new(d.lo * min_f, d.hi * max_f)
        }
        SdfNode::ProjectiveTransform {
            child,
            lipschitz_bound,
            ..
        } => {
            // Conservative: scale by Lipschitz bound
            let child_interval = eval_interval(child, bounds);
            Interval::new(
                child_interval.lo / lipschitz_bound,
                child_interval.hi / lipschitz_bound,
            )
        }
        SdfNode::LatticeDeform { child, .. } => {
            // Conservative: assume worst-case deformation doubles distances
            let child_interval = eval_interval(child, bounds);
            Interval::new(child_interval.lo * 0.5, child_interval.hi * 2.0)
        }
        SdfNode::SdfSkinning { child, .. } => {
            // LBS is approximately distance-preserving
            eval_interval(child, bounds)
        }

        // ============ Modifiers ============
        SdfNode::Twist { child, strength: _ } => {
            // After twist, XZ can reach any rotation → expand to circular
            let max_xz = bounds.length_xz().hi;
            let expanded = Vec3Interval {
                x: Interval::new(-max_xz, max_xz),
                y: bounds.y,
                z: Interval::new(-max_xz, max_xz),
            };
            eval_interval(child, expanded)
        }
        SdfNode::Bend { child, .. } => {
            let max_r = bounds.length().hi;
            eval_interval(
                child,
                Vec3Interval::from_bounds(Vec3::splat(-max_r), Vec3::splat(max_r)),
            )
        }
        SdfNode::RepeatInfinite { child, spacing } => {
            let hs = *spacing * 0.5;
            let cell = Vec3Interval {
                x: if spacing.x > 0.0 {
                    Interval::new(-hs.x, hs.x)
                } else {
                    bounds.x
                },
                y: if spacing.y > 0.0 {
                    Interval::new(-hs.y, hs.y)
                } else {
                    bounds.y
                },
                z: if spacing.z > 0.0 {
                    Interval::new(-hs.z, hs.z)
                } else {
                    bounds.z
                },
            };
            eval_interval(child, cell)
        }
        SdfNode::RepeatFinite { child, spacing, .. } => {
            let hs = *spacing * 0.5;
            let cell = Vec3Interval {
                x: if spacing.x > 0.0 {
                    Interval::new(-hs.x, hs.x)
                } else {
                    bounds.x
                },
                y: if spacing.y > 0.0 {
                    Interval::new(-hs.y, hs.y)
                } else {
                    bounds.y
                },
                z: if spacing.z > 0.0 {
                    Interval::new(-hs.z, hs.z)
                } else {
                    bounds.z
                },
            };
            eval_interval(child, cell)
        }
        SdfNode::Noise {
            child, amplitude, ..
        } => eval_interval(child, bounds).expand(*amplitude),
        SdfNode::Round { child, radius } => eval_interval(child, bounds) - Interval::point(*radius),
        SdfNode::Onion { child, thickness } => {
            eval_interval(child, bounds).abs() - Interval::point(*thickness)
        }
        SdfNode::Elongate { child, amount } => {
            let qx = bounds.x - bounds.x.clamp(-amount.x, amount.x);
            let qy = bounds.y - bounds.y.clamp(-amount.y, amount.y);
            let qz = bounds.z - bounds.z.clamp(-amount.z, amount.z);
            eval_interval(
                child,
                Vec3Interval {
                    x: qx,
                    y: qy,
                    z: qz,
                },
            )
        }
        SdfNode::Mirror { child, axes } => eval_interval(child, bounds.mirror(*axes)),
        SdfNode::OctantMirror { child } => {
            // abs all axes (mirror), then eval with conservative sorted bounds
            eval_interval(child, bounds.mirror(Vec3::ONE))
        }
        SdfNode::Revolution { child, offset } => {
            let qx = bounds.length_xz() - Interval::point(*offset);
            eval_interval(
                child,
                Vec3Interval {
                    x: qx,
                    y: bounds.y,
                    z: Interval::ZERO,
                },
            )
        }
        SdfNode::Extrude { child, half_height } => {
            let flat = Vec3Interval {
                x: bounds.x,
                y: bounds.y,
                z: Interval::ZERO,
            };
            let d = eval_interval(child, flat);
            let wy = bounds.z.abs() - Interval::point(*half_height);
            let ox = d.max(Interval::ZERO);
            let oy = wy.max(Interval::ZERO);
            (ox.sqr() + oy.sqr()).sqrt() + d.max(wy).min(Interval::ZERO)
        }
        SdfNode::Taper { child, factor } => {
            eval_interval(child, bounds).expand(factor.abs() * bounds.y.abs().hi)
        }
        SdfNode::Displacement { child, strength } => {
            eval_interval(child, bounds).expand(strength.abs())
        }
        SdfNode::PolarRepeat { child, count } => {
            let sector = std::f32::consts::TAU / (*count as f32);
            let max_r = bounds.length_xz().hi;
            let ha = sector * 0.5;
            eval_interval(
                child,
                Vec3Interval {
                    x: Interval::new(0.0, max_r),
                    y: bounds.y,
                    z: Interval::new(-max_r * ha.sin(), max_r * ha.sin()),
                },
            )
        }
        SdfNode::SweepBezier { child, p0, p1, p2 } => {
            let bmin_x = p0.x.min(p1.x).min(p2.x);
            let bmax_x = p0.x.max(p1.x).max(p2.x);
            let bmin_z = p0.y.min(p1.y).min(p2.y);
            let bmax_z = p0.y.max(p1.y).max(p2.y);
            // Max perpendicular distance
            let corners = [
                (bounds.x.lo, bounds.z.lo),
                (bounds.x.lo, bounds.z.hi),
                (bounds.x.hi, bounds.z.lo),
                (bounds.x.hi, bounds.z.hi),
            ];
            let curve_corners = [
                (bmin_x, bmin_z),
                (bmin_x, bmax_z),
                (bmax_x, bmin_z),
                (bmax_x, bmax_z),
            ];
            let mut max_d2: f32 = 0.0;
            for &(px, pz) in &corners {
                for &(cx, cz) in &curve_corners {
                    let dx = px - cx;
                    let dz = pz - cz;
                    max_d2 = max_d2.max(dx * dx + dz * dz);
                }
            }
            // Min distance: 0 if overlapping, else min box-to-box distance
            let min_perp = {
                let dx = if bounds.x.hi < bmin_x {
                    bmin_x - bounds.x.hi
                } else if bounds.x.lo > bmax_x {
                    bounds.x.lo - bmax_x
                } else {
                    0.0
                };
                let dz = if bounds.z.hi < bmin_z {
                    bmin_z - bounds.z.hi
                } else if bounds.z.lo > bmax_z {
                    bounds.z.lo - bmax_z
                } else {
                    0.0
                };
                (dx * dx + dz * dz).sqrt()
            };
            eval_interval(
                child,
                Vec3Interval {
                    x: Interval::new(min_perp, max_d2.sqrt()),
                    y: bounds.y,
                    z: Interval::ZERO,
                },
            )
        }

        SdfNode::IcosahedralSymmetry { child } => {
            // Symmetry fold preserves distance bounds
            eval_interval(child, bounds.mirror(Vec3::ONE))
        }
        SdfNode::IFS { child, .. } => {
            // Conservative: IFS may contract, so interval is child's interval
            eval_interval(child, bounds)
        }
        SdfNode::HeightmapDisplacement {
            child, amplitude, ..
        } => {
            let child_iv = eval_interval(child, bounds);
            // Displacement shifts distance by at most amplitude
            let amp = *amplitude;
            Interval::new(child_iv.lo - amp.abs(), child_iv.hi + amp.abs())
        }
        SdfNode::SurfaceRoughness {
            child, amplitude, ..
        } => {
            let child_iv = eval_interval(child, bounds);
            let amp = *amplitude;
            Interval::new(child_iv.lo - amp.abs(), child_iv.hi + amp.abs())
        }

        SdfNode::WithMaterial { child, .. } => eval_interval(child, bounds),

        #[allow(unreachable_patterns)]
        _ => Interval::EVERYTHING,
    }
}

/// Bounding sphere conservative interval: `length(p) - radius`
#[inline]
fn ia_bsphere(bounds: Vec3Interval, radius: f32) -> Interval {
    let l = bounds.length();
    Interval::new(l.lo - radius, l.hi + radius)
}

// ============================================================
// Lipschitz bound tracking
// ============================================================

/// Compute an upper bound on the Lipschitz constant of the SDF.
///
/// The Lipschitz constant L satisfies `|∇SDF(p)| ≤ L` for all p.
/// For exact SDFs, L = 1.0. Modifiers like twist/bend increase L.
/// Used for relaxed sphere tracing step size: `step = distance / L`.
pub fn eval_lipschitz(node: &SdfNode) -> f32 {
    match node {
        // All primitives are exact SDFs with L = 1.0
        SdfNode::Sphere { .. }
        | SdfNode::Box3d { .. }
        | SdfNode::Cylinder { .. }
        | SdfNode::Torus { .. }
        | SdfNode::Plane { .. }
        | SdfNode::Capsule { .. }
        | SdfNode::Cone { .. }
        | SdfNode::Ellipsoid { .. }
        | SdfNode::RoundedCone { .. }
        | SdfNode::Pyramid { .. }
        | SdfNode::Octahedron { .. }
        | SdfNode::HexPrism { .. }
        | SdfNode::Link { .. }
        | SdfNode::Triangle { .. }
        | SdfNode::Bezier { .. }
        | SdfNode::RoundedBox { .. }
        | SdfNode::CappedCone { .. }
        | SdfNode::CappedTorus { .. }
        | SdfNode::RoundedCylinder { .. }
        | SdfNode::TriangularPrism { .. }
        | SdfNode::CutSphere { .. }
        | SdfNode::CutHollowSphere { .. }
        | SdfNode::DeathStar { .. }
        | SdfNode::SolidAngle { .. }
        | SdfNode::Rhombus { .. }
        | SdfNode::Horseshoe { .. }
        | SdfNode::Vesica { .. }
        | SdfNode::InfiniteCylinder { .. }
        | SdfNode::InfiniteCone { .. }
        | SdfNode::Gyroid { .. }
        | SdfNode::Heart { .. }
        | SdfNode::Tube { .. }
        | SdfNode::Barrel { .. }
        | SdfNode::Diamond { .. }
        | SdfNode::ChamferedCube { .. }
        | SdfNode::SchwarzP { .. }
        | SdfNode::Superellipsoid { .. }
        | SdfNode::RoundedX { .. }
        | SdfNode::Pie { .. }
        | SdfNode::Trapezoid { .. }
        | SdfNode::Parallelogram { .. }
        | SdfNode::Tunnel { .. }
        | SdfNode::UnevenCapsule { .. }
        | SdfNode::Egg { .. }
        | SdfNode::ArcShape { .. }
        | SdfNode::Moon { .. }
        | SdfNode::CrossShape { .. }
        | SdfNode::BlobbyCross { .. }
        | SdfNode::ParabolaSegment { .. }
        | SdfNode::RegularPolygon { .. }
        | SdfNode::StarPolygon { .. }
        | SdfNode::Stairs { .. }
        | SdfNode::Helix { .. }
        | SdfNode::Tetrahedron { .. }
        | SdfNode::Dodecahedron { .. }
        | SdfNode::Icosahedron { .. }
        | SdfNode::TruncatedOctahedron { .. }
        | SdfNode::TruncatedIcosahedron { .. }
        | SdfNode::BoxFrame { .. }
        | SdfNode::DiamondSurface { .. }
        | SdfNode::Neovius { .. }
        | SdfNode::Lidinoid { .. }
        | SdfNode::IWP { .. }
        | SdfNode::FRD { .. }
        | SdfNode::FischerKochS { .. }
        | SdfNode::PMY { .. }
        | SdfNode::Circle2D { .. }
        | SdfNode::Rect2D { .. }
        | SdfNode::Segment2D { .. }
        | SdfNode::Polygon2D { .. }
        | SdfNode::RoundedRect2D { .. }
        | SdfNode::Annular2D { .. } => 1.0,

        // Boolean ops: smooth min/max is 1-Lipschitz in (a,b)
        SdfNode::Union { a, b }
        | SdfNode::Intersection { a, b }
        | SdfNode::Subtraction { a, b }
        | SdfNode::SmoothUnion { a, b, .. }
        | SdfNode::SmoothIntersection { a, b, .. }
        | SdfNode::SmoothSubtraction { a, b, .. }
        | SdfNode::ChamferUnion { a, b, .. }
        | SdfNode::ChamferIntersection { a, b, .. }
        | SdfNode::ChamferSubtraction { a, b, .. }
        | SdfNode::StairsUnion { a, b, .. }
        | SdfNode::StairsIntersection { a, b, .. }
        | SdfNode::StairsSubtraction { a, b, .. }
        | SdfNode::XOR { a, b }
        | SdfNode::Morph { a, b, .. }
        | SdfNode::ColumnsUnion { a, b, .. }
        | SdfNode::ColumnsIntersection { a, b, .. }
        | SdfNode::ColumnsSubtraction { a, b, .. }
        | SdfNode::Pipe { a, b, .. }
        | SdfNode::Engrave { a, b, .. }
        | SdfNode::Groove { a, b, .. }
        | SdfNode::Tongue { a, b, .. }
        | SdfNode::ExpSmoothUnion { a, b, .. }
        | SdfNode::ExpSmoothIntersection { a, b, .. }
        | SdfNode::ExpSmoothSubtraction { a, b, .. } => eval_lipschitz(a).max(eval_lipschitz(b)),

        // Distance-preserving transforms
        SdfNode::Translate { child, .. } | SdfNode::Rotate { child, .. } => eval_lipschitz(child),
        // Uniform scale: SDF(p/s)*s, gradient = (1/s)*∇child*s = ∇child
        SdfNode::Scale { child, .. } => eval_lipschitz(child),
        // Non-uniform scale: gradient stretches by max/min factor ratio
        SdfNode::ScaleNonUniform { child, factors } => {
            let s = [factors.x.abs(), factors.y.abs(), factors.z.abs()];
            let min_s = s[0].min(s[1]).min(s[2]).max(1e-6);
            let max_s = s[0].max(s[1]).max(s[2]);
            eval_lipschitz(child) * max_s / min_s
        }
        // Projective transform: use user-provided bound
        SdfNode::ProjectiveTransform {
            child,
            lipschitz_bound,
            ..
        } => eval_lipschitz(child) * lipschitz_bound,
        // Lattice deform: conservative estimate (Jacobian can be large near control points)
        SdfNode::LatticeDeform { child, .. } => eval_lipschitz(child) * 2.0,
        // SDF Skinning: LBS preserves distance (approximately)
        SdfNode::SdfSkinning { child, .. } => eval_lipschitz(child),

        // Modifiers with Jacobian norm ≤ 1
        SdfNode::Round { child, .. }
        | SdfNode::Onion { child, .. }
        | SdfNode::Elongate { child, .. }
        | SdfNode::Mirror { child, .. }
        | SdfNode::OctantMirror { child, .. }
        | SdfNode::Revolution { child, .. }
        | SdfNode::Extrude { child, .. }
        | SdfNode::SweepBezier { child, .. } => eval_lipschitz(child),

        // Repeat: Lipschitz preserved per cell
        SdfNode::RepeatInfinite { child, .. }
        | SdfNode::RepeatFinite { child, .. }
        | SdfNode::PolarRepeat { child, .. } => eval_lipschitz(child),

        // Twist: Jacobian spectral norm = sqrt(1 + strength² * r²)
        // Conservative estimate assumes max XZ radius ≈ 10 units
        SdfNode::Twist { child, strength } => {
            eval_lipschitz(child) * (1.0 + strength * strength * 100.0).sqrt()
        }
        // Bend: similar deformation
        SdfNode::Bend {
            child, curvature, ..
        } => eval_lipschitz(child) * (1.0 + curvature * curvature * 100.0).sqrt(),

        // Noise: adds noise gradient (bounded by amplitude * frequency)
        SdfNode::Noise {
            child,
            amplitude,
            frequency,
            ..
        } => eval_lipschitz(child) + amplitude.abs() * frequency.abs(),
        // Taper: scale varies along Y, gradient stretches
        SdfNode::Taper { child, factor } => eval_lipschitz(child) * (1.0 + factor.abs()),
        // Displacement: procedural perturbation adds gradient
        SdfNode::Displacement { child, strength } => eval_lipschitz(child) + strength.abs(),
        SdfNode::Shear { child, shear } => {
            let max_shear = shear.x.abs().max(shear.y.abs()).max(shear.z.abs());
            eval_lipschitz(child) * (1.0 + max_shear)
        }
        SdfNode::Animated { child, .. } => eval_lipschitz(child),
        SdfNode::IcosahedralSymmetry { child } => eval_lipschitz(child),
        SdfNode::IFS { child, .. } => {
            // IFS can have arbitrary scale factors, conservative estimate
            eval_lipschitz(child) * 2.0
        }
        SdfNode::HeightmapDisplacement {
            child,
            amplitude,
            scale,
            ..
        } => eval_lipschitz(child) + amplitude.abs() * scale.abs(),
        SdfNode::SurfaceRoughness {
            child,
            amplitude,
            frequency,
            ..
        } => eval_lipschitz(child) + amplitude.abs() * frequency.abs(),

        SdfNode::WithMaterial { child, .. } => eval_lipschitz(child),

        #[allow(unreachable_patterns)]
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::eval;

    /// Verify that scalar eval at sampled points falls within interval bounds
    fn check(node: &SdfNode, min: Vec3, max: Vec3, n: usize) {
        let iv = eval_interval(node, Vec3Interval::from_bounds(min, max));
        let step = (max - min) / ((n - 1) as f32);
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let p = min + Vec3::new(ix as f32, iy as f32, iz as f32) * step;
                    let d = eval(node, p);
                    assert!(
                        d >= iv.lo - 1e-4 && d <= iv.hi + 1e-4,
                        "d={d} not in [{lo}, {hi}] at {p:?}",
                        lo = iv.lo,
                        hi = iv.hi,
                    );
                }
            }
        }
    }

    #[test]
    fn test_interval_ops() {
        let a = Interval::new(1.0, 3.0);
        let b = Interval::new(2.0, 5.0);
        assert_eq!((a + b).lo, 3.0);
        assert_eq!((a + b).hi, 8.0);
        assert_eq!((a - b).lo, -4.0);
        assert_eq!((a - b).hi, 1.0);
        assert_eq!((-a).lo, -3.0);
        assert_eq!((-a).hi, -1.0);
    }

    #[test]
    fn test_interval_abs() {
        assert_eq!(Interval::new(1.0, 3.0).abs().lo, 1.0);
        assert_eq!(Interval::new(-3.0, -1.0).abs().hi, 3.0);
        assert_eq!(Interval::new(-2.0, 3.0).abs().lo, 0.0);
    }

    #[test]
    fn test_interval_sqr() {
        let s = Interval::new(-2.0, 3.0).sqr();
        assert_eq!(s.lo, 0.0);
        assert_eq!(s.hi, 9.0);
    }

    #[test]
    fn test_sphere() {
        check(
            &SdfNode::sphere(1.0),
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            5,
        );
    }

    #[test]
    fn test_box3d() {
        check(
            &SdfNode::box3d(1.0, 0.5, 0.75),
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            5,
        );
    }

    #[test]
    fn test_cylinder() {
        check(
            &SdfNode::cylinder(0.5, 1.0),
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            5,
        );
    }

    #[test]
    fn test_torus() {
        check(
            &SdfNode::torus(1.0, 0.25),
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            5,
        );
    }

    #[test]
    fn test_plane() {
        check(
            &SdfNode::plane(Vec3::Y, 0.0),
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            5,
        );
    }

    #[test]
    fn test_rounded_box() {
        check(
            &SdfNode::rounded_box(1.0, 0.5, 0.75, 0.1),
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            4,
        );
    }

    #[test]
    fn test_union() {
        let node = SdfNode::sphere(1.0)
            .translate(1.0, 0.0, 0.0)
            .union(SdfNode::sphere(1.0).translate(-1.0, 0.0, 0.0));
        check(&node, Vec3::splat(-3.0), Vec3::splat(3.0), 4);
    }

    #[test]
    fn test_smooth_union() {
        let node = SdfNode::sphere(1.0)
            .translate(0.5, 0.0, 0.0)
            .smooth_union(SdfNode::sphere(1.0).translate(-0.5, 0.0, 0.0), 0.5);
        check(&node, Vec3::splat(-3.0), Vec3::splat(3.0), 4);
    }

    #[test]
    fn test_subtraction() {
        let node = SdfNode::sphere(1.5).subtract(SdfNode::sphere(1.0).translate(0.5, 0.0, 0.0));
        check(&node, Vec3::splat(-3.0), Vec3::splat(3.0), 4);
    }

    #[test]
    fn test_translate() {
        check(
            &SdfNode::sphere(1.0).translate(2.0, 0.0, 0.0),
            Vec3::new(0.0, -2.0, -2.0),
            Vec3::new(4.0, 2.0, 2.0),
            4,
        );
    }

    #[test]
    fn test_rotate() {
        let node = SdfNode::box3d(1.0, 0.5, 0.3).rotate(Quat::from_rotation_y(0.7));
        check(&node, Vec3::splat(-2.0), Vec3::splat(2.0), 4);
    }

    #[test]
    fn test_scale() {
        check(
            &SdfNode::sphere(1.0).scale(2.0),
            Vec3::splat(-4.0),
            Vec3::splat(4.0),
            4,
        );
    }

    #[test]
    fn test_round() {
        let node = SdfNode::box3d(1.0, 1.0, 1.0).round(0.2);
        check(&node, Vec3::splat(-2.0), Vec3::splat(2.0), 4);
    }

    #[test]
    fn test_onion() {
        let node = SdfNode::sphere(1.0).onion(0.1);
        check(&node, Vec3::splat(-2.0), Vec3::splat(2.0), 4);
    }

    #[test]
    fn test_revolution() {
        let node = SdfNode::Revolution {
            child: std::sync::Arc::new(SdfNode::sphere(0.3)),
            offset: 1.0,
        };
        check(&node, Vec3::splat(-2.0), Vec3::splat(2.0), 4);
    }

    #[test]
    fn test_extrude() {
        let node = SdfNode::Extrude {
            child: std::sync::Arc::new(SdfNode::sphere(0.5)),
            half_height: 1.0,
        };
        check(&node, Vec3::splat(-2.0), Vec3::splat(2.0), 4);
    }

    #[test]
    fn test_mirror() {
        let node = SdfNode::box3d(1.0, 0.5, 0.5).mirror(true, false, false);
        check(&node, Vec3::splat(-2.0), Vec3::splat(2.0), 4);
    }

    #[test]
    fn test_pruning() {
        let sphere = SdfNode::sphere(1.0);
        // Far region: entirely positive
        let far = Vec3Interval::from_bounds(Vec3::splat(5.0), Vec3::splat(6.0));
        assert!(eval_interval(&sphere, far).is_positive());
        // Inside region: entirely negative
        let inside = Vec3Interval::from_bounds(Vec3::splat(-0.1), Vec3::splat(0.1));
        assert!(eval_interval(&sphere, inside).is_negative());
    }

    #[test]
    fn test_chamfer_blend() {
        let node = SdfNode::sphere(1.0).chamfer_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.3);
        check(&node, Vec3::splat(-2.0), Vec3::splat(2.0), 4);
    }

    #[test]
    fn test_stairs_blend() {
        let node = SdfNode::sphere(1.0).stairs_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.3, 4.0);
        check(&node, Vec3::splat(-2.0), Vec3::splat(2.0), 4);
    }

    #[test]
    fn test_conservative_primitives() {
        // These use bounding sphere — should still contain all scalar values
        check(
            &SdfNode::cone(0.5, 1.0),
            Vec3::splat(-3.0),
            Vec3::splat(3.0),
            4,
        );
        check(
            &SdfNode::octahedron(1.0),
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            4,
        );
        check(
            &SdfNode::pyramid(1.0),
            Vec3::splat(-3.0),
            Vec3::splat(3.0),
            4,
        );
    }

    // ============ Lipschitz tests ============

    /// Check that finite-difference gradient ≤ L at sampled points
    fn check_lipschitz(node: &SdfNode, min: Vec3, max: Vec3, n: usize) {
        let lip = eval_lipschitz(node);
        assert!(lip >= 1.0, "Lipschitz should be >= 1.0, got {lip}");
        let eps = 1e-3;
        let step = (max - min) / ((n - 1) as f32);
        let mut max_grad: f32 = 0.0;
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let p = min + step * Vec3::new(ix as f32, iy as f32, iz as f32);
                    let d = eval(node, p);
                    // Finite-difference gradient (central differences)
                    let gx = (eval(node, p + Vec3::X * eps) - eval(node, p - Vec3::X * eps))
                        / (2.0 * eps);
                    let gy = (eval(node, p + Vec3::Y * eps) - eval(node, p - Vec3::Y * eps))
                        / (2.0 * eps);
                    let gz = (eval(node, p + Vec3::Z * eps) - eval(node, p - Vec3::Z * eps))
                        / (2.0 * eps);
                    let grad_mag = (gx * gx + gy * gy + gz * gz).sqrt();
                    max_grad = max_grad.max(grad_mag);
                    // Allow 5% tolerance for finite-difference error
                    assert!(
                        grad_mag <= lip * 1.05 + 0.01,
                        "|∇SDF|={grad_mag:.4} > L={lip:.4} at {p:?} (d={d:.4})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_lipschitz_sphere() {
        check_lipschitz(
            &SdfNode::sphere(1.0),
            Vec3::splat(-3.0),
            Vec3::splat(3.0),
            6,
        );
    }

    #[test]
    fn test_lipschitz_box() {
        check_lipschitz(
            &SdfNode::box3d(1.0, 0.5, 0.75),
            Vec3::splat(-3.0),
            Vec3::splat(3.0),
            6,
        );
    }

    #[test]
    fn test_lipschitz_union() {
        let node = SdfNode::sphere(1.0)
            .translate(1.0, 0.0, 0.0)
            .union(SdfNode::sphere(1.0).translate(-1.0, 0.0, 0.0));
        assert_eq!(eval_lipschitz(&node), 1.0);
        check_lipschitz(&node, Vec3::splat(-3.0), Vec3::splat(3.0), 5);
    }

    #[test]
    fn test_lipschitz_smooth_union() {
        let node = SdfNode::sphere(1.0)
            .translate(0.5, 0.0, 0.0)
            .smooth_union(SdfNode::sphere(1.0).translate(-0.5, 0.0, 0.0), 0.5);
        assert_eq!(eval_lipschitz(&node), 1.0);
        check_lipschitz(&node, Vec3::splat(-3.0), Vec3::splat(3.0), 5);
    }

    #[test]
    fn test_lipschitz_scale() {
        let node = SdfNode::sphere(1.0).scale(2.0);
        assert_eq!(eval_lipschitz(&node), 1.0);
    }

    #[test]
    fn test_lipschitz_non_uniform_scale() {
        let node = SdfNode::ScaleNonUniform {
            child: std::sync::Arc::new(SdfNode::sphere(1.0)),
            factors: Vec3::new(2.0, 1.0, 1.0),
        };
        assert_eq!(eval_lipschitz(&node), 2.0);
    }

    #[test]
    fn test_lipschitz_noise() {
        let node = SdfNode::Noise {
            child: std::sync::Arc::new(SdfNode::sphere(1.0)),
            amplitude: 0.2,
            frequency: 3.0,
            seed: 0,
        };
        // L = 1.0 + 0.2 * 3.0 = 1.6
        assert!((eval_lipschitz(&node) - 1.6).abs() < 1e-5);
    }

    #[test]
    fn test_lipschitz_twist() {
        let node = SdfNode::Twist {
            child: std::sync::Arc::new(SdfNode::sphere(1.0)),
            strength: 0.5,
        };
        // L = 1.0 * sqrt(1 + 0.25 * 100) = sqrt(26)
        let expected = 26.0_f32.sqrt();
        assert!((eval_lipschitz(&node) - expected).abs() < 1e-3);
    }

    #[test]
    fn test_lipschitz_round() {
        let node = SdfNode::box3d(1.0, 1.0, 1.0).round(0.2);
        assert_eq!(eval_lipschitz(&node), 1.0);
        check_lipschitz(&node, Vec3::splat(-3.0), Vec3::splat(3.0), 5);
    }
}
