//! Boolean and blending operations for SdfNode
//!
//! Author: Moroya Sakamoto

use std::sync::Arc;

use super::SdfNode;

impl SdfNode {
    // === Operation methods ===

    /// Union with another shape
    #[must_use]
    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self::Union {
            a: Arc::new(self),
            b: Arc::new(other),
        }
    }

    /// Intersection with another shape
    #[must_use]
    #[inline]
    pub fn intersection(self, other: Self) -> Self {
        Self::Intersection {
            a: Arc::new(self),
            b: Arc::new(other),
        }
    }

    /// Subtract another shape from this one
    #[must_use]
    #[inline]
    pub fn subtract(self, other: Self) -> Self {
        Self::Subtraction {
            a: Arc::new(self),
            b: Arc::new(other),
        }
    }

    /// Smooth union with another shape
    #[must_use]
    #[inline]
    pub fn smooth_union(self, other: Self, k: f32) -> Self {
        Self::SmoothUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Smooth intersection with another shape
    #[must_use]
    #[inline]
    pub fn smooth_intersection(self, other: Self, k: f32) -> Self {
        Self::SmoothIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Smooth subtraction of another shape
    #[must_use]
    #[inline]
    pub fn smooth_subtract(self, other: Self, k: f32) -> Self {
        Self::SmoothSubtraction {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Chamfer union with another shape
    #[must_use]
    #[inline]
    pub fn chamfer_union(self, other: Self, r: f32) -> Self {
        Self::ChamferUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
        }
    }

    /// Chamfer intersection with another shape
    #[must_use]
    #[inline]
    pub fn chamfer_intersection(self, other: Self, r: f32) -> Self {
        Self::ChamferIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
        }
    }

    /// Chamfer subtraction of another shape
    #[must_use]
    #[inline]
    pub fn chamfer_subtract(self, other: Self, r: f32) -> Self {
        Self::ChamferSubtraction {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
        }
    }

    /// Stairs union with another shape
    #[must_use]
    #[inline]
    pub fn stairs_union(self, other: Self, r: f32, n: f32) -> Self {
        Self::StairsUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// Stairs intersection with another shape
    #[must_use]
    #[inline]
    pub fn stairs_intersection(self, other: Self, r: f32, n: f32) -> Self {
        Self::StairsIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// Stairs subtraction of another shape
    #[must_use]
    #[inline]
    pub fn stairs_subtract(self, other: Self, r: f32, n: f32) -> Self {
        Self::StairsSubtraction {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// XOR (symmetric difference) with another shape
    #[must_use]
    #[inline]
    pub fn xor(self, other: Self) -> Self {
        Self::XOR {
            a: Arc::new(self),
            b: Arc::new(other),
        }
    }

    /// Morph with another shape
    #[must_use]
    #[inline]
    pub fn morph(self, other: Self, t: f32) -> Self {
        Self::Morph {
            a: Arc::new(self),
            b: Arc::new(other),
            t,
        }
    }

    /// Columns union with another shape
    #[must_use]
    #[inline]
    pub fn columns_union(self, other: Self, r: f32, n: f32) -> Self {
        Self::ColumnsUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// Columns intersection with another shape
    #[must_use]
    #[inline]
    pub fn columns_intersection(self, other: Self, r: f32, n: f32) -> Self {
        Self::ColumnsIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// Columns subtraction of another shape
    #[must_use]
    #[inline]
    pub fn columns_subtract(self, other: Self, r: f32, n: f32) -> Self {
        Self::ColumnsSubtraction {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// Pipe operation with another shape
    #[must_use]
    #[inline]
    pub fn pipe(self, other: Self, r: f32) -> Self {
        Self::Pipe {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
        }
    }

    /// Engrave another shape into this one
    #[must_use]
    #[inline]
    pub fn engrave(self, other: Self, r: f32) -> Self {
        Self::Engrave {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
        }
    }

    /// Cut a groove of another shape into this one
    #[must_use]
    #[inline]
    pub fn groove(self, other: Self, ra: f32, rb: f32) -> Self {
        Self::Groove {
            a: Arc::new(self),
            b: Arc::new(other),
            ra,
            rb,
        }
    }

    /// Add a tongue protrusion of another shape
    #[must_use]
    #[inline]
    pub fn tongue(self, other: Self, ra: f32, rb: f32) -> Self {
        Self::Tongue {
            a: Arc::new(self),
            b: Arc::new(other),
            ra,
            rb,
        }
    }

    /// Exponential smooth union with another shape
    #[must_use]
    #[inline]
    pub fn exp_smooth_union(self, other: Self, k: f32) -> Self {
        Self::ExpSmoothUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Exponential smooth intersection with another shape
    #[must_use]
    #[inline]
    pub fn exp_smooth_intersection(self, other: Self, k: f32) -> Self {
        Self::ExpSmoothIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Exponential smooth subtraction of another shape
    #[must_use]
    #[inline]
    pub fn exp_smooth_subtract(self, other: Self, k: f32) -> Self {
        Self::ExpSmoothSubtraction {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }
}
