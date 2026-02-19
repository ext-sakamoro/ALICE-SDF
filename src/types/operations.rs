//! Boolean and blending operations for SdfNode
//!
//! Author: Moroya Sakamoto

use std::sync::Arc;

use super::SdfNode;

impl SdfNode {
    // === Operation methods ===

    /// Union with another shape
    #[inline]
    pub fn union(self, other: SdfNode) -> Self {
        SdfNode::Union {
            a: Arc::new(self),
            b: Arc::new(other),
        }
    }

    /// Intersection with another shape
    #[inline]
    pub fn intersection(self, other: SdfNode) -> Self {
        SdfNode::Intersection {
            a: Arc::new(self),
            b: Arc::new(other),
        }
    }

    /// Subtract another shape from this one
    #[inline]
    pub fn subtract(self, other: SdfNode) -> Self {
        SdfNode::Subtraction {
            a: Arc::new(self),
            b: Arc::new(other),
        }
    }

    /// Smooth union with another shape
    #[inline]
    pub fn smooth_union(self, other: SdfNode, k: f32) -> Self {
        SdfNode::SmoothUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Smooth intersection with another shape
    #[inline]
    pub fn smooth_intersection(self, other: SdfNode, k: f32) -> Self {
        SdfNode::SmoothIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Smooth subtraction of another shape
    #[inline]
    pub fn smooth_subtract(self, other: SdfNode, k: f32) -> Self {
        SdfNode::SmoothSubtraction {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Chamfer union with another shape
    #[inline]
    pub fn chamfer_union(self, other: SdfNode, r: f32) -> Self {
        SdfNode::ChamferUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
        }
    }

    /// Chamfer intersection with another shape
    #[inline]
    pub fn chamfer_intersection(self, other: SdfNode, r: f32) -> Self {
        SdfNode::ChamferIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
        }
    }

    /// Chamfer subtraction of another shape
    #[inline]
    pub fn chamfer_subtract(self, other: SdfNode, r: f32) -> Self {
        SdfNode::ChamferSubtraction {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
        }
    }

    /// Stairs union with another shape
    #[inline]
    pub fn stairs_union(self, other: SdfNode, r: f32, n: f32) -> Self {
        SdfNode::StairsUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// Stairs intersection with another shape
    #[inline]
    pub fn stairs_intersection(self, other: SdfNode, r: f32, n: f32) -> Self {
        SdfNode::StairsIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// Stairs subtraction of another shape
    #[inline]
    pub fn stairs_subtract(self, other: SdfNode, r: f32, n: f32) -> Self {
        SdfNode::StairsSubtraction {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// XOR (symmetric difference) with another shape
    #[inline]
    pub fn xor(self, other: SdfNode) -> Self {
        SdfNode::XOR {
            a: Arc::new(self),
            b: Arc::new(other),
        }
    }

    /// Morph with another shape
    #[inline]
    pub fn morph(self, other: SdfNode, t: f32) -> Self {
        SdfNode::Morph {
            a: Arc::new(self),
            b: Arc::new(other),
            t,
        }
    }

    /// Columns union with another shape
    #[inline]
    pub fn columns_union(self, other: SdfNode, r: f32, n: f32) -> Self {
        SdfNode::ColumnsUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// Columns intersection with another shape
    #[inline]
    pub fn columns_intersection(self, other: SdfNode, r: f32, n: f32) -> Self {
        SdfNode::ColumnsIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// Columns subtraction of another shape
    #[inline]
    pub fn columns_subtract(self, other: SdfNode, r: f32, n: f32) -> Self {
        SdfNode::ColumnsSubtraction {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// Pipe operation with another shape
    #[inline]
    pub fn pipe(self, other: SdfNode, r: f32) -> Self {
        SdfNode::Pipe {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
        }
    }

    /// Engrave another shape into this one
    #[inline]
    pub fn engrave(self, other: SdfNode, r: f32) -> Self {
        SdfNode::Engrave {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
        }
    }

    /// Cut a groove of another shape into this one
    #[inline]
    pub fn groove(self, other: SdfNode, ra: f32, rb: f32) -> Self {
        SdfNode::Groove {
            a: Arc::new(self),
            b: Arc::new(other),
            ra,
            rb,
        }
    }

    /// Add a tongue protrusion of another shape
    #[inline]
    pub fn tongue(self, other: SdfNode, ra: f32, rb: f32) -> Self {
        SdfNode::Tongue {
            a: Arc::new(self),
            b: Arc::new(other),
            ra,
            rb,
        }
    }

    /// Exponential smooth union with another shape
    #[inline]
    pub fn exp_smooth_union(self, other: SdfNode, k: f32) -> Self {
        SdfNode::ExpSmoothUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Exponential smooth intersection with another shape
    #[inline]
    pub fn exp_smooth_intersection(self, other: SdfNode, k: f32) -> Self {
        SdfNode::ExpSmoothIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Exponential smooth subtraction of another shape
    #[inline]
    pub fn exp_smooth_subtract(self, other: SdfNode, k: f32) -> Self {
        SdfNode::ExpSmoothSubtraction {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }
}
