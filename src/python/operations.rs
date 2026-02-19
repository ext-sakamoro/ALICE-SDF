//! Boolean and blending operations for PySdfNode.

use pyo3::prelude::*;

use super::node::PySdfNode;

#[pymethods]
impl PySdfNode {
    /// Union with another shape
    fn union(&self, other: &PySdfNode) -> Self {
        PySdfNode {
            inner: self.inner.clone().union(other.inner.clone()),
        }
    }

    /// Intersection with another shape
    fn intersection(&self, other: &PySdfNode) -> Self {
        PySdfNode {
            inner: self.inner.clone().intersection(other.inner.clone()),
        }
    }

    /// Subtract another shape
    fn subtract(&self, other: &PySdfNode) -> Self {
        PySdfNode {
            inner: self.inner.clone().subtract(other.inner.clone()),
        }
    }

    /// Smooth union with another shape
    fn smooth_union(&self, other: &PySdfNode, k: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().smooth_union(other.inner.clone(), k),
        }
    }

    /// Smooth intersection with another shape
    fn smooth_intersection(&self, other: &PySdfNode, k: f32) -> Self {
        PySdfNode {
            inner: self
                .inner
                .clone()
                .smooth_intersection(other.inner.clone(), k),
        }
    }

    /// Smooth subtraction
    fn smooth_subtract(&self, other: &PySdfNode, k: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().smooth_subtract(other.inner.clone(), k),
        }
    }

    /// Exponential smooth union with another shape
    fn exp_smooth_union(&self, other: &PySdfNode, k: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().exp_smooth_union(other.inner.clone(), k),
        }
    }

    /// Exponential smooth intersection with another shape
    fn exp_smooth_intersection(&self, other: &PySdfNode, k: f32) -> Self {
        PySdfNode {
            inner: self
                .inner
                .clone()
                .exp_smooth_intersection(other.inner.clone(), k),
        }
    }

    /// Exponential smooth subtraction of another shape
    fn exp_smooth_subtract(&self, other: &PySdfNode, k: f32) -> Self {
        PySdfNode {
            inner: self
                .inner
                .clone()
                .exp_smooth_subtract(other.inner.clone(), k),
        }
    }

    /// XOR (symmetric difference) with another shape
    fn xor(&self, other: &PySdfNode) -> Self {
        PySdfNode {
            inner: self.inner.clone().xor(other.inner.clone()),
        }
    }

    /// Morph between two shapes (t=0: self, t=1: other)
    fn morph(&self, other: &PySdfNode, t: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().morph(other.inner.clone(), t),
        }
    }

    /// Columns union with another shape
    fn columns_union(&self, other: &PySdfNode, r: f32, n: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().columns_union(other.inner.clone(), r, n),
        }
    }

    /// Columns intersection with another shape
    fn columns_intersection(&self, other: &PySdfNode, r: f32, n: f32) -> Self {
        PySdfNode {
            inner: self
                .inner
                .clone()
                .columns_intersection(other.inner.clone(), r, n),
        }
    }

    /// Columns subtraction of another shape
    fn columns_subtract(&self, other: &PySdfNode, r: f32, n: f32) -> Self {
        PySdfNode {
            inner: self
                .inner
                .clone()
                .columns_subtract(other.inner.clone(), r, n),
        }
    }

    /// Pipe operation with another shape
    fn pipe(&self, other: &PySdfNode, r: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().pipe(other.inner.clone(), r),
        }
    }

    /// Engrave another shape into this one
    fn engrave(&self, other: &PySdfNode, r: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().engrave(other.inner.clone(), r),
        }
    }

    /// Cut a groove of another shape into this one
    fn groove(&self, other: &PySdfNode, ra: f32, rb: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().groove(other.inner.clone(), ra, rb),
        }
    }

    /// Add a tongue protrusion of another shape
    fn tongue(&self, other: &PySdfNode, ra: f32, rb: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().tongue(other.inner.clone(), ra, rb),
        }
    }

    /// Chamfer union with another shape
    fn chamfer_union(&self, other: &PySdfNode, r: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().chamfer_union(other.inner.clone(), r),
        }
    }

    /// Chamfer intersection with another shape
    fn chamfer_intersection(&self, other: &PySdfNode, r: f32) -> Self {
        PySdfNode {
            inner: self
                .inner
                .clone()
                .chamfer_intersection(other.inner.clone(), r),
        }
    }

    /// Chamfer subtraction of another shape
    fn chamfer_subtract(&self, other: &PySdfNode, r: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().chamfer_subtract(other.inner.clone(), r),
        }
    }

    /// Stairs union with another shape
    fn stairs_union(&self, other: &PySdfNode, r: f32, n: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().stairs_union(other.inner.clone(), r, n),
        }
    }

    /// Stairs intersection with another shape
    fn stairs_intersection(&self, other: &PySdfNode, r: f32, n: f32) -> Self {
        PySdfNode {
            inner: self
                .inner
                .clone()
                .stairs_intersection(other.inner.clone(), r, n),
        }
    }

    /// Stairs subtraction of another shape
    fn stairs_subtract(&self, other: &PySdfNode, r: f32, n: f32) -> Self {
        PySdfNode {
            inner: self
                .inner
                .clone()
                .stairs_subtract(other.inner.clone(), r, n),
        }
    }
}
