//! Forward-mode Automatic Differentiation via Dual Numbers
//!
//! Provides `Dual` and `Dual3` types for exact gradient computation
//! without finite-difference approximation. All arithmetic operations
//! propagate derivatives automatically through the chain rule.
//!
//! # Usage
//!
//! ```rust
//! use alice_sdf::autodiff::{Dual3, eval_with_gradient};
//! use alice_sdf::types::SdfNode;
//! use glam::Vec3;
//!
//! let sphere = SdfNode::sphere(1.0);
//! let (distance, gradient) = eval_with_gradient(&sphere, Vec3::new(2.0, 0.0, 0.0));
//! // distance ≈ 1.0, gradient ≈ (1, 0, 0)
//! ```
//!
//! Author: Moroya Sakamoto

use crate::eval::{eval, eval_gradient};
use crate::types::SdfNode;
use glam::Vec3;

// ── Dual Number (1D) ─────────────────────────────────────────

/// 1D dual number: `val + eps * dot`.
///
/// Represents a value and its derivative with respect to a single variable.
#[derive(Debug, Clone, Copy)]
pub struct Dual {
    /// Function value.
    pub val: f32,
    /// Derivative with respect to the tracked variable.
    pub dot: f32,
}

impl Dual {
    /// Constant (derivative = 0).
    #[inline(always)]
    pub fn constant(val: f32) -> Self {
        Self { val, dot: 0.0 }
    }

    /// Variable (derivative = 1).
    #[inline(always)]
    pub fn variable(val: f32) -> Self {
        Self { val, dot: 1.0 }
    }

    /// Absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        if self.val >= 0.0 {
            self
        } else {
            Self {
                val: -self.val,
                dot: -self.dot,
            }
        }
    }

    /// Square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        let r = self.val.max(0.0).sqrt();
        let d = if r > 1e-10 { self.dot / (2.0 * r) } else { 0.0 };
        Self { val: r, dot: d }
    }

    /// Minimum of two dual numbers.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        if self.val <= other.val {
            self
        } else {
            other
        }
    }

    /// Maximum of two dual numbers.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        if self.val >= other.val {
            self
        } else {
            other
        }
    }

    /// Clamp to scalar range.
    #[inline(always)]
    pub fn clamp(self, lo: f32, hi: f32) -> Self {
        if self.val < lo {
            Self::constant(lo)
        } else if self.val > hi {
            Self::constant(hi)
        } else {
            self
        }
    }

    /// Sine.
    #[inline(always)]
    pub fn sin(self) -> Self {
        Self {
            val: self.val.sin(),
            dot: self.dot * self.val.cos(),
        }
    }

    /// Cosine.
    #[inline(always)]
    pub fn cos(self) -> Self {
        Self {
            val: self.val.cos(),
            dot: -self.dot * self.val.sin(),
        }
    }
}

impl std::ops::Add for Dual {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            val: self.val + rhs.val,
            dot: self.dot + rhs.dot,
        }
    }
}

impl std::ops::Sub for Dual {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            val: self.val - rhs.val,
            dot: self.dot - rhs.dot,
        }
    }
}

impl std::ops::Mul for Dual {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            val: self.val * rhs.val,
            dot: self.val * rhs.dot + self.dot * rhs.val,
        }
    }
}

impl std::ops::Mul<f32> for Dual {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self {
            val: self.val * rhs,
            dot: self.dot * rhs,
        }
    }
}

impl std::ops::Div for Dual {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let inv = 1.0 / rhs.val;
        Self {
            val: self.val * inv,
            dot: (self.dot * rhs.val - self.val * rhs.dot) * inv * inv,
        }
    }
}

impl std::ops::Neg for Dual {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            val: -self.val,
            dot: -self.dot,
        }
    }
}

// ── Dual3 (3D gradient) ──────────────────────────────────────

/// 3D dual number: value + gradient (∂f/∂x, ∂f/∂y, ∂f/∂z).
///
/// Tracks partial derivatives with respect to all three spatial coordinates
/// simultaneously, enabling exact gradient computation in a single evaluation.
#[derive(Debug, Clone, Copy)]
pub struct Dual3 {
    /// Function value.
    pub val: f32,
    /// Partial derivative ∂f/∂x.
    pub dx: f32,
    /// Partial derivative ∂f/∂y.
    pub dy: f32,
    /// Partial derivative ∂f/∂z.
    pub dz: f32,
}

impl Dual3 {
    /// Constant (all partial derivatives = 0).
    #[inline(always)]
    pub fn constant(val: f32) -> Self {
        Self {
            val,
            dx: 0.0,
            dy: 0.0,
            dz: 0.0,
        }
    }

    /// Create from value and gradient vector.
    #[inline(always)]
    pub fn from_val_grad(val: f32, grad: Vec3) -> Self {
        Self {
            val,
            dx: grad.x,
            dy: grad.y,
            dz: grad.z,
        }
    }

    /// Extract gradient as Vec3.
    #[inline(always)]
    pub fn gradient(self) -> Vec3 {
        Vec3::new(self.dx, self.dy, self.dz)
    }

    /// Gradient magnitude.
    #[inline(always)]
    pub fn gradient_magnitude(self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy + self.dz * self.dz).sqrt()
    }

    /// Absolute value: |f|, ∇|f| = sign(f) * ∇f.
    #[inline(always)]
    pub fn abs(self) -> Self {
        if self.val >= 0.0 {
            self
        } else {
            Self {
                val: -self.val,
                dx: -self.dx,
                dy: -self.dy,
                dz: -self.dz,
            }
        }
    }

    /// Square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        let r = self.val.max(0.0).sqrt();
        if r > 1e-10 {
            let inv2r = 0.5 / r;
            Self {
                val: r,
                dx: self.dx * inv2r,
                dy: self.dy * inv2r,
                dz: self.dz * inv2r,
            }
        } else {
            Self {
                val: r,
                dx: 0.0,
                dy: 0.0,
                dz: 0.0,
            }
        }
    }

    /// Minimum of two Dual3 values.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        if self.val <= other.val {
            self
        } else {
            other
        }
    }

    /// Maximum of two Dual3 values.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        if self.val >= other.val {
            self
        } else {
            other
        }
    }

    /// Clamp to scalar range.
    #[inline(always)]
    pub fn clamp(self, lo: f32, hi: f32) -> Self {
        if self.val < lo {
            Self::constant(lo)
        } else if self.val > hi {
            Self::constant(hi)
        } else {
            self
        }
    }

    /// 3D length: sqrt(x² + y² + z²) of a Dual3 triple.
    #[inline]
    pub fn length3(x: Dual3, y: Dual3, z: Dual3) -> Dual3 {
        (x * x + y * y + z * z).sqrt()
    }

    /// 2D length: sqrt(x² + z²) of a Dual3 pair.
    #[inline]
    pub fn length2(x: Dual3, z: Dual3) -> Dual3 {
        (x * x + z * z).sqrt()
    }
}

impl std::ops::Add for Dual3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            val: self.val + rhs.val,
            dx: self.dx + rhs.dx,
            dy: self.dy + rhs.dy,
            dz: self.dz + rhs.dz,
        }
    }
}

impl std::ops::Sub for Dual3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            val: self.val - rhs.val,
            dx: self.dx - rhs.dx,
            dy: self.dy - rhs.dy,
            dz: self.dz - rhs.dz,
        }
    }
}

impl std::ops::Mul for Dual3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            val: self.val * rhs.val,
            dx: self.val * rhs.dx + self.dx * rhs.val,
            dy: self.val * rhs.dy + self.dy * rhs.val,
            dz: self.val * rhs.dz + self.dz * rhs.val,
        }
    }
}

impl std::ops::Mul<f32> for Dual3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self {
            val: self.val * rhs,
            dx: self.dx * rhs,
            dy: self.dy * rhs,
            dz: self.dz * rhs,
        }
    }
}

impl std::ops::Div<f32> for Dual3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        let inv = 1.0 / rhs;
        Self {
            val: self.val * inv,
            dx: self.dx * inv,
            dy: self.dy * inv,
            dz: self.dz * inv,
        }
    }
}

impl std::ops::Neg for Dual3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            val: -self.val,
            dx: -self.dx,
            dy: -self.dy,
            dz: -self.dz,
        }
    }
}

// ── SDF evaluation with gradient ─────────────────────────────

/// Evaluate SDF value and gradient simultaneously.
///
/// Returns `(distance, gradient)` using the analytic gradient engine.
/// More convenient than calling `eval()` and `eval_gradient()` separately.
#[inline]
pub fn eval_with_gradient(node: &SdfNode, point: Vec3) -> (f32, Vec3) {
    let val = eval(node, point);
    let grad = eval_gradient(node, point);
    (val, grad)
}

/// Evaluate SDF and return as a `Dual3` combining value and gradient.
#[inline]
pub fn eval_dual3(node: &SdfNode, point: Vec3) -> Dual3 {
    let (val, grad) = eval_with_gradient(node, point);
    Dual3::from_val_grad(val, grad)
}

/// Compute the Hessian (second derivatives) at a point via finite differences
/// on the gradient.
///
/// Returns a 3x3 symmetric matrix as `[dxx, dyy, dzz, dxy, dxz, dyz]`.
pub fn eval_hessian(node: &SdfNode, point: Vec3, epsilon: f32) -> [f32; 6] {
    let gx_pos = eval_gradient(node, point + Vec3::X * epsilon);
    let gx_neg = eval_gradient(node, point - Vec3::X * epsilon);
    let gy_pos = eval_gradient(node, point + Vec3::Y * epsilon);
    let gy_neg = eval_gradient(node, point - Vec3::Y * epsilon);
    let gz_pos = eval_gradient(node, point + Vec3::Z * epsilon);
    let gz_neg = eval_gradient(node, point - Vec3::Z * epsilon);

    let inv_2e = 0.5 / epsilon;
    [
        (gx_pos.x - gx_neg.x) * inv_2e, // dxx
        (gy_pos.y - gy_neg.y) * inv_2e, // dyy
        (gz_pos.z - gz_neg.z) * inv_2e, // dzz
        (gx_pos.y - gx_neg.y) * inv_2e, // dxy
        (gx_pos.z - gx_neg.z) * inv_2e, // dxz
        (gy_pos.z - gy_neg.z) * inv_2e, // dyz
    ]
}

/// Compute mean curvature at a point on the surface.
///
/// Mean curvature H = div(∇f / |∇f|) = (∇²f - ∇f·H·∇f / |∇f|²) / |∇f|
/// Approximated via Laplacian: H ≈ (dxx + dyy + dzz) / |∇f|.
pub fn mean_curvature(node: &SdfNode, point: Vec3, epsilon: f32) -> f32 {
    let grad = eval_gradient(node, point);
    let grad_len = grad.length();
    if grad_len < 1e-10 {
        return 0.0;
    }
    let h = eval_hessian(node, point, epsilon);
    let laplacian = h[0] + h[1] + h[2]; // dxx + dyy + dzz
    laplacian / grad_len
}

// ── Dual3 primitives ─────────────────────────────────────────

/// Evaluate sphere SDF as Dual3: f = |p| - r.
#[inline]
pub fn dual3_sphere(px: Dual3, py: Dual3, pz: Dual3, radius: f32) -> Dual3 {
    Dual3::length3(px, py, pz) - Dual3::constant(radius)
}

/// Evaluate box SDF as Dual3.
#[inline]
pub fn dual3_box(px: Dual3, py: Dual3, pz: Dual3, half: Vec3) -> Dual3 {
    let qx = px.abs() - Dual3::constant(half.x);
    let qy = py.abs() - Dual3::constant(half.y);
    let qz = pz.abs() - Dual3::constant(half.z);
    let zero = Dual3::constant(0.0);
    let outer = Dual3::length3(qx.max(zero), qy.max(zero), qz.max(zero));
    let inner = qx.max(qy).max(qz).min(zero);
    outer + inner
}

/// Evaluate torus SDF as Dual3: f = |q| - r2, q = (|p.xz| - R, p.y).
#[inline]
pub fn dual3_torus(px: Dual3, py: Dual3, pz: Dual3, major_r: f32, minor_r: f32) -> Dual3 {
    let qx = Dual3::length2(px, pz) - Dual3::constant(major_r);
    Dual3::length2(qx, py) - Dual3::constant(minor_r)
}

/// Evaluate plane SDF as Dual3: f = dot(p, n) - d.
#[inline]
pub fn dual3_plane(px: Dual3, py: Dual3, pz: Dual3, normal: Vec3, distance: f32) -> Dual3 {
    px * normal.x + py * normal.y + pz * normal.z - Dual3::constant(distance)
}

/// Create Dual3 seed values for a point (x tracks ∂/∂x, y tracks ∂/∂y, z tracks ∂/∂z).
#[inline(always)]
pub fn dual3_point(point: Vec3) -> (Dual3, Dual3, Dual3) {
    (
        Dual3 {
            val: point.x,
            dx: 1.0,
            dy: 0.0,
            dz: 0.0,
        },
        Dual3 {
            val: point.y,
            dx: 0.0,
            dy: 1.0,
            dz: 0.0,
        },
        Dual3 {
            val: point.z,
            dx: 0.0,
            dy: 0.0,
            dz: 1.0,
        },
    )
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn dual_arithmetic() {
        let a = Dual::variable(3.0);
        let b = Dual::constant(2.0);
        let c = a + b;
        assert_eq!(c.val, 5.0);
        assert_eq!(c.dot, 1.0);

        let d = a * b;
        assert_eq!(d.val, 6.0);
        assert_eq!(d.dot, 2.0);
    }

    #[test]
    fn dual_sqrt() {
        let x = Dual::variable(4.0);
        let r = x.sqrt();
        assert!((r.val - 2.0).abs() < 1e-6);
        assert!((r.dot - 0.25).abs() < 1e-6); // d/dx sqrt(x) = 0.5/sqrt(x)
    }

    #[test]
    fn dual_chain_rule() {
        // f(x) = sqrt(x*x + 1), f'(x) = x / sqrt(x² + 1)
        let x = Dual::variable(3.0);
        let one = Dual::constant(1.0);
        let f = (x * x + one).sqrt();
        let expected = 3.0 / 10.0_f32.sqrt();
        assert!((f.dot - expected).abs() < 1e-5);
    }

    #[test]
    fn dual3_sphere_gradient() {
        let p = Vec3::new(2.0, 0.0, 0.0);
        let (px, py, pz) = dual3_point(p);
        let d = dual3_sphere(px, py, pz, 1.0);
        assert!((d.val - 1.0).abs() < 1e-5);
        assert!((d.dx - 1.0).abs() < 1e-5);
        assert!(d.dy.abs() < 1e-5);
        assert!(d.dz.abs() < 1e-5);
    }

    #[test]
    fn dual3_sphere_diagonal() {
        let p = Vec3::new(1.0, 1.0, 1.0);
        let (px, py, pz) = dual3_point(p);
        let d = dual3_sphere(px, py, pz, 1.0);
        let expected_grad = p.normalize();
        assert!((d.gradient() - expected_grad).length() < 1e-4);
    }

    #[test]
    fn dual3_box_outside() {
        let p = Vec3::new(2.0, 0.0, 0.0);
        let (px, py, pz) = dual3_point(p);
        let d = dual3_box(px, py, pz, Vec3::splat(1.0));
        assert!((d.val - 1.0).abs() < 1e-5);
        assert!((d.dx - 1.0).abs() < 1e-5);
        assert!(d.dy.abs() < 1e-5);
    }

    #[test]
    fn dual3_torus_gradient() {
        let p = Vec3::new(2.0, 0.0, 0.0);
        let (px, py, pz) = dual3_point(p);
        let d = dual3_torus(px, py, pz, 1.5, 0.3);
        let expected = eval(&SdfNode::torus(1.5, 0.3), p);
        assert!((d.val - expected).abs() < 1e-4);
        let expected_grad = eval_gradient(&SdfNode::torus(1.5, 0.3), p);
        assert!((d.gradient() - expected_grad).length() < 0.02);
    }

    #[test]
    fn eval_with_gradient_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let (d, g) = eval_with_gradient(&sphere, Vec3::new(3.0, 0.0, 0.0));
        assert!((d - 2.0).abs() < 1e-4);
        assert!((g.x - 1.0).abs() < 1e-3);
    }

    #[test]
    fn eval_dual3_matches_eval() {
        let shape = SdfNode::sphere(1.0).translate(1.0, 0.0, 0.0);
        let p = Vec3::new(2.5, 0.3, 0.1);
        let d3 = eval_dual3(&shape, p);
        let expected_val = eval(&shape, p);
        let expected_grad = eval_gradient(&shape, p);
        assert!((d3.val - expected_val).abs() < 1e-5);
        assert!((d3.gradient() - expected_grad).length() < 1e-4);
    }

    #[test]
    fn hessian_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let p = Vec3::new(2.0, 0.0, 0.0);
        let h = eval_hessian(&sphere, p, 1e-3);
        // For sphere at (r,0,0): dxx=0, dyy=1/r, dzz=1/r
        assert!(h[0].abs() < 0.05); // dxx near 0
        assert!((h[1] - 0.5).abs() < 0.05); // dyy ≈ 1/r = 0.5
        assert!((h[2] - 0.5).abs() < 0.05); // dzz ≈ 0.5
    }

    #[test]
    fn mean_curvature_sphere() {
        let sphere = SdfNode::sphere(2.0);
        let p = Vec3::new(2.0, 0.0, 0.0);
        let k = mean_curvature(&sphere, p, 1e-3);
        // Mean curvature of sphere = 2/r = 1.0
        assert!((k - 1.0).abs() < 0.1);
    }

    #[test]
    fn dual_trig() {
        let x = Dual::variable(0.0);
        let s = x.sin();
        assert!(s.val.abs() < 1e-6);
        assert!((s.dot - 1.0).abs() < 1e-6); // d/dx sin(x)|_{x=0} = cos(0) = 1

        let c = x.cos();
        assert!((c.val - 1.0).abs() < 1e-6);
        assert!(c.dot.abs() < 1e-6); // d/dx cos(x)|_{x=0} = -sin(0) = 0
    }

    #[test]
    fn dual3_plane_gradient() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let p = Vec3::new(1.0, 3.0, 2.0);
        let (px, py, pz) = dual3_point(p);
        let d = dual3_plane(px, py, pz, n, 0.5);
        assert!((d.val - 2.5).abs() < 1e-5);
        assert!(d.dx.abs() < 1e-5);
        assert!((d.dy - 1.0).abs() < 1e-5);
        assert!(d.dz.abs() < 1e-5);
    }

    #[test]
    fn dual3_constant_zero_gradient() {
        let c = Dual3::constant(42.0);
        assert_eq!(c.val, 42.0);
        assert_eq!(c.gradient(), Vec3::ZERO);
    }
}
