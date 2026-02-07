//! SIMD type definitions for 8-wide evaluation
//!
//! This module provides SIMD-friendly types for evaluating
//! 8 points simultaneously using AVX2/AVX-512/NEON instructions.
//!
//! Author: Moroya Sakamoto

use wide::f32x8;

/// 8 3D vectors packed for SIMD processing
///
/// Structure-of-Arrays (SoA) layout for optimal SIMD performance:
/// - x: [x0, x1, x2, x3, x4, x5, x6, x7]
/// - y: [y0, y1, y2, y3, y4, y5, y6, y7]
/// - z: [z0, z1, z2, z3, z4, z5, z6, z7]
#[derive(Clone, Copy, Debug)]
pub struct Vec3x8 {
    /// X components (8-wide)
    pub x: f32x8,
    /// Y components (8-wide)
    pub y: f32x8,
    /// Z components (8-wide)
    pub z: f32x8,
}

impl Vec3x8 {
    /// Create from 8 separate Vec3 values
    #[inline]
    pub fn from_vecs(v: [glam::Vec3; 8]) -> Self {
        Vec3x8 {
            x: f32x8::new([v[0].x, v[1].x, v[2].x, v[3].x, v[4].x, v[5].x, v[6].x, v[7].x]),
            y: f32x8::new([v[0].y, v[1].y, v[2].y, v[3].y, v[4].y, v[5].y, v[6].y, v[7].y]),
            z: f32x8::new([v[0].z, v[1].z, v[2].z, v[3].z, v[4].z, v[5].z, v[6].z, v[7].z]),
        }
    }

    /// Create with all lanes set to the same vector
    #[inline]
    pub fn splat(v: glam::Vec3) -> Self {
        Vec3x8 {
            x: f32x8::splat(v.x),
            y: f32x8::splat(v.y),
            z: f32x8::splat(v.z),
        }
    }

    /// Create from raw x, y, z arrays
    #[inline]
    pub fn new(x: [f32; 8], y: [f32; 8], z: [f32; 8]) -> Self {
        Vec3x8 {
            x: f32x8::new(x),
            y: f32x8::new(y),
            z: f32x8::new(z),
        }
    }

    /// Zero vector for all 8 lanes
    #[inline]
    pub fn zero() -> Self {
        Vec3x8 {
            x: f32x8::ZERO,
            y: f32x8::ZERO,
            z: f32x8::ZERO,
        }
    }

    /// Compute length of all 8 vectors
    #[inline]
    pub fn length(self) -> f32x8 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Compute squared length of all 8 vectors
    #[inline]
    pub fn length_squared(self) -> f32x8 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Normalize all 8 vectors (zero-safe: returns zero vector for zero-length inputs)
    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        // Branchless zero guard: clamp denominator to epsilon
        let safe_len = len.max(f32x8::splat(1e-10));
        Vec3x8 {
            x: self.x / safe_len,
            y: self.y / safe_len,
            z: self.z / safe_len,
        }
    }

    /// Dot product with another Vec3x8
    #[inline]
    pub fn dot(self, other: Self) -> f32x8 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Component-wise absolute value
    #[inline]
    pub fn abs(self) -> Self {
        Vec3x8 {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    /// Component-wise maximum with zero
    #[inline]
    pub fn max_zero(self) -> Self {
        Vec3x8 {
            x: self.x.max(f32x8::ZERO),
            y: self.y.max(f32x8::ZERO),
            z: self.z.max(f32x8::ZERO),
        }
    }

    /// Component-wise maximum
    #[inline]
    pub fn max(self, other: Self) -> Self {
        Vec3x8 {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    /// Component-wise minimum
    #[inline]
    pub fn min(self, other: Self) -> Self {
        Vec3x8 {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    /// Component-wise clamp
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }

    /// Maximum component of each vector
    #[inline]
    pub fn max_component(self) -> f32x8 {
        self.x.max(self.y).max(self.z)
    }

    /// Minimum component of each vector
    #[inline]
    pub fn min_component(self) -> f32x8 {
        self.x.min(self.y).min(self.z)
    }

    /// Extract results back to array
    #[inline]
    pub fn to_array(self) -> ([f32; 8], [f32; 8], [f32; 8]) {
        (self.x.to_array(), self.y.to_array(), self.z.to_array())
    }
}

// Operator implementations
impl std::ops::Add for Vec3x8 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Vec3x8 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl std::ops::Sub for Vec3x8 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Vec3x8 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl std::ops::Mul<f32x8> for Vec3x8 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f32x8) -> Self {
        Vec3x8 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl std::ops::Div<f32x8> for Vec3x8 {
    type Output = Self;
    #[inline]
    fn div(self, scalar: f32x8) -> Self {
        Vec3x8 {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

impl std::ops::Neg for Vec3x8 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Vec3x8 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Quaternion for 8-wide rotation
#[derive(Clone, Copy, Debug)]
pub struct Quatx8 {
    /// X components (8-wide)
    pub x: f32x8,
    /// Y components (8-wide)
    pub y: f32x8,
    /// Z components (8-wide)
    pub z: f32x8,
    /// W components (8-wide)
    pub w: f32x8,
}

impl Quatx8 {
    /// Create with all lanes set to the same quaternion
    #[inline]
    pub fn splat(q: glam::Quat) -> Self {
        Quatx8 {
            x: f32x8::splat(q.x),
            y: f32x8::splat(q.y),
            z: f32x8::splat(q.z),
            w: f32x8::splat(q.w),
        }
    }

    /// Compute inverse quaternion
    #[inline]
    pub fn inverse(self) -> Self {
        // For unit quaternions, inverse is conjugate
        Quatx8 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// Rotate a Vec3x8 by this quaternion
    #[inline]
    pub fn mul_vec3(self, v: Vec3x8) -> Vec3x8 {
        // Optimized quaternion-vector multiplication
        // q * v * q^-1
        let two = f32x8::splat(2.0);

        let qv_x = self.y * v.z - self.z * v.y;
        let qv_y = self.z * v.x - self.x * v.z;
        let qv_z = self.x * v.y - self.y * v.x;

        let uv_x = self.y * qv_z - self.z * qv_y;
        let uv_y = self.z * qv_x - self.x * qv_z;
        let uv_z = self.x * qv_y - self.y * qv_x;

        Vec3x8 {
            x: v.x + (qv_x * self.w + uv_x) * two,
            y: v.y + (qv_y * self.w + uv_y) * two,
            z: v.z + (qv_z * self.w + uv_z) * two,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_vec3x8_length() {
        let v = Vec3x8::splat(Vec3::new(3.0, 4.0, 0.0));
        let len = v.length();
        let arr = len.to_array();
        for &l in &arr {
            assert!((l - 5.0).abs() < 0.0001);
        }
    }

    #[test]
    fn test_vec3x8_from_vecs() {
        let vecs = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let v = Vec3x8::from_vecs(vecs);
        let (x, y, z) = v.to_array();
        assert_eq!(x[0], 1.0);
        assert_eq!(y[1], 1.0);
        assert_eq!(z[2], 1.0);
    }

    #[test]
    fn test_vec3x8_ops() {
        let a = Vec3x8::splat(Vec3::new(1.0, 2.0, 3.0));
        let b = Vec3x8::splat(Vec3::new(4.0, 5.0, 6.0));

        let sum = a + b;
        let (x, y, z) = sum.to_array();
        assert!((x[0] - 5.0).abs() < 0.0001);
        assert!((y[0] - 7.0).abs() < 0.0001);
        assert!((z[0] - 9.0).abs() < 0.0001);
    }

    #[test]
    fn test_quat_rotation() {
        // 90 degree rotation around Y axis
        let q = glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let qx8 = Quatx8::splat(q);

        let v = Vec3x8::splat(Vec3::new(1.0, 0.0, 0.0));
        let rotated = qx8.mul_vec3(v);
        let (x, y, z) = rotated.to_array();

        // (1, 0, 0) rotated 90° around Y should be approximately (0, 0, -1)
        assert!(x[0].abs() < 0.0001);
        assert!(y[0].abs() < 0.0001);
        assert!((z[0] - (-1.0)).abs() < 0.0001);
    }
}
