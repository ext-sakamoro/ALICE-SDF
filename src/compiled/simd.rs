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
    pub const fn from_vecs(v: [glam::Vec3; 8]) -> Self {
        Self {
            x: f32x8::new([
                v[0].x, v[1].x, v[2].x, v[3].x, v[4].x, v[5].x, v[6].x, v[7].x,
            ]),
            y: f32x8::new([
                v[0].y, v[1].y, v[2].y, v[3].y, v[4].y, v[5].y, v[6].y, v[7].y,
            ]),
            z: f32x8::new([
                v[0].z, v[1].z, v[2].z, v[3].z, v[4].z, v[5].z, v[6].z, v[7].z,
            ]),
        }
    }

    /// Create with all lanes set to the same vector
    #[inline]
    pub fn splat(v: glam::Vec3) -> Self {
        Self {
            x: f32x8::splat(v.x),
            y: f32x8::splat(v.y),
            z: f32x8::splat(v.z),
        }
    }

    /// Create from raw x, y, z arrays
    #[inline]
    pub const fn new(x: [f32; 8], y: [f32; 8], z: [f32; 8]) -> Self {
        Self {
            x: f32x8::new(x),
            y: f32x8::new(y),
            z: f32x8::new(z),
        }
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
        Self {
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
        Self {
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
        Self {
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
        Self {
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
        Self {
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
        let qx8 = Quatx8 {
            x: f32x8::splat(q.x),
            y: f32x8::splat(q.y),
            z: f32x8::splat(q.z),
            w: f32x8::splat(q.w),
        };

        let v = Vec3x8::splat(Vec3::new(1.0, 0.0, 0.0));
        let rotated = qx8.mul_vec3(v);
        let (x, y, z) = rotated.to_array();

        // (1, 0, 0) rotated 90° around Y should be approximately (0, 0, -1)
        assert!(x[0].abs() < 0.0001);
        assert!(y[0].abs() < 0.0001);
        assert!((z[0] - (-1.0)).abs() < 0.0001);
    }
}
