//! Projective (perspective) transform for SDFs
//!
//! Applies a 4x4 projective matrix to warp space with Lipschitz correction.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Apply inverse projective transform to a point
/// matrix is stored as column-major [f32; 16] (inverse of the desired transform)
#[inline(always)]
pub fn projective_transform(p: Vec3, inv_matrix: &[f32; 16]) -> (Vec3, f32) {
    let m = inv_matrix;
    // Homogeneous transform: q = M^-1 * [p, 1]
    let w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
    let inv_w = 1.0 / w;
    let q = Vec3::new(
        (m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12]) * inv_w,
        (m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13]) * inv_w,
        (m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]) * inv_w,
    );
    // Lipschitz correction factor: |dq/dp| ≈ 1/w for projective
    let correction = inv_w.abs();
    (q, correction)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_identity_projective() {
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let p = Vec3::new(1.0, 2.0, 3.0);
        let (q, corr) = projective_transform(p, &identity);
        assert!((q - p).length() < 1e-6);
        assert!((corr - 1.0).abs() < 1e-6);
    }
}
