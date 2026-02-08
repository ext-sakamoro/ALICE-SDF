//! Iterated Function System (IFS) modifier
//!
//! Recursively applies affine transforms to fold space, creating fractal-like
//! self-similar patterns in SDF evaluation.
//!
//! Author: Moroya Sakamoto

use glam::{Mat4, Vec3};

/// Apply IFS folding: for each iteration, apply the transform that maps
/// the point closest to the origin (greedy orbit trap)
#[inline(always)]
pub fn ifs_fold(p: Vec3, transforms: &[[f32; 16]], iterations: u32) -> Vec3 {
    let mut p = p;

    for _ in 0..iterations {
        let mut best_p = p;
        let mut best_dist = p.length_squared();

        for t in transforms {
            let m = Mat4::from_cols_array(t);
            let q = m.transform_point3(p);
            let d = q.length_squared();
            if d < best_dist {
                best_dist = d;
                best_p = q;
            }
        }

        p = best_p;
    }

    p
}

/// Apply IFS with distance scaling correction
/// Returns (folded_point, accumulated_scale) for Lipschitz correction
#[inline(always)]
pub fn ifs_fold_with_scale(p: Vec3, transforms: &[[f32; 16]], iterations: u32) -> (Vec3, f32) {
    let mut p = p;
    let mut scale = 1.0_f32;

    for _ in 0..iterations {
        let mut best_p = p;
        let mut best_dist = p.length_squared();
        let mut best_scale = 1.0;

        for t in transforms {
            let m = Mat4::from_cols_array(t);
            let q = m.transform_point3(p);
            let d = q.length_squared();
            if d < best_dist {
                best_dist = d;
                best_p = q;
                // Approximate scale factor from matrix
                let sx = Vec3::new(m.x_axis.x, m.x_axis.y, m.x_axis.z).length();
                best_scale = sx; // Approximate with X-axis scale
            }
        }

        p = best_p;
        scale *= best_scale;
    }

    (p, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ifs_identity() {
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let p = Vec3::new(1.0, 2.0, 3.0);
        let q = ifs_fold(p, &[identity], 3);
        assert!((q - p).length() < 1e-6);
    }

    #[test]
    fn test_ifs_contraction() {
        // Scale by 0.5 should contract toward origin
        let half = [
            0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let p = Vec3::new(1.0, 1.0, 1.0);
        let q = ifs_fold(p, &[half], 3);
        assert!(
            q.length() < p.length(),
            "IFS with contraction should move toward origin"
        );
    }
}
