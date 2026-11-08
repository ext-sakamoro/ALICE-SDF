//! SDF Skinning — bone-weight based space blending
//!
//! Linear Blend Skinning (LBS) for SDF spatial warping.
//! Each bone has an inverse bind pose and a current transform.
//!
//! Author: Moroya Sakamoto

use glam::{DMat4, Mat4, Vec3};
use serde::{Deserialize, Serialize};

/// A bone transform for SDF skinning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoneTransform {
    /// Inverse bind pose matrix (world space → bone space)
    pub inv_bind_pose: [f32; 16],
    /// Current pose matrix (bone space → world space)
    pub current_pose: [f32; 16],
    /// Vertex weight for this bone
    pub weight: f32,
}

/// The matrix that takes a posed (world) point back to the rest pose for one bone:
/// `inverse(current_pose · inv_bind_pose)` = bind pose · inverse(current pose).
///
/// Composed and inverted once, in f64 (glam's scalar `DMat4`: plain multiplications and
/// additions, no fused operations), then rounded to f32. Every evaluator takes its matrix from
/// here: [`sdf_skinning`], the compiled evaluator (the compiler stores this matrix) and the
/// shader transpilers (they write this matrix's bits as literals), so they all start from the
/// same 16 floats.
///
/// `None` when the composed matrix is not invertible (determinant 0 or not finite) or its inverse
/// is not finite: such a bone has no rest pose to return to and contributes nothing, as a bone of
/// zero weight does.
#[must_use]
pub fn rest_from_posed(bone: &BoneTransform) -> Option<Mat4> {
    let current = DMat4::from_cols_array(&bone.current_pose.map(f64::from));
    let inv_bind = DMat4::from_cols_array(&bone.inv_bind_pose.map(f64::from));
    let posed_from_rest = current * inv_bind;
    let det = posed_from_rest.determinant();
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let m = posed_from_rest.inverse().as_mat4();
    m.is_finite().then_some(m)
}

/// The rest point of `p`: Σ wᵢ Mᵢ p / Σ wᵢ over the bones of weight ≥ 1e-6, where Mᵢ is
/// [`rest_from_posed`]; `p` itself when no bone counts (none, all below the weight, or the
/// weights sum to 1e-6 or less). The one blend every CPU evaluator uses.
#[inline]
#[must_use]
pub fn blend_rest_point(p: Vec3, bones: &[(Mat4, f32)]) -> Vec3 {
    let mut result = Vec3::ZERO;
    let mut total_weight = 0.0;
    for &(m, w) in bones {
        if w < 1e-6 {
            continue;
        }
        result += m.transform_point3(p) * w;
        total_weight += w;
    }
    if total_weight > 1e-6 {
        result / total_weight
    } else {
        p
    }
}

/// Each bone with an invertible pose, as its [`rest_from_posed`] matrix and weight, in order
#[must_use]
pub fn rest_matrices(bones: &[BoneTransform]) -> Vec<(Mat4, f32)> {
    bones
        .iter()
        .filter_map(|b| rest_from_posed(b).map(|m| (m, b.weight)))
        .collect()
}

/// Linear Blend Skinning: take a posed point back to the rest pose
///
/// Each bone maps `p` with [`rest_from_posed`] and the results are averaged by weight
/// ([`blend_rest_point`]). With one bone (or all bones in the same pose) this is the exact inverse
/// of the forward skinning `posed = current · inv_bind · rest`; with several bones in different
/// poses the weighted average of per-bone inverses is an approximation of the inverse.
#[inline]
#[must_use]
pub fn sdf_skinning(p: Vec3, bones: &[BoneTransform]) -> (Vec3, f32) {
    // Lipschitz correction: approximate as 1.0 for LBS (exact for rigid transforms)
    (blend_rest_point(p, &rest_matrices(bones)), 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_identity_skinning() {
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let bone = BoneTransform {
            inv_bind_pose: identity,
            current_pose: identity,
            weight: 1.0,
        };
        let p = Vec3::new(1.0, 2.0, 3.0);
        let (q, _) = sdf_skinning(p, &[bone]);
        assert!((q - p).length() < 1e-5);
    }
}
