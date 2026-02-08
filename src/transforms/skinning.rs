//! SDF Skinning — bone-weight based space blending
//!
//! Linear Blend Skinning (LBS) for SDF spatial warping.
//! Each bone has an inverse bind pose and a current transform.
//!
//! Author: Moroya Sakamoto

use glam::{Mat4, Vec3};
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

/// Apply Linear Blend Skinning to transform point back to rest pose
/// bones: active bones with weights (should sum to ~1.0)
#[inline(always)]
pub fn sdf_skinning(p: Vec3, bones: &[BoneTransform]) -> (Vec3, f32) {
    if bones.is_empty() {
        return (p, 1.0);
    }

    let mut result = Vec3::ZERO;
    let mut total_weight = 0.0;

    for bone in bones {
        let w = bone.weight;
        if w < 1e-6 {
            continue;
        }

        // Compute skinning matrix: inverse(current_pose) * bind_pose
        // This maps from current world space back to rest pose
        let inv_current = mat4_from_array(&bone.current_pose);
        let bind = mat4_from_array(&bone.inv_bind_pose);

        // Transform point: skin_matrix * p
        let skinned = inv_current.transform_point3(p);
        let rest_p = bind.transform_point3(skinned);

        result += rest_p * w;
        total_weight += w;
    }

    if total_weight > 1e-6 {
        result /= total_weight;
    } else {
        result = p;
    }

    // Lipschitz correction: approximate as 1.0 for LBS (exact for rigid transforms)
    (result, 1.0)
}

fn mat4_from_array(m: &[f32; 16]) -> Mat4 {
    Mat4::from_cols_array(m)
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
