//! SDF Skinning: one rigid or affine pose per bone, blended by a weight that is fixed per bone
//!
//! A bone's forward map is `S = current_pose · inv_bind_pose`: a rest-pose point goes to bone space
//! at bind, then to the world by the current pose. The node reads the rest pose at
//! `q = Σ w_i S_i⁻¹ p / Σ w_i`, so a part moves the way its bone moves. The weights do not depend on
//! the position, so the node is one affine map `q = A p + b`: it can pose a part, not bend one. For
//! a joint that bends, see [`crate::transforms::GridWarp`].
//!
//! The inverses are taken in `f64` from the matrices' own entries, so every backend reads the same
//! `f32` constants. A bone with no inverse, or with no weight, takes no part.
//!
//! Author: Moroya Sakamoto

use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

/// A bone transform for SDF skinning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoneTransform {
    /// Inverse bind pose matrix (world space → bone space at bind)
    pub inv_bind_pose: [f32; 16],
    /// Current pose matrix (bone space → world space)
    pub current_pose: [f32; 16],
    /// Weight of this bone, the same at every point
    pub weight: f32,
}

/// The weight under which a bone takes no part
const MIN_WEIGHT: f32 = 1e-6;

/// `S⁻¹` of an affine `S = current_pose · inv_bind_pose` (column-major, the bottom rows are not
/// read), or `None` when it has no finite inverse
fn rest_map(bone: &BoneTransform) -> Option<[f32; 16]> {
    let (c, b) = (&bone.current_pose, &bone.inv_bind_pose);
    let at = |m: &[f32; 16], col: usize, row: usize| f64::from(m[col * 4 + row]);
    let mut s = [[0.0_f64; 3]; 4];
    for (col, out) in s.iter_mut().enumerate() {
        for (row, v) in out.iter_mut().enumerate() {
            *v = at(c, 0, row) * at(b, col, 0)
                + at(c, 1, row) * at(b, col, 1)
                + at(c, 2, row) * at(b, col, 2);
            if col == 3 {
                *v += at(c, 3, row);
            }
        }
    }
    let cof = [
        s[1][1] * s[2][2] - s[1][2] * s[2][1],
        s[1][2] * s[2][0] - s[1][0] * s[2][2],
        s[1][0] * s[2][1] - s[1][1] * s[2][0],
    ];
    let det = s[0][0] * cof[0] + s[0][1] * cof[1] + s[0][2] * cof[2];
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let col0 = [
        cof[0] / det,
        (s[0][2] * s[2][1] - s[0][1] * s[2][2]) / det,
        (s[0][1] * s[1][2] - s[0][2] * s[1][1]) / det,
    ];
    let col1 = [
        cof[1] / det,
        (s[0][0] * s[2][2] - s[0][2] * s[2][0]) / det,
        (s[0][2] * s[1][0] - s[0][0] * s[1][2]) / det,
    ];
    let col2 = [
        cof[2] / det,
        (s[0][1] * s[2][0] - s[0][0] * s[2][1]) / det,
        (s[0][0] * s[1][1] - s[0][1] * s[1][0]) / det,
    ];
    let t = s[3];
    let translate = |row: usize| -(col0[row] * t[0] + col1[row] * t[1] + col2[row] * t[2]);
    let m = [
        col0[0],
        col0[1],
        col0[2],
        0.0,
        col1[0],
        col1[1],
        col1[2],
        0.0,
        col2[0],
        col2[1],
        col2[2],
        0.0,
        translate(0),
        translate(1),
        translate(2),
        1.0,
    ];
    let out = m.map(|v| v as f32);
    out.iter().all(|v| v.is_finite()).then_some(out)
}

/// The bones that take part, each with its `S⁻¹` and weight, and the sum of their weights: the one
/// reading of the node, for the evaluators, the shaders and the bounds. `None` when no bone does,
/// and the node is then the identity.
pub(crate) fn active_bones(bones: &[BoneTransform]) -> Option<(Vec<([f32; 16], f32)>, f32)> {
    let active: Vec<_> = bones
        .iter()
        .filter(|b| b.weight >= MIN_WEIGHT)
        .filter_map(|b| rest_map(b).map(|m| (m, b.weight)))
        .collect();
    let total: f32 = active.iter().map(|(_, w)| w).sum();
    (!active.is_empty() && total > MIN_WEIGHT).then_some((active, total))
}

/// The rest-pose point the node reads at `p`; the second value is 1: the child's value is not
/// divided, and `eval_lipschitz` reports the slope
#[inline(always)]
pub fn sdf_skinning(p: Vec3, bones: &[BoneTransform]) -> (Vec3, f32) {
    let Some((active, total)) = active_bones(bones) else {
        return (p, 1.0);
    };
    let mut result = Vec3::ZERO;
    for (map, weight) in &active {
        result += Mat4::from_cols_array(map).transform_point3(p) * *weight;
    }
    (result / total, 1.0)
}

/// An upper bound of the spectral norm of a 3x3 matrix given by its 9 entries, reached by a rotation
/// times a uniform scale: with `M = AᵀA`, `λ_max(M) ≤ m + s·sqrt(2)`, where `m = tr(M) / 3` and
/// `s² = tr(M²) / 3 − m²` (Wolkowicz and Styan, for any symmetric 3x3).
pub(crate) fn spectral_norm_bound(a: &[f32; 9]) -> f32 {
    let col = |i: usize| {
        [
            f64::from(a[3 * i]),
            f64::from(a[3 * i + 1]),
            f64::from(a[3 * i + 2]),
        ]
    };
    let cols = [col(0), col(1), col(2)];
    let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let m00 = dot(cols[0], cols[0]);
    let m11 = dot(cols[1], cols[1]);
    let m22 = dot(cols[2], cols[2]);
    let m01 = dot(cols[0], cols[1]);
    let m02 = dot(cols[0], cols[2]);
    let m12 = dot(cols[1], cols[2]);
    let mean = (m00 + m11 + m22) / 3.0;
    let square_trace =
        m00 * m00 + m11 * m11 + m22 * m22 + 2.0 * (m01 * m01 + m02 * m02 + m12 * m12);
    let spread = (square_trace / 3.0 - mean * mean).max(0.0).sqrt();
    let bound = (mean + spread * std::f64::consts::SQRT_2).sqrt();
    // rounding to f32 must not go under the bound
    (bound as f32) * (1.0 + f32::EPSILON)
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

    fn bone(current: Mat4, inv_bind: Mat4) -> BoneTransform {
        BoneTransform {
            inv_bind_pose: inv_bind.to_cols_array(),
            current_pose: current.to_cols_array(),
            weight: 1.0,
        }
    }

    /// `S⁻¹ S = I` against the product, for a turn, a scale and a shear at a bind away from the origin
    #[test]
    fn the_rest_map_undoes_the_forward_map() {
        let shear = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, -0.25, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let current =
            Mat4::from_translation(Vec3::new(1.0, -2.0, 3.0)) * Mat4::from_rotation_z(0.7) * shear;
        let inv_bind = Mat4::from_translation(Vec3::new(-0.5, 0.25, 0.0));
        let forward = current * inv_bind;
        let map = rest_map(&bone(current, inv_bind)).expect("an inverse");
        let round = Mat4::from_cols_array(&map) * forward;
        for (got, want) in round
            .to_cols_array()
            .iter()
            .zip(Mat4::IDENTITY.to_cols_array())
        {
            assert!((got - want).abs() < 1e-5, "{round}");
        }
    }

    #[test]
    fn a_bone_with_no_inverse_or_no_weight_takes_no_part() {
        let flat = Mat4::from_scale(Vec3::new(1.0, 0.0, 1.0));
        assert!(rest_map(&bone(flat, Mat4::IDENTITY)).is_none());
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(sdf_skinning(p, &[bone(flat, Mat4::IDENTITY)]).0, p);
        let moved = Mat4::from_translation(Vec3::X);
        let mut none = bone(moved, Mat4::IDENTITY);
        none.weight = 0.0;
        assert_eq!(sdf_skinning(p, &[none]).0, p);
        let (q, _) = sdf_skinning(
            p,
            &[bone(flat, Mat4::IDENTITY), bone(moved, Mat4::IDENTITY)],
        );
        assert!((q - (p - Vec3::X)).length() < 1e-6, "{q}");
    }

    /// Exact for a turn times a uniform scale; never under the norm of a matrix with spread values
    #[test]
    fn the_norm_bound_is_exact_when_it_can_be_and_never_under() {
        let turn = glam::Mat3::from_rotation_z(1.0).to_cols_array();
        assert!((spectral_norm_bound(&turn) - 1.0).abs() < 1e-5);
        let scaled = turn.map(|v| v * 2.5);
        assert!((spectral_norm_bound(&scaled) - 2.5).abs() < 1e-5);
        let diag = [3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5];
        let got = spectral_norm_bound(&diag);
        assert!((3.0..3.2).contains(&got), "{got}");
    }
}
