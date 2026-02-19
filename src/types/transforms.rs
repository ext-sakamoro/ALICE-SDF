//! Spatial transform methods for SdfNode
//!
//! Author: Moroya Sakamoto

use glam::{Quat, Vec3};
use std::sync::Arc;

use super::SdfNode;

impl SdfNode {
    // === Transform methods ===

    /// Translate by offset
    #[inline]
    pub fn translate(self, x: f32, y: f32, z: f32) -> Self {
        SdfNode::Translate {
            child: Arc::new(self),
            offset: Vec3::new(x, y, z),
        }
    }

    /// Translate by vector
    #[inline]
    pub fn translate_vec(self, offset: Vec3) -> Self {
        SdfNode::Translate {
            child: Arc::new(self),
            offset,
        }
    }

    /// Rotate by quaternion
    #[inline]
    pub fn rotate(self, rotation: Quat) -> Self {
        SdfNode::Rotate {
            child: Arc::new(self),
            rotation,
        }
    }

    /// Rotate by Euler angles (radians)
    #[inline]
    pub fn rotate_euler(self, x: f32, y: f32, z: f32) -> Self {
        SdfNode::Rotate {
            child: Arc::new(self),
            rotation: Quat::from_euler(glam::EulerRot::XYZ, x, y, z),
        }
    }

    /// Uniform scale
    #[inline]
    pub fn scale(self, factor: f32) -> Self {
        SdfNode::Scale {
            child: Arc::new(self),
            factor,
        }
    }

    /// Non-uniform scale
    #[inline]
    pub fn scale_xyz(self, x: f32, y: f32, z: f32) -> Self {
        SdfNode::ScaleNonUniform {
            child: Arc::new(self),
            factors: Vec3::new(x, y, z),
        }
    }

    /// Projective (perspective) transform
    #[inline]
    pub fn projective_transform(self, inv_matrix: [f32; 16], lipschitz_bound: f32) -> Self {
        SdfNode::ProjectiveTransform {
            child: Arc::new(self),
            inv_matrix,
            lipschitz_bound,
        }
    }

    /// Lattice deformation (Free-Form Deformation)
    #[inline]
    pub fn lattice_deform(
        self,
        control_points: Vec<Vec3>,
        nx: u32,
        ny: u32,
        nz: u32,
        bbox_min: Vec3,
        bbox_max: Vec3,
    ) -> Self {
        SdfNode::LatticeDeform {
            child: Arc::new(self),
            control_points,
            nx,
            ny,
            nz,
            bbox_min,
            bbox_max,
        }
    }

    /// SDF skinning (bone-weight based deformation)
    #[inline]
    pub fn sdf_skinning(self, bones: Vec<crate::transforms::skinning::BoneTransform>) -> Self {
        SdfNode::SdfSkinning {
            child: Arc::new(self),
            bones,
        }
    }
}
