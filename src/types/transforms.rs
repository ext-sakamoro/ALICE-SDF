//! Spatial transform methods for SdfNode
//!
//! Author: Moroya Sakamoto

use glam::{Quat, Vec3};
use std::sync::Arc;

use super::SdfNode;

impl SdfNode {
    // === Transform methods ===

    /// Translate by offset
    #[must_use]
    #[inline]
    pub fn translate(self, x: f32, y: f32, z: f32) -> Self {
        Self::Translate {
            child: Arc::new(self),
            offset: Vec3::new(x, y, z),
        }
    }

    /// Translate by vector
    #[must_use]
    #[inline]
    pub fn translate_vec(self, offset: Vec3) -> Self {
        Self::Translate {
            child: Arc::new(self),
            offset,
        }
    }

    /// Rotate by quaternion
    #[must_use]
    #[inline]
    pub fn rotate(self, rotation: Quat) -> Self {
        Self::Rotate {
            child: Arc::new(self),
            rotation,
        }
    }

    /// Marks this subtree's top translate and the rotate right below it as a pose ([`SdfNode::Pose`]);
    /// the rest point refuses a pose over any other shape, so prefer [`SdfNode::rigid_pose`]
    #[must_use]
    #[inline]
    pub fn pose(self) -> Self {
        Self::Pose {
            child: Arc::new(self),
        }
    }

    /// A rigid pose: rotate by `rotation`, then translate by `offset`, marked as a pose (the rest
    /// point undoes it; the distance is that of `rotate(rotation).translate_vec(offset)`)
    #[must_use]
    #[inline]
    pub fn rigid_pose(self, rotation: Quat, offset: Vec3) -> Self {
        self.rotate(rotation).translate_vec(offset).pose()
    }

    /// Rotate by Euler angles (radians)
    #[must_use]
    #[inline]
    pub fn rotate_euler(self, x: f32, y: f32, z: f32) -> Self {
        Self::Rotate {
            child: Arc::new(self),
            rotation: Quat::from_euler(glam::EulerRot::XYZ, x, y, z),
        }
    }

    /// Uniform scale
    #[must_use]
    #[inline]
    pub fn scale(self, factor: f32) -> Self {
        Self::Scale {
            child: Arc::new(self),
            factor,
        }
    }

    /// Non-uniform scale
    #[must_use]
    #[inline]
    pub fn scale_xyz(self, x: f32, y: f32, z: f32) -> Self {
        Self::ScaleNonUniform {
            child: Arc::new(self),
            factors: Vec3::new(x, y, z),
        }
    }

    /// Projective (perspective) transform
    #[must_use]
    #[inline]
    pub fn projective_transform(self, inv_matrix: [f32; 16], lipschitz_bound: f32) -> Self {
        Self::ProjectiveTransform {
            child: Arc::new(self),
            inv_matrix,
            lipschitz_bound,
        }
    }

    /// Lattice deformation (Free-Form Deformation)
    #[must_use]
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
        Self::LatticeDeform {
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
    #[must_use]
    #[inline]
    pub fn sdf_skinning(self, bones: Vec<crate::transforms::skinning::BoneTransform>) -> Self {
        Self::SdfSkinning {
            child: Arc::new(self),
            bones,
        }
    }

    /// Domain warp by a grid of affine maps ([`crate::transforms::GridWarp`])
    #[must_use]
    #[inline]
    pub fn grid_warp(self, warp: crate::transforms::GridWarp) -> Self {
        Self::GridWarp {
            child: Arc::new(self),
            warp: Arc::new(warp),
        }
    }
}
