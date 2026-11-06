//! Transform operations for SDFs
//!
//! Transforms modify the input point before evaluating the child SDF.
//!
//! Author: Moroya Sakamoto

pub mod grid_warp;
pub mod grid_warp_build;
pub mod lattice;
pub mod projective;
mod rotate;
mod scale;
pub mod skinning;
mod translate;
pub mod bell_pull;

pub use bell_pull::{BellPull, BellPullData, BellPullError};
pub use grid_warp::{GridWarp, GridWarpData, GridWarpError};
pub use grid_warp_build::{
    affine_posed_to_rest, grid_warp_from_triangles, GridWarpBuildError, Overhang, WarpGrid,
};
pub use lattice::lattice_deform;
pub use projective::projective_transform;
pub use rotate::{transform_rotate, transform_rotate_euler, transform_rotate_inverse};
pub use scale::{transform_scale, transform_scale_inverse, transform_scale_nonuniform};
pub use skinning::{sdf_skinning, BoneTransform};
pub use translate::{transform_translate, transform_translate_inverse};
