//! CSG (Constructive Solid Geometry) Operations (Deep Fried Edition)
//!
//! Boolean operations for combining SDFs.
//!
//! # Deep Fried Optimizations
//! - **Forced Inlining**: All functions use `#[inline(always)]`.
//! - **Branchless Smooth**: Removed safety checks for `k <= 0`.
//!
//! Author: Moroya Sakamoto

mod chamfer;
mod columns;
mod engrave;
mod groove;
mod intersection;
mod morph;
mod pipe;
mod smooth;
mod stairs;
mod subtraction;
mod tongue;
mod union;
mod xor;

pub use chamfer::{
    chamfer_max, chamfer_min, sdf_chamfer_intersection, sdf_chamfer_subtraction, sdf_chamfer_union,
};
pub use columns::{sdf_columns_intersection, sdf_columns_subtraction, sdf_columns_union};
pub use engrave::sdf_engrave;
pub use groove::sdf_groove;
pub use intersection::{sdf_intersection, sdf_intersection_multi};
pub use morph::sdf_morph;
pub use pipe::sdf_pipe;
pub use smooth::{
    sdf_exp_smooth_intersection, sdf_exp_smooth_subtraction, sdf_exp_smooth_union,
    sdf_smooth_intersection, sdf_smooth_intersection_rk, sdf_smooth_subtraction,
    sdf_smooth_subtraction_rk, sdf_smooth_union, sdf_smooth_union_rk, smooth_max, smooth_max_rk,
    smooth_min, smooth_min_cubic, smooth_min_exp, smooth_min_rk, smooth_min_root,
};
pub use stairs::{
    sdf_stairs_intersection, sdf_stairs_subtraction, sdf_stairs_union, stairs_max, stairs_min,
};
pub use subtraction::sdf_subtraction;
pub use tongue::sdf_tongue;
pub use union::{sdf_union, sdf_union_multi};
pub use xor::sdf_xor;

// Generic ([`crate::compiled::real::Real`]) forms — one law for scalar and SIMD evaluators
pub use chamfer::{
    chamfer_max_r, chamfer_min_r, sdf_chamfer_intersection_r, sdf_chamfer_subtraction_r,
    sdf_chamfer_union_r,
};
pub use columns::{sdf_columns_intersection_r, sdf_columns_subtraction_r, sdf_columns_union_r};
pub use engrave::sdf_engrave_r;
pub use groove::sdf_groove_r;
pub use intersection::sdf_intersection_r;
pub use morph::sdf_morph_r;
pub use pipe::sdf_pipe_r;
pub use smooth::{
    sdf_exp_smooth_intersection_r, sdf_exp_smooth_subtraction_r, sdf_exp_smooth_union_r,
    sdf_smooth_intersection_rk_r, sdf_smooth_subtraction_rk_r, sdf_smooth_union_rk_r,
    smooth_max_rk_r, smooth_min_rk_r,
};
pub use stairs::{
    sdf_stairs_intersection_r, sdf_stairs_subtraction_r, sdf_stairs_union_r, stairs_max_r,
    stairs_min_r,
};
pub use subtraction::sdf_subtraction_r;
pub use tongue::sdf_tongue_r;
pub use union::sdf_union_r;
pub use xor::sdf_xor_r;
