//! CSG (Constructive Solid Geometry) Operations (Deep Fried Edition)
//!
//! Boolean operations for combining SDFs.
//!
//! # Deep Fried Optimizations
//! - **Forced Inlining**: All functions use `#[inline(always)]`.
//! - **Branchless Smooth**: Removed safety checks for `k <= 0`.
//!
//! Author: Moroya Sakamoto

mod union;
mod intersection;
mod subtraction;
mod smooth;
mod chamfer;
mod stairs;
mod xor;
mod morph;
mod columns;
mod pipe;
mod engrave;
mod groove;
mod tongue;

pub use union::{sdf_union, sdf_union_multi};
pub use intersection::{sdf_intersection, sdf_intersection_multi};
pub use subtraction::sdf_subtraction;
pub use smooth::{
    sdf_smooth_union, sdf_smooth_intersection, sdf_smooth_subtraction,
    sdf_smooth_union_rk, sdf_smooth_intersection_rk, sdf_smooth_subtraction_rk,
    smooth_min, smooth_max, smooth_min_rk, smooth_max_rk,
    smooth_min_exp, smooth_min_cubic, smooth_min_root,
};
pub use chamfer::{
    sdf_chamfer_union, sdf_chamfer_intersection, sdf_chamfer_subtraction,
    chamfer_min, chamfer_max,
};
pub use stairs::{
    sdf_stairs_union, sdf_stairs_intersection, sdf_stairs_subtraction,
    stairs_min, stairs_max,
};
pub use xor::sdf_xor;
pub use morph::sdf_morph;
pub use columns::{sdf_columns_union, sdf_columns_intersection, sdf_columns_subtraction};
pub use pipe::sdf_pipe;
pub use engrave::sdf_engrave;
pub use groove::sdf_groove;
pub use tongue::sdf_tongue;
