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
