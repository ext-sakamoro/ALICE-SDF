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

pub use union::{sdf_union, sdf_union_multi};
pub use intersection::{sdf_intersection, sdf_intersection_multi};
pub use subtraction::sdf_subtraction;
pub use smooth::{
    sdf_smooth_union, sdf_smooth_intersection, sdf_smooth_subtraction,
    smooth_min, smooth_max, smooth_min_exp, smooth_min_cubic,
};
