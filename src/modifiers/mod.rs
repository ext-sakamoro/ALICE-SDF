//! Modifier operations for SDFs (Deep Fried Edition)
//!
//! Modifiers deform the space before evaluating the child SDF,
//! creating effects like twisting, bending, and repetition.
//!
//! # Deep Fried Optimizations
//! - **Forced Inlining**: All functions use `#[inline(always)]`.
//! - **Simultaneous Trig**: Uses `sin_cos()` for twist/bend.
//! - **Branchless**: Removed safety checks in bend.
//! - **Fast Modulo**: Round-based repeat logic.
//!
//! Author: Moroya Sakamoto

mod bend;
mod displacement;
mod extrude;
mod heightmap_displacement;
mod icosahedral;
mod ifs;
mod mirror;
mod noise;
mod octant_mirror;
mod polar_repeat;
mod repeat;
mod revolution;
mod surface_roughness;
mod sweep;
mod taper;
mod twist;

pub use bend::{modifier_bend, modifier_bend_cheap, modifier_bend_x, modifier_bend_z};
pub use displacement::modifier_displacement;
pub use extrude::{modifier_extrude, modifier_extrude_point};
pub use heightmap_displacement::{bilinear_sample, heightmap_displacement};
pub use icosahedral::icosahedral_fold;
pub use ifs::{ifs_fold, ifs_fold_with_scale};
pub use mirror::{modifier_mirror, modifier_mirror_x, modifier_mirror_y, modifier_mirror_z};
pub use noise::{fbm_noise_3d, modifier_noise_perlin, modifier_noise_simplex, perlin_noise_3d};
pub use octant_mirror::modifier_octant_mirror;
pub use polar_repeat::{modifier_polar_repeat, modifier_polar_repeat_rk};
pub use repeat::{
    modifier_repeat_finite, modifier_repeat_infinite, modifier_repeat_infinite_rk,
    modifier_repeat_polar, modifier_repeat_x, modifier_repeat_y, modifier_repeat_z,
};
pub use revolution::modifier_revolution;
pub use surface_roughness::{fbm, surface_roughness};
pub use sweep::{modifier_sweep_bezier, sweep_bezier_dist_y};
pub use taper::modifier_taper;
pub use twist::{modifier_twist, modifier_twist_x, modifier_twist_z};
