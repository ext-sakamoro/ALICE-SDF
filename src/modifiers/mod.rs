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

mod twist;
mod bend;
mod repeat;
mod noise;
mod mirror;
mod revolution;
mod extrude;

pub use twist::{modifier_twist, modifier_twist_x, modifier_twist_z};
pub use bend::{modifier_bend, modifier_bend_x, modifier_bend_z, modifier_bend_cheap};
pub use repeat::{
    modifier_repeat_infinite, modifier_repeat_finite, modifier_repeat_polar,
    modifier_repeat_x, modifier_repeat_y, modifier_repeat_z,
};
pub use noise::{modifier_noise_perlin, modifier_noise_simplex, perlin_noise_3d, fbm_noise_3d};
pub use mirror::{modifier_mirror, modifier_mirror_x, modifier_mirror_y, modifier_mirror_z};
pub use revolution::modifier_revolution;
pub use extrude::{modifier_extrude, modifier_extrude_point};
