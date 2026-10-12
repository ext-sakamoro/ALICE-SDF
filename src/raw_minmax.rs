//! The platform `max` / `min`, for values that are never stored.
//!
//! `f32::max` / `f32::min` (and the `f64` and `wide::f32x8` forms) leave the
//! sign of an equal pair of zeros unspecified, and the targets differ: aarch64
//! returns `+0` from `max(+0, -0)`, x86_64 returns one of the arguments. A
//! value that reaches a distance, a bound, a mesh or a persisted byte must not
//! depend on that, so the library code is not allowed to call them
//! (`.github/clippy-lib/clippy.toml`) and uses the Law helpers
//! (`law_max_f32` / `law_min_f32`, `Real::max` / `Real::min`) instead.
//!
//! The functions here are the one exception, and the only place in the library
//! where the lint is allowed. **Never use them for a value that reaches a
//! distance, bound, mesh or persisted byte.** They exist for colour, shading,
//! statistics and similar values whose zero sign cannot matter, and each call
//! site has to be visible as an exception: `scripts/raw_minmax_manifest.py`
//! checks that only the modules listed in `scripts/raw_minmax_manifest.toml`
//! call them, with the listed number of calls, and that no geometry module
//! does. Calling one of these does not exempt a module from the libm ban,
//! because nothing here allows anything outside this file.

use wide::f32x8;

/// Platform `f32::max`: the zero sign of an equal pair is target-dependent.
/// Never for a value that reaches a distance, bound, mesh or persisted byte.
#[inline(always)]
#[must_use]
#[allow(
    clippy::disallowed_methods,
    reason = "the one place the platform max is allowed; callers are listed in scripts/raw_minmax_manifest.toml"
)]
pub const fn raw_max_f32(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// Platform `f32::min`: the zero sign of an equal pair is target-dependent.
/// Never for a value that reaches a distance, bound, mesh or persisted byte.
#[inline(always)]
#[must_use]
#[allow(
    clippy::disallowed_methods,
    reason = "the one place the platform min is allowed; callers are listed in scripts/raw_minmax_manifest.toml"
)]
pub const fn raw_min_f32(a: f32, b: f32) -> f32 {
    a.min(b)
}

/// Platform `f64::max`: the zero sign of an equal pair is target-dependent.
/// Never for a value that reaches a distance, bound, mesh or persisted byte.
#[inline(always)]
#[must_use]
#[allow(
    clippy::disallowed_methods,
    reason = "the one place the platform max is allowed; callers are listed in scripts/raw_minmax_manifest.toml"
)]
pub const fn raw_max_f64(a: f64, b: f64) -> f64 {
    a.max(b)
}

/// Platform `f64::min`: the zero sign of an equal pair is target-dependent.
/// Never for a value that reaches a distance, bound, mesh or persisted byte.
#[inline(always)]
#[must_use]
#[allow(
    clippy::disallowed_methods,
    reason = "the one place the platform min is allowed; callers are listed in scripts/raw_minmax_manifest.toml"
)]
pub const fn raw_min_f64(a: f64, b: f64) -> f64 {
    a.min(b)
}

/// Platform `f32x8::max` per lane: the zero sign of an equal pair is
/// target-dependent. Never for a value that reaches a distance, bound, mesh or
/// persisted byte.
#[inline(always)]
#[must_use]
#[allow(
    clippy::disallowed_methods,
    reason = "the one place the platform SIMD max is allowed; callers are listed in scripts/raw_minmax_manifest.toml"
)]
pub fn raw_max_f32x8(a: f32x8, b: f32x8) -> f32x8 {
    a.max(b)
}

/// Platform `f32x8::min` per lane: the zero sign of an equal pair is
/// target-dependent. Never for a value that reaches a distance, bound, mesh or
/// persisted byte.
#[inline(always)]
#[must_use]
#[allow(
    clippy::disallowed_methods,
    reason = "the one place the platform SIMD min is allowed; callers are listed in scripts/raw_minmax_manifest.toml"
)]
pub fn raw_min_f32x8(a: f32x8, b: f32x8) -> f32x8 {
    a.min(b)
}

/// `x.raw_max(y)` / `x.raw_min(y)`: the helpers above in method form. The same
/// rule applies: never for a value that reaches a distance, bound, mesh or
/// persisted byte, and every use is counted in `scripts/raw_minmax_manifest.toml`.
pub trait RawMinMax: Sized {
    /// Platform maximum (target-dependent zero sign).
    #[must_use]
    fn raw_max(self, other: Self) -> Self;
    /// Platform minimum (target-dependent zero sign).
    #[must_use]
    fn raw_min(self, other: Self) -> Self;
}

impl RawMinMax for f32 {
    #[inline(always)]
    fn raw_max(self, other: Self) -> Self {
        raw_max_f32(self, other)
    }
    #[inline(always)]
    fn raw_min(self, other: Self) -> Self {
        raw_min_f32(self, other)
    }
}

impl RawMinMax for f64 {
    #[inline(always)]
    fn raw_max(self, other: Self) -> Self {
        raw_max_f64(self, other)
    }
    #[inline(always)]
    fn raw_min(self, other: Self) -> Self {
        raw_min_f64(self, other)
    }
}

impl RawMinMax for f32x8 {
    #[inline(always)]
    fn raw_max(self, other: Self) -> Self {
        raw_max_f32x8(self, other)
    }
    #[inline(always)]
    fn raw_min(self, other: Self) -> Self {
        raw_min_f32x8(self, other)
    }
}
