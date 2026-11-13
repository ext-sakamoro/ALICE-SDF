//! Bell pull: a domain warp by up to two Gaussian bells, for a shape that gives where it is pressed.
//!
//! A pull is `[g.x, g.y, v.x, v.y]`: the place pressed and how far it is carried, in the xy plane. A
//! point `p` of the deformed shape comes from `x(p) = p − Σ v·exp(−|p.xy − g − v|² / 2σ²)` (z does not
//! change), so the rest point that moves furthest is the one at `g`.
//!
//! **One to one.** `|∇exp| ≤ e^(−1/2)/σ`, so the Jacobian is the identity less a matrix of norm at
//! most `S = Σ|v|·e^(−1/2)/σ`. Its norm is at most `Λ = 1 + S`, and with `S < 1` the map is a
//! contraction away from the identity: one to one, determinant at least `(1 − S)²`. Nothing folds, so
//! no surface appears that the child does not have. [`BellPull::new`] refuses `S ≥ MOST_SLOPE`.
//!
//! **Distance bound.** The value is `child(x(p)) / Λ`: `x` has slope at most `Λ`, so for a 1-Lipschitz
//! child the value never exceeds the distance to the surface, and sphere tracing is safe with it.
//!
//! Every `a * b + c` is rounded twice (no `mul_add`), in the order the emitted shaders use.
//!
//! Author: Moroya Sakamoto

use alice_det_math::exp;
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// `e^(−1/2)`: the largest slope of a unit Gaussian bell, times σ
pub const BELL_SLOPE: f32 = 0.606_530_66;
/// [`BellPull::new`] refuses a slope sum `S` above this
pub const MOST_SLOPE: f32 = 0.95;
/// The narrowest bell width [`BellPull::new`] accepts: `2σ²` stays a normal f32 (below it the
/// exponent `r² / 2σ²` is `0 / 0` at the pull point)
pub const LEAST_SIGMA: f32 = 1e-18;
/// The widest bell width [`BellPull::new`] accepts: `2σ²` stays finite (above it the exponent is
/// `∞ / ∞` far from the pull)
pub const MOST_SIGMA: f32 = 1e18;
/// The most slope sum the live-pull shader lets the page's uniforms reach
///
/// In `GlslShader::transpile_with_live_pull`, pulls above it are scaled down to it. It is just
/// under [`MOST_SLOPE`], so a clamped value is one [`BellPull::new`] accepts after rounding
pub const LIVE_SLOPE: f32 = 0.94;

/// Up to two pulls under one bell width
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(into = "BellPullData")]
pub struct BellPull {
    pulls: [[f32; 4]; 2],
    sigma: f32,
}

/// The stored form of a [`BellPull`]: checked again on load
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BellPullData {
    /// `[g.x, g.y, v.x, v.y]` each
    pub pulls: [[f32; 4]; 2],
    /// Width of the bell, > 0
    pub sigma: f32,
}

/// Why a bell pull was refused
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BellPullError {
    /// A value is not finite, or the width is not positive
    Bad,
    /// The pulls together could fold the shape: their slope sum
    Folds(f32),
}

impl std::fmt::Display for BellPullError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bad => write!(f, "bell pull: values must be finite and the width > 0"),
            Self::Folds(s) => write!(f, "bell pull: slope sum {s} is above {MOST_SLOPE}"),
        }
    }
}

impl std::error::Error for BellPullError {}

/// The reader of a stored bell pull: `SdfNode::BellPull`'s `pull` field reads through it
/// (`#[serde(deserialize_with)]`) rather than a `Deserialize` impl of [`BellPull`], so the type has no
/// reader that code naming the type would reach. It reads [`BellPullData`] and checks it again
/// ([`BellPull::new`]: finite values, a positive width, a slope sum at most [`MOST_SLOPE`])
///
/// # Errors
///
/// The data does not parse, or [`BellPull::new`] refuses it
#[allow(
    clippy::disallowed_methods,
    reason = "stored form other-type: the pulls are read as a field of an SdfNode, whose stored tree goes through the tree loader's identifier check; this reads BellPullData and checks it again (the slope bound is recomputed, none is read)"
)]
// ALLOW-UNWIRED: called by the derived Deserialize of SdfNode through #[serde(deserialize_with = "crate::transforms::bell_pull::deserialize_pull")] on the BellPull variant's pull field (src/types/mod.rs), a string the guard does not read
pub(crate) fn deserialize_pull<'de, D: serde::Deserializer<'de>>(
    de: D,
) -> Result<BellPull, D::Error> {
    BellPull::try_from(BellPullData::deserialize(de)?).map_err(serde::de::Error::custom)
}

impl TryFrom<BellPullData> for BellPull {
    type Error = BellPullError;
    fn try_from(d: BellPullData) -> Result<Self, BellPullError> {
        Self::new(d.pulls, d.sigma)
    }
}

impl From<BellPull> for BellPullData {
    fn from(b: BellPull) -> Self {
        Self {
            pulls: b.pulls,
            sigma: b.sigma,
        }
    }
}

fn length(v: [f32; 4]) -> f32 {
    (v[2] * v[2] + v[3] * v[3]).sqrt()
}

impl BellPull {
    /// # Errors
    ///
    /// A value that is not finite, a width outside [`LEAST_SIGMA`]..=[`MOST_SIGMA`], or pulls that
    /// could fold the shape
    pub fn new(pulls: [[f32; 4]; 2], sigma: f32) -> Result<Self, BellPullError> {
        // σ within [LEAST_SIGMA, MOST_SIGMA] keeps 2σ² a normal, finite f32 (the exponent is never
        // 0/0 or ∞/∞); NaN fails both comparisons
        if !(LEAST_SIGMA..=MOST_SIGMA).contains(&sigma)
            || pulls.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(BellPullError::Bad);
        }
        let s = (length(pulls[0]) + length(pulls[1])) * BELL_SLOPE / sigma;
        if s > MOST_SLOPE {
            return Err(BellPullError::Folds(s));
        }
        Ok(Self { pulls, sigma })
    }

    /// The bound on the map's slope, `1 + S`
    #[must_use]
    pub fn lambda(&self) -> f32 {
        1.0 + (length(self.pulls[0]) + length(self.pulls[1])) * BELL_SLOPE / self.sigma
    }

    /// `x(p)`
    #[must_use]
    pub fn back(&self, p: Vec3) -> Vec3 {
        let mut x = p;
        for u in &self.pulls {
            let (rx, ry) = (p.x - u[0] - u[2], p.y - u[1] - u[3]);
            let w = exp(-(rx * rx + ry * ry) / (2.0 * self.sigma * self.sigma));
            x.x -= u[2] * w;
            x.y -= u[3] * w;
        }
        x
    }

    /// The pulls
    #[must_use]
    pub const fn pulls(&self) -> [[f32; 4]; 2] {
        self.pulls
    }

    /// The bell's width
    #[must_use]
    pub const fn sigma(&self) -> f32 {
        self.sigma
    }
}
