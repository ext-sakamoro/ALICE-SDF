//! Smooth blends with a choice of kernel (`SdfNode::SmoothBlend`)
//!
//! One law for three smooth minima, each written as `smin(a, b) = m − D(x)`
//! with `m = min(a, b)`, `x = |a − b|` and a deficit `D ≥ 0` that is 0 outside
//! the blend. The kernels are our own derivations; the same methods are also
//! known from Inigo Quilez's articles (iquilezles.org/articles/smin).
//!
//! | kernel | `D(x)` for `x < k` (0 beyond) | continuity | deficit at `a = b` |
//! |---|---|---|---|
//! | [`SmoothKernel::Circular`] | `k − x/2 − √(k²/2 − x²/4)` | C1 | `k(1 − 1/√2)` |
//! | [`SmoothKernel::Quartic`] | `k·h³(4 − h)/16`, `h = 1 − x/k` | C2 | `3k/16` |
//! | [`SmoothKernel::Sigmoid`] | `x / (eˣᐟᵏ − 1)` (all `x`, limit `k` at 0) | C∞ | `k` |
//!
//! * **Circular**: every level set is a circular arc of radius `k/√2`
//!   tangent to the two lines of `min`'s level set (in `u = (a+b)/2`,
//!   `v = (a−b)/2` the fillet of the corner `u = c + |v|`).
//! * **Quartic**: the quartic `g(s) = (1 − s)³(3 + s)/16` with
//!   `g(1) = g'(1) = g''(1) = 0` (meets `min` with two continuous
//!   derivatives) and `g'(0) = −1/2`, `g'''(0) = 0` (cancels the kink of `min`
//!   at `a = b` to second order); `D(x) = k·g(x/k)`, written in `h = 1 − x/k`.
//! * **Sigmoid**: `D(x) = x/(eˣᐟᵏ − 1)`, smooth everywhere with unbounded
//!   support, decaying like `x·e^(−x/k)`.
//!
//! In each kernel the two blend weights `∂smin/∂a`, `∂smin/∂b` lie in
//! `[0, 1]` and sum to 1, so blending two 1-Lipschitz fields gives a
//! 1-Lipschitz field.
//!
//! Modes: intersection is `−smin(−a, −b)`, subtraction of `b` from `a` is
//! `−smin(−a, b)`. A width `k ≤ 0` (or NaN) is no blend: `min` / `max`.
//!
//! The kernel table ([`SmoothKernel::deficit_r`]) is the one place each law
//! is written; the tree evaluator, the compiled bytecode and the SIMD lanes
//! all reach it.
//!
//! Oracle: `tests/test_smooth_blend_oracle.rs`.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::LawMinMax;
use crate::compiled::real::Real;
use serde::{Deserialize, Serialize};

/// The kernel of a [`SdfNode::SmoothBlend`](crate::SdfNode::SmoothBlend).
///
/// Serialized by name; a reader that meets a name it does not know fails
/// (serde's unknown-variant error) instead of falling back to a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SmoothKernel {
    /// Circular-arc level sets, C1, deficit `k(1 − 1/√2)` at `a = b`
    Circular,
    /// C2 quartic, deficit `3k/16` at `a = b`
    Quartic,
    /// C∞ logistic-type kernel with unbounded support, deficit `k` at `a = b`
    Sigmoid,
}

/// The boolean a [`SdfNode::SmoothBlend`](crate::SdfNode::SmoothBlend) smooths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlendMode {
    /// `smin(a, b)`
    Union,
    /// `−smin(−a, −b)`
    Intersection,
    /// `b` subtracted from `a`: `−smin(−a, b)`
    Subtraction,
}

impl BlendMode {
    /// Every mode, in the order of its bytecode code ([`Self::code`]).
    pub const ALL: [Self; 3] = [Self::Union, Self::Intersection, Self::Subtraction];

    /// The mode's code in the compiled bytecode and the shaders.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::Union => 0,
            Self::Intersection => 1,
            Self::Subtraction => 2,
        }
    }

    /// The mode with bytecode code `code`, or `None`.
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::Union),
            1 => Some(Self::Intersection),
            2 => Some(Self::Subtraction),
            _ => None,
        }
    }
}

/// The kernel and mode stored in a `SmoothBlend` instruction's
/// `params[1]` / `params[2]` (codes as exact small floats).
///
/// # Panics
///
/// On a code that names no kernel or mode: the bytecode was not produced by
/// the compiler (a corrupted or hand-edited program), and evaluating it with
/// a guessed kernel would return a wrong distance.
#[inline(always)]
#[must_use]
pub fn smooth_blend_decode(kernel_code: f32, mode_code: f32) -> (SmoothKernel, BlendMode) {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (kc, mc) = (kernel_code as u32, mode_code as u32);
    match (
        SmoothKernel::from_code(kc).filter(|_| kc as f32 == kernel_code),
        BlendMode::from_code(mc).filter(|_| mc as f32 == mode_code),
    ) {
        (Some(kn), Some(md)) => (kn, md),
        _ => panic!(
            "SmoothBlend instruction with kernel code {kernel_code} / mode code {mode_code}: \
             not produced by the compiler"
        ),
    }
}

impl SmoothKernel {
    /// Every kernel, in the order of its bytecode code ([`Self::code`]).
    pub const ALL: [Self; 3] = [Self::Circular, Self::Quartic, Self::Sigmoid];

    /// The kernel's code in the compiled bytecode and the shaders.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::Circular => 0,
            Self::Quartic => 1,
            Self::Sigmoid => 2,
        }
    }

    /// The kernel with bytecode code `code`, or `None`.
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::Circular),
            1 => Some(Self::Quartic),
            2 => Some(Self::Sigmoid),
            _ => None,
        }
    }

    /// The deficit `D(x) ≥ 0` below `min` at separation `x = |a − b|`, for a
    /// width `k > 0` (generic over [`Real`]). This is the kernel table.
    #[inline(always)]
    pub fn deficit_r<R: Real>(self, x: R, k: f32) -> R {
        match self {
            Self::Circular => {
                // k − x/2 − √(k²/2 − x²/4) inside, 0 outside (x ≥ k)
                let inside = (R::splat(k)
                    - x * R::splat(0.5)
                    - (R::splat(k * k * 0.5) - x * x * R::splat(0.25))
                        .max(R::zero())
                        .sqrt())
                .max(R::zero());
                R::select(x.lt(R::splat(k)), inside, R::zero())
            }
            Self::Quartic => {
                // k·h³(4 − h)/16, h = max(1 − x/k, 0)
                let h = (R::one() - x * R::splat(1.0 / k)).max(R::zero());
                h * h * h * (R::splat(4.0) - h) * R::splat(k * (1.0 / 16.0))
            }
            Self::Sigmoid => x.map(|x| sigmoid_deficit(x, k)),
        }
    }
}

/// `x / (e^(x/k) − 1)` for `x ≥ 0`, `k > 0`, finite for every input:
/// `k` at `x = 0`; Kahan's `expm1` from `exp` and `ln` for `x/k < 1` (no
/// cancellation in `e^y − 1`); `x·w/(1 − w)` with `w = e^(−x/k)` beyond, which
/// is 0 once `w` underflows (and for `x = ∞`).
#[inline(always)]
fn sigmoid_deficit(x: f32, k: f32) -> f32 {
    if x == 0.0 {
        return k;
    }
    let y = x / k;
    if y < 1.0 {
        let u = alice_det_math::exp(y);
        let em1 = if u == 1.0 {
            y
        } else {
            (u - 1.0) * y / alice_det_math::ln(u)
        };
        x / em1
    } else {
        let w = alice_det_math::exp(-y);
        if w == 0.0 {
            0.0
        } else {
            x * w / (1.0 - w)
        }
    }
}

/// The smooth minimum of `a` and `b` with `kernel` and width `k` (generic over [`Real`]).
#[inline(always)]
pub fn smooth_blend_min_r<R: Real>(a: R, b: R, k: f32, kernel: SmoothKernel) -> R {
    let m = a.min(b);
    if !(k > 0.0) {
        return m;
    }
    m - kernel.deficit_r((a - b).abs(), k)
}

/// The blend of `a` and `b` (generic over [`Real`]): the law of `SdfNode::SmoothBlend`.
#[inline(always)]
pub fn smooth_blend_r<R: Real>(a: R, b: R, k: f32, kernel: SmoothKernel, mode: BlendMode) -> R {
    match mode {
        BlendMode::Union => smooth_blend_min_r(a, b, k, kernel),
        BlendMode::Intersection => -smooth_blend_min_r(-a, -b, k, kernel),
        BlendMode::Subtraction => -smooth_blend_min_r(-a, b, k, kernel),
    }
}

/// The blend of two distances `a` and `b`: the law of `SdfNode::SmoothBlend`.
#[inline(always)]
pub fn smooth_blend(a: f32, b: f32, k: f32, kernel: SmoothKernel, mode: BlendMode) -> f32 {
    smooth_blend_r::<f32>(a, b, k, kernel, mode)
}

/// `D'(x)`, the derivative of the kernel's deficit (`−1/2 ≤ D' ≤ 0`), for `k > 0`.
#[inline(always)]
fn deficit_slope(kernel: SmoothKernel, x: f32, k: f32) -> f32 {
    match kernel {
        SmoothKernel::Circular => {
            if x >= k {
                0.0
            } else {
                let r = (k * k * 0.5 - x * x * 0.25).law_max(0.0).sqrt();
                if r == 0.0 {
                    0.0
                } else {
                    (x * 0.25 / r - 0.5).law_min(0.0).law_max(-0.5)
                }
            }
        }
        SmoothKernel::Quartic => {
            let h = (1.0 - x / k).law_max(0.0);
            -(h * h * (3.0 - h)) * 0.25
        }
        SmoothKernel::Sigmoid => {
            // D = k·B(y), y = x/k, B(y) = y/(e^y − 1): D'(x) = B'(y)
            let y = x / k;
            if y < 0.1 {
                // B'(y) = −1/2 + y/6 − y³/180 + y⁵/5040 − …
                let y2 = y * y;
                -0.5 + y * (1.0 / 6.0 + y2 * (-1.0 / 180.0 + y2 * (1.0 / 5040.0)))
            } else {
                let w = alice_det_math::exp(-y);
                if w == 0.0 {
                    0.0
                } else {
                    let one_w = 1.0 - w;
                    w * (one_w - y) / (one_w * one_w)
                }
            }
        }
    }
}

/// The weights `(∂f/∂a, ∂f/∂b)` of the blend `f = smooth_blend(a, b, ..)`,
/// for the analytic gradient `∇f = wa·∇a + wb·∇b`. For union and
/// intersection both weights lie in `[0, 1]` and sum to 1; for subtraction
/// `wb ≤ 0` (the gradient of `b` enters negated).
#[inline]
#[must_use]
pub fn smooth_blend_weights(
    a: f32,
    b: f32,
    k: f32,
    kernel: SmoothKernel,
    mode: BlendMode,
) -> (f32, f32) {
    // weights of smin(p, q): the nearer argument gets 1 + D', the other −D'
    let smin_w = |p: f32, q: f32| -> (f32, f32) {
        let near_p = p <= q;
        let ds = if k > 0.0 {
            deficit_slope(kernel, (p - q).abs(), k)
        } else {
            0.0
        };
        if near_p {
            (1.0 + ds, -ds)
        } else {
            (-ds, 1.0 + ds)
        }
    };
    match mode {
        BlendMode::Union => smin_w(a, b),
        BlendMode::Intersection => smin_w(-a, -b),
        BlendMode::Subtraction => {
            let (wa, wb) = smin_w(-a, b);
            (wa, -wb)
        }
    }
}

/// The width used where a positive `k` is required (bounds, Lipschitz): `k`
/// if positive, else 0.
#[inline(always)]
#[must_use]
pub fn smooth_blend_width(k: f32) -> f32 {
    k.law_max(0.0)
}

// ---------------------------------------------------------------------------
// Shader forms of the kernel table
// ---------------------------------------------------------------------------

/// The C-family (GLSL / HLSL) text of the three kernels, with `PREC` for the
/// precision qualifier of every local (GLSL `ALICE_PRECISE`, HLSL `precise`).
///
/// Written operation for operation as [`SmoothKernel::deficit_r`] and
/// [`smooth_blend_min_r`] are (the same `k ≤ 0` guard, the same separation
/// `|a − b|`, the same branch points and grouping), so the GPU follows the
/// CPU law up to the device's own rounding; every result goes through a
/// qualified local and every `return` is a name (ALICE-Law G3).
const SMOOTH_BLEND_C_TEMPLATE: [&str; 3] = [
    // Circular
    r"float alice_smin_circular(float a, float b, float k) {
    PREC float m = min(a, b);
    PREC float x = abs(a - b);
    PREC float r = m;
    if (k > 0.0 && x < k) {
        PREC float s = sqrt(max(k * k * 0.5 - x * x * 0.25, 0.0));
        PREC float d = max(k - x * 0.5 - s, 0.0);
        r = m - d;
    }
    return r;
}
",
    // Quartic
    r"float alice_smin_quartic(float a, float b, float k) {
    PREC float m = min(a, b);
    PREC float r = m;
    if (k > 0.0) {
        PREC float x = abs(a - b);
        PREC float h = max(1.0 - x * (1.0 / k), 0.0);
        PREC float d = h * h * h * (4.0 - h) * (k * 0.0625);
        r = m - d;
    }
    return r;
}
",
    // Sigmoid
    r"float alice_smin_sigmoid(float a, float b, float k) {
    PREC float m = min(a, b);
    PREC float r = m;
    if (k > 0.0) {
        PREC float x = abs(a - b);
        PREC float d = k;
        if (x != 0.0) {
            PREC float y = x / k;
            if (y < 1.0) {
                PREC float u = exp(y);
                PREC float em1 = y;
                if (u != 1.0) {
                    em1 = (u - 1.0) * y / log(u);
                }
                d = x / em1;
            } else {
                PREC float w = exp(-y);
                d = 0.0;
                if (w != 0.0) {
                    d = x * w / (1.0 - w);
                }
            }
        }
        r = m - d;
    }
    return r;
}
",
];

/// The WGSL text of the three kernels (same law as the C-family text).
const SMOOTH_BLEND_WGSL: [&str; 3] = [
    r"fn alice_smin_circular(a: f32, b: f32, k: f32) -> f32 {
    let m = min(a, b);
    let x = abs(a - b);
    var r = m;
    if (k > 0.0 && x < k) {
        let s = sqrt(max(k * k * 0.5 - x * x * 0.25, 0.0));
        let d = max(k - x * 0.5 - s, 0.0);
        r = m - d;
    }
    return r;
}
",
    r"fn alice_smin_quartic(a: f32, b: f32, k: f32) -> f32 {
    let m = min(a, b);
    var r = m;
    if (k > 0.0) {
        let x = abs(a - b);
        let h = max(1.0 - x * (1.0 / k), 0.0);
        let d = h * h * h * (4.0 - h) * (k * 0.0625);
        r = m - d;
    }
    return r;
}
",
    r"fn alice_smin_sigmoid(a: f32, b: f32, k: f32) -> f32 {
    let m = min(a, b);
    var r = m;
    if (k > 0.0) {
        let x = abs(a - b);
        var d = k;
        if (x != 0.0) {
            let y = x / k;
            if (y < 1.0) {
                let u = exp(y);
                var em1 = y;
                if (u != 1.0) {
                    em1 = (u - 1.0) * y / log(u);
                }
                d = x / em1;
            } else {
                let w = exp(-y);
                d = 0.0;
                if (w != 0.0) {
                    d = x * w / (1.0 - w);
                }
            }
        }
        r = m - d;
    }
    return r;
}
",
];

/// The shader function that evaluates `kernel`'s smooth minimum.
#[must_use]
pub const fn smooth_blend_helper_name(kernel: SmoothKernel) -> &'static str {
    match kernel {
        SmoothKernel::Circular => "alice_smin_circular",
        SmoothKernel::Quartic => "alice_smin_quartic",
        SmoothKernel::Sigmoid => "alice_smin_sigmoid",
    }
}

fn helper_index(name: &str) -> Option<usize> {
    SmoothKernel::ALL
        .iter()
        .position(|&k| smooth_blend_helper_name(k) == name)
}

/// GLSL source of the helper `name` (one of [`smooth_blend_helper_name`]).
#[must_use]
pub fn smooth_blend_helper_glsl(name: &str) -> Option<&'static str> {
    static TEXT: std::sync::OnceLock<[String; 3]> = std::sync::OnceLock::new();
    let t =
        TEXT.get_or_init(|| SMOOTH_BLEND_C_TEMPLATE.map(|s| s.replace("PREC", "ALICE_PRECISE")));
    helper_index(name).map(|i| t[i].as_str())
}

/// HLSL source of the helper `name` (one of [`smooth_blend_helper_name`]).
#[must_use]
pub fn smooth_blend_helper_hlsl(name: &str) -> Option<&'static str> {
    static TEXT: std::sync::OnceLock<[String; 3]> = std::sync::OnceLock::new();
    let t = TEXT.get_or_init(|| SMOOTH_BLEND_C_TEMPLATE.map(|s| s.replace("PREC", "precise")));
    helper_index(name).map(|i| t[i].as_str())
}

/// WGSL source of the helper `name` (one of [`smooth_blend_helper_name`]).
#[must_use]
pub fn smooth_blend_helper_wgsl(name: &str) -> Option<&'static str> {
    helper_index(name).map(|i| SMOOTH_BLEND_WGSL[i])
}
