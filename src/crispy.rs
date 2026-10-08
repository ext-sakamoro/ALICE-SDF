//! Internal math helpers shared by the evaluators
//!
//! Not part of the public API (crate-private since 5.0.0).
//!
//! Author: Moroya Sakamoto

/// Fast inverse square root (Quake III style, one Newton-Raphson iteration)
///
/// Accuracy: ~0.175% relative error. Sufficient for normal estimation,
/// gradient normalization, and lighting math.
#[cfg(feature = "terrain")]
#[inline(always)]
pub fn fast_inv_sqrt(x: f32) -> f32 {
    let half = 0.5 * x;
    let i = 0x5f375a86u32.wrapping_sub(f32::to_bits(x) >> 1);
    let y = f32::from_bits(i);
    y * ((half * y) * -y + 1.5)
}

/// Normalize a 2D gradient (gx, gz) using fast inverse square root.
///
/// Returns `(gx * inv_len, gz * inv_len)`. Returns `(0.0, 0.0)` if near zero.
#[cfg(feature = "terrain")]
#[inline(always)]
pub fn fast_normalize_2d(gx: f32, gz: f32) -> (f32, f32) {
    let len_sq = gx * gx + (gz * gz);
    if len_sq < 1e-12 {
        return (0.0, 0.0);
    }
    let inv_len = fast_inv_sqrt(len_sq);
    (gx * inv_len, gz * inv_len)
}

/// Round to nearest integer with ties toward `+∞`: `floor(x + 0.5)`.
///
/// This is the **only** rounding rule used by the repeat / polar / helix laws.
/// `f32::round` (ties away from zero), `wide::f32x8::round` (ties to even on
/// AVX / NEON, away from zero on the SSE2 fallback), Cranelift `nearest`
/// (ties to even), GLSL `round` (implementation-defined), WGSL `round`
/// (ties to even) and HLSL `round` (ties away from zero) all disagree at
/// exact `.5` inputs, and a marching-cubes grid whose step divides the
/// repeat spacing lands on those inputs systematically. `floor` is
/// bit-identical on every path, so every evaluator (tree, compiled scalar,
/// SIMD, JIT, GLSL / WGSL / HLSL) must call this or emit `floor(x + 0.5)`.
#[inline(always)]
pub fn round_half_up(x: f32) -> f32 {
    (x + 0.5).floor()
}

/// FNV-1a hash (64-bit) — fast, well-distributed, no dependencies.
#[inline]
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x00000100000001B3;
    let mut hash = OFFSET;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}
#[allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_half_up_ties_toward_positive_infinity() {
        // Exact .5 inputs: the whole point of the helper.
        for (x, want) in [
            (0.5f32, 1.0f32),
            (-0.5, 0.0),
            (1.5, 2.0),
            (-1.5, -1.0),
            (2.5, 3.0),
            (-2.5, -2.0),
        ] {
            assert_eq!(round_half_up(x), want, "round_half_up({x})");
        }
        // Away from ties it agrees with `f32::round`.
        for x in [0.0f32, 0.25, 0.75, -0.25, -0.75, 3.1, -3.9, 1e5 + 0.3] {
            assert_eq!(round_half_up(x), x.round(), "round_half_up({x})");
        }
    }

    #[cfg(feature = "terrain")]
    #[test]
    fn test_fast_inv_sqrt_accuracy() {
        let test_values = [0.25f32, 1.0, 4.0, 16.0, 100.0, 0.01];
        for x in test_values {
            let expected = 1.0 / x.sqrt();
            let got = fast_inv_sqrt(x);
            let error = ((got - expected) / expected).abs();
            assert!(
                error < 0.002,
                "fast_inv_sqrt({}) = {}, expected {}, error = {:.4}%",
                x,
                got,
                expected,
                error * 100.0,
            );
        }
    }

    #[cfg(feature = "terrain")]
    #[test]
    fn test_fast_inv_sqrt_large() {
        let x = 10000.0f32;
        let expected = 1.0 / x.sqrt();
        let got = fast_inv_sqrt(x);
        assert!((got - expected).abs() / expected < 0.002);
    }

    #[cfg(feature = "terrain")]
    #[test]
    fn test_fast_normalize_2d() {
        let (nx, nz) = fast_normalize_2d(3.0, 4.0);
        let len = nx.hypot(nz);
        assert!(
            (len - 1.0).abs() < 0.01,
            "Should be unit length, got {}",
            len
        );
        assert!((nx - 0.6).abs() < 0.01);
        assert!((nz - 0.8).abs() < 0.01);
    }

    #[cfg(feature = "terrain")]
    #[test]
    fn test_fast_normalize_2d_zero() {
        let (nx, nz) = fast_normalize_2d(0.0, 0.0);
        assert_eq!(nx, 0.0);
        assert_eq!(nz, 0.0);
    }

    #[cfg(feature = "terrain")]
    #[test]
    fn test_fast_normalize_2d_small() {
        let (nx, nz) = fast_normalize_2d(1e-7, 0.0);
        assert_eq!(nx, 0.0);
        assert_eq!(nz, 0.0);
    }

    #[test]
    fn test_fnv1a_hash_deterministic() {
        let h1 = fnv1a_hash(b"Sphere");
        let h2 = fnv1a_hash(b"Sphere");
        assert_eq!(h1, h2);
        // Different inputs produce different hashes
        let h3 = fnv1a_hash(b"Box");
        assert_ne!(h1, h3);
    }
}
