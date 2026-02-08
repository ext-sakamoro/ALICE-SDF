//! Morph (linear interpolation) operation for SDFs (Deep Fried Edition)
//!
//! Smoothly blends between two shapes using linear interpolation.
//!
//! Author: Moroya Sakamoto

/// Morph between two SDFs
///
/// Linear interpolation: `a * (1 - t) + b * t`
/// - `t = 0.0` gives shape a
/// - `t = 1.0` gives shape b
/// - `t = 0.5` gives midpoint blend
#[inline(always)]
pub fn sdf_morph(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morph_zero() {
        assert!((sdf_morph(1.0, 2.0, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_morph_one() {
        assert!((sdf_morph(1.0, 2.0, 1.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_morph_half() {
        assert!((sdf_morph(1.0, 3.0, 0.5) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_morph_quarter() {
        assert!((sdf_morph(0.0, 4.0, 0.25) - 1.0).abs() < 1e-6);
    }
}
