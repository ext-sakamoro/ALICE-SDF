//! Vertex Attribute Quantization (meshoptimizer §quantization.cpp 移植)
//!
//! GPU / glTF-KHR_mesh_quantization / storage 用の汎用量子化 helper
//! attribute 種別に応じて適切な encoding を選択する:
//!
//! | Attribute | 推奨 encoding | ビット数 | 精度 |
//! |-----------|--------------|---------|------|
//! | Position | i16 (`snorm_i16` or 整数化) | 16 | ~0.003% of extent |
//! | Normal | i8 snorm (`snorm_i8`) | 8×3 | 1/127 ≈ 0.8% |
//! | Tangent | i8 snorm | 8×4 | 同上 |
//! | UV (0-1) | u16 unorm (`unorm_u16`) | 16 | 1/65535 ≈ 0.0015% |
//! | Color | u8 unorm (`unorm_u8`) | 8 | 1/255 ≈ 0.4% |
//! | Skinning weight | u16 unorm | 16 | 高精度必須 |
//!
//! # Reference
//!
//! - zeux/meshoptimizer §quantization.cpp `meshopt_quantizeUnorm/Snorm/Half/Float`
//! - glTF-KHR_mesh_quantization spec
//! - IEEE 754 Half precision (binary16)
//!
//! Author: Moroya Sakamoto

// ============================================================================
// snorm (Signed Normalized): [-1.0, 1.0] ↔ signed integer
// ============================================================================

/// f32 `[-1.0, 1.0]` を i8 snorm (`[-127, 127]`) に量子化
///
/// glTF 仕様 (KHR_mesh_quantization): SBYTE normalized で normal / tangent に使用
/// -128 は使わず (-127 が最小値)、0 は 0.0 に正確に対応
#[must_use]
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub fn snorm_i8_encode(v: f32) -> i8 {
    let clamped = v.clamp(-1.0, 1.0);
    (clamped * 127.0).round() as i8
}

/// i8 snorm (`[-127, 127]`) を f32 `[-1.0, 1.0]` に復元
#[must_use]
#[inline]
pub fn snorm_i8_decode(q: i8) -> f32 {
    (q as f32 / 127.0).clamp(-1.0, 1.0)
}

/// f32 `[-1.0, 1.0]` を i16 snorm (`[-32767, 32767]`) に量子化
///
/// glTF 仕様: SHORT normalized、精度 ~0.003% で高精度 attribute (position range など)
#[must_use]
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub fn snorm_i16_encode(v: f32) -> i16 {
    let clamped = v.clamp(-1.0, 1.0);
    (clamped * 32767.0).round() as i16
}

/// i16 snorm (`[-32767, 32767]`) を f32 `[-1.0, 1.0]` に復元
#[must_use]
#[inline]
pub fn snorm_i16_decode(q: i16) -> f32 {
    (q as f32 / 32767.0).clamp(-1.0, 1.0)
}

// ============================================================================
// unorm (Unsigned Normalized): [0.0, 1.0] ↔ unsigned integer
// ============================================================================

/// f32 `[0.0, 1.0]` を u8 unorm (`[0, 255]`) に量子化
///
/// glTF 仕様: UBYTE normalized、color 用
#[must_use]
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn unorm_u8_encode(v: f32) -> u8 {
    let clamped = v.clamp(0.0, 1.0);
    (clamped * 255.0).round() as u8
}

/// u8 unorm (`[0, 255]`) を f32 `[0.0, 1.0]` に復元
#[must_use]
#[inline]
pub fn unorm_u8_decode(q: u8) -> f32 {
    (q as f32 / 255.0).clamp(0.0, 1.0)
}

/// f32 `[0.0, 1.0]` を u16 unorm (`[0, 65535]`) に量子化
///
/// glTF 仕様: USHORT normalized、UV / skinning weight / 中精度 attribute
#[must_use]
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn unorm_u16_encode(v: f32) -> u16 {
    let clamped = v.clamp(0.0, 1.0);
    (clamped * 65535.0).round() as u16
}

/// u16 unorm (`[0, 65535]`) を f32 `[0.0, 1.0]` に復元
#[must_use]
#[inline]
pub fn unorm_u16_decode(q: u16) -> f32 {
    (q as f32 / 65535.0).clamp(0.0, 1.0)
}

// ============================================================================
// FP16 (IEEE 754 binary16 / Half precision)
// ============================================================================

/// f32 → IEEE 754 binary16 (half precision) の bit representation を返す
///
/// # 特徴
///
/// - 5-bit exponent + 10-bit mantissa + 1-bit sign = 16-bit
/// - 数値範囲: ±6.55e4、精度 ~0.1% (10-bit mantissa)
/// - subnormal / infinity / NaN サポート
///
/// # 用途
///
/// - GPU で float attribute の memory bandwidth 半減
/// - EXT_meshopt_compression でも使用
#[must_use]
pub fn half_encode(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_biased = (bits >> 23) & 0xFF;
    let mantissa = bits & 0x7FFFFF;

    // Special: NaN
    if exp_biased == 0xFF {
        if mantissa != 0 {
            return sign | 0x7E00; // NaN
        }
        return sign | 0x7C00; // Infinity
    }

    // Zero or subnormal that becomes zero in FP16
    if exp_biased == 0 {
        return sign;
    }

    let exp_unbiased = (exp_biased as i32) - 127;
    let new_exp = exp_unbiased + 15;

    if new_exp >= 31 {
        return sign | 0x7C00; // overflow → infinity
    }
    if new_exp <= 0 {
        // subnormal or underflow
        if new_exp < -10 {
            return sign; // underflow → 0
        }
        let mantissa_shift = mantissa | 0x800000;
        let shift = 14 - new_exp;
        let shifted = mantissa_shift >> shift;
        // round to nearest even
        let rounded = (shifted + 1) >> 1;
        return sign | (rounded as u16);
    }

    let new_exp_u = new_exp as u16;
    let new_mantissa = (mantissa >> 13) as u16;
    // round to nearest even (下位 13 bit で判定)
    let round_bit = (mantissa >> 12) & 1;
    let sticky = (mantissa & 0x0FFF) != 0;
    let round_up = round_bit == 1 && (sticky || (new_mantissa & 1) == 1);
    let base = sign | (new_exp_u << 10) | new_mantissa;
    if round_up {
        base + 1
    } else {
        base
    }
}

/// IEEE 754 binary16 (half precision) の bit → f32
#[must_use]
pub fn half_decode(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x03FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign); // signed zero
        }
        // subnormal
        let m = mantissa as f32;
        let val = m * (2.0_f32).powi(-24);
        return if sign != 0 { -val } else { val };
    }

    if exp == 31 {
        if mantissa == 0 {
            return f32::from_bits(sign | 0x7F80_0000); // infinity
        }
        return f32::from_bits(sign | 0x7FC0_0000); // NaN
    }

    let new_exp = (exp as i32 - 15 + 127) as u32;
    let new_mantissa = mantissa << 13;
    f32::from_bits(sign | (new_exp << 23) | new_mantissa)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // snorm_i8
    // ------------------------------------------------------------------------

    #[test]
    fn test_snorm_i8_boundary() {
        assert_eq!(snorm_i8_encode(1.0), 127);
        assert_eq!(snorm_i8_encode(-1.0), -127);
        assert_eq!(snorm_i8_encode(0.0), 0);
    }

    #[test]
    fn test_snorm_i8_roundtrip() {
        for &v in &[-1.0_f32, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0] {
            let q = snorm_i8_encode(v);
            let back = snorm_i8_decode(q);
            assert!((v - back).abs() < 0.01, "roundtrip failed: {v} → {q} → {back}");
        }
    }

    #[test]
    fn test_snorm_i8_clamp_out_of_range() {
        assert_eq!(snorm_i8_encode(2.0), 127);
        assert_eq!(snorm_i8_encode(-2.0), -127);
    }

    // ------------------------------------------------------------------------
    // snorm_i16
    // ------------------------------------------------------------------------

    #[test]
    fn test_snorm_i16_boundary() {
        assert_eq!(snorm_i16_encode(1.0), 32767);
        assert_eq!(snorm_i16_encode(-1.0), -32767);
        assert_eq!(snorm_i16_encode(0.0), 0);
    }

    #[test]
    fn test_snorm_i16_roundtrip_precision() {
        // i16 は ~0.003% 精度、tolerance 5e-5
        for &v in &[-0.9_f32, -0.3, 0.0, 0.3, 0.7, 0.999] {
            let q = snorm_i16_encode(v);
            let back = snorm_i16_decode(q);
            assert!((v - back).abs() < 5e-5, "i16 roundtrip: {v} → {q} → {back}");
        }
    }

    // ------------------------------------------------------------------------
    // unorm_u8
    // ------------------------------------------------------------------------

    #[test]
    fn test_unorm_u8_boundary() {
        assert_eq!(unorm_u8_encode(0.0), 0);
        assert_eq!(unorm_u8_encode(1.0), 255);
        assert_eq!(unorm_u8_encode(0.5), 128); // 127.5 → round → 128
    }

    #[test]
    fn test_unorm_u8_roundtrip() {
        for &v in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let q = unorm_u8_encode(v);
            let back = unorm_u8_decode(q);
            assert!((v - back).abs() < 0.005, "u8 roundtrip: {v} → {q} → {back}");
        }
    }

    // ------------------------------------------------------------------------
    // unorm_u16
    // ------------------------------------------------------------------------

    #[test]
    fn test_unorm_u16_boundary() {
        assert_eq!(unorm_u16_encode(0.0), 0);
        assert_eq!(unorm_u16_encode(1.0), 65535);
    }

    #[test]
    fn test_unorm_u16_precision() {
        for &v in &[0.0_f32, 0.123, 0.5, 0.876543, 1.0] {
            let q = unorm_u16_encode(v);
            let back = unorm_u16_decode(q);
            assert!((v - back).abs() < 2e-5, "u16 roundtrip: {v} → {q} → {back}");
        }
    }

    // ------------------------------------------------------------------------
    // FP16
    // ------------------------------------------------------------------------

    #[test]
    fn test_half_zero() {
        assert_eq!(half_encode(0.0), 0);
        assert_eq!(half_encode(-0.0), 0x8000);
        assert_eq!(half_decode(0), 0.0);
        assert!(half_decode(0x8000).is_sign_negative());
    }

    #[test]
    fn test_half_one() {
        // 1.0 in FP16 = 0x3C00 (sign=0, exp=15+15=30 → biased=15, mantissa=0)
        assert_eq!(half_encode(1.0), 0x3C00);
        assert!((half_decode(0x3C00) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_half_negative_one() {
        assert_eq!(half_encode(-1.0), 0xBC00);
        assert!((half_decode(0xBC00) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_half_roundtrip_precision() {
        // FP16 精度 ~0.1%、typical value でチェック
        for &v in &[
            0.5_f32, 1.5, 3.14159, 100.0, -50.0, 0.0001, 65000.0,
        ] {
            let bits = half_encode(v);
            let back = half_decode(bits);
            let rel_err = ((v - back) / v).abs();
            assert!(
                rel_err < 1e-3,
                "FP16 rel err too large: {v} → {bits:#x} → {back} (err={rel_err})"
            );
        }
    }

    #[test]
    fn test_half_overflow_to_infinity() {
        // FP16 max ≈ 6.55e4、それ以上は infinity
        let bits = half_encode(1e10);
        assert_eq!(bits, 0x7C00); // +infinity
        assert!(half_decode(bits).is_infinite());
        assert!(half_decode(bits).is_sign_positive());
    }

    #[test]
    fn test_half_negative_overflow() {
        let bits = half_encode(-1e10);
        assert_eq!(bits, 0xFC00); // -infinity
        assert!(half_decode(bits).is_infinite());
        assert!(half_decode(bits).is_sign_negative());
    }

    #[test]
    fn test_half_underflow_to_zero() {
        // FP16 subnormal 未満は 0
        let bits = half_encode(1e-30);
        assert_eq!(bits, 0);
    }
}
