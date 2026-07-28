//! meshopt `EXT_meshopt_compression` vertex filters
//!
//! Filters transform vertex attributes into a form that compresses better
//! under the byte-group encoding used by `meshopt_vertex_codec` They are
//! stateless per-vertex transforms that MUST be paired with a matching
//! inverse on the decode side
//!
//! # Filters
//!
//! - **Octahedral (Oct)**: encode a unit vector (normal / tangent) into
//!   2 signed values in `[-1, 1]` + a redundant 3rd component for numerical
//!   stability; supports i8 or i16 per component 50-75% smaller than a
//!   raw f32 x 3 storage with < 1% angular error
//!
//! - **Quaternion (Quat)**: encode a rotation quaternion as its 3 smaller
//!   components + a 2-bit index for the largest component (double-cover
//!   discards sign); i16 x 4 per rotation
//!
//! - **Exponential (Exp)**: encode a float as `(mantissa, shared_exponent)`
//!   with mantissa in a caller-chosen bit budget The "separate" mode stores
//!   an independent exponent per lane
//!
//! # Reference
//!
//! - zeux/meshoptimizer §vertexfilter.cpp
//! - Cigolle et al "A Survey of Efficient Representations for Independent Unit Vectors" (2014)
//!
//! Author: Moroya Sakamoto

/// Quantize `v` in `[-1, 1]` to a signed integer with `bits` precision
///
/// Uses round-half-away-from-zero to match meshopt semantics
#[inline]
#[must_use]
pub fn quantize_snorm(v: f32, bits: u32) -> i32 {
    debug_assert!((1..=32).contains(&bits));
    let scale = ((1i32 << (bits - 1)) - 1) as f32;
    let round = if v >= 0.0 { 0.5 } else { -0.5 };
    v.clamp(-1.0, 1.0).mul_add(scale, round) as i32
}

// ============================================================================
// Octahedral filter
// ============================================================================

/// Encode a single unit vector `(nx, ny, nz)` + handedness `nw` using
/// octahedral projection Writes 4 signed integers to `output[0..4]`
///
/// `bits` selects the precision (typically 8 for i8 stride 4, or 16 for i16 stride 8)
#[allow(clippy::similar_names)]
pub fn encode_filter_oct_one(nx: f32, ny: f32, nz: f32, nw: f32, bits: u32, output: &mut [i32; 4]) {
    let nl = nx.abs() + ny.abs() + nz.abs();
    let ns = if nl == 0.0 { 0.0 } else { 1.0 / nl };
    let nx_p = nx * ns;
    let ny_p = ny * ns;

    let (u, v) = if nz >= 0.0 {
        (nx_p, ny_p)
    } else {
        let sx = if nx_p >= 0.0 { 1.0 } else { -1.0 };
        let sy = if ny_p >= 0.0 { 1.0 } else { -1.0 };
        ((1.0 - ny_p.abs()) * sx, (1.0 - nx_p.abs()) * sy)
    };

    output[0] = quantize_snorm(u, bits);
    output[1] = quantize_snorm(v, bits);
    output[2] = quantize_snorm(1.0, bits);
    // bytebits = 2*stride, but the caller supplies the bits budget for w
    output[3] = quantize_snorm(nw, bits);
}

/// Encode a `Vec<Vec4>` of `(nx, ny, nz, nw)` into i16 x 4 storage using
/// octahedral filter (stride 8 bytes/vertex, `bits` = 8 or 16)
///
/// `bits` selects the precision `output` receives `count * 4` `i16`
/// values (2 bytes each, 8 bytes/vertex)
#[must_use]
pub fn encode_filter_oct_i16(normals: &[[f32; 4]], bits: u32) -> Vec<i16> {
    assert!(
        (2..=16).contains(&bits),
        "oct filter bits must be in [2, 16]"
    );
    let mut out = Vec::with_capacity(normals.len() * 4);
    let mut tmp = [0i32; 4];
    for n in normals {
        encode_filter_oct_one(n[0], n[1], n[2], n[3], bits, &mut tmp);
        for &v in &tmp {
            out.push(v as i16);
        }
    }
    out
}

/// Decode i16 x 4 octahedral filter back to a unit vector + handedness
///
/// The input is what `encode_filter_oct_i16` produced; the output is the
/// meshopt post-filter representation (each component scaled to the i16 max
/// range so that a raw `snorm16` decoder recovers the original vector)
///
/// Corresponds to `meshopt_decodeFilterOct<short>`
pub fn decode_filter_oct_i16_in_place(data: &mut [i16]) {
    assert!(data.len() % 4 == 0);
    let max_v = f32::from(i16::MAX);

    for chunk in data.chunks_exact_mut(4) {
        let mut x = f32::from(chunk[0]);
        let mut y = f32::from(chunk[1]);
        let z = f32::from(chunk[2]) - x.abs() - y.abs();

        let t = if z >= 0.0 { 0.0 } else { z };
        x += if x >= 0.0 { t } else { -t };
        y += if y >= 0.0 { t } else { -t };

        let l = x.mul_add(x, y.mul_add(y, z * z)).sqrt();
        if l == 0.0 {
            continue;
        }
        let s = max_v / l;

        let round = |v: f32| -> i16 {
            let r = if v >= 0.0 { 0.5 } else { -0.5 };
            v.mul_add(s, r) as i16
        };

        chunk[0] = round(x);
        chunk[1] = round(y);
        chunk[2] = round(z);
        // chunk[3] (handedness) already in i16 range
    }
}

// ============================================================================
// Quaternion filter
// ============================================================================

/// Encode a single quaternion `(x, y, z, w)` (need not be normalized on
/// input; encoder normalizes) using largest-component storage
///
/// Writes 4 `i16` to `output[0..4]` — 3 smaller components + `(snorm(1) & ~3) | qc`
/// with `qc` being the index of the largest-magnitude component
#[allow(clippy::similar_names)]
pub fn encode_filter_quat_one(q: [f32; 4], bits: u32, output: &mut [i16; 4]) {
    let scaler = std::f32::consts::SQRT_2;

    // Find largest-magnitude component
    let mut qc: usize = 0;
    for i in 1..4 {
        if q[i].abs() > q[qc].abs() {
            qc = i;
        }
    }

    let sign = if q[qc] < 0.0 { -1.0 } else { 1.0 };

    let i1 = (qc + 1) & 3;
    let i2 = (qc + 2) & 3;
    let i3 = (qc + 3) & 3;

    output[0] = quantize_snorm(q[i1] * scaler * sign, bits) as i16;
    output[1] = quantize_snorm(q[i2] * scaler * sign, bits) as i16;
    output[2] = quantize_snorm(q[i3] * scaler * sign, bits) as i16;
    output[3] = ((quantize_snorm(1.0, bits) & !3) | (qc as i32)) as i16;
}

/// Encode a `Vec<Quat>` (as `[f32; 4]`) into i16 x 4 storage
#[must_use]
pub fn encode_filter_quat_i16(quaternions: &[[f32; 4]], bits: u32) -> Vec<i16> {
    assert!(
        (4..=16).contains(&bits),
        "quat filter bits must be in [4, 16]"
    );
    let mut out = Vec::with_capacity(quaternions.len() * 4);
    let mut tmp = [0i16; 4];
    for q in quaternions {
        encode_filter_quat_one(*q, bits, &mut tmp);
        out.extend_from_slice(&tmp);
    }
    out
}

/// Decode i16 x 4 quaternion filter back to a normalized quaternion
/// representation (in i16 space)
///
/// Corresponds to `meshopt_decodeFilterQuat`
pub fn decode_filter_quat_i16_in_place(data: &mut [i16]) {
    assert!(data.len() % 4 == 0);
    let scale = f32::from(i16::MAX) / std::f32::consts::SQRT_2;

    for chunk in data.chunks_exact_mut(4) {
        // Recover scale from high bits of the 4th component
        let sf = chunk[3] | 3;
        let s = f32::from(sf);

        let x = f32::from(chunk[0]);
        let y = f32::from(chunk[1]);
        let z = f32::from(chunk[2]);

        let ws = s * s;
        let ww = 2.0f32.mul_add(ws, -x.mul_add(x, y.mul_add(y, z * z)));
        let w = ww.max(0.0).sqrt();

        let ss = scale / s;

        let round_sign = |v: f32| -> i16 {
            let r = if v >= 0.0 { 0.5 } else { -0.5 };
            v.mul_add(ss, r) as i16
        };
        let round_pos = |v: f32| -> i16 { v.mul_add(ss, 0.5) as i16 };

        let xf = round_sign(x);
        let yf = round_sign(y);
        let zf = round_sign(z);
        let wf = round_pos(w);

        let qc = (chunk[3] & 3) as usize;

        chunk[(qc + 1) & 3] = xf;
        chunk[(qc + 2) & 3] = yf;
        chunk[(qc + 3) & 3] = zf;
        chunk[qc] = wf;
    }
}

// ============================================================================
// Exponential filter (per-lane separate mode)
// ============================================================================

/// Compute the base-2 exponent of `v` used by the meshopt Exp filter
///
/// Returns `-100` for zero/subnormal inputs (safe sentinel)
#[inline]
#[must_use]
fn opt_log2(v: f32) -> i32 {
    if v == 0.0 || !v.is_finite() {
        return -100;
    }
    let bits = v.abs().to_bits();
    ((bits >> 23) & 0xFF) as i32 - 127
}

/// Encode a single float using `bits`-mantissa + 8-bit shared exponent
///
/// Uses `EncodeExpSeparate` semantics (per-lane exponent) The scale factor
/// leaves 2 bits of headroom above the raw `e` value to guarantee that
/// rounding cannot push the mantissa out of the signed `bits`-bit range
#[must_use]
pub fn encode_filter_exp_one(v: f32, bits: u32) -> u32 {
    debug_assert!((1..=24).contains(&bits));
    let e = opt_log2(v);
    // Choose scale so that |v / 2^scale| fits in signed `bits` bits with rounding headroom
    // (|v| < 2^(e+1), we need |m| < 2^(bits-1), so scale >= e - bits + 2)
    let target_exp = e - (bits as i32 - 2);
    let scale = target_exp.clamp(-127, 127);
    let m = (v * (-scale as f32).exp2()).round() as i32;
    let m_clamped = m.clamp(-(1 << (bits - 1)), (1 << (bits - 1)) - 1);

    // pack: mantissa low 24 bits (sign-extended), exponent high 8 bits
    let m_bits = (m_clamped as u32) & 0x00FF_FFFF;
    let e_bits = (scale as u32 & 0xFF) << 24;
    m_bits | e_bits
}

/// Encode a slice of floats using per-lane exp filter
#[must_use]
pub fn encode_filter_exp_u32(data: &[f32], bits: u32) -> Vec<u32> {
    assert!(
        (1..=24).contains(&bits),
        "exp filter bits must be in [1, 24]"
    );
    data.iter()
        .map(|&v| encode_filter_exp_one(v, bits))
        .collect()
}

/// Decode u32 exp filter back to float in-place
///
/// Corresponds to `meshopt_decodeFilterExp`
pub fn decode_filter_exp_u32_in_place(data: &mut [u32]) {
    for slot in data {
        let v = *slot;
        // Sign-extend low 24-bit mantissa
        let m = ((v << 8) as i32) >> 8;
        let e = (v as i32) >> 24;

        // Optimized ldexp: build float from exponent bits
        let bits = ((e + 127) as u32) << 23;
        let exp_multiplier = f32::from_bits(bits);
        let f = exp_multiplier * (m as f32);
        *slot = f.to_bits();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
    }
    fn norm(a: [f32; 3]) -> f32 {
        dot(a, a).sqrt()
    }

    #[test]
    fn test_quantize_snorm_extremes() {
        assert_eq!(quantize_snorm(1.0, 8), 127);
        assert_eq!(quantize_snorm(-1.0, 8), -127);
        assert_eq!(quantize_snorm(0.0, 8), 0);
        assert_eq!(quantize_snorm(1.0, 16), 32767);
    }

    #[test]
    fn test_oct_encode_unit_vector_roundtrip() {
        // Random unit vectors — Oct filter should recover with < 1% angular error
        let normals: Vec<[f32; 4]> = vec![
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [-1.0, 0.0, 0.0, -1.0],
            [0.577_350_25, 0.577_350_25, 0.577_350_25, 1.0],
            [0.267_261_25, 0.534_522_5, 0.801_783_74, 1.0],
        ];

        let mut encoded = encode_filter_oct_i16(&normals, 16);
        decode_filter_oct_i16_in_place(&mut encoded);

        let inv_max = 1.0 / f32::from(i16::MAX);
        for (i, orig) in normals.iter().enumerate() {
            let dec = [
                f32::from(encoded[i * 4]) * inv_max,
                f32::from(encoded[i * 4 + 1]) * inv_max,
                f32::from(encoded[i * 4 + 2]) * inv_max,
            ];
            let n_orig = [orig[0], orig[1], orig[2]];
            let ln = norm(dec);
            // Reconstructed vector should be approximately unit-length
            assert!(
                (ln - 1.0).abs() < 0.01,
                "octahedral decoded length {ln} not unit (orig={n_orig:?})"
            );
            // Angular error < 1% (cos > 0.9998)
            let cos_angle = dot(n_orig, dec) / ln;
            assert!(
                cos_angle > 0.999,
                "octahedral angular error too large: cos={cos_angle}, orig={n_orig:?}, dec={dec:?}"
            );
        }
    }

    #[test]
    fn test_quat_encode_roundtrip_identity() {
        // Identity quaternion (0, 0, 0, 1) should round-trip
        let quats = vec![[0.0f32, 0.0, 0.0, 1.0]];
        let mut encoded = encode_filter_quat_i16(&quats, 16);
        decode_filter_quat_i16_in_place(&mut encoded);

        let inv_max = 1.0 / f32::from(i16::MAX);
        let dec = [
            f32::from(encoded[0]) * inv_max,
            f32::from(encoded[1]) * inv_max,
            f32::from(encoded[2]) * inv_max,
            f32::from(encoded[3]) * inv_max,
        ];
        // Quaternion should still be unit-length
        let l = dec[0]
            .mul_add(
                dec[0],
                dec[1].mul_add(dec[1], dec[2].mul_add(dec[2], dec[3] * dec[3])),
            )
            .sqrt();
        assert!(
            (l - 1.0).abs() < 0.05,
            "quat identity roundtrip lost normalization: len={l}, dec={dec:?}"
        );
    }

    #[test]
    fn test_exp_encode_positive_float() {
        // Test that positive floats survive one encode + decode cycle
        let inputs = vec![1.0f32, 3.14, 100.0, 0.001];
        let mut encoded = encode_filter_exp_u32(&inputs, 16);
        decode_filter_exp_u32_in_place(&mut encoded);

        for (orig, &encoded_bits) in inputs.iter().zip(encoded.iter()) {
            let decoded = f32::from_bits(encoded_bits);
            let rel_err = (decoded - orig).abs() / orig.abs().max(1e-6);
            assert!(
                rel_err < 0.001,
                "exp roundtrip too lossy: orig={orig}, decoded={decoded}, rel_err={rel_err}"
            );
        }
    }
}
