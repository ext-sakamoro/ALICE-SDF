//! Attribute quantization vs closed forms (`mesh::quantization`).
//!
//! - snorm / unorm decode: `q / (2^(n-1) - 1)` (signed) or `q / (2^n - 1)`
//!   (unsigned), clamped to the range. The quotient of two small integers is
//!   computed in `f64` and rounded once to `f32` (double rounding through
//!   `f64` is exact for a single division), so the comparison is bit for bit,
//!   over every code of every width.
//! - encode → decode: the error is at most half a step (`0.5 / scale`) plus
//!   one `f32` rounding.
//! - binary16: the reference value of every one of the 65 536 codes is built
//!   from the IEEE 754 definition in `f64` (`(-1)^s · 2^(e-15) · (1 + m/1024)`,
//!   subnormals `2^-24 · m`). `half_decode` must give that value exactly, and
//!   `half_encode` must give the nearest code with ties to the even code
//!   (IEEE 754 roundTiesToEven), overflow to infinity at `65520`, for a sweep
//!   of `f32` values that includes every code value and every midpoint
//!   between neighbouring codes.
//!
//! Author: Moroya Sakamoto
#![allow(clippy::float_cmp, clippy::manual_midpoint)]

use alice_sdf::mesh::quantization::{
    half_decode, half_encode, snorm_i16_decode, snorm_i16_encode, snorm_i8_decode, snorm_i8_encode,
    unorm_u16_decode, unorm_u16_encode, unorm_u8_decode, unorm_u8_encode,
};

fn reference_decode(q: i64, scale: f64, lo: f64) -> f32 {
    ((q as f64 / scale).clamp(lo, 1.0)) as f32
}

#[test]
fn snorm_and_unorm_decode_are_the_closed_form_for_every_code() {
    let mut n = 0;
    for q in i8::MIN..=i8::MAX {
        assert_eq!(
            snorm_i8_decode(q).to_bits(),
            reference_decode(q.into(), 127.0, -1.0).to_bits(),
            "snorm_i8_decode({q})"
        );
        n += 1;
    }
    for q in i16::MIN..=i16::MAX {
        assert_eq!(
            snorm_i16_decode(q).to_bits(),
            reference_decode(q.into(), 32767.0, -1.0).to_bits(),
            "snorm_i16_decode({q})"
        );
        n += 1;
    }
    for q in u8::MIN..=u8::MAX {
        assert_eq!(
            unorm_u8_decode(q).to_bits(),
            reference_decode(q.into(), 255.0, 0.0).to_bits(),
            "unorm_u8_decode({q})"
        );
        n += 1;
    }
    for q in u16::MIN..=u16::MAX {
        assert_eq!(
            unorm_u16_decode(q).to_bits(),
            reference_decode(q.into(), 65535.0, 0.0).to_bits(),
            "unorm_u16_decode({q})"
        );
        n += 1;
    }
    assert_eq!(n, 256 + 65536 + 256 + 65536);
}

#[test]
fn encode_then_decode_is_within_half_a_step() {
    let mut n = 0;
    for i in 0..=20_000 {
        let s = -1.25 + 2.5 * f64::from(i) / 20_000.0; // also outside [-1, 1]
        let v = s as f32;
        let checks: [(&str, f32, f64, f64); 4] = [
            ("snorm_i8", snorm_i8_decode(snorm_i8_encode(v)), 127.0, -1.0),
            (
                "snorm_i16",
                snorm_i16_decode(snorm_i16_encode(v)),
                32767.0,
                -1.0,
            ),
            ("unorm_u8", unorm_u8_decode(unorm_u8_encode(v)), 255.0, 0.0),
            (
                "unorm_u16",
                unorm_u16_decode(unorm_u16_encode(v)),
                65535.0,
                0.0,
            ),
        ];
        for (name, got, scale, lo) in checks {
            let want = f64::from(v).clamp(lo, 1.0);
            let bound = 0.5 / scale + f64::from(f32::EPSILON);
            assert!(
                (f64::from(got) - want).abs() <= bound,
                "{name}: {v} -> {got} (want {want} ± {bound})"
            );
            n += 1;
        }
    }
    assert_eq!(n, 4 * 20_001);
}

/// The exact value of a binary16 code (`None` for NaN).
fn half_value(h: u16) -> Option<f64> {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1F);
    let m = f64::from(h & 0x3FF);
    match e {
        0 => Some(sign * m * 2f64.powi(-24)),
        31 if m == 0.0 => Some(sign * f64::INFINITY),
        31 => None,
        _ => Some(sign * 2f64.powi(e - 15) * (1.0 + m / 1024.0)),
    }
}

/// IEEE 754 roundTiesToEven of `v` to binary16, by search over the ordered
/// table of finite non-negative codes `0x0000..=0x7BFF`.
fn half_round_reference(v: f32) -> u16 {
    let sign: u16 = if v.is_sign_negative() { 0x8000 } else { 0 };
    let a = f64::from(v).abs();
    if v.is_nan() {
        unreachable!()
    }
    // 65504 + half an ulp (16) ties to the even neighbour 65536, which is out of range.
    if a >= 65520.0 {
        return sign | 0x7C00;
    }
    // binary search for the largest code with value <= a
    let (mut lo, mut hi) = (0u16, 0x7BFFu16);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if half_value(mid).unwrap() <= a {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let below = lo;
    let vb = half_value(below).unwrap();
    if vb == a {
        return sign | below;
    }
    let above = below + 1;
    let va = if above == 0x7C00 {
        65536.0
    } else {
        half_value(above).unwrap()
    };
    let pick = if a - vb < va - a {
        below
    } else if a - vb > va - a {
        above
    } else if below & 1 == 0 {
        below
    } else {
        above
    };
    sign | pick
}

#[test]
fn half_decode_is_the_ieee_value_of_every_code() {
    let mut n = 0;
    for h in 0..=u16::MAX {
        let got = half_decode(h);
        match half_value(h) {
            None => assert!(got.is_nan(), "half_decode({h:#06x}) = {got}"),
            Some(want) => {
                assert_eq!(
                    got.to_bits(),
                    (want as f32).to_bits(),
                    "half_decode({h:#06x})"
                );
            }
        }
        n += 1;
    }
    assert_eq!(n, 65536);
}

#[test]
fn half_encode_rounds_to_nearest_even_including_subnormals() {
    let mut values: Vec<f32> = Vec::new();
    // every finite code value and every midpoint to the next code (exact in f32)
    for h in 0..0x7C00u16 {
        let v = half_value(h).unwrap();
        let next = if h == 0x7BFF {
            65536.0
        } else {
            half_value(h + 1).unwrap()
        };
        for x in [
            v,
            (v + next) / 2.0,
            v + (next - v) * 0.25,
            v + (next - v) * 0.75,
        ] {
            values.push(x as f32);
            values.push(-(x as f32));
        }
    }
    // f32 values below the smallest half subnormal and around it
    for k in 0..64 {
        values.push(2f32.powi(-26) * (1.0 + k as f32 / 16.0));
    }
    values.extend([
        65519.0,
        65520.0,
        65521.0,
        1e6,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ]);

    let mut n = 0;
    for &v in &values {
        let got = half_encode(v);
        let want = half_round_reference(v);
        assert_eq!(
            got,
            want,
            "half_encode({v:e}) = {got:#06x}, want {want:#06x} ({:?})",
            half_value(want)
        );
        n += 1;
    }
    assert!(n > 200_000, "compared {n}");
    assert!(half_decode(half_encode(f32::NAN)).is_nan());
}

#[test]
fn half_encode_inverts_half_decode_on_every_non_nan_code() {
    let mut n = 0;
    for h in 0..=u16::MAX {
        if half_value(h).is_none() {
            continue;
        }
        assert_eq!(half_encode(half_decode(h)), h, "code {h:#06x}");
        n += 1;
    }
    assert_eq!(n, 65536 - 2 * 1023);
}
