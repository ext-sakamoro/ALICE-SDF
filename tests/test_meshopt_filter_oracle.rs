//! meshopt vertex filters vs closed forms (`mesh::meshopt_filter`).
//!
//! - `quantize_snorm(v, b)`: round half away from zero of
//!   `clamp(v) · (2^(b-1) - 1)`; the error is at most half a step.
//! - Octahedral (Cigolle et al. 2014): the encoder stores the octahedral
//!   projection `(u, v)` of `n / |n|₁` (folded for `z < 0`) quantized to `b`
//!   bits. Checked in `f64`: the stored `(u, v)` is within half a step of the
//!   exact projection, and the decoder output is the normalised inverse
//!   projection of the stored code, scaled to `i16::MAX`, within one unit.
//!   The decoded vector is then within the angle bound that follows from the
//!   half-step grid error.
//! - Quaternion (largest-component, √2 scaling): after decode, each component
//!   of `q / |q|` (sign chosen so the largest component is positive) is
//!   recovered within the bound that follows from the `b`-bit step: the three
//!   stored components carry `0.5 / (√2 · (2^(b-1)-1))`, the reconstructed
//!   largest one at most three times that (it is ≥ 1/2), plus one i16 unit.
//! - Exponential: `decode(encode(v))` is `m · 2^e` with `|m| < 2^(b-1)`; the
//!   error is at most half of `2^e`, where `e = max(floor(log2 |v|) - b + 2,
//!   -126)`, or below one step when the rounding reaches `2^(b-1)` and is
//!   clamped.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
#![allow(clippy::float_cmp)]

use alice_sdf::mesh::meshopt_filter::{
    decode_filter_exp_u32_in_place, decode_filter_oct_i16_in_place,
    decode_filter_quat_i16_in_place, encode_filter_exp_one, encode_filter_exp_u32,
    encode_filter_oct_i16, encode_filter_oct_one, encode_filter_quat_i16, encode_filter_quat_one,
    quantize_snorm, try_decode_filter_oct_i16_in_place, try_encode_filter_oct_i16,
    try_encode_filter_quat_i16,
};

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

fn scale(bits: u32) -> f64 {
    f64::from((1i32 << (bits - 1)) - 1)
}

#[test]
fn quantize_snorm_is_within_half_a_step_and_rounds_halves_away_from_zero() {
    let mut n = 0;
    for bits in [2u32, 8, 12, 16, 24] {
        let s = scale(bits);
        for i in -2100..=2100 {
            let v = (f64::from(i) / 2000.0) as f32; // includes |v| > 1
            let q = quantize_snorm(v, bits);
            let want = f64::from(v).clamp(-1.0, 1.0) * s;
            assert!(
                (f64::from(q) - want).abs() <= 0.5 + 1e-6 * s,
                "{v} {bits}: {q}"
            );
            n += 1;
        }
        // exact halves: k + 0.5 steps
        for k in [0i32, 1, 2, 5] {
            if f64::from(k) + 0.5 > s {
                continue;
            }
            let v = ((f64::from(k) + 0.5) / s) as f32;
            if f64::from(v) * s == f64::from(k) + 0.5 {
                assert_eq!(quantize_snorm(v, bits), k + 1);
                assert_eq!(quantize_snorm(-v, bits), -(k + 1));
                n += 2;
            }
        }
    }
    assert!(n > 20_000);
}

fn oct_project(n: [f64; 3]) -> (f64, f64) {
    let l1 = n[0].abs() + n[1].abs() + n[2].abs();
    let (x, y) = (n[0] / l1, n[1] / l1);
    if n[2] >= 0.0 {
        (x, y)
    } else {
        (
            (1.0 - y.abs()) * x.signum_or_one(),
            (1.0 - x.abs()) * y.signum_or_one(),
        )
    }
}

trait SignumOrOne {
    fn signum_or_one(self) -> f64;
}
impl SignumOrOne for f64 {
    fn signum_or_one(self) -> f64 {
        if self >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}

fn oct_unproject(u: f64, v: f64) -> [f64; 3] {
    let z = 1.0 - u.abs() - v.abs();
    let t = z.min(0.0);
    let x = u + if u >= 0.0 { t } else { -t };
    let y = v + if v >= 0.0 { t } else { -t };
    let l = (x * x + y * y + z * z).sqrt();
    [x / l, y / l, z / l]
}

fn normals() -> Vec<[f32; 4]> {
    let mut rng = Rng(0x0123_4567_89AB_CDEF);
    let mut out = vec![
        [1.0, 0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0, -1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, -1.0, 1.0],
        [0.577, -0.577, -0.577, -1.0],
    ];
    while out.len() < 4000 {
        let p = [rng.unit(), rng.unit(), rng.unit()];
        let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if !(0.1..=1.0).contains(&l) {
            continue;
        }
        let w = if rng.unit() >= 0.0 { 1.0 } else { -1.0 };
        out.push([(p[0] / l) as f32, (p[1] / l) as f32, (p[2] / l) as f32, w]);
    }
    out
}

#[test]
fn octahedral_filter_matches_the_projection_and_its_inverse() {
    let mut n = 0;
    for bits in [8u32, 10, 12, 16] {
        let s = scale(bits);
        let data = normals();
        let enc = encode_filter_oct_i16(&data, bits);
        assert_eq!(enc, try_encode_filter_oct_i16(&data, bits).unwrap());
        let mut dec = enc.clone();
        decode_filter_oct_i16_in_place(&mut dec);
        // half-step grid error: |Δ(u, v)|₂ <= √2 · 0.5/s. The map (u, v) -> point
        // on the octahedron |p|₁ = 1 is piecewise linear with Jacobian rows
        // (±1, 0), (0, ±1), (±1, ±1) in both halves, largest singular value √3;
        // normalising a point with |p|₂ >= 1/√3 stretches by at most √3. So the
        // direction moves by at most 3 · √2 · 0.5/s (plus the i16 rounding).
        let max_angle = 3.0 * std::f64::consts::SQRT_2 * 0.5 / s + 4.0 / 32767.0;
        for (i, nrm) in data.iter().enumerate() {
            let c = &enc[4 * i..4 * i + 4];
            let mut one = [0i32; 4];
            encode_filter_oct_one(nrm[0], nrm[1], nrm[2], nrm[3], bits, &mut one);
            assert_eq!(one.map(|x| x as i16), [c[0], c[1], c[2], c[3]]);

            let exact = [f64::from(nrm[0]), f64::from(nrm[1]), f64::from(nrm[2])];
            let (u, v) = oct_project(exact);
            assert!(
                (f64::from(c[0]) / s - u).abs() <= 0.5 / s + 1e-6,
                "u {nrm:?}"
            );
            assert!(
                (f64::from(c[1]) / s - v).abs() <= 0.5 / s + 1e-6,
                "v {nrm:?}"
            );
            assert_eq!(f64::from(c[2]), s);
            assert_eq!(f64::from(c[3]), f64::from(nrm[3]).signum() * s);

            let want = oct_unproject(f64::from(c[0]) / s, f64::from(c[1]) / s);
            let d = &dec[4 * i..4 * i + 4];
            for k in 0..3 {
                assert!(
                    (f64::from(d[k]) - want[k] * 32767.0).abs() <= 1.0,
                    "decode {nrm:?} bits {bits}: {d:?} vs {want:?}"
                );
            }
            assert_eq!(d[3], c[3]);
            let got = [d[0], d[1], d[2]].map(|x| f64::from(x) / 32767.0);
            let gl = (got[0] * got[0] + got[1] * got[1] + got[2] * got[2]).sqrt();
            let el = (exact[0] * exact[0] + exact[1] * exact[1] + exact[2] * exact[2]).sqrt();
            let cos = (got[0] * exact[0] + got[1] * exact[1] + got[2] * exact[2]) / (gl * el);
            assert!(
                cos.min(1.0).acos() <= max_angle,
                "angle {nrm:?} bits {bits}"
            );
            n += 1;
        }
    }
    assert_eq!(n, 4 * 4000);
}

#[test]
fn octahedral_filter_rejects_bad_input() {
    assert!(try_encode_filter_oct_i16(&[[0.0, 0.0, 1.0, 1.0]], 1).is_err());
    assert!(try_encode_filter_oct_i16(&[[0.0, 0.0, 1.0, 1.0]], 17).is_err());
    let mut odd = [1i16, 2, 3];
    assert!(try_decode_filter_oct_i16_in_place(&mut odd).is_err());
    assert_eq!(odd, [1, 2, 3], "nothing is modified on error");
}

#[test]
fn quaternion_filter_recovers_the_normalised_rotation() {
    let mut rng = Rng(0xFACE_B00C_1234_0042);
    let mut quats: Vec<[f32; 4]> = vec![
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
        [-0.5, 0.5, -0.5, 0.5],
    ];
    while quats.len() < 4000 {
        let q = [rng.unit(), rng.unit(), rng.unit(), rng.unit()];
        let l = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        if l < 0.1 {
            continue;
        }
        let k = 0.5 + rng.unit().abs() * 2.0; // the encoder normalises any scale
        quats.push(q.map(|x| (x / l * k) as f32));
    }
    let mut n = 0;
    for bits in [8u32, 12, 16] {
        let s = scale(bits);
        let enc = encode_filter_quat_i16(&quats, bits);
        assert_eq!(enc, try_encode_filter_quat_i16(&quats, bits).unwrap());
        let mut dec = enc.clone();
        decode_filter_quat_i16_in_place(&mut dec);
        let dq = 0.5 / (s * std::f64::consts::SQRT_2);
        let bound = 4.0 * dq + 2.0 / 32767.0;
        for (i, q) in quats.iter().enumerate() {
            let mut one = [0i16; 4];
            encode_filter_quat_one(*q, bits, &mut one);
            assert_eq!(&one[..], &enc[4 * i..4 * i + 4]);

            let qd = q.map(f64::from);
            let l = qd.iter().map(|x| x * x).sum::<f64>().sqrt();
            let mut qc = 0;
            for k in 1..4 {
                if q[k].abs() > q[qc].abs() {
                    qc = k;
                }
            }
            let sign = if q[qc] < 0.0 { -1.0 } else { 1.0 };
            for k in 0..4 {
                let want = qd[k] / l * sign;
                let got = f64::from(dec[4 * i + k]) / 32767.0;
                assert!(
                    (got - want).abs() <= bound,
                    "q {q:?} bits {bits} comp {k}: {got} vs {want} (bound {bound})"
                );
            }
            n += 1;
        }
    }
    assert_eq!(n, 3 * 4000);
}

#[test]
fn exponential_filter_error_is_half_of_the_shared_exponent_step() {
    let mut rng = Rng(0x7777_0000_1111_2222);
    let mut values: Vec<f32> = vec![0.0, -0.0, 1.0, -1.0, 0.5, 3.0, 1e-30, -1e30, 2.0e-38];
    for _ in 0..5000 {
        let m = rng.unit();
        let e = (rng.unit() * 120.0) as i32;
        values.push((m * 2f64.powi(e)) as f32);
    }
    let mut n = 0;
    for bits in [8u32, 12, 16, 24] {
        let enc = encode_filter_exp_u32(&values, bits);
        let mut dec = enc.clone();
        decode_filter_exp_u32_in_place(&mut dec);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(enc[i], encode_filter_exp_one(v, bits));
            let got = f64::from(f32::from_bits(dec[i]));
            let a = f64::from(v).abs();
            if a == 0.0 {
                assert_eq!(got, 0.0);
                n += 1;
                continue;
            }
            // the format decodes `m · 2^e` for e in [-126, 127] (the exponent is
            // rebuilt as the bits of a normal f32), so the step cannot go below 2^-126
            let e = (a.log2().floor() as i32 - bits as i32 + 2).max(-126);
            // half a step, except where |v| rounds up to 2^(b-1) steps, which is
            // out of the signed b-bit range and is clamped to 2^(b-1) - 1: then
            // the error is below one step (meshopt's encoder does the same)
            let step = 2f64.powi(e);
            let bound = if a / step >= 2f64.powi(bits as i32 - 1) - 0.5 {
                step
            } else {
                0.5 * step
            };
            assert!(
                (got - f64::from(v)).abs() <= bound,
                "exp {v:e} bits {bits}: {got:e} (bound {bound:e})"
            );
            n += 1;
        }
    }
    assert_eq!(n, 4 * values.len());
}
