//! CPU side of the texture noise — the crate's one value-noise law
//! (`modifiers::hash_noise_3d`, PCG lattice hash) in scalar and 8-lane form.
//!
//! Until 1.13.0 this file carried a second law (`fract(sin(dot) · 43758.5)`)
//! that the emitted shaders duplicated under the same `hash_noise_3d` name;
//! the sin hash is not reproducible across GPUs (it amplifies 1 ulp of `sin`
//! into a different corner value) and it clashed with the transpilers'
//! helper of the same name. The PCG law is integer-only up to the final
//! `u32 → f32`, so CPU, SIMD and the WGSL / HLSL / GLSL helpers agree.
//!
//! ## Deep Fried
//! - `hash_noise_3d_simd`: 8-lane `wide::f32x8` evaluation (the corner hash
//!   is scalar per lane — 64 integer hashes per 8 pixels — the blend is SIMD)
//! - `eval_octave_simd`: 8-pixel simultaneous octave evaluation

use glam::Vec3;
use wide::f32x8;

// ──────────────────────────── Scalar ────────────────────────────

/// The crate's value noise (`modifiers::hash_noise_3d`) at `(px, py, pz)`,
/// in `[-1, 1]`. Kept as a free function over three floats for the fitter.
#[inline]
pub fn hash_noise_3d_cpu(px: f32, py: f32, pz: f32, seed: u32) -> f32 {
    crate::modifiers::hash_noise_3d(Vec3::new(px, py, pz), seed)
}

/// Evaluate noise for 2D UV with z=0 (texture fitting shortcut)
#[inline]
pub fn hash_noise_2d(u: f32, v: f32, seed: u32) -> f32 {
    hash_noise_3d_cpu(u, v, 0.0, seed)
}

/// Evaluate a single octave: amplitude * noise(uv * frequency + phase, seed)
///
/// Same operation order as `eval_octave_simd` (see [`hash_noise_3d_cpu`]).
#[allow(clippy::suboptimal_flops)]
#[inline]
pub fn eval_octave(
    u: f32,
    v: f32,
    amplitude: f32,
    frequency: f32,
    phase: [f32; 2],
    seed: u32,
    rotation: f32,
) -> f32 {
    let (sin_r, cos_r) = rotation.sin_cos();
    let ru = u * cos_r - v * sin_r;
    let rv = u * sin_r + v * cos_r;
    amplitude * hash_noise_2d(ru * frequency + phase[0], rv * frequency + phase[1], seed)
}

// ──────────────────────────── SIMD (Deep Fried) ────────────────────────────

/// [Deep Fried] SIMD lerp: a + (b - a) * t
#[inline(always)]
fn lerp_simd(a: f32x8, b: f32x8, t: f32x8) -> f32x8 {
    a + (b - a) * t
}

/// [Deep Fried] Lattice-corner hash for 8 lanes: the scalar PCG hash per
/// lane (integer ops, `wide` has no per-lane variable shift), same law as
/// `modifiers::hash_noise_3d`.
#[inline(always)]
fn hash_corner_simd(ix: f32x8, iy: f32x8, iz: f32x8, seed: u32) -> f32x8 {
    let x: [f32; 8] = ix.into();
    let y: [f32; 8] = iy.into();
    let z: [f32; 8] = iz.into();
    let mut out = [0.0f32; 8];
    for lane in 0..8 {
        out[lane] = crate::modifiers::surface_roughness_hash3(x[lane], y[lane], z[lane], seed);
    }
    f32x8::new(out)
}

/// [Deep Fried] 8-lane SIMD hash noise evaluation.
///
/// Evaluates `hash_noise_3d` at 8 different (px, py, pz) points
/// simultaneously. All 8 points share the same seed.
/// Output: 8 values in [-1, 1].
#[inline]
pub fn hash_noise_3d_simd(px: f32x8, py: f32x8, pz: f32x8, seed: u32) -> f32x8 {
    let one = f32x8::splat(1.0);
    let two = f32x8::splat(2.0);
    let three = f32x8::splat(3.0);

    let ix = px.floor();
    let iy = py.floor();
    let iz = pz.floor();
    let fx = px - ix;
    let fy = py - iy;
    let fz = pz - iz;

    // smoothstep: u = f * f * (3 - 2*f)
    let ux = fx * fx * (three - two * fx);
    let uy = fy * fy * (three - two * fy);
    let uz = fz * fz * (three - two * fz);

    // Hash 8 corners (each is 8-wide SIMD)
    let ix1 = ix + one;
    let iy1 = iy + one;
    let iz1 = iz + one;

    let n000 = hash_corner_simd(ix, iy, iz, seed);
    let n100 = hash_corner_simd(ix1, iy, iz, seed);
    let n010 = hash_corner_simd(ix, iy1, iz, seed);
    let n110 = hash_corner_simd(ix1, iy1, iz, seed);
    let n001 = hash_corner_simd(ix, iy, iz1, seed);
    let n101 = hash_corner_simd(ix1, iy, iz1, seed);
    let n011 = hash_corner_simd(ix, iy1, iz1, seed);
    let n111 = hash_corner_simd(ix1, iy1, iz1, seed);

    // Trilinear interpolation
    let c00 = lerp_simd(n000, n100, ux);
    let c10 = lerp_simd(n010, n110, ux);
    let c01 = lerp_simd(n001, n101, ux);
    let c11 = lerp_simd(n011, n111, ux);
    let c0 = lerp_simd(c00, c10, uy);
    let c1 = lerp_simd(c01, c11, uy);

    lerp_simd(c0, c1, uz) * two - one
}

/// [Deep Fried] 8-lane SIMD octave evaluation for 2D UV (z=0).
///
/// Evaluates 8 pixels at once with shared octave parameters.
#[inline]
pub fn eval_octave_simd(
    u: f32x8,
    v: f32x8,
    amplitude: f32,
    frequency: f32,
    phase: [f32; 2],
    seed: u32,
    rotation: f32,
) -> f32x8 {
    let (sin_r, cos_r) = rotation.sin_cos();
    let sin_v = f32x8::splat(sin_r);
    let cos_v = f32x8::splat(cos_r);
    let freq = f32x8::splat(frequency);
    let ph_u = f32x8::splat(phase[0]);
    let ph_v = f32x8::splat(phase[1]);
    let amp = f32x8::splat(amplitude);
    let zero = f32x8::splat(0.0);

    // Rotation
    let ru = u * cos_v - v * sin_v;
    let rv = u * sin_v + v * cos_v;

    // noise(ru * freq + phase, rv * freq + phase, 0, seed)
    let nx = ru * freq + ph_u;
    let ny = rv * freq + ph_v;

    amp * hash_noise_3d_simd(nx, ny, zero, seed)
}

/// [Deep Fried] Reduce f32x8 to scalar sum
#[inline(always)]
pub fn f32x8_sum(v: f32x8) -> f32 {
    let arr: [f32; 8] = v.into();
    arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_range() {
        for i in 0..1000 {
            let x = (i as f32) * 0.137;
            let y = (i as f32) * 0.291;
            let z = (i as f32) * 0.473;
            let v = hash_noise_3d_cpu(x, y, z, 42);
            assert!((-1.0..=1.0).contains(&v), "Out of range: {}", v);
        }
    }

    #[test]
    fn test_noise_deterministic() {
        let a = hash_noise_3d_cpu(1.5, 2.3, 0.7, 42);
        let b = hash_noise_3d_cpu(1.5, 2.3, 0.7, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_noise_different_seeds() {
        let a = hash_noise_3d_cpu(1.0, 2.0, 3.0, 0);
        let b = hash_noise_3d_cpu(1.0, 2.0, 3.0, 1);
        assert!(
            (a - b).abs() > 0.001,
            "Different seeds should give different values"
        );
    }

    #[test]
    fn test_noise_continuity() {
        let a = hash_noise_3d_cpu(1.0, 2.0, 3.0, 42);
        let b = hash_noise_3d_cpu(1.001, 2.0, 3.0, 42);
        assert!((a - b).abs() < 0.1, "Noise should be continuous");
    }

    #[test]
    fn test_simd_matches_scalar() {
        // SIMD must produce identical results to scalar
        let xs = [0.5, 1.2, 3.7, 0.1, 2.9, 4.4, 0.8, 6.1];
        let ys = [1.0, 0.3, 2.5, 4.0, 1.7, 0.9, 3.3, 2.2];
        let zs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let seed = 42u32;

        let simd_result = hash_noise_3d_simd(f32x8::new(xs), f32x8::new(ys), f32x8::new(zs), seed);
        let simd_arr: [f32; 8] = simd_result.into();

        for i in 0..8 {
            let scalar = hash_noise_3d_cpu(xs[i], ys[i], zs[i], seed);
            assert!(
                (simd_arr[i] - scalar).abs() < 1e-5,
                "Lane {} mismatch: simd={}, scalar={}",
                i,
                simd_arr[i],
                scalar
            );
        }
    }

    #[test]
    fn test_simd_octave_matches_scalar() {
        let us = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let vs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2];
        let amp = 0.5;
        let freq = 4.0;
        let phase = [0.3, 0.7];
        let seed = 7u32;
        let rot = 0.4;

        let simd_result =
            eval_octave_simd(f32x8::new(us), f32x8::new(vs), amp, freq, phase, seed, rot);
        let simd_arr: [f32; 8] = simd_result.into();

        for i in 0..8 {
            let scalar = eval_octave(us[i], vs[i], amp, freq, phase, seed, rot);
            assert!(
                (simd_arr[i] - scalar).abs() < 1e-5,
                "Lane {} mismatch: simd={}, scalar={}",
                i,
                simd_arr[i],
                scalar
            );
        }
    }

    #[test]
    fn test_simd_noise_range() {
        for batch in 0..125 {
            let base = batch as f32 * 0.8;
            let px = f32x8::new([
                base,
                base + 0.1,
                base + 0.2,
                base + 0.3,
                base + 0.4,
                base + 0.5,
                base + 0.6,
                base + 0.7,
            ]);
            let py = px * f32x8::splat(1.37);
            let pz = f32x8::splat(0.0);

            let result = hash_noise_3d_simd(px, py, pz, 99);
            let arr: [f32; 8] = result.into();
            for &v in &arr {
                assert!((-1.0..=1.0).contains(&v), "SIMD out of range: {}", v);
            }
        }
    }
}
