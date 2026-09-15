//! Surface roughness modifier using Fractal Brownian Motion (FBM)
//!
//! Adds micro-detail noise to SDF surfaces without mesh generation.
//! Uses multi-octave value noise for natural-looking roughness.
//!
//! The noise law lives here once ([`hash_noise_3d`] / [`fbm`]) and the GLSL /
//! WGSL / HLSL transpilers emit the same function, so CPU, SIMD, JIT and
//! shader evaluations of `SurfaceRoughness` agree (portable PCG hash; the
//! previous `fract(sin(·) · 43758)` hash differed between GPU and CPU `sin`).
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// PCG integer hash (Jarzynski & Olano), bit-exact on CPU and GPU.
#[inline(always)]
pub const fn pcg(v: u32) -> u32 {
    let v = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let w = ((v >> ((v >> 28) + 4)) ^ v).wrapping_mul(277_803_737);
    (w >> 22) ^ w
}

/// Lattice-corner hash in `[0, 1)`: PCG over the raw bits of the (integer
/// valued) corner coordinates and `seed`. Only integer ops and a `u32 → f32`
/// conversion, so every evaluation path — CPU, SIMD, JIT and the GLSL / WGSL /
/// HLSL `hash_noise_3d` helpers — produces the same value.
#[inline(always)]
fn hash3(i: Vec3, seed: u32) -> f32 {
    let h = pcg(i.x.to_bits() ^ pcg(i.y.to_bits() ^ pcg(i.z.to_bits() ^ seed)));
    (h as f32) * (1.0 / 4_294_967_295.0)
}

/// Value noise in `[-1, 1]`: trilinear (smoothstep-weighted) blend of the
/// eight lattice-corner hashes, mapped with `2v - 1`.
///
/// This is the single definition of the noise law; the shader transpilers
/// emit exactly this function as `hash_noise_3d(p, seed)`.
#[inline(always)]
pub fn hash_noise_3d(p: Vec3, seed: u32) -> f32 {
    let i = p.floor();
    let f = p - i;
    let u = f * f * (Vec3::splat(3.0) - f * 2.0);

    let n000 = hash3(i, seed);
    let n100 = hash3(i + Vec3::X, seed);
    let n010 = hash3(i + Vec3::Y, seed);
    let n110 = hash3(i + Vec3::new(1.0, 1.0, 0.0), seed);
    let n001 = hash3(i + Vec3::Z, seed);
    let n101 = hash3(i + Vec3::new(1.0, 0.0, 1.0), seed);
    let n011 = hash3(i + Vec3::new(0.0, 1.0, 1.0), seed);
    let n111 = hash3(i + Vec3::ONE, seed);

    let c00 = (n100 - n000).mul_add(u.x, n000);
    let c10 = (n110 - n010).mul_add(u.x, n010);
    let c01 = (n101 - n001).mul_add(u.x, n001);
    let c11 = (n111 - n011).mul_add(u.x, n011);
    let c0 = (c10 - c00).mul_add(u.y, c00);
    let c1 = (c11 - c01).mul_add(u.y, c01);
    (c1 - c0).mul_add(u.z, c0).mul_add(2.0, -1.0)
}

/// Seed the transpilers hard-code for `SurfaceRoughness`.
pub const SURFACE_ROUGHNESS_SEED: u32 = 42;

/// Fractal Brownian Motion: `Σ_{i<octaves} 0.5^i · hash_noise_3d(p · 2^i, 42)`.
///
/// Same structure as the transpiled shader expansion (amplitude starts at 1
/// and halves, frequency doubles, no per-octave rotation), so
/// `|fbm| ≤ 2 - 2^(1 - octaves)`.
#[inline(always)]
pub fn fbm(p: Vec3, octaves: u32) -> f32 {
    let mut value = 0.0_f32;
    let mut amplitude = 1.0_f32;
    for i in 0..octaves {
        let scale = (1u32 << i) as f32;
        value = amplitude.mul_add(hash_noise_3d(p * scale, SURFACE_ROUGHNESS_SEED), value);
        amplitude *= 0.5;
    }
    value
}

/// Upper bound of `|fbm(_, octaves)|`.
#[inline(always)]
pub fn fbm_bound(octaves: u32) -> f32 {
    2.0 - 0.5_f32.powi(octaves as i32 - 1)
}

/// Apply surface roughness to an SDF distance value
#[inline(always)]
pub fn surface_roughness(
    p: Vec3,
    distance: f32,
    frequency: f32,
    amplitude: f32,
    octaves: u32,
) -> f32 {
    let noise = fbm(p * frequency, octaves);
    noise.mul_add(amplitude, distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fbm_bounded() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let n = fbm(p, 4);
        assert!(n.abs() < 2.0, "FBM should be bounded: got {}", n);
    }

    #[test]
    fn test_roughness_preserves_sign() {
        let p = Vec3::new(0.0, 0.0, 0.0);
        // Large positive distance should remain positive with small roughness
        let d = surface_roughness(p, 10.0, 1.0, 0.01, 3);
        assert!(
            d > 0.0,
            "Large positive distance should stay positive with small roughness"
        );
    }
}
