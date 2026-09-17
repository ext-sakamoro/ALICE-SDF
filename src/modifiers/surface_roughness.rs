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

/// The lattice-corner hash as three floats (for the texture module's SIMD
/// lanes); same law as the private `hash3`.
#[cfg(feature = "texture-fit")]
#[inline(always)]
pub fn hash3_xyz(x: f32, y: f32, z: f32, seed: u32) -> f32 {
    hash3(Vec3::new(x, y, z), seed)
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

    let c00 = (n100 - n000) * u.x + n000;
    let c10 = (n110 - n010) * u.x + n010;
    let c01 = (n101 - n001) * u.x + n001;
    let c11 = (n111 - n011) * u.x + n011;
    let c0 = (c10 - c00) * u.y + c00;
    let c1 = (c11 - c01) * u.y + c01;
    ((c1 - c0) * u.z + c0) * 2.0 + -1.0
}

/// WGSL source of `hash_noise_3d` — the same law as [`hash_noise_3d`],
/// emitted by the WGSL transpiler and by `texture::generate_shader`.
pub const HASH_NOISE_WGSL: &str = r"fn alice_pcg(v0: u32) -> u32 {
    let v = v0 * 747796405u + 2891336453u;
    let w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}
fn alice_hash3(i: vec3<f32>, seed: u32) -> f32 {
    let q = bitcast<vec3<u32>>(i);
    return f32(alice_pcg(q.x ^ alice_pcg(q.y ^ alice_pcg(q.z ^ seed)))) * (1.0 / 4294967295.0);
}
// Same law as `alice_sdf::modifiers::surface_roughness::hash_noise_3d` (PCG lattice hash, bit-exact with the CPU).
fn hash_noise_3d(p: vec3<f32>, seed: u32) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = alice_hash3(i, seed);
    let n100 = alice_hash3(i + vec3<f32>(1.0, 0.0, 0.0), seed);
    let n010 = alice_hash3(i + vec3<f32>(0.0, 1.0, 0.0), seed);
    let n110 = alice_hash3(i + vec3<f32>(1.0, 1.0, 0.0), seed);
    let n001 = alice_hash3(i + vec3<f32>(0.0, 0.0, 1.0), seed);
    let n101 = alice_hash3(i + vec3<f32>(1.0, 0.0, 1.0), seed);
    let n011 = alice_hash3(i + vec3<f32>(0.0, 1.0, 1.0), seed);
    let n111 = alice_hash3(i + vec3<f32>(1.0, 1.0, 1.0), seed);
    let c00 = n000 + (n100 - n000) * u.x;
    let c10 = n010 + (n110 - n010) * u.x;
    let c01 = n001 + (n101 - n001) * u.x;
    let c11 = n011 + (n111 - n011) * u.x;
    let c0 = c00 + (c10 - c00) * u.y;
    let c1 = c01 + (c11 - c01) * u.y;
    return (c0 + (c1 - c0) * u.z) * 2.0 - 1.0;
}";

/// GLSL source of `hash_noise_3d` (see [`HASH_NOISE_WGSL`]).
pub const HASH_NOISE_GLSL: &str = r"uint alice_pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}
float alice_hash3(vec3 i, uint seed) {
    uvec3 q = floatBitsToUint(i);
    return float(alice_pcg(q.x ^ alice_pcg(q.y ^ alice_pcg(q.z ^ seed)))) * (1.0 / 4294967295.0);
}
// Same law as `alice_sdf::modifiers::surface_roughness::hash_noise_3d` (PCG lattice hash, bit-exact with the CPU).
float hash_noise_3d(vec3 p, uint seed) {
    vec3 i = floor(p);
    vec3 f = p - i;
    vec3 u = f * f * (3.0 - 2.0 * f);
    float n000 = alice_hash3(i, seed);
    float n100 = alice_hash3(i + vec3(1.0, 0.0, 0.0), seed);
    float n010 = alice_hash3(i + vec3(0.0, 1.0, 0.0), seed);
    float n110 = alice_hash3(i + vec3(1.0, 1.0, 0.0), seed);
    float n001 = alice_hash3(i + vec3(0.0, 0.0, 1.0), seed);
    float n101 = alice_hash3(i + vec3(1.0, 0.0, 1.0), seed);
    float n011 = alice_hash3(i + vec3(0.0, 1.0, 1.0), seed);
    float n111 = alice_hash3(i + vec3(1.0, 1.0, 1.0), seed);
    float c00 = n000 + (n100 - n000) * u.x;
    float c10 = n010 + (n110 - n010) * u.x;
    float c01 = n001 + (n101 - n001) * u.x;
    float c11 = n011 + (n111 - n011) * u.x;
    float c0 = c00 + (c10 - c00) * u.y;
    float c1 = c01 + (c11 - c01) * u.y;
    return (c0 + (c1 - c0) * u.z) * 2.0 - 1.0;
}";

/// HLSL source of `hash_noise_3d` (see [`HASH_NOISE_WGSL`]).
pub const HASH_NOISE_HLSL: &str = r"uint alice_pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}
float alice_hash3(float3 i, uint seed) {
    uint3 q = asuint(i);
    return (float)alice_pcg(q.x ^ alice_pcg(q.y ^ alice_pcg(q.z ^ seed))) * (1.0 / 4294967295.0);
}
// Same law as `alice_sdf::modifiers::surface_roughness::hash_noise_3d` (PCG lattice hash, bit-exact with the CPU).
float hash_noise_3d(float3 p, uint seed) {
    float3 i = floor(p);
    float3 f = p - i;
    float3 u = f * f * (3.0 - 2.0 * f);
    float n000 = alice_hash3(i, seed);
    float n100 = alice_hash3(i + float3(1.0, 0.0, 0.0), seed);
    float n010 = alice_hash3(i + float3(0.0, 1.0, 0.0), seed);
    float n110 = alice_hash3(i + float3(1.0, 1.0, 0.0), seed);
    float n001 = alice_hash3(i + float3(0.0, 0.0, 1.0), seed);
    float n101 = alice_hash3(i + float3(1.0, 0.0, 1.0), seed);
    float n011 = alice_hash3(i + float3(0.0, 1.0, 1.0), seed);
    float n111 = alice_hash3(i + float3(1.0, 1.0, 1.0), seed);
    float c00 = n000 + (n100 - n000) * u.x;
    float c10 = n010 + (n110 - n010) * u.x;
    float c01 = n001 + (n101 - n001) * u.x;
    float c11 = n011 + (n111 - n011) * u.x;
    float c0 = c00 + (c10 - c00) * u.y;
    float c1 = c01 + (c11 - c01) * u.y;
    return (c0 + (c1 - c0) * u.z) * 2.0 - 1.0;
}";

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
        value += amplitude * hash_noise_3d(p * scale, SURFACE_ROUGHNESS_SEED);
        amplitude *= 0.5;
    }
    value
}

/// Upper bound of `|fbm(_, octaves)|`.
#[inline(always)]
pub fn fbm_bound(octaves: u32) -> f32 {
    2.0 - alice_det_math::powi(0.5_f32, octaves as i32 - 1)
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
    noise * amplitude + distance
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
