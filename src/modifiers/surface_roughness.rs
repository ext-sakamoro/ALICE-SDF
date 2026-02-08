//! Surface roughness modifier using Fractal Brownian Motion (FBM)
//!
//! Adds micro-detail noise to SDF surfaces without mesh generation.
//! Uses multi-octave value noise for natural-looking roughness.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Hash function for 3D noise
#[inline(always)]
fn hash(p: Vec3) -> f32 {
    let h = p.dot(Vec3::new(127.1, 311.7, 74.7));
    h.sin().fract() * 43758.5453
}

/// Smooth value noise
#[inline(always)]
fn value_noise(p: Vec3) -> f32 {
    let i = Vec3::new(p.x.floor(), p.y.floor(), p.z.floor());
    let f = p - i;

    // Smoothstep
    let u = f * f * (Vec3::splat(3.0) - f * 2.0);

    let n000 = hash(i);
    let n100 = hash(i + Vec3::X);
    let n010 = hash(i + Vec3::Y);
    let n110 = hash(i + Vec3::new(1.0, 1.0, 0.0));
    let n001 = hash(i + Vec3::Z);
    let n101 = hash(i + Vec3::new(1.0, 0.0, 1.0));
    let n011 = hash(i + Vec3::new(0.0, 1.0, 1.0));
    let n111 = hash(i + Vec3::ONE);

    // Trilinear interpolation
    let a = n000 * (1.0 - u.x) + n100 * u.x;
    let b = n010 * (1.0 - u.x) + n110 * u.x;
    let c = n001 * (1.0 - u.x) + n101 * u.x;
    let d = n011 * (1.0 - u.x) + n111 * u.x;

    let e = a * (1.0 - u.y) + b * u.y;
    let f_val = c * (1.0 - u.y) + d * u.y;

    (e * (1.0 - u.z) + f_val * u.z).fract()
}

/// Fractal Brownian Motion
#[inline(always)]
pub fn fbm(p: Vec3, octaves: u32) -> f32 {
    let mut value = 0.0_f32;
    let mut amplitude = 0.5_f32;
    let mut frequency = 1.0_f32;
    let mut p = p;

    for _ in 0..octaves {
        value += amplitude * (value_noise(p * frequency) * 2.0 - 1.0);
        amplitude *= 0.5;
        frequency *= 2.0;
        // Rotate to break axis alignment
        p = Vec3::new(p.x * 0.8 - p.z * 0.6, p.y, p.x * 0.6 + p.z * 0.8);
    }

    value
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
    distance + noise * amplitude
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
