//! Noise field abstraction for NPR primitives that need a random sample
//!
//! Provides a [`NoiseField`] trait so callers can plug in any noise
//! generator (Perlin, Simplex, worley, blue-noise, hash-based, etc.).
//! A default deterministic hash-based value noise is provided as
//! [`HashNoise`] for callers who do not need a specific distribution.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// A source of deterministic scalar / 3D noise samples
///
/// Implementations must be pure (same input -> same output) so results
/// are reproducible across runs and threads.
pub trait NoiseField {
    /// Scalar sample in `[0, 1]` at the given 3D point
    fn sample_scalar(&self, point: Vec3) -> f32;

    /// 3D vector sample with components in `[-1, 1]`
    ///
    /// Default implementation stitches three orthogonally offset scalar
    /// samples; override for a more efficient direct implementation.
    fn sample_vec3(&self, point: Vec3) -> Vec3 {
        let sx = self.sample_scalar(point + Vec3::new(0.0, 0.0, 0.0));
        let sy = self.sample_scalar(point + Vec3::new(37.0, 91.0, 13.0));
        let sz = self.sample_scalar(point + Vec3::new(71.0, 29.0, 53.0));
        Vec3::new(sx * 2.0 - 1.0, sy * 2.0 - 1.0, sz * 2.0 - 1.0)
    }
}

/// Deterministic hash-based value noise
///
/// Uses a small PCG-like integer hash on the quantized point, giving
/// blocky value noise suitable for `paper_grain`, `hand_drawn_jitter`
/// and other coarse-detail primitives. Not intended as a high-quality
/// gradient noise; callers wanting smooth noise should implement
/// [`NoiseField`] with their own generator.
#[derive(Debug, Clone, Copy)]
pub struct HashNoise {
    /// Seed offset applied before hashing
    pub seed: u32,
    /// Spatial frequency: points are multiplied by this before quantization
    pub frequency: f32,
}

impl HashNoise {
    /// Construct with the given seed and unit frequency
    #[inline]
    #[must_use]
    pub const fn new(seed: u32) -> Self {
        Self {
            seed,
            frequency: 1.0,
        }
    }

    /// Set spatial frequency (higher = finer detail)
    #[inline]
    #[must_use]
    pub const fn with_frequency(mut self, frequency: f32) -> Self {
        self.frequency = frequency;
        self
    }
}

impl Default for HashNoise {
    fn default() -> Self {
        Self {
            seed: 0,
            frequency: 1.0,
        }
    }
}

/// PCG-like 32-bit hash for three integer coordinates
#[inline]
fn pcg_hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut state = seed
        .wrapping_add((x as u32).wrapping_mul(0x9E37_79B1))
        .wrapping_add((y as u32).wrapping_mul(0x85EB_CA6B))
        .wrapping_add((z as u32).wrapping_mul(0xC2B2_AE35));
    state = state.wrapping_mul(0x7FEB_352D);
    state ^= state >> 15;
    state = state.wrapping_mul(0x846C_A68B);
    state ^= state >> 16;
    state
}

impl NoiseField for HashNoise {
    fn sample_scalar(&self, point: Vec3) -> f32 {
        let scaled = point * self.frequency;
        let ix = scaled.x.floor() as i32;
        let iy = scaled.y.floor() as i32;
        let iz = scaled.z.floor() as i32;
        let h = pcg_hash3(ix, iy, iz, self.seed);
        (h as f32) / (u32::MAX as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_noise_deterministic() {
        let noise = HashNoise::new(42);
        let a = noise.sample_scalar(Vec3::new(1.5, 2.5, 3.5));
        let b = noise.sample_scalar(Vec3::new(1.5, 2.5, 3.5));
        assert!((a - b).abs() < 1e-6);
    }

    #[test]
    fn hash_noise_output_bounded() {
        let noise = HashNoise::new(7);
        for i in 0..100 {
            let p = Vec3::new(i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3);
            let s = noise.sample_scalar(p);
            assert!((0.0..=1.0).contains(&s), "out of range at i={i}: {s}");
        }
    }

    #[test]
    fn hash_noise_different_seeds_differ() {
        let a = HashNoise::new(1).sample_scalar(Vec3::new(1.0, 2.0, 3.0));
        let b = HashNoise::new(2).sample_scalar(Vec3::new(1.0, 2.0, 3.0));
        assert!(
            (a - b).abs() > 1e-4,
            "different seeds should give different noise"
        );
    }

    #[test]
    fn hash_noise_different_cells_differ() {
        let noise = HashNoise::new(0);
        let a = noise.sample_scalar(Vec3::new(0.5, 0.5, 0.5));
        let b = noise.sample_scalar(Vec3::new(1.5, 0.5, 0.5));
        assert!((a - b).abs() > 1e-4, "different cells should differ");
    }

    #[test]
    fn hash_noise_frequency_changes_output() {
        let a = HashNoise::new(0)
            .with_frequency(1.0)
            .sample_scalar(Vec3::new(0.3, 0.3, 0.3));
        let b = HashNoise::new(0)
            .with_frequency(10.0)
            .sample_scalar(Vec3::new(0.3, 0.3, 0.3));
        assert!((a - b).abs() > 1e-4, "different frequencies should differ");
    }

    #[test]
    fn sample_vec3_default_bounded() {
        let noise = HashNoise::new(0);
        for i in 0..50 {
            let p = Vec3::splat(i as f32 * 0.1);
            let v = noise.sample_vec3(p);
            assert!((-1.0..=1.0).contains(&v.x));
            assert!((-1.0..=1.0).contains(&v.y));
            assert!((-1.0..=1.0).contains(&v.z));
        }
    }
}
