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

// ============================================================================
// Perlin gradient noise
// ============================================================================

/// Classic 3D gradient noise on the integer lattice
///
/// Smooth, spatially coherent noise with continuous derivatives.
/// Suitable for `puffy_cloud_layer` (soft cloud edges), `paper_grain`
/// (fine paper texture), and general procedural detail.
#[derive(Debug, Clone, Copy)]
pub struct PerlinNoise {
    /// Seed offset applied before hashing lattice corners
    pub seed: u32,
    /// Spatial frequency (higher = finer detail)
    pub frequency: f32,
}

impl PerlinNoise {
    /// Construct with unit frequency
    #[inline]
    #[must_use]
    pub const fn new(seed: u32) -> Self {
        Self {
            seed,
            frequency: 1.0,
        }
    }

    /// Set spatial frequency
    #[inline]
    #[must_use]
    pub const fn with_frequency(mut self, frequency: f32) -> Self {
        self.frequency = frequency;
        self
    }
}

impl Default for PerlinNoise {
    fn default() -> Self {
        Self {
            seed: 0,
            frequency: 1.0,
        }
    }
}

/// Fifth-order smoothstep (`6t^5 - 15t^4 + 10t^3`), C² continuous
#[inline]
fn perlin_fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Twelve edge-midpoint gradients selected by a hash's low bits
#[inline]
fn perlin_grad(hash: u32, x: f32, y: f32, z: f32) -> f32 {
    match hash & 15 {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        3 => -x - y,
        4 => x + z,
        5 => -x + z,
        6 => x - z,
        7 => -x - z,
        8 => y + z,
        9 => -y + z,
        10 => y - z,
        11 => -y - z,
        12 => x + y,
        13 => -y + z,
        14 => -x + y,
        _ => -y - z,
    }
}

impl NoiseField for PerlinNoise {
    fn sample_scalar(&self, point: Vec3) -> f32 {
        let p = point * self.frequency;
        let xi = p.x.floor() as i32;
        let yi = p.y.floor() as i32;
        let zi = p.z.floor() as i32;
        let xf = p.x - xi as f32;
        let yf = p.y - yi as f32;
        let zf = p.z - zi as f32;

        let u = perlin_fade(xf);
        let v = perlin_fade(yf);
        let w = perlin_fade(zf);

        let n000 = perlin_grad(pcg_hash3(xi, yi, zi, self.seed), xf, yf, zf);
        let n100 = perlin_grad(pcg_hash3(xi + 1, yi, zi, self.seed), xf - 1.0, yf, zf);
        let n010 = perlin_grad(pcg_hash3(xi, yi + 1, zi, self.seed), xf, yf - 1.0, zf);
        let n110 = perlin_grad(
            pcg_hash3(xi + 1, yi + 1, zi, self.seed),
            xf - 1.0,
            yf - 1.0,
            zf,
        );
        let n001 = perlin_grad(pcg_hash3(xi, yi, zi + 1, self.seed), xf, yf, zf - 1.0);
        let n101 = perlin_grad(
            pcg_hash3(xi + 1, yi, zi + 1, self.seed),
            xf - 1.0,
            yf,
            zf - 1.0,
        );
        let n011 = perlin_grad(
            pcg_hash3(xi, yi + 1, zi + 1, self.seed),
            xf,
            yf - 1.0,
            zf - 1.0,
        );
        let n111 = perlin_grad(
            pcg_hash3(xi + 1, yi + 1, zi + 1, self.seed),
            xf - 1.0,
            yf - 1.0,
            zf - 1.0,
        );

        let nx00 = lerp_f32(n000, n100, u);
        let nx10 = lerp_f32(n010, n110, u);
        let nx01 = lerp_f32(n001, n101, u);
        let nx11 = lerp_f32(n011, n111, u);
        let nxy0 = lerp_f32(nx00, nx10, v);
        let nxy1 = lerp_f32(nx01, nx11, v);
        let raw = lerp_f32(nxy0, nxy1, w);

        // grad values sit in [-sqrt(2)/2 * 2, sqrt(2)/2 * 2] roughly [-1.4, 1.4]
        // clamp then remap [-1, 1] -> [0, 1]
        (raw.clamp(-1.0, 1.0) * 0.5 + 0.5).clamp(0.0, 1.0)
    }
}

// ============================================================================
// Worley (cellular) noise
// ============================================================================

/// Cellular noise returning the distance to the nearest feature point
///
/// Samples a random feature point per unit cell and returns the shortest
/// Euclidean distance from `point` to any nearby feature point. Suitable
/// for `puffy_cloud_layer` (cell-like clouds), stippling / mosaic
/// patterns, and Voronoi-based textures. Output is normalized to `[0, 1]`
/// with a soft clamp at expected maximum distance.
#[derive(Debug, Clone, Copy)]
pub struct WorleyNoise {
    /// Seed offset applied to the cell hash
    pub seed: u32,
    /// Spatial frequency (higher = smaller cells)
    pub frequency: f32,
}

impl WorleyNoise {
    /// Construct with unit frequency
    #[inline]
    #[must_use]
    pub const fn new(seed: u32) -> Self {
        Self {
            seed,
            frequency: 1.0,
        }
    }

    /// Set spatial frequency
    #[inline]
    #[must_use]
    pub const fn with_frequency(mut self, frequency: f32) -> Self {
        self.frequency = frequency;
        self
    }
}

impl Default for WorleyNoise {
    fn default() -> Self {
        Self {
            seed: 0,
            frequency: 1.0,
        }
    }
}

impl NoiseField for WorleyNoise {
    fn sample_scalar(&self, point: Vec3) -> f32 {
        let p = point * self.frequency;
        let xi = p.x.floor() as i32;
        let yi = p.y.floor() as i32;
        let zi = p.z.floor() as i32;

        let mut nearest_sq: f32 = f32::MAX;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let cx = xi + dx;
                    let cy = yi + dy;
                    let cz = zi + dz;
                    // Feature point inside the cell, jittered by two hashes
                    let hx = pcg_hash3(cx, cy, cz, self.seed);
                    let hy = pcg_hash3(cx, cy, cz, self.seed.wrapping_add(0x1000_0001));
                    let hz = pcg_hash3(cx, cy, cz, self.seed.wrapping_add(0x2000_0001));
                    let jitter_x = (hx as f32) / (u32::MAX as f32);
                    let jitter_y = (hy as f32) / (u32::MAX as f32);
                    let jitter_z = (hz as f32) / (u32::MAX as f32);
                    let fp_x = cx as f32 + jitter_x;
                    let fp_y = cy as f32 + jitter_y;
                    let fp_z = cz as f32 + jitter_z;
                    let dx_f = p.x - fp_x;
                    let dy_f = p.y - fp_y;
                    let dz_f = p.z - fp_z;
                    let sq = dx_f * dx_f + dy_f * dy_f + dz_f * dz_f;
                    if sq < nearest_sq {
                        nearest_sq = sq;
                    }
                }
            }
        }
        // Maximum reachable distance for a jittered point inside a 3x3x3 search
        // is roughly sqrt(3) (unit cell diagonal). Normalize into [0, 1].
        let d = nearest_sq.sqrt();
        (d / 3f32.sqrt()).clamp(0.0, 1.0)
    }
}

// ============================================================================
// Fractional Brownian motion (fbm) over any NoiseField
// ============================================================================

/// Multi-octave sum of a base noise field
///
/// Composes `octaves` samples of `base` at doubling frequencies with
/// halving amplitude. Produces a richer / more natural noise field
/// suitable for large-scale features like clouds and terrain.
///
/// # Panics
/// Panics if `octaves == 0`.
#[must_use]
pub fn fbm<N: NoiseField>(base: &N, point: Vec3, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    assert!(octaves > 0, "octaves must be > 0");
    let mut freq = 1.0_f32;
    let mut amp = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut norm = 0.0_f32;
    for _ in 0..octaves {
        sum += base.sample_scalar(point * freq) * amp;
        norm += amp;
        freq *= lacunarity;
        amp *= gain;
    }
    if norm > 0.0 {
        (sum / norm).clamp(0.0, 1.0)
    } else {
        0.0
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

    // ------------------------------------------------------------------
    // Perlin
    // ------------------------------------------------------------------

    #[test]
    fn perlin_output_bounded() {
        let noise = PerlinNoise::new(0);
        for i in 0..100 {
            let p = Vec3::new(i as f32 * 0.03, i as f32 * 0.05, i as f32 * 0.07);
            let s = noise.sample_scalar(p);
            assert!((0.0..=1.0).contains(&s), "out of range at i={i}: {s}");
        }
    }

    #[test]
    fn perlin_smooth_across_cells() {
        let noise = PerlinNoise::new(0);
        let a = noise.sample_scalar(Vec3::new(0.5, 0.5, 0.5));
        let b = noise.sample_scalar(Vec3::new(0.501, 0.5, 0.5));
        // Perlin is C^2 continuous; a small step should give a small delta
        assert!(
            (a - b).abs() < 0.05,
            "expected smooth transition, got {a} vs {b}"
        );
    }

    #[test]
    fn perlin_deterministic() {
        let noise = PerlinNoise::new(7);
        let a = noise.sample_scalar(Vec3::new(1.3, 2.4, 3.5));
        let b = noise.sample_scalar(Vec3::new(1.3, 2.4, 3.5));
        assert!((a - b).abs() < 1e-6);
    }

    #[test]
    fn perlin_lattice_midvalue() {
        // At integer lattice points all gradient dots are zero, so output is 0.5
        let noise = PerlinNoise::new(0);
        let out = noise.sample_scalar(Vec3::new(3.0, 5.0, 7.0));
        assert!(
            (out - 0.5).abs() < 1e-4,
            "lattice point should be mid, got {out}"
        );
    }

    // ------------------------------------------------------------------
    // Worley
    // ------------------------------------------------------------------

    #[test]
    fn worley_output_bounded() {
        let noise = WorleyNoise::new(0);
        for i in 0..100 {
            let p = Vec3::new(i as f32 * 0.13, i as f32 * 0.17, i as f32 * 0.19);
            let s = noise.sample_scalar(p);
            assert!((0.0..=1.0).contains(&s), "out of range at i={i}: {s}");
        }
    }

    #[test]
    fn worley_deterministic() {
        let noise = WorleyNoise::new(42);
        let a = noise.sample_scalar(Vec3::new(2.3, 4.5, 6.7));
        let b = noise.sample_scalar(Vec3::new(2.3, 4.5, 6.7));
        assert!((a - b).abs() < 1e-6);
    }

    #[test]
    fn worley_frequency_creates_more_cells() {
        let low = WorleyNoise::new(0).with_frequency(1.0);
        let high = WorleyNoise::new(0).with_frequency(10.0);
        // Different frequency should produce different noise at same point
        let a = low.sample_scalar(Vec3::new(1.0, 1.0, 1.0));
        let b = high.sample_scalar(Vec3::new(1.0, 1.0, 1.0));
        assert!((a - b).abs() > 1e-3);
    }

    // ------------------------------------------------------------------
    // fbm
    // ------------------------------------------------------------------

    #[test]
    fn fbm_bounded() {
        let base = PerlinNoise::new(0);
        for i in 0..50 {
            let p = Vec3::new(i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3);
            let s = fbm(&base, p, 4, 2.0, 0.5);
            assert!((0.0..=1.0).contains(&s), "out of range at i={i}: {s}");
        }
    }

    #[test]
    fn fbm_single_octave_matches_base() {
        let base = PerlinNoise::new(0);
        let p = Vec3::new(1.5, 2.5, 3.5);
        let single = fbm(&base, p, 1, 2.0, 0.5);
        let raw = base.sample_scalar(p);
        assert!(
            (single - raw).abs() < 1e-4,
            "1-octave fbm should equal base, got {single} vs {raw}"
        );
    }

    #[test]
    #[should_panic(expected = "octaves must be > 0")]
    fn fbm_zero_octaves_panics() {
        let base = PerlinNoise::new(0);
        let _ = fbm(&base, Vec3::ZERO, 0, 2.0, 0.5);
    }
}
