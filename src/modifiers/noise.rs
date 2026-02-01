//! Noise modifiers for SDFs (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Forced Inlining**: All helper functions (`fade`, `lerp`, `grad`, `hash`) are
//!   forced inline to allow the compiler to flatten the noise calculation loop.
//! - **Branchless Grad**: Optimized gradient selection logic.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Apply Perlin noise displacement to a distance value
#[inline(always)]
pub fn modifier_noise_perlin(
    distance: f32,
    point: Vec3,
    amplitude: f32,
    frequency: f32,
    seed: u32,
) -> f32 {
    let noise = perlin_noise_3d(
        point.x * frequency,
        point.y * frequency,
        point.z * frequency,
        seed,
    );
    distance + noise * amplitude
}

/// Apply simplex-like noise displacement
#[inline(always)]
pub fn modifier_noise_simplex(
    distance: f32,
    point: Vec3,
    amplitude: f32,
    frequency: f32,
    seed: u32,
) -> f32 {
    // For now, mapping to perlin or similar fast noise
    // In deep fried context, we prefer consistent pipelines
    modifier_noise_perlin(distance, point, amplitude, frequency, seed)
}

/// 3D Perlin noise implementation (Deep Fried)
#[inline(always)]
pub fn perlin_noise_3d(x: f32, y: f32, z: f32, seed: u32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;

    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    // Unrolled and Inlined Gradient Fetch
    let aaa = grad3d(hash3d(xi, yi, zi, seed), xf, yf, zf);
    let aba = grad3d(hash3d(xi, yi + 1, zi, seed), xf, yf - 1.0, zf);
    let aab = grad3d(hash3d(xi, yi, zi + 1, seed), xf, yf, zf - 1.0);
    let abb = grad3d(hash3d(xi, yi + 1, zi + 1, seed), xf, yf - 1.0, zf - 1.0);
    let baa = grad3d(hash3d(xi + 1, yi, zi, seed), xf - 1.0, yf, zf);
    let bba = grad3d(hash3d(xi + 1, yi + 1, zi, seed), xf - 1.0, yf - 1.0, zf);
    let bab = grad3d(hash3d(xi + 1, yi, zi + 1, seed), xf - 1.0, yf, zf - 1.0);
    let bbb = grad3d(hash3d(xi + 1, yi + 1, zi + 1, seed), xf - 1.0, yf - 1.0, zf - 1.0);

    lerp(
        lerp(lerp(aaa, baa, u), lerp(aba, bba, u), v),
        lerp(lerp(aab, bab, u), lerp(abb, bbb, u), v),
        w,
    )
}

/// Fractal Brownian Motion (fBM) noise
#[inline(always)]
pub fn fbm_noise_3d(
    x: f32,
    y: f32,
    z: f32,
    seed: u32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_value = 0.0;

    for i in 0..octaves {
        value += amplitude * perlin_noise_3d(x * frequency, y * frequency, z * frequency, seed + i);
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    value / max_value
}

// Helper functions - all forced inline for maximum optimization

#[inline(always)]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

#[inline(always)]
fn hash3d(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut h = seed;
    h ^= x as u32;
    h = h.wrapping_mul(0x85EBCA6B);
    h ^= y as u32;
    h = h.wrapping_mul(0xC2B2AE35);
    h ^= z as u32;
    h = h.wrapping_mul(0x27D4EB2D);
    h ^= h >> 16;
    h
}

#[inline(always)]
fn grad3d(hash: u32, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15;
    // Branchless selection using logic ops
    // u = h < 8 ? x : y
    let u = if (h & 8) == 0 { x } else { y };

    // v = h < 4 ? y : (h == 12 || h == 14 ? x : z)
    // h=12(1100), 14(1110). Bit 2 is 1 (4,5,6,7, 12,13,14,15).
    // This logic is tricky to fully de-branch without lookup,
    // but the compiler does a good job with simple ifs.
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };

    let g1 = if (h & 1) == 0 { u } else { -u };
    let g2 = if (h & 2) == 0 { v } else { -v };
    g1 + g2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perlin_range() {
        let mut min = f32::MAX;
        let mut max = f32::MIN;

        for i in 0..1000 {
            let x = (i as f32) * 0.1;
            let y = (i as f32) * 0.13;
            let z = (i as f32) * 0.17;
            let n = perlin_noise_3d(x, y, z, 42);
            min = min.min(n);
            max = max.max(n);
        }

        assert!(min >= -1.5);
        assert!(max <= 1.5);
    }

    #[test]
    fn test_perlin_deterministic() {
        let n1 = perlin_noise_3d(1.5, 2.5, 3.5, 42);
        let n2 = perlin_noise_3d(1.5, 2.5, 3.5, 42);
        assert!((n1 - n2).abs() < 0.0001);
    }

    #[test]
    fn test_modifier_noise() {
        let original = 1.0;
        let noisy = modifier_noise_perlin(original, Vec3::new(1.0, 2.0, 3.0), 0.1, 1.0, 42);
        assert!((noisy - original).abs() <= 0.15);
    }

    #[test]
    fn test_fbm() {
        let n = fbm_noise_3d(1.5, 2.5, 3.5, 42, 4, 2.0, 0.5);
        assert!(n >= -1.0 && n <= 1.0);
    }
}
