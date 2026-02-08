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

    // Division Exorcism: reciprocal multiplication instead of division
    value * (1.0 / max_value)
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

/// Branchless gradient selection via lookup table.
///
/// The 16 Perlin gradient vectors are encoded as (u_src, v_src, u_sign, v_sign):
/// - u_src/v_src: 0=x, 1=y, 2=z
/// - u_sign/v_sign: 0=positive, 1=negative
///
/// This eliminates all conditional branches from the inner loop.
#[inline(always)]
fn grad3d(hash: u32, x: f32, y: f32, z: f32) -> f32 {
    // Packed LUT: each entry is (u_src << 4 | v_src << 2 | u_sign << 1 | v_sign)
    // Derived from Perlin's original gradient table
    const LUT: [u8; 16] = [
        0b_00_01_00, // h=0:  +x +y
        0b_00_01_01, // h=1:  +x -y
        0b_00_01_10, // h=2:  -x +y
        0b_00_01_11, // h=3:  -x -y
        0b_00_10_00, // h=4:  +x +z
        0b_00_10_01, // h=5:  +x -z
        0b_00_10_10, // h=6:  -x +z
        0b_00_10_11, // h=7:  -x -z
        0b_01_10_00, // h=8:  +y +z
        0b_01_10_01, // h=9:  +y -z
        0b_01_10_10, // h=10: -y +z
        0b_01_10_11, // h=11: -y -z
        0b_01_00_00, // h=12: +y +x
        0b_00_10_10, // h=13: -x +z  (same as h=6)
        0b_01_00_10, // h=14: -y +x
        0b_00_10_01, // h=15: +x -z  (same as h=5)
    ];

    let entry = LUT[(hash & 15) as usize];
    let coords = [x, y, z];

    let u_src = ((entry >> 4) & 3) as usize;
    let v_src = ((entry >> 2) & 3) as usize;
    let u_neg = (entry >> 1) & 1;
    let v_neg = entry & 1;

    // Branchless sign application: val * (1 - 2 * sign_bit)
    let u = coords[u_src] * (1.0 - 2.0 * u_neg as f32);
    let v = coords[v_src] * (1.0 - 2.0 * v_neg as f32);
    u + v
}

/// [Deep Fried v2] Batch Perlin noise for 8 points
///
/// Evaluates noise at 8 points simultaneously. While the inner computation
/// is scalar (hash lookups prevent true SIMD), the batch interface enables
/// the caller to amortize function call overhead and allows the compiler
/// to interleave independent computations across the 8 evaluations.
#[allow(dead_code)] // batch8 API reserved for compiled SIMD evaluator
#[inline(always)]
pub fn perlin_noise_3d_batch8(
    points: &[(f32, f32, f32); 8],
    seed: u32,
) -> [f32; 8] {
    [
        perlin_noise_3d(points[0].0, points[0].1, points[0].2, seed),
        perlin_noise_3d(points[1].0, points[1].1, points[1].2, seed),
        perlin_noise_3d(points[2].0, points[2].1, points[2].2, seed),
        perlin_noise_3d(points[3].0, points[3].1, points[3].2, seed),
        perlin_noise_3d(points[4].0, points[4].1, points[4].2, seed),
        perlin_noise_3d(points[5].0, points[5].1, points[5].2, seed),
        perlin_noise_3d(points[6].0, points[6].1, points[6].2, seed),
        perlin_noise_3d(points[7].0, points[7].1, points[7].2, seed),
    ]
}

/// [Deep Fried v2] Batch noise modifier for 8 distances
///
/// Applies Perlin noise displacement to 8 distance values at once.
#[allow(dead_code)] // batch8 API reserved for compiled SIMD evaluator
#[inline(always)]
pub fn modifier_noise_perlin_batch8(
    distances: &[f32; 8],
    points: &[Vec3; 8],
    amplitude: f32,
    frequency: f32,
    seed: u32,
) -> [f32; 8] {
    let scaled: [(f32, f32, f32); 8] = [
        (points[0].x * frequency, points[0].y * frequency, points[0].z * frequency),
        (points[1].x * frequency, points[1].y * frequency, points[1].z * frequency),
        (points[2].x * frequency, points[2].y * frequency, points[2].z * frequency),
        (points[3].x * frequency, points[3].y * frequency, points[3].z * frequency),
        (points[4].x * frequency, points[4].y * frequency, points[4].z * frequency),
        (points[5].x * frequency, points[5].y * frequency, points[5].z * frequency),
        (points[6].x * frequency, points[6].y * frequency, points[6].z * frequency),
        (points[7].x * frequency, points[7].y * frequency, points[7].z * frequency),
    ];
    let noise = perlin_noise_3d_batch8(&scaled, seed);
    let mut result = [0.0f32; 8];
    for i in 0..8 {
        result[i] = distances[i] + noise[i] * amplitude;
    }
    result
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
