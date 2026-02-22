//! Volume, surface area, and center-of-mass estimation
//!
//! Uses Monte Carlo integration to estimate geometric properties of
//! arbitrary SDF shapes. The estimates converge as `1/sqrt(N)` with
//! the number of samples.
//!
//! Author: Moroya Sakamoto

use crate::eval::eval;
use crate::types::{Aabb, SdfNode};
use glam::Vec3;

// ── Result types ─────────────────────────────────────────────

/// Result of a volume estimation.
#[derive(Debug, Clone)]
pub struct VolumeEstimate {
    /// Estimated volume in cubic units.
    pub volume: f64,
    /// Standard error of the estimate.
    pub std_error: f64,
    /// Number of samples used.
    pub sample_count: u64,
    /// Fraction of samples that were inside the surface.
    pub fill_ratio: f64,
}

/// Result of a surface area estimation.
#[derive(Debug, Clone)]
pub struct AreaEstimate {
    /// Estimated surface area in square units.
    pub area: f64,
    /// Standard error of the estimate.
    pub std_error: f64,
    /// Number of samples used.
    pub sample_count: u64,
}

/// Result of a center-of-mass estimation.
#[derive(Debug, Clone)]
pub struct CenterOfMass {
    /// Estimated center of mass.
    pub center: Vec3,
    /// Number of interior samples used.
    pub interior_count: u64,
}

// ── Deterministic RNG ────────────────────────────────────────

/// Simple deterministic PRNG (xorshift64) for reproducible Monte Carlo.
struct Rng64 {
    state: u64,
}

impl Rng64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    #[inline(always)]
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    #[inline(always)]
    fn next_f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform f32 in [lo, hi).
    #[inline(always)]
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

// ── Volume estimation ────────────────────────────────────────

/// Estimate the volume enclosed by an SDF surface using Monte Carlo sampling.
///
/// Points with `eval(p) < 0` are considered inside.
///
/// # Arguments
/// * `node` - SDF tree to measure
/// * `aabb` - Bounding box to sample within
/// * `samples` - Number of random samples (higher = more accurate)
/// * `seed` - Random seed for reproducibility
pub fn estimate_volume(node: &SdfNode, aabb: Aabb, samples: u64, seed: u64) -> VolumeEstimate {
    if samples == 0 {
        return VolumeEstimate {
            volume: 0.0,
            std_error: 0.0,
            sample_count: 0,
            fill_ratio: 0.0,
        };
    }

    let mut rng = Rng64::new(seed);
    let box_volume = (aabb.max.x - aabb.min.x) as f64
        * (aabb.max.y - aabb.min.y) as f64
        * (aabb.max.z - aabb.min.z) as f64;

    let mut inside_count: u64 = 0;

    for _ in 0..samples {
        let p = Vec3::new(
            rng.next_range(aabb.min.x, aabb.max.x),
            rng.next_range(aabb.min.y, aabb.max.y),
            rng.next_range(aabb.min.z, aabb.max.z),
        );
        if eval(node, p) < 0.0 {
            inside_count += 1;
        }
    }

    let ratio = inside_count as f64 / samples as f64;
    let volume = ratio * box_volume;
    // Standard error via binomial proportion
    let variance = ratio * (1.0 - ratio) / samples as f64;
    let std_error = variance.sqrt() * box_volume;

    VolumeEstimate {
        volume,
        std_error,
        sample_count: samples,
        fill_ratio: ratio,
    }
}

/// Estimate the surface area using the epsilon-layer method.
///
/// Counts samples where `|eval(p)| < epsilon` and estimates
/// area ≈ (count / N) * box_volume / (2 * epsilon).
pub fn estimate_surface_area(
    node: &SdfNode,
    aabb: Aabb,
    samples: u64,
    epsilon: f32,
    seed: u64,
) -> AreaEstimate {
    if samples == 0 || epsilon <= 0.0 {
        return AreaEstimate {
            area: 0.0,
            std_error: 0.0,
            sample_count: 0,
        };
    }

    let mut rng = Rng64::new(seed);
    let box_volume = (aabb.max.x - aabb.min.x) as f64
        * (aabb.max.y - aabb.min.y) as f64
        * (aabb.max.z - aabb.min.z) as f64;

    let mut near_count: u64 = 0;

    for _ in 0..samples {
        let p = Vec3::new(
            rng.next_range(aabb.min.x, aabb.max.x),
            rng.next_range(aabb.min.y, aabb.max.y),
            rng.next_range(aabb.min.z, aabb.max.z),
        );
        if eval(node, p).abs() < epsilon {
            near_count += 1;
        }
    }

    let ratio = near_count as f64 / samples as f64;
    let area = ratio * box_volume / (2.0 * epsilon as f64);
    let variance = ratio * (1.0 - ratio) / samples as f64;
    let std_error = variance.sqrt() * box_volume / (2.0 * epsilon as f64);

    AreaEstimate {
        area,
        std_error,
        sample_count: samples,
    }
}

/// Estimate the center of mass (assuming uniform density).
pub fn estimate_center_of_mass(
    node: &SdfNode,
    aabb: Aabb,
    samples: u64,
    seed: u64,
) -> CenterOfMass {
    if samples == 0 {
        return CenterOfMass {
            center: Vec3::ZERO,
            interior_count: 0,
        };
    }

    let mut rng = Rng64::new(seed);
    let mut sum = Vec3::ZERO;
    let mut count: u64 = 0;

    for _ in 0..samples {
        let p = Vec3::new(
            rng.next_range(aabb.min.x, aabb.max.x),
            rng.next_range(aabb.min.y, aabb.max.y),
            rng.next_range(aabb.min.z, aabb.max.z),
        );
        if eval(node, p) < 0.0 {
            sum += p;
            count += 1;
        }
    }

    let center = if count > 0 {
        sum / count as f32
    } else {
        Vec3::ZERO
    };

    CenterOfMass {
        center,
        interior_count: count,
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere_aabb(r: f32) -> Aabb {
        Aabb {
            min: Vec3::splat(-r * 1.1),
            max: Vec3::splat(r * 1.1),
        }
    }

    #[test]
    fn volume_unit_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let est = estimate_volume(&sphere, sphere_aabb(1.0), 100_000, 42);
        let expected = 4.0 / 3.0 * std::f64::consts::PI;
        let error = (est.volume - expected).abs() / expected;
        assert!(
            error < 0.05,
            "Volume error {:.1}%, expected < 5%",
            error * 100.0
        );
    }

    #[test]
    fn volume_box() {
        // box3d(w, h, d) uses half_extents = (w/2, h/2, d/2)
        // So box3d(2.0, 2.0, 2.0) spans [-1, 1]^3, volume = 8.0
        let b = SdfNode::box3d(2.0, 2.0, 2.0);
        let aabb = Aabb {
            min: Vec3::splat(-1.5),
            max: Vec3::splat(1.5),
        };
        let est = estimate_volume(&b, aabb, 100_000, 123);
        let expected = 8.0; // 2×2×2
        let error = (est.volume - expected).abs() / expected;
        assert!(error < 0.05, "Volume error {:.1}%", error * 100.0);
    }

    #[test]
    fn volume_zero_samples() {
        let sphere = SdfNode::sphere(1.0);
        let est = estimate_volume(&sphere, sphere_aabb(1.0), 0, 0);
        assert_eq!(est.volume, 0.0);
        assert_eq!(est.sample_count, 0);
    }

    #[test]
    fn volume_deterministic() {
        let sphere = SdfNode::sphere(1.0);
        let e1 = estimate_volume(&sphere, sphere_aabb(1.0), 10_000, 42);
        let e2 = estimate_volume(&sphere, sphere_aabb(1.0), 10_000, 42);
        assert_eq!(e1.volume, e2.volume);
    }

    #[test]
    fn surface_area_unit_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let est = estimate_surface_area(&sphere, sphere_aabb(1.0), 500_000, 0.05, 42);
        let expected = 4.0 * std::f64::consts::PI;
        let error = (est.area - expected).abs() / expected;
        assert!(
            error < 0.15,
            "Area error {:.1}%, expected < 15%",
            error * 100.0
        );
    }

    #[test]
    fn surface_area_zero_epsilon() {
        let sphere = SdfNode::sphere(1.0);
        let est = estimate_surface_area(&sphere, sphere_aabb(1.0), 1000, 0.0, 0);
        assert_eq!(est.area, 0.0);
    }

    #[test]
    fn center_of_mass_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let com = estimate_center_of_mass(&sphere, sphere_aabb(1.0), 100_000, 42);
        // Sphere centered at origin: COM should be near (0,0,0)
        assert!(
            com.center.length() < 0.05,
            "COM {:?} too far from origin",
            com.center
        );
        assert!(com.interior_count > 0);
    }

    #[test]
    fn center_of_mass_translated() {
        let sphere = SdfNode::sphere(0.5).translate(2.0, 0.0, 0.0);
        let aabb = Aabb {
            min: Vec3::new(1.0, -1.0, -1.0),
            max: Vec3::new(3.0, 1.0, 1.0),
        };
        let com = estimate_center_of_mass(&sphere, aabb, 50_000, 42);
        assert!(
            (com.center.x - 2.0).abs() < 0.1,
            "COM.x {:?} should be near 2.0",
            com.center
        );
    }

    #[test]
    fn fill_ratio_full_box() {
        // box3d(2.0, 2.0, 2.0) → half_extents = (1,1,1) → spans [-1, 1]^3
        // AABB also [-1, 1]^3 → fill_ratio should be ~1.0
        let b = SdfNode::box3d(2.0, 2.0, 2.0);
        let aabb = Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        let est = estimate_volume(&b, aabb, 10_000, 42);
        assert!(est.fill_ratio > 0.95, "fill_ratio={}", est.fill_ratio);
    }

    #[test]
    fn std_error_decreases_with_samples() {
        let sphere = SdfNode::sphere(1.0);
        let e_low = estimate_volume(&sphere, sphere_aabb(1.0), 1_000, 42);
        let e_high = estimate_volume(&sphere, sphere_aabb(1.0), 100_000, 42);
        assert!(e_high.std_error < e_low.std_error);
    }
}
