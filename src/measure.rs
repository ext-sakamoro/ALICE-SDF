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
    const fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    #[inline(always)]
    const fn next(&mut self) -> u64 {
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
        self.next_f32() * (hi - lo) + lo
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
// ── to append to src/measure.rs ──────────────────────────────

/// Result of a tension measurement — what the field actually does over a
/// region, as opposed to what its Lipschitz bound allows.
#[derive(Debug, Clone)]
pub struct TensionEstimate {
    /// Largest measured `|f(p) − f(q)| / |p − q|`. Above `1` the field
    /// over-reports distance somewhere in the region and a sphere-tracing
    /// step of `f` can cross the surface there.
    pub max_quotient: f32,
    /// Smallest measured quotient. Below `1` the field is slack: correct,
    /// but a marcher spends more steps covering the same ground.
    pub min_quotient: f32,
    /// The sample point where `max_quotient` occurred.
    pub at: Vec3,
    /// Its partner, so the pair can be replayed.
    pub toward: Vec3,
    /// Pairs that contributed (pairs entirely inside the solid are skipped —
    /// the Lipschitz claim is an exterior one).
    pub sample_count: u64,
    /// The separation used between the two points of a pair.
    pub probe: f32,
}

impl TensionEstimate {
    /// The factor by which the field exceeds a 1-Lipschitz law — the number
    /// to put on screen. `1` or below is safe, above `1` tears.
    #[must_use]
    pub const fn tension(&self) -> f32 {
        self.max_quotient
    }

    /// True when the measurement found over-reporting.
    #[must_use]
    pub fn tears(&self) -> bool {
        self.max_quotient > 1.0
    }
}

/// Measure the largest and smallest difference quotient of a field over a
/// region.
///
/// This is the measured counterpart of
/// [`distance_fidelity`](crate::fidelity::distance_fidelity): the bound says
/// what the law allows anywhere, this says what the field does *here*. Two
/// things only the measurement can see:
///
/// * the slack (`min_quotient`), which no Lipschitz bound reports, and which
///   is what makes a correct field slow rather than wrong;
/// * the actual tension of a *blended* region, where two laws meet over a
///   transition and the bound of neither one describes the seam.
///
/// Difference quotients rather than gradients, because a gradient sampled by
/// central differences smooths over exactly the discontinuities that break a
/// marcher. Pairs where both points are strictly inside the solid are
/// skipped: the Lipschitz claim is about the exterior, and a field is allowed
/// to be discontinuous deep inside.
///
/// # Arguments
/// * `node` — SDF tree to measure
/// * `aabb` — region to sample within
/// * `samples` — number of point pairs
/// * `seed` — random seed, for reproducibility
/// * `probe` — separation between the two points of a pair. Small enough to
///   be local, large enough that `f`'s own precision does not dominate;
///   `1e-3` of the region's size is a reasonable default.
#[must_use]
pub fn measure_tension(
    node: &SdfNode,
    aabb: Aabb,
    samples: u64,
    seed: u64,
    probe: f32,
) -> TensionEstimate {
    let mut rng = Rng64::new(seed);
    let mut max_quotient = 0.0_f32;
    let mut min_quotient = f32::INFINITY;
    let mut at = Vec3::ZERO;
    let mut toward = Vec3::ZERO;
    let mut counted: u64 = 0;

    for _ in 0..samples {
        let p = Vec3::new(
            rng.next_range(aabb.min.x, aabb.max.x),
            rng.next_range(aabb.min.y, aabb.max.y),
            rng.next_range(aabb.min.z, aabb.max.z),
        );
        // a direction on the sphere, rejection sampled so the distribution
        // does not favour the cube's corners
        let dir = loop {
            let c = Vec3::new(
                rng.next_range(-1.0, 1.0),
                rng.next_range(-1.0, 1.0),
                rng.next_range(-1.0, 1.0),
            );
            let len_sq = c.length_squared();
            if len_sq > 1e-6 && len_sq <= 1.0 {
                break c / len_sq.sqrt();
            }
        };
        let q = p + dir * probe;

        let fp = eval(node, p);
        let fq = eval(node, q);
        // exterior claim only
        if fp < 0.0 && fq < 0.0 {
            continue;
        }
        if !fp.is_finite() || !fq.is_finite() {
            continue;
        }
        let quotient = (fp - fq).abs() / probe;
        counted += 1;
        if quotient > max_quotient {
            max_quotient = quotient;
            at = p;
            toward = q;
        }
        if quotient < min_quotient {
            min_quotient = quotient;
        }
    }

    if counted == 0 {
        min_quotient = 0.0;
    }

    TensionEstimate {
        max_quotient,
        min_quotient,
        at,
        toward,
        sample_count: counted,
        probe,
    }
}
