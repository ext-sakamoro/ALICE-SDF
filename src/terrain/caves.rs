//! Cave Generation: SDF-based subterranean features (Deep Fried Edition)
//!
//! Generates cave/overhang SDFs that can be combined with the heightmap
//! terrain using CSG operations. Uses layered 3D noise.
//!
//! Author: Moroya Sakamoto

use crate::types::SdfNode;
use glam::Vec3;

/// Configuration for cave generation
#[derive(Debug, Clone)]
pub struct CaveConfig {
    /// Maximum depth below surface for caves
    pub max_depth: f32,
    /// Minimum depth below surface for caves to start
    pub min_depth: f32,
    /// Cave tunnel radius
    pub tunnel_radius: f32,
    /// Number of noise octaves for cave shape
    pub octaves: u32,
    /// Noise frequency
    pub frequency: f32,
    /// Cave density (0.0 = no caves, 1.0 = very cavernous)
    pub density: f32,
    /// Random seed
    pub seed: u64,
}

impl Default for CaveConfig {
    fn default() -> Self {
        CaveConfig {
            max_depth: 50.0,
            min_depth: 5.0,
            tunnel_radius: 3.0,
            octaves: 3,
            frequency: 0.05,
            density: 0.3,
            seed: 42,
        }
    }
}

/// Generate a cave SDF from configuration
///
/// Creates a series of connected SDF spheres forming cave tunnels,
/// then applies noise displacement for organic shapes.
///
/// The result is an SdfNode that can be subtracted from terrain:
/// `terrain_sdf = max(height_sdf, -cave_sdf)`
pub fn generate_cave_sdf(config: &CaveConfig) -> SdfNode {
    let mut rng = config.seed;
    let num_tunnels = (config.density * 10.0).max(1.0) as u32;
    let segments_per_tunnel = 8u32;

    let mut cave_parts: Vec<SdfNode> = Vec::new();

    for _ in 0..num_tunnels {
        // Random starting point
        rng = lcg_next(rng);
        let mut px = (lcg_float(rng) * 2.0 - 1.0) * 20.0;
        rng = lcg_next(rng);
        let mut py = -(config.min_depth + lcg_float(rng) * (config.max_depth - config.min_depth));
        rng = lcg_next(rng);
        let mut pz = (lcg_float(rng) * 2.0 - 1.0) * 20.0;

        for _ in 0..segments_per_tunnel {
            // Random direction bias (mostly horizontal)
            rng = lcg_next(rng);
            let dx = (lcg_float(rng) * 2.0 - 1.0) * 5.0;
            rng = lcg_next(rng);
            let dy = (lcg_float(rng) * 2.0 - 1.0) * 2.0;
            rng = lcg_next(rng);
            let dz = (lcg_float(rng) * 2.0 - 1.0) * 5.0;

            rng = lcg_next(rng);
            let radius = config.tunnel_radius * (0.5 + lcg_float(rng) * 1.0);

            // Create a capsule segment
            let segment = SdfNode::sphere(radius).translate(px, py, pz);
            cave_parts.push(segment);

            px += dx;
            py = (py + dy).clamp(-config.max_depth, -config.min_depth);
            pz += dz;
        }
    }

    // Combine all cave segments with smooth union
    if cave_parts.is_empty() {
        // Return a sphere at infinity (no caves)
        return SdfNode::sphere(0.01).translate(0.0, -10000.0, 0.0);
    }

    let mut result = cave_parts.remove(0);
    for part in cave_parts {
        result = result.smooth_union(part, config.tunnel_radius * 0.5);
    }

    // Add noise displacement for organic look
    if config.octaves > 0 {
        result = result.noise(config.frequency, 1.0 / config.frequency, config.seed as u32);
    }

    result
}

/// Generate a simple chamber (large open cave room)
pub fn generate_chamber(center: Vec3, radius: f32, height: f32) -> SdfNode {
    SdfNode::box3d(radius, height * 0.5, radius)
        .round(radius * 0.3)
        .translate(center.x, center.y, center.z)
}

#[inline]
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

#[inline]
fn lcg_float(state: u64) -> f32 {
    ((state >> 16) as u32 as f32) / (u32::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::eval;

    #[test]
    fn test_generate_cave_sdf() {
        let config = CaveConfig {
            density: 0.2,
            octaves: 0, // No noise for deterministic test
            ..Default::default()
        };
        let cave = generate_cave_sdf(&config);

        // Cave should be somewhere below the surface
        let d = eval(&cave, Vec3::new(0.0, -20.0, 0.0));
        // Just verify it evaluates without panicking and returns finite
        assert!(d.is_finite(), "Cave SDF should return finite distance");
    }

    #[test]
    fn test_cave_config_default() {
        let config = CaveConfig::default();
        assert_eq!(config.max_depth, 50.0);
        assert_eq!(config.min_depth, 5.0);
        assert!(config.density > 0.0);
    }

    #[test]
    fn test_generate_chamber() {
        let chamber = generate_chamber(Vec3::new(0.0, -10.0, 0.0), 5.0, 4.0);

        // Center of chamber should be inside (negative)
        let d = eval(&chamber, Vec3::new(0.0, -10.0, 0.0));
        assert!(d < 0.0, "Chamber center should be inside, got {}", d);

        // Far away should be outside
        let d = eval(&chamber, Vec3::new(100.0, 100.0, 100.0));
        assert!(d > 0.0);
    }

    #[test]
    fn test_cave_deterministic() {
        let config = CaveConfig {
            seed: 123,
            octaves: 0,
            ..Default::default()
        };

        let cave1 = generate_cave_sdf(&config);
        let cave2 = generate_cave_sdf(&config);

        let p = Vec3::new(5.0, -15.0, 3.0);
        let d1 = eval(&cave1, p);
        let d2 = eval(&cave2, p);
        assert!(
            (d1 - d2).abs() < 0.001,
            "Same seed should produce same cave"
        );
    }

    #[test]
    fn test_cave_zero_density() {
        let config = CaveConfig {
            density: 0.0,
            octaves: 0,
            ..Default::default()
        };
        let cave = generate_cave_sdf(&config);
        // Should still return a valid SDF (single tiny sphere at -10000)
        let d = eval(&cave, Vec3::ZERO);
        assert!(d.is_finite());
    }
}
