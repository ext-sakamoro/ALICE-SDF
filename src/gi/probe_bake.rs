//! Probe Baking: Fill irradiance probes via cone tracing (Deep Fried Edition)
//!
//! Bakes indirect lighting into the probe grid by running cone traces
//! from each probe position. Uses Rayon for parallel baking.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use rayon::prelude::*;

use super::cone_trace::{trace_hemisphere, ConeTraceConfig};
use super::irradiance::{IrradianceGrid, IrradianceProbe};
use super::DirectionalLight;
use crate::svo::SparseVoxelOctree;

/// Configuration for GI probe baking
#[derive(Debug, Clone)]
pub struct BakeGiConfig {
    /// Probe grid dimensions [X, Y, Z]
    pub grid_size: [u32; 3],
    /// World-space min bounds for probe grid
    pub bounds_min: Vec3,
    /// World-space max bounds for probe grid
    pub bounds_max: Vec3,
    /// Number of sample directions per probe
    pub samples_per_probe: u32,
    /// Cone trace configuration
    pub cone_config: ConeTraceConfig,
    /// Optional directional light for direct lighting contribution
    pub sun: Option<DirectionalLight>,
}

impl Default for BakeGiConfig {
    fn default() -> Self {
        BakeGiConfig {
            grid_size: [8, 8, 8],
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            samples_per_probe: 32,
            cone_config: ConeTraceConfig {
                num_cones: 5,
                max_distance: 10.0,
                ..Default::default()
            },
            sun: Some(DirectionalLight::default()),
        }
    }
}

/// Bake an irradiance grid from an SVO
///
/// For each probe, traces cones in uniformly distributed directions
/// and accumulates the results into L1 spherical harmonics.
/// Uses Rayon for parallel probe baking.
pub fn bake_irradiance_grid(svo: &SparseVoxelOctree, config: &BakeGiConfig) -> IrradianceGrid {
    let mut grid = IrradianceGrid::new(config.grid_size, config.bounds_min, config.bounds_max);

    // Generate uniform sample directions on sphere
    let directions = generate_uniform_directions(config.samples_per_probe);

    // Bake probes in parallel
    let baked_probes: Vec<IrradianceProbe> = grid
        .probes
        .par_iter()
        .map(|probe| {
            bake_single_probe(
                svo,
                probe.position,
                &directions,
                &config.cone_config,
                config.sun.as_ref(),
            )
        })
        .collect();

    grid.probes = baked_probes;
    grid
}

/// Bake a single probe at a given position
fn bake_single_probe(
    svo: &SparseVoxelOctree,
    position: Vec3,
    directions: &[(Vec3, f32)],
    cone_config: &ConeTraceConfig,
    light: Option<&DirectionalLight>,
) -> IrradianceProbe {
    let mut probe = IrradianceProbe {
        position,
        ..Default::default()
    };

    // Check if probe is inside geometry (skip if so)
    let dist = crate::svo::svo_query_point(svo, position);
    if dist < -0.1 {
        // Deep inside geometry, skip
        return probe;
    }

    for &(dir, weight) in directions {
        let result = super::cone_trace::cone_trace(svo, position, dir, cone_config, light);

        probe.add_sample(dir, result.color * weight);
    }

    probe.normalize(directions.len() as u32);
    probe
}

/// Generate uniformly distributed directions on a sphere
///
/// Uses Fibonacci sphere distribution for even coverage.
fn generate_uniform_directions(count: u32) -> Vec<(Vec3, f32)> {
    let mut dirs = Vec::with_capacity(count as usize);
    let golden_ratio = (1.0 + 5.0f32.sqrt()) / 2.0;
    let inv_golden = 1.0 / golden_ratio;
    let inv_count = 1.0 / count as f32;
    let weight = 1.0; // Uniform weight for spherical sampling

    for i in 0..count {
        let theta = std::f32::consts::TAU * i as f32 * inv_golden;
        let phi = (1.0 - 2.0 * (i as f32 + 0.5) * inv_count).acos();

        let x = phi.sin() * theta.cos();
        let y = phi.sin() * theta.sin();
        let z = phi.cos();

        dirs.push((Vec3::new(x, y, z), weight));
    }

    dirs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svo::SvoBuildConfig;
    use crate::types::SdfNode;

    fn make_test_svo() -> SparseVoxelOctree {
        let sphere = SdfNode::sphere(1.0);
        let config = SvoBuildConfig {
            max_depth: 3,
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            ..Default::default()
        };
        SparseVoxelOctree::build(&sphere, &config)
    }

    #[test]
    fn test_bake_irradiance_grid() {
        let svo = make_test_svo();
        let config = BakeGiConfig {
            grid_size: [2, 2, 2],
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            samples_per_probe: 8,
            cone_config: ConeTraceConfig {
                max_distance: 5.0,
                num_cones: 3,
                ..Default::default()
            },
            sun: Some(DirectionalLight::default()),
        };

        let grid = bake_irradiance_grid(&svo, &config);

        assert_eq!(grid.probe_count(), 8);

        // At least some probes should have non-zero irradiance
        let mut any_nonzero = false;
        for probe in &grid.probes {
            let irr = probe.evaluate(Vec3::Y);
            if irr.length() > 0.001 {
                any_nonzero = true;
            }
        }
        assert!(any_nonzero, "Some probes should have non-zero irradiance");
    }

    #[test]
    fn test_bake_config_default() {
        let config = BakeGiConfig::default();
        assert_eq!(config.grid_size, [8, 8, 8]);
        assert!(config.samples_per_probe > 0);
        assert!(config.sun.is_some());
    }

    #[test]
    fn test_uniform_directions() {
        let dirs = generate_uniform_directions(16);
        assert_eq!(dirs.len(), 16);

        // All should be unit vectors
        for (d, w) in &dirs {
            assert!(
                (d.length() - 1.0).abs() < 0.01,
                "Direction should be unit, got {}",
                d.length()
            );
            assert!(*w > 0.0);
        }
    }

    #[test]
    fn test_uniform_directions_coverage() {
        let dirs = generate_uniform_directions(32);

        // Should cover all octants
        let mut octant_counts = [0u32; 8];
        for (d, _) in &dirs {
            let mut oct = 0u8;
            if d.x > 0.0 {
                oct |= 1;
            }
            if d.y > 0.0 {
                oct |= 2;
            }
            if d.z > 0.0 {
                oct |= 4;
            }
            octant_counts[oct as usize] += 1;
        }

        for (i, &count) in octant_counts.iter().enumerate() {
            assert!(count > 0, "Octant {} should have directions", i);
        }
    }

    #[test]
    fn test_grid_sample_after_bake() {
        let svo = make_test_svo();
        let config = BakeGiConfig {
            grid_size: [2, 2, 2],
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            samples_per_probe: 8,
            cone_config: ConeTraceConfig {
                max_distance: 5.0,
                num_cones: 3,
                ..Default::default()
            },
            sun: Some(DirectionalLight::default()),
        };

        let grid = bake_irradiance_grid(&svo, &config);

        // Sample at grid center
        let irr = grid.sample(Vec3::ZERO, Vec3::Y);
        // Should return something finite
        assert!(irr.x.is_finite() && irr.y.is_finite() && irr.z.is_finite());
    }

    #[test]
    fn test_bake_probe_inside_geometry() {
        let svo = make_test_svo();
        let dirs = generate_uniform_directions(4);
        let config = ConeTraceConfig::default();

        // Probe at center (inside sphere) should still return valid probe
        let probe = bake_single_probe(
            &svo,
            Vec3::ZERO, // inside the sphere
            &dirs,
            &config,
            None,
        );

        // Should be default (zero) since it's inside geometry
        let irr = probe.evaluate(Vec3::Y);
        assert!(
            irr.length() < 0.1,
            "Probe inside geometry should have near-zero irradiance"
        );
    }
}
