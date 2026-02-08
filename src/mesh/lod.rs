//! LOD (Level of Detail) Generation
//!
//! Generates multiple levels of detail from SDF for efficient rendering
//! at different distances.
//!
//! # Features
//!
//! - Progressive LOD chain generation
//! - Error metric computation
//! - Screen-space error based selection
//! - Continuous LOD blending support
//!
//! Author: Moroya Sakamoto

use crate::mesh::{sdf_to_mesh, MarchingCubesConfig, Mesh, Vertex};
use crate::types::SdfNode;
use glam::Vec3;

/// LOD entry with mesh and metadata
#[derive(Debug, Clone)]
pub struct LodMesh {
    /// LOD level (0 = highest detail)
    pub level: u32,
    /// Resolution used for generation
    pub resolution: u32,
    /// The mesh at this LOD
    pub mesh: Mesh,
    /// Maximum geometric error
    pub max_error: f32,
    /// Screen-space error threshold for this LOD
    pub screen_threshold: f32,
    /// Minimum distance for this LOD
    pub min_distance: f32,
    /// Maximum distance for this LOD
    pub max_distance: f32,
}

impl LodMesh {
    /// Check if this LOD should be used at given distance
    #[inline]
    pub fn is_active(&self, distance: f32) -> bool {
        distance >= self.min_distance && distance < self.max_distance
    }

    /// Compute blend factor for continuous LOD (0 = this LOD, 1 = next LOD)
    #[inline]
    pub fn blend_factor(&self, distance: f32) -> f32 {
        let range = self.max_distance - self.min_distance;
        if range <= 0.0 {
            return 0.0;
        }
        ((distance - self.min_distance) / range).clamp(0.0, 1.0)
    }
}

/// LOD chain containing multiple detail levels
#[derive(Debug)]
pub struct LodChain {
    /// All LOD levels (sorted by level, 0 = highest detail)
    pub levels: Vec<LodMesh>,
    /// Base bounds of the model
    pub bounds_min: Vec3,
    /// Base bounds of the model
    pub bounds_max: Vec3,
    /// Total bounding sphere radius
    pub bounding_radius: f32,
}

impl LodChain {
    /// Get LOD level for a given distance
    pub fn get_lod(&self, distance: f32) -> Option<&LodMesh> {
        self.levels.iter().find(|l| l.is_active(distance))
    }

    /// Get LOD level by index
    pub fn get_level(&self, level: u32) -> Option<&LodMesh> {
        self.levels.iter().find(|l| l.level == level)
    }

    /// Get the best LOD for screen-space error threshold
    pub fn select_by_error(&self, distance: f32, error_threshold: f32) -> Option<&LodMesh> {
        // Start from highest detail and find first that meets threshold
        for lod in self.levels.iter().rev() {
            let screen_error = lod.max_error / distance.max(0.001);
            if screen_error <= error_threshold {
                return Some(lod);
            }
        }

        // Return lowest detail if nothing meets threshold
        self.levels.last()
    }

    /// Get pair of LODs for blending at given distance
    pub fn get_blend_pair(&self, distance: f32) -> Option<(&LodMesh, &LodMesh, f32)> {
        for i in 0..self.levels.len().saturating_sub(1) {
            if self.levels[i].is_active(distance) {
                let blend = self.levels[i].blend_factor(distance);
                return Some((&self.levels[i], &self.levels[i + 1], blend));
            }
        }
        None
    }

    /// Get total triangle count at LOD 0
    pub fn base_triangle_count(&self) -> usize {
        self.levels
            .first()
            .map(|l| l.mesh.triangle_count())
            .unwrap_or(0)
    }

    /// Get memory usage in bytes (approximate)
    pub fn memory_usage(&self) -> usize {
        self.levels
            .iter()
            .map(|l| {
                l.mesh.vertices.len() * std::mem::size_of::<Vertex>()
                    + l.mesh.indices.len() * std::mem::size_of::<u32>()
            })
            .sum()
    }
}

/// Configuration for LOD generation
#[derive(Debug, Clone)]
pub struct LodConfig {
    /// Number of LOD levels to generate
    pub num_levels: u32,
    /// Base resolution (LOD 0)
    pub base_resolution: u32,
    /// Resolution reduction factor per level (0.5 = halve each level)
    pub reduction_factor: f32,
    /// Distance multiplier per LOD level
    pub distance_multiplier: f32,
    /// Base distance for LOD 0
    pub base_distance: f32,
    /// Compute vertex normals
    pub compute_normals: bool,
}

impl Default for LodConfig {
    fn default() -> Self {
        LodConfig {
            num_levels: 5,
            base_resolution: 64,
            reduction_factor: 0.5,
            distance_multiplier: 2.0,
            base_distance: 1.0,
            compute_normals: true,
        }
    }
}

impl LodConfig {
    /// Create config for high quality
    pub fn high_quality() -> Self {
        LodConfig {
            num_levels: 6,
            base_resolution: 128,
            reduction_factor: 0.6,
            distance_multiplier: 2.0,
            base_distance: 0.5,
            compute_normals: true,
        }
    }

    /// Create config for balanced quality/performance
    pub fn balanced() -> Self {
        LodConfig {
            num_levels: 4,
            base_resolution: 48,
            reduction_factor: 0.5,
            distance_multiplier: 2.5,
            base_distance: 1.0,
            compute_normals: true,
        }
    }

    /// Create config for performance
    pub fn fast() -> Self {
        LodConfig {
            num_levels: 3,
            base_resolution: 32,
            reduction_factor: 0.5,
            distance_multiplier: 3.0,
            base_distance: 2.0,
            compute_normals: false,
        }
    }

    /// Get resolution at a specific LOD level
    pub fn resolution_at_level(&self, level: u32) -> u32 {
        let factor = self.reduction_factor.powi(level as i32);
        ((self.base_resolution as f32 * factor) as u32).max(4)
    }

    /// Get distance range for a specific LOD level
    pub fn distance_range(&self, level: u32) -> (f32, f32) {
        let min = if level == 0 {
            0.0 // LOD 0 starts at distance 0
        } else {
            self.base_distance * self.distance_multiplier.powi(level as i32)
        };
        let max = if level < self.num_levels - 1 {
            self.base_distance * self.distance_multiplier.powi((level + 1) as i32)
        } else {
            f32::INFINITY
        };
        (min, max)
    }
}

/// Configuration for decimation-based LOD generation
#[derive(Debug, Clone)]
pub struct DecimationLodConfig {
    /// Number of LOD levels to generate
    pub num_levels: u32,
    /// Base resolution for LOD 0 mesh generation
    pub base_resolution: u32,
    /// Decimation ratio per level (0.5 = halve triangle count each level)
    pub decimation_ratio: f32,
    /// Distance multiplier per LOD level
    pub distance_multiplier: f32,
    /// Base distance for LOD 0
    pub base_distance: f32,
    /// Compute vertex normals
    pub compute_normals: bool,
    /// Preserve material boundaries during decimation
    pub preserve_materials: bool,
}

impl Default for DecimationLodConfig {
    fn default() -> Self {
        DecimationLodConfig {
            num_levels: 5,
            base_resolution: 64,
            decimation_ratio: 0.5,
            distance_multiplier: 2.0,
            base_distance: 1.0,
            compute_normals: true,
            preserve_materials: true,
        }
    }
}

impl DecimationLodConfig {
    /// High quality decimation LOD
    pub fn high_quality() -> Self {
        DecimationLodConfig {
            num_levels: 6,
            base_resolution: 128,
            decimation_ratio: 0.5,
            distance_multiplier: 2.0,
            base_distance: 0.5,
            compute_normals: true,
            preserve_materials: true,
        }
    }

    /// Fast decimation LOD
    pub fn fast() -> Self {
        DecimationLodConfig {
            num_levels: 3,
            base_resolution: 32,
            decimation_ratio: 0.4,
            distance_multiplier: 3.0,
            base_distance: 2.0,
            compute_normals: true,
            preserve_materials: false,
        }
    }

    /// Get distance range for a specific LOD level
    pub fn distance_range(&self, level: u32) -> (f32, f32) {
        let min = if level == 0 {
            0.0
        } else {
            self.base_distance * self.distance_multiplier.powi(level as i32)
        };
        let max = if level < self.num_levels - 1 {
            self.base_distance * self.distance_multiplier.powi((level + 1) as i32)
        } else {
            f32::INFINITY
        };
        (min, max)
    }
}

/// Generate LOD chain using QEM decimation from a high-res base mesh
///
/// Unlike `generate_lod_chain` which re-runs marching cubes at lower resolutions,
/// this function generates LOD 0 once and progressively decimates it.
/// This produces higher-quality lower LODs because the decimation preserves
/// the original surface topology and attributes (UVs, normals, tangents).
pub fn generate_lod_chain_decimated(
    sdf: &SdfNode,
    min_bounds: Vec3,
    max_bounds: Vec3,
    config: &DecimationLodConfig,
) -> LodChain {
    use crate::mesh::decimate::{decimate, DecimateConfig};

    let bounds_radius = (max_bounds - min_bounds).length() * 0.5;

    // Generate high-res base mesh (LOD 0)
    let mc_config = MarchingCubesConfig {
        resolution: config.base_resolution as usize,
        iso_level: 0.0,
        compute_normals: config.compute_normals,
        ..Default::default()
    };

    let base_mesh = sdf_to_mesh(sdf, min_bounds, max_bounds, &mc_config);
    let base_resolution = config.base_resolution;

    let mut levels = Vec::with_capacity(config.num_levels as usize);

    // LOD 0: full resolution
    let (min_dist, max_dist) = config.distance_range(0);
    let max_error = compute_lod_error(&base_mesh, base_resolution, bounds_radius);

    levels.push(LodMesh {
        level: 0,
        resolution: base_resolution,
        mesh: base_mesh.clone(),
        max_error,
        screen_threshold: max_error / min_dist.max(0.001),
        min_distance: 0.0,
        max_distance: max_dist,
    });

    // LOD 1..N: progressive decimation
    let mut current_mesh = base_mesh;

    for level in 1..config.num_levels {
        let (min_dist, max_dist) = config.distance_range(level);

        let dec_config = DecimateConfig {
            target_ratio: config.decimation_ratio,
            max_error: f32::MAX,
            preserve_boundary: true,
            preserve_materials: config.preserve_materials,
            locked_materials: Vec::new(),
        };

        let mut lod_mesh = current_mesh.clone();
        decimate(&mut lod_mesh, &dec_config);

        let effective_res =
            (base_resolution as f32 * config.decimation_ratio.powi(level as i32)) as u32;
        let max_error = compute_lod_error(&lod_mesh, effective_res.max(4), bounds_radius);

        levels.push(LodMesh {
            level,
            resolution: effective_res.max(4),
            mesh: lod_mesh.clone(),
            max_error,
            screen_threshold: max_error / min_dist.max(0.001),
            min_distance: if level == 0 { 0.0 } else { min_dist },
            max_distance: max_dist,
        });

        current_mesh = lod_mesh;
    }

    LodChain {
        levels,
        bounds_min: min_bounds,
        bounds_max: max_bounds,
        bounding_radius: bounds_radius,
    }
}

/// Generate LOD chain from SDF
pub fn generate_lod_chain(
    sdf: &SdfNode,
    min_bounds: Vec3,
    max_bounds: Vec3,
    config: &LodConfig,
) -> LodChain {
    let mut levels = Vec::with_capacity(config.num_levels as usize);

    let _center = (min_bounds + max_bounds) * 0.5;
    let bounds_radius = (max_bounds - min_bounds).length() * 0.5;

    for level in 0..config.num_levels {
        let resolution = config.resolution_at_level(level);
        let (min_dist, max_dist) = config.distance_range(level);

        let mc_config = MarchingCubesConfig {
            resolution: resolution as usize,
            iso_level: 0.0,
            compute_normals: config.compute_normals,
            ..Default::default()
        };

        let mesh = sdf_to_mesh(sdf, min_bounds, max_bounds, &mc_config);

        let max_error = compute_lod_error(&mesh, resolution, bounds_radius);

        levels.push(LodMesh {
            level,
            resolution,
            mesh,
            max_error,
            screen_threshold: max_error / min_dist.max(0.001),
            min_distance: if level == 0 { 0.0 } else { min_dist },
            max_distance: max_dist,
        });
    }

    LodChain {
        levels,
        bounds_min: min_bounds,
        bounds_max: max_bounds,
        bounding_radius: bounds_radius,
    }
}

/// Compute geometric error for a LOD level
fn compute_lod_error(_mesh: &Mesh, resolution: u32, bounds_radius: f32) -> f32 {
    // Error is proportional to cell size
    let cell_size = (bounds_radius * 2.0) / resolution as f32;
    cell_size * 0.5 // Half cell size as max error estimate
}

/// Screen-space LOD selector
pub struct LodSelector {
    /// Field of view in radians
    pub fov_y: f32,
    /// Screen height in pixels
    pub screen_height: f32,
    /// Pixel error threshold
    pub pixel_threshold: f32,
}

impl Default for LodSelector {
    fn default() -> Self {
        LodSelector {
            fov_y: std::f32::consts::FRAC_PI_3, // 60 degrees
            screen_height: 1080.0,
            pixel_threshold: 1.0, // 1 pixel error threshold
        }
    }
}

impl LodSelector {
    /// Create selector for 4K display
    pub fn high_res() -> Self {
        LodSelector {
            fov_y: std::f32::consts::FRAC_PI_3,
            screen_height: 2160.0,
            pixel_threshold: 1.0,
        }
    }

    /// Compute screen-space error for a geometric error at distance
    pub fn screen_error(&self, geometric_error: f32, distance: f32) -> f32 {
        let proj_factor = self.screen_height / (2.0 * (self.fov_y * 0.5).tan());
        (geometric_error / distance.max(0.001)) * proj_factor
    }

    /// Check if LOD is acceptable at distance
    pub fn is_acceptable(&self, lod: &LodMesh, distance: f32) -> bool {
        let screen_err = self.screen_error(lod.max_error, distance);
        screen_err <= self.pixel_threshold
    }

    /// Select best LOD from chain
    pub fn select<'a>(&self, chain: &'a LodChain, distance: f32) -> Option<&'a LodMesh> {
        // Start from lowest detail and find first acceptable
        for lod in chain.levels.iter().rev() {
            if self.is_acceptable(lod, distance) {
                return Some(lod);
            }
        }
        chain.levels.first() // Fall back to highest detail
    }
}

/// Continuous LOD blending support
pub struct ContinuousLod {
    /// LOD chain
    chain: LodChain,
    /// Current LOD level (can be fractional for blending)
    current_lod: f32,
    /// Transition speed
    transition_speed: f32,
}

impl ContinuousLod {
    /// Create from LOD chain
    pub fn new(chain: LodChain, transition_speed: f32) -> Self {
        ContinuousLod {
            chain,
            current_lod: 0.0,
            transition_speed,
        }
    }

    /// Update LOD based on distance
    pub fn update(&mut self, distance: f32, delta_time: f32) {
        let target_lod = self.compute_target_lod(distance);
        let diff = target_lod - self.current_lod;

        if diff.abs() > 0.01 {
            self.current_lod += diff.signum() * self.transition_speed * delta_time;
            self.current_lod = self
                .current_lod
                .clamp(0.0, (self.chain.levels.len() - 1) as f32);
        }
    }

    /// Compute target LOD level for distance
    fn compute_target_lod(&self, distance: f32) -> f32 {
        for (i, lod) in self.chain.levels.iter().enumerate() {
            if lod.is_active(distance) {
                return i as f32 + lod.blend_factor(distance);
            }
        }
        (self.chain.levels.len() - 1) as f32
    }

    /// Get current LOD meshes for rendering (for blending)
    pub fn get_render_meshes(&self) -> (&Mesh, Option<(&Mesh, f32)>) {
        let base_level = self.current_lod.floor() as usize;
        let blend = self.current_lod.fract();

        let base_mesh = &self.chain.levels[base_level].mesh;

        if blend > 0.01 && base_level + 1 < self.chain.levels.len() {
            let next_mesh = &self.chain.levels[base_level + 1].mesh;
            (base_mesh, Some((next_mesh, blend)))
        } else {
            (base_mesh, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lod_config() {
        let config = LodConfig::default();

        assert_eq!(config.resolution_at_level(0), 64);
        assert_eq!(config.resolution_at_level(1), 32);
        assert_eq!(config.resolution_at_level(2), 16);

        let (min0, max0) = config.distance_range(0);
        let (min1, _max1) = config.distance_range(1);

        assert_eq!(min0, 0.0); // LOD 0 starts at 0
        assert_eq!(min1, max0); // LOD 1 starts where LOD 0 ends
    }

    #[test]
    fn test_generate_lod_chain() {
        let sphere = SdfNode::sphere(1.0);
        let config = LodConfig::fast();

        let chain = generate_lod_chain(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        assert_eq!(chain.levels.len(), config.num_levels as usize);

        // Higher LOD levels should have fewer triangles
        let tri_counts: Vec<usize> = chain
            .levels
            .iter()
            .map(|l| l.mesh.triangle_count())
            .collect();

        for i in 1..tri_counts.len() {
            assert!(
                tri_counts[i] <= tri_counts[i - 1] || tri_counts[i - 1] == 0,
                "LOD {} has more triangles than LOD {}",
                i,
                i - 1
            );
        }
    }

    #[test]
    fn test_lod_selection() {
        let sphere = SdfNode::sphere(1.0);
        let config = LodConfig::default();

        let chain = generate_lod_chain(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // Close distance should select LOD 0
        let close_lod = chain.get_lod(0.5);
        assert!(close_lod.is_some());
        assert_eq!(close_lod.unwrap().level, 0);

        // Far distance should select higher LOD
        let far_lod = chain.get_lod(100.0);
        assert!(far_lod.is_some());
        assert!(far_lod.unwrap().level > 0);
    }

    #[test]
    fn test_screen_error_selector() {
        let selector = LodSelector::default();

        let error = selector.screen_error(0.1, 10.0);
        assert!(error > 0.0);

        // Closer distance should give larger screen error
        let close_error = selector.screen_error(0.1, 1.0);
        assert!(close_error > error);
    }

    #[test]
    fn test_decimation_lod_chain() {
        let sphere = SdfNode::sphere(1.0);
        let config = DecimationLodConfig::fast();

        let chain =
            generate_lod_chain_decimated(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        assert_eq!(chain.levels.len(), config.num_levels as usize);

        // Higher LOD levels should have fewer triangles (progressive decimation)
        for i in 1..chain.levels.len() {
            let prev = chain.levels[i - 1].mesh.triangle_count();
            let curr = chain.levels[i].mesh.triangle_count();
            assert!(
                curr <= prev,
                "LOD {} ({} tris) should have <= LOD {} ({} tris)",
                i,
                curr,
                i - 1,
                prev
            );
        }

        // LOD 0 should have the most triangles
        let lod0_tris = chain.levels[0].mesh.triangle_count();
        let last_lod_tris = chain.levels.last().unwrap().mesh.triangle_count();
        assert!(
            last_lod_tris < lod0_tris,
            "Last LOD ({} tris) should be less than LOD 0 ({} tris)",
            last_lod_tris,
            lod0_tris
        );
    }

    #[test]
    fn test_decimation_lod_vs_resolution_lod() {
        let sphere = SdfNode::sphere(1.0);

        // Resolution-based LOD
        let res_config = LodConfig::fast();
        let res_chain =
            generate_lod_chain(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &res_config);

        // Decimation-based LOD
        let dec_config = DecimationLodConfig::fast();
        let dec_chain =
            generate_lod_chain_decimated(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &dec_config);

        // Both should produce valid LOD chains
        assert!(!res_chain.levels.is_empty());
        assert!(!dec_chain.levels.is_empty());

        // Both LOD 0 should have meshes
        assert!(res_chain.levels[0].mesh.triangle_count() > 0);
        assert!(dec_chain.levels[0].mesh.triangle_count() > 0);
    }

    #[test]
    fn test_continuous_lod() {
        let sphere = SdfNode::sphere(1.0);
        let config = LodConfig::fast();

        let chain = generate_lod_chain(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let mut clod = ContinuousLod::new(chain, 2.0);

        // Update towards far distance
        for _ in 0..10 {
            clod.update(100.0, 0.1);
        }

        // Should have moved towards higher LOD
        assert!(clod.current_lod > 0.0);

        let (base, blend) = clod.get_render_meshes();
        assert!(base.triangle_count() > 0 || blend.is_some());
    }
}
