//! Splatmap: Multi-layer material blending for terrain (Deep Fried Edition)
//!
//! Supports up to 16 material layers with per-texel blend weights.
//! Used for painting grass, rock, sand, snow etc. onto terrain.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

use super::Heightmap;

/// A single material splat layer
#[derive(Debug, Clone)]
pub struct SplatLayer {
    /// Layer name (e.g. "grass", "rock")
    pub name: String,
    /// Material ID for this layer
    pub material_id: u16,
    /// Blend weights (flat, `x + z * width`)
    pub weights: Vec<f32>,
}

/// Multi-layer splatmap for terrain material blending
pub struct Splatmap {
    /// Material layers (up to 16)
    pub layers: Vec<SplatLayer>,
    /// Map width
    pub width: u32,
    /// Map depth
    pub depth: u32,
}

impl Splatmap {
    /// Create an empty splatmap
    pub fn new(width: u32, depth: u32) -> Self {
        Splatmap {
            layers: Vec::new(),
            width,
            depth,
        }
    }

    /// Add a layer with uniform weight
    pub fn add_layer(&mut self, name: &str, material_id: u16, initial_weight: f32) -> usize {
        let idx = self.layers.len();
        self.layers.push(SplatLayer {
            name: name.to_string(),
            material_id,
            weights: vec![initial_weight; (self.width * self.depth) as usize],
        });
        idx
    }

    /// Get weight of a layer at grid position
    #[inline]
    pub fn get_weight(&self, layer: usize, x: u32, z: u32) -> f32 {
        if layer >= self.layers.len() || x >= self.width || z >= self.depth {
            return 0.0;
        }
        self.layers[layer].weights[(x + z * self.width) as usize]
    }

    /// Set weight of a layer at grid position
    #[inline]
    pub fn set_weight(&mut self, layer: usize, x: u32, z: u32, weight: f32) {
        if layer < self.layers.len() && x < self.width && z < self.depth {
            self.layers[layer].weights[(x + z * self.width) as usize] = weight;
        }
    }

    /// Get the dominant material ID at a position
    pub fn dominant_material(&self, x: u32, z: u32) -> u16 {
        let mut best_weight = 0.0f32;
        let mut best_id = 0u16;

        for layer in &self.layers {
            let idx = (x + z * self.width) as usize;
            if idx < layer.weights.len() && layer.weights[idx] > best_weight {
                best_weight = layer.weights[idx];
                best_id = layer.material_id;
            }
        }

        best_id
    }

    /// Normalize weights so they sum to 1.0 at each texel
    pub fn normalize(&mut self) {
        let total = (self.width * self.depth) as usize;
        for i in 0..total {
            let sum: f32 = self.layers.iter().map(|l| l.weights[i]).sum();
            if sum > 1e-6 {
                for layer in &mut self.layers {
                    layer.weights[i] /= sum;
                }
            }
        }
    }

    /// Auto-splat based on heightmap slope and altitude
    ///
    /// Simple rules:
    /// - Layer 0: Flat areas (slope < threshold)
    /// - Layer 1: Steep areas (slope > threshold)
    /// - Layer 2 (if exists): High altitude
    /// - Layer 3 (if exists): Low altitude
    pub fn auto_splat_from_heightmap(
        &mut self,
        heightmap: &Heightmap,
        slope_threshold: f32,
        altitude_mid: f32,
    ) {
        if self.layers.is_empty() {
            return;
        }

        let has_steep = self.layers.len() > 1;
        let has_high = self.layers.len() > 2;
        let has_low = self.layers.len() > 3;

        for z in 0..self.depth {
            for x in 0..self.width {
                let wx = x as f32 / self.width as f32 * heightmap.world_width;
                let wz = z as f32 / self.depth as f32 * heightmap.world_depth;

                let normal = heightmap.normal_at(wx, wz);
                let slope = 1.0 - normal.y; // 0 = flat, 1 = vertical
                let height = heightmap.sample(wx, wz);

                let idx = (x + z * self.width) as usize;

                // Reset all weights
                for layer in &mut self.layers {
                    layer.weights[idx] = 0.0;
                }

                if has_steep && slope > slope_threshold {
                    // Steep: mostly rock
                    let steep_blend = ((slope - slope_threshold) / (1.0 - slope_threshold)).clamp(0.0, 1.0);
                    self.layers[1].weights[idx] = steep_blend;
                    self.layers[0].weights[idx] = 1.0 - steep_blend;
                } else {
                    self.layers[0].weights[idx] = 1.0;
                }

                // Altitude blending
                if has_high && height > altitude_mid {
                    let alt_blend = ((height - altitude_mid) / altitude_mid).clamp(0.0, 1.0) * 0.5;
                    self.layers[2].weights[idx] = alt_blend;
                    // Reduce other weights
                    for l in 0..2 {
                        self.layers[l].weights[idx] *= 1.0 - alt_blend;
                    }
                }

                if has_low && height < -altitude_mid {
                    let low_blend = ((-height - altitude_mid) / altitude_mid).clamp(0.0, 1.0) * 0.5;
                    self.layers[3].weights[idx] = low_blend;
                    for l in 0..3 {
                        self.layers[l].weights[idx] *= 1.0 - low_blend;
                    }
                }
            }
        }
    }

    /// Number of layers
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splatmap_new() {
        let splatmap = Splatmap::new(16, 16);
        assert_eq!(splatmap.layer_count(), 0);
    }

    #[test]
    fn test_add_layer() {
        let mut splatmap = Splatmap::new(8, 8);
        let idx = splatmap.add_layer("grass", 0, 1.0);
        assert_eq!(idx, 0);
        assert_eq!(splatmap.layer_count(), 1);
        assert_eq!(splatmap.get_weight(0, 4, 4), 1.0);
    }

    #[test]
    fn test_set_weight() {
        let mut splatmap = Splatmap::new(8, 8);
        splatmap.add_layer("grass", 0, 0.0);
        splatmap.set_weight(0, 3, 3, 0.75);
        assert_eq!(splatmap.get_weight(0, 3, 3), 0.75);
    }

    #[test]
    fn test_dominant_material() {
        let mut splatmap = Splatmap::new(8, 8);
        splatmap.add_layer("grass", 0, 0.3);
        splatmap.add_layer("rock", 1, 0.7);

        let dominant = splatmap.dominant_material(4, 4);
        assert_eq!(dominant, 1); // Rock has higher weight
    }

    #[test]
    fn test_normalize() {
        let mut splatmap = Splatmap::new(4, 4);
        splatmap.add_layer("a", 0, 2.0);
        splatmap.add_layer("b", 1, 3.0);
        splatmap.normalize();

        let sum = splatmap.get_weight(0, 0, 0) + splatmap.get_weight(1, 0, 0);
        assert!((sum - 1.0).abs() < 0.001);
        assert!((splatmap.get_weight(0, 0, 0) - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_auto_splat() {
        let mut hm = Heightmap::new(16, 16, 16.0, 16.0);
        hm.generate_fbm(3, 0.5, 2.0, 42);
        hm.scale_heights(10.0);

        let mut splatmap = Splatmap::new(16, 16);
        splatmap.add_layer("grass", 0, 0.0);
        splatmap.add_layer("rock", 1, 0.0);

        splatmap.auto_splat_from_heightmap(&hm, 0.3, 5.0);

        // At least some weights should be non-zero
        let mut any_grass = false;
        for z in 0..16 {
            for x in 0..16 {
                if splatmap.get_weight(0, x, z) > 0.0 { any_grass = true; }
            }
        }
        assert!(any_grass, "Should have some grass");
    }
}
