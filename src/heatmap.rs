//! Distance field cross-section heatmap visualization
//!
//! Generates 2D cross-section slices of an SDF tree along arbitrary planes,
//! producing distance maps suitable for visualization as color images.
//!
//! Author: Moroya Sakamoto

use crate::eval::eval;
use crate::types::SdfNode;
use glam::Vec3;

// ── Slice Plane ──────────────────────────────────────────────

/// Defines the slicing plane for cross-section generation.
#[derive(Debug, Clone, Copy)]
pub enum SlicePlane {
    /// XY plane at the given Z coordinate.
    XY(f32),
    /// XZ plane at the given Y coordinate.
    XZ(f32),
    /// YZ plane at the given X coordinate.
    YZ(f32),
    /// Arbitrary plane defined by origin and normal.
    Custom {
        /// A point on the plane.
        origin: Vec3,
        /// Plane normal (will be normalized internally).
        normal: Vec3,
    },
}

// ── Heatmap Config ───────────────────────────────────────────

/// Configuration for heatmap generation.
#[derive(Debug, Clone)]
pub struct HeatmapConfig {
    /// Resolution in pixels (width = height = resolution).
    pub resolution: u32,
    /// Half-extent of the slice region in world units.
    pub range: f32,
    /// Which plane to slice.
    pub plane: SlicePlane,
}

impl Default for HeatmapConfig {
    fn default() -> Self {
        Self {
            resolution: 256,
            range: 2.0,
            plane: SlicePlane::XY(0.0),
        }
    }
}

// ── Heatmap ──────────────────────────────────────────────────

/// A 2D grid of distance values from a cross-section slice.
#[derive(Debug, Clone)]
pub struct Heatmap {
    /// Distance values in row-major order (width * height).
    pub pixels: Vec<f32>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Minimum distance value in the grid.
    pub min_val: f32,
    /// Maximum distance value in the grid.
    pub max_val: f32,
}

/// Color map for converting distance values to RGBA.
#[derive(Debug, Clone, Copy)]
pub enum ColorMap {
    /// Blue (inside) → white (surface) → red (outside).
    Coolwarm,
    /// Black (inside) → white (surface) → black (outside, dimmer).
    Binary,
    /// Perceptually uniform blue → green → yellow.
    Viridis,
    /// Perceptually uniform dark → hot pink → yellow.
    Magma,
}

// ── Generation ───────────────────────────────────────────────

/// Generate a distance-field heatmap by slicing an SDF tree.
pub fn generate_heatmap(node: &SdfNode, config: &HeatmapConfig) -> Heatmap {
    let res = config.resolution as usize;
    let mut pixels = vec![0.0f32; res * res];
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    let step = 2.0 * config.range / config.resolution as f32;

    for iy in 0..res {
        for ix in 0..res {
            let u = -config.range + (ix as f32 + 0.5) * step;
            let v = -config.range + (iy as f32 + 0.5) * step;

            let point = plane_to_world(config.plane, u, v);
            let d = eval(node, point);

            pixels[iy * res + ix] = d;
            if d < min_val {
                min_val = d;
            }
            if d > max_val {
                max_val = d;
            }
        }
    }

    Heatmap {
        pixels,
        width: config.resolution,
        height: config.resolution,
        min_val,
        max_val,
    }
}

/// Convert (u, v) coordinates on a slice plane to a 3D world point.
fn plane_to_world(plane: SlicePlane, u: f32, v: f32) -> Vec3 {
    match plane {
        SlicePlane::XY(z) => Vec3::new(u, v, z),
        SlicePlane::XZ(y) => Vec3::new(u, y, v),
        SlicePlane::YZ(x) => Vec3::new(x, u, v),
        SlicePlane::Custom { origin, normal } => {
            let n = normal.normalize();
            // Build orthonormal basis
            let up = if n.y.abs() < 0.99 { Vec3::Y } else { Vec3::X };
            let right = up.cross(n).normalize();
            let forward = n.cross(right);
            origin + right * u + forward * v
        }
    }
}

/// Convert a heatmap to RGBA pixel data using the specified color map.
pub fn heatmap_to_rgba(heatmap: &Heatmap, colormap: ColorMap) -> Vec<[u8; 4]> {
    let scale = heatmap.max_val.abs().max(heatmap.min_val.abs()).max(1e-6);
    heatmap
        .pixels
        .iter()
        .map(|&d| {
            let t = (d / scale).clamp(-1.0, 1.0);
            distance_to_color(t, colormap)
        })
        .collect()
}

/// Map a normalized distance [-1, 1] to an RGBA color.
fn distance_to_color(t: f32, colormap: ColorMap) -> [u8; 4] {
    match colormap {
        ColorMap::Coolwarm => {
            if t >= 0.0 {
                // Surface → outside: white → red
                let r = 255;
                let g = (255.0 * (1.0 - t)) as u8;
                let b = (255.0 * (1.0 - t)) as u8;
                [r, g, b, 255]
            } else {
                // Inside → surface: blue → white
                let s = -t;
                let r = (255.0 * (1.0 - s)) as u8;
                let g = (255.0 * (1.0 - s)) as u8;
                let b = 255;
                [r, g, b, 255]
            }
        }
        ColorMap::Binary => {
            let v = (255.0 * (1.0 - t.abs())) as u8;
            [v, v, v, 255]
        }
        ColorMap::Viridis => {
            let s = (t + 1.0) * 0.5; // [0, 1]
            let r = (68.0 + s * (187.0)) as u8;
            let g = (1.0 + s * (254.0)) as u8;
            let b = (84.0 + s * (80.0) - s * s * 100.0).clamp(0.0, 255.0) as u8;
            [r, g, b, 255]
        }
        ColorMap::Magma => {
            let s = (t + 1.0) * 0.5; // [0, 1]
            let r = (s * 255.0).min(255.0) as u8;
            let g = (s * s * 200.0).min(255.0) as u8;
            let b = (80.0 + s * 100.0).min(255.0) as u8;
            [r, g, b, 255]
        }
    }
}

impl Heatmap {
    /// Sample distance at pixel coordinates.
    #[inline]
    pub fn sample(&self, x: u32, y: u32) -> f32 {
        if x < self.width && y < self.height {
            self.pixels[(y * self.width + x) as usize]
        } else {
            f32::MAX
        }
    }

    /// Count pixels where the SDF is negative (inside).
    pub fn inside_pixel_count(&self) -> u32 {
        self.pixels.iter().filter(|&&d| d < 0.0).count() as u32
    }

    /// Count pixels within epsilon of the surface.
    pub fn surface_pixel_count(&self, epsilon: f32) -> u32 {
        self.pixels.iter().filter(|&&d| d.abs() < epsilon).count() as u32
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_xy_center_negative() {
        let sphere = SdfNode::sphere(1.0);
        let config = HeatmapConfig {
            resolution: 32,
            range: 2.0,
            plane: SlicePlane::XY(0.0),
        };
        let hm = generate_heatmap(&sphere, &config);
        // Center pixel should be negative (inside sphere)
        let center = hm.sample(16, 16);
        assert!(
            center < 0.0,
            "Center should be inside sphere, got {}",
            center
        );
    }

    #[test]
    fn sphere_xy_corner_positive() {
        let sphere = SdfNode::sphere(1.0);
        let config = HeatmapConfig {
            resolution: 32,
            range: 2.0,
            plane: SlicePlane::XY(0.0),
        };
        let hm = generate_heatmap(&sphere, &config);
        // Corner pixel should be positive (outside sphere)
        let corner = hm.sample(0, 0);
        assert!(
            corner > 0.0,
            "Corner should be outside sphere, got {}",
            corner
        );
    }

    #[test]
    fn heatmap_resolution() {
        let sphere = SdfNode::sphere(1.0);
        let config = HeatmapConfig {
            resolution: 64,
            range: 2.0,
            plane: SlicePlane::XY(0.0),
        };
        let hm = generate_heatmap(&sphere, &config);
        assert_eq!(hm.width, 64);
        assert_eq!(hm.height, 64);
        assert_eq!(hm.pixels.len(), 64 * 64);
    }

    #[test]
    fn heatmap_min_max() {
        let sphere = SdfNode::sphere(1.0);
        let config = HeatmapConfig {
            resolution: 32,
            range: 2.0,
            plane: SlicePlane::XY(0.0),
        };
        let hm = generate_heatmap(&sphere, &config);
        assert!(hm.min_val < 0.0);
        assert!(hm.max_val > 0.0);
    }

    #[test]
    fn heatmap_xz_plane() {
        let sphere = SdfNode::sphere(1.0);
        let config = HeatmapConfig {
            resolution: 16,
            range: 2.0,
            plane: SlicePlane::XZ(0.0),
        };
        let hm = generate_heatmap(&sphere, &config);
        let center = hm.sample(8, 8);
        assert!(center < 0.0, "XZ center should be inside");
    }

    #[test]
    fn heatmap_yz_plane() {
        let sphere = SdfNode::sphere(1.0);
        let config = HeatmapConfig {
            resolution: 16,
            range: 2.0,
            plane: SlicePlane::YZ(0.0),
        };
        let hm = generate_heatmap(&sphere, &config);
        let center = hm.sample(8, 8);
        assert!(center < 0.0);
    }

    #[test]
    fn heatmap_to_rgba_length() {
        let sphere = SdfNode::sphere(1.0);
        let config = HeatmapConfig {
            resolution: 8,
            range: 2.0,
            plane: SlicePlane::XY(0.0),
        };
        let hm = generate_heatmap(&sphere, &config);
        let rgba = heatmap_to_rgba(&hm, ColorMap::Coolwarm);
        assert_eq!(rgba.len(), 64);
        // All pixels should have alpha = 255
        for px in &rgba {
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn heatmap_inside_pixel_count() {
        let sphere = SdfNode::sphere(1.0);
        let config = HeatmapConfig {
            resolution: 32,
            range: 2.0,
            plane: SlicePlane::XY(0.0),
        };
        let hm = generate_heatmap(&sphere, &config);
        let inside = hm.inside_pixel_count();
        // Circle of radius 1 in a 4x4 grid: area fraction ≈ pi/16 ≈ 0.196
        let total = 32 * 32;
        let ratio = inside as f32 / total as f32;
        assert!(
            ratio > 0.1 && ratio < 0.35,
            "Inside ratio {} outside expected range",
            ratio
        );
    }

    #[test]
    fn custom_plane() {
        let sphere = SdfNode::sphere(1.0);
        let config = HeatmapConfig {
            resolution: 16,
            range: 2.0,
            plane: SlicePlane::Custom {
                origin: Vec3::ZERO,
                normal: Vec3::new(1.0, 1.0, 0.0),
            },
        };
        let hm = generate_heatmap(&sphere, &config);
        // Center should still be inside sphere
        let center = hm.sample(8, 8);
        assert!(
            center < 0.0,
            "Custom plane center should be inside, got {}",
            center
        );
    }

    #[test]
    fn surface_pixel_count() {
        let sphere = SdfNode::sphere(1.0);
        let config = HeatmapConfig {
            resolution: 64,
            range: 2.0,
            plane: SlicePlane::XY(0.0),
        };
        let hm = generate_heatmap(&sphere, &config);
        let on_surface = hm.surface_pixel_count(0.15);
        assert!(on_surface > 0, "Should detect surface pixels");
    }

    #[test]
    fn colormap_binary() {
        let sphere = SdfNode::sphere(1.0);
        let config = HeatmapConfig {
            resolution: 8,
            range: 2.0,
            plane: SlicePlane::XY(0.0),
        };
        let hm = generate_heatmap(&sphere, &config);
        let rgba = heatmap_to_rgba(&hm, ColorMap::Binary);
        assert_eq!(rgba.len(), 64);
    }
}
