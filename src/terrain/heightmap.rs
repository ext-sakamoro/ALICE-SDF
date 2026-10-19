//! Heightmap: 2D height grid with interpolation and procedural generation (Deep Fried Edition)
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::LawMinMax;
#[cfg(feature = "image")]
use crate::io::IoError;
#[cfg(feature = "image")]
use crate::provenance::{image_format, import_verdict, Checked, Provenance};
use glam::Vec3;

/// 2D heightmap storing terrain elevation data
///
/// Heights are stored in a flat array indexed as `[x + z * width]`.
/// World-space coordinates map to the grid using bilinear or bicubic
/// interpolation.
pub struct Heightmap {
    /// Height values (flat, `x + z * width`)
    pub heights: Vec<f32>,
    /// Grid width (number of samples in X)
    pub width: u32,
    /// Grid depth (number of samples in Z)
    pub depth: u32,
    /// World-space width (X extent)
    pub world_width: f32,
    /// World-space depth (Z extent)
    pub world_depth: f32,
    /// Precomputed reciprocal of world_width
    inv_world_width: f32,
    /// Precomputed reciprocal of world_depth
    inv_world_depth: f32,
}

impl Heightmap {
    /// Create a flat heightmap (all zeros)
    pub fn new(width: u32, depth: u32, world_width: f32, world_depth: f32) -> Self {
        Self {
            heights: vec![0.0; (width * depth) as usize],
            width,
            depth,
            world_width,
            world_depth,
            inv_world_width: 1.0 / world_width,
            inv_world_depth: 1.0 / world_depth,
        }
    }

    /// Create from existing height data
    pub fn from_data(
        heights: Vec<f32>,
        width: u32,
        depth: u32,
        world_width: f32,
        world_depth: f32,
    ) -> Self {
        assert_eq!(heights.len(), (width * depth) as usize);
        Self {
            heights,
            width,
            depth,
            world_width,
            world_depth,
            inv_world_width: 1.0 / world_width,
            inv_world_depth: 1.0 / world_depth,
        }
    }

    /// Get height at grid coordinates
    #[inline]
    pub fn get_height(&self, x: u32, z: u32) -> f32 {
        let x = x.min(self.width - 1);
        let z = z.min(self.depth - 1);
        self.heights[(x + z * self.width) as usize]
    }

    /// Set height at grid coordinates
    #[inline]
    pub fn set_height(&mut self, x: u32, z: u32, h: f32) {
        if x < self.width && z < self.depth {
            self.heights[(x + z * self.width) as usize] = h;
        }
    }

    /// Sample height at world coordinates using bilinear interpolation
    pub fn sample(&self, world_x: f32, world_z: f32) -> f32 {
        let fx = (world_x * self.inv_world_width * self.width as f32)
            .clamp(0.0, (self.width - 1) as f32);
        let fz = (world_z * self.inv_world_depth * self.depth as f32)
            .clamp(0.0, (self.depth - 1) as f32);

        let x0 = fx.floor() as u32;
        let z0 = fz.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let z1 = (z0 + 1).min(self.depth - 1);

        let tx = fx - fx.floor();
        let tz = fz - fz.floor();

        let h00 = self.get_height(x0, z0);
        let h10 = self.get_height(x1, z0);
        let h01 = self.get_height(x0, z1);
        let h11 = self.get_height(x1, z1);

        let h0 = (h10 - h00) * tx + h00;
        let h1 = (h11 - h01) * tx + h01;

        (h1 - h0) * tz + h0
    }

    /// Sample height using bicubic interpolation (smoother)
    pub fn sample_bicubic(&self, world_x: f32, world_z: f32) -> f32 {
        let fx = (world_x * self.inv_world_width * self.width as f32)
            .clamp(1.0, (self.width - 2) as f32);
        let fz = (world_z * self.inv_world_depth * self.depth as f32)
            .clamp(1.0, (self.depth - 2) as f32);

        let ix = fx.floor() as i32;
        let iz = fz.floor() as i32;
        let tx = fx - fx.floor();
        let tz = fz - fz.floor();

        let mut result = 0.0;
        for dz in -1..=2i32 {
            let z = (iz + dz).clamp(0, self.depth as i32 - 1) as u32;
            let wz = cubic_weight(tz - dz as f32);
            for dx in -1..=2i32 {
                let x = (ix + dx).clamp(0, self.width as i32 - 1) as u32;
                let wx = cubic_weight(tx - dx as f32);
                result += (self.get_height(x, z) * wx) * wz;
            }
        }
        result
    }

    /// Compute normal at world coordinates
    pub fn normal_at(&self, world_x: f32, world_z: f32) -> Vec3 {
        let eps_x = self.world_width * (1.0 / self.width as f32);
        let eps_z = self.world_depth * (1.0 / self.depth as f32);

        let hx0 = self.sample(world_x - eps_x, world_z);
        let hx1 = self.sample(world_x + eps_x, world_z);
        let hz0 = self.sample(world_x, world_z - eps_z);
        let hz1 = self.sample(world_x, world_z + eps_z);

        let inv_2_eps_x = 0.5 * self.width as f32 * self.inv_world_width;
        let inv_2_eps_z = 0.5 * self.depth as f32 * self.inv_world_depth;
        let dx = (hx1 - hx0) * inv_2_eps_x;
        let dz = (hz1 - hz0) * inv_2_eps_z;

        Vec3::new(-dx, 1.0, -dz).normalize()
    }

    /// Generate fractal Brownian motion (fBm) terrain
    ///
    /// Layered simplex-like noise using a hash-based approach.
    pub fn generate_fbm(&mut self, octaves: u32, persistence: f32, lacunarity: f32, seed: u64) {
        let scale = 1.0 / self.width.max(self.depth) as f32;

        for z in 0..self.depth {
            for x in 0..self.width {
                let wx = x as f32 * scale;
                let wz = z as f32 * scale;

                let mut value = 0.0f32;
                let mut amplitude = 1.0f32;
                let mut frequency = 1.0f32;
                let mut max_amp = 0.0f32;

                for oct in 0..octaves {
                    let nx = wx * frequency + ((oct as f32) * 31.7 + (seed as f32));
                    let nz = wz * frequency + ((seed as f32) * 0.7 + (oct as f32 * 17.3));
                    let n = hash_noise_2d(nx, nz);
                    value += n * amplitude;
                    max_amp += amplitude;
                    amplitude *= persistence;
                    frequency *= lacunarity;
                }

                let normalized = value / max_amp; // [-1, 1] range
                self.set_height(x, z, normalized);
            }
        }
    }

    /// Compute min and max height values
    pub fn height_range(&self) -> (f32, f32) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &h in &self.heights {
            min = min.law_min(h);
            max = max.law_max(h);
        }
        (min, max)
    }

    /// Normalize heights to [0, 1] range
    pub fn normalize(&mut self) {
        let (min, max) = self.height_range();
        let range = max - min;
        if range > 1e-6 {
            for h in &mut self.heights {
                *h = (*h - min) / range;
            }
        }
    }

    /// Scale all heights by a factor
    pub fn scale_heights(&mut self, factor: f32) {
        for h in &mut self.heights {
            *h *= factor;
        }
    }

    /// Total number of height samples
    #[inline]
    pub const fn sample_count(&self) -> usize {
        (self.width * self.depth) as usize
    }

    /// Load a heightmap from an image file (PNG, JPEG), with what is known about the arithmetic
    /// it was written under
    ///
    /// As [`Heightmap::from_image_bytes`], on the bytes of the file at `path`.
    ///
    /// Requires the `image` feature to be enabled.
    ///
    /// # Errors
    ///
    /// [`IoError::Io`] when the file cannot be read, and the errors of
    /// [`Heightmap::from_image_bytes`].
    #[cfg(feature = "image")]
    pub fn from_image(
        path: impl AsRef<std::path::Path>,
        config: &HeightmapImageConfig,
    ) -> Result<Checked<Self>, IoError> {
        Self::from_image_bytes(&std::fs::read(path.as_ref())?, config)
    }

    /// As [`Heightmap::from_image`], refusing an image without a semantics identifier
    ///
    /// # Errors
    ///
    /// As [`Heightmap::from_image_bytes_pinned`].
    #[cfg(feature = "image")]
    pub fn from_image_pinned(
        path: impl AsRef<std::path::Path>,
        config: &HeightmapImageConfig,
    ) -> Result<Self, IoError> {
        Self::from_image_bytes_pinned(&std::fs::read(path.as_ref())?, config)
    }

    /// Load a heightmap from in-memory image bytes (PNG, JPEG), with what is known about the
    /// arithmetic it was written under
    ///
    /// The identifier is read from the same bytes before the pixels are decoded: a PNG `tEXt`
    /// chunk or a JPEG comment `alice:semantics_id` (see [`crate::provenance::from_bytes`]), as
    /// [`Heightmap::to_png_with_provenance`] writes it. An image without one (a paint tool's) is
    /// read and marked [`crate::provenance::StoredSemantics::Unpinned`]; this crate's identifier
    /// gives `Current`, an earlier one it reproduces `VerifiedPrevious`.
    ///
    /// An 8-bit image maps a grey level `p` to `p · height_scale / 255`, a 16-bit one to
    /// `p · height_scale / 65535` (colour images are converted to grey first).
    ///
    /// Requires the `image` feature to be enabled.
    ///
    /// # Errors
    ///
    /// [`IoError::SemanticsMismatch`] (naming both identifiers) for an image stamped under another
    /// identifier, [`IoError::InvalidFormat`] when the identifier is present but malformed or the
    /// PNG / JPEG structure cannot be walked, or when the pixels cannot be decoded.
    #[cfg(feature = "image")]
    pub fn from_image_bytes(
        bytes: &[u8],
        config: &HeightmapImageConfig,
    ) -> Result<Checked<Self>, IoError> {
        let semantics = import_verdict(bytes, image_format(bytes), true)?;
        Ok(Checked::new(Self::decode_image(bytes, config)?, semantics))
    }

    /// As [`Heightmap::from_image_bytes`], refusing an image without a semantics identifier
    ///
    /// # Errors
    ///
    /// [`IoError::Unpinned`] for an image without an identifier, and the errors of
    /// [`Heightmap::from_image_bytes`].
    #[cfg(feature = "image")]
    pub fn from_image_bytes_pinned(
        bytes: &[u8],
        config: &HeightmapImageConfig,
    ) -> Result<Self, IoError> {
        import_verdict(bytes, image_format(bytes), false)?;
        Self::decode_image(bytes, config)
    }

    /// Decodes the pixels of an image the caller has judged
    #[cfg(feature = "image")]
    fn decode_image(bytes: &[u8], config: &HeightmapImageConfig) -> Result<Self, IoError> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| IoError::InvalidFormat(format!("image: {e}")))?;
        let sixteen = matches!(
            img,
            image::DynamicImage::ImageLuma16(_)
                | image::DynamicImage::ImageLumaA16(_)
                | image::DynamicImage::ImageRgb16(_)
                | image::DynamicImage::ImageRgba16(_)
        );
        let (width, height, heights) = if sixteen {
            let gray = img.to_luma16();
            let scale = config.height_scale / 65535.0;
            let heights = gray
                .as_raw()
                .iter()
                .map(|&p| f32::from(p) * scale)
                .collect();
            (gray.width(), gray.height(), heights)
        } else {
            let gray = img.to_luma8();
            let scale = config.height_scale / 255.0;
            let heights = gray
                .as_raw()
                .iter()
                .map(|&p| f32::from(p) * scale)
                .collect();
            (gray.width(), gray.height(), heights)
        };
        Ok(Self::from_grey_heights(heights, width, height, config))
    }

    /// Encodes the heightmap as a 16-bit greyscale PNG under this crate's semantics identifier
    ///
    /// As [`Heightmap::to_png_with_provenance`] with [`Provenance::sdf`].
    ///
    /// # Errors
    ///
    /// As [`Heightmap::to_png_with_provenance`].
    #[cfg(feature = "image")]
    pub fn to_png(&self, config: &HeightmapImageConfig) -> Result<Vec<u8>, IoError> {
        self.to_png_with_provenance(config, &Provenance::sdf())
    }

    /// Encodes the heightmap as a 16-bit greyscale PNG, storing `provenance` in a `tEXt` chunk
    /// `alice:semantics_id` before the image data
    ///
    /// A height `h` is stored as the grey level `round(h · 65535 / height_scale)`, clamped to
    /// `0..=65535` (a NaN height is stored as 0), so [`Heightmap::from_image_bytes`] with the same
    /// `height_scale` reads back `p · height_scale / 65535`: heights read from such a PNG are
    /// written back to the same bytes. The world size is not stored (it comes from the
    /// config when the image is read).
    ///
    /// # Errors
    ///
    /// [`IoError::Serialization`] when the PNG encoder fails (a zero width or depth).
    #[cfg(feature = "image")]
    pub fn to_png_with_provenance(
        &self,
        config: &HeightmapImageConfig,
        provenance: &Provenance,
    ) -> Result<Vec<u8>, IoError> {
        let to_level = 65535.0 / config.height_scale;
        let mut data = Vec::with_capacity(self.heights.len() * 2);
        for &h in &self.heights {
            // `as` saturates, and maps NaN to 0
            let level = (h * to_level).round().clamp(0.0, 65535.0) as u16;
            data.extend_from_slice(&level.to_be_bytes());
        }
        let encode_error = |e: png::EncodingError| IoError::Serialization(format!("PNG: {e}"));
        let mut out = Vec::new();
        let mut encoder = png::Encoder::new(&mut out, self.width, self.depth);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Sixteen);
        encoder
            .add_text_chunk(crate::provenance::KEY.to_string(), provenance.hex())
            .map_err(encode_error)?;
        let mut writer = encoder.write_header().map_err(encode_error)?;
        writer.write_image_data(&data).map_err(encode_error)?;
        writer.finish().map_err(encode_error)?;
        Ok(out)
    }

    /// Build a heightmap from the heights of a greyscale image
    #[cfg(feature = "image")]
    fn from_grey_heights(
        heights: Vec<f32>,
        width: u32,
        height: u32,
        config: &HeightmapImageConfig,
    ) -> Self {
        Self {
            heights,
            width,
            depth: height,
            world_width: config.world_width.unwrap_or(width as f32),
            world_depth: config.world_depth.unwrap_or(height as f32),
            inv_world_width: 1.0 / config.world_width.unwrap_or(width as f32),
            inv_world_depth: 1.0 / config.world_depth.unwrap_or(height as f32),
        }
    }
}

/// Configuration for loading a heightmap from an image
#[cfg(feature = "image")]
#[derive(Debug, Clone)]
pub struct HeightmapImageConfig {
    /// Maximum height value (luminance 255 maps to this). Default: 100.0
    pub height_scale: f32,
    /// World-space width. If None, uses image width in pixels.
    pub world_width: Option<f32>,
    /// World-space depth. If None, uses image height in pixels.
    pub world_depth: Option<f32>,
}

#[cfg(feature = "image")]
impl Default for HeightmapImageConfig {
    fn default() -> Self {
        Self {
            height_scale: 100.0,
            world_width: None,
            world_depth: None,
        }
    }
}

#[cfg(feature = "image")]
impl HeightmapImageConfig {
    /// Create config with custom height scale and world dimensions
    pub const fn new(height_scale: f32, world_width: f32, world_depth: f32) -> Self {
        Self {
            height_scale,
            world_width: Some(world_width),
            world_depth: Some(world_depth),
        }
    }
}

/// Catmull-Rom cubic interpolation weight
#[inline]
fn cubic_weight(t: f32) -> f32 {
    let t = t.abs();
    if t <= 1.0 {
        ((1.5f32 * t + -2.5) * t) * t + 1.0
    } else if t <= 2.0 {
        (((-0.5f32) * t + 2.5) * t + -4.0) * t + 2.0
    } else {
        0.0
    }
}

/// 2D hash noise (value noise)
fn hash_noise_2d(x: f32, z: f32) -> f32 {
    let ix = x.floor() as i32;
    let iz = z.floor() as i32;
    let fx = x - x.floor();
    let fz = z - z.floor();

    // Smoothstep
    let sx = fx * fx * (2.0f32 * -fx + 3.0);
    let sz = fz * fz * (2.0f32 * -fz + 3.0);

    let h00 = hash_2d(ix, iz);
    let h10 = hash_2d(ix + 1, iz);
    let h01 = hash_2d(ix, iz + 1);
    let h11 = hash_2d(ix + 1, iz + 1);

    let h0 = (h10 - h00) * sx + h00;
    let h1 = (h11 - h01) * sx + h01;

    (h1 - h0) * sz + h0
}

/// Integer hash -> float in [-1, 1]
#[inline]
fn hash_2d(x: i32, z: i32) -> f32 {
    let mut h = (x as u32)
        .wrapping_mul(374761393)
        .wrapping_add((z as u32).wrapping_mul(668265263));
    h = (h ^ (h >> 13)).wrapping_mul(1103515245);
    h = h ^ (h >> 16);
    (h as f32) / (u32::MAX as f32 / 2.0) - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heightmap_new() {
        let hm = Heightmap::new(16, 16, 100.0, 100.0);
        assert_eq!(hm.sample_count(), 256);
        assert_eq!(hm.get_height(0, 0), 0.0);
    }

    #[test]
    fn test_heightmap_set_get() {
        let mut hm = Heightmap::new(8, 8, 10.0, 10.0);
        hm.set_height(3, 4, 5.0);
        assert_eq!(hm.get_height(3, 4), 5.0);
    }

    #[test]
    fn test_bilinear_sample() {
        let mut hm = Heightmap::new(4, 4, 4.0, 4.0);
        hm.set_height(0, 0, 0.0);
        hm.set_height(1, 0, 10.0);
        hm.set_height(0, 1, 0.0);
        hm.set_height(1, 1, 10.0);

        // oracle: the bilinear closed form is EXACT at the midpoint of 0 and
        // 10, so the tolerance is f32 rounding. It used to be 1.0 (20%), which
        // accepts an implementation returning 4.1.
        let h = hm.sample(0.5, 0.0);
        assert!((h - 5.0).abs() < 1e-5, "Expected exactly 5.0, got {}", h);
    }

    #[test]
    fn test_normal_at() {
        let hm = Heightmap::new(8, 8, 8.0, 8.0);
        // oracle: on a flat field both finite differences are exactly zero, so
        // the normal is exactly +Y (was 0.1, which accepts a 25 degree tilt)
        let n = hm.normal_at(4.0, 4.0);
        assert!(
            (n - Vec3::Y).length() < 1e-6,
            "Flat terrain normal must be exactly (0,1,0), got {:?}",
            n
        );
    }

    #[test]
    fn test_generate_fbm() {
        let mut hm = Heightmap::new(32, 32, 100.0, 100.0);
        hm.generate_fbm(4, 0.5, 2.0, 42);

        let (min, max) = hm.height_range();
        assert!(max > min, "fBm should produce variation");
        assert!(
            min >= -1.0 && max <= 1.0,
            "fBm should be in [-1,1], got [{}, {}]",
            min,
            max
        );
    }

    #[test]
    fn test_normalize() {
        let mut hm = Heightmap::new(8, 8, 10.0, 10.0);
        hm.generate_fbm(3, 0.5, 2.0, 123);
        hm.normalize();

        let (min, max) = hm.height_range();
        assert!((min - 0.0).abs() < 0.001);
        assert!((max - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bicubic_sample() {
        let hm = {
            let mut h = Heightmap::new(8, 8, 8.0, 8.0);
            for z in 0..8 {
                for x in 0..8 {
                    h.set_height(x, z, f32::midpoint(x as f32, z as f32));
                }
            }
            h
        };

        // oracle: h = (x + z)/2 is linear and Catmull-Rom reproduces linear
        // fields exactly, so at the grid node (4,4) the answer is exactly 4.0
        // (was 1.0, which accepts 3.1)
        let h = hm.sample_bicubic(4.0, 4.0);
        assert!((h - 4.0).abs() < 1e-5, "Expected exactly 4.0, got {}", h);
    }

    #[test]
    fn test_height_range() {
        let mut hm = Heightmap::new(4, 4, 4.0, 4.0);
        hm.set_height(0, 0, -5.0);
        hm.set_height(1, 1, 10.0);
        let (min, max) = hm.height_range();
        assert_eq!(min, -5.0);
        assert_eq!(max, 10.0);
    }

    #[cfg(feature = "image")]
    #[test]
    fn test_from_image_bytes() {
        // Minimal 2x2 grayscale PNG (hand-crafted)
        // Use image crate to create in-memory PNG
        use image::{GrayImage, Luma};

        let mut img = GrayImage::new(4, 4);
        // Set corners: TL=0, TR=128, BL=64, BR=255
        img.put_pixel(0, 0, Luma([0u8]));
        img.put_pixel(3, 0, Luma([128u8]));
        img.put_pixel(0, 3, Luma([64u8]));
        img.put_pixel(3, 3, Luma([255u8]));

        // Encode to PNG bytes
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let png_bytes = buf.into_inner();

        let config = super::HeightmapImageConfig {
            height_scale: 100.0,
            world_width: Some(400.0),
            world_depth: Some(400.0),
        };
        let (hm, status) = Heightmap::from_image_bytes(&png_bytes, &config)
            .unwrap()
            .into_parts();
        assert_eq!(status, crate::provenance::StoredSemantics::Unpinned);

        assert_eq!(hm.width, 4);
        assert_eq!(hm.depth, 4);
        assert_eq!(hm.sample_count(), 16);
        assert!((hm.world_width - 400.0).abs() < 0.001);

        // Top-left should be 0
        assert!((hm.get_height(0, 0) - 0.0).abs() < 0.01);
        // Bottom-right should be 100.0
        assert!((hm.get_height(3, 3) - 100.0).abs() < 0.01);
    }
}
