//! Codec Bridge: ALICE-SDF × ALICE-Codec Integration
//!
//! Evaluate SDF fields on voxel grids and encode the resulting
//! 3D volume data using ALICE-Codec's wavelet compression.
//!
//! # Architecture
//!
//! ```text
//! SdfNode ──→ voxelize_sdf() ──→ SdfVolume ──→ encode_sdf_volume()
//!              (eval_grid)        (f32 data)    ├── f32 → i32 (fixed-point)
//!                                               ├── Wavelet3D forward
//!                                               ├── Quantization (RDO)
//!                                               └── rANS entropy coding
//! ```
//!
//! # Example
//!
//! ```ignore
//! use alice_sdf::prelude::*;
//! use alice_sdf::codec_bridge::{voxelize_sdf, encode_sdf_volume, decode_sdf_volume, EncodeConfig};
//!
//! // Create SDF shape
//! let node = SdfNode::sphere(1.0)
//!     .smooth_union(SdfNode::box3d(0.5, 2.0, 0.5), 0.2);
//!
//! // Voxelize on 64^3 grid
//! let volume = voxelize_sdf(
//!     &node,
//!     Vec3::splat(-2.0),
//!     Vec3::splat(2.0),
//!     [64, 64, 64],
//! );
//!
//! // Encode with wavelet compression
//! let encoded = encode_sdf_volume(&volume, &EncodeConfig::default());
//!
//! // Decode back
//! let decoded = decode_sdf_volume(&encoded);
//! ```
//!
//! Author: Moroya Sakamoto

use alice_codec::quant::Quantizer;
use alice_codec::rans::{FrequencyTable, RansDecoder, RansEncoder};
use alice_codec::wavelet::Wavelet3D;

use crate::eval::eval_batch_parallel;
use crate::types::SdfNode;
use glam::Vec3;

// ============================================================================
// Volume Types
// ============================================================================

/// Voxelized SDF volume for codec encoding.
///
/// Distance values are stored in row-major order: `data[z * H * W + y * W + x]`
/// matching ALICE-Codec's 3D wavelet layout (x, y, z/depth).
#[derive(Clone, Debug)]
pub struct SdfVolume {
    /// Distance values in row-major order `[z][y][x]`.
    pub data: Vec<f32>,
    /// Volume width (X axis).
    pub width: usize,
    /// Volume height (Y axis).
    pub height: usize,
    /// Volume depth (Z axis).
    pub depth: usize,
    /// World-space minimum corner.
    pub origin: Vec3,
    /// Distance between adjacent voxels.
    pub voxel_size: f32,
}

impl SdfVolume {
    /// Total number of voxels.
    #[inline]
    pub fn len(&self) -> usize {
        self.width * self.height * self.depth
    }

    /// Check if volume is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Sample a voxel at (x, y, z) indices.
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        self.data[z * self.height * self.width + y * self.width + x]
    }

    /// World-space position for voxel indices.
    #[inline]
    pub fn world_pos(&self, x: usize, y: usize, z: usize) -> Vec3 {
        self.origin
            + Vec3::new(
                x as f32 * self.voxel_size,
                y as f32 * self.voxel_size,
                z as f32 * self.voxel_size,
            )
    }
}

/// Statistics about a voxelized SDF volume.
#[derive(Clone, Debug)]
pub struct VolumeStats {
    /// Minimum distance value in the volume.
    pub min_distance: f32,
    /// Maximum distance value in the volume.
    pub max_distance: f32,
    /// Number of sign changes between adjacent voxels (surface crossings).
    pub zero_crossings: usize,
    /// Total number of voxels.
    pub total_voxels: usize,
    /// Ratio of surface voxels to total voxels.
    pub surface_ratio: f64,
    /// Root-mean-square of distance values.
    pub rms: f64,
}

/// Configuration for SDF volume encoding.
#[derive(Clone, Debug)]
pub struct EncodeConfig {
    /// Quality factor (0-100). Higher = better quality, larger output.
    pub quality: u8,
    /// Fixed-point scale factor: `i32_value = (f32_value * scale).round()`.
    /// Higher scale preserves more precision but increases coefficient magnitude.
    pub fixed_point_scale: f32,
    /// Use CDF 5/3 wavelet (lossless integers) instead of CDF 9/7.
    pub lossless_wavelet: bool,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        Self {
            quality: 75,
            fixed_point_scale: 1024.0,
            lossless_wavelet: false,
        }
    }
}

impl EncodeConfig {
    /// High quality preset (quality 95).
    pub fn high_quality() -> Self {
        Self {
            quality: 95,
            fixed_point_scale: 4096.0,
            lossless_wavelet: true,
        }
    }

    /// Fast / small preset (quality 50).
    pub fn fast() -> Self {
        Self {
            quality: 50,
            fixed_point_scale: 512.0,
            lossless_wavelet: false,
        }
    }

    /// Lossless wavelet preset (CDF 5/3).
    pub fn lossless() -> Self {
        Self {
            quality: 100,
            fixed_point_scale: 4096.0,
            lossless_wavelet: true,
        }
    }
}

/// Header stored at the beginning of an encoded SDF volume bitstream.
#[derive(Clone, Debug)]
struct EncodedHeader {
    width: u32,
    height: u32,
    depth: u32,
    origin: [f32; 3],
    voxel_size: f32,
    fixed_point_scale: f32,
    quality: u8,
    /// Flags byte: bit 0 = lossless_wavelet, bit 1 = rans_compressed.
    flags: u8,
    /// Number of quantized coefficients (voxel count).
    symbol_count: u32,
}

impl EncodedHeader {
    const SIZE: usize = 4 * 3 + 4 * 3 + 4 + 4 + 1 + 1 + 4; // 34 bytes

    const FLAG_LOSSLESS_WAVELET: u8 = 0x01;
    const FLAG_RANS_COMPRESSED: u8 = 0x02;

    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::SIZE);
        buf.extend_from_slice(&self.width.to_le_bytes());
        buf.extend_from_slice(&self.height.to_le_bytes());
        buf.extend_from_slice(&self.depth.to_le_bytes());
        buf.extend_from_slice(&self.origin[0].to_le_bytes());
        buf.extend_from_slice(&self.origin[1].to_le_bytes());
        buf.extend_from_slice(&self.origin[2].to_le_bytes());
        buf.extend_from_slice(&self.voxel_size.to_le_bytes());
        buf.extend_from_slice(&self.fixed_point_scale.to_le_bytes());
        buf.push(self.quality);
        buf.push(self.flags);
        buf.extend_from_slice(&self.symbol_count.to_le_bytes());
        buf
    }

    fn from_bytes(data: &[u8]) -> Self {
        let read_u32 = |offset: usize| -> u32 {
            u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        };
        let read_f32 = |offset: usize| -> f32 {
            f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        };

        Self {
            width: read_u32(0),
            height: read_u32(4),
            depth: read_u32(8),
            origin: [read_f32(12), read_f32(16), read_f32(20)],
            voxel_size: read_f32(24),
            fixed_point_scale: read_f32(28),
            quality: data[32],
            flags: data[33],
            symbol_count: read_u32(34),
        }
    }

    fn lossless_wavelet(&self) -> bool {
        self.flags & Self::FLAG_LOSSLESS_WAVELET != 0
    }

    fn rans_compressed(&self) -> bool {
        self.flags & Self::FLAG_RANS_COMPRESSED != 0
    }
}

// ============================================================================
// Voxelization
// ============================================================================

/// Voxelize an SDF node onto a regular 3D grid.
///
/// Evaluates the SDF at every grid point using parallel batch evaluation.
///
/// # Arguments
/// * `sdf` - The SDF tree to evaluate
/// * `origin` - World-space minimum corner of the volume
/// * `extent` - World-space maximum corner of the volume
/// * `dims` - Grid dimensions `[width, height, depth]`
///
/// # Returns
/// An `SdfVolume` with distance values at each grid point.
pub fn voxelize_sdf(sdf: &SdfNode, origin: Vec3, extent: Vec3, dims: [usize; 3]) -> SdfVolume {
    let [w, h, d] = dims;
    let size = extent - origin;
    let voxel_size_vec = Vec3::new(
        if w > 1 {
            size.x / (w - 1) as f32
        } else {
            size.x
        },
        if h > 1 {
            size.y / (h - 1) as f32
        } else {
            size.y
        },
        if d > 1 {
            size.z / (d - 1) as f32
        } else {
            size.z
        },
    );
    // Use the smallest axis step as the uniform voxel_size for metadata
    let voxel_size = voxel_size_vec.x.min(voxel_size_vec.y).min(voxel_size_vec.z);

    let total = w * h * d;
    let mut points = Vec::with_capacity(total);

    // Build point list in z-major, y-minor, x-innermost order
    // matching Wavelet3D layout: volume[z * H * W + y * W + x]
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                points.push(Vec3::new(
                    origin.x + x as f32 * voxel_size_vec.x,
                    origin.y + y as f32 * voxel_size_vec.y,
                    origin.z + z as f32 * voxel_size_vec.z,
                ));
            }
        }
    }

    let data = eval_batch_parallel(sdf, &points);

    SdfVolume {
        data,
        width: w,
        height: h,
        depth: d,
        origin,
        voxel_size,
    }
}

/// Voxelize an SDF using a uniform cubic grid at the given resolution.
///
/// Convenience wrapper that computes dims from a single resolution value.
pub fn voxelize_sdf_uniform(
    sdf: &SdfNode,
    origin: Vec3,
    extent: Vec3,
    resolution: usize,
) -> SdfVolume {
    voxelize_sdf(sdf, origin, extent, [resolution, resolution, resolution])
}

// ============================================================================
// Encoding / Decoding
// ============================================================================

/// Encode a voxelized SDF volume using 3D wavelet transform + rANS.
///
/// Pipeline: f32 -> fixed-point i32 -> Wavelet3D forward -> Quantize -> rANS
///
/// # Arguments
/// * `volume` - The SDF volume to encode
/// * `config` - Encoding parameters (quality, scale, wavelet type)
///
/// # Returns
/// Compressed byte stream including header for decoding.
pub fn encode_sdf_volume(volume: &SdfVolume, config: &EncodeConfig) -> Vec<u8> {
    let total = volume.len();
    let scale = config.fixed_point_scale;

    // 1. Convert f32 distances to fixed-point i32
    let mut coeffs: Vec<i32> = volume
        .data
        .iter()
        .map(|&d| (d * scale).round() as i32)
        .collect();

    // 2. Apply 3D wavelet transform
    let wavelet = if config.lossless_wavelet {
        Wavelet3D::cdf53()
    } else {
        Wavelet3D::cdf97()
    };
    wavelet.forward(&mut coeffs, volume.width, volume.height, volume.depth);

    // 3. Quantize
    // Compute step size from quality: quality 100 -> step 1, dead_zone 0
    // (near-lossless). quality 0 -> aggressive quantization.
    // SDF-specific: wavelet detail coefficients are small but crucial for
    // sign preservation, so dead_zone must be kept very small at high quality.
    let quantizer = {
        if config.quality >= 95 {
            // Near-lossless: no quantization loss, only wavelet rounding
            Quantizer::with_dead_zone(1, 0)
        } else {
            let max_abs = coeffs.iter().map(|c| c.abs()).max().unwrap_or(1).max(1) as f32;
            // Cubic curve: high quality stays near step=1
            let inv_quality = (100 - config.quality.min(100)) as f32 / 100.0;
            let quality_factor = inv_quality * inv_quality * inv_quality;
            let step = (1.0 + quality_factor * (max_abs / 4.0)).round() as i32;
            let step = step.max(1);
            // Dead zone proportional to step but capped to avoid zeroing detail
            let dead_zone = step / 2;
            Quantizer::with_dead_zone(step, dead_zone)
        }
    };

    let mut quantized = vec![0i32; total];
    quantizer.quantize_buffer(&coeffs, &mut quantized);

    // 4. Serialize quantized coefficients as i16 little-endian bytes, then
    //    compress those bytes with rANS. This avoids the u8-symbol clamping
    //    issue of alice_codec::quant::to_symbols which is designed for video
    //    pixel data (0-255 range), not for wide-range SDF wavelet coefficients.
    let raw_bytes: Vec<u8> = quantized
        .iter()
        .flat_map(|&v| {
            let clamped = v.max(i16::MIN as i32).min(i16::MAX as i32) as i16;
            clamped.to_le_bytes()
        })
        .collect();

    // 5. Build byte-level histogram and compress with rANS.
    //    If the histogram is too skewed (any symbol dominates > 75% of data),
    //    store raw bytes to avoid u32 overflow in the rANS encoder.
    let mut histogram = [0u32; 256];
    for &b in &raw_bytes {
        histogram[b as usize] += 1;
    }

    let max_freq = *histogram.iter().max().unwrap_or(&0);
    // rANS encoder overflows when x_max = ((RANS32_L >> PROB_BITS) << 8) * freq
    // where freq is the *normalized* frequency (max = PROB_SCALE = 4096).
    // The product limit is u32::MAX. With RANS32_L=1<<23, PROB_BITS=12:
    //   ((1<<23 >> 12) << 8) * 4096 = 2^31 which barely overflows.
    // Safe limit: if max_freq dominates > 50% of data, normalized freq approaches
    // PROB_SCALE and can overflow. Skip rANS for small or highly-skewed data.
    let use_rans = raw_bytes.len() >= 512 && max_freq < (raw_bytes.len() as u32 / 2);

    let payload: Vec<u8> = if use_rans {
        let freq_table = FrequencyTable::from_histogram(&histogram);
        let mut encoder = RansEncoder::with_capacity(raw_bytes.len());
        encoder.encode_symbols(&raw_bytes, &freq_table);
        encoder.finish()
    } else {
        // Store raw (rANS can overflow with extreme histograms on small data)
        raw_bytes.clone()
    };

    // 6. Assemble output: header + quantizer params + histogram + payload
    let mut flags = 0u8;
    if config.lossless_wavelet {
        flags |= EncodedHeader::FLAG_LOSSLESS_WAVELET;
    }
    if use_rans {
        flags |= EncodedHeader::FLAG_RANS_COMPRESSED;
    }

    let header = EncodedHeader {
        width: volume.width as u32,
        height: volume.height as u32,
        depth: volume.depth as u32,
        origin: [volume.origin.x, volume.origin.y, volume.origin.z],
        voxel_size: volume.voxel_size,
        fixed_point_scale: scale,
        quality: config.quality,
        flags,
        symbol_count: total as u32,
    };

    let mut output = header.to_bytes();

    // Write quantizer step + dead_zone (8 bytes)
    output.extend_from_slice(&quantizer.step.to_le_bytes());
    output.extend_from_slice(&quantizer.dead_zone.to_le_bytes());

    if use_rans {
        // Write histogram (256 * 4 = 1024 bytes) - only needed for rANS decoding
        for &count in &histogram {
            output.extend_from_slice(&count.to_le_bytes());
        }
    }

    // Write payload length + data
    let payload_len = payload.len() as u32;
    output.extend_from_slice(&payload_len.to_le_bytes());
    output.extend(payload);

    output
}

/// Decode a compressed SDF volume back to an `SdfVolume`.
///
/// Pipeline: rANS decode -> un-quantize -> Wavelet3D inverse -> i32 -> f32
///
/// # Arguments
/// * `data` - Compressed byte stream produced by `encode_sdf_volume`.
///
/// # Returns
/// Reconstructed `SdfVolume` (with lossy reconstruction error from quantization).
pub fn decode_sdf_volume(data: &[u8]) -> SdfVolume {
    // 1. Read header
    let header = EncodedHeader::from_bytes(data);
    let (w, h, d) = (
        header.width as usize,
        header.height as usize,
        header.depth as usize,
    );
    let total = w * h * d;
    let scale = header.fixed_point_scale;

    let mut offset = EncodedHeader::SIZE;

    // 2. Read quantizer params
    let step = i32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]);
    offset += 4;
    let dead_zone = i32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]);
    offset += 4;
    let quantizer = Quantizer::with_dead_zone(step, dead_zone);

    // 3. Read histogram if rANS compressed
    let use_rans = header.rans_compressed();

    let mut histogram = [0u32; 256];
    if use_rans {
        for h in &mut histogram {
            *h = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;
        }
    }

    // 4. Read payload
    let payload_len = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as usize;
    offset += 4;
    let payload_bytes = &data[offset..offset + payload_len];

    // 5. Decode raw bytes (2 bytes per coefficient = i16 LE)
    let raw_byte_count = total * 2;
    let raw_bytes: Vec<u8> = if use_rans {
        let freq_table = FrequencyTable::from_histogram(&histogram);
        let mut decoder = RansDecoder::new(payload_bytes);
        decoder.decode_n(raw_byte_count, &freq_table)
    } else {
        payload_bytes[..raw_byte_count].to_vec()
    };

    // 6. Convert i16 LE bytes back to quantized i32 coefficients
    let quantized: Vec<i32> = raw_bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as i32)
        .collect();

    // 7. Dequantize
    let mut coeffs = vec![0i32; total];
    quantizer.dequantize_buffer(&quantized, &mut coeffs);

    // 8. Inverse wavelet transform
    let wavelet = if header.lossless_wavelet() {
        Wavelet3D::cdf53()
    } else {
        Wavelet3D::cdf97()
    };
    wavelet.inverse(&mut coeffs, w, h, d);

    // 9. Convert fixed-point i32 back to f32 distances
    let inv_scale = 1.0 / scale;
    let float_data: Vec<f32> = coeffs.iter().map(|&c| c as f32 * inv_scale).collect();

    SdfVolume {
        data: float_data,
        width: w,
        height: h,
        depth: d,
        origin: Vec3::new(header.origin[0], header.origin[1], header.origin[2]),
        voxel_size: header.voxel_size,
    }
}

// ============================================================================
// Volume Statistics
// ============================================================================

/// Compute statistics about an SDF volume.
///
/// Useful for estimating compression characteristics before encoding.
pub fn volume_stats(volume: &SdfVolume) -> VolumeStats {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum_sq: f64 = 0.0;

    for &v in &volume.data {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum_sq += (v as f64) * (v as f64);
    }

    // Count zero-crossings along X axis (sign changes = surface voxels)
    let mut zero_crossing_count = 0usize;
    for z in 0..volume.depth {
        for y in 0..volume.height {
            for x in 1..volume.width {
                let idx = z * volume.height * volume.width + y * volume.width + x;
                if volume.data[idx] * volume.data[idx - 1] < 0.0 {
                    zero_crossing_count += 1;
                }
            }
        }
    }

    let n = volume.data.len();
    let rms = if n > 0 {
        (sum_sq / n as f64).sqrt()
    } else {
        0.0
    };

    VolumeStats {
        min_distance: min,
        max_distance: max,
        zero_crossings: zero_crossing_count,
        total_voxels: n,
        surface_ratio: if n > 0 {
            zero_crossing_count as f64 / n as f64
        } else {
            0.0
        },
        rms,
    }
}

/// Compute the compression ratio for an encoded SDF volume.
///
/// Returns `uncompressed_size / compressed_size`.
pub fn compression_ratio(volume: &SdfVolume, encoded: &[u8]) -> f64 {
    let uncompressed = volume.len() * std::mem::size_of::<f32>();
    uncompressed as f64 / encoded.len() as f64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;

    #[test]
    fn test_voxelize_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let volume = voxelize_sdf(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), [8, 8, 8]);

        assert_eq!(volume.len(), 512);
        assert_eq!(volume.width, 8);
        assert_eq!(volume.height, 8);
        assert_eq!(volume.depth, 8);

        // Center voxel (3,3,3) should be inside the sphere (negative distance)
        let center_d = volume.get(3, 3, 3);
        assert!(
            center_d < 0.0,
            "Center should be inside sphere, got {}",
            center_d
        );

        // Corner voxel (0,0,0) at (-2,-2,-2) should be outside
        let corner_d = volume.get(0, 0, 0);
        assert!(
            corner_d > 0.0,
            "Corner should be outside sphere, got {}",
            corner_d
        );
    }

    #[test]
    fn test_voxelize_uniform() {
        let box_sdf = SdfNode::box3d(1.0, 1.0, 1.0);
        let volume = voxelize_sdf_uniform(&box_sdf, Vec3::splat(-2.0), Vec3::splat(2.0), 4);
        assert_eq!(volume.len(), 64);
    }

    #[test]
    fn test_volume_stats() {
        let sphere = SdfNode::sphere(1.0);
        let volume = voxelize_sdf_uniform(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), 8);

        let stats = volume_stats(&volume);

        assert!(
            stats.min_distance < 0.0,
            "Sphere should have negative interior"
        );
        assert!(
            stats.max_distance > 0.0,
            "Sphere should have positive exterior"
        );
        assert!(
            stats.zero_crossings > 0,
            "Sphere should have surface crossings"
        );
        assert!(stats.surface_ratio > 0.0 && stats.surface_ratio < 1.0);
        assert!(stats.rms > 0.0);
        assert_eq!(stats.total_voxels, 512);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        // Use a small 8^3 sphere volume for fast testing
        let sphere = SdfNode::sphere(1.0);
        let volume = voxelize_sdf_uniform(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), 8);

        let config = EncodeConfig {
            quality: 99,
            fixed_point_scale: 4096.0,
            lossless_wavelet: true, // CDF 5/3 for better roundtrip
        };

        let encoded = encode_sdf_volume(&volume, &config);
        assert!(!encoded.is_empty(), "Encoded data should not be empty");

        let decoded = decode_sdf_volume(&encoded);

        // Verify dimensions match
        assert_eq!(decoded.width, volume.width);
        assert_eq!(decoded.height, volume.height);
        assert_eq!(decoded.depth, volume.depth);
        assert_eq!(decoded.data.len(), volume.data.len());

        // Verify origin preserved
        assert!((decoded.origin - volume.origin).length() < 1e-6);

        // Verify distances are approximately correct
        // (Quantization introduces some error)
        let mut max_error: f32 = 0.0;
        for (orig, recon) in volume.data.iter().zip(decoded.data.iter()) {
            let err = (orig - recon).abs();
            if err > max_error {
                max_error = err;
            }
        }

        // With quality 99 and scale 4096, max error should be very small
        assert!(
            max_error < 0.5,
            "Max reconstruction error {} is too large",
            max_error
        );
    }

    #[test]
    fn test_encode_preserves_sign() {
        // The sign of the SDF (inside/outside) must be preserved after encoding
        let sphere = SdfNode::sphere(1.0);
        let volume = voxelize_sdf_uniform(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), 8);

        let config = EncodeConfig::high_quality();
        let encoded = encode_sdf_volume(&volume, &config);
        let decoded = decode_sdf_volume(&encoded);

        // Count sign mismatches (excluding near-zero values where sign is ambiguous)
        let threshold = 0.1;
        let mut sign_mismatches = 0;
        let mut checked = 0;
        for (&orig, &recon) in volume.data.iter().zip(decoded.data.iter()) {
            if orig.abs() > threshold {
                checked += 1;
                if orig.signum() != recon.signum() {
                    sign_mismatches += 1;
                }
            }
        }

        let mismatch_rate = if checked > 0 {
            sign_mismatches as f64 / checked as f64
        } else {
            0.0
        };

        assert!(
            mismatch_rate < 0.05,
            "Sign mismatch rate {} is too high ({}/{} voxels)",
            mismatch_rate,
            sign_mismatches,
            checked
        );
    }

    #[test]
    fn test_compression_ratio_positive() {
        let sphere = SdfNode::sphere(1.0);
        let volume = voxelize_sdf_uniform(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), 8);

        let encoded = encode_sdf_volume(&volume, &EncodeConfig::default());
        let ratio = compression_ratio(&volume, &encoded);

        // Ratio should be positive (always is by definition)
        assert!(ratio > 0.0, "Compression ratio should be positive");
    }

    #[test]
    fn test_world_pos() {
        let volume = SdfVolume {
            data: vec![0.0; 8],
            width: 2,
            height: 2,
            depth: 2,
            origin: Vec3::new(1.0, 2.0, 3.0),
            voxel_size: 0.5,
        };

        let p = volume.world_pos(1, 1, 1);
        assert!((p.x - 1.5).abs() < 1e-6);
        assert!((p.y - 2.5).abs() < 1e-6);
        assert!((p.z - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_encode_config_presets() {
        let fast = EncodeConfig::fast();
        assert_eq!(fast.quality, 50);

        let hq = EncodeConfig::high_quality();
        assert_eq!(hq.quality, 95);

        let lossless = EncodeConfig::lossless();
        assert!(lossless.lossless_wavelet);
        assert_eq!(lossless.quality, 100);
    }
}
