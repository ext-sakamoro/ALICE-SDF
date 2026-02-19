//! Volume Export: Raw Binary and DDS 3D Texture Formats
//!
//! Export Volume3D to formats consumable by game engines and GPU APIs.
//!
//! - **Raw**: Flat binary dump (f32 or f16), importable by UE5/Unity
//! - **DDS 3D**: DirectX DDS container with 3D texture header, loadable
//!   by DirectX, Vulkan (via KTX conversion), and most game engines
//!
//! Author: Moroya Sakamoto

use std::io::{self, Write};
use std::path::Path;

use super::Volume3D;

/// Export volume as raw binary file (f32 per voxel)
///
/// Layout: Z-major flat array of f32, same as `Volume3D::data`.
/// A companion `.meta` JSON file is written with resolution and bounds.
///
/// # Arguments
/// * `volume` - The volume to export
/// * `path` - Output file path (e.g., "volume.raw")
pub fn export_raw(volume: &Volume3D<f32>, path: &str) -> io::Result<()> {
    let path = Path::new(path);

    // Write raw data
    let mut file = std::fs::File::create(path)?;
    // SAFETY: `volume.data` is a contiguous Vec<f32>. f32 is POD with no padding,
    // so reinterpreting as &[u8] with byte length = len * 4 is sound.
    // The resulting slice borrows `volume.data` and is valid for its lifetime.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            volume.data.as_ptr() as *const u8,
            volume.data.len() * std::mem::size_of::<f32>(),
        )
    };
    file.write_all(bytes)?;

    // Write metadata JSON
    let meta_path = path.with_extension("raw.meta.json");
    let meta = format!(
        r#"{{
  "format": "float32",
  "resolution": [{}, {}, {}],
  "world_min": [{}, {}, {}],
  "world_max": [{}, {}, {}],
  "voxel_count": {},
  "byte_size": {},
  "mip_levels": {}
}}"#,
        volume.resolution[0],
        volume.resolution[1],
        volume.resolution[2],
        volume.world_min.x,
        volume.world_min.y,
        volume.world_min.z,
        volume.world_max.x,
        volume.world_max.y,
        volume.world_max.z,
        volume.voxel_count(),
        volume.data.len() * 4,
        volume.mip_count(),
    );
    std::fs::write(meta_path, meta)?;

    Ok(())
}

/// Export volume as raw binary with mip chain
///
/// Layout: [mip0_data][mip1_data][mip2_data]...
/// Metadata includes byte offsets for each mip level.
pub fn export_raw_with_mips(volume: &Volume3D<f32>, path: &str) -> io::Result<()> {
    let path = Path::new(path);
    let mut file = std::fs::File::create(path)?;

    // Write mip 0 (base level)
    // SAFETY: `volume.data` is a contiguous Vec<f32>. f32 is POD with no padding,
    // so reinterpreting as &[u8] with byte length = len * 4 is sound.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            volume.data.as_ptr() as *const u8,
            volume.data.len() * std::mem::size_of::<f32>(),
        )
    };
    file.write_all(bytes)?;

    // Write each mip level
    let mut mip_offsets = vec![0u64];
    let mut offset = volume.data.len() as u64 * 4;

    for mip in &volume.mips {
        mip_offsets.push(offset);
        // SAFETY: Each mip level is a contiguous Vec<f32>. f32 is POD,
        // so reinterpreting as &[u8] with byte length = len * 4 is sound.
        let mip_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                mip.as_ptr() as *const u8,
                mip.len() * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(mip_bytes)?;
        offset += mip.len() as u64 * 4;
    }

    // Write metadata JSON with mip offsets
    let meta_path = path.with_extension("raw.meta.json");
    let offsets_str = mip_offsets
        .iter()
        .map(|o| o.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let meta = format!(
        r#"{{
  "format": "float32",
  "resolution": [{}, {}, {}],
  "world_min": [{}, {}, {}],
  "world_max": [{}, {}, {}],
  "voxel_count": {},
  "byte_size": {},
  "mip_levels": {},
  "mip_byte_offsets": [{}]
}}"#,
        volume.resolution[0],
        volume.resolution[1],
        volume.resolution[2],
        volume.world_min.x,
        volume.world_min.y,
        volume.world_min.z,
        volume.world_max.x,
        volume.world_max.y,
        volume.world_max.z,
        volume.voxel_count(),
        offset,
        volume.mip_count(),
        offsets_str,
    );
    std::fs::write(meta_path, meta)?;

    Ok(())
}

/// DDS 3D texture pixel format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DdsFormat {
    /// 32-bit float per voxel (R32_FLOAT, DXGI_FORMAT_R32_FLOAT = 41)
    R32Float,
    /// 16-bit float per voxel (R16_FLOAT, DXGI_FORMAT_R16_FLOAT = 54)
    R16Float,
    /// 4-channel 32-bit float (R32G32B32A32_FLOAT = 2) for dist+gradient
    R32G32B32A32Float,
}

/// Export volume as DDS 3D texture
///
/// Creates a DirectX DDS file with DX10 header extension for 3D textures.
/// Compatible with DirectX 11/12, Vulkan (via conversion), and most engines.
///
/// # Arguments
/// * `volume` - The volume to export
/// * `path` - Output file path (e.g., "volume.dds")
/// * `format` - Pixel format
pub fn export_dds_3d(volume: &Volume3D<f32>, path: &str, format: DdsFormat) -> io::Result<()> {
    let path = Path::new(path);
    let mut file = std::fs::File::create(path)?;

    let (dxgi_format, bytes_per_pixel) = match format {
        DdsFormat::R32Float => (41u32, 4u32),
        DdsFormat::R16Float => (54, 2),
        DdsFormat::R32G32B32A32Float => (2, 16),
    };

    // DDS magic number
    file.write_all(&0x20534444u32.to_le_bytes())?; // "DDS "

    // DDS_HEADER (124 bytes)
    let pitch_or_linear = volume.resolution[0] * bytes_per_pixel;
    let header = DdsHeader {
        size: 124,
        flags: 0x00000001
            | 0x00000002
            | 0x00000004
            | 0x00000008
            | 0x00001000
            | 0x00080000
            | 0x00800000,
        // DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_MIPMAPCOUNT | DDSD_LINEARSIZE | DDSD_DEPTH
        height: volume.resolution[1],
        width: volume.resolution[0],
        pitch_or_linear_size: pitch_or_linear,
        depth: volume.resolution[2],
        mip_map_count: volume.mip_count() as u32,
        reserved1: [0u32; 11],
        // DDS_PIXELFORMAT (32 bytes)
        pf_size: 32,
        pf_flags: 0x00000004, // DDPF_FOURCC
        pf_four_cc: u32::from_le_bytes(*b"DX10"),
        pf_rgb_bit_count: 0,
        pf_r_bitmask: 0,
        pf_g_bitmask: 0,
        pf_b_bitmask: 0,
        pf_a_bitmask: 0,
        // DDS_HEADER caps
        caps: 0x00001000 | 0x00000008 | 0x00400000, // DDSCAPS_TEXTURE | DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
        caps2: 0x00200000,                          // DDSCAPS2_VOLUME
        caps3: 0,
        caps4: 0,
        reserved2: 0,
    };

    write_dds_header(&mut file, &header)?;

    // DDS_HEADER_DXT10 (20 bytes)
    let dxt10 = DdsHeaderDxt10 {
        dxgi_format,
        resource_dimension: 4, // D3D10_RESOURCE_DIMENSION_TEXTURE3D
        misc_flag: 0,
        array_size: 1,
        misc_flags2: 0,
    };
    write_dds_header_dxt10(&mut file, &dxt10)?;

    // Write pixel data
    match format {
        DdsFormat::R32Float => {
            // SAFETY: `volume.data` is a contiguous Vec<f32>. f32 is POD,
            // so reinterpreting as &[u8] with byte length = len * 4 is sound.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(volume.data.as_ptr() as *const u8, volume.data.len() * 4)
            };
            file.write_all(bytes)?;

            // Write mip levels
            for mip in &volume.mips {
                // SAFETY: Each mip is a contiguous Vec<f32>. f32 is POD,
                // so reinterpreting as &[u8] with byte length = len * 4 is sound.
                let mip_bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(mip.as_ptr() as *const u8, mip.len() * 4) };
                file.write_all(mip_bytes)?;
            }
        }
        DdsFormat::R16Float => {
            // Convert f32 -> f16
            for &val in &volume.data {
                let h = f32_to_f16(val);
                file.write_all(&h.to_le_bytes())?;
            }
            for mip in &volume.mips {
                for &val in mip {
                    let h = f32_to_f16(val);
                    file.write_all(&h.to_le_bytes())?;
                }
            }
        }
        DdsFormat::R32G32B32A32Float => {
            // Pad distance to RGBA (dist, 0, 0, 0)
            for &val in &volume.data {
                file.write_all(&val.to_le_bytes())?;
                file.write_all(&0.0f32.to_le_bytes())?;
                file.write_all(&0.0f32.to_le_bytes())?;
                file.write_all(&0.0f32.to_le_bytes())?;
            }
            for mip in &volume.mips {
                for &val in mip {
                    file.write_all(&val.to_le_bytes())?;
                    file.write_all(&0.0f32.to_le_bytes())?;
                    file.write_all(&0.0f32.to_le_bytes())?;
                    file.write_all(&0.0f32.to_le_bytes())?;
                }
            }
        }
    }

    Ok(())
}

/// Export distance+gradient volume as DDS 3D (RGBA32F)
pub fn export_dds_3d_distgrad(
    volume: &Volume3D<super::VoxelDistGrad>,
    path: &str,
) -> io::Result<()> {
    let path = Path::new(path);
    let mut file = std::fs::File::create(path)?;

    let dxgi_format = 2u32; // R32G32B32A32_FLOAT
    let bytes_per_pixel = 16u32;

    // DDS magic
    file.write_all(&0x20534444u32.to_le_bytes())?;

    let header = DdsHeader {
        size: 124,
        flags: 0x00000001
            | 0x00000002
            | 0x00000004
            | 0x00000008
            | 0x00001000
            | 0x00080000
            | 0x00800000,
        height: volume.resolution[1],
        width: volume.resolution[0],
        pitch_or_linear_size: volume.resolution[0] * bytes_per_pixel,
        depth: volume.resolution[2],
        mip_map_count: volume.mip_count() as u32,
        reserved1: [0u32; 11],
        pf_size: 32,
        pf_flags: 0x00000004,
        pf_four_cc: u32::from_le_bytes(*b"DX10"),
        pf_rgb_bit_count: 0,
        pf_r_bitmask: 0,
        pf_g_bitmask: 0,
        pf_b_bitmask: 0,
        pf_a_bitmask: 0,
        caps: 0x00001000 | 0x00000008 | 0x00400000,
        caps2: 0x00200000,
        caps3: 0,
        caps4: 0,
        reserved2: 0,
    };
    write_dds_header(&mut file, &header)?;

    let dxt10 = DdsHeaderDxt10 {
        dxgi_format,
        resource_dimension: 4,
        misc_flag: 0,
        array_size: 1,
        misc_flags2: 0,
    };
    write_dds_header_dxt10(&mut file, &dxt10)?;

    // Write RGBA32F data (dist, nx, ny, nz)
    for voxel in &volume.data {
        file.write_all(&voxel.distance.to_le_bytes())?;
        file.write_all(&voxel.nx.to_le_bytes())?;
        file.write_all(&voxel.ny.to_le_bytes())?;
        file.write_all(&voxel.nz.to_le_bytes())?;
    }

    for mip in &volume.mips {
        for voxel in mip {
            file.write_all(&voxel.distance.to_le_bytes())?;
            file.write_all(&voxel.nx.to_le_bytes())?;
            file.write_all(&voxel.ny.to_le_bytes())?;
            file.write_all(&voxel.nz.to_le_bytes())?;
        }
    }

    Ok(())
}

// --- Internal helpers ---

#[allow(dead_code)]
struct DdsHeader {
    size: u32,
    flags: u32,
    height: u32,
    width: u32,
    pitch_or_linear_size: u32,
    depth: u32,
    mip_map_count: u32,
    reserved1: [u32; 11],
    pf_size: u32,
    pf_flags: u32,
    pf_four_cc: u32,
    pf_rgb_bit_count: u32,
    pf_r_bitmask: u32,
    pf_g_bitmask: u32,
    pf_b_bitmask: u32,
    pf_a_bitmask: u32,
    caps: u32,
    caps2: u32,
    caps3: u32,
    caps4: u32,
    reserved2: u32,
}

fn write_dds_header(w: &mut impl Write, h: &DdsHeader) -> io::Result<()> {
    w.write_all(&h.size.to_le_bytes())?;
    w.write_all(&h.flags.to_le_bytes())?;
    w.write_all(&h.height.to_le_bytes())?;
    w.write_all(&h.width.to_le_bytes())?;
    w.write_all(&h.pitch_or_linear_size.to_le_bytes())?;
    w.write_all(&h.depth.to_le_bytes())?;
    w.write_all(&h.mip_map_count.to_le_bytes())?;
    for r in &h.reserved1 {
        w.write_all(&r.to_le_bytes())?;
    }
    w.write_all(&h.pf_size.to_le_bytes())?;
    w.write_all(&h.pf_flags.to_le_bytes())?;
    w.write_all(&h.pf_four_cc.to_le_bytes())?;
    w.write_all(&h.pf_rgb_bit_count.to_le_bytes())?;
    w.write_all(&h.pf_r_bitmask.to_le_bytes())?;
    w.write_all(&h.pf_g_bitmask.to_le_bytes())?;
    w.write_all(&h.pf_b_bitmask.to_le_bytes())?;
    w.write_all(&h.pf_a_bitmask.to_le_bytes())?;
    w.write_all(&h.caps.to_le_bytes())?;
    w.write_all(&h.caps2.to_le_bytes())?;
    w.write_all(&h.caps3.to_le_bytes())?;
    w.write_all(&h.caps4.to_le_bytes())?;
    w.write_all(&h.reserved2.to_le_bytes())?;
    Ok(())
}

struct DdsHeaderDxt10 {
    dxgi_format: u32,
    resource_dimension: u32,
    misc_flag: u32,
    array_size: u32,
    misc_flags2: u32,
}

fn write_dds_header_dxt10(w: &mut impl Write, h: &DdsHeaderDxt10) -> io::Result<()> {
    w.write_all(&h.dxgi_format.to_le_bytes())?;
    w.write_all(&h.resource_dimension.to_le_bytes())?;
    w.write_all(&h.misc_flag.to_le_bytes())?;
    w.write_all(&h.array_size.to_le_bytes())?;
    w.write_all(&h.misc_flags2.to_le_bytes())?;
    Ok(())
}

/// Convert f32 to IEEE 754 half-precision (f16)
///
/// Software implementation for portability (no `half` crate dependency).
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007FFFFF;

    if exponent == 255 {
        // Inf or NaN
        if mantissa != 0 {
            return (sign | 0x7E00) as u16; // NaN
        }
        return (sign | 0x7C00) as u16; // Inf
    }

    let new_exp = exponent - 127 + 15;

    if new_exp >= 31 {
        return (sign | 0x7C00) as u16; // Overflow -> Inf
    }

    if new_exp <= 0 {
        if new_exp < -10 {
            return sign as u16; // Underflow -> 0
        }
        // Denormalized
        let m = (mantissa | 0x00800000) >> (1 - new_exp);
        return (sign | (m >> 13)) as u16;
    }

    (sign | ((new_exp as u32) << 10) | (mantissa >> 13)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_export_raw() {
        let vol = Volume3D::<f32>::new([4, 4, 4], Vec3::splat(-1.0), Vec3::splat(1.0));

        let dir = std::env::temp_dir();
        let path = dir.join("alice_test_volume.raw");
        let path_str = path.to_str().unwrap();

        export_raw(&vol, path_str).unwrap();

        // Check file exists and has correct size
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 4 * 4 * 4 * 4); // 64 voxels * 4 bytes

        // Check metadata exists
        let meta_path = path.with_extension("raw.meta.json");
        assert!(meta_path.exists());

        // Cleanup
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&meta_path).ok();
    }

    #[test]
    fn test_export_dds_3d() {
        let vol = Volume3D::<f32>::new([4, 4, 4], Vec3::splat(-1.0), Vec3::splat(1.0));

        let dir = std::env::temp_dir();
        let path = dir.join("alice_test_volume.dds");
        let path_str = path.to_str().unwrap();

        export_dds_3d(&vol, path_str, DdsFormat::R32Float).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        // DDS magic + header (124) + DXT10 (20) + pixel data
        assert!(bytes.len() > 4 + 124 + 20);
        assert_eq!(&bytes[0..4], &0x20534444u32.to_le_bytes()); // "DDS "

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_f32_to_f16() {
        // Zero
        assert_eq!(f32_to_f16(0.0), 0x0000);

        // One
        assert_eq!(f32_to_f16(1.0), 0x3C00);

        // Negative one
        assert_eq!(f32_to_f16(-1.0), 0xBC00);

        // Infinity
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xFC00);
    }
}
