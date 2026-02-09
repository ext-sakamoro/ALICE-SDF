//! ALICE Binary Mesh (.abm) format (Deep Fried Edition)
//!
//! A compact binary format for storing polygon meshes with optional vertex
//! attributes and LOD chains.
//!
//! # Header (32 bytes)
//!
//! | Offset | Size | Field          | Description                        |
//! |--------|------|----------------|------------------------------------|
//! | 0      | 8    | magic          | `"ALICEBM\0"`                     |
//! | 8      | 2    | version        | Format version (currently 1)       |
//! | 10     | 2    | flags          | Feature flags (bitfield)           |
//! | 12     | 4    | vertex_count   | Number of vertices in primary mesh |
//! | 16     | 4    | index_count    | Number of indices in primary mesh  |
//! | 20     | 1    | lod_count      | Number of LOD levels (0 = single)  |
//! | 21     | 7    | _reserved      | Reserved for future use            |
//! | 28     | 4    | crc32          | CRC32 of entire payload            |
//!
//! # Payload (SoA layout, little-endian)
//!
//! 1. Positions:    `[f32; vertex_count * 3]` (always present)
//! 2. Normals:      `[f32; vertex_count * 3]` (if `HAS_NORMALS`)
//! 3. UVs:          `[f32; vertex_count * 2]` (if `HAS_UVS`)
//! 4. UV2:          `[f32; vertex_count * 2]` (if `HAS_UV2`)
//! 5. Tangents:     `[f32; vertex_count * 4]` (if `HAS_TANGENTS`)
//! 6. Colors:       `[f32; vertex_count * 4]` (if `HAS_COLORS`)
//! 7. Material IDs: `[u32; vertex_count]`     (if `HAS_MATERIAL_IDS`)
//! 8. Indices:      `[u32; index_count]`
//! 9. LOD entries   (if `HAS_LODS`)
//!
//! # Deep Fried Optimizations
//! - **SoA Payload**: Vertex attributes stored as separate arrays for cache-friendly
//!   SIMD batch reads (positions separate from normals, etc.).
//! - **Streaming Write**: CRC computed on-the-fly with seek-patch (same as asdf.rs).
//! - **CRC-First Read**: Validates integrity before parsing payload.
//! - **Buffered I/O**: BufReader/BufWriter for reduced syscalls.
//! - **Feature Flags**: Only non-zero attributes are stored (minimal file size).
//!
//! Author: Moroya Sakamoto

use crate::io::IoError;
use crate::mesh::{Mesh, Vertex};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic bytes identifying the ABM format
pub const ABM_MAGIC: [u8; 8] = *b"ALICEBM\0";

/// Current format version
pub const ABM_VERSION: u16 = 1;

/// ABM header size in bytes
const ABM_HEADER_SIZE: usize = 32;

// Feature flags (bitfield)

/// Vertex normals present
pub const ABM_FLAG_HAS_NORMALS: u16 = 1 << 0;
/// Primary UV coordinates present
pub const ABM_FLAG_HAS_UVS: u16 = 1 << 1;
/// Secondary UV (lightmap) coordinates present
pub const ABM_FLAG_HAS_UV2: u16 = 1 << 2;
/// Tangent vectors present
pub const ABM_FLAG_HAS_TANGENTS: u16 = 1 << 3;
/// Vertex colors present
pub const ABM_FLAG_HAS_COLORS: u16 = 1 << 4;
/// Per-vertex material IDs present
pub const ABM_FLAG_HAS_MATERIAL_IDS: u16 = 1 << 5;
/// LOD chain present after primary mesh
pub const ABM_FLAG_HAS_LODS: u16 = 1 << 6;

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// ABM file header (32 bytes, little-endian)
#[derive(Debug, Clone, Copy)]
pub struct AbmHeader {
    /// Magic number `"ALICEBM\0"`
    pub magic: [u8; 8],
    /// Format version
    pub version: u16,
    /// Feature flags (bitfield of `ABM_FLAG_*`)
    pub flags: u16,
    /// Number of vertices in the primary mesh
    pub vertex_count: u32,
    /// Number of indices in the primary mesh
    pub index_count: u32,
    /// Number of LOD levels (0 = single mesh, no LODs)
    pub lod_count: u8,
    /// Reserved for future use
    pub _reserved: [u8; 7],
    /// CRC32 of the entire payload (everything after the header)
    pub crc32: u32,
}

impl AbmHeader {
    /// Serialize header to a 32-byte little-endian array
    #[inline(always)]
    pub fn to_bytes(&self) -> [u8; ABM_HEADER_SIZE] {
        let mut b = [0u8; ABM_HEADER_SIZE];
        b[0..8].copy_from_slice(&self.magic);
        b[8..10].copy_from_slice(&self.version.to_le_bytes());
        b[10..12].copy_from_slice(&self.flags.to_le_bytes());
        b[12..16].copy_from_slice(&self.vertex_count.to_le_bytes());
        b[16..20].copy_from_slice(&self.index_count.to_le_bytes());
        b[20] = self.lod_count;
        b[21..28].copy_from_slice(&self._reserved);
        b[28..32].copy_from_slice(&self.crc32.to_le_bytes());
        b
    }

    /// Parse header from a 32-byte little-endian array
    #[inline(always)]
    pub fn from_bytes(b: &[u8; ABM_HEADER_SIZE]) -> Result<Self, IoError> {
        let magic: [u8; 8] = b[0..8].try_into().unwrap();
        if magic != ABM_MAGIC {
            return Err(IoError::InvalidFormat(format!(
                "Invalid ABM magic: {:?}",
                magic
            )));
        }

        let version = u16::from_le_bytes(b[8..10].try_into().unwrap());
        if version > ABM_VERSION {
            return Err(IoError::UnsupportedVersion(version));
        }

        let flags = u16::from_le_bytes(b[10..12].try_into().unwrap());
        let vertex_count = u32::from_le_bytes(b[12..16].try_into().unwrap());
        let index_count = u32::from_le_bytes(b[16..20].try_into().unwrap());
        let lod_count = b[20];
        let mut reserved = [0u8; 7];
        reserved.copy_from_slice(&b[21..28]);
        let crc32 = u32::from_le_bytes(b[28..32].try_into().unwrap());

        Ok(AbmHeader {
            magic,
            version,
            flags,
            vertex_count,
            index_count,
            lod_count,
            _reserved: reserved,
            crc32,
        })
    }
}

// ---------------------------------------------------------------------------
// CRC streaming writer (same pattern as asdf.rs)
// ---------------------------------------------------------------------------

/// Writer wrapper that computes CRC32 on-the-fly
struct CrcWriter<W: Write> {
    inner: W,
    hasher: crc32fast::Hasher,
}

impl<W: Write> CrcWriter<W> {
    #[inline(always)]
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: crc32fast::Hasher::new(),
        }
    }

    #[inline(always)]
    fn finalize(self) -> u32 {
        self.hasher.finalize()
    }
}

impl<W: Write> Write for CrcWriter<W> {
    #[inline(always)]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }

    #[inline(always)]
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

// ---------------------------------------------------------------------------
// Helpers: detect which attributes are non-default
// ---------------------------------------------------------------------------

/// Check if a vertex has a meaningful (non-zero) normal
#[inline(always)]
fn has_nonzero_normal(v: &Vertex) -> bool {
    v.normal.x != 0.0 || v.normal.y != 0.0 || v.normal.z != 0.0
}

/// Check if a vertex has meaningful UV coordinates
#[inline(always)]
fn has_nonzero_uv(v: &Vertex) -> bool {
    v.uv.x != 0.0 || v.uv.y != 0.0
}

/// Check if a vertex has meaningful UV2 coordinates
#[inline(always)]
fn has_nonzero_uv2(v: &Vertex) -> bool {
    v.uv2.x != 0.0 || v.uv2.y != 0.0
}

/// Check if a vertex has a non-default tangent
#[inline(always)]
fn has_nondefault_tangent(v: &Vertex) -> bool {
    // Default tangent is (1,0,0,1); anything else is meaningful
    !(v.tangent.x == 1.0 && v.tangent.y == 0.0 && v.tangent.z == 0.0 && v.tangent.w == 1.0)
}

/// Check if a vertex has a non-white color
#[inline(always)]
fn has_nondefault_color(v: &Vertex) -> bool {
    v.color != [1.0, 1.0, 1.0, 1.0]
}

/// Check if a vertex has a non-zero material ID
#[inline(always)]
fn has_nonzero_material_id(v: &Vertex) -> bool {
    v.material_id != 0
}

/// Determine feature flags from mesh vertices
fn compute_flags(mesh: &Mesh) -> u16 {
    let mut flags: u16 = 0;
    for v in &mesh.vertices {
        if has_nonzero_normal(v) {
            flags |= ABM_FLAG_HAS_NORMALS;
        }
        if has_nonzero_uv(v) {
            flags |= ABM_FLAG_HAS_UVS;
        }
        if has_nonzero_uv2(v) {
            flags |= ABM_FLAG_HAS_UV2;
        }
        if has_nondefault_tangent(v) {
            flags |= ABM_FLAG_HAS_TANGENTS;
        }
        if has_nondefault_color(v) {
            flags |= ABM_FLAG_HAS_COLORS;
        }
        if has_nonzero_material_id(v) {
            flags |= ABM_FLAG_HAS_MATERIAL_IDS;
        }
        // Early exit if all flags already set
        if flags
            == (ABM_FLAG_HAS_NORMALS
                | ABM_FLAG_HAS_UVS
                | ABM_FLAG_HAS_UV2
                | ABM_FLAG_HAS_TANGENTS
                | ABM_FLAG_HAS_COLORS
                | ABM_FLAG_HAS_MATERIAL_IDS)
        {
            break;
        }
    }
    flags
}

// ---------------------------------------------------------------------------
// Payload write helpers
// ---------------------------------------------------------------------------

/// Write f32 values as little-endian bytes
#[inline(always)]
fn write_f32<W: Write>(w: &mut W, val: f32) -> std::io::Result<()> {
    w.write_all(&val.to_le_bytes())
}

/// Write u32 values as little-endian bytes
#[inline(always)]
fn write_u32<W: Write>(w: &mut W, val: u32) -> std::io::Result<()> {
    w.write_all(&val.to_le_bytes())
}

/// Write the SoA payload for a single mesh
fn write_mesh_payload<W: Write>(w: &mut W, mesh: &Mesh, flags: u16) -> Result<(), IoError> {
    // 1. Positions (always present)
    for v in &mesh.vertices {
        write_f32(w, v.position.x)?;
        write_f32(w, v.position.y)?;
        write_f32(w, v.position.z)?;
    }

    // 2. Normals
    if flags & ABM_FLAG_HAS_NORMALS != 0 {
        for v in &mesh.vertices {
            write_f32(w, v.normal.x)?;
            write_f32(w, v.normal.y)?;
            write_f32(w, v.normal.z)?;
        }
    }

    // 3. UVs
    if flags & ABM_FLAG_HAS_UVS != 0 {
        for v in &mesh.vertices {
            write_f32(w, v.uv.x)?;
            write_f32(w, v.uv.y)?;
        }
    }

    // 4. UV2
    if flags & ABM_FLAG_HAS_UV2 != 0 {
        for v in &mesh.vertices {
            write_f32(w, v.uv2.x)?;
            write_f32(w, v.uv2.y)?;
        }
    }

    // 5. Tangents
    if flags & ABM_FLAG_HAS_TANGENTS != 0 {
        for v in &mesh.vertices {
            write_f32(w, v.tangent.x)?;
            write_f32(w, v.tangent.y)?;
            write_f32(w, v.tangent.z)?;
            write_f32(w, v.tangent.w)?;
        }
    }

    // 6. Colors
    if flags & ABM_FLAG_HAS_COLORS != 0 {
        for v in &mesh.vertices {
            write_f32(w, v.color[0])?;
            write_f32(w, v.color[1])?;
            write_f32(w, v.color[2])?;
            write_f32(w, v.color[3])?;
        }
    }

    // 7. Material IDs
    if flags & ABM_FLAG_HAS_MATERIAL_IDS != 0 {
        for v in &mesh.vertices {
            write_u32(w, v.material_id)?;
        }
    }

    // 8. Indices
    for &idx in &mesh.indices {
        write_u32(w, idx)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Payload read helpers
// ---------------------------------------------------------------------------

/// Read a little-endian f32 from a byte slice at the given offset, advancing it
#[inline(always)]
fn read_f32(data: &[u8], offset: &mut usize) -> Result<f32, IoError> {
    if *offset + 4 > data.len() {
        return Err(IoError::InvalidFormat("Unexpected end of payload".into()));
    }
    let val = f32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    Ok(val)
}

/// Read a little-endian u32 from a byte slice at the given offset, advancing it
#[inline(always)]
fn read_u32(data: &[u8], offset: &mut usize) -> Result<u32, IoError> {
    if *offset + 4 > data.len() {
        return Err(IoError::InvalidFormat("Unexpected end of payload".into()));
    }
    let val = u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    Ok(val)
}

/// Read a mesh from the SoA payload
fn read_mesh_payload(
    data: &[u8],
    offset: &mut usize,
    vertex_count: u32,
    index_count: u32,
    flags: u16,
) -> Result<Mesh, IoError> {
    let vc = vertex_count as usize;
    let ic = index_count as usize;

    // Pre-allocate vertices with defaults
    let mut vertices = vec![Vertex::default(); vc];

    // 1. Positions (always)
    for v in vertices.iter_mut() {
        v.position.x = read_f32(data, offset)?;
        v.position.y = read_f32(data, offset)?;
        v.position.z = read_f32(data, offset)?;
    }

    // 2. Normals
    if flags & ABM_FLAG_HAS_NORMALS != 0 {
        for v in vertices.iter_mut() {
            v.normal.x = read_f32(data, offset)?;
            v.normal.y = read_f32(data, offset)?;
            v.normal.z = read_f32(data, offset)?;
        }
    }

    // 3. UVs
    if flags & ABM_FLAG_HAS_UVS != 0 {
        for v in vertices.iter_mut() {
            v.uv.x = read_f32(data, offset)?;
            v.uv.y = read_f32(data, offset)?;
        }
    }

    // 4. UV2
    if flags & ABM_FLAG_HAS_UV2 != 0 {
        for v in vertices.iter_mut() {
            v.uv2.x = read_f32(data, offset)?;
            v.uv2.y = read_f32(data, offset)?;
        }
    }

    // 5. Tangents
    if flags & ABM_FLAG_HAS_TANGENTS != 0 {
        for v in vertices.iter_mut() {
            v.tangent.x = read_f32(data, offset)?;
            v.tangent.y = read_f32(data, offset)?;
            v.tangent.z = read_f32(data, offset)?;
            v.tangent.w = read_f32(data, offset)?;
        }
    }

    // 6. Colors
    if flags & ABM_FLAG_HAS_COLORS != 0 {
        for v in vertices.iter_mut() {
            v.color[0] = read_f32(data, offset)?;
            v.color[1] = read_f32(data, offset)?;
            v.color[2] = read_f32(data, offset)?;
            v.color[3] = read_f32(data, offset)?;
        }
    }

    // 7. Material IDs
    if flags & ABM_FLAG_HAS_MATERIAL_IDS != 0 {
        for v in vertices.iter_mut() {
            v.material_id = read_u32(data, offset)?;
        }
    }

    // 8. Indices
    let mut indices = Vec::with_capacity(ic);
    for _ in 0..ic {
        indices.push(read_u32(data, offset)?);
    }

    // Restore default tangent for vertices without tangent data
    if flags & ABM_FLAG_HAS_TANGENTS == 0 {
        for v in vertices.iter_mut() {
            v.tangent = glam::Vec4::new(1.0, 0.0, 0.0, 1.0);
        }
    }

    // Restore default color for vertices without color data
    if flags & ABM_FLAG_HAS_COLORS == 0 {
        for v in vertices.iter_mut() {
            v.color = [1.0, 1.0, 1.0, 1.0];
        }
    }

    Ok(Mesh { vertices, indices })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Save a mesh to ALICE Binary Mesh (.abm) format
///
/// Uses streaming CRC with seek-patch: writes placeholder header, streams the
/// SoA payload while computing CRC32 on-the-fly, then patches the header.
///
/// # Arguments
/// * `mesh` - The mesh to save
/// * `path` - Destination file path
///
/// # Errors
/// Returns `IoError::Io` on filesystem errors.
pub fn save_abm(mesh: &Mesh, path: impl AsRef<Path>) -> Result<(), IoError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let flags = compute_flags(mesh);

    // 1. Write placeholder header (CRC unknown yet)
    let placeholder = AbmHeader {
        magic: ABM_MAGIC,
        version: ABM_VERSION,
        flags,
        vertex_count: mesh.vertices.len() as u32,
        index_count: mesh.indices.len() as u32,
        lod_count: 0,
        _reserved: [0u8; 7],
        crc32: 0,
    };
    writer.write_all(&placeholder.to_bytes())?;

    // 2. Stream payload through CRC writer
    let mut crc_writer = CrcWriter::new(&mut writer);
    write_mesh_payload(&mut crc_writer, mesh, flags)?;
    let crc = crc_writer.finalize();

    // 3. Seek back and patch CRC in header
    writer.seek(SeekFrom::Start(0))?;
    let real_header = AbmHeader {
        crc32: crc,
        ..placeholder
    };
    writer.write_all(&real_header.to_bytes())?;
    writer.flush()?;

    Ok(())
}

/// Load a mesh from ALICE Binary Mesh (.abm) format
///
/// Reads the entire payload, validates CRC32 before parsing (fail-fast on
/// corruption), then reconstructs the mesh from SoA layout.
///
/// # Arguments
/// * `path` - Source file path
///
/// # Errors
/// Returns `IoError::CrcMismatch` on integrity failure, `IoError::InvalidFormat`
/// on malformed data.
pub fn load_abm(path: impl AsRef<Path>) -> Result<Mesh, IoError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // 1. Read header
    let mut hdr_bytes = [0u8; ABM_HEADER_SIZE];
    reader.read_exact(&mut hdr_bytes)?;
    let header = AbmHeader::from_bytes(&hdr_bytes)?;

    // 2. Read entire payload
    let mut payload = Vec::new();
    reader.read_to_end(&mut payload)?;

    // 3. CRC-first validation
    let actual_crc = crc32fast::hash(&payload);
    if actual_crc != header.crc32 {
        return Err(IoError::CrcMismatch {
            expected: header.crc32,
            actual: actual_crc,
        });
    }

    // 4. Parse payload
    let mut offset = 0;
    read_mesh_payload(
        &payload,
        &mut offset,
        header.vertex_count,
        header.index_count,
        header.flags,
    )
}

/// Save multiple meshes as an ABM file with LOD chain
///
/// The first mesh in `meshes` is the primary (LOD 0). Additional meshes are
/// stored as LOD levels with corresponding `transition_distances`.
///
/// # Arguments
/// * `meshes` - LOD meshes from highest to lowest detail. Must have at least 1.
/// * `transition_distances` - Screen-space transition distance for each LOD
///   level after the first. Length must equal `meshes.len() - 1`.
/// * `path` - Destination file path
///
/// # Errors
/// Returns `IoError::InvalidFormat` if `meshes` is empty or distances length
/// does not match.
pub fn save_abm_with_lods(
    meshes: &[Mesh],
    transition_distances: &[f32],
    path: impl AsRef<Path>,
) -> Result<(), IoError> {
    if meshes.is_empty() {
        return Err(IoError::InvalidFormat("No meshes provided".into()));
    }
    if transition_distances.len() != meshes.len().saturating_sub(1) {
        return Err(IoError::InvalidFormat(format!(
            "Expected {} transition distances for {} LODs, got {}",
            meshes.len().saturating_sub(1),
            meshes.len(),
            transition_distances.len()
        )));
    }

    let primary = &meshes[0];
    let mut flags = compute_flags(primary);
    // Also scan LOD meshes for flags
    for lod in &meshes[1..] {
        flags |= compute_flags(lod);
    }

    let lod_count = if meshes.len() > 1 {
        flags |= ABM_FLAG_HAS_LODS;
        (meshes.len() - 1) as u8
    } else {
        0
    };

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // 1. Write placeholder header
    let placeholder = AbmHeader {
        magic: ABM_MAGIC,
        version: ABM_VERSION,
        flags,
        vertex_count: primary.vertices.len() as u32,
        index_count: primary.indices.len() as u32,
        lod_count,
        _reserved: [0u8; 7],
        crc32: 0,
    };
    writer.write_all(&placeholder.to_bytes())?;

    // 2. Stream payload through CRC writer
    let mut crc_writer = CrcWriter::new(&mut writer);

    // Primary mesh
    write_mesh_payload(&mut crc_writer, primary, flags)?;

    // LOD entries
    for (i, lod_mesh) in meshes[1..].iter().enumerate() {
        // LOD header: vertex_count, index_count, transition_distance
        write_u32(&mut crc_writer, lod_mesh.vertices.len() as u32)?;
        write_u32(&mut crc_writer, lod_mesh.indices.len() as u32)?;
        write_f32(&mut crc_writer, transition_distances[i])?;

        // LOD mesh payload
        write_mesh_payload(&mut crc_writer, lod_mesh, flags)?;
    }

    let crc = crc_writer.finalize();

    // 3. Patch CRC
    writer.seek(SeekFrom::Start(0))?;
    let real_header = AbmHeader {
        crc32: crc,
        ..placeholder
    };
    writer.write_all(&real_header.to_bytes())?;
    writer.flush()?;

    Ok(())
}

/// Load an ABM file containing a LOD chain
///
/// Returns a vector of meshes (LOD 0 first) and the transition distances
/// between them.
///
/// # Arguments
/// * `path` - Source file path
///
/// # Errors
/// Returns `IoError::CrcMismatch` on integrity failure.
pub fn load_abm_with_lods(
    path: impl AsRef<Path>,
) -> Result<(Vec<Mesh>, Vec<f32>), IoError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // 1. Read header
    let mut hdr_bytes = [0u8; ABM_HEADER_SIZE];
    reader.read_exact(&mut hdr_bytes)?;
    let header = AbmHeader::from_bytes(&hdr_bytes)?;

    // 2. Read entire payload
    let mut payload = Vec::new();
    reader.read_to_end(&mut payload)?;

    // 3. CRC validation
    let actual_crc = crc32fast::hash(&payload);
    if actual_crc != header.crc32 {
        return Err(IoError::CrcMismatch {
            expected: header.crc32,
            actual: actual_crc,
        });
    }

    // 4. Parse primary mesh
    let mut offset = 0;
    let primary = read_mesh_payload(
        &payload,
        &mut offset,
        header.vertex_count,
        header.index_count,
        header.flags,
    )?;

    let mut meshes = vec![primary];
    let mut distances = Vec::new();

    // 5. Parse LOD entries
    if header.flags & ABM_FLAG_HAS_LODS != 0 {
        for _ in 0..header.lod_count {
            let lod_vc = read_u32(&payload, &mut offset)?;
            let lod_ic = read_u32(&payload, &mut offset)?;
            let dist = read_f32(&payload, &mut offset)?;

            let lod_mesh = read_mesh_payload(
                &payload,
                &mut offset,
                lod_vc,
                lod_ic,
                header.flags,
            )?;

            meshes.push(lod_mesh);
            distances.push(dist);
        }
    }

    Ok((meshes, distances))
}

/// Read only the ABM header without loading the full mesh
///
/// Useful for inspecting file metadata (vertex/index count, flags, LOD count)
/// without the cost of reading the entire payload.
///
/// # Arguments
/// * `path` - Source file path
///
/// # Errors
/// Returns `IoError::InvalidFormat` on bad magic, `IoError::UnsupportedVersion`
/// on unknown version.
pub fn read_abm_header(path: impl AsRef<Path>) -> Result<AbmHeader, IoError> {
    let mut file = File::open(path)?;
    let mut hdr_bytes = [0u8; ABM_HEADER_SIZE];
    file.read_exact(&mut hdr_bytes)?;
    AbmHeader::from_bytes(&hdr_bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec3, Vec4};
    use std::fs;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("alice_sdf_abm_{}", name));
        path
    }

    /// Build a simple triangle mesh with normals and UVs
    fn make_triangle_mesh() -> Mesh {
        let v0 = Vertex::with_all(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec2::new(0.0, 0.0),
            Vec2::ZERO,
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            [1.0, 1.0, 1.0, 1.0],
            0,
        );
        let v1 = Vertex::with_all(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::ZERO,
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            [1.0, 1.0, 1.0, 1.0],
            0,
        );
        let v2 = Vertex::with_all(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec2::new(0.0, 1.0),
            Vec2::ZERO,
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            [1.0, 1.0, 1.0, 1.0],
            0,
        );
        Mesh {
            vertices: vec![v0, v1, v2],
            indices: vec![0, 1, 2],
        }
    }

    #[test]
    fn test_roundtrip_triangle() {
        let mesh = make_triangle_mesh();
        let path = temp_path("roundtrip_tri.abm");

        save_abm(&mesh, &path).unwrap();
        let loaded = load_abm(&path).unwrap();

        assert_eq!(loaded.vertices.len(), 3);
        assert_eq!(loaded.indices.len(), 3);

        // Verify positions
        for (orig, load) in mesh.vertices.iter().zip(loaded.vertices.iter()) {
            assert!((orig.position - load.position).length() < 1e-6);
            assert!((orig.normal - load.normal).length() < 1e-6);
            assert!((orig.uv - load.uv).length() < 1e-6);
        }

        // Verify indices
        assert_eq!(loaded.indices, vec![0, 1, 2]);

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_roundtrip_with_lods() {
        // LOD 0: full triangle
        let lod0 = make_triangle_mesh();
        // LOD 1: simplified (single point, degenerate)
        let lod1 = Mesh {
            vertices: vec![Vertex::new(Vec3::ZERO, Vec3::Y)],
            indices: vec![0, 0, 0],
        };

        let path = temp_path("roundtrip_lods.abm");
        let distances = vec![10.0];

        save_abm_with_lods(&[lod0.clone(), lod1.clone()], &distances, &path).unwrap();
        let (loaded_meshes, loaded_distances) = load_abm_with_lods(&path).unwrap();

        assert_eq!(loaded_meshes.len(), 2);
        assert_eq!(loaded_distances.len(), 1);
        assert!((loaded_distances[0] - 10.0).abs() < 1e-6);

        // LOD 0 check
        assert_eq!(loaded_meshes[0].vertices.len(), 3);
        assert_eq!(loaded_meshes[0].indices.len(), 3);

        // LOD 1 check
        assert_eq!(loaded_meshes[1].vertices.len(), 1);
        assert_eq!(loaded_meshes[1].indices.len(), 3);

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_crc_integrity() {
        let mesh = make_triangle_mesh();
        let path = temp_path("crc_corrupt.abm");

        save_abm(&mesh, &path).unwrap();

        // Corrupt a byte in the payload (after the 32-byte header)
        let mut data = fs::read(&path).unwrap();
        assert!(data.len() > ABM_HEADER_SIZE + 4);
        data[ABM_HEADER_SIZE + 2] ^= 0xFF;
        fs::write(&path, &data).unwrap();

        // Load should fail with CRC mismatch
        let result = load_abm(&path);
        assert!(
            matches!(result, Err(IoError::CrcMismatch { .. })),
            "Expected CrcMismatch, got {:?}",
            result
        );

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_header_only_read() {
        let mesh = make_triangle_mesh();
        let path = temp_path("header_only.abm");

        save_abm(&mesh, &path).unwrap();

        let header = read_abm_header(&path).unwrap();
        assert_eq!(header.magic, ABM_MAGIC);
        assert_eq!(header.version, ABM_VERSION);
        assert_eq!(header.vertex_count, 3);
        assert_eq!(header.index_count, 3);
        assert_eq!(header.lod_count, 0);
        // Should have normals and UVs flags set
        assert!(header.flags & ABM_FLAG_HAS_NORMALS != 0);
        assert!(header.flags & ABM_FLAG_HAS_UVS != 0);

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_empty_mesh() {
        let mesh = Mesh::new();
        let path = temp_path("empty.abm");

        save_abm(&mesh, &path).unwrap();
        let loaded = load_abm(&path).unwrap();

        assert_eq!(loaded.vertices.len(), 0);
        assert_eq!(loaded.indices.len(), 0);

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_mesh_with_all_attributes() {
        let v0 = Vertex::with_all(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec2::new(0.5, 0.5),
            Vec2::new(0.1, 0.2),
            Vec4::new(0.0, 0.0, 1.0, -1.0),
            [1.0, 0.0, 0.0, 1.0],
            42,
        );
        let v1 = Vertex::with_all(
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.3, 0.4),
            Vec4::new(0.0, 1.0, 0.0, 1.0),
            [0.0, 1.0, 0.0, 0.5],
            7,
        );
        let mesh = Mesh {
            vertices: vec![v0, v1],
            indices: vec![0, 1, 0],
        };
        let path = temp_path("all_attrs.abm");

        save_abm(&mesh, &path).unwrap();

        // Check header flags
        let header = read_abm_header(&path).unwrap();
        assert!(header.flags & ABM_FLAG_HAS_NORMALS != 0);
        assert!(header.flags & ABM_FLAG_HAS_UVS != 0);
        assert!(header.flags & ABM_FLAG_HAS_UV2 != 0);
        assert!(header.flags & ABM_FLAG_HAS_TANGENTS != 0);
        assert!(header.flags & ABM_FLAG_HAS_COLORS != 0);
        assert!(header.flags & ABM_FLAG_HAS_MATERIAL_IDS != 0);

        // Full round-trip
        let loaded = load_abm(&path).unwrap();
        assert_eq!(loaded.vertices.len(), 2);
        for (orig, load) in mesh.vertices.iter().zip(loaded.vertices.iter()) {
            assert!((orig.position - load.position).length() < 1e-6);
            assert!((orig.normal - load.normal).length() < 1e-6);
            assert!((orig.uv - load.uv).length() < 1e-6);
            assert!((orig.uv2 - load.uv2).length() < 1e-6);
            assert!((orig.tangent - load.tangent).length() < 1e-6);
            assert!((orig.color[0] - load.color[0]).abs() < 1e-6);
            assert!((orig.color[1] - load.color[1]).abs() < 1e-6);
            assert!((orig.color[2] - load.color[2]).abs() < 1e-6);
            assert!((orig.color[3] - load.color[3]).abs() < 1e-6);
            assert_eq!(orig.material_id, load.material_id);
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_invalid_magic() {
        let mut bytes = [0u8; ABM_HEADER_SIZE];
        bytes[0..8].copy_from_slice(b"INVALID\0");
        let result = AbmHeader::from_bytes(&bytes);
        assert!(matches!(result, Err(IoError::InvalidFormat(_))));
    }

    #[test]
    fn test_header_roundtrip() {
        let header = AbmHeader {
            magic: ABM_MAGIC,
            version: ABM_VERSION,
            flags: ABM_FLAG_HAS_NORMALS | ABM_FLAG_HAS_UVS,
            vertex_count: 1000,
            index_count: 3000,
            lod_count: 3,
            _reserved: [0u8; 7],
            crc32: 0xDEADBEEF,
        };
        let bytes = header.to_bytes();
        let parsed = AbmHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.magic, ABM_MAGIC);
        assert_eq!(parsed.version, ABM_VERSION);
        assert_eq!(parsed.flags, ABM_FLAG_HAS_NORMALS | ABM_FLAG_HAS_UVS);
        assert_eq!(parsed.vertex_count, 1000);
        assert_eq!(parsed.index_count, 3000);
        assert_eq!(parsed.lod_count, 3);
        assert_eq!(parsed.crc32, 0xDEADBEEF);
    }
}
