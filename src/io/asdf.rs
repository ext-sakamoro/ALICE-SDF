//! ASDF binary format (Deep Fried Edition)
//!
//! Header (16 bytes):
//!   - Magic: "ASDF" (4 bytes)
//!   - Version: u16 (2 bytes)
//!   - Flags: u16 (2 bytes)
//!   - Node count: u32 (4 bytes)
//!   - CRC32: u32 (4 bytes)
//!
//! Body:
//!   - Bincode-serialized SdfTree
//!
//! # Deep Fried Optimizations
//! - **Streaming Write**: Uses `BufWriter` with on-the-fly CRC and seek patching.
//! - **CRC-First Read**: Validates CRC before deserialization (fail-fast on corruption).
//! - **Buffered I/O**: Uses `BufReader`/`BufWriter` for reduced syscalls.
//!
//! Author: Moroya Sakamoto

use crate::io::IoError;
use crate::types::SdfTree;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Magic bytes for ASDF format
pub const ASDF_MAGIC: [u8; 4] = *b"ASDF";

/// Current format version
pub const ASDF_VERSION: u16 = 1;

/// ASDF file header
#[derive(Debug, Clone, Copy)]
pub struct AsdfHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub flags: u16,
    pub node_count: u32,
    pub crc32: u32,
}

impl AsdfHeader {
    /// Create a new header for a tree
    pub fn new(tree: &SdfTree, body_crc: u32) -> Self {
        AsdfHeader {
            magic: ASDF_MAGIC,
            version: ASDF_VERSION,
            flags: 0,
            node_count: tree.node_count(),
            crc32: body_crc,
        }
    }

    /// Serialize header to bytes
    #[inline(always)]
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..6].copy_from_slice(&self.version.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.flags.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.node_count.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.crc32.to_le_bytes());
        bytes
    }

    /// Parse header from bytes
    #[inline(always)]
    pub fn from_bytes(bytes: &[u8; 16]) -> Result<Self, IoError> {
        let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
        if magic != ASDF_MAGIC {
            return Err(IoError::InvalidFormat(format!(
                "Invalid magic bytes: {:?}",
                magic
            )));
        }

        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version > ASDF_VERSION {
            return Err(IoError::UnsupportedVersion(version));
        }

        let flags = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        let node_count = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let crc32 = u32::from_le_bytes(bytes[12..16].try_into().unwrap());

        Ok(AsdfHeader {
            magic,
            version,
            flags,
            node_count,
            crc32,
        })
    }
}

/// Writer wrapper that calculates CRC32 on the fly (Deep Fried)
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

/// Reader wrapper that calculates CRC32 on the fly (Deep Fried)
struct CrcReader<R: Read> {
    inner: R,
    hasher: crc32fast::Hasher,
}

impl<R: Read> CrcReader<R> {
    #[inline(always)]
    fn new(inner: R) -> Self {
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

impl<R: Read> Read for CrcReader<R> {
    #[inline(always)]
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
}

/// Save an SDF tree to binary ASDF format (Streaming Optimized)
///
/// Uses seek patching: writes placeholder header, streams body with CRC,
/// then patches header with actual CRC. Avoids loading entire body into memory.
pub fn save_asdf(tree: &SdfTree, path: impl AsRef<Path>) -> Result<(), IoError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // 1. Write placeholder header (CRC unknown yet)
    let placeholder = AsdfHeader {
        magic: ASDF_MAGIC,
        version: ASDF_VERSION,
        flags: 0,
        node_count: tree.node_count(),
        crc32: 0,
    };
    writer.write_all(&placeholder.to_bytes())?;

    // 2. Stream write body and calculate CRC on-the-fly
    let mut crc_writer = CrcWriter::new(&mut writer);
    bincode::serialize_into(&mut crc_writer, tree)
        .map_err(|e| IoError::Serialization(e.to_string()))?;

    let crc = crc_writer.finalize();

    // 3. Patch header with actual CRC (seek back to start)
    writer.seek(SeekFrom::Start(0))?;
    let real_header = AsdfHeader::new(tree, crc);
    writer.write_all(&real_header.to_bytes())?;

    writer.flush()?;

    Ok(())
}

/// Load an SDF tree from binary ASDF format (CRC-First Validation)
///
/// Reads body into buffer, validates CRC before deserialization.
/// This protects against corrupted data causing bincode to hang.
pub fn load_asdf(path: impl AsRef<Path>) -> Result<SdfTree, IoError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // 1. Read header
    let mut header_bytes = [0u8; 16];
    reader.read_exact(&mut header_bytes)?;
    let header = AsdfHeader::from_bytes(&header_bytes)?;

    // 2. Read entire body into buffer (necessary for CRC validation before deserialize)
    let mut body = Vec::new();
    reader.read_to_end(&mut body)?;

    // 3. Verify CRC BEFORE deserialization (fail-fast on corruption)
    let actual_crc = crc32fast::hash(&body);
    if actual_crc != header.crc32 {
        return Err(IoError::CrcMismatch {
            expected: header.crc32,
            actual: actual_crc,
        });
    }

    // 4. Deserialize (safe because CRC verified)
    let tree: SdfTree =
        bincode::deserialize(&body).map_err(|e| IoError::Serialization(e.to_string()))?;

    Ok(tree)
}

/// Read only the header from an ASDF file
pub fn read_header(path: impl AsRef<Path>) -> Result<AsdfHeader, IoError> {
    let mut file = File::open(path)?;
    let mut header_bytes = [0u8; 16];
    file.read_exact(&mut header_bytes)?;
    AsdfHeader::from_bytes(&header_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;
    use std::fs;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("alice_sdf_asdf_{}", name));
        path
    }

    #[test]
    fn test_header_roundtrip() {
        let tree = SdfTree::new(SdfNode::sphere(1.0));
        let header = AsdfHeader::new(&tree, 0x12345678);

        let bytes = header.to_bytes();
        let parsed = AsdfHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.magic, ASDF_MAGIC);
        assert_eq!(parsed.version, ASDF_VERSION);
        assert_eq!(parsed.node_count, 1);
        assert_eq!(parsed.crc32, 0x12345678);
    }

    #[test]
    fn test_save_load_sphere() {
        let tree = SdfTree::new(SdfNode::sphere(2.5));
        let path = temp_path("sphere.asdf");

        save_asdf(&tree, &path).unwrap();
        let loaded = load_asdf(&path).unwrap();

        assert_eq!(loaded.node_count(), 1);
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_load_complex() {
        let shape = SdfNode::sphere(1.0)
            .subtract(SdfNode::box3d(1.0, 1.0, 1.0))
            .translate(1.0, 2.0, 3.0)
            .twist(0.5);
        let tree = SdfTree::new(shape);
        let path = temp_path("complex.asdf");

        save_asdf(&tree, &path).unwrap();
        let loaded = load_asdf(&path).unwrap();

        assert_eq!(loaded.node_count(), tree.node_count());
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_crc_verification() {
        let tree = SdfTree::new(SdfNode::sphere(1.0));
        let path = temp_path("crc_test.asdf");

        save_asdf(&tree, &path).unwrap();

        // Corrupt the file
        let mut data = fs::read(&path).unwrap();
        if data.len() > 20 {
            data[20] ^= 0xFF; // Flip some bits in the body
        }
        fs::write(&path, &data).unwrap();

        // Should fail CRC check
        let result = load_asdf(&path);
        assert!(matches!(result, Err(IoError::CrcMismatch { .. })));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_invalid_magic() {
        let bytes = [b'X', b'X', b'X', b'X', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let result = AsdfHeader::from_bytes(&bytes);
        assert!(matches!(result, Err(IoError::InvalidFormat(_))));
    }

    #[test]
    fn test_streaming_large_tree() {
        // Test with a moderately complex tree to verify streaming works
        let mut shape = SdfNode::sphere(1.0);
        for i in 0..10 {
            shape = shape.union(SdfNode::box3d(1.0, 1.0, 1.0).translate(i as f32, 0.0, 0.0));
        }
        let tree = SdfTree::new(shape);
        let path = temp_path("streaming_large.asdf");

        save_asdf(&tree, &path).unwrap();
        let loaded = load_asdf(&path).unwrap();

        assert_eq!(loaded.node_count(), tree.node_count());
        fs::remove_file(&path).ok();
    }
}
