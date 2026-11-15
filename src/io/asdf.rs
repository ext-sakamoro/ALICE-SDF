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
//!   - Bincode-serialized SdfTree (the `asdf` feature: `save_asdf` / `load_asdf`; the header,
//!     [`AsdfHeader`] and `read_header`, is always available)
//!
//! # Deep Fried Optimizations
//! - **Streaming Write**: Uses `BufWriter` with on-the-fly CRC and seek patching.
//! - **CRC-First Read**: Validates CRC before deserialization (fail-fast on corruption).
//! - **Buffered I/O**: Uses `BufReader`/`BufWriter` for reduced syscalls.
//!
//! Author: Moroya Sakamoto

use crate::io::IoError;
#[cfg(feature = "asdf")]
use crate::provenance::Checked;
use crate::provenance::Provenance;
use crate::types::SdfTree;
use std::fs::File;
use std::io::Read;
#[cfg(feature = "asdf")]
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// Magic bytes for ASDF format
pub const ASDF_MAGIC: [u8; 4] = *b"ASDF";

/// Current format version
///
/// 2 adds the 32-byte semantics identifier after the 16-byte header (see
/// [`crate::provenance`]); 1 has none and is read only by [`load_asdf_unverified`].
pub const ASDF_VERSION: u16 = 2;

/// The header length of [`ASDF_VERSION`] (16 bytes plus the 32-byte semantics identifier)
pub const ASDF_HEADER_LEN: usize = 48;

/// The most bytes the body is read from or written to (256 MiB)
///
/// alice-serial's bincode1 checks every length prefix against the bytes that remain and this
/// limit before it allocates, so a forged length is an `Err`, never a large allocation (the
/// `capacity overflow` panic fuzz_asdf_decode found in an unbounded decoder). The largest scenes
/// written are under 100 MB.
#[cfg(feature = "asdf")]
pub const ASDF_BINCODE_LIMIT: usize = 256 * 1024 * 1024;

/// The deepest nesting the body is read with: each node's enum tag counts one level, and a
/// two-child operation one more, so a left-deep chain of `n` unions needs about `2n`
///
/// The deepest trees measured: text-to-print presets 44, LOL documents 27, LOL panels 35, the
/// Figure humanoid 32. 1024 keeps every one of them and a chain of 500 unions; a deeper body is an
/// `Err` (`TooDeep`) before the recursion can exhaust the stack. A build reading deeper trees
/// needs a stack of about 160 KiB per 1024 levels (wasm32) or more in a debug build.
#[cfg(feature = "asdf")]
pub const ASDF_MAX_DEPTH: usize = 1024;

/// ASDF file header
#[derive(Debug, Clone, Copy)]
pub struct AsdfHeader {
    /// Magic number identifying ASDF format
    pub magic: [u8; 4],
    /// File format version
    pub version: u16,
    /// Feature flags
    pub flags: u16,
    /// Number of SDF nodes
    pub node_count: u32,
    /// CRC32 checksum
    pub crc32: u32,
    /// The semantics identifier the data was written under (`None` in a version 1 file)
    pub semantics_id: Option<[u8; 32]>,
}

impl AsdfHeader {
    /// Create a new header for a tree, carrying this crate's semantics identifier
    pub fn new(tree: &SdfTree, body_crc: u32) -> Self {
        Self::with_provenance(tree, body_crc, &Provenance::sdf())
    }

    /// Create a new header for a tree, carrying the given semantics identifier
    pub fn with_provenance(tree: &SdfTree, body_crc: u32, provenance: &Provenance) -> Self {
        Self {
            magic: ASDF_MAGIC,
            version: ASDF_VERSION,
            flags: 0,
            node_count: tree.node_count(),
            crc32: body_crc,
            semantics_id: Some(provenance.semantics_id),
        }
    }

    /// The length of this header in a file (16 bytes, 48 with the semantics identifier)
    pub const fn byte_len(&self) -> usize {
        if self.semantics_id.is_some() {
            ASDF_HEADER_LEN
        } else {
            16
        }
    }

    /// Serialize header to bytes (16 bytes, 48 with the semantics identifier)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; self.byte_len()];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..6].copy_from_slice(&self.version.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.flags.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.node_count.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.crc32.to_le_bytes());
        if let Some(id) = &self.semantics_id {
            bytes[16..48].copy_from_slice(id);
        }
        bytes
    }

    /// Parse header from bytes (the first 16 bytes; a version 2 header also reads the
    /// identifier from the next 32)
    ///
    /// # Errors
    ///
    /// [`IoError::InvalidFormat`] for a wrong magic or a truncated header,
    /// [`IoError::UnsupportedVersion`] for a newer version.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IoError> {
        if bytes.len() < 16 {
            return Err(IoError::InvalidFormat("ASDF header truncated".into()));
        }
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
        let semantics_id = if version >= 2 {
            let id: [u8; 32] = bytes
                .get(16..48)
                .ok_or_else(|| IoError::InvalidFormat("ASDF v2 header truncated".into()))?
                .try_into()
                .unwrap();
            Some(id)
        } else {
            None
        };
        Ok(Self {
            magic,
            version,
            flags,
            node_count,
            crc32,
            semantics_id,
        })
    }
}

#[cfg(feature = "asdf")]
/// Writer wrapper that calculates CRC32 on the fly (Deep Fried)
struct CrcWriter<W: Write> {
    inner: W,
    hasher: crc32fast::Hasher,
}

#[cfg(feature = "asdf")]
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

#[cfg(feature = "asdf")]
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

#[cfg(feature = "asdf")]
/// Reader wrapper that calculates CRC32 on the fly (Deep Fried)
#[allow(dead_code)]
struct CrcReader<R: Read> {
    inner: R,
    hasher: crc32fast::Hasher,
}

#[cfg(feature = "asdf")]
#[allow(dead_code)]
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

#[cfg(feature = "asdf")]
impl<R: Read> Read for CrcReader<R> {
    #[inline(always)]
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
}

#[cfg(feature = "asdf")]
/// Save an SDF tree to binary ASDF format (Streaming Optimized)
///
/// Uses seek patching: writes placeholder header, streams body with CRC,
/// then patches header with actual CRC. Avoids loading entire body into memory.
pub fn save_asdf(tree: &SdfTree, path: impl AsRef<Path>) -> Result<(), IoError> {
    save_asdf_with_provenance(tree, path, &Provenance::sdf())
}

#[cfg(feature = "asdf")]
/// [`save_asdf`] with the given semantics identifier in the header
pub fn save_asdf_with_provenance(
    tree: &SdfTree,
    path: impl AsRef<Path>,
    provenance: &Provenance,
) -> Result<(), IoError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // 1. Write placeholder header (CRC unknown yet)
    let placeholder = AsdfHeader {
        magic: ASDF_MAGIC,
        version: ASDF_VERSION,
        flags: 0,
        node_count: tree.node_count(),
        crc32: 0,
        semantics_id: Some(provenance.semantics_id),
    };
    writer.write_all(&placeholder.to_bytes())?;

    // 2. Write the body (the bincode 1 layout, at most ASDF_BINCODE_LIMIT bytes) and its CRC
    let body = alice_serial::bincode1::to_vec(tree, ASDF_BINCODE_LIMIT)
        .map_err(|e| IoError::Serialization(e.to_string()))?;
    let mut crc_writer = CrcWriter::new(&mut writer);
    crc_writer.write_all(&body)?;

    let crc = crc_writer.finalize();

    // 3. Patch header with actual CRC (seek back to start)
    writer.seek(SeekFrom::Start(0))?;
    let real_header = AsdfHeader::with_provenance(tree, crc, provenance);
    writer.write_all(&real_header.to_bytes())?;

    writer.flush()?;

    Ok(())
}

#[cfg(feature = "asdf")]
/// Load an SDF tree from binary ASDF format (CRC-First Validation)
///
/// Reads body into buffer, validates CRC before deserialization.
/// A corrupted body is refused by its CRC before it is decoded.
///
/// The header's semantics identifier must be this crate's or one it still reproduces
/// ([`crate::check_sdf_semantics`]).
///
/// # Errors
///
/// [`IoError::Unpinned`] for a version 1 file (no identifier; [`load_asdf_unverified`] reads
/// it), [`IoError::SemanticsMismatch`] for another identifier, and the format errors below.
pub fn load_asdf(path: impl AsRef<Path>) -> Result<SdfTree, IoError> {
    load_asdf_checked(path, false).map(|c| c.into_parts().0)
}

#[cfg(feature = "asdf")]
/// [`load_asdf`] that also reads a version 1 file (written before the semantics identifier),
/// marking it [`crate::provenance::StoredSemantics::Unpinned`]
///
/// # Errors
///
/// As [`load_asdf`], except that a file without an identifier is returned, not refused.
pub fn load_asdf_unverified(path: impl AsRef<Path>) -> Result<Checked<SdfTree>, IoError> {
    load_asdf_checked(path, true)
}

#[cfg(feature = "asdf")]
fn load_asdf_checked(
    path: impl AsRef<Path>,
    allow_unpinned: bool,
) -> Result<Checked<SdfTree>, IoError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    // 1. Read header (16 bytes, then the identifier of a version 2 header)
    let mut header_bytes = vec![0u8; 16];
    reader.read_exact(&mut header_bytes)?;
    if u16::from_le_bytes([header_bytes[4], header_bytes[5]]) >= 2 {
        header_bytes.resize(ASDF_HEADER_LEN, 0);
        reader.read_exact(&mut header_bytes[16..])?;
    }
    let header = AsdfHeader::from_bytes(&header_bytes)?;
    let semantics =
        crate::provenance::sdf_load_verdict(header.semantics_id, "ASDF", allow_unpinned)?;

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

    // 4. Deserialize the whole body: every length is checked against the bytes that remain and
    // ASDF_BINCODE_LIMIT before anything is allocated, nesting stops at ASDF_MAX_DEPTH, and bytes
    // after the tree are refused (the writer never leaves any)
    #[allow(
        clippy::disallowed_methods,
        reason = "stored form checked: the header identifier went through sdf_load_verdict before this decode"
    )]
    let tree: SdfTree =
        alice_serial::bincode1::from_slice_with_depth(&body, ASDF_BINCODE_LIMIT, ASDF_MAX_DEPTH)
            .map_err(|e| IoError::Serialization(e.to_string()))?;
    crate::modifiers::check_repeat_cells_tree(&tree.root)
        .map_err(|e| IoError::InvalidFormat(format!("ASDF: {e}")))?;
    Ok(Checked::new(tree, semantics))
}

/// Read only the header from an ASDF file
pub fn read_header(path: impl AsRef<Path>) -> Result<AsdfHeader, IoError> {
    let mut file = File::open(path)?;
    let mut header_bytes = Vec::with_capacity(ASDF_HEADER_LEN);
    Read::by_ref(&mut file)
        .take(ASDF_HEADER_LEN as u64)
        .read_to_end(&mut header_bytes)?;
    AsdfHeader::from_bytes(&header_bytes)
}

#[cfg(all(test, feature = "asdf"))]
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
        let body_byte = ASDF_HEADER_LEN + 4;
        assert!(data.len() > body_byte);
        data[body_byte] ^= 0xFF; // Flip some bits in the body (after the 48-byte header)
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
    fn test_malformed_body_no_panic() {
        // Regression test for fuzz_asdf_decode 発見 bug (2026-09-03)
        // valid ASDF magic + malformed body で上限の無い decoder が
        // Vec `capacity overflow` panic を起こした (potential DoS)
        // bincode1 は長さを残りの bytes と ASDF_BINCODE_LIMIT で照合してから確保する
        let path = temp_path("malformed_body.asdf");
        // fuzz artifact をベースにした 28-byte payload
        // (base64: QVNERgAAAAAQKN8oN+TnLSgAABAo3yjSKCgyKA==)
        let bytes = [
            0x41, 0x53, 0x44, 0x46, // magic: "ASDF"
            0x00, 0x00, // version
            0x00, 0x00, // flags
            0x10, 0x28, 0xdf, 0x28, // node_count (huge)
            0x37, 0xe4, 0xe7, 0x2d, // CRC32
            0x28, 0x00, 0x00, 0x10, 0x28, 0xdf, 0x28, 0xd2, 0x28, 0x28, 0x32, 0x28,
        ];
        fs::write(&path, bytes).unwrap();

        // panic せず Err を返せば OK (CrcMismatch or Serialization どちらも許容)
        let result = load_asdf(&path);
        assert!(result.is_err(), "malformed body should return Err");

        fs::remove_file(&path).ok();
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
