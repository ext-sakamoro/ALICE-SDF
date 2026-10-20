//! Provenance: the identifier of the meaning of the bytes a writer stores (ALICE-Law G8)
//!
//! Every durable format this crate writes carries a 32-byte semantics identifier, so a file read
//! back later (or by another program) can be checked against the arithmetic that produced it.
//! The plain writers store [`Provenance::sdf`] (this crate's [`crate::SDF_SEMANTICS_ID`]); a caller
//! whose own identifier folds more meaning in (a language that turns its programs into meshes)
//! passes its identifier through the `*_with_provenance` writers.
//!
//! | format | where the identifier is stored |
//! |---|---|
//! | binary STL | the 80-byte header: `ALICESDF`, version byte `1`, the 32 raw identifier bytes, zeros |
//! | ASCII STL | the solid name: `solid alice:semantics_id=<64 hex>` |
//! | 3MF | `<metadata name="alice:semantics_id">` in `3D/3dmodel.model`, namespace [`NAMESPACE_3MF`] |
//! | FBX (binary and ASCII) | the `Creator` string of `FBXHeaderExtension`: `ALICE-SDF alice:semantics_id=<64 hex>` |
//! | OBJ (and its MTL) | a header comment: `# alice:semantics_id=<64 hex>` |
//! | PLY (binary and ASCII) | a header comment: `comment alice:semantics_id=<64 hex>` |
//! | glTF (`.glb`, `.gltf`, meshopt-compressed `.glb`) | `asset.extras`: `{"alice:semantics_id": "<64 hex>"}` |
//! | ASDF JSON (`.asdf.json`, [`crate::io::save_asdf_json`]) and the other JSON documents of this crate (the LOD chain sidecar `.abm.meta.json`) | the top-level field `"alice:semantics_id": "<64 hex>"` next to the document's fields |
//! | Alembic (`.abc`, [`crate::io::alembic::export_alembic`]) | the archive metadata (the fourth child of the root group, `key=value;...`): `alice:semantics_id=<64 hex>` |
//! | USD (`.usda`) | the layer's `customLayerData`: `string "alice:semantics_id" = "<64 hex>"` |
//! | STEP | the second string of `FILE_DESCRIPTION`: `'alice:semantics_id=<64 hex>'` |
//! | IGES | two Start-section lines, `alice:semantics_id=` and the 64 hex digits (joined they read as one) |
//! | ASDF (`.asdf`, version 2) | the 32 bytes after the 16-byte header; this crate's loaders refuse another identifier and, by default, a version 1 file without one |
//! | ABM (`.abm`, version 2) | the 32 bytes after the 32-byte header; this crate's loaders refuse another identifier and, by default, a version 1 file without one |
//! | DDS 3D texture | the header's reserved dwords: tag `ALSD`, version `1`, the 32 identifier bytes |
//! | raw volume data (no header) | a sidecar file `<path>.semantics`: `alice:semantics_id=<64 hex>` |
//!
//! [`read`] finds the identifier in a file of any of these formats, and [`check`] compares it with
//! the expected one (or an earlier one the caller has verified) and names both on a mismatch.
//!
//! A caller that turns its own programs into meshes (its identifier folds more meaning than this
//! crate's) can store its identifier as a second, *producer* identifier next to this crate's
//! through the `*_with_stamp` writers ([`Stamp`]); [`read_with_producer`] returns both. Without a
//! producer the bytes are exactly those of the `*_with_provenance` writers.
//!
//! | format | where the producer identifier is stored |
//! |---|---|
//! | binary STL | header version `2` (instead of `1`), bytes `41..48` zero, bytes `48..80` the producer's 32 raw bytes |
//! | ASCII STL | the solid name continues: ` alice:producer_semantics_id=<64 hex>` |
//! | 3MF | a second `<metadata name="alice:producer_semantics_id">` in `3D/3dmodel.model` |
//! | FBX (binary and ASCII) | a `ProducerSemanticsId` string in `FBXHeaderExtension`: `alice:producer_semantics_id=<64 hex>` (not in `Creator`) |

use std::fmt;
use std::path::Path;

/// The key under which the identifier is stored in text formats
pub const KEY: &str = "alice:semantics_id";

/// The XML namespace of the 3MF metadata name (`xmlns:alice`)
pub const NAMESPACE_3MF: &str = "https://alicelaw.net/ns/3mf/provenance/2026";

/// The first 8 bytes of a binary STL header written by this crate
pub const STL_MAGIC: &[u8; 8] = b"ALICESDF";

/// The layout version of the binary STL header
pub const STL_HEADER_VERSION: u8 = 1;

/// The layout version of a binary STL header that also carries a producer identifier
pub const STL_HEADER_VERSION_PRODUCER: u8 = 2;

/// The key under which a producer identifier is stored in text formats
pub const PRODUCER_KEY: &str = "alice:producer_semantics_id";

/// The identifier of the meaning of the stored bytes
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Provenance {
    /// 32-byte semantics identifier
    pub semantics_id: [u8; 32],
}

impl Provenance {
    /// This crate's own identifier ([`crate::SDF_SEMANTICS_ID`])
    #[must_use]
    pub const fn sdf() -> Self {
        Self {
            semantics_id: crate::SDF_SEMANTICS_ID,
        }
    }

    /// A caller's identifier
    #[must_use]
    pub const fn new(semantics_id: [u8; 32]) -> Self {
        Self { semantics_id }
    }

    /// Lower-case hex of the identifier (64 characters)
    #[must_use]
    pub fn hex(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::with_capacity(64);
        for b in self.semantics_id {
            let _ = write!(s, "{b:02x}");
        }
        s
    }

    /// `alice:semantics_id=<64 hex>`, the form text formats store
    #[must_use]
    pub fn tagged(&self) -> String {
        format!("{KEY}={}", self.hex())
    }

    /// The 80-byte binary STL header carrying this identifier
    #[must_use]
    pub fn stl_header(&self) -> [u8; 80] {
        let mut h = [0u8; 80];
        h[..8].copy_from_slice(STL_MAGIC);
        h[8] = STL_HEADER_VERSION;
        h[9..41].copy_from_slice(&self.semantics_id);
        h
    }

    pub(crate) fn from_hex(s: &str) -> Option<Self> {
        let s = s.as_bytes();
        if s.len() != 64 {
            return None;
        }
        let mut id = [0u8; 32];
        for (i, pair) in s.chunks_exact(2).enumerate() {
            let hi = (pair[0] as char).to_digit(16)?;
            let lo = (pair[1] as char).to_digit(16)?;
            id[i] = u8::try_from(hi * 16 + lo).ok()?;
        }
        Some(Self::new(id))
    }
}

impl fmt::Debug for Provenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Provenance({})", self.hex())
    }
}

impl fmt::Display for Provenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.hex())
    }
}

/// What a `*_with_stamp` writer stores
///
/// `primary` is the identifier of the meaning of the bytes, stored where the `*_with_provenance`
/// writers store theirs; `producer` is optionally the identifier of the program that produced them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Stamp {
    /// Stored where the `*_with_provenance` writers store their identifier; [`check`] reads it
    pub primary: Provenance,
    /// Stored in a separate slot; `None` writes the same bytes as `*_with_provenance(primary)`
    pub producer: Option<Provenance>,
}

impl Stamp {
    /// A stamp with this crate's identifier as `primary` and `producer` beside it
    #[must_use]
    pub const fn sdf_with_producer(producer: Provenance) -> Self {
        Self {
            primary: Provenance::sdf(),
            producer: Some(producer),
        }
    }

    /// A stamp from both parts
    #[must_use]
    pub const fn new(primary: Provenance, producer: Option<Provenance>) -> Self {
        Self { primary, producer }
    }

    /// The 80-byte binary STL header: version `1` without a producer (the bytes of
    /// [`Provenance::stl_header`]), version `2` with the producer in bytes `48..80`
    #[must_use]
    pub fn stl_header(&self) -> [u8; 80] {
        let mut h = self.primary.stl_header();
        if let Some(producer) = self.producer {
            h[8] = STL_HEADER_VERSION_PRODUCER;
            h[48..80].copy_from_slice(&producer.semantics_id);
        }
        h
    }

    /// The text form: `alice:semantics_id=<hex>`, then ` alice:producer_semantics_id=<hex>` when a
    /// producer is present
    #[must_use]
    pub fn tagged(&self) -> String {
        match self.producer {
            None => self.primary.tagged(),
            Some(p) => format!("{} {PRODUCER_KEY}={}", self.primary.tagged(), p.hex()),
        }
    }
}

impl From<Provenance> for Stamp {
    fn from(primary: Provenance) -> Self {
        Self::new(primary, None)
    }
}

/// Reading or checking a stored identifier failed
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ProvenanceError {
    /// The file could not be read
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// The file is in a format this module reads but carries no identifier
    #[error("no {KEY} in {format}")]
    Missing {
        /// The format recognised
        format: &'static str,
    },
    /// The format is not one this module reads, or its identifier cannot be read
    #[error("cannot read an identifier: {0}")]
    Unreadable(String),
    /// The stored identifier is not the expected one
    #[error("{KEY} mismatch: stored {stored}, expected {expected}")]
    Mismatch {
        /// The identifier in the file
        stored: Provenance,
        /// The identifier the caller expected
        expected: Provenance,
    },
    /// The file carries no producer identifier where one was expected
    #[error("no {PRODUCER_KEY} in the file")]
    ProducerMissing,
    /// The stored producer identifier is not the expected one
    #[error("{PRODUCER_KEY} mismatch: stored {stored}, expected {expected}")]
    ProducerMismatch {
        /// The producer identifier in the file
        stored: Provenance,
        /// The producer identifier the caller expected
        expected: Provenance,
    },
}

/// The identifier stored in the file at `path`
///
/// # Errors
///
/// [`ProvenanceError::Missing`] when the format is recognised but carries no identifier,
/// [`ProvenanceError::Unreadable`] when the format is not recognised (or a 3MF entry is
/// compressed), [`ProvenanceError::Io`] when the file cannot be read.
pub fn read(path: impl AsRef<Path>) -> Result<Provenance, ProvenanceError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    match from_bytes(&bytes) {
        Err(ProvenanceError::Unreadable(why)) => {
            // a format with no place for the identifier (raw volume data, .splat) has it in a
            // sidecar; one that binds it to the file's SHA-256 is checked against these bytes
            let sidecar = sidecar_path(path);
            if !sidecar.is_file() {
                return Err(ProvenanceError::Unreadable(why));
            }
            let text = std::fs::read(&sidecar)?;
            // the bound form (identifier + the file's SHA-256) is the default, as load_splat
            // demands; only the raw volume exports keep the plain sidecar, so a .splat under any
            // name with its SHA-256 line removed is refused here too
            let plain = path.extension().is_some_and(|e| {
                PLAIN_SIDECAR_EXTENSIONS
                    .iter()
                    .any(|x| e.eq_ignore_ascii_case(x))
            });
            if !plain || find(&text, CONTENT_SHA256_KEY.as_bytes()).is_some() {
                match bound_sidecar_id(path, &bytes) {
                    Ok(Some(id)) => Ok(Provenance::new(id)),
                    Ok(None) => Err(ProvenanceError::Unreadable(
                        "sidecar: removed while reading".into(),
                    )),
                    Err(e) => Err(ProvenanceError::Unreadable(format!("sidecar: {e}"))),
                }
            } else {
                tagged_in(&text, "sidecar")
            }
        }
        other => other,
    }
}

/// The RIFF chunk id of the identifier in a MagicaVoxel `.vox` file: a child of `MAIN` holding a
/// layout version (u32 = 1) and the 32-byte identifier; readers that do not know it skip it
pub const VOX_CHUNK: &[u8; 4] = b"ALSD";

/// The identifier in the `ALSD` chunk of a `.vox` file: `Ok(None)` when there is none (a file of
/// another program, or written before identifiers were stored); an `ALSD` chunk that is cut short,
/// has another length or layout version, or occurs twice is an error, never "no identifier"
pub(crate) fn vox_stored_id(bytes: &[u8]) -> Result<Option<[u8; 32]>, String> {
    let mut found: Option<[u8; 32]> = None;
    let mut off = vox_main_children(bytes)?;
    while off < bytes.len() {
        let rest = &bytes[off..];
        if rest.len() < 12 {
            if rest.starts_with(VOX_CHUNK) || (rest.len() < 4 && VOX_CHUNK.starts_with(rest)) {
                return Err("ALSD chunk: truncated header".into());
            }
            return Err(format!("chunk header at byte {off} is cut short"));
        }
        let id = &rest[0..4];
        let content = u32::from_le_bytes([rest[4], rest[5], rest[6], rest[7]]) as usize;
        let children = u32::from_le_bytes([rest[8], rest[9], rest[10], rest[11]]) as usize;
        let end = 12usize.saturating_add(content).saturating_add(children);
        if id == VOX_CHUNK {
            if found.is_some() {
                return Err("more than one ALSD chunk".into());
            }
            if content != 36 || children != 0 {
                return Err(format!(
                    "ALSD chunk: length {content} (children {children}), expected 36"
                ));
            }
            let body = rest
                .get(12..48)
                .ok_or_else(|| String::from("ALSD chunk: truncated body"))?;
            let layout = u32::from_le_bytes([body[0], body[1], body[2], body[3]]);
            if layout != 1 {
                return Err(format!("ALSD chunk: layout version {layout} is not known"));
            }
            let mut id = [0u8; 32];
            id.copy_from_slice(&body[4..36]);
            found = Some(id);
        }
        if end > rest.len() {
            return Err(format!(
                "chunk {} at byte {off} overruns the file",
                String::from_utf8_lossy(id)
            ));
        }
        off += end;
    }
    Ok(found)
}

/// The offset of MAIN's first child in a `.vox` file, after checking that MAIN's declared children
/// size reaches exactly the end of the file (MagicaVoxel writes it so): a file cut short, or one
/// with bytes after MAIN, is an error, so a chunk that is missing is never read as never written
pub(crate) fn vox_main_children(bytes: &[u8]) -> Result<usize, String> {
    if bytes.len() < 8 || &bytes[0..4] != b"VOX " {
        return Err("not a VOX file".into());
    }
    if bytes.len() < 20 || &bytes[8..12] != b"MAIN" {
        return Err("missing MAIN".into());
    }
    let content = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let children = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
    let start = 20usize.saturating_add(content);
    let have = bytes.len().saturating_sub(start);
    if start > bytes.len() || children != have {
        return Err(format!(
            "MAIN children size {children}, the file has {have} bytes after MAIN's header"
        ));
    }
    Ok(start)
}

/// The identifier stored in the bytes of a file
///
/// # Errors
///
/// As [`read`].
pub fn from_bytes(bytes: &[u8]) -> Result<Provenance, ProvenanceError> {
    if bytes.starts_with(b"PK\x03\x04") {
        return read_3mf(bytes);
    }
    if bytes.starts_with(b"Kaydara FBX Binary") {
        return tagged_in(bytes, "FBX");
    }
    if bytes.starts_with(b"; FBX") {
        return tagged_in(bytes, "FBX");
    }
    if bytes.starts_with(b"ASDF") {
        // ASDF: version 2 and later carry the identifier after the 16-byte header
        let version = bytes
            .get(4..6)
            .map(|v| u16::from_le_bytes([v[0], v[1]]))
            .ok_or_else(|| ProvenanceError::Unreadable("ASDF: truncated header".into()))?;
        if version < 2 {
            return Err(ProvenanceError::Missing { format: "ASDF" });
        }
        let id: [u8; 32] = bytes
            .get(16..48)
            .ok_or_else(|| ProvenanceError::Unreadable("ASDF: truncated header".into()))?
            .try_into()
            .unwrap();
        return Ok(Provenance::new(id));
    }
    if bytes.starts_with(PNG_SIGNATURE) {
        return png_tagged(bytes);
    }
    if bytes.starts_with(JPEG_SOI) {
        return jpeg_tagged(bytes);
    }
    if bytes.starts_with(b"DDS ") {
        // DDS: the identifier block sits in the header's reserved dwords (bytes 32..76)
        let r = bytes
            .get(32..76)
            .ok_or_else(|| ProvenanceError::Unreadable("DDS: truncated header".into()))?;
        if &r[..4] != DDS_TAG || r[4..8] != 1u32.to_le_bytes() {
            return Err(ProvenanceError::Missing { format: "DDS" });
        }
        let mut id = [0u8; 32];
        id.copy_from_slice(&r[8..40]);
        return Ok(Provenance::new(id));
    }
    if bytes.starts_with(b"ALICEBM\0") {
        // ABM: version 2 carries the identifier after the 32-byte header; a version this crate
        // does not know is refused rather than read at the version 2 offset
        let version = bytes
            .get(8..10)
            .map(|v| u16::from_le_bytes([v[0], v[1]]))
            .ok_or_else(|| ProvenanceError::Unreadable("ABM: truncated header".into()))?;
        return match version {
            1 => Err(ProvenanceError::Missing { format: "ABM" }),
            2 => {
                let id: [u8; 32] = bytes
                    .get(32..64)
                    .ok_or_else(|| ProvenanceError::Unreadable("ABM: truncated header".into()))?
                    .try_into()
                    .unwrap();
                Ok(Provenance::new(id))
            }
            v => Err(ProvenanceError::Unreadable(format!(
                "ABM: version {v} is not known"
            ))),
        };
    }
    if bytes.starts_with(b"NSDF") {
        // NSDF (neural weights): version 2 carries the identifier after the version
        let version = bytes
            .get(4..8)
            .map(|v| u32::from_le_bytes([v[0], v[1], v[2], v[3]]))
            .ok_or_else(|| ProvenanceError::Unreadable("NSDF: truncated header".into()))?;
        return match version {
            1 => Err(ProvenanceError::Missing { format: "NSDF" }),
            2 => {
                let id: [u8; 32] = bytes
                    .get(8..40)
                    .ok_or_else(|| ProvenanceError::Unreadable("NSDF: truncated header".into()))?
                    .try_into()
                    .unwrap();
                Ok(Provenance::new(id))
            }
            v => Err(ProvenanceError::Unreadable(format!(
                "NSDF: version {v} is not known"
            ))),
        };
    }
    if bytes.starts_with(b"Ogawa") {
        return read_alembic(bytes);
    }
    if bytes.starts_with(b"ALICEVDB") {
        // ALICE-VDB dense grid: `ALICEVDB2` carries the identifier after the 9-byte magic
        return match bytes.get(8) {
            Some(b'1') => Err(ProvenanceError::Missing {
                format: "ALICE-VDB",
            }),
            Some(b'2') => {
                let id: [u8; 32] = bytes
                    .get(9..41)
                    .ok_or_else(|| {
                        ProvenanceError::Unreadable("ALICE-VDB: truncated header".into())
                    })?
                    .try_into()
                    .unwrap();
                Ok(Provenance::new(id))
            }
            Some(&v) => Err(ProvenanceError::Unreadable(format!(
                "ALICE-VDB: version {:?} is not known",
                char::from(v)
            ))),
            None => Err(ProvenanceError::Unreadable(
                "ALICE-VDB: truncated header".into(),
            )),
        };
    }
    if bytes.starts_with(b"ASVC") {
        // SVO chunk wire form: version 2 carries the identifier after the version (version 1 has
        // no magic and is not recognised here)
        let version = bytes
            .get(4..8)
            .map(|v| u32::from_le_bytes([v[0], v[1], v[2], v[3]]))
            .ok_or_else(|| ProvenanceError::Unreadable("SVO chunk: truncated header".into()))?;
        if version != 2 {
            return Err(ProvenanceError::Unreadable(format!(
                "SVO chunk: version {version} is not known"
            )));
        }
        let mut id = [0u8; 32];
        id.copy_from_slice(
            bytes
                .get(8..40)
                .ok_or_else(|| ProvenanceError::Unreadable("SVO chunk: truncated header".into()))?,
        );
        return Ok(Provenance::new(id));
    }
    if bytes.starts_with(b"VOX ") {
        // MagicaVoxel: the private `ALSD` chunk among MAIN's children
        return match vox_stored_id(bytes) {
            Ok(Some(id)) => Ok(Provenance::new(id)),
            Ok(None) => Err(ProvenanceError::Missing { format: "VOX" }),
            Err(e) => Err(ProvenanceError::Unreadable(format!("VOX: {e}"))),
        };
    }
    if bytes.starts_with(b"glTF") {
        // GLB: 12-byte header, then the JSON chunk (length, type `JSON`, data)
        let len = bytes
            .get(12..16)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
            .ok_or_else(|| ProvenanceError::Unreadable("GLB: truncated header".into()))?;
        if bytes.get(16..20) != Some(b"JSON") {
            return Err(ProvenanceError::Unreadable(
                "GLB: first chunk is not JSON".into(),
            ));
        }
        let json = bytes
            .get(20..20 + len)
            .ok_or_else(|| ProvenanceError::Unreadable("GLB: truncated JSON chunk".into()))?;
        return json_tagged(json, "GLB");
    }
    if bytes.first() == Some(&b'{') {
        return json_tagged(bytes, "glTF");
    }
    if bytes.starts_with(b"#usda") {
        // USDA: the layer metadata (the parenthesised block after the first line) holds
        // `customLayerData = { string "alice:semantics_id" = "<hex>" }`
        let end = find(bytes, b"\n)")
            .ok_or_else(|| ProvenanceError::Unreadable("USDA: no layer metadata block".into()))?;
        let head = std::str::from_utf8(&bytes[..end])
            .map_err(|_| ProvenanceError::Unreadable("USDA: header is not UTF-8".into()))?;
        let key = format!("string \"{KEY}\" = \"");
        return match head.find(&key) {
            None => Err(ProvenanceError::Missing { format: "USDA" }),
            Some(at) => head
                .get(at + key.len()..at + key.len() + 64)
                .and_then(Provenance::from_hex)
                .ok_or_else(|| {
                    ProvenanceError::Unreadable(format!("USDA: {KEY} is not 64 hex digits"))
                }),
        };
    }
    if bytes.starts_with(b"ISO-10303-21;") {
        // STEP: the identifier is one of the strings of FILE_DESCRIPTION's description list
        let start = find(bytes, b"FILE_DESCRIPTION((")
            .ok_or(ProvenanceError::Missing { format: "STEP" })?
            + b"FILE_DESCRIPTION((".len();
        let len = find(&bytes[start..], b"),").ok_or_else(|| {
            ProvenanceError::Unreadable("STEP: FILE_DESCRIPTION not closed".into())
        })?;
        return tagged_in(&bytes[start..start + len], "STEP");
    }
    if is_iges_start(bytes) {
        // IGES: the Start section's text (columns 1-72 of the `S` lines) joined
        let mut joined = Vec::new();
        for line in bytes.split(|&b| b == b'\n') {
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if line.len() < 73 || line[72] != b'S' {
                break;
            }
            let text = &line[..72];
            let trimmed = text
                .iter()
                .rposition(|&c| c != b' ')
                .map_or(&text[..0], |i| &text[..=i]);
            joined.extend_from_slice(trimmed);
        }
        return tagged_in(&joined, "IGES");
    }
    if bytes.starts_with(b"ply\n") || bytes.starts_with(b"ply\r\n") {
        // the header ends at `end_header`; the identifier is one of its comment lines
        let end = find(bytes, b"end_header")
            .ok_or_else(|| ProvenanceError::Unreadable("PLY: no end_header".into()))?;
        return tagged_in(&bytes[..end], "PLY");
    }
    if bytes.starts_with(b"#") {
        // OBJ (and MTL): the identifier is one of the leading comment lines
        let mut end = 0;
        for line in bytes.split_inclusive(|&b| b == b'\n') {
            if !line.starts_with(b"#") {
                break;
            }
            end += line.len();
        }
        return tagged_in(&bytes[..end], "OBJ");
    }
    if bytes.starts_with(b"solid") {
        let first = bytes.split(|&b| b == b'\n').next().unwrap_or(&[]);
        return tagged_in(first, "ASCII STL");
    }
    if bytes.len() >= 84 {
        let count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        if count.checked_mul(50).and_then(|n| n.checked_add(84)) == Some(bytes.len()) {
            if &bytes[..8] == STL_MAGIC {
                // this crate's header: a version it does not know, or a version 2 header whose
                // padding is not zero, is refused rather than read as a file without one
                if bytes[8] == STL_HEADER_VERSION
                    || (bytes[8] == STL_HEADER_VERSION_PRODUCER && bytes[41..48] == [0; 7])
                {
                    let mut id = [0u8; 32];
                    id.copy_from_slice(&bytes[9..41]);
                    return Ok(Provenance::new(id));
                }
                return Err(ProvenanceError::Unreadable(format!(
                    "binary STL: header version {} (with this padding) is not one this crate knows",
                    bytes[8]
                )));
            }
            return Err(ProvenanceError::Missing {
                format: "binary STL",
            });
        }
    }
    Err(ProvenanceError::Unreadable("unrecognised format".into()))
}

/// The identifier and the producer identifier stored in the file at `path`
///
/// The first is what [`read`] returns; the second is `None` when the file carries no producer
/// identifier.
///
/// # Errors
///
/// As [`read`], and [`ProvenanceError::Unreadable`] when a producer slot is malformed.
pub fn read_with_producer(
    path: impl AsRef<Path>,
) -> Result<(Provenance, Option<Provenance>), ProvenanceError> {
    let path = path.as_ref();
    let primary = read(path)?;
    let producer = producer_in(&std::fs::read(path)?)?;
    Ok((primary, producer))
}

/// As [`read_with_producer`], from the bytes of a file
///
/// # Errors
///
/// As [`from_bytes`], and [`ProvenanceError::Unreadable`] when a producer slot is malformed.
pub fn from_bytes_with_producer(
    bytes: &[u8],
) -> Result<(Provenance, Option<Provenance>), ProvenanceError> {
    Ok((from_bytes(bytes)?, producer_in(bytes)?))
}

/// Check the producer identifier of the file at `path` against `expected`
///
/// The primary identifier is [`check`]'s; this compares only the producer slot: `Ok(())` when it
/// holds `expected`.
///
/// # Errors
///
/// [`ProvenanceError::ProducerMissing`] when the file carries no producer identifier,
/// [`ProvenanceError::ProducerMismatch`] (naming both) when it carries another one, and the
/// errors of [`read_with_producer`].
pub fn check_producer(
    path: impl AsRef<Path>,
    expected: &Provenance,
) -> Result<(), ProvenanceError> {
    match read_with_producer(path)?.1 {
        None => Err(ProvenanceError::ProducerMissing),
        Some(stored) if stored == *expected => Ok(()),
        Some(stored) => Err(ProvenanceError::ProducerMismatch {
            stored,
            expected: *expected,
        }),
    }
}

/// The producer identifier in the bytes of a file, if any
fn producer_in(bytes: &[u8]) -> Result<Option<Provenance>, ProvenanceError> {
    if bytes.starts_with(b"PK\x03\x04") {
        let Some(model) = zip_entry(bytes, "3D/3dmodel.model")? else {
            return Ok(None);
        };
        let text = std::str::from_utf8(model)
            .map_err(|_| ProvenanceError::Unreadable("3MF: model is not UTF-8".into()))?;
        let open = format!("<metadata name=\"{PRODUCER_KEY}\">");
        let mut values = text.match_indices(&open).map(|(at, _)| {
            let start = at + open.len();
            text.get(start..start + 64)
                .and_then(Provenance::from_hex)
                .filter(|_| text[start + 64..].starts_with("</metadata>"))
        });
        return match (values.next(), values.next()) {
            (None, _) => Ok(None),
            (Some(Some(p)), None) => Ok(Some(p)),
            _ => Err(ProvenanceError::Unreadable(format!(
                "3MF: {PRODUCER_KEY} is malformed or repeated"
            ))),
        };
    }
    if !bytes.starts_with(b"solid") && bytes.len() >= 84 && &bytes[..8] == STL_MAGIC {
        // binary STL written by this crate: version 2 carries the producer in bytes 48..80
        return Ok((bytes[8] == STL_HEADER_VERSION_PRODUCER).then(|| {
            let mut id = [0u8; 32];
            id.copy_from_slice(&bytes[48..80]);
            Provenance::new(id)
        }));
    }
    // text and FBX: the tagged form, at most one distinct value
    let key = format!("{PRODUCER_KEY}=");
    let key = key.as_bytes();
    let mut found: Option<Provenance> = None;
    let mut i = 0;
    while i + key.len() + 64 <= bytes.len() {
        if &bytes[i..i + key.len()] == key {
            let start = i + key.len();
            let p = std::str::from_utf8(&bytes[start..start + 64])
                .ok()
                .and_then(Provenance::from_hex)
                .ok_or_else(|| ProvenanceError::Unreadable(format!("{PRODUCER_KEY} is not hex")))?;
            match found {
                Some(q) if q != p => {
                    return Err(ProvenanceError::Unreadable(format!(
                        "two different {PRODUCER_KEY} values"
                    )))
                }
                _ => found = Some(p),
            }
            i = start + 64;
        } else {
            i += 1;
        }
    }
    Ok(found)
}

/// What a read knows about the arithmetic stored data was written under
///
/// The same three cases as the stores of the other ALICE crates (`Current` / `VerifiedPrevious` /
/// `Unpinned`): [`check`] returns the first two, and this crate's own loaders return
/// `Unpinned` only through their explicit `*_unverified` variant, wrapped in [`Checked`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StoredSemantics {
    /// The current identifier
    Current,
    /// An earlier identifier that is verified to give the same bits
    VerifiedPrevious(Provenance),
    /// No identifier: the data was written before identifiers were stored, so nothing says which
    /// arithmetic it was written under
    Unpinned,
}

impl StoredSemantics {
    /// The weaker of two statuses, for a value read from more than one stored part (a mesh file
    /// and its metadata sidecar): `Unpinned` < `VerifiedPrevious` < `Current`. Of two
    /// `VerifiedPrevious` with different identifiers, the receiver's is returned.
    ///
    /// The match names every variant, so a variant added later does not compile here until its
    /// place in the order is decided.
    #[must_use]
    pub const fn weaker(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unpinned, _) | (_, Self::Unpinned) => Self::Unpinned,
            (Self::VerifiedPrevious(p), _) => Self::VerifiedPrevious(p),
            (Self::Current, Self::VerifiedPrevious(p)) => Self::VerifiedPrevious(p),
            (Self::Current, Self::Current) => Self::Current,
        }
    }
}

/// A value read from a file, with what is known about the arithmetic it was written under
///
/// Returned by the explicit `*_unverified` loaders, which also read files that carry no
/// identifier; the value is only reachable through [`Checked::into_parts`], next to its
/// [`StoredSemantics`].
#[must_use = "a loaded value carries its semantics status; take both with into_parts"]
#[derive(Debug)]
pub struct Checked<T> {
    value: T,
    semantics: StoredSemantics,
}

impl<T> Checked<T> {
    pub(crate) const fn new(value: T, semantics: StoredSemantics) -> Self {
        Self { value, semantics }
    }

    /// What is known about the arithmetic the value was written under
    pub const fn semantics(&self) -> StoredSemantics {
        self.semantics
    }

    /// The value and its semantics status
    pub fn into_parts(self) -> (T, StoredSemantics) {
        (self.value, self.semantics)
    }
}

/// The verdict of this crate's own loaders on a stored identifier (`None`: the format version
/// predates identifiers)
///
/// # Errors
///
/// [`crate::io::IoError::SemanticsMismatch`] for an identifier that is neither this crate's nor
/// one it still reproduces; with `allow_unpinned == false`, [`crate::io::IoError::Unpinned`] for
/// data without an identifier.
pub(crate) fn sdf_load_verdict(
    stored: Option<[u8; 32]>,
    format: &'static str,
    allow_unpinned: bool,
) -> Result<StoredSemantics, crate::io::IoError> {
    match stored {
        None if allow_unpinned => Ok(StoredSemantics::Unpinned),
        None => Err(crate::io::IoError::Unpinned { format }),
        Some(id) => match crate::check_sdf_semantics(&id) {
            alice_det_math::SemanticsCheck::Verified => Ok(StoredSemantics::Current),
            alice_det_math::SemanticsCheck::VerifiedPrevious(p) => {
                Ok(StoredSemantics::VerifiedPrevious(Provenance::new(p)))
            }
            _ => Err(crate::io::IoError::SemanticsMismatch {
                format,
                stored: Provenance::new(id),
                expected: Provenance::sdf(),
            }),
        },
    }
}

/// The verdict of an importer (a mesh or an image) on the bytes of a file of `format`: the
/// identifier this crate's writers store ([`from_bytes`]), judged as [`sdf_load_verdict`] judges it
///
/// A file without an identifier (a format this module reads, carrying none, or a file whose
/// layout [`from_bytes`] does not place, such as an OBJ that does not start with a comment) has
/// none; a file that holds the key (or the binary STL header magic), and a PNG or JPEG whose
/// structure cannot be walked, is an error, never read as unstamped.
///
/// # Errors
///
/// As [`sdf_load_verdict`], and [`crate::io::IoError::InvalidFormat`] for an identifier that is
/// present but malformed.
pub(crate) fn import_verdict(
    bytes: &[u8],
    format: &'static str,
    allow_unpinned: bool,
) -> Result<StoredSemantics, crate::io::IoError> {
    let stored = match from_bytes(bytes) {
        Ok(p) => Some(p.semantics_id),
        Err(ProvenanceError::Missing { .. }) => None,
        // a file that holds the key, or one whose own walk this module owns (this crate's binary
        // STL header, the PNG chunks, the JPEG markers) and failed, is never read as unstamped;
        // for PNG and JPEG the decoder refuses the same malformed files today, the guard keeps
        // that from depending on the decoder
        Err(e)
            if find(bytes, KEY.as_bytes()).is_some()
                || bytes.starts_with(STL_MAGIC)
                || bytes.starts_with(PNG_SIGNATURE)
                || bytes.starts_with(JPEG_SOI) =>
        {
            return Err(crate::io::IoError::InvalidFormat(format!("{format}: {e}")))
        }
        Err(_) => None,
    };
    sdf_load_verdict(stored, format, allow_unpinned)
}

/// The identifier of a JSON document: `asset.extras["alice:semantics_id"]` of a glTF document
/// (one with an `asset` object; `gltf` names it in errors), the top-level `"alice:semantics_id"` of
/// any other JSON document of this crate (ASDF JSON, the LOD chain sidecar), named `JSON` in errors
fn json_tagged(bytes: &[u8], gltf: &'static str) -> Result<Provenance, ProvenanceError> {
    #[allow(
        clippy::disallowed_methods,
        reason = "stored form other-type: a serde_json::Value of a glTF or ASDF JSON document, read only for its stored identifier"
    )]
    let doc: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| ProvenanceError::Unreadable(format!("JSON: {e}")))?;
    let (format, field) = match doc.get("asset") {
        Some(asset) => (gltf, asset.get("extras").and_then(|x| x.get(KEY))),
        None => ("JSON", doc.get(KEY)),
    };
    match field {
        None => Err(ProvenanceError::Missing { format }),
        Some(v) => v.as_str().and_then(Provenance::from_hex).ok_or_else(|| {
            ProvenanceError::Unreadable(format!("{format}: {KEY} is not 64 hex digits"))
        }),
    }
}

/// An IGES file starts with a Start-section line: 80 columns, `S` in column 73, sequence 1
fn is_iges_start(bytes: &[u8]) -> bool {
    let first = bytes.split(|&b| b == b'\n').next().unwrap_or(&[]);
    let first = first.strip_suffix(b"\r").unwrap_or(first);
    first.len() == 80 && first[72] == b'S' && first[73..].trim_ascii() == b"1"
}

/// The tag of the identifier block in the DDS header's reserved dwords
pub const DDS_TAG: &[u8; 4] = b"ALSD";

/// The DDS header's 11 reserved dwords carrying the identifier: the tag [`DDS_TAG`], a layout
/// version `1`, the 32 identifier bytes (little-endian dwords), one zero dword
#[cfg(feature = "volume")]
pub(crate) fn dds_reserved(p: &Provenance) -> [u32; 11] {
    let mut r = [0u32; 11];
    r[0] = u32::from_le_bytes(*DDS_TAG);
    r[1] = 1;
    for (i, c) in p.semantics_id.chunks_exact(4).enumerate() {
        r[2 + i] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    r
}

/// The sidecar of a file whose format has no place for the identifier: `<path>.semantics`
#[must_use]
pub fn sidecar_path(path: &Path) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".semantics");
    std::path::PathBuf::from(name)
}

/// Writes the sidecar `<path>.semantics` (`alice:semantics_id=<64 hex>` and a newline)
#[cfg(feature = "volume")]
pub(crate) fn write_sidecar(path: &Path, p: &Provenance) -> std::io::Result<()> {
    std::fs::write(sidecar_path(path), format!("{}\n", p.tagged()))
}

/// The extensions whose sidecar is the plain `alice:semantics_id=` line
///
/// These are the raw volume exports (`volume::export_raw`, `volume::export_dds_3d`); every other
/// sidecar is read in the bound form with the file's SHA-256.
pub const PLAIN_SIDECAR_EXTENSIONS: &[&str] = &["raw", "dds"];

/// The key of the SHA-256 of the file in a sidecar that binds the identifier to the bytes
pub const CONTENT_SHA256_KEY: &str = "alice:content_sha256";

/// Writes the sidecar `<path>.semantics` of a file whose format has no place for the identifier and
/// whose bytes other programs rewrite: the identifier and the SHA-256 of `bytes`, one
/// `key=<64 hex>` line each
pub(crate) fn write_bound_sidecar(
    path: &Path,
    p: &Provenance,
    bytes: &[u8],
) -> std::io::Result<()> {
    use sha2::{Digest, Sha256};
    let hex = Provenance::new(Sha256::digest(bytes).into()).hex();
    std::fs::write(
        sidecar_path(path),
        format!("{}\n{CONTENT_SHA256_KEY}={hex}\n", p.tagged()),
    )
}

/// The identifier in the sidecar of the file at `path` whose bytes are `bytes`: `Ok(None)` when
/// there is no sidecar; a sidecar that is not exactly the two lines of [`write_bound_sidecar`], or
/// whose SHA-256 is not the SHA-256 of `bytes` (the file was rewritten without it), is an error,
/// never "no identifier"
pub(crate) fn bound_sidecar_id(path: &Path, bytes: &[u8]) -> Result<Option<[u8; 32]>, String> {
    use sha2::{Digest, Sha256};
    let sidecar = sidecar_path(path);
    let text = match std::fs::read(&sidecar) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("sidecar {}: {e}", sidecar.display())),
    };
    let text = std::str::from_utf8(&text).map_err(|_| String::from("sidecar is not UTF-8"))?;
    let hex32 = |v: &str| -> Option<[u8; 32]> {
        if v.len() != 64 || !v.bytes().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, o) in out.iter_mut().enumerate() {
            *o = u8::from_str_radix(&v[2 * i..2 * i + 2], 16).ok()?;
        }
        Some(out)
    };
    let (mut id, mut sha) = (None, None);
    for line in text.lines() {
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("sidecar line {line:?} is not key=value"))?;
        let slot = if key == KEY {
            &mut id
        } else if key == CONTENT_SHA256_KEY {
            &mut sha
        } else {
            return Err(format!("sidecar key {key:?} is not known"));
        };
        if slot.is_some() {
            return Err(format!("sidecar has {key} twice"));
        }
        *slot = Some(hex32(value).ok_or_else(|| format!("sidecar {key} is not 64 hex"))?);
    }
    let id = id.ok_or_else(|| format!("sidecar has no {KEY}"))?;
    let sha = sha.ok_or_else(|| format!("sidecar has no {CONTENT_SHA256_KEY}"))?;
    if sha[..] != Sha256::digest(bytes)[..] {
        return Err(format!(
            "the file's SHA-256 is not the sidecar's {CONTENT_SHA256_KEY} (the file was rewritten after its sidecar)"
        ));
    }
    Ok(Some(id))
}

/// The name of an image format in errors, from its signature (`PNG`, `JPEG`, else `image`)
#[cfg(feature = "image")]
pub(crate) fn image_format(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(PNG_SIGNATURE) {
        "PNG"
    } else if bytes.starts_with(JPEG_SOI) {
        "JPEG"
    } else {
        "image"
    }
}

/// The PNG file signature
const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";

/// The JPEG start-of-image marker
const JPEG_SOI: &[u8; 2] = b"\xff\xd8";

/// The CRC-32 of PNG chunks (ISO 3309, reflected polynomial `0xEDB88320`) over `parts` in order
fn png_crc(parts: &[&[u8]]) -> u32 {
    let mut crc = !0u32;
    for part in parts {
        for &b in *part {
            crc ^= u32::from(b);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
    }
    !crc
}

/// The identifier of a PNG file: a `tEXt` chunk with the keyword [`KEY`] whose text is the 64 hex
/// digits (written by [`crate::terrain::Heightmap::to_png_with_provenance`] before the image data)
///
/// The chunks are walked from the signature to `IEND`; a chunk whose length runs past the end, a
/// cut chunk header and a file without `IEND` are unreadable (a truncated stamped file is never
/// read as one without an identifier). The key chunk's CRC is checked, so a changed byte in the
/// identifier is unreadable rather than a different identifier. The key only counts in `tEXt`
/// (uncompressed): the keyword in a `zTXt` or `iTXt` chunk, a second key chunk, or a text that is
/// not exactly 64 hex digits is unreadable.
fn png_tagged(bytes: &[u8]) -> Result<Provenance, ProvenanceError> {
    let bad = |what: &str| ProvenanceError::Unreadable(format!("PNG: {what}"));
    let mut found = None;
    let mut at = PNG_SIGNATURE.len();
    loop {
        let header = bytes
            .get(at..at + 8)
            .ok_or_else(|| bad("the file ends before IEND"))?;
        let len = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let kind = &header[4..8];
        let data_end = (at + 8)
            .checked_add(len)
            .filter(|&e| e.checked_add(4).is_some_and(|c| c <= bytes.len()))
            .ok_or_else(|| bad("a chunk runs past the end of the file"))?;
        let data = &bytes[at + 8..data_end];
        let keyword = data.split(|&b| b == 0).next().unwrap_or(data);
        if keyword == KEY.as_bytes() {
            if kind != b"tEXt" {
                return Err(bad(&format!(
                    "{KEY} is in a {} chunk (only tEXt is read)",
                    String::from_utf8_lossy(kind)
                )));
            }
            let stored = &bytes[data_end..data_end + 4];
            if png_crc(&[kind, data]).to_be_bytes() != stored {
                return Err(bad(&format!("the CRC of the {KEY} chunk does not match")));
            }
            let p = data
                .get(KEY.len() + 1..)
                .and_then(|text| std::str::from_utf8(text).ok())
                .filter(|t| t.len() == 64)
                .and_then(Provenance::from_hex)
                .ok_or_else(|| bad(&format!("{KEY} is not 64 hex digits")))?;
            if found.replace(p).is_some() {
                return Err(bad(&format!("{KEY} appears twice")));
            }
        }
        at = data_end + 4;
        if kind == b"IEND" {
            return found.ok_or(ProvenanceError::Missing { format: "PNG" });
        }
    }
}

/// The identifier of a JPEG file: a comment segment (`COM`) whose payload is
/// `alice:semantics_id=<64 hex>`
///
/// The marker segments are walked from the start of image to the start of scan (after it the
/// bytes are entropy-coded data); a segment whose length runs past the end, a file that ends
/// before the start of scan, a second key comment, or a comment that starts with the key but is
/// not exactly the tagged form is unreadable.
fn jpeg_tagged(bytes: &[u8]) -> Result<Provenance, ProvenanceError> {
    let bad = |what: &str| ProvenanceError::Unreadable(format!("JPEG: {what}"));
    let mut found = None;
    let mut at = JPEG_SOI.len();
    loop {
        if bytes.get(at) != Some(&0xFF) {
            return Err(bad("a marker is expected before the start of scan"));
        }
        // fill bytes (0xFF) may precede a marker
        while bytes.get(at + 1) == Some(&0xFF) {
            at += 1;
        }
        let marker = *bytes
            .get(at + 1)
            .ok_or_else(|| bad("the file ends before the start of scan"))?;
        match marker {
            // the start of scan: the header segments end here
            0xDA => return found.ok_or(ProvenanceError::Missing { format: "JPEG" }),
            // markers without a length (TEM, RST0-7)
            0x01 | 0xD0..=0xD7 => {
                at += 2;
                continue;
            }
            0xD9 => return Err(bad("the image ends before the start of scan")),
            _ => {}
        }
        let len = bytes
            .get(at + 2..at + 4)
            .map(|l| usize::from(u16::from_be_bytes([l[0], l[1]])))
            .ok_or_else(|| bad("the file ends inside a segment header"))?;
        let end = at + 2 + len;
        if len < 2 || end > bytes.len() {
            return Err(bad("a segment runs past the end of the file"));
        }
        if marker == 0xFE && bytes[at + 4..end].starts_with(KEY.as_bytes()) {
            let p = std::str::from_utf8(&bytes[at + 4..end])
                .ok()
                .and_then(|t| t.strip_prefix(KEY))
                .and_then(|t| t.strip_prefix('='))
                .filter(|t| t.len() == 64)
                .and_then(Provenance::from_hex)
                .ok_or_else(|| bad(&format!("the {KEY} comment is not {KEY}=<64 hex digits>")))?;
            if found.replace(p).is_some() {
                return Err(bad(&format!("{KEY} appears twice")));
            }
        }
        at = end;
    }
}

/// The position of the first occurrence of `needle` in `haystack`
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Compares the identifier stored in the file at `path` with `expected`, or with one of the
/// earlier identifiers in `previous` that the caller has verified
///
/// # Errors
///
/// [`ProvenanceError::Mismatch`] naming the stored and the expected identifier when the stored one
/// is neither, otherwise as [`read`].
pub fn check(
    path: impl AsRef<Path>,
    expected: &Provenance,
    previous: &[[u8; 32]],
) -> Result<StoredSemantics, ProvenanceError> {
    let stored = read(path)?;
    if stored == *expected {
        Ok(StoredSemantics::Current)
    } else if previous.contains(&stored.semantics_id) {
        Ok(StoredSemantics::VerifiedPrevious(stored))
    } else {
        Err(ProvenanceError::Mismatch {
            stored,
            expected: *expected,
        })
    }
}

/// [`check`] against this crate's identifier and the earlier ones it still reproduces
/// ([`crate::PREVIOUS_SDF_SEMANTICS_IDS`], the same rule as [`crate::check_sdf_semantics`])
///
/// # Errors
///
/// As [`check`].
pub fn check_sdf(path: impl AsRef<Path>) -> Result<StoredSemantics, ProvenanceError> {
    check(path, &Provenance::sdf(), crate::PREVIOUS_SDF_SEMANTICS_IDS)
}

/// `alice:semantics_id=<64 hex>` anywhere in `bytes`; exactly one distinct value is accepted
fn tagged_in(bytes: &[u8], format: &'static str) -> Result<Provenance, ProvenanceError> {
    let key = format!("{KEY}=");
    let key = key.as_bytes();
    let mut found: Option<Provenance> = None;
    let mut i = 0;
    while i + key.len() + 64 <= bytes.len() {
        if &bytes[i..i + key.len()] == key {
            let start = i + key.len();
            let hex = std::str::from_utf8(&bytes[start..start + 64])
                .map_err(|_| ProvenanceError::Unreadable(format!("{format}: {KEY} is not hex")))?;
            let p = Provenance::from_hex(hex).ok_or_else(|| {
                ProvenanceError::Unreadable(format!("{format}: {KEY} is not hex"))
            })?;
            match found {
                Some(q) if q != p => {
                    return Err(ProvenanceError::Unreadable(format!(
                        "{format}: two different {KEY} values"
                    )))
                }
                _ => found = Some(p),
            }
            i = start + 64;
        } else {
            i += 1;
        }
    }
    found.ok_or(ProvenanceError::Missing { format })
}

/// The identifier in the archive metadata of an Alembic (Ogawa) archive, as
/// [`crate::io::alembic::export_alembic`] writes it: the header (`Ogawa`, the frozen byte `0xFF`,
/// version `00 01`, the `u64` position of the root group), the root group's fourth child (a data
/// block, the metadata `key=value;key=value`) and in it `alice:semantics_id=<64 hex>`. An archive
/// of another program without that key has none (Missing); the layout this crate wrote up to
/// 7.0.0 (`Ogawa` `00 01 00`, which no Alembic reader opens) is unreadable.
fn read_alembic(bytes: &[u8]) -> Result<Provenance, ProvenanceError> {
    let bad = |why: &str| ProvenanceError::Unreadable(format!("Alembic: {why}"));
    let u64_at = |at: usize| -> Result<u64, ProvenanceError> {
        let w: [u8; 8] = bytes
            .get(at..at.checked_add(8).ok_or_else(|| bad("truncated archive"))?)
            .ok_or_else(|| bad("truncated archive"))?
            .try_into()
            .map_err(|_| bad("truncated archive"))?;
        Ok(u64::from_le_bytes(w))
    };
    if bytes.get(5..8) == Some(&[0x00, 0x01, 0x00]) {
        return Err(bad(
            "the archive layout ALICE-SDF 7.0.0 and earlier wrote (not the Alembic Ogawa layout); export the mesh again",
        ));
    }
    if bytes.get(6..8) != Some(&[0x00, 0x01]) {
        return Err(bad("not Ogawa version 0.1"));
    }
    if bytes.get(5) != Some(&0xFF) {
        return Err(bad(
            "the archive was not finished (frozen byte is not 0xFF)",
        ));
    }
    let root = usize::try_from(u64_at(8)?).map_err(|_| bad("truncated archive"))?;
    let children = u64_at(root)?;
    if children < 4 {
        return Err(bad("the root group has no archive metadata"));
    }
    let child = u64_at(root + 8 + 3 * 8)?;
    if child & (1 << 63) == 0 {
        return Err(bad("the archive metadata is not a data block"));
    }
    let meta: &[u8] = if child == 1 << 63 {
        &[]
    } else {
        let at = usize::try_from(child & !(1 << 63)).map_err(|_| bad("truncated archive"))?;
        let len = usize::try_from(u64_at(at)?).map_err(|_| bad("truncated archive"))?;
        let start = at + 8;
        bytes
            .get(
                start
                    ..start
                        .checked_add(len)
                        .ok_or_else(|| bad("truncated archive"))?,
            )
            .ok_or_else(|| bad("truncated archive"))?
    };
    let meta = std::str::from_utf8(meta).map_err(|_| bad("archive metadata is not UTF-8"))?;
    let mut values = meta
        .split(';')
        .filter_map(|kv| kv.split_once('='))
        .filter(|(k, _)| *k == KEY)
        .map(|(_, v)| Provenance::from_hex(v));
    match (values.next(), values.next()) {
        (None, _) => Err(ProvenanceError::Missing { format: "Alembic" }),
        (Some(Some(p)), None) => Ok(p),
        (Some(None), _) => Err(bad(&format!("{KEY} is not 64 hex digits"))),
        (Some(Some(_)), Some(_)) => Err(bad(&format!("more than one {KEY}"))),
    }
}

/// The `alice:semantics_id` metadata of `3D/3dmodel.model` in a 3MF (zip) archive
fn read_3mf(bytes: &[u8]) -> Result<Provenance, ProvenanceError> {
    let model = zip_entry(bytes, "3D/3dmodel.model")?
        .ok_or_else(|| ProvenanceError::Unreadable("3MF: no 3D/3dmodel.model".into()))?;
    let open = format!("<metadata name=\"{KEY}\">");
    let text = std::str::from_utf8(model)
        .map_err(|_| ProvenanceError::Unreadable("3MF: model is not UTF-8".into()))?;
    let mut values = text.match_indices(&open).map(|(at, _)| {
        let start = at + open.len();
        text.get(start..start + 64)
            .and_then(Provenance::from_hex)
            .filter(|_| text[start + 64..].starts_with("</metadata>"))
    });
    match (values.next(), values.next()) {
        (None, _) => Err(ProvenanceError::Missing { format: "3MF" }),
        (Some(Some(p)), None) => Ok(p),
        (Some(None), _) => Err(ProvenanceError::Unreadable(format!(
            "3MF: {KEY} is not 64 hex digits"
        ))),
        (Some(Some(_)), Some(_)) => Err(ProvenanceError::Unreadable(format!(
            "3MF: more than one {KEY}"
        ))),
    }
}

/// The stored (uncompressed) data of the zip entry `name`, walking the local file headers
fn zip_entry<'a>(bytes: &'a [u8], name: &str) -> Result<Option<&'a [u8]>, ProvenanceError> {
    let short = || ProvenanceError::Unreadable("3MF: truncated zip".into());
    let u16_at = |at: usize| -> Option<usize> {
        bytes
            .get(at..at + 2)
            .map(|b| usize::from(u16::from_le_bytes([b[0], b[1]])))
    };
    let u32_at = |at: usize| -> Option<usize> {
        bytes
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    };
    let mut at = 0;
    while bytes.get(at..at + 4) == Some(b"PK\x03\x04") {
        let flags = u16_at(at + 6).ok_or_else(short)?;
        let method = u16_at(at + 8).ok_or_else(short)?;
        let compressed = u32_at(at + 18).ok_or_else(short)?;
        let name_len = u16_at(at + 26).ok_or_else(short)?;
        let extra_len = u16_at(at + 28).ok_or_else(short)?;
        let name_at = at + 30;
        let data_at = name_at + name_len + extra_len;
        if flags & 0x08 != 0 {
            return Err(ProvenanceError::Unreadable(
                "3MF: entry sizes in a data descriptor are not read".into(),
            ));
        }
        let entry = bytes.get(name_at..name_at + name_len).ok_or_else(short)?;
        let data = bytes.get(data_at..data_at + compressed).ok_or_else(short)?;
        if entry == name.as_bytes() {
            if method != 0 {
                return Err(ProvenanceError::Unreadable(format!(
                    "3MF: {name} is compressed (method {method}); only stored entries are read"
                )));
            }
            return Ok(Some(data));
        }
        at = data_at + compressed;
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weaker_orders_unpinned_previous_current() {
        use StoredSemantics::{Current, Unpinned, VerifiedPrevious};
        let (a, b) = (VerifiedPrevious(id(1)), VerifiedPrevious(id(2)));
        let cases = [
            (Current, Current, Current),
            (Current, a, a),
            (a, Current, a),
            (Current, Unpinned, Unpinned),
            (Unpinned, Current, Unpinned),
            (a, Unpinned, Unpinned),
            (Unpinned, a, Unpinned),
            (Unpinned, Unpinned, Unpinned),
            (a, a, a),
            (a, b, a),
            (b, a, b),
        ];
        for (x, y, want) in cases {
            assert_eq!(x.weaker(y), want, "{x:?}.weaker({y:?})");
        }
    }

    fn id(seed: u8) -> Provenance {
        let mut b = [0u8; 32];
        for (i, v) in b.iter_mut().enumerate() {
            *v = seed.wrapping_add(u8::try_from(i).unwrap()).wrapping_mul(37);
        }
        Provenance::new(b)
    }

    #[test]
    fn hex_round_trips() {
        let p = id(3);
        assert_eq!(Provenance::from_hex(&p.hex()), Some(p));
        assert_eq!(p.hex().len(), 64);
        assert!(Provenance::from_hex(&p.hex()[..63]).is_none());
        assert!(Provenance::from_hex(&"g".repeat(64)).is_none());
    }

    #[test]
    fn the_stl_header_never_reads_as_ascii() {
        let h = id(0).stl_header();
        assert!(!h.starts_with(b"solid"));
        assert_eq!(&h[..8], STL_MAGIC);
        assert_eq!(h[8], STL_HEADER_VERSION);
        assert!(h[41..].iter().all(|&b| b == 0));
    }

    #[test]
    fn two_different_tags_are_refused() {
        let text = format!("{} {}", id(1).tagged(), id(2).tagged());
        assert!(matches!(
            tagged_in(text.as_bytes(), "t"),
            Err(ProvenanceError::Unreadable(_))
        ));
        let same = format!("{} {}", id(1).tagged(), id(1).tagged());
        assert_eq!(tagged_in(same.as_bytes(), "t").unwrap(), id(1));
    }

    #[test]
    fn a_binary_stl_without_the_magic_is_missing_not_unreadable() {
        let mut bytes = vec![0u8; 84];
        bytes.extend_from_slice(&[0u8; 50]);
        bytes[80] = 1;
        assert!(matches!(
            from_bytes(&bytes),
            Err(ProvenanceError::Missing {
                format: "binary STL"
            })
        ));
    }

    #[test]
    fn unknown_bytes_are_unreadable() {
        assert!(matches!(
            from_bytes(b"hello"),
            Err(ProvenanceError::Unreadable(_))
        ));
    }
}
