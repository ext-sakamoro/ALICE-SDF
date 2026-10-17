//! Provenance: the identifier of the meaning of the bytes a writer stores (ALICE-Law G8)
//!
//! Every durable format this crate writes carries a 32-byte semantics identifier, so a file read
//! back later (or by another program) can be checked against the arithmetic that produced it.
//! The plain writers store [`Provenance::sdf`] (this crate's [`crate::SEMANTICS_ID`]); a caller
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
//! | USD (`.usda`) | the layer's `customLayerData`: `string "alice:semantics_id" = "<64 hex>"` |
//! | STEP | the second string of `FILE_DESCRIPTION`: `'alice:semantics_id=<64 hex>'` |
//! | IGES | two Start-section lines, `alice:semantics_id=` and the 64 hex digits (joined they read as one) |
//!
//! [`read`] finds the identifier in a file of any of these formats, and [`check`] compares it with
//! the expected one (or an earlier one the caller has verified) and names both on a mismatch.

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

/// The identifier of the meaning of the stored bytes
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Provenance {
    /// 32-byte semantics identifier
    pub semantics_id: [u8; 32],
}

impl Provenance {
    /// This crate's own identifier ([`crate::SEMANTICS_ID`])
    #[must_use]
    pub const fn sdf() -> Self {
        Self {
            semantics_id: crate::SEMANTICS_ID,
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

    fn from_hex(s: &str) -> Option<Self> {
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
}

/// The identifier stored in the file at `path`
///
/// # Errors
///
/// [`ProvenanceError::Missing`] when the format is recognised but carries no identifier,
/// [`ProvenanceError::Unreadable`] when the format is not recognised (or a 3MF entry is
/// compressed), [`ProvenanceError::Io`] when the file cannot be read.
pub fn read(path: impl AsRef<Path>) -> Result<Provenance, ProvenanceError> {
    from_bytes(&std::fs::read(path)?)
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
            if &bytes[..8] == STL_MAGIC && bytes[8] == STL_HEADER_VERSION {
                let mut id = [0u8; 32];
                id.copy_from_slice(&bytes[9..41]);
                return Ok(Provenance::new(id));
            }
            return Err(ProvenanceError::Missing {
                format: "binary STL",
            });
        }
    }
    Err(ProvenanceError::Unreadable("unrecognised format".into()))
}

/// Which accepted identifier a stored one matched
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Accepted {
    /// The current identifier
    Current,
    /// An earlier identifier the caller lists as verified (its outputs are reproduced bit for bit)
    VerifiedPrevious(Provenance),
}

/// Identifiers of earlier releases of this crate whose stored outputs this release reproduces
///
/// Empty: [`Provenance::sdf`] is the identifier of the arithmetic (`alice-det-math`'s
/// `SEMANTICS_ID`) and does not yet fold in this crate's own algorithms, so an identifier that
/// `alice-det-math` lists as a verified earlier one says nothing about whether an earlier release
/// of this crate wrote the same bytes. An entry is added only with a test that shows the outputs
/// of that release reproduced.
pub const PREVIOUS_SDF_IDS: &[[u8; 32]] = &[];

/// `asset.extras["alice:semantics_id"]` of a glTF JSON document
fn json_tagged(bytes: &[u8], format: &'static str) -> Result<Provenance, ProvenanceError> {
    let doc: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| ProvenanceError::Unreadable(format!("{format}: JSON: {e}")))?;
    match doc
        .get("asset")
        .and_then(|a| a.get("extras"))
        .and_then(|x| x.get(KEY))
    {
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
) -> Result<Accepted, ProvenanceError> {
    let stored = read(path)?;
    if stored == *expected {
        Ok(Accepted::Current)
    } else if previous.contains(&stored.semantics_id) {
        Ok(Accepted::VerifiedPrevious(stored))
    } else {
        Err(ProvenanceError::Mismatch {
            stored,
            expected: *expected,
        })
    }
}

/// [`check`] against this crate's identifier and [`PREVIOUS_SDF_IDS`]
///
/// # Errors
///
/// As [`check`].
pub fn check_sdf(path: impl AsRef<Path>) -> Result<Accepted, ProvenanceError> {
    check(path, &Provenance::sdf(), PREVIOUS_SDF_IDS)
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
