//! The two stored forms `bake_assets` writes itself, each with the semantics identifier
//! (ALICE-Law G8): the compiled bytecode `law.bytecode` and `manifest.json`. The other baked files
//! are written by the library writers, which store it already.
//!
//! `law.bytecode` (little-endian): magic `ASDFBC02`, the 32-byte identifier, `u32` instruction
//! count, `u32` aux length, `f32` Lipschitz bound, then per instruction `7 × f32` params, `u8`
//! opcode, `u8` flags, `u16` child count, `u32` skip offset, `u32` aux offset, `u32` aux length
//! (44 bytes), then the aux `f32`s. `ASDFBC01` (the same layout without the identifier) is
//! refused. `manifest.json` holds the identifier as its top-level `alice:semantics_id`.
//!
//! The readers accept this crate's identifier and the earlier ones it still reproduces
//! (`PREVIOUS_SDF_SEMANTICS_IDS`), judged by `alice_sdf::check_sdf_semantics`, the rule of
//! `provenance::check_sdf`.

use alice_sdf::prelude::CompiledSdf;
use alice_sdf::provenance::{self, Provenance, StoredSemantics, KEY};
use serde_json::Value;
use std::path::Path;

/// The magic of `law.bytecode` with the identifier
pub const BYTECODE_MAGIC: &[u8; 8] = b"ASDFBC02";
/// The magic of the earlier `law.bytecode`, which has none
pub const BYTECODE_MAGIC_V1: &[u8; 8] = b"ASDFBC01";

/// The bytes of `law.bytecode` for `c` with the identifier `id`
pub fn bytecode_bytes(c: &CompiledSdf, id: &Provenance) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(BYTECODE_MAGIC);
    out.extend_from_slice(&id.semantics_id);
    out.extend_from_slice(&(c.instructions().len() as u32).to_le_bytes());
    out.extend_from_slice(&(c.aux_data().len() as u32).to_le_bytes());
    out.extend_from_slice(&c.lipschitz().to_le_bytes());
    for ins in c.instructions() {
        for p in ins.params {
            out.extend_from_slice(&p.to_le_bytes());
        }
        out.push(ins.opcode as u8);
        out.push(ins.flags);
        out.extend_from_slice(&ins.child_count.to_le_bytes());
        out.extend_from_slice(&ins.skip_offset.to_le_bytes());
        out.extend_from_slice(&ins.aux_offset.to_le_bytes());
        out.extend_from_slice(&ins.aux_len.to_le_bytes());
    }
    for a in c.aux_data() {
        out.extend_from_slice(&a.to_le_bytes());
    }
    out
}

/// Writes `law.bytecode` for `c` with this crate's identifier
pub fn write_bytecode(path: &Path, c: &CompiledSdf) -> Result<(), String> {
    write_bytecode_with_provenance(path, c, &Provenance::sdf())
}

/// Writes `law.bytecode` for `c` with the caller's identifier
pub fn write_bytecode_with_provenance(
    path: &Path,
    c: &CompiledSdf,
    id: &Provenance,
) -> Result<(), String> {
    std::fs::write(path, bytecode_bytes(c, id)).map_err(|e| format!("{}: {e}", path.display()))
}

/// The status of a stored identifier under the rule of `provenance::check_sdf`: the crate's
/// `check_sdf_semantics`, not a copy of it
fn judge(stored: [u8; 32], what: &str) -> Result<StoredSemantics, String> {
    match alice_sdf::check_sdf_semantics(&stored) {
        alice_det_math::SemanticsCheck::Verified => Ok(StoredSemantics::Current),
        alice_det_math::SemanticsCheck::VerifiedPrevious(p) => {
            Ok(StoredSemantics::VerifiedPrevious(Provenance::new(p)))
        }
        _ => Err(format!(
            "{what}: semantics identifier {} is not this build's {}",
            Provenance::new(stored).hex(),
            Provenance::sdf().hex()
        )),
    }
}

/// The identifier status of `law.bytecode` bytes: the bytes without the identifier (`ASDFBC01`), a
/// cut header, an unknown magic and another identifier are refused
pub fn bytecode_semantics(bytes: &[u8]) -> Result<StoredSemantics, String> {
    if bytes.starts_with(BYTECODE_MAGIC_V1) {
        return Err("law.bytecode: ASDFBC01 stores no semantics identifier".into());
    }
    if !bytes.starts_with(BYTECODE_MAGIC) {
        return Err("law.bytecode: not ASDFBC02".into());
    }
    let id: [u8; 32] = bytes
        .get(8..40)
        .and_then(|s| s.try_into().ok())
        .ok_or("law.bytecode: the header is cut before the identifier")?;
    judge(id, "law.bytecode")
}

/// Writes `manifest.json` with this crate's identifier as its top-level `alice:semantics_id`
pub fn write_manifest(path: &Path, manifest: &Value) -> Result<(), String> {
    write_manifest_with_provenance(path, manifest, &Provenance::sdf())
}

/// Writes `manifest.json` with the caller's identifier
pub fn write_manifest_with_provenance(
    path: &Path,
    manifest: &Value,
    id: &Provenance,
) -> Result<(), String> {
    let mut doc = manifest.clone();
    doc.as_object_mut()
        .ok_or("manifest is not a JSON object")?
        .insert(KEY.to_string(), Value::String(id.hex()));
    let text = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?;
    std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
}

/// The identifier status of the manifest at `path`: one without the identifier and another
/// identifier are refused
pub fn manifest_semantics(path: &Path) -> Result<StoredSemantics, String> {
    let stored = provenance::read(path).map_err(|e| format!("manifest.json: {e}"))?;
    judge(stored.id, "manifest.json")
}
