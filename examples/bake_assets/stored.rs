//! The two stored forms `bake_assets` writes itself, each with the semantics identifier
//! (ALICE-Law G8): the compiled bytecode `law.bytecode` and `manifest.json`. The other baked files
//! are written by the library writers, which store it already.
//!
//! `law.bytecode` (little-endian): magic `ASDFBC02`, the 32-byte meaning identifier of the form
//! `StoredForm::BakedBytecode` (in an earlier file, the raw semantics identifier), `u32` instruction
//! count, `u32` aux length, `f32` Lipschitz bound, then per instruction `7 × f32` params, `u8`
//! opcode, `u8` flags, `u16` child count, `u32` skip offset, `u32` aux offset, `u32` aux length
//! (44 bytes), then the aux `f32`s. `ASDFBC01` (the same layout without the identifier) is
//! refused. `manifest.json` holds the meaning identifier of a JSON document as its top-level
//! `alice:meaning_id` (an earlier manifest, the raw one as `alice:semantics_id`).
//!
//! The readers judge the stored identifier with `provenance::Stored::judge`, the rule of
//! `provenance::check_sdf`: this build's meaning identifier is current, a raw identifier this
//! crate reproduces (an earlier file) is the previous generation, anything else is refused.

use alice_sdf::prelude::CompiledSdf;
use alice_sdf::provenance::{self, Provenance, Stored, StoredForm, StoredSemantics};
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
    out.extend_from_slice(&id.slot_for(StoredForm::BakedBytecode));
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

/// The status of a stored identifier under the rule of `provenance::check_sdf`
/// (`Stored::judge` against this build and `PREVIOUS_SDF_SEMANTICS_IDS`), not a copy of it
fn judge(stored: &Stored, what: &str) -> Result<StoredSemantics, String> {
    stored
        .judge(&Provenance::sdf(), alice_sdf::PREVIOUS_SDF_SEMANTICS_IDS)
        .map_err(|e| format!("{what}: {e}"))
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
    judge(
        &Stored::in_slot(StoredForm::BakedBytecode, id),
        "law.bytecode",
    )
}

/// Writes `manifest.json` with this crate's meaning identifier as its top-level `alice:meaning_id`
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
    let (key, meaning) = id.named_for(StoredForm::JsonDocument);
    doc.as_object_mut()
        .ok_or("manifest is not a JSON object")?
        .insert(key.to_string(), Value::String(meaning));
    let text = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?;
    std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
}

/// The identifier status of the manifest at `path`: one without the identifier and another
/// identifier are refused
pub fn manifest_semantics(path: &Path) -> Result<StoredSemantics, String> {
    let stored = provenance::read(path).map_err(|e| format!("manifest.json: {e}"))?;
    judge(&stored, "manifest.json")
}
