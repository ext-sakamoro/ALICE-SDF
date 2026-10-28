//! Texture-to-Formula Conversion Tool
//!
//! Converts bitmap textures (PNG/JPG) into resolution-independent
//! procedural noise formulas:
//!
//! `texture(u,v) ≈ bias + Σᵢ aᵢ · noise(uv · fᵢ + φᵢ, seedᵢ)`
//!
//! The noise is the crate's one value-noise law (`modifiers::hash_noise_3d`,
//! PCG lattice hash: integer ops up to the final `u32 → f32`), on the CPU
//! (scalar and SIMD) and in the `hash_noise_3d` helper the generated WGSL /
//! HLSL / GLSL shaders embed — the same text the SDF transpilers emit, so a
//! fitted texture can live in the same shader as an SDF. CPU ≡ GPU is
//! measured: `tests/test_texture_shader_gpu_parity.rs` renders the emitted
//! WGSL on the GPU and matches `reconstruct` to 3e-7 (CI `gpu-parity` job).
//! Fits made before 1.14.0 used a sin hash and must be regenerated.

mod fitting;
mod noise_cpu;
mod optimizer;
mod shader;
mod spectrum;

pub use fitting::{fit_texture, fit_texture_pinned, reconstruct};
pub use noise_cpu::{eval_octave, hash_noise_3d_cpu};
pub use optimizer::{nelder_mead, OptimizeResult};
pub use shader::{generate_shader, ShaderLanguage};

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::io::IoError;
use crate::provenance::{
    Checked, Provenance, Stored, StoredForm, StoredSemantics, KEY, MEANING_KEY,
};

/// Why [`fit_texture`] failed
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TextureFitError {
    /// The image could not be read: I/O, its semantics identifier (another one, malformed, or
    /// none for [`fit_texture_pinned`]) or its pixels
    #[error(transparent)]
    Io(#[from] crate::io::IoError),
    /// The fit itself failed
    #[error("{0}")]
    Fit(String),
}

/// Result of fitting a texture to procedural noise octaves
///
/// The fitted parameters only reproduce the texture under the noise law they were fitted
/// against, so the stored form carries the semantics identifier: write it with
/// [`save_texture_fit`] and read it with [`load_texture_fit`]. The type has no serde impls of
/// its own, so a stored fit cannot be read without the identifier check.
#[derive(Debug, Clone, PartialEq)]
pub struct TextureFitResult {
    /// Source image width
    pub width: u32,
    /// Source image height
    pub height: u32,
    /// Number of channels (1=grayscale, 3=RGB)
    pub channels: u32,
    /// DC bias per channel
    pub bias: Vec<f32>,
    /// Fitted octaves per channel
    pub octaves: Vec<Vec<FittedOctave>>,
    /// Peak Signal-to-Noise Ratio (dB)
    pub psnr_db: f32,
    /// Normalized Mean Squared Error
    pub nmse: f32,
}

/// A single fitted noise octave
#[derive(Debug, Clone, PartialEq)]
pub struct FittedOctave {
    /// Amplitude (weight)
    pub amplitude: f32,
    /// Frequency multiplier
    pub frequency: f32,
    /// Phase offset (u, v)
    pub phase: [f32; 2],
    /// Noise seed
    pub seed: u32,
    /// Rotation in radians (anisotropy)
    pub rotation: f32,
}

/// Configuration for texture fitting
pub struct TextureFitConfig {
    /// Maximum number of octaves to fit (default: 8)
    pub max_octaves: u32,
    /// Target PSNR in dB — stop fitting when reached (default: 28.0)
    pub target_psnr_db: f32,
    /// Nelder-Mead iterations per octave (default: 500)
    pub iterations_per_octave: u32,
    /// Whether the texture should tile seamlessly (default: true)
    ///
    /// Currently has no effect: the hash-lattice noise the fit is expressed
    /// in does not wrap, so no choice of parameters makes the reconstruction
    /// seamless. Kept so that the field can be honoured once the noise
    /// lattice is made periodic (see CHANGELOG).
    pub tileable: bool,
}

impl Default for TextureFitConfig {
    fn default() -> Self {
        Self {
            max_octaves: 8,
            target_psnr_db: 28.0,
            iterations_per_octave: 500,
            tileable: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Stored form (JSON)
// ---------------------------------------------------------------------------

/// The format of the stored fit: 1 carries the semantics identifier; a document without
/// `format` was written by the CLI of 7.0.0 or earlier (no identifier)
const FIT_FORMAT: u32 = 1;

/// The JSON document of a stored fit: the fields of [`TextureFitResult`] next to `format` and
/// `alice:meaning_id` (the meaning identifier of the JSON document form; a document written before
/// the form stored meaning identifiers holds the raw `alice:semantics_id`)
#[derive(Serialize, Deserialize)]
struct FitDocument {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    format: Option<u32>,
    #[serde(
        rename = "alice:meaning_id",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    meaning_id: Option<String>,
    #[serde(
        rename = "alice:semantics_id",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    semantics_id: Option<String>,
    width: u32,
    height: u32,
    channels: u32,
    bias: Vec<f32>,
    octaves: Vec<Vec<OctaveDocument>>,
    psnr_db: f32,
    nmse: f32,
}

/// One octave in a [`FitDocument`]
#[derive(Serialize, Deserialize)]
struct OctaveDocument {
    amplitude: f32,
    frequency: f32,
    phase: [f32; 2],
    seed: u32,
    rotation: f32,
}

impl FitDocument {
    fn new(result: &TextureFitResult, provenance: &Provenance) -> Self {
        Self {
            format: Some(FIT_FORMAT),
            meaning_id: Some(provenance.named_for(StoredForm::JsonDocument).1),
            semantics_id: None,
            width: result.width,
            height: result.height,
            channels: result.channels,
            bias: result.bias.clone(),
            octaves: result
                .octaves
                .iter()
                .map(|ch| {
                    ch.iter()
                        .map(|o| OctaveDocument {
                            amplitude: o.amplitude,
                            frequency: o.frequency,
                            phase: o.phase,
                            seed: o.seed,
                            rotation: o.rotation,
                        })
                        .collect()
                })
                .collect(),
            psnr_db: result.psnr_db,
            nmse: result.nmse,
        }
    }

    /// The stored identifier (`None` for a document written before `format` 1)
    fn stored_id(&self) -> Result<Option<Stored>, IoError> {
        let stored = crate::provenance::json_fields_stored(
            self.meaning_id.as_deref(),
            self.semantics_id.as_deref(),
            StoredForm::JsonDocument,
            "texture fit",
        )?;
        match (self.format, stored.is_some()) {
            (None, false) | (Some(FIT_FORMAT), true) => Ok(stored),
            (Some(FIT_FORMAT), false) | (None, true) => Err(IoError::InvalidFormat(format!(
                "texture fit: `format` and {MEANING_KEY} (or {KEY}) must be both present or both absent"
            ))),
            (Some(f), _) => Err(IoError::UnsupportedVersion(
                u16::try_from(f).unwrap_or(u16::MAX),
            )),
        }
    }

    fn into_result(self) -> TextureFitResult {
        TextureFitResult {
            width: self.width,
            height: self.height,
            channels: self.channels,
            bias: self.bias,
            octaves: self
                .octaves
                .into_iter()
                .map(|ch| {
                    ch.into_iter()
                        .map(|o| FittedOctave {
                            amplitude: o.amplitude,
                            frequency: o.frequency,
                            phase: o.phase,
                            seed: o.seed,
                            rotation: o.rotation,
                        })
                        .collect()
                })
                .collect(),
            psnr_db: self.psnr_db,
            nmse: self.nmse,
        }
    }
}

/// Write a fit as JSON with this crate's semantics identifier
///
/// The document holds the fields of [`TextureFitResult`] next to `"format": 1` and
/// `"alice:meaning_id"` (the meaning identifier of the JSON document form); [`load_texture_fit`]
/// reads it back and checks the identifier.
///
/// # Errors
///
/// The I/O and JSON errors.
pub fn save_texture_fit(result: &TextureFitResult, path: impl AsRef<Path>) -> Result<(), IoError> {
    save_texture_fit_with_provenance(result, path, &Provenance::sdf())
}

/// [`save_texture_fit`] storing the given semantics identifier instead of this crate's
///
/// # Errors
///
/// As [`save_texture_fit`].
pub fn save_texture_fit_with_provenance(
    result: &TextureFitResult,
    path: impl AsRef<Path>,
    provenance: &Provenance,
) -> Result<(), IoError> {
    let json = serde_json::to_string_pretty(&FitDocument::new(result, provenance))
        .map_err(|e| IoError::Serialization(format!("texture fit: {e}")))?;
    std::fs::write(path, json + "\n")?;
    Ok(())
}

/// Read a fit written by [`save_texture_fit`]
///
/// The fitted parameters are values computed under the noise law, so a document without an
/// identifier (written by the CLI of 7.0.0 or earlier, possibly under an earlier noise law) is
/// refused; [`load_texture_fit_unverified`] reads it and reports it as
/// [`StoredSemantics::Unpinned`].
///
/// # Errors
///
/// [`IoError::Unpinned`] for a document without an identifier, [`IoError::SemanticsMismatch`]
/// for another identifier, [`IoError::InvalidFormat`] / [`IoError::UnsupportedVersion`] for a
/// malformed identifier or another format, and the I/O and JSON errors.
pub fn load_texture_fit(path: impl AsRef<Path>) -> Result<TextureFitResult, IoError> {
    load_fit(path.as_ref(), false).map(|c| c.into_parts().0)
}

/// [`load_texture_fit`] that also reads a document without an identifier, marking it
/// [`StoredSemantics::Unpinned`]
///
/// # Errors
///
/// As [`load_texture_fit`], except [`IoError::Unpinned`].
pub fn load_texture_fit_unverified(
    path: impl AsRef<Path>,
) -> Result<Checked<TextureFitResult>, IoError> {
    load_fit(path.as_ref(), true)
}

fn load_fit(path: &Path, allow_unpinned: bool) -> Result<Checked<TextureFitResult>, IoError> {
    let doc = read_fit_document(path)?;
    let semantics: StoredSemantics = crate::provenance::sdf_stored_load_verdict(
        doc.stored_id()?,
        "texture fit",
        allow_unpinned,
    )?;
    Ok(Checked::new(doc.into_result(), semantics))
}

#[allow(
    clippy::disallowed_methods,
    reason = "stored form checked: the document's format and its identifier field go through sdf_stored_load_verdict in load_fit before any value is returned"
)]
fn read_fit_document(path: &Path) -> Result<FitDocument, IoError> {
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|e| IoError::Serialization(format!("texture fit: {e}")))
}
