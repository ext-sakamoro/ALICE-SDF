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

pub use fitting::{fit_texture, reconstruct};
pub use noise_cpu::{eval_octave, hash_noise_3d_cpu};
pub use optimizer::{nelder_mead, OptimizeResult};
pub use shader::{generate_shader, ShaderLanguage};

use serde::{Deserialize, Serialize};

/// Result of fitting a texture to procedural noise octaves
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
