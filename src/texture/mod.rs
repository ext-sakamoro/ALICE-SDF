//! Texture-to-Formula Conversion Tool
//!
//! Converts bitmap textures (PNG/JPG) into resolution-independent
//! procedural noise formulas:
//!
//! `texture(u,v) ≈ bias + Σᵢ aᵢ · noise(uv · fᵢ + φᵢ, seedᵢ)`
//!
//! The CPU noise implementation exactly matches the GPU `hash_noise_3d`
//! used in WGSL/HLSL/GLSL shaders, guaranteeing CPU fitting = GPU rendering.

mod noise_cpu;
mod optimizer;
mod spectrum;
mod fitting;
mod shader;

pub use noise_cpu::hash_noise_3d_cpu;
pub use fitting::fit_texture;
pub use shader::{generate_shader, ShaderLanguage};

use serde::{Serialize, Deserialize};

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
