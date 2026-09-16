//! Shader code generation from TextureFitResult
//!
//! Generates standalone WGSL, HLSL, or GLSL shader functions
//! that reproduce the fitted texture using hash_noise_3d.

use super::TextureFitResult;
use crate::modifiers::{HASH_NOISE_GLSL, HASH_NOISE_HLSL, HASH_NOISE_WGSL};
use std::fmt::Write;

/// Target shader language
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderLanguage {
    /// WebGPU Shading Language
    Wgsl,
    /// High Level Shading Language (DirectX / UE5)
    Hlsl,
    /// OpenGL Shading Language (Unity / Vulkan)
    Glsl,
}

impl std::str::FromStr for ShaderLanguage {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "wgsl" => Ok(Self::Wgsl),
            "hlsl" => Ok(Self::Hlsl),
            "glsl" => Ok(Self::Glsl),
            _ => Err(format!(
                "Unknown shader language: '{}'. Expected wgsl, hlsl, or glsl",
                s
            )),
        }
    }
}

/// Generate a standalone shader function from a fit result
pub fn generate_shader(
    result: &TextureFitResult,
    lang: ShaderLanguage,
    source_name: &str,
) -> String {
    match lang {
        ShaderLanguage::Wgsl => generate_wgsl(result, source_name),
        ShaderLanguage::Hlsl => generate_hlsl(result, source_name),
        ShaderLanguage::Glsl => generate_glsl(result, source_name),
    }
}

fn generate_wgsl(result: &TextureFitResult, source_name: &str) -> String {
    let mut s = String::with_capacity(2048);
    let total_octaves: usize = result.octaves.iter().map(|o| o.len()).sum();

    writeln!(
        s,
        "// ALICE-SDF Procedural Texture (fitted from {})",
        source_name
    )
    .unwrap();
    writeln!(
        s,
        "// PSNR: {:.1} dB, {} octaves",
        result.psnr_db, total_octaves
    )
    .unwrap();
    writeln!(s).unwrap();

    // hash_noise_3d function
    s.push_str(HASH_NOISE_WGSL);
    writeln!(s).unwrap();
    writeln!(s).unwrap();

    // procedural_texture function
    writeln!(s, "fn procedural_texture(uv: vec2<f32>) -> f32 {{").unwrap();

    if let Some(octaves) = result.octaves.first() {
        let bias = result.bias.first().copied().unwrap_or(0.0);
        writeln!(s, "    var value: f32 = {:.6};", bias).unwrap();

        for oct in octaves {
            if oct.rotation.abs() > 1e-6 {
                writeln!(
                    s,
                    "    // rotated octave (freq={:.1}, seed={})",
                    oct.frequency, oct.seed
                )
                .unwrap();
                writeln!(s, "    {{").unwrap();
                writeln!(s, "        let s = sin({:.6});", oct.rotation).unwrap();
                writeln!(s, "        let c = cos({:.6});", oct.rotation).unwrap();
                writeln!(
                    s,
                    "        let ruv = vec2<f32>(uv.x * c - uv.y * s, uv.x * s + uv.y * c);"
                )
                .unwrap();
                writeln!(s, "        value += {:.6} * hash_noise_3d(vec3<f32>(ruv * {:.6} + vec2<f32>({:.6}, {:.6}), 0.0), {}u);",
                    oct.amplitude, oct.frequency, oct.phase[0], oct.phase[1], oct.seed).unwrap();
                writeln!(s, "    }}").unwrap();
            } else {
                writeln!(s, "    value += {:.6} * hash_noise_3d(vec3<f32>(uv * {:.6} + vec2<f32>({:.6}, {:.6}), 0.0), {}u);",
                    oct.amplitude, oct.frequency, oct.phase[0], oct.phase[1], oct.seed).unwrap();
            }
        }
    }

    writeln!(s, "    return clamp(value, 0.0, 1.0);").unwrap();
    writeln!(s, "}}").unwrap();
    s
}

fn generate_hlsl(result: &TextureFitResult, source_name: &str) -> String {
    let mut s = String::with_capacity(2048);
    let total_octaves: usize = result.octaves.iter().map(|o| o.len()).sum();

    writeln!(
        s,
        "// ALICE-SDF Procedural Texture (fitted from {})",
        source_name
    )
    .unwrap();
    writeln!(
        s,
        "// PSNR: {:.1} dB, {} octaves",
        result.psnr_db, total_octaves
    )
    .unwrap();
    writeln!(s).unwrap();

    s.push_str(HASH_NOISE_HLSL);
    writeln!(s).unwrap();
    writeln!(s).unwrap();

    writeln!(s, "float procedural_texture(float2 uv) {{").unwrap();

    if let Some(octaves) = result.octaves.first() {
        let bias = result.bias.first().copied().unwrap_or(0.0);
        writeln!(s, "    float value = {:.6};", bias).unwrap();

        for oct in octaves {
            if oct.rotation.abs() > 1e-6 {
                writeln!(s, "    {{").unwrap();
                writeln!(s, "        float s = sin({:.6});", oct.rotation).unwrap();
                writeln!(s, "        float c = cos({:.6});", oct.rotation).unwrap();
                writeln!(
                    s,
                    "        float2 ruv = float2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);"
                )
                .unwrap();
                writeln!(s, "        value += {:.6} * hash_noise_3d(float3(ruv * {:.6} + float2({:.6}, {:.6}), 0.0), {}u);",
                    oct.amplitude, oct.frequency, oct.phase[0], oct.phase[1], oct.seed).unwrap();
                writeln!(s, "    }}").unwrap();
            } else {
                writeln!(s, "    value += {:.6} * hash_noise_3d(float3(uv * {:.6} + float2({:.6}, {:.6}), 0.0), {}u);",
                    oct.amplitude, oct.frequency, oct.phase[0], oct.phase[1], oct.seed).unwrap();
            }
        }
    }

    writeln!(s, "    return saturate(value);").unwrap();
    writeln!(s, "}}").unwrap();
    s
}

fn generate_glsl(result: &TextureFitResult, source_name: &str) -> String {
    let mut s = String::with_capacity(2048);
    let total_octaves: usize = result.octaves.iter().map(|o| o.len()).sum();

    writeln!(
        s,
        "// ALICE-SDF Procedural Texture (fitted from {})",
        source_name
    )
    .unwrap();
    writeln!(
        s,
        "// PSNR: {:.1} dB, {} octaves",
        result.psnr_db, total_octaves
    )
    .unwrap();
    writeln!(s).unwrap();

    s.push_str(HASH_NOISE_GLSL);
    writeln!(s).unwrap();
    writeln!(s).unwrap();

    writeln!(s, "float procedural_texture(vec2 uv) {{").unwrap();

    if let Some(octaves) = result.octaves.first() {
        let bias = result.bias.first().copied().unwrap_or(0.0);
        writeln!(s, "    float value = {:.6};", bias).unwrap();

        for oct in octaves {
            if oct.rotation.abs() > 1e-6 {
                writeln!(s, "    {{").unwrap();
                writeln!(s, "        float s = sin({:.6});", oct.rotation).unwrap();
                writeln!(s, "        float c = cos({:.6});", oct.rotation).unwrap();
                writeln!(
                    s,
                    "        vec2 ruv = vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);"
                )
                .unwrap();
                writeln!(s, "        value += {:.6} * hash_noise_3d(vec3(ruv * {:.6} + vec2({:.6}, {:.6}), 0.0), {}u);",
                    oct.amplitude, oct.frequency, oct.phase[0], oct.phase[1], oct.seed).unwrap();
                writeln!(s, "    }}").unwrap();
            } else {
                writeln!(s, "    value += {:.6} * hash_noise_3d(vec3(uv * {:.6} + vec2({:.6}, {:.6}), 0.0), {}u);",
                    oct.amplitude, oct.frequency, oct.phase[0], oct.phase[1], oct.seed).unwrap();
            }
        }
    }

    writeln!(s, "    return clamp(value, 0.0, 1.0);").unwrap();
    writeln!(s, "}}").unwrap();
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::FittedOctave;

    fn make_test_result() -> TextureFitResult {
        TextureFitResult {
            width: 256,
            height: 256,
            channels: 1,
            bias: vec![0.4523],
            octaves: vec![vec![
                FittedOctave {
                    amplitude: 0.312,
                    frequency: 4.0,
                    phase: [0.5, 0.3],
                    seed: 42,
                    rotation: 0.0,
                },
                FittedOctave {
                    amplitude: 0.156,
                    frequency: 8.0,
                    phase: [0.1, 0.7],
                    seed: 7,
                    rotation: 0.3,
                },
            ]],
            psnr_db: 28.5,
            nmse: 0.15,
        }
    }

    #[test]
    fn test_wgsl_output() {
        let result = make_test_result();
        let shader = generate_shader(&result, ShaderLanguage::Wgsl, "granite.png");
        assert!(shader.contains("fn hash_noise_3d("));
        assert!(shader.contains("fn procedural_texture("));
        assert!(shader.contains("0.452300"));
        assert!(shader.contains("42u"));
        assert!(shader.contains("return clamp(value, 0.0, 1.0);"));
    }

    #[test]
    fn test_hlsl_output() {
        let result = make_test_result();
        let shader = generate_shader(&result, ShaderLanguage::Hlsl, "granite.png");
        assert!(shader.contains("float hash_noise_3d(float3"));
        assert!(shader.contains("float procedural_texture(float2"));
        assert!(shader.contains("return saturate(value);"));
    }

    #[test]
    fn test_glsl_output() {
        let result = make_test_result();
        let shader = generate_shader(&result, ShaderLanguage::Glsl, "granite.png");
        assert!(shader.contains("float hash_noise_3d(vec3"));
        assert!(shader.contains("float procedural_texture(vec2"));
        assert!(shader.contains("return clamp(value, 0.0, 1.0);"));
    }

    #[test]
    fn test_rotated_octave_wgsl() {
        let result = make_test_result();
        let shader = generate_shader(&result, ShaderLanguage::Wgsl, "test.png");
        // Second octave has rotation, should generate rotation code
        assert!(shader.contains("let s = sin("));
        assert!(shader.contains("let c = cos("));
        assert!(shader.contains("let ruv ="));
    }

    #[test]
    fn test_shader_language_parse() {
        assert_eq!(
            "wgsl".parse::<ShaderLanguage>().unwrap(),
            ShaderLanguage::Wgsl
        );
        assert_eq!(
            "HLSL".parse::<ShaderLanguage>().unwrap(),
            ShaderLanguage::Hlsl
        );
        assert_eq!(
            "Glsl".parse::<ShaderLanguage>().unwrap(),
            ShaderLanguage::Glsl
        );
        assert!("foobar".parse::<ShaderLanguage>().is_err());
    }
}
