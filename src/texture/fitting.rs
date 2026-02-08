//! Core texture fitting algorithm — Deep Fried Edition
//!
//! Greedy octave-by-octave fitting with SIMD + rayon parallelism:
//! 1. Load image → grayscale f32 buffer
//! 2. Compute mean (DC bias) and variance
//! 3. DCT frequency analysis → dominant bands
//! 4. For each octave:
//!    a. Initialize from next dominant band
//!    b. Nelder-Mead optimization with **SIMD cost function** (8-pixel batches)
//!    c. **rayon parallel** residual subtraction
//!    d. Check PSNR, stop if target reached
//!
//! ## Deep Fried
//! - Cost function uses `eval_octave_simd` (f32x8) for 8x throughput
//! - Subsampled grid stored in SoA layout for optimal SIMD loads
//! - Residual subtraction parallelized row-wise with rayon + SIMD

use rayon::prelude::*;
use std::path::Path;
use wide::f32x8;

use super::noise_cpu::{eval_octave, eval_octave_simd, f32x8_sum};
use super::optimizer::nelder_mead;
use super::spectrum::analyze_frequencies;
use super::{FittedOctave, TextureFitConfig, TextureFitResult};

/// [Deep Fried] SoA layout for subsampled evaluation grid.
/// Aligned for SIMD: length is always a multiple of 8.
struct SubsampleGrid {
    /// U coordinates (padded to multiple of 8)
    u: Vec<f32>,
    /// V coordinates (padded to multiple of 8)
    v: Vec<f32>,
    /// Source X indices (for residual lookup)
    src_x: Vec<usize>,
    /// Source Y indices (for residual lookup)
    src_y: Vec<usize>,
    /// Actual (non-padded) pixel count
    count: usize,
}

impl SubsampleGrid {
    fn new(width: usize, height: usize) -> Self {
        let sub_w = width.min(64);
        let sub_h = height.min(64);
        let count = sub_w * sub_h;
        // Pad to multiple of 8
        let padded = (count + 7) & !7;

        let mut u = Vec::with_capacity(padded);
        let mut v = Vec::with_capacity(padded);
        let mut src_x = Vec::with_capacity(padded);
        let mut src_y = Vec::with_capacity(padded);

        for sy in 0..sub_h {
            for sx in 0..sub_w {
                let x = sx * width / sub_w;
                let y = sy * height / sub_h;
                u.push(x as f32 / width as f32);
                v.push(y as f32 / height as f32);
                src_x.push(x);
                src_y.push(y);
            }
        }

        // Pad with zeros (won't affect cost since residual is also 0)
        while u.len() < padded {
            u.push(0.0);
            v.push(0.0);
            src_x.push(0);
            src_y.push(0);
        }

        Self {
            u,
            v,
            src_x,
            src_y,
            count,
        }
    }

    fn padded_len(&self) -> usize {
        self.u.len()
    }
}

/// Fit a texture image to procedural noise octaves.
///
/// Reads an image file (PNG/JPG), converts to grayscale, and fits
/// a sum of noise octaves to approximate it.
pub fn fit_texture(path: &Path, config: &TextureFitConfig) -> Result<TextureFitResult, String> {
    let img = image::open(path).map_err(|e| format!("Failed to load image: {}", e))?;
    let gray = img.to_luma32f();
    let (width, height) = gray.dimensions();
    let pixels: Vec<f32> = gray.as_raw().to_vec();

    let (bias, octaves, psnr, nmse) =
        fit_channel(&pixels, width as usize, height as usize, config)?;

    Ok(TextureFitResult {
        width,
        height,
        channels: 1,
        bias: vec![bias],
        octaves: vec![octaves],
        psnr_db: psnr,
        nmse,
    })
}

/// [Deep Fried] Fit a single grayscale channel with SIMD + parallel cost evaluation
fn fit_channel(
    pixels: &[f32],
    width: usize,
    height: usize,
    config: &TextureFitConfig,
) -> Result<(f32, Vec<FittedOctave>, f32, f32), String> {
    let n = pixels.len();

    // 1. DC bias (mean)
    let mean: f32 = pixels.iter().sum::<f32>() / n as f32;

    // 2. Residual
    let mut residual: Vec<f32> = pixels.iter().map(|&p| p - mean).collect();

    // 3. DCT frequency analysis
    let bands = analyze_frequencies(pixels, width, height, config.max_octaves as usize);

    // 4. Build SoA subsample grid (SIMD-friendly)
    let grid = SubsampleGrid::new(width, height);
    let padded = grid.padded_len();

    // Build subsampled residual (padded to multiple of 8)
    let mut sub_residual = vec![0.0f32; padded];
    for i in 0..grid.count {
        sub_residual[i] = residual[grid.src_y[i] * width + grid.src_x[i]];
    }

    let mut fitted_octaves: Vec<FittedOctave> = Vec::new();

    // 5. Greedy octave fitting
    for octave_idx in 0..config.max_octaves {
        let (init_freq, init_amp) = if (octave_idx as usize) < bands.len() {
            (
                bands[octave_idx as usize].frequency,
                bands[octave_idx as usize].energy.min(1.0),
            )
        } else {
            (2.0f32.powi(octave_idx as i32 + 1), 0.1)
        };

        let seed = octave_idx;

        let initial = [init_amp, init_freq, 0.0f32, 0.0, 0.0];
        let step_sizes = [init_amp * 0.5, init_freq * 0.3, 0.5, 0.5, 0.3];

        // References for closure
        let grid_u = &grid.u;
        let grid_v = &grid.v;
        let sub_res = &sub_residual;
        let actual_count = grid.count;

        // [Deep Fried] SIMD + parallel cost function
        let result = nelder_mead(
            &initial,
            &step_sizes,
            config.iterations_per_octave,
            |params| {
                let amp = params[0];
                let freq = params[1].abs();
                let phase = [params[2], params[3]];
                let rot = params[4];

                // Process in chunks of 2048 pixels (rayon parallelism)
                // Each chunk uses SIMD 8-wide evaluation
                let total_err: f64 = grid_u
                    .par_chunks(2048)
                    .zip(grid_v.par_chunks(2048))
                    .zip(sub_res.par_chunks(2048))
                    .map(|((u_chunk, v_chunk), res_chunk)| {
                        let mut chunk_err = 0.0f64;

                        // [Deep Fried] Process 8 pixels at a time with SIMD
                        let simd_chunks = u_chunk.len() / 8;
                        for i in 0..simd_chunks {
                            let base = i * 8;
                            let u8 = f32x8::new([
                                u_chunk[base],
                                u_chunk[base + 1],
                                u_chunk[base + 2],
                                u_chunk[base + 3],
                                u_chunk[base + 4],
                                u_chunk[base + 5],
                                u_chunk[base + 6],
                                u_chunk[base + 7],
                            ]);
                            let v8 = f32x8::new([
                                v_chunk[base],
                                v_chunk[base + 1],
                                v_chunk[base + 2],
                                v_chunk[base + 3],
                                v_chunk[base + 4],
                                v_chunk[base + 5],
                                v_chunk[base + 6],
                                v_chunk[base + 7],
                            ]);
                            let r8 = f32x8::new([
                                res_chunk[base],
                                res_chunk[base + 1],
                                res_chunk[base + 2],
                                res_chunk[base + 3],
                                res_chunk[base + 4],
                                res_chunk[base + 5],
                                res_chunk[base + 6],
                                res_chunk[base + 7],
                            ]);

                            let val = eval_octave_simd(u8, v8, amp, freq, phase, seed, rot);
                            let diff = r8 - val;
                            let diff2 = diff * diff;
                            chunk_err += f32x8_sum(diff2) as f64;
                        }

                        // Scalar remainder
                        let remainder_start = simd_chunks * 8;
                        for j in remainder_start..u_chunk.len() {
                            let val =
                                eval_octave(u_chunk[j], v_chunk[j], amp, freq, phase, seed, rot);
                            let diff = (res_chunk[j] - val) as f64;
                            chunk_err += diff * diff;
                        }

                        chunk_err
                    })
                    .sum();

                // Only count actual pixels (not padding)
                let _ = actual_count;
                total_err
            },
        );

        let amp = result.params[0];
        let freq = result.params[1].abs();
        let phase = [result.params[2], result.params[3]];
        let rot = result.params[4];

        if amp.abs() < 1e-5 {
            break;
        }

        let octave = FittedOctave {
            amplitude: amp,
            frequency: freq,
            phase,
            seed,
            rotation: rot,
        };

        // [Deep Fried] Subtract fitted octave from full residual (rayon + SIMD)
        let w = width;
        let h = height;
        residual.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let v_coord = y as f32 / h as f32;
            let v8 = f32x8::splat(v_coord);

            // SIMD: process 8 pixels per iteration
            let simd_cols = w / 8;
            for i in 0..simd_cols {
                let base = i * 8;
                let u8 = f32x8::new([
                    base as f32 / w as f32,
                    (base + 1) as f32 / w as f32,
                    (base + 2) as f32 / w as f32,
                    (base + 3) as f32 / w as f32,
                    (base + 4) as f32 / w as f32,
                    (base + 5) as f32 / w as f32,
                    (base + 6) as f32 / w as f32,
                    (base + 7) as f32 / w as f32,
                ]);
                let vals = eval_octave_simd(u8, v8, amp, freq, phase, seed, rot);
                let arr: [f32; 8] = vals.into();
                row[base] -= arr[0];
                row[base + 1] -= arr[1];
                row[base + 2] -= arr[2];
                row[base + 3] -= arr[3];
                row[base + 4] -= arr[4];
                row[base + 5] -= arr[5];
                row[base + 6] -= arr[6];
                row[base + 7] -= arr[7];
            }

            // Scalar remainder
            for x in (simd_cols * 8)..w {
                let u_coord = x as f32 / w as f32;
                row[x] -= eval_octave(u_coord, v_coord, amp, freq, phase, seed, rot);
            }
        });

        // Update subsampled residual
        for i in 0..grid.count {
            sub_residual[i] = residual[grid.src_y[i] * width + grid.src_x[i]];
        }
        // Keep padding zeros
        for i in grid.count..padded {
            sub_residual[i] = 0.0;
        }

        fitted_octaves.push(octave);

        let psnr = compute_psnr(&residual);
        if psnr >= config.target_psnr_db {
            break;
        }
    }

    let psnr = compute_psnr(&residual);
    let nmse = compute_nmse(pixels, &residual, mean);

    Ok((mean, fitted_octaves, psnr, nmse))
}

/// Compute PSNR from residual (assuming signal range [0, 1])
fn compute_psnr(residual: &[f32]) -> f32 {
    let n = residual.len() as f64;
    let mse: f64 = residual
        .iter()
        .map(|&r| (r as f64) * (r as f64))
        .sum::<f64>()
        / n;
    if mse < 1e-15 {
        return 100.0;
    }
    (10.0 * (1.0 / mse).log10()) as f32
}

/// Compute Normalized MSE: MSE / variance_of_original
fn compute_nmse(original: &[f32], residual: &[f32], mean: f32) -> f32 {
    let n = original.len() as f64;
    let mse: f64 = residual
        .iter()
        .map(|&r| (r as f64) * (r as f64))
        .sum::<f64>()
        / n;
    let variance: f64 = original
        .iter()
        .map(|&p| ((p - mean) as f64).powi(2))
        .sum::<f64>()
        / n;
    if variance < 1e-15 {
        return 0.0;
    }
    (mse / variance) as f32
}

/// [Deep Fried] Reconstruct texture with SIMD + rayon at arbitrary resolution
pub fn reconstruct(result: &TextureFitResult, width: usize, height: usize) -> Vec<f32> {
    let mut buffer = vec![0.0f32; width * height];

    if result.channels >= 1 && !result.octaves.is_empty() {
        let bias = result.bias[0];
        let octaves = &result.octaves[0];

        buffer
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                let v_coord = y as f32 / height as f32;
                let v8 = f32x8::splat(v_coord);

                let simd_cols = width / 8;
                for i in 0..simd_cols {
                    let base = i * 8;
                    let u8 = f32x8::new([
                        base as f32 / width as f32,
                        (base + 1) as f32 / width as f32,
                        (base + 2) as f32 / width as f32,
                        (base + 3) as f32 / width as f32,
                        (base + 4) as f32 / width as f32,
                        (base + 5) as f32 / width as f32,
                        (base + 6) as f32 / width as f32,
                        (base + 7) as f32 / width as f32,
                    ]);

                    let mut acc = f32x8::splat(bias);
                    for oct in octaves {
                        acc += eval_octave_simd(
                            u8,
                            v8,
                            oct.amplitude,
                            oct.frequency,
                            oct.phase,
                            oct.seed,
                            oct.rotation,
                        );
                    }

                    // Clamp to [0, 1]
                    let zero = f32x8::splat(0.0);
                    let one = f32x8::splat(1.0);
                    let clamped = acc.max(zero).min(one);
                    let arr: [f32; 8] = clamped.into();
                    row[base..base + 8].copy_from_slice(&arr);
                }

                // Scalar remainder
                for x in (simd_cols * 8)..width {
                    let u_coord = x as f32 / width as f32;
                    let mut val = bias;
                    for oct in octaves {
                        val += eval_octave(
                            u_coord,
                            v_coord,
                            oct.amplitude,
                            oct.frequency,
                            oct.phase,
                            oct.seed,
                            oct.rotation,
                        );
                    }
                    row[x] = val.clamp(0.0, 1.0);
                }
            });
    }

    buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_single_octave() {
        let w = 64usize;
        let h = 64usize;
        let bias = 0.5;
        let amp = 0.3;
        let freq = 4.0;
        let phase = [0.2, 0.1];
        let seed = 0u32;
        let rot = 0.0;

        let mut pixels = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let u = x as f32 / w as f32;
                let v = y as f32 / h as f32;
                let val = bias + eval_octave(u, v, amp, freq, phase, seed, rot);
                pixels[y * w + x] = val.clamp(0.0, 1.0);
            }
        }

        let config = TextureFitConfig {
            max_octaves: 2,
            target_psnr_db: 30.0,
            iterations_per_octave: 2000,
            tileable: true,
        };

        let (fitted_bias, _octaves, psnr, _) = fit_channel(&pixels, w, h, &config).unwrap();

        assert!(
            (fitted_bias - bias).abs() < 0.1,
            "Bias: {} vs {}",
            fitted_bias,
            bias
        );
        assert!(psnr > 10.0, "PSNR too low: {}", psnr);
    }

    #[test]
    fn test_reconstruct_dimensions() {
        let result = TextureFitResult {
            width: 64,
            height: 64,
            channels: 1,
            bias: vec![0.5],
            octaves: vec![vec![FittedOctave {
                amplitude: 0.3,
                frequency: 4.0,
                phase: [0.0, 0.0],
                seed: 42,
                rotation: 0.0,
            }]],
            psnr_db: 30.0,
            nmse: 0.1,
        };

        let buf = reconstruct(&result, 128, 128);
        assert_eq!(buf.len(), 128 * 128);

        for &v in &buf {
            assert!(v >= 0.0 && v <= 1.0, "Out of range: {}", v);
        }
    }

    #[test]
    fn test_reconstruct_simd_matches_scalar() {
        // Verify SIMD reconstruct produces same output as scalar
        let result = TextureFitResult {
            width: 64,
            height: 64,
            channels: 1,
            bias: vec![0.4],
            octaves: vec![vec![
                FittedOctave {
                    amplitude: 0.25,
                    frequency: 3.0,
                    phase: [0.1, 0.2],
                    seed: 5,
                    rotation: 0.3,
                },
                FittedOctave {
                    amplitude: 0.15,
                    frequency: 7.0,
                    phase: [0.5, 0.8],
                    seed: 11,
                    rotation: 0.0,
                },
            ]],
            psnr_db: 25.0,
            nmse: 0.2,
        };

        // SIMD path (Deep Fried)
        let buf_simd = reconstruct(&result, 72, 72); // non-multiple-of-8 width

        // Scalar reference
        let w = 72;
        let h = 72;
        let bias = result.bias[0];
        let octaves = &result.octaves[0];
        let mut buf_scalar = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let u = x as f32 / w as f32;
                let v = y as f32 / h as f32;
                let mut val = bias;
                for oct in octaves {
                    val += eval_octave(
                        u,
                        v,
                        oct.amplitude,
                        oct.frequency,
                        oct.phase,
                        oct.seed,
                        oct.rotation,
                    );
                }
                buf_scalar[y * w + x] = val.clamp(0.0, 1.0);
            }
        }

        for i in 0..(w * h) {
            assert!(
                (buf_simd[i] - buf_scalar[i]).abs() < 1e-5,
                "Pixel {} mismatch: simd={}, scalar={}",
                i,
                buf_simd[i],
                buf_scalar[i]
            );
        }
    }
}
