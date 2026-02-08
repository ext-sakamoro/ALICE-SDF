//! DCT frequency analysis for initial parameter estimation — Deep Fried Edition
//!
//! Performs a partial DCT-II (first N coefficients only) to extract
//! dominant frequency bands from the texture. These are used as
//! initial guesses for the octave fitting.
//!
//! ## Deep Fried
//! - Row-level and column-level DCT parallelized with rayon
//! - Parallel reduce for energy accumulation

use rayon::prelude::*;

/// Dominant frequency band extracted from the image
#[derive(Debug, Clone)]
pub struct FrequencyBand {
    /// Frequency in cycles per image width
    pub frequency: f32,
    /// Relative energy (amplitude estimate)
    pub energy: f32,
}

/// [Deep Fried] Analyze dominant frequency bands with rayon parallelism.
///
/// Returns frequency bands sorted by energy (strongest first).
///
/// - `data`: grayscale pixel values in [0, 1], row-major
/// - `width`: image width
/// - `height`: image height
/// - `max_bands`: maximum number of bands to return
pub fn analyze_frequencies(
    data: &[f32],
    width: usize,
    height: usize,
    max_bands: usize,
) -> Vec<FrequencyBand> {
    let max_coeffs = 16usize;

    // DC removal
    let n_total = (width * height) as f64;
    let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n_total;

    // [Deep Fried] Parallel DCT-II along rows
    let row_energy: Vec<f64> = (0..height)
        .into_par_iter()
        .map(|y| {
            let row = &data[y * width..(y + 1) * width];
            let mut local_energy = vec![0.0f64; max_coeffs];
            for k in 1..max_coeffs.min(width) {
                let mut sum = 0.0f64;
                for n in 0..width {
                    let angle = std::f64::consts::PI * (2.0 * n as f64 + 1.0) * k as f64
                        / (2.0 * width as f64);
                    sum += (row[n] as f64 - mean) * angle.cos();
                }
                let coeff = sum / width as f64;
                local_energy[k] = coeff * coeff;
            }
            local_energy
        })
        .reduce(
            || vec![0.0f64; max_coeffs],
            |mut a, b| {
                for i in 0..max_coeffs {
                    a[i] += b[i];
                }
                a
            },
        );

    // [Deep Fried] Parallel DCT-II along columns
    let col_energy: Vec<f64> = (0..width)
        .into_par_iter()
        .map(|x| {
            let mut local_energy = vec![0.0f64; max_coeffs];
            for k in 1..max_coeffs.min(height) {
                let mut sum = 0.0f64;
                for n in 0..height {
                    let angle = std::f64::consts::PI * (2.0 * n as f64 + 1.0) * k as f64
                        / (2.0 * height as f64);
                    sum += (data[n * width + x] as f64 - mean) * angle.cos();
                }
                let coeff = sum / height as f64;
                local_energy[k] = coeff * coeff;
            }
            local_energy
        })
        .reduce(
            || vec![0.0f64; max_coeffs],
            |mut a, b| {
                for i in 0..max_coeffs {
                    a[i] += b[i];
                }
                a
            },
        );

    // Combine row and column energies
    let mut bands: Vec<FrequencyBand> = Vec::new();
    for k in 1..max_coeffs {
        let total_energy = (row_energy[k] + col_energy[k]) as f32;
        if total_energy > 1e-8 {
            bands.push(FrequencyBand {
                frequency: k as f32,
                energy: total_energy.sqrt(),
            });
        }
    }

    bands.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap());
    bands.truncate(max_bands);
    bands
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dc_signal() {
        let data = vec![0.5f32; 64 * 64];
        let bands = analyze_frequencies(&data, 64, 64, 8);
        for band in &bands {
            assert!(band.energy < 0.1, "Unexpected energy: {}", band.energy);
        }
    }

    #[test]
    fn test_periodic_signal() {
        let w = 64;
        let h = 64;
        let mut data = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let u = x as f32 / w as f32;
                data[y * w + x] = 0.5 + 0.4 * (u * 4.0 * 2.0 * std::f32::consts::PI).sin();
            }
        }
        let bands = analyze_frequencies(&data, w, h, 8);
        assert!(!bands.is_empty());
        // DCT-II coefficient k=8 corresponds to 4 full cycles in 64 pixels
        // (basis cos(π(2n+1)k/2N) has period 2N/k = 128/8 = 16 pixels = 4 cycles)
        assert!(
            bands[0].frequency >= 7.0 && bands[0].frequency <= 9.0,
            "Expected freq ~8 (DCT-II k for 4 cycles), got {}",
            bands[0].frequency
        );
    }
}
