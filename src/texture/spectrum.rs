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
    /// DCT-II coefficient index `k` of the band (`k / 2` cycles across the
    /// image, so a lattice noise with `f` cells across the image peaks around
    /// `k ≈ f / 2` … `f`); a starting point for the fit, not a measurement.
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

    let (row_energy, col_energy) = dct_energies(data, width, height, mean, max_coeffs);

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

    bands.sort_by(|a, b| b.energy.total_cmp(&a.energy));
    bands.truncate(max_bands);
    bands
}
/// Coefficient `k` (1 ≤ k < `max_coeffs`) of a DCT-II over `samples`, squared;
/// index 0 stays 0
fn dct_energy(
    samples: impl Iterator<Item = f32> + Clone,
    len: usize,
    mean: f64,
    max_coeffs: usize,
) -> Vec<f64> {
    let mut local_energy = vec![0.0f64; max_coeffs];
    for (k, energy) in local_energy[1..max_coeffs.min(len)].iter_mut().enumerate() {
        let k = k + 1;
        let mut sum = 0.0f64;
        for (n, val) in samples.clone().enumerate() {
            let angle =
                std::f64::consts::PI * (2.0f64 * (n as f64) + 1.0) * k as f64 / (2.0 * len as f64);
            sum += (f64::from(val) - mean) * alice_det_math::cos64(angle);
        }
        let coeff = sum / len as f64;
        *energy = coeff * coeff;
    }
    local_energy
}

/// The squared DCT-II coefficients of every row and of every column, summed per
/// coefficient. The rows and columns are transformed in parallel and collected in
/// index order, then summed in that order: a parallel reduce would add them in an
/// order that depends on how rayon splits the work, so the f64 sums would depend
/// on the thread count (ALICE-Law G13).
fn dct_energies(
    data: &[f32],
    width: usize,
    height: usize,
    mean: f64,
    max_coeffs: usize,
) -> (Vec<f64>, Vec<f64>) {
    // [Deep Fried] Parallel DCT-II along rows, then along columns
    let rows: Vec<Vec<f64>> = (0..height)
        .into_par_iter()
        .map(|y| {
            dct_energy(
                data[y * width..(y + 1) * width].iter().copied(),
                width,
                mean,
                max_coeffs,
            )
        })
        .collect();
    let cols: Vec<Vec<f64>> = (0..width)
        .into_par_iter()
        .map(|x| {
            dct_energy(
                data.chunks(width).map(move |row| row[x]),
                height,
                mean,
                max_coeffs,
            )
        })
        .collect();
    (
        sum_in_order(&rows, max_coeffs),
        sum_in_order(&cols, max_coeffs),
    )
}

/// Per-coefficient sum of `parts`, in the order of `parts`
fn sum_in_order(parts: &[Vec<f64>], max_coeffs: usize) -> Vec<f64> {
    let mut total = vec![0.0f64; max_coeffs];
    for part in parts {
        for (t, p) in total.iter_mut().zip(part) {
            *t += p;
        }
    }
    total
}
#[allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
#[cfg(test)]
mod tests {
    use super::*;

    /// The energies are the rows' (and columns') spectra summed in index order, not
    /// in an order that depends on how rayon splits the work (ALICE-Law G13): on
    /// pools of 1, 2, 3 and 8 threads, 3 runs each, the f64 sums are the bits of the
    /// in-order sum of the per-row and per-column spectra computed one by one
    #[test]
    fn the_energy_sums_are_in_index_order_on_every_thread_count() {
        let (w, h) = (48usize, 1500usize);
        let data: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                // rows of very different energy, so the order of the additions shows
                let scale = if (i / w) % 7 == 0 { 1.0e3 } else { 1.0e-3 };
                0.5 + scale * alice_det_math::sin(x * 0.37 + y * 0.011) + 1e-3 * (i % 97) as f32
            })
            .collect();
        let mean = data.iter().map(|&v| f64::from(v)).sum::<f64>() / (w * h) as f64;
        let serial = |n: usize, part: &dyn Fn(usize) -> Vec<f64>| {
            let mut total = [0.0f64; 16];
            for i in 0..n {
                for (t, p) in total.iter_mut().zip(part(i)) {
                    *t += p;
                }
            }
            total.iter().map(|v| v.to_bits()).collect::<Vec<u64>>()
        };
        let want_rows = serial(h, &|y| {
            dct_energy(data[y * w..(y + 1) * w].iter().copied(), w, mean, 16)
        });
        let want_cols = serial(w, &|x| {
            dct_energy(data.chunks(w).map(move |r| r[x]), h, mean, 16)
        });
        for threads in [1, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for _ in 0..3 {
                let (rows, cols) = pool.install(|| dct_energies(&data, w, h, mean, 16));
                let bits = |v: &[f64]| v.iter().map(|v| v.to_bits()).collect::<Vec<u64>>();
                assert_eq!(bits(&rows), want_rows, "rows, {threads} threads");
                assert_eq!(bits(&cols), want_cols, "columns, {threads} threads");
            }
        }
    }

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
                data[y * w + x] = 0.4f32.mul_add((u * 4.0 * 2.0 * std::f32::consts::PI).sin(), 0.5);
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
