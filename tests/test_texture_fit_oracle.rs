//! Texture-fit oracle: the optimizer is checked on functions with a known
//! minimum, and the fitter on an image synthesized from the very law it
//! fits (`bias + Σ aᵢ · noise(uv · fᵢ + φᵢ, seedᵢ)`), so the answer is
//! known exactly. The reported PSNR / NMSE must agree with the values
//! recomputed from `reconstruct` at the source resolution.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
#![cfg(feature = "texture-fit")]

#[path = "common/temp.rs"]
mod temp;

use alice_sdf::io::IoError;
use alice_sdf::provenance::{self, meaning_id, Provenance, StoredForm, StoredSemantics};
use alice_sdf::texture::{
    eval_octave, fit_texture, generate_shader, hash_noise_3d_cpu, load_texture_fit,
    load_texture_fit_unverified, nelder_mead, reconstruct, save_texture_fit,
    save_texture_fit_with_provenance, FittedOctave, ShaderLanguage, TextureFitConfig,
    TextureFitResult,
};

fn temp_png(name: &str, width: u32, height: u32, pixels: &[f32]) -> temp::TempPath {
    let path = temp::TempPath::new("alice_sdf_texture_fit_oracle", name);
    let bytes: Vec<u8> = pixels
        .iter()
        .map(|&p| (p.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    image::GrayImage::from_raw(width, height, bytes)
        .unwrap()
        .save(&path)
        .unwrap();
    path
}

/// Independent evaluation of one octave, written from the module doc rather
/// than by calling `eval_octave`, so the exported evaluator is checked too.
/// Plain arithmetic on purpose (the crate uses `mul_add`; agreement to 1e-4
/// is part of what is checked).
#[allow(clippy::suboptimal_flops)]
fn octave_reference(
    u: f32,
    v: f32,
    amp: f32,
    freq: f32,
    phase: [f32; 2],
    seed: u32,
    rot: f32,
) -> f32 {
    let (s, c) = rot.sin_cos();
    let ru = u * c - v * s;
    let rv = u * s + v * c;
    amp * hash_noise_3d_cpu(ru * freq + phase[0], rv * freq + phase[1], 0.0, seed)
}

fn psnr(pixels: &[f32], recon: &[f32]) -> f64 {
    let mse = pixels
        .iter()
        .zip(recon)
        .map(|(&p, &r)| ((p - r) as f64).powi(2))
        .sum::<f64>()
        / pixels.len() as f64;
    10.0 * (1.0 / mse).log10()
}

// ───────────────────────────── optimizer ─────────────────────────────

#[test]
fn nelder_mead_finds_the_minimum_of_a_quadratic_bowl() {
    let target = [1.5f32, -2.0, 0.25];
    let r = nelder_mead(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], 2000, |p| {
        p.iter()
            .zip(&target)
            .map(|(&x, &t)| ((x - t) as f64).powi(2))
            .sum()
    });
    for (x, t) in r.params.iter().zip(&target) {
        assert!(
            (x - t).abs() < 1e-3,
            "bowl: got {:?}, want {:?}",
            r.params,
            target
        );
    }
    assert!(r.cost < 1e-6, "cost {}", r.cost);
    assert!(r.iterations <= 2000);
}

#[test]
#[allow(clippy::suboptimal_flops)] // textbook form of the function
fn nelder_mead_reaches_the_rosenbrock_valley_floor() {
    // f(x, y) = (1 − x)² + 100 (y − x²)², minimum 0 at (1, 1); the classic
    // curved-valley test that a broken contraction / shrink step fails.
    let r = nelder_mead(&[-1.2, 1.0], &[0.5, 0.5], 5000, |p| {
        let (x, y) = (p[0] as f64, p[1] as f64);
        (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
    });
    assert!(
        (r.params[0] - 1.0).abs() < 2e-2 && (r.params[1] - 1.0).abs() < 4e-2,
        "rosenbrock: got {:?} cost {}",
        r.params,
        r.cost
    );
    assert!(r.cost < 1e-3, "cost {}", r.cost);
}

#[test]
#[allow(clippy::suboptimal_flops)]
fn nelder_mead_never_returns_a_point_worse_than_the_start() {
    // Monotone: the returned cost is the best simplex vertex, which can never
    // exceed the initial point (the initial point is a vertex).
    let f = |p: &[f32]| {
        ((p[0] as f64).sin() * 3.0 + (p[1] as f64).cos()).abs() + 0.1 * (p[0] as f64).powi(2)
    };
    for start in [[0.3f32, 0.7], [2.0, -1.0], [-3.0, 3.0]] {
        let r = nelder_mead(&start, &[0.2, 0.2], 300, f);
        assert!(
            r.cost <= f(&start) + 1e-12,
            "start {:?}: {} > {}",
            start,
            r.cost,
            f(&start)
        );
    }
}

// ───────────────────────────── noise law ─────────────────────────────

#[test]
fn eval_octave_matches_the_documented_law() {
    let mut worst = 0.0f32;
    for i in 0..500 {
        let t = i as f32 * 0.618_034;
        let u = t.fract();
        let v = (t * 1.7).fract();
        let amp = 0.3 + (t * 0.1).fract();
        let freq = 6.0f32.mul_add((t * 0.37).fract(), 1.0);
        let phase = [(t * 3.1).fract() * 4.0, (t * 2.3).fract() * 4.0];
        let seed = i % 7;
        let rot = (t * 0.91).fract() * 6.0;
        let a = eval_octave(u, v, amp, freq, phase, seed, rot);
        let b = octave_reference(u, v, amp, freq, phase, seed, rot);
        worst = worst.max((a - b).abs());
    }
    // mul_add vs separate multiply-add: a few ulp, nothing structural.
    assert!(worst < 1e-4, "eval_octave vs reference: worst {}", worst);
}

#[test]
fn hash_noise_is_in_range_and_continuous() {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    let mut worst_jump = 0.0f32;
    let h = 1e-3;
    for i in 0..2000 {
        let t = i as f32 * 0.013;
        let (x, y, z) = (t * 3.7, t * 2.1 + 0.5, t * 0.9);
        let n = hash_noise_3d_cpu(x, y, z, 3);
        lo = lo.min(n);
        hi = hi.max(n);
        let n2 = hash_noise_3d_cpu(x + h, y, z, 3);
        worst_jump = worst_jump.max((n2 - n).abs());
    }
    assert!(lo >= -1.0 && hi <= 1.0, "range [{}, {}]", lo, hi);
    assert!(
        hi - lo > 1.0,
        "noise should use most of its range, got [{}, {}]",
        lo,
        hi
    );
    // Trilinear interpolation between corners in [0, 1]: slope ≤ 2 per cell
    // in each axis, so |Δn| ≤ 2 h · (1 + O(h)).
    assert!(
        worst_jump < 4.0 * h,
        "noise is not continuous: jump {} over h {}",
        worst_jump,
        h
    );
}

// ───────────────────────────── fitter ─────────────────────────────

#[test]
fn flat_image_fits_with_no_octaves_and_the_bias_alone() {
    let (w, h) = (48u32, 40u32);
    let pixels = vec![0.6f32; (w * h) as usize];
    let path = temp_png("flat.png", w, h, &pixels);
    let r = fit_texture(&path, &TextureFitConfig::default())
        .unwrap()
        .into_parts()
        .0;
    assert_eq!(r.width, w);
    assert_eq!(r.height, h);
    assert_eq!(r.channels, 1);
    // 0.6 · 255 = 153 exactly, so the decoded bias is exact too.
    assert!(
        (r.bias[0] - 153.0 / 255.0).abs() < 1e-6,
        "bias {}",
        r.bias[0]
    );
    assert!(
        r.octaves[0].is_empty(),
        "flat image should need no octaves: {:?}",
        r.octaves[0]
    );
    assert!(r.psnr_db >= 100.0, "psnr {}", r.psnr_db);
    assert_eq!(r.nmse, 0.0);
    let recon = reconstruct(&r, w as usize, h as usize);
    assert!(recon.iter().all(|&p| (p - 153.0 / 255.0).abs() < 1e-6));
}

/// One octave synthesized from the fitter's own law (seed 0, the seed the
/// fitter assigns to its first octave) must be recovered well enough that
/// the residual is small, and the reported metrics must be the metrics of
/// the reconstruction.
#[test]
fn synthesized_single_octave_is_recovered_and_metrics_are_honest() {
    let (w, h) = (64usize, 64usize);
    let (bias, amp, freq, phase, seed, rot) = (0.5f32, 0.3f32, 4.0f32, [0.7f32, 1.3], 0u32, 0.0f32);
    let mut pixels = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
            pixels.push(bias + octave_reference(u, v, amp, freq, phase, seed, rot));
        }
    }
    let path = temp_png("one_octave.png", w as u32, h as u32, &pixels);
    // Re-read the 8-bit quantized image: that is what the fitter sees.
    let quantized: Vec<f32> = pixels
        .iter()
        .map(|&p| (p * 255.0).round() / 255.0)
        .collect();

    let config = TextureFitConfig {
        max_octaves: 3,
        ..TextureFitConfig::default()
    };
    let r = fit_texture(&path, &config).unwrap().into_parts().0;
    let n = r.octaves[0].len();
    assert!(n >= 1, "no octave fitted");

    let recon = reconstruct(&r, w, h);
    let measured = psnr(&quantized, &recon);
    eprintln!(
        "one octave: fitted {} octave(s), reported psnr {:.2} dB, measured {:.2} dB, nmse {:.4}, first = {:?}",
        n, r.psnr_db, measured, r.nmse, r.octaves[0][0]
    );

    // (1) The reported PSNR is the PSNR of the reconstruction (clamp aside).
    assert!(
        (r.psnr_db as f64 - measured).abs() < 0.5,
        "reported psnr {} vs measured {}",
        r.psnr_db,
        measured
    );
    // (2) NMSE = MSE / Var is consistent with the PSNR.
    let mse = 10f64.powf(-(r.psnr_db as f64) / 10.0);
    let mean = quantized.iter().map(|&p| p as f64).sum::<f64>() / quantized.len() as f64;
    let var = quantized
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / quantized.len() as f64;
    assert!(
        ((r.nmse as f64) - mse / var).abs() < 0.02,
        "nmse {} vs mse/var {}",
        r.nmse,
        mse / var
    );
    // (3) A texture that *is* one octave of the law must be explained well:
    //     most of the variance gone. The greedy fit starts from the DCT band
    //     estimate and Nelder-Mead in 5 parameters; it does not always land
    //     on the exact (amp, freq, phase), so the bar is the residual, not
    //     the parameters.
    assert!(
        r.nmse < 0.15,
        "nmse {} — the fitter did not explain its own law (measured 0.085 at 1.13.0)",
        r.nmse
    );
    assert!(r.psnr_db > 27.0, "psnr {}", r.psnr_db);
}

#[test]
fn more_octaves_never_raise_the_residual() {
    let (w, h) = (64usize, 64usize);
    let mut pixels = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
            let p = 0.5
                + octave_reference(u, v, 0.25, 3.0, [0.2, 0.9], 0, 0.0)
                + octave_reference(u, v, 0.12, 7.0, [1.1, 0.4], 1, 0.3)
                + octave_reference(u, v, 0.06, 13.0, [0.0, 2.2], 2, 0.0);
            pixels.push(p);
        }
    }
    let path = temp_png("three_octaves.png", w as u32, h as u32, &pixels);
    let mut last = f32::INFINITY;
    for max_octaves in [1u32, 2, 4] {
        let config = TextureFitConfig {
            max_octaves,
            target_psnr_db: 100.0, // never stop early
            ..TextureFitConfig::default()
        };
        let r = fit_texture(&path, &config).unwrap().into_parts().0;
        eprintln!(
            "max_octaves {}: fitted {} nmse {:.4} psnr {:.2}",
            max_octaves,
            r.octaves[0].len(),
            r.nmse,
            r.psnr_db
        );
        assert!(
            r.nmse <= last + 1e-4,
            "nmse rose from {} to {} at max_octaves {}",
            last,
            r.nmse,
            max_octaves
        );
        assert!(r.octaves[0].len() as u32 <= max_octaves);
        last = r.nmse;
    }
}

#[test]
fn fit_is_deterministic() {
    let (w, h) = (40usize, 40usize);
    let pixels: Vec<f32> = (0..w * h)
        .map(|i| {
            let (u, v) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
            0.5 + octave_reference(u, v, 0.3, 5.0, [0.3, 0.3], 0, 0.0)
        })
        .collect();
    let path = temp_png("det.png", w as u32, h as u32, &pixels);
    let config = TextureFitConfig {
        max_octaves: 2,
        iterations_per_octave: 200,
        ..TextureFitConfig::default()
    };
    let a = fit_texture(&path, &config).unwrap().into_parts().0;
    let b = fit_texture(&path, &config).unwrap().into_parts().0;
    assert_eq!(a.psnr_db.to_bits(), b.psnr_db.to_bits());
    assert_eq!(a.octaves[0].len(), b.octaves[0].len());
    for (x, y) in a.octaves[0].iter().zip(&b.octaves[0]) {
        assert_eq!(x.amplitude.to_bits(), y.amplitude.to_bits());
        assert_eq!(x.frequency.to_bits(), y.frequency.to_bits());
        assert_eq!(x.phase, y.phase);
        assert_eq!(x.seed, y.seed);
    }
}

/// Non-multiple-of-8 sample counts exercise the padded SIMD lanes of the
/// cost function: 45 × 45 = 2025 samples, 7 padding lanes. The fit must
/// be as good as on a 48 × 48 (no padding) version of the same texture;
/// until 1.13.0 the padding lanes contributed `amp² · noise(phase)²` to
/// the cost and biased the optimizer.
#[test]
fn padded_sample_grid_fits_as_well_as_an_aligned_one() {
    let make = |w: usize, h: usize| -> Vec<f32> {
        (0..w * h)
            .map(|i| {
                let (u, v) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
                0.5 + octave_reference(u, v, 0.3, 4.0, [0.7, 1.3], 0, 0.0)
            })
            .collect()
    };
    let config = TextureFitConfig {
        max_octaves: 1,
        ..TextureFitConfig::default()
    };
    let padded = fit_texture(&temp_png("pad45.png", 45, 45, &make(45, 45)), &config)
        .unwrap()
        .into_parts()
        .0;
    let aligned = fit_texture(&temp_png("pad48.png", 48, 48, &make(48, 48)), &config)
        .unwrap()
        .into_parts()
        .0;
    eprintln!(
        "padded nmse {:.4} aligned nmse {:.4}",
        padded.nmse, aligned.nmse
    );
    assert!(
        padded.nmse < aligned.nmse + 0.05,
        "padded grid fit is worse: {} vs {}",
        padded.nmse,
        aligned.nmse
    );
}

// ───────────────────────────── generated shaders ─────────────────────────────

/// The WGSL / GLSL the tool emits must at least parse and validate (naga);
/// HLSL has no naga front end. (A GPU parity run of these against
/// `reconstruct` is still pending — see CHANGELOG.)
#[test]
fn generated_shaders_validate_with_naga() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let (w, h) = (32usize, 32usize);
    let pixels: Vec<f32> = (0..w * h)
        .map(|i| {
            let (u, v) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
            0.5 + octave_reference(u, v, 0.3, 3.0, [0.2, 0.9], 0, 0.4)
                + octave_reference(u, v, 0.1, 9.0, [1.0, 0.1], 1, 0.0)
        })
        .collect();
    let path = temp_png("shader.png", w as u32, h as u32, &pixels);
    let config = TextureFitConfig {
        max_octaves: 2,
        target_psnr_db: 100.0,
        iterations_per_octave: 100,
        ..TextureFitConfig::default()
    };
    let r = fit_texture(&path, &config).unwrap().into_parts().0;
    assert_eq!(
        r.octaves[0].len(),
        2,
        "want both octaves (rotated and axis-aligned emit paths)"
    );
    let validate = |m: &naga::Module| {
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(m)
            .map(|_| ())
            .map_err(|e| format!("{e:?}"))
    };

    let wgsl = generate_shader(&r, ShaderLanguage::Wgsl, "shader.png");
    let entry = format!(
        "{wgsl}\n@fragment fn alice_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {{ return vec4<f32>(procedural_texture(uv)); }}\n"
    );
    let module = naga::front::wgsl::parse_str(&entry)
        .unwrap_or_else(|e| panic!("WGSL parse: {e:?}\n{wgsl}"));
    validate(&module).unwrap_or_else(|e| panic!("WGSL validate: {e}\n{wgsl}"));

    let glsl = generate_shader(&r, ShaderLanguage::Glsl, "shader.png");
    let entry = format!(
        "{glsl}\nin vec2 alice_uv;\nout vec4 alice_frag;\nvoid main() {{ alice_frag = vec4(procedural_texture(alice_uv)); }}\n"
    );
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options {
        stage: naga::ShaderStage::Fragment,
        defines: naga::FastHashMap::default(),
    };
    let module = frontend
        .parse(&options, &entry)
        .unwrap_or_else(|e| panic!("GLSL parse: {e:?}\n{glsl}"));
    validate(&module).unwrap_or_else(|e| panic!("GLSL validate: {e}\n{glsl}"));

    let hlsl = generate_shader(&r, ShaderLanguage::Hlsl, "shader.png");
    assert!(
        hlsl.contains("float procedural_texture(float2 uv)"),
        "{hlsl}"
    );
    assert_eq!(
        hlsl.matches("hash_noise_3d(").count(),
        2 + 1,
        "one call per octave plus the definition:\n{hlsl}"
    );
}

// ---------------------------------------------------------------------------
// Stored fit: the semantics identifier (ALICE-Law G8)
// ---------------------------------------------------------------------------

/// (writer name as the scanner reads it, what it writes): the stored fit is written, read back
/// through `provenance::read` and judged by the loaders below
const FIT_WRITERS: &[(&str, &str)] =
    &[("src/texture/mod.rs::save_texture_fit", "texture fit JSON")];

fn sample_fit() -> TextureFitResult {
    TextureFitResult {
        width: 16,
        height: 8,
        channels: 1,
        bias: vec![0.25],
        octaves: vec![vec![
            FittedOctave {
                amplitude: 0.5,
                frequency: 3.0,
                phase: [0.125, -0.75],
                seed: 7,
                rotation: 0.3,
            },
            FittedOctave {
                amplitude: -0.0625,
                frequency: 11.5,
                phase: [1.0e-7, 2.5],
                seed: 4_000_000_000,
                rotation: -1.25,
            },
        ]],
        psnr_db: 31.5,
        nmse: 0.0123,
    }
}

fn fit_path(name: &str) -> temp::TempPath {
    temp::TempPath::new("alice_sdf_texture_fit_provenance", name)
}

/// `save_texture_fit` stores this crate's identifier, `save_texture_fit_with_provenance` the
/// caller's; this crate's loads back bit for bit, another one is refused by both loaders
#[test]
fn texture_fit_writer_stores_and_loaders_check_the_identifier() {
    assert!(FIT_WRITERS
        .iter()
        .any(|(w, _)| *w == "src/texture/mod.rs::save_texture_fit"));
    let fit = sample_fit();
    let own = fit_path("own.json");
    save_texture_fit(&fit, &own).unwrap();
    assert!(provenance::read(&own)
        .unwrap()
        .is_written_under(&Provenance::sdf()));
    let back = load_texture_fit(&own).unwrap();
    assert_eq!(back, fit);
    let bits = |r: &TextureFitResult| -> Vec<u32> {
        r.octaves
            .iter()
            .flatten()
            .flat_map(|o| [o.amplitude, o.frequency, o.phase[0], o.phase[1], o.rotation])
            .chain(r.bias.iter().copied())
            .chain([r.psnr_db, r.nmse])
            .map(f32::to_bits)
            .collect()
    };
    assert_eq!(bits(&back), bits(&fit), "the fit reads back bit for bit");
    assert_eq!(
        load_texture_fit_unverified(&own).unwrap().semantics(),
        StoredSemantics::Current
    );

    // an identifier this crate published before and no longer reproduces (`6e98f0be…`,
    // alice-det-math 0.5.2 at 39507d1b): the writer stores the document form's meaning identifier
    // under it (ALICE-Law G8), refused by both loaders naming both meaning identifiers; the shared
    // judge given it as an earlier identifier reads the document as VerifiedPrevious
    let hex = "6e98f0bed3f8691ca470ae2fe2c404c893856a3e3f6a9cda9d67528ae8074b37";
    let earlier = Provenance::new(core::array::from_fn(|i| {
        u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()
    }));
    assert_ne!(earlier, Provenance::sdf());
    assert!(!alice_sdf::PREVIOUS_SDF_SEMANTICS_IDS.contains(&earlier.semantics_id));
    let prev = fit_path("earlier.json");
    save_texture_fit_with_provenance(&fit, &prev, &earlier).unwrap();
    for err in [
        load_texture_fit(&prev).map(|_| ()).unwrap_err(),
        load_texture_fit_unverified(&prev).map(|_| ()).unwrap_err(),
    ] {
        match err {
            IoError::MeaningMismatch {
                form,
                stored,
                expected,
                ..
            } => {
                assert_eq!(form, StoredForm::JsonDocument);
                assert_eq!(
                    (stored, expected),
                    (
                        Provenance::new(meaning_id(form, &earlier)).hex(),
                        Provenance::new(meaning_id(form, &Provenance::sdf())).hex()
                    )
                );
            }
            other => panic!("the earlier identifier was not refused: {other:?}"),
        }
    }
    assert_eq!(
        provenance::check(&prev, &Provenance::sdf(), &[earlier.semantics_id]).unwrap(),
        StoredSemantics::VerifiedPrevious(earlier)
    );

    let other_id = Provenance::new([0xA5; 32]);
    let other = fit_path("other.json");
    save_texture_fit_with_provenance(&fit, &other, &other_id).unwrap();
    assert!(provenance::read(&other)
        .unwrap()
        .is_written_under(&other_id));
    for err in [
        load_texture_fit(&other).unwrap_err(),
        load_texture_fit_unverified(&other).unwrap_err(),
    ] {
        match err {
            IoError::MeaningMismatch {
                form,
                stored,
                expected,
                ..
            } => {
                assert_eq!(stored, Provenance::new(meaning_id(form, &other_id)).hex());
                assert_eq!(
                    expected,
                    Provenance::new(meaning_id(form, &Provenance::sdf())).hex()
                );
            }
            e => panic!("expected MeaningMismatch, got {e:?}"),
        }
    }
}

/// A fit the CLI wrote before the identifier (the bare fields, no `format`) is refused by
/// default and read as `Unpinned` by the explicit loader; a half-stamped or unknown-format
/// document is an error, never read as unstamped
#[test]
fn texture_fit_without_identifier_is_unpinned_and_malformed_is_refused() {
    let fit = sample_fit();
    let own = fit_path("for_legacy.json");
    save_texture_fit(&fit, &own).unwrap();
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&own).unwrap()).unwrap();
    let write = |name: &str, edit: &dyn Fn(&mut serde_json::Map<String, serde_json::Value>)| {
        let mut d = doc.as_object().unwrap().clone();
        edit(&mut d);
        let p = fit_path(name);
        std::fs::write(&p, serde_json::to_string_pretty(&d).unwrap()).unwrap();
        p
    };

    let legacy = write("legacy.json", &|d| {
        d.remove("format");
        d.remove(provenance::MEANING_KEY);
    });
    assert!(matches!(
        load_texture_fit(&legacy),
        Err(IoError::Unpinned { .. })
    ));
    let (back, semantics) = load_texture_fit_unverified(&legacy).unwrap().into_parts();
    assert_eq!(semantics, StoredSemantics::Unpinned);
    assert_eq!(back, fit);

    let no_id = write("no_id.json", &|d| {
        d.remove(provenance::MEANING_KEY);
    });
    let no_format = write("no_format.json", &|d| {
        d.remove("format");
    });
    let bad_hex = write("bad_hex.json", &|d| {
        d.insert(provenance::MEANING_KEY.into(), "zz".into());
    });
    for p in [&no_id, &no_format, &bad_hex] {
        assert!(
            matches!(
                load_texture_fit_unverified(p),
                Err(IoError::InvalidFormat(_))
            ),
            "{}",
            p.display()
        );
    }
    let format_2 = write("format_2.json", &|d| {
        d.insert("format".into(), 2.into());
    });
    assert!(matches!(
        load_texture_fit_unverified(&format_2),
        Err(IoError::UnsupportedVersion(2))
    ));
}
