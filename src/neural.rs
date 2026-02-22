//! Neural SDF — small MLP that approximates a Signed Distance Field.
//!
//! Pure-Rust inference and training (no external ML dependencies).
//! Train from an existing `SdfNode` tree, then evaluate the neural
//! approximation ~10-100x faster for complex trees.
//!
//! # Example
//!
//! ```rust,no_run
//! use alice_sdf::neural::{NeuralSdf, NeuralSdfConfig};
//! use alice_sdf::SdfNode;
//! use glam::Vec3;
//!
//! let scene = SdfNode::sphere(1.0).smooth_union(
//!     SdfNode::box3d(0.5, 0.5, 0.5).translate(1.5, 0.0, 0.0), 0.3);
//!
//! let config = NeuralSdfConfig::default();
//! let nsdf = NeuralSdf::train(&scene, Vec3::splat(-3.0), Vec3::splat(3.0), &config);
//!
//! let d = nsdf.eval(Vec3::new(0.0, 0.0, 0.0));
//! ```
//!
//! Author: Moroya Sakamoto

use crate::eval::eval;
use crate::SdfNode;
use glam::Vec3;
use std::io::{Read, Write};

// ============================================================
// Configuration
// ============================================================

/// Neural SDF training/architecture configuration
#[derive(Debug, Clone)]
pub struct NeuralSdfConfig {
    /// Number of hidden layers (default 3)
    pub hidden_layers: usize,
    /// Neurons per hidden layer (default 64)
    pub hidden_width: usize,
    /// Positional encoding frequencies (default 6, 0 = disabled)
    pub pos_encoding_freqs: usize,
    /// Learning rate for Adam optimizer (default 1e-3)
    pub learning_rate: f32,
    /// Training batch size (default 4096)
    pub batch_size: usize,
    /// Number of training epochs (default 100)
    pub epochs: usize,
    /// Random seed (default 42)
    pub seed: u64,
}

impl Default for NeuralSdfConfig {
    fn default() -> Self {
        Self {
            hidden_layers: 3,
            hidden_width: 64,
            pos_encoding_freqs: 6,
            learning_rate: 1e-3,
            batch_size: 4096,
            epochs: 100,
            seed: 42,
        }
    }
}

// ============================================================
// Internal structures
// ============================================================

/// Dense layer: output = W * input + b
struct Layer {
    w: Vec<f32>, // [out_dim * in_dim] row-major
    b: Vec<f32>, // [out_dim]
    in_dim: usize,
    out_dim: usize,
}

/// Adam optimizer per-layer state
struct AdamState {
    m_w: Vec<f32>,
    v_w: Vec<f32>,
    m_b: Vec<f32>,
    v_b: Vec<f32>,
}

/// Xorshift64 PRNG
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform f32 in [0, 1)
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16777216.0
    }

    /// Uniform f32 in [lo, hi)
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Approximate normal distribution (Box-Muller)
    fn normal(&mut self, mean: f32, std: f32) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::TAU * u2).cos();
        mean + std * z
    }
}

// ============================================================
// NeuralSdf
// ============================================================

/// A small MLP that approximates a Signed Distance Field.
pub struct NeuralSdf {
    layers: Vec<Layer>,
    adam: Vec<AdamState>,
    #[allow(dead_code)] // stored for serialization round-trip; encode() derives from pos_freqs
    input_dim: usize,
    pos_freqs: usize,
    adam_t: u32,
}

impl NeuralSdf {
    /// Create a new NeuralSdf with random Xavier-initialized weights.
    pub fn new(config: &NeuralSdfConfig) -> Self {
        let input_dim = 3 + 3 * 2 * config.pos_encoding_freqs;
        let mut rng = Rng::new(config.seed);
        let mut layers = Vec::new();
        let mut adam = Vec::new();

        let mut prev_dim = input_dim;
        for i in 0..config.hidden_layers {
            let out_dim = config.hidden_width;
            let _ = i;
            layers.push(Self::make_layer(prev_dim, out_dim, &mut rng));
            adam.push(Self::make_adam(prev_dim, out_dim));
            prev_dim = out_dim;
        }
        // Output layer (linear, no activation)
        layers.push(Self::make_layer(prev_dim, 1, &mut rng));
        adam.push(Self::make_adam(prev_dim, 1));

        Self {
            layers,
            adam,
            input_dim,
            pos_freqs: config.pos_encoding_freqs,
            adam_t: 0,
        }
    }

    fn make_layer(in_dim: usize, out_dim: usize, rng: &mut Rng) -> Layer {
        // Xavier initialization: std = sqrt(2 / (in + out))
        let std = (2.0 / (in_dim + out_dim) as f32).sqrt();
        let n = in_dim * out_dim;
        let w: Vec<f32> = (0..n).map(|_| rng.normal(0.0, std)).collect();
        let b = vec![0.0; out_dim];
        Layer {
            w,
            b,
            in_dim,
            out_dim,
        }
    }

    fn make_adam(in_dim: usize, out_dim: usize) -> AdamState {
        let n = in_dim * out_dim;
        AdamState {
            m_w: vec![0.0; n],
            v_w: vec![0.0; n],
            m_b: vec![0.0; out_dim],
            v_b: vec![0.0; out_dim],
        }
    }

    /// Positional encoding: [x, y, z, sin(π·x), cos(π·x), ..., sin(2^(L-1)π·x), ...]
    fn encode(&self, p: Vec3) -> Vec<f32> {
        let mut enc = vec![p.x, p.y, p.z];
        for k in 0..self.pos_freqs {
            let freq = (1 << k) as f32 * std::f32::consts::PI;
            for &v in &[p.x, p.y, p.z] {
                enc.push((freq * v).sin());
                enc.push((freq * v).cos());
            }
        }
        enc
    }

    /// Evaluate the neural SDF at a 3D point.
    pub fn eval(&self, p: Vec3) -> f32 {
        let enc = self.encode(p);
        let mut buf_a = enc;
        let mut buf_b = Vec::new();
        let n_layers = self.layers.len();

        for (i, layer) in self.layers.iter().enumerate() {
            buf_b.resize(layer.out_dim, 0.0);
            let is_last = i == n_layers - 1;
            layer_forward(layer, &buf_a, &mut buf_b, !is_last);
            std::mem::swap(&mut buf_a, &mut buf_b);
        }

        buf_a[0]
    }

    /// Forward pass returning cached activations for backprop.
    fn forward_cached(&self, encoded: &[f32]) -> (f32, Vec<Vec<f32>>) {
        let n_layers = self.layers.len();
        // activations[0] = input, activations[i+1] = output of layer i
        let mut activations = Vec::with_capacity(n_layers + 1);
        activations.push(encoded.to_vec());

        for (i, layer) in self.layers.iter().enumerate() {
            let prev = &activations[i];
            let mut output = vec![0.0; layer.out_dim];
            let is_last = i == n_layers - 1;
            layer_forward(layer, prev, &mut output, !is_last);
            activations.push(output);
        }

        let out = activations.last().unwrap()[0];
        (out, activations)
    }

    /// Backward pass: accumulate gradients into grad_w / grad_b.
    fn backward(
        &self,
        activations: &[Vec<f32>],
        d_output: f32,
        grad_w: &mut [Vec<f32>],
        grad_b: &mut [Vec<f32>],
    ) {
        let n_layers = self.layers.len();
        let mut d_act = vec![d_output];

        for i in (0..n_layers).rev() {
            let layer = &self.layers[i];
            let input = &activations[i];
            let output = &activations[i + 1];
            let is_last = i == n_layers - 1;

            // d_pre_act = d_act * relu_derivative (hidden) or d_act (output)
            let d_pre_act: Vec<f32> = if !is_last {
                d_act
                    .iter()
                    .zip(output.iter())
                    .map(|(&da, &o)| if o > 0.0 { da } else { 0.0 })
                    .collect()
            } else {
                d_act.clone()
            };

            // Accumulate weight and bias gradients
            for r in 0..layer.out_dim {
                let row_off = r * layer.in_dim;
                for c in 0..layer.in_dim {
                    grad_w[i][row_off + c] += d_pre_act[r] * input[c];
                }
                grad_b[i][r] += d_pre_act[r];
            }

            // Propagate gradient to previous layer
            if i > 0 {
                d_act = vec![0.0; layer.in_dim];
                #[allow(clippy::needless_range_loop)]
                for r in 0..layer.out_dim {
                    let row_off = r * layer.in_dim;
                    for c in 0..layer.in_dim {
                        d_act[c] += d_pre_act[r] * layer.w[row_off + c];
                    }
                }
            }
        }
    }

    /// Adam optimizer update step.
    fn adam_update(&mut self, grad_w: &[Vec<f32>], grad_b: &[Vec<f32>], lr: f32) {
        self.adam_t += 1;
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let eps: f32 = 1e-8;
        let bc1 = 1.0 - beta1.powi(self.adam_t as i32);
        let bc2 = 1.0 - beta2.powi(self.adam_t as i32);

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let st = &mut self.adam[i];

            #[allow(clippy::needless_range_loop)]
            for j in 0..layer.w.len() {
                let g = grad_w[i][j];
                st.m_w[j] = beta1 * st.m_w[j] + (1.0 - beta1) * g;
                st.v_w[j] = beta2 * st.v_w[j] + (1.0 - beta2) * g * g;
                let m_hat = st.m_w[j] / bc1;
                let v_hat = st.v_w[j] / bc2;
                layer.w[j] -= lr * m_hat / (v_hat.sqrt() + eps);
            }

            #[allow(clippy::needless_range_loop)]
            for j in 0..layer.b.len() {
                let g = grad_b[i][j];
                st.m_b[j] = beta1 * st.m_b[j] + (1.0 - beta1) * g;
                st.v_b[j] = beta2 * st.v_b[j] + (1.0 - beta2) * g * g;
                let m_hat = st.m_b[j] / bc1;
                let v_hat = st.v_b[j] / bc2;
                layer.b[j] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
    }

    /// Train a neural SDF from an SdfNode tree.
    ///
    /// Samples random points in `[min_bounds, max_bounds]`, evaluates the
    /// tree SDF at each point, and trains the network to approximate it.
    pub fn train(
        node: &SdfNode,
        min_bounds: Vec3,
        max_bounds: Vec3,
        config: &NeuralSdfConfig,
    ) -> Self {
        let mut nsdf = Self::new(config);
        let mut rng = Rng::new(config.seed.wrapping_add(12345));

        // Pre-allocate gradient accumulators
        let _n_layers = nsdf.layers.len();
        let mut grad_w: Vec<Vec<f32>> = nsdf.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
        let mut grad_b: Vec<Vec<f32>> = nsdf.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();

        let inv_batch = 1.0 / config.batch_size as f32;

        for _epoch in 0..config.epochs {
            // Zero gradients
            for g in grad_w.iter_mut() {
                g.fill(0.0);
            }
            for g in grad_b.iter_mut() {
                g.fill(0.0);
            }

            let mut _epoch_loss = 0.0;

            for _s in 0..config.batch_size {
                // Sample random point
                let p = Vec3::new(
                    rng.uniform(min_bounds.x, max_bounds.x),
                    rng.uniform(min_bounds.y, max_bounds.y),
                    rng.uniform(min_bounds.z, max_bounds.z),
                );
                let target = eval(node, p);

                // Forward
                let encoded = nsdf.encode(p);
                let (pred, activations) = nsdf.forward_cached(&encoded);

                // Loss = MSE
                let diff = pred - target;
                _epoch_loss += diff * diff;

                // Backward: d_loss/d_pred = 2 * (pred - target) / batch_size
                let d_output = 2.0 * diff * inv_batch;
                nsdf.backward(&activations, d_output, &mut grad_w, &mut grad_b);
            }

            // Adam update
            nsdf.adam_update(&grad_w, &grad_b, config.learning_rate);
        }

        // Drop Adam state to save memory (inference only)
        for st in nsdf.adam.iter_mut() {
            st.m_w.clear();
            st.m_w.shrink_to_fit();
            st.v_w.clear();
            st.v_w.shrink_to_fit();
            st.m_b.clear();
            st.m_b.shrink_to_fit();
            st.v_b.clear();
            st.v_b.shrink_to_fit();
        }

        nsdf
    }

    /// Save weights to a binary writer.
    ///
    /// Format: `b"NSDF"` + version(u32) + pos_freqs(u32) + n_layers(u32)
    /// + for each layer: in_dim(u32) + out_dim(u32) + weights + biases
    pub fn save<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b"NSDF")?;
        w.write_all(&1u32.to_le_bytes())?; // version
        w.write_all(&(self.pos_freqs as u32).to_le_bytes())?;
        w.write_all(&(self.layers.len() as u32).to_le_bytes())?;

        for layer in &self.layers {
            w.write_all(&(layer.in_dim as u32).to_le_bytes())?;
            w.write_all(&(layer.out_dim as u32).to_le_bytes())?;
            for &val in &layer.w {
                w.write_all(&val.to_le_bytes())?;
            }
            for &val in &layer.b {
                w.write_all(&val.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Load weights from a binary reader.
    pub fn load<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"NSDF" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Not an NSDF file",
            ));
        }

        let version = read_u32(r)?;
        if version != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Unsupported version",
            ));
        }

        let pos_freqs = read_u32(r)? as usize;
        let n_layers = read_u32(r)? as usize;
        let input_dim = 3 + 3 * 2 * pos_freqs;

        let mut layers = Vec::with_capacity(n_layers);
        let mut adam = Vec::with_capacity(n_layers);

        for _ in 0..n_layers {
            let in_dim = read_u32(r)? as usize;
            let out_dim = read_u32(r)? as usize;

            let n_w = in_dim * out_dim;
            let mut w = vec![0.0f32; n_w];
            for val in w.iter_mut() {
                *val = read_f32(r)?;
            }
            let mut b = vec![0.0f32; out_dim];
            for val in b.iter_mut() {
                *val = read_f32(r)?;
            }

            layers.push(Layer {
                w,
                b,
                in_dim,
                out_dim,
            });
            adam.push(NeuralSdf::make_adam(in_dim, out_dim));
        }

        Ok(Self {
            layers,
            adam,
            input_dim,
            pos_freqs,
            adam_t: 0,
        })
    }

    /// Number of trainable parameters.
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.w.len() + l.b.len()).sum()
    }
}

// ============================================================
// Utility functions
// ============================================================

#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn layer_forward(layer: &Layer, input: &[f32], output: &mut [f32], relu: bool) {
    for r in 0..layer.out_dim {
        let mut sum = layer.b[r];
        let row = &layer.w[r * layer.in_dim..(r + 1) * layer.in_dim];
        for c in 0..layer.in_dim {
            sum += row[c] * input[c];
        }
        output[r] = if relu { sum.max(0.0) } else { sum };
    }
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> std::io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positional_encoding() {
        let nsdf = NeuralSdf::new(&NeuralSdfConfig {
            pos_encoding_freqs: 2,
            ..Default::default()
        });
        let enc = nsdf.encode(Vec3::ZERO);
        // 3 raw + 2 freqs * 3 dims * 2 (sin/cos) = 3 + 12 = 15
        assert_eq!(enc.len(), 15);
        assert_eq!(enc[0], 0.0); // x
        assert_eq!(enc[1], 0.0); // y
        assert_eq!(enc[2], 0.0); // z
    }

    #[test]
    fn test_forward_shape() {
        let config = NeuralSdfConfig {
            hidden_layers: 2,
            hidden_width: 32,
            pos_encoding_freqs: 4,
            ..Default::default()
        };
        let nsdf = NeuralSdf::new(&config);
        let d = nsdf.eval(Vec3::new(1.0, 2.0, 3.0));
        assert!(d.is_finite(), "Output should be finite, got {}", d);
    }

    #[test]
    fn test_param_count() {
        let config = NeuralSdfConfig {
            hidden_layers: 2,
            hidden_width: 32,
            pos_encoding_freqs: 0,
            ..Default::default()
        };
        let nsdf = NeuralSdf::new(&config);
        // Layer 0: 3*32 + 32 = 128
        // Layer 1: 32*32 + 32 = 1056
        // Output: 32*1 + 1 = 33
        assert_eq!(nsdf.param_count(), 128 + 1056 + 33);
    }

    #[test]
    fn test_train_sphere() {
        let sphere = SdfNode::sphere(1.0);
        // Small network for fast debug-mode test
        let config = NeuralSdfConfig {
            hidden_layers: 2,
            hidden_width: 32,
            pos_encoding_freqs: 0, // raw xyz only for speed
            learning_rate: 5e-3,
            batch_size: 512,
            epochs: 1000,
            seed: 42,
        };

        let nsdf = NeuralSdf::train(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // Check that the neural SDF approximates the sphere reasonably
        let test_points = [
            Vec3::new(0.0, 0.0, 0.0), // inside, d = -1
            Vec3::new(1.0, 0.0, 0.0), // on surface, d = 0
            Vec3::new(2.0, 0.0, 0.0), // outside, d = 1
            Vec3::new(0.5, 0.5, 0.0), // inside
        ];

        let mut max_err: f32 = 0.0;
        for &p in &test_points {
            let expected = eval(&sphere, p);
            let predicted = nsdf.eval(p);
            let err = (expected - predicted).abs();
            max_err = max_err.max(err);
        }

        assert!(
            max_err < 0.3,
            "Max error should be < 0.3, got {:.4}",
            max_err
        );
    }

    #[test]
    fn test_save_load_roundtrip() {
        let config = NeuralSdfConfig {
            hidden_layers: 2,
            hidden_width: 16,
            pos_encoding_freqs: 2,
            ..Default::default()
        };
        let nsdf = NeuralSdf::new(&config);

        // Evaluate before save
        let p = Vec3::new(0.5, -0.3, 0.7);
        let d_before = nsdf.eval(p);

        // Save
        let mut buf = Vec::new();
        nsdf.save(&mut buf).unwrap();

        // Load
        let nsdf2 = NeuralSdf::load(&mut &buf[..]).unwrap();

        // Evaluate after load
        let d_after = nsdf2.eval(p);

        assert!(
            (d_before - d_after).abs() < 1e-6,
            "Before={}, After={}",
            d_before,
            d_after
        );
        assert_eq!(nsdf.param_count(), nsdf2.param_count());
    }

    #[test]
    fn test_gradient_check() {
        let config = NeuralSdfConfig {
            hidden_layers: 1,
            hidden_width: 4,
            pos_encoding_freqs: 0,
            seed: 123,
            ..Default::default()
        };
        let mut nsdf = NeuralSdf::new(&config);

        let p = Vec3::new(0.5, -0.3, 0.7);
        let target = 0.42_f32;
        let encoded = nsdf.encode(p);
        let (pred, activations) = nsdf.forward_cached(&encoded);
        let diff = pred - target;

        // Analytical gradient
        let mut grad_w: Vec<Vec<f32>> = nsdf.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
        let mut grad_b: Vec<Vec<f32>> = nsdf.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();
        nsdf.backward(&activations, 2.0 * diff, &mut grad_w, &mut grad_b);

        // Numerical gradient check on first layer weights
        let eps = 1e-4;
        for j in 0..nsdf.layers[0].w.len() {
            let orig = nsdf.layers[0].w[j];

            nsdf.layers[0].w[j] = orig + eps;
            let loss_p = (nsdf.eval(p) - target).powi(2);
            nsdf.layers[0].w[j] = orig - eps;
            let loss_m = (nsdf.eval(p) - target).powi(2);
            nsdf.layers[0].w[j] = orig;

            let numerical = (loss_p - loss_m) / (2.0 * eps);
            let analytical = grad_w[0][j];
            let scale = numerical.abs().max(analytical.abs()).max(1e-5);
            let rel_err = (numerical - analytical).abs() / scale;

            assert!(rel_err < 0.05,
                "Gradient w[0][{j}]: analytical={analytical:.6}, numerical={numerical:.6}, rel_err={rel_err:.4}");
        }
    }

    #[test]
    fn test_save_load_invalid_magic() {
        let bad_data = b"BADDxxxxxxxx";
        let result = NeuralSdf::load(&mut &bad_data[..]);
        assert!(result.is_err());
    }
}
