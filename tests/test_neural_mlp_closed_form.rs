//! Neural SDF evaluator against hand-built networks with a closed form.
//!
//! The `NSDF` file format is documented on `NeuralSdf::save`, so a network
//! with chosen weights can be written byte by byte and loaded. Two such
//! networks have exact answers:
//! - no hidden layer, no encoding: `f(p) = n·p − c`, a plane (gradient `n`)
//! - one ReLU layer with rows `+x` and `−x`, output `[1, 1]`, bias `−h`:
//!   `f(p) = relu(x) + relu(−x) − h = |x| − h`, a slab of half-width `h`
//!
//! Also checked: parameter count `Σ (in·out + out)`, input dimension
//! `3 + 6·freqs`, hidden layer count, the saved byte length
//! `16 + Σ (8 + 4·(in·out + out))`, a lossless save / load, and that
//! `eval_batch` returns exactly what `eval` returns.
//!
//! Author: Moroya Sakamoto

use alice_sdf::neural::{NeuralSdf, NeuralSdfConfig};
use glam::Vec3;

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

/// `layers`: (in, out, weights row-major, biases)
fn nsdf(freqs: u32, layers: &[(u32, u32, Vec<f32>, Vec<f32>)]) -> NeuralSdf {
    let mut b = b"NSDF".to_vec();
    put_u32(&mut b, 1);
    put_u32(&mut b, freqs);
    put_u32(&mut b, layers.len() as u32);
    for (i, o, w, bias) in layers {
        put_u32(&mut b, *i);
        put_u32(&mut b, *o);
        for v in w.iter().chain(bias) {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    NeuralSdf::load(&mut b.as_slice()).expect("well-formed NSDF")
}

fn points() -> Vec<Vec3> {
    (0..50)
        .map(|i| {
            let t = i as f32 * 0.61;
            Vec3::new(t.sin() * 2.0, (t * 1.7).cos() * 1.5, (t * 0.3).sin() - 0.2)
        })
        .collect()
}

#[test]
fn single_linear_layer_is_a_plane() {
    let n = Vec3::new(0.6, 0.0, 0.8);
    let c = 0.5;
    let net = nsdf(0, &[(3, 1, vec![n.x, n.y, n.z], vec![-c])]);
    assert_eq!(net.input_dimension(), 3);
    assert_eq!(net.hidden_layer_count(), 0);
    assert_eq!(net.param_count(), 4);
    let mut compared = 0;
    for p in points() {
        let expected = n.dot(p) - c;
        assert!((net.eval(p) - expected).abs() < 1e-6, "p={p}");
        let (d, g) = net.eval_with_gradient(p, 1e-2);
        assert!((d - expected).abs() < 1e-6);
        assert!((g - n).length() < 1e-4, "gradient {g} vs {n}");
        compared += 1;
    }
    assert!(compared > 0);
}

#[test]
fn relu_pair_is_a_slab() {
    let h = 0.75;
    let net = nsdf(
        0,
        &[
            (3, 2, vec![1.0, 0.0, 0.0, -1.0, 0.0, 0.0], vec![0.0, 0.0]),
            (2, 1, vec![1.0, 1.0], vec![-h]),
        ],
    );
    assert_eq!(net.hidden_layer_count(), 1);
    assert_eq!(net.param_count(), 3 * 2 + 2 + 2 + 1);
    let pts = points();
    let batch = net.eval_batch(&pts);
    let mut compared = 0;
    for (p, b) in pts.iter().zip(&batch) {
        let expected = p.x.abs() - h;
        assert!((net.eval(*p) - expected).abs() < 1e-6, "p={p}");
        assert_eq!(b.to_bits(), net.eval(*p).to_bits());
        compared += 1;
    }
    assert!(compared > 0);
}

#[test]
fn architecture_counts_and_save_length() {
    let mut compared = 0;
    for &(layers, width, freqs) in &[(1_usize, 8_usize, 0_usize), (2, 16, 2), (3, 70, 4)] {
        let cfg = NeuralSdfConfig {
            hidden_layers: layers,
            hidden_width: width,
            pos_encoding_freqs: freqs,
            ..Default::default()
        };
        let net = NeuralSdf::new(&cfg);
        let input = 3 + 6 * freqs;
        let mut dims = vec![input];
        dims.extend(std::iter::repeat_n(width, layers));
        dims.push(1);
        let params: usize = dims.windows(2).map(|w| w[0] * w[1] + w[1]).sum();
        assert_eq!(net.input_dimension(), input);
        assert_eq!(net.hidden_layer_count(), layers);
        assert_eq!(net.param_count(), params);

        let mut bytes = Vec::new();
        net.save(&mut bytes).unwrap();
        assert_eq!(bytes.len(), 16 + 8 * (layers + 1) + 4 * params);
        let back = NeuralSdf::load(&mut bytes.as_slice()).unwrap();
        let pts = points();
        // hidden_width 70 > 64 also exercises the batch buffers' growth
        let batch = back.eval_batch(&pts);
        for (p, b) in pts.iter().zip(&batch) {
            assert_eq!(back.eval(*p).to_bits(), net.eval(*p).to_bits());
            assert_eq!(b.to_bits(), net.eval(*p).to_bits());
            compared += 1;
        }
    }
    assert!(compared > 0);
}

#[test]
fn malformed_files_are_rejected() {
    assert!(NeuralSdf::load(&mut &b"NOPE\x01\0\0\0"[..]).is_err());
    let mut b = b"NSDF".to_vec();
    put_u32(&mut b, 2);
    assert!(NeuralSdf::load(&mut b.as_slice()).is_err());
    // truncated weights
    let mut b = b"NSDF".to_vec();
    for v in [1, 0, 1, 3, 1] {
        put_u32(&mut b, v);
    }
    b.extend_from_slice(&1.0f32.to_le_bytes());
    assert!(NeuralSdf::load(&mut b.as_slice()).is_err());
}
