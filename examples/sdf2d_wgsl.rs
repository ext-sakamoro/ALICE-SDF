//! 2D SDF as WGSL
//!
//! A flat shape (`Sdf2dNode`) becomes WGSL: `Wgsl2dShader::source` is the fragment
//! `fn sdf2d_eval(p: vec2<f32>) -> f32` to paste into a host shader, and `to_compute_shader` the
//! whole compute shader with the bindings of the 3D one, which `GpuEvaluator` runs. With a GPU
//! the example compares a few distances with the CPU law `eval_2d` (they agree within a
//! tolerance, not bit for bit: the GPU's `length` / `atan2` / `sin` / `cos` are its own).
//!
//! ```bash
//! cargo run --example sdf2d_wgsl --features gpu            # print the compute shader
//! cargo run --example sdf2d_wgsl --features gpu -- compare # also run it on the GPU
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::wgsl::{Wgsl2dError, Wgsl2dShader};
use alice_sdf::compiled::GpuEvaluator;
use alice_sdf::prelude::Vec3;
use alice_sdf::sdf2d::{eval_2d, Sdf2dNode};

fn main() {
    // a ring with a five-pointed star in it, the star turned a little
    let shape = Sdf2dNode::ring(0.8, 0.05).union(Sdf2dNode::star(0.6, 0.25, 5).rotate(0.3));
    let shader = Wgsl2dShader::transpile(&shape).expect("finite parameters");
    eprintln!(
        "[sdf2d_wgsl] fragment {} bytes, compute shader {} bytes",
        shader.source.len(),
        shader.to_compute_shader().len()
    );

    // a parameter that is not a finite number is refused, not emitted
    match Wgsl2dShader::transpile(&Sdf2dNode::circle(f32::NAN)) {
        Err(Wgsl2dError::NonFinite(v)) => eprintln!("[sdf2d_wgsl] refused a radius of {v}"),
        other => panic!("expected NonFinite, got {other:?}"),
    }

    if std::env::args().nth(1).as_deref() == Some("compare") {
        let points: Vec<Vec3> = (0..16)
            .map(|i| {
                // points on a circle of radius 0.45, between the star's points and the ring
                let (s, c) = alice_det_math::sin_cos(i as f32 / 16.0 * std::f32::consts::TAU);
                Vec3::new(0.45 * c, 0.45 * s, 0.0)
            })
            .collect();
        match GpuEvaluator::from_wgsl_pinned(&shader.to_compute_shader()) {
            Ok(gpu) => {
                let got = gpu.eval_batch(&points).expect("gpu eval");
                for (p, g) in points.iter().zip(&got) {
                    let c = eval_2d(&shape, [p.x, p.y]);
                    println!("({:+.3}, {:+.3})  cpu {c:+.6}  gpu {g:+.6}", p.x, p.y);
                }
            }
            Err(e) => eprintln!("[sdf2d_wgsl] no GPU here: {e}"),
        }
    } else {
        print!("{}", shader.to_compute_shader());
    }
}
