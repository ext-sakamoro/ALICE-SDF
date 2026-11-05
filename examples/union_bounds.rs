//! Union operand bounds: a union of many parts evaluates only the parts near the point
//!
//! A row of 16 parts is evaluated on a grid of points along the row. On the CPU
//! (`eval_union_bounded_stats`) each point evaluates the parts whose bounding sphere could hold a
//! value below the nearest one found so far and skips the rest, with the value bit-equal to the
//! plain union. On the GPU the shader that counts the skips (`transpile_union_skip_count`) gives
//! the same measure. The example prints both counts.
//!
//! Run: `cargo run --example union_bounds --features gpu`

use alice_sdf::compiled::{GpuEvaluator, TranspileMode, WgslShader};
use alice_sdf::eval::{eval, eval_union_bounded, eval_union_bounded_stats, UnionBoundStats};
use alice_sdf::prelude::*;

fn main() {
    let parts: Vec<SdfNode> = (0..16)
        .map(|i| {
            let x = i as f32 * 0.5;
            if i % 2 == 0 {
                SdfNode::sphere(0.2).translate(x, 0.0, 0.0)
            } else {
                SdfNode::box3d(0.15, 0.2, 0.15).translate(x, 0.0, 0.0)
            }
        })
        .collect();
    let node = parts
        .iter()
        .cloned()
        .reduce(|a, b| a.union(b))
        .expect("parts");

    let points: Vec<Vec3> = (0..400)
        .map(|i| Vec3::new(i as f32 * 0.02 - 0.5, 0.25, 0.05))
        .collect();

    let mut total = UnionBoundStats::default();
    let mut equal = 0;
    for p in &points {
        let (v, st) = eval_union_bounded_stats(&node, *p);
        total.evaluated += st.evaluated;
        total.skipped += st.skipped;
        if v.to_bits() == eval(&node, *p).to_bits()
            && eval_union_bounded(&node, *p).to_bits() == v.to_bits()
        {
            equal += 1;
        }
    }
    println!(
        "cpu: {} points, evaluated {} skipped {} of {} part evaluations, bit-equal {equal}",
        points.len(),
        total.evaluated,
        total.skipped,
        points.len() * parts.len()
    );

    let counter = WgslShader::transpile_union_skip_count(&node, TranspileMode::Hardcoded);
    match GpuEvaluator::from_shader(&counter) {
        Ok(g) => {
            let skips = g.eval_batch(&points).expect("gpu eval");
            let skipped: f32 = skips.iter().sum();
            println!("gpu: skipped {skipped} part evaluations");
        }
        Err(e) => println!("gpu: no adapter ({e})"),
    }
}
