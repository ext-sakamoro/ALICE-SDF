//! Emit an SDF scene as dependency-free Rust source.
//!
//! This is what a consuming crate's `build.rs` does: the same `SdfNode` that
//! feeds the shader transpilers becomes `fn sdf(x, y, z) -> f32` and
//! `fn normal(x, y, z) -> (f32, f32, f32)`, which the runtime crate
//! `include!`s and evaluates on the CPU with only `alice-det-math` as a
//! dependency (bit-identical to `eval_compiled`).
//!
//! ```text
//! cargo run --example rust_transpile --features rust            # print to stdout
//! cargo run --example rust_transpile --features rust -- out.rs  # write a file
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::rust::{is_supported, RustSource, RustTranspileOptions};
use alice_sdf::compiled::{eval_compiled, CompiledSdf};
use alice_sdf::prelude::*;

fn main() {
    let shape = SdfNode::sphere(0.6)
        .smooth_union(
            SdfNode::box3d(0.5, 0.4, 0.3)
                .rotate_euler(0.3, -0.7, 1.1)
                .translate(0.5, 0.1, -0.2),
            0.3,
        )
        .twist(0.8);

    let compiled = CompiledSdf::compile(&shape);
    if let Some(i) = compiled
        .instructions()
        .iter()
        .find(|i| !is_supported(i.opcode))
    {
        eprintln!("scene uses {:?}, which has no Rust emitter", i.opcode);
        std::process::exit(1);
    }

    // `det_math_path` is how the consuming crate names its alice-det-math
    // dependency (here the default name, spelled out).
    let options = RustTranspileOptions::default()
        .with_normal_epsilon(1e-3)
        .with_det_math_path("::alice_det_math");
    let src = match RustSource::transpile_compiled(&compiled, &options) {
        Ok(src) => src,
        Err(e) => {
            eprintln!("transpile failed: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "[rust_transpile] {} instructions -> {} bytes of Rust; eval_compiled(0.5, 0.2, 0.1) = {}",
        src.instruction_count(),
        src.source().len(),
        eval_compiled(&compiled, Vec3::new(0.5, 0.2, 0.1))
    );

    match std::env::args().nth(1) {
        Some(path) => {
            if let Err(e) = std::fs::write(&path, src.source()) {
                eprintln!("write {path}: {e}");
                std::process::exit(1);
            }
            eprintln!("[rust_transpile] wrote {path}");
        }
        None => print!("{}", src.source()),
    }
}
