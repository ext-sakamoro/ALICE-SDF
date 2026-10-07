//! Metal kernel from an SDF scene
//!
//! The MSL is not a separate transpiler: the WGSL compute shader is
//! translated by naga, so the law has a single source. `MslShader` carries
//! what a Metal host needs to dispatch the kernel — the renamed entry point,
//! the threadgroup size and the buffer slot for the runtime array lengths.
//! `tests/test_msl_metal_oracle.rs` compiles and runs these kernels on Metal
//! and compares them with the CPU.
//!
//! ```bash
//! cargo run --example msl_emit --features msl              # print to stdout
//! cargo run --example msl_emit --features msl -- out.metal # write a file
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::msl::{MslError, MslShader};
use alice_sdf::compiled::{TranspileMode, WgslShader};
use alice_sdf::prelude::*;

fn main() {
    let shape = SdfNode::sphere(0.8)
        .smooth_union(SdfNode::torus(1.0, 0.2), 0.15)
        .translate(0.0, 0.25, 0.0);

    // Distance kernel straight from the tree.
    let kernel = MslShader::transpile(&shape, TranspileMode::Hardcoded).expect("MSL emit");
    assert_ne!(
        kernel.entry_point, "main",
        "`main` is reserved in MSL and is renamed"
    );
    assert_eq!(kernel.wgsl_entry_point, "main");
    assert_eq!(kernel.workgroup_size, [256, 1, 1]);
    // bindings 0..=2 (points, distances, count) -> sizes buffer in slot 3
    assert_eq!(kernel.sizes_buffer_slot, 3);
    assert!(kernel.source.contains(&kernel.entry_point));
    eprintln!(
        "[msl_emit] kernel `{}`, threads per threadgroup {:?}, sizes buffer at slot {}, {} bytes of MSL",
        kernel.entry_point,
        kernel.workgroup_size,
        kernel.sizes_buffer_slot,
        kernel.source.len()
    );

    // Any WGSL compute shader: the distance + normal variant, Dynamic mode
    // (parameters in a uniform buffer at binding 3).
    let normals =
        WgslShader::transpile(&shape, TranspileMode::Dynamic).to_compute_shader_with_normals();
    let with_normals = MslShader::from_wgsl(&normals).expect("MSL emit (normals)");
    assert_eq!(with_normals.sizes_buffer_slot, 4);
    eprintln!(
        "[msl_emit] normals kernel `{}`, {} bytes of MSL",
        with_normals.entry_point,
        with_normals.source.len()
    );

    // Input that is not valid WGSL is reported, not translated.
    match MslShader::from_wgsl("fn broken( {") {
        Err(MslError::WgslParse(msg)) => eprintln!(
            "[msl_emit] rejected invalid WGSL: {}",
            msg.lines().next().unwrap_or("")
        ),
        other => panic!("expected a WGSL parse error, got {other:?}"),
    }

    match std::env::args().nth(1) {
        Some(path) => {
            std::fs::write(&path, &kernel.source).expect("write MSL");
            eprintln!("[msl_emit] wrote {path}");
        }
        None => print!("{}", kernel.source),
    }
}
