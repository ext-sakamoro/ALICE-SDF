//! Stack canary for a browser background thread: the tree walkers on a chain `depth` levels deep
//!
//! Built as a wasm module (`cargo build --release --example stack_canary_wasm --target
//! wasm32-unknown-unknown --no-default-features`) with plain exports, and driven by
//! `scripts/browser_stack_canary.py`: a background thread in headless Chromium calls each export
//! with growing depths and records the deepest chain that ran before the engine stopped it
//! (a `RangeError` for the engine stack, a trap for the module's own 1 MiB stack). Each export
//! returns the value at the origin, the centre of the chain's deepest leaf, so a call that ran is
//! also checked to have produced a finite negative distance.
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::{eval_compiled, CompiledSdf};
use alice_sdf::{eval, SdfNode};
use glam::Vec3;

/// A chain of `depth` smooth unions of spheres along x, folded one at a time (the deepest leaf is
/// the first sphere, at the origin)
fn sphere_chain(depth: u32) -> SdfNode {
    let mut result = SdfNode::sphere(1.0);
    for i in 1..=depth {
        result = result.smooth_union(
            SdfNode::sphere(1.0).translate(i as f32 * 1.5, 0.0, 0.0),
            0.5,
        );
    }
    result
}

/// `eval` at the origin of a chain `depth` deep
#[no_mangle]
pub extern "C" fn canary_eval(depth: u32) -> f32 {
    let tree = sphere_chain(depth);
    // the tree is dropped here: dropping a deep tree recurses too, as it does in the browser
    eval(&tree, Vec3::ZERO)
}

/// `CompiledSdf::compile` (validation, stack depths, Lipschitz bound, bytecode) of a chain `depth`
/// deep, then the bytecode at the origin
#[no_mangle]
pub extern "C" fn canary_compile(depth: u32) -> f32 {
    let tree = sphere_chain(depth);
    let compiled = CompiledSdf::compile(&tree);
    eval_compiled(&compiled, Vec3::ZERO)
}
