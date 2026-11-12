//! A shape pressed and pulled by Gaussian bells (`transforms::BellPull`), on the CPU and as the GLSL a
//! page moves every frame: `GlslShader::transpile_with_live_pull` reads the pulls from the uniforms
//! `uPull`, `uPull2` and `uPullSigma`, so the shader is compiled once and the pull is a uniform write.
//!
//! The CPU value is the child at the pulled-back point divided by the slope bound: one pull of
//! 0.15 along x under a bell of width 0.34 carries the surface point under the press by the whole
//! pull, and the value there is 0.
//!
//!     cargo run --example bell_pull_live --features glsl
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::GlslShader;
use alice_sdf::eval::{eval, eval_scaled};
use alice_sdf::prelude::Vec3;
use alice_sdf::transforms::BellPull;
use alice_sdf::SdfNode;

fn main() {
    // a slab 1 wide in x, its +x face at x = 0.5; the press at the middle of that face
    let slab = SdfNode::box3d(1.0, 1.0, 0.04);
    let still = BellPull::new([[0.0; 4]; 2], 0.34).expect("no pull folds nothing");
    let pull = BellPull::new([[0.5, 0.0, 0.15, 0.0], [0.0; 4]], 0.34).expect("one to one");

    // the shader: compiled once, with the pull read from uniforms
    let shader = GlslShader::transpile_with_live_pull(&slab.clone().bell_pull(still));
    for name in [
        "uniform vec4 uPull;",
        "uniform vec4 uPull2;",
        "uniform float uPullSigma;",
        "sdf_eval_scaled(",
    ] {
        assert!(
            shader.source.contains(name),
            "the live shader declares {name}"
        );
    }
    println!(
        "live-pull GLSL: {} bytes, {} helpers",
        shader.source.len(),
        shader.helper_count
    );

    // the CPU law: the face point under the press moves out by the whole pull
    let pulled = slab.bell_pull(pull);
    let carried = Vec3::new(0.5 + 0.15, 0.0, 0.0);
    let d = eval(&pulled, carried);
    let (d2, scale) = eval_scaled(&pulled, carried);
    assert!(
        d.abs() < 1e-5,
        "the carried surface point is on the surface: {d}"
    );
    assert_eq!(d.to_bits(), d2.to_bits(), "eval_scaled gives eval's value");
    println!(
        "value at the carried point {d:+.2e}, local scale {scale:.4}, slope bound {:.4}",
        pull.lambda()
    );
}
