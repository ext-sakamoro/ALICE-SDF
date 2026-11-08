//! A grid warp whose maps change every frame, for WebGL 2: one GLSL ES shader, a new texture per
//! frame.
//!
//! ```text
//! cargo run --example grid_warp_texture --features glsl -- [frames]
//! ```
//!
//! Transpiles a box under a twisting grid warp once with
//! `GlslShader::transpile_with_node_texture`, then for each frame builds the warp with another twist
//! (the same grid), takes `GlslNodeTexture::frame` (the texels to upload to the R32F texture
//! `alice_node_tex` and the Lipschitz bound to march with). Prints the shader's size, the texture's
//! size and one line per frame with the frame's CPU distance at one point (what the shader returns
//! there with the frame's texels; `tests/test_grid_warp_glsl_texture.rs` runs that comparison on a
//! GPU).
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::glsl::{GlslNodeTexture, GlslShader};
use alice_sdf::prelude::*;
use alice_sdf::transforms::{GridWarp, GridWarpData};

fn warp(twist: f32) -> Result<GridWarp, Box<dyn std::error::Error>> {
    let (dims, cell, origin) = ([8u32, 8, 8], 0.4_f32, [-1.4_f32; 3]);
    let mut maps = Vec::new();
    for k in 0..dims[2] {
        for _ in 0..dims[1] * dims[0] {
            let z = origin[2] + k as f32 * cell;
            let (s, c) = alice_det_math::sin_cos(twist * z);
            maps.extend_from_slice(&[c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0]);
        }
    }
    Ok(GridWarp::new(GridWarpData {
        origin,
        cell,
        dims,
        maps,
        reach: 0.8,
        floor: Vec::new(),
    })?)
}

fn tree(twist: f32) -> Result<SdfNode, Box<dyn std::error::Error>> {
    Ok(SdfNode::box3d(0.8, 0.5, 1.0).grid_warp(warp(twist)?))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frames: u32 = std::env::args().nth(1).map_or(Ok(4), |s| s.parse())?;
    let tex: GlslNodeTexture = GlslShader::transpile_with_node_texture(&tree(0.0)?, 300);
    let es = tex.es_source();
    let (w, h) = tex.texture_size();
    println!(
        "[grid_warp_texture] shader {} bytes, {} node floats, texture {w}x{h}",
        es.len(),
        tex.node_data.len()
    );
    for f in 0..frames {
        let twist = 0.4 * f as f32;
        let t = tree(twist)?;
        let frame = tex.frame(&t)?;
        let p = Vec3::new(0.3, 0.2, 0.9);
        println!(
            "[grid_warp_texture] frame {f}: twist {twist:.2}, {} texels, lipschitz {:.4}, d(p) {:.5}",
            frame.texels.len(),
            frame.lipschitz,
            eval(&t, p)
        );
    }
    Ok(())
}
