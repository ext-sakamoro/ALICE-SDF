//! WebAssembly bindings for ALICE-SDF
//!
//! Minimal browser-facing API for SDF evaluation + 2D slice rendering.
//! Enabled via `--features wasm` + `cargo build --target wasm32-unknown-unknown`.
//!
//! # JavaScript usage
//!
//! ```js
//! import init, { sdf_sphere, sdf_box, op_smooth_union, render_slice_2d } from './alice_sdf.js';
//! await init();
//! const d = sdf_sphere(1.0, 0.0, 0.0, /* center */ 0, 0, 0, /* radius */ 1.0);
//! ```

use crate::primitives::{sdf_box3d_at, sdf_cylinder, sdf_plane, sdf_sphere_at, sdf_torus};
use glam::Vec3;
use wasm_bindgen::prelude::*;

// === Primitives ===

#[wasm_bindgen]
pub fn sdf_sphere(px: f32, py: f32, pz: f32, cx: f32, cy: f32, cz: f32, radius: f32) -> f32 {
    sdf_sphere_at(Vec3::new(px, py, pz), Vec3::new(cx, cy, cz), radius)
}

#[wasm_bindgen]
pub fn sdf_box(
    px: f32,
    py: f32,
    pz: f32,
    cx: f32,
    cy: f32,
    cz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
) -> f32 {
    sdf_box3d_at(
        Vec3::new(px, py, pz),
        Vec3::new(cx, cy, cz),
        Vec3::new(hx, hy, hz),
    )
}

#[wasm_bindgen]
pub fn sdf_torus_w(px: f32, py: f32, pz: f32, major_radius: f32, minor_radius: f32) -> f32 {
    sdf_torus(Vec3::new(px, py, pz), major_radius, minor_radius)
}

#[wasm_bindgen]
pub fn sdf_cylinder_w(px: f32, py: f32, pz: f32, radius: f32, half_height: f32) -> f32 {
    sdf_cylinder(Vec3::new(px, py, pz), radius, half_height)
}

#[wasm_bindgen]
pub fn sdf_plane_w(px: f32, py: f32, pz: f32, nx: f32, ny: f32, nz: f32, distance: f32) -> f32 {
    sdf_plane(Vec3::new(px, py, pz), Vec3::new(nx, ny, nz), distance)
}

// === Operations ===

#[wasm_bindgen]
pub fn op_union(a: f32, b: f32) -> f32 {
    crate::operations::sdf_union(a, b)
}

#[wasm_bindgen]
pub fn op_intersection(a: f32, b: f32) -> f32 {
    crate::operations::sdf_intersection(a, b)
}

#[wasm_bindgen]
pub fn op_subtraction(a: f32, b: f32) -> f32 {
    crate::operations::sdf_subtraction(a, b)
}

#[wasm_bindgen]
pub fn op_smooth_union(a: f32, b: f32, k: f32) -> f32 {
    crate::operations::sdf_smooth_union(a, b, k)
}

#[wasm_bindgen]
pub fn op_smooth_intersection(a: f32, b: f32, k: f32) -> f32 {
    crate::operations::sdf_smooth_intersection(a, b, k)
}

#[wasm_bindgen]
pub fn op_smooth_subtraction(a: f32, b: f32, k: f32) -> f32 {
    crate::operations::sdf_smooth_subtraction(a, b, k)
}

// === 2D slice rendering ===

/// 2D スライス (z=0 平面) を 1 球の SDF で uint8 RGBA バッファに描画
///
/// JavaScript canvas に putImageData で描画可能
#[wasm_bindgen]
pub fn render_sphere_slice_rgba(
    width: u32,
    height: u32,
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
    half_range: f32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut buf = vec![0u8; w * h * 4];
    let center = Vec3::new(cx, cy, cz);
    for y in 0..h {
        for x in 0..w {
            let px = (x as f32 / w as f32 * 2.0 - 1.0) * half_range;
            let py = (1.0 - y as f32 / h as f32 * 2.0) * half_range;
            let p = Vec3::new(px, py, 0.0);
            let d = sdf_sphere_at(p, center, radius);
            let idx = (y * w + x) * 4;
            if d < 0.0 {
                let t = (-d).min(1.0);
                buf[idx] = (51.0 + 100.0 * t) as u8;
                buf[idx + 1] = (102.0 + 130.0 * t) as u8;
                buf[idx + 2] = 230;
            } else {
                let t = (d * 0.5).min(1.0);
                let g = (12.0 + 100.0 * t) as u8;
                buf[idx] = g;
                buf[idx + 1] = g;
                buf[idx + 2] = (26.0 + 100.0 * t) as u8;
            }
            buf[idx + 3] = 255;
        }
    }
    buf
}

#[wasm_bindgen]
pub fn alice_sdf_version() -> String {
    "1.5.0".to_string()
}
