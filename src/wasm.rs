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
    "1.6.0".to_string()
}

// === WebXR (VR/AR) helpers ===
//
// JavaScript 側で WebXR Device API のフレームコールバック (XRSession.requestAnimationFrame)
// から 1 frame ごとに以下を呼び出して、コントローラ位置 / ハンド位置の SDF クエリを行う。

/// 球の表面にぶつかる最短距離を ray march (WebXR コントローラ / hand pose 用)
///
/// 最大 64 step、エプシロン 0.001。ヒットしなければ -1.0 を返す。
#[wasm_bindgen]
pub fn raymarch_sphere(
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
    max_dist: f32,
) -> f32 {
    let origin = Vec3::new(ox, oy, oz);
    let dir = Vec3::new(dx, dy, dz).normalize_or_zero();
    if dir.length_squared() < 1e-12 {
        return -1.0;
    }
    let center = Vec3::new(cx, cy, cz);
    let mut t = 0.0_f32;
    for _ in 0..64 {
        let p = origin + dir * t;
        let d = sdf_sphere_at(p, center, radius);
        if d.abs() < 0.001 {
            return t;
        }
        t += d.max(0.001);
        if t > max_dist {
            return -1.0;
        }
    }
    -1.0
}

/// 2 球 smooth-union 形状への最短距離 ray march (簡易シーン例、VR デモ用)
#[wasm_bindgen]
pub fn raymarch_two_spheres_smooth(
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    c1x: f32,
    c1y: f32,
    c1z: f32,
    r1: f32,
    c2x: f32,
    c2y: f32,
    c2z: f32,
    r2: f32,
    k: f32,
    max_dist: f32,
) -> f32 {
    let origin = Vec3::new(ox, oy, oz);
    let dir = Vec3::new(dx, dy, dz).normalize_or_zero();
    if dir.length_squared() < 1e-12 {
        return -1.0;
    }
    let c1 = Vec3::new(c1x, c1y, c1z);
    let c2 = Vec3::new(c2x, c2y, c2z);
    let mut t = 0.0_f32;
    for _ in 0..96 {
        let p = origin + dir * t;
        let d1 = sdf_sphere_at(p, c1, r1);
        let d2 = sdf_sphere_at(p, c2, r2);
        let d = crate::operations::sdf_smooth_union(d1, d2, k);
        if d.abs() < 0.001 {
            return t;
        }
        t += d.max(0.001);
        if t > max_dist {
            return -1.0;
        }
    }
    -1.0
}

/// バッチ評価 (WebXR ハンドメッシュ全頂点に対する SDF クエリ等)
///
/// `points_xyz` は [x0,y0,z0, x1,y1,z1, ...] のフラット配列、長さ = N*3。
/// 戻り値は長さ N の距離配列。
#[wasm_bindgen]
pub fn sphere_batch_flat(points_xyz: &[f32], cx: f32, cy: f32, cz: f32, radius: f32) -> Vec<f32> {
    let center = Vec3::new(cx, cy, cz);
    let n = points_xyz.len() / 3;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let p = Vec3::new(
            points_xyz[i * 3],
            points_xyz[i * 3 + 1],
            points_xyz[i * 3 + 2],
        );
        out.push(sdf_sphere_at(p, center, radius));
    }
    out
}
