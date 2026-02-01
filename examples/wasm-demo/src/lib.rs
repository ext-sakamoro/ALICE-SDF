//! ALICE-SDF WebAssembly Demo
//!
//! This module provides a WebAssembly interface for ALICE-SDF,
//! enabling browser-based SDF visualization and evaluation.
//!
//! # Features
//! - WebGPU compute shader evaluation (modern browsers)
//! - Canvas2D fallback (legacy browsers)
//! - Real-time raymarching visualization
//!
//! # Building
//! ```bash
//! wasm-pack build --target web
//! ```
//!
//! Author: Moroya Sakamoto

use wasm_bindgen::prelude::*;
use glam::Vec3;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format!($($t)*)))
}

/// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
    console_log!("ALICE-SDF WASM initialized");
}

/// SDF Node types for JavaScript interop
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum SdfType {
    Sphere,
    Box,
    Cylinder,
    Torus,
}

/// JavaScript-accessible SDF evaluator
#[wasm_bindgen]
pub struct SdfEvaluator {
    shape_type: SdfType,
    params: [f32; 4],
    transform: [f32; 3],
}

#[wasm_bindgen]
impl SdfEvaluator {
    /// Create a sphere SDF
    #[wasm_bindgen(constructor)]
    pub fn new(shape_type: SdfType) -> SdfEvaluator {
        SdfEvaluator {
            shape_type,
            params: [1.0, 1.0, 1.0, 0.0],
            transform: [0.0, 0.0, 0.0],
        }
    }

    /// Set shape parameters
    pub fn set_params(&mut self, p0: f32, p1: f32, p2: f32, p3: f32) {
        self.params = [p0, p1, p2, p3];
    }

    /// Set translation
    pub fn set_translation(&mut self, x: f32, y: f32, z: f32) {
        self.transform = [x, y, z];
    }

    /// Evaluate SDF at a point
    pub fn eval(&self, x: f32, y: f32, z: f32) -> f32 {
        let p = Vec3::new(x, y, z) - Vec3::from_array(self.transform);

        match self.shape_type {
            SdfType::Sphere => {
                p.length() - self.params[0]
            }
            SdfType::Box => {
                let b = Vec3::new(self.params[0], self.params[1], self.params[2]);
                let q = p.abs() - b;
                q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
            }
            SdfType::Cylinder => {
                let r = self.params[0];
                let h = self.params[1];
                let d = Vec3::new(p.xz().length() - r, p.y.abs() - h, 0.0);
                d.x.max(d.y).min(0.0) + d.xy().max(glam::Vec2::ZERO).length()
            }
            SdfType::Torus => {
                let major = self.params[0];
                let minor = self.params[1];
                let q = glam::Vec2::new(p.xz().length() - major, p.y);
                q.length() - minor
            }
        }
    }

    /// Evaluate batch of points (for performance)
    pub fn eval_batch(&self, points: &[f32]) -> Vec<f32> {
        let count = points.len() / 3;
        let mut results = Vec::with_capacity(count);

        for i in 0..count {
            let x = points[i * 3];
            let y = points[i * 3 + 1];
            let z = points[i * 3 + 2];
            results.push(self.eval(x, y, z));
        }

        results
    }
}

/// Raymarcher for 2D canvas rendering
#[wasm_bindgen]
pub struct Raymarcher {
    width: u32,
    height: u32,
    camera_pos: [f32; 3],
    camera_dir: [f32; 3],
    fov: f32,
}

#[wasm_bindgen]
impl Raymarcher {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Raymarcher {
        Raymarcher {
            width,
            height,
            camera_pos: [0.0, 0.0, -5.0],
            camera_dir: [0.0, 0.0, 1.0],
            fov: 60.0,
        }
    }

    /// Set camera position
    pub fn set_camera(&mut self, x: f32, y: f32, z: f32) {
        self.camera_pos = [x, y, z];
    }

    /// Render to RGBA buffer (for Canvas2D)
    pub fn render(&self, sdf: &SdfEvaluator) -> Vec<u8> {
        let mut buffer = vec![0u8; (self.width * self.height * 4) as usize];
        let fov_tan = (self.fov.to_radians() * 0.5).tan();
        let aspect = self.width as f32 / self.height as f32;

        let cam_pos = Vec3::from_array(self.camera_pos);
        let cam_dir = Vec3::from_array(self.camera_dir).normalize();
        let cam_right = cam_dir.cross(Vec3::Y).normalize();
        let cam_up = cam_right.cross(cam_dir);

        for y in 0..self.height {
            for x in 0..self.width {
                let u = (x as f32 / self.width as f32) * 2.0 - 1.0;
                let v = 1.0 - (y as f32 / self.height as f32) * 2.0;

                let ray_dir = (cam_dir + cam_right * u * fov_tan * aspect + cam_up * v * fov_tan).normalize();

                // Raymarch
                let mut t = 0.0f32;
                let mut hit = false;
                let max_dist = 100.0;
                let epsilon = 0.001;

                for _ in 0..64 {
                    let p = cam_pos + ray_dir * t;
                    let d = sdf.eval(p.x, p.y, p.z);

                    if d < epsilon {
                        hit = true;
                        break;
                    }

                    if t > max_dist {
                        break;
                    }

                    t += d;
                }

                let idx = ((y * self.width + x) * 4) as usize;

                if hit {
                    // Calculate normal
                    let p = cam_pos + ray_dir * t;
                    let eps = 0.001;
                    let n = Vec3::new(
                        sdf.eval(p.x + eps, p.y, p.z) - sdf.eval(p.x - eps, p.y, p.z),
                        sdf.eval(p.x, p.y + eps, p.z) - sdf.eval(p.x, p.y - eps, p.z),
                        sdf.eval(p.x, p.y, p.z + eps) - sdf.eval(p.x, p.y, p.z - eps),
                    ).normalize();

                    // Simple lighting
                    let light_dir = Vec3::new(1.0, 1.0, -1.0).normalize();
                    let diffuse = n.dot(light_dir).max(0.0);
                    let ambient = 0.2;
                    let intensity = ambient + diffuse * 0.8;

                    buffer[idx] = (200.0 * intensity) as u8;
                    buffer[idx + 1] = (180.0 * intensity) as u8;
                    buffer[idx + 2] = (160.0 * intensity) as u8;
                    buffer[idx + 3] = 255;
                } else {
                    // Sky gradient
                    let sky = (v * 0.5 + 0.5) * 0.3 + 0.4;
                    buffer[idx] = (sky * 128.0) as u8;
                    buffer[idx + 1] = (sky * 178.0) as u8;
                    buffer[idx + 2] = (sky * 255.0) as u8;
                    buffer[idx + 3] = 255;
                }
            }
        }

        buffer
    }

    /// Get render dimensions
    pub fn get_width(&self) -> u32 {
        self.width
    }

    pub fn get_height(&self) -> u32 {
        self.height
    }
}

/// Benchmark utility
#[wasm_bindgen]
pub fn benchmark_eval(iterations: u32) -> f64 {
    let sdf = SdfEvaluator::new(SdfType::Sphere);

    let start = web_sys::window()
        .unwrap()
        .performance()
        .unwrap()
        .now();

    let mut sum = 0.0f32;
    for i in 0..iterations {
        let t = i as f32 / iterations as f32;
        sum += sdf.eval(
            (t * 123.456).sin() * 2.0,
            (t * 234.567).sin() * 2.0,
            (t * 345.678).sin() * 2.0,
        );
    }

    let end = web_sys::window()
        .unwrap()
        .performance()
        .unwrap()
        .now();

    // Prevent optimization
    if sum.abs() > 1e10 {
        console_log!("Unexpected sum: {}", sum);
    }

    end - start
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_sdf() {
        let sdf = SdfEvaluator::new(SdfType::Sphere);
        assert!((sdf.eval(0.0, 0.0, 0.0) + 1.0).abs() < 0.001);
        assert!(sdf.eval(1.0, 0.0, 0.0).abs() < 0.001);
        assert!((sdf.eval(2.0, 0.0, 0.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_box_sdf() {
        let mut sdf = SdfEvaluator::new(SdfType::Box);
        sdf.set_params(1.0, 1.0, 1.0, 0.0);
        assert!(sdf.eval(0.0, 0.0, 0.0) < 0.0); // Inside
        assert!(sdf.eval(2.0, 0.0, 0.0) > 0.0); // Outside
    }
}
