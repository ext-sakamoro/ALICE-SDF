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
use glam::{Vec3, Vec3Swizzles};

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

// ============================================================================
// SoA Zero-Copy Pipeline (Project PLASMA)
// ============================================================================

/// SoA (Structure of Arrays) buffer for zero-copy JS → WASM → WebGPU pipeline
///
/// Stores 3D positions in SoA layout (separate X, Y, Z arrays) for
/// direct GPU buffer upload without memory copies or layout conversions.
///
/// ## Usage (JavaScript)
/// ```js
/// const soa = new SoABuffer(10000);
/// // Set positions directly
/// for (let i = 0; i < 10000; i++) {
///     soa.set_point(i, x[i], y[i], z[i]);
/// }
/// // Get typed array views (zero-copy)
/// const xBuf = soa.x_buffer();  // Float32Array view
/// const yBuf = soa.y_buffer();
/// const zBuf = soa.z_buffer();
/// // Upload directly to WebGPU
/// device.queue.writeBuffer(gpuBuffer, 0, xBuf);
/// ```
#[wasm_bindgen]
pub struct SoABuffer {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    distances: Vec<f32>,
    count: usize,
}

#[wasm_bindgen]
impl SoABuffer {
    /// Create a new SoA buffer with the given capacity
    #[wasm_bindgen(constructor)]
    pub fn new(count: u32) -> SoABuffer {
        let count = count as usize;
        SoABuffer {
            x: vec![0.0; count],
            y: vec![0.0; count],
            z: vec![0.0; count],
            distances: vec![0.0; count],
            count,
        }
    }

    /// Set a single point's coordinates
    pub fn set_point(&mut self, index: u32, x: f32, y: f32, z: f32) {
        let i = index as usize;
        if i < self.count {
            self.x[i] = x;
            self.y[i] = y;
            self.z[i] = z;
        }
    }

    /// Get the number of points
    pub fn len(&self) -> u32 {
        self.count as u32
    }

    /// Get a pointer to the X array (for JS Float32Array view)
    pub fn x_ptr(&self) -> *const f32 {
        self.x.as_ptr()
    }

    /// Get a pointer to the Y array
    pub fn y_ptr(&self) -> *const f32 {
        self.y.as_ptr()
    }

    /// Get a pointer to the Z array
    pub fn z_ptr(&self) -> *const f32 {
        self.z.as_ptr()
    }

    /// Get a pointer to the distances (output) array
    pub fn distances_ptr(&self) -> *const f32 {
        self.distances.as_ptr()
    }

    /// Get distances as mutable pointer (for writing results)
    pub fn distances_mut_ptr(&mut self) -> *mut f32 {
        self.distances.as_mut_ptr()
    }

    /// Get the byte length of each array (for WebGPU buffer sizing)
    pub fn byte_length(&self) -> u32 {
        (self.count * std::mem::size_of::<f32>()) as u32
    }

    /// Evaluate all points against a sphere SDF (CPU fallback)
    ///
    /// For WebGPU path, use the WGSL shader instead.
    pub fn eval_sphere(&mut self, radius: f32) {
        for i in 0..self.count {
            let len = (self.x[i] * self.x[i] + self.y[i] * self.y[i] + self.z[i] * self.z[i]).sqrt();
            self.distances[i] = len - radius;
        }
    }

    /// Evaluate all points against a box SDF (CPU fallback)
    pub fn eval_box(&mut self, hx: f32, hy: f32, hz: f32) {
        for i in 0..self.count {
            let qx = self.x[i].abs() - hx;
            let qy = self.y[i].abs() - hy;
            let qz = self.z[i].abs() - hz;

            let outside = (qx.max(0.0) * qx.max(0.0)
                + qy.max(0.0) * qy.max(0.0)
                + qz.max(0.0) * qz.max(0.0))
            .sqrt();
            let inside = qx.max(qy.max(qz)).min(0.0);

            self.distances[i] = outside + inside;
        }
    }

    /// Get the distance at a specific index
    pub fn get_distance(&self, index: u32) -> f32 {
        let i = index as usize;
        if i < self.count {
            self.distances[i]
        } else {
            f32::MAX
        }
    }

    /// Set all points from interleaved AoS data [x0,y0,z0,x1,y1,z1,...]
    pub fn set_from_aos(&mut self, data: &[f32]) {
        let point_count = (data.len() / 3).min(self.count);
        for i in 0..point_count {
            self.x[i] = data[i * 3];
            self.y[i] = data[i * 3 + 1];
            self.z[i] = data[i * 3 + 2];
        }
    }

    /// Apply a translation to all points (for instanced animation)
    pub fn translate_all(&mut self, tx: f32, ty: f32, tz: f32) {
        for i in 0..self.count {
            self.x[i] += tx;
            self.y[i] += ty;
            self.z[i] += tz;
        }
    }
}

/// Generate instanced SDF compute shader for WebGPU
///
/// Returns WGSL source code for a compute shader that evaluates
/// one SDF shape against N instance transforms in SoA layout.
#[wasm_bindgen]
pub fn generate_instanced_wgsl(shape_type: SdfType) -> String {
    let sdf_body = match shape_type {
        SdfType::Sphere => {
            "fn sdf_eval(p: vec3<f32>) -> f32 { return length(p) - uniforms.params.x; }"
        }
        SdfType::Box => {
            r#"fn sdf_eval(p: vec3<f32>) -> f32 {
    let b = uniforms.params.xyz;
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}"#
        }
        SdfType::Cylinder => {
            r#"fn sdf_eval(p: vec3<f32>) -> f32 {
    let r = uniforms.params.x;
    let h = uniforms.params.y;
    let d = vec2<f32>(length(p.xz) - r, abs(p.y) - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}"#
        }
        SdfType::Torus => {
            r#"fn sdf_eval(p: vec3<f32>) -> f32 {
    let major = uniforms.params.x;
    let minor = uniforms.params.y;
    let q = vec2<f32>(length(p.xz) - major, p.y);
    return length(q) - minor;
}"#
        }
    };

    format!(
        r#"// ALICE-SDF Instanced Compute Shader (WASM/WebGPU Zero-Copy Pipeline)
// Generated by Project PLASMA

struct Uniforms {{
    params: vec4<f32>,       // Shape params (radius, hx, hy, hz)
    instance_count: u32,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
}}

struct InstanceTransform {{
    translate: vec3<f32>,
    scale: f32,
}}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> instances: array<InstanceTransform>;
@group(0) @binding(2) var<storage, read> points_x: array<f32>;
@group(0) @binding(3) var<storage, read> points_y: array<f32>;
@group(0) @binding(4) var<storage, read> points_z: array<f32>;
@group(0) @binding(5) var<storage, read_write> distances: array<f32>;

{sdf_fn}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= uniforms.point_count) {{
        return;
    }}

    let p = vec3<f32>(points_x[idx], points_y[idx], points_z[idx]);
    var min_dist: f32 = 1e20;

    for (var i: u32 = 0u; i < uniforms.instance_count; i = i + 1u) {{
        let inst = instances[i];
        var q = (p - inst.translate);
        if (abs(inst.scale - 1.0) > 1e-6) {{
            q = q / inst.scale;
        }}
        var d = sdf_eval(q);
        if (abs(inst.scale - 1.0) > 1e-6) {{
            d = d * inst.scale;
        }}
        min_dist = min(min_dist, d);
    }}

    distances[idx] = min_dist;
}}
"#,
        sdf_fn = sdf_body,
    )
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

    #[test]
    fn test_soa_buffer_create() {
        let soa = SoABuffer::new(100);
        assert_eq!(soa.len(), 100);
        assert_eq!(soa.byte_length(), 400); // 100 * 4 bytes
    }

    #[test]
    fn test_soa_buffer_set_point() {
        let mut soa = SoABuffer::new(10);
        soa.set_point(0, 1.0, 2.0, 3.0);
        soa.set_point(1, 4.0, 5.0, 6.0);

        assert_eq!(soa.x[0], 1.0);
        assert_eq!(soa.y[0], 2.0);
        assert_eq!(soa.z[0], 3.0);
        assert_eq!(soa.x[1], 4.0);
    }

    #[test]
    fn test_soa_buffer_eval_sphere() {
        let mut soa = SoABuffer::new(3);
        soa.set_point(0, 0.0, 0.0, 0.0); // origin
        soa.set_point(1, 1.0, 0.0, 0.0); // on surface
        soa.set_point(2, 2.0, 0.0, 0.0); // outside

        soa.eval_sphere(1.0);

        assert!((soa.get_distance(0) + 1.0).abs() < 0.001);
        assert!(soa.get_distance(1).abs() < 0.001);
        assert!((soa.get_distance(2) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_soa_buffer_from_aos() {
        let mut soa = SoABuffer::new(3);
        let aos = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        soa.set_from_aos(&aos);

        assert_eq!(soa.x[0], 1.0);
        assert_eq!(soa.y[0], 2.0);
        assert_eq!(soa.z[0], 3.0);
        assert_eq!(soa.x[2], 7.0);
    }

    #[test]
    fn test_soa_buffer_translate() {
        let mut soa = SoABuffer::new(2);
        soa.set_point(0, 1.0, 2.0, 3.0);
        soa.set_point(1, 4.0, 5.0, 6.0);
        soa.translate_all(10.0, 20.0, 30.0);

        assert_eq!(soa.x[0], 11.0);
        assert_eq!(soa.y[0], 22.0);
        assert_eq!(soa.z[0], 33.0);
    }

    #[test]
    fn test_soa_zero_copy_pointers() {
        let soa = SoABuffer::new(10);
        // Verify pointers are non-null and point to valid memory
        assert!(!soa.x_ptr().is_null());
        assert!(!soa.y_ptr().is_null());
        assert!(!soa.z_ptr().is_null());
        assert!(!soa.distances_ptr().is_null());
    }

    #[test]
    fn test_generate_instanced_wgsl() {
        let wgsl = generate_instanced_wgsl(SdfType::Sphere);
        assert!(wgsl.contains("sdf_eval"));
        assert!(wgsl.contains("@compute"));
        assert!(wgsl.contains("instances"));
        assert!(wgsl.contains("points_x"));
        assert!(wgsl.contains("distances"));
    }
}
