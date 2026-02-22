//! Instanced SDF Rendering
//!
//! Evaluates one SDF shape at N instance positions simultaneously.
//! Instead of compiling N separate SDF trees (one per particle),
//! compile once and evaluate with per-instance transforms.
//!
//! # Use Case
//!
//! Text particle systems: 10,000 identical SDF glyphs at different
//! positions, rotations, and scales. Instancing reduces memory from
//! N × shape_size to 1 × shape_size + N × 36 bytes (AnimationParams).
//!
//! # Performance
//!
//! | Approach | Memory | Compile Time | Eval Speed |
//! |----------|--------|-------------|------------|
//! | N separate SDFs | O(N × S) | O(N × 0.1ms) | 1x |
//! | Instanced (scalar) | O(S + N×36B) | O(0.1ms) | ~1x per instance |
//! | Instanced (SIMD/SoA) | O(S + N×36B) | O(0.1ms) | ~8x throughput |
//!
//! Author: Moroya Sakamoto

use crate::animation::AnimationParams;
use crate::compiled::eval_simd::eval_compiled_simd;
use crate::compiled::simd::{Quatx8, Vec3x8};
use crate::compiled::CompiledSdf;
use wide::f32x8;

/// SIMD-ready instance data in Structure-of-Arrays layout
///
/// Each `Vec<f32x8>` element holds data for 8 instances, packed for
/// zero-cost SIMD register loads. Padding lanes in the last chunk
/// use identity transforms that produce `f32::MAX` distances.
#[derive(Debug, Clone)]
struct InstanceDataSoA {
    translate_x: Vec<f32x8>,
    translate_y: Vec<f32x8>,
    translate_z: Vec<f32x8>,
    inv_quat_x: Vec<f32x8>,
    inv_quat_y: Vec<f32x8>,
    inv_quat_z: Vec<f32x8>,
    inv_quat_w: Vec<f32x8>,
    inv_scale: Vec<f32x8>,
    scale_correction: Vec<f32x8>,
    count: usize,
}

impl InstanceDataSoA {
    fn new() -> Self {
        InstanceDataSoA {
            translate_x: Vec::new(),
            translate_y: Vec::new(),
            translate_z: Vec::new(),
            inv_quat_x: Vec::new(),
            inv_quat_y: Vec::new(),
            inv_quat_z: Vec::new(),
            inv_quat_w: Vec::new(),
            inv_scale: Vec::new(),
            scale_correction: Vec::new(),
            count: 0,
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        let chunks = capacity.div_ceil(8);
        InstanceDataSoA {
            translate_x: Vec::with_capacity(chunks),
            translate_y: Vec::with_capacity(chunks),
            translate_z: Vec::with_capacity(chunks),
            inv_quat_x: Vec::with_capacity(chunks),
            inv_quat_y: Vec::with_capacity(chunks),
            inv_quat_z: Vec::with_capacity(chunks),
            inv_quat_w: Vec::with_capacity(chunks),
            inv_scale: Vec::with_capacity(chunks),
            scale_correction: Vec::with_capacity(chunks),
            count: 0,
        }
    }

    /// Append one instance, precomputing inverse transforms
    fn push(&mut self, params: &AnimationParams) {
        let lane = self.count % 8;

        if lane == 0 {
            // Start new chunk — padding lanes translate far away (1e20)
            // so their SDF distance is huge and won't win the min
            self.translate_x.push(f32x8::splat(1e20));
            self.translate_y.push(f32x8::splat(1e20));
            self.translate_z.push(f32x8::splat(1e20));
            self.inv_quat_x.push(f32x8::ZERO);
            self.inv_quat_y.push(f32x8::ZERO);
            self.inv_quat_z.push(f32x8::ZERO);
            self.inv_quat_w.push(f32x8::ONE);
            self.inv_scale.push(f32x8::ONE);
            self.scale_correction.push(f32x8::ONE);
        }

        let chunk = self.count / 8;

        // Precompute inverse rotation quaternion
        let inv_quat = if params.has_rotation() {
            glam::Quat::from_euler(
                glam::EulerRot::XYZ,
                params.rotate_x,
                params.rotate_y,
                params.rotate_z,
            )
            .inverse()
        } else {
            glam::Quat::IDENTITY
        };

        // Precompute inverse scale
        let (inv_scale, sc) = if params.has_scale() {
            (1.0 / params.scale, params.scale)
        } else {
            (1.0, 1.0)
        };

        Self::set_lane(&mut self.translate_x[chunk], lane, params.translate_x);
        Self::set_lane(&mut self.translate_y[chunk], lane, params.translate_y);
        Self::set_lane(&mut self.translate_z[chunk], lane, params.translate_z);
        Self::set_lane(&mut self.inv_quat_x[chunk], lane, inv_quat.x);
        Self::set_lane(&mut self.inv_quat_y[chunk], lane, inv_quat.y);
        Self::set_lane(&mut self.inv_quat_z[chunk], lane, inv_quat.z);
        Self::set_lane(&mut self.inv_quat_w[chunk], lane, inv_quat.w);
        Self::set_lane(&mut self.inv_scale[chunk], lane, inv_scale);
        Self::set_lane(&mut self.scale_correction[chunk], lane, sc);

        self.count += 1;
    }

    #[inline]
    fn set_lane(v: &mut f32x8, lane: usize, value: f32) {
        let mut arr = v.to_array();
        arr[lane] = value;
        *v = f32x8::new(arr);
    }

    #[inline]
    fn chunk_count(&self) -> usize {
        self.translate_x.len()
    }
}

/// Instanced SDF: one compiled shape + N instance transforms
///
/// Stores a single pre-compiled SDF shape and per-instance transforms
/// in both AoS (for scalar eval) and SoA (for SIMD eval) layouts.
/// The SoA layout enables zero-cost SIMD register loads.
#[derive(Debug, Clone)]
pub struct InstancedSdf {
    /// The compiled base shape (shared across all instances)
    pub compiled: CompiledSdf,
    /// Per-instance transforms — AoS layout for scalar evaluation
    instances: Vec<AnimationParams>,
    /// Per-instance transforms — SoA layout for SIMD evaluation
    soa: InstanceDataSoA,
}

impl InstancedSdf {
    /// Create a new instanced SDF from a compiled shape
    pub fn new(compiled: CompiledSdf) -> Self {
        InstancedSdf {
            compiled,
            instances: Vec::new(),
            soa: InstanceDataSoA::new(),
        }
    }

    /// Create with pre-allocated capacity for instances
    pub fn with_capacity(compiled: CompiledSdf, capacity: usize) -> Self {
        InstancedSdf {
            compiled,
            instances: Vec::with_capacity(capacity),
            soa: InstanceDataSoA::with_capacity(capacity),
        }
    }

    /// Add an instance with the given transform
    pub fn add_instance(&mut self, params: AnimationParams) {
        self.soa.push(&params);
        self.instances.push(params);
    }

    /// Add an instance at a position (no rotation, unit scale)
    pub fn add_at(&mut self, x: f32, y: f32, z: f32) {
        let params = AnimationParams {
            translate_x: x,
            translate_y: y,
            translate_z: z,
            scale: 1.0,
            ..Default::default()
        };
        self.soa.push(&params);
        self.instances.push(params);
    }

    /// Number of instances
    pub fn instance_count(&self) -> usize {
        self.soa.count
    }

    /// Evaluate the minimum distance to any instance (scalar fallback)
    ///
    /// Returns the signed distance to the closest instance.
    pub fn eval_min(&self, point: glam::Vec3) -> f32 {
        let mut min_dist = f32::MAX;
        for params in &self.instances {
            let d = crate::animation::eval_animated_compiled(&self.compiled, params, point);
            if d < min_dist {
                min_dist = d;
            }
        }
        min_dist
    }

    /// Evaluate minimum distance using SIMD with SoA zero-cost loads
    ///
    /// Processes 8 instances simultaneously. The SoA layout eliminates
    /// gather/transpose overhead — each `f32x8` loads directly from
    /// contiguous memory into a SIMD register.
    pub fn eval_min_simd(&self, point: glam::Vec3) -> f32 {
        let n = self.soa.count;
        if n == 0 {
            return f32::MAX;
        }

        let p_x = f32x8::splat(point.x);
        let p_y = f32x8::splat(point.y);
        let p_z = f32x8::splat(point.z);

        let mut min_simd = f32x8::splat(f32::MAX);

        for i in 0..self.soa.chunk_count() {
            // Zero-cost SIMD loads from SoA — no gather/transpose!
            let tx = self.soa.translate_x[i];
            let ty = self.soa.translate_y[i];
            let tz = self.soa.translate_z[i];

            // p_local = p - translate
            let local = Vec3x8 {
                x: p_x - tx,
                y: p_y - ty,
                z: p_z - tz,
            };

            // Apply precomputed inverse rotation
            let inv_quat = Quatx8 {
                x: self.soa.inv_quat_x[i],
                y: self.soa.inv_quat_y[i],
                z: self.soa.inv_quat_z[i],
                w: self.soa.inv_quat_w[i],
            };
            let rotated = inv_quat.mul_vec3(local);

            // Apply precomputed inverse scale (multiplication, not division)
            let inv_s = self.soa.inv_scale[i];
            let scaled = Vec3x8 {
                x: rotated.x * inv_s,
                y: rotated.y * inv_s,
                z: rotated.z * inv_s,
            };

            // Evaluate SDF at 8 transformed points
            let raw_dist = eval_compiled_simd(&self.compiled, scaled);

            // Scale correction (precomputed)
            // Padding lanes have scale_correction = f32::MAX → distance = huge
            let corrected = raw_dist * self.soa.scale_correction[i];

            // SIMD min across chunks (reduce at end)
            min_simd = min_simd.min(corrected);
        }

        // Final horizontal reduction
        let arr = min_simd.to_array();
        let mut min_dist = arr[0];
        for &d in &arr[1..] {
            if d < min_dist {
                min_dist = d;
            }
        }
        min_dist
    }

    /// Batch evaluate using SIMD inner loop
    ///
    /// For each query point, evaluates all instances using 8-wide SIMD
    /// and returns the minimum distance.
    pub fn eval_min_batch_simd(&self, points: &[glam::Vec3]) -> Vec<f32> {
        use rayon::prelude::*;

        if points.len() < 256 {
            points.iter().map(|&p| self.eval_min_simd(p)).collect()
        } else {
            points.par_iter().map(|&p| self.eval_min_simd(p)).collect()
        }
    }

    /// Evaluate distance to each instance separately (SoA output)
    ///
    /// Returns a vector of distances, one per instance, for the given query point.
    pub fn eval_per_instance(&self, point: glam::Vec3) -> Vec<f32> {
        self.instances
            .iter()
            .map(|params| crate::animation::eval_animated_compiled(&self.compiled, params, point))
            .collect()
    }

    /// Batch evaluate: for each query point, find minimum distance to any instance
    ///
    /// Uses scalar evaluation for the inner loop.
    pub fn eval_min_batch(&self, points: &[glam::Vec3]) -> Vec<f32> {
        use rayon::prelude::*;

        if points.len() < 256 {
            points.iter().map(|&p| self.eval_min(p)).collect()
        } else {
            points.par_iter().map(|&p| self.eval_min(p)).collect()
        }
    }

    /// Generate WGSL compute shader for instanced SDF evaluation
    ///
    /// Produces a compute shader that:
    /// 1. Reads N instance transforms from a storage buffer
    /// 2. For each query point, evaluates the SDF against all instances
    /// 3. Outputs the minimum distance
    ///
    /// Requires the original SdfNode since CompiledSdf doesn't store the tree.
    #[cfg(feature = "gpu")]
    pub fn to_instanced_wgsl(node: &crate::types::SdfNode) -> String {
        let base_wgsl =
            crate::compiled::WgslShader::transpile(node, crate::compiled::TranspileMode::Hardcoded);

        format!(
            r#"// ALICE-SDF Instanced Compute Shader
// Generated by ALICE-SDF (Project PLASMA)

struct InstanceParams {{
    translate_x: f32,
    translate_y: f32,
    translate_z: f32,
    rotate_x: f32,
    rotate_y: f32,
    rotate_z: f32,
    scale: f32,
    twist: f32,
    bend: f32,
}}

@group(0) @binding(0) var<storage, read> instances: array<InstanceParams>;
@group(0) @binding(1) var<storage, read> points_x: array<f32>;
@group(0) @binding(2) var<storage, read> points_y: array<f32>;
@group(0) @binding(3) var<storage, read> points_z: array<f32>;
@group(0) @binding(4) var<storage, read_write> distances: array<f32>;
@group(0) @binding(5) var<uniform> params: vec2<u32>; // (point_count, instance_count)

{base_sdf}

fn apply_inverse_transform(p: vec3<f32>, inst: InstanceParams) -> vec3<f32> {{
    var q = p;
    q = q - vec3<f32>(inst.translate_x, inst.translate_y, inst.translate_z);
    let cx = cos(-inst.rotate_x); let sx = sin(-inst.rotate_x);
    let cy = cos(-inst.rotate_y); let sy = sin(-inst.rotate_y);
    let cz = cos(-inst.rotate_z); let sz = sin(-inst.rotate_z);
    let rz = mat3x3<f32>(vec3(cz, -sz, 0.0), vec3(sz, cz, 0.0), vec3(0.0, 0.0, 1.0));
    let ry = mat3x3<f32>(vec3(cy, 0.0, sy), vec3(0.0, 1.0, 0.0), vec3(-sy, 0.0, cy));
    let rx = mat3x3<f32>(vec3(1.0, 0.0, 0.0), vec3(0.0, cx, -sx), vec3(0.0, sx, cx));
    q = rx * ry * rz * q;
    if (abs(inst.scale - 1.0) > 1e-6) {{
        q = q / inst.scale;
    }}
    return q;
}}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let point_idx = gid.x;
    let point_count = params.x;
    let instance_count = params.y;

    if (point_idx >= point_count) {{
        return;
    }}

    let p = vec3<f32>(points_x[point_idx], points_y[point_idx], points_z[point_idx]);
    var min_dist: f32 = 1e20;

    for (var i: u32 = 0u; i < instance_count; i = i + 1u) {{
        let inst = instances[i];
        let q = apply_inverse_transform(p, inst);
        var d = sdf_eval(q);
        if (abs(inst.scale - 1.0) > 1e-6) {{
            d = d * inst.scale;
        }}
        min_dist = min(min_dist, d);
    }}

    distances[point_idx] = min_dist;
}}
"#,
            base_sdf = base_wgsl.source,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;

    #[test]
    fn test_instanced_sdf_basic() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);

        let mut instanced = InstancedSdf::new(compiled);
        instanced.add_at(0.0, 0.0, 0.0);
        instanced.add_at(5.0, 0.0, 0.0);

        assert_eq!(instanced.instance_count(), 2);

        // At origin: inside first instance
        let d = instanced.eval_min(glam::Vec3::ZERO);
        assert!((d + 1.0).abs() < 0.01);

        // At (5,0,0): inside second instance
        let d = instanced.eval_min(glam::Vec3::new(5.0, 0.0, 0.0));
        assert!((d + 1.0).abs() < 0.01);

        // At (2.5,0,0): equidistant, should be ~1.5
        let d = instanced.eval_min(glam::Vec3::new(2.5, 0.0, 0.0));
        assert!((d - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_instanced_sdf_scaled() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);

        let mut instanced = InstancedSdf::new(compiled);
        instanced.add_instance(AnimationParams {
            translate_x: 0.0,
            translate_y: 0.0,
            translate_z: 0.0,
            scale: 2.0,
            ..Default::default()
        });

        // At origin: inside scaled sphere (radius=2, distance=-2)
        let d = instanced.eval_min(glam::Vec3::ZERO);
        assert!((d + 2.0).abs() < 0.01);

        // At (2,0,0): on surface
        let d = instanced.eval_min(glam::Vec3::new(2.0, 0.0, 0.0));
        assert!(d.abs() < 0.01);
    }

    #[test]
    fn test_instanced_sdf_batch() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);

        let mut instanced = InstancedSdf::new(compiled);
        instanced.add_at(0.0, 0.0, 0.0);

        let points = vec![
            glam::Vec3::ZERO,
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::new(2.0, 0.0, 0.0),
        ];

        let distances = instanced.eval_min_batch(&points);
        assert_eq!(distances.len(), 3);
        assert!((distances[0] + 1.0).abs() < 0.01);
        assert!(distances[1].abs() < 0.01);
        assert!((distances[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_instanced_per_instance() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);

        let mut instanced = InstancedSdf::new(compiled);
        instanced.add_at(0.0, 0.0, 0.0);
        instanced.add_at(5.0, 0.0, 0.0);

        // At origin: close to first, far from second
        let dists = instanced.eval_per_instance(glam::Vec3::ZERO);
        assert_eq!(dists.len(), 2);
        assert!((dists[0] + 1.0).abs() < 0.01); // inside first
        assert!((dists[1] - 4.0).abs() < 0.01); // far from second
    }

    #[test]
    fn test_instanced_simd_basic() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);

        let mut instanced = InstancedSdf::new(compiled);
        instanced.add_at(0.0, 0.0, 0.0);
        instanced.add_at(5.0, 0.0, 0.0);

        // SIMD should match scalar
        let points = [
            glam::Vec3::ZERO,
            glam::Vec3::new(5.0, 0.0, 0.0),
            glam::Vec3::new(2.5, 0.0, 0.0),
        ];
        for &p in &points {
            let scalar = instanced.eval_min(p);
            let simd = instanced.eval_min_simd(p);
            assert!(
                (scalar - simd).abs() < 0.01,
                "Mismatch at {:?}: scalar={}, simd={}",
                p,
                scalar,
                simd
            );
        }
    }

    #[test]
    fn test_instanced_simd_many() {
        let sphere = SdfNode::sphere(0.5);
        let compiled = CompiledSdf::compile(&sphere);

        let mut instanced = InstancedSdf::new(compiled);
        // Add 20 instances (tests full chunks + remainder)
        for i in 0..20 {
            instanced.add_at(i as f32 * 2.0, 0.0, 0.0);
        }

        let test_points = [
            glam::Vec3::ZERO,
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::new(10.0, 0.0, 0.0),
            glam::Vec3::new(38.0, 0.0, 0.0),
        ];

        for &p in &test_points {
            let scalar = instanced.eval_min(p);
            let simd = instanced.eval_min_simd(p);
            assert!(
                (scalar - simd).abs() < 0.01,
                "Mismatch at {:?}: scalar={}, simd={}",
                p,
                scalar,
                simd
            );
        }
    }

    #[test]
    fn test_instanced_simd_scaled() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);

        let mut instanced = InstancedSdf::new(compiled);
        instanced.add_instance(AnimationParams {
            scale: 2.0,
            ..Default::default()
        });
        instanced.add_instance(AnimationParams {
            translate_x: 5.0,
            scale: 0.5,
            ..Default::default()
        });

        let test_points = [
            glam::Vec3::ZERO,
            glam::Vec3::new(5.0, 0.0, 0.0),
            glam::Vec3::new(2.5, 0.0, 0.0),
        ];

        for &p in &test_points {
            let scalar = instanced.eval_min(p);
            let simd = instanced.eval_min_simd(p);
            assert!(
                (scalar - simd).abs() < 0.01,
                "Mismatch at {:?}: scalar={}, simd={}",
                p,
                scalar,
                simd
            );
        }
    }

    #[test]
    fn test_instanced_batch_simd() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);

        let mut instanced = InstancedSdf::new(compiled);
        instanced.add_at(0.0, 0.0, 0.0);
        instanced.add_at(5.0, 0.0, 0.0);

        let points = vec![
            glam::Vec3::ZERO,
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::new(5.0, 0.0, 0.0),
        ];

        let scalar_results = instanced.eval_min_batch(&points);
        let simd_results = instanced.eval_min_batch_simd(&points);

        for i in 0..points.len() {
            assert!(
                (scalar_results[i] - simd_results[i]).abs() < 0.01,
                "Mismatch at point {}: scalar={}, simd={}",
                i,
                scalar_results[i],
                simd_results[i]
            );
        }
    }

    #[test]
    fn test_instanced_soa_layout() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);

        let mut instanced = InstancedSdf::new(compiled);

        // Add 10 instances — should create 2 chunks (8 + 2)
        for i in 0..10 {
            instanced.add_at(i as f32, 0.0, 0.0);
        }

        assert_eq!(instanced.soa.chunk_count(), 2);
        assert_eq!(instanced.soa.count, 10);

        // Verify SoA data for first instance
        let tx_arr = instanced.soa.translate_x[0].to_array();
        assert_eq!(tx_arr[0], 0.0);
        assert_eq!(tx_arr[7], 7.0);

        // Verify second chunk has 2 valid + 6 padding
        let tx_arr2 = instanced.soa.translate_x[1].to_array();
        assert_eq!(tx_arr2[0], 8.0);
        assert_eq!(tx_arr2[1], 9.0);
        // Padding lanes have translate far away (1e20)
        let tx_pad = instanced.soa.translate_x[1].to_array();
        assert_eq!(tx_pad[2], 1e20);
    }

    #[test]
    fn test_instanced_memory_size() {
        // AnimationParams should be 36 bytes
        assert_eq!(std::mem::size_of::<AnimationParams>(), 36);

        // 10,000 instances = 360KB (vs megabytes for 10K separate SDF trees)
        let total_bytes = 10_000 * std::mem::size_of::<AnimationParams>();
        assert_eq!(total_bytes, 360_000);
    }
}
