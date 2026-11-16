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
use crate::compiled::real::LawMinMax;
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
    const fn new() -> Self {
        Self {
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
        Self {
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
    const fn chunk_count(&self) -> usize {
        self.translate_x.len()
    }
}

/// The most instances an [`InstancedSdf`] holds.
///
/// The instanced WGSL (`to_instanced_wgsl`) loops over at most this
/// many instances (a literal bound, so the shader limits its own work whoever dispatches it, and stops at its
/// `instance_count` uniform); [`InstancedSdf::try_add_instance`] / [`InstancedSdf::try_add_at`] refuse past it.
/// 4096 is a stated ceiling: this crate's tests and examples use at most 6 instances, and no other repository
/// depends on `InstancedSdf`.
pub const MAX_INSTANCES: usize = 4096;

/// An instance past [`MAX_INSTANCES`] was refused
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TooManyInstances;

impl std::fmt::Display for TooManyInstances {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "instanced: more than {MAX_INSTANCES} instances (compiled::MAX_INSTANCES)"
        )
    }
}

impl std::error::Error for TooManyInstances {}

/// The instance limit for a base whose own per-invocation loop product is `base_worst` (see
/// [`InstancedSdf::gpu_instance_limit`]): `floor(MAX_TRIP_PRODUCT / base_worst) - 1`, at most [`MAX_INSTANCES`],
/// refused when 0
#[cfg(feature = "gpu")]
fn limit_for_base_worst(base_worst: u64) -> Result<usize, InstancedGpuRefusal> {
    let base_worst = base_worst.max(1);
    let fit = (crate::gpu_loop_bound::MAX_TRIP_PRODUCT / base_worst).saturating_sub(1);
    if fit == 0 {
        return Err(InstancedGpuRefusal::BaseOverCeiling { base_worst });
    }
    Ok(usize::try_from(fit)
        .unwrap_or(usize::MAX)
        .min(MAX_INSTANCES))
}

/// The instanced WGSL for a base tree, refused for GPU evaluation (typed; nothing is emitted or uploaded)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstancedGpuRefusal {
    /// The base's own loops leave no instance under the per-invocation ceiling: one instance would already cost
    /// more than [`crate::gpu_loop_bound::MAX_TRIP_PRODUCT`] loop iterations with the instance loop around it
    BaseOverCeiling {
        /// The base's own largest per-invocation loop product
        base_worst: u64,
    },
    /// The base's shader has a loop without a stated bound
    BaseUnbounded {
        /// The loop checker's reason
        reason: String,
    },
    /// More instances are held than this base allows ([`InstancedSdf::gpu_instance_limit`])
    TooManyInstances {
        /// Instances held
        count: usize,
        /// The base's limit
        limit: usize,
    },
}

#[cfg(feature = "gpu")]
impl std::fmt::Display for InstancedGpuRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::gpu_loop_bound::MAX_TRIP_PRODUCT;
        match self {
            Self::BaseOverCeiling { base_worst } => write!(
                f,
                "instanced: the base alone may run {base_worst} loop iterations per evaluation, so no instance fits \
                 under the ceiling {MAX_TRIP_PRODUCT} (gpu_loop_bound::MAX_TRIP_PRODUCT); evaluate it on the CPU path"
            ),
            Self::BaseUnbounded { reason } => {
                write!(f, "instanced: the base has an unbounded loop: {reason}")
            }
            Self::TooManyInstances { count, limit } => write!(
                f,
                "instanced: {count} instances, over this base's limit {limit} (InstancedSdf::gpu_instance_limit: \
                 the base's loops times the instance loop stay under gpu_loop_bound::MAX_TRIP_PRODUCT)"
            ),
        }
    }
}

#[cfg(feature = "gpu")]
impl std::error::Error for InstancedGpuRefusal {}

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
    pub const fn new(compiled: CompiledSdf) -> Self {
        Self {
            compiled,
            instances: Vec::new(),
            soa: InstanceDataSoA::new(),
        }
    }

    /// Create with pre-allocated capacity for instances
    pub fn with_capacity(compiled: CompiledSdf, capacity: usize) -> Self {
        Self {
            compiled,
            instances: Vec::with_capacity(capacity),
            soa: InstanceDataSoA::with_capacity(capacity),
        }
    }

    /// Add an instance with the given transform
    ///
    /// Past [`MAX_INSTANCES`] the instance is still added (this form cannot refuse), and the instanced WGSL evaluates
    /// only the first [`MAX_INSTANCES`]; [`Self::try_add_instance`] refuses it.
    #[deprecated(
        note = "use try_add_instance, which refuses an instance past compiled::MAX_INSTANCES"
    )]
    // ALLOW-UNWIRED: deprecated 7.x form kept for compatibility, callers moved to InstancedSdf::try_add_instance
    pub fn add_instance(&mut self, params: AnimationParams) {
        self.soa.push(&params);
        self.instances.push(params);
    }

    /// Add an instance with the given transform, at most [`MAX_INSTANCES`] in all
    ///
    /// # Errors
    ///
    /// [`TooManyInstances`] when [`MAX_INSTANCES`] instances are already held (nothing is added).
    pub fn try_add_instance(&mut self, params: AnimationParams) -> Result<(), TooManyInstances> {
        if self.instances.len() >= MAX_INSTANCES {
            return Err(TooManyInstances);
        }
        self.soa.push(&params);
        self.instances.push(params);
        Ok(())
    }

    /// Add an instance at a position (no rotation, unit scale), at most [`MAX_INSTANCES`] in all
    ///
    /// # Errors
    ///
    /// [`TooManyInstances`] when [`MAX_INSTANCES`] instances are already held (nothing is added).
    pub fn try_add_at(&mut self, x: f32, y: f32, z: f32) -> Result<(), TooManyInstances> {
        self.try_add_instance(AnimationParams {
            translate_x: x,
            translate_y: y,
            translate_z: z,
            scale: 1.0,
            ..Default::default()
        })
    }

    /// Add an instance at a position (no rotation, unit scale)
    ///
    /// As [`Self::add_instance`] past [`MAX_INSTANCES`]; [`Self::try_add_at`] refuses it.
    #[deprecated(note = "use try_add_at, which refuses an instance past compiled::MAX_INSTANCES")]
    // ALLOW-UNWIRED: deprecated 7.x form kept for compatibility, callers moved to InstancedSdf::try_add_at
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

    /// The data the instanced WGSL reads (`to_instanced_wgsl`): the `instances` storage array, 9 `f32` per instance
    /// in the order of its `InstanceParams` (translate x y z, rotate x y z, scale, twist, bend), and the instance
    /// count for the `y` of its `params` uniform. This is the point where an [`InstancedSdf`] becomes GPU input, so
    /// an over-count added through the deprecated `add_instance` / `add_at` is refused here, before any upload or
    /// dispatch, rather than cut silently by the shader's loop bound.
    ///
    /// # Errors
    ///
    /// [`TooManyInstances`] when more than [`MAX_INSTANCES`] instances are held.
    ///
    /// [`MAX_INSTANCES`] is the instanced shader's limit only for a base without loops of its own. This form cannot
    /// see the base's loops (the held [`CompiledSdf`] is the CPU form, not the tree the shader is made from), so for
    /// GPU input use `instance_upload_for` with that tree: it refuses past the base's
    /// `gpu_instance_limit`. A caller knows the two agree when `gpu_instance_limit(&node)` is
    /// [`MAX_INSTANCES`].
    ///
    /// Deprecated because its safe use depends on a check the caller makes elsewhere. It is still correct for a base
    /// without loops of its own, which is when `gpu_instance_limit(&node)` returns [`MAX_INSTANCES`].
    #[deprecated(
        note = "use instance_upload_for(&node), which refuses past the base's own instance limit; this form is \
                correct only for a base without loops (gpu_instance_limit(&node) == MAX_INSTANCES)"
    )]
    // ALLOW-UNWIRED: deprecated 7.x form kept for compatibility, callers moved to InstancedSdf::instance_upload_for
    pub fn instance_upload(&self) -> Result<(Vec<f32>, u32), TooManyInstances> {
        if self.instances.len() > MAX_INSTANCES {
            return Err(TooManyInstances);
        }
        Ok(self.upload_data())
    }

    /// The flat instance data and the count (no check; the callers check the count first)
    fn upload_data(&self) -> (Vec<f32>, u32) {
        let flat = self
            .instances
            .iter()
            .flat_map(|a| {
                [
                    a.translate_x,
                    a.translate_y,
                    a.translate_z,
                    a.rotate_x,
                    a.rotate_y,
                    a.rotate_z,
                    a.scale,
                    a.twist,
                    a.bend,
                ]
            })
            .collect();
        // the callers refuse more than MAX_INSTANCES (4096), so the count fits u32
        (flat, self.instances.len() as u32)
    }

    /// [`Self::instance_upload`] for the base tree the instanced WGSL is made from ([`Self::to_instanced_wgsl`] /
    /// [`Self::try_to_instanced_wgsl`] with the same `node`): refused past that base's
    /// [`Self::gpu_instance_limit`], which is under [`MAX_INSTANCES`] when the base has loops of its own.
    ///
    /// Use this form for GPU input. The held [`CompiledSdf`] is the CPU form and does not show the loops of the
    /// shader made from the tree, so the caller passes the same `node` it passes to the shader.
    ///
    /// # Errors
    ///
    /// [`InstancedGpuRefusal`]: the base is refused, or more instances are held than its limit.
    #[cfg(feature = "gpu")]
    pub fn instance_upload_for(
        &self,
        node: &crate::types::SdfNode,
    ) -> Result<(Vec<f32>, u32), InstancedGpuRefusal> {
        let limit = Self::gpu_instance_limit(node)?;
        let count = self.instances.len();
        if count > limit {
            return Err(InstancedGpuRefusal::TooManyInstances { count, limit });
        }
        // the limit is at most MAX_INSTANCES
        Ok(self.upload_data())
    }

    /// The most instances the instanced WGSL of `node` evaluates: the loop literal [`Self::to_instanced_wgsl`]
    /// emits, and the limit [`Self::instance_upload_for`] refuses past (one function computes it for both).
    ///
    /// The instance loop calls the base's `sdf_eval` once per instance, so the per-invocation loop product is
    /// (limit + 1) × the base's own product (the loop checker counts a loop bounded by `B` as `B + 1` iterations).
    /// The limit is the largest that keeps this at or under [`crate::gpu_loop_bound::MAX_TRIP_PRODUCT`]:
    /// `floor(MAX_TRIP_PRODUCT / base) - 1`, at most [`MAX_INSTANCES`]. A base without loops (product 1) gets
    /// [`MAX_INSTANCES`].
    ///
    /// # Errors
    ///
    /// [`InstancedGpuRefusal::BaseOverCeiling`] when the limit would be 0 (the base alone is over half the
    /// ceiling), with the base's product; [`InstancedGpuRefusal::BaseUnbounded`] for a base with an unbounded loop.
    #[cfg(feature = "gpu")]
    pub fn gpu_instance_limit(node: &crate::types::SdfNode) -> Result<usize, InstancedGpuRefusal> {
        let base =
            crate::compiled::WgslShader::transpile(node, crate::compiled::TranspileMode::Hardcoded);
        Self::limit_of_base(&base)
    }

    /// The limit for an already transpiled base (see [`Self::gpu_instance_limit`])
    #[cfg(feature = "gpu")]
    fn limit_of_base(base: &crate::compiled::WgslShader) -> Result<usize, InstancedGpuRefusal> {
        use crate::gpu_loop_bound::{check_module, parse_source, HostMaxima};
        let module = parse_source(&wgpu::ShaderSource::Wgsl(base.to_compute_shader().into()))
            .map_err(|reason| InstancedGpuRefusal::BaseUnbounded { reason })?;
        let report = check_module(&module, &HostMaxima::new());
        if let Some(u) = report.unbounded.first() {
            return Err(InstancedGpuRefusal::BaseUnbounded {
                reason: format!("fn {}: {}", u.function, u.reason),
            });
        }
        limit_for_base_worst(report.worst)
    }

    /// Number of instances
    pub const fn instance_count(&self) -> usize {
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
            min_simd = min_simd.law_min(corrected);
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
    ///
    /// The text is a whole shader, so it carries this crate's semantics identifier on its first
    /// line (`// alice:semantics_id=<64 hex>`, see [`crate::provenance`]).
    ///
    /// The text is self-contained: a tree's data arrays (lofts, section stacks, height maps,
    /// lattices, polygon outlines) are module constants in it, and its bindings 0-5 are taken (so
    /// it has no node data buffer). A driver that copies a whole constant array at each read at a
    /// run-time index makes a large tree slow here, or fails to build it; to evaluate such a tree
    /// in this crate use [`crate::compiled::GpuEvaluator::new`], which reads the data from a
    /// storage buffer.
    ///
    /// The instance loop's literal bound is [`Self::gpu_instance_limit`] of `node`. For a base that limit refuses,
    /// the loop bound is 1 so the module is over the per-invocation ceiling and is refused where it is created;
    /// [`Self::try_to_instanced_wgsl`] refuses it here, typed.
    #[cfg(feature = "gpu")]
    pub fn to_instanced_wgsl(node: &crate::types::SdfNode) -> String {
        let base_wgsl =
            crate::compiled::WgslShader::transpile(node, crate::compiled::TranspileMode::Hardcoded);
        let limit = Self::limit_of_base(&base_wgsl).unwrap_or(1);
        Self::instanced_text(&base_wgsl, limit)
    }

    /// [`Self::to_instanced_wgsl`], refused (typed) for a base [`Self::gpu_instance_limit`] refuses
    ///
    /// # Errors
    ///
    /// As [`Self::gpu_instance_limit`].
    #[cfg(feature = "gpu")]
    pub fn try_to_instanced_wgsl(
        node: &crate::types::SdfNode,
    ) -> Result<String, InstancedGpuRefusal> {
        let base_wgsl =
            crate::compiled::WgslShader::transpile(node, crate::compiled::TranspileMode::Hardcoded);
        let limit = Self::limit_of_base(&base_wgsl)?;
        Ok(Self::instanced_text(&base_wgsl, limit))
    }

    #[cfg(feature = "gpu")]
    fn instanced_text(base_wgsl: &crate::compiled::WgslShader, limit: usize) -> String {
        crate::provenance::tagged_shader(&format!(
            r"// ALICE-SDF Instanced Compute Shader
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
    // The columns above build Rx(rotate_x) Ry(rotate_y) Rz(rotate_z), the
    // forward XYZ Euler rotation (glam `Quat::from_euler(EulerRot::XYZ, ..)`);
    // the query point needs its inverse, the transpose.
    q = transpose(rx * ry * rz) * q;
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

    // at most this base's instance limit (a literal bound: the shader limits its own work), and no further than
    // the count
    for (var i: u32 = 0u; i < {max_instances}u; i = i + 1u) {{
        if (i >= instance_count) {{
            break;
        }}
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
",
            base_sdf = base_wgsl.source,
            max_instances = limit,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;

    #[test]
    #[allow(
        deprecated,
        reason = "the deprecated form is the way past the maximum this test needs"
    )]
    fn an_over_count_added_through_the_deprecated_form_is_refused_at_upload() {
        let compiled = CompiledSdf::compile(&SdfNode::sphere(0.5));
        let mut inst = InstancedSdf::new(compiled);
        for i in 0..MAX_INSTANCES {
            inst.add_at(i as f32, 0.0, 0.0);
        }
        let (flat, count) = inst.instance_upload().expect("at the maximum");
        assert_eq!(
            (flat.len(), count as usize),
            (9 * MAX_INSTANCES, MAX_INSTANCES)
        );
        inst.add_at(0.0, 0.0, 0.0);
        assert_eq!(inst.instance_upload(), Err(TooManyInstances));
    }

    #[test]
    fn instances_past_the_maximum_are_refused() {
        let compiled = CompiledSdf::compile(&SdfNode::sphere(0.5));
        let mut inst = InstancedSdf::new(compiled);
        for i in 0..MAX_INSTANCES {
            inst.try_add_at(i as f32, 0.0, 0.0)
                .expect("under MAX_INSTANCES");
        }
        assert_eq!(inst.try_add_at(0.0, 0.0, 0.0), Err(TooManyInstances));
        assert_eq!(
            inst.instance_count(),
            MAX_INSTANCES,
            "nothing was added by the refused call"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn the_instanced_shader_bounds_its_own_loop() {
        // the emitted loop's bound is the constant, whatever the instance_count uniform says
        let wgsl = InstancedSdf::to_instanced_wgsl(&SdfNode::sphere(0.5));
        let bound = format!("i < {MAX_INSTANCES}u;");
        assert_eq!(
            InstancedSdf::gpu_instance_limit(&SdfNode::sphere(0.5)),
            Ok(MAX_INSTANCES)
        );
        assert_eq!(wgsl.matches(&bound).count(), 1, "{bound}");
        let r = crate::gpu_loop_bound::check_module(
            &crate::gpu_loop_bound::parse_source(&wgpu::ShaderSource::Wgsl(wgsl.as_str().into()))
                .expect("parses"),
            &crate::gpu_loop_bound::HostMaxima::new(),
        );
        assert!(r.unbounded.is_empty(), "{r:?}");
    }

    #[cfg(feature = "gpu")]
    fn polygons(rows: usize, verts: usize) -> SdfNode {
        let ring = |k: usize| {
            let v: Vec<glam::Vec2> = (0..verts)
                .map(|i| {
                    // a convex arc of y = x^2 (no trigonometry)
                    let x = -1.0 + 2.0 * i as f32 / verts as f32;
                    glam::Vec2::new(x, x * x)
                })
                .collect();
            SdfNode::polygon_2d(v, 0.2).translate(k as f32, 0.0, 0.0)
        };
        (1..rows).fold(ring(0), |acc, k| acc.union(ring(k)))
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn a_base_with_loops_lowers_the_instance_limit_and_the_emitted_bound_with_it() {
        let base = polygons(2, 64);
        let limit = InstancedSdf::gpu_instance_limit(&base).expect("fits");
        assert!(limit < MAX_INSTANCES, "{limit}");
        let wgsl = InstancedSdf::try_to_instanced_wgsl(&base).expect("fits");
        assert_eq!(wgsl, InstancedSdf::to_instanced_wgsl(&base));
        assert_eq!(wgsl.matches(&format!("i < {limit}u;")).count(), 1);
        let r = crate::gpu_loop_bound::check_module(
            &crate::gpu_loop_bound::parse_source(&wgpu::ShaderSource::Wgsl(wgsl.as_str().into()))
                .expect("parses"),
            &crate::gpu_loop_bound::HostMaxima::new(),
        );
        assert!(
            r.worst <= crate::gpu_loop_bound::MAX_TRIP_PRODUCT,
            "{}",
            r.worst
        );
        // one more instance than the limit is refused at the GPU upload, the plain form still accepts it
        let mut inst = InstancedSdf::new(CompiledSdf::compile(&base));
        for i in 0..=limit {
            inst.try_add_at(i as f32, 0.0, 0.0)
                .expect("under MAX_INSTANCES");
        }
        #[allow(
            deprecated,
            reason = "the deprecated form's gap is what this test shows"
        )]
        let plain = inst.instance_upload();
        assert!(plain.is_ok());
        assert_eq!(
            inst.instance_upload_for(&base),
            Err(InstancedGpuRefusal::TooManyInstances {
                count: limit + 1,
                limit
            })
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn a_base_over_half_the_ceiling_is_refused_with_its_product() {
        // the arithmetic at its edges (a tree whose base product is over half the ceiling needs tens of thousands of
        // vertices, whose construction checks take minutes in a debug build; the census covers the corpus trees)
        use crate::gpu_loop_bound::MAX_TRIP_PRODUCT as C;
        assert_eq!(limit_for_base_worst(0), Ok(MAX_INSTANCES));
        assert_eq!(limit_for_base_worst(1), Ok(MAX_INSTANCES));
        assert_eq!(limit_for_base_worst(4097), Ok(14));
        assert_eq!(limit_for_base_worst(C / 2), Ok(1));
        for w in [C / 2 + 1, C, C + 1, u64::MAX] {
            let e = limit_for_base_worst(w).unwrap_err();
            assert_eq!(e, InstancedGpuRefusal::BaseOverCeiling { base_worst: w });
            assert!(e.to_string().contains(&w.to_string()), "{e}");
        }
        // every limit keeps the instanced product, (limit + 1) x base, at or under the ceiling, and is the largest
        for w in 1..=C / 2 {
            let l = limit_for_base_worst(w).expect("fits") as u64;
            assert!((l + 1) * w <= C, "{w}: {l}");
            assert!(
                l == MAX_INSTANCES as u64 || (l + 2) * w > C,
                "{w}: {l} is not the largest"
            );
        }
    }

    #[test]
    fn test_instanced_sdf_basic() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);

        let mut instanced = InstancedSdf::new(compiled);
        instanced
            .try_add_at(0.0, 0.0, 0.0)
            .expect("under MAX_INSTANCES");
        instanced
            .try_add_at(5.0, 0.0, 0.0)
            .expect("under MAX_INSTANCES");

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
        instanced
            .try_add_instance(AnimationParams {
                translate_x: 0.0,
                translate_y: 0.0,
                translate_z: 0.0,
                scale: 2.0,
                ..Default::default()
            })
            .expect("under MAX_INSTANCES");

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
        instanced
            .try_add_at(0.0, 0.0, 0.0)
            .expect("under MAX_INSTANCES");

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
        instanced
            .try_add_at(0.0, 0.0, 0.0)
            .expect("under MAX_INSTANCES");
        instanced
            .try_add_at(5.0, 0.0, 0.0)
            .expect("under MAX_INSTANCES");

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
        instanced
            .try_add_at(0.0, 0.0, 0.0)
            .expect("under MAX_INSTANCES");
        instanced
            .try_add_at(5.0, 0.0, 0.0)
            .expect("under MAX_INSTANCES");

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
            instanced
                .try_add_at(i as f32 * 2.0, 0.0, 0.0)
                .expect("under MAX_INSTANCES");
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
        instanced
            .try_add_instance(AnimationParams {
                scale: 2.0,
                ..Default::default()
            })
            .expect("under MAX_INSTANCES");
        instanced
            .try_add_instance(AnimationParams {
                translate_x: 5.0,
                scale: 0.5,
                ..Default::default()
            })
            .expect("under MAX_INSTANCES");

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
        instanced
            .try_add_at(0.0, 0.0, 0.0)
            .expect("under MAX_INSTANCES");
        instanced
            .try_add_at(5.0, 0.0, 0.0)
            .expect("under MAX_INSTANCES");

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
            instanced
                .try_add_at(i as f32, 0.0, 0.0)
                .expect("under MAX_INSTANCES");
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
