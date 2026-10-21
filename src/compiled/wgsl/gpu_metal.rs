//! The Metal backend of [`GpuEvaluator`](super::GpuEvaluator) (macOS, `msl` feature)
//!
//! wgpu 24 compiles naga's MSL with Metal's default options, which keep fast
//! math on: the Metal compiler then fuses `a * b + c` into one rounding and drops
//! the sign of a zero (measured on Apple silicon, `tests/test_gpu_fma_contraction.rs`).
//! Neither can be switched off through wgpu. This backend runs the same WGSL
//! kernel as MSL from [`MslShader`], whose source starts with
//! `#pragma METAL fp contract(off)`, compiled with
//! [`MslShader::REQUIRED_COMPILE_OPTIONS`] (fast math off), the single source of
//! the options. Bindings follow `MslShader`'s identity map: WGSL `@binding(N)` is
//! Metal `buffer(N)`, and the runtime array lengths go in the buffer at
//! `sizes_buffer_slot`.
//!
//! Author: Moroya Sakamoto

use std::sync::Mutex;

use glam::Vec3;
use metal::{
    CommandQueue, ComputePipelineState, Device, MTLCommandBufferStatus, MTLLanguageVersion,
    MTLResourceOptions, MTLSize,
};

use super::gpu_eval::GpuError;
use super::transpiler::{TranspileMode, WgslShader};
use crate::compiled::msl::MslShader;
use crate::types::SdfNode;

/// Bytes per point on both sides: `InputPoint` (x, y, z, pad) and
/// `OutputDistance` / `OutputFull` are each four `f32`.
const POINT_BYTES: usize = 16;

/// Size of the dynamic parameter buffer, as on the wgpu path (1024 `vec4<f32>`).
const PARAM_FLOATS: usize = 1024 * 4;

/// `_mslBufferSizes` slots filled per dispatch. Every runtime-sized array these
/// kernels declare (input points, output records) has the same byte length, so
/// filling every slot with it is correct whichever member naga maps to which
/// global.
const SIZES_SLOTS: usize = 8;

/// `MTLCompileOptions` built from [`MslShader::REQUIRED_COMPILE_OPTIONS`] and
/// nothing else.
fn compile_options() -> Result<metal::CompileOptions, GpuError> {
    let req = MslShader::REQUIRED_COMPILE_OPTIONS;
    let version = match req.language_version {
        (2, 0) => MTLLanguageVersion::V2_0,
        (2, 1) => MTLLanguageVersion::V2_1,
        (2, 2) => MTLLanguageVersion::V2_2,
        (2, 3) => MTLLanguageVersion::V2_3,
        (2, 4) => MTLLanguageVersion::V2_4,
        (3, 0) => MTLLanguageVersion::V3_0,
        (major, minor) => {
            return Err(GpuError::ShaderCompilation(format!(
                "MSL language version {major}.{minor} has no MTLLanguageVersion mapping"
            )))
        }
    };
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(req.fast_math_enabled);
    options.set_language_version(version);
    Ok(options)
}

/// No Metal device is the same error a caller gets from wgpu without an adapter.
fn require_device(device: Option<Device>) -> Result<Device, GpuError> {
    device.ok_or(GpuError::NoAdapter)
}

/// One compiled kernel and how to dispatch it.
struct Kernel {
    pipeline: ComputePipelineState,
    /// Threads per threadgroup (`@workgroup_size` x).
    threads: u64,
    /// Buffer slot of the runtime array lengths.
    sizes_slot: u64,
}

impl Kernel {
    fn compile(device: &Device, wgsl: &str) -> Result<Self, GpuError> {
        // the text was judged by the GpuEvaluator constructor (or is this crate's own), so a
        // mismatch never reaches here; the second read of its leading block only marks it
        let shader = MslShader::from_wgsl(wgsl)
            .map(|c| c.into_parts().0)
            .map_err(|e| GpuError::ShaderCompilation(format!("MSL: {e}")))?;
        let options = compile_options()?;
        let library = device
            .new_library_with_source(&shader.source, &options)
            .map_err(|e| GpuError::ShaderCompilation(format!("Metal: {e}")))?;
        let function = library
            .get_function(&shader.entry_point, None)
            .map_err(|e| {
                GpuError::ShaderCompilation(format!(
                    "Metal: entry point {} not found: {e}",
                    shader.entry_point
                ))
            })?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| GpuError::ShaderCompilation(format!("Metal pipeline: {e}")))?;
        Ok(Self {
            pipeline,
            threads: u64::from(shader.workgroup_size[0].max(1)),
            sizes_slot: u64::from(shader.sizes_buffer_slot),
        })
    }
}

/// Reusable output and size buffers of a [`GpuBufferPool`](super::GpuBufferPool)
/// on the Metal backend.
///
/// The input points and the point count are passed per call through
/// `newBufferWithBytes` (a copy), so the pool never writes into buffer memory.
pub(super) struct MetalPool {
    capacity: usize,
    output: metal::Buffer,
}

impl MetalPool {
    fn new(device: &Device, capacity: usize) -> Self {
        let capacity = capacity.max(256);
        let output = device.new_buffer(
            (capacity * POINT_BYTES) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Self { capacity, output }
    }

    pub(super) const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Grow by 1.5x or to `required`, whichever is larger, as the wgpu pool does.
    fn ensure_capacity(&mut self, device: &Device, required: usize) {
        if required > self.capacity {
            *self = Self::new(device, (self.capacity * 3 / 2).max(required));
        }
    }
}

/// The Metal backend: a device, its queue, the distance kernel and, in
/// Dynamic mode, the distance + normal kernel and the parameter buffer.
pub(super) struct MetalEngine {
    device: Device,
    queue: CommandQueue,
    distance: Kernel,
    full: Option<Kernel>,
    /// Replaced whole by `update_params` (a fresh buffer copied from the
    /// values), so no buffer memory is written in place.
    params: Option<Mutex<metal::Buffer>>,
}

impl MetalEngine {
    fn device() -> Result<Device, GpuError> {
        require_device(Device::system_default())
    }

    pub(super) fn from_wgsl(wgsl: &str) -> Result<Self, GpuError> {
        let device = Self::device()?;
        let distance = Kernel::compile(&device, wgsl)?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            distance,
            full: None,
            params: None,
        })
    }

    pub(super) fn new_dynamic(node: &SdfNode) -> Result<Self, GpuError> {
        let shader = WgslShader::transpile(node, TranspileMode::Dynamic);
        let device = Self::device()?;
        let distance = Kernel::compile(&device, &shader.to_compute_shader())?;
        let full = Kernel::compile(&device, &shader.to_compute_shader_with_normals())?;
        let params = Self::param_buffer(&device, shader.param_layout);
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            distance,
            full: Some(full),
            params: Some(Mutex::new(params)),
        })
    }

    /// The parameter values, zero-padded to the fixed uniform size.
    fn param_buffer(device: &Device, mut values: Vec<f32>) -> metal::Buffer {
        values.resize(PARAM_FLOATS.max(values.len().next_multiple_of(4)), 0.0);
        device.new_buffer_with_data(
            values.as_ptr().cast(),
            (values.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub(super) fn device_info(&self) -> String {
        format!(
            "Metal device {} (MslShader, fast math off)",
            self.device.name()
        )
    }

    pub(super) fn create_buffer_pool(&self, initial_capacity: usize) -> MetalPool {
        MetalPool::new(&self.device, initial_capacity)
    }

    pub(super) fn update_params(&self, node: &SdfNode) {
        let params = self
            .params
            .as_ref()
            .expect("update_params requires Dynamic mode (use new_dynamic)");
        let fresh = Self::param_buffer(&self.device, WgslShader::extract_params(node));
        *params
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = fresh;
    }

    pub(super) fn eval_batch(&self, points: &[Vec3]) -> Result<Vec<f32>, GpuError> {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let output = self.device.new_buffer(
            (points.len() * POINT_BYTES) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let records = self.dispatch(&self.distance, points, &output)?;
        Ok(records.chunks_exact(4).map(|r| r[0]).collect())
    }

    pub(super) fn eval_batch_pooled(
        &self,
        points: &[Vec3],
        pool: &mut MetalPool,
    ) -> Result<Vec<f32>, GpuError> {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        pool.ensure_capacity(&self.device, points.len());
        let records = self.dispatch(&self.distance, points, &pool.output)?;
        Ok(records.chunks_exact(4).map(|r| r[0]).collect())
    }

    pub(super) fn eval_batch_full(&self, points: &[Vec3]) -> Result<Vec<(f32, Vec3)>, GpuError> {
        let full = self
            .full
            .as_ref()
            .expect("eval_batch_full requires Dynamic mode (use new_dynamic)");
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let output = self.device.new_buffer(
            (points.len() * POINT_BYTES) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let records = self.dispatch(full, points, &output)?;
        Ok(records
            .chunks_exact(4)
            .map(|r| (r[0], Vec3::new(r[1], r[2], r[3])))
            .collect())
    }

    /// Run `kernel` over `points`, writing one four-`f32` record per point into
    /// `output` (at least `points.len() * 16` bytes), and return those records.
    fn dispatch(
        &self,
        kernel: &Kernel,
        points: &[Vec3],
        output: &metal::Buffer,
    ) -> Result<Vec<f32>, GpuError> {
        let n = points.len();
        let byte_len = n * POINT_BYTES;
        let count = u32::try_from(n).map_err(|_| {
            GpuError::BufferMapping(format!("{n} points exceed the u32 point count"))
        })?;
        let array_bytes = u32::try_from(byte_len).map_err(|_| {
            GpuError::BufferMapping(format!("{byte_len} bytes exceed a u32 array length"))
        })?;

        let input: Vec<[f32; 4]> = points.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let input_buf = self.device.new_buffer_with_data(
            input.as_ptr().cast(),
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let count_buf = self.device.new_buffer_with_data(
            std::ptr::addr_of!(count).cast(),
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let sizes = [array_bytes; SIZES_SLOTS];
        let sizes_buf = self.device.new_buffer_with_data(
            sizes.as_ptr().cast(),
            (SIZES_SLOTS * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = self
            .params
            .as_ref()
            .map(|p| p.lock().unwrap_or_else(std::sync::PoisonError::into_inner));

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.pipeline);
        enc.set_buffer(0, Some(&input_buf), 0);
        enc.set_buffer(1, Some(output), 0);
        enc.set_buffer(2, Some(&count_buf), 0);
        if let Some(params) = &params {
            enc.set_buffer(3, Some(params), 0);
        }
        enc.set_buffer(kernel.sizes_slot, Some(&sizes_buf), 0);
        let groups = (n as u64).div_ceil(kernel.threads);
        enc.dispatch_thread_groups(
            MTLSize::new(groups, 1, 1),
            MTLSize::new(kernel.threads, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        drop(params);

        let status = cmd.status();
        if status != MTLCommandBufferStatus::Completed {
            return Err(GpuError::BufferMapping(format!(
                "Metal command buffer ended as {status:?}"
            )));
        }
        // SAFETY: `output` is a shared-storage buffer of at least `byte_len` bytes
        // (allocated by the caller for `n` points), Metal returns its contents
        // aligned for any scalar type, and the command buffer that wrote it has
        // completed, so the GPU no longer writes it while this slice is read and
        // copied out. Shared storage needs no blit to be visible to the CPU.
        let records = unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), n * 4) };
        Ok(records.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without a Metal device a caller sees what wgpu gave without an adapter:
    /// `GpuError::NoAdapter`, displayed as before.
    #[test]
    fn missing_metal_device_is_no_adapter() {
        let err = require_device(None).expect_err("no device must be an error");
        assert!(matches!(err, GpuError::NoAdapter), "got {err:?}");
        assert_eq!(err.to_string(), "Failed to create GPU adapter");
    }

    /// The options come from `REQUIRED_COMPILE_OPTIONS` alone.
    #[test]
    fn compile_options_follow_the_required_ones() {
        let req = MslShader::REQUIRED_COMPILE_OPTIONS;
        let options = compile_options().expect("the required language version maps");
        assert_eq!(options.is_fast_math_enabled(), req.fast_math_enabled);
    }
}
