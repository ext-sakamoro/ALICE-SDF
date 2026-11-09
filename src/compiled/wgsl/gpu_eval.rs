//! GPU Evaluator: WebGPU-based SDF Evaluation (Deep Fried Edition)
//!
//! This module provides GPU-accelerated SDF evaluation.
//! It can evaluate millions of points in parallel on the GPU.
//!
//! # Backends and ALICE-Law G3
//!
//! The WGSL kernel must keep `a * b + c` as two roundings and keep the sign of a
//! zero, as the CPU does. Measured on Apple silicon:
//!
//! - **macOS with the `msl` feature**: the kernel runs on Metal as
//!   [`crate::compiled::msl::MslShader`]'s MSL (`#pragma METAL fp contract(off)`),
//!   compiled with `MslShader::REQUIRED_COMPILE_OPTIONS` (fast math off)
//!   (`gpu_metal.rs`, oracle `tests/test_gpu_fma_contraction.rs`).
//! - **wgpu** (every other target, and macOS without `msl`): wgpu 24 compiles
//!   naga's MSL with Metal's default fast math and offers no option for it, so on
//!   Metal `a * b + c` is fused and the sign of a zero is lost; this is the
//!   non-conforming configuration. Vulkan, DX12 and GL are not measured.
//! - [`GpuEvaluator::from_glsl_compute`] always runs through wgpu.
//!
//! # Deep Fried Optimizations
//!
//! - **Async Execution**: Non-blocking GPU dispatch for CPU/GPU overlap
//! - **Division Exorcism**: WGSL code uses pre-computed reciprocals
//! - **Double Buffering Ready**: Architecture supports ping-pong buffers
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use thiserror::Error;
use wgpu::util::DeviceExt;

use super::transpiler::WgslShader;
use crate::gpu_context::{GpuContext, Held};
use crate::types::SdfNode;

/// Bytes past the output for the dispatch marker (the output's stride, so the output stays a whole array)
const MARKER_BYTES: u64 = 16;

/// Whether a read's mapped bytes hold its dispatch's marker at `end`. wgpu's wait gives up after 60 s and
/// the mapping then completes as though the work had finished (`Device::maintain` does not look at the
/// wait's result), with the output still zeros: the mapping is no proof, the marker is. It is written
/// after the output and is new for each dispatch, so a reused staging buffer cannot pass with the last one
fn dispatch_finished(data: &[u8], end: usize, nonce: u32) -> Result<(), GpuError> {
    let found = data
        .get(end..end + 4)
        .map(|b| u32::from_ne_bytes([b[0], b[1], b[2], b[3]]));
    if found == Some(nonce) {
        Ok(())
    } else {
        Err(GpuError::BufferMapping(format!(
            "the GPU work did not finish: dispatch marker {found:?}, expected {nonce}; its output is not read and the device should not be used again"
        )))
    }
}

/// Error type for GPU evaluation
#[derive(Error, Debug)]
pub enum GpuError {
    /// A batch needs more workgroups than one dispatch of this device can have: a shader with a
    /// small declared size over many points. Split the points, or declare a larger size
    #[error("{points} points at workgroup size {workgroup_size} need more than the device's {limit} workgroups in a dispatch")]
    DispatchTooLarge {
        /// Points in the batch
        points: usize,
        /// The size the shader declares
        workgroup_size: u32,
        /// The device's limit per dimension
        limit: u32,
    },
    /// Failed to create GPU adapter
    #[error("Failed to create GPU adapter")]
    NoAdapter,

    /// Failed to create GPU device
    #[error("Failed to create GPU device: {0}")]
    DeviceCreation(String),

    /// Shader compilation error
    #[error("Shader compilation error: {0}")]
    ShaderCompilation(String),

    /// Buffer mapping error
    #[error("Buffer mapping error: {0}")]
    BufferMapping(String),

    /// The shader text carries another semantics identifier in its leading `//` block (see
    /// [`crate::provenance`]): it was transpiled under another arithmetic
    #[error("shader text: semantics identifier mismatch: stored {stored}, expected {expected}")]
    SemanticsMismatch {
        /// The identifier in the text
        stored: crate::provenance::Provenance,
        /// This crate's identifier
        expected: crate::provenance::Provenance,
    },

    /// The shader text carries no semantics identifier in its leading `//` block, and the
    /// constructor is a `*_pinned` one
    #[error("shader text: no semantics identifier in the leading // block (use the constructor that is not *_pinned to run it unpinned)")]
    Unpinned,

    /// The identifier in the leading `//` block is malformed, or there are two different ones
    #[error("shader text: {0}")]
    InvalidFormat(String),
}

impl From<crate::provenance::ShaderRefusal> for GpuError {
    fn from(r: crate::provenance::ShaderRefusal) -> Self {
        use crate::provenance::ShaderRefusal;
        match r {
            ShaderRefusal::Unpinned => Self::Unpinned,
            ShaderRefusal::Mismatch { stored, expected } => {
                Self::SemanticsMismatch { stored, expected }
            }
            ShaderRefusal::Invalid(e) => Self::InvalidFormat(e),
        }
    }
}

#[cfg(test)]
thread_local! {
    /// The engine builds started on this thread (a unit test checks that a refused text starts
    /// none)
    static ENGINE_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Counts an engine build in the unit tests
#[cfg(test)]
fn note_engine_build() {
    ENGINE_BUILDS.with(|n| n.set(n.get() + 1));
}

/// Nothing in a normal build
#[cfg(not(test))]
#[inline]
const fn note_engine_build() {}

/// Input point for GPU evaluation (16 bytes, aligned)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuInputPoint {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}

impl From<Vec3> for GpuInputPoint {
    fn from(v: Vec3) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            _pad: 0.0,
        }
    }
}

/// Output distance from GPU evaluation (16 bytes, aligned)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuOutputDistance {
    distance: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

/// Output distance + normal from GPU evaluation (16 bytes, aligned)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuOutputFull {
    dist: f32,
    nx: f32,
    ny: f32,
    nz: f32,
}

/// Size of the dynamic parameter buffer (1024 vec4<f32> = 4096 floats = 16KB)
const PARAM_BUFFER_SIZE: u64 = 1024 * 16;

/// The wgpu backend of [`GpuEvaluator`]: every target, and macOS without the `msl`
/// feature (where Metal contracts `a * b + c`, see the module docs).
#[cfg_attr(
    all(target_os = "macos", feature = "msl"),
    allow(
        dead_code,
        reason = "on macOS with msl the WGSL constructors build the Metal engine; wgpu is left for from_glsl_compute"
    )
)]
struct WgpuResources {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Dynamic parameter buffer (present in Dynamic mode)
    param_buffer: Option<wgpu::Buffer>,
    /// Digest of the tree's Dynamic-mode shape: `update_params` refuses another (present in Dynamic mode)
    dynamic_shape: Option<[u8; 32]>,
    /// The last dispatch's marker ([`WgpuEngine::stamp_finished`])
    dispatch_marker: std::sync::atomic::AtomicU32,
    /// A dispatch did not finish: the device may still be running its commands and is not used again
    unusable: std::sync::atomic::AtomicBool,
    /// Distance + normals pipeline (present in Dynamic mode)
    full_pipeline: Option<wgpu::ComputePipeline>,
    /// Bind group layout for 4-binding pipelines
    dynamic_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// [Deep Fried v2] Workgroup size used in shader (matches dispatch calculation)
    workgroup_size: u32,
    /// Node data storage buffer (binding 4), present when the shader reads `alice_node_data`
    node_data: Option<wgpu::Buffer>,
    /// The checked reads' counter (binding 5, one `atomic<u32>`), present for a shader from
    /// [`WgslShader::try_transpile_with_node_data_checked`]
    read_hits: Option<wgpu::Buffer>,
}

type WgpuEngine = Held<WgpuResources>;

#[cfg_attr(
    all(target_os = "macos", feature = "msl"),
    allow(
        dead_code,
        reason = "on macOS with msl the WGSL constructors build the Metal engine; wgpu is left for from_glsl_compute"
    )
)]
impl WgpuEngine {
    fn from_source(source: wgpu::ShaderSource<'_>) -> Result<Self, GpuError> {
        Self::from_source_with_data(source, &[])
    }

    /// [`WgpuEngine::from_source`] with the node data buffer (binding 4) when `node_data` is not
    /// empty
    fn from_source_with_data(
        source: wgpu::ShaderSource<'_>,
        node_data: &[f32],
    ) -> Result<Self, GpuError> {
        Self::from_source_checked(source, node_data, false)
    }

    /// [`WgpuEngine::from_source_with_data`] with, when `checked`, the read counter at binding 5
    fn from_source_checked(
        source: wgpu::ShaderSource<'_>,
        node_data: &[f32],
        checked: bool,
    ) -> Result<Self, GpuError> {
        let workgroup_size = declared_workgroup_size(&source)?;
        let context = GpuContext::request("ALICE-SDF Device")?;
        let device = context.device();

        // Create shader module
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ALICE-SDF Shader"),
            source,
        });

        // Create bind group layout
        let mut layout_entries = vec![
            // Input points buffer
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Output distances buffer
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Point count uniform
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        if !node_data.is_empty() {
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        if checked {
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ALICE-SDF Bind Group Layout"),
            entries: &layout_entries,
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ALICE-SDF Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ALICE-SDF Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let node_data = (!node_data.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ALICE-SDF Node Data Buffer"),
                contents: bytemuck::cast_slice(node_data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        });
        let read_hits = checked.then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ALICE-SDF Read Counter"),
                contents: bytemuck::cast_slice(&[0u32, 0u32]),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        });
        Ok(context.hold(WgpuResources {
            pipeline,
            bind_group_layout,
            param_buffer: None,
            dynamic_shape: None,
            dispatch_marker: std::sync::atomic::AtomicU32::new(0),
            unusable: std::sync::atomic::AtomicBool::new(false),
            full_pipeline: None,
            dynamic_bind_group_layout: None,
            workgroup_size,
            node_data,
            read_hits,
        }))
    }

    /// Evaluate the SDF at multiple points using the GPU
    ///
    /// This is highly efficient for large numbers of points (1000+).
    /// For small numbers of points, CPU evaluation may be faster due to
    /// GPU dispatch overhead.
    fn eval_batch(&self, points: &[Vec3]) -> Result<Vec<f32>, GpuError> {
        self.check_usable()?;
        let workgroup_count = dispatch_count(
            points.len(),
            self.workgroup_size,
            self.context()
                .device()
                .limits()
                .max_compute_workgroups_per_dimension,
        )?;
        if points.is_empty() {
            return Ok(Vec::new());
        }

        let point_count = points.len();

        // Convert input points
        let input_data: Vec<GpuInputPoint> = points.iter().map(|&p| p.into()).collect();

        // Create input buffer
        let input_buffer =
            self.context()
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Input Points Buffer"),
                    contents: bytemuck::cast_slice(&input_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        // Create output buffer
        let output_buffer_size = (point_count * std::mem::size_of::<GpuOutputDistance>()) as u64;
        let output_buffer = self
            .context()
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output Distances Buffer"),
                size: output_buffer_size + MARKER_BYTES,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        // Create staging buffer for reading results
        let staging_buffer = self
            .context()
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: output_buffer_size + MARKER_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        // Create uniform buffer for point count
        let count_buffer =
            self.context()
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Point Count Buffer"),
                    contents: bytemuck::cast_slice(&[point_count as u32]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        // Create bind group
        let bind_group = self.distance_bind_group(
            "ALICE-SDF Bind Group",
            &input_buffer,
            &output_buffer,
            &count_buffer,
        );

        // Create command encoder
        let mut encoder =
            self.context()
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ALICE-SDF Command Encoder"),
                });

        // Dispatch compute shader
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ALICE-SDF Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
        let nonce = self.stamp_finished(
            &mut encoder,
            &output_buffer,
            &staging_buffer,
            output_buffer_size,
        );

        // Submit commands
        self.context().before_host_read(&mut encoder);
        self.context()
            .queue()
            .submit(std::iter::once(encoder.finish()));

        // Read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.wait_for_mapping(&receiver)?;

        let data = buffer_slice.get_mapped_range();
        let end = output_buffer_size as usize;
        self.check_finished(&data, end, nonce)?;
        let output: &[GpuOutputDistance] = bytemuck::cast_slice(&data[..end]);
        let distances: Vec<f32> = output.iter().map(|d| d.distance).collect();

        drop(data);
        staging_buffer.unmap();

        Ok(distances)
    }

    /// [`dispatch_finished`], remembering a failure: after it wgpu has released work the device may
    /// still be running, so this engine refuses further work ([`Self::check_usable`]) instead of reading more
    fn check_finished(&self, data: &[u8], end: usize, nonce: u32) -> Result<(), GpuError> {
        let r = dispatch_finished(data, end, nonce);
        if r.is_err() {
            self.unusable
                .store(true, std::sync::atomic::Ordering::Relaxed);
            self.context().abandon();
        }
        r
    }

    /// [`GpuContext::wait_for_mapping`], and on an error this engine refuses further work
    fn wait_for_mapping(
        &self,
        mapped: &std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ) -> Result<(), GpuError> {
        let r = self.context().wait_for_mapping(mapped);
        if r.is_err() {
            self.unusable
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        r
    }

    /// Err once a dispatch of this engine did not finish
    fn check_usable(&self) -> Result<(), GpuError> {
        if self.unusable.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(GpuError::BufferMapping(
                "an earlier dispatch did not finish: this evaluator is not usable, create another"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Add to `encoder` the writing of a new marker at `end` of `output` and its copy to `end` of `staging`,
    /// after the output's own copy: the changes of `output`'s use order them after the dispatch.
    /// The marker ([`dispatch_finished`])
    fn stamp_finished(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::Buffer,
        staging: &wgpu::Buffer,
        end: u64,
    ) -> u32 {
        let nonce = self
            .dispatch_marker
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            .wrapping_add(1)
            .max(1);
        let source =
            self.context()
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ALICE-SDF Dispatch Marker"),
                    contents: bytemuck::cast_slice(&[nonce]),
                    usage: wgpu::BufferUsages::COPY_SRC,
                });
        encoder.copy_buffer_to_buffer(&source, 0, output, end, 4);
        encoder.copy_buffer_to_buffer(output, end, staging, end, 4);
        nonce
    }

    /// The checked reads that left their proven range since the engine was built (`None` without
    /// the counter)
    fn read_hits(&self) -> Result<Option<u32>, GpuError> {
        self.check_usable()?;
        let Some(hits) = &self.read_hits else {
            return Ok(None);
        };
        let staging = self
            .context()
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("ALICE-SDF Read Counter Staging"),
                size: 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        let mut encoder =
            self.context()
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ALICE-SDF Read Counter Copy"),
                });
        encoder.copy_buffer_to_buffer(hits, 0, &staging, 0, 4);
        let nonce = self.stamp_finished(&mut encoder, hits, &staging, 4);
        self.context().before_host_read(&mut encoder);
        self.context()
            .queue()
            .submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.wait_for_mapping(&receiver)?;
        let n = {
            let data = slice.get_mapped_range();
            self.check_finished(&data, 4, nonce)?;
            bytemuck::cast_slice::<u8, u32>(&data[..4])[0]
        };
        staging.unmap();
        Ok(Some(n))
    }

    /// Bind group for the distance pipeline (`pipeline`).
    ///
    /// A Dynamic evaluator's distance pipeline is built from a shader that
    /// also reads the parameter uniform at binding 3, so its bind group must use
    /// the 4-binding layout and include the parameter buffer, exactly as
    /// `eval_batch_full` does; the 3-binding layout is only compatible with a
    /// Hardcoded pipeline.
    fn distance_bind_group(
        &self,
        label: &str,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: count.as_entire_binding(),
            },
        ];
        let layout = match (&self.dynamic_bind_group_layout, &self.param_buffer) {
            (Some(layout), Some(params)) => {
                entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                });
                layout
            }
            _ => &self.bind_group_layout,
        };
        if let Some(data) = &self.node_data {
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: data.as_entire_binding(),
            });
        }
        if let Some(hits) = &self.read_hits {
            entries.push(wgpu::BindGroupEntry {
                binding: 5,
                resource: hits.as_entire_binding(),
            });
        }
        self.context()
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &entries,
            })
    }

    /// Create a persistent buffer pool for repeated evaluations
    ///
    /// Avoids buffer allocation overhead when calling `eval_batch` many times.
    /// The pool reuses GPU buffers, only re-allocating when capacity is exceeded.
    fn create_buffer_pool(&self, initial_capacity: usize) -> WgpuPool {
        WgpuPool::new(self.context(), initial_capacity)
    }

    /// Evaluate using a persistent buffer pool (avoids re-allocation)
    ///
    /// This is significantly faster than `eval_batch` for repeated calls
    /// with similar-sized point batches (e.g., marching cubes, raymarching).
    fn eval_batch_pooled(
        &self,
        points: &[Vec3],
        pool: &mut WgpuPool,
    ) -> Result<Vec<f32>, GpuError> {
        self.check_usable()?;
        let workgroup_count = dispatch_count(
            points.len(),
            self.workgroup_size,
            self.context()
                .device()
                .limits()
                .max_compute_workgroups_per_dimension,
        )?;
        if points.is_empty() {
            return Ok(Vec::new());
        }

        let point_count = points.len();

        // Ensure pool has enough capacity
        pool.ensure_capacity(self.context(), point_count);

        // Write input data
        let input_data: Vec<GpuInputPoint> = points.iter().map(|&p| p.into()).collect();
        self.context().queue().write_buffer(
            &pool.input_buffer,
            0,
            bytemuck::cast_slice(&input_data),
        );

        // Write point count
        self.context().queue().write_buffer(
            &pool.count_buffer,
            0,
            bytemuck::cast_slice(&[point_count as u32]),
        );

        // Create bind group (lightweight, references existing buffers)
        let bind_group = self.distance_bind_group(
            "ALICE-SDF Pooled Bind Group",
            &pool.input_buffer,
            &pool.output_buffer,
            &pool.count_buffer,
        );

        // Dispatch
        let mut encoder =
            self.context()
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ALICE-SDF Pooled Encoder"),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ALICE-SDF Pooled Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        let read_size = (point_count * std::mem::size_of::<GpuOutputDistance>()) as u64;
        encoder.copy_buffer_to_buffer(&pool.output_buffer, 0, &pool.staging_buffer, 0, read_size);
        let nonce = self.stamp_finished(
            &mut encoder,
            &pool.output_buffer,
            &pool.staging_buffer,
            read_size,
        );

        self.context().before_host_read(&mut encoder);

        self.context()
            .queue()
            .submit(std::iter::once(encoder.finish()));

        // Read results
        let buffer_slice = pool.staging_buffer.slice(..read_size + MARKER_BYTES);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.wait_for_mapping(&receiver)?;

        let data = buffer_slice.get_mapped_range();
        let end = read_size as usize;
        // the pool is used again: its staging buffer is unmapped whatever the marker says
        let finished = self.check_finished(&data, end, nonce);
        let output: &[GpuOutputDistance] = bytemuck::cast_slice(&data[..end]);
        let distances: Vec<f32> = output[..point_count].iter().map(|d| d.distance).collect();

        drop(data);
        pool.staging_buffer.unmap();

        finished?;
        Ok(distances)
    }

    /// Create a GPU evaluator from raw WGSL source (async)
    async fn from_wgsl_async(wgsl_source: &str) -> Result<Self, GpuError> {
        let workgroup_size =
            declared_workgroup_size(&wgpu::ShaderSource::Wgsl(wgsl_source.into()))?;
        let context = GpuContext::request_async("ALICE-SDF Device").await?;
        let device = context.device();

        // Create shader module
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ALICE-SDF Shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ALICE-SDF Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ALICE-SDF Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ALICE-SDF Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(context.hold(WgpuResources {
            pipeline,
            bind_group_layout,
            param_buffer: None,
            dynamic_shape: None,
            dispatch_marker: std::sync::atomic::AtomicU32::new(0),
            unusable: std::sync::atomic::AtomicBool::new(false),
            full_pipeline: None,
            dynamic_bind_group_layout: None,
            workgroup_size,
            node_data: None,
            read_hits: None,
        }))
    }

    /// Evaluate the SDF at multiple points using the GPU (async)
    ///
    /// This is the async version that allows CPU work while GPU computes.
    /// Use this for optimal CPU/GPU overlap in real-time applications.
    #[allow(
        clippy::unused_async,
        clippy::unused_async_trait_impl,
        reason = "the public signature is async; `poll` waits for the work, so nothing is left to await"
    )]
    async fn eval_batch_async(&self, points: &[Vec3]) -> Result<Vec<f32>, GpuError> {
        self.check_usable()?;
        let workgroup_count = dispatch_count(
            points.len(),
            self.workgroup_size,
            self.context()
                .device()
                .limits()
                .max_compute_workgroups_per_dimension,
        )?;
        if points.is_empty() {
            return Ok(Vec::new());
        }

        let point_count = points.len();

        // Convert input points
        let input_data: Vec<GpuInputPoint> = points.iter().map(|&p| p.into()).collect();

        // Create input buffer
        let input_buffer =
            self.context()
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Input Points Buffer"),
                    contents: bytemuck::cast_slice(&input_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        // Create output buffer
        let output_buffer_size = (point_count * std::mem::size_of::<GpuOutputDistance>()) as u64;
        let output_buffer = self
            .context()
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output Distances Buffer"),
                size: output_buffer_size + MARKER_BYTES,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        // Create staging buffer for reading results
        let staging_buffer = self
            .context()
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: output_buffer_size + MARKER_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        // Create uniform buffer for point count
        let count_buffer =
            self.context()
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Point Count Buffer"),
                    contents: bytemuck::cast_slice(&[point_count as u32]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        // Create bind group
        let bind_group = self.distance_bind_group(
            "ALICE-SDF Bind Group",
            &input_buffer,
            &output_buffer,
            &count_buffer,
        );

        // Create command encoder
        let mut encoder =
            self.context()
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ALICE-SDF Command Encoder"),
                });

        // Dispatch compute shader
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ALICE-SDF Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
        let nonce = self.stamp_finished(
            &mut encoder,
            &output_buffer,
            &staging_buffer,
            output_buffer_size,
        );

        // Submit commands (non-blocking)
        self.context().before_host_read(&mut encoder);
        self.context()
            .queue()
            .submit(std::iter::once(encoder.finish()));

        // Map buffer asynchronously
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.wait_for_mapping(&receiver)?;

        let data = buffer_slice.get_mapped_range();
        let end = output_buffer_size as usize;
        self.check_finished(&data, end, nonce)?;
        let output: &[GpuOutputDistance] = bytemuck::cast_slice(&data[..end]);
        let distances: Vec<f32> = output.iter().map(|d| d.distance).collect();

        drop(data);
        staging_buffer.unmap();

        Ok(distances)
    }

    // ========== Dynamic Parameter Binding ==========

    /// Create a new GPU evaluator with Dynamic parameter binding
    ///
    /// In Dynamic mode, SDF parameters (radius, size, blend ratio, etc.)
    /// can be updated at 60fps without shader recompilation via `update_params()`.
    fn new_dynamic(node: &SdfNode) -> Result<Self, GpuError> {
        let shader = WgslShader::transpile_dynamic(node);
        let shape = WgslShader::dynamic_shape(node).0;

        // Build distance-only compute shader (4 bindings)
        let compute_source = shader.to_compute_shader();
        // Build distance+normals compute shader (4 bindings)
        let full_source = shader.to_compute_shader_with_normals();

        let context = GpuContext::request("ALICE-SDF Device (Dynamic)")?;
        let device = context.device();

        // 4-binding layout (shared by both pipelines)
        let mut dynamic_entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        // the data arrays in the node data buffer (see `WgslShader::transpile_dynamic`)
        if !shader.node_data.is_empty() {
            dynamic_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let dynamic_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ALICE-SDF Dynamic Bind Group Layout"),
                entries: &dynamic_entries,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ALICE-SDF Dynamic Pipeline Layout"),
            bind_group_layouts: &[&dynamic_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Distance-only pipeline
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ALICE-SDF Dynamic Shader"),
            source: wgpu::ShaderSource::Wgsl(compute_source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ALICE-SDF Dynamic Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Distance + Normals pipeline
        let full_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ALICE-SDF Full Shader"),
            source: wgpu::ShaderSource::Wgsl(full_source.into()),
        });
        let full_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ALICE-SDF Full Pipeline"),
            layout: Some(&pipeline_layout),
            module: &full_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let node_data = (!shader.node_data.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SDF Node Data Buffer (Dynamic)"),
                contents: bytemuck::cast_slice(&shader.node_data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        });

        // Create parameter buffer with initial values
        let mut param_data = shader.param_layout.clone();
        while !param_data.len().is_multiple_of(4) {
            param_data.push(0.0);
        }
        // Ensure minimum size for the uniform buffer
        param_data.resize(PARAM_BUFFER_SIZE as usize / 4, 0.0);
        let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SDF Params Buffer"),
            contents: bytemuck::cast_slice(&param_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Also create a 3-binding layout for backward compat (unused in Dynamic, but required by struct)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ALICE-SDF Legacy Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        Ok(context.hold(WgpuResources {
            pipeline,
            bind_group_layout,
            param_buffer: Some(param_buffer),
            dynamic_shape: Some(shape),
            dispatch_marker: std::sync::atomic::AtomicU32::new(0),
            unusable: std::sync::atomic::AtomicBool::new(false),
            full_pipeline: Some(full_pipeline),
            dynamic_bind_group_layout: Some(dynamic_bind_group_layout),
            workgroup_size: shader.workgroup_size,
            node_data,
            read_hits: None,
        }))
    }

    /// Update SDF parameters without recompiling shaders
    ///
    /// Traverses the node tree in the same order as transpilation to extract
    /// current values, then uploads them to the GPU via `write_buffer`.
    ///
    /// No shader is recompiled, but the tree is transpiled again to read its values and check its shape:
    /// about 70 µs for 65 primitives (fit for 60fps animation) and 16 ms for a loft of 64,000 values
    /// (a frame at 60fps).
    ///
    /// # Panics
    /// Panics if the evaluator was not created with `new_dynamic()`, or if `node` differs from the tree
    /// it was created with in anything but parameter values (its form, or non-parameter data such as
    /// a loft's rows or a stack's sections, which are compiled into the shader): recreate the
    /// evaluator for that tree.
    fn update_params(&self, node: &SdfNode) {
        let (shape, params) = WgslShader::dynamic_shape(node);
        let expected = self
            .dynamic_shape
            .expect("update_params requires Dynamic mode (use new_dynamic)");
        assert!(
            shape == expected,
            "update_params: the tree differs from the evaluator's in more than parameter values"
        );
        let mut padded = params;
        while !padded.len().is_multiple_of(4) {
            padded.push(0.0);
        }
        padded.resize(PARAM_BUFFER_SIZE as usize / 4, 0.0);
        self.context().queue().write_buffer(
            self.param_buffer
                .as_ref()
                .expect("update_params requires Dynamic mode (use new_dynamic)"),
            0,
            bytemuck::cast_slice(&padded),
        );
    }

    /// Evaluate the SDF at multiple points, returning both distance and normal
    ///
    /// Uses GPU-side Tetrahedral Method for normal estimation (4 SDF evaluations).
    /// Output is `(distance, normal)` per point.
    ///
    /// # Panics
    /// Panics if the evaluator was not created with `new_dynamic()`.
    fn eval_batch_full(&self, points: &[Vec3]) -> Result<Vec<(f32, Vec3)>, GpuError> {
        self.check_usable()?;
        let workgroup_count = dispatch_count(
            points.len(),
            self.workgroup_size,
            self.context()
                .device()
                .limits()
                .max_compute_workgroups_per_dimension,
        )?;
        if points.is_empty() {
            return Ok(Vec::new());
        }

        let full_pipeline = self
            .full_pipeline
            .as_ref()
            .expect("eval_batch_full requires Dynamic mode (use new_dynamic)");
        let dyn_layout = self
            .dynamic_bind_group_layout
            .as_ref()
            .expect("eval_batch_full requires Dynamic mode (use new_dynamic)");
        let param_buf = self
            .param_buffer
            .as_ref()
            .expect("eval_batch_full requires Dynamic mode (use new_dynamic)");

        let point_count = points.len();
        let input_data: Vec<GpuInputPoint> = points.iter().map(|&p| p.into()).collect();

        let input_buffer =
            self.context()
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Input Points Buffer (Full)"),
                    contents: bytemuck::cast_slice(&input_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let output_buffer_size = (point_count * std::mem::size_of::<GpuOutputFull>()) as u64;
        let output_buffer = self
            .context()
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output Full Buffer"),
                size: output_buffer_size + MARKER_BYTES,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        let staging_buffer = self
            .context()
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer (Full)"),
                size: output_buffer_size + MARKER_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        let count_buffer =
            self.context()
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Point Count Buffer"),
                    contents: bytemuck::cast_slice(&[point_count as u32]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context()
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ALICE-SDF Full Bind Group"),
                layout: dyn_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: count_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buf.as_entire_binding(),
                    },
                ]
                .into_iter()
                .chain(self.node_data.as_ref().map(|d| wgpu::BindGroupEntry {
                    binding: 4,
                    resource: d.as_entire_binding(),
                }))
                .collect::<Vec<_>>(),
            });

        let mut encoder =
            self.context()
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ALICE-SDF Full Encoder"),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ALICE-SDF Full Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(full_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
        let nonce = self.stamp_finished(
            &mut encoder,
            &output_buffer,
            &staging_buffer,
            output_buffer_size,
        );
        self.context().before_host_read(&mut encoder);
        self.context()
            .queue()
            .submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.wait_for_mapping(&receiver)?;

        let data = buffer_slice.get_mapped_range();
        let end = output_buffer_size as usize;
        self.check_finished(&data, end, nonce)?;
        let output: &[GpuOutputFull] = bytemuck::cast_slice(&data[..end]);
        let results: Vec<(f32, Vec3)> = output
            .iter()
            .map(|o| (o.dist, Vec3::new(o.nx, o.ny, o.nz)))
            .collect();

        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }
}

/// Auto-tune threshold: below this, single dispatch is optimal
const AUTO_TUNE_THRESHOLD: usize = 262144; // 256K points

/// Determine optimal chunk size for large batches
///
/// Balances GPU occupancy against buffer transfer overhead.
/// Larger chunks = fewer dispatches but more latency per chunk.
const fn optimal_chunk_size(total_points: usize) -> usize {
    if total_points <= 262144 {
        total_points // Single dispatch
    } else if total_points <= 1_048_576 {
        262144 // 256K chunks (good for most GPUs)
    } else {
        524288 // 512K chunks (for very large batches)
    }
}

/// Persistent GPU buffer pool for repeated evaluations
///
/// Avoids the overhead of creating and destroying GPU buffers for every
/// `eval_batch` call. Buffers are allocated once and reused, only growing
/// when the batch size exceeds current capacity.
///
/// # Performance
///
/// Buffer creation is one of the most expensive GPU operations.
/// By reusing buffers, `eval_batch_pooled` can be 2-5x faster than
/// `eval_batch` for repeated calls with similar-sized batches.
#[cfg_attr(
    all(target_os = "macos", feature = "msl"),
    allow(
        dead_code,
        reason = "on macOS with msl the WGSL constructors build the Metal engine; wgpu is left for from_glsl_compute"
    )
)]
struct WgpuPoolBuffers {
    /// Current capacity (number of points)
    capacity: usize,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    count_buffer: wgpu::Buffer,
}

/// A pool can outlive its evaluator, and its buffers hold the device
type WgpuPool = Held<WgpuPoolBuffers>;

#[cfg_attr(
    all(target_os = "macos", feature = "msl"),
    allow(
        dead_code,
        reason = "on macOS with msl the WGSL constructors build the Metal engine; wgpu is left for from_glsl_compute"
    )
)]
impl WgpuPool {
    /// Create a new buffer pool with the given initial capacity
    fn new(context: &GpuContext, capacity: usize) -> Self {
        let device = context.device();
        let cap = capacity.max(256); // Minimum 256 points
        let input_size = (cap * std::mem::size_of::<GpuInputPoint>()) as u64;
        let output_size = (cap * std::mem::size_of::<GpuOutputDistance>()) as u64;

        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pool Input Buffer"),
            size: input_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pool Output Buffer"),
            size: output_size + MARKER_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pool Staging Buffer"),
            size: output_size + MARKER_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pool Count Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        context.clone().hold(WgpuPoolBuffers {
            capacity: cap,
            input_buffer,
            output_buffer,
            staging_buffer,
            count_buffer,
        })
    }

    /// Ensure the pool can hold at least `required` points
    ///
    /// If the current capacity is insufficient, new larger buffers are allocated.
    /// Uses 1.5x growth factor to amortize allocation cost.
    fn ensure_capacity(&mut self, context: &GpuContext, required: usize) {
        if required <= self.capacity {
            return;
        }

        // Grow by 1.5x or to required size, whichever is larger
        let new_cap = (self.capacity * 3 / 2).max(required);
        *self = Self::new(context, new_cap);
    }
}

impl std::fmt::Debug for WgpuPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuPool")
            .field("capacity", &self.capacity)
            .field(
                "input_bytes",
                &(self.capacity * std::mem::size_of::<GpuInputPoint>()),
            )
            .field(
                "output_bytes",
                &(self.capacity * std::mem::size_of::<GpuOutputDistance>()),
            )
            .finish_non_exhaustive()
    }
}

/// The workgroups a batch needs in one dispatch
///
/// # Errors
/// [`GpuError::DispatchTooLarge`] over the device's limit, where the dispatch would be refused by
/// validation and panic.
fn dispatch_count(points: usize, workgroup_size: u32, limit: u32) -> Result<u32, GpuError> {
    let too_large = || GpuError::DispatchTooLarge {
        points,
        workgroup_size,
        limit,
    };
    // a size of 0 does not get here on a device; no count is right for it
    let count = (points as u64).div_ceil(u64::from(workgroup_size.max(1)));
    u32::try_from(count)
        .ok()
        .filter(|c| *c <= limit)
        .ok_or_else(too_large)
}

/// The `x` of the `@workgroup_size` of the compute entry point `main`, since a dispatch is counted in
/// these: a size other than the shader's own leaves points unevaluated, at 0, with no error. An
/// override counts with its default, which is what the pipeline runs with.
///
/// # Errors
/// [`GpuError::ShaderCompilation`] when the source does not parse, has no such entry point, or its
/// size is an override with no constant default.
fn declared_workgroup_size(source: &wgpu::ShaderSource<'_>) -> Result<u32, GpuError> {
    let failed = GpuError::ShaderCompilation;
    let module = match source {
        wgpu::ShaderSource::Wgsl(text) => {
            wgpu::naga::front::wgsl::parse_str(text).map_err(|e| failed(e.emit_to_string(text)))?
        }
        #[cfg(feature = "glsl")]
        wgpu::ShaderSource::Glsl {
            shader,
            stage,
            defines,
        } => wgpu::naga::front::glsl::Frontend::default()
            .parse(
                &wgpu::naga::front::glsl::Options {
                    stage: *stage,
                    defines: defines.clone(),
                },
                shader,
            )
            .map_err(|e| failed(e.emit_to_string(shader)))?,
        _ => return Err(failed("unsupported shader source".to_owned())),
    };
    let entry = module
        .entry_points
        .iter()
        .find(|e| e.name == "main" && e.stage == wgpu::naga::ShaderStage::Compute)
        .ok_or_else(|| failed("no compute entry point `main`".to_owned()))?;
    let Some(overrides) = entry.workgroup_size_overrides else {
        return Ok(entry.workgroup_size[0]);
    };
    let Some(x) = overrides[0] else {
        return Ok(entry.workgroup_size[0]);
    };
    let unknown = || failed("`@workgroup_size` is an override with no constant default".to_owned());
    let mut expression = &module.global_expressions[x];
    if let wgpu::naga::Expression::Override(handle) = expression {
        let init = module.overrides[*handle].init.ok_or_else(unknown)?;
        expression = &module.global_expressions[init];
    }
    match expression {
        wgpu::naga::Expression::Literal(wgpu::naga::Literal::U32(v)) => Ok(*v),
        wgpu::naga::Expression::Literal(wgpu::naga::Literal::I32(v)) => {
            u32::try_from(*v).map_err(|_| unknown())
        }
        wgpu::naga::Expression::Literal(wgpu::naga::Literal::AbstractInt(v)) => {
            u32::try_from(*v).map_err(|_| unknown())
        }
        _ => Err(unknown()),
    }
}

/// The backend a [`GpuEvaluator`] runs on.
enum Engine {
    #[cfg_attr(
        all(target_os = "macos", feature = "msl", not(feature = "glsl")),
        allow(
            dead_code,
            reason = "on macOS with msl only from_glsl_compute builds the wgpu engine"
        )
    )]
    Wgpu(WgpuEngine),
    #[cfg(all(target_os = "macos", feature = "msl"))]
    Metal(super::gpu_metal::MetalEngine),
}

/// GPU-based SDF evaluator
///
/// Evaluates the WGSL compute kernel of an SDF at many points in parallel. On
/// macOS with the `msl` feature the kernel runs on Metal through
/// [`crate::compiled::msl::MslShader`] with
/// [`MslShader::REQUIRED_COMPILE_OPTIONS`](crate::compiled::msl::MslShader::REQUIRED_COMPILE_OPTIONS),
/// so `a * b + c` keeps its two roundings and a zero keeps its sign (ALICE-Law G3);
/// elsewhere it runs through wgpu (see the module docs for what is measured per
/// backend).
pub struct GpuEvaluator {
    engine: Engine,
}

impl GpuEvaluator {
    /// Create a new GPU evaluator for the given SDF
    ///
    /// This compiles the SDF to a WGSL compute shader and creates
    /// the necessary GPU resources. A union chain at the top skips operands by their bounding
    /// spheres ([`WgslShader::transpile_with_union_bounds`]); the values are those of the plain
    /// shader.
    pub fn new(node: &SdfNode) -> Result<Self, GpuError> {
        // per-node data (lattice control points, height fields, grid warp maps, lofts, stacks) goes
        // to the node data buffer on both engines
        let shader = WgslShader::try_transpile_with_node_data(
            node,
            super::transpiler::TranspileMode::Hardcoded,
        )
        .map_err(|e| GpuError::InvalidFormat(e.to_string()))?;
        Self::from_shader(&shader)
    }

    /// [`GpuEvaluator::new`] with the loft and stack data reads checked: every read outside the
    /// index range the emit proved for it is counted ([`GpuEvaluator::read_hits`]), with the same
    /// values. For tests and diagnostics, on either engine
    ///
    /// # Errors
    ///
    /// As [`GpuEvaluator::new`], and a tree whose reads the emit cannot prove in range
    /// ([`WgslShader::try_transpile_with_node_data_checked`])
    pub fn new_checked(node: &SdfNode) -> Result<Self, GpuError> {
        let shader = WgslShader::try_transpile_with_node_data_checked(
            node,
            super::transpiler::TranspileMode::Hardcoded,
        )
        .map_err(|e| GpuError::InvalidFormat(e.to_string()))?;
        let text = shader.to_compute_shader();
        crate::provenance::shader_text_verdict(&text, false)?;
        note_engine_build();
        #[cfg(all(target_os = "macos", feature = "msl"))]
        {
            super::gpu_metal::MetalEngine::from_wgsl_with_data(&text, &shader.node_data, true).map(
                |m| Self {
                    engine: Engine::Metal(m),
                },
            )
        }
        #[cfg(not(all(target_os = "macos", feature = "msl")))]
        {
            let mut evaluator = WgpuEngine::from_source_checked(
                wgpu::ShaderSource::Wgsl(text.into()),
                &shader.node_data,
                true,
            )
            .map(|w| Self {
                engine: Engine::Wgpu(w),
            })?;
            evaluator.set_wgpu_workgroup_size(shader.workgroup_size);
            Ok(evaluator)
        }
    }

    /// The checked reads that left their proven range in every evaluation so far (an evaluator from
    /// [`GpuEvaluator::new_checked`]; `None` otherwise). Zero means every loft and stack read stayed
    /// inside the range the emit proved, so the buffer clamps never changed a value
    ///
    /// # Errors
    ///
    /// The counter could not be read back
    pub fn read_hits(&self) -> Result<Option<u32>, GpuError> {
        match &self.engine {
            Engine::Wgpu(w) => w.read_hits(),
            #[cfg(all(target_os = "macos", feature = "msl"))]
            Engine::Metal(m) => Ok(m.read_hits()),
        }
    }

    /// Create a GPU evaluator from a pre-generated WGSL shader
    ///
    /// The shader is this crate's own, so its text carries this crate's identifier; it is read
    /// through [`GpuEvaluator::from_wgsl_pinned`].
    pub fn from_shader(shader: &WgslShader) -> Result<Self, GpuError> {
        if !shader.node_data.is_empty() {
            let text = shader.to_compute_shader();
            crate::provenance::shader_text_verdict(&text, false)?;
            note_engine_build();
            #[cfg(all(target_os = "macos", feature = "msl"))]
            {
                let mut evaluator = super::gpu_metal::MetalEngine::from_wgsl_with_data(
                    &text,
                    &shader.node_data,
                    false,
                )
                .map(|m| Self {
                    engine: Engine::Metal(m),
                })?;
                evaluator.set_wgpu_workgroup_size(shader.workgroup_size);
                return Ok(evaluator);
            }
            #[cfg(not(all(target_os = "macos", feature = "msl")))]
            {
                let mut evaluator = WgpuEngine::from_source_with_data(
                    wgpu::ShaderSource::Wgsl(text.into()),
                    &shader.node_data,
                )
                .map(|w| Self {
                    engine: Engine::Wgpu(w),
                })?;
                evaluator.set_wgpu_workgroup_size(shader.workgroup_size);
                return Ok(evaluator);
            }
        }
        let mut evaluator = Self::from_wgsl_pinned(&shader.to_compute_shader())?;
        evaluator.set_wgpu_workgroup_size(shader.workgroup_size);
        Ok(evaluator)
    }

    /// Create a GPU evaluator from raw WGSL source, with what is known about the arithmetic it was
    /// transpiled under
    ///
    /// The identifier is read from the leading `//` block of the text (see
    /// [`crate::provenance`]) before any device is requested: this crate's text is
    /// [`StoredSemantics::Current`](crate::provenance::StoredSemantics::Current), text without
    /// the key anywhere (another program's WGSL) is `Unpinned`, text under another identifier is
    /// refused, and the key outside the leading block (below it) is refused as invalid.
    /// [`GpuEvaluator::from_wgsl_pinned`] also refuses `Unpinned`.
    ///
    /// # Errors
    ///
    /// [`GpuError::SemanticsMismatch`] for another identifier, [`GpuError::InvalidFormat`] for a
    /// malformed one, and the device and compile errors.
    pub fn from_wgsl(wgsl_source: &str) -> Result<crate::provenance::Checked<Self>, GpuError> {
        let semantics = crate::provenance::shader_text_verdict(wgsl_source, true)?;
        Self::build_from_wgsl(wgsl_source).map(|e| crate::provenance::Checked::new(e, semantics))
    }

    /// [`GpuEvaluator::from_wgsl`] that refuses text without this crate's identifier
    ///
    /// # Errors
    ///
    /// [`GpuError::Unpinned`] for text without an identifier, otherwise as
    /// [`GpuEvaluator::from_wgsl`].
    pub fn from_wgsl_pinned(wgsl_source: &str) -> Result<Self, GpuError> {
        crate::provenance::shader_text_verdict(wgsl_source, false)?;
        Self::build_from_wgsl(wgsl_source)
    }

    fn build_from_wgsl(wgsl_source: &str) -> Result<Self, GpuError> {
        note_engine_build();
        #[cfg(all(target_os = "macos", feature = "msl"))]
        {
            super::gpu_metal::MetalEngine::from_wgsl(wgsl_source).map(|m| Self {
                engine: Engine::Metal(m),
            })
        }
        #[cfg(not(all(target_os = "macos", feature = "msl")))]
        {
            WgpuEngine::from_source(wgpu::ShaderSource::Wgsl(wgsl_source.into())).map(|w| Self {
                engine: Engine::Wgpu(w),
            })
        }
    }

    /// Create a GPU evaluator from a GLSL compute shader (`#version 450`,
    /// `main` entry, the same three bindings as the WGSL compute wrapper).
    /// wgpu compiles it through naga's GLSL front end, so the GLSL
    /// transpiler's output can be run on the GPU and compared with the CPU
    /// — the parity oracle for the GLSL path (`tests/test_gpu_law_parity.rs`).
    ///
    /// This always runs through wgpu, also on macOS: Metal may contract
    /// `a * b + c` here (see the module docs).
    ///
    /// The identifier is read from the leading `//` block as in [`GpuEvaluator::from_wgsl`]
    /// (for GLSL the block comes right after the `#version` line, which GLSL ES needs first), and
    /// [`GpuEvaluator::from_glsl_compute_pinned`] also refuses `Unpinned`.
    ///
    /// # Errors
    ///
    /// As [`GpuEvaluator::from_wgsl`].
    #[cfg(feature = "glsl")]
    pub fn from_glsl_compute(
        glsl_source: &str,
    ) -> Result<crate::provenance::Checked<Self>, GpuError> {
        let semantics = crate::provenance::shader_text_verdict(glsl_source, true)?;
        Self::build_from_glsl_compute(glsl_source)
            .map(|e| crate::provenance::Checked::new(e, semantics))
    }

    /// [`GpuEvaluator::from_glsl_compute`] that refuses text without this crate's identifier
    ///
    /// # Errors
    ///
    /// [`GpuError::Unpinned`] for text without an identifier, otherwise as
    /// [`GpuEvaluator::from_glsl_compute`].
    #[cfg(feature = "glsl")]
    pub fn from_glsl_compute_pinned(glsl_source: &str) -> Result<Self, GpuError> {
        crate::provenance::shader_text_verdict(glsl_source, false)?;
        Self::build_from_glsl_compute(glsl_source)
    }

    #[cfg(feature = "glsl")]
    fn build_from_glsl_compute(glsl_source: &str) -> Result<Self, GpuError> {
        note_engine_build();
        WgpuEngine::from_source(wgpu::ShaderSource::Glsl {
            shader: glsl_source.into(),
            stage: wgpu::naga::ShaderStage::Compute,
            defines: wgpu::naga::FastHashMap::default(),
        })
        .map(|w| Self {
            engine: Engine::Wgpu(w),
        })
    }

    /// The wgpu path dispatches with the transpiler's workgroup size; the Metal
    /// path reads it from the kernel itself.
    fn set_wgpu_workgroup_size(&mut self, size: u32) {
        match &mut self.engine {
            Engine::Wgpu(w) => w.workgroup_size = size,
            #[cfg(all(target_os = "macos", feature = "msl"))]
            Engine::Metal(_) => {}
        }
    }

    /// Evaluate the SDF at multiple points using the GPU
    ///
    /// This is highly efficient for large numbers of points (1000+).
    /// For small numbers of points, CPU evaluation may be faster due to
    /// GPU dispatch overhead.
    pub fn eval_batch(&self, points: &[Vec3]) -> Result<Vec<f32>, GpuError> {
        match &self.engine {
            Engine::Wgpu(w) => w.eval_batch(points),
            #[cfg(all(target_os = "macos", feature = "msl"))]
            Engine::Metal(m) => m.eval_batch(points),
        }
    }

    /// Set how long a read on this evaluator waits for its GPU work before it returns an error.
    /// The default is 10 minutes. A first dispatch can take long on a driver that compiles the
    /// shader there. After that error the evaluator refuses further work and its device is kept for
    /// the life of the process, because the work may still be running.
    ///
    /// The Metal engine waits for its command buffer without a limit, and this does not change it.
    pub fn set_wait_limit(&self, limit: std::time::Duration) {
        match &self.engine {
            Engine::Wgpu(w) => w.context().set_wait_limit(limit),
            #[cfg(all(feature = "msl", target_vendor = "apple"))]
            Engine::Metal(_) => {}
        }
    }

    /// Get information about the GPU being used
    pub fn device_info(&self) -> String {
        match &self.engine {
            Engine::Wgpu(_) => "WebGPU Device (via wgpu)".to_string(),
            #[cfg(all(target_os = "macos", feature = "msl"))]
            Engine::Metal(m) => m.device_info(),
        }
    }

    /// Create a persistent buffer pool for repeated evaluations
    ///
    /// Avoids buffer allocation overhead when calling `eval_batch` many times.
    /// The pool reuses GPU buffers, only re-allocating when capacity is exceeded.
    pub fn create_buffer_pool(&self, initial_capacity: usize) -> GpuBufferPool {
        match &self.engine {
            Engine::Wgpu(w) => GpuBufferPool::from_wgpu(w.create_buffer_pool(initial_capacity)),
            #[cfg(all(target_os = "macos", feature = "msl"))]
            Engine::Metal(m) => GpuBufferPool::from_metal(m.create_buffer_pool(initial_capacity)),
        }
    }

    /// Evaluate using a persistent buffer pool (avoids re-allocation)
    ///
    /// This is significantly faster than `eval_batch` for repeated calls
    /// with similar-sized point batches (e.g., marching cubes, raymarching).
    ///
    /// # Panics
    ///
    /// Panics if `pool` was created by an evaluator on another backend.
    pub fn eval_batch_pooled(
        &self,
        points: &[Vec3],
        pool: &mut GpuBufferPool,
    ) -> Result<Vec<f32>, GpuError> {
        let result = match (&self.engine, &mut pool.buffers) {
            (Engine::Wgpu(w), PoolBuffers::Wgpu(p)) => w.eval_batch_pooled(points, p),
            #[cfg(all(target_os = "macos", feature = "msl"))]
            (Engine::Metal(m), PoolBuffers::Metal(p)) => m.eval_batch_pooled(points, p),
            #[cfg(all(target_os = "macos", feature = "msl"))]
            _ => panic!("GpuBufferPool was created by an evaluator on another backend"),
        };
        pool.sync_capacity();
        result
    }

    /// Auto-tuned batch evaluation
    ///
    /// Splits large batches into optimal chunk sizes to balance GPU
    /// occupancy with buffer transfer overhead.
    pub fn eval_batch_auto(
        &self,
        points: &[Vec3],
        pool: &mut GpuBufferPool,
    ) -> Result<Vec<f32>, GpuError> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        // For small batches, use single dispatch
        if points.len() <= AUTO_TUNE_THRESHOLD {
            return self.eval_batch_pooled(points, pool);
        }

        // Split into optimal chunks
        let chunk_size = optimal_chunk_size(points.len());
        let mut all_distances = Vec::with_capacity(points.len());

        for chunk in points.chunks(chunk_size) {
            let distances = self.eval_batch_pooled(chunk, pool)?;
            all_distances.extend_from_slice(&distances);
        }

        Ok(all_distances)
    }

    // ========== Deep Fried Edition: Async API ==========

    /// Create a new GPU evaluator asynchronously
    ///
    /// Use this when you want to initialize the GPU without blocking.
    pub async fn new_async(node: &SdfNode) -> Result<Self, GpuError> {
        let shader = WgslShader::transpile(node, super::transpiler::TranspileMode::Hardcoded);
        Self::from_shader_async(&shader).await
    }

    /// Create a GPU evaluator from a pre-generated WGSL shader (async)
    pub async fn from_shader_async(shader: &WgslShader) -> Result<Self, GpuError> {
        if !shader.node_data.is_empty() {
            return Err(GpuError::InvalidFormat(
                "node data buffer (binding 4) is bound by the synchronous constructor only (GpuEvaluator::from_shader)".into(),
            ));
        }
        let compute_shader = shader.to_compute_shader();
        let mut evaluator = Self::from_wgsl_async_pinned(&compute_shader).await?;
        evaluator.set_wgpu_workgroup_size(shader.workgroup_size);
        Ok(evaluator)
    }

    /// Create a GPU evaluator from raw WGSL source (async)
    ///
    /// The identifier is read from the leading `//` block as in [`GpuEvaluator::from_wgsl`],
    /// before any device is requested; [`GpuEvaluator::from_wgsl_async_pinned`] also refuses
    /// `Unpinned`.
    ///
    /// On the Metal path the device and the pipeline are made synchronously
    /// (Metal has no asynchronous device request).
    ///
    /// # Errors
    ///
    /// As [`GpuEvaluator::from_wgsl`].
    pub async fn from_wgsl_async(
        wgsl_source: &str,
    ) -> Result<crate::provenance::Checked<Self>, GpuError> {
        let semantics = crate::provenance::shader_text_verdict(wgsl_source, true)?;
        Self::build_from_wgsl_async(wgsl_source)
            .await
            .map(|e| crate::provenance::Checked::new(e, semantics))
    }

    /// [`GpuEvaluator::from_wgsl_async`] that refuses text without this crate's identifier
    ///
    /// # Errors
    ///
    /// [`GpuError::Unpinned`] for text without an identifier, otherwise as
    /// [`GpuEvaluator::from_wgsl_async`].
    pub async fn from_wgsl_async_pinned(wgsl_source: &str) -> Result<Self, GpuError> {
        crate::provenance::shader_text_verdict(wgsl_source, false)?;
        Self::build_from_wgsl_async(wgsl_source).await
    }

    #[cfg_attr(
        all(target_os = "macos", feature = "msl"),
        allow(
            clippy::unused_async,
            clippy::unused_async_trait_impl,
            reason = "the async signature serves the wgpu path; Metal has no async device request"
        )
    )]
    async fn build_from_wgsl_async(wgsl_source: &str) -> Result<Self, GpuError> {
        #[cfg(all(target_os = "macos", feature = "msl"))]
        {
            Self::build_from_wgsl(wgsl_source)
        }
        #[cfg(not(all(target_os = "macos", feature = "msl")))]
        {
            note_engine_build();
            WgpuEngine::from_wgsl_async(wgsl_source)
                .await
                .map(|w| Self {
                    engine: Engine::Wgpu(w),
                })
        }
    }

    /// Evaluate the SDF at multiple points using the GPU (async)
    ///
    /// This is the async version that allows CPU work while GPU computes.
    /// Use this for optimal CPU/GPU overlap in real-time applications.
    pub async fn eval_batch_async(&self, points: &[Vec3]) -> Result<Vec<f32>, GpuError> {
        match &self.engine {
            Engine::Wgpu(w) => w.eval_batch_async(points).await,
            #[cfg(all(target_os = "macos", feature = "msl"))]
            Engine::Metal(m) => m.eval_batch(points),
        }
    }

    // ========== Dynamic Parameter Binding ==========

    /// Create a new GPU evaluator with Dynamic parameter binding
    ///
    /// In Dynamic mode, SDF parameters (radius, size, blend ratio, etc.)
    /// can be updated at 60fps without shader recompilation via `update_params()`.
    pub fn new_dynamic(node: &SdfNode) -> Result<Self, GpuError> {
        #[cfg(all(target_os = "macos", feature = "msl"))]
        {
            super::gpu_metal::MetalEngine::new_dynamic(node).map(|m| Self {
                engine: Engine::Metal(m),
            })
        }
        #[cfg(not(all(target_os = "macos", feature = "msl")))]
        {
            WgpuEngine::new_dynamic(node).map(|w| Self {
                engine: Engine::Wgpu(w),
            })
        }
    }

    /// Update SDF parameters without recompiling shaders
    ///
    /// Traverses the node tree in the same order as transpilation to extract
    /// current values, then uploads them to the GPU.
    ///
    /// No shader is recompiled, but the tree is transpiled again to read its values and check its shape:
    /// about 70 µs for 65 primitives (fit for 60fps animation) and 16 ms for a loft of 64,000 values
    /// (a frame at 60fps).
    ///
    /// # Panics
    /// Panics if the evaluator was not created with `new_dynamic()`, or if `node` differs from the tree
    /// it was created with in anything but parameter values (its form, or non-parameter data such as
    /// a loft's rows or a stack's sections, which are compiled into the shader): recreate the
    /// evaluator for that tree.
    pub fn update_params(&self, node: &SdfNode) {
        match &self.engine {
            Engine::Wgpu(w) => w.update_params(node),
            #[cfg(all(target_os = "macos", feature = "msl"))]
            Engine::Metal(m) => m.update_params(node),
        }
    }

    /// Evaluate the SDF at multiple points, returning both distance and normal
    ///
    /// Uses GPU-side Tetrahedral Method for normal estimation (4 SDF evaluations).
    /// Output is `(distance, normal)` per point.
    ///
    /// # Panics
    /// Panics if the evaluator was not created with `new_dynamic()`.
    pub fn eval_batch_full(&self, points: &[Vec3]) -> Result<Vec<(f32, Vec3)>, GpuError> {
        match &self.engine {
            Engine::Wgpu(w) => w.eval_batch_full(points),
            #[cfg(all(target_os = "macos", feature = "msl"))]
            Engine::Metal(m) => m.eval_batch_full(points),
        }
    }

    /// Package a batch for GPU evaluation and return a handle to run it
    ///
    /// Nothing is dispatched here: the batch is evaluated when the handle is
    /// consumed, by [`GpuEvalFuture::wait`] (blocking, `eval_batch`) or
    /// [`GpuEvalFuture::resolve`] (async, `eval_batch_async`). The handle
    /// borrows the evaluator, so the borrow checker keeps the evaluator alive
    /// until the handle is consumed (until 5.0.0 the handle held a raw pointer
    /// and dropping the evaluator first was a use-after-free in safe code).
    ///
    /// # Example
    /// ```no_run
    /// # use alice_sdf::prelude::*;
    /// # use alice_sdf::compiled::GpuEvaluator;
    /// let gpu = GpuEvaluator::new(&SdfNode::sphere(1.0)).unwrap();
    /// let future = gpu.eval_batch_submit(vec![Vec3::ZERO, Vec3::X]);
    /// // Do other CPU work here...
    /// let distances = future.wait().unwrap();
    /// assert_eq!(distances.len(), 2);
    /// ```
    pub const fn eval_batch_submit(&self, points: Vec<Vec3>) -> GpuEvalFuture<'_> {
        GpuEvalFuture::new(self, points)
    }
}

/// The buffers behind a [`GpuBufferPool`], on the backend of the evaluator that made it.
enum PoolBuffers {
    #[cfg_attr(
        all(target_os = "macos", feature = "msl", not(feature = "glsl")),
        allow(
            dead_code,
            reason = "on macOS with msl only from_glsl_compute builds the wgpu engine"
        )
    )]
    Wgpu(WgpuPool),
    #[cfg(all(target_os = "macos", feature = "msl"))]
    Metal(super::gpu_metal::MetalPool),
}

/// Persistent GPU buffer pool for repeated evaluations
///
/// Avoids the overhead of creating and destroying GPU buffers for every
/// `eval_batch` call. Buffers are allocated once and reused, only growing
/// when the batch size exceeds current capacity.
///
/// # Performance
///
/// Buffer creation is one of the most expensive GPU operations.
/// By reusing buffers, `eval_batch_pooled` can be 2-5x faster than
/// `eval_batch` for repeated calls with similar-sized batches.
pub struct GpuBufferPool {
    /// Current capacity (number of points)
    pub capacity: usize,
    buffers: PoolBuffers,
}

impl GpuBufferPool {
    fn from_wgpu(pool: WgpuPool) -> Self {
        Self {
            capacity: pool.capacity,
            buffers: PoolBuffers::Wgpu(pool),
        }
    }

    #[cfg(all(target_os = "macos", feature = "msl"))]
    const fn from_metal(pool: super::gpu_metal::MetalPool) -> Self {
        Self {
            capacity: pool.capacity(),
            buffers: PoolBuffers::Metal(pool),
        }
    }

    /// Mirror the backend pool's capacity after it may have grown.
    fn sync_capacity(&mut self) {
        self.capacity = match &self.buffers {
            PoolBuffers::Wgpu(p) => p.capacity,
            #[cfg(all(target_os = "macos", feature = "msl"))]
            PoolBuffers::Metal(p) => p.capacity(),
        };
    }
}

impl std::fmt::Debug for GpuBufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBufferPool")
            .field("capacity", &self.capacity)
            .field(
                "input_bytes",
                &(self.capacity * std::mem::size_of::<GpuInputPoint>()),
            )
            .field(
                "output_bytes",
                &(self.capacity * std::mem::size_of::<GpuOutputDistance>()),
            )
            .finish_non_exhaustive()
    }
}

/// Future for GPU evaluation result
///
/// Allows CPU work while GPU computes. The handle borrows the
/// [`GpuEvaluator`] that made it, so the evaluator cannot be dropped or moved
/// while the handle exists:
///
/// ```compile_fail,E0505
/// # use alice_sdf::prelude::*;
/// # use alice_sdf::compiled::GpuEvaluator;
/// let gpu = GpuEvaluator::new(&SdfNode::sphere(1.0)).unwrap();
/// let future = gpu.eval_batch_submit(vec![Vec3::ZERO]);
/// drop(gpu); // error[E0505]: cannot move out of `gpu` because it is borrowed
/// let _ = future.wait();
/// ```
///
/// Consuming the handle first and dropping the evaluator afterwards compiles:
///
/// ```no_run
/// # use alice_sdf::prelude::*;
/// # use alice_sdf::compiled::GpuEvaluator;
/// let gpu = GpuEvaluator::new(&SdfNode::sphere(1.0)).unwrap();
/// let future = gpu.eval_batch_submit(vec![Vec3::ZERO]);
/// let _ = future.wait();
/// drop(gpu);
/// ```
///
/// Until 5.0.0 this type had no lifetime and held a raw pointer, so the first
/// snippet compiled and `wait` read freed memory.
#[must_use = "the batch is evaluated only when the handle is consumed by `wait` or `resolve`"]
pub struct GpuEvalFuture<'a> {
    evaluator: &'a GpuEvaluator,
    points: Vec<Vec3>,
}

impl<'a> GpuEvalFuture<'a> {
    const fn new(evaluator: &'a GpuEvaluator, points: Vec<Vec3>) -> Self {
        Self { evaluator, points }
    }

    /// Wait for the GPU result (blocking)
    ///
    /// # Errors
    ///
    /// The errors of [`GpuEvaluator::eval_batch`].
    pub fn wait(self) -> Result<Vec<f32>, GpuError> {
        self.evaluator.eval_batch(&self.points)
    }

    /// Get the GPU result asynchronously
    ///
    /// # Errors
    ///
    /// The errors of [`GpuEvaluator::eval_batch_async`].
    pub async fn resolve(self) -> Result<Vec<f32>, GpuError> {
        self.evaluator.eval_batch_async(&self.points).await
    }
}

impl std::fmt::Debug for GpuEvalFuture<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuEvalFuture")
            .field("points", &self.points.len())
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for GpuEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuEvaluator")
            .field("device", &self.device_info())
            .finish()
    }
}
#[cfg(test)]
mod dispatch_limit_tests {
    use super::*;

    #[test]
    fn a_dispatch_past_the_limit_is_an_error() {
        assert_eq!(dispatch_count(65_535, 1, 65_535).ok(), Some(65_535));
        assert_eq!(dispatch_count(0, 64, 65_535).ok(), Some(0));
        assert_eq!(dispatch_count(65, 64, 65_535).ok(), Some(2));
        assert_eq!(dispatch_count(131_070, 2, 65_535).ok(), Some(65_535));
        assert_eq!(dispatch_count(10, 0, 65_535).ok(), Some(10));
        for (points, size) in [(65_536, 1), (131_071, 2), (usize::MAX, 1)] {
            assert!(matches!(
                dispatch_count(points, size, 65_535),
                Err(GpuError::DispatchTooLarge { limit: 65_535, .. })
            ));
        }
    }
}

#[allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_engine_whose_dispatch_did_not_finish_refuses_further_work() {
        if !crate::gpu_test_support::gpu_ready("an_engine_whose_dispatch_did_not_finish") {
            return;
        }
        let gpu = GpuEvaluator::new(&SdfNode::Sphere { radius: 1.0 }).unwrap();
        #[allow(
            irrefutable_let_patterns,
            reason = "the Metal engine exists on one target only"
        )]
        let Engine::Wgpu(w) = &gpu.engine
        else {
            return;
        };
        assert!(gpu.eval_batch(&[Vec3::ZERO]).is_ok());
        assert!(w.check_finished(&[0; 4], 0, 7).is_err());
        let e = gpu.eval_batch(&[Vec3::ZERO]).unwrap_err().to_string();
        assert!(e.contains("not usable"), "{e}");
    }

    #[test]
    fn a_read_needs_its_own_dispatch_marker() {
        let bytes = |m: u32| {
            let mut v = vec![0u8; 8];
            v.extend_from_slice(&m.to_ne_bytes());
            v
        };
        assert!(dispatch_finished(&bytes(7), 8, 7).is_ok());
        // the zeros of work that did not finish
        let e = dispatch_finished(&bytes(0), 8, 7).unwrap_err().to_string();
        assert!(e.contains("did not finish"), "{e}");
        // the last dispatch's marker in a reused staging buffer
        assert!(dispatch_finished(&bytes(6), 8, 7).is_err());
        // no room for a marker
        assert!(dispatch_finished(&bytes(7)[..10], 8, 7).is_err());
    }

    use crate::eval::eval;

    /// A refused text starts no engine build (so no device is requested): the verdict comes from
    /// the leading `//` block alone. Without a GPU every other result of these constructors would
    /// be a device error, so the count is the only way to see the order on a machine with one
    #[test]
    fn verdict_runs_before_any_device_request() {
        let own = WgslShader::transpile(
            &SdfNode::sphere(1.0),
            crate::compiled::wgsl::TranspileMode::Hardcoded,
        )
        .to_compute_shader();
        let key = format!("// {}=", crate::provenance::KEY);
        let untagged: String = own
            .lines()
            .filter(|l| !l.starts_with(&key))
            .flat_map(|l| [l, "\n"])
            .collect();
        let other = format!(
            "// {}={}\n{untagged}",
            crate::provenance::KEY,
            crate::provenance::Provenance::new([0x5a; 32]).hex()
        );
        let before = ENGINE_BUILDS.with(std::cell::Cell::get);
        assert!(matches!(
            GpuEvaluator::from_wgsl(&other),
            Err(GpuError::SemanticsMismatch { .. })
        ));
        assert!(matches!(
            GpuEvaluator::from_wgsl_pinned(&untagged),
            Err(GpuError::Unpinned)
        ));
        assert!(matches!(
            pollster::block_on(GpuEvaluator::from_wgsl_async(&other)),
            Err(GpuError::SemanticsMismatch { .. })
        ));
        assert!(matches!(
            pollster::block_on(GpuEvaluator::from_wgsl_async_pinned(&untagged)),
            Err(GpuError::Unpinned)
        ));
        #[cfg(feature = "glsl")]
        {
            assert!(matches!(
                GpuEvaluator::from_glsl_compute(&other),
                Err(GpuError::SemanticsMismatch { .. })
            ));
            assert!(matches!(
                GpuEvaluator::from_glsl_compute_pinned(&untagged),
                Err(GpuError::Unpinned)
            ));
        }
        assert_eq!(
            ENGINE_BUILDS.with(std::cell::Cell::get),
            before,
            "a refused text started an engine build"
        );
        // the counter itself counts: an accepted text starts one (whatever the device answers)
        let _ = GpuEvaluator::from_wgsl(&own);
        assert_eq!(ENGINE_BUILDS.with(std::cell::Cell::get), before + 1);
    }

    #[test]
    fn test_gpu_eval_sphere() {
        if !crate::gpu_test_support::gpu_ready("test_gpu_eval_sphere") {
            return;
        }

        let sphere = SdfNode::Sphere { radius: 1.0 };
        let gpu = GpuEvaluator::new(&sphere).unwrap();

        let points = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];

        let gpu_results = gpu.eval_batch(&points).unwrap();

        for (i, p) in points.iter().enumerate() {
            let cpu_result = eval(&sphere, *p);
            assert!(
                (gpu_results[i] - cpu_result).abs() < 0.001,
                "Mismatch at {:?}: gpu={}, cpu={}",
                p,
                gpu_results[i],
                cpu_result
            );
        }
    }

    #[test]
    fn test_gpu_eval_complex() {
        if !crate::gpu_test_support::gpu_ready("test_gpu_eval_complex") {
            return;
        }

        let shape = SdfNode::Sphere { radius: 1.0 }.union(
            SdfNode::Box3d {
                half_extents: Vec3::splat(0.5),
            }
            .translate(2.0, 0.0, 0.0),
        );

        let gpu = GpuEvaluator::new(&shape).unwrap();

        let points: Vec<Vec3> = (0..100)
            .map(|i| Vec3::new((i as f32).mul_add(0.05, -2.5), 0.0, 0.0))
            .collect();

        let gpu_results = gpu.eval_batch(&points).unwrap();

        for (i, p) in points.iter().enumerate() {
            let cpu_result = eval(&shape, *p);
            assert!(
                (gpu_results[i] - cpu_result).abs() < 0.01,
                "Mismatch at {:?}: gpu={}, cpu={}",
                p,
                gpu_results[i],
                cpu_result
            );
        }
    }

    #[test]
    fn test_gpu_eval_large_batch() {
        if !crate::gpu_test_support::gpu_ready("test_gpu_eval_large_batch") {
            return;
        }

        let sphere = SdfNode::Sphere { radius: 1.0 };
        // the field size is part of the entry in tests/test_gpu_submit_limit.rs for this many points
        assert_eq!(sphere.node_count(), 1);
        let gpu = GpuEvaluator::new(&sphere).unwrap();

        // Test with a large batch
        let points: Vec<Vec3> = (0..10000)
            .map(|i| {
                let t = i as f32 / 10000.0;
                Vec3::new(t * 4.0 - 2.0, (t * 10.0).sin(), (t * 10.0).cos())
            })
            .collect();

        let gpu_results = gpu.eval_batch(&points).unwrap();
        assert_eq!(gpu_results.len(), 10000);

        // Spot check a few results
        for i in [0, 1000, 5000, 9999] {
            let cpu_result = eval(&sphere, points[i]);
            assert!(
                (gpu_results[i] - cpu_result).abs() < 0.01,
                "Mismatch at index {}: gpu={}, cpu={}",
                i,
                gpu_results[i],
                cpu_result
            );
        }
    }
}
