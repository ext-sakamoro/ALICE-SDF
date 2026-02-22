//! GPU Evaluator: WebGPU-based SDF Evaluation (Deep Fried Edition)
//!
//! This module provides GPU-accelerated SDF evaluation using wgpu.
//! It can evaluate millions of points in parallel on the GPU.
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
use crate::types::SdfNode;

/// Error type for GPU evaluation
#[derive(Error, Debug)]
pub enum GpuError {
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
}

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
        GpuInputPoint {
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

/// GPU-based SDF evaluator
///
/// Uses WebGPU compute shaders to evaluate SDF at many points in parallel.
pub struct GpuEvaluator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Dynamic parameter buffer (present in Dynamic mode)
    param_buffer: Option<wgpu::Buffer>,
    /// Distance + normals pipeline (present in Dynamic mode)
    full_pipeline: Option<wgpu::ComputePipeline>,
    /// Bind group layout for 4-binding pipelines
    dynamic_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// [Deep Fried v2] Workgroup size used in shader (matches dispatch calculation)
    workgroup_size: u32,
}

impl GpuEvaluator {
    /// Create a new GPU evaluator for the given SDF
    ///
    /// This compiles the SDF to a WGSL compute shader and creates
    /// the necessary GPU resources.
    pub fn new(node: &SdfNode) -> Result<Self, GpuError> {
        let shader = WgslShader::transpile(node, super::transpiler::TranspileMode::Hardcoded);
        Self::from_shader(&shader)
    }

    /// Create a GPU evaluator from a pre-generated WGSL shader
    pub fn from_shader(shader: &WgslShader) -> Result<Self, GpuError> {
        let compute_shader = shader.to_compute_shader();
        let mut evaluator = Self::from_wgsl(&compute_shader)?;
        evaluator.workgroup_size = shader.workgroup_size;
        Ok(evaluator)
    }

    /// Create a GPU evaluator from raw WGSL source
    pub fn from_wgsl(wgsl_source: &str) -> Result<Self, GpuError> {
        // Initialize wgpu synchronously using pollster
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(GpuError::NoAdapter)?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ALICE-SDF Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;

        // Create shader module
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ALICE-SDF Shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ALICE-SDF Bind Group Layout"),
            entries: &[
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

        Ok(GpuEvaluator {
            device,
            queue,
            pipeline,
            bind_group_layout,
            param_buffer: None,
            full_pipeline: None,
            dynamic_bind_group_layout: None,
            workgroup_size: 256,
        })
    }

    /// Evaluate the SDF at multiple points using the GPU
    ///
    /// This is highly efficient for large numbers of points (1000+).
    /// For small numbers of points, CPU evaluation may be faster due to
    /// GPU dispatch overhead.
    pub fn eval_batch(&self, points: &[Vec3]) -> Result<Vec<f32>, GpuError> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        let point_count = points.len();

        // Convert input points
        let input_data: Vec<GpuInputPoint> = points.iter().map(|&p| p.into()).collect();

        // Create input buffer
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input Points Buffer"),
                contents: bytemuck::cast_slice(&input_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create output buffer
        let output_buffer_size = (point_count * std::mem::size_of::<GpuOutputDistance>()) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Distances Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create uniform buffer for point count
        let count_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Point Count Buffer"),
                contents: bytemuck::cast_slice(&[point_count as u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ALICE-SDF Bind Group"),
            layout: &self.bind_group_layout,
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
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
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

            // Calculate workgroup count (adaptive threads per workgroup)
            let wg = self.workgroup_size;
            let workgroup_count = (point_count as u32).div_ceil(wg);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);

        // Submit commands
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|_| GpuError::BufferMapping("GPU channel closed unexpectedly".to_string()))?
            .map_err(|e| GpuError::BufferMapping(format!("{:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let output: &[GpuOutputDistance] = bytemuck::cast_slice(&data);
        let distances: Vec<f32> = output.iter().map(|d| d.distance).collect();

        drop(data);
        staging_buffer.unmap();

        Ok(distances)
    }

    /// Get information about the GPU being used
    pub fn device_info(&self) -> String {
        "WebGPU Device (via wgpu)".to_string()
    }

    /// Create a persistent buffer pool for repeated evaluations
    ///
    /// Avoids buffer allocation overhead when calling `eval_batch` many times.
    /// The pool reuses GPU buffers, only re-allocating when capacity is exceeded.
    pub fn create_buffer_pool(&self, initial_capacity: usize) -> GpuBufferPool {
        GpuBufferPool::new(&self.device, initial_capacity)
    }

    /// Evaluate using a persistent buffer pool (avoids re-allocation)
    ///
    /// This is significantly faster than `eval_batch` for repeated calls
    /// with similar-sized point batches (e.g., marching cubes, raymarching).
    pub fn eval_batch_pooled(
        &self,
        points: &[Vec3],
        pool: &mut GpuBufferPool,
    ) -> Result<Vec<f32>, GpuError> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        let point_count = points.len();

        // Ensure pool has enough capacity
        pool.ensure_capacity(&self.device, point_count);

        // Write input data
        let input_data: Vec<GpuInputPoint> = points.iter().map(|&p| p.into()).collect();
        self.queue
            .write_buffer(&pool.input_buffer, 0, bytemuck::cast_slice(&input_data));

        // Write point count
        self.queue.write_buffer(
            &pool.count_buffer,
            0,
            bytemuck::cast_slice(&[point_count as u32]),
        );

        // Create bind group (lightweight, references existing buffers)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ALICE-SDF Pooled Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pool.input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pool.output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pool.count_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let mut encoder = self
            .device
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

            let workgroup_count = (point_count as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        let read_size = (point_count * std::mem::size_of::<GpuOutputDistance>()) as u64;
        encoder.copy_buffer_to_buffer(&pool.output_buffer, 0, &pool.staging_buffer, 0, read_size);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read results
        let buffer_slice = pool.staging_buffer.slice(..read_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|_| GpuError::BufferMapping("GPU channel closed unexpectedly".to_string()))?
            .map_err(|e| GpuError::BufferMapping(format!("{:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let output: &[GpuOutputDistance] = bytemuck::cast_slice(&data);
        let distances: Vec<f32> = output[..point_count].iter().map(|d| d.distance).collect();

        drop(data);
        pool.staging_buffer.unmap();

        Ok(distances)
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
        let compute_shader = shader.to_compute_shader();
        let mut evaluator = Self::from_wgsl_async(&compute_shader).await?;
        evaluator.workgroup_size = shader.workgroup_size;
        Ok(evaluator)
    }

    /// Create a GPU evaluator from raw WGSL source (async)
    pub async fn from_wgsl_async(wgsl_source: &str) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ALICE-SDF Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;

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

        Ok(GpuEvaluator {
            device,
            queue,
            pipeline,
            bind_group_layout,
            param_buffer: None,
            full_pipeline: None,
            dynamic_bind_group_layout: None,
            workgroup_size: 256,
        })
    }

    /// Evaluate the SDF at multiple points using the GPU (async)
    ///
    /// This is the async version that allows CPU work while GPU computes.
    /// Use this for optimal CPU/GPU overlap in real-time applications.
    pub async fn eval_batch_async(&self, points: &[Vec3]) -> Result<Vec<f32>, GpuError> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        let point_count = points.len();

        // Convert input points
        let input_data: Vec<GpuInputPoint> = points.iter().map(|&p| p.into()).collect();

        // Create input buffer
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input Points Buffer"),
                contents: bytemuck::cast_slice(&input_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create output buffer
        let output_buffer_size = (point_count * std::mem::size_of::<GpuOutputDistance>()) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Distances Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create uniform buffer for point count
        let count_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Point Count Buffer"),
                contents: bytemuck::cast_slice(&[point_count as u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ALICE-SDF Bind Group"),
            layout: &self.bind_group_layout,
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
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
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

            let workgroup_count = (point_count as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);

        // Submit commands (non-blocking)
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map buffer asynchronously
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Poll device until mapping is complete
        self.device.poll(wgpu::Maintain::Wait);

        // Wait for mapping result
        receiver
            .await
            .map_err(|_| GpuError::BufferMapping("Channel closed".to_string()))?
            .map_err(|e| GpuError::BufferMapping(format!("{:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let output: &[GpuOutputDistance] = bytemuck::cast_slice(&data);
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
    pub fn new_dynamic(node: &SdfNode) -> Result<Self, GpuError> {
        use super::transpiler::TranspileMode;
        let shader = WgslShader::transpile(node, TranspileMode::Dynamic);

        // Build distance-only compute shader (4 bindings)
        let compute_source = shader.to_compute_shader();
        // Build distance+normals compute shader (4 bindings)
        let full_source = shader.to_compute_shader_with_normals();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(GpuError::NoAdapter)?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ALICE-SDF Device (Dynamic)"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;

        // 4-binding layout (shared by both pipelines)
        let dynamic_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ALICE-SDF Dynamic Bind Group Layout"),
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
                ],
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

        // Create parameter buffer with initial values
        let mut param_data = shader.param_layout.clone();
        while param_data.len() % 4 != 0 {
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

        Ok(GpuEvaluator {
            device,
            queue,
            pipeline,
            bind_group_layout,
            param_buffer: Some(param_buffer),
            full_pipeline: Some(full_pipeline),
            dynamic_bind_group_layout: Some(dynamic_bind_group_layout),
            workgroup_size: shader.workgroup_size,
        })
    }

    /// Update SDF parameters without recompiling shaders
    ///
    /// Traverses the node tree in the same order as transpilation to extract
    /// current values, then uploads them to the GPU via `write_buffer`.
    ///
    /// This is a zero-latency operation suitable for 60fps animation.
    ///
    /// # Panics
    /// Panics if the evaluator was not created with `new_dynamic()`.
    pub fn update_params(&self, node: &SdfNode) {
        let params = WgslShader::extract_params(node);
        let mut padded = params;
        while padded.len() % 4 != 0 {
            padded.push(0.0);
        }
        padded.resize(PARAM_BUFFER_SIZE as usize / 4, 0.0);
        self.queue.write_buffer(
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
    pub fn eval_batch_full(&self, points: &[Vec3]) -> Result<Vec<(f32, Vec3)>, GpuError> {
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

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input Points Buffer (Full)"),
                contents: bytemuck::cast_slice(&input_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buffer_size = (point_count * std::mem::size_of::<GpuOutputFull>()) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Full Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer (Full)"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Point Count Buffer"),
                contents: bytemuck::cast_slice(&[point_count as u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            ],
        });

        let mut encoder = self
            .device
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
            let workgroup_count = (point_count as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|_| GpuError::BufferMapping("GPU channel closed unexpectedly".to_string()))?
            .map_err(|e| GpuError::BufferMapping(format!("{:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let output: &[GpuOutputFull] = bytemuck::cast_slice(&data);
        let results: Vec<(f32, Vec3)> = output
            .iter()
            .map(|o| (o.dist, Vec3::new(o.nx, o.ny, o.nz)))
            .collect();

        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }

    /// Submit a batch for GPU evaluation and return a future
    ///
    /// This is the most flexible async API - it submits the work to the GPU
    /// and returns immediately, allowing you to do other work while waiting.
    ///
    /// # Example
    /// ```ignore
    /// let future = gpu.eval_batch_submit(points);
    /// // Do other CPU work here...
    /// let distances = future.await?;
    /// ```
    pub fn eval_batch_submit(&self, points: Vec<Vec3>) -> GpuEvalFuture {
        GpuEvalFuture::new(self, points)
    }
}

/// Auto-tune threshold: below this, single dispatch is optimal
const AUTO_TUNE_THRESHOLD: usize = 262144; // 256K points

/// Determine optimal chunk size for large batches
///
/// Balances GPU occupancy against buffer transfer overhead.
/// Larger chunks = fewer dispatches but more latency per chunk.
fn optimal_chunk_size(total_points: usize) -> usize {
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
pub struct GpuBufferPool {
    /// Current capacity (number of points)
    pub capacity: usize,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    count_buffer: wgpu::Buffer,
}

impl GpuBufferPool {
    /// Create a new buffer pool with the given initial capacity
    fn new(device: &wgpu::Device, capacity: usize) -> Self {
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
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pool Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pool Count Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        GpuBufferPool {
            capacity: cap,
            input_buffer,
            output_buffer,
            staging_buffer,
            count_buffer,
        }
    }

    /// Ensure the pool can hold at least `required` points
    ///
    /// If the current capacity is insufficient, new larger buffers are allocated.
    /// Uses 1.5x growth factor to amortize allocation cost.
    fn ensure_capacity(&mut self, device: &wgpu::Device, required: usize) {
        if required <= self.capacity {
            return;
        }

        // Grow by 1.5x or to required size, whichever is larger
        let new_cap = (self.capacity * 3 / 2).max(required);
        *self = Self::new(device, new_cap);
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
/// Allows CPU work while GPU computes.
pub struct GpuEvalFuture {
    evaluator: *const GpuEvaluator,
    points: Vec<Vec3>,
}

// SAFETY: GpuEvalFuture holds a raw pointer to GpuEvaluator, which itself
// contains only wgpu handles (Device, Queue, Pipeline) that are Send + Sync.
// The pointer is derived from a shared reference and is only dereferenced in
// `wait()` / `resolve()`, where the evaluator is guaranteed to outlive the future
// (it is created from `&GpuEvaluator` with the same lifetime scope).
unsafe impl Send for GpuEvalFuture {}
unsafe impl Sync for GpuEvalFuture {}

impl GpuEvalFuture {
    fn new(evaluator: &GpuEvaluator, points: Vec<Vec3>) -> Self {
        GpuEvalFuture {
            evaluator: evaluator as *const _,
            points,
        }
    }

    /// Wait for the GPU result (blocking)
    pub fn wait(self) -> Result<Vec<f32>, GpuError> {
        // SAFETY: The evaluator pointer was created from a valid `&GpuEvaluator`
        // reference in `eval_batch_submit()`. The caller must ensure the
        // GpuEvaluator outlives this future, which is the documented contract.
        let evaluator = unsafe { &*self.evaluator };
        evaluator.eval_batch(&self.points)
    }

    /// Get the GPU result asynchronously
    pub async fn resolve(self) -> Result<Vec<f32>, GpuError> {
        // SAFETY: The evaluator pointer was created from a valid `&GpuEvaluator`
        // reference in `eval_batch_submit()`. The caller must ensure the
        // GpuEvaluator outlives this future, which is the documented contract.
        let evaluator = unsafe { &*self.evaluator };
        evaluator.eval_batch_async(&self.points).await
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
mod tests {
    use super::*;
    use crate::eval::eval;

    fn has_gpu() -> bool {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .is_some()
    }

    #[test]
    fn test_gpu_eval_sphere() {
        if !has_gpu() {
            eprintln!("Skipping GPU test: no GPU available");
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
        if !has_gpu() {
            eprintln!("Skipping GPU test: no GPU available");
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
            .map(|i| Vec3::new(i as f32 * 0.05 - 2.5, 0.0, 0.0))
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
        if !has_gpu() {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        }

        let sphere = SdfNode::Sphere { radius: 1.0 };
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
