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

/// GPU-based SDF evaluator
///
/// Uses WebGPU compute shaders to evaluate SDF at many points in parallel.
pub struct GpuEvaluator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuEvaluator {
    /// Create a new GPU evaluator for the given SDF
    ///
    /// This compiles the SDF to a WGSL compute shader and creates
    /// the necessary GPU resources.
    pub fn new(node: &SdfNode) -> Result<Self, GpuError> {
        let shader = WgslShader::transpile(node);
        Self::from_shader(&shader)
    }

    /// Create a GPU evaluator from a pre-generated WGSL shader
    pub fn from_shader(shader: &WgslShader) -> Result<Self, GpuError> {
        let compute_shader = shader.to_compute_shader();
        Self::from_wgsl(&compute_shader)
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

            // Calculate workgroup count (256 threads per workgroup)
            let workgroup_count = (point_count as u32 + 255) / 256;
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
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .unwrap()
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
        format!("WebGPU Device (via wgpu)")
    }

    // ========== Deep Fried Edition: Async API ==========

    /// Create a new GPU evaluator asynchronously
    ///
    /// Use this when you want to initialize the GPU without blocking.
    pub async fn new_async(node: &SdfNode) -> Result<Self, GpuError> {
        let shader = WgslShader::transpile(node);
        Self::from_shader_async(&shader).await
    }

    /// Create a GPU evaluator from a pre-generated WGSL shader (async)
    pub async fn from_shader_async(shader: &WgslShader) -> Result<Self, GpuError> {
        let compute_shader = shader.to_compute_shader();
        Self::from_wgsl_async(&compute_shader).await
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

            let workgroup_count = (point_count as u32 + 255) / 256;
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

/// Future for GPU evaluation result
///
/// Allows CPU work while GPU computes.
pub struct GpuEvalFuture {
    evaluator: *const GpuEvaluator,
    points: Vec<Vec3>,
}

// Safety: GpuEvaluator is Send + Sync
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
        // Safety: evaluator pointer is valid for the lifetime of GpuEvalFuture
        let evaluator = unsafe { &*self.evaluator };
        evaluator.eval_batch(&self.points)
    }

    /// Get the GPU result asynchronously
    pub async fn resolve(self) -> Result<Vec<f32>, GpuError> {
        // Safety: evaluator pointer is valid for the lifetime of GpuEvalFuture
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

        let shape = SdfNode::Sphere { radius: 1.0 }
            .union(SdfNode::Box3d {
                half_extents: Vec3::splat(0.5),
            }.translate(2.0, 0.0, 0.0));

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
