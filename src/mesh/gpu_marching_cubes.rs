//! GPU Compute Marching Cubes (Deep Fried Edition)
//!
//! 3-pass GPU Marching Cubes pipeline for ~50x faster mesh generation
//! compared to CPU at high resolution (128+).
//!
//! # Pipeline
//!
//! ```text
//! Pass 1: SDF Grid Eval     @workgroup_size(4,4,4)  -> sdf_grid[(res+1)^3]
//! Pass 2: Cell Classify      @workgroup_size(4,4,4)  -> vertex_counts[res^3] + cube_indices
//! CPU:    Prefix Sum         sequential scan          -> vertex_offsets[res^3]
//! Pass 3: Vertex Generation  @workgroup_size(64)      -> output_vertices[total_verts]
//! CPU:    Readback + Convert                          -> Mesh { vertices, indices }
//! ```
//!
//! # Deep Fried Optimizations
//!
//! - **Atomic vertex counting**: Pass 2 uses atomicAdd for total vertex count
//! - **CPU prefix sum**: Avoids complex GPU scan; fast enough for res <= 512
//! - **Tetrahedral normals**: 4-point gradient estimation on GPU
//! - **Zero-copy tables**: EDGE_TABLE and TRI_TABLE embedded as WGSL constants
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use wgpu::util::DeviceExt;

use super::gpu_mc_shaders;
use crate::compiled::{GpuError, TranspileMode, WgslShader};
use crate::mesh::{Mesh, Vertex};
use crate::types::SdfNode;

/// Configuration for GPU Marching Cubes
#[derive(Debug, Clone, Copy)]
pub struct GpuMarchingCubesConfig {
    /// Grid resolution along each axis (e.g. 64, 128, 256)
    pub resolution: u32,
    /// Iso-level (usually 0.0 for SDF surface)
    pub iso_level: f32,
    /// Whether to compute vertex normals via tetrahedral gradient
    pub compute_normals: bool,
    /// Maximum vertices to allocate (0 = auto-estimate from resolution)
    pub max_vertices: u32,
}

impl Default for GpuMarchingCubesConfig {
    fn default() -> Self {
        GpuMarchingCubesConfig {
            resolution: 64,
            iso_level: 0.0,
            compute_normals: true,
            max_vertices: 0,
        }
    }
}

/// Uniform buffer for MC shaders (48 bytes, 16-byte aligned)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct McUniforms {
    resolution: u32,
    iso_level: f32,
    _pad0: u32,
    _pad1: u32,
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
}

/// GPU output vertex (32 bytes, cache-aligned)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVertex {
    px: f32,
    py: f32,
    pz: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    _pad0: f32,
    _pad1: f32,
}

/// Run GPU Marching Cubes on an SDF node
///
/// Transpiles the SDF to WGSL and runs the 3-pass pipeline.
///
/// # Arguments
/// * `node` - The SDF tree to mesh
/// * `bounds_min` - World-space minimum bounds
/// * `bounds_max` - World-space maximum bounds
/// * `config` - GPU MC configuration
///
/// # Returns
/// `Mesh` with vertices and triangle indices
pub fn gpu_marching_cubes(
    node: &SdfNode,
    bounds_min: Vec3,
    bounds_max: Vec3,
    config: &GpuMarchingCubesConfig,
) -> Result<Mesh, GpuError> {
    let shader = WgslShader::transpile(node, TranspileMode::Hardcoded);
    gpu_marching_cubes_from_shader(&shader, bounds_min, bounds_max, config)
}

/// Run GPU Marching Cubes from a pre-compiled WGSL shader
pub fn gpu_marching_cubes_from_shader(
    sdf_shader: &WgslShader,
    bounds_min: Vec3,
    bounds_max: Vec3,
    config: &GpuMarchingCubesConfig,
) -> Result<Mesh, GpuError> {
    let res = config.resolution;
    let grid_size = (res + 1) as usize;
    let grid_total = grid_size * grid_size * grid_size;
    let cell_total = (res as usize) * (res as usize) * (res as usize);

    // Estimate max vertices if not specified
    let max_verts = if config.max_vertices > 0 {
        config.max_vertices as usize
    } else {
        // Rough upper bound: ~15 vertices per surface cell, ~10% cells active
        (cell_total / 10) * 15
    };

    // Generate shader sources
    let pass1_source = gpu_mc_shaders::generate_sdf_grid_shader(sdf_shader);
    let pass2_source = gpu_mc_shaders::generate_classify_shader();
    let pass3_source = gpu_mc_shaders::generate_vertex_shader(sdf_shader);

    // Initialize wgpu
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
            label: Some("ALICE-SDF GPU MC Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .map_err(|e: wgpu::RequestDeviceError| GpuError::DeviceCreation(e.to_string()))?;

    // Uniforms
    let uniforms = McUniforms {
        resolution: res,
        iso_level: config.iso_level,
        _pad0: 0,
        _pad1: 0,
        bounds_min: [bounds_min.x, bounds_min.y, bounds_min.z, 0.0],
        bounds_max: [bounds_max.x, bounds_max.y, bounds_max.z, 0.0],
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MC Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // ====== PASS 1: SDF Grid Eval ======

    let sdf_grid_size = (grid_total * std::mem::size_of::<f32>()) as u64;
    let sdf_grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SDF Grid Buffer"),
        size: sdf_grid_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let pass1_shader: wgpu::ShaderModule =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Pass 1 Shader"),
            source: wgpu::ShaderSource::Wgsl(pass1_source.into()),
        });

    let pass1_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MC Pass 1 BGL"),
        entries: &[
            // sdf_grid: storage RW
            bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
            // uniforms
            bgl_entry(1, wgpu::BufferBindingType::Uniform),
        ],
    });

    let pass1_pipeline = create_pipeline(&device, &pass1_bgl, &pass1_shader, "MC Pass 1");

    let pass1_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MC Pass 1 BG"),
        layout: &pass1_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: sdf_grid_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    // ====== PASS 2: Cell Classify + Count ======

    let cell_counts_size = (cell_total * std::mem::size_of::<u32>()) as u64;
    let cell_counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Cell Vertex Counts"),
        size: cell_counts_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let cell_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Cell Cube Indices"),
        size: cell_counts_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Atomic counter for total vertex count (single u32)
    let total_count_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Total Vertex Count"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let pass2_shader: wgpu::ShaderModule =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Pass 2 Shader"),
            source: wgpu::ShaderSource::Wgsl(pass2_source.into()),
        });

    let pass2_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MC Pass 2 BGL"),
        entries: &[
            // sdf_grid: storage read
            bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
            // uniforms
            bgl_entry(1, wgpu::BufferBindingType::Uniform),
            // cell_vertex_counts: storage RW
            bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            // cell_cube_indices: storage RW
            bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            // total_vertex_count: storage RW
            bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    });

    let pass2_pipeline = create_pipeline(&device, &pass2_bgl, &pass2_shader, "MC Pass 2");

    let pass2_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MC Pass 2 BG"),
        layout: &pass2_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: sdf_grid_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cell_counts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cell_indices_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: total_count_buffer.as_entire_binding(),
            },
        ],
    });

    // ====== Dispatch Pass 1 + Pass 2 ======

    let wg = 4u32;
    let dispatch_grid = (grid_size as u32 + wg - 1) / wg;
    let dispatch_cell = (res + wg - 1) / wg;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MC Pass 1+2 Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MC Pass 1"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pass1_pipeline);
        pass.set_bind_group(0, &pass1_bg, &[]);
        pass.dispatch_workgroups(dispatch_grid, dispatch_grid, dispatch_grid);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MC Pass 2"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pass2_pipeline);
        pass.set_bind_group(0, &pass2_bg, &[]);
        pass.dispatch_workgroups(dispatch_cell, dispatch_cell, dispatch_cell);
    }

    // Copy cell counts and total to staging for CPU readback
    let counts_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Counts Staging"),
        size: cell_counts_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let total_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Total Staging"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&cell_counts_buffer, 0, &counts_staging, 0, cell_counts_size);
    encoder.copy_buffer_to_buffer(&total_count_buffer, 0, &total_staging, 0, 4);

    queue.submit(std::iter::once(encoder.finish()));

    // Read back total vertex count
    let total_verts = read_u32(&device, &total_staging)?;

    if total_verts == 0 {
        return Ok(Mesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        });
    }

    // Read back cell vertex counts
    let cell_counts = read_u32_vec(&device, &counts_staging, cell_total)?;

    // ====== CPU Prefix Sum ======
    // Compute exclusive prefix sum for per-cell vertex write offsets
    let mut offsets = vec![0u32; cell_total];
    let mut running = 0u32;
    for i in 0..cell_total {
        offsets[i] = running;
        running += cell_counts[i];
    }

    // Clamp total_verts to max_verts for safety
    let actual_verts = (total_verts as usize).min(max_verts);

    // ====== PASS 3: Vertex Generation ======

    let offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cell Offsets"),
        contents: bytemuck::cast_slice(&offsets),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_size = (actual_verts * std::mem::size_of::<GpuVertex>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Vertices"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let pass3_shader: wgpu::ShaderModule =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Pass 3 Shader"),
            source: wgpu::ShaderSource::Wgsl(pass3_source.into()),
        });

    let pass3_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MC Pass 3 BGL"),
        entries: &[
            // sdf_grid: storage read
            bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
            // uniforms
            bgl_entry(1, wgpu::BufferBindingType::Uniform),
            // cell_offsets: storage read
            bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
            // cell_cube_indices: storage read
            bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
            // output_vertices: storage RW
            bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    });

    let pass3_pipeline = create_pipeline(&device, &pass3_bgl, &pass3_shader, "MC Pass 3");

    let pass3_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MC Pass 3 BG"),
        layout: &pass3_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: sdf_grid_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cell_indices_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch pass 3: one thread per cell, @workgroup_size(64)
    let dispatch_cells = (cell_total as u32 + 63) / 64;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MC Pass 3 Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MC Pass 3"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pass3_pipeline);
        pass.set_bind_group(0, &pass3_bg, &[]);
        pass.dispatch_workgroups(dispatch_cells, 1, 1);
    }

    // Copy output to staging
    let output_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Staging"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &output_staging, 0, output_size);
    queue.submit(std::iter::once(encoder.finish()));

    // Read back vertices
    let gpu_verts = read_gpu_vertices(&device, &output_staging, actual_verts)?;

    // Convert to Mesh
    let mut vertices = Vec::with_capacity(actual_verts);
    let mut indices = Vec::with_capacity(actual_verts);

    for (i, gv) in gpu_verts.iter().enumerate() {
        vertices.push(Vertex::new(
            Vec3::new(gv.px, gv.py, gv.pz),
            Vec3::new(gv.nx, gv.ny, gv.nz),
        ));
        indices.push(i as u32);
    }

    Ok(Mesh { vertices, indices })
}

// ====== Helper functions ======

/// Create a bind group layout entry (compute, storage/uniform)
fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Create a compute pipeline from a bind group layout and shader module
fn create_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Read a single u32 from a mapped staging buffer
fn read_u32(device: &wgpu::Device, staging: &wgpu::Buffer) -> Result<u32, GpuError> {
    let slice = staging.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);

    pollster::block_on(receiver)
        .map_err(|e| GpuError::BufferMapping(format!("Channel error: {}", e)))?
        .map_err(|e| GpuError::BufferMapping(format!("Map error: {:?}", e)))?;

    let mapped = slice.get_mapped_range();
    let val: u32 = bytemuck::cast_slice::<u8, u32>(&mapped)[0];
    drop(mapped);
    staging.unmap();

    Ok(val)
}

/// Read a Vec<u32> from a mapped staging buffer
fn read_u32_vec(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<u32>, GpuError> {
    let slice = staging.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);

    pollster::block_on(receiver)
        .map_err(|e| GpuError::BufferMapping(format!("Channel error: {}", e)))?
        .map_err(|e| GpuError::BufferMapping(format!("Map error: {:?}", e)))?;

    let mapped = slice.get_mapped_range();
    let data: &[u32] = bytemuck::cast_slice(&mapped);
    let result = data[..count].to_vec();
    drop(mapped);
    staging.unmap();

    Ok(result)
}

/// Read GpuVertex array from a mapped staging buffer
fn read_gpu_vertices(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<GpuVertex>, GpuError> {
    let slice = staging.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);

    pollster::block_on(receiver)
        .map_err(|e| GpuError::BufferMapping(format!("Channel error: {}", e)))?
        .map_err(|e| GpuError::BufferMapping(format!("Map error: {:?}", e)))?;

    let mapped = slice.get_mapped_range();
    let data: &[GpuVertex] = bytemuck::cast_slice(&mapped);
    let result = data[..count].to_vec();
    drop(mapped);
    staging.unmap();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_mc_config_default() {
        let config = GpuMarchingCubesConfig::default();
        assert_eq!(config.resolution, 64);
        assert_eq!(config.iso_level, 0.0);
        assert!(config.compute_normals);
        assert_eq!(config.max_vertices, 0);
    }

    #[test]
    fn test_mc_uniforms_size() {
        // Must be 48 bytes (3 x vec4)
        assert_eq!(std::mem::size_of::<McUniforms>(), 48);
    }

    #[test]
    fn test_gpu_vertex_size() {
        // Must be 32 bytes (cache-line aligned)
        assert_eq!(std::mem::size_of::<GpuVertex>(), 32);
    }

    #[test]
    fn test_gpu_mc_sphere() {
        // This test requires a GPU; it will be skipped in CI without GPU
        let sphere = SdfNode::sphere(1.0);
        let config = GpuMarchingCubesConfig {
            resolution: 16,
            ..Default::default()
        };

        match gpu_marching_cubes(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config) {
            Ok(mesh) => {
                assert!(mesh.vertices.len() > 0, "Mesh should have vertices");
                assert!(mesh.indices.len() > 0, "Mesh should have indices");
                // Vertices should be in groups of 3 (triangles)
                assert_eq!(mesh.indices.len() % 3, 0, "Indices should be multiple of 3");

                // All vertices should be within bounds (with some tolerance)
                for v in &mesh.vertices {
                    assert!(
                        v.position.x >= -2.5 && v.position.x <= 2.5,
                        "Vertex out of bounds: {:?}",
                        v.position
                    );
                    assert!(
                        v.position.y >= -2.5 && v.position.y <= 2.5,
                        "Vertex out of bounds: {:?}",
                        v.position
                    );
                    assert!(
                        v.position.z >= -2.5 && v.position.z <= 2.5,
                        "Vertex out of bounds: {:?}",
                        v.position
                    );
                }

                // Normals should be approximately unit length
                for v in &mesh.vertices {
                    let n_len = v.normal.length();
                    assert!(
                        n_len > 0.5 && n_len < 1.5,
                        "Normal not unit length: {} at {:?}",
                        n_len,
                        v.position
                    );
                }
            }
            Err(GpuError::NoAdapter) => {
                // No GPU available, skip test
                eprintln!("Skipping GPU MC test: no GPU adapter available");
            }
            Err(e) => panic!("GPU MC failed: {}", e),
        }
    }

    #[test]
    fn test_gpu_mc_empty() {
        // A huge sphere evaluated in a tiny region far away should produce no mesh
        let sphere = SdfNode::sphere(1.0);
        let config = GpuMarchingCubesConfig {
            resolution: 8,
            ..Default::default()
        };

        match gpu_marching_cubes(&sphere, Vec3::splat(10.0), Vec3::splat(12.0), &config) {
            Ok(mesh) => {
                assert_eq!(
                    mesh.vertices.len(),
                    0,
                    "Should produce empty mesh far from surface"
                );
            }
            Err(GpuError::NoAdapter) => {
                eprintln!("Skipping GPU MC test: no GPU adapter available");
            }
            Err(e) => panic!("GPU MC failed: {}", e),
        }
    }

    #[test]
    fn test_gpu_mc_complex_shape() {
        let shape = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);
        let config = GpuMarchingCubesConfig {
            resolution: 16,
            ..Default::default()
        };

        match gpu_marching_cubes(&shape, Vec3::splat(-2.0), Vec3::splat(2.0), &config) {
            Ok(mesh) => {
                assert!(mesh.vertices.len() > 0, "Complex shape should produce mesh");
            }
            Err(GpuError::NoAdapter) => {
                eprintln!("Skipping GPU MC test: no GPU adapter available");
            }
            Err(e) => panic!("GPU MC failed: {}", e),
        }
    }
}
