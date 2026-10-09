//! GPU Compute Marching Cubes (Deep Fried Edition)
//!
//! GPU Marching Cubes pipeline for faster mesh generation compared to CPU
//! at high resolution (128+). The mesh shares each vertex between the cells
//! around its lattice edge, like the CPU [`crate::mesh::marching_cubes`].
//!
//! # Pipeline
//!
//! ```text
//! Pass 1:  SDF Grid Eval      @workgroup_size(4,4,4) -> sdf_grid[(res+1)^3]
//! Pass 2:  Cell Classify      @workgroup_size(4,4,4) -> index_counts[res^3] + cube_indices
//! Pass 2b: Edge Count         @workgroup_size(4,4,4) -> edge_counts[(res+1)^3]
//! Prefix sums (CPU, or GPU above res 128)            -> cell / grid point offsets
//! Pass 3:  Edge Vertices      @workgroup_size(4,4,4) -> vertices[total_verts]
//! Pass 4:  Triangle Indices   @workgroup_size(4,4,4) -> indices[total_indices]
//! CPU:     Readback                                  -> Mesh { vertices, indices }
//! ```
//!
//! # Deep Fried Optimizations
//!
//! - **Atomic counting**: Pass 2 / 2b use atomicAdd for the index / vertex totals
//! - **CPU prefix sum**: Avoids complex GPU scan; fast enough for res <= 512
//! - **Tetrahedral normals**: 4-point gradient estimation on GPU
//! - **Zero-copy tables**: EDGE_TABLE and TRI_TABLE embedded as WGSL constants
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use wgpu::util::DeviceExt;

use super::gpu_mc_shaders;
use crate::compiled::{GpuError, TranspileMode, WgslShader};
use crate::mesh::sdf_to_mesh::lattice_point;
use crate::mesh::{Mesh, Vertex};
use crate::types::SdfNode;

/// Resolution threshold: use GPU prefix sum when res > this value
const GPU_PREFIX_SUM_THRESHOLD: u32 = 128;

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
        Self {
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
///
/// The mesh has one vertex per sign-changing lattice edge, shared by every
/// triangle that uses it, in the vertex order of the CPU
/// [`crate::mesh::marching_cubes`] (lattice point, then axis); a grid value
/// equal to the iso-level is outside on both paths and the lattice
/// coordinates are computed on the host with the CPU formula. Wherever the
/// two evaluators give the same signs the two meshes have the same index
/// buffer. As on the CPU, a lattice point exactly on the surface gives one
/// vertex per sign-changing edge at that point: several vertices can share a
/// position and zero-area triangles can occur (see
/// [`crate::mesh::marching_cubes`]). Until 5.0 the GPU path returned one vertex per triangle corner
/// whose copies differed in the last bits, and callers had to weld by
/// distance.
///
/// The vertex and index buffers are sized from the counts of Pass 2 / 2b.
/// A non-zero `config.max_vertices` is a cap: a surface with more vertices
/// fails with [`GpuError::BufferMapping`] instead of being truncated.
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

    // Lattice coordinates per axis with the CPU marching cubes formula, so
    // both paths evaluate the field at bit-identical points.
    let cell = (bounds_max - bounds_min) / res as f32;
    let mut axis_coords = Vec::with_capacity(3 * grid_size);
    for axis in 0..3 {
        for i in 0..grid_size {
            let mut g = [0usize; 3];
            g[axis] = i;
            axis_coords.push(lattice_point(bounds_min, cell, g)[axis]);
        }
    }

    // Initialize wgpu
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
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
    let axis_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MC Axis Coords"),
        contents: bytemuck::cast_slice(&axis_coords),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let storage = |label: &str, bytes: usize| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let counter = |label: &str| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    };

    let sdf_grid_buffer = storage("SDF Grid Buffer", grid_total * 4);
    let cell_counts_buffer = storage("Cell Index Counts", cell_total * 4);
    let cell_indices_buffer = storage("Cell Cube Indices", cell_total * 4);
    let point_counts_buffer = storage("Point Edge Counts", grid_total * 4);
    let total_index_buffer = counter("Total Index Count");
    let total_vertex_buffer = counter("Total Vertex Count");

    // (read_only, buffer) per binding, in binding order
    let ro = wgpu::BufferBindingType::Storage { read_only: true };
    let rw = wgpu::BufferBindingType::Storage { read_only: false };
    let uni = wgpu::BufferBindingType::Uniform;
    let stage = |label: &str,
                 source: String,
                 bindings: &[(wgpu::BufferBindingType, &wgpu::Buffer)]|
     -> (wgpu::ComputePipeline, wgpu::BindGroup) {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let entries: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(i, (ty, _))| bgl_entry(i as u32, *ty))
            .collect();
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        });
        let pipeline = create_pipeline(&device, &bgl, &module, label);
        let group_entries: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(i, (_, b))| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &bgl,
            entries: &group_entries,
        });
        (pipeline, bg)
    };

    // ====== PASS 1: SDF grid, PASS 2: cells, PASS 2b: edges per grid point ======
    let pass1 = stage(
        "MC Pass 1",
        gpu_mc_shaders::generate_sdf_grid_shader(sdf_shader),
        &[
            (rw, &sdf_grid_buffer),
            (uni, &uniform_buffer),
            (ro, &axis_buffer),
        ],
    );
    let pass2 = stage(
        "MC Pass 2",
        gpu_mc_shaders::generate_classify_shader(),
        &[
            (ro, &sdf_grid_buffer),
            (uni, &uniform_buffer),
            (rw, &cell_counts_buffer),
            (rw, &cell_indices_buffer),
            (rw, &total_index_buffer),
        ],
    );
    let pass2b = stage(
        "MC Pass 2b",
        gpu_mc_shaders::generate_edge_count_shader(),
        &[
            (ro, &sdf_grid_buffer),
            (uni, &uniform_buffer),
            (rw, &point_counts_buffer),
            (rw, &total_vertex_buffer),
        ],
    );

    let wg = 4u32;
    let dispatch_grid = (grid_size as u32).div_ceil(wg);
    let dispatch_cell = res.div_ceil(wg);
    let use_gpu_prefix_sum = res > GPU_PREFIX_SUM_THRESHOLD;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MC Pass 1+2 Encoder"),
    });
    for (p, n) in [
        (&pass1, dispatch_grid),
        (&pass2, dispatch_cell),
        (&pass2b, dispatch_grid),
    ] {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&p.0);
        pass.set_bind_group(0, &p.1, &[]);
        pass.dispatch_workgroups(n, n, n);
    }
    let staging = |label: &str, bytes: usize| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let totals_staging = staging("Totals Staging", 8);
    encoder.copy_buffer_to_buffer(&total_index_buffer, 0, &totals_staging, 0, 4);
    encoder.copy_buffer_to_buffer(&total_vertex_buffer, 0, &totals_staging, 4, 4);
    // For the CPU prefix sum, also read back the per-cell and per-point counts
    let counts_staging = if use_gpu_prefix_sum {
        None
    } else {
        let c = staging("Cell Counts Staging", cell_total * 4);
        let p = staging("Point Counts Staging", grid_total * 4);
        encoder.copy_buffer_to_buffer(&cell_counts_buffer, 0, &c, 0, (cell_total * 4) as u64);
        encoder.copy_buffer_to_buffer(&point_counts_buffer, 0, &p, 0, (grid_total * 4) as u64);
        Some((c, p))
    };
    queue.submit(std::iter::once(encoder.finish()));

    let totals = read_u32_vec(&device, &totals_staging, 2)?;
    let (total_indices, total_verts) = (totals[0] as usize, totals[1] as usize);
    if total_indices == 0 {
        return Ok(Mesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        });
    }
    if config.max_vertices > 0 && total_verts > config.max_vertices as usize {
        return Err(GpuError::BufferMapping(format!(
            "marching cubes surface has {total_verts} vertices, more than max_vertices {}",
            config.max_vertices
        )));
    }

    // ====== Prefix sums: cell index offsets and grid point vertex offsets ======
    let (cell_offsets_buffer, point_offsets_buffer) = match &counts_staging {
        None => (
            dispatch_gpu_prefix_sum(&device, &queue, &cell_counts_buffer, cell_total)?,
            dispatch_gpu_prefix_sum(&device, &queue, &point_counts_buffer, grid_total)?,
        ),
        Some((c, p)) => {
            let upload = |label: &str, counts: Vec<u32>| {
                let mut running = 0u32;
                let offsets: Vec<u32> = counts
                    .iter()
                    .map(|&n| {
                        let o = running;
                        running += n;
                        o
                    })
                    .collect();
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(&offsets),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            (
                upload("Cell Offsets", read_u32_vec(&device, c, cell_total)?),
                upload("Point Offsets", read_u32_vec(&device, p, grid_total)?),
            )
        }
    };

    // ====== PASS 3: edge vertices, PASS 4: triangle indices ======
    let vertex_bytes = total_verts * std::mem::size_of::<GpuVertex>();
    let index_bytes = total_indices * 4;
    let vertex_buffer = storage("Output Vertices", vertex_bytes);
    let index_buffer = storage("Output Indices", index_bytes);
    let tri_table = gpu_mc_shaders::tri_table_flat();
    let tri_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MC Tri Table"),
        contents: bytemuck::cast_slice(&tri_table),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let pass3 = stage(
        "MC Pass 3",
        gpu_mc_shaders::generate_vertex_shader(sdf_shader),
        &[
            (ro, &sdf_grid_buffer),
            (uni, &uniform_buffer),
            (ro, &point_offsets_buffer),
            (rw, &vertex_buffer),
            (ro, &axis_buffer),
        ],
    );
    let pass4 = stage(
        "MC Pass 4",
        gpu_mc_shaders::generate_cell_index_shader(),
        &[
            (ro, &sdf_grid_buffer),
            (uni, &uniform_buffer),
            (ro, &cell_offsets_buffer),
            (ro, &cell_indices_buffer),
            (rw, &index_buffer),
            (ro, &tri_table_buffer),
            (ro, &point_offsets_buffer),
        ],
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MC Pass 3+4 Encoder"),
    });
    for (p, n) in [(&pass3, dispatch_grid), (&pass4, dispatch_cell)] {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&p.0);
        pass.set_bind_group(0, &p.1, &[]);
        pass.dispatch_workgroups(n, n, n);
    }
    let vertex_staging = staging("Vertex Staging", vertex_bytes);
    let index_staging = staging("Index Staging", index_bytes);
    encoder.copy_buffer_to_buffer(&vertex_buffer, 0, &vertex_staging, 0, vertex_bytes as u64);
    encoder.copy_buffer_to_buffer(&index_buffer, 0, &index_staging, 0, index_bytes as u64);
    queue.submit(std::iter::once(encoder.finish()));

    let gpu_verts = read_gpu_vertices(&device, &vertex_staging, total_verts)?;
    let indices = read_u32_vec(&device, &index_staging, total_indices)?;
    let vertices = gpu_verts
        .iter()
        .map(|gv| {
            Vertex::new(
                Vec3::new(gv.px, gv.py, gv.pz),
                Vec3::new(gv.nx, gv.ny, gv.nz),
            )
        })
        .collect();

    Ok(Mesh { vertices, indices })
}

// ====== Helper functions ======

/// Create a bind group layout entry (compute, storage/uniform)
const fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
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

/// Dispatch recursive GPU prefix sum (Hillis-Steele)
///
/// Computes exclusive prefix sum of `input_buffer` (u32 × `count` elements)
/// entirely on the GPU. Returns a STORAGE buffer containing the offsets.
fn dispatch_gpu_prefix_sum(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_buffer: &wgpu::Buffer,
    count: usize,
) -> Result<wgpu::Buffer, GpuError> {
    let wg = gpu_mc_shaders::PREFIX_SUM_WG;
    let n = count as u32;
    let num_blocks = n.div_ceil(wg);

    // Compile scan and propagate shaders
    let scan_source = gpu_mc_shaders::generate_prefix_sum_scan_shader();
    let propagate_source = gpu_mc_shaders::generate_prefix_sum_propagate_shader();

    let scan_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Prefix Sum Scan Shader"),
        source: wgpu::ShaderSource::Wgsl(scan_source.into()),
    });
    let propagate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Prefix Sum Propagate Shader"),
        source: wgpu::ShaderSource::Wgsl(propagate_source.into()),
    });

    // Scan BGL: input(read), output(rw), block_sums(rw), params(uniform)
    let scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Prefix Sum Scan BGL"),
        entries: &[
            bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
            bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
            bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            bgl_entry(3, wgpu::BufferBindingType::Uniform),
        ],
    });
    let scan_pipeline = create_pipeline(device, &scan_bgl, &scan_shader, "Prefix Sum Scan");

    // Propagate BGL: data(rw), block_offsets(read), params(uniform)
    let propagate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Prefix Sum Propagate BGL"),
        entries: &[
            bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
            bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
            bgl_entry(2, wgpu::BufferBindingType::Uniform),
        ],
    });
    let propagate_pipeline = create_pipeline(
        device,
        &propagate_bgl,
        &propagate_shader,
        "Prefix Sum Propagate",
    );

    // Recursive prefix sum implementation
    gpu_prefix_sum_recursive(
        device,
        queue,
        input_buffer,
        n,
        num_blocks,
        &scan_bgl,
        &scan_pipeline,
        &propagate_bgl,
        &propagate_pipeline,
    )
}

/// Recursive helper for multi-level GPU prefix sum
#[allow(clippy::too_many_arguments)]
fn gpu_prefix_sum_recursive(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_buffer: &wgpu::Buffer,
    n: u32,
    num_blocks: u32,
    scan_bgl: &wgpu::BindGroupLayout,
    scan_pipeline: &wgpu::ComputePipeline,
    propagate_bgl: &wgpu::BindGroupLayout,
    propagate_pipeline: &wgpu::ComputePipeline,
) -> Result<wgpu::Buffer, GpuError> {
    let wg = gpu_mc_shaders::PREFIX_SUM_WG;

    // Output buffer for scanned results
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Prefix Sum Output"),
        size: (n as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Block sums buffer
    let block_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Prefix Sum Block Sums"),
        size: (num_blocks as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Params uniform (element count)
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Prefix Sum Params"),
        contents: bytemuck::cast_slice(&[n, 0u32, 0u32, 0u32]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Scan pass: scan each block, extract block sums
    let scan_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Prefix Sum Scan BG"),
        layout: scan_bgl,
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
                resource: block_sums_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Prefix Sum Scan Encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Prefix Sum Scan"),
            timestamp_writes: None,
        });
        pass.set_pipeline(scan_pipeline);
        pass.set_bind_group(0, &scan_bg, &[]);
        pass.dispatch_workgroups(num_blocks, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    // If only 1 block, no propagation needed
    if num_blocks <= 1 {
        return Ok(output_buffer);
    }

    // Recursively scan block sums
    let next_num_blocks = num_blocks.div_ceil(wg);
    let scanned_block_sums = gpu_prefix_sum_recursive(
        device,
        queue,
        &block_sums_buffer,
        num_blocks,
        next_num_blocks,
        scan_bgl,
        scan_pipeline,
        propagate_bgl,
        propagate_pipeline,
    )?;

    // Propagate: add scanned block sums back to each element
    let propagate_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Prefix Sum Propagate BG"),
        layout: propagate_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scanned_block_sums.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Prefix Sum Propagate Encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Prefix Sum Propagate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(propagate_pipeline);
        pass.set_bind_group(0, &propagate_bg, &[]);
        pass.dispatch_workgroups(num_blocks, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    Ok(output_buffer)
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

    let result = {
        let mapped = slice.get_mapped_range();
        let data: &[u32] = bytemuck::cast_slice(&mapped);
        if data.len() < count {
            return Err(GpuError::BufferMapping(format!(
                "read_u32_vec: buffer has {} u32 elements, expected {}",
                data.len(),
                count
            )));
        }
        data[..count].to_vec()
    };
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

    let result = {
        let mapped = slice.get_mapped_range();
        let data: &[GpuVertex] = bytemuck::cast_slice(&mapped);
        if data.len() < count {
            return Err(GpuError::BufferMapping(format!(
                "read_gpu_vertices: buffer has {} vertices, expected {}",
                data.len(),
                count
            )));
        }
        data[..count].to_vec()
    };
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
                assert!(!mesh.vertices.is_empty(), "Mesh should have vertices");
                assert!(!mesh.indices.is_empty(), "Mesh should have indices");
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
                assert!(
                    !mesh.vertices.is_empty(),
                    "Complex shape should produce mesh"
                );
            }
            Err(GpuError::NoAdapter) => {
                eprintln!("Skipping GPU MC test: no GPU adapter available");
            }
            Err(e) => panic!("GPU MC failed: {}", e),
        }
    }
}
