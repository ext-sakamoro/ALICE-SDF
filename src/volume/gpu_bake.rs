//! GPU-based SDF Volume Baking (Deep Fried Edition)
//!
//! Bakes SDF to 3D volume texture using a compute shader with 3D dispatch.
//! Typically ~50x faster than CPU baking at high resolution (128^3+).
//!
//! # GPU Pipeline
//!
//! 1. Transpile SDF to WGSL volume shader (3D grid dispatch)
//! 2. Create storage buffer for output voxels
//! 3. Dispatch `@workgroup_size(4,4,4)` with `(res/4, res/4, res/4)` workgroups
//! 4. Readback storage buffer to CPU
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use wgpu::util::DeviceExt;

use super::{BakeConfig, Volume3D, VoxelDistGrad};
use crate::compiled::{GpuError, TranspileMode, WgslShader};
use crate::types::SdfNode;

/// Uniform buffer for volume bake parameters (48 bytes, 16-byte aligned)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VolumeBakeUniforms {
    /// Resolution (xyz) + padding
    resolution: [u32; 4],
    /// World-space minimum bounds + padding
    bounds_min: [f32; 4],
    /// World-space maximum bounds + padding
    bounds_max: [f32; 4],
}

/// Bake SDF to volume on the GPU
///
/// # Arguments
/// * `node` - The SDF tree to bake
/// * `config` - Baking configuration
///
/// # Returns
/// Volume3D<f32> containing signed distances
pub fn gpu_bake_volume(node: &SdfNode, config: &BakeConfig) -> Result<Volume3D<f32>, GpuError> {
    let shader = WgslShader::transpile(node, TranspileMode::Hardcoded);
    gpu_bake_volume_from_shader(&shader, config)
}

/// Bake volume from a pre-compiled WGSL shader
pub fn gpu_bake_volume_from_shader(
    shader: &WgslShader,
    config: &BakeConfig,
) -> Result<Volume3D<f32>, GpuError> {
    let res = config.resolution;
    let total = res[0] as usize * res[1] as usize * res[2] as usize;

    // Compute padded bounds
    let padding_world = if config.padding > 0 {
        let size = config.bounds_max - config.bounds_min;
        Vec3::new(
            size.x / res[0] as f32 * config.padding as f32,
            size.y / res[1] as f32 * config.padding as f32,
            size.z / res[2] as f32 * config.padding as f32,
        )
    } else {
        Vec3::ZERO
    };
    let world_min = config.bounds_min - padding_world;
    let world_max = config.bounds_max + padding_world;

    // Generate 3D volume bake shader
    let compute_source = generate_volume_bake_shader(shader);

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
            label: Some("ALICE-SDF Volume Bake Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .map_err(|e: wgpu::RequestDeviceError| GpuError::DeviceCreation(e.to_string()))?;

    // Create shader module
    let shader_module: wgpu::ShaderModule =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Volume Bake Shader"),
            source: wgpu::ShaderSource::Wgsl(compute_source.into()),
        });

    // Create output buffer
    let output_buffer_size = (total * std::mem::size_of::<f32>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Volume Output Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create staging buffer
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Volume Staging Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create uniform buffer
    let uniforms = VolumeBakeUniforms {
        resolution: [res[0], res[1], res[2], 0],
        bounds_min: [world_min.x, world_min.y, world_min.z, 0.0],
        bounds_max: [world_max.x, world_max.y, world_max.z, 0.0],
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Volume Uniforms Buffer"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Volume Bake Bind Group Layout"),
        entries: &[
            // Output volume buffer
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Uniforms
            wgpu::BindGroupLayoutEntry {
                binding: 1,
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

    // Pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Volume Bake Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Volume Bake Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Volume Bake Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch 3D compute
    let wg = 4u32; // workgroup_size(4, 4, 4)
    let dispatch_x = (res[0] + wg - 1) / wg;
    let dispatch_y = (res[1] + wg - 1) / wg;
    let dispatch_z = (res[2] + wg - 1) / wg;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Volume Bake Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Volume Bake Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
    queue.submit(std::iter::once(encoder.finish()));

    // Readback
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);

    pollster::block_on(receiver)
        .map_err(|e| GpuError::BufferMapping(format!("Channel error: {}", e)))?
        .map_err(|e| GpuError::BufferMapping(format!("Map error: {:?}", e)))?;

    let mapped = buffer_slice.get_mapped_range();
    let data: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging_buffer.unmap();

    let mut volume = Volume3D {
        data,
        resolution: res,
        world_min,
        world_max,
        mips: Vec::new(),
    };

    if config.generate_mips {
        volume.mips = super::mipchain::generate_mip_chain(&volume);
    }

    Ok(volume)
}

/// Bake distance + gradient volume on GPU
pub fn gpu_bake_volume_with_normals(
    node: &SdfNode,
    config: &BakeConfig,
    _gradient_epsilon: f32,
) -> Result<Volume3D<VoxelDistGrad>, GpuError> {
    let shader = WgslShader::transpile(node, TranspileMode::Hardcoded);
    let res = config.resolution;
    let total = res[0] as usize * res[1] as usize * res[2] as usize;

    let padding_world = if config.padding > 0 {
        let size = config.bounds_max - config.bounds_min;
        Vec3::new(
            size.x / res[0] as f32 * config.padding as f32,
            size.y / res[1] as f32 * config.padding as f32,
            size.z / res[2] as f32 * config.padding as f32,
        )
    } else {
        Vec3::ZERO
    };
    let world_min = config.bounds_min - padding_world;
    let world_max = config.bounds_max + padding_world;

    let compute_source = generate_volume_bake_shader_with_normals(&shader);

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
            label: Some("ALICE-SDF Volume Bake Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Volume Bake Normal Shader"),
        source: wgpu::ShaderSource::Wgsl(compute_source.into()),
    });

    // Output: 4 floats per voxel (dist, nx, ny, nz)
    let output_buffer_size = (total * std::mem::size_of::<VoxelDistGrad>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Volume Output Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Volume Staging Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniforms = VolumeBakeUniforms {
        resolution: [res[0], res[1], res[2], 0],
        bounds_min: [world_min.x, world_min.y, world_min.z, 0.0],
        bounds_max: [world_max.x, world_max.y, world_max.z, 0.0],
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Volume Uniforms Buffer"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Volume Bake Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
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
        label: Some("Volume Bake Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Volume Bake Normal Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Volume Bake Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let wg = 4u32;
    let dispatch_x = (res[0] + wg - 1) / wg;
    let dispatch_y = (res[1] + wg - 1) / wg;
    let dispatch_z = (res[2] + wg - 1) / wg;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Volume Bake Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Volume Bake Normal Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);

    pollster::block_on(receiver)
        .map_err(|e| GpuError::BufferMapping(format!("Channel error: {}", e)))?
        .map_err(|e| GpuError::BufferMapping(format!("Map error: {:?}", e)))?;

    let mapped = buffer_slice.get_mapped_range();
    let data: Vec<VoxelDistGrad> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging_buffer.unmap();

    let mut volume = Volume3D {
        data,
        resolution: res,
        world_min,
        world_max,
        mips: Vec::new(),
    };

    if config.generate_mips {
        volume.mips = super::mipchain::generate_mip_chain_distgrad(&volume);
    }

    Ok(volume)
}

/// Generate WGSL compute shader for 3D volume baking (distance only)
fn generate_volume_bake_shader(shader: &WgslShader) -> String {
    format!(
        r#"// ALICE-SDF Volume Bake Shader (3D Dispatch)

struct VolumeUniforms {{
    resolution: vec4<u32>,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
}}

@group(0) @binding(0) var<storage, read_write> output_volume: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: VolumeUniforms;

{sdf_func}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let res = uniforms.resolution.xyz;

    // Bounds check
    if (gid.x >= res.x || gid.y >= res.y || gid.z >= res.z) {{
        return;
    }}

    // Compute world-space position
    let fres = vec3<f32>(f32(res.x) - 1.0, f32(res.y) - 1.0, f32(res.z) - 1.0);
    let t = vec3<f32>(f32(gid.x), f32(gid.y), f32(gid.z)) / max(fres, vec3<f32>(1.0));
    let p = mix(uniforms.bounds_min.xyz, uniforms.bounds_max.xyz, t);

    // Evaluate SDF
    let distance = sdf_eval(p);

    // Write to flat array (Z-major order)
    let idx = gid.x + gid.y * res.x + gid.z * res.x * res.y;
    output_volume[idx] = distance;
}}
"#,
        sdf_func = shader.source,
    )
}

/// Generate WGSL compute shader for 3D volume baking (distance + normals)
fn generate_volume_bake_shader_with_normals(shader: &WgslShader) -> String {
    format!(
        r#"// ALICE-SDF Volume Bake Shader - Distance + Normals (3D Dispatch)

struct VolumeUniforms {{
    resolution: vec4<u32>,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
}}

struct VoxelDistGrad {{
    distance: f32,
    nx: f32,
    ny: f32,
    nz: f32,
}}

@group(0) @binding(0) var<storage, read_write> output_volume: array<VoxelDistGrad>;
@group(0) @binding(1) var<uniform> uniforms: VolumeUniforms;

{sdf_func}

fn estimate_normal(p: vec3<f32>) -> vec3<f32> {{
    let e = 0.001;
    let k0 = vec3<f32>(1.0, -1.0, -1.0);
    let k1 = vec3<f32>(-1.0, -1.0, 1.0);
    let k2 = vec3<f32>(-1.0, 1.0, -1.0);
    let k3 = vec3<f32>(1.0, 1.0, 1.0);

    let d0 = sdf_eval(p + k0 * e);
    let d1 = sdf_eval(p + k1 * e);
    let d2 = sdf_eval(p + k2 * e);
    let d3 = sdf_eval(p + k3 * e);

    return normalize(k0 * d0 + k1 * d1 + k2 * d2 + k3 * d3);
}}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let res = uniforms.resolution.xyz;

    if (gid.x >= res.x || gid.y >= res.y || gid.z >= res.z) {{
        return;
    }}

    let fres = vec3<f32>(f32(res.x) - 1.0, f32(res.y) - 1.0, f32(res.z) - 1.0);
    let t = vec3<f32>(f32(gid.x), f32(gid.y), f32(gid.z)) / max(fres, vec3<f32>(1.0));
    let p = mix(uniforms.bounds_min.xyz, uniforms.bounds_max.xyz, t);

    let distance = sdf_eval(p);
    let n = estimate_normal(p);

    let idx = gid.x + gid.y * res.x + gid.z * res.x * res.y;
    output_volume[idx].distance = distance;
    output_volume[idx].nx = n.x;
    output_volume[idx].ny = n.y;
    output_volume[idx].nz = n.z;
}}
"#,
        sdf_func = shader.source,
    )
}
