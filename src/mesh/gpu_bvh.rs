//! Prototype: a [`MeshBvh`] on the GPU, flattened to storage buffers, sphere traced by one kernel
//!
//! Author: Moroya Sakamoto
use super::bvh::{BvhNode, MeshBvh};
use crate::compiled::GpuError;
use crate::gpu_context::GpuContext;
use std::time::Instant;
use wgpu::util::DeviceExt;

/// The kernel's text
pub const KERNEL: &str = include_str!("gpu_bvh.wgsl");

/// The BVH as the kernel reads it
pub struct FlatBvh {
    /// 8 words a node: min, a, max, b; a leaf has the top bit of `a` set, `b` its count
    pub nodes: Vec<u32>,
    /// Triangle indices of the leaves
    pub leaf_tris: Vec<u32>,
    /// 9 floats a triangle
    pub tris: Vec<f32>,
    /// The deepest node, the root at 1
    pub depth: u32,
}

fn push(node: &BvhNode, out: &mut FlatBvh, depth: u32) -> u32 {
    let at = (out.nodes.len() / 8) as u32;
    out.depth = out.depth.max(depth);
    let a = node.aabb();
    let head = [
        a.min.x.to_bits(),
        a.min.y.to_bits(),
        a.min.z.to_bits(),
        0,
        a.max.x.to_bits(),
        a.max.y.to_bits(),
        a.max.z.to_bits(),
        0,
    ];
    out.nodes.extend_from_slice(&head);
    let (w3, w7) = match node {
        BvhNode::Leaf { triangles, .. } => {
            let first = out.leaf_tris.len() as u32;
            out.leaf_tris.extend(triangles.iter().map(|t| *t as u32));
            (first | 0x8000_0000, triangles.len() as u32)
        }
        BvhNode::Internal { left, right, .. } => {
            let l = push(left, out, depth + 1);
            let r = push(right, out, depth + 1);
            (l, r)
        }
    };
    out.nodes[at as usize * 8 + 3] = w3;
    out.nodes[at as usize * 8 + 7] = w7;
    at
}

/// Flatten a built BVH
#[must_use]
pub fn flatten(bvh: &MeshBvh) -> FlatBvh {
    let mut out = FlatBvh {
        nodes: Vec::new(),
        leaf_tris: Vec::new(),
        tris: Vec::with_capacity(bvh.triangles.len() * 9),
        depth: 0,
    };
    for t in &bvh.triangles {
        for v in [t.v0, t.v1, t.v2] {
            out.tris.extend_from_slice(&[v.x, v.y, v.z]);
        }
    }
    if let Some(root) = &bvh.root {
        push(root, &mut out, 1);
    }
    out
}

/// One ray's answer: s, triangle (`u32::MAX` none), evaluations, hit
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, PartialEq, Debug)]
pub struct RayOut {
    /// Distance along the ray at the hit
    pub s: f32,
    /// Nearest triangle at the hit
    pub tri: u32,
    /// Evaluations
    pub evals: u32,
    /// 1 on a hit
    pub hit: u32,
}

/// A ready kernel and its buffers
pub struct Tracer {
    context: GpuContext,
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    params: wgpu::Buffer,
    output: wgpu::Buffer,
    staging: wgpu::Buffer,
    shell: f32,
    hit: f32,
    max_evals: u32,
    tile: u32,
    /// Seconds to compile the module and the pipeline
    pub compile_s: f64,
}

impl Tracer {
    /// Build the kernel and upload the data; `rays` is 7 floats a ray
    ///
    /// # Errors
    /// [`GpuError`] when there is no device or the kernel does not compile.
    pub fn new(
        flat: &FlatBvh,
        rays: &[f32],
        shell: f32,
        hit: f32,
        max_evals: u32,
        tile: u32,
    ) -> Result<Self, GpuError> {
        let context = GpuContext::request("mesh bvh prototype")?;
        let started = Instant::now();
        let text = std::env::var("MG_KERNEL").map_or_else(
            |_| KERNEL.to_owned(),
            |p| std::fs::read_to_string(p).expect("kernel"),
        );
        let device = context.device();
        if let Ok(path) = std::env::var("MG_DUMP_SPV") {
            let words = crate::gpu_context::no_contraction_spirv(wgpu::ShaderSource::Wgsl(
                text.clone().into(),
            ))?;
            std::fs::write(path, bytemuck::cast_slice::<u32, u8>(&words)).expect("dump");
        }
        let module = if std::env::var_os("MG_PLAIN").is_some() {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mesh bvh"),
                source: wgpu::ShaderSource::Wgsl(text.into()),
            })
        } else {
            context.shader_module("mesh bvh", wgpu::ShaderSource::Wgsl(text.into()))?
        };
        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh bvh"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh bvh"),
            bind_group_layouts: &[&group],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mesh bvh"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let compile_s = started.elapsed().as_secs_f64();
        let storage = |label: &str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let b_rays = storage("rays", bytemuck::cast_slice(rays));
        let b_nodes = storage("nodes", bytemuck::cast_slice(&flat.nodes));
        let b_leaf = storage("leaf", bytemuck::cast_slice(&flat.leaf_tris));
        let b_tris = storage("tris", bytemuck::cast_slice(&flat.tris));
        let size = u64::from(tile) * 16;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh bvh"),
            layout: &group,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_rays.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_leaf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_tris.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            context,
            pipeline,
            bind,
            params,
            output,
            staging,
            shell,
            hit,
            max_evals,
            tile,
            compile_s,
        })
    }

    /// Trace `count` rays in tiles of the size given to [`Tracer::new`]; also returns the longest
    /// tile, in seconds, from submit to the mapped result
    ///
    /// # Errors
    /// [`GpuError::BufferMapping`] when a read does not complete.
    pub fn trace(&self, count: u32) -> Result<(Vec<RayOut>, f64), GpuError> {
        let (device, queue) = (self.context.device(), self.context.queue());
        let mut all = Vec::with_capacity(count as usize);
        let mut longest = 0.0_f64;
        let mut first = 0;
        while first < count {
            let n = self.tile.min(count - first);
            let words = [
                first,
                n,
                self.shell.to_bits(),
                self.hit.to_bits(),
                self.max_evals,
                0,
                0,
                0,
            ];
            queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&words));
            let started = Instant::now();
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind, &[]);
                pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
            }
            let bytes = u64::from(n) * 16;
            encoder.copy_buffer_to_buffer(&self.output, 0, &self.staging, 0, bytes);
            queue.submit(std::iter::once(encoder.finish()));
            let slice = self.staging.slice(..bytes);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = sender.send(r);
            });
            device.poll(wgpu::Maintain::Wait);
            receiver
                .recv()
                .map_err(|e| GpuError::BufferMapping(e.to_string()))?
                .map_err(|e| GpuError::BufferMapping(format!("{e:?}")))?;
            all.extend_from_slice(bytemuck::cast_slice::<u8, RayOut>(
                &slice.get_mapped_range(),
            ));
            self.staging.unmap();
            longest = longest.max(started.elapsed().as_secs_f64());
            first += n;
        }
        Ok((all, longest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses, validates and goes through the marking pass; no device
    #[test]
    fn the_kernel_goes_through_the_marking_pass() {
        let words =
            crate::gpu_context::no_contraction_spirv(wgpu::ShaderSource::Wgsl(KERNEL.into()))
                .expect("marked SPIR-V");
        assert!(words.len() > 100);
    }
}
