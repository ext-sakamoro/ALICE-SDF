//! A [`MeshBvh`] on the GPU: the first hit of rays on a mesh thickened by a radius
//!
//! [`ShellRenderer`] answers what [`MeshBvh::shell_first_hit`] answers, for many rays at once: on a
//! GPU through one compute kernel (`gpu_bvh_shell.wgsl`, the CPU method's operations in the same
//! order over the BVH flattened to storage buffers and walked by parent links), or on the CPU when
//! no device is available.
//!
//! The GPU path compiles through [`GpuContext`]'s shader path, which keeps the driver from fusing
//! `a * b + c`. WGSL still allows `sqrt` and division a few ulps from the correctly rounded result,
//! so `t` is not claimed to match the CPU bit for bit; the comparison test measures it.
//!
//! Author: Moroya Sakamoto

use super::bvh::{BvhNode, MeshBvh};
use super::bvh_shell::{ShellFeature, ShellFrame, ShellHit, ShellRay};
use crate::compiled::GpuError;
use crate::gpu_context::GpuContext;
use wgpu::util::DeviceExt;

/// The kernel's text: the closest-point helpers, then the shell walk
pub const SHELL_KERNEL: &str = concat!(
    include_str!("gpu_bvh_closest.wgsl"),
    "\n",
    include_str!("gpu_bvh_shell.wgsl")
);

const WORKGROUP: u32 = 64;
/// Rays in one submit, so one submit's work stays short on a shared GPU
const RAYS_PER_SUBMIT: u32 = 1 << 16;

/// The BVH as the kernel reads it
pub struct FlatBvh {
    /// 8 words a node: min, a, max, b; a leaf has the top bit of `a` set and `b` its count, an
    /// internal node has its left child in `a` and its right in `b`. The root is node 0.
    pub nodes: Vec<u32>,
    /// Triangle indices of the leaves
    pub leaf_tris: Vec<u32>,
    /// 9 floats a triangle: v0, v1, v2
    pub tris: Vec<f32>,
    /// The parent of each node (the root's is itself)
    pub parents: Vec<u32>,
    /// 4 floats a triangle: its box centre (the triangle frame, see [`MeshBvh::shell_first_hit`])
    pub tri_centres: Vec<f32>,
    /// 9 floats a triangle: `v - c`, `c` its own box centre
    pub tri_local: Vec<f32>,
}

fn local(out: &mut Vec<f32>, t: &super::bvh::BvhTriangle, c: glam::Vec3) {
    for v in [t.v0 - c, t.v1 - c, t.v2 - c] {
        out.extend_from_slice(&[v.x, v.y, v.z]);
    }
}

fn push(node: &BvhNode, out: &mut FlatBvh) -> u32 {
    let at = (out.nodes.len() / 8) as u32;
    out.parents.push(at);
    let a = node.aabb();
    out.nodes.extend_from_slice(&[
        a.min.x.to_bits(),
        a.min.y.to_bits(),
        a.min.z.to_bits(),
        0,
        a.max.x.to_bits(),
        a.max.y.to_bits(),
        a.max.z.to_bits(),
        0,
    ]);
    let (w3, w7) = match node {
        BvhNode::Leaf { triangles, .. } => {
            let first = out.leaf_tris.len() as u32;
            out.leaf_tris.extend(triangles.iter().map(|&t| t as u32));
            (first | 0x8000_0000, triangles.len() as u32)
        }
        BvhNode::Internal { left, right, .. } => {
            let l = push(left, out);
            let r = push(right, out);
            out.parents[l as usize] = at;
            out.parents[r as usize] = at;
            (l, r)
        }
    };
    out.nodes[at as usize * 8 + 3] = w3;
    out.nodes[at as usize * 8 + 7] = w7;
    at
}

/// Flatten a built BVH; an empty one has no nodes
#[must_use]
pub fn flatten(bvh: &MeshBvh) -> FlatBvh {
    let mut out = FlatBvh {
        nodes: Vec::new(),
        leaf_tris: Vec::new(),
        tris: Vec::with_capacity(bvh.triangles.len() * 9),
        parents: Vec::new(),
        tri_centres: Vec::new(),
        tri_local: Vec::new(),
    };
    for t in &bvh.triangles {
        for v in [t.v0, t.v1, t.v2] {
            out.tris.extend_from_slice(&[v.x, v.y, v.z]);
        }
        let c = (t.aabb.min + t.aabb.max) * 0.5;
        out.tri_centres.extend_from_slice(&[c.x, c.y, c.z, 0.0]);
        local(&mut out.tri_local, t, c);
    }
    if let Some(root) = &bvh.root {
        push(root, &mut out);
    }
    out
}

/// Where a [`ShellRenderer`] computes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellBackend {
    /// The compute kernel on a GPU
    Gpu,
    /// [`MeshBvh::shell_first_hit`] on the CPU, one ray after another
    Cpu,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    first: u32,
    count: u32,
    r: f32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RawHit {
    t_base: f32,
    t_local: f32,
    tri: u32,
    feature: u32,
}

struct Gpu {
    context: GpuContext,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    nodes: wgpu::Buffer,
    leaf_tris: wgpu::Buffer,
    tris: wgpu::Buffer,
    parents: wgpu::Buffer,
    centres: wgpu::Buffer,
    local: wgpu::Buffer,
}

/// First hits of rays on a mesh thickened by `r`: the boundary of the set of points within `r` of
/// a triangle (see [`MeshBvh::shell_first_hit`] for the shape, ties and the inside rule)
pub struct ShellRenderer<'a> {
    bvh: &'a MeshBvh,
    r: f32,
    frame: ShellFrame,
    gpu: Option<Gpu>,
}

impl<'a> ShellRenderer<'a> {
    /// On a GPU when one is available, otherwise on the CPU ([`Self::backend`] says which)
    ///
    /// # Errors
    /// [`GpuError`] when a device exists but the kernel does not compile or the backend cannot keep
    /// the driver from fusing operations; no device is not an error.
    pub fn new(bvh: &'a MeshBvh, r: f32) -> Result<Self, GpuError> {
        Self::new_in(ShellFrame::Triangle, bvh, r)
    }

    /// [`Self::new`], solving each hit in `frame`
    ///
    /// # Errors
    /// As [`Self::new`].
    pub(crate) fn new_in(frame: ShellFrame, bvh: &'a MeshBvh, r: f32) -> Result<Self, GpuError> {
        match GpuContext::request("mesh shell renderer") {
            Ok(context) => Self::on_in(context, frame, bvh, r),
            Err(GpuError::NoAdapter | GpuError::DeviceCreation(_)) => {
                Ok(Self::cpu_in(frame, bvh, r))
            }
            Err(e) => Err(e),
        }
    }

    /// On the given device
    ///
    /// # Errors
    /// [`GpuError`] when the kernel does not compile or the backend cannot keep the driver from
    /// fusing operations.
    pub fn on(context: GpuContext, bvh: &'a MeshBvh, r: f32) -> Result<Self, GpuError> {
        Self::on_in(context, ShellFrame::Triangle, bvh, r)
    }

    /// [`Self::on`], solving each hit in `frame`
    ///
    /// # Errors
    /// As [`Self::on`].
    pub(crate) fn on_in(
        context: GpuContext,
        frame: ShellFrame,
        bvh: &'a MeshBvh,
        r: f32,
    ) -> Result<Self, GpuError> {
        let flat = flatten(bvh);
        if flat.nodes.is_empty() {
            // nothing to hit: every ray misses, with no buffer to bind
            return Ok(Self::cpu_in(frame, bvh, r));
        }
        let device = context.device();
        let code = match frame {
            ShellFrame::World => 0,
            ShellFrame::Triangle => 1,
        };
        let text = SHELL_KERNEL.replacen(
            "const FRAME: u32 = 0u;",
            &format!("const FRAME: u32 = {code}u;"),
            1,
        );
        let module = context.shader_module("mesh shell", wgpu::ShaderSource::Wgsl(text.into()))?;
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh shell"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, read),
                entry(2, read),
                entry(3, read),
                entry(4, read),
                entry(5, read),
                entry(6, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(7, read),
                entry(8, read),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh shell"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mesh shell"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let storage = |label: &str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        // a leaf list is never empty when there are nodes; the mesh has at least one triangle
        let nodes = storage("mesh shell nodes", bytemuck::cast_slice(&flat.nodes));
        let leaf_tris = storage("mesh shell leaves", bytemuck::cast_slice(&flat.leaf_tris));
        let tris = storage("mesh shell triangles", bytemuck::cast_slice(&flat.tris));
        let parents = storage("mesh shell parents", bytemuck::cast_slice(&flat.parents));
        // the world frame reads neither; bind something non-empty
        let (c, l): (&[f32], &[f32]) = match frame {
            ShellFrame::World => (&[0.0; 4], &[0.0; 9]),
            ShellFrame::Triangle => (&flat.tri_centres, &flat.tri_local),
        };
        let centres = storage("mesh shell frame centres", bytemuck::cast_slice(c));
        let local = storage("mesh shell local triangles", bytemuck::cast_slice(l));
        Ok(Self {
            bvh,
            r,
            frame,
            gpu: Some(Gpu {
                context,
                pipeline,
                layout,
                nodes,
                leaf_tris,
                tris,
                parents,
                centres,
                local,
            }),
        })
    }

    /// On the CPU only
    #[must_use]
    pub const fn cpu(bvh: &'a MeshBvh, r: f32) -> Self {
        Self::cpu_in(ShellFrame::Triangle, bvh, r)
    }

    /// [`Self::cpu`], solving each hit in `frame`
    #[must_use]
    pub(crate) const fn cpu_in(frame: ShellFrame, bvh: &'a MeshBvh, r: f32) -> Self {
        Self {
            bvh,
            r,
            frame,
            gpu: None,
        }
    }

    /// Where [`Self::first_hits`] computes
    #[must_use]
    pub const fn backend(&self) -> ShellBackend {
        if self.gpu.is_some() {
            ShellBackend::Gpu
        } else {
            ShellBackend::Cpu
        }
    }

    /// The first hit of each ray, in the order of `rays`
    ///
    /// # Errors
    /// [`GpuError`] when the GPU path fails (buffer mapping, a device lost); the CPU path does not
    /// fail.
    pub fn first_hits(&self, rays: &[ShellRay]) -> Result<Vec<Option<ShellHit>>, GpuError> {
        match &self.gpu {
            Some(gpu) => self.gpu_hits(gpu, rays),
            None => Ok(self.cpu_hits(rays)),
        }
    }

    /// The first hit of each ray on the CPU, whichever backend this renderer has
    #[must_use]
    pub fn cpu_hits(&self, rays: &[ShellRay]) -> Vec<Option<ShellHit>> {
        rays.iter()
            .map(|ray| {
                self.bvh
                    .shell_first_hit_in(self.frame, ray.origin, ray.dir, self.r, ray.tmax)
            })
            .collect()
    }

    fn gpu_hits(&self, gpu: &Gpu, rays: &[ShellRay]) -> Result<Vec<Option<ShellHit>>, GpuError> {
        if rays.is_empty() {
            return Ok(Vec::new());
        }
        let count = u32::try_from(rays.len())
            .map_err(|_| GpuError::BufferMapping(format!("{} rays in one call", rays.len())))?;
        let device = gpu.context.device();
        let queue = gpu.context.queue();
        let words: Vec<f32> = rays
            .iter()
            .flat_map(|r| {
                {
                    // 1 / dir on the CPU: the kernel then divides nowhere in the walk
                    let inv = r.dir.recip();
                    [
                        r.origin.x, r.origin.y, r.origin.z, r.dir.x, r.dir.y, r.dir.z, r.tmax,
                        inv.x, inv.y, inv.z,
                    ]
                }
            })
            .collect();
        let ray_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh shell rays"),
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_bytes = u64::from(count) * std::mem::size_of::<RawHit>() as u64;
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh shell hits"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh shell staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh shell params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh shell"),
            layout: &gpu.layout,
            entries: &[
                (0, &params),
                (1, &ray_buf),
                (2, &gpu.nodes),
                (3, &gpu.leaf_tris),
                (4, &gpu.tris),
                (5, &gpu.parents),
                (6, &out),
                (7, &gpu.centres),
                (8, &gpu.local),
            ]
            .map(|(binding, b)| wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            }),
        });
        let max_groups = device.limits().max_compute_workgroups_per_dimension;
        let per_submit = RAYS_PER_SUBMIT.min(max_groups.saturating_mul(WORKGROUP));
        let mut first = 0u32;
        while first < count {
            let n = per_submit.min(count - first);
            // a write lands before the next submit, so each submit reads its own range
            queue.write_buffer(
                &params,
                0,
                bytemuck::bytes_of(&Params {
                    first,
                    count: n,
                    r: self.r,
                    pad: 0,
                }),
            );
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mesh shell"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mesh shell"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&gpu.pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(n.div_ceil(WORKGROUP), 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));
            first += n;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mesh shell readback"),
        });
        encoder.copy_buffer_to_buffer(&out, 0, &staging, 0, out_bytes);
        gpu.context.before_host_read(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver)
            .map_err(|e| GpuError::BufferMapping(format!("Channel error: {e}")))?
            .map_err(|e| GpuError::BufferMapping(format!("Map error: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let raw: &[RawHit] = bytemuck::cast_slice(&mapped);
        let hits = raw.iter().map(decode).collect::<Result<Vec<_>, _>>();
        drop(mapped);
        staging.unmap();
        hits
    }
}

fn decode(h: &RawHit) -> Result<Option<ShellHit>, GpuError> {
    if h.tri == u32::MAX {
        return Ok(None);
    }
    let feature = match h.feature {
        1 => ShellFeature::Face,
        2 => ShellFeature::Edge,
        3 => ShellFeature::Vertex,
        4 => ShellFeature::Inside,
        f => {
            return Err(GpuError::BufferMapping(format!(
                "the kernel wrote feature {f} for triangle {}",
                h.tri
            )))
        }
    };
    Ok(Some(ShellHit {
        // the same f32 sum the kernel compared by
        t: h.t_base + h.t_local,
        t_base: h.t_base,
        t_local: h.t_local,
        triangle: h.tri,
        feature,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn the_shell_kernel_parses_and_validates() {
        let module = naga::front::wgsl::parse_str(SHELL_KERNEL).expect("the kernel parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("the kernel validates");
        assert!(module.entry_points.iter().any(|e| e.name == "main"));
    }

    #[test]
    fn flattening_keeps_every_triangle_once_with_parent_links() {
        let v: Vec<Vec3> = (0..60)
            .map(|i| Vec3::new((i % 7) as f32, (i % 5) as f32 * 0.5, (i / 7) as f32 * 0.3))
            .collect();
        let ix: Vec<u32> = (0..60).collect();
        let bvh = MeshBvh::build(&v, &ix, 2);
        let flat = flatten(&bvh);
        let mut seen = flat.leaf_tris.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..20).collect::<Vec<u32>>());
        assert_eq!(flat.parents[0], 0);
        for at in 0..flat.nodes.len() / 8 {
            let (a, b) = (flat.nodes[at * 8 + 3], flat.nodes[at * 8 + 7]);
            if a & 0x8000_0000 == 0 {
                assert_eq!(flat.parents[a as usize], at as u32);
                assert_eq!(flat.parents[b as usize], at as u32);
            }
        }
        assert!(flatten(&MeshBvh::build(&[], &[], 2)).nodes.is_empty());
    }
}
