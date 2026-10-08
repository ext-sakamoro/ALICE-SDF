//! `InstancedSdf::to_instanced_wgsl`: the compute shader parses and validates
//! with naga, and dispatched on a GPU it returns the CPU
//! `InstancedSdf::eval_min` for translated, rotated and scaled instances.
//!
//! CI's gpu-parity job (lavapipe) sets ALICE_SDF_REQUIRE_GPU=1 so a missing
//! adapter fails instead of skipping.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
#![cfg(feature = "gpu")]

mod common;

use alice_sdf::animation::AnimationParams;
use alice_sdf::compiled::{CompiledSdf, InstancedSdf};
use alice_sdf::prelude::*;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

/// GPU `sin` / `cos` / `sqrt` are not correctly rounded; the shader rebuilds
/// the rotation from Euler angles while the CPU uses a quaternion.
const TOL: f32 = 2e-4;

fn base() -> SdfNode {
    SdfNode::box3d(0.5, 0.3, 0.2).smooth_union(SdfNode::sphere(0.2).translate(0.2, 0.1, 0.0), 0.05)
}

fn instances() -> Vec<AnimationParams> {
    (0..13)
        .map(|i| {
            let f = i as f32;
            AnimationParams {
                translate_x: (f * 1.1).cos() * 1.8,
                translate_y: (f * 0.5).sin(),
                translate_z: f * 0.15 - 1.0,
                rotate_x: f * 0.37,
                rotate_y: -0.21 * f,
                rotate_z: 0.5 + f * 0.11,
                scale: if i % 3 == 0 { 1.0 } else { 0.6 + f * 0.04 },
                ..Default::default()
            }
        })
        .collect()
}

fn shader() -> String {
    InstancedSdf::to_instanced_wgsl(&base())
}

#[test]
fn instanced_wgsl_parses_and_validates() {
    let src = shader();
    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
        panic!(
            "instanced WGSL does not parse: {}\n{src}",
            e.emit_to_string(&src)
        )
    });
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("instanced WGSL does not validate: {e:?}"));
    assert!(module.entry_points.iter().any(|e| e.name == "main"));
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

fn gpu_or_skip() -> Option<Gpu> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));
    let Some(adapter) = adapter else {
        assert!(
            std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
            "ALICE_SDF_REQUIRE_GPU is set but no GPU adapter was found"
        );
        eprintln!("skipping instanced WGSL GPU parity: no adapter");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("instanced parity"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .expect("request device");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("instanced sdf"),
        source: wgpu::ShaderSource::Wgsl(shader().into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("instanced sdf"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    Some(Gpu {
        device,
        queue,
        pipeline,
    })
}

impl Gpu {
    fn run(&self, inst: &[AnimationParams], pts: &[Vec3]) -> Vec<f32> {
        let dev = &self.device;
        let flat: Vec<f32> = inst
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
        let storage = |label: &str, data: &[f32]| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let instances = storage("instances", &flat);
        let xs: Vec<f32> = pts.iter().map(|p| p.x).collect();
        let ys: Vec<f32> = pts.iter().map(|p| p.y).collect();
        let zs: Vec<f32> = pts.iter().map(|p| p.z).collect();
        let (bx, by, bz) = (storage("x", &xs), storage("y", &ys), storage("z", &zs));
        let out_bytes = (pts.len() * 4) as u64;
        let distances = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("distances"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let counts = [pts.len() as u32, inst.len() as u32];
        let counts = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("counts"),
            contents: bytemuck::cast_slice(&counts),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let entries: Vec<wgpu::BindGroupEntry> = [&instances, &bx, &by, &bz, &distances, &counts]
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("instanced"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((pts.len() as u32).div_ceil(256), 1, 1);
        }
        enc.copy_buffer_to_buffer(&distances, 0, &staging, 0, out_bytes);
        self.queue.submit(std::iter::once(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map callback").expect("map buffer");
        let data: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        data
    }
}

#[test]
fn instanced_wgsl_on_the_gpu_matches_eval_min() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let params = instances();
    let mut inst = InstancedSdf::new(CompiledSdf::compile(&base()));
    for a in &params {
        inst.add_instance(*a);
    }
    let pts: Vec<Vec3> = common::test_grid_points(17)
        .into_iter()
        .map(|p| p * 2.5)
        .collect();
    let gpu_d = gpu.run(&params, &pts);
    assert_eq!(gpu_d.len(), pts.len());
    let mut worst = 0.0f32;
    let mut compared = 0usize;
    for (p, g) in pts.iter().zip(&gpu_d) {
        let c = inst.eval_min(*p);
        let err = (g - c).abs() / c.abs().max(1.0);
        worst = worst.max(err);
        assert!(err <= TOL, "{p}: gpu {g} cpu {c}");
        compared += 1;
    }
    eprintln!("instanced WGSL vs eval_min: {compared} points, worst rel err {worst:e}");
    assert!(compared > 4_000, "compared {compared}");
}
