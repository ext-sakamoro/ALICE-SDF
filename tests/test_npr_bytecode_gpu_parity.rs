//! GPU ↔ CPU parity for the NPR colour bytecode
//! (`npr::compiled_color::{CompiledColorPipeline::serialize,
//! emit_wgsl_bytecode_evaluator}`).
//!
//! The colour laws are defined once on the CPU (`npr::toon`, `npr::hatch`,
//! `npr::motion`, `npr::composition`, `npr::palette`); the shader-library
//! helpers in `npr::shader_glue` follow them line by line. The WGSL stack
//! machine returned by `emit_wgsl_bytecode_evaluator` must evaluate a
//! serialised program to the same colour as the CPU tree walker
//! (`NprColorNode::eval`) does for the tree it came from. This test runs the
//! emitted evaluator in a compute shader over a sweep of shading contexts and
//! compares every program of a corpus that uses each of the 18 opcodes and
//! each of the 5 palette sources.
//!
//! Several laws are step functions (toon bands, two-tone, posterise, bloom
//! threshold, hatch and speed lines): next to a step a 1-ulp difference in
//! `sin` / `atan2` between CPU and GPU legitimately picks the other side.
//! A context is compared only when the CPU result does not move under a
//! ±1e-3 perturbation of every input scalar; the test fails if fewer than
//! 60 % of the contexts of a program are compared.
//!
//! CI's gpu-parity job (lavapipe) sets ALICE_SDF_REQUIRE_GPU=1 so a missing
//! adapter is a failure there; elsewhere it is a skip.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
#![cfg(feature = "gpu")]

use alice_sdf::npr::compiled_color::{emit_wgsl_bytecode_evaluator, CompiledColorPipeline};
use alice_sdf::npr::dsl::{NprColorContext, NprColorNode, PaletteSource};
use glam::{Vec2, Vec3};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

const TOL: f32 = 2e-4;
const PERTURB: f32 = 1e-3;

/// The six context scalars the bytecode evaluator reads.
#[derive(Clone, Copy, Debug)]
struct Scalars {
    n_dot_l: f32,
    n_dot_v: f32,
    sdf: f32,
    uv: Vec2,
    time: f32,
}

impl Scalars {
    /// A CPU context whose `n . l` / `n . v` are exactly these scalars
    /// (normal = +Z, light and view carry the value in z).
    const fn ctx(self) -> NprColorContext {
        NprColorContext {
            sdf: self.sdf,
            normal: Vec3::Z,
            view: Vec3::new(0.0, 0.0, self.n_dot_v),
            light: Vec3::new(0.0, 0.0, self.n_dot_l),
            uv: self.uv,
            time: self.time,
        }
    }

    const fn as_array(self) -> [f32; 6] {
        [
            self.n_dot_l,
            self.n_dot_v,
            self.sdf,
            self.uv.x,
            self.uv.y,
            self.time,
        ]
    }

    const fn from_array(a: [f32; 6]) -> Self {
        Self {
            n_dot_l: a[0],
            n_dot_v: a[1],
            sdf: a[2],
            uv: Vec2::new(a[3], a[4]),
            time: a[5],
        }
    }
}

fn contexts(n: usize) -> Vec<Scalars> {
    let mut state: u64 = 0x0b17_e00d_5eed_0001;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    };
    (0..n)
        .map(|_| Scalars {
            n_dot_l: next().mul_add(2.4, -1.2),
            n_dot_v: next().mul_add(2.4, -1.2),
            sdf: next().mul_add(3.0, -1.5),
            uv: Vec2::new(next(), next()),
            time: next().mul_add(10.0, -2.0),
        })
        .collect()
}

const fn c(r: f32, g: f32, b: f32) -> NprColorNode {
    NprColorNode::Constant(Vec3::new(r, g, b))
}

const fn palette3(source: PaletteSource) -> NprColorNode {
    NprColorNode::Palette3 {
        source,
        c0: Vec3::new(0.1, 0.2, 0.9),
        c1: Vec3::new(0.9, 0.8, 0.1),
        c2: Vec3::new(0.2, 0.9, 0.3),
    }
}

/// Programs covering every opcode and every palette source.
fn corpus() -> Vec<(&'static str, NprColorNode)> {
    let shadow = Vec3::new(0.15, 0.1, 0.3);
    let light = Vec3::new(0.95, 0.85, 0.7);
    vec![
        (
            "toon 4 bands",
            NprColorNode::Toon {
                shadow,
                light,
                bands: 4,
            },
        ),
        (
            "toon 1 band",
            NprColorNode::Toon {
                shadow,
                light,
                bands: 1,
            },
        ),
        (
            "soft toon",
            NprColorNode::SoftToon {
                shadow,
                light,
                bands: 3,
                smoothness: 0.08,
            },
        ),
        (
            "two tone",
            NprColorNode::TwoTone {
                shadow,
                light,
                threshold: 0.45,
            },
        ),
        (
            "multiply plus scale",
            c(0.8, 0.6, 0.4)
                .multiply(c(0.5, 0.9, 0.7))
                .plus(c(0.05, 0.1, 0.0))
                .scale(1.3),
        ),
        (
            "outline over",
            c(0.7, 0.7, 0.2).with_outline(Vec3::new(0.0, 0.0, 0.1), 0.6),
        ),
        (
            "fresnel",
            c(0.2, 0.3, 0.8).with_fresnel(Vec3::new(1.0, 0.9, 0.8), 2.5),
        ),
        (
            "saturate",
            NprColorNode::Saturate {
                child: Box::new(c(0.9, 0.3, 0.2)),
                factor: 1.6,
            },
        ),
        (
            "bloom over a toon base",
            NprColorNode::Toon {
                shadow,
                light,
                bands: 3,
            }
            .bloom(0.6, 1.8),
        ),
        (
            "posterize a palette",
            palette3(PaletteSource::NDotL).posterize(4),
        ),
        ("vignette", c(0.9, 0.9, 0.9).vignetted(0.3, 0.2)),
        ("palette3 n.v", palette3(PaletteSource::NDotV)),
        ("palette3 sdf", palette3(PaletteSource::Sdf)),
        ("palette3 uv.y", palette3(PaletteSource::UvY)),
        ("palette3 time", palette3(PaletteSource::TimeCycle)),
        (
            "palette5 n.l",
            NprColorNode::Palette5 {
                source: PaletteSource::NDotL,
                c0: Vec3::new(0.0, 0.0, 0.2),
                c1: Vec3::new(0.3, 0.0, 0.5),
                c2: Vec3::new(0.9, 0.2, 0.3),
                c3: Vec3::new(1.0, 0.7, 0.2),
                c4: Vec3::new(1.0, 1.0, 0.9),
            },
        ),
        (
            "hatch",
            c(0.95, 0.92, 0.85).with_hatch(0.6, 7.0, 0.15, Vec3::new(0.1, 0.1, 0.15)),
        ),
        (
            "speed lines",
            c(1.0, 1.0, 1.0).with_speed_lines(Vec2::new(0.5, 0.45), 24, 0.2, Vec3::ZERO),
        ),
        ("tonemap", c(2.0, 0.8, 5.0).tonemap_reinhard(1.5)),
    ]
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

const WORKGROUP: u32 = 64;

fn shader_source() -> String {
    format!(
        r"{evaluator}

@group(0) @binding(0) var<storage, read> program: array<u32>;
@group(0) @binding(1) var<storage, read> contexts: array<f32>;
@group(0) @binding(2) var<storage, read_write> colors: array<f32>;
@group(0) @binding(3) var<uniform> sizes: vec4<u32>;

fn alice_npr_load(index: u32) -> u32 {{
    return program[index];
}}

@compute @workgroup_size({WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    if (i >= sizes.y) {{ return; }}
    let b = i * 6u;
    let ctx = AliceNprBytecodeCtx(
        contexts[b],
        contexts[b + 1u],
        contexts[b + 2u],
        vec2<f32>(contexts[b + 3u], contexts[b + 4u]),
        contexts[b + 5u],
    );
    let color = alice_npr_eval_bytecode(sizes.x, ctx);
    colors[i * 3u] = color.x;
    colors[i * 3u + 1u] = color.y;
    colors[i * 3u + 2u] = color.z;
}}
",
        evaluator = emit_wgsl_bytecode_evaluator()
    )
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
        eprintln!("skipping NPR bytecode GPU parity: no adapter");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("npr bytecode parity"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .expect("request device");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("npr bytecode evaluator"),
        source: wgpu::ShaderSource::Wgsl(shader_source().into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("npr bytecode evaluator"),
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
    fn run(&self, words: &[u32], ctxs: &[Scalars]) -> Vec<Vec3> {
        let dev = &self.device;
        let flat: Vec<f32> = ctxs.iter().flat_map(|s| s.as_array()).collect();
        let program = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("program"),
            contents: bytemuck::cast_slice(words),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let contexts = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("contexts"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_bytes = (ctxs.len() * 3 * 4) as u64;
        let colors = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("colors"),
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
        let sizes = [words.len() as u32, ctxs.len() as u32, 0, 0];
        let sizes = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sizes"),
            contents: bytemuck::cast_slice(&sizes),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("npr bytecode"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: program.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: contexts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: colors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sizes.as_entire_binding(),
                },
            ],
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((ctxs.len() as u32).div_ceil(WORKGROUP), 1, 1);
        }
        enc.copy_buffer_to_buffer(&colors, 0, &staging, 0, out_bytes);
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
        data.chunks_exact(3)
            .map(|v| Vec3::new(v[0], v[1], v[2]))
            .collect()
    }
}

/// True when the CPU colour does not move under a small perturbation of
/// every input scalar, i.e. the context is not next to a step of a law.
fn stable(node: &NprColorNode, s: Scalars, at: Vec3) -> bool {
    let base = s.as_array();
    for k in 0..6 {
        for sign in [-1.0f32, 1.0] {
            let mut a = base;
            a[k] += sign * PERTURB * a[k].abs().max(1.0);
            let moved = node.eval(&Scalars::from_array(a).ctx());
            if (moved - at).abs().max_element() > 0.05 {
                return false;
            }
        }
    }
    true
}

#[test]
fn bytecode_evaluator_on_the_gpu_matches_the_cpu_tree_walker() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let ctxs = contexts(1024);
    let mut total = 0usize;
    for (name, node) in corpus() {
        let program = CompiledColorPipeline::compile(&node)
            .serialize()
            .expect("corpus programs have no Fallback");
        let got = gpu.run(program.as_words(), &ctxs);
        assert_eq!(got.len(), ctxs.len());
        let mut compared = 0usize;
        let mut worst = (0.0f32, Vec3::ZERO, Vec3::ZERO, None);
        for (s, g) in ctxs.iter().zip(&got) {
            let cpu = node.eval(&s.ctx());
            if !stable(&node, *s, cpu) {
                continue;
            }
            compared += 1;
            let d = (cpu - *g).abs().max_element();
            if d.is_nan() || d > worst.0 {
                worst = (d, cpu, *g, Some(*s));
            }
        }
        assert!(
            compared * 10 >= ctxs.len() * 6,
            "{name}: only {compared} of {} contexts are away from a step",
            ctxs.len()
        );
        assert!(
            worst.0 <= TOL,
            "{name}: GPU {:?} vs CPU {:?} (|d| = {:e}) at {:?}",
            worst.2,
            worst.1,
            worst.0,
            worst.3
        );
        eprintln!(
            "{name}: max |gpu - cpu| = {:.2e} over {compared} contexts",
            worst.0
        );
        total += compared;
    }
    assert!(total > 10_000, "compared {total} contexts in all");
}
