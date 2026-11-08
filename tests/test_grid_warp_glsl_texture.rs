//! GLSL that reads a grid warp's maps and bounds from a texture at run time
//! (`GlslShader::transpile_with_node_texture`): a new map of the same layout is a new texture, not a
//! new shader.
//!
//! oracle: the CPU evaluation of the frame's tree (`eval`) at sample points, within a per-point
//! bound on the f32 rounding of the grid warp's expression (`tolerance`), and
//! `interval::eval_lipschitz` of that tree for the frame's bound. The shader is compiled once from
//! the first tree; the second tree's texels go into the same texture of the same pipeline and the
//! results must follow the second tree, not the first. The grid has more node data than one texture
//! row, so the index layout on the texture is exercised.
//!
//! The GLSL ES text (WebGL 2) is checked for its sampler and `texelFetch` here; it is compiled by
//! glslang in CI. The GPU run uses GLSL 4.50 (naga reads no GLSL ES) with the same body and the same
//! index layout (`GlslNodeTexture::reader`), reading the texture with `texelFetch`.
//!
//! Author: Moroya Sakamoto
#![cfg(all(feature = "glsl", feature = "gpu"))]

use alice_sdf::compiled::glsl::{
    GlslNodeTexture, GlslNodeTextureError, GlslShader, NODE_TEXTURE_WIDTH,
};
use alice_sdf::interval::eval_lipschitz;
use alice_sdf::prelude::*;
use alice_sdf::transforms::{GridWarp, GridWarpData};

const DIMS: [u32; 3] = [6, 6, 6];
const CELL: f32 = 0.5;
const ORIGIN: [f32; 3] = [-1.25, -1.25, -1.25];

/// A grid that turns points about y by `twist` · z and scales them by 1 + `twist` / 4 · (z − z₀) (a
/// different twist is a different map of the same layout, with another bound)
fn warp(twist: f32, dims: [u32; 3]) -> GridWarp {
    let mut maps = Vec::new();
    for k in 0..dims[2] {
        for _j in 0..dims[1] {
            for _i in 0..dims[0] {
                let z = ORIGIN[2] + k as f32 * CELL;
                let (s, c) = alice_det_math::sin_cos(twist * z);
                // a scale growing along z with the twist: the bound depends on the map
                let g = 1.0 + 0.25 * twist * (z - ORIGIN[2]);
                maps.extend_from_slice(&[
                    g * c,
                    0.0,
                    g * s,
                    0.0,
                    0.0,
                    g,
                    0.0,
                    0.0,
                    -g * s,
                    0.0,
                    g * c,
                    0.0,
                ]);
            }
        }
    }
    GridWarp::new(GridWarpData {
        origin: ORIGIN,
        cell: CELL,
        dims,
        maps,
        reach: 0.5,
        floor: Vec::new(),
    })
    .expect("a valid grid")
}

fn tree(twist: f32) -> SdfNode {
    SdfNode::box3d(BOX_HALF.x * 2.0, BOX_HALF.y * 2.0, BOX_HALF.z * 2.0)
        .grid_warp(warp(twist, DIMS))
}

fn points(n: usize) -> Vec<Vec3> {
    let mut state: u64 = 0x5eed_0000_0000_0041;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((state >> 40) as f32) / ((1u64 << 24) as f32)) * 3.6 - 1.8
    };
    (0..n).map(|_| Vec3::new(next(), next(), next())).collect()
}

#[test]
fn the_source_holds_no_map_value_and_the_data_spans_more_than_one_row() {
    let tex = GlslShader::transpile_with_node_texture(&tree(0.6), 450);
    assert!(tex.shader.source.contains("alice_node_at("));
    assert!(
        !tex.shader.source.contains("const float alice_gw"),
        "a grid warp array was written into the source"
    );
    let floats = DIMS.iter().product::<u32>() as usize * 13 + 1;
    assert!(tex.node_data.len() >= floats, "{}", tex.node_data.len());
    let (w, h) = tex.texture_size();
    assert_eq!(w, NODE_TEXTURE_WIDTH);
    assert!(
        h >= 2,
        "the data fits one row: the layout on the texture is not exercised"
    );
    assert_eq!(tex.texels().len(), (w * h) as usize);
}

#[test]
fn the_webgl2_source_reads_the_texture() {
    let es = GlslShader::transpile_with_node_texture(&tree(0.6), 300).es_source();
    assert!(es.contains("uniform highp sampler2D alice_node_tex;"));
    assert!(es.contains("texelFetch(alice_node_tex, ivec2(i % 2048, i / 2048), 0).r"));
    assert!(es.contains("float alice_node_at(int i)"));
}

#[test]
fn a_new_map_of_the_same_layout_is_a_new_frame_with_its_bound() {
    let tex = GlslShader::transpile_with_node_texture(&tree(0.6), 450);
    let b = tree(1.3);
    let frame = tex.frame(&b).expect("same layout");
    assert_eq!(frame.lipschitz, eval_lipschitz(&b));
    assert_ne!(
        frame.lipschitz,
        eval_lipschitz(&tree(0.6)),
        "the bound did not move"
    );
    assert_ne!(frame.texels, tex.texels(), "the texels did not move");
    // another grid size, another reach or another node is another shader
    let other_dims = SdfNode::box3d(0.75, 0.4, 0.9).grid_warp(warp(0.6, [6, 6, 7]));
    let other_child = SdfNode::sphere(0.8).grid_warp(warp(0.6, DIMS));
    for other in [other_dims, other_child] {
        assert_eq!(
            tex.frame(&other).err(),
            Some(GlslNodeTextureError::LayoutChanged)
        );
    }
}

/// A compute pipeline over GLSL 4.50 with the node data in an R32F texture
struct Gpu {
    ctx: alice_sdf::gpu_context::GpuContext,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    texture: wgpu::Texture,
    sampler: wgpu::Sampler,
    size: (u32, u32),
}

fn gpu_or_skip(source: &str, size: (u32, u32)) -> Option<Gpu> {
    let ctx = match alice_sdf::gpu_context::GpuContext::request("grid warp texture") {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "ALICE_SDF_REQUIRE_GPU is set but no GPU adapter was found: {e}"
            );
            eprintln!("skipping: {e}");
            return None;
        }
    };
    let device = ctx.device();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("grid warp texture"),
        source: wgpu::ShaderSource::Glsl {
            shader: source.into(),
            stage: wgpu::naga::ShaderStage::Compute,
            defines: wgpu::naga::FastHashMap::default(),
        },
    });
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    // R32F is not filterable without a feature: the texture is unfilterable and the sampler
    // non-filtering (texelFetch does not filter); WebGL 2 reads it the same way
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            storage(0, true),
            storage(1, false),
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
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("alice_node_tex"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
    Some(Gpu {
        ctx,
        pipeline,
        layout,
        texture,
        sampler,
        size,
    })
}

impl Gpu {
    fn upload(&self, texels: &[f32]) {
        self.ctx.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.size.0 * 4),
                rows_per_image: Some(self.size.1),
            },
            wgpu::Extent3d {
                width: self.size.0,
                height: self.size.1,
                depth_or_array_layers: 1,
            },
        );
    }

    fn eval(&self, pts: &[Vec3]) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let input: Vec<f32> = pts.iter().flat_map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let n = u32::try_from(pts.len()).unwrap();
        let inb = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let size = u64::from(n) * 4;
        let outb = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let countb = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[n, 0, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let readb = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let view = self
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let bind = self
            .ctx
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: inb.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: outb.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: countb.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
        let mut enc = self
            .ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&outb, 0, &readb, 0, size);
        self.ctx.queue().submit([enc.finish()]);
        let slice = readb.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.ctx.device().poll(wgpu::Maintain::Wait);
        let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        readb.unmap();
        out
    }
}

fn compute_wrapper(tex: &GlslNodeTexture) -> String {
    format!(
        r"#version 450
layout(local_size_x = 64) in;
layout(std430, set = 0, binding = 0) readonly buffer InputPoints {{ vec4 input_points[]; }};
layout(std430, set = 0, binding = 1) buffer OutputDistances {{ float output_distances[]; }};
layout(std140, set = 0, binding = 2) uniform PointCount {{ uint point_count; }};
layout(set = 0, binding = 3) uniform texture2D alice_node_t;
layout(set = 0, binding = 4) uniform sampler alice_node_s;
{}
{}
void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= point_count) {{ return; }}
    output_distances[idx] = sdf_eval(input_points[idx].xyz);
}}
",
        GlslNodeTexture::reader(|c| format!(
            "texelFetch(sampler2D(alice_node_t, alice_node_s), {c}, 0).r"
        )),
        tex.shader.source
    )
}

/// Half extents of the warped box (the child of every frame)
const BOX_HALF: Vec3 = Vec3::new(0.375, 0.2, 0.45);

/// The unit roundoff of f32
const U: f32 = f32::EPSILON / 2.0;

/// A bound on |GPU − CPU| at `p` for the grid warp over the box, from the f32 evaluation of the
/// same expression on both sides (standard model: each `+ − ×` within `U` relative, as Vulkan
/// rounds them; a GPU division within 2.5 ulp = `5U`, a GPU `sqrt` taken as `1 / inversesqrt`
/// within 2 + 2.5 ulp = `9U`; the CPU rounds each correctly). Every magnitude is this point's: its
/// cell coordinate, the 8 corner maps of its cell and how much they differ, its warped point.
/// Terms, in the evaluation order of `transforms::grid_warp`:
/// 1. the cell coordinate `g = (q − b₀)·(1/h)`: 2 roundings of `|g|`; the fraction `f = g − ⌊g⌋`
///    is exact ⇒ `Δf ≤ 2U·|g|∞`
/// 2. the corner weights: a shift `Δf` moves weight between opposite corners, which changes a
///    blended entry by at most `3·D·Δf` (D = the largest difference of an entry between two
///    corners of the cell); rounding `1 − f` and the 2 products: `3U` per weight
/// 3. a blended entry `Σ₈ w·m` (8 products, 7 sums, Σw = 1):
///    `ΔM ≤ 3DΔf + m·(8·3U + 15U)` (m = the largest entry magnitude at the cell's corners); the
///    blended bound Λ alike, with its corner values
/// 4. the warped point `x = M q + t`, per axis: `Δx ≤ (3|q|∞ + 1)ΔM + 6U·X`
///    (X = `Σ|Mᵢ qᵢ| + |t|`)
/// 5. the box at `x` (1-Lipschitz, `√3·Δx` from the point; `|x| − h`, the length of 3 squares
///    summed, its `sqrt` and the outer sum: `(1 + 5 + 9 + 1)U·C`, C = |x|₂ + |h|₂)
/// 6. `c / Λ`: `Δc/Λ + |c|ΔΛ/Λ² + 5U|c/Λ|`; the clamps are exact
/// 7. the extension and clip outside the box: `e = |p − q|` (`(5 + 9)U·e`), `sqrt(e² + d⁺²)`
///    (`(3 + 9)U·r`, r = that root) and 2 sums (`2U·(|d| + r)`)
///
/// Both sides round, so the bound is twice one side's. It assumes every rounding error has the
/// worst sign at once, so it is far above the differences seen (the test prints the ratio).
fn tolerance(warp: &GridWarp, p: Vec3) -> f32 {
    let data = warp.data();
    let (x, b, lam, _) = warp.warp(p);
    let q = p.clamp(warp.box_min(), warp.box_max());
    let g = (q - warp.box_min()) / data.cell;
    let [nx, ny, nz] = data.dims.map(|n| n as usize);
    let cell = |v: f32, n: usize| (v.floor().max(0.0) as usize).min(n - 2);
    let (i0, j0, k0) = (cell(g.x, nx), cell(g.y, ny), cell(g.z, nz));
    let corners: Vec<usize> = (0..8)
        .map(|c| (i0 + (c & 1)) + nx * ((j0 + ((c >> 1) & 1)) + ny * (k0 + ((c >> 2) & 1))))
        .collect();
    let entry = |n: usize, t: usize| data.maps[n * 12 + t];
    let mut m = 0.0f32;
    let mut d = 0.0f32;
    for t in 0..12 {
        let vals: Vec<f32> = corners.iter().map(|&n| entry(n, t)).collect();
        let (lo, hi) = vals
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
                (a.min(*v), b.max(*v))
            });
        m = m.max(lo.abs()).max(hi.abs());
        d = d.max(hi - lo);
    }
    let lams: Vec<f32> = corners.iter().map(|&n| warp.node_lambda()[n]).collect();
    let lam_max = lams.iter().fold(0.0f32, |a, v| a.max(*v));
    let lam_d = lams.iter().fold(0.0f32, |a, v| a.max(*v))
        - lams.iter().fold(f32::INFINITY, |a, v| a.min(*v));
    let df = 2.0 * U * g.abs().max_element();
    let dm = 3.0 * d * df + m * (24.0 * U + 15.0 * U);
    let dlam = 3.0 * lam_d * df + lam_max * (24.0 * U + 15.0 * U);
    let qn = q.abs().max_element();
    let xs = 3.0 * m * qn + m;
    let dx = (3.0 * qn + 1.0) * dm + 6.0 * U * xs;
    let c =
        (x.abs() - BOX_HALF).max(Vec3::ZERO).length() + (x.abs() - BOX_HALF).max_element().min(0.0);
    let cc = x.length() + BOX_HALF.length();
    let dc = 3f32.sqrt() * dx + 16.0 * U * cc;
    let v = c / lam;
    let dv = dc / lam + c.abs() * dlam / (lam * lam) + 5.0 * U * v.abs();
    let e = (p - q).length();
    let dp = v.max(0.0);
    let r = (e * e + dp * dp).sqrt();
    let ext = 14.0 * U * e + 12.0 * U * r + 2.0 * U * (v.abs() + r) + 2.0 * U * b.abs();
    2.0 * (dv + ext)
}

/// The largest |GPU − CPU| and the largest ratio of it to [`tolerance`] over the points
fn compare(got: &[f32], node: &SdfNode, warp: &GridWarp, pts: &[Vec3]) -> (f32, f32, Vec3) {
    let mut out = (0.0f32, 0.0f32, Vec3::ZERO);
    for (g, p) in got.iter().zip(pts) {
        let diff = (g - eval(node, *p)).abs();
        let ratio = diff / tolerance(warp, *p);
        out.0 = out.0.max(diff);
        if ratio > out.1 {
            out = (out.0, ratio, *p);
        }
    }
    out
}

#[test]
fn one_shader_follows_the_texture_of_each_frame() {
    let (a, b) = (tree(0.6), tree(1.3));
    let tex = GlslShader::transpile_with_node_texture(&a, 450);
    let Some(gpu) = gpu_or_skip(&compute_wrapper(&tex), tex.texture_size()) else {
        return;
    };
    let pts = points(4096);
    gpu.upload(&tex.texels());
    let got_a = gpu.eval(&pts);
    let (da, ra, pa) = compare(&got_a, &a, &warp(0.6, DIMS), &pts);
    eprintln!("frame a: largest |GPU − CPU| {da:e}, largest ratio to the bound {ra:.3}");
    assert!(ra <= 1.0, "frame a: {ra} of the bound at {pa}");
    gpu.upload(&tex.frame(&b).expect("same layout").texels);
    let got_b = gpu.eval(&pts);
    let (db, rb, pb) = compare(&got_b, &b, &warp(1.3, DIMS), &pts);
    eprintln!("frame b: largest |GPU − CPU| {db:e}, largest ratio to the bound {rb:.3}");
    assert!(rb <= 1.0, "frame b: {rb} of the bound at {pb}");
    // the second frame is not the first: the change is observed
    let moved = got_a
        .iter()
        .zip(&got_b)
        .filter(|(x, y)| (*x - *y).abs() > 1e-3)
        .count();
    assert!(
        moved > pts.len() / 10,
        "only {moved} of {} values moved",
        pts.len()
    );
}

/// The bound's slack hides no real error: one map entry of one node moved by `DELTA` in the
/// texture (the translation along x of a node in the middle of the box, about 840 ulp of 1, the
/// scale of the scene) puts some point near that node above the bound
#[test]
fn one_node_off_by_a_small_step_is_above_the_bound() {
    const DELTA: f32 = 1e-4;
    let a = tree(0.6);
    let w = warp(0.6, DIMS);
    let tex = GlslShader::transpile_with_node_texture(&a, 450);
    let Some(gpu) = gpu_or_skip(&compute_wrapper(&tex), tex.texture_size()) else {
        return;
    };
    let [nx, ny, _] = DIMS.map(|n| n as usize);
    let (i, j, k) = (3usize, 3usize, 3usize);
    let node = i + nx * (j + ny * k);
    let mut texels = tex.texels();
    // the maps are the first array of the node data (12 floats per node, x fastest)
    assert_eq!(texels[node * 12 + 3], w.data().maps[node * 12 + 3]);
    texels[node * 12 + 3] += DELTA;
    gpu.upload(&texels);
    let centre = Vec3::from(ORIGIN) + Vec3::new(i as f32, j as f32, k as f32) * CELL;
    let near: Vec<Vec3> = points(512)
        .into_iter()
        .map(|p| centre + p * (0.4 * CELL / 1.8))
        .collect();
    let got = gpu.eval(&near);
    let (diff, ratio, at) = compare(&got, &a, &w, &near);
    eprintln!("one node off by {DELTA:e}: largest |GPU − CPU| {diff:e}, ratio to the bound {ratio:.2} at {at}");
    assert!(
        ratio > 1.0,
        "a {DELTA:e} error at one node stays under the bound ({ratio})"
    );
}
