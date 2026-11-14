//! GPU evaluator and marcher for [`SheetField`] (WGSL compute), the same arithmetic and the same BVH walks as
//! the CPU reference.
//!
//! Data: storage buffers for the sheets (box, coarsest cell, level range), the levels (origin, cell, bricks per
//! axis, offsets of its brick index and of each channel's rows / bytes / scale-offset pairs), one `u32` table
//! holding the dense brick indices, the channel rows and the flattened sheet-box BVH (nodes, leaf entries,
//! parent links; see [`crate::mesh::gpu_bvh`]) at bases passed in the uniforms (a device may bind only 8
//! storage buffers to a stage), the channel bytes packed four to a `u32` (WGSL has no byte storage) and the
//! scale-offset pairs. One invocation per point (evaluation) or per ray (march).
//!
//! The union is the BVH walk of [`SheetField::eval`]; the march keeps, per ray, the previous signed value of
//! each sheet whose box holds the current point (found by the BVH walk of [`SheetField::sheets_containing`])
//! in a list compiled per field: [`max_live_for`] of [`live_bound`] entries, at least [`MAX_LIVE_SHEETS`].
//! The bound: a point held by the boxes of a set S of sheets lies in every box of S, so every box of S meets
//! every other box of S (inclusive intersection of the same `f32` boxes the shader tests), and |S| is at most
//! the count of boxes meeting any one of them, itself included; [`live_bound`] is the largest such count over
//! the sheets. The list length is not in the arithmetic, so a field whose bound is at most 64 compiles the same
//! program as the earlier fixed 64 list and gives the same results. A ray that would still need more (it
//! cannot, by the bound) reports [`GpuError`] for the batch (never a guessed answer); a bound over
//! [`MAX_LIVE_CAP`] is refused at upload.

use glam::Vec3;
use wgpu::util::DeviceExt;

use super::{Hit, MarchResult, SheetField, ValidRule, BRICK_NODES};
use crate::compiled::GpuError;
use crate::gpu_context::GpuContext;

/// The smallest per-ray list of live sheets the march compiles (fields whose [`live_bound`] is at most this).
pub const MAX_LIVE_SHEETS: u32 = 64;
/// The largest per-ray list the march compiles; a field whose [`live_bound`] is over it is refused.
pub const MAX_LIVE_CAP: u32 = 1024;

/// The most sheet boxes that can hold one point of `field`.
///
/// Over the sheets, the count of boxes meeting that sheet's box (inclusive, itself included). See the module docs for why no point is in more boxes.
#[must_use]
pub fn live_bound(field: &SheetField) -> u32 {
    let s = &field.sheets;
    let mut order: Vec<usize> = (0..s.len()).collect();
    order.sort_by(|&a, &b| s[a].lo.x.total_cmp(&s[b].lo.x));
    let mut count = vec![1u32; s.len()];
    // sweep in x: a pair can meet only if their x ranges overlap
    for (i, &a) in order.iter().enumerate() {
        for &b in &order[i + 1..] {
            if s[b].lo.x > s[a].hi.x {
                break;
            }
            if s[a].lo.cmple(s[b].hi).all() && s[b].lo.cmple(s[a].hi).all() {
                count[a] += 1;
                count[b] += 1;
            }
        }
    }
    count.into_iter().max().unwrap_or(0)
}

/// The list size compiled for a bound.
///
/// [`MAX_LIVE_SHEETS`] up to it, then the next power of two up to [`MAX_LIVE_CAP`] (at most 5 programs per process: 64, 128, 256, 512, 1024). Refuses a bound over the cap.
pub fn max_live_for(bound: u32) -> Result<u32, GpuError> {
    if bound > MAX_LIVE_CAP {
        return Err(GpuError::InvalidFormat(format!(
            "sheet field: {bound} sheet boxes can hold one point, over the march's cap {MAX_LIVE_CAP}"
        )));
    }
    Ok(bound.max(MAX_LIVE_SHEETS).next_power_of_two())
}

/// The march program's text for a list of `max_live` live sheets.
#[must_use]
pub fn march_shader(max_live: u32) -> String {
    SHADER
        .replace("__MAX_LIVE__", &max_live.to_string())
        .replace("__MIN_STEP__", &format!("{:?}", super::MIN_STEP))
}
const WG: u32 = 64;

/// What the GPU says at one point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuSample {
    /// Union unsigned distance.
    pub f: f32,
    /// Union signed companion.
    pub g: f32,
    /// Whether the minimizing sheet's value was replaced by the validity rule.
    pub masked: bool,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SheetInfo {
    lo: [f32; 3],
    hc: f32,
    hi: [f32; 3],
    level_start: u32,
    level_count: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LevelInfo {
    lo: [f32; 3],
    h: f32,
    nb: [u32; 3],
    index_off: u32,
    nch: u32,
    _pad: [u32; 3],
    rows_off: [u32; 8],
    q_off: [u32; 8],
    sc_off: [u32; 8],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    n_sheets: u32,
    rule: u32,
    count: u32,
    steps: u32,
    dir: [f32; 3],
    tmax: f32,
    index_base: u32,
    rows_base: u32,
    bn_base: u32,
    bl_base: u32,
    bp_base: u32,
    bn_len: u32,
    _pad: [u32; 2],
}

/// Where each table starts in the merged `u32` buffer.
#[derive(Clone, Copy)]
struct Bases {
    index: u32,
    rows: u32,
    bn: u32,
    bl: u32,
    bp: u32,
    bn_len: u32,
}

/// A [`SheetField`] uploaded to the GPU.
pub struct GpuSheetField {
    context: GpuContext,
    sheets: wgpu::Buffer,
    levels: wgpu::Buffer,
    ix: wgpu::Buffer,
    qwords: wgpu::Buffer,
    scoff: wgpu::Buffer,
    bases: Bases,
    n_sheets: u32,
    rule: u32,
    eval_pipeline: wgpu::ComputePipeline,
    march_pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    max_live: u32,
}

fn storage_init(device: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    let data: Vec<u8> = if bytes.is_empty() {
        vec![0u8; 4]
    } else {
        bytes.to_vec()
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &data,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

const fn entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
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

/// The largest storage binding a sheet field can use on this machine's GPU, in bytes.
///
/// The adapter's limits as [`GpuContext::request_with_adapter_storage_limits`] requests them (the
/// smaller of the binding and buffer sizes). A chunk of sheets whose channel bytes exceed it cannot be uploaded.
pub fn storage_binding_limit() -> Result<u64, GpuError> {
    let context = GpuContext::request_with_adapter_storage_limits("ALICE-SDF sheet field limits")?;
    let l = context.device().limits();
    Ok(u64::from(l.max_storage_buffer_binding_size).min(l.max_buffer_size))
}

impl GpuSheetField {
    /// Upload `field` and build the pipelines.
    pub fn new(field: &SheetField) -> Result<Self, GpuError> {
        let mut sheets = Vec::new();
        let mut levels = Vec::new();
        let (mut index, mut rows, mut qbytes, mut scoff) = (
            Vec::<i32>::new(),
            Vec::<i32>::new(),
            Vec::<u8>::new(),
            Vec::<f32>::new(),
        );
        for sh in &field.sheets {
            sheets.push(SheetInfo {
                lo: sh.lo.to_array(),
                hc: sh.hc,
                hi: sh.hi.to_array(),
                level_start: levels.len() as u32,
                level_count: sh.levels.len() as u32,
                _pad: [0; 3],
            });
            for lv in &sh.levels {
                if !matches!(lv.channels.len(), 2 | 4 | 7) {
                    return Err(GpuError::InvalidFormat(format!(
                        "sheet field level with {} channels",
                        lv.channels.len()
                    )));
                }
                let mut li = LevelInfo {
                    lo: lv.lo.to_array(),
                    h: lv.h,
                    nb: lv.nb,
                    index_off: index.len() as u32,
                    nch: lv.channels.len() as u32,
                    _pad: [0; 3],
                    rows_off: [0; 8],
                    q_off: [0; 8],
                    sc_off: [0; 8],
                };
                index.extend_from_slice(&lv.index);
                for (c, ch) in lv.channels.iter().enumerate() {
                    li.rows_off[c] = rows.len() as u32;
                    rows.extend_from_slice(&ch.rows);
                    li.q_off[c] = qbytes.len() as u32;
                    debug_assert_eq!(ch.q.len(), ch.scale.len() * BRICK_NODES);
                    qbytes.extend_from_slice(&ch.q);
                    li.sc_off[c] = (scoff.len() / 2) as u32;
                    for (s, o) in ch.scale.iter().zip(&ch.offset) {
                        scoff.push(*s);
                        scoff.push(*o);
                    }
                }
                levels.push(li);
            }
        }
        while qbytes.len() % 4 != 0 {
            qbytes.push(0);
        }
        // the merged u32 table: brick indices, channel rows, BVH nodes, leaf entries, parent links
        let bvh = field.index();
        let mut ix: Vec<u32> = Vec::with_capacity(
            index.len() + rows.len() + bvh.nodes.len() + bvh.leaf_entries.len() + bvh.parents.len(),
        );
        let base = |v: &Vec<u32>| v.len() as u32;
        let b_index = base(&ix);
        ix.extend(index.iter().map(|&x| x as u32));
        let b_rows = base(&ix);
        ix.extend(rows.iter().map(|&x| x as u32));
        let b_bn = base(&ix);
        ix.extend_from_slice(&bvh.nodes);
        let b_bl = base(&ix);
        ix.extend_from_slice(&bvh.leaf_entries);
        let b_bp = base(&ix);
        ix.extend_from_slice(&bvh.parents);
        let bases = Bases {
            index: b_index,
            rows: b_rows,
            bn: b_bn,
            bl: b_bl,
            bp: b_bp,
            bn_len: bvh.nodes.len() as u32,
        };
        let max_live = max_live_for(live_bound(field))?;
        let context = GpuContext::request_with_adapter_storage_limits("ALICE-SDF sheet field")?;
        let device = context.device();
        let max = u64::from(device.limits().max_storage_buffer_binding_size);
        for (name, len) in [
            ("bytes", qbytes.len()),
            ("index table", ix.len() * 4),
            ("scale-offset", scoff.len() * 4),
        ] {
            if len as u64 > max {
                return Err(GpuError::InvalidFormat(format!(
                    "sheet field {name} buffer {len} B exceeds the device limit {max} B"
                )));
            }
        }
        let b_sheets = storage_init(device, "sheet field sheets", bytemuck::cast_slice(&sheets));
        let b_levels = storage_init(device, "sheet field levels", bytemuck::cast_slice(&levels));
        let b_ix = storage_init(device, "sheet field index table", bytemuck::cast_slice(&ix));
        let b_q = storage_init(device, "sheet field bytes", &qbytes);
        let b_sc = storage_init(
            device,
            "sheet field scale-offset",
            bytemuck::cast_slice(&scoff),
        );
        use wgpu::BufferBindingType as T;
        let ro = T::Storage { read_only: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sheet field layout"),
            entries: &[
                entry(0, T::Uniform),
                entry(1, ro),
                entry(2, ro),
                entry(3, ro),
                entry(4, ro),
                entry(5, ro),
                entry(6, ro),
                entry(7, T::Storage { read_only: false }),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sheet field"),
            source: wgpu::ShaderSource::Wgsl(march_shader(max_live).into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sheet field"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipe = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let eval_pipeline = pipe("eval_main");
        let march_pipeline = pipe("march_main");
        Ok(Self {
            n_sheets: sheets.len() as u32,
            rule: u32::from(field.rule == ValidRule::CellV1),
            sheets: b_sheets,
            levels: b_levels,
            ix: b_ix,
            qwords: b_q,
            scoff: b_sc,
            bases,
            eval_pipeline,
            march_pipeline,
            layout,
            context,
            max_live,
        })
    }

    const fn uniforms(&self, count: u32, steps: u32, dir: Vec3, tmax: f32) -> Uniforms {
        let b = self.bases;
        Uniforms {
            n_sheets: self.n_sheets,
            rule: self.rule,
            count,
            steps,
            dir: dir.to_array(),
            tmax,
            index_base: b.index,
            rows_base: b.rows,
            bn_base: b.bn,
            bl_base: b.bl,
            bp_base: b.bp,
            bn_len: b.bn_len,
            _pad: [0; 2],
        }
    }

    fn run(
        &self,
        pipeline: &wgpu::ComputePipeline,
        input: &[[f32; 4]],
        out_vec4s: usize,
        u: Uniforms,
    ) -> Result<Vec<[f32; 4]>, GpuError> {
        let device = self.context.device();
        let queue = self.context.queue();
        let n = input.len().max(1);
        let b_u = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sheet field uniforms"),
            contents: bytemuck::cast_slice(&[u]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let b_in = storage_init(device, "sheet field input", bytemuck::cast_slice(input));
        let out_bytes = (n * out_vec4s * 16) as u64;
        let b_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sheet field output"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sheet field staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sheet field"),
            layout: &self.layout,
            entries: &[
                bind(0, &b_u),
                bind(1, &self.sheets),
                bind(2, &self.levels),
                bind(3, &self.ix),
                bind(4, &self.qwords),
                bind(5, &self.scoff),
                bind(6, &b_in),
                bind(7, &b_out),
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sheet field"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sheet field"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WG), 1, 1);
        }
        enc.copy_buffer_to_buffer(&b_out, 0, &staging, 0, out_bytes);
        // the GPU's writes -> the CPU's read of the mapped staging buffer (see GpuContext::before_host_read)
        self.context.before_host_read(&mut enc);
        queue.submit(std::iter::once(enc.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.context.wait_for_mapping(&receiver)?;
        let out = {
            let mapped = slice.get_mapped_range();
            let data: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
            data[..input.len() * out_vec4s].to_vec()
        };
        staging.unmap();
        Ok(out)
    }

    /// The per-ray list size this field's march was compiled with ([`max_live_for`] of [`live_bound`]).
    #[must_use]
    pub const fn max_live(&self) -> u32 {
        self.max_live
    }

    /// The union value at each point.
    pub fn eval_batch(&self, points: &[Vec3]) -> Result<Vec<GpuSample>, GpuError> {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let input: Vec<[f32; 4]> = points.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let out = self.run(
            &self.eval_pipeline,
            &input,
            1,
            self.uniforms(points.len() as u32, 0, Vec3::ZERO, 0.0),
        )?;
        Ok(out
            .iter()
            .map(|o| GpuSample {
                f: o[0],
                g: o[1],
                masked: o[2] != 0.0,
            })
            .collect())
    }

    /// [`Self::march_batch`] with the budget from [`super::decided_steps`]: every result is a hit or a miss.
    pub fn march_batch_decided(
        &self,
        origins: &[Vec3],
        dir: Vec3,
        tmax: f32,
    ) -> Result<Vec<MarchResult>, GpuError> {
        self.march_batch(origins, dir, tmax, super::decided_steps(tmax))
    }

    /// March rays from `origins` along the unit `dir` up to `tmax` with at most `steps` steps.
    pub fn march_batch(
        &self,
        origins: &[Vec3],
        dir: Vec3,
        tmax: f32,
        steps: u32,
    ) -> Result<Vec<MarchResult>, GpuError> {
        if origins.is_empty() {
            return Ok(Vec::new());
        }
        let input: Vec<[f32; 4]> = origins.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let out = self.run(
            &self.march_pipeline,
            &input,
            4,
            self.uniforms(origins.len() as u32, steps, dir, tmax),
        )?;
        let mut res = Vec::with_capacity(origins.len());
        let (rows, rest) = out.as_chunks::<4>();
        debug_assert!(rest.is_empty(), "4 output words a ray");
        for r in rows {
            let kind = r[0][1] as i32;
            res.push(match kind {
                1 => MarchResult::Hit(Hit {
                    t: r[0][0],
                    sheet: r[0][2] as usize,
                    normal: Vec3::new(r[1][0], r[1][1], r[1][2]),
                    uv: (r[2][2] != 0.0).then_some([r[2][0], r[2][1]]),
                    attr_normal: (r[3][3] != 0.0).then(|| Vec3::new(r[3][0], r[3][1], r[3][2])),
                }),
                0 => MarchResult::Miss,
                2 => MarchResult::Undecided,
                _ => {
                    return Err(GpuError::InvalidFormat(format!(
                        "sheet field march: more than {} sheet boxes hold one point",
                        self.max_live
                    )))
                }
            });
        }
        Ok(res)
    }
}

const SHADER: &str = r"
struct SheetInfo { lo: vec3<f32>, hc: f32, hi: vec3<f32>, level_start: u32, level_count: u32, p0: u32, p1: u32, p2: u32 };
struct LevelInfo { lo: vec3<f32>, h: f32, nb: vec3<u32>, index_off: u32, nch: u32, p0: u32, p1: u32, p2: u32,
                   rows_off: array<vec4<u32>, 2>, q_off: array<vec4<u32>, 2>, sc_off: array<vec4<u32>, 2> };
struct Uniforms { n_sheets: u32, rule: u32, count: u32, steps: u32, dir: vec3<f32>, tmax: f32,
                  index_base: u32, rows_base: u32, bn_base: u32, bl_base: u32, bp_base: u32, bn_len: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> SHEETS: array<SheetInfo>;
@group(0) @binding(2) var<storage, read> LEVELS: array<LevelInfo>;
@group(0) @binding(3) var<storage, read> IX: array<u32>;
@group(0) @binding(4) var<storage, read> QW: array<u32>;
@group(0) @binding(5) var<storage, read> SC: array<f32>;
@group(0) @binding(6) var<storage, read> INPUT: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> OUT: array<vec4<f32>>;
const LEAF: u32 = 0x80000000u;
const NONE: u32 = 0xffffffffu;

const BRICK: u32 = 8u;
const NODES: u32 = 9u;
const BRICK_NODES: u32 = 729u;
const HALF_DIAG: f32 = 0.8660254;
const JUMP_S: f32 = 0.02;
const MAX_LIVE: u32 = __MAX_LIVE__u;

struct Sample { f: f32, g: f32, inside: bool, masked: bool, uv: vec2<f32>, has_uv: bool, s: f32, db: f32, db_step: f32, an: vec3<f32>, has_an: bool };

fn qbyte(byte_index: u32) -> f32 {
    let w = QW[byte_index >> 2u];
    return f32((w >> ((byte_index & 3u) * 8u)) & 255u);
}

fn box_distance(p: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> f32 {
    return length(max(max(lo - p, p - hi), vec3<f32>(0.0)));
}

fn outside_bound(p: vec3<f32>, s: SheetInfo) -> f32 {
    return max(box_distance(p, s.lo, s.hi), 3.0 * s.hc);
}

// sheet k at p, interpolating the first nmax channels only (2 while marching; attributes once, at the hit)
fn sample_sheet(k: u32, p: vec3<f32>, nmax: u32) -> Sample {
    let sh = SHEETS[k];
    var out: Sample;
    for (var li = 0u; li < sh.level_count; li = li + 1u) {
        let lv = LEVELS[sh.level_start + li];
        let u = (p - lv.lo) / lv.h;
        let b = f32(BRICK);
        let bi = floor(u / b);
        if (any(bi < vec3<f32>(0.0)) || any(bi >= vec3<f32>(lv.nb))) { continue; }
        let bu = vec3<u32>(bi);
        let k_ = bitcast<i32>(IX[U.index_base + lv.index_off + (bu.x * lv.nb.y + bu.y) * lv.nb.z + bu.z]);
        if (k_ < 0) { continue; }
        let w = u - bi * b;
        let c = min(vec3<u32>(floor(w)), vec3<u32>(BRICK - 1u));
        let t = w - vec3<f32>(c);
        var vals = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        var umin = 1e30;
        var q0 = 0.0;
        var q1 = 0.0;
        let nch = min(lv.nch, nmax);
        for (var ci = 0u; ci < nch; ci = ci + 1u) {
            let row = bitcast<i32>(IX[U.rows_base + lv.rows_off[ci / 4u][ci % 4u] + u32(k_)]);
            var acc = 0.0;
            if (row >= 0) {
                let sc = SC[2u * (lv.sc_off[ci / 4u][ci % 4u] + u32(row))];
                let off = SC[2u * (lv.sc_off[ci / 4u][ci % 4u] + u32(row)) + 1u];
                for (var dx = 0u; dx < 2u; dx = dx + 1u) {
                    for (var dy = 0u; dy < 2u; dy = dy + 1u) {
                        for (var dz = 0u; dz < 2u; dz = dz + 1u) {
                            let wx = select(1.0 - t.x, t.x, dx == 1u);
                            let wy = select(1.0 - t.y, t.y, dy == 1u);
                            let wz = select(1.0 - t.z, t.z, dz == 1u);
                            let node = ((c.x + dx) * NODES + (c.y + dy)) * NODES + (c.z + dz);
                            let v = qbyte(lv.q_off[ci / 4u][ci % 4u] + u32(row) * BRICK_NODES + node) * sc + off;
                            acc = acc + (wx * wy * wz) * v;
                            if (ci == 0u) { umin = min(umin, abs(v)); }
                        }
                    }
                }
                if (ci == 0u) { q0 = sc; }
                if (ci == 1u) { q1 = sc; }
            }
            vals[ci] = acc;
        }
        let s = vals[0];
        let db = max(vals[1], 0.0);
        var f = sqrt(s * s + db * db);
        var g = select(f, -f, s < 0.0);
        out.masked = false;
        if (U.rule == 1u && umin > HALF_DIAG * lv.h + q0) {
            f = max(umin - q0 - HALF_DIAG * lv.h, 0.05);
            g = f;
            out.masked = true;
        }
        out.f = f; out.g = g; out.inside = true;
        out.s = s; out.db = vals[1]; out.db_step = q1;
        out.has_uv = nch >= 4u;
        out.has_an = nch >= 7u;
        out.an = vec3<f32>(vals[4], vals[5], vals[6]);
        out.uv = vec2<f32>(vals[2], vals[3]);
        return out;
    }
    let v = outside_bound(p, sh);
    out.f = v; out.g = v; out.inside = false; out.masked = false; out.has_uv = false; out.has_an = false; out.an = vec3<f32>(0.0);
    out.s = 0.0; out.db = 0.0; out.db_step = 0.0;
    return out;
}

// a bisected sign change of sheet k's g at p is a crossing of the sheet: answered, unmasked, |s| <= JUMP_S,
// d_b <= step/2 (inside the boundary up to the rounding of the stored d_b), as SheetField::march
fn crossing_inside(k: u32, p: vec3<f32>) -> bool {
    let c = sample_sheet(k, p, 2u);
    return c.inside && !c.masked && abs(c.s) <= JUMP_S && c.db <= 0.5 * c.db_step;
}

fn in_box(k: u32, p: vec3<f32>) -> bool {
    let sh = SHEETS[k];
    return all(p >= sh.lo) && all(p <= sh.hi);
}

struct Union { f: f32, g: f32, k: u32, masked: bool };

fn nlo(i: u32) -> vec3<f32> { return vec3<f32>(bitcast<f32>(IX[U.bn_base + 8u * i]), bitcast<f32>(IX[U.bn_base + 8u * i + 1u]), bitcast<f32>(IX[U.bn_base + 8u * i + 2u])); }
fn nhi(i: u32) -> vec3<f32> { return vec3<f32>(bitcast<f32>(IX[U.bn_base + 8u * i + 4u]), bitcast<f32>(IX[U.bn_base + 8u * i + 5u]), bitcast<f32>(IX[U.bn_base + 8u * i + 6u])); }

// min over sheets, lowest index on ties; a node farther than the best value so far is skipped (as the CPU walk)
fn eval_union(p: vec3<f32>) -> Union {
    var r: Union;
    r.f = 1e30; r.g = 1e30; r.k = NONE; r.masked = false;
    if (U.bn_len < 8u || U.n_sheets == 0u) { r.k = 0u; return r; }
    var at = 0u;
    var prev = NONE;
    loop {
        let a = IX[U.bn_base + 8u * at + 3u];
        let b = IX[U.bn_base + 8u * at + 7u];
        var up = true;
        if ((a & LEAF) != 0u) {
            let first = a & ~LEAF;
            for (var j = first; j < first + b; j = j + 1u) {
                let k = IX[U.bl_base + j];
                let sh = SHEETS[k];
                var v: f32; var w: f32; var m = false;
                if (box_distance(p, sh.lo, sh.hi) == 0.0) {
                    let s = sample_sheet(k, p, 2u);
                    v = s.f; w = s.g; m = s.masked;
                } else {
                    v = outside_bound(p, sh); w = v;
                }
                if (v < r.f || (v == r.f && k < r.k)) { r.f = v; r.g = w; r.k = k; r.masked = m; }
            }
        } else {
            let dl = box_distance(p, nlo(a), nhi(a));
            let dr = box_distance(p, nlo(b), nhi(b));
            var first = a; var df = dl; var second = b; var ds = dr;
            if (dr < dl) { first = b; df = dr; second = a; ds = dl; }
            if (prev == NONE || prev == IX[U.bp_base + at]) {
                if (df <= r.f) { prev = at; at = first; up = false; }
                else if (ds <= r.f) { prev = at; at = second; up = false; }
            } else if (prev == first && ds <= r.f) { prev = at; at = second; up = false; }
        }
        if (up) {
            if (at == 0u) { break; }
            prev = at;
            at = IX[U.bp_base + at];
        }
    }
    if (r.k == NONE) { r.k = 0u; }
    return r;
}

// the sheets whose box holds p, ascending, into L (count returned; MAX_LIVE + 1 means overflow)
fn live_sheets(p: vec3<f32>, L: ptr<function, array<u32, __MAX_LIVE__>>) -> u32 {
    var n = 0u;
    if (U.bn_len < 8u || U.n_sheets == 0u) { return 0u; }
    if (!(all(p >= nlo(0u)) && all(p <= nhi(0u)))) { return 0u; }
    var at = 0u;
    var prev = NONE;
    loop {
        let a = IX[U.bn_base + 8u * at + 3u];
        let b = IX[U.bn_base + 8u * at + 7u];
        var up = true;
        if ((a & LEAF) != 0u) {
            let first = a & ~LEAF;
            for (var j = first; j < first + b; j = j + 1u) {
                let k = IX[U.bl_base + j];
                if (in_box(k, p)) {
                    if (n == MAX_LIVE) { return MAX_LIVE + 1u; }
                    (*L)[n] = k; n = n + 1u;
                }
            }
        } else {
            let in_l = all(p >= nlo(a)) && all(p <= nhi(a));
            let in_r = all(p >= nlo(b)) && all(p <= nhi(b));
            if (prev == NONE || prev == IX[U.bp_base + at]) {
                if (in_l) { prev = at; at = a; up = false; }
                else if (in_r) { prev = at; at = b; up = false; }
            } else if (prev == a && in_r) { prev = at; at = b; up = false; }
        }
        if (up) {
            if (at == 0u) { break; }
            prev = at;
            at = IX[U.bp_base + at];
        }
    }
    for (var i = 1u; i < n; i = i + 1u) {
        let x = (*L)[i];
        var j = i;
        loop {
            if (j == 0u || (*L)[j - 1u] <= x) { break; }
            (*L)[j] = (*L)[j - 1u];
            j = j - 1u;
        }
        (*L)[j] = x;
    }
    return n;
}

@compute @workgroup_size(64)
fn eval_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= U.count) { return; }
    let r = eval_union(INPUT[i].xyz);
    OUT[i] = vec4<f32>(r.f, r.g, select(0.0, 1.0, r.masked), f32(r.k));
}

@compute @workgroup_size(64)
fn march_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= U.count) { return; }
    let o = INPUT[i].xyz;
    let d = U.dir;
    var prev_k: array<u32, __MAX_LIVE__>;
    var prev_g: array<f32, __MAX_LIVE__>;
    var n_prev = 0u;
    var overflow = false;
    var lv: array<u32, __MAX_LIVE__>;
    let n0 = live_sheets(o, &lv);
    if (n0 > MAX_LIVE) { overflow = true; }
    else {
        for (var j = 0u; j < n0; j = j + 1u) { prev_k[j] = lv[j]; prev_g[j] = sample_sheet(lv[j], o, 2u).g; }
        n_prev = n0;
    }
    var t = 0.0;
    var tprev = 0.0;
    var kind = 2;
    var own = 0u;
    for (var step = 0u; step < U.steps && !overflow; step = step + 1u) {
        let p = o + d * t;
        let dd = eval_union(p).f;
        var thit = 1e30;
        var khit = 0u;
        var cur_k: array<u32, __MAX_LIVE__>;
        var cur_g: array<f32, __MAX_LIVE__>;
        var n_cur = 0u;
        let nl = live_sheets(p, &lv);
        if (nl > MAX_LIVE) { overflow = true; break; }
        for (var li = 0u; li < nl; li = li + 1u) {
            let k = lv[li];
            let gc = sample_sheet(k, p, 2u).g;
            cur_k[n_cur] = k; cur_g[n_cur] = gc; n_cur = n_cur + 1u;
            if (t <= 0.0) { continue; }
            var found = false; var gp = 0.0;
            for (var j = 0u; j < n_prev; j = j + 1u) { if (prev_k[j] == k) { found = true; gp = prev_g[j]; } }
            if (!found) { continue; }
            if ((gc < 0.0) == (gp < 0.0)) { continue; }
            var a = tprev; var b = t; var ga = gp;
            for (var it = 0; it < 30; it = it + 1) {
                let mid = 0.5 * (a + b);
                let gm = sample_sheet(k, o + d * mid, 2u).g;
                if ((gm < 0.0) == (ga < 0.0)) { a = mid; ga = gm; } else { b = mid; }
            }
            let tm = 0.5 * (a + b);
            if (tm < thit && crossing_inside(k, o + d * tm)) { thit = tm; khit = k; }
        }
        if (overflow) { break; }
        if (thit < 1e29) { t = thit; own = khit; kind = 1; break; }
        tprev = t;
        for (var j = 0u; j < n_cur; j = j + 1u) { prev_k[j] = cur_k[j]; prev_g[j] = cur_g[j]; }
        n_prev = n_cur;
        t = t + max(dd, __MIN_STEP__);
        if (t >= U.tmax) { kind = 0; break; }
    }
    if (overflow) { OUT[4u * i] = vec4<f32>(0.0, 3.0, 0.0, 0.0); return; }
    var nrm = vec3<f32>(0.0, 0.0, 1.0);
    var uv = vec3<f32>(0.0);
    var an = vec4<f32>(0.0);
    if (kind == 1) {
        let p = o + d * t;
        let e = 0.05;
        // normal = gradient of the sheet's side channel s (the distance g bends toward the rim), as SheetField
        let n = vec3<f32>(
            sample_sheet(own, p + vec3<f32>(e, 0.0, 0.0), 2u).s - sample_sheet(own, p - vec3<f32>(e, 0.0, 0.0), 2u).s,
            sample_sheet(own, p + vec3<f32>(0.0, e, 0.0), 2u).s - sample_sheet(own, p - vec3<f32>(0.0, e, 0.0), 2u).s,
            sample_sheet(own, p + vec3<f32>(0.0, 0.0, e), 2u).s - sample_sheet(own, p - vec3<f32>(0.0, 0.0, e), 2u).s);
        if (length(n) > 1e-12) { nrm = normalize(n); }
        let s = sample_sheet(own, p, 8u);
        if (s.has_uv && s.inside) { uv = vec3<f32>(s.uv, 1.0); }
        if (s.has_an && s.inside && length(s.an) > 1e-12) { an = vec4<f32>(normalize(s.an), 1.0); }
    }
    OUT[4u * i] = vec4<f32>(t, f32(kind), f32(own), 0.0);
    OUT[4u * i + 1u] = vec4<f32>(nrm, 0.0);
    OUT[4u * i + 2u] = vec4<f32>(uv, 0.0);
    OUT[4u * i + 3u] = an;
}
";
