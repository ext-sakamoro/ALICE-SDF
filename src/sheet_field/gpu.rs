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

use super::{Hit, MarchResult, SheetField, ValidRule};
use crate::compiled::GpuError;
use crate::gpu_context::{GpuContext, Held};

/// The smallest per-ray list of live sheets the march compiles (fields whose [`live_bound`] is at most this).
pub const MAX_LIVE_SHEETS: u32 = 64;
/// The largest per-ray list the march compiles; a field whose [`live_bound`] is over it is refused.
pub const MAX_LIVE_CAP: u32 = 1024;

/// The most levels one sheet may have on the GPU (every loop over a sheet's levels is bounded by it); a field with
/// more is refused at upload.
pub const MAX_SHEET_LEVELS: u32 = 16;
/// The most triangles one kink cell may refer to on the GPU; a field with more is refused at upload.
pub const MAX_KINK_CELL_TRIS: u32 = 256;
/// The most segments one kink cell may refer to on the GPU; a field with more is refused at upload.
pub const MAX_KINK_CELL_SEGS: u32 = 256;
/// The most sheets one uploaded field may have (it bounds the BVH walks: at most 3 visits a node, 2n - 1 nodes).
pub const MAX_FIELD_SHEETS: u32 = 1 << 16;
/// The most steps one march may take (`steps` of [`GpuSheetField::march_batch`]); [`super::decided_steps`] reaches it
/// at a `tmax` of about 10.4 m.
pub const MAX_MARCH_STEPS: u32 = 1 << 20;
/// The most visits of one BVH walk: 3 a node over the most nodes a field may have.
const MAX_BVH_VISITS: u32 = 3 * (2 * MAX_FIELD_SHEETS - 1) + 1;

/// The march kernel's label (listed with its ceiling in the GPU lifecycle census).
pub const MARCH_KERNEL: &str = "sheet field march";

/// The stated maxima of the host values the march's loops are bounded by.
///
/// Each is checked on the host before a dispatch: `steps` by [`GpuSheetField::march_batch`], the budget by
/// [`GpuSheetField::set_work_budget`], the visits by the upload's sheet count.
#[must_use]
pub fn march_maxima() -> crate::gpu_loop_bound::HostMaxima {
    crate::gpu_loop_bound::HostMaxima::from([
        ("U.steps".to_string(), u64::from(MAX_MARCH_STEPS)),
        ("U.bn_visits".to_string(), u64::from(MAX_BVH_VISITS)),
        ("U.work_budget".to_string(), u64::from(MARCH_WORK_BUDGET)),
    ])
}

/// The most loop entries between two ticks of the march's work counter (loops one loop body enters directly, calls
/// expanded): stated here, checked against the shader by the counted-loop check.
pub const MARCH_UNIT_LOOPS: u64 = 16;
/// The most loop instances in the march's expanded call tree (each entered at most once after the budget is spent):
/// stated here, checked against the shader by the counted-loop check.
pub const MARCH_LOOP_INSTANCES: u64 = 512;

/// The march kernel's ceiling: every loop iteration ticks the work counter, so one invocation runs at most
/// `(MARCH_WORK_BUDGET + MARCH_LOOP_INSTANCES + 1) x (MARCH_UNIT_LOOPS + 1)` loop iterations.
#[must_use]
pub const fn march_ceiling() -> crate::gpu_loop_bound::KernelCeiling {
    crate::gpu_loop_bound::KernelCeiling {
        max_trip_product: (MARCH_WORK_BUDGET as u64 + MARCH_LOOP_INSTANCES + 1) * (MARCH_UNIT_LOOPS + 1),
        reason: "the sheet march: every loop iteration ticks the per-ray work counter, bounded by MARCH_WORK_BUDGET",
        counter: crate::gpu_loop_bound::WorkCounter {
            tick: "tick",
            budget: "U.work_budget",
        },
    }
}

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
    march_shader_margin(max_live, 0.0)
}

/// [`march_shader`] with a step margin.
///
/// Every step is taken from the union value less `margin` ([`super::kink::KINK_STEP_MARGIN`] for a field with kink
/// cells, 0 otherwise; the CPU `SheetField::step_margin`).
#[must_use]
pub fn march_shader_margin(max_live: u32, margin: f32) -> String {
    SHADER
        .replace("__STEP_MARGIN__", &format!("{margin:?}"))
        .replace("__MAX_LIVE__", &max_live.to_string())
        .replace("__MIN_STEP__", &format!("{:?}", super::MIN_STEP))
        .replace("__MASKED_FLOOR__", &format!("{:?}", super::MASKED_FLOOR))
        .replace(
            "__MASKED_FLOOR_PER_H__",
            &format!("{:?}", super::MASKED_FLOOR_PER_H),
        )
        .replace(
            "__CELL_WALK_CAP__",
            &format!("{}u", super::cellroot::CELL_WALK_CAP),
        )
        .replace("__MAX_LEVELS__", &MAX_SHEET_LEVELS.to_string())
        .replace("__MAX_KINK_TRIS__", &MAX_KINK_CELL_TRIS.to_string())
        .replace("__MAX_KINK_SEGS__", &MAX_KINK_CELL_SEGS.to_string())
        .replace("__BVH_LEAF__", &super::SHEET_BVH_LEAF.to_string())
}
const WG: u32 = 64;

/// A BVH walk of point or ray `i` used its visit bound without returning to the root (a malformed tree).
fn unfinished_walk(i: usize) -> GpuError {
    GpuError::InvalidFormat(format!(
        "sheet field: the BVH walk of item {i} took more than 3 visits a node without returning to the root"
    ))
}

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
    /// Where this level's band masks start (relative to the band table), when a channel is a band.
    band_off: u32,
    _pad: [u32; 2],
    rows_off: [u32; 8],
    q_off: [u32; 8],
    sc_off: [u32; 8],
    /// Nodes per brick axis of each channel (9, a coarse attribute grid, or 0 = band).
    nodes: [u32; 8],
    /// For a band channel: where its row starts are in the band table.
    st_off: [u32; 8],
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
    /// Where the band table (masks and row starts) starts in the merged `u32` buffer.
    band_base: u32,
    /// [`MARCH_WORK_BUDGET`] or the field's own setting.
    work_budget: u32,
    /// The most visits of one BVH walk (3 a node), at most [`MAX_BVH_VISITS`].
    bn_visits: u32,
    _pad: [u32; 3],
}

/// Where each table starts in the merged `u32` buffer.
#[derive(Clone, Copy)]
struct Bases {
    band: u32,
    index: u32,
    rows: u32,
    bn: u32,
    bl: u32,
    bp: u32,
    bn_len: u32,
}

/// A [`SheetField`] uploaded to the GPU: its buffers and pipelines, held with the context that made them
/// ([`Held`] drops them before the context).
pub type GpuSheetField = Held<SheetFieldResources>;

/// The buffers and pipelines of a [`GpuSheetField`] (no context of their own).
pub struct SheetFieldResources {
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
    control: std::sync::Mutex<DispatchControl>,
    marker: std::sync::atomic::AtomicU32,
}

/// How long one march dispatch is allowed to take: the sizes adapt to keep each dispatch near this.
///
/// This is a bound on dispatch duration, chosen against one measured device: on the GPU it was measured
/// on, a single dispatch that would take about 4.5-4.9 s never returned while the same rays split into
/// dispatches of up to 4.24 s all finished (the cause of that device behaviour is inferred, not measured).
/// Other devices are unmeasured; browsers stop long dispatches sooner.
pub const DISPATCH_TARGET: std::time::Duration = std::time::Duration::from_secs(1);

/// How long a read waits for one march dispatch before it is a [`GpuError::DispatchDidNotFinish`]
/// (a few times [`DISPATCH_TARGET`], below the measured duration where a dispatch never returned).
pub const DISPATCH_WAIT: std::time::Duration = std::time::Duration::from_secs(4);

/// Rays in the first march dispatch of a field; later sizes follow the measured times.
pub const FIRST_DISPATCH: usize = 16_384;

/// The smallest march dispatch.
pub const MIN_DISPATCH: usize = 1_024;

/// The work one ray's march may do, in ticks of its work counter.
///
/// One tick per loop iteration of the march kernel, calls expanded: a step, a sheet sample, a corner read, a BVH
/// visit and a bisection step each tick once.
///
/// A ray over this ends as [`MarchResult::Undecided`], never as a hit. The value is the same on every device, so a
/// ray's result never depends on the device's speed; a device too slow to run one ray at this budget within the wait
/// bound is refused when a field is uploaded ([`check_device_speed`], with the adapter's measured speed from
/// [`adapter_speeds`]).
///
/// Derived from the worst ray measured on real fields (the per-ray work in [`GpuSheetField::last_march_stats`]): the
/// census over the scanned fields gave at most 670,569 ticks (a head scan, face front view), and 152,473 on a rock scan
/// with kink cells at 1600 x 1600 rays. 2^21 = 2,097,152 is 3.1 times the census worst. A ray grazing a flat sheet
/// takes about 3,000 ticks per mm, so a graze over about 700 mm ends undecided.
pub const MARCH_WORK_BUDGET: u32 = 1 << 21;

/// The half lengths (mm) of the flat sheets the device-speed measurement grazes ([`measure_ns_per_tick`]): about
/// 30,000 and 240,000 ticks.
pub const SPEED_PROBE_HALF_MM: [f32; 2] = [5.0, 40.0];

/// Whether a device whose march costs `ns_per_tick` runs one ray at the full [`MARCH_WORK_BUDGET`] within a third of
/// `wait` (the read's wait bound for one dispatch, [`DISPATCH_WAIT`]).
///
/// The third leaves room for the rays that share the ray's dispatch and for the measurement's error. The budget is
/// never scaled to the device: a slower device is refused, so the same ray has the same result everywhere.
///
/// # Errors
///
/// [`GpuError::DeviceTooSlow`] naming both times; [`GpuError::InvalidFormat`] for a speed that is not a finite
/// positive number (not measured).
pub fn check_device_speed(ns_per_tick: f64, wait: std::time::Duration) -> Result<(), GpuError> {
    if !(ns_per_tick.is_finite() && ns_per_tick > 0.0) {
        return Err(GpuError::InvalidFormat(format!(
            "sheet field march: device speed {ns_per_tick} ns per tick is not a measured time"
        )));
    }
    let full = ns_per_tick * f64::from(MARCH_WORK_BUDGET) * 1e-9;
    let limit = wait.as_secs_f64() / 3.0;
    if full > limit {
        return Err(GpuError::DeviceTooSlow {
            ns_per_tick,
            budget: MARCH_WORK_BUDGET,
            budget_ms: (full * 1e3).ceil() as u64,
            limit_ms: (limit * 1e3).floor() as u64,
        });
    }
    Ok(())
}

/// The flat sheet the device-speed probe grazes: `z = 0` over `|x|, |y| <= half`.
fn speed_probe_field(half: f32) -> Result<SheetField, GpuError> {
    use super::build::{level_from_fn, sheet};
    let h = 0.25;
    let nbx = ((2.0 * half + 8.0) / (8.0 * h)).ceil() as u32;
    let lo = Vec3::new(-half - 4.0 + 0.013, -2.0 + 0.021, -1.0 + 0.007);
    let db = move |x: f32, y: f32| {
        let dx = x.abs() - half;
        let dy = y.abs() - half;
        Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0)
    };
    let lv = level_from_fn(
        lo,
        h,
        [nbx, 2, 1],
        |_c, _e| true,
        move |p| vec![p.z, db(p.x, p.y)],
    )
    .map_err(|e| GpuError::InvalidFormat(e.to_string()))?;
    let sh = sheet(vec![lv]).map_err(|e| GpuError::InvalidFormat(e.to_string()))?;
    Ok(SheetField::new(vec![sh], ValidRule::CellV1))
}

/// The command that measures this machine's adapter and prints its row of [`adapter_speeds`].
pub const ADAPTER_SPEED_COMMAND: &str =
    "cargo run --release --features gpu --example sheet_field_adapter_speed -- $(date -u +%F)";

/// The environment variable naming a further speed table file (the same format as the crate's), read at upload.
///
/// The crate's table holds rows for software adapters only; a machine with a hardware GPU keeps its rows in its
/// own file and names it here. With the variable unset only the crate's table is read.
pub const ADAPTER_SPEEDS_VAR: &str = "ALICE_SDF_ADAPTER_SPEEDS";

/// The measured march speeds of the software adapters (`src/sheet_field/adapter_speeds.tsv`).
const ADAPTER_SPEEDS_TSV: &str = include_str!("adapter_speeds.tsv");

/// One adapter's measured march speed: its identity as wgpu reports it, the speed, when and how it was measured.
#[derive(Debug, Clone, PartialEq)]
pub struct AdapterSpeed {
    /// PCI vendor id (or the backend's equivalent).
    pub vendor: u32,
    /// PCI device id (or the backend's equivalent).
    pub device: u32,
    /// The backend (`Vulkan`, `Metal`, `Dx12`, `Gl`).
    pub backend: String,
    /// The adapter's name.
    pub name: String,
    /// The driver's description, its version where the backend gives one (empty where it gives none).
    pub driver_info: String,
    /// March time per work tick (see [`measure_ns_per_tick`]).
    pub ns_per_tick: f64,
    /// The date it was measured (`YYYY-MM-DD`).
    pub measured: String,
    /// The command that measured it ([`ADAPTER_SPEED_COMMAND`]).
    pub command: String,
}

impl AdapterSpeed {
    /// The row as a line of `adapter_speeds.tsv`.
    #[must_use]
    pub fn tsv(&self) -> String {
        format!(
            "{:#06x}\t{:#06x}\t{}\t{}\t{}\t{:.2}\t{}\t{}",
            self.vendor,
            self.device,
            self.backend,
            self.name,
            self.driver_info,
            self.ns_per_tick,
            self.measured,
            self.command
        )
    }

    fn matches(&self, info: &wgpu::AdapterInfo) -> bool {
        self.vendor == info.vendor
            && self.device == info.device
            && self.backend == format!("{:?}", info.backend)
            && self.name == info.name
            && self.driver_info == info.driver_info
    }
}

/// The rows of a speed table: tab-separated `vendor device backend name driver_info ns_per_tick measured command`,
/// `#` lines and blank lines skipped; ids in hex (`0x10de`) or decimal.
///
/// # Errors
///
/// [`GpuError::InvalidFormat`] for a malformed row, a speed that is not finite and positive, or a table with no row
/// (an empty table would refuse every adapter).
pub fn parse_adapter_speeds(text: &str) -> Result<Vec<AdapterSpeed>, GpuError> {
    let bad = |n: usize, why: &str| {
        GpuError::InvalidFormat(format!("adapter speed table line {n}: {why}"))
    };
    let id = |n: usize, f: &str| {
        let v = match f.strip_prefix("0x") {
            Some(h) => u32::from_str_radix(h, 16),
            None => f.parse(),
        };
        v.map_err(|_| bad(n, &format!("id {f:?} is not a number")))
    };
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let n = i + 1;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let [vendor, device, backend, name, driver_info, ns, measured, command] = f[..] else {
            return Err(bad(n, &format!("{} fields, not 8", f.len())));
        };
        let ns_per_tick: f64 = ns
            .parse()
            .map_err(|_| bad(n, &format!("speed {ns:?} is not a number")))?;
        if !(ns_per_tick.is_finite() && ns_per_tick > 0.0) {
            return Err(bad(
                n,
                &format!("speed {ns_per_tick} is not a measured time"),
            ));
        }
        rows.push(AdapterSpeed {
            vendor: id(n, vendor)?,
            device: id(n, device)?,
            backend: backend.to_string(),
            name: name.to_string(),
            driver_info: driver_info.to_string(),
            ns_per_tick,
            measured: measured.to_string(),
            command: command.to_string(),
        });
    }
    if rows.is_empty() {
        return Err(GpuError::InvalidFormat(
            "adapter speed table: no row".into(),
        ));
    }
    Ok(rows)
}

/// Whether a table text holds any row (a line that is neither blank nor a `#` comment).
fn has_rows(text: &str) -> bool {
    text.lines()
        .any(|l| !(l.trim().is_empty() || l.starts_with('#')))
}

/// The speed tables read at upload: the crate's (`src/sheet_field/adapter_speeds.tsv`, software adapters), then
/// the file [`ADAPTER_SPEEDS_VAR`] names, when it is set.
///
/// # Errors
///
/// As [`parse_adapter_speeds`] for either table that has rows; the named file unreadable, or with no row.
pub fn adapter_speeds() -> Result<Vec<AdapterSpeed>, GpuError> {
    let mut rows = if has_rows(ADAPTER_SPEEDS_TSV) {
        parse_adapter_speeds(ADAPTER_SPEEDS_TSV)?
    } else {
        Vec::new()
    };
    if let Some(path) = std::env::var_os(ADAPTER_SPEEDS_VAR) {
        let text = std::fs::read_to_string(&path).map_err(|e| {
            GpuError::InvalidFormat(format!(
                "{ADAPTER_SPEEDS_VAR} {}: {e}",
                path.to_string_lossy()
            ))
        })?;
        rows.extend(parse_adapter_speeds(&text).map_err(|e| {
            GpuError::InvalidFormat(format!(
                "{ADAPTER_SPEEDS_VAR} {}: {e}",
                path.to_string_lossy()
            ))
        })?);
    }
    Ok(rows)
}

/// The measured march speed of `info`'s adapter in `rows`.
///
/// The row whose identity (vendor, device, backend, name and driver) is exactly the adapter's. Never measured at
/// run time, so whether an adapter is admitted does not depend on what else the device is doing.
///
/// # Errors
///
/// [`GpuError::AdapterSpeedUnknown`] when no row is the adapter's, naming [`ADAPTER_SPEED_COMMAND`].
pub fn table_ns_per_tick(rows: &[AdapterSpeed], info: &wgpu::AdapterInfo) -> Result<f64, GpuError> {
    rows.iter()
        .find(|r| r.matches(info))
        .map(|r| r.ns_per_tick)
        .ok_or_else(|| GpuError::AdapterSpeedUnknown {
            adapter: format!(
                "{:#06x} {:#06x} {:?} {:?} {:?}",
                info.vendor, info.device, info.backend, info.name, info.driver_info
            ),
            command: ADAPTER_SPEED_COMMAND.to_string(),
            var: ADAPTER_SPEEDS_VAR.to_string(),
        })
}

/// This machine's adapter measured: its row of the speed table, dated `measured` (`YYYY-MM-DD`, given by the
/// caller: the library reads no clock).
///
/// # Errors
///
/// As [`measure_ns_per_tick`].
pub fn measure_adapter_speed(measured: &str) -> Result<AdapterSpeed, GpuError> {
    let (ns_per_tick, info) = measure_ns_per_tick()?;
    Ok(AdapterSpeed {
        vendor: info.vendor,
        device: info.device,
        backend: format!("{:?}", info.backend),
        name: info.name,
        driver_info: info.driver_info,
        ns_per_tick,
        measured: measured.to_string(),
        command: ADAPTER_SPEED_COMMAND.to_string(),
    })
}

/// The device's march time per work tick, and the adapter it was measured on.
///
/// Run on an idle device: it is the measurement for the speed table ([`ADAPTER_SPEED_COMMAND`]), never taken at
/// upload. One ray alone in its dispatch grazes a probe sheet for its full length, at two lengths
/// ([`SPEED_PROBE_HALF_MM`]); each time is the least of 5 timed runs (after one untimed). The figure is the slope
/// over the 5 mm and 40 mm half-lengths (time over work), so the dispatch's fixed cost (a few ms, which a short ray
/// would count as per-tick time) is left out.
///
/// # Errors
///
/// The upload's or the march's error; [`GpuError::InvalidFormat`] when the longer graze did not take more work.
pub fn measure_ns_per_tick() -> Result<(f64, wgpu::AdapterInfo), GpuError> {
    let mut info = None;
    let mut pts = [(0u32, 0.0f64); 2];
    for (pt, half) in pts.iter_mut().zip(SPEED_PROBE_HALF_MM) {
        let g = GpuSheetField::build(&speed_probe_field(half)?, &|src| src)?;
        info = Some(g.context().adapter_info().clone());
        let o = [Vec3::new(-half - 1.0, 0.003, 0.02)];
        let tmax = 2.0 * half + 2.0;
        g.march_batch_decided(&o, Vec3::X, tmax)?;
        let mut best = f64::INFINITY;
        let mut work = 0u32;
        for _ in 0..5 {
            g.march_batch_decided(&o, Vec3::X, tmax)?;
            let st = g.last_march_stats();
            work = st.max_work;
            best = best.min(st.dispatches.iter().map(|d| d.1).sum::<f64>());
        }
        *pt = (work, best);
    }
    let [(w0, t0), (w1, t1)] = pts;
    if w1 <= w0 {
        return Err(GpuError::InvalidFormat(format!(
            "sheet field march: the speed probe's longer graze took {w1} ticks, not more than the shorter's {w0}"
        )));
    }
    // a slope at or under 0 (timer noise on a very fast device) is the shorter ray's whole time per tick instead
    let slope = (t1 - t0) * 1e9 / f64::from(w1 - w0);
    let ns = if slope > 0.0 {
        slope
    } else {
        t0 * 1e9 / f64::from(w0.max(1))
    };
    Ok((ns, info.expect("two probe uploads")))
}

/// Rounds of re-issue after a march dispatch leaves results unwritten (each round halves the dispatch size); a ray
/// still unwritten after these is a [`GpuError::DispatchLeftUnwritten`].
///
/// Measured: on one device a dispatch that ran about 2 s (or longer, or while another process used the GPU) wrote
/// only part of its rays while the copies after it in the same submit ran; why the device cuts it is not measured.
pub const MAX_REISSUE_ROUNDS: u32 = 8;

/// The dispatch sizes, their times and the per-ray work of the last march call.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MarchStats {
    /// `(rays, seconds)` of each dispatch, in order, re-issues included.
    pub dispatches: Vec<(usize, f64)>,
    /// The summed per-ray work (see [`MARCH_WORK_BUDGET`]) of each dispatch, in the order of `dispatches` (the rays
    /// it wrote; re-issued rays count in the dispatch that wrote them).
    pub dispatch_work: Vec<u64>,
    /// The re-issue round of each dispatch, in the order of `dispatches`: 0 for a first issue, k for the k-th round of
    /// re-issue after a cut (those follow the halving of the cut size, not the size rule of [`next_dispatch`]).
    pub dispatch_round: Vec<u32>,
    /// Dispatches that left some of their rays unwritten (each such ray was issued again in a smaller dispatch).
    pub cut_dispatches: u64,
    /// Rays issued again because a dispatch left them unwritten (a ray counts once per re-issue).
    pub reissued_rays: u64,
    /// The size of the smallest re-issue dispatch (0 when nothing was re-issued).
    pub smallest_reissue: usize,
    /// The largest work of one ray (see [`MARCH_WORK_BUDGET`]).
    pub max_work: u32,
    /// The batch index of a ray with that work (for profiling it; `None` before any ray was read).
    pub max_work_ray: Option<usize>,
    /// Rays whose work was at least half the budget.
    pub over_half_budget: u64,
    /// Rays that ran out of the budget (returned as undecided).
    pub over_budget: u64,
    /// Hits that needed an attribute node outside its band: that attribute is `None` in the hit (as on the CPU).
    /// A field whose kink cells carry the banded attributes ([`super::kink::KinkSheet::covers_band`]) reads 0.
    pub band_misses: u64,
}

#[derive(Debug)]
struct DispatchControl {
    next: usize,
    target: std::time::Duration,
    wait: std::time::Duration,
    work_budget: u32,
    ns_per_tick: f64,
    spread_block: usize,
    stats: MarchStats,
    /// test only: the next `cut.1` march dispatches use `cut.0` workgroups whatever their size (a dispatch the
    /// device cut short, made on purpose)
    #[cfg(test)]
    cut: (u32, u32),
}

/// Bytes of a ray's march result in the output binding (4 `vec4<f32>`).
pub const MARCH_OUT_BYTES_PER_RAY: u64 = 64;

/// The most rays one march dispatch may hold on a device.
///
/// The workgroups one dispatch dimension allows times the workgroup size, and the rays whose results fit one storage
/// binding (`binding_limit`, the smaller of the device's
/// largest storage binding and largest buffer, over [`MARCH_OUT_BYTES_PER_RAY`]). Never below one workgroup.
#[must_use]
pub fn dispatch_cap(max_workgroups_per_dim: u32, binding_limit: u64) -> usize {
    let by_groups = u64::from(max_workgroups_per_dim) * u64::from(WG);
    let by_binding = binding_limit / MARCH_OUT_BYTES_PER_RAY;
    usize::try_from(by_groups.min(by_binding).max(u64::from(WG))).unwrap_or(usize::MAX)
}

/// The next dispatch size after one of `rays` took `secs`.
///
/// The rays whose predicted time (at the measured seconds per ray) is the smaller of the target and twice the last
/// dispatch's time, and after a dispatch over the target at most half of it; never below [`MIN_DISPATCH`]. The rays of
/// a march are dispatched in an order spread over the whole batch ([`spread_stride`]), so the seconds per ray of
/// consecutive dispatches stay close and the prediction holds.
#[must_use]
pub fn next_dispatch(rays: usize, secs: f64, target: std::time::Duration) -> usize {
    let t = target.as_secs_f64();
    if rays == 0 || secs <= 0.0 {
        return rays.saturating_mul(2).max(MIN_DISPATCH);
    }
    let per_ray = secs / rays as f64;
    let mut next = (t.min(2.0 * secs) / per_ray).floor() as usize;
    if secs > t {
        next = next.min(rays / 2);
    }
    next.max(MIN_DISPATCH)
}

/// Rays in one contiguous block of the march order.
///
/// A block keeps neighbouring rays (neighbouring pixels) together, so the rays of one workgroup touch the same sheets
/// and take similar steps; the blocks are visited in the spread order so that each dispatch still samples the whole
/// batch. Measured on one device and one view (10.24 M rays, a 15-sheet-list body field): single rays in the spread
/// order 144.2 s, contiguous 47.5 s, blocks of 1024 / 4096 / 16384 rays 48.9 / 47.7 / 47.9 s, results byte-identical in
/// all five; 4096 is the smallest block measured within 1 % of contiguous (1024 was 3 % over in one run). The blocks
/// did not make the cost per ray of consecutive dispatches flatter than the contiguous order (mean |ln ratio| 0.36 vs
/// 0.35; single rays 0.11).
pub const SPREAD_BLOCK: usize = 4_096;

/// The march order over `n` rays.
///
/// Contiguous blocks of `block` rays (the last one shorter), the blocks visited in the spread order ([`spread_stride`]
/// over the block count). `block` 0 or at least `n` is the contiguous order. A permutation of `0..n` for every `n` and
/// `block`.
#[must_use]
pub fn spread_blocks(n: usize, block: usize) -> Vec<usize> {
    if block == 0 || block >= n {
        return (0..n).collect();
    }
    let nb = n.div_ceil(block);
    let sb = spread_stride(nb);
    let mut out = Vec::with_capacity(n);
    for k in 0..nb {
        let b = ((k as u128 * sb as u128) % nb as u128) as usize;
        out.extend(b * block..((b + 1) * block).min(n));
    }
    out
}

/// The stride of the spread order over `n` rays.
///
/// Ray `j` of the order is `(j * stride) % n`, a permutation (the stride is coprime with `n`), near `0.618 n` so that
/// consecutive dispatches sample the whole batch.
#[must_use]
pub fn spread_stride(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let gcd = |mut a: usize, mut b: usize| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let mut s = ((n as f64) * 0.618_033_988_749_895).round() as usize % n;
    s = s.max(1);
    while gcd(s, n) != 1 {
        s += 1;
        if s >= n {
            s = 1;
        }
    }
    s
}

/// The bits every output word holds before a dispatch writes it: a signalling NaN with a payload, which no
/// arithmetic of the kernels produces (arithmetic NaNs are quiet)
const NOT_WRITTEN: u32 = 0x7FA5_A5A5;

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
        Self::new_with_mode(field, field.root_mode())
    }

    /// [`Self::new`] for `mode` whatever the field's own. The GPU program runs the cell walk only; any other mode
    /// (the CPU's fixed-step reference) is refused, never run as something else.
    pub fn new_with_mode(field: &SheetField, mode: super::RootMode) -> Result<Self, GpuError> {
        if mode != super::RootMode::CellWalk {
            return Err(GpuError::InvalidFormat(format!(
                "the GPU sheet march runs the cell walk only, not {mode:?}"
            )));
        }
        let g = Self::build(field, &|src| src)?;
        let ns = table_ns_per_tick(&adapter_speeds()?, g.context().adapter_info())?;
        check_device_speed(ns, DISPATCH_WAIT)?;
        if let Ok(mut c) = g.control.lock() {
            c.ns_per_tick = ns;
        }
        Ok(g)
    }

    /// [`Self::new`] on a device whose march time per work tick is `ns_per_tick` in place of the speed table's (a
    /// test gives it to stand for a slower or faster device): refused by [`check_device_speed`] against
    /// [`DISPATCH_WAIT`]. The work budget is [`MARCH_WORK_BUDGET`] whatever the speed.
    #[doc(hidden)]
    pub fn new_with_ns_per_tick(field: &SheetField, ns_per_tick: f64) -> Result<Self, GpuError> {
        check_device_speed(ns_per_tick, DISPATCH_WAIT)?;
        let g = Self::build(field, &|src| src)?;
        if let Ok(mut c) = g.control.lock() {
            c.ns_per_tick = ns_per_tick;
        }
        Ok(g)
    }

    /// The march time per work tick this field's device was admitted with (see [`check_device_speed`]).
    #[must_use]
    pub fn ns_per_tick(&self) -> f64 {
        self.control.lock().map_or(f64::NAN, |c| c.ns_per_tick)
    }

    /// [`Self::new`] with the march shader's text passed through `edit` first (the module's tests use it to break
    /// one property of the shader on purpose).
    fn build(field: &SheetField, edit: &dyn Fn(String) -> String) -> Result<Self, GpuError> {
        let mut sheets = Vec::new();
        let mut levels = Vec::new();
        let (mut index, mut rows, mut qbytes, mut scoff) = (
            Vec::<i32>::new(),
            Vec::<i32>::new(),
            Vec::<u8>::new(),
            Vec::<f32>::new(),
        );
        let mut bands = Vec::<u32>::new();
        // the kink blocks ([`super::kink`]), appended to the index table last; a sheet's block offset is relative to
        // the kink table until the table's base is known, NONE when the sheet has none
        let mut kinks = Vec::<u32>::new();
        // an empty field has no sheet or level record to bind (a binding smaller than one record fails at dispatch):
        // refused here, typed, before any device work; the CPU answers it (every ray a miss)
        if field.sheets.is_empty() {
            return Err(GpuError::InvalidFormat(
                "sheet field: no sheets to upload (march an empty field on the CPU: every ray misses)".into(),
            ));
        }
        if field.sheets.len() > MAX_FIELD_SHEETS as usize {
            return Err(GpuError::InvalidFormat(format!(
                "sheet field: {} sheets, over the upload's MAX_FIELD_SHEETS {MAX_FIELD_SHEETS}",
                field.sheets.len()
            )));
        }
        for (i, sh) in field.sheets.iter().enumerate() {
            sh.validate()
                .map_err(|e| GpuError::InvalidFormat(format!("sheet field: sheet {i}: {e}")))?;
            if sh.levels.len() > MAX_SHEET_LEVELS as usize {
                return Err(GpuError::InvalidFormat(format!(
                    "sheet field: a sheet with {} levels, over MAX_SHEET_LEVELS {MAX_SHEET_LEVELS}",
                    sh.levels.len()
                )));
            }
            if let Some(ks) = &sh.kink {
                let (t, g) = ks.max_cell_refs();
                if t > MAX_KINK_CELL_TRIS || g > MAX_KINK_CELL_SEGS {
                    return Err(GpuError::InvalidFormat(format!(
                        "sheet field: a kink cell with {t} triangles and {g} segments, over MAX_KINK_CELL_TRIS \
                         {MAX_KINK_CELL_TRIS} / MAX_KINK_CELL_SEGS {MAX_KINK_CELL_SEGS}"
                    )));
                }
            }
            let kink_at = match &sh.kink {
                None => u32::MAX,
                Some(ks) => {
                    let block = ks
                        .gpu_block()
                        .map_err(|e| GpuError::InvalidFormat(format!("sheet field: {e}")))?;
                    // the shader adds the block's own offsets to its start: the end must stay a u32 offset
                    let at = u32::try_from(kinks.len())
                        .ok()
                        .filter(|&at| u32::try_from(u64::from(at) + block.len() as u64).is_ok())
                        .ok_or_else(|| {
                            GpuError::InvalidFormat(format!(
                                "sheet field: kink blocks of {} + {} words, over u32 offsets",
                                kinks.len(),
                                block.len()
                            ))
                        })?;
                    kinks.extend(block);
                    at
                }
            };
            sheets.push(SheetInfo {
                lo: sh.lo.to_array(),
                hc: sh.hc,
                hi: sh.hi.to_array(),
                level_start: levels.len() as u32,
                level_count: sh.levels.len() as u32,
                _pad: [kink_at, 0, 0],
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
                    band_off: bands.len() as u32,
                    _pad: [0; 2],
                    rows_off: [0; 8],
                    q_off: [0; 8],
                    sc_off: [0; 8],
                    nodes: [0; 8],
                    st_off: [0; 8],
                };
                bands.extend_from_slice(&lv.band);
                index.extend_from_slice(&lv.index);
                for (c, ch) in lv.channels.iter().enumerate() {
                    li.rows_off[c] = rows.len() as u32;
                    rows.extend_from_slice(&ch.rows);
                    li.q_off[c] = qbytes.len() as u32;
                    li.nodes[c] = ch.nodes;
                    li.st_off[c] = bands.len() as u32;
                    bands.extend_from_slice(&ch.starts);
                    debug_assert!(
                        ch.nodes == super::BAND
                            || ch.q.len() == ch.scale.len() * (ch.nodes as usize).pow(3)
                    );
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
            index.len()
                + rows.len()
                + bvh.nodes.len()
                + bvh.leaf_entries.len()
                + bvh.parents.len()
                + bands.len()
                + kinks.len(),
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
        let b_band = base(&ix);
        ix.extend_from_slice(&bands);
        let b_kink = base(&ix);
        ix.extend_from_slice(&kinks);
        for si in &mut sheets {
            if si._pad[0] != u32::MAX {
                si._pad[0] += b_kink;
            }
        }
        let bases = Bases {
            band: b_band,
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
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sheet field layout"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, ro),
                entry(2, ro),
                entry(3, ro),
                entry(4, ro),
                entry(5, ro),
                entry(6, ro),
                entry(7, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let module = context.shader_module_with_ceiling(
            MARCH_KERNEL,
            wgpu::ShaderSource::Wgsl(
                edit(march_shader_margin(max_live, field.step_margin())).into(),
            ),
            &march_maxima(),
            march_ceiling(),
        )?;
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
        let resources = SheetFieldResources {
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
            max_live,
            control: std::sync::Mutex::new(DispatchControl {
                next: FIRST_DISPATCH,
                target: DISPATCH_TARGET,
                wait: DISPATCH_WAIT,
                work_budget: MARCH_WORK_BUDGET,
                ns_per_tick: f64::NAN,
                spread_block: SPREAD_BLOCK,
                stats: MarchStats::default(),
                #[cfg(test)]
                cut: (0, 0),
            }),
            marker: std::sync::atomic::AtomicU32::new(0),
        };
        context.set_wait_limit(DISPATCH_WAIT);
        Ok(context.hold(resources))
    }

    fn uniforms(&self, count: u32, steps: u32, dir: Vec3, tmax: f32) -> Uniforms {
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
            band_base: b.band,
            work_budget: self
                .control
                .lock()
                .map_or(MARCH_WORK_BUDGET, |c| c.work_budget),
            bn_visits: 3 * (b.bn_len / 8) + 1,
            _pad: [0; 3],
        }
    }

    fn run(
        &self,
        pipeline: &wgpu::ComputePipeline,
        input: &[[f32; 4]],
        out_vec4s: usize,
        u: Uniforms,
    ) -> Result<Vec<[f32; 4]>, GpuError> {
        self.run_groups(pipeline, input, out_vec4s, u, None)
    }

    /// [`Self::run`] with the workgroup count taken as given when `groups` is set (only the module's tests
    /// do that, to make the device refuse a dispatch past the size check).
    fn run_groups(
        &self,
        pipeline: &wgpu::ComputePipeline,
        input: &[[f32; 4]],
        out_vec4s: usize,
        u: Uniforms,
        groups: Option<u32>,
    ) -> Result<Vec<[f32; 4]>, GpuError> {
        let (out, unwritten) = self.run_groups_raw(pipeline, input, out_vec4s, u, groups)?;
        if let Some(&first) = unwritten.first() {
            return Err(GpuError::BufferMapping(format!(
                "the sheet field dispatch left {} of {} results unwritten (first {first}): its output is not read",
                unwritten.len(),
                input.len()
            )));
        }
        Ok(out)
    }

    /// [`Self::run_groups`] returning, with the output, the results the dispatch did not write at all (their first
    /// vec4 still holds [`NOT_WRITTEN`]); the caller decides what to do with them.
    fn run_groups_raw(
        &self,
        pipeline: &wgpu::ComputePipeline,
        input: &[[f32; 4]],
        out_vec4s: usize,
        u: Uniforms,
        groups: Option<u32>,
    ) -> Result<(Vec<[f32; 4]>, Vec<usize>), GpuError> {
        // a dispatch of this field that did not return may still be running: no more work goes to this device
        if self.context().is_abandoned() {
            return Err(GpuError::BufferMapping(
                "an earlier dispatch of this sheet field did not finish: it is not usable, create another".to_string(),
            ));
        }
        let device = self.context().device();
        let queue = self.context().queue();
        let n = input.len().max(1);
        // a dispatch over the device's workgroups per dimension is invalid: the submit is dropped and the output
        // would be read unwritten, so it is refused here
        let limit = device.limits().max_compute_workgroups_per_dimension;
        if groups.is_none() && (n as u64).div_ceil(u64::from(WG)) > u64::from(limit) {
            return Err(GpuError::DispatchTooLarge {
                points: n,
                workgroup_size: WG,
                limit,
            });
        }
        // a validation error (an invalid dispatch, buffer or binding) goes to this scope and becomes an error, instead
        // of the uncaptured-error handler (a panic by default, or nothing when a handler swallows it)
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let b_u = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sheet field uniforms"),
            contents: bytemuck::cast_slice(&[u]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let b_in = storage_init(device, "sheet field input", bytemuck::cast_slice(input));
        let out_bytes = (n * out_vec4s * 16) as u64;
        // every output word starts as NOT_WRITTEN (a NaN no arithmetic produces): a zeroed slot would decode as a
        // miss at t 0, so a result the dispatch did not write could pass as an answer; it is refused below instead
        let b_out = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sheet field output"),
            contents: bytemuck::cast_slice(&vec![NOT_WRITTEN; n * out_vec4s * 4]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        // the staging buffer ends with a marker the same submit writes after the dispatch: a submit the device drops
        // (a validation error is reported only to the uncaptured-error handler) leaves it zero, and the read is refused
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sheet field staging"),
            size: out_bytes + 16,
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
            pass.dispatch_workgroups(groups.unwrap_or_else(|| (n as u32).div_ceil(WG)), 1, 1);
        }
        enc.copy_buffer_to_buffer(&b_out, 0, &staging, 0, out_bytes);
        let nonce = self
            .marker
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            .wrapping_add(1)
            .max(1);
        let b_mark = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sheet field dispatch marker"),
            contents: bytemuck::cast_slice(&[nonce, 0u32, 0, 0]),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        enc.copy_buffer_to_buffer(&b_mark, 0, &staging, out_bytes, 16);
        // the GPU's writes -> the CPU's read of the mapped staging buffer (see GpuContext::before_host_read)
        self.context().before_host_read(&mut enc);
        queue.submit(std::iter::once(enc.finish()));
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(e) = pollster::block_on(device.pop_error_scope()) {
            return Err(GpuError::BufferMapping(format!(
                "the sheet field dispatch was refused by the device: {e}"
            )));
        }
        #[cfg(target_arch = "wasm32")]
        drop(device.pop_error_scope());
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        if let Err(e) = self.context().wait_for_mapping(&receiver) {
            if self.context().is_abandoned() {
                let limit_ms = self.control.lock().map_or(0, |c| c.wait.as_millis() as u64);
                return Err(GpuError::DispatchDidNotFinish {
                    rays: input.len(),
                    limit_ms,
                });
            }
            return Err(e);
        }
        let (out, found) = {
            let mapped = slice.get_mapped_range();
            let data: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
            let found = data[n * out_vec4s][0].to_bits();
            (data[..input.len() * out_vec4s].to_vec(), found)
        };
        staging.unmap();
        if found != nonce {
            return Err(GpuError::BufferMapping(format!(
                "the sheet field dispatch did not run (marker {found}, expected {nonce}): its output is not read"
            )));
        }
        // the first vec4 of each result is written by every path of both kernels (eval and march, overflow included)
        let unwritten: Vec<usize> = (0..input.len())
            .filter(|&k| {
                out[k * out_vec4s]
                    .iter()
                    .any(|w| w.to_bits() == NOT_WRITTEN)
            })
            .collect();
        Ok((out, unwritten))
    }

    /// The dispatch sizes, their times and the per-ray work of the last march call.
    #[must_use]
    pub fn last_march_stats(&self) -> MarchStats {
        self.control
            .lock()
            .map(|c| c.stats.clone())
            .unwrap_or_default()
    }

    /// Another target duration for one march dispatch (sizes adapt toward it).
    pub fn set_dispatch_target(&self, target: std::time::Duration) {
        if let Ok(mut c) = self.control.lock() {
            c.target = target;
        }
    }

    /// Another wait bound for one dispatch; over it, the read is a [`GpuError::DispatchDidNotFinish`] and the
    /// device is not used again.
    pub fn set_dispatch_wait(&self, wait: std::time::Duration) {
        if let Ok(mut c) = self.control.lock() {
            c.wait = wait;
        }
        self.context().set_wait_limit(wait);
    }

    /// Another block size of the march order (see [`SPREAD_BLOCK`]; 0 = contiguous).
    pub fn set_spread_block(&self, block: usize) {
        if let Ok(mut c) = self.control.lock() {
            c.spread_block = block;
        }
    }

    /// Another per-ray work budget (see [`MARCH_WORK_BUDGET`]), at most [`MARCH_WORK_BUDGET`] (the kernel's ceiling is
    /// stated for it).
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidFormat`] for a budget over [`MARCH_WORK_BUDGET`].
    pub fn set_work_budget(&self, budget: u32) -> Result<(), GpuError> {
        if u64::from(budget) > u64::from(MARCH_WORK_BUDGET) {
            return Err(GpuError::InvalidFormat(format!(
                "sheet field march: work budget {budget} over MARCH_WORK_BUDGET {MARCH_WORK_BUDGET}"
            )));
        }
        if let Ok(mut c) = self.control.lock() {
            c.work_budget = budget;
        }
        Ok(())
    }

    /// The per-ray list size this field's march was compiled with ([`max_live_for`] of [`live_bound`]).
    #[must_use]
    pub fn max_live(&self) -> u32 {
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
        if let Some(i) = out.iter().position(|o| o[2] == 2.0) {
            return Err(unfinished_walk(i));
        }
        if let Some(i) = out.iter().position(|o| o[2] == 3.0) {
            return Err(GpuError::InvalidFormat(format!(
                "sheet field: the evaluation of point {i} ran over the work budget"
            )));
        }
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
        let steps =
            super::decided_steps(tmax).map_err(|e| GpuError::InvalidFormat(e.to_string()))?;
        self.march_batch(origins, dir, tmax, steps)
    }

    /// March rays from `origins` along the unit `dir` up to `tmax` with at most `steps` steps.
    ///
    /// The rays go to the GPU in dispatches whose size follows the measured time of the previous one (see
    /// [`DISPATCH_TARGET`] and [`next_dispatch`]); the results do not depend on the sizes. A dispatch that does
    /// not return within the wait bound is a [`GpuError::DispatchDidNotFinish`]. The sizes, times and per-ray
    /// work are in [`Self::last_march_stats`].
    pub fn march_batch(
        &self,
        origins: &[Vec3],
        dir: Vec3,
        tmax: f32,
        steps: u32,
    ) -> Result<Vec<MarchResult>, GpuError> {
        if steps > MAX_MARCH_STEPS {
            return Err(GpuError::InvalidFormat(format!(
                "sheet field march: {steps} steps, over MAX_MARCH_STEPS {MAX_MARCH_STEPS}"
            )));
        }
        if origins.is_empty() {
            return Ok(Vec::new());
        }
        let (target, budget) = self
            .control
            .lock()
            .map(|c| (c.target, c.work_budget))
            .map_err(|_| GpuError::InvalidFormat("sheet field dispatch control poisoned".into()))?;
        let mut stats = MarchStats::default();
        let n = origins.len();
        let block = self.control.lock().map_or(SPREAD_BLOCK, |c| c.spread_block);
        let order = spread_blocks(n, block);
        let mut res = vec![MarchResult::Miss; n];
        let mut at = 0;
        while at < n {
            let limits = self.context().device().limits();
            let most = dispatch_cap(
                limits.max_compute_workgroups_per_dimension,
                u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size),
            );
            let size = self
                .control
                .lock()
                .map(|c| c.next)
                .unwrap_or(FIRST_DISPATCH)
                .max(MIN_DISPATCH)
                .min(most)
                .min(n - at);
            let mut pending: Vec<usize> = order[at..at + size].to_vec();
            let mut piece_size = size;
            let mut round = 0u32;
            loop {
                let mut left = Vec::new();
                for piece in pending.chunks(piece_size) {
                    let part: Vec<Vec3> = piece.iter().map(|&i| origins[i]).collect();
                    #[allow(
                        clippy::disallowed_methods,
                        reason = "dispatch sizes follow the measured time of the last dispatch; the results do not depend on it (each ray is marched whole in one dispatch)"
                    )]
                    let t0 = std::time::Instant::now();
                    stats.dispatch_work.push(0);
                    stats.dispatch_round.push(round);
                    let unwritten = self.march_dispatch(
                        &part, piece, dir, tmax, steps, budget, &mut res, &mut stats,
                    )?;
                    #[allow(
                        clippy::disallowed_methods,
                        reason = "dispatch sizes follow the measured time of the last dispatch; the results do not depend on it (each ray is marched whole in one dispatch)"
                    )]
                    let secs = t0.elapsed().as_secs_f64();
                    stats.dispatches.push((piece.len(), secs));
                    if round > 0 {
                        stats.smallest_reissue = if stats.smallest_reissue == 0 {
                            piece.len()
                        } else {
                            stats.smallest_reissue.min(piece.len())
                        };
                    }
                    if let Ok(mut c) = self.control.lock() {
                        c.next = if unwritten.is_empty() {
                            next_dispatch(piece.len(), secs, target).min(most)
                        } else {
                            // a dispatch the device cut short: the next ones are at most half of it
                            (piece.len() / 2).max(MIN_DISPATCH).min(c.next)
                        };
                    }
                    if !unwritten.is_empty() {
                        stats.cut_dispatches += 1;
                        stats.reissued_rays += unwritten.len() as u64;
                        left.extend(unwritten);
                    }
                }
                if left.is_empty() {
                    break;
                }
                round += 1;
                if round > MAX_REISSUE_ROUNDS {
                    if let Ok(mut c) = self.control.lock() {
                        c.stats = stats;
                    }
                    return Err(GpuError::DispatchLeftUnwritten {
                        unwritten: left.len(),
                        rays: size,
                        rounds: MAX_REISSUE_ROUNDS,
                    });
                }
                // the rays left unwritten go again, in dispatches half the size of the cut ones (never below one
                // workgroup)
                piece_size = (piece_size / 2).max(WG as usize);
                pending = left;
            }
            at += size;
        }
        if let Ok(mut c) = self.control.lock() {
            c.stats = stats;
        }
        Ok(res)
    }

    #[allow(clippy::too_many_arguments)]
    fn march_dispatch(
        &self,
        origins: &[Vec3],
        index: &[usize],
        dir: Vec3,
        tmax: f32,
        steps: u32,
        budget: u32,
        res: &mut [MarchResult],
        stats: &mut MarchStats,
    ) -> Result<Vec<usize>, GpuError> {
        let input: Vec<[f32; 4]> = origins.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        #[cfg(test)]
        let groups = self.control.lock().ok().and_then(|mut c| {
            (c.cut.1 > 0).then(|| {
                c.cut.1 -= 1;
                c.cut.0
            })
        });
        #[cfg(not(test))]
        let groups = None;
        let (out, _) = self.run_groups_raw(
            &self.march_pipeline,
            &input,
            4,
            self.uniforms(origins.len() as u32, steps, dir, tmax),
            groups,
        )?;
        let (rows, rest) = out.as_chunks::<4>();
        debug_assert!(rest.is_empty(), "4 output words a ray");
        let mut unwritten = Vec::new();
        for (j, r) in rows.iter().enumerate() {
            let i = index[j];
            // a ray's result is 4 stores (the overflow path only the first, and it is refused below): a ray with
            // any of its first vec4's words, or of the others' unless it overflowed, still NOT_WRITTEN was not
            // (fully) written by this dispatch and is issued again
            let poisoned = |v: &[f32; 4]| v.iter().any(|w| w.to_bits() == NOT_WRITTEN);
            let early = matches!(r[0][1] as i32, 3 | 5 | 6); // overflow, unsorted and an unfinished walk write only the first vec4
            if poisoned(&r[0]) || (!early && r[1..].iter().any(poisoned)) {
                unwritten.push(i);
                continue;
            }
            let kind = r[0][1] as i32;
            let work = r[1][3].to_bits();
            *stats.dispatch_work.last_mut().expect("a dispatch slot") += u64::from(work);
            if work > stats.max_work || stats.max_work_ray.is_none() {
                stats.max_work = work;
                stats.max_work_ray = Some(i);
            }
            if u64::from(work) * 2 >= u64::from(budget) {
                stats.over_half_budget += 1;
            }
            if kind == 1 && r[0][3] != 0.0 {
                // an attribute the hit needed is outside its band: that attribute is None for this ray (as on the
                // CPU), counted, and the batch goes on
                stats.band_misses += 1;
            }
            res[i] = match kind {
                1 => MarchResult::Hit(Hit {
                    t: r[0][0],
                    sheet: r[0][2] as usize,
                    normal: Vec3::new(r[1][0], r[1][1], r[1][2]),
                    uv: (r[2][2] != 0.0).then_some([r[2][0], r[2][1]]),
                    attr_normal: (r[3][3] != 0.0).then(|| Vec3::new(r[3][0], r[3][1], r[3][2])),
                }),
                0 => MarchResult::Miss,
                2 => MarchResult::Undecided,
                6 => return Err(unfinished_walk(i)),
                5 => {
                    return Err(GpuError::InvalidFormat(format!(
                        "sheet field march: ray {i} met a live-sheet list that is not ascending (the crossing check pairs \
                         the two lists by a merge and would pair the wrong sheets)"
                    )))
                }
                4 => {
                    stats.over_budget += 1;
                    MarchResult::Undecided
                }
                _ => {
                    return Err(GpuError::InvalidFormat(format!(
                        "sheet field march: more than {} sheet boxes hold one point",
                        self.max_live
                    )))
                }
            };
        }
        Ok(unwritten)
    }
}

const SHADER: &str = r"
struct SheetInfo { lo: vec3<f32>, hc: f32, hi: vec3<f32>, level_start: u32, level_count: u32, p0: u32, p1: u32, p2: u32 };
struct LevelInfo { lo: vec3<f32>, h: f32, nb: vec3<u32>, index_off: u32, nch: u32, band_off: u32, p1: u32, p2: u32,
                   rows_off: array<vec4<u32>, 2>, q_off: array<vec4<u32>, 2>, sc_off: array<vec4<u32>, 2>,
                   nodes: array<vec4<u32>, 2>, st_off: array<vec4<u32>, 2> };
struct Uniforms { n_sheets: u32, rule: u32, count: u32, steps: u32, dir: vec3<f32>, tmax: f32,
                  index_base: u32, rows_base: u32, bn_base: u32, bl_base: u32, bp_base: u32, bn_len: u32, band_base: u32, work_budget: u32,
                  bn_visits: u32, u_p0: u32, u_p1: u32, u_p2: u32 };
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
const BAND_WORDS: u32 = 23u;
const HALF_DIAG: f32 = 0.8660254;
const SQRT3: f32 = 1.7320508;
const JUMP_S: f32 = 0.02;
const MASKED_FLOOR: f32 = __MASKED_FLOOR__;
const MASKED_FLOOR_PER_H: f32 = __MASKED_FLOOR_PER_H__;
const MAX_LIVE: u32 = __MAX_LIVE__u;
const MAX_LEVELS: u32 = __MAX_LEVELS__u;
const MAX_CHANNELS: u32 = 8u;
const MAX_KINK_TRIS: u32 = __MAX_KINK_TRIS__u;
const MAX_KINK_SEGS: u32 = __MAX_KINK_SEGS__u;
const BVH_LEAF: u32 = __BVH_LEAF__u;
const MASK_WORDS: u32 = 16u;

// the invocation's work: one tick per loop iteration, whatever the loop; over U.work_budget every loop leaves at its
// next iteration and the result is refused (a march ray reports undecided, an evaluation is an error)
var<private> work: u32;
fn tick() -> bool {
    work = work + 1u;
    return work <= U.work_budget;
}

struct Sample { f: f32, g: f32, inside: bool, masked: bool, uv: vec2<f32>, has_uv: bool, s: f32, db: f32, db_step: f32, an: vec3<f32>, has_an: bool, band_miss: u32, lip: f32 };

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

// kink cells (the CPU sheet_field::kink): the exact d_b from the cell's triangles and rim segments. A sheet's block
// starts at SheetInfo.p0 in IX (NONE: no kink cells); layout in kink::gpu_layout.
fn kf(i: u32) -> f32 { return bitcast<f32>(IX[i]); }
fn kv(hdr: u32, v: u32) -> vec3<f32> { let b = hdr + IX[hdr + 4u] + 7u * v; return vec3<f32>(kf(b), kf(b + 1u), kf(b + 2u)); }

// closest point of triangle abc to p and its feature: xyz the point, w the feature (0 face, 1 + e edge e = ab, bc, ca,
// 4 + i corner i) (Ericson 5.1.5, line for line with the CPU closest_on_tri; a degenerate triangle by its edges)
fn k_closest(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec4<f32> {
    let ab = b - a;
    let ac = c - a;
    let cr = cross(ab, ac);
    if (dot(cr, cr) <= 0.0) {
        var best = vec4<f32>(a, 4.0);
        var bd = 1e30;
        for (var i = 0u; i < 3u; i = i + 1u) { if (!tick()) { break; }
            var x = a; var y = b;
            if (i == 1u) { x = b; y = c; }
            if (i == 2u) { x = c; y = a; }
            let e = y - x;
            let ee = dot(e, e);
            var t = 0.0;
            if (ee > 0.0) { t = clamp(dot(p - x, e) / ee, 0.0, 1.0); }
            let q = x + e * t;
            let d = dot(p - q, p - q);
            if (d < bd) {
                bd = d;
                var f = 1.0 + f32(i);
                if (t <= 0.0) { f = 4.0 + f32(i); }
                if (t >= 1.0) { f = 4.0 + f32((i + 1u) % 3u); }
                best = vec4<f32>(q, f);
            }
        }
        return best;
    }
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return vec4<f32>(a, 4.0); }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return vec4<f32>(b, 5.0); }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { return vec4<f32>(a + ab * (d1 / (d1 - d3)), 1.0); }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return vec4<f32>(c, 6.0); }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { return vec4<f32>(a + ac * (d2 / (d2 - d6)), 3.0); }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        return vec4<f32>(b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))), 2.0);
    }
    let den = 1.0 / (va + vb + vc);
    return vec4<f32>(a + ab * (vb * den) + ac * (vc * den), 0.0);
}

// the cell record (x = 1 found, y = ref offset, z = triangles | segments << 16, w = flags: 1 exact d_b, 2 exact s) of
// (level li, active brick bk, cell ci) of the sheet whose kink block starts at hdr
fn kink_cell(hdr: u32, li: u32, bk: u32, ci: u32) -> vec4<u32> {
    if (li >= IX[hdr + 9u]) { return vec4<u32>(0u); }
    let soff = IX[hdr + 12u + 2u * li];
    let slen = IX[hdr + 13u + 2u * li];
    if (bk >= slen) { return vec4<u32>(0u); }
    let slot = IX[hdr + soff + bk];
    if (slot == NONE) { return vec4<u32>(0u); }
    let kb = hdr + soff + slen + 17u * slot;
    let wi = ci / 32u;
    let bit = ci % 32u;
    let m = IX[kb + wi];
    if (((m >> bit) & 1u) == 0u) { return vec4<u32>(0u); }
    var below = countOneBits(m & ((1u << bit) - 1u));
    for (var j = 0u; j < MASK_WORDS; j = j + 1u) { if (!tick()) { break; } if (j >= wi) { break; } below = below + countOneBits(IX[kb + j]); }
    let rec = hdr + IX[hdr + 7u] + 3u * (IX[kb + 16u] + below);
    return vec4<u32>(1u, IX[rec], IX[rec + 1u], IX[rec + 2u]);
}

// the exact values at world point pw in the flagged cell rc (kink_cell): x s, y d_b, z the unsigned distance, w the
// flags (the CPU KinkSheet::exact_at, line for line)
// the nearest face (its index) and the closest point (local) of the last kink_exact, for the attributes
var<private> k_face: u32;
var<private> k_q: vec3<f32>;

// barycentric weights of q in triangle abc (the CPU kink::bary, the builder's formula)
fn k_bary(q: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let v0 = b - a; let v1 = c - a; let v2 = q - a;
    let d00 = dot(v0, v0); let d01 = dot(v0, v1); let d11 = dot(v1, v1); let d20 = dot(v2, v0); let d21 = dot(v2, v1);
    let den = d00 * d11 - d01 * d01;
    if (den == 0.0) { return vec3<f32>(1.0, 0.0, 0.0); }
    let v = (d11 * d20 - d01 * d21) / den;
    let w = (d00 * d21 - d01 * d20) / den;
    return vec3<f32>(1.0 - v - w, v, w);
}

fn kink_exact(hdr: u32, rc: vec4<u32>, pw: vec3<f32>) -> vec4<f32> {
    let ntri = rc.z & 0xffffu;
    let nseg = rc.z >> 16u;
    let refs = hdr + IX[hdr + 8u] + rc.y;
    // local coordinates: the origin is a whole number of mm, so p - origin is exact
    let p = pw - vec3<f32>(kf(hdr + 1u), kf(hdr + 2u), kf(hdr + 3u));
    var best = 1e30;
    var fb = NONE;
    var q = vec3<f32>(0.0);
    var feat = 0.0;
    for (var i = 0u; i < MAX_KINK_TRIS; i = i + 1u) { if (!tick()) { break; }
        if (i >= ntri) { break; }
        let fo = hdr + IX[hdr + 5u] + 16u * IX[refs + i];
        let cq = k_closest(p, kv(hdr, IX[fo]), kv(hdr, IX[fo + 1u]), kv(hdr, IX[fo + 2u]));
        let d = length(p - cq.xyz);
        if (d < best) { best = d; fb = fo; q = cq.xyz; feat = cq.w; }
    }
    if (fb == NONE) { k_face = NONE; return vec4<f32>(1e3, 1e3, 1e3, f32(rc.w)); }
    k_face = (fb - (hdr + IX[hdr + 5u])) / 16u;
    k_q = q;
    let r = p - q;
    let ud = length(r);
    let fn_ = vec3<f32>(kf(fb + 3u), kf(fb + 4u), kf(fb + 5u));
    var bnd = false;
    var nb = fn_;
    var ns = fn_;
    if (feat >= 0.5 && feat < 3.5) {
        let e = u32(feat) - 1u;
        bnd = ((IX[fb + 6u] >> e) & 1u) == 1u;
        // edge e (ab, bc, ca) is opposite corner (e + 2) % 3
        let eo = fb + 7u + 3u * ((e + 2u) % 3u);
        ns = vec3<f32>(kf(eo), kf(eo + 1u), kf(eo + 2u));
    } else if (feat >= 3.5) {
        let vb = hdr + IX[hdr + 4u] + 7u * IX[fb + u32(feat) - 4u];
        ns = vec3<f32>(kf(vb + 3u), kf(vb + 4u), kf(vb + 5u));
        if (IX[vb + 6u] == 1u) { bnd = true; nb = ns / max(length(ns), 1e-12); }
    }
    var s = 0.0;
    var db = 0.0;
    if ((rc.w & 2u) != 0u) {
        if (bnd) { s = dot(r, nb); }
        else if (ud >= 1e-12) { let sn = dot(r, ns); s = select(select(0.0, -ud, sn < 0.0), ud, sn > 0.0); }
    }
    if ((rc.w & 1u) != 0u) {
        if (bnd) {
            let sn = dot(r, nb);
            db = sqrt(max(dot(r, r) - sn * sn, 0.0));
        } else {
            var dr = 1e3;
            for (var i = 0u; i < MAX_KINK_SEGS; i = i + 1u) { if (!tick()) { break; }
                if (i >= nseg) { break; }
                let so = hdr + IX[hdr + 6u] + 2u * IX[refs + ntri + i];
                let a = kv(hdr, IX[so]);
                let e = kv(hdr, IX[so + 1u]) - a;
                let ee = dot(e, e);
                var t = 0.0;
                if (ee > 0.0) { t = clamp(dot(q - a, e) / ee, 0.0, 1.0); }
                dr = min(dr, length(q - (a + e * t)));
            }
            db = -max(min(dr, kf(hdr) - ud), 0.0);
        }
    }
    return vec4<f32>(s, db, ud, f32(rc.w));
}

// the exact (s, unsigned distance, 1) at p when p's answering cell of sheet k is an s cell, else z = 0
fn exact_s_here(k: u32, p: vec3<f32>) -> vec3<f32> {
    let sh = SHEETS[k];
    if (sh.p0 == NONE) { return vec3<f32>(0.0); }
    for (var li = 0u; li < MAX_LEVELS; li = li + 1u) { if (!tick()) { break; }
        if (li >= sh.level_count) { break; }
        let lv = LEVELS[sh.level_start + li];
        let u = (p - lv.lo) / lv.h;
        let b = f32(BRICK);
        let bi = floor(u / b);
        if (any(bi < vec3<f32>(0.0)) || any(bi >= vec3<f32>(lv.nb))) { continue; }
        let bu = vec3<u32>(bi);
        let k_ = bitcast<i32>(IX[U.index_base + lv.index_off + (bu.x * lv.nb.y + bu.y) * lv.nb.z + bu.z]);
        if (k_ < 0) { continue; }
        let c = min(vec3<u32>(floor(u - bi * b)), vec3<u32>(BRICK - 1u));
        let rc = kink_cell(sh.p0, li, u32(k_), (c.x * BRICK + c.y) * BRICK + c.z);
        if (rc.x == 0u || (rc.w & 2u) == 0u) { return vec3<f32>(0.0); }
        let e = kink_exact(sh.p0, rc, p);
        return vec3<f32>(e.x, e.z, 1.0);
    }
    return vec3<f32>(0.0);
}

// the first accepted crossing of the exact s in [ta, tb] inside one s cell (the CPU Sheet::exact_cell_root): x 1 root /
// 0 clear / 2 capped, y the root t, z s at the last sample
fn exact_or_sample(k: u32, p: vec3<f32>) -> vec2<f32> {
    let e = exact_s_here(k, p);
    if (e.z == 1.0) { return e.xy; }
    let c = sample_sheet(k, p, 2u);
    return vec2<f32>(c.s, c.f);
}

fn exact_cell_root(k: u32, o: vec3<f32>, d: vec3<f32>, ta: f32, tb: f32, h: f32) -> vec3<f32> {
    let sub = h / 512.0;
    // past the s cell (a sample that rounds into a cell that is not an s cell) the sheet's own sample stands in, so the
    // last sub-step's sign change is still seen (the CPU exact_cell_root)
    let e0 = exact_or_sample(k, o + d * ta);
    var a = ta; var sa = e0.x; var ua = e0.y;
    var n = 0u;
    for (var walk = 0u; walk < 2048u; walk = walk + 1u) { if (!tick()) { break; }
        if (a >= tb) { break; }
        if (n >= 2048u) { return vec3<f32>(2.0, 0.0, sa); }
        n = n + 1u;
        let b = min(a + max(ua - JUMP_S, sub), tb);
        let eb = exact_or_sample(k, o + d * b);
        if ((sa < 0.0) != (eb.x < 0.0)) {
            var l = a; var r = b; var sl = sa;
            for (var it = 0; it < 40; it = it + 1) { if (!tick()) { break; }
                let m = 0.5 * (l + r);
                let em = exact_or_sample(k, o + d * m);
                n = n + 1u;
                if ((em.x < 0.0) == (sl < 0.0)) { l = m; sl = em.x; } else { r = m; }
            }
            let tm = 0.5 * (l + r);
            if (crossing_inside(k, o + d * tm)) { return vec3<f32>(1.0, tm, eb.x); }
        }
        a = b; sa = eb.x; ua = eb.y;
    }
    // every pass evaluates at least once, so n reaches 2048 no later than the pass counter: past the loop is past the cap
    if (a < tb) { return vec3<f32>(2.0, 0.0, sa); }
    return vec3<f32>(0.0, 0.0, sa);
}

// sheet k at p, interpolating the first nmax channels only (2 while marching; attributes once, at the hit)
fn sample_sheet(k: u32, p: vec3<f32>, nmax: u32) -> Sample {
    let sh = SHEETS[k];
    var out: Sample;
    for (var li = 0u; li < MAX_LEVELS; li = li + 1u) { if (!tick()) { break; }
        if (li >= sh.level_count) { break; }
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
        var ucorner = -1e30;
        // the attribute channels a corner with weight > 0 needed and the band did not store (bit = channel)
        var band_miss = 0u;
        var q0 = 0.0;
        var q1 = 0.0;
        let nch = min(lv.nch, nmax);
        for (var ci = 0u; ci < MAX_CHANNELS; ci = ci + 1u) { if (!tick()) { break; }
            if (ci >= nch) { break; }
            let row = bitcast<i32>(IX[U.rows_base + lv.rows_off[ci / 4u][ci % 4u] + u32(k_)]);
            var acc = 0.0;
            let nn = lv.nodes[ci / 4u][ci % 4u];
            if (row >= 0 && nn == 0u) {
                // a band: only the nodes in the brick's mask are stored, packed; a node outside it is a miss
                let sc = SC[2u * (lv.sc_off[ci / 4u][ci % 4u] + u32(row))];
                let off = SC[2u * (lv.sc_off[ci / 4u][ci % 4u] + u32(row)) + 1u];
                let mw = U.band_base + lv.band_off + u32(k_) * BAND_WORDS;
                let start = IX[U.band_base + lv.st_off[ci / 4u][ci % 4u] + u32(row)];
                for (var dx = 0u; dx < 2u; dx = dx + 1u) { if (!tick()) { break; }
                    for (var dy = 0u; dy < 2u; dy = dy + 1u) { if (!tick()) { break; }
                        for (var dz = 0u; dz < 2u; dz = dz + 1u) { if (!tick()) { break; }
                            let wx = select(1.0 - t.x, t.x, dx == 1u);
                            let wy = select(1.0 - t.y, t.y, dy == 1u);
                            let wz = select(1.0 - t.z, t.z, dz == 1u);
                            let node = ((c.x + dx) * NODES + (c.y + dy)) * NODES + (c.z + dz);
                            let word = IX[mw + node / 32u];
                            let bit = node % 32u;
                            if (((word >> bit) & 1u) == 0u) { if (wx * wy * wz > 0.0) { band_miss = band_miss | (1u << ci); } continue; }
                            var rank = countOneBits(word & ((1u << bit) - 1u));
                            for (var j = 0u; j < BAND_WORDS; j = j + 1u) { if (!tick()) { break; } if (j >= node / 32u) { break; } rank = rank + countOneBits(IX[mw + j]); }
                            acc = acc + (wx * wy * wz) * (qbyte(lv.q_off[ci / 4u][ci % 4u] + start + rank) * sc + off);
                        }
                    }
                }
            } else if (row >= 0 && nn != NODES) {
                // an attribute on a coarse grid: trilinear on its own nodes (every BRICK / (nn - 1)-th corner)
                let sc = SC[2u * (lv.sc_off[ci / 4u][ci % 4u] + u32(row))];
                let off = SC[2u * (lv.sc_off[ci / 4u][ci % 4u] + u32(row)) + 1u];
                let wc = w / f32(BRICK / (nn - 1u));
                let cc = min(vec3<u32>(max(floor(wc), vec3<f32>(0.0))), vec3<u32>(nn - 2u));
                let tc = wc - vec3<f32>(cc);
                let base = lv.q_off[ci / 4u][ci % 4u] + u32(row) * nn * nn * nn;
                for (var dx = 0u; dx < 2u; dx = dx + 1u) { if (!tick()) { break; }
                    for (var dy = 0u; dy < 2u; dy = dy + 1u) { if (!tick()) { break; }
                        for (var dz = 0u; dz < 2u; dz = dz + 1u) { if (!tick()) { break; }
                            let wx = select(1.0 - tc.x, tc.x, dx == 1u);
                            let wy = select(1.0 - tc.y, tc.y, dy == 1u);
                            let wz = select(1.0 - tc.z, tc.z, dz == 1u);
                            let node = ((cc.x + dx) * nn + (cc.y + dy)) * nn + (cc.z + dz);
                            acc = acc + (wx * wy * wz) * (qbyte(base + node) * sc + off);
                        }
                    }
                }
            } else if (row >= 0) {
                let sc = SC[2u * (lv.sc_off[ci / 4u][ci % 4u] + u32(row))];
                let off = SC[2u * (lv.sc_off[ci / 4u][ci % 4u] + u32(row)) + 1u];
                for (var dx = 0u; dx < 2u; dx = dx + 1u) { if (!tick()) { break; }
                    for (var dy = 0u; dy < 2u; dy = dy + 1u) { if (!tick()) { break; }
                        for (var dz = 0u; dz < 2u; dz = dz + 1u) { if (!tick()) { break; }
                            let wx = select(1.0 - t.x, t.x, dx == 1u);
                            let wy = select(1.0 - t.y, t.y, dy == 1u);
                            let wz = select(1.0 - t.z, t.z, dz == 1u);
                            let node = ((c.x + dx) * NODES + (c.y + dy)) * NODES + (c.z + dz);
                            let v = qbyte(lv.q_off[ci / 4u][ci % 4u] + u32(row) * BRICK_NODES + node) * sc + off;
                            acc = acc + (wx * wy * wz) * v;
                            if (ci == 0u) {
                                umin = min(umin, abs(v));
                                let e = t - vec3<f32>(f32(dx), f32(dy), f32(dz));
                                let dist = lv.h * sqrt(e.x * e.x + e.y * e.y + e.z * e.z);
                                ucorner = max(ucorner, abs(v) - dist);
                            }
                        }
                    }
                }
                if (ci == 0u) { q0 = sc; }
                if (ci == 1u) { q1 = sc; }
            }
            vals[ci] = acc;
        }
        if (sh.p0 != NONE && nch >= 2u) {
            let rc = kink_cell(sh.p0, li, u32(k_), (c.x * BRICK + c.y) * BRICK + c.z);
            if (rc.x == 1u) {
                let e = kink_exact(sh.p0, rc, p);
                if ((rc.w & 2u) != 0u) { vals[0] = e.x; }
                if ((rc.w & 1u) != 0u) { vals[1] = e.y; }
                // the attributes of the exact closest point (the CPU Sheet::sample: the face's corners by its weights)
                let hdr = sh.p0;
                if (k_face != NONE && (nch >= 4u)) {
                    let fo = hdr + IX[hdr + 5u] + 16u * k_face;
                    let w3 = k_bary(k_q, kv(hdr, IX[fo]), kv(hdr, IX[fo + 1u]), kv(hdr, IX[fo + 2u]));
                    if (IX[hdr + 10u] != NONE) {
                        let ub = hdr + IX[hdr + 10u] + 6u * k_face;
                        vals[2] = w3.x * kf(ub) + w3.y * kf(ub + 2u) + w3.z * kf(ub + 4u);
                        vals[3] = w3.x * kf(ub + 1u) + w3.y * kf(ub + 3u) + w3.z * kf(ub + 5u);
                        // the UV comes from the kink face, not from the band corners read above
                        band_miss = band_miss & ~0xCu;
                    }
                    if (nch >= 7u && IX[hdr + 11u] != NONE) {
                        let nb = hdr + IX[hdr + 11u] + 9u * k_face;
                        vals[4] = w3.x * kf(nb) + w3.y * kf(nb + 3u) + w3.z * kf(nb + 6u);
                        vals[5] = w3.x * kf(nb + 1u) + w3.y * kf(nb + 4u) + w3.z * kf(nb + 7u);
                        vals[6] = w3.x * kf(nb + 2u) + w3.y * kf(nb + 5u) + w3.z * kf(nb + 8u);
                        band_miss = band_miss & ~0x70u;
                    }
                }
            }
        }
        let s = vals[0];
        let db = max(vals[1], 0.0);
        var f = sqrt(s * s + db * db);
        var g = select(f, -f, s < 0.0);
        out.masked = false;
        if (U.rule == 1u && umin > HALF_DIAG * lv.h + q0) {
            // the per-corner bound (module docs of the CPU sheet field): max over corners of |s_c| - |p - c|
            // the floor never exceeds the distance from a point of a masked cell to the surface (over h/2):
            // SheetField's masked_floor
            f = max(max(umin - q0 - HALF_DIAG * lv.h, ucorner - q0), min(MASKED_FLOOR, MASKED_FLOOR_PER_H * lv.h));
            g = f;
            out.masked = true;
        }
        out.f = f; out.g = g; out.inside = true;
        out.s = s; out.db = vals[1]; out.db_step = q1;
        out.lip = SQRT3 * (1.0 + 2.0 * max(q0, q1) / lv.h);
        out.has_uv = nch >= 4u;
        out.band_miss = band_miss;
        out.has_an = nch >= 7u;
        out.an = vec3<f32>(vals[4], vals[5], vals[6]);
        out.uv = vec2<f32>(vals[2], vals[3]);
        return out;
    }
    let v = outside_bound(p, sh);
    out.f = v; out.g = v; out.inside = false; out.masked = false; out.has_uv = false; out.has_an = false; out.an = vec3<f32>(0.0);
    out.band_miss = 0u;
    out.s = 0.0; out.db = 0.0; out.db_step = 0.0; out.lip = 0.0;
    return out;
}

// a bisected sign change of sheet k's g at p is a crossing of the sheet: answered, unmasked, |s| <= JUMP_S,
// d_b <= step/2 (inside the boundary up to the rounding of the stored d_b), as SheetField::march
// how far a sample is from an accepted crossing (CPU Sample::reach)
fn reach(c: Sample) -> f32 {
    if (c.inside) { return max(abs(c.s), c.db - 0.5 * c.db_step); }
    return c.f;
}

fn crossing_inside(k: u32, p: vec3<f32>) -> bool {
    let c = sample_sheet(k, p, 2u);
    return c.inside && !c.masked && abs(c.s) <= JUMP_S && c.db <= 0.5 * c.db_step;
}

// M2 (CPU cellroot, same rules, f32 in cell units): the cell of sheet k's answering level at p, with the 8 stored s
// corners; x of the result: 0 none answers, 1 a cell (masked in w), lo in yzw of the second
// q: the point's coordinate inside the cell from the lookup's own arithmetic (w - c, in [0, 1] for the cell the lookup
// chose), so the walk's exit cannot be computed against a differently rounded cell origin
struct SCell { found: bool, masked: bool, lo: vec3<f32>, h: f32, c: array<f32, 8>, q: vec3<f32> };
fn s_cell(k: u32, p: vec3<f32>) -> SCell {
    let sh = SHEETS[k];
    var out: SCell;
    out.found = false;
    for (var li = 0u; li < MAX_LEVELS; li = li + 1u) { if (!tick()) { break; }
        if (li >= sh.level_count) { break; }
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
        out.q = clamp(w - vec3<f32>(c), vec3<f32>(0.0), vec3<f32>(1.0));
        let row = bitcast<i32>(IX[U.rows_base + lv.rows_off[0][0] + u32(k_)]);
        var umin = 1e30;
        var q0 = 0.0;
        if (row >= 0) {
            let sc = SC[2u * (lv.sc_off[0][0] + u32(row))];
            let off = SC[2u * (lv.sc_off[0][0] + u32(row)) + 1u];
            q0 = sc;
            for (var i = 0u; i < 8u; i = i + 1u) { if (!tick()) { break; }
                let node = ((c.x + ((i >> 2u) & 1u)) * NODES + (c.y + ((i >> 1u) & 1u))) * NODES + (c.z + (i & 1u));
                let v = qbyte(lv.q_off[0][0] + u32(row) * BRICK_NODES + node) * sc + off;
                out.c[i] = v;
                umin = min(umin, abs(v));
            }
        } else {
            for (var i = 0u; i < 8u; i = i + 1u) { if (!tick()) { break; } out.c[i] = 0.0; }
            umin = 0.0;
        }
        out.found = true;
        out.masked = U.rule == 1u && umin > HALF_DIAG * lv.h + q0;
        out.h = lv.h;
        out.lo = lv.lo + (bi * b + vec3<f32>(c)) * lv.h;
        return out;
    }
    return out;
}

fn cub(c: vec4<f32>, u: f32) -> f32 { return ((c.w * u + c.z) * u + c.y) * u + c.x; }

fn mulc(a: vec4<f32>, b: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(a.x * b.x, a.y * b.x + a.x * b.y, a.z * b.x + a.y * b.y, a.w * b.x + a.z * b.y);
}

// the first zero with a sign change of c in (0, ue]: x 1 found (u in y), 2 ambiguous, 0 none
fn first_root(c: vec4<f32>, ue: f32) -> vec2<f32> {
    let a = 3.0 * c.w; let b = 2.0 * c.z; let cc = c.y;
    var cuts = array<f32, 4>(0.0, ue, ue, ue);
    var n = 1u;
    if (abs(a) > 0.0) {
        let disc = b * b - 4.0 * a * cc;
        if (disc >= 0.0) {
            let r = sqrt(disc);
            let x0 = min((-b - r) / (2.0 * a), (-b + r) / (2.0 * a));
            let x1 = max((-b - r) / (2.0 * a), (-b + r) / (2.0 * a));
            if (x0 > 0.0 && x0 < ue) { cuts[n] = x0; n = n + 1u; }
            if (x1 > 0.0 && x1 < ue) { cuts[n] = x1; n = n + 1u; }
        }
    } else if (abs(b) > 0.0) {
        let x = -cc / b;
        if (x > 0.0 && x < ue) { cuts[n] = x; n = n + 1u; }
    }
    cuts[n] = ue;
    let scale = abs(c.x) + abs(c.y) + abs(c.z) + abs(c.w);
    for (var j = 0u; j < 3u; j = j + 1u) { if (!tick()) { break; }
        if (j >= n) { break; }
        var l = cuts[j]; var r = cuts[j + 1u];
        let fl = cub(c, l); let fr = cub(c, r);
        if ((fl < 0.0) == (fr < 0.0)) { continue; }
        for (var it = 0u; it < 24u; it = it + 1u) { if (!tick()) { break; }
            let m = 0.5 * (l + r);
            if ((cub(c, m) < 0.0) == (fl < 0.0)) { l = m; } else { r = m; }
        }
        let u = 0.5 * (l + r);
        let sm = cub(c, u - 1.52587890625e-5); let sp = cub(c, u + 1.52587890625e-5);
        if (abs(sm) <= 9.5367431640625e-7 * scale || abs(sp) <= 9.5367431640625e-7 * scale) { return vec2<f32>(2.0, u); }
        if ((sm < 0.0) != (sp < 0.0)) { return vec2<f32>(1.0, u); }
    }
    return vec2<f32>(0.0, 0.0);
}

// the cell walk of sheet k over [ta, tb]: x 1 root at t = y, 0 clear, 3 exhausted; z counts ambiguous roots
fn walk_first_root(k: u32, o: vec3<f32>, d: vec3<f32>, ta: f32, tb: f32) -> vec3<f32> {
    var t = ta;
    var amb = 0.0;
    // the previous cell's exit (t, s, valid), for a sign change across a face (the CPU rule: both |s| <= JUMP_S)
    var prev = vec3<f32>(0.0, 0.0, 0.0);
    var past = false;
    // the hair stepped past the last cell's exit (the CPU walk's last_hair)
    var last_hair = 0.0;
    for (var i = 0u; i < __CELL_WALK_CAP__; i = i + 1u) { if (!tick()) { break; }
        if (t >= tb) {
            // the span ended on a cell face: enter that one next cell for its face rule only (the CPU walk)
            if (past || !(prev.z == 1.0 && prev.x <= tb + last_hair)) { return vec3<f32>(0.0, 0.0, amb); }
            past = true;
        }
        let p = o + d * t;
        let cl = s_cell(k, p);
        if (!cl.found) { t = t + outside_bound(p, SHEETS[k]); prev.z = 0.0; continue; }
        let q0 = cl.q;
        var ux = 1e30;
        // a direction component below PARALLEL (1e-9) is the rounding of a zero: it cannot leave the cell through that
        // face within the walk (the CPU cellroot::PARALLEL)
        // the exit axis's point coordinate over its direction (|p_a| / |d_a|), for the hair past the face (CPU walk)
        var pd = 0.0;
        for (var a = 0; a < 3; a = a + 1) { if (!tick()) { break; }
            var ua = 1e30;
            if (d[a] > 1e-9) { ua = (1.0 - q0[a]) / d[a]; } else if (d[a] < -1e-9) { ua = -q0[a] / d[a]; }
            if (ua < ux) { ux = ua; pd = abs(p[a]) / abs(d[a]); }
        }
        ux = max(ux, 0.0);
        let ue = max(min(ux, (tb - t) / cl.h), 0.0);
        // an s cell: the exact s from the cell's triangles (the CPU walk, line for line)
        let ex = exact_s_here(k, p);
        if (ex.z == 1.0) {
            if (prev.z == 1.0 && ((prev.y < 0.0) != (ex.x < 0.0)) && abs(prev.y) <= JUMP_S && abs(ex.x) <= JUMP_S) {
                var fa = prev.x; var fb = t;
                let gfa = sample_sheet(k, o + d * fa, 2u).g;
                for (var it = 0; it < 30; it = it + 1) { if (!tick()) { break; }
                    let m = 0.5 * (fa + fb);
                    if ((sample_sheet(k, o + d * m, 2u).g < 0.0) == (gfa < 0.0)) { fa = m; } else { fb = m; }
                }
                let tm = 0.5 * (fa + fb);
                if (crossing_inside(k, o + d * tm)) { return vec3<f32>(1.0, tm, amb); }
            }
            let tn = t + ux * cl.h;
            prev.z = 0.0;
            if (!cl.masked && !past) {
                let r = exact_cell_root(k, o, d, t, t + ue * cl.h, cl.h);
                if (r.x == 1.0) { return vec3<f32>(1.0, r.y, amb); }
                if (r.x == 2.0) { amb = amb + 1.0; } else { prev = vec3<f32>(tn, r.z, 1.0); }
            }
            last_hair = max(max(9.5367431640625e-7 * cl.h, 4.0 * 1.1920929e-7 * abs(tn)), 4.0 * 1.1920929e-7 * pd);
            t = tn + last_hair;
            continue;
        }
        var cu = vec4<f32>(0.0);
        for (var c8 = 0u; c8 < 8u; c8 = c8 + 1u) { if (!tick()) { break; }
            var poly = vec4<f32>(1.0, 0.0, 0.0, 0.0);
            poly = mulc(poly, select(vec2<f32>(1.0 - q0.x, -d.x), vec2<f32>(q0.x, d.x), ((c8 >> 2u) & 1u) == 1u));
            poly = mulc(poly, select(vec2<f32>(1.0 - q0.y, -d.y), vec2<f32>(q0.y, d.y), ((c8 >> 1u) & 1u) == 1u));
            poly = mulc(poly, select(vec2<f32>(1.0 - q0.z, -d.z), vec2<f32>(q0.z, d.z), (c8 & 1u) == 1u));
            cu = cu + cl.c[c8] * poly;
        }
        let s_in = cu.x;
        if (prev.z == 1.0 && ((prev.y < 0.0) != (s_in < 0.0)) && abs(prev.y) <= JUMP_S && abs(s_in) <= JUMP_S) {
            var fa = prev.x; var fb = t;
            let gfa = sample_sheet(k, o + d * fa, 2u).g;
            for (var it = 0; it < 30; it = it + 1) { if (!tick()) { break; }
                let m = 0.5 * (fa + fb);
                if ((sample_sheet(k, o + d * m, 2u).g < 0.0) == (gfa < 0.0)) { fa = m; } else { fb = m; }
            }
            let tm = 0.5 * (fa + fb);
            if (crossing_inside(k, o + d * tm)) { return vec3<f32>(1.0, tm, amb); }
        }
        prev = vec3<f32>(t + ux * cl.h, cub(cu, ux), 1.0);
        if (!cl.masked && !past) {
            let r = first_root(cu, ue);
            if (r.x == 1.0) {
                let tr = t + r.y * cl.h;
                if (crossing_inside(k, o + d * tr)) { return vec3<f32>(1.0, tr, amb); }
            } else if (r.x == 2.0) { amb = amb + 1.0; }
        }
        // past the face by a hair that t can carry: 2^-20 h, but at least 4 ulp of t (t is f32 in mm, up to ~1000)
        let tn = t + ux * cl.h;
        last_hair = max(max(9.5367431640625e-7 * cl.h, 4.0 * 1.1920929e-7 * abs(tn)), 4.0 * 1.1920929e-7 * pd);
        t = tn + last_hair;
    }
    if (t >= tb) { return vec3<f32>(0.0, 0.0, amb); }
    return vec3<f32>(3.0, 0.0, amb);
}

fn in_box(k: u32, p: vec3<f32>) -> bool {
    let sh = SHEETS[k];
    return all(p >= sh.lo) && all(p <= sh.hi);
}

struct Union { f: f32, g: f32, k: u32, masked: bool };

fn nlo(i: u32) -> vec3<f32> { return vec3<f32>(bitcast<f32>(IX[U.bn_base + 8u * i]), bitcast<f32>(IX[U.bn_base + 8u * i + 1u]), bitcast<f32>(IX[U.bn_base + 8u * i + 2u])); }
fn nhi(i: u32) -> vec3<f32> { return vec3<f32>(bitcast<f32>(IX[U.bn_base + 8u * i + 4u]), bitcast<f32>(IX[U.bn_base + 8u * i + 5u]), bitcast<f32>(IX[U.bn_base + 8u * i + 6u])); }

// set when a BVH walk used its visit bound without returning to the root (a malformed tree): the result is refused
var<private> walk_unfinished: bool;

// min over sheets, lowest index on ties; a node farther than the best value so far is skipped (as the CPU walk)
fn eval_union(p: vec3<f32>) -> Union {
    var r: Union;
    r.f = 1e30; r.g = 1e30; r.k = NONE; r.masked = false;
    if (U.bn_len < 8u || U.n_sheets == 0u) { r.k = 0u; return r; }
    var at = 0u;
    var prev = NONE;
    var done = false;
    for (var visit = 0u; visit < U.bn_visits; visit = visit + 1u) { if (!tick()) { break; }
        let a = IX[U.bn_base + 8u * at + 3u];
        let b = IX[U.bn_base + 8u * at + 7u];
        var up = true;
        if ((a & LEAF) != 0u) {
            let first = a & ~LEAF;
            for (var e = 0u; e < BVH_LEAF; e = e + 1u) { if (!tick()) { break; }
                if (e >= b) { break; }
                let j = first + e;
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
            if (at == 0u) { done = true; break; }
            prev = at;
            at = IX[U.bp_base + at];
        }
    }
    if (!done) { walk_unfinished = true; }
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
    var done = false;
    for (var visit = 0u; visit < U.bn_visits; visit = visit + 1u) { if (!tick()) { break; }
        let a = IX[U.bn_base + 8u * at + 3u];
        let b = IX[U.bn_base + 8u * at + 7u];
        var up = true;
        if ((a & LEAF) != 0u) {
            let first = a & ~LEAF;
            for (var e = 0u; e < BVH_LEAF; e = e + 1u) { if (!tick()) { break; }
                if (e >= b) { break; }
                let j = first + e;
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
            if (at == 0u) { done = true; break; }
            prev = at;
            at = IX[U.bp_base + at];
        }
    }
    if (!done) { walk_unfinished = true; return 0u; }
    for (var i = 1u; i < MAX_LIVE; i = i + 1u) { if (!tick()) { break; }
        if (i >= n) { break; }
        let x = (*L)[i];
        var j = i;
        for (var shift = 0u; shift < MAX_LIVE; shift = shift + 1u) { if (!tick()) { break; }
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
    OUT[i] = vec4<f32>(r.f, r.g, select(select(select(0.0, 1.0, r.masked), 2.0, walk_unfinished), 3.0, work > U.work_budget), f32(r.k));
}

@compute @workgroup_size(64)
fn march_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= U.count) { return; }
    let o = INPUT[i].xyz;
    let d = U.dir;
    var prev_k: array<u32, __MAX_LIVE__>;
    var prev_g: array<f32, __MAX_LIVE__>;
    var prev_s: array<f32, __MAX_LIVE__>;
    var prev_l: array<f32, __MAX_LIVE__>;
    var n_prev = 0u;
    var overflow = false;
    var unsorted = false;
    var lv: array<u32, __MAX_LIVE__>;
    let n0 = live_sheets(o, &lv);
    if (n0 > MAX_LIVE) { overflow = true; }
    else {
        for (var j = 0u; j < MAX_LIVE; j = j + 1u) { if (!tick()) { break; }
            if (j >= n0) { break; }
            if (j > 0u && lv[j] <= lv[j - 1u]) { unsorted = true; }
            let c0 = sample_sheet(lv[j], o, 2u);
            prev_k[j] = lv[j]; prev_g[j] = c0.g; prev_s[j] = reach(c0); prev_l[j] = c0.lip;
        }
        n_prev = n0;
    }
    var t = 0.0;
    var tprev = 0.0;
    var kind = 2;
    var own = 0u;
    var walk_out = false;
    var floor_step = false;
    for (var step = 0u; step < U.steps; step = step + 1u) { if (!tick()) { break; }
        if (overflow) { break; }
        let p = o + d * t;
        let dd = eval_union(p).f;
        var thit = 1e30;
        var khit = 0u;
        var cur_k: array<u32, __MAX_LIVE__>;
        var cur_g: array<f32, __MAX_LIVE__>;
        var cur_s: array<f32, __MAX_LIVE__>;
        var cur_l: array<f32, __MAX_LIVE__>;
        var n_cur = 0u;
        let nl = live_sheets(p, &lv);
        if (walk_unfinished) { break; }
        if (nl > MAX_LIVE) { overflow = true; break; }
        // both lists are ascending by sheet (live_sheets sorts): one pointer walks the previous list once while the
        // current list is read, so a sheet's previous value is found in O(n_live + n_prev) per step; a list that is
        // not strictly ascending would pair the wrong sheets, so it is refused (kind 5), never marched
        var jp = 0u;
        for (var li = 0u; li < MAX_LIVE; li = li + 1u) { if (!tick()) { break; }
            if (li >= nl) { break; }
            let k = lv[li];
            if (li > 0u && k <= lv[li - 1u]) { unsorted = true; }
            let cs = sample_sheet(k, p, 2u);
            let gc = cs.g;
            cur_k[n_cur] = k; cur_g[n_cur] = gc; cur_s[n_cur] = reach(cs); cur_l[n_cur] = cs.lip; n_cur = n_cur + 1u;
            if (t <= 0.0) { continue; }
            for (var adv = 0u; adv < MAX_LIVE; adv = adv + 1u) { if (!tick()) { break; } if (!(jp < n_prev && prev_k[jp] < k)) { break; } jp = jp + 1u; }
            if (jp >= n_prev || prev_k[jp] != k) { continue; }
            let gp = prev_g[jp];
            if (floor_step) {
                let wr = walk_first_root(k, o, d, tprev, t);
                if (wr.x == 3.0) { kind = 4; walk_out = true; break; }
                if (wr.x == 1.0 && wr.y < thit) { thit = wr.y; khit = k; }
                // a crossing exists when the step's ends have opposite signs, whatever the walk located (the CPU march)
                if (wr.x == 0.0 && ((gc < 0.0) != (gp < 0.0))) {
                    var fa = tprev; var fb = t; var gfa = gp;
                    for (var it = 0; it < 30; it = it + 1) { if (!tick()) { break; }
                        let m = 0.5 * (fa + fb);
                        let gm = sample_sheet(k, o + d * m, 2u).g;
                        if ((gm < 0.0) == (gfa < 0.0)) { fa = m; gfa = gm; } else { fb = m; }
                    }
                    let tm = 0.5 * (fa + fb);
                    if (tm < thit && crossing_inside(k, o + d * tm)) { thit = tm; khit = k; }
                }
                continue;
            }
            // not a floor step: the step stayed inside the free ball, so ends of one sign hold no zero
            if ((gc < 0.0) == (gp < 0.0)) { continue; }
            var a = tprev; var b = t; var ga = gp;
            for (var it = 0; it < 30; it = it + 1) { if (!tick()) { break; }
                let mid = 0.5 * (a + b);
                let gm = sample_sheet(k, o + d * mid, 2u).g;
                if ((gm < 0.0) == (ga < 0.0)) { a = mid; ga = gm; } else { b = mid; }
            }
            let tm = 0.5 * (a + b);
            if (tm < thit && crossing_inside(k, o + d * tm)) { thit = tm; khit = k; }
        }
        if (overflow || walk_out) { break; }
        if (thit < 1e29) { t = thit; own = khit; kind = 1; break; }
        tprev = t;
        for (var j = 0u; j < MAX_LIVE; j = j + 1u) { if (!tick()) { break; } if (j >= n_cur) { break; } prev_k[j] = cur_k[j]; prev_g[j] = cur_g[j]; prev_s[j] = cur_s[j]; prev_l[j] = cur_l[j]; }
        n_prev = n_cur;
        // the step margin (CPU SheetField::step_margin): with kink cells the step is taken from dd less the cover's position
        // bound, so a step is free of every crossing the field can show whatever the rounding of t
        let dd_step = dd - __STEP_MARGIN__;
        floor_step = dd_step < __MIN_STEP__;
        t = t + max(dd_step, __MIN_STEP__);
        if (t >= U.tmax) { kind = 0; break; }
    }
    // over the budget, every loop left early: nothing else this invocation computed is trusted
    if (work > U.work_budget) { OUT[4u * i] = vec4<f32>(0.0, 4.0, 0.0, 0.0); OUT[4u * i + 1u] = vec4<f32>(0.0, 0.0, 0.0, bitcast<f32>(work)); OUT[4u * i + 2u] = vec4<f32>(0.0); OUT[4u * i + 3u] = vec4<f32>(0.0); return; }
    if (walk_unfinished) { OUT[4u * i] = vec4<f32>(0.0, 6.0, 0.0, 0.0); return; }
    if (overflow) { OUT[4u * i] = vec4<f32>(0.0, 3.0, 0.0, 0.0); return; }
    if (unsorted) { OUT[4u * i] = vec4<f32>(0.0, 5.0, 0.0, 0.0); return; }
    var nrm = vec3<f32>(0.0, 0.0, 1.0);
    var uv = vec3<f32>(0.0);
    var an = vec4<f32>(0.0);
    var miss = 0.0;
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
        // a genuine band miss: that attribute is None for this ray (the CPU's too), and the host counts the ray
        if (s.band_miss != 0u) { miss = 1.0; }
        if (s.has_uv && s.inside && (s.band_miss & 0xCu) == 0u) { uv = vec3<f32>(s.uv, 1.0); }
        if (s.has_an && s.inside && (s.band_miss & 0x70u) == 0u && length(s.an) > 1e-12) { an = vec4<f32>(normalize(s.an), 1.0); }
    }
    OUT[4u * i] = vec4<f32>(t, f32(kind), f32(own), miss);
    OUT[4u * i + 1u] = vec4<f32>(nrm, bitcast<f32>(work));
    OUT[4u * i + 2u] = vec4<f32>(uv, 0.0);
    OUT[4u * i + 3u] = an;
}
";

#[cfg(test)]
mod tests {
    use super::{GpuSheetField, WG};
    use crate::compiled::GpuError;
    use crate::sheet_field::build::{level_from_fn, sheet};
    use crate::sheet_field::MarchResult;
    use crate::sheet_field::{SheetField, ValidRule};
    use glam::Vec3;

    /// A dispatch the device refuses (more workgroups than one dimension may have, sent past the size check)
    /// is reported only to the uncaptured-error handler and the submit is dropped: the read must be an error
    /// (the marker the same submit writes is missing), never values read from an unwritten buffer.
    #[test]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    fn a_dispatch_the_device_refuses_is_an_error_not_unwritten_values() {
        let lv = level_from_fn(
            Vec3::new(-2.0, -2.0, -1.0),
            0.25,
            [2, 2, 1],
            |_c, _e| true,
            |p| vec![p.z, (p.x.abs().max(p.y.abs())) - 1.0],
        )
        .unwrap();
        let f = SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1);
        let g = match GpuSheetField::new(&f) {
            Ok(g) => g,
            Err(e) => {
                assert!(
                    std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                    "ALICE_SDF_REQUIRE_GPU set but no GPU: {e}"
                );
                eprintln!("gpu-test: skipped (no GPU: {e})");
                return;
            }
        };
        let input = vec![[0.0f32, 0.0, 0.5, 0.0]; 64];
        // the same input through the checked path is fine
        let ok = g.run(
            &g.eval_pipeline,
            &input,
            1,
            g.uniforms(64, 0, Vec3::ZERO, 0.0),
        );
        assert!(ok.is_ok(), "{ok:?}");
        let limit = g
            .context()
            .device()
            .limits()
            .max_compute_workgroups_per_dimension;
        let bad = g.run_groups(
            &g.eval_pipeline,
            &input,
            1,
            g.uniforms(64, 0, Vec3::ZERO, 0.0),
            Some(limit + 1),
        );
        assert!(
            bad.is_err(),
            "a refused dispatch was read as values ({} of them)",
            bad.map(|v| v.len()).unwrap_or(0)
        );
        let _ = WG;
    }

    /// 100 overlapping flat squares (a deep live list, so a march does real work) and rays over them.
    fn stack_and_rays(n: usize) -> (SheetField, Vec<Vec3>) {
        let square = |z: f32| {
            let lv = level_from_fn(
                Vec3::new(-8.0 + 0.013, -8.0 + 0.021, z - 1.0 + 0.007),
                0.25,
                [8, 8, 1],
                |_c, _e| true,
                |p| {
                    let (dx, dy) = (p.x.abs() - 5.0, p.y.abs() - 5.0);
                    vec![
                        p.z - z,
                        Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0),
                    ]
                },
            )
            .unwrap();
            sheet(vec![lv]).unwrap()
        };
        let f = SheetField::new(
            (0..100).map(|k| square(0.01 * k as f32)).collect(),
            ValidRule::CellV1,
        );
        let os = (0..n)
            .map(|i| {
                let (a, b) = (
                    (i % 97) as f32 / 97.0,
                    (i / 97) as f32 / (n / 97 + 1) as f32,
                );
                Vec3::new(-6.0 + 12.0 * a, -6.0 + 12.0 * b, 5.0)
            })
            .collect();
        (f, os)
    }

    fn gpu_or_skip(f: &SheetField) -> Option<GpuSheetField> {
        match GpuSheetField::new(f) {
            Ok(g) => Some(g),
            Err(e) => {
                assert!(
                    std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                    "ALICE_SDF_REQUIRE_GPU set but no GPU: {e}"
                );
                eprintln!("gpu-test: skipped (no GPU: {e})");
                None
            }
        }
    }

    /// A march whose first dispatch the device cuts short (made here: one workgroup for a dispatch of thousands of
    /// rays) re-issues the unwritten rays and ends with the results, bit for bit, of a march made only of the
    /// smallest dispatches; the cut and the re-issued rays are counted.
    /// Oracle: the march of each ray does not depend on its neighbours or its dispatch (the smallest-dispatch run).
    #[test]
    #[allow(clippy::unwrap_used)]
    fn a_cut_dispatch_is_reissued_and_the_result_is_the_uncut_one() {
        let (f, os) = stack_and_rays(5_000);
        let (Some(a), Some(b)) = (gpu_or_skip(&f), gpu_or_skip(&f)) else {
            return;
        };
        a.set_dispatch_target(std::time::Duration::from_nanos(1));
        let small = a.march_batch_decided(&os, -Vec3::Z, 10.0).unwrap();
        let sa = a.last_march_stats();
        assert_eq!(
            (sa.cut_dispatches, sa.reissued_rays, sa.smallest_reissue),
            (0, 0, 0),
            "{sa:?}"
        );
        assert!(
            small.iter().any(|m| matches!(m, MarchResult::Hit(_))),
            "no hits: the scene is wrong"
        );
        b.control.lock().unwrap().cut = (1, 1);
        let cut = b.march_batch_decided(&os, -Vec3::Z, 10.0).unwrap();
        let sb = b.last_march_stats();
        let first = sb.dispatches[0].0;
        assert!(
            first > WG as usize,
            "the first dispatch was not larger than one workgroup: {first}"
        );
        assert_eq!(sb.cut_dispatches, 1, "{sb:?}");
        assert_eq!(sb.reissued_rays, (first - WG as usize) as u64, "{sb:?}");
        let reissues = &sb.dispatches[1..];
        assert!(
            reissues.iter().all(|d| d.0 <= first / 2),
            "a re-issue over half the cut size: {sb:?}"
        );
        assert_eq!(
            reissues.iter().map(|d| d.0).sum::<usize>(),
            os.len() - WG as usize,
            "the re-issues are not exactly the unwritten rays: {sb:?}"
        );
        assert_eq!(
            sb.smallest_reissue,
            reissues.iter().map(|d| d.0).min().unwrap(),
            "{sb:?}"
        );
        let differ = (0..os.len()).filter(|&i| small[i] != cut[i]).count();
        assert_eq!(
            differ,
            0,
            "{differ} of {} rays differ after the re-issue",
            os.len()
        );
    }

    /// A device that cuts every dispatch: after MAX_REISSUE_ROUNDS the march is the typed error naming the rays
    /// left, and every cut is counted (never a result with unwritten slots read as misses).
    #[test]
    #[allow(clippy::unwrap_used)]
    fn a_device_that_always_cuts_is_a_typed_error_after_the_rounds() {
        let (f, os) = stack_and_rays(3_000);
        let Some(g) = gpu_or_skip(&f) else { return };
        // zero workgroups: the dispatch writes nothing, every round
        g.control.lock().unwrap().cut = (0, u32::MAX);
        match g.march_batch_decided(&os, -Vec3::Z, 10.0) {
            Err(GpuError::DispatchLeftUnwritten {
                unwritten,
                rays,
                rounds,
            }) => {
                assert_eq!(rounds, super::MAX_REISSUE_ROUNDS);
                assert_eq!(unwritten, rays, "unwritten {unwritten} of {rays}");
            }
            other => panic!("not the typed error: {other:?}"),
        }
        let st = g.last_march_stats();
        assert!(
            st.cut_dispatches > u64::from(super::MAX_REISSUE_ROUNDS),
            "{st:?}"
        );
        // every round re-issues all the rays it was given: 3000 x the rounds
        assert_eq!(
            st.reissued_rays,
            os.len() as u64 * u64::from(super::MAX_REISSUE_ROUNDS + 1),
            "{st:?}"
        );
        assert!(st.smallest_reissue > 0, "{st:?}");
    }

    /// The crossing check pairs the current and previous live lists by a merge, which needs both ascending: a shader
    /// whose list sort is turned around (descending) must make the march an error, never pair the wrong sheets.
    /// Oracle: the same rays on the unedited shader hit (the scene is deep enough that every ray has several sheets).
    #[test]
    #[allow(clippy::unwrap_used)]
    fn a_live_list_out_of_order_is_an_error_not_a_wrong_pairing() {
        let (f, os) = stack_and_rays(2_000);
        let Some(ok) = gpu_or_skip(&f) else { return };
        let good = ok.march_batch_decided(&os, -Vec3::Z, 10.0).unwrap();
        assert!(
            good.iter().any(|m| matches!(m, MarchResult::Hit(_))),
            "no hits: the scene is wrong"
        );
        let sort = "if (j == 0u || (*L)[j - 1u] <= x) { break; }";
        let bad = GpuSheetField::build(&f, &|src: String| {
            assert_eq!(src.matches(sort).count(), 1, "the sort line moved");
            src.replace(sort, "if (j == 0u || (*L)[j - 1u] >= x) { break; }")
        })
        .unwrap();
        match bad.march_batch_decided(&os, -Vec3::Z, 10.0) {
            Err(e) => assert!(
                e.to_string().contains("not ascending"),
                "another error: {e}"
            ),
            Ok(_) => panic!("a march with descending live lists returned results"),
        }
    }

    /// A dispatch that runs but covers only part of the rays (one workgroup for 3 workgroups of input) leaves the
    /// rest unwritten: the read is an error naming them, never a miss decoded from an untouched slot.
    #[test]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    fn results_a_dispatch_did_not_write_are_an_error_not_misses() {
        let lv = level_from_fn(
            Vec3::new(-2.0, -2.0, -1.0),
            0.25,
            [2, 2, 1],
            |_c, _e| true,
            |p| vec![p.z, (p.x.abs().max(p.y.abs())) - 1.0],
        )
        .unwrap();
        let f = SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1);
        let g = match GpuSheetField::new(&f) {
            Ok(g) => g,
            Err(e) => {
                assert!(
                    std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                    "ALICE_SDF_REQUIRE_GPU set but no GPU: {e}"
                );
                eprintln!("gpu-test: skipped (no GPU: {e})");
                return;
            }
        };
        let n = 3 * WG as usize;
        let input = vec![[0.3f32, 0.2, 0.5, 0.0]; n];
        let u = g.uniforms(n as u32, 64, -Vec3::Z, 2.0);
        let full = g.run_groups(&g.march_pipeline, &input, 4, u, None).unwrap();
        assert_eq!(full.len(), 4 * n);
        let part = g.run_groups(&g.march_pipeline, &input, 4, u, Some(1));
        match part {
            Err(e) => assert!(
                e.to_string().contains(&format!(
                    "{} of {n} results unwritten (first {WG})",
                    n - WG as usize
                )),
                "{e}"
            ),
            Ok(v) => panic!(
                "{} results read from a dispatch that wrote {WG}",
                v.len() / 4
            ),
        }
    }
}

#[cfg(test)]
mod shader_validate {
    /// The march program parses and validates on the CPU (naga), before any device runs it.
    #[test]
    fn the_march_program_validates_without_a_device() {
        let src = super::march_shader(super::MAX_LIVE_SHEETS);
        let module = wgpu::naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}"));
    }

    fn parsed(src: &str) -> wgpu::naga::Module {
        crate::gpu_loop_bound::parse_source(&wgpu::ShaderSource::Wgsl(src.into()))
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Every loop of every compiled march (each list size, with and without the kink margin) has its own bound and
    /// ticks the work counter, and the counted bound is under the stated ceiling, which is over the budget.
    #[test]
    fn the_march_counts_every_loop_under_its_stated_ceiling() {
        let c = super::march_ceiling();
        assert!(u64::from(super::MARCH_WORK_BUDGET) <= c.max_trip_product);
        let mut compared = 0;
        let mut ml = super::MAX_LIVE_SHEETS;
        while ml <= super::MAX_LIVE_CAP {
            for margin in [0.0, crate::sheet_field::kink::KINK_STEP_MARGIN] {
                let m = parsed(&super::march_shader_margin(ml, margin));
                let loops = crate::gpu_loop_bound::check_module(&m, &super::march_maxima());
                assert!(loops.unbounded.is_empty(), "{ml}: {:?}", loops.unbounded);
                let r = crate::gpu_loop_bound::check_counted(&m, &super::march_maxima(), c.counter);
                assert!(r.problems.is_empty(), "{ml}: {:?}", r.problems);
                assert_eq!(r.budget, u64::from(super::MARCH_WORK_BUDGET));
                assert!(r.instances <= super::MARCH_LOOP_INSTANCES, "{ml}: {r:?}");
                assert!(r.unit <= super::MARCH_UNIT_LOOPS, "{ml}: {r:?}");
                assert!(r.bound <= c.max_trip_product, "{ml}: {r:?}");
                compared += 1;
            }
            ml *= 2;
        }
        assert_eq!(compared, 10);
    }

    /// A sheet with more levels than the GPU loops are bounded by is refused at upload, before any device.
    #[test]
    #[allow(clippy::unwrap_used)]
    fn a_sheet_over_the_level_maximum_is_refused_before_any_device() {
        use crate::sheet_field::build::{level_from_fn, sheet};
        let lv = || {
            level_from_fn(
                glam::Vec3::new(-2.0, -2.0, -1.0),
                0.25,
                [2, 2, 1],
                |_c, _e| true,
                |p| vec![p.z, (p.x.abs().max(p.y.abs())) - 1.0],
            )
            .unwrap()
        };
        let levels = |n: u32| (0..n).map(|_| lv()).collect::<Vec<_>>();
        let over = crate::sheet_field::SheetField::new(
            vec![sheet(levels(super::MAX_SHEET_LEVELS + 1)).unwrap()],
            crate::sheet_field::ValidRule::CellV1,
        );
        let e = super::GpuSheetField::new(&over)
            .err()
            .map(|e| e.to_string());
        assert!(
            e.as_deref().is_some_and(|e| e.contains("MAX_SHEET_LEVELS")),
            "{e:?}"
        );
    }

    /// The check's teeth on the real kernel: dropping any one loop's tick is refused.
    #[test]
    fn a_march_loop_without_its_tick_is_refused() {
        let src = super::march_shader(super::MAX_LIVE_SHEETS);
        let tick = " if (!tick()) { break; }";
        let sites: Vec<usize> = src.match_indices(tick).map(|(i, _)| i).collect();
        let loops =
            crate::gpu_loop_bound::check_module(&parsed(&src), &super::march_maxima()).loops;
        assert!(
            loops > 0 && sites.len() == loops,
            "{} ticks for {loops} loops",
            sites.len()
        );
        for at in sites {
            let cut = format!("{}{}", &src[..at], &src[at + tick.len()..]);
            let r = crate::gpu_loop_bound::require_bounded_under(
                "march",
                &wgpu::ShaderSource::Wgsl(cut.into()),
                &super::march_maxima(),
                Some(super::march_ceiling()),
            );
            assert!(r.is_err(), "the loop at byte {at} without its tick passed");
        }
    }
}
