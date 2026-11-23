//! The process's one `wgpu::Instance`, and the lock every device is created and destroyed under.
//!
//! A driver's instance, device and context calls are not all safe to overlap: with one driver, on
//! Vulkan and EGL, threads that each built and dropped their own instance and device
//! blocked for good inside instance creation, context creation, device destruction and context
//! destruction. Recording, submitting and waiting did not take part, so they run outside the lock.
//!
//! [`GpuContext::request`] is the only way to a device in this crate and its tests
//! (`tests/test_gpu_lifecycle_oracle.rs` holds that from the source).
//!
//! A device ends when its last holder goes, and every wgpu resource holds it. So the resources
//! have to end before their context, which then destroys the device inside the lock. To keep them
//! together, build a [`Held`] with [`GpuContext::hold`]; a local context is declared before the
//! resources made from it, and Rust drops it after them.
//!
//! This covers the devices made here. It cannot reach an application that also creates its own
//! `wgpu::Instance` or device: those calls are outside the lock and can overlap with these. Such an
//! application should take its device from [`GpuContext::request`].
//!
//! Author: Moroya Sakamoto

#![allow(
    clippy::disallowed_types,
    reason = "the one module that may name the instance; the calls are allowed on their functions"
)]

use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, PoisonError};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

use crate::compiled::GpuError;

/// What the lock guards: the instance, made on first use and kept for the life of the process.
struct Lifecycle {
    instance: Option<wgpu::Instance>,
}

/// The only value of [`Lifecycle`], so a `&mut Lifecycle` means the lock is held.
static LIFECYCLE: Mutex<Lifecycle> = Mutex::new(Lifecycle { instance: None });

/// Run `f` holding the lock. The guard stays in here, so no caller can let it go early. A panic in a
/// driver call leaves nothing here half written, so a poisoned lock is still good.
fn locked<R>(f: impl FnOnce(&mut Lifecycle) -> R) -> R {
    f(&mut LIFECYCLE.lock().unwrap_or_else(PoisonError::into_inner))
}

#[cfg(target_arch = "wasm32")]
const ADAPTER_OPTIONS: wgpu::RequestAdapterOptions<'static, 'static> =
    wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    };

/// The adapter a native request asks for: the high performance one, or wgpu's software fallback when this crate's
/// own tests ask for it ([`slot_guard`])
#[cfg(not(target_arch = "wasm32"))]
fn adapter_options() -> wgpu::RequestAdapterOptions<'static, 'static> {
    wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: slot_guard::fallback_requested(),
    }
}

/// Set when a device of this process was lost (not destroyed by this crate): the driver's state is unknown, so no
/// device is created again in this process ([`refusal::DEVICE_ABANDONED`]).
static PROCESS_DEVICE_LOST: AtomicBool = AtomicBool::new(false);

/// The contexts whose wait went over its limit. Their work may still be running on the GPU, and a new device on the
/// same GPU would queue more work behind it (2026-11-16: each test of one binary made its own device and stacked
/// another dispatch that did not end on the stalled GPU). While any of them has work in flight, no device is
/// created ([`refusal::DEVICE_ABANDONED`]); one whose queue has drained leaves the list. (Native only: in a browser
/// the page owns the GPU and no wait is abandoned here.)
#[cfg(not(target_arch = "wasm32"))]
static OUTSTANDING: Mutex<Vec<GpuContext>> = Mutex::new(Vec::new());

/// Drop from `list` the entries that have `drained`, and return how many are left
#[cfg(any(not(target_arch = "wasm32"), test))]
fn still_running<T>(list: &Mutex<Vec<T>>, drained: impl Fn(&T) -> bool) -> usize {
    let mut list = list.lock().unwrap_or_else(PoisonError::into_inner);
    list.retain(|entry| !drained(entry));
    list.len()
}

/// The abandoned contexts of this process that still have work in flight
#[cfg(not(target_arch = "wasm32"))]
fn outstanding_work() -> usize {
    still_running(&OUTSTANDING, GpuContext::queue_drained)
}

/// In a browser no wait is abandoned (the page owns the GPU)
#[cfg(target_arch = "wasm32")]
const fn outstanding_work() -> usize {
    0
}

/// The stable start of the messages of the guard's errors.
///
/// They are carried in existing [`GpuError`] variants (a new variant would be a breaking change of the enum), so a
/// caller tells them apart by these prefixes; a test pins each one.
pub mod refusal {
    /// [`GpuError::DeviceCreation`](crate::compiled::GpuError::DeviceCreation): a wait of this process went over
    /// its limit, or a device was lost
    pub const DEVICE_ABANDONED: &str = "gpu device abandoned: ";
    /// [`GpuError::ShaderCompilation`](crate::compiled::GpuError::ShaderCompilation): a shader module has a loop
    /// without a stated bound
    pub const UNBOUNDED_SHADER_LOOP: &str = "unbounded shader loop: ";
    /// [`GpuError::DeviceCreation`](crate::compiled::GpuError::DeviceCreation): the GPU slot guard refused the
    /// adapter (this crate's own tests only)
    pub const ADAPTER_REFUSED: &str = "gpu adapter refused: ";
}

/// The abandoned-device error ([`refusal::DEVICE_ABANDONED`]) when `lost` is set or `running` abandoned
/// dispatches are still in flight
fn refuse_if_abandoned(lost: &AtomicBool, running: usize) -> Result<(), GpuError> {
    if lost.load(Ordering::Acquire) {
        return Err(GpuError::DeviceCreation(format!(
            "{}a device of this process was lost: no new GPU work is started in this process",
            refusal::DEVICE_ABANDONED
        )));
    }
    if running > 0 {
        return Err(GpuError::DeviceCreation(format!(
            "{}{running} earlier GPU dispatch(es) of this process did not finish and are still running: no new GPU work is started until they end",
            refusal::DEVICE_ABANDONED
        )));
    }
    Ok(())
}

/// The device-lost callback: a loss the driver reports (not this crate's own destroy) sets `flag`
fn on_device_lost(
    flag: &'static AtomicBool,
) -> impl Fn(wgpu::DeviceLostReason, String) + Send + 'static {
    move |reason, _message| {
        if reason != wgpu::DeviceLostReason::Destroyed {
            flag.store(true, Ordering::Release);
        }
    }
}

/// The system's Metal device, under the same rules as a wgpu device.
///
/// None while an abandoned dispatch runs or after a device loss in this process, and (`gpu-slot-guard`) only inside
/// a GPU slot, since Metal has no software device. The only place the crate and its tests take a Metal device
/// (`tests/test_gpu_lifecycle_oracle.rs`).
///
/// # Errors
///
/// [`GpuError::DeviceCreation`] ([`refusal::DEVICE_ABANDONED`] / [`refusal::ADAPTER_REFUSED`]), or
/// [`GpuError::NoAdapter`] when there is none.
#[cfg(all(target_os = "macos", feature = "msl"))]
pub fn metal_system_device() -> Result<metal::Device, GpuError> {
    refuse_if_abandoned(&PROCESS_DEVICE_LOST, outstanding_work())?;
    // the slot is decided before the system's Metal device is touched (Metal has no software device)
    slot_guard::admit("Metal system default device", false)?;
    metal::Device::system_default().ok_or(GpuError::NoAdapter)
}

/// Which adapters this crate's own tests may use (`gpu-slot-guard` feature, on only through this crate's
/// dev-dependency on itself: an application that depends on the crate never has it).
///
/// - `ALICE_SDF_TEST_FALLBACK_ADAPTER=1` asks wgpu for its software fallback adapter; the adapter found must be a
///   software one (device type `Cpu`), else the request is refused (a system without a fallback adapter, such as
///   macOS, then has no GPU tests rather than tests on the real GPU read as software runs).
/// - Otherwise a software adapter (lavapipe, SwiftShader: type `Cpu`) is admitted, and a real one only inside a
///   GPU slot of the machine: `ALICE_GPU_SLOT` set and not empty (the machine's GPU queue sets it, or
///   `ci-hosted` on a hosted CI runner, which is its own machine).
///
/// The adapter is printed to stderr on every admitted request, so a test log says which adapter ran.
pub mod slot_guard {
    use crate::compiled::GpuError;

    /// Whether this build has the guard (only this crate's own tests, examples and benches)
    pub const ENABLED: bool = cfg!(feature = "gpu-slot-guard");

    /// The variable that asks for wgpu's software fallback adapter
    #[cfg(feature = "gpu-slot-guard")]
    pub const FALLBACK_VAR: &str = "ALICE_SDF_TEST_FALLBACK_ADAPTER";
    /// The variable a GPU slot of the machine sets
    #[cfg(feature = "gpu-slot-guard")]
    pub const SLOT_VAR: &str = "ALICE_GPU_SLOT";

    /// Whether the fallback adapter is asked for (always `false` without the feature)
    // Not `const`: with the feature the body reads the environment, and the public signature
    // does not change with a test-only feature
    #[must_use]
    #[cfg_attr(not(feature = "gpu-slot-guard"), allow(clippy::missing_const_for_fn))]
    pub fn fallback_requested() -> bool {
        #[cfg(feature = "gpu-slot-guard")]
        {
            std::env::var(FALLBACK_VAR).is_ok_and(|v| v == "1")
        }
        #[cfg(not(feature = "gpu-slot-guard"))]
        {
            false
        }
    }

    /// The decision for one adapter, from the two variables' values
    ///
    /// # Errors
    ///
    /// [`GpuError::DeviceCreation`] ([`super::refusal::ADAPTER_REFUSED`]) as described in the [module docs](self).
    #[cfg(feature = "gpu-slot-guard")]
    pub fn decide(
        name: &str,
        software: bool,
        fallback: Option<&str>,
        slot: Option<&str>,
    ) -> Result<&'static str, GpuError> {
        if fallback == Some("1") {
            return if software {
                Ok("fallback (software)")
            } else {
                Err(GpuError::DeviceCreation(format!(
                    "{}{FALLBACK_VAR}=1 but the adapter found is {name}, not a software adapter: this system offers no fallback adapter (not run on the real GPU)",
                    super::refusal::ADAPTER_REFUSED
                )))
            };
        }
        if software {
            return Ok("software");
        }
        if slot.is_some_and(|v| !v.is_empty()) {
            Ok("real GPU in a slot")
        } else {
            Err(GpuError::DeviceCreation(format!(
                "{}real GPU adapter needs a GPU slot of this machine (ALICE_GPU_SLOT unset)",
                super::refusal::ADAPTER_REFUSED
            )))
        }
    }

    /// [`decide`] on the process's variables; always `Ok` without the feature
    ///
    /// # Errors
    ///
    /// [`GpuError::DeviceCreation`] ([`super::refusal::ADAPTER_REFUSED`]).
    // Not `const` for the same reason as `fallback_requested`
    #[cfg_attr(
        not(feature = "gpu-slot-guard"),
        allow(clippy::unnecessary_wraps, clippy::missing_const_for_fn)
    )]
    pub fn admit(name: &str, software: bool) -> Result<(), GpuError> {
        if !ENABLED {
            let _ = (name, software);
            return Ok(());
        }
        #[cfg(feature = "gpu-slot-guard")]
        {
            let fallback = std::env::var(FALLBACK_VAR).ok();
            let slot = std::env::var(SLOT_VAR).ok();
            // the decision only: the library prints nothing (a test that wants the adapter prints it itself)
            decide(name, software, fallback.as_deref(), slot.as_deref())?;
        }
        Ok(())
    }
}

#[allow(
    clippy::disallowed_methods,
    reason = "the instance is created here, under the lock its caller holds"
)]
fn instance(held: &mut Lifecycle) -> &wgpu::Instance {
    held.instance.get_or_insert_with(|| {
        wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        })
    })
}

/// Which limits a device asks for
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Wanted {
    /// wgpu's defaults (a storage binding of 128 MiB)
    Default,
    /// The defaults, with the storage binding and buffer sizes raised to what the adapter allows
    AdapterStorage,
}

/// The limits `wanted` names on `adapter`
fn limits_for(adapter: &wgpu::Adapter, wanted: Wanted) -> wgpu::Limits {
    match wanted {
        Wanted::Default => wgpu::Limits::default(),
        Wanted::AdapterStorage => {
            let a = adapter.limits();
            wgpu::Limits {
                max_storage_buffer_binding_size: a.max_storage_buffer_binding_size,
                max_buffer_size: a.max_buffer_size,
                ..wgpu::Limits::default()
            }
        }
    }
}

const fn device_descriptor(label: &str, limits: wgpu::Limits) -> wgpu::DeviceDescriptor<'_> {
    wgpu::DeviceDescriptor {
        label: Some(label),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        memory_hints: wgpu::MemoryHints::Performance,
    }
}

/// How long a read waits for its work unless the evaluator says otherwise, in milliseconds. It is
/// far over any dispatch this crate makes (the 64,000-value loft takes 16 ms) and over a slow first
/// compile (214 s measured for one shader on one native driver): reaching it means something is stuck
const WAIT_LIMIT_MS: u64 = 600_000;

/// What a wait ended with
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, PartialEq)]
enum Waited {
    /// The mapping's result
    Mapped(Result<(), wgpu::BufferAsyncError>),
    /// The sender went without a result
    Closed,
    /// The limit passed first
    OverLimit,
}

/// Why nothing was taken from a mapping channel
#[cfg(not(target_arch = "wasm32"))]
enum NotTaken {
    Empty,
    Closed,
}

#[cfg(not(target_arch = "wasm32"))]
impl From<std::sync::mpsc::TryRecvError> for NotTaken {
    fn from(e: std::sync::mpsc::TryRecvError) -> Self {
        match e {
            std::sync::mpsc::TryRecvError::Empty => Self::Empty,
            std::sync::mpsc::TryRecvError::Disconnected => Self::Closed,
        }
    }
}

/// Polls made without a pause before a wait starts to sleep, so a short dispatch costs no sleep
#[cfg(not(target_arch = "wasm32"))]
const POLLS_WITHOUT_PAUSE: u32 = 1000;

/// The pause between later polls. A long wait costs no core, and the limit is counted in these, so
/// no clock is read: it is a lower bound on the time waited
#[cfg(not(target_arch = "wasm32"))]
const POLL_PAUSE: Duration = Duration::from_micros(500);

/// Poll and look for the mapping until it arrives or the pauses reach `limit`. The mapping is looked
/// for before the limit, so a result that is there is never refused, with a limit of zero as well
#[cfg(not(target_arch = "wasm32"))]
fn wait_until(
    mut poll: impl FnMut(),
    mut take: impl FnMut() -> Result<Result<(), wgpu::BufferAsyncError>, NotTaken>,
    mut pause: impl FnMut(Duration),
    limit: Duration,
) -> Waited {
    let mut polls = 0u32;
    let mut paused = Duration::ZERO;
    loop {
        poll();
        match take() {
            Ok(result) => return Waited::Mapped(result),
            Err(NotTaken::Closed) => return Waited::Closed,
            Err(NotTaken::Empty) => {}
        }
        if paused >= limit {
            return Waited::OverLimit;
        }
        if polls < POLLS_WITHOUT_PAUSE {
            polls += 1;
            std::thread::yield_now();
        } else {
            pause(POLL_PAUSE);
            paused += POLL_PAUSE;
        }
    }
}

/// A device and its queue
struct GpuDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// Destroys what it holds inside the lock, unless work on it is still running
struct Serialized {
    device: ManuallyDrop<GpuDevice>,
    /// A read gave up on this device. Its work may still be running, and destroying a queue with
    /// work running panics after a few seconds, so the device is kept for the life of the process
    abandoned: AtomicBool,
    /// How long a read waits, in milliseconds
    wait_limit_ms: AtomicU64,
    /// The adapter the device is on
    adapter: wgpu::AdapterInfo,
}

impl Drop for Serialized {
    fn drop(&mut self) {
        if !self.abandoned.load(Ordering::Acquire) {
            locked(|held| destroy(held, &mut self.device));
        }
    }
}

/// The destruction of a device. It takes `held` as proof of the lock, and is the only drop of a
/// [`GpuDevice`].
fn destroy(_held: &mut Lifecycle, device: &mut ManuallyDrop<GpuDevice>) {
    // SAFETY: the one call is from `Serialized::drop`, which does not use the value after it.
    unsafe { ManuallyDrop::drop(device) };
}

/// The creation of a device. It takes `held` as proof of the lock, and the disallowed call is
/// allowed only here, so moving it out of the lock does not lint.
#[allow(
    clippy::disallowed_methods,
    reason = "the device is created here, under the lock"
)]
fn create(held: &mut Lifecycle, label: &str, wanted: Wanted) -> Result<GpuContext, GpuError> {
    create_unless(
        &PROCESS_DEVICE_LOST,
        outstanding_work,
        held,
        label,
        |lost, held, label| make(lost, held, label, wanted),
    )
}

/// `make` a context unless `lost` is set or `running` reports abandoned work in flight (then nothing reaches the
/// driver)
fn create_unless(
    lost: &'static AtomicBool,
    running: impl FnOnce() -> usize,
    held: &mut Lifecycle,
    label: &str,
    make: impl FnOnce(&'static AtomicBool, &mut Lifecycle, &str) -> Result<GpuContext, GpuError>,
) -> Result<GpuContext, GpuError> {
    refuse_if_abandoned(lost, running())?;
    make(lost, held, label)
}

#[allow(
    clippy::disallowed_methods,
    reason = "the device is created here, under the lock"
)]
fn make(
    abandoned: &'static AtomicBool,
    held: &mut Lifecycle,
    label: &str,
    wanted: Wanted,
) -> Result<GpuContext, GpuError> {
    let adapter = pollster::block_on(instance(held).request_adapter(&adapter_options()))
        .ok_or(GpuError::NoAdapter)?;
    let info = adapter.get_info();
    slot_guard::admit(&info.name, info.device_type == wgpu::DeviceType::Cpu)?;
    let limits = limits_for(&adapter, wanted);
    let (device, queue) =
        pollster::block_on(adapter.request_device(&device_descriptor(label, limits), None))
            .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;
    device.set_device_lost_callback(on_device_lost(abandoned));
    Ok(GpuContext::holding(device, queue, info))
}

/// A shared handle on a device. See the [module docs](self) for how long it must live.
#[derive(Clone)]
pub struct GpuContext(Arc<Serialized>);

impl GpuContext {
    /// Records, at the end of `encoder`, the dependency from everything the GPU wrote to the CPU's
    /// reads of a mapped buffer. Call it on every encoder whose results the CPU maps and reads.
    ///
    /// wgpu 24 records none when a buffer is mapped for reading (a TODO in `wgpu-core`'s
    /// `Buffer::map_async`, still there in 29.0.1; <https://github.com/gfx-rs/wgpu/issues/9306>).
    /// Vulkan requires one; on one integrated-GPU Vulkan driver (540.4.0) its absence let the first
    /// mapped read return another tile's old results at 16,384 points a dispatch and up, while the
    /// memory held the new ones (`tests/test_gpu_readback_volume.rs`). A device on another backend
    /// gives no Vulkan handle and nothing is recorded; other backends synchronise mapping
    /// themselves, and a browser has no such access. Remove this once wgpu records the dependency
    /// (the issue above).
    // wgpu 24 builds its Vulkan backend only for the targets of this predicate (its Cargo.toml
    // gives wgpu-core the `vulkan` feature for exactly these), so `wgpu::hal::api::Vulkan` exists
    // only there. Elsewhere (Apple targets, a browser) the call records nothing, and the
    // signature is still the one the Vulkan build needs
    #[cfg_attr(
        not(any(
            windows,
            all(unix, not(target_os = "emscripten"), not(target_vendor = "apple"))
        )),
        allow(
            clippy::unused_self,
            clippy::needless_pass_by_ref_mut,
            clippy::missing_const_for_fn
        )
    )]
    pub(crate) fn before_host_read(&self, encoder: &mut wgpu::CommandEncoder) {
        #[cfg(any(
            windows,
            all(unix, not(target_os = "emscripten"), not(target_vendor = "apple"))
        ))]
        {
            use ash::vk;
            // SAFETY: the device is this context's own and outlives the call; the handle is only
            // cloned to record into an encoder of the same device, never used to destroy anything.
            let device = unsafe {
                self.device()
                    .as_hal::<wgpu::hal::api::Vulkan, _, _>(|d| d.map(|d| d.raw_device().clone()))
            };
            let Some(device) = device else {
                return;
            };
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            // SAFETY: the command buffer is the one wgpu is recording for `encoder`, still open;
            // a pipeline barrier changes no state wgpu tracks (no binding, no layout, no resource
            // usage), and the handle is not kept or destroyed.
            unsafe {
                encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|e| {
                    if let Some(e) = e {
                        device.cmd_pipeline_barrier(
                            e.raw_handle(),
                            vk::PipelineStageFlags::ALL_COMMANDS,
                            vk::PipelineStageFlags::HOST,
                            vk::DependencyFlags::empty(),
                            &barrier,
                            &[],
                            &[],
                        );
                    }
                });
            }
        }
        #[cfg(not(any(
            windows,
            all(unix, not(target_os = "emscripten"), not(target_vendor = "apple"))
        )))]
        let _ = encoder;
    }

    /// Create a device on the high performance adapter, with the default limits and no extra
    /// features.
    ///
    /// wgpu's native futures complete without yielding, so this blocks only while the lock is
    /// held by another create or drop. On `wasm32` they wait on the browser: see
    /// [`GpuContext::request_async`].
    pub fn request(label: &str) -> Result<Self, GpuError> {
        locked(|held| create(held, label, Wanted::Default))
    }

    /// [`GpuContext::request`], with the storage binding and buffer size limits raised to what the
    /// adapter allows (the defaults cap one storage binding at 128 MiB; an integrated GPU can allow
    /// 4 GiB). Opt-in for data that needs it (a sheet field whose single sheet is larger); every
    /// other limit stays at the default, and [`GpuContext::request`] is unchanged.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn request_with_adapter_storage_limits(label: &str) -> Result<Self, GpuError> {
        locked(|held| create(held, label, Wanted::AdapterStorage))
    }

    /// [`GpuContext::request`] for an async caller. Native futures complete without yielding, so
    /// this is that call with the result ready.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn request_async(label: &str) -> std::future::Ready<Result<Self, GpuError>> {
        std::future::ready(Self::request(label))
    }

    /// [`GpuContext::request`] for `wasm32`, where the wait has to go back to the browser. The
    /// lock is not held across it: a drop on the same thread would wait for a create that cannot
    /// finish. The browser owns the driver there, and this thread is the only caller.
    #[cfg(target_arch = "wasm32")]
    #[allow(
        clippy::disallowed_methods,
        reason = "no thread waits here, and the browser owns the driver"
    )]
    pub async fn request_async(label: &str) -> Result<Self, GpuError> {
        refuse_if_abandoned(&PROCESS_DEVICE_LOST, outstanding_work())?;
        let instance = locked(|held| instance(held).clone());
        let adapter = instance
            .request_adapter(&ADAPTER_OPTIONS)
            .await
            .ok_or(GpuError::NoAdapter)?;
        let (device, queue) = adapter
            .request_device(&device_descriptor(label, wgpu::Limits::default()), None)
            .await
            .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;
        let info = adapter.get_info();
        device.set_device_lost_callback(on_device_lost(&PROCESS_DEVICE_LOST));
        Ok(Self::holding(device, queue, info))
    }

    /// The device
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.0.device.device
    }

    /// The device's queue
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.0.device.queue
    }

    /// Wait for a buffer mapping, polling the device, until it arrives or the wait limit passes.
    ///
    /// `Maintain::Wait` is not used: wgpu's wait stops at 60 s and then treats the submission as
    /// finished, running the mapping callback and releasing what the running work still uses. A
    /// first dispatch can pass 60 s by itself on a driver that compiles the shader there. A poll
    /// only looks at the fence, so a slow dispatch gives its values when it finishes.
    ///
    /// # Errors
    ///
    /// [`GpuError::BufferMapping`] when the limit passes (the device is then kept for good, see
    /// [`Serialized`]) or the mapping fails. Nothing of the output may be read after an error.
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(
        clippy::disallowed_methods,
        reason = "the one place that polls: the wait is bounded here"
    )]
    pub(crate) fn wait_for_mapping(
        &self,
        mapped: &Receiver<Result<(), wgpu::BufferAsyncError>>,
    ) -> Result<(), GpuError> {
        let limit = Duration::from_millis(self.0.wait_limit_ms.load(Ordering::Relaxed));
        let waited = wait_until(
            || {
                self.device().poll(wgpu::Maintain::Poll);
            },
            || mapped.try_recv().map_err(NotTaken::from),
            std::thread::sleep,
            limit,
        );
        self.after_wait(waited, limit)
    }

    /// [`GpuContext::wait_for_mapping`] on `wasm32`, where the browser runs the work and this thread
    /// cannot wait for it: the mapping is there or the read fails
    #[cfg(target_arch = "wasm32")]
    #[allow(
        clippy::disallowed_methods,
        reason = "no thread waits here, and the browser owns the driver"
    )]
    pub(crate) fn wait_for_mapping(
        &self,
        mapped: &Receiver<Result<(), wgpu::BufferAsyncError>>,
    ) -> Result<(), GpuError> {
        self.device().poll(wgpu::Maintain::Poll);
        match mapped.try_recv() {
            Ok(r) => r.map_err(|e| GpuError::BufferMapping(format!("{e:?}"))),
            Err(_) => Err(GpuError::BufferMapping(
                "the GPU work has not finished: its output is not read".to_string(),
            )),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn after_wait(&self, waited: Waited, limit: Duration) -> Result<(), GpuError> {
        match waited {
            Waited::Mapped(Ok(())) => Ok(()),
            Waited::Mapped(Err(e)) => Err(GpuError::BufferMapping(format!("{e:?}"))),
            Waited::Closed => Err(GpuError::BufferMapping(
                "GPU channel closed unexpectedly".to_string(),
            )),
            Waited::OverLimit => {
                self.abandon();
                Err(GpuError::BufferMapping(format!(
                    "the GPU work did not finish within {limit:?} on {}: its output is not read and the device should not be used again",
                    self.adapter_info().name
                )))
            }
        }
    }

    /// Whether a wait on this device went over its limit (the device is kept and not used again)
    pub(crate) fn is_abandoned(&self) -> bool {
        self.0.abandoned.load(Ordering::Acquire)
    }

    /// Keep this device for the life of the process: work on it may still be running
    pub(crate) fn abandon(&self) {
        self.0.abandoned.store(true, Ordering::Release);
        #[cfg(not(target_arch = "wasm32"))]
        OUTSTANDING
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(self.clone());
    }

    /// Whether this device has no submission in flight (a poll looks at the fence and does not wait)
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(
        clippy::disallowed_methods,
        reason = "a non-blocking poll: it only reads whether the abandoned work has ended"
    )]
    fn queue_drained(&self) -> bool {
        self.device().poll(wgpu::Maintain::Poll).is_queue_empty()
    }

    /// Set how long a read on this device waits for its work
    pub(crate) fn set_wait_limit(&self, limit: std::time::Duration) {
        let ms = u64::try_from(limit.as_millis()).unwrap_or(u64::MAX);
        self.0.wait_limit_ms.store(ms, Ordering::Relaxed);
    }

    /// Keep `resources` with this context, so that they end before it does
    #[must_use]
    pub const fn hold<T>(self, resources: T) -> Held<T> {
        Held {
            resources,
            context: self,
        }
    }

    fn holding(device: wgpu::Device, queue: wgpu::Queue, adapter: wgpu::AdapterInfo) -> Self {
        Self(Arc::new(Serialized {
            device: ManuallyDrop::new(GpuDevice { device, queue }),
            abandoned: AtomicBool::new(false),
            wait_limit_ms: AtomicU64::new(WAIT_LIMIT_MS),
            adapter,
        }))
    }

    /// Create a shader module on this device, after checking that every loop of `source` is bounded
    /// ([`crate::gpu_loop_bound`]): a module with an unbounded loop is never compiled. This is the only place in
    /// the crate that creates a shader module (`tests/test_gpu_lifecycle_oracle.rs`).
    ///
    /// # Errors
    ///
    /// [`GpuError::ShaderCompilation`]: [`refusal::UNBOUNDED_SHADER_LOOP`], or the parse error.
    pub fn shader_module(
        &self,
        label: &str,
        source: wgpu::ShaderSource<'_>,
        maxima: &crate::gpu_loop_bound::HostMaxima,
    ) -> Result<wgpu::ShaderModule, GpuError> {
        self.checked_module(label, source, maxima, None)
    }

    /// [`Self::shader_module`] against a stated [`crate::gpu_loop_bound::KernelCeiling`] instead of
    /// [`crate::gpu_loop_bound::MAX_TRIP_PRODUCT`]: for a kernel whose loops are all bounded and whose real work is
    /// bounded by its own counter. Every caller's label is listed in `tests/test_gpu_lifecycle_oracle.rs`; the
    /// method is crate-private so no caller outside this crate can raise a ceiling.
    ///
    /// # Errors
    ///
    /// As [`Self::shader_module`]; also a ceiling without a reason.
    pub(crate) fn shader_module_with_ceiling(
        &self,
        label: &str,
        source: wgpu::ShaderSource<'_>,
        maxima: &crate::gpu_loop_bound::HostMaxima,
        ceiling: crate::gpu_loop_bound::KernelCeiling,
    ) -> Result<wgpu::ShaderModule, GpuError> {
        self.checked_module(label, source, maxima, Some(ceiling))
    }

    #[allow(
        clippy::disallowed_methods,
        reason = "the one place that creates shader modules, after the loop check"
    )]
    fn checked_module(
        &self,
        label: &str,
        source: wgpu::ShaderSource<'_>,
        maxima: &crate::gpu_loop_bound::HostMaxima,
        ceiling: Option<crate::gpu_loop_bound::KernelCeiling>,
    ) -> Result<wgpu::ShaderModule, GpuError> {
        crate::gpu_loop_bound::require_bounded_under(label, &source, maxima, ceiling)?;
        Ok(self
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source,
            }))
    }

    /// The adapter this context's device is on (its name and device type say whether a test ran on a real GPU
    /// or a software one)
    #[must_use]
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.0.adapter
    }
}

/// Resources made from a device, with the context that made it.
///
/// `resources` is declared first, so it ends first: this one declaration is what orders the drop
/// for every holder of a device.
pub struct Held<T, C = GpuContext> {
    resources: T,
    context: C,
}

impl<T, C> Held<T, C> {
    /// The context the resources were made from
    pub const fn context(&self) -> &C {
        &self.context
    }
}

impl<T, C> Deref for Held<T, C> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.resources
    }
}

impl<T, C> DerefMut for Held<T, C> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.resources
    }
}

#[cfg(test)]
mod tests {
    use super::{locked, wait_until, GpuContext, Held, NotTaken, Waited, POLLS_WITHOUT_PAUSE};
    use std::sync::mpsc::{channel, RecvTimeoutError};
    use std::sync::Mutex;
    use std::time::Duration;

    #[test]
    fn a_wait_returns_the_mapping_when_it_arrives_and_polls_until_then() {
        let polls = std::cell::Cell::new(0);
        let waited = wait_until(
            || polls.set(polls.get() + 1),
            || {
                if polls.get() < 5 {
                    Err(NotTaken::Empty)
                } else {
                    Ok(Ok(()))
                }
            },
            |_| {},
            Duration::from_secs(1),
        );
        assert_eq!(waited, Waited::Mapped(Ok(())));
        assert_eq!(polls.get(), 5);
    }

    /// The count of polls and pauses is the limit's: 20 ms at 0.5 ms a pause, with the look that
    /// finds the limit reached
    #[test]
    fn a_wait_ends_at_its_limit_and_not_before() {
        let polls = std::cell::Cell::new(0u32);
        let paused = std::cell::Cell::new(Duration::ZERO);
        let waited = wait_until(
            || polls.set(polls.get() + 1),
            || Err(NotTaken::Empty),
            |d| paused.set(paused.get() + d),
            Duration::from_millis(20),
        );
        assert_eq!(waited, Waited::OverLimit);
        assert_eq!(paused.get(), Duration::from_millis(20));
        assert_eq!(polls.get(), POLLS_WITHOUT_PAUSE + 40 + 1);
    }

    /// A mapping that is there is taken at any limit: the limit bounds the wait, not the result
    #[test]
    fn a_wait_takes_a_mapping_that_is_there_at_a_limit_of_zero() {
        let waited = wait_until(|| {}, || Ok(Ok(())), |_| {}, Duration::ZERO);
        assert_eq!(waited, Waited::Mapped(Ok(())));
        let polls = std::cell::Cell::new(0);
        let none = wait_until(
            || polls.set(polls.get() + 1),
            || Err(NotTaken::Empty),
            |_| {},
            Duration::ZERO,
        );
        assert_eq!(none, Waited::OverLimit);
        assert_eq!(polls.get(), 1, "a limit of zero looks once");
    }

    #[test]
    fn a_wait_reports_a_closed_channel_and_a_failed_mapping() {
        let closed = wait_until(
            || {},
            || Err(NotTaken::Closed),
            |_| {},
            Duration::from_secs(1),
        );
        assert_eq!(closed, Waited::Closed);
        let failed = wait_until(
            || {},
            || Ok(Err(wgpu::BufferAsyncError)),
            |_| {},
            Duration::from_secs(1),
        );
        assert_eq!(failed, Waited::Mapped(Err(wgpu::BufferAsyncError)));
    }

    /// With the lock held, a drop and a request both wait, and both finish once it is free. Finishing
    /// early is the failure, so a slow machine can only miss it, not fail falsely. This shows on any
    /// driver; the overlap that blocks for good shows on one.
    #[test]
    fn test_gpu_create_and_destroy_wait_for_the_lock() {
        let name = "test_gpu_create_and_destroy_wait_for_the_lock";
        if !crate::gpu_test_support::gpu_ready(name) {
            return;
        }
        let context = GpuContext::request(name).unwrap();
        let (done, finished) = channel();
        let threads = locked(|_| {
            let destroyed = done.clone();
            let destroying = std::thread::spawn(move || {
                drop(context);
                let _ = destroyed.send("destroy");
            });
            let created = done.clone();
            let creating = std::thread::spawn(move || {
                // sent with the context alive: its drop needs the lock as well, and would hide a
                // request made outside it
                let context = GpuContext::request(name).unwrap();
                let _ = created.send("create");
                drop(context);
            });
            drop(done);
            let early = finished.recv_timeout(Duration::from_secs(2));
            assert_eq!(
                early,
                Err(RecvTimeoutError::Timeout),
                "ran outside the lock"
            );
            [destroying, creating]
        });
        for _ in 0..2 {
            finished
                .recv_timeout(Duration::from_secs(120))
                .expect("a create or destroy never finished after the lock was freed");
        }
        // a driver still destroying a device while the process exits is another way to crash
        for thread in threads {
            thread.join().unwrap();
        }
    }

    struct Ends<'a>(&'static str, &'a Mutex<Vec<&'static str>>);

    impl Drop for Ends<'_> {
        fn drop(&mut self) {
            self.1.lock().unwrap().push(self.0);
        }
    }

    /// A device outlives whatever holds it, so resources that end after their context would
    /// destroy the device outside the lock
    #[test]
    fn held_resources_end_before_their_context() {
        let order = Mutex::new(Vec::new());
        drop(Held {
            resources: Ends("resources", &order),
            context: Ends("context", &order),
        });
        assert_eq!(*order.lock().unwrap(), ["resources", "context"]);
    }
}

#[cfg(test)]
mod guard_tests {
    #[cfg(feature = "gpu-slot-guard")]
    use super::slot_guard::decide;
    use super::{create_unless, locked, on_device_lost};
    use crate::compiled::GpuError;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    fn flag(set: bool) -> &'static AtomicBool {
        Box::leak(Box::new(AtomicBool::new(set)))
    }

    #[test]
    fn while_abandoned_work_runs_no_context_is_made() {
        // oracle: abandoned work in flight stops the request before the driver: the maker runs 0 times
        let made = AtomicUsize::new(0);
        let r = locked(|held| {
            create_unless(
                flag(false),
                || 1,
                held,
                "poisoned",
                |_, _, _| {
                    made.fetch_add(1, Ordering::SeqCst);
                    Err(GpuError::NoAdapter)
                },
            )
        });
        assert!(
            matches!(&r, Err(GpuError::DeviceCreation(m)) if m.starts_with(super::refusal::DEVICE_ABANDONED)),
            "{:?}",
            r.err()
        );
        assert_eq!(made.load(Ordering::SeqCst), 0);
        // and once it has ended the maker is reached (the guard is not a blanket refusal)
        let r = locked(|held| {
            create_unless(
                flag(false),
                || 0,
                held,
                "fresh",
                |_, _, _| {
                    made.fetch_add(1, Ordering::SeqCst);
                    Err(GpuError::NoAdapter)
                },
            )
        });
        assert!(matches!(r, Err(GpuError::NoAdapter)));
        assert_eq!(made.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_device_the_driver_lost_poisons_and_one_this_crate_destroyed_does_not() {
        let f = flag(false);
        on_device_lost(f)(wgpu::DeviceLostReason::Destroyed, String::new());
        assert!(!f.load(Ordering::SeqCst));
        on_device_lost(f)(wgpu::DeviceLostReason::Unknown, "driver".into());
        assert!(f.load(Ordering::SeqCst));
        let made = AtomicUsize::new(0);
        let r = locked(|held| {
            create_unless(
                f,
                || 0,
                held,
                "after loss",
                |_, _, _| {
                    made.fetch_add(1, Ordering::SeqCst);
                    Err(GpuError::NoAdapter)
                },
            )
        });
        assert!(
            matches!(&r, Err(GpuError::DeviceCreation(m)) if m.starts_with(super::refusal::DEVICE_ABANDONED))
        );
        assert_eq!(made.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn an_abandoned_dispatch_that_never_drains_refuses_every_later_device() {
        // oracle: this morning's case, a kernel that never ends: its context never drains, so every later request
        // for the life of the process is refused before the driver (the maker runs 0 times in all of them)
        let list = std::sync::Mutex::new(vec!["endless kernel"]);
        let made = AtomicUsize::new(0);
        for round in 0..5 {
            let r = locked(|held| {
                create_unless(
                    flag(false),
                    || super::still_running(&list, |_| false),
                    held,
                    "after an endless kernel",
                    |_, _, _| {
                        made.fetch_add(1, Ordering::SeqCst);
                        Err(GpuError::NoAdapter)
                    },
                )
            });
            assert!(
                matches!(&r, Err(GpuError::DeviceCreation(m)) if m.starts_with(super::refusal::DEVICE_ABANDONED)),
                "round {round}: {:?}",
                r.err()
            );
        }
        assert_eq!(made.load(Ordering::SeqCst), 0);
        assert_eq!(list.lock().unwrap().len(), 1, "the endless entry stays");
    }

    #[test]
    fn drained_entries_leave_the_outstanding_list() {
        // oracle: entries whose work has ended are dropped, the rest counted (stand-ins for contexts)
        let list = std::sync::Mutex::new(vec![1, 2, 3, 4]);
        assert_eq!(super::still_running(&list, |e| e % 2 == 0), 2);
        assert_eq!(*list.lock().unwrap(), vec![1, 3]);
        assert_eq!(super::still_running(&list, |_| true), 0);
        assert_eq!(super::still_running(&list, |_| false), 0);
    }

    #[test]
    #[allow(
        clippy::assertions_on_constants,
        reason = "the constant is the feature this build has: the assertion is the check that it is on"
    )]
    fn this_crates_own_test_builds_have_the_guard() {
        assert!(
            super::slot_guard::ENABLED,
            "built without gpu-slot-guard (the dev-dependency on itself is missing)"
        );
    }

    #[cfg(feature = "gpu-slot-guard")]
    #[test]
    fn the_slot_decides_real_adapters_and_the_fallback_must_be_software() {
        // a real adapter: only with a non-empty slot token
        assert!(matches!(
            decide("Apple M2 Pro", false, None, None),
            Err(GpuError::DeviceCreation(m)) if m.starts_with(super::refusal::ADAPTER_REFUSED)
        ));
        assert!(matches!(
            decide("Apple M2 Pro", false, None, Some("")),
            Err(GpuError::DeviceCreation(m)) if m.starts_with(super::refusal::ADAPTER_REFUSED)
        ));
        assert_eq!(
            decide("Apple M2 Pro", false, None, Some("t")).unwrap(),
            "real GPU in a slot"
        );
        // a software adapter needs no slot
        assert_eq!(decide("llvmpipe", true, None, None).unwrap(), "software");
        // the fallback asked for: software is admitted, a real adapter is refused even inside a slot
        assert_eq!(
            decide("llvmpipe", true, Some("1"), None).unwrap(),
            "fallback (software)"
        );
        let e = decide("Apple M2 Pro", false, Some("1"), Some("t")).unwrap_err();
        assert!(e.to_string().contains("not a software adapter"), "{e}");
        // the refusal text is the agreed one, after the stable prefix
        let Err(GpuError::DeviceCreation(m)) = decide("x", false, None, None) else {
            panic!("a real adapter without a slot is refused as DeviceCreation");
        };
        assert_eq!(
            m,
            format!(
                "{}real GPU adapter needs a GPU slot of this machine (ALICE_GPU_SLOT unset)",
                super::refusal::ADAPTER_REFUSED
            )
        );
    }

    #[test]
    fn the_refusal_prefixes_are_stable_and_distinct() {
        // callers and log watchers match these strings: changing one is a breaking change of the messages
        use super::refusal::{ADAPTER_REFUSED, DEVICE_ABANDONED, UNBOUNDED_SHADER_LOOP};
        assert_eq!(DEVICE_ABANDONED, "gpu device abandoned: ");
        assert_eq!(UNBOUNDED_SHADER_LOOP, "unbounded shader loop: ");
        assert_eq!(ADAPTER_REFUSED, "gpu adapter refused: ");
        for (a, b) in [
            (DEVICE_ABANDONED, UNBOUNDED_SHADER_LOOP),
            (DEVICE_ABANDONED, ADAPTER_REFUSED),
            (UNBOUNDED_SHADER_LOOP, ADAPTER_REFUSED),
        ] {
            assert!(!a.starts_with(b) && !b.starts_with(a));
        }
    }
}
