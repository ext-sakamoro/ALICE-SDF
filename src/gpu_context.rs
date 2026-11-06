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
use std::time::{Duration, Instant};

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

const ADAPTER_OPTIONS: wgpu::RequestAdapterOptions<'static, 'static> =
    wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    };

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

fn device_descriptor(label: &str) -> wgpu::DeviceDescriptor<'_> {
    wgpu::DeviceDescriptor {
        label: Some(label),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
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

/// Poll and look for the mapping until it arrives or `elapsed` passes `limit`. The mapping is looked
/// for before the limit, so a result that is there is never refused, with a limit of zero as well.
/// After about a millisecond of yielding it sleeps between polls, so a long wait costs no core
#[cfg(not(target_arch = "wasm32"))]
fn wait_until(
    mut poll: impl FnMut(),
    mut take: impl FnMut() -> Result<Result<(), wgpu::BufferAsyncError>, NotTaken>,
    elapsed: impl Fn() -> Duration,
    limit: Duration,
) -> Waited {
    loop {
        poll();
        match take() {
            Ok(result) => return Waited::Mapped(result),
            Err(NotTaken::Closed) => return Waited::Closed,
            Err(NotTaken::Empty) => {}
        }
        let waited = elapsed();
        if waited >= limit {
            return Waited::OverLimit;
        }
        if waited < Duration::from_millis(1) {
            std::thread::yield_now();
        } else {
            std::thread::sleep(Duration::from_micros(200));
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
fn create(held: &mut Lifecycle, label: &str) -> Result<GpuContext, GpuError> {
    let adapter = pollster::block_on(instance(held).request_adapter(&ADAPTER_OPTIONS))
        .ok_or(GpuError::NoAdapter)?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&device_descriptor(label), None))
            .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;
    Ok(GpuContext::holding(device, queue))
}

/// A shared handle on a device. See the [module docs](self) for how long it must live.
#[derive(Clone)]
pub struct GpuContext(Arc<Serialized>);

impl GpuContext {
    /// Create a device on the high performance adapter, with the default limits and no extra
    /// features.
    ///
    /// wgpu's native futures complete without yielding, so this blocks only while the lock is
    /// held by another create or drop. On `wasm32` they wait on the browser: see
    /// [`GpuContext::request_async`].
    pub fn request(label: &str) -> Result<Self, GpuError> {
        locked(|held| create(held, label))
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
        let instance = locked(|held| instance(held).clone());
        let adapter = instance
            .request_adapter(&ADAPTER_OPTIONS)
            .await
            .ok_or(GpuError::NoAdapter)?;
        let (device, queue) = adapter
            .request_device(&device_descriptor(label), None)
            .await
            .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;
        Ok(Self::holding(device, queue))
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
        reason = "the one place that polls: the wait is bounded here, by the clock of a blocking call"
    )]
    pub(crate) fn wait_for_mapping(
        &self,
        mapped: &Receiver<Result<(), wgpu::BufferAsyncError>>,
    ) -> Result<(), GpuError> {
        let limit = Duration::from_millis(self.0.wait_limit_ms.load(Ordering::Relaxed));
        let started = Instant::now();
        let waited = wait_until(
            || {
                self.device().poll(wgpu::Maintain::Poll);
            },
            || mapped.try_recv().map_err(|e| e.into()),
            || started.elapsed(),
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
                    "the GPU work did not finish within {limit:?}: its output is not read and the device should not be used again"
                )))
            }
        }
    }

    /// Keep this device for the life of the process: work on it may still be running
    pub(crate) fn abandon(&self) {
        self.0.abandoned.store(true, Ordering::Release);
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

    fn holding(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self(Arc::new(Serialized {
            device: ManuallyDrop::new(GpuDevice { device, queue }),
            abandoned: AtomicBool::new(false),
            wait_limit_ms: AtomicU64::new(WAIT_LIMIT_MS),
        }))
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
    use super::{locked, wait_until, GpuContext, Held, NotTaken, Waited};
    use std::sync::mpsc::{channel, RecvTimeoutError};
    use std::sync::Mutex;
    use std::time::Duration;

    /// A clock that moves by one millisecond for each look at it
    fn ticking() -> impl Fn() -> Duration {
        let now = std::cell::Cell::new(0u64);
        move || {
            now.set(now.get() + 1);
            Duration::from_millis(now.get())
        }
    }

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
            ticking(),
            Duration::from_secs(1),
        );
        assert_eq!(waited, Waited::Mapped(Ok(())));
        assert_eq!(polls.get(), 5);
    }

    #[test]
    fn a_wait_ends_at_its_limit_and_not_before() {
        let polls = std::cell::Cell::new(0);
        let waited = wait_until(
            || polls.set(polls.get() + 1),
            || Err(NotTaken::Empty),
            ticking(),
            Duration::from_millis(40),
        );
        assert_eq!(waited, Waited::OverLimit);
        assert_eq!(
            polls.get(),
            40,
            "one poll for each millisecond of the clock"
        );
    }

    /// A mapping that is there is taken at any limit: the limit bounds the wait, not the result
    #[test]
    fn a_wait_takes_a_mapping_that_is_there_at_a_limit_of_zero() {
        let waited = wait_until(|| {}, || Ok(Ok(())), ticking(), Duration::ZERO);
        assert_eq!(waited, Waited::Mapped(Ok(())));
    }

    #[test]
    fn a_wait_reports_a_closed_channel_and_a_failed_mapping() {
        let closed = wait_until(
            || {},
            || Err(NotTaken::Closed),
            ticking(),
            Duration::from_secs(1),
        );
        assert_eq!(closed, Waited::Closed);
        let failed = wait_until(
            || {},
            || Ok(Err(wgpu::BufferAsyncError)),
            ticking(),
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
