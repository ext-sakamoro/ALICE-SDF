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
    clippy::disallowed_methods,
    reason = "the one module that may make an instance or a device"
)]

use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, PoisonError};

use crate::compiled::GpuError;

/// The instance, made on first use and kept for the life of the process. Holding the lock is the
/// only way to reach it, and creating or destroying a device needs the lock as well.
static LIFECYCLE: Mutex<Option<wgpu::Instance>> = Mutex::new(None);

/// Run `f` holding the lock. The guard stays in here, so no caller can let it go early. A panic in a
/// driver call leaves nothing here half written, so a poisoned lock is still good.
fn locked<R>(f: impl FnOnce(&mut Option<wgpu::Instance>) -> R) -> R {
    f(&mut LIFECYCLE.lock().unwrap_or_else(PoisonError::into_inner))
}

const ADAPTER_OPTIONS: wgpu::RequestAdapterOptions<'static, 'static> =
    wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    };

fn new_instance() -> wgpu::Instance {
    wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
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

/// A device and its queue
struct GpuDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// Destroys what it holds inside the lock
struct Serialized(ManuallyDrop<GpuDevice>);

impl Drop for Serialized {
    fn drop(&mut self) {
        // SAFETY: this is the only place the value is dropped, and `self` is not used after it.
        locked(|_| unsafe { ManuallyDrop::drop(&mut self.0) });
    }
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
        locked(|slot| {
            let instance = slot.get_or_insert_with(new_instance);
            let adapter = pollster::block_on(instance.request_adapter(&ADAPTER_OPTIONS))
                .ok_or(GpuError::NoAdapter)?;
            let (device, queue) =
                pollster::block_on(adapter.request_device(&device_descriptor(label), None))
                    .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;
            Ok(Self::holding(device, queue))
        })
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
    pub async fn request_async(label: &str) -> Result<Self, GpuError> {
        let instance = locked(|slot| slot.get_or_insert_with(new_instance).clone());
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
        &self.0 .0.device
    }

    /// The device's queue
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.0 .0.queue
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
        Self(Arc::new(Serialized(ManuallyDrop::new(GpuDevice {
            device,
            queue,
        }))))
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
    use super::{locked, GpuContext, Held};
    use std::sync::mpsc::{channel, RecvTimeoutError};
    use std::sync::Mutex;
    use std::time::Duration;

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
