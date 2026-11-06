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
use std::sync::{Arc, Mutex, PoisonError};

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

/// The feature [`GpuContext::shader_module`] reads a marked SPIR-V module through, requested where
/// the adapter has it (the Vulkan backend).
#[cfg(not(target_arch = "wasm32"))]
const SPIRV_PASSTHROUGH: wgpu::Features = wgpu::Features::SPIRV_SHADER_PASSTHROUGH;

fn device_descriptor(label: &str, features: wgpu::Features) -> wgpu::DeviceDescriptor<'_> {
    wgpu::DeviceDescriptor {
        label: Some(label),
        required_features: features,
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
        locked(|held| destroy(held, &mut self.0));
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
    let (device, queue) = pollster::block_on(adapter.request_device(
        &device_descriptor(label, adapter.features() & SPIRV_PASSTHROUGH),
        None,
    ))
    .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;
    Ok(GpuContext::holding(device, queue))
}

/// The SPIR-V of a shader source, its floating-point arithmetic marked `NoContraction`. The writer
/// options are wgpu's own for the Vulkan backend, except that the bounds checks stay on where wgpu
/// would drop them for a robust device: a passthrough module cannot ask which it is on.
#[cfg(not(target_arch = "wasm32"))]
fn no_contraction_spirv(source: wgpu::ShaderSource<'_>) -> Result<Vec<u32>, GpuError> {
    use naga::back::spv;
    use naga::proc::{BoundsCheckPolicies, BoundsCheckPolicy};
    let failed = |e: String| GpuError::ShaderCompilation(e);
    let module = match source {
        wgpu::ShaderSource::Wgsl(text) => {
            naga::front::wgsl::parse_str(&text).map_err(|e| failed(e.emit_to_string(&text)))?
        }
        wgpu::ShaderSource::Glsl {
            shader,
            stage,
            defines,
        } => naga::front::glsl::Frontend::default()
            .parse(&naga::front::glsl::Options { stage, defines }, &shader)
            .map_err(|e| failed(e.emit_to_string(&shader)))?,
        _ => return Err(failed("unsupported shader source".to_owned())),
    };
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .map_err(|e| failed(format!("{e:?}")))?;
    let options = spv::Options {
        lang_version: (1, 0),
        flags: spv::WriterFlags::empty(),
        bounds_check_policies: BoundsCheckPolicies {
            index: BoundsCheckPolicy::Restrict,
            buffer: BoundsCheckPolicy::Restrict,
            image_load: BoundsCheckPolicy::Restrict,
            binding_array: BoundsCheckPolicy::Unchecked,
        },
        zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::Polyfill,
        ..spv::Options::default()
    };
    let words = spv::write_vec(&module, &info, &options, None)
        .map_err(|e| failed(format!("SPIR-V writer: {e}")))?;
    crate::compiled::wgsl::spirv_no_contraction::mark_no_contraction(&words)
        .map_err(|e| failed(format!("SPIR-V pass: {e:?}")))
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
            .request_device(&device_descriptor(label, wgpu::Features::empty()), None)
            .await
            .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;
        Ok(Self::holding(device, queue))
    }

    /// Compile a shader. On a device with SPIR-V passthrough every floating-point arithmetic
    /// result is marked `NoContraction` (see [`crate::compiled::wgsl::spirv_no_contraction`]), so
    /// the driver cannot fuse `a * b + c`.
    ///
    /// # Errors
    /// [`GpuError::ShaderCompilation`] when the source does not parse or validate.
    pub fn shader_module(
        &self,
        label: &str,
        source: wgpu::ShaderSource<'_>,
    ) -> Result<wgpu::ShaderModule, GpuError> {
        let device = self.device();
        #[cfg(not(target_arch = "wasm32"))]
        if device.features().contains(SPIRV_PASSTHROUGH) {
            let words = no_contraction_spirv(source)?;
            // SAFETY: naga wrote the module from IR its validator accepted, and the pass only adds
            // decorations to it. Every pipeline built on it names an explicit layout.
            return Ok(unsafe {
                device.create_shader_module_spirv(&wgpu::ShaderModuleDescriptorSpirV {
                    label: Some(label),
                    source: words.into(),
                })
            });
        }
        Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source,
        }))
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
