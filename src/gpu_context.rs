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

/// The adapter with SPIR-V passthrough when the default choice lacks it: a machine with both DX12
/// and Vulkan on one GPU keeps working where the DX12 path would be refused
/// ([`GpuContext::shader_module`]). The best device type wins, as in the default choice.
#[cfg(not(target_arch = "wasm32"))]
fn prefer_passthrough(instance: &wgpu::Instance, chosen: wgpu::Adapter) -> wgpu::Adapter {
    if chosen.features().contains(SPIRV_PASSTHROUGH) {
        return chosen;
    }
    let rank = |adapter: &wgpu::Adapter| match adapter.get_info().device_type {
        wgpu::DeviceType::DiscreteGpu => 3,
        wgpu::DeviceType::IntegratedGpu => 2,
        wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Other => 1,
        wgpu::DeviceType::Cpu => 0,
    };
    instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .filter(|adapter| adapter.features().contains(SPIRV_PASSTHROUGH))
        .max_by_key(rank)
        .unwrap_or(chosen)
}

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
    backend: wgpu::Backend,
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
    let instance = instance(held);
    let adapter = pollster::block_on(instance.request_adapter(&ADAPTER_OPTIONS))
        .ok_or(GpuError::NoAdapter)?;
    let adapter = prefer_passthrough(instance, adapter);
    let backend = adapter.get_info().backend;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &device_descriptor(label, adapter.features() & SPIRV_PASSTHROUGH),
        None,
    ))
    .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;
    Ok(GpuContext::holding(device, queue, backend))
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

/// Whether a driver can fuse `a * b + c` on a path: what a result that needs the same bits
/// everywhere checks before it trusts the path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Contraction {
    /// The driver is forbidden to: the shader is marked SPIR-V, or MSL with fast math off.
    Forbidden,
    /// The path runs and the driver may fuse, so its bits can depend on the driver and on what else
    /// the shader computes.
    NotControlled,
    /// The path refuses to compile a shader on this backend.
    Refused,
}

/// A way a shader reaches the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPath {
    /// `GpuEvaluator` from WGSL, a tree or a shader: on macOS it runs as MSL with fast math off,
    /// not through wgpu.
    Evaluator,
    /// A shader through wgpu: the volume bake, marching cubes and `GpuEvaluator` from GLSL compute.
    WgpuShader,
}

/// The answer for `path` on `backend`.
#[must_use]
pub const fn contraction(path: GpuPath, backend: wgpu::Backend) -> Contraction {
    match (backend, path) {
        (wgpu::Backend::Vulkan, _) => Contraction::Forbidden,
        (wgpu::Backend::Metal, GpuPath::Evaluator) if cfg!(target_os = "macos") => {
            Contraction::Forbidden
        }
        // no MSL input and no fast-math setting through wgpu; in a browser the browser owns the
        // compiler
        (wgpu::Backend::Metal | wgpu::Backend::BrowserWebGpu, _) => Contraction::NotControlled,
        _ => Contraction::Refused,
    }
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
    // where there is no Vulkan device (macOS, a browser) the call records nothing, and the
    // signature is still the one the Vulkan build needs
    #[cfg_attr(
        any(target_arch = "wasm32", target_os = "macos"),
        allow(
            clippy::unused_self,
            clippy::needless_pass_by_ref_mut,
            clippy::missing_const_for_fn
        )
    )]
    pub(crate) fn before_host_read(&self, encoder: &mut wgpu::CommandEncoder) {
        #[cfg(not(any(target_arch = "wasm32", target_os = "macos")))]
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
        #[cfg(any(target_arch = "wasm32", target_os = "macos"))]
        let _ = encoder;
    }

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
        let backend = adapter.get_info().backend;
        let (device, queue) = adapter
            .request_device(&device_descriptor(label, wgpu::Features::empty()), None)
            .await
            .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;
        Ok(Self::holding(device, queue, backend))
    }

    /// Compile a shader. On a device with SPIR-V passthrough the builtins with a sum inside are
    /// written out and every floating-point arithmetic result is marked `NoContraction` (see
    /// `compiled::wgsl::spirv_no_contraction`), so the driver cannot fuse `a * b + c`.
    ///
    /// DX12 and GL have no such path and are refused rather than run with contraction. Metal through
    /// wgpu and a browser run as they are, and say so: see [`Contraction`].
    ///
    /// # Errors
    /// [`GpuError::ShaderCompilation`] when the source does not parse or validate, or holds a
    /// builtin with no expansion; [`GpuError::BackendWithoutContractionControl`] on DX12 or GL.
    pub(crate) fn shader_module(
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
        // a Vulkan device always has the feature through `create`; without it the table's answer
        // would not hold, so it is refused too
        if self.0 .0.backend == wgpu::Backend::Vulkan
            || contraction(GpuPath::WgpuShader, self.0 .0.backend) == Contraction::Refused
        {
            return Err(GpuError::BackendWithoutContractionControl(
                self.0 .0.backend.to_string(),
            ));
        }
        Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source,
        }))
    }

    /// The device's backend
    #[must_use]
    pub fn backend(&self) -> wgpu::Backend {
        self.0 .0.backend
    }

    /// Whether the driver of this device can fuse `a * b + c` on `path` ([`contraction`]).
    #[must_use]
    pub fn contraction(&self, path: GpuPath) -> Contraction {
        contraction(path, self.backend())
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

    fn holding(device: wgpu::Device, queue: wgpu::Queue, backend: wgpu::Backend) -> Self {
        Self(Arc::new(Serialized(ManuallyDrop::new(GpuDevice {
            device,
            queue,
            backend,
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
