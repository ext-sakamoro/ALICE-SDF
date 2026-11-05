//! GPU contexts are created and dropped safely from many threads.
//!
//! Evaluators built on separate threads each made their own `wgpu::Instance` and device, and a
//! driver's instance, device and context calls are not all safe to overlap: with one driver, threads
//! blocked for good inside instance creation, device destruction and context destruction, with no
//! frame in a shader, a submit or a fence wait.
//!
//! `overlapping_create_and_drop_finish` makes the overlap happen on purpose. Its threads meet at a
//! barrier before every create and every drop, and a watchdog aborts the process at its limit, so a
//! deadlock is a failure and not a run that never ends. Whether an unserialized path deadlocks
//! depends on the driver, so [`one_module_owns_the_gpu_lifecycle`] holds the same rule on every
//! machine, from the source: only [`LIFECYCLE_MODULE`] creates an instance or a device.
//!
//! Author: Moroya Sakamoto

#![allow(
    clippy::disallowed_methods,
    reason = "the wall clock only limits how long the threads may stay blocked"
)]

use std::path::Path;

/// The only file of the crate that creates a `wgpu::Instance` or a device
const LIFECYCLE_MODULE: &str = "src/gpu_context.rs";

/// The calls only [`LIFECYCLE_MODULE`] may make, and a context as a struct field, whose place among
/// the fields would order the drop: `Held` is the one struct that holds one
const LIFECYCLE_CALLS: &[&str] = &["Instance::new(", ".request_device(", CONTEXT_FIELD];

const CONTEXT_FIELD: &str = ": GpuContext,";

/// What is scanned for them
const SCANNED: &[&str] = &["src", "tests", "examples", "benches"];

/// Every `(file, call)` of `dir`'s Rust files outside comments
fn lifecycle_sites(dir: &Path, sites: &mut Vec<(String, &'static str)>) {
    for entry in std::fs::read_dir(dir).expect("read a source directory") {
        let path = entry.expect("read a source directory entry").path();
        if path.is_dir() {
            lifecycle_sites(&path, sites);
        } else if path.extension().is_some_and(|e| e == "rs") {
            if path
                .file_name()
                .is_some_and(|f| Path::new(file!()).ends_with(f))
            {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("read a source file");
            for line in text.lines().filter(|l| !l.trim_start().starts_with("//")) {
                for call in LIFECYCLE_CALLS {
                    if line.contains(call) {
                        sites.push((path.display().to_string(), call));
                    }
                }
            }
        }
    }
}

#[test]
fn one_module_owns_the_gpu_lifecycle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sites = Vec::new();
    for dir in SCANNED {
        lifecycle_sites(&root.join(dir), &mut sites);
    }
    let owner = root.join(LIFECYCLE_MODULE).display().to_string();
    let others: Vec<_> = sites.iter().filter(|(file, _)| *file != owner).collect();
    assert!(others.is_empty(), "outside {LIFECYCLE_MODULE}: {others:#?}");
    // the field is forbidden everywhere and appears nowhere, so only the calls can show the read
    for call in LIFECYCLE_CALLS.iter().filter(|c| **c != CONTEXT_FIELD) {
        assert!(
            sites.iter().any(|(_, c)| c == call),
            "{LIFECYCLE_MODULE} makes no `{call}` call: the check read nothing"
        );
    }
}

#[cfg(feature = "gpu")]
mod overlap {
    use alice_sdf::compiled::GpuEvaluator;
    use alice_sdf::types::SdfNode;
    use glam::Vec3;
    use std::sync::mpsc::{channel, RecvTimeoutError};
    use std::sync::{Arc, Barrier};
    use std::time::{Duration, Instant};

    const THREADS: usize = 8;
    const ROUNDS: usize = 10;
    /// Past this the process is aborted: the rounds take seconds on a software rasterizer
    const LIMIT: Duration = Duration::from_secs(180);
    /// `|p - c|` is 3 here, so one f32 ulp of the result is 2.4e-7; the shader's `length` may be a
    /// few ulp off
    const DIST_TOL: f32 = 1e-5;

    /// Ends the process, not the test: a blocked thread would keep it alive, and an orderly exit
    /// could block in the driver's own teardown
    fn fail(why: &str) -> ! {
        eprintln!("GPU lifecycle: {why}");
        std::process::abort();
    }

    #[test]
    fn overlapping_create_and_drop_finish() {
        let sphere = SdfNode::sphere(1.0);
        if let Err(e) = GpuEvaluator::new(&sphere) {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "ALICE_SDF_REQUIRE_GPU is set but the GPU evaluator failed: {e}"
            );
            println!("skipped: no GPU evaluator: {e}");
            return;
        }

        let barrier = Arc::new(Barrier::new(THREADS));
        let (done, finished) = channel::<Result<usize, String>>();
        for thread in 0..THREADS {
            let (barrier, done, sphere) = (barrier.clone(), done.clone(), sphere.clone());
            std::thread::spawn(move || {
                let mut evals = 0;
                for _ in 0..ROUNDS {
                    barrier.wait();
                    // half of the threads take the other constructor, which has its own path
                    let made = if thread % 2 == 0 {
                        GpuEvaluator::new(&sphere)
                    } else {
                        GpuEvaluator::new_dynamic(&sphere)
                    };
                    let gpu = match made {
                        Ok(gpu) => gpu,
                        Err(e) => {
                            let _ = done.send(Err(format!("create failed: {e}")));
                            return;
                        }
                    };
                    match gpu.eval_batch(&[Vec3::new(3.0, 0.0, 0.0)]) {
                        Ok(d) if (d[0] - 2.0).abs() <= DIST_TOL => evals += 1,
                        other => {
                            let _ = done.send(Err(format!("wanted 2 at (3,0,0): {other:?}")));
                            return;
                        }
                    }
                    barrier.wait();
                    drop(gpu);
                }
                let _ = done.send(Ok(evals));
            });
        }
        drop(done);

        let start = Instant::now();
        let mut evals = 0;
        for _ in 0..THREADS {
            let left = LIMIT.saturating_sub(start.elapsed());
            match finished.recv_timeout(left) {
                Ok(Ok(n)) => evals += n,
                Ok(Err(why)) => fail(&why),
                Err(RecvTimeoutError::Timeout) => {
                    fail(&format!("threads still blocked after {LIMIT:?}"))
                }
                Err(RecvTimeoutError::Disconnected) => fail("a thread died without a result"),
            }
        }
        assert_eq!(evals, THREADS * ROUNDS);
    }
}
