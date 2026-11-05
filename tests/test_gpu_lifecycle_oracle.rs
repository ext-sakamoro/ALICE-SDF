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

/// The calls only [`LIFECYCLE_MODULE`] makes
const LIFECYCLE_CALLS: &[&str] = &["Instance::new(", ".request_device("];

/// A kept clone of these holds the device past its context
const KEPT_HANDLES: &[&str] = &[".device().clone()", ".queue().clone()"];

/// What is scanned
const SCANNED: &[&str] = &["src", "tests", "examples", "benches"];

/// Each struct declaration of `text`, comments left out
fn struct_declarations(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut open: Option<String> = None;
    for line in text.lines().map(str::trim).filter(|l| !l.starts_with("//")) {
        let starts = ["struct ", "pub struct ", "pub(crate) struct "]
            .iter()
            .any(|s| line.starts_with(s));
        if open.is_none() && starts {
            open = Some(String::new());
        }
        if let Some(body) = open.as_mut() {
            body.push_str(line);
            body.push('\n');
            if line == "}" || line.ends_with(';') || line.ends_with("{}") {
                found.push(open.take().expect("open"));
            }
        }
    }
    found
}

/// What a source check finds in one file. Only [`LIFECYCLE_MODULE`] may have these.
#[derive(Default)]
struct Scan {
    files: usize,
    structs: usize,
    calls: Vec<(String, &'static str)>,
    others: Vec<(String, String)>,
}

fn scan(dir: &Path, owner: &Path, found: &mut Scan) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let path = entry.expect("read a source directory entry").path();
        if path.is_dir() {
            scan(&path, owner, found);
            continue;
        }
        let this_file = path
            .file_name()
            .is_some_and(|f| Path::new(file!()).ends_with(f));
        if path.extension().is_none_or(|e| e != "rs") || this_file {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("read a source file");
        let file = path.display().to_string();
        found.files += 1;
        for line in text.lines().filter(|l| !l.trim_start().starts_with("//")) {
            for call in LIFECYCLE_CALLS {
                if line.contains(call) {
                    found.calls.push((file.clone(), call));
                }
            }
            for handle in KEPT_HANDLES {
                if line.contains(handle) && path != owner {
                    found.others.push((file.clone(), (*handle).to_string()));
                }
            }
        }
        let structs = struct_declarations(&text);
        found.structs += structs.len();
        if path != owner {
            for body in structs {
                let beside = body.contains("Held<") && body.contains("wgpu::");
                if body.contains("GpuContext") || beside {
                    found.others.push((file.clone(), body));
                }
            }
        }
    }
}

/// The text half of the rule. Clippy's disallowed lists hold the instance and device calls by their
/// resolved names, aliases included. Neither sees a context returned in a tuple with a resource, a
/// resource bound late to a local declared before its context, or a handle kept through a variable:
/// those are for review.
#[test]
fn one_module_owns_the_gpu_lifecycle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let owner = root.join(LIFECYCLE_MODULE);
    let mut found = Scan::default();
    for dir in SCANNED {
        scan(&root.join(dir), &owner, &mut found);
    }
    assert!(
        found.files > 0 && found.structs > 0,
        "the check read nothing"
    );
    let owner = owner.display().to_string();
    let calls: Vec<_> = found
        .calls
        .iter()
        .filter(|(file, _)| *file != owner)
        .collect();
    assert!(calls.is_empty(), "outside {LIFECYCLE_MODULE}: {calls:#?}");
    assert!(found.others.is_empty(), "drop order: {:#?}", found.others);
    for call in LIFECYCLE_CALLS {
        assert!(
            found.calls.iter().any(|(_, c)| c == call),
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
