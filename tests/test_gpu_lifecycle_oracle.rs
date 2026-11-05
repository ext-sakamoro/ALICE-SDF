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

use std::path::{Path, PathBuf};

/// The only file of the crate that creates a `wgpu::Instance` or a device
const LIFECYCLE_MODULE: &str = "src/gpu_context.rs";

/// The calls only [`LIFECYCLE_MODULE`] makes
const LIFECYCLE_CALLS: &[&str] = &["Instance::new(", ".request_device("];

/// With the white space gone: a kept clone of these holds the device past its context
const KEPT_HANDLES: &[&str] = &[".device().clone()", ".queue().clone()"];

/// What is scanned
const SCANNED: &[&str] = &["src", "tests", "examples", "benches"];

/// The name of the struct or enum declared on `line`, attributes and visibility passed over
fn declared_name(line: &str) -> Option<&str> {
    let mut rest = line;
    while let Some(after) = rest.strip_prefix("#[") {
        rest = after.split_once(']')?.1.trim_start();
    }
    if let Some(after) = rest.strip_prefix("pub") {
        rest = after.trim_start();
        if rest.starts_with('(') {
            rest = rest.split_once(')')?.1.trim_start();
        }
    }
    let name = rest
        .strip_prefix("struct ")
        .or_else(|| rest.strip_prefix("enum "))?;
    name.split(|c: char| !c.is_alphanumeric() && c != '_')
        .next()
}

/// `(name, body)` of each struct and enum in `code`. Shader text in a string reads as a declaration
/// too, and has no `wgpu::` or context in it.
fn declarations(code: &[&str]) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let mut open: Option<(String, String)> = None;
    for line in code.iter().map(|l| l.trim()) {
        if open.is_none() {
            open = declared_name(line).map(|name| (name.to_string(), String::new()));
        }
        if let Some((_, body)) = open.as_mut() {
            body.push_str(line);
            body.push('\n');
            if line.starts_with('}') || line.ends_with(';') || line.ends_with("{}") {
                found.push(open.take().expect("open"));
            }
        }
    }
    found
}

/// What the rule forbids outside [`LIFECYCLE_MODULE`] in one file, besides the calls. A borrowed
/// field (`&'a`) ends before what it borrows, so it is not a holder.
fn drop_order_problems(code: &[&str], decls: &[(String, String)]) -> Vec<String> {
    let text = code.join("\n");
    let squeezed: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let mut problems = Vec::new();
    for handle in KEPT_HANDLES {
        if squeezed.contains(handle) {
            problems.push(format!("a kept handle: {handle}"));
        }
    }
    if squeezed.contains("=GpuContext;") {
        problems.push("an alias of GpuContext".to_string());
    }
    for line in code.iter().map(|l| l.trim()) {
        if line.starts_with("use wgpu::") && !line.starts_with("use wgpu::util::") {
            problems.push(format!("a wgpu type under another name: {line}"));
        }
    }
    for (name, body) in decls {
        let holds = |what: &str| body.lines().any(|l| l.contains(what) && !l.contains("&'"));
        if holds("GpuContext") {
            problems.push(format!(
                "{name} holds a context: `Held` is the one that does"
            ));
        }
        if holds("wgpu::") && !squeezed.contains(&format!("Held<{name}>")) {
            problems.push(format!(
                "{name} holds a wgpu resource and is not `Held<{name}>`"
            ));
        }
    }
    problems
}

/// What the source check finds
#[derive(Default)]
struct Scan {
    declarations: usize,
    calls: Vec<(PathBuf, &'static str)>,
    problems: Vec<(String, String)>,
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
        let code: Vec<&str> = text
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect();
        let file = path.display().to_string();
        for line in &code {
            for call in LIFECYCLE_CALLS {
                if line.contains(call) {
                    found.calls.push((path.clone(), call));
                }
            }
        }
        let decls = declarations(&code);
        found.declarations += decls.len();
        if path != owner {
            for problem in drop_order_problems(&code, &decls) {
                found.problems.push((file.clone(), problem));
            }
        }
    }
}

/// The text half of the rule. Clippy's disallowed lists hold the instance and device calls by their
/// resolved names, aliases included, and its allow for the device call is on the one function that
/// takes the lock as proof. A rename in wgpu of a call the module makes fails to compile there and
/// fails this check; `create_device_from_hal` is not called, so a rename of it would drop that entry
/// silently. Neither half sees a context returned in a tuple with a resource, a resource bound late
/// to a local declared before its context, or a handle kept through a variable: those are for review.
#[test]
fn one_module_owns_the_gpu_lifecycle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let owner = root.join(LIFECYCLE_MODULE);
    let mut found = Scan::default();
    for dir in SCANNED {
        scan(&root.join(dir), &owner, &mut found);
    }
    assert!(found.declarations > 0, "the check read nothing");
    let calls: Vec<_> = found
        .calls
        .iter()
        .filter(|(file, _)| *file != owner)
        .collect();
    assert!(calls.is_empty(), "outside {LIFECYCLE_MODULE}: {calls:#?}");
    assert!(
        found.problems.is_empty(),
        "drop order: {:#?}",
        found.problems
    );
    for call in LIFECYCLE_CALLS {
        assert!(
            found.calls.iter().any(|(_, c)| c == call),
            "{LIFECYCLE_MODULE} makes no `{call}` call: the check read nothing"
        );
    }
}

/// A device is dropped in one place, the function that takes the lock as proof. A second drop would
/// wait for the lock and run outside it, which the lock test cannot tell apart.
#[test]
fn the_one_device_drop_takes_the_lock_as_proof() {
    let text =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(LIFECYCLE_MODULE))
            .expect("read the lifecycle module");
    let drops: Vec<_> = text.match_indices("ManuallyDrop::drop(").collect();
    assert_eq!(drops.len(), 1, "{LIFECYCLE_MODULE}: {drops:?}");
    let before = &text[..drops[0].0];
    let last_fn = before.rfind("fn ").expect("a function before the drop");
    assert!(
        before[last_fn..].starts_with("fn destroy(_held: &mut Lifecycle,"),
        "the drop is in: {}",
        &before[last_fn..last_fn + 40]
    );
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
