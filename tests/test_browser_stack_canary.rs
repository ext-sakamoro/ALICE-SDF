//! The tree evaluator's depth in a browser background thread
//!
//! `scripts/browser_stack_canary.py` builds `examples/stack_canary_wasm.rs` for
//! wasm32-unknown-unknown and runs it in a headless Chromium background thread: each export must
//! evaluate a chain at least as deep as the wasm stack budget test holds (1792 levels, the 1 MiB
//! module stack at 512 B per level). The compile path is held to it by `scripts/stack_budget.sh`
//! (CI `stack-budget`); `eval` is not yet, and this test names that gap. It needs Chromium and
//! the wasm32-unknown-unknown target (`cargo test --test test_browser_stack_canary -- --ignored`).
//!
//! Author: Moroya Sakamoto

use std::process::Command;

#[test]
#[ignore = "known defect: recursive eval reaches 1096 in V8; closed by explicit work stacks"]
fn eval_reaches_the_wasm_canary_depth_in_a_browser_thread() {
    let status = Command::new("python3")
        .args(["scripts/browser_stack_canary.py", "--only", "canary_eval"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("run scripts/browser_stack_canary.py");
    assert!(
        status.success(),
        "canary_eval stayed below the wasm canary depth in the browser"
    );
}
