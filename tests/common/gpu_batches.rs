//! GPU evaluation of a point list in bounded submits: one `eval_batch` is one compute dispatch, and
//! tests stay at a dispatch size every GPU in the test matrix handles without stalling its driver.
//! `tests/test_gpu_submit_limit.rs` refuses a test that dispatches more than this directly.
//!
//! Author: Moroya Sakamoto
#![allow(dead_code, reason = "each test binary uses part of this module")]

use glam::Vec3;

/// The most points a test sends to the GPU in one submit
pub const MAX_POINTS_PER_SUBMIT: usize = 2048;

/// `submit` over `pts` in order, at most [`MAX_POINTS_PER_SUBMIT`] points at a time, the results
/// joined in the order of `pts`
///
/// # Errors
///
/// The first error of `submit`
pub fn in_batches<T, E>(
    pts: &[Vec3],
    mut submit: impl FnMut(&[Vec3]) -> Result<Vec<T>, E>,
) -> Result<Vec<T>, E> {
    let mut out = Vec::with_capacity(pts.len());
    for chunk in pts.chunks(MAX_POINTS_PER_SUBMIT) {
        out.extend(submit(chunk)?);
    }
    Ok(out)
}

/// [`in_batches`] on a GPU evaluator's `eval_batch`
///
/// # Errors
///
/// The first error of `eval_batch`
#[cfg(feature = "gpu")]
pub fn eval_in_batches(
    gpu: &alice_sdf::compiled::GpuEvaluator,
    pts: &[Vec3],
) -> Result<Vec<f32>, alice_sdf::compiled::GpuError> {
    in_batches(pts, |chunk| gpu.eval_batch(chunk))
}
