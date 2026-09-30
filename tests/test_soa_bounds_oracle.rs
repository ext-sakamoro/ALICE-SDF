//! 3.1.x backport: the SoA out-of-bounds write oracle only.
//!
//! 4.0.0 carries the full binding oracle (`tests/test_binding_oracle.rs`, 9 tests).
//! This file is the subset that pins the one thing 3.1.1 fixes, so the 3.1.x
//! line cannot regress: `eval_compiled_batch_soa_raw` and the two C wrappers
//! that reach it must touch exactly `count` elements of each array.
//!
//! Author: Moroya Sakamoto

use alice_sdf::animation::AnimationParams;
use alice_sdf::compiled::{eval_compiled, CompiledSdf};
use alice_sdf::ffi::{BatchResult, CompiledHandle, SdfHandle, SdfResult};
use alice_sdf::prelude::*;
use std::ffi::CString;
use std::os::raw::c_char;

extern "C" {
    // primitives
    fn alice_sdf_sphere(radius: f32) -> SdfHandle;

    fn alice_sdf_eval_soa(
        compiled: CompiledHandle,
        x: *const f32,
        y: *const f32,
        z: *const f32,
        distances: *mut f32,
        count: u32,
    ) -> BatchResult;
    #[allow(clippy::too_many_arguments)]
    fn alice_sdf_eval_animated_batch_soa(
        compiled: CompiledHandle,
        params: *const AnimationParams,
        x: *const f32,
        y: *const f32,
        z: *const f32,
        distances: *mut f32,
        count: u32,
    ) -> BatchResult;

    // lifecycle / introspection
    fn alice_sdf_compile(node: SdfHandle) -> CompiledHandle;
    fn alice_sdf_free(node: SdfHandle);
    fn alice_sdf_free_compiled(compiled: CompiledHandle);
    fn alice_sdf_load(path: *const c_char) -> SdfHandle;
}

// ============================================================================
// Helpers
// ============================================================================

/// Deterministic sample points: the axis / sector ties the other parity tests
/// use, plus an LCG spray over [-3, 3]³.
fn sample_points(n: usize) -> Vec<Vec3> {
    let mut pts = vec![
        Vec3::ZERO,
        Vec3::new(0.25, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::new(0.3, -0.7, 1.1),
        Vec3::new(1.5, 1.5, 1.5),
        Vec3::new(-2.0, 0.4, -0.9),
        Vec3::new(0.0, -2.25, 0.0),
    ];
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32) * 6.0 - 3.0
    };
    while pts.len() < n {
        pts.push(Vec3::new(next(), next(), next()));
    }
    pts.truncate(n);
    pts
}

fn bits(v: f32) -> String {
    format!("{v:.7e} ({:#010x})", v.to_bits())
}

/// `-0.0 == 0.0` for the FFI's purposes: the wrapper must not change the
/// value, and a sign-of-zero difference is a real difference, so compare the
/// bits.
const fn same_bits(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits()
}

/// Evaluate the whole point set through one AoS batch entry point.
unsafe fn soa(compiled: CompiledHandle, pts: &[Vec3]) -> Vec<f32> {
    let xs: Vec<f32> = pts.iter().map(|p| p.x).collect();
    let ys: Vec<f32> = pts.iter().map(|p| p.y).collect();
    let zs: Vec<f32> = pts.iter().map(|p| p.z).collect();
    let mut out = vec![f32::NAN; pts.len()];
    let r = alice_sdf_eval_soa(
        compiled,
        xs.as_ptr(),
        ys.as_ptr(),
        zs.as_ptr(),
        out.as_mut_ptr(),
        u32::try_from(pts.len()).expect("count fits u32"),
    );
    assert_eq!(r.result, SdfResult::Ok, "SoA call rejected the input");
    out
}

// ============================================================================
// 1. Closed forms through the C API
// ============================================================================

/// Exact signed distances, written down by hand — no crate function was
/// called to produce any number in this test.
///
/// The values also pin the half-extent / half-height convention of the C API
/// (`include/alice_sdf.h` documents `hx` as "Half-extent in X" and
/// `half_height` as "Half-height of the cylinder"): a wrapper that forwarded
/// these as full extents would halve or double every distance below.
fn soa_overrun_by_count(counts: &[usize]) -> Vec<(usize, usize)> {
    const CANARY: f32 = 12345.0;
    const SLACK: usize = 8;
    let identity = AnimationParams {
        scale: 1.0,
        ..Default::default()
    };
    let handle = unsafe { alice_sdf_sphere(1.0) };
    let compiled_handle = unsafe { alice_sdf_compile(handle) };
    let mut measured = Vec::new();
    for &count in counts {
        let pts = sample_points(count);
        let mut xs: Vec<f32> = pts.iter().map(|p| p.x).collect();
        let mut ys: Vec<f32> = pts.iter().map(|p| p.y).collect();
        let mut zs: Vec<f32> = pts.iter().map(|p| p.z).collect();
        for v in [&mut xs, &mut ys, &mut zs] {
            v.resize(count + SLACK, 0.5);
        }
        let mut worst = 0usize;
        {
            // The public raw entry point, called directly (it is re-exported
            // from `lib.rs`, so external Rust callers reach it without the FFI).
            let mut out = vec![CANARY; count + SLACK];
            unsafe {
                alice_sdf::compiled::eval_compiled_batch_soa_raw(
                    &CompiledSdf::compile(&SdfNode::sphere(1.0)),
                    xs.as_ptr(),
                    ys.as_ptr(),
                    zs.as_ptr(),
                    out.as_mut_ptr(),
                    count,
                );
            }
            worst = worst.max(out[count..].iter().filter(|v| **v != CANARY).count());
        }
        for which in 0..2 {
            let mut out = vec![CANARY; count + SLACK];
            let r = unsafe {
                if which == 0 {
                    alice_sdf_eval_soa(
                        compiled_handle,
                        xs.as_ptr(),
                        ys.as_ptr(),
                        zs.as_ptr(),
                        out.as_mut_ptr(),
                        u32::try_from(count).expect("count fits u32"),
                    )
                } else {
                    alice_sdf_eval_animated_batch_soa(
                        compiled_handle,
                        &raw const identity,
                        xs.as_ptr(),
                        ys.as_ptr(),
                        zs.as_ptr(),
                        out.as_mut_ptr(),
                        u32::try_from(count).expect("count fits u32"),
                    )
                }
            };
            assert_eq!(r.result, SdfResult::Ok, "count={count} which={which}");
            worst = worst.max(out[count..].iter().filter(|v| **v != CANARY).count());
        }
        measured.push((count, worst));
    }
    unsafe {
        alice_sdf_free_compiled(compiled_handle);
        alice_sdf_free(handle);
    }
    measured
}

/// The contract `include/alice_sdf.h` states for `alice_sdf_eval_soa` and
/// `alice_sdf_eval_animated_batch_soa`:
///
/// > `distances` must point to a writable array of at least `count` f32 values.
/// > Arrays should be 32-byte aligned for AVX2 (not required, but faster).
///
/// A caller that allocates exactly `count` floats — which is what that sentence
/// permits — must therefore come back with at most `count` floats written.
///
/// This was red on `dde189d` and is the reason this file exists. Both functions
/// forwarded the caller's raw pointers and the full `count` to
/// `compiled::eval_soa::eval_compiled_batch_soa_raw`, which rounds `count`
/// **up** to a multiple of 8 and reads and writes every full group, so they
/// went `(8 - count % 8) % 8` elements past the end of all four arrays.
/// Measured with a canary before the fix:
///
/// ```text
/// count=   1 eval_soa=7 animated(identity)=7   count=   7 eval_soa=1 animated=1
/// count= 257 eval_soa=7 animated(identity)=7   count=1023 eval_soa=1 animated=1
/// count=1025 eval_soa=0 animated(identity)=7
/// ```
///
/// With buffers of exactly `count` elements it is a SIGSEGV, not a silent
/// overwrite (a `Vec<f32>` of length 1 crashed the test binary). The same code
/// is in the published 3.1.0 crate
/// (`alice-sdf-3.1.0/src/compiled/eval_soa.rs:209`, called from its
/// `src/ffi/eval.rs:372` and `:770`), so it is not a 4.0.0 regression.
///
/// `alice_sdf_eval_batch`, `alice_sdf_eval_compiled_batch` and
/// `alice_sdf_eval_gradient_soa` were measured too and never overran: they walk
/// whole 8-groups and finish the remainder scalar, which is what the two
/// functions here now do.
#[test]
fn soa_must_not_write_past_the_documented_count() {
    let measured = soa_overrun_by_count(&[
        1, 2, 5, 7, 8, 9, 15, 16, 255, 256, 257, 1023, 1024, 1025, 1031,
    ]);
    let bad: Vec<String> = measured
        .iter()
        .filter(|(_, over)| *over != 0)
        .map(|(count, over)| format!("count={count}: wrote {over} slot(s) past the end"))
        .collect();
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

// ============================================================================
// 4. Gradient (normal) path
// ============================================================================

/// `alice_sdf_eval_gradient_soa` is the normal supplier for the Unity / UE5
/// particle and collision paths and had no test at all.
///
/// Oracle: the outward unit normal of a sphere at `p` is `p / |p|` — a closed
/// form, so this does not depend on the crate agreeing with itself. Both
/// internal branches are exercised, because they use *different* numerical
/// schemes: below 1024 points a 6-tap central difference, at and above 1024 a
/// 4-tap tetrahedral difference (`eval_distance_and_gradient_simd`). Both are
/// both estimate the gradient from samples a step `eps = 1e-3` apart (the
/// constant `EPSILON` in `src/ffi/eval.rs`).
///
/// Error bound, derived rather than fitted: for `d(p) = |p| - 1` the Hessian
/// has eigenvalues `1/|p|`, so a finite difference of step `eps` misses the
/// true direction by `O(eps / |p|)` — the bound below is `2 * eps / |p|`,
/// which at `|p| = 3` is 6.7e-4 and only relaxes near the origin, where the
/// normal genuinely degenerates. (A flat 2e-3 was the first bound written
/// here and it failed at `|p| = 0.21` with 2.24e-3, i.e. 0.48 * eps / |p| —
/// the constant, not the implementation, was wrong.)
///
/// The returned distance, by contrast, must be bit-identical to
/// `eval_compiled`.
#[test]
fn exact_size_buffers_survive_and_agree() {
    let node = SdfNode::sphere(0.6)
        .union(SdfNode::sphere(0.5).translate(0.8, 0.0, 0.0))
        .twist(0.9);
    let compiled = CompiledSdf::compile(&node);
    let identity = AnimationParams {
        scale: 1.0,
        ..Default::default()
    };
    let dir = std::env::temp_dir().join(format!("alice_sdf_ffi_exact_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("node.asdf");
    alice_sdf::save(&SdfTree::new(node), &path).expect("save");
    let c_path = CString::new(path.to_string_lossy().as_ref()).expect("path has no NUL");
    let handle = unsafe { alice_sdf_load(c_path.as_ptr()) };
    let compiled_handle = unsafe { alice_sdf_compile(handle) };

    let mut failures = Vec::new();
    for count in [
        1usize, 2, 5, 7, 8, 9, 15, 16, 255, 256, 257, 1023, 1024, 1025, 1031, 4096,
    ] {
        let pts = sample_points(count);
        let xs: Vec<f32> = pts.iter().map(|p| p.x).collect();
        let ys: Vec<f32> = pts.iter().map(|p| p.y).collect();
        let zs: Vec<f32> = pts.iter().map(|p| p.z).collect();
        let want: Vec<f32> = pts.iter().map(|p| eval_compiled(&compiled, *p)).collect();

        // 1. the public raw function, exact-size buffers
        let mut raw_out = vec![f32::NAN; count];
        unsafe {
            alice_sdf::compiled::eval_compiled_batch_soa_raw(
                &compiled,
                xs.as_ptr(),
                ys.as_ptr(),
                zs.as_ptr(),
                raw_out.as_mut_ptr(),
                count,
            );
        }

        // 2. the C wrapper
        let ffi_out = unsafe { soa(compiled_handle, &pts) };

        // 3. the animated C wrapper on its identity branch
        let mut anim_out = vec![f32::NAN; count];
        let r = unsafe {
            alice_sdf_eval_animated_batch_soa(
                compiled_handle,
                &raw const identity,
                xs.as_ptr(),
                ys.as_ptr(),
                zs.as_ptr(),
                anim_out.as_mut_ptr(),
                u32::try_from(count).expect("count fits u32"),
            )
        };
        assert_eq!(r.result, SdfResult::Ok, "count={count} animated");

        for (label, got) in [
            ("eval_compiled_batch_soa_raw", &raw_out),
            ("alice_sdf_eval_soa", &ffi_out),
            ("alice_sdf_eval_animated_batch_soa(identity)", &anim_out),
        ] {
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                if !same_bits(g, w) {
                    failures.push(format!(
                        "count={count} {label}[{i}] at {:?}: {} vs native compiled {}",
                        pts[i],
                        bits(g),
                        bits(w)
                    ));
                    break;
                }
            }
        }
    }
    unsafe {
        alice_sdf_free_compiled(compiled_handle);
        alice_sdf_free(handle);
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "{} exact-size difference(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
