//! Binding-layer oracles for the C ABI (`--features ffi`).
//!
//! The C ABI is what Unity (`bindings/AliceSdf.cs`), Unreal
//! (`unreal-plugin/`) and every ctypes/cffi caller actually execute, and until
//! now nothing in `cargo test` looked at the values it returns: `src/ffi/**`
//! had 15 behavioural unit tests (registry bookkeeping, panic sentinels, a
//! unit sphere at three points) and no comparison against either a closed
//! form or the native Rust API it wraps.
//!
//! Two independent expectation sources are used here, never the FFI itself:
//!
//! 1. **closed forms** — a sphere, box, cylinder, torus, plane and capsule
//!    have exact signed distances that were written down by hand (see each
//!    assertion's comment). These also pin the *convention* of the C API:
//!    `alice_sdf_box` takes half-extents while `SdfNode::box3d` takes full
//!    extents, and `alice_sdf_cylinder` takes a half-height while
//!    `SdfNode::cylinder` takes the full height. Passing the wrong one
//!    produces a shape of the wrong size, which the numbers below catch.
//! 2. **the native Rust API** — `eval` / `eval_compiled` /
//!    `eval_animated_compiled`. The FFI functions are documented as thin
//!    wrappers over these, so "thin" is testable to the bit: the crate's own
//!    determinism rule (`tests/test_det_parity.rs`) makes the tree, scalar
//!    and SIMD evaluators bit-identical, so any difference the FFI introduces
//!    is the FFI's own (wrong argument, wrong registry, a lost lane in the
//!    threshold-dependent batch paths).
//!
//! Scope note: the *declared types* of the exported functions are compared
//! against `include/alice_sdf.h` and `bindings/AliceSdf.cs` by
//! `scripts/unreal-abi-check.sh` step 2a (added in `7853907` after
//! `alice_sdf_mirror` was declared `float` for three major versions while
//! Rust took `u8`). This file deliberately does not repeat that textual
//! comparison — it calls the exported symbols and checks the values, which is
//! the half a declaration diff cannot see.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "ffi")]

mod common;

use alice_sdf::animation::AnimationParams;
use alice_sdf::compiled::{eval_compiled, CompiledSdf};
use alice_sdf::ffi::{BatchResult, CompiledHandle, SdfHandle, SdfResult};
use alice_sdf::prelude::*;
use common::corpus::corpus;
use std::ffi::CString;
use std::os::raw::c_char;

// ============================================================================
// Exported symbols
// ============================================================================
//
// Declared here rather than reached through `alice_sdf::ffi::…` because the
// modules under `src/ffi/` are private: the only way into them is the symbol
// the dynamic library exports, which is also the only way Unity and Unreal
// get in. A missing `#[no_mangle]` therefore fails this file at link time.
//
// The signatures are transcribed from the `extern "C" fn` definitions in
// `src/ffi/`; the header's agreement with those is step 2a's job (see the
// module comment).

extern "C" {
    // primitives
    fn alice_sdf_sphere(radius: f32) -> SdfHandle;
    fn alice_sdf_box(hx: f32, hy: f32, hz: f32) -> SdfHandle;
    fn alice_sdf_cylinder(radius: f32, half_height: f32) -> SdfHandle;
    fn alice_sdf_torus(major_radius: f32, minor_radius: f32) -> SdfHandle;
    fn alice_sdf_plane(nx: f32, ny: f32, nz: f32, distance: f32) -> SdfHandle;
    fn alice_sdf_capsule(
        ax: f32,
        ay: f32,
        az: f32,
        bx: f32,
        by: f32,
        bz: f32,
        radius: f32,
    ) -> SdfHandle;
    fn alice_sdf_ellipsoid(rx: f32, ry: f32, rz: f32) -> SdfHandle;
    fn alice_sdf_octahedron(size: f32) -> SdfHandle;
    fn alice_sdf_rounded_box(hx: f32, hy: f32, hz: f32, round_radius: f32) -> SdfHandle;
    fn alice_sdf_hex_prism(hex_radius: f32, half_height: f32) -> SdfHandle;
    fn alice_sdf_cone(radius: f32, half_height: f32) -> SdfHandle;
    fn alice_sdf_tube(outer_radius: f32, thickness: f32, half_height: f32) -> SdfHandle;

    // operations
    fn alice_sdf_union(a: SdfHandle, b: SdfHandle) -> SdfHandle;
    fn alice_sdf_intersection(a: SdfHandle, b: SdfHandle) -> SdfHandle;
    fn alice_sdf_subtract(a: SdfHandle, b: SdfHandle) -> SdfHandle;
    fn alice_sdf_smooth_union(a: SdfHandle, b: SdfHandle, k: f32) -> SdfHandle;
    fn alice_sdf_smooth_subtract(a: SdfHandle, b: SdfHandle, k: f32) -> SdfHandle;
    fn alice_sdf_chamfer_union(a: SdfHandle, b: SdfHandle, r: f32) -> SdfHandle;
    fn alice_sdf_morph(a: SdfHandle, b: SdfHandle, t: f32) -> SdfHandle;
    fn alice_sdf_xor(a: SdfHandle, b: SdfHandle) -> SdfHandle;

    // modifiers
    fn alice_sdf_round(node: SdfHandle, radius: f32) -> SdfHandle;
    fn alice_sdf_onion(node: SdfHandle, thickness: f32) -> SdfHandle;
    fn alice_sdf_twist(node: SdfHandle, strength: f32) -> SdfHandle;
    fn alice_sdf_bend(node: SdfHandle, curvature: f32) -> SdfHandle;
    fn alice_sdf_repeat(node: SdfHandle, sx: f32, sy: f32, sz: f32) -> SdfHandle;
    fn alice_sdf_mirror(node: SdfHandle, mx: u8, my: u8, mz: u8) -> SdfHandle;
    fn alice_sdf_elongate(node: SdfHandle, x: f32, y: f32, z: f32) -> SdfHandle;
    fn alice_sdf_polar_repeat(node: SdfHandle, count: u32) -> SdfHandle;
    // ⚠️ 引数順は counts (u32 x3) が先、spacing (f32 x3) が後。
    // 2026-09-30: この宣言は floats を先に書いていて Windows CI だけが red に
    // なった (run 36680136653)。SysV (macOS / Linux) は整数と浮動小数で
    // レジスタバンクが分かれ、それぞれ独立に採番するので、順序を入れ替えても
    // 値が偶然正しいレジスタに着いて test が通る。Microsoft x64 は引数の
    // 「位置」でスロットを決めるため、入れ替えると callee が無関係な
    // レジスタとスタックを読む。宣言の順序誤りは SysV では検出できない。
    fn alice_sdf_repeat_finite(
        node: SdfHandle,
        cx: u32,
        cy: u32,
        cz: u32,
        sx: f32,
        sy: f32,
        sz: f32,
    ) -> SdfHandle;
    fn alice_sdf_shear(node: SdfHandle, xy: f32, xz: f32, yz: f32) -> SdfHandle;

    // transforms
    fn alice_sdf_translate(node: SdfHandle, x: f32, y: f32, z: f32) -> SdfHandle;
    fn alice_sdf_rotate_euler(node: SdfHandle, x: f32, y: f32, z: f32) -> SdfHandle;
    fn alice_sdf_scale(node: SdfHandle, factor: f32) -> SdfHandle;
    fn alice_sdf_scale_xyz(node: SdfHandle, x: f32, y: f32, z: f32) -> SdfHandle;

    // evaluation
    fn alice_sdf_eval(node: SdfHandle, x: f32, y: f32, z: f32) -> f32;
    fn alice_sdf_eval_compiled(compiled: CompiledHandle, x: f32, y: f32, z: f32) -> f32;
    fn alice_sdf_eval_batch(
        node: SdfHandle,
        points: *const f32,
        distances: *mut f32,
        count: u32,
    ) -> BatchResult;
    fn alice_sdf_eval_compiled_batch(
        compiled: CompiledHandle,
        points: *const f32,
        distances: *mut f32,
        count: u32,
    ) -> BatchResult;
    fn alice_sdf_eval_soa(
        compiled: CompiledHandle,
        x: *const f32,
        y: *const f32,
        z: *const f32,
        distances: *mut f32,
        count: u32,
    ) -> BatchResult;
    #[allow(clippy::too_many_arguments)]
    fn alice_sdf_eval_gradient_soa(
        compiled: CompiledHandle,
        x: *const f32,
        y: *const f32,
        z: *const f32,
        nx: *mut f32,
        ny: *mut f32,
        nz: *mut f32,
        dist: *mut f32,
        count: u32,
    ) -> BatchResult;
    fn alice_sdf_eval_animated_compiled(
        compiled: CompiledHandle,
        params: *const AnimationParams,
        x: f32,
        y: f32,
        z: f32,
    ) -> f32;
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
    fn alice_sdf_clone(node: SdfHandle) -> SdfHandle;
    fn alice_sdf_node_count(node: SdfHandle) -> u32;
    fn alice_sdf_is_valid(node: SdfHandle) -> bool;
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
unsafe fn batch_aos(
    call: unsafe extern "C" fn(*mut std::ffi::c_void, *const f32, *mut f32, u32) -> BatchResult,
    handle: *mut std::ffi::c_void,
    pts: &[Vec3],
) -> Vec<f32> {
    let flat: Vec<f32> = pts.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    let mut out = vec![f32::NAN; pts.len()];
    let r = call(
        handle,
        flat.as_ptr(),
        out.as_mut_ptr(),
        u32::try_from(pts.len()).expect("count fits u32"),
    );
    assert_eq!(r.result, SdfResult::Ok, "batch call rejected the input");
    assert_eq!(r.count as usize, pts.len(), "batch count");
    out
}

/// Evaluate the whole point set through `alice_sdf_eval_soa`.
///
/// The buffers are exactly `count` long, which is all the C header asks callers
/// for. That matters: before the `count & !7` truncation in
/// `src/ffi/eval.rs`, a `Vec` of exactly 1 element made this a SIGSEGV
/// (see `soa_must_not_write_past_the_documented_count`), so every caller of
/// this helper doubles as a crash-level regression detector.
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
#[test]
fn ffi_primitives_match_closed_forms() {
    unsafe {
        // Unit sphere: d(p) = |p| - 1.
        let s = alice_sdf_sphere(1.0);
        assert_eq!(alice_sdf_eval(s, 0.5, 0.0, 0.0), -0.5, "sphere inside");
        assert_eq!(alice_sdf_eval(s, 2.0, 0.0, 0.0), 1.0, "sphere outside");
        assert_eq!(alice_sdf_eval(s, 0.0, 0.0, 0.0), -1.0, "sphere centre");
        assert_eq!(alice_sdf_eval(s, 1.0, 0.0, 0.0), 0.0, "sphere surface");

        // Box with half-extents (1,1,1): -1 at the centre, +1 one unit off a
        // face, sqrt(3) at the diagonal corner direction (2,2,2).
        let b = alice_sdf_box(1.0, 1.0, 1.0);
        assert_eq!(alice_sdf_eval(b, 0.0, 0.0, 0.0), -1.0, "box centre");
        assert_eq!(alice_sdf_eval(b, 2.0, 0.0, 0.0), 1.0, "box face");
        let corner = alice_sdf_eval(b, 2.0, 2.0, 2.0);
        assert!(
            (corner - 3.0_f32.sqrt()).abs() < 1e-6,
            "box corner: {} vs sqrt(3)",
            bits(corner)
        );

        // Cylinder radius 1, half-height 1 (so the full height is 2): the
        // centre is 1 from both the side and the caps.
        let c = alice_sdf_cylinder(1.0, 1.0);
        assert_eq!(alice_sdf_eval(c, 0.0, 0.0, 0.0), -1.0, "cylinder centre");
        assert_eq!(alice_sdf_eval(c, 2.0, 0.0, 0.0), 1.0, "cylinder side");
        assert_eq!(alice_sdf_eval(c, 0.0, 2.0, 0.0), 1.0, "cylinder cap");

        // Torus in XZ, major 1, minor 0.25: the circle of revolution is at
        // radius 1, so (1,0,0) is 0.25 inside and (1.5,0,0) is 0.25 outside;
        // the hole centre is 1 - 0.25 = 0.75 away.
        let t = alice_sdf_torus(1.0, 0.25);
        assert_eq!(alice_sdf_eval(t, 1.0, 0.0, 0.0), -0.25, "torus tube centre");
        assert_eq!(alice_sdf_eval(t, 1.5, 0.0, 0.0), 0.25, "torus outside");
        assert_eq!(alice_sdf_eval(t, 0.0, 0.0, 0.0), 0.75, "torus hole");

        // Plane with normal +Y through the origin: d = p·n - distance.
        let pl = alice_sdf_plane(0.0, 1.0, 0.0, 0.0);
        assert_eq!(alice_sdf_eval(pl, 0.0, 3.0, 0.0), 3.0, "plane above");
        assert_eq!(alice_sdf_eval(pl, 5.0, -2.0, 7.0), -2.0, "plane below");

        // Capsule from (0,-1,0) to (0,1,0) with radius 0.5: on the axis the
        // distance is -0.5, one unit sideways it is 0.5, and one unit past
        // the end cap it is 0.5.
        let cap = alice_sdf_capsule(0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.5);
        assert_eq!(alice_sdf_eval(cap, 0.0, 0.0, 0.0), -0.5, "capsule axis");
        assert_eq!(alice_sdf_eval(cap, 1.0, 0.0, 0.0), 0.5, "capsule side");
        assert_eq!(alice_sdf_eval(cap, 0.0, 2.0, 0.0), 0.5, "capsule cap");

        // Boolean laws, exactly: union = min, intersection = max,
        // subtraction = max(a, -b).
        let s2 = alice_sdf_translate(alice_sdf_sphere(1.0), 1.0, 0.0, 0.0);
        let u = alice_sdf_union(s, s2);
        assert_eq!(
            alice_sdf_eval(u, 0.5, 0.0, 0.0),
            -0.5,
            "union takes the min"
        );
        let i = alice_sdf_intersection(s, s2);
        assert_eq!(
            alice_sdf_eval(i, 0.5, 0.0, 0.0),
            -0.5,
            "intersection takes the max"
        );
        let sub = alice_sdf_subtract(s, s2);
        // At (-0.5,0,0): a = -0.5, b = |(-1.5,0,0)| - 1 = 0.5, so
        // max(a, -b) = max(-0.5, -0.5) = -0.5.
        assert_eq!(alice_sdf_eval(sub, -0.5, 0.0, 0.0), -0.5, "subtract");
        // At (0.5,0,0): a = -0.5, b = -0.5, so max(-0.5, 0.5) = 0.5.
        assert_eq!(alice_sdf_eval(sub, 0.5, 0.0, 0.0), 0.5, "subtract carves");

        // Onion of thickness t turns a solid into a shell: |d| - t.
        let shell = alice_sdf_onion(alice_sdf_sphere(1.0), 0.1);
        assert_eq!(alice_sdf_eval(shell, 0.0, 0.0, 0.0), 0.9, "onion centre");
        assert_eq!(alice_sdf_eval(shell, 1.0, 0.0, 0.0), -0.1, "onion on shell");

        // Round of radius r inflates by r: |p| - 1 - r.
        let rounded = alice_sdf_round(alice_sdf_sphere(1.0), 0.25);
        assert_eq!(
            alice_sdf_eval(rounded, 2.0, 0.0, 0.0),
            0.75,
            "round inflates"
        );

        // Uniform scale by 2 maps d(p) to 2·d(p/2).
        let big = alice_sdf_scale(alice_sdf_sphere(1.0), 2.0);
        assert_eq!(alice_sdf_eval(big, 4.0, 0.0, 0.0), 2.0, "scale 2 sphere");
        assert_eq!(alice_sdf_eval(big, 0.0, 0.0, 0.0), -2.0, "scale 2 centre");

        for h in [s, b, c, t, pl, cap, s2, u, i, sub, shell, rounded, big] {
            alice_sdf_free(h);
        }
    }
}

// ============================================================================
// 2. Constructor parity: the C constructors build the documented node
// ============================================================================

/// Every C constructor is paired with the `SdfNode` the header says it builds,
/// and the two are compared through the *same* evaluator, so a difference can
/// only come from the constructor: a swapped argument, a forgotten `* 0.5`, a
/// wrong enum variant, an integer argument read as a float.
///
/// The parameters are pairwise distinct on purpose — `(0.31, 0.47, 0.59)`
/// rather than `(0.5, 0.5, 0.5)` — so that a transposition changes the field.
///
/// Expected side: the `SdfNode` variants and constructors of
/// `src/types/constructors.rs`, which the header documents as the meaning of
/// each C function (`hx` = half-extent, `half_height` = half-height, hence the
/// struct literals rather than `SdfNode::box3d` / `SdfNode::cylinder`, which
/// take full extents).
#[test]
fn ffi_constructors_build_the_documented_node() {
    let pts = sample_points(96);
    let leaf_a = || SdfNode::sphere(0.6);
    let leaf_b = || SdfNode::sphere(0.5).translate(0.8, 0.0, 0.0);

    // (name, handle built through the C API, node the C API documents)
    let cases: Vec<(&str, SdfHandle, SdfNode)> = unsafe {
        let a = || alice_sdf_sphere(0.6);
        let b = || alice_sdf_translate(alice_sdf_sphere(0.5), 0.8, 0.0, 0.0);
        vec![
            // --- primitives ---
            (
                "sphere",
                alice_sdf_sphere(0.31),
                SdfNode::Sphere { radius: 0.31 },
            ),
            (
                "box",
                alice_sdf_box(0.31, 0.47, 0.59),
                SdfNode::Box3d {
                    half_extents: Vec3::new(0.31, 0.47, 0.59),
                },
            ),
            (
                "cylinder",
                alice_sdf_cylinder(0.31, 0.47),
                SdfNode::Cylinder {
                    radius: 0.31,
                    half_height: 0.47,
                },
            ),
            (
                "torus",
                alice_sdf_torus(0.71, 0.23),
                SdfNode::Torus {
                    major_radius: 0.71,
                    minor_radius: 0.23,
                },
            ),
            (
                "plane",
                alice_sdf_plane(0.0, 1.0, 0.0, 0.2),
                SdfNode::plane(Vec3::Y, 0.2),
            ),
            (
                "capsule",
                alice_sdf_capsule(-0.5, 0.0, 0.0, 0.5, 0.2, 0.0, 0.25),
                SdfNode::capsule(Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.2, 0.0), 0.25),
            ),
            (
                "cone",
                alice_sdf_cone(0.5, 0.4),
                SdfNode::Cone {
                    radius: 0.5,
                    half_height: 0.4,
                },
            ),
            (
                "ellipsoid",
                alice_sdf_ellipsoid(0.61, 0.43, 0.29),
                SdfNode::ellipsoid(0.61, 0.43, 0.29),
            ),
            (
                "octahedron",
                alice_sdf_octahedron(0.6),
                SdfNode::octahedron(0.6),
            ),
            (
                "rounded_box",
                alice_sdf_rounded_box(0.31, 0.47, 0.59, 0.07),
                SdfNode::rounded_box(0.31, 0.47, 0.59, 0.07),
            ),
            (
                "hex_prism",
                alice_sdf_hex_prism(0.5, 0.3),
                SdfNode::HexPrism {
                    hex_radius: 0.5,
                    half_height: 0.3,
                },
            ),
            (
                "tube",
                alice_sdf_tube(0.5, 0.1, 0.3),
                SdfNode::Tube {
                    outer_radius: 0.5,
                    thickness: 0.1,
                    half_height: 0.3,
                },
            ),
            // --- operations ---
            ("union", alice_sdf_union(a(), b()), leaf_a().union(leaf_b())),
            (
                "intersection",
                alice_sdf_intersection(a(), b()),
                leaf_a().intersection(leaf_b()),
            ),
            (
                "subtract",
                alice_sdf_subtract(a(), b()),
                leaf_a().subtract(leaf_b()),
            ),
            (
                "smooth_union",
                alice_sdf_smooth_union(a(), b(), 0.2),
                leaf_a().smooth_union(leaf_b(), 0.2),
            ),
            (
                "smooth_subtract",
                alice_sdf_smooth_subtract(a(), b(), 0.2),
                leaf_a().smooth_subtract(leaf_b(), 0.2),
            ),
            (
                "chamfer_union",
                alice_sdf_chamfer_union(a(), b(), 0.15),
                leaf_a().chamfer_union(leaf_b(), 0.15),
            ),
            (
                "morph",
                alice_sdf_morph(a(), b(), 0.35),
                leaf_a().morph(leaf_b(), 0.35),
            ),
            ("xor", alice_sdf_xor(a(), b()), leaf_a().xor(leaf_b())),
            // --- modifiers ---
            ("round", alice_sdf_round(a(), 0.07), leaf_a().round(0.07)),
            ("onion", alice_sdf_onion(a(), 0.05), leaf_a().onion(0.05)),
            ("twist", alice_sdf_twist(a(), 1.3), leaf_a().twist(1.3)),
            ("bend", alice_sdf_bend(a(), 0.7), leaf_a().bend(0.7)),
            (
                "repeat",
                alice_sdf_repeat(a(), 1.1, 1.3, 1.7),
                leaf_a().repeat_infinite(1.1, 1.3, 1.7),
            ),
            // mx / my / mz are `uint8_t` in the header and `u8` in Rust; the
            // 1.7.2-4.0.0 drift was a `float` declaration here. A boolean
            // argument read from the wrong register mirrors the wrong axes,
            // which the asymmetric child (b(), offset in X) exposes.
            (
                "mirror_x",
                alice_sdf_mirror(b(), 1, 0, 0),
                leaf_b().mirror(true, false, false),
            ),
            (
                "mirror_yz",
                alice_sdf_mirror(b(), 0, 1, 1),
                leaf_b().mirror(false, true, true),
            ),
            (
                "elongate",
                alice_sdf_elongate(a(), 0.2, 0.3, 0.4),
                leaf_a().elongate(0.2, 0.3, 0.4),
            ),
            (
                "polar_repeat",
                alice_sdf_polar_repeat(b(), 7),
                leaf_b().polar_repeat(7),
            ),
            (
                "repeat_finite",
                alice_sdf_repeat_finite(b(), 2, 1, 3, 1.1, 1.3, 1.7),
                leaf_b().repeat_finite([2, 1, 3], Vec3::new(1.1, 1.3, 1.7)),
            ),
            (
                "shear",
                alice_sdf_shear(a(), 0.2, 0.3, 0.4),
                leaf_a().shear(0.2, 0.3, 0.4),
            ),
            // --- transforms ---
            (
                "translate",
                alice_sdf_translate(a(), 0.3, -0.7, 1.1),
                leaf_a().translate(0.3, -0.7, 1.1),
            ),
            (
                "rotate_euler",
                alice_sdf_rotate_euler(b(), 0.3, -0.7, 1.1),
                leaf_b().rotate_euler(0.3, -0.7, 1.1),
            ),
            ("scale", alice_sdf_scale(a(), 1.7), leaf_a().scale(1.7)),
            (
                "scale_xyz",
                alice_sdf_scale_xyz(a(), 1.3, 0.7, 2.1),
                leaf_a().scale_xyz(1.3, 0.7, 2.1),
            ),
        ]
    };

    let mut failures = Vec::new();
    for (name, handle, node) in &cases {
        assert!(
            !handle.is_null(),
            "{name}: constructor returned the null handle"
        );
        let native_count = node.node_count() as usize;
        let ffi_count = unsafe { alice_sdf_node_count(*handle) } as usize;
        if ffi_count != native_count {
            failures.push(format!(
                "{name}: node_count FFI {ffi_count} vs native {native_count}"
            ));
        }
        for p in &pts {
            let got = unsafe { alice_sdf_eval(*handle, p.x, p.y, p.z) };
            let want = eval(node, *p);
            if !same_bits(got, want) {
                failures.push(format!(
                    "{name} at {p:?}: FFI {} vs native {}",
                    bits(got),
                    bits(want)
                ));
                break;
            }
        }
    }
    for (_, handle, _) in &cases {
        unsafe { alice_sdf_free(*handle) };
    }
    assert!(
        failures.is_empty(),
        "{} C constructor(s) disagree with the node the header documents:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert_eq!(cases.len(), 35, "case count (update when adding a row)");
}

// ============================================================================
// 3. Every eval path, over the shared corpus
// ============================================================================

/// The corpus that `test_det_parity.rs` / `test_evaluator_opcode_parity.rs`
/// use (one node per compilable `SdfNode` variant) is pushed through the C
/// ABI and compared to the native evaluators to the bit.
///
/// Trees are handed to the C side the way the UE5 plugin does it — write
/// `.asdf` with the native API, `alice_sdf_load` it — because the handle
/// registry is private, so there is no other way to give the C API a node it
/// did not build itself. `tests/test_asdf_roundtrip_parity.rs` is the oracle
/// for that channel (the round trip keeps every evaluator bit-identical), so a
/// difference seen here is the FFI's.
#[test]
fn ffi_eval_paths_are_bit_identical_to_the_native_evaluators() {
    let dir = std::env::temp_dir().join(format!("alice_sdf_ffi_oracle_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let pts = sample_points(80);
    let mut failures = Vec::new();
    let mut compared = 0usize;
    let mut bytecode_unsupported = Vec::new();

    for (name, node) in corpus() {
        let path = dir.join(format!("{name}.asdf"));
        alice_sdf::save(&SdfTree::new(node.clone()), &path).expect("save");
        let c_path = CString::new(path.to_string_lossy().as_ref()).expect("path has no NUL");
        let handle = unsafe { alice_sdf_load(c_path.as_ptr()) };
        assert!(!handle.is_null(), "{name}: alice_sdf_load returned null");

        // Tree path.
        for p in &pts {
            let got = unsafe { alice_sdf_eval(handle, p.x, p.y, p.z) };
            let want = eval(&node, *p);
            if !same_bits(got, want) {
                failures.push(format!(
                    "{name} alice_sdf_eval at {p:?}: {} vs tree {}",
                    bits(got),
                    bits(want)
                ));
                break;
            }
        }

        // Compiled paths. The nodes with no bytecode law (Triangle, Bezier,
        // Terrain) make `CompiledSdf::compile` panic, which `ffi_guard` turns
        // into the null handle — pin that pairing rather than skipping it.
        let compiled_handle = unsafe { alice_sdf_compile(handle) };
        match CompiledSdf::try_compile(&node) {
            Err(_) => {
                if !compiled_handle.is_null() {
                    failures.push(format!(
                        "{name}: native try_compile rejects the node but alice_sdf_compile \
                         returned a handle"
                    ));
                }
                bytecode_unsupported.push(name);
                unsafe { alice_sdf_free(handle) };
                continue;
            }
            Ok(compiled) => {
                assert!(
                    !compiled_handle.is_null(),
                    "{name}: alice_sdf_compile returned null for a compilable node"
                );
                let want: Vec<f32> = pts.iter().map(|p| eval_compiled(&compiled, *p)).collect();

                let mut check = |path: &str, got: &[f32]| {
                    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                        if !same_bits(g, w) {
                            failures.push(format!(
                                "{name} {path} at {:?}: {} vs native compiled {}",
                                pts[i],
                                bits(g),
                                bits(w)
                            ));
                            break;
                        }
                    }
                };

                let single: Vec<f32> = pts
                    .iter()
                    .map(|p| unsafe { alice_sdf_eval_compiled(compiled_handle, p.x, p.y, p.z) })
                    .collect();
                check("eval_compiled", &single);
                let aos =
                    unsafe { batch_aos(alice_sdf_eval_compiled_batch, compiled_handle, &pts) };
                check("eval_compiled_batch", &aos);
                let aos_node = unsafe { batch_aos(alice_sdf_eval_batch, handle, &pts) };
                check("eval_batch", &aos_node);
                let soa_out = unsafe { soa(compiled_handle, &pts) };
                check("eval_soa", &soa_out);
                compared += 1;
            }
        }
        unsafe {
            alice_sdf_free_compiled(compiled_handle);
            alice_sdf_free(handle);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        failures.is_empty(),
        "{} FFI/native difference(s) over the corpus:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        compared > 100,
        "only {compared} corpus nodes reached the compiled FFI paths \
         (bytecode-unsupported: {bytecode_unsupported:?})"
    );
}

/// The batch entry points change strategy with `count` — `alice_sdf_eval_batch`
/// and `alice_sdf_eval_compiled_batch` go parallel at 256,
/// `alice_sdf_eval_soa` at 1024 (and inside each branch an 8-wide SIMD loop
/// handles all but the last `count % 8` points). Those boundaries are where a
/// lane or a remainder gets lost, and nothing was exercising them: the
/// existing unit tests call every batch function with 3 or 4 points.
#[test]
fn ffi_batch_paths_agree_at_every_threshold() {
    let node = SdfNode::sphere(0.6)
        .union(SdfNode::sphere(0.5).translate(0.8, 0.0, 0.0))
        .twist(0.9);
    let compiled = CompiledSdf::compile(&node);
    let dir = std::env::temp_dir().join(format!("alice_sdf_ffi_thresh_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("node.asdf");
    alice_sdf::save(&SdfTree::new(node), &path).expect("save");
    let c_path = CString::new(path.to_string_lossy().as_ref()).expect("path has no NUL");
    let handle = unsafe { alice_sdf_load(c_path.as_ptr()) };
    let compiled_handle = unsafe { alice_sdf_compile(handle) };

    let mut failures = Vec::new();
    for count in [1usize, 7, 8, 9, 255, 256, 257, 1023, 1024, 1025, 4097] {
        let pts = sample_points(count);
        assert_eq!(pts.len(), count);
        let want: Vec<f32> = pts.iter().map(|p| eval_compiled(&compiled, *p)).collect();
        let paths: Vec<(&str, Vec<f32>)> = unsafe {
            vec![
                ("eval_batch", batch_aos(alice_sdf_eval_batch, handle, &pts)),
                (
                    "eval_compiled_batch",
                    batch_aos(alice_sdf_eval_compiled_batch, compiled_handle, &pts),
                ),
                ("eval_soa", soa(compiled_handle, &pts)),
            ]
        };
        for (label, got) in paths {
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                if !same_bits(g, w) {
                    failures.push(format!(
                        "count={count} {label}[{i}] at {:?}: {} vs native {}",
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
        "{} threshold difference(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// How far past `count` the two SoA entry points actually write, per count.
///
/// Returns `(count, slots written past count)` measured with a canary, using
/// buffers that carry 8 slots of slack so the measurement itself stays inside
/// memory this test owns. `alice_sdf_eval_animated_batch_soa` is measured with
/// identity parameters, which is the branch that forwards to the same raw SoA
/// path.
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
fn ffi_gradient_soa_matches_the_analytic_sphere_normal() {
    let node = SdfNode::sphere(1.0);
    let compiled = CompiledSdf::compile(&node);
    let handle = unsafe { alice_sdf_sphere(1.0) };
    let compiled_handle = unsafe { alice_sdf_compile(handle) };

    let mut failures = Vec::new();
    for count in [512usize, 2048] {
        // Keep every point well away from the origin, where the normal is
        // undefined.
        let pts: Vec<Vec3> = sample_points(count * 3)
            .into_iter()
            .filter(|p| p.length() > 0.2)
            .take(count)
            .collect();
        assert_eq!(pts.len(), count, "enough points away from the origin");

        let xs: Vec<f32> = pts.iter().map(|p| p.x).collect();
        let ys: Vec<f32> = pts.iter().map(|p| p.y).collect();
        let zs: Vec<f32> = pts.iter().map(|p| p.z).collect();
        let mut nx = vec![f32::NAN; count];
        let mut ny = vec![f32::NAN; count];
        let mut nz = vec![f32::NAN; count];
        let mut dist = vec![f32::NAN; count];
        let r = unsafe {
            alice_sdf_eval_gradient_soa(
                compiled_handle,
                xs.as_ptr(),
                ys.as_ptr(),
                zs.as_ptr(),
                nx.as_mut_ptr(),
                ny.as_mut_ptr(),
                nz.as_mut_ptr(),
                dist.as_mut_ptr(),
                u32::try_from(count).expect("count fits u32"),
            )
        };
        assert_eq!(r.result, SdfResult::Ok, "count={count}");
        assert_eq!(r.count as usize, count, "count={count}");

        for (i, p) in pts.iter().enumerate() {
            let want = p.normalize();
            let got = Vec3::new(nx[i], ny[i], nz[i]);
            let err = (got - want).length();
            // `EPSILON` in src/ffi/eval.rs; see the doc comment for the bound.
            const FD_STEP: f32 = 1e-3;
            let bound = 2.0 * FD_STEP / p.length();
            if err > bound {
                failures.push(format!(
                    "count={count} normal[{i}] at {p:?}: got {got:?} vs analytic {want:?} \
                     (|err| {err:e} > bound {bound:e})"
                ));
            }
            if (got.length() - 1.0).abs() > 1e-5 {
                failures.push(format!(
                    "count={count} normal[{i}] is not a unit vector: |n| = {}",
                    got.length()
                ));
            }
            let want_d = eval_compiled(&compiled, *p);
            if !same_bits(dist[i], want_d) {
                failures.push(format!(
                    "count={count} dist[{i}] at {p:?}: {} vs native compiled {}",
                    bits(dist[i]),
                    bits(want_d)
                ));
            }
            if failures.len() > 8 {
                break;
            }
        }
    }
    unsafe {
        alice_sdf_free_compiled(compiled_handle);
        alice_sdf_free(handle);
    }
    assert!(
        failures.is_empty(),
        "{} gradient difference(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ============================================================================
// 5. Animated path
// ============================================================================

/// `alice_sdf_eval_animated_compiled` / `…_batch_soa` apply the inverse of an
/// `AnimationParams` transform to the query point instead of rebuilding the
/// tree. The existing unit tests check one translation with `> 0.0` and
/// `abs() < 0.01`; here every component is exercised and the result is held to
/// the bit against `animation::eval_animated_compiled`, plus one closed form:
/// translating a unit sphere by 5 in X puts the origin 4 units outside it.
#[test]
fn ffi_animated_paths_match_the_native_animation() {
    let node = SdfNode::sphere(1.0);
    let compiled = CompiledSdf::compile(&node);
    let handle = unsafe { alice_sdf_sphere(1.0) };
    let compiled_handle = unsafe { alice_sdf_compile(handle) };
    let pts = sample_points(64);

    let param_sets = [
        AnimationParams {
            scale: 1.0,
            ..Default::default()
        },
        AnimationParams {
            translate_x: 5.0,
            scale: 1.0,
            ..Default::default()
        },
        AnimationParams {
            translate_x: 0.3,
            translate_y: -0.7,
            translate_z: 1.1,
            rotate_x: 0.4,
            rotate_y: -0.9,
            rotate_z: 1.3,
            scale: 1.7,
            twist: 0.6,
            bend: 0.35,
        },
    ];

    let mut failures = Vec::new();
    for (k, params) in param_sets.iter().enumerate() {
        for p in &pts {
            let got =
                unsafe { alice_sdf_eval_animated_compiled(compiled_handle, params, p.x, p.y, p.z) };
            let want = alice_sdf::animation::eval_animated_compiled(&compiled, params, *p);
            if !same_bits(got, want) {
                failures.push(format!(
                    "params[{k}] single at {p:?}: {} vs native {}",
                    bits(got),
                    bits(want)
                ));
                break;
            }
        }

        let xs: Vec<f32> = pts.iter().map(|p| p.x).collect();
        let ys: Vec<f32> = pts.iter().map(|p| p.y).collect();
        let zs: Vec<f32> = pts.iter().map(|p| p.z).collect();
        let mut out = vec![f32::NAN; pts.len()];
        let r = unsafe {
            alice_sdf_eval_animated_batch_soa(
                compiled_handle,
                params,
                xs.as_ptr(),
                ys.as_ptr(),
                zs.as_ptr(),
                out.as_mut_ptr(),
                u32::try_from(pts.len()).expect("count fits u32"),
            )
        };
        assert_eq!(r.result, SdfResult::Ok, "params[{k}] batch");
        for (i, p) in pts.iter().enumerate() {
            let want = alice_sdf::animation::eval_animated_compiled(&compiled, params, *p);
            if !same_bits(out[i], want) {
                failures.push(format!(
                    "params[{k}] batch[{i}] at {p:?}: {} vs native {}",
                    bits(out[i]),
                    bits(want)
                ));
                break;
            }
        }
    }

    // Closed form: unit sphere translated to (5,0,0), queried at the origin.
    let shifted = AnimationParams {
        translate_x: 5.0,
        scale: 1.0,
        ..Default::default()
    };
    let d = unsafe {
        alice_sdf_eval_animated_compiled(compiled_handle, &raw const shifted, 0.0, 0.0, 0.0)
    };
    assert_eq!(d, 4.0, "translated unit sphere, distance from the origin");
    let d_in = unsafe {
        alice_sdf_eval_animated_compiled(compiled_handle, &raw const shifted, 5.0, 0.0, 0.0)
    };
    assert_eq!(d_in, -1.0, "translated unit sphere, its own centre");

    unsafe {
        alice_sdf_free_compiled(compiled_handle);
        alice_sdf_free(handle);
    }
    assert!(
        failures.is_empty(),
        "{} animated difference(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ============================================================================
// 6. Handle lifecycle
// ============================================================================

/// A host that frees a handle and keeps using it, or passes a node handle
/// where a compiled handle belongs, must get the documented sentinel — not a
/// stale shape and not an abort. `f32::MAX` is the sentinel for the scalar
/// entry points and `SdfResult::InvalidHandle` for the batch ones.
#[test]
fn ffi_stale_and_crossed_handles_return_sentinels() {
    unsafe {
        let node = alice_sdf_sphere(1.0);
        assert!(alice_sdf_is_valid(node));
        let compiled = alice_sdf_compile(node);

        // A node handle is not a compiled handle and vice versa: the two
        // registries use disjoint id ranges, so crossing them must be caught.
        assert_eq!(
            alice_sdf_eval_compiled(node, 0.5, 0.0, 0.0),
            f32::MAX,
            "node handle accepted as a compiled handle"
        );
        assert_eq!(
            alice_sdf_eval(compiled, 0.5, 0.0, 0.0),
            f32::MAX,
            "compiled handle accepted as a node handle"
        );

        // Cloning gives an independent handle that evaluates the same.
        let cloned = alice_sdf_clone(node);
        assert!(!cloned.is_null() && cloned != node);
        assert_eq!(
            alice_sdf_eval(cloned, 0.31, -0.47, 0.59),
            alice_sdf_eval(node, 0.31, -0.47, 0.59)
        );

        // Freeing one does not disturb the other.
        alice_sdf_free(cloned);
        assert!(!alice_sdf_is_valid(cloned), "freed handle still valid");
        assert_eq!(
            alice_sdf_eval(cloned, 0.5, 0.0, 0.0),
            f32::MAX,
            "use after free"
        );
        assert_eq!(
            alice_sdf_eval(node, 0.5, 0.0, 0.0),
            -0.5,
            "sibling survived"
        );

        // Null everywhere.
        assert_eq!(
            alice_sdf_eval(std::ptr::null_mut(), 0.0, 0.0, 0.0),
            f32::MAX
        );
        assert_eq!(
            alice_sdf_eval_compiled(std::ptr::null_mut(), 0.0, 0.0, 0.0),
            f32::MAX
        );
        assert_eq!(alice_sdf_node_count(std::ptr::null_mut()), 0);
        assert!(!alice_sdf_is_valid(std::ptr::null_mut()));

        let mut out = [f32::NAN; 4];
        let flat = [0.0f32; 12];
        let r = alice_sdf_eval_soa(
            std::ptr::null_mut(),
            flat.as_ptr(),
            flat.as_ptr(),
            flat.as_ptr(),
            out.as_mut_ptr(),
            4,
        );
        assert_eq!(r.result, SdfResult::InvalidHandle, "SoA with a null handle");
        let r = alice_sdf_eval_compiled_batch(node, flat.as_ptr(), out.as_mut_ptr(), 4);
        assert_eq!(
            r.result,
            SdfResult::InvalidHandle,
            "compiled batch with a node handle"
        );

        alice_sdf_free_compiled(compiled);
        alice_sdf_free(node);
    }
}

/// The reproduction, kept as a regression test: exactly `count` elements
/// allocated, no slack, for the counts that used to walk off the end.
///
/// A `Vec<f32>` of length 1 put the overrun right at the end of a small
/// allocation and the test binary died with SIGSEGV (signal 11), so this test
/// does not merely check values — it checks that the process survives. All
/// three entry points that reach the raw SoA path are exercised: the public
/// `eval_compiled_batch_soa_raw` itself, `alice_sdf_eval_soa` (which used to
/// overrun below its 1024 parallel threshold) and
/// `alice_sdf_eval_animated_batch_soa` with identity parameters (which had no
/// threshold, so it overran at 1025 too).
///
/// The values are held to the bit against `eval_compiled`, which also covers
/// the other half of the question: the 8-wide fast path must be unchanged for a
/// `count` that is a multiple of 8. Measured directly as well — the raw
/// function's output bits for `count` = 8 / 64 / 256 / 1024 / 4096 are
/// identical before and after the fix (FNV digest over every returned word:
/// `count=8 digest=5ce40a654075eaa6`, and likewise for the rest).
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
