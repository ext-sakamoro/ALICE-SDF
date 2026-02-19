//! JIT-SIMD Compiler for ALICE-SDF
//!
//! Generates native machine code that processes 8 points in parallel using SIMD instructions.
//! Eliminates all interpreter overhead and leverages CPU vector units directly.
//!
//! # Architecture
//!
//! This module uses Cranelift's F32X4 type (128-bit SIMD) with 2 lanes to process
//! 8 points simultaneously. On x86_64, this generates AVX/AVX2 instructions.
//! On AArch64, it generates NEON instructions.
//!
//! # Usage
//!
//! ```rust,ignore
//! use alice_sdf::prelude::*;
//! use alice_sdf::compiled::{CompiledSdf, jit_simd::JitSimd};
//! use alice_sdf::soa::SoAPoints;
//!
//! let sphere = SdfNode::sphere(1.0);
//! let compiled = CompiledSdf::compile(&sphere);
//! let jit = JitSimd::compile(&compiled).unwrap();
//!
//! // SoA buffer (alignment-aware)
//! let points = SoAPoints::from_vec3_slice(&points_vec);
//! let mut dists = vec![0.0f32; points.len()];
//!
//! // Process 8 points at a time
//! unsafe {
//!     let (px, py, pz) = points.as_ptrs();
//!     let pout = dists.as_mut_ptr();
//!
//!     for i in (0..points.padded_len()).step_by(8) {
//!         jit.eval(px.add(i), py.add(i), pz.add(i), pout.add(i));
//!     }
//! }
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(feature = "jit")]
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlags, Value};
#[cfg(feature = "jit")]
use cranelift_codegen::settings::{self, Configurable};
#[cfg(feature = "jit")]
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
#[cfg(feature = "jit")]
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(feature = "jit")]
use cranelift_module::{Linkage, Module};

#[cfg(feature = "jit")]
use super::{CompiledSdf, OpCode};

/// JIT-compiled SIMD SDF evaluator.
///
/// Processes 8 points in parallel using native SIMD instructions.
/// The generated function takes SoA-layout pointers and writes results directly.
#[cfg(feature = "jit")]
pub struct JitSimd {
    #[allow(dead_code)]
    module: JITModule,
    func_ptr: *const u8,
}

// SAFETY: JitSimd holds a read-only function pointer to JIT-compiled machine code.
// After compilation completes, the code is immutable and safe to call from any thread.
// The JITModule is held alive by the struct, preventing use-after-free.
#[cfg(feature = "jit")]
unsafe impl Send for JitSimd {}
// SAFETY: See Send impl above. The function pointer is read-only and thread-safe.
#[cfg(feature = "jit")]
unsafe impl Sync for JitSimd {}

/// Saved coordinate state for transform restoration
#[cfg(feature = "jit")]
struct CoordState {
    x: (Value, Value),
    y: (Value, Value),
    z: (Value, Value),
    scale: (Value, Value),
    opcode: OpCode,
    params: [f32; 6],
}

// ============ SIMD Helpers for Branchless Selection ============

/// SIMD branchless select: returns `if_neg` where `cond < 0`, else `if_pos`
///
/// Uses sign-bit extraction: raw_bitcast to I32X4, arithmetic shift right by 31,
/// then bitselect. Zero overhead on SSE/AVX/NEON.
#[cfg(feature = "jit")]
fn simd_select_neg(
    builder: &mut FunctionBuilder,
    cond: Value,   // F32X4 - select based on sign
    if_neg: Value, // F32X4 - selected when cond < 0
    if_pos: Value, // F32X4 - selected when cond >= 0
) -> Value {
    let mask_i32 = builder.ins().bitcast(types::I32X4, MemFlags::new(), cond);
    let sign_mask = builder.ins().sshr_imm(mask_i32, 31);
    let true_bits = builder.ins().bitcast(types::I32X4, MemFlags::new(), if_neg);
    let false_bits = builder.ins().bitcast(types::I32X4, MemFlags::new(), if_pos);
    let selected = builder.ins().bitselect(sign_mask, true_bits, false_bits);
    builder
        .ins()
        .bitcast(types::F32X4, MemFlags::new(), selected)
}

/// SIMD 2D vector length: sqrt(x² + y²)
#[cfg(feature = "jit")]
fn simd_length2(builder: &mut FunctionBuilder, x: Value, y: Value) -> Value {
    let xx = builder.ins().fmul(x, x);
    let yy = builder.ins().fmul(y, y);
    let sum = builder.ins().fadd(xx, yy);
    builder.ins().sqrt(sum)
}

/// SIMD 3D vector length: sqrt(x² + y² + z²)
#[cfg(feature = "jit")]
fn simd_length3(builder: &mut FunctionBuilder, x: Value, y: Value, z: Value) -> Value {
    let xx = builder.ins().fmul(x, x);
    let yy = builder.ins().fmul(y, y);
    let zz = builder.ins().fmul(z, z);
    let xy = builder.ins().fadd(xx, yy);
    let sum = builder.ins().fadd(xy, zz);
    builder.ins().sqrt(sum)
}

/// SIMD sin/cos Taylor-series approximation
///
/// sin(x) ≈ x - x³/6 + x⁵/120
/// cos(x) ≈ 1 - x²/2 + x⁴/24
///
/// Returns (cos, sin)
#[cfg(feature = "jit")]
fn simd_sincos_approx_js(
    builder: &mut FunctionBuilder,
    angle: Value,
    simd_type: types::Type,
) -> (Value, Value) {
    let one_s = builder.ins().f32const(1.0);
    let half_s = builder.ins().f32const(0.5);
    let sixth_s = builder.ins().f32const(1.0 / 6.0);
    let tf_s = builder.ins().f32const(1.0 / 24.0);
    let ot_s = builder.ins().f32const(1.0 / 120.0);
    let one_v = builder.ins().splat(simd_type, one_s);
    let half_v = builder.ins().splat(simd_type, half_s);
    let sixth_v = builder.ins().splat(simd_type, sixth_s);
    let tf_v = builder.ins().splat(simd_type, tf_s);
    let ot_v = builder.ins().splat(simd_type, ot_s);

    let x2 = builder.ins().fmul(angle, angle);
    let x3 = builder.ins().fmul(x2, angle);
    let x4 = builder.ins().fmul(x2, x2);
    let x5 = builder.ins().fmul(x4, angle);

    // sin(x) = x - x³/6 + x⁵/120
    let st1 = builder.ins().fmul(x3, sixth_v);
    let st2 = builder.ins().fmul(x5, ot_v);
    let sin_r = builder.ins().fsub(angle, st1);
    let sin_r = builder.ins().fadd(sin_r, st2);

    // cos(x) = 1 - x²/2 + x⁴/24
    let ct1 = builder.ins().fmul(x2, half_v);
    let ct2 = builder.ins().fmul(x4, tf_v);
    let cos_r = builder.ins().fsub(one_v, ct1);
    let cos_r = builder.ins().fadd(cos_r, ct2);

    (cos_r, sin_r)
}

/// Emit SIMD stairs_min for one F32X4 lane (constants baked in)
#[cfg(feature = "jit")]
fn emit_jitsimd_stairs_min(
    builder: &mut FunctionBuilder,
    simd_type: types::Type,
    a: Value,
    b: Value,
    r: f32,
    n: f32,
) -> Value {
    let half_s = builder.ins().f32const(0.5);
    let half = builder.ins().splat(simd_type, half_s);
    let s_s = builder.ins().f32const(std::f32::consts::FRAC_1_SQRT_2);
    let s_v = builder.ins().splat(simd_type, s_s);
    let s2_s = builder.ins().f32const(std::f32::consts::SQRT_2);
    let s2_v = builder.ins().splat(simd_type, s2_s);

    let rn_f = r / n;
    let rn_s = builder.ins().f32const(rn_f);
    let rn_v = builder.ins().splat(simd_type, rn_s);

    let off_f = (r - rn_f) * 0.5 * std::f32::consts::SQRT_2;
    let off_s = builder.ins().f32const(off_f);
    let off_v = builder.ins().splat(simd_type, off_s);

    let step_f = r * std::f32::consts::SQRT_2 / n;
    let step_s = builder.ins().f32const(step_f);
    let step_v = builder.ins().splat(simd_type, step_s);

    // px = (b - a) * s - off
    let b_minus_a = builder.ins().fsub(b, a);
    let px_rot = builder.ins().fmul(b_minus_a, s_v);
    let px = builder.ins().fsub(px_rot, off_v);
    // py = (a + b) * s - off
    let a_plus_b = builder.ins().fadd(a, b);
    let py_rot = builder.ins().fmul(a_plus_b, s_v);
    let py = builder.ins().fsub(py_rot, off_v);

    // pMod1
    let half_s2 = builder.ins().fmul(half, s2_v);
    let half_s2_rn = builder.ins().fmul(half_s2, rn_v);
    let px2 = builder.ins().fadd(px, half_s2_rn);
    let step_half = builder.ins().fmul(step_v, half);
    let t = builder.ins().fadd(px2, step_half);

    let t_div_step = builder.ins().fdiv(t, step_v);
    let t_floor = builder.ins().floor(t_div_step);
    let t_mod = builder.ins().fmul(step_v, t_floor);
    let t_sub = builder.ins().fsub(t, t_mod);
    let px3 = builder.ins().fsub(t_sub, step_half);

    let min_ab = builder.ins().fmin(a, b);
    let d2 = builder.ins().fmin(min_ab, py);

    let px3_plus_py = builder.ins().fadd(px3, py);
    let npx = builder.ins().fmul(px3_plus_py, s_v);
    let py_minus_px3 = builder.ins().fsub(py, px3);
    let npy = builder.ins().fmul(py_minus_px3, s_v);

    let edge = builder.ins().fmul(half, rn_v);
    let npx_e = builder.ins().fsub(npx, edge);
    let npy_e = builder.ins().fsub(npy, edge);
    let vmax = builder.ins().fmax(npx_e, npy_e);
    builder.ins().fmin(d2, vmax)
}

#[cfg(feature = "jit")]
impl JitSimd {
    /// Compile the SDF bytecode into a native SIMD function.
    ///
    /// The generated function signature is:
    /// `unsafe extern "C" fn(ptr_x: *const f32, ptr_y: *const f32, ptr_z: *const f32, ptr_out: *mut f32)`
    ///
    /// # Arguments
    /// * `sdf` - The compiled SDF bytecode to JIT compile
    ///
    /// # Returns
    /// A JitSimd instance containing the compiled native code
    pub fn compile(sdf: &CompiledSdf) -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| e.to_string())?;
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|e| e.to_string())?;

        // Enable SIMD on x86_64
        if cfg!(target_arch = "x86_64") {
            let _ = flag_builder.set("enable_simd", "true");
        }

        let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| e.to_string())?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);
        let mut ctx = module.make_context();
        let mut func_ctx = FunctionBuilderContext::new();

        // Signature: (ptr_x, ptr_y, ptr_z, ptr_out) -> void
        let ptr_type = module.target_config().pointer_type();
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // px
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // py
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // pz
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // out

        let func_id = module
            .declare_function("eval_simd_8", Linkage::Export, &ctx.func.signature)
            .map_err(|e| e.to_string())?;

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);

            // Fetch arguments
            let px = builder.block_params(entry_block)[0];
            let py = builder.block_params(entry_block)[1];
            let pz = builder.block_params(entry_block)[2];
            let pout = builder.block_params(entry_block)[3];

            // Define SIMD type (F32X4 is safest cross-platform baseline)
            let simd_type = types::F32X4;
            let mem_flags = MemFlags::new();

            // Load 8 floats as two F32X4 vectors
            let x0 = builder.ins().load(simd_type, mem_flags, px, 0);
            let x1 = builder.ins().load(simd_type, mem_flags, px, 16);
            let y0 = builder.ins().load(simd_type, mem_flags, py, 0);
            let y1 = builder.ins().load(simd_type, mem_flags, py, 16);
            let z0 = builder.ins().load(simd_type, mem_flags, pz, 0);
            let z1 = builder.ins().load(simd_type, mem_flags, pz, 16);

            // Create zero and one constants
            let zero_s = builder.ins().f32const(0.0);
            let zero = builder.ins().splat(simd_type, zero_s);
            let one_s = builder.ins().f32const(1.0);
            let one = builder.ins().splat(simd_type, one_s);

            // Current coordinates (mutable during transforms)
            let mut curr_x = (x0, x1);
            let mut curr_y = (y0, y1);
            let mut curr_z = (z0, z1);
            let mut curr_scale = (one, one);

            // Compilation state stacks
            let mut value_stack: Vec<(Value, Value)> = Vec::with_capacity(64);
            let mut coord_stack: Vec<CoordState> = Vec::with_capacity(32);

            for inst in &sdf.instructions {
                match inst.opcode {
                    OpCode::Sphere => {
                        let radius = inst.params[0];
                        let r_s = builder.ins().f32const(radius);
                        let r0 = builder.ins().splat(simd_type, r_s);
                        let r1 = builder.ins().splat(simd_type, r_s);

                        // length(p) - r
                        // Lane 0
                        let xx0 = builder.ins().fmul(curr_x.0, curr_x.0);
                        let yy0 = builder.ins().fmul(curr_y.0, curr_y.0);
                        let zz0 = builder.ins().fmul(curr_z.0, curr_z.0);
                        let xy0 = builder.ins().fadd(xx0, yy0);
                        let sum0 = builder.ins().fadd(xy0, zz0);
                        let len0 = builder.ins().sqrt(sum0);
                        let d0 = builder.ins().fsub(len0, r0);

                        // Lane 1
                        let xx1 = builder.ins().fmul(curr_x.1, curr_x.1);
                        let yy1 = builder.ins().fmul(curr_y.1, curr_y.1);
                        let zz1 = builder.ins().fmul(curr_z.1, curr_z.1);
                        let xy1 = builder.ins().fadd(xx1, yy1);
                        let sum1 = builder.ins().fadd(xy1, zz1);
                        let len1 = builder.ins().sqrt(sum1);
                        let d1 = builder.ins().fsub(len1, r1);

                        // Apply scale correction
                        let ds0 = builder.ins().fmul(d0, curr_scale.0);
                        let ds1 = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((ds0, ds1));
                    }

                    OpCode::Box3d => {
                        let bx_s = builder.ins().f32const(inst.params[0]);
                        let by_s = builder.ins().f32const(inst.params[1]);
                        let bz_s = builder.ins().f32const(inst.params[2]);
                        let bx = builder.ins().splat(simd_type, bx_s);
                        let by = builder.ins().splat(simd_type, by_s);
                        let bz = builder.ins().splat(simd_type, bz_s);

                        // Lane 0: q = abs(p) - b
                        let ax0 = builder.ins().fabs(curr_x.0);
                        let ay0 = builder.ins().fabs(curr_y.0);
                        let az0 = builder.ins().fabs(curr_z.0);
                        let qx0 = builder.ins().fsub(ax0, bx);
                        let qy0 = builder.ins().fsub(ay0, by);
                        let qz0 = builder.ins().fsub(az0, bz);

                        // max(q, 0)
                        let mx0 = builder.ins().fmax(qx0, zero);
                        let my0 = builder.ins().fmax(qy0, zero);
                        let mz0 = builder.ins().fmax(qz0, zero);

                        // outside = length(max(q, 0))
                        let mxx0 = builder.ins().fmul(mx0, mx0);
                        let myy0 = builder.ins().fmul(my0, my0);
                        let mzz0 = builder.ins().fmul(mz0, mz0);
                        let mxy0 = builder.ins().fadd(mxx0, myy0);
                        let len_sq0 = builder.ins().fadd(mxy0, mzz0);
                        let outside0 = builder.ins().sqrt(len_sq0);

                        // inside = min(max(q.x, max(q.y, q.z)), 0)
                        let max_yz0 = builder.ins().fmax(qy0, qz0);
                        let max_xyz0 = builder.ins().fmax(qx0, max_yz0);
                        let inside0 = builder.ins().fmin(max_xyz0, zero);
                        let d0 = builder.ins().fadd(outside0, inside0);

                        // Lane 1
                        let ax1 = builder.ins().fabs(curr_x.1);
                        let ay1 = builder.ins().fabs(curr_y.1);
                        let az1 = builder.ins().fabs(curr_z.1);
                        let qx1 = builder.ins().fsub(ax1, bx);
                        let qy1 = builder.ins().fsub(ay1, by);
                        let qz1 = builder.ins().fsub(az1, bz);

                        let mx1 = builder.ins().fmax(qx1, zero);
                        let my1 = builder.ins().fmax(qy1, zero);
                        let mz1 = builder.ins().fmax(qz1, zero);

                        let mxx1 = builder.ins().fmul(mx1, mx1);
                        let myy1 = builder.ins().fmul(my1, my1);
                        let mzz1 = builder.ins().fmul(mz1, mz1);
                        let mxy1 = builder.ins().fadd(mxx1, myy1);
                        let len_sq1 = builder.ins().fadd(mxy1, mzz1);
                        let outside1 = builder.ins().sqrt(len_sq1);

                        let max_yz1 = builder.ins().fmax(qy1, qz1);
                        let max_xyz1 = builder.ins().fmax(qx1, max_yz1);
                        let inside1 = builder.ins().fmin(max_xyz1, zero);
                        let d1 = builder.ins().fadd(outside1, inside1);

                        // Scale correction
                        let ds0 = builder.ins().fmul(d0, curr_scale.0);
                        let ds1 = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((ds0, ds1));
                    }

                    OpCode::Cylinder => {
                        let radius = inst.params[0];
                        let half_height = inst.params[1];

                        let r_s = builder.ins().f32const(radius);
                        let h_s = builder.ins().f32const(half_height);
                        let r = builder.ins().splat(simd_type, r_s);
                        let h = builder.ins().splat(simd_type, h_s);

                        // Lane 0: d = vec2(length(p.xz) - r, abs(p.y) - h)
                        let xx0 = builder.ins().fmul(curr_x.0, curr_x.0);
                        let zz0 = builder.ins().fmul(curr_z.0, curr_z.0);
                        let xz0 = builder.ins().fadd(xx0, zz0);
                        let len_xz0 = builder.ins().sqrt(xz0);
                        let dx0 = builder.ins().fsub(len_xz0, r);
                        let ay0 = builder.ins().fabs(curr_y.0);
                        let dy0 = builder.ins().fsub(ay0, h);

                        // length(max(d, 0)) + min(max(d.x, d.y), 0)
                        let mdx0 = builder.ins().fmax(dx0, zero);
                        let mdy0 = builder.ins().fmax(dy0, zero);
                        let mdxx0 = builder.ins().fmul(mdx0, mdx0);
                        let mdyy0 = builder.ins().fmul(mdy0, mdy0);
                        let len_sq0 = builder.ins().fadd(mdxx0, mdyy0);
                        let outside0 = builder.ins().sqrt(len_sq0);
                        let max_d0 = builder.ins().fmax(dx0, dy0);
                        let inside0 = builder.ins().fmin(max_d0, zero);
                        let d0 = builder.ins().fadd(outside0, inside0);

                        // Lane 1
                        let xx1 = builder.ins().fmul(curr_x.1, curr_x.1);
                        let zz1 = builder.ins().fmul(curr_z.1, curr_z.1);
                        let xz1 = builder.ins().fadd(xx1, zz1);
                        let len_xz1 = builder.ins().sqrt(xz1);
                        let dx1 = builder.ins().fsub(len_xz1, r);
                        let ay1 = builder.ins().fabs(curr_y.1);
                        let dy1 = builder.ins().fsub(ay1, h);

                        let mdx1 = builder.ins().fmax(dx1, zero);
                        let mdy1 = builder.ins().fmax(dy1, zero);
                        let mdxx1 = builder.ins().fmul(mdx1, mdx1);
                        let mdyy1 = builder.ins().fmul(mdy1, mdy1);
                        let len_sq1 = builder.ins().fadd(mdxx1, mdyy1);
                        let outside1 = builder.ins().sqrt(len_sq1);
                        let max_d1 = builder.ins().fmax(dx1, dy1);
                        let inside1 = builder.ins().fmin(max_d1, zero);
                        let d1 = builder.ins().fadd(outside1, inside1);

                        let ds0 = builder.ins().fmul(d0, curr_scale.0);
                        let ds1 = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((ds0, ds1));
                    }

                    OpCode::Plane => {
                        // Plane: dot(p, normal) - distance
                        let nx = inst.params[0];
                        let ny = inst.params[1];
                        let nz = inst.params[2];
                        let dist = inst.params[3];

                        let nx_s = builder.ins().f32const(nx);
                        let ny_s = builder.ins().f32const(ny);
                        let nz_s = builder.ins().f32const(nz);
                        let d_s = builder.ins().f32const(dist);

                        let nx_v = builder.ins().splat(simd_type, nx_s);
                        let ny_v = builder.ins().splat(simd_type, ny_s);
                        let nz_v = builder.ins().splat(simd_type, nz_s);
                        let d_v = builder.ins().splat(simd_type, d_s);

                        // Lane 0
                        let dotx0 = builder.ins().fmul(curr_x.0, nx_v);
                        let doty0 = builder.ins().fmul(curr_y.0, ny_v);
                        let dotz0 = builder.ins().fmul(curr_z.0, nz_v);
                        let dotxy0 = builder.ins().fadd(dotx0, doty0);
                        let dot0 = builder.ins().fadd(dotxy0, dotz0);
                        let d0 = builder.ins().fsub(dot0, d_v);

                        // Lane 1
                        let dotx1 = builder.ins().fmul(curr_x.1, nx_v);
                        let doty1 = builder.ins().fmul(curr_y.1, ny_v);
                        let dotz1 = builder.ins().fmul(curr_z.1, nz_v);
                        let dotxy1 = builder.ins().fadd(dotx1, doty1);
                        let dot1 = builder.ins().fadd(dotxy1, dotz1);
                        let d1 = builder.ins().fsub(dot1, d_v);

                        let ds0 = builder.ins().fmul(d0, curr_scale.0);
                        let ds1 = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((ds0, ds1));
                    }

                    OpCode::Torus => {
                        let major = inst.params[0];
                        let minor = inst.params[1];

                        let major_s = builder.ins().f32const(major);
                        let minor_s = builder.ins().f32const(minor);
                        let major_v = builder.ins().splat(simd_type, major_s);
                        let minor_v = builder.ins().splat(simd_type, minor_s);

                        // Lane 0: q = vec2(length(p.xz) - major, p.y)
                        let xx0 = builder.ins().fmul(curr_x.0, curr_x.0);
                        let zz0 = builder.ins().fmul(curr_z.0, curr_z.0);
                        let xz0 = builder.ins().fadd(xx0, zz0);
                        let len_xz0 = builder.ins().sqrt(xz0);
                        let qx0 = builder.ins().fsub(len_xz0, major_v);
                        // length(q) - minor
                        let qxx0 = builder.ins().fmul(qx0, qx0);
                        let qyy0 = builder.ins().fmul(curr_y.0, curr_y.0);
                        let q_len_sq0 = builder.ins().fadd(qxx0, qyy0);
                        let q_len0 = builder.ins().sqrt(q_len_sq0);
                        let d0 = builder.ins().fsub(q_len0, minor_v);

                        // Lane 1
                        let xx1 = builder.ins().fmul(curr_x.1, curr_x.1);
                        let zz1 = builder.ins().fmul(curr_z.1, curr_z.1);
                        let xz1 = builder.ins().fadd(xx1, zz1);
                        let len_xz1 = builder.ins().sqrt(xz1);
                        let qx1 = builder.ins().fsub(len_xz1, major_v);
                        let qxx1 = builder.ins().fmul(qx1, qx1);
                        let qyy1 = builder.ins().fmul(curr_y.1, curr_y.1);
                        let q_len_sq1 = builder.ins().fadd(qxx1, qyy1);
                        let q_len1 = builder.ins().sqrt(q_len_sq1);
                        let d1 = builder.ins().fsub(q_len1, minor_v);

                        let ds0 = builder.ins().fmul(d0, curr_scale.0);
                        let ds1 = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((ds0, ds1));
                    }

                    OpCode::Ellipsoid => {
                        let inv_rx_s = builder.ins().f32const(1.0 / inst.params[0]);
                        let inv_ry_s = builder.ins().f32const(1.0 / inst.params[1]);
                        let inv_rz_s = builder.ins().f32const(1.0 / inst.params[2]);
                        let inv_rx = builder.ins().splat(simd_type, inv_rx_s);
                        let inv_ry = builder.ins().splat(simd_type, inv_ry_s);
                        let inv_rz = builder.ins().splat(simd_type, inv_rz_s);

                        let inv_rx2_s = builder
                            .ins()
                            .f32const(1.0 / (inst.params[0] * inst.params[0]));
                        let inv_ry2_s = builder
                            .ins()
                            .f32const(1.0 / (inst.params[1] * inst.params[1]));
                        let inv_rz2_s = builder
                            .ins()
                            .f32const(1.0 / (inst.params[2] * inst.params[2]));
                        let inv_rx2 = builder.ins().splat(simd_type, inv_rx2_s);
                        let inv_ry2 = builder.ins().splat(simd_type, inv_ry2_s);
                        let inv_rz2 = builder.ins().splat(simd_type, inv_rz2_s);

                        let eps_s = builder.ins().f32const(1e-10);
                        let eps = builder.ins().splat(simd_type, eps_s);

                        // Lane 0: k0 = length(p / radii)
                        let px0 = builder.ins().fmul(curr_x.0, inv_rx);
                        let py0 = builder.ins().fmul(curr_y.0, inv_ry);
                        let pz0 = builder.ins().fmul(curr_z.0, inv_rz);
                        let xx0 = builder.ins().fmul(px0, px0);
                        let yy0 = builder.ins().fmul(py0, py0);
                        let zz0 = builder.ins().fmul(pz0, pz0);
                        let xy0 = builder.ins().fadd(xx0, yy0);
                        let sum0 = builder.ins().fadd(xy0, zz0);
                        let k0_0 = builder.ins().sqrt(sum0);

                        // k1 = length(p / (radii*radii))
                        let qx0 = builder.ins().fmul(curr_x.0, inv_rx2);
                        let qy0 = builder.ins().fmul(curr_y.0, inv_ry2);
                        let qz0 = builder.ins().fmul(curr_z.0, inv_rz2);
                        let qxx0 = builder.ins().fmul(qx0, qx0);
                        let qyy0 = builder.ins().fmul(qy0, qy0);
                        let qzz0 = builder.ins().fmul(qz0, qz0);
                        let qxy0 = builder.ins().fadd(qxx0, qyy0);
                        let qsum0 = builder.ins().fadd(qxy0, qzz0);
                        let k1_0 = builder.ins().sqrt(qsum0);

                        // d = k0 * (k0 - 1) / (k1 + eps)
                        let k1_safe0 = builder.ins().fadd(k1_0, eps);
                        let k0_m1_0 = builder.ins().fsub(k0_0, one);
                        let num0 = builder.ins().fmul(k0_0, k0_m1_0);
                        let d0 = builder.ins().fdiv(num0, k1_safe0);

                        // Lane 1
                        let px1 = builder.ins().fmul(curr_x.1, inv_rx);
                        let py1 = builder.ins().fmul(curr_y.1, inv_ry);
                        let pz1 = builder.ins().fmul(curr_z.1, inv_rz);
                        let xx1 = builder.ins().fmul(px1, px1);
                        let yy1 = builder.ins().fmul(py1, py1);
                        let zz1 = builder.ins().fmul(pz1, pz1);
                        let xy1 = builder.ins().fadd(xx1, yy1);
                        let sum1 = builder.ins().fadd(xy1, zz1);
                        let k0_1 = builder.ins().sqrt(sum1);

                        let qx1 = builder.ins().fmul(curr_x.1, inv_rx2);
                        let qy1 = builder.ins().fmul(curr_y.1, inv_ry2);
                        let qz1 = builder.ins().fmul(curr_z.1, inv_rz2);
                        let qxx1 = builder.ins().fmul(qx1, qx1);
                        let qyy1 = builder.ins().fmul(qy1, qy1);
                        let qzz1 = builder.ins().fmul(qz1, qz1);
                        let qxy1 = builder.ins().fadd(qxx1, qyy1);
                        let qsum1 = builder.ins().fadd(qxy1, qzz1);
                        let k1_1 = builder.ins().sqrt(qsum1);

                        let k1_safe1 = builder.ins().fadd(k1_1, eps);
                        let k0_m1_1 = builder.ins().fsub(k0_1, one);
                        let num1 = builder.ins().fmul(k0_1, k0_m1_1);
                        let d1 = builder.ins().fdiv(num1, k1_safe1);

                        let ds0 = builder.ins().fmul(d0, curr_scale.0);
                        let ds1 = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((ds0, ds1));
                    }

                    OpCode::Link => {
                        let hl_s = builder.ins().f32const(inst.params[0]);
                        let r1_s = builder.ins().f32const(inst.params[1]);
                        let r2_s = builder.ins().f32const(inst.params[2]);
                        let hl = builder.ins().splat(simd_type, hl_s);
                        let r1_v = builder.ins().splat(simd_type, r1_s);
                        let r2_v = builder.ins().splat(simd_type, r2_s);

                        // Lane 0: qy = max(abs(y) - half_length, 0)
                        let abs_y0 = builder.ins().fabs(curr_y.0);
                        let y_sub0 = builder.ins().fsub(abs_y0, hl);
                        let qy0 = builder.ins().fmax(y_sub0, zero);
                        // xy_len = sqrt(x² + qy²) - r1
                        let xx0 = builder.ins().fmul(curr_x.0, curr_x.0);
                        let qyy0 = builder.ins().fmul(qy0, qy0);
                        let xy_sq0 = builder.ins().fadd(xx0, qyy0);
                        let xy_len0 = builder.ins().sqrt(xy_sq0);
                        let xy_sub0 = builder.ins().fsub(xy_len0, r1_v);
                        // d = sqrt(xy_sub² + z²) - r2
                        let xys0 = builder.ins().fmul(xy_sub0, xy_sub0);
                        let zz0 = builder.ins().fmul(curr_z.0, curr_z.0);
                        let d_sq0 = builder.ins().fadd(xys0, zz0);
                        let d_len0 = builder.ins().sqrt(d_sq0);
                        let d0 = builder.ins().fsub(d_len0, r2_v);

                        // Lane 1
                        let abs_y1 = builder.ins().fabs(curr_y.1);
                        let y_sub1 = builder.ins().fsub(abs_y1, hl);
                        let qy1 = builder.ins().fmax(y_sub1, zero);
                        let xx1 = builder.ins().fmul(curr_x.1, curr_x.1);
                        let qyy1 = builder.ins().fmul(qy1, qy1);
                        let xy_sq1 = builder.ins().fadd(xx1, qyy1);
                        let xy_len1 = builder.ins().sqrt(xy_sq1);
                        let xy_sub1 = builder.ins().fsub(xy_len1, r1_v);
                        let xys1 = builder.ins().fmul(xy_sub1, xy_sub1);
                        let zz1 = builder.ins().fmul(curr_z.1, curr_z.1);
                        let d_sq1 = builder.ins().fadd(xys1, zz1);
                        let d_len1 = builder.ins().sqrt(d_sq1);
                        let d1 = builder.ins().fsub(d_len1, r2_v);

                        let ds0 = builder.ins().fmul(d0, curr_scale.0);
                        let ds1 = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((ds0, ds1));
                    }

                    OpCode::Capsule => {
                        // params: [ax, ay, az, bx, by, bz], radius in skip_offset
                        let ax_s = builder.ins().f32const(inst.params[0]);
                        let ay_s = builder.ins().f32const(inst.params[1]);
                        let az_s = builder.ins().f32const(inst.params[2]);
                        // b endpoint used only for ba computation below
                        let radius = inst.get_capsule_radius();
                        let r_s = builder.ins().f32const(radius);

                        let ax = builder.ins().splat(simd_type, ax_s);
                        let ay = builder.ins().splat(simd_type, ay_s);
                        let az = builder.ins().splat(simd_type, az_s);
                        let r_v = builder.ins().splat(simd_type, r_s);

                        // Pre-compute ba and inv_ba_dot (Division Exorcism)
                        let ba_x = inst.params[3] - inst.params[0];
                        let ba_y = inst.params[4] - inst.params[1];
                        let ba_z = inst.params[5] - inst.params[2];
                        let ba_dot_val = ba_x * ba_x + ba_y * ba_y + ba_z * ba_z;
                        let inv_ba_dot = if ba_dot_val.abs() < 1e-10 {
                            1.0
                        } else {
                            1.0 / ba_dot_val
                        };

                        let bax_s = builder.ins().f32const(ba_x);
                        let bay_s = builder.ins().f32const(ba_y);
                        let baz_s = builder.ins().f32const(ba_z);
                        let inv_bd_s = builder.ins().f32const(inv_ba_dot);
                        let bax = builder.ins().splat(simd_type, bax_s);
                        let bay = builder.ins().splat(simd_type, bay_s);
                        let baz = builder.ins().splat(simd_type, baz_s);
                        let inv_bd = builder.ins().splat(simd_type, inv_bd_s);

                        // Lane 0: pa = p - a, h = clamp(dot(pa,ba) * inv_ba_dot, 0, 1)
                        let pax0 = builder.ins().fsub(curr_x.0, ax);
                        let pay0 = builder.ins().fsub(curr_y.0, ay);
                        let paz0 = builder.ins().fsub(curr_z.0, az);
                        let dot_x0 = builder.ins().fmul(pax0, bax);
                        let dot_y0 = builder.ins().fmul(pay0, bay);
                        let dot_z0 = builder.ins().fmul(paz0, baz);
                        let dot_xy0 = builder.ins().fadd(dot_x0, dot_y0);
                        let dot0 = builder.ins().fadd(dot_xy0, dot_z0);
                        let h_raw0 = builder.ins().fmul(dot0, inv_bd);
                        let h_min0 = builder.ins().fmin(h_raw0, one);
                        let h0 = builder.ins().fmax(h_min0, zero);
                        // d = length(pa - ba*h) - radius
                        let bhx0 = builder.ins().fmul(bax, h0);
                        let bhy0 = builder.ins().fmul(bay, h0);
                        let bhz0 = builder.ins().fmul(baz, h0);
                        let dx0 = builder.ins().fsub(pax0, bhx0);
                        let dy0 = builder.ins().fsub(pay0, bhy0);
                        let dz0 = builder.ins().fsub(paz0, bhz0);
                        let len0 = simd_length3(&mut builder, dx0, dy0, dz0);
                        let d0 = builder.ins().fsub(len0, r_v);

                        // Lane 1
                        let pax1 = builder.ins().fsub(curr_x.1, ax);
                        let pay1 = builder.ins().fsub(curr_y.1, ay);
                        let paz1 = builder.ins().fsub(curr_z.1, az);
                        let dot_x1 = builder.ins().fmul(pax1, bax);
                        let dot_y1 = builder.ins().fmul(pay1, bay);
                        let dot_z1 = builder.ins().fmul(paz1, baz);
                        let dot_xy1 = builder.ins().fadd(dot_x1, dot_y1);
                        let dot1 = builder.ins().fadd(dot_xy1, dot_z1);
                        let h_raw1 = builder.ins().fmul(dot1, inv_bd);
                        let h_min1 = builder.ins().fmin(h_raw1, one);
                        let h1 = builder.ins().fmax(h_min1, zero);
                        let bhx1 = builder.ins().fmul(bax, h1);
                        let bhy1 = builder.ins().fmul(bay, h1);
                        let bhz1 = builder.ins().fmul(baz, h1);
                        let dx1 = builder.ins().fsub(pax1, bhx1);
                        let dy1 = builder.ins().fsub(pay1, bhy1);
                        let dz1 = builder.ins().fsub(paz1, bhz1);
                        let len1 = simd_length3(&mut builder, dx1, dy1, dz1);
                        let d1 = builder.ins().fsub(len1, r_v);

                        let ds0 = builder.ins().fmul(d0, curr_scale.0);
                        let ds1 = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((ds0, ds1));
                    }

                    OpCode::Cone => {
                        let radius_val = inst.params[0];
                        let half_height = inst.params[1];

                        let r_s = builder.ins().f32const(radius_val);
                        let h_s = builder.ins().f32const(half_height);
                        let r = builder.ins().splat(simd_type, r_s);
                        let h = builder.ins().splat(simd_type, h_s);

                        // Pre-compute k2 and inv_k2_dot (Division Exorcism)
                        let k2x_val = -radius_val;
                        let k2y_val = 2.0 * half_height;
                        let k2_dot_val = k2x_val * k2x_val + k2y_val * k2y_val;
                        let inv_k2d_val = if k2_dot_val.abs() < 1e-10 {
                            1.0
                        } else {
                            1.0 / k2_dot_val
                        };

                        let k2x_s = builder.ins().f32const(k2x_val);
                        let k2y_s = builder.ins().f32const(k2y_val);
                        let inv_k2d_s = builder.ins().f32const(inv_k2d_val);
                        let k2x = builder.ins().splat(simd_type, k2x_s);
                        let k2y = builder.ins().splat(simd_type, k2y_s);
                        let inv_k2d = builder.ins().splat(simd_type, inv_k2d_s);
                        let neg_one_s = builder.ins().f32const(-1.0);
                        let neg_one_v = builder.ins().splat(simd_type, neg_one_s);

                        // Lane 0: q_x = length(p.xz), q_y = p.y
                        let q_x0 = simd_length2(&mut builder, curr_x.0, curr_z.0);
                        let q_y0 = curr_y.0;

                        // ca_r = q_y < 0 ? radius : 0 (branchless)
                        let ca_r0 = simd_select_neg(&mut builder, q_y0, r, zero);
                        let min_q_ca0 = builder.ins().fmin(q_x0, ca_r0);
                        let ca_x0 = builder.ins().fsub(q_x0, min_q_ca0);
                        let abs_qy0 = builder.ins().fabs(q_y0);
                        let ca_y0 = builder.ins().fsub(abs_qy0, h);

                        // t = clamp(dot(-q_x, h-q_y, k2) * inv_k2_dot, 0, 1)
                        let neg_qx0 = builder.ins().fneg(q_x0);
                        let diff_y0 = builder.ins().fsub(h, q_y0);
                        let dy_k2y0 = builder.ins().fmul(diff_y0, k2y);
                        let nqx_k2x0 = builder.ins().fmul(neg_qx0, k2x);
                        let dot0 = builder.ins().fadd(nqx_k2x0, dy_k2y0);
                        let t_raw0 = builder.ins().fmul(dot0, inv_k2d);
                        let t_min0 = builder.ins().fmin(t_raw0, one);
                        let t0 = builder.ins().fmax(zero, t_min0);

                        // cb
                        let k2x_t0 = builder.ins().fmul(k2x, t0);
                        let cb_x0 = builder.ins().fadd(q_x0, k2x_t0);
                        let qy_h0 = builder.ins().fsub(q_y0, h);
                        let k2y_t0 = builder.ins().fmul(k2y, t0);
                        let cb_y0 = builder.ins().fadd(qy_h0, k2y_t0);

                        // s = (cb_x < 0 && ca_y < 0) ? -1 : 1
                        // Both < 0 iff max(cb_x, ca_y) < 0
                        let both_neg_cond0 = builder.ins().fmax(cb_x0, ca_y0);
                        let s0 = simd_select_neg(&mut builder, both_neg_cond0, neg_one_v, one);

                        // d2 = min(ca², cb²)
                        let ca_xx0 = builder.ins().fmul(ca_x0, ca_x0);
                        let ca_yy0 = builder.ins().fmul(ca_y0, ca_y0);
                        let ca_sq0 = builder.ins().fadd(ca_xx0, ca_yy0);
                        let cb_xx0 = builder.ins().fmul(cb_x0, cb_x0);
                        let cb_yy0 = builder.ins().fmul(cb_y0, cb_y0);
                        let cb_sq0 = builder.ins().fadd(cb_xx0, cb_yy0);
                        let d2_0 = builder.ins().fmin(ca_sq0, cb_sq0);
                        let dist0 = builder.ins().sqrt(d2_0);
                        let d0 = builder.ins().fmul(s0, dist0);

                        // Lane 1
                        let q_x1 = simd_length2(&mut builder, curr_x.1, curr_z.1);
                        let q_y1 = curr_y.1;
                        let ca_r1 = simd_select_neg(&mut builder, q_y1, r, zero);
                        let min_q_ca1 = builder.ins().fmin(q_x1, ca_r1);
                        let ca_x1 = builder.ins().fsub(q_x1, min_q_ca1);
                        let abs_qy1 = builder.ins().fabs(q_y1);
                        let ca_y1 = builder.ins().fsub(abs_qy1, h);
                        let neg_qx1 = builder.ins().fneg(q_x1);
                        let diff_y1 = builder.ins().fsub(h, q_y1);
                        let dy_k2y1 = builder.ins().fmul(diff_y1, k2y);
                        let nqx_k2x1 = builder.ins().fmul(neg_qx1, k2x);
                        let dot1 = builder.ins().fadd(nqx_k2x1, dy_k2y1);
                        let t_raw1 = builder.ins().fmul(dot1, inv_k2d);
                        let t_min1 = builder.ins().fmin(t_raw1, one);
                        let t1 = builder.ins().fmax(zero, t_min1);
                        let k2x_t1 = builder.ins().fmul(k2x, t1);
                        let cb_x1 = builder.ins().fadd(q_x1, k2x_t1);
                        let qy_h1 = builder.ins().fsub(q_y1, h);
                        let k2y_t1 = builder.ins().fmul(k2y, t1);
                        let cb_y1 = builder.ins().fadd(qy_h1, k2y_t1);
                        let both_neg_cond1 = builder.ins().fmax(cb_x1, ca_y1);
                        let s1 = simd_select_neg(&mut builder, both_neg_cond1, neg_one_v, one);
                        let ca_xx1 = builder.ins().fmul(ca_x1, ca_x1);
                        let ca_yy1 = builder.ins().fmul(ca_y1, ca_y1);
                        let ca_sq1 = builder.ins().fadd(ca_xx1, ca_yy1);
                        let cb_xx1 = builder.ins().fmul(cb_x1, cb_x1);
                        let cb_yy1 = builder.ins().fmul(cb_y1, cb_y1);
                        let cb_sq1 = builder.ins().fadd(cb_xx1, cb_yy1);
                        let d2_1 = builder.ins().fmin(ca_sq1, cb_sq1);
                        let dist1 = builder.ins().sqrt(d2_1);
                        let d1 = builder.ins().fmul(s1, dist1);

                        let ds0 = builder.ins().fmul(d0, curr_scale.0);
                        let ds1 = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((ds0, ds1));
                    }

                    OpCode::RoundedCone => {
                        let r1_val = inst.params[0];
                        let r2_val = inst.params[1];
                        let half_height = inst.params[2];

                        // Pre-compute constants (Division Exorcism)
                        let h_val = half_height * 2.0;
                        let inv_h = if h_val.abs() < 1e-10 {
                            1.0
                        } else {
                            1.0 / h_val
                        };
                        let b_val = (r1_val - r2_val) * inv_h;
                        let a_val = (1.0 - b_val * b_val).max(0.0).sqrt();
                        let ah_val = a_val * h_val;

                        let r1_s = builder.ins().f32const(r1_val);
                        let r2_s = builder.ins().f32const(r2_val);
                        let hh_s = builder.ins().f32const(half_height);
                        let h_s = builder.ins().f32const(h_val);
                        let b_s = builder.ins().f32const(b_val);
                        let a_s = builder.ins().f32const(a_val);
                        let ah_s = builder.ins().f32const(ah_val);

                        let r1_v = builder.ins().splat(simd_type, r1_s);
                        let r2_v = builder.ins().splat(simd_type, r2_s);
                        let hh_v = builder.ins().splat(simd_type, hh_s);
                        let h_v = builder.ins().splat(simd_type, h_s);
                        let b_v = builder.ins().splat(simd_type, b_s);
                        let a_v = builder.ins().splat(simd_type, a_s);
                        let ah_v = builder.ins().splat(simd_type, ah_s);

                        // Lane 0
                        let q_x0 = simd_length2(&mut builder, curr_x.0, curr_z.0);
                        let q_y0 = builder.ins().fadd(curr_y.0, hh_v);
                        // k = q_x * (-b) + q_y * a
                        let neg_b = builder.ins().fneg(b_v);
                        let qx_nb0 = builder.ins().fmul(q_x0, neg_b);
                        let qy_a0 = builder.ins().fmul(q_y0, a_v);
                        let k0 = builder.ins().fadd(qx_nb0, qy_a0);
                        // Case 1 (k < 0): length(q_x, q_y) - r1
                        let len1_0 = simd_length2(&mut builder, q_x0, q_y0);
                        let d1_0 = builder.ins().fsub(len1_0, r1_v);
                        // Case 2 (k > ah): length(q_x, q_y - h) - r2
                        let qy_h0 = builder.ins().fsub(q_y0, h_v);
                        let len2_0 = simd_length2(&mut builder, q_x0, qy_h0);
                        let d2_0 = builder.ins().fsub(len2_0, r2_v);
                        // Case 3: q_x * a + q_y * b - r1
                        let qxa0 = builder.ins().fmul(q_x0, a_v);
                        let qyb0 = builder.ins().fmul(q_y0, b_v);
                        let qxaqyb0 = builder.ins().fadd(qxa0, qyb0);
                        let d3_0 = builder.ins().fsub(qxaqyb0, r1_v);
                        // Branchless: select(k > ah, d2, d3) then select(k < 0, d1, inner)
                        let k_ah0 = builder.ins().fsub(ah_v, k0); // ah - k: negative when k > ah
                        let inner0 = simd_select_neg(&mut builder, k_ah0, d2_0, d3_0);
                        let d0 = simd_select_neg(&mut builder, k0, d1_0, inner0);

                        // Lane 1
                        let q_x1 = simd_length2(&mut builder, curr_x.1, curr_z.1);
                        let q_y1 = builder.ins().fadd(curr_y.1, hh_v);
                        let qx_nb1 = builder.ins().fmul(q_x1, neg_b);
                        let qy_a1 = builder.ins().fmul(q_y1, a_v);
                        let k1 = builder.ins().fadd(qx_nb1, qy_a1);
                        let len1_1 = simd_length2(&mut builder, q_x1, q_y1);
                        let d1_1 = builder.ins().fsub(len1_1, r1_v);
                        let qy_h1 = builder.ins().fsub(q_y1, h_v);
                        let len2_1 = simd_length2(&mut builder, q_x1, qy_h1);
                        let d2_1 = builder.ins().fsub(len2_1, r2_v);
                        let qxa1 = builder.ins().fmul(q_x1, a_v);
                        let qyb1 = builder.ins().fmul(q_y1, b_v);
                        let qxaqyb1 = builder.ins().fadd(qxa1, qyb1);
                        let d3_1 = builder.ins().fsub(qxaqyb1, r1_v);
                        let k_ah1 = builder.ins().fsub(ah_v, k1);
                        let inner1 = simd_select_neg(&mut builder, k_ah1, d2_1, d3_1);
                        let d1 = simd_select_neg(&mut builder, k1, d1_1, inner1);

                        let ds0 = builder.ins().fmul(d0, curr_scale.0);
                        let ds1 = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((ds0, ds1));
                    }

                    OpCode::Pyramid => {
                        let half_height = inst.params[0];
                        let h_val = half_height * 2.0;
                        let m2_val = h_val * h_val + 0.25;
                        let inv_m2_val = 1.0 / m2_val;
                        let inv_m2_025_val = 1.0 / (m2_val + 0.25);

                        let hh_s = builder.ins().f32const(half_height);
                        let h_s = builder.ins().f32const(h_val);
                        let m2_s = builder.ins().f32const(m2_val);
                        let inv_m2_s = builder.ins().f32const(inv_m2_val);
                        let inv_m2_025_s = builder.ins().f32const(inv_m2_025_val);
                        let half_s = builder.ins().f32const(0.5);
                        let neg_half_s = builder.ins().f32const(-0.5);
                        let neg_one_s = builder.ins().f32const(-1.0);

                        let hh_v = builder.ins().splat(simd_type, hh_s);
                        let h_v = builder.ins().splat(simd_type, h_s);
                        let m2_v = builder.ins().splat(simd_type, m2_s);
                        let inv_m2_v = builder.ins().splat(simd_type, inv_m2_s);
                        let inv_m2_025_v = builder.ins().splat(simd_type, inv_m2_025_s);
                        let half_v = builder.ins().splat(simd_type, half_s);
                        let neg_half_v = builder.ins().splat(simd_type, neg_half_s);
                        let neg_one_v = builder.ins().splat(simd_type, neg_one_s);

                        // Lane 0
                        let py0 = builder.ins().fadd(curr_y.0, hh_v);
                        let abs_px0 = builder.ins().fabs(curr_x.0);
                        let abs_pz0 = builder.ins().fabs(curr_z.0);
                        // Branchless swap
                        let px_s0 = builder.ins().fmax(abs_px0, abs_pz0);
                        let pz_s0 = builder.ins().fmin(abs_px0, abs_pz0);
                        let px_adj0 = builder.ins().fsub(px_s0, half_v);
                        let pz_adj0 = builder.ins().fsub(pz_s0, half_v);
                        // q = (pz_adj, h*py - 0.5*px_adj, h*px_adj + 0.5*py)
                        let qx0 = pz_adj0;
                        let nhalf_px0 = builder.ins().fmul(neg_half_v, px_adj0);
                        let h_py0 = builder.ins().fmul(h_v, py0);
                        let qy0 = builder.ins().fadd(h_py0, nhalf_px0);
                        let half_py0 = builder.ins().fmul(half_v, py0);
                        let h_px0 = builder.ins().fmul(h_v, px_adj0);
                        let qz0 = builder.ins().fadd(h_px0, half_py0);
                        // s = max(-qx, 0)
                        let neg_qx0 = builder.ins().fneg(qx0);
                        let s0 = builder.ins().fmax(neg_qx0, zero);
                        // t = clamp((qy - 0.5*pz) * inv_m2_025, 0, 1)
                        let half_pz0 = builder.ins().fmul(half_v, pz_adj0);
                        let qy_sub0 = builder.ins().fsub(qy0, half_pz0);
                        let t_raw0 = builder.ins().fmul(qy_sub0, inv_m2_025_v);
                        let t_min0 = builder.ins().fmin(t_raw0, one);
                        let t0 = builder.ins().fmax(zero, t_min0);
                        // a = m2*(qx+s)² + qy²
                        let qx_s0 = builder.ins().fadd(qx0, s0);
                        let qx_s_sq0 = builder.ins().fmul(qx_s0, qx_s0);
                        let m2_qxs0 = builder.ins().fmul(m2_v, qx_s_sq0);
                        let qy_sq0 = builder.ins().fmul(qy0, qy0);
                        let a0 = builder.ins().fadd(m2_qxs0, qy_sq0);
                        // b = m2*(qx+0.5t)² + (qy-m2*t)²
                        let half_t0 = builder.ins().fmul(half_v, t0);
                        let qx_ht0 = builder.ins().fadd(qx0, half_t0);
                        let qx_ht_sq0 = builder.ins().fmul(qx_ht0, qx_ht0);
                        let m2_t0 = builder.ins().fmul(m2_v, t0);
                        let qy_m2t0 = builder.ins().fsub(qy0, m2_t0);
                        let qy_m2t_sq0 = builder.ins().fmul(qy_m2t0, qy_m2t0);
                        let m2_qxht0 = builder.ins().fmul(m2_v, qx_ht_sq0);
                        let b0 = builder.ins().fadd(m2_qxht0, qy_m2t_sq0);
                        // d2 = (min(qy, -qx*m2 - qy*0.5) > 0) ? 0 : min(a, b)
                        let neg_qx_m2_0 = builder.ins().fmul(neg_qx0, m2_v);
                        let half_qy0 = builder.ins().fmul(half_v, qy0);
                        let cond0 = builder.ins().fsub(neg_qx_m2_0, half_qy0);
                        let min_cond0 = builder.ins().fmin(qy0, cond0);
                        let ab_min0 = builder.ins().fmin(a0, b0);
                        // min_cond > 0 → d2 = 0, else → d2 = ab_min
                        // Using sign: negate min_cond: if original > 0, negated < 0 → select zero
                        let neg_min_cond0 = builder.ins().fneg(min_cond0);
                        let d2_0 = simd_select_neg(&mut builder, neg_min_cond0, zero, ab_min0);
                        // result = sqrt((d2 + qz²) * inv_m2) * sign(max(qz, -py))
                        let qz_sq0 = builder.ins().fmul(qz0, qz0);
                        let d2_qz0 = builder.ins().fadd(d2_0, qz_sq0);
                        let d2_sc0 = builder.ins().fmul(d2_qz0, inv_m2_v);
                        let dist0 = builder.ins().sqrt(d2_sc0);
                        let neg_py0 = builder.ins().fneg(py0);
                        let sign_arg0 = builder.ins().fmax(qz0, neg_py0);
                        let signed0 = simd_select_neg(&mut builder, sign_arg0, neg_one_v, one);
                        let d0 = builder.ins().fmul(signed0, dist0);

                        // Lane 1
                        let py1 = builder.ins().fadd(curr_y.1, hh_v);
                        let abs_px1 = builder.ins().fabs(curr_x.1);
                        let abs_pz1 = builder.ins().fabs(curr_z.1);
                        let px_s1 = builder.ins().fmax(abs_px1, abs_pz1);
                        let pz_s1 = builder.ins().fmin(abs_px1, abs_pz1);
                        let px_adj1 = builder.ins().fsub(px_s1, half_v);
                        let pz_adj1 = builder.ins().fsub(pz_s1, half_v);
                        let qx1 = pz_adj1;
                        let nhalf_px1 = builder.ins().fmul(neg_half_v, px_adj1);
                        let h_py1 = builder.ins().fmul(h_v, py1);
                        let qy1 = builder.ins().fadd(h_py1, nhalf_px1);
                        let half_py1 = builder.ins().fmul(half_v, py1);
                        let h_px1 = builder.ins().fmul(h_v, px_adj1);
                        let qz1 = builder.ins().fadd(h_px1, half_py1);
                        let neg_qx1 = builder.ins().fneg(qx1);
                        let s1 = builder.ins().fmax(neg_qx1, zero);
                        let half_pz1 = builder.ins().fmul(half_v, pz_adj1);
                        let qy_sub1 = builder.ins().fsub(qy1, half_pz1);
                        let t_raw1 = builder.ins().fmul(qy_sub1, inv_m2_025_v);
                        let t_min1 = builder.ins().fmin(t_raw1, one);
                        let t1 = builder.ins().fmax(zero, t_min1);
                        let qx_s1 = builder.ins().fadd(qx1, s1);
                        let qx_s_sq1 = builder.ins().fmul(qx_s1, qx_s1);
                        let m2_qxs1 = builder.ins().fmul(m2_v, qx_s_sq1);
                        let qy_sq1 = builder.ins().fmul(qy1, qy1);
                        let a1 = builder.ins().fadd(m2_qxs1, qy_sq1);
                        let half_t1 = builder.ins().fmul(half_v, t1);
                        let qx_ht1 = builder.ins().fadd(qx1, half_t1);
                        let qx_ht_sq1 = builder.ins().fmul(qx_ht1, qx_ht1);
                        let m2_t1 = builder.ins().fmul(m2_v, t1);
                        let qy_m2t1 = builder.ins().fsub(qy1, m2_t1);
                        let qy_m2t_sq1 = builder.ins().fmul(qy_m2t1, qy_m2t1);
                        let m2_qxht1 = builder.ins().fmul(m2_v, qx_ht_sq1);
                        let b1 = builder.ins().fadd(m2_qxht1, qy_m2t_sq1);
                        let neg_qx_m2_1 = builder.ins().fmul(neg_qx1, m2_v);
                        let half_qy1 = builder.ins().fmul(half_v, qy1);
                        let cond1 = builder.ins().fsub(neg_qx_m2_1, half_qy1);
                        let min_cond1 = builder.ins().fmin(qy1, cond1);
                        let ab_min1 = builder.ins().fmin(a1, b1);
                        let neg_min_cond1 = builder.ins().fneg(min_cond1);
                        let d2_1 = simd_select_neg(&mut builder, neg_min_cond1, zero, ab_min1);
                        let qz_sq1 = builder.ins().fmul(qz1, qz1);
                        let d2_qz1 = builder.ins().fadd(d2_1, qz_sq1);
                        let d2_sc1 = builder.ins().fmul(d2_qz1, inv_m2_v);
                        let dist1 = builder.ins().sqrt(d2_sc1);
                        let neg_py1 = builder.ins().fneg(py1);
                        let sign_arg1 = builder.ins().fmax(qz1, neg_py1);
                        let signed1 = simd_select_neg(&mut builder, sign_arg1, neg_one_v, one);
                        let d1 = builder.ins().fmul(signed1, dist1);

                        let ds0 = builder.ins().fmul(d0, curr_scale.0);
                        let ds1 = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((ds0, ds1));
                    }

                    OpCode::Octahedron => {
                        let size_val = inst.params[0];
                        let s_s = builder.ins().f32const(size_val);
                        let three_s = builder.ins().f32const(3.0);
                        let half_s = builder.ins().f32const(0.5);
                        let inv_sqrt3_s = builder.ins().f32const(0.57735027);
                        let s_v = builder.ins().splat(simd_type, s_s);
                        let three_v = builder.ins().splat(simd_type, three_s);
                        let half_v = builder.ins().splat(simd_type, half_s);
                        let inv_sqrt3_v = builder.ins().splat(simd_type, inv_sqrt3_s);

                        // Lane 0
                        let apx0 = builder.ins().fabs(curr_x.0);
                        let apy0 = builder.ins().fabs(curr_y.0);
                        let apz0 = builder.ins().fabs(curr_z.0);
                        let sum_xy0 = builder.ins().fadd(apx0, apy0);
                        let sum_xyz0 = builder.ins().fadd(sum_xy0, apz0);
                        let m0 = builder.ins().fsub(sum_xyz0, s_v);
                        // cond = 3*p - m (negative means 3*p < m → case applies)
                        let tpx0 = builder.ins().fmul(three_v, apx0);
                        let tpy0 = builder.ins().fmul(three_v, apy0);
                        let tpz0 = builder.ins().fmul(three_v, apz0);
                        let c1_0 = builder.ins().fsub(tpx0, m0);
                        let c2_0 = builder.ins().fsub(tpy0, m0);
                        let c3_0 = builder.ins().fsub(tpz0, m0);
                        // Cascading select for q (case3 default, override case2, override case1)
                        let qx_c2_0 = simd_select_neg(&mut builder, c2_0, apy0, apz0);
                        let qy_c2_0 = simd_select_neg(&mut builder, c2_0, apz0, apx0);
                        let qz_c2_0 = simd_select_neg(&mut builder, c2_0, apx0, apy0);
                        let qx0 = simd_select_neg(&mut builder, c1_0, apx0, qx_c2_0);
                        let qy0 = simd_select_neg(&mut builder, c1_0, apy0, qy_c2_0);
                        let qz0 = simd_select_neg(&mut builder, c1_0, apz0, qz_c2_0);
                        // k = clamp(0.5*(qz - qy + s), 0, s)
                        let qz_qy0 = builder.ins().fsub(qz0, qy0);
                        let qz_qy_s0 = builder.ins().fadd(qz_qy0, s_v);
                        let hv0 = builder.ins().fmul(half_v, qz_qy_s0);
                        let k_min0 = builder.ins().fmin(hv0, s_v);
                        let k0 = builder.ins().fmax(zero, k_min0);
                        // detail = length(qx, qy-s+k, qz-k)
                        let qy_s0 = builder.ins().fsub(qy0, s_v);
                        let qy_sk0 = builder.ins().fadd(qy_s0, k0);
                        let qz_k0 = builder.ins().fsub(qz0, k0);
                        let detail0 = simd_length3(&mut builder, qx0, qy_sk0, qz_k0);
                        // early = m * inv_sqrt3
                        let early0 = builder.ins().fmul(m0, inv_sqrt3_v);
                        // any_case = min(c1, min(c2, c3)) < 0
                        let min_c23_0 = builder.ins().fmin(c2_0, c3_0);
                        let any_cond0 = builder.ins().fmin(c1_0, min_c23_0);
                        let d0 = simd_select_neg(&mut builder, any_cond0, detail0, early0);

                        // Lane 1
                        let apx1 = builder.ins().fabs(curr_x.1);
                        let apy1 = builder.ins().fabs(curr_y.1);
                        let apz1 = builder.ins().fabs(curr_z.1);
                        let sum_xy1 = builder.ins().fadd(apx1, apy1);
                        let sum_xyz1 = builder.ins().fadd(sum_xy1, apz1);
                        let m1 = builder.ins().fsub(sum_xyz1, s_v);
                        let tpx1 = builder.ins().fmul(three_v, apx1);
                        let tpy1 = builder.ins().fmul(three_v, apy1);
                        let tpz1 = builder.ins().fmul(three_v, apz1);
                        let c1_1 = builder.ins().fsub(tpx1, m1);
                        let c2_1 = builder.ins().fsub(tpy1, m1);
                        let c3_1 = builder.ins().fsub(tpz1, m1);
                        let qx_c2_1 = simd_select_neg(&mut builder, c2_1, apy1, apz1);
                        let qy_c2_1 = simd_select_neg(&mut builder, c2_1, apz1, apx1);
                        let qz_c2_1 = simd_select_neg(&mut builder, c2_1, apx1, apy1);
                        let qx1 = simd_select_neg(&mut builder, c1_1, apx1, qx_c2_1);
                        let qy1 = simd_select_neg(&mut builder, c1_1, apy1, qy_c2_1);
                        let qz1 = simd_select_neg(&mut builder, c1_1, apz1, qz_c2_1);
                        let qz_qy1 = builder.ins().fsub(qz1, qy1);
                        let qz_qy_s1 = builder.ins().fadd(qz_qy1, s_v);
                        let hv1 = builder.ins().fmul(half_v, qz_qy_s1);
                        let k_min1 = builder.ins().fmin(hv1, s_v);
                        let k1 = builder.ins().fmax(zero, k_min1);
                        let qy_s1 = builder.ins().fsub(qy1, s_v);
                        let qy_sk1 = builder.ins().fadd(qy_s1, k1);
                        let qz_k1 = builder.ins().fsub(qz1, k1);
                        let detail1 = simd_length3(&mut builder, qx1, qy_sk1, qz_k1);
                        let early1 = builder.ins().fmul(m1, inv_sqrt3_v);
                        let min_c23_1 = builder.ins().fmin(c2_1, c3_1);
                        let any_cond1 = builder.ins().fmin(c1_1, min_c23_1);
                        let d1 = simd_select_neg(&mut builder, any_cond1, detail1, early1);

                        let ds0 = builder.ins().fmul(d0, curr_scale.0);
                        let ds1 = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((ds0, ds1));
                    }

                    OpCode::HexPrism => {
                        let hex_radius = inst.params[0];
                        let half_height = inst.params[1];

                        let kx_val: f32 = -0.8660254;
                        let ky_val: f32 = 0.5;
                        let kz_val: f32 = 0.57735027;

                        let hr_s = builder.ins().f32const(hex_radius);
                        let hh_s = builder.ins().f32const(half_height);
                        let kx_s = builder.ins().f32const(kx_val);
                        let ky_s = builder.ins().f32const(ky_val);
                        let two_s = builder.ins().f32const(2.0);
                        let neg_one_s = builder.ins().f32const(-1.0);
                        let kz_hr = kz_val * hex_radius;
                        let neg_kz_hr_s = builder.ins().f32const(-kz_hr);
                        let kz_hr_s = builder.ins().f32const(kz_hr);

                        let hr_v = builder.ins().splat(simd_type, hr_s);
                        let hh_v = builder.ins().splat(simd_type, hh_s);
                        let kx_v = builder.ins().splat(simd_type, kx_s);
                        let ky_v = builder.ins().splat(simd_type, ky_s);
                        let two_v = builder.ins().splat(simd_type, two_s);
                        let neg_one_v = builder.ins().splat(simd_type, neg_one_s);
                        let neg_kz_hr_v = builder.ins().splat(simd_type, neg_kz_hr_s);
                        let kz_hr_v = builder.ins().splat(simd_type, kz_hr_s);

                        // Lane 0
                        let apx0 = builder.ins().fabs(curr_x.0);
                        let apy0 = builder.ins().fabs(curr_y.0);
                        let apz0 = builder.ins().fabs(curr_z.0);
                        // dot(k.xy, p.xy) = kx*px + ky*py
                        let kx_px0 = builder.ins().fmul(kx_v, apx0);
                        let ky_py0 = builder.ins().fmul(ky_v, apy0);
                        let dot0 = builder.ins().fadd(kx_px0, ky_py0);
                        let dot_min0 = builder.ins().fmin(dot0, zero);
                        let reflect0 = builder.ins().fmul(two_v, dot_min0);
                        // px -= reflect * kx, py -= reflect * ky
                        let rkx0 = builder.ins().fmul(reflect0, kx_v);
                        let rky0 = builder.ins().fmul(reflect0, ky_v);
                        let px_r0 = builder.ins().fsub(apx0, rkx0);
                        let py_r0 = builder.ins().fsub(apy0, rky0);
                        // clamp(px, -kz*hr, kz*hr)
                        let px_cl0 = builder.ins().fmin(px_r0, kz_hr_v);
                        let clamped0 = builder.ins().fmax(neg_kz_hr_v, px_cl0);
                        // d_xy = length(px-clamped, py-hr) * sign(py-hr)
                        let dx0 = builder.ins().fsub(px_r0, clamped0);
                        let dy0 = builder.ins().fsub(py_r0, hr_v);
                        let len_dxy0 = simd_length2(&mut builder, dx0, dy0);
                        let sign_dy0 = simd_select_neg(&mut builder, dy0, neg_one_v, one);
                        let d_xy0 = builder.ins().fmul(len_dxy0, sign_dy0);
                        // d_z = |pz| - hh
                        let d_z0 = builder.ins().fsub(apz0, hh_v);
                        // result = max(d_xy, d_z).min(0) + length(max(d_xy,0), max(d_z,0))
                        let max_dd0 = builder.ins().fmax(d_xy0, d_z0);
                        let interior0 = builder.ins().fmin(max_dd0, zero);
                        let d_xy_p0 = builder.ins().fmax(d_xy0, zero);
                        let d_z_p0 = builder.ins().fmax(d_z0, zero);
                        let exterior0 = simd_length2(&mut builder, d_xy_p0, d_z_p0);
                        let d0 = builder.ins().fadd(interior0, exterior0);

                        // Lane 1
                        let apx1 = builder.ins().fabs(curr_x.1);
                        let apy1 = builder.ins().fabs(curr_y.1);
                        let apz1 = builder.ins().fabs(curr_z.1);
                        let kx_px1 = builder.ins().fmul(kx_v, apx1);
                        let ky_py1 = builder.ins().fmul(ky_v, apy1);
                        let dot1 = builder.ins().fadd(kx_px1, ky_py1);
                        let dot_min1 = builder.ins().fmin(dot1, zero);
                        let reflect1 = builder.ins().fmul(two_v, dot_min1);
                        let rkx1 = builder.ins().fmul(reflect1, kx_v);
                        let rky1 = builder.ins().fmul(reflect1, ky_v);
                        let px_r1 = builder.ins().fsub(apx1, rkx1);
                        let py_r1 = builder.ins().fsub(apy1, rky1);
                        let px_cl1 = builder.ins().fmin(px_r1, kz_hr_v);
                        let clamped1 = builder.ins().fmax(neg_kz_hr_v, px_cl1);
                        let dx1 = builder.ins().fsub(px_r1, clamped1);
                        let dy1 = builder.ins().fsub(py_r1, hr_v);
                        let len_dxy1 = simd_length2(&mut builder, dx1, dy1);
                        let sign_dy1 = simd_select_neg(&mut builder, dy1, neg_one_v, one);
                        let d_xy1 = builder.ins().fmul(len_dxy1, sign_dy1);
                        let d_z1 = builder.ins().fsub(apz1, hh_v);
                        let max_dd1 = builder.ins().fmax(d_xy1, d_z1);
                        let interior1 = builder.ins().fmin(max_dd1, zero);
                        let d_xy_p1 = builder.ins().fmax(d_xy1, zero);
                        let d_z_p1 = builder.ins().fmax(d_z1, zero);
                        let exterior1 = simd_length2(&mut builder, d_xy_p1, d_z_p1);
                        let d1 = builder.ins().fadd(interior1, exterior1);

                        let ds0 = builder.ins().fmul(d0, curr_scale.0);
                        let ds1 = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((ds0, ds1));
                    }

                    OpCode::Union => {
                        let b = value_stack.pop().unwrap();
                        let a = value_stack.pop().unwrap();
                        let r0 = builder.ins().fmin(a.0, b.0);
                        let r1 = builder.ins().fmin(a.1, b.1);
                        value_stack.push((r0, r1));
                    }

                    OpCode::Intersection => {
                        let b = value_stack.pop().unwrap();
                        let a = value_stack.pop().unwrap();
                        let r0 = builder.ins().fmax(a.0, b.0);
                        let r1 = builder.ins().fmax(a.1, b.1);
                        value_stack.push((r0, r1));
                    }

                    OpCode::Subtraction => {
                        let b = value_stack.pop().unwrap();
                        let a = value_stack.pop().unwrap();
                        // max(a, -b)
                        let neg_b0 = builder.ins().fneg(b.0);
                        let neg_b1 = builder.ins().fneg(b.1);
                        let r0 = builder.ins().fmax(a.0, neg_b0);
                        let r1 = builder.ins().fmax(a.1, neg_b1);
                        value_stack.push((r0, r1));
                    }

                    OpCode::SmoothUnion => {
                        let k = inst.params[0];
                        let b = value_stack.pop().unwrap();
                        let a = value_stack.pop().unwrap();

                        let k_s = builder.ins().f32const(k);
                        let k_v = builder.ins().splat(simd_type, k_s);
                        let half_s = builder.ins().f32const(0.5);
                        let half = builder.ins().splat(simd_type, half_s);

                        // Lane 0: h = max(k - abs(a - b), 0) / k
                        let diff0 = builder.ins().fsub(a.0, b.0);
                        let abs_diff0 = builder.ins().fabs(diff0);
                        let k_minus0 = builder.ins().fsub(k_v, abs_diff0);
                        let h_num0 = builder.ins().fmax(k_minus0, zero);
                        let h0 = builder.ins().fdiv(h_num0, k_v);
                        // min(a, b) - h*h*k*0.25
                        let min_ab0 = builder.ins().fmin(a.0, b.0);
                        let hh0 = builder.ins().fmul(h0, h0);
                        let hhk0 = builder.ins().fmul(hh0, k_v);
                        let corr0 = builder.ins().fmul(hhk0, half);
                        let corr0 = builder.ins().fmul(corr0, half);
                        let d0 = builder.ins().fsub(min_ab0, corr0);

                        // Lane 1
                        let diff1 = builder.ins().fsub(a.1, b.1);
                        let abs_diff1 = builder.ins().fabs(diff1);
                        let k_minus1 = builder.ins().fsub(k_v, abs_diff1);
                        let h_num1 = builder.ins().fmax(k_minus1, zero);
                        let h1 = builder.ins().fdiv(h_num1, k_v);
                        let min_ab1 = builder.ins().fmin(a.1, b.1);
                        let hh1 = builder.ins().fmul(h1, h1);
                        let hhk1 = builder.ins().fmul(hh1, k_v);
                        let corr1 = builder.ins().fmul(hhk1, half);
                        let corr1 = builder.ins().fmul(corr1, half);
                        let d1 = builder.ins().fsub(min_ab1, corr1);

                        value_stack.push((d0, d1));
                    }

                    OpCode::SmoothIntersection => {
                        let k = inst.params[0];
                        let b = value_stack.pop().unwrap();
                        let a = value_stack.pop().unwrap();

                        let k_s = builder.ins().f32const(k);
                        let k_v = builder.ins().splat(simd_type, k_s);
                        let half_s = builder.ins().f32const(0.5);
                        let half = builder.ins().splat(simd_type, half_s);

                        // Lane 0
                        let diff0 = builder.ins().fsub(a.0, b.0);
                        let abs_diff0 = builder.ins().fabs(diff0);
                        let k_minus0 = builder.ins().fsub(k_v, abs_diff0);
                        let h_num0 = builder.ins().fmax(k_minus0, zero);
                        let h0 = builder.ins().fdiv(h_num0, k_v);
                        let max_ab0 = builder.ins().fmax(a.0, b.0);
                        let hh0 = builder.ins().fmul(h0, h0);
                        let hhk0 = builder.ins().fmul(hh0, k_v);
                        let corr0 = builder.ins().fmul(hhk0, half);
                        let corr0 = builder.ins().fmul(corr0, half);
                        let d0 = builder.ins().fadd(max_ab0, corr0);

                        // Lane 1
                        let diff1 = builder.ins().fsub(a.1, b.1);
                        let abs_diff1 = builder.ins().fabs(diff1);
                        let k_minus1 = builder.ins().fsub(k_v, abs_diff1);
                        let h_num1 = builder.ins().fmax(k_minus1, zero);
                        let h1 = builder.ins().fdiv(h_num1, k_v);
                        let max_ab1 = builder.ins().fmax(a.1, b.1);
                        let hh1 = builder.ins().fmul(h1, h1);
                        let hhk1 = builder.ins().fmul(hh1, k_v);
                        let corr1 = builder.ins().fmul(hhk1, half);
                        let corr1 = builder.ins().fmul(corr1, half);
                        let d1 = builder.ins().fadd(max_ab1, corr1);

                        value_stack.push((d0, d1));
                    }

                    OpCode::SmoothSubtraction => {
                        let k = inst.params[0];
                        let b = value_stack.pop().unwrap();
                        let a = value_stack.pop().unwrap();

                        let k_s = builder.ins().f32const(k);
                        let k_v = builder.ins().splat(simd_type, k_s);
                        let half_s = builder.ins().f32const(0.5);
                        let half = builder.ins().splat(simd_type, half_s);

                        // Lane 0: smooth intersection of a and -b
                        let neg_b0 = builder.ins().fneg(b.0);
                        let diff0 = builder.ins().fsub(a.0, neg_b0);
                        let abs_diff0 = builder.ins().fabs(diff0);
                        let k_minus0 = builder.ins().fsub(k_v, abs_diff0);
                        let h_num0 = builder.ins().fmax(k_minus0, zero);
                        let h0 = builder.ins().fdiv(h_num0, k_v);
                        let max_ab0 = builder.ins().fmax(a.0, neg_b0);
                        let hh0 = builder.ins().fmul(h0, h0);
                        let hhk0 = builder.ins().fmul(hh0, k_v);
                        let corr0 = builder.ins().fmul(hhk0, half);
                        let corr0 = builder.ins().fmul(corr0, half);
                        let d0 = builder.ins().fadd(max_ab0, corr0);

                        // Lane 1
                        let neg_b1 = builder.ins().fneg(b.1);
                        let diff1 = builder.ins().fsub(a.1, neg_b1);
                        let abs_diff1 = builder.ins().fabs(diff1);
                        let k_minus1 = builder.ins().fsub(k_v, abs_diff1);
                        let h_num1 = builder.ins().fmax(k_minus1, zero);
                        let h1 = builder.ins().fdiv(h_num1, k_v);
                        let max_ab1 = builder.ins().fmax(a.1, neg_b1);
                        let hh1 = builder.ins().fmul(h1, h1);
                        let hhk1 = builder.ins().fmul(hh1, k_v);
                        let corr1 = builder.ins().fmul(hhk1, half);
                        let corr1 = builder.ins().fmul(corr1, half);
                        let d1 = builder.ins().fadd(max_ab1, corr1);

                        value_stack.push((d0, d1));
                    }

                    OpCode::ChamferUnion => {
                        let r = inst.params[0];
                        let b = value_stack.pop().unwrap();
                        let a = value_stack.pop().unwrap();

                        let r_s = builder.ins().f32const(r);
                        let r_v = builder.ins().splat(simd_type, r_s);
                        let s_s = builder.ins().f32const(std::f32::consts::FRAC_1_SQRT_2);
                        let s_v = builder.ins().splat(simd_type, s_s);

                        let min0 = builder.ins().fmin(a.0, b.0);
                        let sum0 = builder.ins().fadd(a.0, b.0);
                        let sc0 = builder.ins().fmul(sum0, s_v);
                        let ch0 = builder.ins().fsub(sc0, r_v);
                        let d0 = builder.ins().fmin(min0, ch0);

                        let min1 = builder.ins().fmin(a.1, b.1);
                        let sum1 = builder.ins().fadd(a.1, b.1);
                        let sc1 = builder.ins().fmul(sum1, s_v);
                        let ch1 = builder.ins().fsub(sc1, r_v);
                        let d1 = builder.ins().fmin(min1, ch1);

                        value_stack.push((d0, d1));
                    }

                    OpCode::ChamferIntersection => {
                        let r = inst.params[0];
                        let b = value_stack.pop().unwrap();
                        let a = value_stack.pop().unwrap();

                        let r_s = builder.ins().f32const(r);
                        let r_v = builder.ins().splat(simd_type, r_s);
                        let s_s = builder.ins().f32const(std::f32::consts::FRAC_1_SQRT_2);
                        let s_v = builder.ins().splat(simd_type, s_s);

                        let na0 = builder.ins().fneg(a.0);
                        let nb0 = builder.ins().fneg(b.0);
                        let min0 = builder.ins().fmin(na0, nb0);
                        let sum0 = builder.ins().fadd(na0, nb0);
                        let sc0 = builder.ins().fmul(sum0, s_v);
                        let ch0 = builder.ins().fsub(sc0, r_v);
                        let cm0 = builder.ins().fmin(min0, ch0);
                        let d0 = builder.ins().fneg(cm0);

                        let na1 = builder.ins().fneg(a.1);
                        let nb1 = builder.ins().fneg(b.1);
                        let min1 = builder.ins().fmin(na1, nb1);
                        let sum1 = builder.ins().fadd(na1, nb1);
                        let sc1 = builder.ins().fmul(sum1, s_v);
                        let ch1 = builder.ins().fsub(sc1, r_v);
                        let cm1 = builder.ins().fmin(min1, ch1);
                        let d1 = builder.ins().fneg(cm1);

                        value_stack.push((d0, d1));
                    }

                    OpCode::ChamferSubtraction => {
                        let r = inst.params[0];
                        let b = value_stack.pop().unwrap();
                        let a = value_stack.pop().unwrap();

                        let r_s = builder.ins().f32const(r);
                        let r_v = builder.ins().splat(simd_type, r_s);
                        let s_s = builder.ins().f32const(std::f32::consts::FRAC_1_SQRT_2);
                        let s_v = builder.ins().splat(simd_type, s_s);

                        let na0 = builder.ins().fneg(a.0);
                        let min0 = builder.ins().fmin(na0, b.0);
                        let sum0 = builder.ins().fadd(na0, b.0);
                        let sc0 = builder.ins().fmul(sum0, s_v);
                        let ch0 = builder.ins().fsub(sc0, r_v);
                        let cm0 = builder.ins().fmin(min0, ch0);
                        let d0 = builder.ins().fneg(cm0);

                        let na1 = builder.ins().fneg(a.1);
                        let min1 = builder.ins().fmin(na1, b.1);
                        let sum1 = builder.ins().fadd(na1, b.1);
                        let sc1 = builder.ins().fmul(sum1, s_v);
                        let ch1 = builder.ins().fsub(sc1, r_v);
                        let cm1 = builder.ins().fmin(min1, ch1);
                        let d1 = builder.ins().fneg(cm1);

                        value_stack.push((d0, d1));
                    }

                    OpCode::StairsUnion => {
                        let r = inst.params[0];
                        let n = inst.params[1];
                        let b = value_stack.pop().unwrap();
                        let a = value_stack.pop().unwrap();

                        let d0 = emit_jitsimd_stairs_min(&mut builder, simd_type, a.0, b.0, r, n);
                        let d1 = emit_jitsimd_stairs_min(&mut builder, simd_type, a.1, b.1, r, n);
                        value_stack.push((d0, d1));
                    }

                    OpCode::StairsIntersection => {
                        let r = inst.params[0];
                        let n = inst.params[1];
                        let b = value_stack.pop().unwrap();
                        let a = value_stack.pop().unwrap();

                        let na0 = builder.ins().fneg(a.0);
                        let nb0 = builder.ins().fneg(b.0);
                        let sm0 = emit_jitsimd_stairs_min(&mut builder, simd_type, na0, nb0, r, n);
                        let d0 = builder.ins().fneg(sm0);

                        let na1 = builder.ins().fneg(a.1);
                        let nb1 = builder.ins().fneg(b.1);
                        let sm1 = emit_jitsimd_stairs_min(&mut builder, simd_type, na1, nb1, r, n);
                        let d1 = builder.ins().fneg(sm1);

                        value_stack.push((d0, d1));
                    }

                    OpCode::StairsSubtraction => {
                        let r = inst.params[0];
                        let n = inst.params[1];
                        let b = value_stack.pop().unwrap();
                        let a = value_stack.pop().unwrap();

                        let na0 = builder.ins().fneg(a.0);
                        let sm0 = emit_jitsimd_stairs_min(&mut builder, simd_type, na0, b.0, r, n);
                        let d0 = builder.ins().fneg(sm0);

                        let na1 = builder.ins().fneg(a.1);
                        let sm1 = emit_jitsimd_stairs_min(&mut builder, simd_type, na1, b.1, r, n);
                        let d1 = builder.ins().fneg(sm1);

                        value_stack.push((d0, d1));
                    }

                    OpCode::Translate => {
                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Translate,
                            params: inst.params,
                        });

                        let tx_s = builder.ins().f32const(inst.params[0]);
                        let ty_s = builder.ins().f32const(inst.params[1]);
                        let tz_s = builder.ins().f32const(inst.params[2]);
                        let tx = builder.ins().splat(simd_type, tx_s);
                        let ty = builder.ins().splat(simd_type, ty_s);
                        let tz = builder.ins().splat(simd_type, tz_s);

                        let new_x0 = builder.ins().fsub(curr_x.0, tx);
                        let new_x1 = builder.ins().fsub(curr_x.1, tx);
                        let new_y0 = builder.ins().fsub(curr_y.0, ty);
                        let new_y1 = builder.ins().fsub(curr_y.1, ty);
                        let new_z0 = builder.ins().fsub(curr_z.0, tz);
                        let new_z1 = builder.ins().fsub(curr_z.1, tz);

                        curr_x = (new_x0, new_x1);
                        curr_y = (new_y0, new_y1);
                        curr_z = (new_z0, new_z1);
                    }

                    OpCode::Scale => {
                        // params[0] = 1/factor (precomputed inverse)
                        // params[1] = factor (original)
                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Scale,
                            params: inst.params,
                        });

                        let inv_s = builder.ins().f32const(inst.params[0]);
                        let inv_v = builder.ins().splat(simd_type, inv_s);
                        let f_s = builder.ins().f32const(inst.params[1]);
                        let f_v = builder.ins().splat(simd_type, f_s);

                        // Division Exorcism: p *= inv_factor (no division)
                        let new_x0 = builder.ins().fmul(curr_x.0, inv_v);
                        let new_x1 = builder.ins().fmul(curr_x.1, inv_v);
                        let new_y0 = builder.ins().fmul(curr_y.0, inv_v);
                        let new_y1 = builder.ins().fmul(curr_y.1, inv_v);
                        let new_z0 = builder.ins().fmul(curr_z.0, inv_v);
                        let new_z1 = builder.ins().fmul(curr_z.1, inv_v);

                        curr_x = (new_x0, new_x1);
                        curr_y = (new_y0, new_y1);
                        curr_z = (new_z0, new_z1);

                        // scale_correction *= factor
                        let new_scale0 = builder.ins().fmul(curr_scale.0, f_v);
                        let new_scale1 = builder.ins().fmul(curr_scale.1, f_v);
                        curr_scale = (new_scale0, new_scale1);
                    }

                    OpCode::Round => {
                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Round,
                            params: inst.params,
                        });
                    }

                    OpCode::Onion => {
                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Onion,
                            params: inst.params,
                        });
                    }

                    OpCode::Mirror => {
                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Mirror,
                            params: inst.params,
                        });

                        // Conditionally apply abs (axes known at JIT compile time)
                        if inst.params[0] != 0.0 {
                            curr_x = (builder.ins().fabs(curr_x.0), builder.ins().fabs(curr_x.1));
                        }
                        if inst.params[1] != 0.0 {
                            curr_y = (builder.ins().fabs(curr_y.0), builder.ins().fabs(curr_y.1));
                        }
                        if inst.params[2] != 0.0 {
                            curr_z = (builder.ins().fabs(curr_z.0), builder.ins().fabs(curr_z.1));
                        }
                    }

                    OpCode::SweepBezier => {
                        return Err("SweepBezier not supported in SIMD JIT".to_string());
                    }

                    OpCode::Revolution => {
                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Revolution,
                            params: inst.params,
                        });

                        let off_s = builder.ins().f32const(inst.params[0]);
                        let off = builder.ins().splat(simd_type, off_s);

                        // Lane 0: q = sqrt(x² + z²) - offset
                        let xx0 = builder.ins().fmul(curr_x.0, curr_x.0);
                        let zz0 = builder.ins().fmul(curr_z.0, curr_z.0);
                        let xz0 = builder.ins().fadd(xx0, zz0);
                        let len0 = builder.ins().sqrt(xz0);
                        let q0 = builder.ins().fsub(len0, off);

                        // Lane 1
                        let xx1 = builder.ins().fmul(curr_x.1, curr_x.1);
                        let zz1 = builder.ins().fmul(curr_z.1, curr_z.1);
                        let xz1 = builder.ins().fadd(xx1, zz1);
                        let len1 = builder.ins().sqrt(xz1);
                        let q1 = builder.ins().fsub(len1, off);

                        curr_x = (q0, q1);
                        // curr_y unchanged
                        curr_z = (zero, zero);
                    }

                    OpCode::Extrude => {
                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z, // saves original z for post-processing
                            scale: curr_scale,
                            opcode: OpCode::Extrude,
                            params: inst.params, // params[0] = half_height
                        });

                        // Evaluate child at (x, y, 0)
                        curr_z = (zero, zero);
                    }

                    OpCode::Rotate => {
                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Rotate,
                            params: inst.params,
                        });

                        // Quaternion → rotation matrix (compile-time)
                        let qx = inst.params[0];
                        let qy = inst.params[1];
                        let qz = inst.params[2];
                        let qw = inst.params[3];
                        // Inverse quaternion: negate xyz
                        let qx = -qx;
                        let qy = -qy;
                        let qz = -qz;
                        let m00 = 1.0 - 2.0 * (qy * qy + qz * qz);
                        let m01 = 2.0 * (qx * qy - qz * qw);
                        let m02 = 2.0 * (qx * qz + qy * qw);
                        let m10 = 2.0 * (qx * qy + qz * qw);
                        let m11 = 1.0 - 2.0 * (qx * qx + qz * qz);
                        let m12 = 2.0 * (qy * qz - qx * qw);
                        let m20 = 2.0 * (qx * qz - qy * qw);
                        let m21 = 2.0 * (qy * qz + qx * qw);
                        let m22 = 1.0 - 2.0 * (qx * qx + qy * qy);

                        let m00_s = builder.ins().f32const(m00);
                        let m00v = builder.ins().splat(simd_type, m00_s);
                        let m01_s = builder.ins().f32const(m01);
                        let m01v = builder.ins().splat(simd_type, m01_s);
                        let m02_s = builder.ins().f32const(m02);
                        let m02v = builder.ins().splat(simd_type, m02_s);
                        let m10_s = builder.ins().f32const(m10);
                        let m10v = builder.ins().splat(simd_type, m10_s);
                        let m11_s = builder.ins().f32const(m11);
                        let m11v = builder.ins().splat(simd_type, m11_s);
                        let m12_s = builder.ins().f32const(m12);
                        let m12v = builder.ins().splat(simd_type, m12_s);
                        let m20_s = builder.ins().f32const(m20);
                        let m20v = builder.ins().splat(simd_type, m20_s);
                        let m21_s = builder.ins().f32const(m21);
                        let m21v = builder.ins().splat(simd_type, m21_s);
                        let m22_s = builder.ins().f32const(m22);
                        let m22v = builder.ins().splat(simd_type, m22_s);

                        // Lane 0: p' = M * p (standard order FMA chain)
                        // x' = m00*x + m01*y + m02*z
                        let t0 = builder.ins().fmul(m00v, curr_x.0);
                        let t0 = builder.ins().fma(m01v, curr_y.0, t0);
                        let nx0 = builder.ins().fma(m02v, curr_z.0, t0);
                        let t0 = builder.ins().fmul(m10v, curr_x.0);
                        let t0 = builder.ins().fma(m11v, curr_y.0, t0);
                        let ny0 = builder.ins().fma(m12v, curr_z.0, t0);
                        let t0 = builder.ins().fmul(m20v, curr_x.0);
                        let t0 = builder.ins().fma(m21v, curr_y.0, t0);
                        let nz0 = builder.ins().fma(m22v, curr_z.0, t0);

                        // Lane 1
                        let t1 = builder.ins().fmul(m00v, curr_x.1);
                        let t1 = builder.ins().fma(m01v, curr_y.1, t1);
                        let nx1 = builder.ins().fma(m02v, curr_z.1, t1);
                        let t1 = builder.ins().fmul(m10v, curr_x.1);
                        let t1 = builder.ins().fma(m11v, curr_y.1, t1);
                        let ny1 = builder.ins().fma(m12v, curr_z.1, t1);
                        let t1 = builder.ins().fmul(m20v, curr_x.1);
                        let t1 = builder.ins().fma(m21v, curr_y.1, t1);
                        let nz1 = builder.ins().fma(m22v, curr_z.1, t1);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        curr_z = (nz0, nz1);
                    }

                    OpCode::ScaleNonUniform => {
                        // params[0..2] = precomputed 1/sx, 1/sy, 1/sz
                        // params[3] = min(sx, sy, sz) for scale correction
                        let inv_sx = inst.params[0];
                        let inv_sy = inst.params[1];
                        let inv_sz = inst.params[2];
                        let min_factor = inst.params[3];

                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::ScaleNonUniform,
                            params: inst.params,
                        });

                        let isx_s = builder.ins().f32const(inv_sx);
                        let isx = builder.ins().splat(simd_type, isx_s);
                        let isy_s = builder.ins().f32const(inv_sy);
                        let isy = builder.ins().splat(simd_type, isy_s);
                        let isz_s = builder.ins().f32const(inv_sz);
                        let isz = builder.ins().splat(simd_type, isz_s);
                        let mf_s = builder.ins().f32const(min_factor);
                        let mf = builder.ins().splat(simd_type, mf_s);

                        curr_x = (
                            builder.ins().fmul(curr_x.0, isx),
                            builder.ins().fmul(curr_x.1, isx),
                        );
                        curr_y = (
                            builder.ins().fmul(curr_y.0, isy),
                            builder.ins().fmul(curr_y.1, isy),
                        );
                        curr_z = (
                            builder.ins().fmul(curr_z.0, isz),
                            builder.ins().fmul(curr_z.1, isz),
                        );
                        curr_scale = (
                            builder.ins().fmul(curr_scale.0, mf),
                            builder.ins().fmul(curr_scale.1, mf),
                        );
                    }

                    OpCode::Twist => {
                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Twist,
                            params: inst.params,
                        });

                        let k_s = builder.ins().f32const(inst.params[0]);
                        let k_v = builder.ins().splat(simd_type, k_s);

                        // Lane 0: angle = y * k, rotate XZ
                        let angle0 = builder.ins().fmul(curr_y.0, k_v);
                        let (cos0, sin0) = simd_sincos_approx_js(&mut builder, angle0, simd_type);
                        let cx0 = builder.ins().fmul(cos0, curr_x.0);
                        let sz0 = builder.ins().fmul(sin0, curr_z.0);
                        let nx0 = builder.ins().fsub(cx0, sz0);
                        let sx0 = builder.ins().fmul(sin0, curr_x.0);
                        let cz0 = builder.ins().fmul(cos0, curr_z.0);
                        let nz0 = builder.ins().fadd(sx0, cz0);

                        // Lane 1
                        let angle1 = builder.ins().fmul(curr_y.1, k_v);
                        let (cos1, sin1) = simd_sincos_approx_js(&mut builder, angle1, simd_type);
                        let cx1 = builder.ins().fmul(cos1, curr_x.1);
                        let sz1 = builder.ins().fmul(sin1, curr_z.1);
                        let nx1 = builder.ins().fsub(cx1, sz1);
                        let sx1 = builder.ins().fmul(sin1, curr_x.1);
                        let cz1 = builder.ins().fmul(cos1, curr_z.1);
                        let nz1 = builder.ins().fadd(sx1, cz1);

                        curr_x = (nx0, nx1);
                        // curr_y unchanged
                        curr_z = (nz0, nz1);
                    }

                    OpCode::Bend => {
                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Bend,
                            params: inst.params,
                        });

                        let k_s = builder.ins().f32const(inst.params[0]);
                        let k_v = builder.ins().splat(simd_type, k_s);

                        // Lane 0: angle = k * x, rotate XY
                        // x' = cos*x - sin*y, y' = sin*x + cos*y
                        let angle0 = builder.ins().fmul(k_v, curr_x.0);
                        let (cos0, sin0) = simd_sincos_approx_js(&mut builder, angle0, simd_type);
                        let cx0 = builder.ins().fmul(cos0, curr_x.0);
                        let sy0 = builder.ins().fmul(sin0, curr_y.0);
                        let nx0 = builder.ins().fsub(cx0, sy0);
                        let sx0 = builder.ins().fmul(sin0, curr_x.0);
                        let cy0 = builder.ins().fmul(cos0, curr_y.0);
                        let ny0 = builder.ins().fadd(sx0, cy0);

                        // Lane 1
                        let angle1 = builder.ins().fmul(k_v, curr_x.1);
                        let (cos1, sin1) = simd_sincos_approx_js(&mut builder, angle1, simd_type);
                        let cx1 = builder.ins().fmul(cos1, curr_x.1);
                        let sy1 = builder.ins().fmul(sin1, curr_y.1);
                        let nx1 = builder.ins().fsub(cx1, sy1);
                        let sx1 = builder.ins().fmul(sin1, curr_x.1);
                        let cy1 = builder.ins().fmul(cos1, curr_y.1);
                        let ny1 = builder.ins().fadd(sx1, cy1);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        // curr_z unchanged
                    }

                    OpCode::RepeatInfinite => {
                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::RepeatInfinite,
                            params: inst.params,
                        });

                        // Division Exorcism: pre-compute reciprocals at compile time
                        let sx = inst.params[0];
                        let sy = inst.params[1];
                        let sz = inst.params[2];
                        let isx = if sx.abs() < 1e-7 { 0.0 } else { 1.0 / sx };
                        let isy = if sy.abs() < 1e-7 { 0.0 } else { 1.0 / sy };
                        let isz = if sz.abs() < 1e-7 { 0.0 } else { 1.0 / sz };

                        let sx_s = builder.ins().f32const(sx);
                        let sx_v = builder.ins().splat(simd_type, sx_s);
                        let sy_s = builder.ins().f32const(sy);
                        let sy_v = builder.ins().splat(simd_type, sy_s);
                        let sz_s = builder.ins().f32const(sz);
                        let sz_v = builder.ins().splat(simd_type, sz_s);
                        let isx_s = builder.ins().f32const(isx);
                        let isx_v = builder.ins().splat(simd_type, isx_s);
                        let isy_s = builder.ins().f32const(isy);
                        let isy_v = builder.ins().splat(simd_type, isy_s);
                        let isz_s = builder.ins().f32const(isz);
                        let isz_v = builder.ins().splat(simd_type, isz_s);

                        // Lane 0: p - s * round(p * inv_s)
                        let rx0 = builder.ins().fmul(curr_x.0, isx_v);
                        let ry0 = builder.ins().fmul(curr_y.0, isy_v);
                        let rz0 = builder.ins().fmul(curr_z.0, isz_v);
                        let rx0 = builder.ins().nearest(rx0);
                        let ry0 = builder.ins().nearest(ry0);
                        let rz0 = builder.ins().nearest(rz0);
                        let ox0 = builder.ins().fmul(sx_v, rx0);
                        let oy0 = builder.ins().fmul(sy_v, ry0);
                        let oz0 = builder.ins().fmul(sz_v, rz0);
                        let nx0 = builder.ins().fsub(curr_x.0, ox0);
                        let ny0 = builder.ins().fsub(curr_y.0, oy0);
                        let nz0 = builder.ins().fsub(curr_z.0, oz0);

                        // Lane 1
                        let rx1 = builder.ins().fmul(curr_x.1, isx_v);
                        let ry1 = builder.ins().fmul(curr_y.1, isy_v);
                        let rz1 = builder.ins().fmul(curr_z.1, isz_v);
                        let rx1 = builder.ins().nearest(rx1);
                        let ry1 = builder.ins().nearest(ry1);
                        let rz1 = builder.ins().nearest(rz1);
                        let ox1 = builder.ins().fmul(sx_v, rx1);
                        let oy1 = builder.ins().fmul(sy_v, ry1);
                        let oz1 = builder.ins().fmul(sz_v, rz1);
                        let nx1 = builder.ins().fsub(curr_x.1, ox1);
                        let ny1 = builder.ins().fsub(curr_y.1, oy1);
                        let nz1 = builder.ins().fsub(curr_z.1, oz1);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        curr_z = (nz0, nz1);
                    }

                    OpCode::RepeatFinite => {
                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::RepeatFinite,
                            params: inst.params,
                        });

                        let cx = inst.params[0]; // count_x
                        let cy = inst.params[1]; // count_y
                        let cz = inst.params[2]; // count_z
                        let sx = inst.params[3];
                        let sy = inst.params[4];
                        let sz = inst.params[5];
                        let isx = if sx.abs() < 1e-7 { 0.0 } else { 1.0 / sx };
                        let isy = if sy.abs() < 1e-7 { 0.0 } else { 1.0 / sy };
                        let isz = if sz.abs() < 1e-7 { 0.0 } else { 1.0 / sz };
                        let lx = cx * 0.5;
                        let ly = cy * 0.5;
                        let lz = cz * 0.5;

                        let sx_s = builder.ins().f32const(sx);
                        let sx_v = builder.ins().splat(simd_type, sx_s);
                        let sy_s = builder.ins().f32const(sy);
                        let sy_v = builder.ins().splat(simd_type, sy_s);
                        let sz_s = builder.ins().f32const(sz);
                        let sz_v = builder.ins().splat(simd_type, sz_s);
                        let isx_s = builder.ins().f32const(isx);
                        let isx_v = builder.ins().splat(simd_type, isx_s);
                        let isy_s = builder.ins().f32const(isy);
                        let isy_v = builder.ins().splat(simd_type, isy_s);
                        let isz_s = builder.ins().f32const(isz);
                        let isz_v = builder.ins().splat(simd_type, isz_s);
                        let lx_s = builder.ins().f32const(lx);
                        let lx_v = builder.ins().splat(simd_type, lx_s);
                        let ly_s = builder.ins().f32const(ly);
                        let ly_v = builder.ins().splat(simd_type, ly_s);
                        let lz_s = builder.ins().f32const(lz);
                        let lz_v = builder.ins().splat(simd_type, lz_s);
                        let nlx_s = builder.ins().f32const(-lx);
                        let nlx_v = builder.ins().splat(simd_type, nlx_s);
                        let nly_s = builder.ins().f32const(-ly);
                        let nly_v = builder.ins().splat(simd_type, nly_s);
                        let nlz_s = builder.ins().f32const(-lz);
                        let nlz_v = builder.ins().splat(simd_type, nlz_s);

                        // Lane 0: clamp(round(p * inv_s), -limit, limit), then p - cell * s
                        let pix0 = builder.ins().fmul(curr_x.0, isx_v);
                        let rx0 = builder.ins().nearest(pix0);
                        let piy0 = builder.ins().fmul(curr_y.0, isy_v);
                        let ry0 = builder.ins().nearest(piy0);
                        let piz0 = builder.ins().fmul(curr_z.0, isz_v);
                        let rz0 = builder.ins().nearest(piz0);
                        let cx0 = builder.ins().fmin(rx0, lx_v);
                        let rx0 = builder.ins().fmax(nlx_v, cx0);
                        let cy0 = builder.ins().fmin(ry0, ly_v);
                        let ry0 = builder.ins().fmax(nly_v, cy0);
                        let cz0 = builder.ins().fmin(rz0, lz_v);
                        let rz0 = builder.ins().fmax(nlz_v, cz0);
                        let ofx0 = builder.ins().fmul(rx0, sx_v);
                        let nx0 = builder.ins().fsub(curr_x.0, ofx0);
                        let ofy0 = builder.ins().fmul(ry0, sy_v);
                        let ny0 = builder.ins().fsub(curr_y.0, ofy0);
                        let ofz0 = builder.ins().fmul(rz0, sz_v);
                        let nz0 = builder.ins().fsub(curr_z.0, ofz0);

                        // Lane 1
                        let pix1 = builder.ins().fmul(curr_x.1, isx_v);
                        let rx1 = builder.ins().nearest(pix1);
                        let piy1 = builder.ins().fmul(curr_y.1, isy_v);
                        let ry1 = builder.ins().nearest(piy1);
                        let piz1 = builder.ins().fmul(curr_z.1, isz_v);
                        let rz1 = builder.ins().nearest(piz1);
                        let cx1 = builder.ins().fmin(rx1, lx_v);
                        let rx1 = builder.ins().fmax(nlx_v, cx1);
                        let cy1 = builder.ins().fmin(ry1, ly_v);
                        let ry1 = builder.ins().fmax(nly_v, cy1);
                        let cz1 = builder.ins().fmin(rz1, lz_v);
                        let rz1 = builder.ins().fmax(nlz_v, cz1);
                        let ofx1 = builder.ins().fmul(rx1, sx_v);
                        let nx1 = builder.ins().fsub(curr_x.1, ofx1);
                        let ofy1 = builder.ins().fmul(ry1, sy_v);
                        let ny1 = builder.ins().fsub(curr_y.1, ofy1);
                        let ofz1 = builder.ins().fmul(rz1, sz_v);
                        let nz1 = builder.ins().fsub(curr_z.1, ofz1);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        curr_z = (nz0, nz1);
                    }

                    OpCode::Elongate => {
                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Elongate,
                            params: inst.params,
                        });

                        let hx_s = builder.ins().f32const(inst.params[0]);
                        let hx = builder.ins().splat(simd_type, hx_s);
                        let hy_s = builder.ins().f32const(inst.params[1]);
                        let hy = builder.ins().splat(simd_type, hy_s);
                        let hz_s = builder.ins().f32const(inst.params[2]);
                        let hz = builder.ins().splat(simd_type, hz_s);
                        let nhx = builder.ins().fneg(hx);
                        let nhy = builder.ins().fneg(hy);
                        let nhz = builder.ins().fneg(hz);

                        // Lane 0: p - clamp(p, -h, h)
                        let cx0 = builder.ins().fmin(curr_x.0, hx);
                        let cx0 = builder.ins().fmax(nhx, cx0);
                        let cy0 = builder.ins().fmin(curr_y.0, hy);
                        let cy0 = builder.ins().fmax(nhy, cy0);
                        let cz0 = builder.ins().fmin(curr_z.0, hz);
                        let cz0 = builder.ins().fmax(nhz, cz0);
                        let nx0 = builder.ins().fsub(curr_x.0, cx0);
                        let ny0 = builder.ins().fsub(curr_y.0, cy0);
                        let nz0 = builder.ins().fsub(curr_z.0, cz0);

                        // Lane 1
                        let cx1 = builder.ins().fmin(curr_x.1, hx);
                        let cx1 = builder.ins().fmax(nhx, cx1);
                        let cy1 = builder.ins().fmin(curr_y.1, hy);
                        let cy1 = builder.ins().fmax(nhy, cy1);
                        let cz1 = builder.ins().fmin(curr_z.1, hz);
                        let cz1 = builder.ins().fmax(nhz, cz1);
                        let nx1 = builder.ins().fsub(curr_x.1, cx1);
                        let ny1 = builder.ins().fsub(curr_y.1, cy1);
                        let nz1 = builder.ins().fsub(curr_z.1, cz1);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        curr_z = (nz0, nz1);
                    }

                    OpCode::Noise => {
                        // Noise cannot be evaluated in JIT (perlin_noise_3d not available)
                        // Treat as nop — just save state for PopTransform to restore
                        coord_stack.push(CoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Noise,
                            params: inst.params,
                        });
                    }

                    OpCode::PopTransform => {
                        if let Some(state) = coord_stack.pop() {
                            match state.opcode {
                                OpCode::Round => {
                                    // d -= radius
                                    let r_s = builder.ins().f32const(state.params[0]);
                                    let r = builder.ins().splat(simd_type, r_s);
                                    let d = value_stack.pop().unwrap();
                                    let d0 = builder.ins().fsub(d.0, r);
                                    let d1 = builder.ins().fsub(d.1, r);
                                    value_stack.push((d0, d1));
                                }
                                OpCode::Onion => {
                                    // d = abs(d) - thickness
                                    let t_s = builder.ins().f32const(state.params[0]);
                                    let t = builder.ins().splat(simd_type, t_s);
                                    let d = value_stack.pop().unwrap();
                                    let abs0 = builder.ins().fabs(d.0);
                                    let abs1 = builder.ins().fabs(d.1);
                                    let d0 = builder.ins().fsub(abs0, t);
                                    let d1 = builder.ins().fsub(abs1, t);
                                    value_stack.push((d0, d1));
                                }
                                OpCode::Extrude => {
                                    // w = vec2(child_d, |original_z| - half_height)
                                    // d = min(max(w.x, w.y), 0) + length(max(w, 0))
                                    let h_s = builder.ins().f32const(state.params[0]);
                                    let h = builder.ins().splat(simd_type, h_s);
                                    let d = value_stack.pop().unwrap();

                                    // |original_z| - half_height
                                    let abs_z0 = builder.ins().fabs(state.z.0);
                                    let abs_z1 = builder.ins().fabs(state.z.1);
                                    let wy0 = builder.ins().fsub(abs_z0, h);
                                    let wy1 = builder.ins().fsub(abs_z1, h);

                                    // inside = min(max(d, w_y), 0)
                                    let inner0 = builder.ins().fmax(d.0, wy0);
                                    let inner1 = builder.ins().fmax(d.1, wy1);
                                    let inside0 = builder.ins().fmin(inner0, zero);
                                    let inside1 = builder.ins().fmin(inner1, zero);

                                    // outside = length(max(d, 0), max(w_y, 0))
                                    let mx0 = builder.ins().fmax(d.0, zero);
                                    let mx1 = builder.ins().fmax(d.1, zero);
                                    let my0 = builder.ins().fmax(wy0, zero);
                                    let my1 = builder.ins().fmax(wy1, zero);
                                    let mxx0 = builder.ins().fmul(mx0, mx0);
                                    let mxx1 = builder.ins().fmul(mx1, mx1);
                                    let myy0 = builder.ins().fmul(my0, my0);
                                    let myy1 = builder.ins().fmul(my1, my1);
                                    let len_sq0 = builder.ins().fadd(mxx0, myy0);
                                    let len_sq1 = builder.ins().fadd(mxx1, myy1);
                                    let outside0 = builder.ins().sqrt(len_sq0);
                                    let outside1 = builder.ins().sqrt(len_sq1);

                                    let r0 = builder.ins().fadd(inside0, outside0);
                                    let r1 = builder.ins().fadd(inside1, outside1);
                                    value_stack.push((r0, r1));
                                }
                                _ => {}
                            }

                            // Restore coordinates
                            curr_x = state.x;
                            curr_y = state.y;
                            curr_z = state.z;
                            curr_scale = state.scale;
                        }
                    }

                    OpCode::End => break,

                    _ => {
                        // Unimplemented opcodes: push MAX distance
                        let max_s = builder.ins().f32const(f32::MAX);
                        let max_v = builder.ins().splat(simd_type, max_s);
                        value_stack.push((max_v, max_v));
                    }
                }
            }

            // Store result
            if let Some(res) = value_stack.pop() {
                builder.ins().store(mem_flags, res.0, pout, 0);
                builder.ins().store(mem_flags, res.1, pout, 16);
            } else {
                // If stack empty, write MAX
                let max_s = builder.ins().f32const(f32::MAX);
                let max_v = builder.ins().splat(simd_type, max_s);
                builder.ins().store(mem_flags, max_v, pout, 0);
                builder.ins().store(mem_flags, max_v, pout, 16);
            }

            builder.ins().return_(&[]);
            builder.seal_block(entry_block);
            builder.finalize();
        }

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().map_err(|e| e.to_string())?;

        let code = module.get_finalized_function(func_id);

        Ok(JitSimd {
            module,
            func_ptr: code,
        })
    }

    /// Execute the JIT function on a batch of 8 points (SoA layout).
    ///
    /// # Safety
    /// - Pointers must point to valid memory with at least 8 f32 values
    /// - Pointers should be 32-byte aligned for best performance
    /// - Output pointer must be writable
    #[inline(always)]
    pub unsafe fn eval(&self, px: *const f32, py: *const f32, pz: *const f32, pout: *mut f32) {
        // SAFETY: func_ptr was compiled by Cranelift with verified extern "C" ABI signature
        // matching (px: *const f32, py: *const f32, pz: *const f32, pout: *mut f32).
        // The JITModule is held alive by the struct, preventing use-after-free.
        let func: unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32) =
            std::mem::transmute(self.func_ptr);
        func(px, py, pz, pout);
    }

    /// Evaluate many points using SoA layout
    ///
    /// # Arguments
    /// * `points` - SoA points structure
    ///
    /// # Returns
    /// Vector of distances
    pub fn eval_soa(&self, points: &crate::soa::SoAPoints) -> Vec<f32> {
        if points.is_empty() {
            return Vec::new();
        }

        let padded_len = points.padded_len();
        let mut results = vec![0.0f32; padded_len];

        let (px, py, pz) = points.as_ptrs();

        for i in (0..padded_len).step_by(8) {
            // SAFETY: Loop bounds ensure i + 8 <= padded_len. The SoA buffers are
            // allocated with padded_len elements, so ptr.add(i) stays within bounds.
            // results is allocated with padded_len elements, so the output pointer is valid.
            unsafe {
                self.eval(px.add(i), py.add(i), pz.add(i), results.as_mut_ptr().add(i));
            }
        }

        results.truncate(points.len());
        results
    }
}

#[cfg(all(test, feature = "jit"))]
mod tests {
    use super::*;
    use crate::compiled::CompiledSdf;
    use crate::eval::eval;
    use crate::soa::SoAPoints;
    use crate::types::SdfNode;
    use glam::Vec3;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_jit_simd_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);
        let jit = JitSimd::compile(&compiled).unwrap();

        let points = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.5, 0.0, 0.0),
        ];

        let soa = SoAPoints::from_vec3_slice(&points);
        let results = jit.eval_soa(&soa);

        for (i, p) in points.iter().enumerate() {
            let expected = eval(&sphere, *p);
            assert!(
                approx_eq(results[i], expected, 0.001),
                "Sphere mismatch at {}: got {}, expected {}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_box() {
        let box3d = SdfNode::box3d(1.0, 0.5, 0.5);
        let compiled = CompiledSdf::compile(&box3d);
        let jit = JitSimd::compile(&compiled).unwrap();

        let points = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.5, 0.25, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, 0.0, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
        ];

        let soa = SoAPoints::from_vec3_slice(&points);
        let results = jit.eval_soa(&soa);

        for (i, p) in points.iter().enumerate() {
            let expected = eval(&box3d, *p);
            assert!(
                approx_eq(results[i], expected, 0.001),
                "Box mismatch at {}: got {}, expected {}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_union() {
        let shape =
            SdfNode::sphere(1.0).union(SdfNode::box3d(0.5, 0.5, 0.5).translate(2.0, 0.0, 0.0));
        let compiled = CompiledSdf::compile(&shape);
        let jit = JitSimd::compile(&compiled).unwrap();

        let points = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.5, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];

        let soa = SoAPoints::from_vec3_slice(&points);
        let results = jit.eval_soa(&soa);

        for (i, p) in points.iter().enumerate() {
            let expected = eval(&shape, *p);
            assert!(
                approx_eq(results[i], expected, 0.001),
                "Union mismatch at {}: got {}, expected {}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_smooth_union() {
        let shape = SdfNode::sphere(1.0).smooth_union(SdfNode::cylinder(0.5, 1.0), 0.2);
        let compiled = CompiledSdf::compile(&shape);
        let jit = JitSimd::compile(&compiled).unwrap();

        let points = vec![
            Vec3::ZERO,
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.5, 0.0, 0.5),
            Vec3::new(0.0, 0.0, 0.5),
        ];

        let soa = SoAPoints::from_vec3_slice(&points);
        let results = jit.eval_soa(&soa);

        for (i, p) in points.iter().enumerate() {
            let expected = eval(&shape, *p);
            assert!(
                approx_eq(results[i], expected, 0.01),
                "SmoothUnion mismatch at {}: got {}, expected {}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_translate_scale() {
        let shape = SdfNode::sphere(1.0).scale(2.0).translate(1.0, 0.0, 0.0);
        let compiled = CompiledSdf::compile(&shape);
        let jit = JitSimd::compile(&compiled).unwrap();

        let points = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(1.0, 0.0, 2.0),
            Vec3::new(4.0, 0.0, 0.0),
        ];

        let soa = SoAPoints::from_vec3_slice(&points);
        let results = jit.eval_soa(&soa);

        for (i, p) in points.iter().enumerate() {
            let expected = eval(&shape, *p);
            assert!(
                approx_eq(results[i], expected, 0.01),
                "Translate+Scale mismatch at {}: got {}, expected {}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_large_batch() {
        let shape = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&shape);
        let jit = JitSimd::compile(&compiled).unwrap();

        let points: Vec<Vec3> = (0..1000)
            .map(|i| {
                let t = i as f32 / 1000.0 * std::f32::consts::TAU;
                Vec3::new(t.cos() * 2.0, t.sin() * 2.0, 0.0)
            })
            .collect();

        let soa = SoAPoints::from_vec3_slice(&points);
        let results = jit.eval_soa(&soa);

        assert_eq!(results.len(), points.len());

        for i in [0, 100, 500, 999] {
            let expected = eval(&shape, points[i]);
            assert!(
                approx_eq(results[i], expected, 0.01),
                "Large batch mismatch at {}: got {}, expected {}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_raw_eval() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);
        let jit = JitSimd::compile(&compiled).unwrap();

        let x = [0.0f32, 1.0, 2.0, 0.5, -1.0, 0.0, 0.0, 0.5];
        let y = [0.0f32; 8];
        let z = [0.0f32; 8];
        let mut out = [0.0f32; 8];

        // SAFETY: All arrays are [f32; 8], providing exactly 8 contiguous elements
        // as required by the JIT-compiled SIMD function.
        unsafe {
            jit.eval(x.as_ptr(), y.as_ptr(), z.as_ptr(), out.as_mut_ptr());
        }

        let expected = [-1.0, 0.0, 1.0, -0.5, 0.0, -1.0, -1.0, -0.5];
        for i in 0..8 {
            assert!(
                approx_eq(out[i], expected[i], 0.001),
                "Raw eval mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected[i]
            );
        }
    }
}
