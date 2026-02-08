//! JIT + SIMD Integration: Deep Fried Crispy Edition (FMA + Reciprocal Mul + Zero Copy)
//!
//! This module generates native SIMD machine code using Cranelift,
//! optimized for maximum throughput by eliminating division and memory overhead.
//!
//! # Optimizations
//!
//! - **Division Exorcism**: Replaces `div` with `mul(1/k)` for Smooth ops and Scale.
//! - **FMA**: Fused Multiply-Add for dot products and length calculations.
//! - **Trusted Alignment**: Assumes 16-byte aligned input pointers for faster loads.
//! - **Zero-Copy Batching**: Direct pointer arithmetic for batch processing.
//!
//! # Performance
//!
//! | Method | Dispatch | SIMD | Expected Speedup |
//! |--------|----------|------|------------------|
//! | Interpreter | match loop | Software | 1x |
//! | SIMD Interpreter | match loop | f32x8 | ~6x |
//! | JIT Scalar | Native | None | ~3x |
//! | JIT + SIMD | Native | Hardware | ~15-20x |
//! | **Deep Fried** | **Native + No Div** | **Hardware** | **~25-35x** |
//!
//! Author: Moroya Sakamoto

use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlags, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::mem;

use super::super::opcode::OpCode;
use super::super::CompiledSdf;

/// Function signature for SIMD JIT: processes 8 points at once
type SimdSdfFn =
    unsafe extern "C" fn(px: *const f32, py: *const f32, pz: *const f32, out: *mut f32);

/// JIT-compiled SIMD SDF evaluator (Deep Fried Crispy Edition)
pub struct JitSimdSdf {
    #[allow(dead_code)]
    module: JITModule,
    func_ptr: *const u8,
}

unsafe impl Send for JitSimdSdf {}
unsafe impl Sync for JitSimdSdf {}

/// SIMD coordinate state (2 lanes of F32X4)
struct SimdCoordState {
    x: (Value, Value),
    y: (Value, Value),
    z: (Value, Value),
    scale: (Value, Value),
    opcode: OpCode,
    params: [f32; 4],
    /// Flag indicating this transform was folded (no-op)
    folded: bool,
}

/// Epsilon for constant folding
const FOLD_EPSILON: f32 = 1e-6;

/// FMA-optimized SIMD 2D length: sqrt(x² + y²)
fn simd_length2_fma(builder: &mut FunctionBuilder, x: Value, y: Value) -> Value {
    let yy = builder.ins().fmul(y, y);
    let len_sq = builder.ins().fma(x, x, yy);
    builder.ins().sqrt(len_sq)
}

/// FMA-optimized SIMD 3D length: sqrt(x² + y² + z²)
fn simd_length3_fma(builder: &mut FunctionBuilder, x: Value, y: Value, z: Value) -> Value {
    let zz = builder.ins().fmul(z, z);
    let yy_zz = builder.ins().fma(y, y, zz);
    let len_sq = builder.ins().fma(x, x, yy_zz);
    builder.ins().sqrt(len_sq)
}

/// Branchless SIMD select: returns if_neg where cond < 0, else if_pos
fn simd_select_neg(
    builder: &mut FunctionBuilder,
    cond: Value,
    if_neg: Value,
    if_pos: Value,
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

/// SIMD Taylor-series sin/cos approximation for F32X4 vectors.
/// sin(x) ≈ x - x³/6 + x⁵/120, cos(x) ≈ 1 - x²/2 + x⁴/24
fn simd_sincos_approx(
    builder: &mut FunctionBuilder,
    angle: Value,
    vec_type: types::Type,
) -> (Value, Value) {
    let one_s = builder.ins().f32const(1.0);
    let half_s = builder.ins().f32const(0.5);
    let sixth_s = builder.ins().f32const(1.0 / 6.0);
    let tf_s = builder.ins().f32const(1.0 / 24.0);
    let ot_s = builder.ins().f32const(1.0 / 120.0);
    let one_v = builder.ins().splat(vec_type, one_s);
    let half_v = builder.ins().splat(vec_type, half_s);
    let sixth_v = builder.ins().splat(vec_type, sixth_s);
    let tf_v = builder.ins().splat(vec_type, tf_s);
    let ot_v = builder.ins().splat(vec_type, ot_s);

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

/// Emit SIMD stairs_min for one F32X4 lane (Hardcoded mode — constants baked in)
fn emit_simd_stairs_min(
    builder: &mut FunctionBuilder,
    vec_type: types::Type,
    a: Value,
    b: Value,
    r: f32,
    n: f32,
) -> Value {
    let half_s = builder.ins().f32const(0.5);
    let half = builder.ins().splat(vec_type, half_s);
    let s_s = builder.ins().f32const(std::f32::consts::FRAC_1_SQRT_2);
    let s_v = builder.ins().splat(vec_type, s_s);
    let s2_s = builder.ins().f32const(std::f32::consts::SQRT_2);
    let s2_v = builder.ins().splat(vec_type, s2_s);

    let rn_f = r / n;
    let rn_s = builder.ins().f32const(rn_f);
    let rn_v = builder.ins().splat(vec_type, rn_s);

    let off_f = (r - rn_f) * 0.5 * std::f32::consts::SQRT_2;
    let off_s = builder.ins().f32const(off_f);
    let off_v = builder.ins().splat(vec_type, off_s);

    let step_f = r * std::f32::consts::SQRT_2 / n;
    let step_s = builder.ins().f32const(step_f);
    let step_v = builder.ins().splat(vec_type, step_s);

    // px = (b - a) * s - off
    let b_minus_a = builder.ins().fsub(b, a);
    let px_rot = builder.ins().fmul(b_minus_a, s_v);
    let px = builder.ins().fsub(px_rot, off_v);
    // py = (a + b) * s - off
    let a_plus_b = builder.ins().fadd(a, b);
    let py_rot = builder.ins().fmul(a_plus_b, s_v);
    let py = builder.ins().fsub(py_rot, off_v);

    // pMod1: px2 = px + 0.5 * sqrt(2) * rn
    let half_s2 = builder.ins().fmul(half, s2_v);
    let half_s2_rn = builder.ins().fmul(half_s2, rn_v);
    let px2 = builder.ins().fadd(px, half_s2_rn);
    let step_half = builder.ins().fmul(step_v, half);
    let t = builder.ins().fadd(px2, step_half);

    // px3 = t - step * floor(t / step) - step * 0.5
    let t_div_step = builder.ins().fdiv(t, step_v);
    let t_floor = builder.ins().floor(t_div_step);
    let t_mod = builder.ins().fmul(step_v, t_floor);
    let t_sub = builder.ins().fsub(t, t_mod);
    let px3 = builder.ins().fsub(t_sub, step_half);

    // d2 = min(min(a, b), py)
    let min_ab = builder.ins().fmin(a, b);
    let d2 = builder.ins().fmin(min_ab, py);

    // Rotate back
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

/// Emit SIMD stairs_min for one F32X4 lane using pre-splatted constants
fn emit_simd_stairs_min_lane(
    builder: &mut FunctionBuilder,
    a: Value,
    b: Value,
    s_v: Value,
    half: Value,
    s2_v: Value,
    rn_v: Value,
    off_v: Value,
    step_v: Value,
) -> Value {
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

impl JitSimdSdf {
    /// Compile SDF to native SIMD machine code with Aggressive optimizations
    pub fn compile(sdf: &CompiledSdf) -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| e.to_string())?;

        // Use colocated libcalls to avoid PLT overhead
        flag_builder
            .set("use_colocated_libcalls", "true")
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

        let ptr_type = module.target_config().pointer_type();
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // px
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // py
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // pz
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // out

        let func_id = module
            .declare_function(
                "eval_sdf_simd_deep_fried",
                Linkage::Export,
                &ctx.func.signature,
            )
            .map_err(|e| e.to_string())?;

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let ptr_x = builder.block_params(entry_block)[0];
            let ptr_y = builder.block_params(entry_block)[1];
            let ptr_z = builder.block_params(entry_block)[2];
            let ptr_out = builder.block_params(entry_block)[3];

            let vec_type = types::F32X4;

            // Trusted Alignment: Assume caller provides aligned pointers
            let mut mem_flags = MemFlags::new();
            mem_flags.set_aligned();
            mem_flags.set_notrap();

            // Load inputs (8 floats -> 2 x F32X4)
            let x0 = builder.ins().load(vec_type, mem_flags, ptr_x, 0);
            let y0 = builder.ins().load(vec_type, mem_flags, ptr_y, 0);
            let z0 = builder.ins().load(vec_type, mem_flags, ptr_z, 0);
            let x1 = builder.ins().load(vec_type, mem_flags, ptr_x, 16);
            let y1 = builder.ins().load(vec_type, mem_flags, ptr_y, 16);
            let z1 = builder.ins().load(vec_type, mem_flags, ptr_z, 16);

            // Constants
            let zero_s = builder.ins().f32const(0.0);
            let zero_vec = builder.ins().splat(vec_type, zero_s);
            let one_s = builder.ins().f32const(1.0);
            let one_vec = builder.ins().splat(vec_type, one_s);

            // State
            let mut value_stack: Vec<(Value, Value)> = Vec::with_capacity(64);
            let mut coord_stack: Vec<SimdCoordState> = Vec::with_capacity(32);

            let mut curr_x = (x0, x1);
            let mut curr_y = (y0, y1);
            let mut curr_z = (z0, z1);
            let mut curr_scale = (one_vec, one_vec);

            for inst in &sdf.instructions {
                match inst.opcode {
                    OpCode::Sphere => {
                        let r_s = builder.ins().f32const(inst.params[0]);
                        let r = builder.ins().splat(vec_type, r_s);

                        // FMA optimized length
                        let zz0 = builder.ins().fmul(curr_z.0, curr_z.0);
                        let yy_zz0 = builder.ins().fma(curr_y.0, curr_y.0, zz0);
                        let len_sq0 = builder.ins().fma(curr_x.0, curr_x.0, yy_zz0);
                        let len0 = builder.ins().sqrt(len_sq0);
                        let d0 = builder.ins().fsub(len0, r);
                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);

                        let zz1 = builder.ins().fmul(curr_z.1, curr_z.1);
                        let yy_zz1 = builder.ins().fma(curr_y.1, curr_y.1, zz1);
                        let len_sq1 = builder.ins().fma(curr_x.1, curr_x.1, yy_zz1);
                        let len1 = builder.ins().sqrt(len_sq1);
                        let d1 = builder.ins().fsub(len1, r);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);

                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Box3d => {
                        let hx_s = builder.ins().f32const(inst.params[0]);
                        let hy_s = builder.ins().f32const(inst.params[1]);
                        let hz_s = builder.ins().f32const(inst.params[2]);
                        let hx = builder.ins().splat(vec_type, hx_s);
                        let hy = builder.ins().splat(vec_type, hy_s);
                        let hz = builder.ins().splat(vec_type, hz_s);

                        // Lane 0
                        let ax0 = builder.ins().fabs(curr_x.0);
                        let ay0 = builder.ins().fabs(curr_y.0);
                        let az0 = builder.ins().fabs(curr_z.0);
                        let qx0 = builder.ins().fsub(ax0, hx);
                        let qy0 = builder.ins().fsub(ay0, hy);
                        let qz0 = builder.ins().fsub(az0, hz);
                        let mx0 = builder.ins().fmax(qx0, zero_vec);
                        let my0 = builder.ins().fmax(qy0, zero_vec);
                        let mz0 = builder.ins().fmax(qz0, zero_vec);

                        let mzz0 = builder.ins().fmul(mz0, mz0);
                        let myy_mzz0 = builder.ins().fma(my0, my0, mzz0);
                        let len_sq0 = builder.ins().fma(mx0, mx0, myy_mzz0);
                        let outside0 = builder.ins().sqrt(len_sq0);

                        let max_yz0 = builder.ins().fmax(qy0, qz0);
                        let max_xyz0 = builder.ins().fmax(qx0, max_yz0);
                        let inside0 = builder.ins().fmin(max_xyz0, zero_vec);
                        let d0 = builder.ins().fadd(outside0, inside0);
                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);

                        // Lane 1
                        let ax1 = builder.ins().fabs(curr_x.1);
                        let ay1 = builder.ins().fabs(curr_y.1);
                        let az1 = builder.ins().fabs(curr_z.1);
                        let qx1 = builder.ins().fsub(ax1, hx);
                        let qy1 = builder.ins().fsub(ay1, hy);
                        let qz1 = builder.ins().fsub(az1, hz);
                        let mx1 = builder.ins().fmax(qx1, zero_vec);
                        let my1 = builder.ins().fmax(qy1, zero_vec);
                        let mz1 = builder.ins().fmax(qz1, zero_vec);

                        let mzz1 = builder.ins().fmul(mz1, mz1);
                        let myy_mzz1 = builder.ins().fma(my1, my1, mzz1);
                        let len_sq1 = builder.ins().fma(mx1, mx1, myy_mzz1);
                        let outside1 = builder.ins().sqrt(len_sq1);

                        let max_yz1 = builder.ins().fmax(qy1, qz1);
                        let max_xyz1 = builder.ins().fmax(qx1, max_yz1);
                        let inside1 = builder.ins().fmin(max_xyz1, zero_vec);
                        let d1 = builder.ins().fadd(outside1, inside1);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);

                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Cylinder => {
                        let r_s = builder.ins().f32const(inst.params[0]);
                        let h_s = builder.ins().f32const(inst.params[1]);
                        let r = builder.ins().splat(vec_type, r_s);
                        let h = builder.ins().splat(vec_type, h_s);

                        // Lane 0
                        let zz0 = builder.ins().fmul(curr_z.0, curr_z.0);
                        let xz_sq0 = builder.ins().fma(curr_x.0, curr_x.0, zz0);
                        let xz_len0 = builder.ins().sqrt(xz_sq0);
                        let dx0 = builder.ins().fsub(xz_len0, r);
                        let ay0 = builder.ins().fabs(curr_y.0);
                        let dy0 = builder.ins().fsub(ay0, h);

                        let mx0 = builder.ins().fmax(dx0, zero_vec);
                        let my0 = builder.ins().fmax(dy0, zero_vec);
                        let mzz0 = builder.ins().fmul(my0, my0);
                        let len_sq0 = builder.ins().fma(mx0, mx0, mzz0);
                        let outside0 = builder.ins().sqrt(len_sq0);

                        let im0 = builder.ins().fmax(dx0, dy0);
                        let inside0 = builder.ins().fmin(im0, zero_vec);
                        let d0 = builder.ins().fadd(outside0, inside0);
                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);

                        // Lane 1
                        let zz1 = builder.ins().fmul(curr_z.1, curr_z.1);
                        let xz_sq1 = builder.ins().fma(curr_x.1, curr_x.1, zz1);
                        let xz_len1 = builder.ins().sqrt(xz_sq1);
                        let dx1 = builder.ins().fsub(xz_len1, r);
                        let ay1 = builder.ins().fabs(curr_y.1);
                        let dy1 = builder.ins().fsub(ay1, h);

                        let mx1 = builder.ins().fmax(dx1, zero_vec);
                        let my1 = builder.ins().fmax(dy1, zero_vec);
                        let mzz1 = builder.ins().fmul(my1, my1);
                        let len_sq1 = builder.ins().fma(mx1, mx1, mzz1);
                        let outside1 = builder.ins().sqrt(len_sq1);

                        let im1 = builder.ins().fmax(dx1, dy1);
                        let inside1 = builder.ins().fmin(im1, zero_vec);
                        let d1 = builder.ins().fadd(outside1, inside1);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);

                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Plane => {
                        let nx_s = builder.ins().f32const(inst.params[0]);
                        let ny_s = builder.ins().f32const(inst.params[1]);
                        let nz_s = builder.ins().f32const(inst.params[2]);
                        let dist_s = builder.ins().f32const(inst.params[3]);
                        let nx = builder.ins().splat(vec_type, nx_s);
                        let ny = builder.ins().splat(vec_type, ny_s);
                        let nz = builder.ins().splat(vec_type, nz_s);
                        let dist = builder.ins().splat(vec_type, dist_s);

                        let z_nz0 = builder.ins().fmul(curr_z.0, nz);
                        let y_ny0 = builder.ins().fma(curr_y.0, ny, z_nz0);
                        let dot0 = builder.ins().fma(curr_x.0, nx, y_ny0);
                        let d0 = builder.ins().fadd(dot0, dist);
                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);

                        let z_nz1 = builder.ins().fmul(curr_z.1, nz);
                        let y_ny1 = builder.ins().fma(curr_y.1, ny, z_nz1);
                        let dot1 = builder.ins().fma(curr_x.1, nx, y_ny1);
                        let d1 = builder.ins().fadd(dot1, dist);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);

                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Torus => {
                        let maj_s = builder.ins().f32const(inst.params[0]);
                        let min_s = builder.ins().f32const(inst.params[1]);
                        let maj = builder.ins().splat(vec_type, maj_s);
                        let min = builder.ins().splat(vec_type, min_s);

                        // Lane 0
                        let zz0 = builder.ins().fmul(curr_z.0, curr_z.0);
                        let xz_sq0 = builder.ins().fma(curr_x.0, curr_x.0, zz0);
                        let xz0 = builder.ins().sqrt(xz_sq0);
                        let qx0 = builder.ins().fsub(xz0, maj);
                        let yy0 = builder.ins().fmul(curr_y.0, curr_y.0);
                        let q_sq0 = builder.ins().fma(qx0, qx0, yy0);
                        let q0 = builder.ins().sqrt(q_sq0);
                        let d0 = builder.ins().fsub(q0, min);
                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);

                        // Lane 1
                        let zz1 = builder.ins().fmul(curr_z.1, curr_z.1);
                        let xz_sq1 = builder.ins().fma(curr_x.1, curr_x.1, zz1);
                        let xz1 = builder.ins().sqrt(xz_sq1);
                        let qx1 = builder.ins().fsub(xz1, maj);
                        let yy1 = builder.ins().fmul(curr_y.1, curr_y.1);
                        let q_sq1 = builder.ins().fma(qx1, qx1, yy1);
                        let q1 = builder.ins().sqrt(q_sq1);
                        let d1 = builder.ins().fsub(q1, min);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);

                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Capsule => {
                        let ax_s = builder.ins().f32const(inst.params[0]);
                        let ay_s = builder.ins().f32const(inst.params[1]);
                        let az_s = builder.ins().f32const(inst.params[2]);
                        let radius = inst.get_capsule_radius();
                        let r_s = builder.ins().f32const(radius);
                        let ax = builder.ins().splat(vec_type, ax_s);
                        let ay = builder.ins().splat(vec_type, ay_s);
                        let az = builder.ins().splat(vec_type, az_s);
                        let r_v = builder.ins().splat(vec_type, r_s);

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
                        let bax = builder.ins().splat(vec_type, bax_s);
                        let bay = builder.ins().splat(vec_type, bay_s);
                        let baz = builder.ins().splat(vec_type, baz_s);
                        let inv_bd = builder.ins().splat(vec_type, inv_bd_s);

                        // Lane 0: pa = p - a, h = clamp(dot(pa,ba) * inv_ba_dot, 0, 1)
                        let pax0 = builder.ins().fsub(curr_x.0, ax);
                        let pay0 = builder.ins().fsub(curr_y.0, ay);
                        let paz0 = builder.ins().fsub(curr_z.0, az);
                        let dot_z0 = builder.ins().fmul(paz0, baz);
                        let dot_yz0 = builder.ins().fma(pay0, bay, dot_z0);
                        let dot0 = builder.ins().fma(pax0, bax, dot_yz0);
                        let h_raw0 = builder.ins().fmul(dot0, inv_bd);
                        let h_min0 = builder.ins().fmin(h_raw0, one_vec);
                        let h0 = builder.ins().fmax(h_min0, zero_vec);
                        let bhx0 = builder.ins().fmul(bax, h0);
                        let bhy0 = builder.ins().fmul(bay, h0);
                        let bhz0 = builder.ins().fmul(baz, h0);
                        let dx0 = builder.ins().fsub(pax0, bhx0);
                        let dy0 = builder.ins().fsub(pay0, bhy0);
                        let dz0 = builder.ins().fsub(paz0, bhz0);
                        let len0 = simd_length3_fma(&mut builder, dx0, dy0, dz0);
                        let d0 = builder.ins().fsub(len0, r_v);
                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);

                        // Lane 1
                        let pax1 = builder.ins().fsub(curr_x.1, ax);
                        let pay1 = builder.ins().fsub(curr_y.1, ay);
                        let paz1 = builder.ins().fsub(curr_z.1, az);
                        let dot_z1 = builder.ins().fmul(paz1, baz);
                        let dot_yz1 = builder.ins().fma(pay1, bay, dot_z1);
                        let dot1 = builder.ins().fma(pax1, bax, dot_yz1);
                        let h_raw1 = builder.ins().fmul(dot1, inv_bd);
                        let h_min1 = builder.ins().fmin(h_raw1, one_vec);
                        let h1 = builder.ins().fmax(h_min1, zero_vec);
                        let bhx1 = builder.ins().fmul(bax, h1);
                        let bhy1 = builder.ins().fmul(bay, h1);
                        let bhz1 = builder.ins().fmul(baz, h1);
                        let dx1 = builder.ins().fsub(pax1, bhx1);
                        let dy1 = builder.ins().fsub(pay1, bhy1);
                        let dz1 = builder.ins().fsub(paz1, bhz1);
                        let len1 = simd_length3_fma(&mut builder, dx1, dy1, dz1);
                        let d1 = builder.ins().fsub(len1, r_v);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);

                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Cone => {
                        let radius_val = inst.params[0];
                        let half_height = inst.params[1];

                        let r_s = builder.ins().f32const(radius_val);
                        let h_s = builder.ins().f32const(half_height);
                        let r = builder.ins().splat(vec_type, r_s);
                        let h = builder.ins().splat(vec_type, h_s);

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
                        let k2x = builder.ins().splat(vec_type, k2x_s);
                        let k2y = builder.ins().splat(vec_type, k2y_s);
                        let inv_k2d = builder.ins().splat(vec_type, inv_k2d_s);
                        let neg_one_s = builder.ins().f32const(-1.0);
                        let neg_one_v = builder.ins().splat(vec_type, neg_one_s);

                        // Lane 0
                        let q_x0 = simd_length2_fma(&mut builder, curr_x.0, curr_z.0);
                        let q_y0 = curr_y.0;
                        let ca_r0 = simd_select_neg(&mut builder, q_y0, r, zero_vec);
                        let min_q_ca0 = builder.ins().fmin(q_x0, ca_r0);
                        let ca_x0 = builder.ins().fsub(q_x0, min_q_ca0);
                        let abs_qy0 = builder.ins().fabs(q_y0);
                        let ca_y0 = builder.ins().fsub(abs_qy0, h);
                        let neg_qx0 = builder.ins().fneg(q_x0);
                        let diff_y0 = builder.ins().fsub(h, q_y0);
                        let dy_k2y0 = builder.ins().fmul(diff_y0, k2y);
                        let nqx_k2x0 = builder.ins().fma(neg_qx0, k2x, dy_k2y0);
                        let t_raw0 = builder.ins().fmul(nqx_k2x0, inv_k2d);
                        let t_min0 = builder.ins().fmin(t_raw0, one_vec);
                        let t0 = builder.ins().fmax(zero_vec, t_min0);
                        let k2x_t0 = builder.ins().fmul(k2x, t0);
                        let cb_x0 = builder.ins().fadd(q_x0, k2x_t0);
                        let qy_h0 = builder.ins().fsub(q_y0, h);
                        let k2y_t0 = builder.ins().fmul(k2y, t0);
                        let cb_y0 = builder.ins().fadd(qy_h0, k2y_t0);
                        let both_neg_cond0 = builder.ins().fmax(cb_x0, ca_y0);
                        let s0 = simd_select_neg(&mut builder, both_neg_cond0, neg_one_v, one_vec);
                        let ca_sq0 = {
                            let xx = builder.ins().fmul(ca_x0, ca_x0);
                            builder.ins().fma(ca_y0, ca_y0, xx)
                        };
                        let cb_sq0 = {
                            let xx = builder.ins().fmul(cb_x0, cb_x0);
                            builder.ins().fma(cb_y0, cb_y0, xx)
                        };
                        let d2_0 = builder.ins().fmin(ca_sq0, cb_sq0);
                        let dist0 = builder.ins().sqrt(d2_0);
                        let d0 = builder.ins().fmul(s0, dist0);

                        // Lane 1
                        let q_x1 = simd_length2_fma(&mut builder, curr_x.1, curr_z.1);
                        let q_y1 = curr_y.1;
                        let ca_r1 = simd_select_neg(&mut builder, q_y1, r, zero_vec);
                        let min_q_ca1 = builder.ins().fmin(q_x1, ca_r1);
                        let ca_x1 = builder.ins().fsub(q_x1, min_q_ca1);
                        let abs_qy1 = builder.ins().fabs(q_y1);
                        let ca_y1 = builder.ins().fsub(abs_qy1, h);
                        let neg_qx1 = builder.ins().fneg(q_x1);
                        let diff_y1 = builder.ins().fsub(h, q_y1);
                        let dy_k2y1 = builder.ins().fmul(diff_y1, k2y);
                        let nqx_k2x1 = builder.ins().fma(neg_qx1, k2x, dy_k2y1);
                        let t_raw1 = builder.ins().fmul(nqx_k2x1, inv_k2d);
                        let t_min1 = builder.ins().fmin(t_raw1, one_vec);
                        let t1 = builder.ins().fmax(zero_vec, t_min1);
                        let k2x_t1 = builder.ins().fmul(k2x, t1);
                        let cb_x1 = builder.ins().fadd(q_x1, k2x_t1);
                        let qy_h1 = builder.ins().fsub(q_y1, h);
                        let k2y_t1 = builder.ins().fmul(k2y, t1);
                        let cb_y1 = builder.ins().fadd(qy_h1, k2y_t1);
                        let both_neg_cond1 = builder.ins().fmax(cb_x1, ca_y1);
                        let s1 = simd_select_neg(&mut builder, both_neg_cond1, neg_one_v, one_vec);
                        let ca_sq1 = {
                            let xx = builder.ins().fmul(ca_x1, ca_x1);
                            builder.ins().fma(ca_y1, ca_y1, xx)
                        };
                        let cb_sq1 = {
                            let xx = builder.ins().fmul(cb_x1, cb_x1);
                            builder.ins().fma(cb_y1, cb_y1, xx)
                        };
                        let d2_1 = builder.ins().fmin(ca_sq1, cb_sq1);
                        let dist1 = builder.ins().sqrt(d2_1);
                        let d1 = builder.ins().fmul(s1, dist1);

                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Ellipsoid => {
                        let inv_rx_s = builder.ins().f32const(1.0 / inst.params[0]);
                        let inv_ry_s = builder.ins().f32const(1.0 / inst.params[1]);
                        let inv_rz_s = builder.ins().f32const(1.0 / inst.params[2]);
                        let inv_rx = builder.ins().splat(vec_type, inv_rx_s);
                        let inv_ry = builder.ins().splat(vec_type, inv_ry_s);
                        let inv_rz = builder.ins().splat(vec_type, inv_rz_s);

                        let inv_rx2_s = builder
                            .ins()
                            .f32const(1.0 / (inst.params[0] * inst.params[0]));
                        let inv_ry2_s = builder
                            .ins()
                            .f32const(1.0 / (inst.params[1] * inst.params[1]));
                        let inv_rz2_s = builder
                            .ins()
                            .f32const(1.0 / (inst.params[2] * inst.params[2]));
                        let inv_rx2 = builder.ins().splat(vec_type, inv_rx2_s);
                        let inv_ry2 = builder.ins().splat(vec_type, inv_ry2_s);
                        let inv_rz2 = builder.ins().splat(vec_type, inv_rz2_s);

                        let eps_s = builder.ins().f32const(1e-10);
                        let eps = builder.ins().splat(vec_type, eps_s);

                        // Lane 0: k0 = length(p / radii)
                        let px0 = builder.ins().fmul(curr_x.0, inv_rx);
                        let py0 = builder.ins().fmul(curr_y.0, inv_ry);
                        let pz0 = builder.ins().fmul(curr_z.0, inv_rz);
                        let k0_0 = simd_length3_fma(&mut builder, px0, py0, pz0);
                        // k1 = length(p / (radii²))
                        let qx0 = builder.ins().fmul(curr_x.0, inv_rx2);
                        let qy0 = builder.ins().fmul(curr_y.0, inv_ry2);
                        let qz0 = builder.ins().fmul(curr_z.0, inv_rz2);
                        let k1_0 = simd_length3_fma(&mut builder, qx0, qy0, qz0);
                        // d = k0 * (k0 - 1) / (k1 + eps)
                        let k1_safe0 = builder.ins().fadd(k1_0, eps);
                        let k0_m1_0 = builder.ins().fsub(k0_0, one_vec);
                        let num0 = builder.ins().fmul(k0_0, k0_m1_0);
                        let d0 = builder.ins().fdiv(num0, k1_safe0);

                        // Lane 1
                        let px1 = builder.ins().fmul(curr_x.1, inv_rx);
                        let py1 = builder.ins().fmul(curr_y.1, inv_ry);
                        let pz1 = builder.ins().fmul(curr_z.1, inv_rz);
                        let k0_1 = simd_length3_fma(&mut builder, px1, py1, pz1);
                        let qx1 = builder.ins().fmul(curr_x.1, inv_rx2);
                        let qy1 = builder.ins().fmul(curr_y.1, inv_ry2);
                        let qz1 = builder.ins().fmul(curr_z.1, inv_rz2);
                        let k1_1 = simd_length3_fma(&mut builder, qx1, qy1, qz1);
                        let k1_safe1 = builder.ins().fadd(k1_1, eps);
                        let k0_m1_1 = builder.ins().fsub(k0_1, one_vec);
                        let num1 = builder.ins().fmul(k0_1, k0_m1_1);
                        let d1 = builder.ins().fdiv(num1, k1_safe1);

                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((d0_scaled, d1_scaled));
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

                        let r1_v = builder.ins().splat(vec_type, r1_s);
                        let r2_v = builder.ins().splat(vec_type, r2_s);
                        let hh_v = builder.ins().splat(vec_type, hh_s);
                        let h_v = builder.ins().splat(vec_type, h_s);
                        let b_v = builder.ins().splat(vec_type, b_s);
                        let a_v = builder.ins().splat(vec_type, a_s);
                        let ah_v = builder.ins().splat(vec_type, ah_s);
                        let neg_b = builder.ins().fneg(b_v);

                        // Lane 0
                        let q_x0 = simd_length2_fma(&mut builder, curr_x.0, curr_z.0);
                        let q_y0 = builder.ins().fadd(curr_y.0, hh_v);
                        let qx_nb0 = builder.ins().fmul(q_x0, neg_b);
                        let k0 = builder.ins().fma(q_y0, a_v, qx_nb0);
                        let len1_0 = simd_length2_fma(&mut builder, q_x0, q_y0);
                        let d1_0 = builder.ins().fsub(len1_0, r1_v);
                        let qy_h0 = builder.ins().fsub(q_y0, h_v);
                        let len2_0 = simd_length2_fma(&mut builder, q_x0, qy_h0);
                        let d2_0 = builder.ins().fsub(len2_0, r2_v);
                        let qyb0 = builder.ins().fmul(q_y0, b_v);
                        let d3_0 = builder.ins().fma(q_x0, a_v, qyb0);
                        let d3_0 = builder.ins().fsub(d3_0, r1_v);
                        let k_ah0 = builder.ins().fsub(ah_v, k0);
                        let inner0 = simd_select_neg(&mut builder, k_ah0, d2_0, d3_0);
                        let d0 = simd_select_neg(&mut builder, k0, d1_0, inner0);

                        // Lane 1
                        let q_x1 = simd_length2_fma(&mut builder, curr_x.1, curr_z.1);
                        let q_y1 = builder.ins().fadd(curr_y.1, hh_v);
                        let qx_nb1 = builder.ins().fmul(q_x1, neg_b);
                        let k1 = builder.ins().fma(q_y1, a_v, qx_nb1);
                        let len1_1 = simd_length2_fma(&mut builder, q_x1, q_y1);
                        let d1_1 = builder.ins().fsub(len1_1, r1_v);
                        let qy_h1 = builder.ins().fsub(q_y1, h_v);
                        let len2_1 = simd_length2_fma(&mut builder, q_x1, qy_h1);
                        let d2_1 = builder.ins().fsub(len2_1, r2_v);
                        let qyb1 = builder.ins().fmul(q_y1, b_v);
                        let d3_1 = builder.ins().fma(q_x1, a_v, qyb1);
                        let d3_1 = builder.ins().fsub(d3_1, r1_v);
                        let k_ah1 = builder.ins().fsub(ah_v, k1);
                        let inner1 = simd_select_neg(&mut builder, k_ah1, d2_1, d3_1);
                        let d1 = simd_select_neg(&mut builder, k1, d1_1, inner1);

                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((d0_scaled, d1_scaled));
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

                        let hh_v = builder.ins().splat(vec_type, hh_s);
                        let h_v = builder.ins().splat(vec_type, h_s);
                        let m2_v = builder.ins().splat(vec_type, m2_s);
                        let inv_m2_v = builder.ins().splat(vec_type, inv_m2_s);
                        let inv_m2_025_v = builder.ins().splat(vec_type, inv_m2_025_s);
                        let half_v = builder.ins().splat(vec_type, half_s);
                        let neg_half_v = builder.ins().splat(vec_type, neg_half_s);
                        let neg_one_v = builder.ins().splat(vec_type, neg_one_s);

                        // Lane 0
                        let py0 = builder.ins().fadd(curr_y.0, hh_v);
                        let abs_px0 = builder.ins().fabs(curr_x.0);
                        let abs_pz0 = builder.ins().fabs(curr_z.0);
                        let px_s0 = builder.ins().fmax(abs_px0, abs_pz0);
                        let pz_s0 = builder.ins().fmin(abs_px0, abs_pz0);
                        let px_adj0 = builder.ins().fsub(px_s0, half_v);
                        let pz_adj0 = builder.ins().fsub(pz_s0, half_v);
                        let qx0 = pz_adj0;
                        let nhalf_px0 = builder.ins().fmul(neg_half_v, px_adj0);
                        let qy0 = builder.ins().fma(h_v, py0, nhalf_px0);
                        let half_py0 = builder.ins().fmul(half_v, py0);
                        let qz0 = builder.ins().fma(h_v, px_adj0, half_py0);
                        let neg_qx0 = builder.ins().fneg(qx0);
                        let s0 = builder.ins().fmax(neg_qx0, zero_vec);
                        let half_pz0 = builder.ins().fmul(half_v, pz_adj0);
                        let qy_sub0 = builder.ins().fsub(qy0, half_pz0);
                        let t_raw0 = builder.ins().fmul(qy_sub0, inv_m2_025_v);
                        let t_min0 = builder.ins().fmin(t_raw0, one_vec);
                        let t0 = builder.ins().fmax(zero_vec, t_min0);
                        let qx_s0 = builder.ins().fadd(qx0, s0);
                        let qx_s_sq0 = builder.ins().fmul(qx_s0, qx_s0);
                        let m2_qxs0 = builder.ins().fmul(m2_v, qx_s_sq0);
                        let a0 = builder.ins().fma(qy0, qy0, m2_qxs0);
                        let half_t0 = builder.ins().fmul(half_v, t0);
                        let qx_ht0 = builder.ins().fadd(qx0, half_t0);
                        let qx_ht_sq0 = builder.ins().fmul(qx_ht0, qx_ht0);
                        let m2_t0 = builder.ins().fmul(m2_v, t0);
                        let qy_m2t0 = builder.ins().fsub(qy0, m2_t0);
                        let m2_qxht0 = builder.ins().fmul(m2_v, qx_ht_sq0);
                        let b0 = builder.ins().fma(qy_m2t0, qy_m2t0, m2_qxht0);
                        let neg_qx_m2_0 = builder.ins().fmul(neg_qx0, m2_v);
                        let half_qy0 = builder.ins().fmul(half_v, qy0);
                        let cond0 = builder.ins().fsub(neg_qx_m2_0, half_qy0);
                        let min_cond0 = builder.ins().fmin(qy0, cond0);
                        let ab_min0 = builder.ins().fmin(a0, b0);
                        let neg_min_cond0 = builder.ins().fneg(min_cond0);
                        let d2_0 = simd_select_neg(&mut builder, neg_min_cond0, zero_vec, ab_min0);
                        let qz_sq0 = builder.ins().fmul(qz0, qz0);
                        let d2_qz0 = builder.ins().fadd(d2_0, qz_sq0);
                        let d2_sc0 = builder.ins().fmul(d2_qz0, inv_m2_v);
                        let dist0 = builder.ins().sqrt(d2_sc0);
                        let neg_py0 = builder.ins().fneg(py0);
                        let sign_arg0 = builder.ins().fmax(qz0, neg_py0);
                        let signed0 = simd_select_neg(&mut builder, sign_arg0, neg_one_v, one_vec);
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
                        let qy1 = builder.ins().fma(h_v, py1, nhalf_px1);
                        let half_py1 = builder.ins().fmul(half_v, py1);
                        let qz1 = builder.ins().fma(h_v, px_adj1, half_py1);
                        let neg_qx1 = builder.ins().fneg(qx1);
                        let s1 = builder.ins().fmax(neg_qx1, zero_vec);
                        let half_pz1 = builder.ins().fmul(half_v, pz_adj1);
                        let qy_sub1 = builder.ins().fsub(qy1, half_pz1);
                        let t_raw1 = builder.ins().fmul(qy_sub1, inv_m2_025_v);
                        let t_min1 = builder.ins().fmin(t_raw1, one_vec);
                        let t1 = builder.ins().fmax(zero_vec, t_min1);
                        let qx_s1 = builder.ins().fadd(qx1, s1);
                        let qx_s_sq1 = builder.ins().fmul(qx_s1, qx_s1);
                        let m2_qxs1 = builder.ins().fmul(m2_v, qx_s_sq1);
                        let a1 = builder.ins().fma(qy1, qy1, m2_qxs1);
                        let half_t1 = builder.ins().fmul(half_v, t1);
                        let qx_ht1 = builder.ins().fadd(qx1, half_t1);
                        let qx_ht_sq1 = builder.ins().fmul(qx_ht1, qx_ht1);
                        let m2_t1 = builder.ins().fmul(m2_v, t1);
                        let qy_m2t1 = builder.ins().fsub(qy1, m2_t1);
                        let m2_qxht1 = builder.ins().fmul(m2_v, qx_ht_sq1);
                        let b1 = builder.ins().fma(qy_m2t1, qy_m2t1, m2_qxht1);
                        let neg_qx_m2_1 = builder.ins().fmul(neg_qx1, m2_v);
                        let half_qy1 = builder.ins().fmul(half_v, qy1);
                        let cond1 = builder.ins().fsub(neg_qx_m2_1, half_qy1);
                        let min_cond1 = builder.ins().fmin(qy1, cond1);
                        let ab_min1 = builder.ins().fmin(a1, b1);
                        let neg_min_cond1 = builder.ins().fneg(min_cond1);
                        let d2_1 = simd_select_neg(&mut builder, neg_min_cond1, zero_vec, ab_min1);
                        let qz_sq1 = builder.ins().fmul(qz1, qz1);
                        let d2_qz1 = builder.ins().fadd(d2_1, qz_sq1);
                        let d2_sc1 = builder.ins().fmul(d2_qz1, inv_m2_v);
                        let dist1 = builder.ins().sqrt(d2_sc1);
                        let neg_py1 = builder.ins().fneg(py1);
                        let sign_arg1 = builder.ins().fmax(qz1, neg_py1);
                        let signed1 = simd_select_neg(&mut builder, sign_arg1, neg_one_v, one_vec);
                        let d1 = builder.ins().fmul(signed1, dist1);

                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Octahedron => {
                        let size_val = inst.params[0];
                        let s_s = builder.ins().f32const(size_val);
                        let three_s = builder.ins().f32const(3.0);
                        let half_s = builder.ins().f32const(0.5);
                        let inv_sqrt3_s = builder.ins().f32const(0.57735027);
                        let s_v = builder.ins().splat(vec_type, s_s);
                        let three_v = builder.ins().splat(vec_type, three_s);
                        let half_v = builder.ins().splat(vec_type, half_s);
                        let inv_sqrt3_v = builder.ins().splat(vec_type, inv_sqrt3_s);

                        // Lane 0
                        let apx0 = builder.ins().fabs(curr_x.0);
                        let apy0 = builder.ins().fabs(curr_y.0);
                        let apz0 = builder.ins().fabs(curr_z.0);
                        let sum_xy0 = builder.ins().fadd(apx0, apy0);
                        let sum_xyz0 = builder.ins().fadd(sum_xy0, apz0);
                        let m0 = builder.ins().fsub(sum_xyz0, s_v);
                        let tpx0 = builder.ins().fmul(three_v, apx0);
                        let tpy0 = builder.ins().fmul(three_v, apy0);
                        let tpz0 = builder.ins().fmul(three_v, apz0);
                        let c1_0 = builder.ins().fsub(tpx0, m0);
                        let c2_0 = builder.ins().fsub(tpy0, m0);
                        let c3_0 = builder.ins().fsub(tpz0, m0);
                        let qx_c2_0 = simd_select_neg(&mut builder, c2_0, apy0, apz0);
                        let qy_c2_0 = simd_select_neg(&mut builder, c2_0, apz0, apx0);
                        let qz_c2_0 = simd_select_neg(&mut builder, c2_0, apx0, apy0);
                        let qx0 = simd_select_neg(&mut builder, c1_0, apx0, qx_c2_0);
                        let qy0 = simd_select_neg(&mut builder, c1_0, apy0, qy_c2_0);
                        let qz0 = simd_select_neg(&mut builder, c1_0, apz0, qz_c2_0);
                        let qz_qy0 = builder.ins().fsub(qz0, qy0);
                        let qz_qy_s0 = builder.ins().fadd(qz_qy0, s_v);
                        let hv0 = builder.ins().fmul(half_v, qz_qy_s0);
                        let k_min0 = builder.ins().fmin(hv0, s_v);
                        let k0 = builder.ins().fmax(zero_vec, k_min0);
                        let qy_s0 = builder.ins().fsub(qy0, s_v);
                        let qy_sk0 = builder.ins().fadd(qy_s0, k0);
                        let qz_k0 = builder.ins().fsub(qz0, k0);
                        let detail0 = simd_length3_fma(&mut builder, qx0, qy_sk0, qz_k0);
                        let early0 = builder.ins().fmul(m0, inv_sqrt3_v);
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
                        let k1 = builder.ins().fmax(zero_vec, k_min1);
                        let qy_s1 = builder.ins().fsub(qy1, s_v);
                        let qy_sk1 = builder.ins().fadd(qy_s1, k1);
                        let qz_k1 = builder.ins().fsub(qz1, k1);
                        let detail1 = simd_length3_fma(&mut builder, qx1, qy_sk1, qz_k1);
                        let early1 = builder.ins().fmul(m1, inv_sqrt3_v);
                        let min_c23_1 = builder.ins().fmin(c2_1, c3_1);
                        let any_cond1 = builder.ins().fmin(c1_1, min_c23_1);
                        let d1 = simd_select_neg(&mut builder, any_cond1, detail1, early1);

                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((d0_scaled, d1_scaled));
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

                        let hr_v = builder.ins().splat(vec_type, hr_s);
                        let hh_v = builder.ins().splat(vec_type, hh_s);
                        let kx_v = builder.ins().splat(vec_type, kx_s);
                        let ky_v = builder.ins().splat(vec_type, ky_s);
                        let two_v = builder.ins().splat(vec_type, two_s);
                        let neg_one_v = builder.ins().splat(vec_type, neg_one_s);
                        let neg_kz_hr_v = builder.ins().splat(vec_type, neg_kz_hr_s);
                        let kz_hr_v = builder.ins().splat(vec_type, kz_hr_s);

                        // Lane 0
                        let apx0 = builder.ins().fabs(curr_x.0);
                        let apy0 = builder.ins().fabs(curr_y.0);
                        let apz0 = builder.ins().fabs(curr_z.0);
                        let ky_py0 = builder.ins().fmul(ky_v, apy0);
                        let dot0 = builder.ins().fma(kx_v, apx0, ky_py0);
                        let dot_min0 = builder.ins().fmin(dot0, zero_vec);
                        let reflect0 = builder.ins().fmul(two_v, dot_min0);
                        let rkx0 = builder.ins().fmul(reflect0, kx_v);
                        let rky0 = builder.ins().fmul(reflect0, ky_v);
                        let px_r0 = builder.ins().fsub(apx0, rkx0);
                        let py_r0 = builder.ins().fsub(apy0, rky0);
                        let px_cl0 = builder.ins().fmin(px_r0, kz_hr_v);
                        let clamped0 = builder.ins().fmax(neg_kz_hr_v, px_cl0);
                        let dx0 = builder.ins().fsub(px_r0, clamped0);
                        let dy0 = builder.ins().fsub(py_r0, hr_v);
                        let len_dxy0 = simd_length2_fma(&mut builder, dx0, dy0);
                        let sign_dy0 = simd_select_neg(&mut builder, dy0, neg_one_v, one_vec);
                        let d_xy0 = builder.ins().fmul(len_dxy0, sign_dy0);
                        let d_z0 = builder.ins().fsub(apz0, hh_v);
                        let max_dd0 = builder.ins().fmax(d_xy0, d_z0);
                        let interior0 = builder.ins().fmin(max_dd0, zero_vec);
                        let d_xy_p0 = builder.ins().fmax(d_xy0, zero_vec);
                        let d_z_p0 = builder.ins().fmax(d_z0, zero_vec);
                        let exterior0 = simd_length2_fma(&mut builder, d_xy_p0, d_z_p0);
                        let d0 = builder.ins().fadd(interior0, exterior0);

                        // Lane 1
                        let apx1 = builder.ins().fabs(curr_x.1);
                        let apy1 = builder.ins().fabs(curr_y.1);
                        let apz1 = builder.ins().fabs(curr_z.1);
                        let ky_py1 = builder.ins().fmul(ky_v, apy1);
                        let dot1 = builder.ins().fma(kx_v, apx1, ky_py1);
                        let dot_min1 = builder.ins().fmin(dot1, zero_vec);
                        let reflect1 = builder.ins().fmul(two_v, dot_min1);
                        let rkx1 = builder.ins().fmul(reflect1, kx_v);
                        let rky1 = builder.ins().fmul(reflect1, ky_v);
                        let px_r1 = builder.ins().fsub(apx1, rkx1);
                        let py_r1 = builder.ins().fsub(apy1, rky1);
                        let px_cl1 = builder.ins().fmin(px_r1, kz_hr_v);
                        let clamped1 = builder.ins().fmax(neg_kz_hr_v, px_cl1);
                        let dx1 = builder.ins().fsub(px_r1, clamped1);
                        let dy1 = builder.ins().fsub(py_r1, hr_v);
                        let len_dxy1 = simd_length2_fma(&mut builder, dx1, dy1);
                        let sign_dy1 = simd_select_neg(&mut builder, dy1, neg_one_v, one_vec);
                        let d_xy1 = builder.ins().fmul(len_dxy1, sign_dy1);
                        let d_z1 = builder.ins().fsub(apz1, hh_v);
                        let max_dd1 = builder.ins().fmax(d_xy1, d_z1);
                        let interior1 = builder.ins().fmin(max_dd1, zero_vec);
                        let d_xy_p1 = builder.ins().fmax(d_xy1, zero_vec);
                        let d_z_p1 = builder.ins().fmax(d_z1, zero_vec);
                        let exterior1 = simd_length2_fma(&mut builder, d_xy_p1, d_z_p1);
                        let d1 = builder.ins().fadd(interior1, exterior1);

                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Link => {
                        let hl_s = builder.ins().f32const(inst.params[0]);
                        let r1_s = builder.ins().f32const(inst.params[1]);
                        let r2_s = builder.ins().f32const(inst.params[2]);
                        let hl = builder.ins().splat(vec_type, hl_s);
                        let r1_v = builder.ins().splat(vec_type, r1_s);
                        let r2_v = builder.ins().splat(vec_type, r2_s);

                        // Lane 0: qy = max(abs(y) - half_length, 0)
                        let abs_y0 = builder.ins().fabs(curr_y.0);
                        let y_sub0 = builder.ins().fsub(abs_y0, hl);
                        let qy0 = builder.ins().fmax(y_sub0, zero_vec);
                        let qyy0 = builder.ins().fmul(qy0, qy0);
                        let xy_sq0 = builder.ins().fma(curr_x.0, curr_x.0, qyy0);
                        let xy_len0 = builder.ins().sqrt(xy_sq0);
                        let xy_sub0 = builder.ins().fsub(xy_len0, r1_v);
                        let zz0 = builder.ins().fmul(curr_z.0, curr_z.0);
                        let d_sq0 = builder.ins().fma(xy_sub0, xy_sub0, zz0);
                        let d_len0 = builder.ins().sqrt(d_sq0);
                        let d0 = builder.ins().fsub(d_len0, r2_v);
                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);

                        // Lane 1
                        let abs_y1 = builder.ins().fabs(curr_y.1);
                        let y_sub1 = builder.ins().fsub(abs_y1, hl);
                        let qy1 = builder.ins().fmax(y_sub1, zero_vec);
                        let qyy1 = builder.ins().fmul(qy1, qy1);
                        let xy_sq1 = builder.ins().fma(curr_x.1, curr_x.1, qyy1);
                        let xy_len1 = builder.ins().sqrt(xy_sq1);
                        let xy_sub1 = builder.ins().fsub(xy_len1, r1_v);
                        let zz1 = builder.ins().fmul(curr_z.1, curr_z.1);
                        let d_sq1 = builder.ins().fma(xy_sub1, xy_sub1, zz1);
                        let d_len1 = builder.ins().sqrt(d_sq1);
                        let d1 = builder.ins().fsub(d_len1, r2_v);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);

                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Union => {
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        value_stack
                            .push((builder.ins().fmin(a.0, b.0), builder.ins().fmin(a.1, b.1)));
                    }

                    OpCode::Intersection => {
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        value_stack
                            .push((builder.ins().fmax(a.0, b.0), builder.ins().fmax(a.1, b.1)));
                    }

                    OpCode::Subtraction => {
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let neg_b0 = builder.ins().fneg(b.0);
                        let neg_b1 = builder.ins().fneg(b.1);
                        value_stack.push((
                            builder.ins().fmax(a.0, neg_b0),
                            builder.ins().fmax(a.1, neg_b1),
                        ));
                    }

                    // Division Exorcism: Smooth Ops use mul(1/k) instead of div(k)
                    OpCode::SmoothUnion => {
                        let k = inst.params[0];
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        if k.abs() < FOLD_EPSILON {
                            value_stack
                                .push((builder.ins().fmin(a.0, b.0), builder.ins().fmin(a.1, b.1)));
                            continue;
                        }

                        // Pre-compute reciprocal at compile time
                        let inv_k = 1.0 / k;

                        let k_s = builder.ins().f32const(k);
                        let k_v = builder.ins().splat(vec_type, k_s);
                        let inv_k_s = builder.ins().f32const(inv_k);
                        let inv_k_v = builder.ins().splat(vec_type, inv_k_s);
                        let quarter_s = builder.ins().f32const(0.25);
                        let quarter = builder.ins().splat(vec_type, quarter_s);

                        // Lane 0: fmul(inv_k) instead of fdiv(k)
                        let diff0 = builder.ins().fsub(a.0, b.0);
                        let abs_diff0 = builder.ins().fabs(diff0);
                        let h_num0 = builder.ins().fsub(k_v, abs_diff0);
                        let h_num0 = builder.ins().fmax(h_num0, zero_vec);
                        let h0 = builder.ins().fmul(h_num0, inv_k_v);
                        let hh0 = builder.ins().fmul(h0, h0);
                        let hhk0 = builder.ins().fmul(hh0, k_v);
                        let off0 = builder.ins().fmul(hhk0, quarter);
                        let min0 = builder.ins().fmin(a.0, b.0);
                        let res0 = builder.ins().fsub(min0, off0);

                        // Lane 1
                        let diff1 = builder.ins().fsub(a.1, b.1);
                        let abs_diff1 = builder.ins().fabs(diff1);
                        let h_num1 = builder.ins().fsub(k_v, abs_diff1);
                        let h_num1 = builder.ins().fmax(h_num1, zero_vec);
                        let h1 = builder.ins().fmul(h_num1, inv_k_v);
                        let hh1 = builder.ins().fmul(h1, h1);
                        let hhk1 = builder.ins().fmul(hh1, k_v);
                        let off1 = builder.ins().fmul(hhk1, quarter);
                        let min1 = builder.ins().fmin(a.1, b.1);
                        let res1 = builder.ins().fsub(min1, off1);

                        value_stack.push((res0, res1));
                    }

                    OpCode::SmoothIntersection => {
                        let k = inst.params[0];
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        if k.abs() < FOLD_EPSILON {
                            value_stack
                                .push((builder.ins().fmax(a.0, b.0), builder.ins().fmax(a.1, b.1)));
                            continue;
                        }

                        let inv_k = 1.0 / k;
                        let k_s = builder.ins().f32const(k);
                        let k_v = builder.ins().splat(vec_type, k_s);
                        let inv_k_s = builder.ins().f32const(inv_k);
                        let inv_k_v = builder.ins().splat(vec_type, inv_k_s);
                        let quarter_s = builder.ins().f32const(0.25);
                        let quarter = builder.ins().splat(vec_type, quarter_s);

                        // Lane 0
                        let diff0 = builder.ins().fsub(a.0, b.0);
                        let abs_diff0 = builder.ins().fabs(diff0);
                        let h_num0 = builder.ins().fsub(k_v, abs_diff0);
                        let h_num0 = builder.ins().fmax(h_num0, zero_vec);
                        let h0 = builder.ins().fmul(h_num0, inv_k_v);
                        let hh0 = builder.ins().fmul(h0, h0);
                        let hhk0 = builder.ins().fmul(hh0, k_v);
                        let off0 = builder.ins().fmul(hhk0, quarter);
                        let max0 = builder.ins().fmax(a.0, b.0);
                        let res0 = builder.ins().fadd(max0, off0);

                        // Lane 1
                        let diff1 = builder.ins().fsub(a.1, b.1);
                        let abs_diff1 = builder.ins().fabs(diff1);
                        let h_num1 = builder.ins().fsub(k_v, abs_diff1);
                        let h_num1 = builder.ins().fmax(h_num1, zero_vec);
                        let h1 = builder.ins().fmul(h_num1, inv_k_v);
                        let hh1 = builder.ins().fmul(h1, h1);
                        let hhk1 = builder.ins().fmul(hh1, k_v);
                        let off1 = builder.ins().fmul(hhk1, quarter);
                        let max1 = builder.ins().fmax(a.1, b.1);
                        let res1 = builder.ins().fadd(max1, off1);

                        value_stack.push((res0, res1));
                    }

                    OpCode::SmoothSubtraction => {
                        let k = inst.params[0];
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        if k.abs() < FOLD_EPSILON {
                            let neg_b0 = builder.ins().fneg(b.0);
                            let neg_b1 = builder.ins().fneg(b.1);
                            value_stack.push((
                                builder.ins().fmax(a.0, neg_b0),
                                builder.ins().fmax(a.1, neg_b1),
                            ));
                            continue;
                        }

                        let inv_k = 1.0 / k;
                        let k_s = builder.ins().f32const(k);
                        let k_v = builder.ins().splat(vec_type, k_s);
                        let inv_k_s = builder.ins().f32const(inv_k);
                        let inv_k_v = builder.ins().splat(vec_type, inv_k_s);
                        let quarter_s = builder.ins().f32const(0.25);
                        let quarter = builder.ins().splat(vec_type, quarter_s);

                        // Lane 0
                        let neg_b0 = builder.ins().fneg(b.0);
                        let diff0 = builder.ins().fsub(a.0, neg_b0);
                        let abs_diff0 = builder.ins().fabs(diff0);
                        let h_num0 = builder.ins().fsub(k_v, abs_diff0);
                        let h_num0 = builder.ins().fmax(h_num0, zero_vec);
                        let h0 = builder.ins().fmul(h_num0, inv_k_v);
                        let hh0 = builder.ins().fmul(h0, h0);
                        let hhk0 = builder.ins().fmul(hh0, k_v);
                        let off0 = builder.ins().fmul(hhk0, quarter);
                        let max0 = builder.ins().fmax(a.0, neg_b0);
                        let res0 = builder.ins().fadd(max0, off0);

                        // Lane 1
                        let neg_b1 = builder.ins().fneg(b.1);
                        let diff1 = builder.ins().fsub(a.1, neg_b1);
                        let abs_diff1 = builder.ins().fabs(diff1);
                        let h_num1 = builder.ins().fsub(k_v, abs_diff1);
                        let h_num1 = builder.ins().fmax(h_num1, zero_vec);
                        let h1 = builder.ins().fmul(h_num1, inv_k_v);
                        let hh1 = builder.ins().fmul(h1, h1);
                        let hhk1 = builder.ins().fmul(hh1, k_v);
                        let off1 = builder.ins().fmul(hhk1, quarter);
                        let max1 = builder.ins().fmax(a.1, neg_b1);
                        let res1 = builder.ins().fadd(max1, off1);

                        value_stack.push((res0, res1));
                    }

                    OpCode::ChamferUnion => {
                        let r = inst.params[0];
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        if r.abs() < FOLD_EPSILON {
                            value_stack
                                .push((builder.ins().fmin(a.0, b.0), builder.ins().fmin(a.1, b.1)));
                            continue;
                        }

                        let r_s = builder.ins().f32const(r);
                        let r_v = builder.ins().splat(vec_type, r_s);
                        let s_s = builder.ins().f32const(std::f32::consts::FRAC_1_SQRT_2);
                        let s_v = builder.ins().splat(vec_type, s_s);

                        // Lane 0: min(min(a,b), (a+b)*s - r)
                        let min0 = builder.ins().fmin(a.0, b.0);
                        let sum0 = builder.ins().fadd(a.0, b.0);
                        let sc0 = builder.ins().fmul(sum0, s_v);
                        let ch0 = builder.ins().fsub(sc0, r_v);
                        let res0 = builder.ins().fmin(min0, ch0);

                        // Lane 1
                        let min1 = builder.ins().fmin(a.1, b.1);
                        let sum1 = builder.ins().fadd(a.1, b.1);
                        let sc1 = builder.ins().fmul(sum1, s_v);
                        let ch1 = builder.ins().fsub(sc1, r_v);
                        let res1 = builder.ins().fmin(min1, ch1);

                        value_stack.push((res0, res1));
                    }

                    OpCode::ChamferIntersection => {
                        let r = inst.params[0];
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        if r.abs() < FOLD_EPSILON {
                            value_stack
                                .push((builder.ins().fmax(a.0, b.0), builder.ins().fmax(a.1, b.1)));
                            continue;
                        }

                        let r_s = builder.ins().f32const(r);
                        let r_v = builder.ins().splat(vec_type, r_s);
                        let s_s = builder.ins().f32const(std::f32::consts::FRAC_1_SQRT_2);
                        let s_v = builder.ins().splat(vec_type, s_s);

                        // -chamfer_min(-a, -b, r)
                        let na0 = builder.ins().fneg(a.0);
                        let nb0 = builder.ins().fneg(b.0);
                        let min0 = builder.ins().fmin(na0, nb0);
                        let sum0 = builder.ins().fadd(na0, nb0);
                        let sc0 = builder.ins().fmul(sum0, s_v);
                        let ch0 = builder.ins().fsub(sc0, r_v);
                        let cm0 = builder.ins().fmin(min0, ch0);
                        let res0 = builder.ins().fneg(cm0);

                        let na1 = builder.ins().fneg(a.1);
                        let nb1 = builder.ins().fneg(b.1);
                        let min1 = builder.ins().fmin(na1, nb1);
                        let sum1 = builder.ins().fadd(na1, nb1);
                        let sc1 = builder.ins().fmul(sum1, s_v);
                        let ch1 = builder.ins().fsub(sc1, r_v);
                        let cm1 = builder.ins().fmin(min1, ch1);
                        let res1 = builder.ins().fneg(cm1);

                        value_stack.push((res0, res1));
                    }

                    OpCode::ChamferSubtraction => {
                        let r = inst.params[0];
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        if r.abs() < FOLD_EPSILON {
                            let neg_b0 = builder.ins().fneg(b.0);
                            let neg_b1 = builder.ins().fneg(b.1);
                            value_stack.push((
                                builder.ins().fmax(a.0, neg_b0),
                                builder.ins().fmax(a.1, neg_b1),
                            ));
                            continue;
                        }

                        let r_s = builder.ins().f32const(r);
                        let r_v = builder.ins().splat(vec_type, r_s);
                        let s_s = builder.ins().f32const(std::f32::consts::FRAC_1_SQRT_2);
                        let s_v = builder.ins().splat(vec_type, s_s);

                        // -chamfer_min(-a, b, r)
                        let na0 = builder.ins().fneg(a.0);
                        let min0 = builder.ins().fmin(na0, b.0);
                        let sum0 = builder.ins().fadd(na0, b.0);
                        let sc0 = builder.ins().fmul(sum0, s_v);
                        let ch0 = builder.ins().fsub(sc0, r_v);
                        let cm0 = builder.ins().fmin(min0, ch0);
                        let res0 = builder.ins().fneg(cm0);

                        let na1 = builder.ins().fneg(a.1);
                        let min1 = builder.ins().fmin(na1, b.1);
                        let sum1 = builder.ins().fadd(na1, b.1);
                        let sc1 = builder.ins().fmul(sum1, s_v);
                        let ch1 = builder.ins().fsub(sc1, r_v);
                        let cm1 = builder.ins().fmin(min1, ch1);
                        let res1 = builder.ins().fneg(cm1);

                        value_stack.push((res0, res1));
                    }

                    OpCode::StairsUnion => {
                        let r = inst.params[0];
                        let n = inst.params[1];
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        let res0 = emit_simd_stairs_min(&mut builder, vec_type, a.0, b.0, r, n);
                        let res1 = emit_simd_stairs_min(&mut builder, vec_type, a.1, b.1, r, n);
                        value_stack.push((res0, res1));
                    }

                    OpCode::StairsIntersection => {
                        let r = inst.params[0];
                        let n = inst.params[1];
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        let na0 = builder.ins().fneg(a.0);
                        let nb0 = builder.ins().fneg(b.0);
                        let sm0 = emit_simd_stairs_min(&mut builder, vec_type, na0, nb0, r, n);
                        let res0 = builder.ins().fneg(sm0);

                        let na1 = builder.ins().fneg(a.1);
                        let nb1 = builder.ins().fneg(b.1);
                        let sm1 = emit_simd_stairs_min(&mut builder, vec_type, na1, nb1, r, n);
                        let res1 = builder.ins().fneg(sm1);

                        value_stack.push((res0, res1));
                    }

                    OpCode::StairsSubtraction => {
                        let r = inst.params[0];
                        let n = inst.params[1];
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        let na0 = builder.ins().fneg(a.0);
                        let sm0 = emit_simd_stairs_min(&mut builder, vec_type, na0, b.0, r, n);
                        let res0 = builder.ins().fneg(sm0);

                        let na1 = builder.ins().fneg(a.1);
                        let sm1 = emit_simd_stairs_min(&mut builder, vec_type, na1, b.1, r, n);
                        let res1 = builder.ins().fneg(sm1);

                        value_stack.push((res0, res1));
                    }

                    OpCode::Translate => {
                        let tx = inst.params[0];
                        let ty = inst.params[1];
                        let tz = inst.params[2];

                        if tx.abs() < FOLD_EPSILON
                            && ty.abs() < FOLD_EPSILON
                            && tz.abs() < FOLD_EPSILON
                        {
                            coord_stack.push(SimdCoordState {
                                x: curr_x,
                                y: curr_y,
                                z: curr_z,
                                scale: curr_scale,
                                opcode: OpCode::Translate,
                                params: [0.0; 4],
                                folded: true,
                            });
                            continue;
                        }

                        coord_stack.push(SimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Translate,
                            params: [0.0; 4],
                            folded: false,
                        });

                        let tx_s = builder.ins().f32const(tx);
                        let ty_s = builder.ins().f32const(ty);
                        let tz_s = builder.ins().f32const(tz);
                        let tx_v = builder.ins().splat(vec_type, tx_s);
                        let ty_v = builder.ins().splat(vec_type, ty_s);
                        let tz_v = builder.ins().splat(vec_type, tz_s);

                        let nx0 = builder.ins().fsub(curr_x.0, tx_v);
                        let ny0 = builder.ins().fsub(curr_y.0, ty_v);
                        let nz0 = builder.ins().fsub(curr_z.0, tz_v);
                        let nx1 = builder.ins().fsub(curr_x.1, tx_v);
                        let ny1 = builder.ins().fsub(curr_y.1, ty_v);
                        let nz1 = builder.ins().fsub(curr_z.1, tz_v);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        curr_z = (nz0, nz1);
                    }

                    // Division Exorcism: Scale uses mul(1/s) instead of div(s)
                    OpCode::Scale => {
                        // params[0] = 1/factor (precomputed inverse)
                        // params[1] = factor (original)
                        let inv_factor = inst.params[0];
                        let factor = inst.params[1];

                        if (factor - 1.0).abs() < FOLD_EPSILON {
                            coord_stack.push(SimdCoordState {
                                x: curr_x,
                                y: curr_y,
                                z: curr_z,
                                scale: curr_scale,
                                opcode: OpCode::Scale,
                                params: [0.0; 4],
                                folded: true,
                            });
                            continue;
                        }

                        coord_stack.push(SimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Scale,
                            params: [0.0; 4],
                            folded: false,
                        });

                        // Division Exorcism: p *= inv_factor (no division)
                        let inv_s = builder.ins().f32const(inv_factor);
                        let inv_v = builder.ins().splat(vec_type, inv_s);

                        let f_s = builder.ins().f32const(factor);
                        let f_v = builder.ins().splat(vec_type, f_s);

                        let nx0 = builder.ins().fmul(curr_x.0, inv_v);
                        let ny0 = builder.ins().fmul(curr_y.0, inv_v);
                        let nz0 = builder.ins().fmul(curr_z.0, inv_v);
                        let nx1 = builder.ins().fmul(curr_x.1, inv_v);
                        let ny1 = builder.ins().fmul(curr_y.1, inv_v);
                        let nz1 = builder.ins().fmul(curr_z.1, inv_v);

                        // scale_correction *= factor
                        let ns0 = builder.ins().fmul(curr_scale.0, f_v);
                        let ns1 = builder.ins().fmul(curr_scale.1, f_v);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        curr_z = (nz0, nz1);
                        curr_scale = (ns0, ns1);
                    }

                    OpCode::Rotate => {
                        coord_stack.push(SimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Rotate,
                            params: [0.0; 4],
                            folded: false,
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
                        // Rotation matrix from quaternion
                        let m00 = 1.0 - 2.0 * (qy * qy + qz * qz);
                        let m01 = 2.0 * (qx * qy - qz * qw);
                        let m02 = 2.0 * (qx * qz + qy * qw);
                        let m10 = 2.0 * (qx * qy + qz * qw);
                        let m11 = 1.0 - 2.0 * (qx * qx + qz * qz);
                        let m12 = 2.0 * (qy * qz - qx * qw);
                        let m20 = 2.0 * (qx * qz - qy * qw);
                        let m21 = 2.0 * (qy * qz + qx * qw);
                        let m22 = 1.0 - 2.0 * (qx * qx + qy * qy);

                        let _ts1 = builder.ins().f32const(m00);
                        let m00v = builder.ins().splat(vec_type, _ts1);
                        let _ts2 = builder.ins().f32const(m01);
                        let m01v = builder.ins().splat(vec_type, _ts2);
                        let _ts3 = builder.ins().f32const(m02);
                        let m02v = builder.ins().splat(vec_type, _ts3);
                        let _ts4 = builder.ins().f32const(m10);
                        let m10v = builder.ins().splat(vec_type, _ts4);
                        let _ts5 = builder.ins().f32const(m11);
                        let m11v = builder.ins().splat(vec_type, _ts5);
                        let _ts6 = builder.ins().f32const(m12);
                        let m12v = builder.ins().splat(vec_type, _ts6);
                        let _ts7 = builder.ins().f32const(m20);
                        let m20v = builder.ins().splat(vec_type, _ts7);
                        let _ts8 = builder.ins().f32const(m21);
                        let m21v = builder.ins().splat(vec_type, _ts8);
                        let _ts9 = builder.ins().f32const(m22);
                        let m22v = builder.ins().splat(vec_type, _ts9);

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

                        coord_stack.push(SimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::ScaleNonUniform,
                            params: [0.0; 4],
                            folded: false,
                        });

                        let _ts10 = builder.ins().f32const(inv_sx);
                        let isx = builder.ins().splat(vec_type, _ts10);
                        let _ts11 = builder.ins().f32const(inv_sy);
                        let isy = builder.ins().splat(vec_type, _ts11);
                        let _ts12 = builder.ins().f32const(inv_sz);
                        let isz = builder.ins().splat(vec_type, _ts12);
                        let _ts13 = builder.ins().f32const(min_factor);
                        let mf = builder.ins().splat(vec_type, _ts13);

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
                        let k = inst.params[0];
                        if k.abs() < FOLD_EPSILON {
                            coord_stack.push(SimdCoordState {
                                x: curr_x,
                                y: curr_y,
                                z: curr_z,
                                scale: curr_scale,
                                opcode: OpCode::Twist,
                                params: [0.0; 4],
                                folded: true,
                            });
                            continue;
                        }

                        coord_stack.push(SimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Twist,
                            params: [0.0; 4],
                            folded: false,
                        });

                        let k_s = builder.ins().f32const(k);
                        let k_v = builder.ins().splat(vec_type, k_s);

                        // Lane 0: angle = y * k, (cos, sin) = approx(angle)
                        let angle0 = builder.ins().fmul(curr_y.0, k_v);
                        let (cos0, sin0) = simd_sincos_approx(&mut builder, angle0, vec_type);
                        let cx0 = builder.ins().fmul(cos0, curr_x.0);
                        let sz0 = builder.ins().fmul(sin0, curr_z.0);
                        let nx0 = builder.ins().fsub(cx0, sz0);
                        let sx0 = builder.ins().fmul(sin0, curr_x.0);
                        let cz0 = builder.ins().fmul(cos0, curr_z.0);
                        let nz0 = builder.ins().fadd(sx0, cz0);

                        // Lane 1
                        let angle1 = builder.ins().fmul(curr_y.1, k_v);
                        let (cos1, sin1) = simd_sincos_approx(&mut builder, angle1, vec_type);
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
                        let k = inst.params[0];
                        if k.abs() < FOLD_EPSILON {
                            coord_stack.push(SimdCoordState {
                                x: curr_x,
                                y: curr_y,
                                z: curr_z,
                                scale: curr_scale,
                                opcode: OpCode::Bend,
                                params: [0.0; 4],
                                folded: true,
                            });
                            continue;
                        }

                        coord_stack.push(SimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Bend,
                            params: [0.0; 4],
                            folded: false,
                        });

                        let k_s = builder.ins().f32const(k);
                        let k_v = builder.ins().splat(vec_type, k_s);

                        // Lane 0: angle = k * x, rotate XY
                        // x' = cos*x - sin*y, y' = sin*x + cos*y
                        let angle0 = builder.ins().fmul(k_v, curr_x.0);
                        let (cos0, sin0) = simd_sincos_approx(&mut builder, angle0, vec_type);
                        let cx0 = builder.ins().fmul(cos0, curr_x.0);
                        let sy0 = builder.ins().fmul(sin0, curr_y.0);
                        let nx0 = builder.ins().fsub(cx0, sy0);
                        let sx0 = builder.ins().fmul(sin0, curr_x.0);
                        let cy0 = builder.ins().fmul(cos0, curr_y.0);
                        let ny0 = builder.ins().fadd(sx0, cy0);

                        // Lane 1
                        let angle1 = builder.ins().fmul(k_v, curr_x.1);
                        let (cos1, sin1) = simd_sincos_approx(&mut builder, angle1, vec_type);
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
                        coord_stack.push(SimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::RepeatInfinite,
                            params: [0.0; 4],
                            folded: false,
                        });

                        // Division Exorcism: pre-compute reciprocals at compile time
                        let sx = inst.params[0];
                        let sy = inst.params[1];
                        let sz = inst.params[2];
                        let isx = if sx.abs() < 1e-7 { 0.0 } else { 1.0 / sx };
                        let isy = if sy.abs() < 1e-7 { 0.0 } else { 1.0 / sy };
                        let isz = if sz.abs() < 1e-7 { 0.0 } else { 1.0 / sz };

                        let _ts14 = builder.ins().f32const(sx);
                        let sx_v = builder.ins().splat(vec_type, _ts14);
                        let _ts15 = builder.ins().f32const(sy);
                        let sy_v = builder.ins().splat(vec_type, _ts15);
                        let _ts16 = builder.ins().f32const(sz);
                        let sz_v = builder.ins().splat(vec_type, _ts16);
                        let _ts17 = builder.ins().f32const(isx);
                        let isx_v = builder.ins().splat(vec_type, _ts17);
                        let _ts18 = builder.ins().f32const(isy);
                        let isy_v = builder.ins().splat(vec_type, _ts18);
                        let _ts19 = builder.ins().f32const(isz);
                        let isz_v = builder.ins().splat(vec_type, _ts19);

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
                        coord_stack.push(SimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::RepeatFinite,
                            params: [0.0; 4],
                            folded: false,
                        });

                        let cx = inst.params[0] as f32; // count already as f32
                        let cy = inst.params[1] as f32;
                        let cz = inst.params[2] as f32;
                        let sx = inst.params[3];
                        let sy = inst.params[4];
                        let sz = inst.params[5];
                        let isx = if sx.abs() < 1e-7 { 0.0 } else { 1.0 / sx };
                        let isy = if sy.abs() < 1e-7 { 0.0 } else { 1.0 / sy };
                        let isz = if sz.abs() < 1e-7 { 0.0 } else { 1.0 / sz };
                        let lx = cx * 0.5;
                        let ly = cy * 0.5;
                        let lz = cz * 0.5;

                        let _ts20 = builder.ins().f32const(sx);
                        let sx_v = builder.ins().splat(vec_type, _ts20);
                        let _ts21 = builder.ins().f32const(sy);
                        let sy_v = builder.ins().splat(vec_type, _ts21);
                        let _ts22 = builder.ins().f32const(sz);
                        let sz_v = builder.ins().splat(vec_type, _ts22);
                        let _ts23 = builder.ins().f32const(isx);
                        let isx_v = builder.ins().splat(vec_type, _ts23);
                        let _ts24 = builder.ins().f32const(isy);
                        let isy_v = builder.ins().splat(vec_type, _ts24);
                        let _ts25 = builder.ins().f32const(isz);
                        let isz_v = builder.ins().splat(vec_type, _ts25);
                        let _ts26 = builder.ins().f32const(lx);
                        let lx_v = builder.ins().splat(vec_type, _ts26);
                        let _ts27 = builder.ins().f32const(ly);
                        let ly_v = builder.ins().splat(vec_type, _ts27);
                        let _ts28 = builder.ins().f32const(lz);
                        let lz_v = builder.ins().splat(vec_type, _ts28);
                        let nlx_s = builder.ins().f32const(-lx);
                        let nlx_v = builder.ins().splat(vec_type, nlx_s);
                        let nly_s = builder.ins().f32const(-ly);
                        let nly_v = builder.ins().splat(vec_type, nly_s);
                        let nlz_s = builder.ins().f32const(-lz);
                        let nlz_v = builder.ins().splat(vec_type, nlz_s);

                        // Lane 0: clamp(round(p * inv_s), -limit, limit), then p - cell * s
                        let _tm29 = builder.ins().fmul(curr_x.0, isx_v);
                        let rx0 = builder.ins().nearest(_tm29);
                        let _tm30 = builder.ins().fmul(curr_y.0, isy_v);
                        let ry0 = builder.ins().nearest(_tm30);
                        let _tm31 = builder.ins().fmul(curr_z.0, isz_v);
                        let rz0 = builder.ins().nearest(_tm31);
                        let _tn32 = builder.ins().fmin(rx0, lx_v);
                        let rx0 = builder.ins().fmax(nlx_v, _tn32);
                        let _tn33 = builder.ins().fmin(ry0, ly_v);
                        let ry0 = builder.ins().fmax(nly_v, _tn33);
                        let _tn34 = builder.ins().fmin(rz0, lz_v);
                        let rz0 = builder.ins().fmax(nlz_v, _tn34);
                        let _tm35 = builder.ins().fmul(rx0, sx_v);
                        let nx0 = builder.ins().fsub(curr_x.0, _tm35);
                        let _tm36 = builder.ins().fmul(ry0, sy_v);
                        let ny0 = builder.ins().fsub(curr_y.0, _tm36);
                        let _tm37 = builder.ins().fmul(rz0, sz_v);
                        let nz0 = builder.ins().fsub(curr_z.0, _tm37);

                        // Lane 1
                        let _tm38 = builder.ins().fmul(curr_x.1, isx_v);
                        let rx1 = builder.ins().nearest(_tm38);
                        let _tm39 = builder.ins().fmul(curr_y.1, isy_v);
                        let ry1 = builder.ins().nearest(_tm39);
                        let _tm40 = builder.ins().fmul(curr_z.1, isz_v);
                        let rz1 = builder.ins().nearest(_tm40);
                        let _tn41 = builder.ins().fmin(rx1, lx_v);
                        let rx1 = builder.ins().fmax(nlx_v, _tn41);
                        let _tn42 = builder.ins().fmin(ry1, ly_v);
                        let ry1 = builder.ins().fmax(nly_v, _tn42);
                        let _tn43 = builder.ins().fmin(rz1, lz_v);
                        let rz1 = builder.ins().fmax(nlz_v, _tn43);
                        let _tm44 = builder.ins().fmul(rx1, sx_v);
                        let nx1 = builder.ins().fsub(curr_x.1, _tm44);
                        let _tm45 = builder.ins().fmul(ry1, sy_v);
                        let ny1 = builder.ins().fsub(curr_y.1, _tm45);
                        let _tm46 = builder.ins().fmul(rz1, sz_v);
                        let nz1 = builder.ins().fsub(curr_z.1, _tm46);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        curr_z = (nz0, nz1);
                    }

                    OpCode::Elongate => {
                        coord_stack.push(SimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Elongate,
                            params: [0.0; 4],
                            folded: false,
                        });

                        let _ts47 = builder.ins().f32const(inst.params[0]);
                        let hx = builder.ins().splat(vec_type, _ts47);
                        let _ts48 = builder.ins().f32const(inst.params[1]);
                        let hy = builder.ins().splat(vec_type, _ts48);
                        let _ts49 = builder.ins().f32const(inst.params[2]);
                        let hz = builder.ins().splat(vec_type, _ts49);
                        let nhx = builder.ins().fneg(hx);
                        let nhy = builder.ins().fneg(hy);
                        let nhz = builder.ins().fneg(hz);

                        // Lane 0: p - clamp(p, -h, h)
                        let _tn50 = builder.ins().fmin(curr_x.0, hx);
                        let cx0 = builder.ins().fmax(nhx, _tn50);
                        let _tn51 = builder.ins().fmin(curr_y.0, hy);
                        let cy0 = builder.ins().fmax(nhy, _tn51);
                        let _tn52 = builder.ins().fmin(curr_z.0, hz);
                        let cz0 = builder.ins().fmax(nhz, _tn52);
                        let nx0 = builder.ins().fsub(curr_x.0, cx0);
                        let ny0 = builder.ins().fsub(curr_y.0, cy0);
                        let nz0 = builder.ins().fsub(curr_z.0, cz0);

                        // Lane 1
                        let _tn53 = builder.ins().fmin(curr_x.1, hx);
                        let cx1 = builder.ins().fmax(nhx, _tn53);
                        let _tn54 = builder.ins().fmin(curr_y.1, hy);
                        let cy1 = builder.ins().fmax(nhy, _tn54);
                        let _tn55 = builder.ins().fmin(curr_z.1, hz);
                        let cz1 = builder.ins().fmax(nhz, _tn55);
                        let nx1 = builder.ins().fsub(curr_x.1, cx1);
                        let ny1 = builder.ins().fsub(curr_y.1, cy1);
                        let nz1 = builder.ins().fsub(curr_z.1, cz1);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        curr_z = (nz0, nz1);
                    }

                    OpCode::Mirror => {
                        coord_stack.push(SimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Mirror,
                            params: [0.0; 4],
                            folded: false,
                        });

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

                    OpCode::Revolution => {
                        coord_stack.push(SimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Revolution,
                            params: [0.0; 4],
                            folded: false,
                        });

                        let _ts56 = builder.ins().f32const(inst.params[0]);
                        let off = builder.ins().splat(vec_type, _ts56);

                        // Lane 0: q = sqrt(x² + z²) - offset
                        let len0 = simd_length2_fma(&mut builder, curr_x.0, curr_z.0);
                        let q0 = builder.ins().fsub(len0, off);
                        // Lane 1
                        let len1 = simd_length2_fma(&mut builder, curr_x.1, curr_z.1);
                        let q1 = builder.ins().fsub(len1, off);

                        curr_x = (q0, q1);
                        // curr_y unchanged
                        curr_z = (zero_vec, zero_vec);
                    }

                    OpCode::SweepBezier => {
                        return Err("SweepBezier not supported in hardcoded SIMD JIT".to_string());
                    }

                    OpCode::Extrude => {
                        // Store half_height in params[0] for PopTransform
                        coord_stack.push(SimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Extrude,
                            params: [inst.params[0], 0.0, 0.0, 0.0],
                            folded: false,
                        });

                        // Evaluate child at (x, y, 0)
                        curr_z = (zero_vec, zero_vec);
                    }

                    OpCode::Noise => {
                        // Store noise params for PopTransform (nop in JIT - perlin not available)
                        coord_stack.push(SimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Noise,
                            params: [0.0; 4],
                            folded: true,
                        });
                    }

                    OpCode::Round => {
                        let r = inst.params[0];
                        coord_stack.push(SimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Round,
                            params: [r, 0.0, 0.0, 0.0],
                            folded: r.abs() < FOLD_EPSILON,
                        });
                    }

                    OpCode::Onion => {
                        let t = inst.params[0];
                        coord_stack.push(SimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Onion,
                            params: [t, 0.0, 0.0, 0.0],
                            folded: t.abs() < FOLD_EPSILON,
                        });
                    }

                    OpCode::PopTransform => {
                        if let Some(state) = coord_stack.pop() {
                            if !state.folded {
                                match state.opcode {
                                    OpCode::Round => {
                                        let d = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                                        let r_s = builder.ins().f32const(state.params[0]);
                                        let r = builder.ins().splat(vec_type, r_s);
                                        value_stack.push((
                                            builder.ins().fsub(d.0, r),
                                            builder.ins().fsub(d.1, r),
                                        ));
                                    }
                                    OpCode::Onion => {
                                        let d = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                                        let t_s = builder.ins().f32const(state.params[0]);
                                        let t = builder.ins().splat(vec_type, t_s);
                                        let a0 = builder.ins().fabs(d.0);
                                        let a1 = builder.ins().fabs(d.1);
                                        value_stack.push((
                                            builder.ins().fsub(a0, t),
                                            builder.ins().fsub(a1, t),
                                        ));
                                    }
                                    OpCode::Extrude => {
                                        // w = (child_d, |orig_z| - half_height)
                                        // d = min(max(w.x, w.y), 0) + length(max(w, 0))
                                        let hh_s = builder.ins().f32const(state.params[0]);
                                        let hh = builder.ins().splat(vec_type, hh_s);
                                        let d = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                                        // Lane 0
                                        let az0 = builder.ins().fabs(state.z.0);
                                        let wy0 = builder.ins().fsub(az0, hh);
                                        let inner0 = builder.ins().fmax(d.0, wy0);
                                        let inside0 = builder.ins().fmin(inner0, zero_vec);
                                        let mx0 = builder.ins().fmax(d.0, zero_vec);
                                        let my0 = builder.ins().fmax(wy0, zero_vec);
                                        let outside0 = simd_length2_fma(&mut builder, mx0, my0);
                                        let r0 = builder.ins().fadd(inside0, outside0);

                                        // Lane 1
                                        let az1 = builder.ins().fabs(state.z.1);
                                        let wy1 = builder.ins().fsub(az1, hh);
                                        let inner1 = builder.ins().fmax(d.1, wy1);
                                        let inside1 = builder.ins().fmin(inner1, zero_vec);
                                        let mx1 = builder.ins().fmax(d.1, zero_vec);
                                        let my1 = builder.ins().fmax(wy1, zero_vec);
                                        let outside1 = simd_length2_fma(&mut builder, mx1, my1);
                                        let r1 = builder.ins().fadd(inside1, outside1);

                                        value_stack.push((r0, r1));
                                    }
                                    _ => {}
                                }
                            }
                            curr_x = state.x;
                            curr_y = state.y;
                            curr_z = state.z;
                            curr_scale = state.scale;
                        }
                    }

                    OpCode::End => break,

                    _ => {
                        // Unimplemented / fallback
                        let max_s = builder.ins().f32const(f32::MAX);
                        let max_v = builder.ins().splat(vec_type, max_s);
                        value_stack.push((max_v, max_v));
                    }
                }
            }

            let result = value_stack.pop().unwrap_or((zero_vec, zero_vec));

            // Store results (2 x F32X4 -> 8 floats)
            builder.ins().store(mem_flags, result.0, ptr_out, 0);
            builder.ins().store(mem_flags, result.1, ptr_out, 16);

            builder.ins().return_(&[]);
            builder.finalize();
        }

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().map_err(|e| e.to_string())?;

        let code = module.get_finalized_function(func_id);

        Ok(JitSimdSdf {
            module,
            func_ptr: code,
        })
    }

    /// Evaluate 8 points using native SIMD (raw pointer interface)
    #[inline]
    pub unsafe fn eval_8_raw(&self, x: *const f32, y: *const f32, z: *const f32, out: *mut f32) {
        let func: SimdSdfFn = mem::transmute(self.func_ptr);
        func(x, y, z, out);
    }

    /// Evaluate 8 points using native SIMD (array interface for convenience)
    #[inline]
    pub unsafe fn eval_8(&self, x: &[f32; 8], y: &[f32; 8], z: &[f32; 8]) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        self.eval_8_raw(x.as_ptr(), y.as_ptr(), z.as_ptr(), out.as_mut_ptr());
        out
    }

    /// Evaluate many points (Zero-copy optimization)
    pub fn eval_batch(&self, x: &[f32], y: &[f32], z: &[f32]) -> Vec<f32> {
        let len = x.len();
        assert_eq!(y.len(), len);
        assert_eq!(z.len(), len);

        let mut results = vec![0.0f32; len];
        let chunk_size = 8;
        let loop_len = len - (len % chunk_size);

        // Main Loop: Zero-copy pointer arithmetic
        if loop_len > 0 {
            let x_ptr = x.as_ptr();
            let y_ptr = y.as_ptr();
            let z_ptr = z.as_ptr();
            let out_ptr = results.as_mut_ptr();

            for i in (0..loop_len).step_by(chunk_size) {
                unsafe {
                    self.eval_8_raw(x_ptr.add(i), y_ptr.add(i), z_ptr.add(i), out_ptr.add(i));
                }
            }
        }

        // Handle remainder with padding
        let remainder = len % chunk_size;
        if remainder > 0 {
            let offset = loop_len;
            let mut x_pad = [0.0f32; 8];
            let mut y_pad = [0.0f32; 8];
            let mut z_pad = [0.0f32; 8];
            let mut out_pad = [0.0f32; 8];

            unsafe {
                std::ptr::copy_nonoverlapping(
                    x.as_ptr().add(offset),
                    x_pad.as_mut_ptr(),
                    remainder,
                );
                std::ptr::copy_nonoverlapping(
                    y.as_ptr().add(offset),
                    y_pad.as_mut_ptr(),
                    remainder,
                );
                std::ptr::copy_nonoverlapping(
                    z.as_ptr().add(offset),
                    z_pad.as_mut_ptr(),
                    remainder,
                );

                self.eval_8_raw(
                    x_pad.as_ptr(),
                    y_pad.as_ptr(),
                    z_pad.as_ptr(),
                    out_pad.as_mut_ptr(),
                );

                std::ptr::copy_nonoverlapping(
                    out_pad.as_ptr(),
                    results.as_mut_ptr().add(offset),
                    remainder,
                );
            }
        }

        results
    }

    /// Evaluate SoAPoints using native SIMD
    pub fn eval_soa(&self, points: &crate::soa::SoAPoints) -> Vec<f32> {
        let (x, y, z) = points.as_slices();
        self.eval_batch(x, y, z)
    }
}

// ============ Dynamic Parameter Support ============

/// Emitter for SIMD JIT parameters (dynamic mode)
struct SimdParamEmitter {
    params_ptr: Value,
    param_index: usize,
    params: Vec<f32>,
}

impl SimdParamEmitter {
    fn new(params_ptr: Value) -> Self {
        Self {
            params_ptr,
            param_index: 0,
            params: Vec::new(),
        }
    }

    /// Emit a parameter: load scalar from params_ptr buffer
    fn emit(&mut self, builder: &mut FunctionBuilder, value: f32) -> Value {
        self.params.push(value);
        let idx = self.param_index;
        self.param_index += 1;
        let mut flags = MemFlags::new();
        flags.set_notrap();
        builder
            .ins()
            .load(types::F32, flags, self.params_ptr, (idx * 4) as i32)
    }

    /// Emit a parameter and splat to F32X4 SIMD vector
    fn emit_splat(&mut self, builder: &mut FunctionBuilder, value: f32) -> Value {
        let scalar = self.emit(builder, value);
        builder.ins().splat(types::F32X4, scalar)
    }
}

/// Dynamic SIMD coordinate state (stores splatted Value for Round/Onion)
struct DynSimdCoordState {
    x: (Value, Value),
    y: (Value, Value),
    z: (Value, Value),
    scale: (Value, Value),
    opcode: OpCode,
    /// Pre-splatted parameter vector for Round/Onion/Extrude
    param_vec: Value,
    /// Extra float params for Extrude half_height etc.
    params: [f32; 4],
}

/// Function signature for dynamic SIMD JIT: 5 pointer args
type SimdSdfDynamicFn = unsafe extern "C" fn(
    px: *const f32,
    py: *const f32,
    pz: *const f32,
    out: *mut f32,
    params: *const f32,
);

/// JIT-compiled SIMD SDF evaluator with dynamic parameter support
///
/// Unlike `JitSimdSdf`, this variant loads shape parameters from a runtime
/// buffer instead of baking them as constants. This enables zero-latency
/// parameter updates without recompilation.
pub struct JitSimdSdfDynamic {
    #[allow(dead_code)]
    module: JITModule,
    func_ptr: *const u8,
    params: Vec<f32>,
}

unsafe impl Send for JitSimdSdfDynamic {}
unsafe impl Sync for JitSimdSdfDynamic {}

impl JitSimdSdfDynamic {
    /// Compile SDF to native SIMD machine code with dynamic parameter support
    pub fn compile(sdf: &CompiledSdf) -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| e.to_string())?;
        flag_builder
            .set("use_colocated_libcalls", "true")
            .map_err(|e| e.to_string())?;
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

        let ptr_type = module.target_config().pointer_type();
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // px
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // py
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // pz
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // out
        ctx.func.signature.params.push(AbiParam::new(ptr_type)); // params_ptr

        let func_id = module
            .declare_function(
                "eval_sdf_simd_dynamic",
                Linkage::Export,
                &ctx.func.signature,
            )
            .map_err(|e| e.to_string())?;

        let initial_params;

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let ptr_x = builder.block_params(entry_block)[0];
            let ptr_y = builder.block_params(entry_block)[1];
            let ptr_z = builder.block_params(entry_block)[2];
            let ptr_out = builder.block_params(entry_block)[3];
            let params_ptr = builder.block_params(entry_block)[4];

            let vec_type = types::F32X4;

            let mut mem_flags = MemFlags::new();
            mem_flags.set_aligned();
            mem_flags.set_notrap();

            let x0 = builder.ins().load(vec_type, mem_flags, ptr_x, 0);
            let y0 = builder.ins().load(vec_type, mem_flags, ptr_y, 0);
            let z0 = builder.ins().load(vec_type, mem_flags, ptr_z, 0);
            let x1 = builder.ins().load(vec_type, mem_flags, ptr_x, 16);
            let y1 = builder.ins().load(vec_type, mem_flags, ptr_y, 16);
            let z1 = builder.ins().load(vec_type, mem_flags, ptr_z, 16);

            let zero_s = builder.ins().f32const(0.0);
            let zero_vec = builder.ins().splat(vec_type, zero_s);
            let one_s = builder.ins().f32const(1.0);
            let one_vec = builder.ins().splat(vec_type, one_s);

            let mut emitter = SimdParamEmitter::new(params_ptr);
            let mut value_stack: Vec<(Value, Value)> = Vec::with_capacity(64);
            let mut coord_stack: Vec<DynSimdCoordState> = Vec::with_capacity(32);

            let mut curr_x = (x0, x1);
            let mut curr_y = (y0, y1);
            let mut curr_z = (z0, z1);
            let mut curr_scale = (one_vec, one_vec);

            for inst in &sdf.instructions {
                match inst.opcode {
                    OpCode::Sphere => {
                        let r = emitter.emit_splat(&mut builder, inst.params[0]);

                        let zz0 = builder.ins().fmul(curr_z.0, curr_z.0);
                        let yy_zz0 = builder.ins().fma(curr_y.0, curr_y.0, zz0);
                        let len_sq0 = builder.ins().fma(curr_x.0, curr_x.0, yy_zz0);
                        let len0 = builder.ins().sqrt(len_sq0);
                        let d0 = builder.ins().fsub(len0, r);
                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);

                        let zz1 = builder.ins().fmul(curr_z.1, curr_z.1);
                        let yy_zz1 = builder.ins().fma(curr_y.1, curr_y.1, zz1);
                        let len_sq1 = builder.ins().fma(curr_x.1, curr_x.1, yy_zz1);
                        let len1 = builder.ins().sqrt(len_sq1);
                        let d1 = builder.ins().fsub(len1, r);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);

                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Box3d => {
                        let hx = emitter.emit_splat(&mut builder, inst.params[0]);
                        let hy = emitter.emit_splat(&mut builder, inst.params[1]);
                        let hz = emitter.emit_splat(&mut builder, inst.params[2]);

                        let ax0 = builder.ins().fabs(curr_x.0);
                        let ay0 = builder.ins().fabs(curr_y.0);
                        let az0 = builder.ins().fabs(curr_z.0);
                        let qx0 = builder.ins().fsub(ax0, hx);
                        let qy0 = builder.ins().fsub(ay0, hy);
                        let qz0 = builder.ins().fsub(az0, hz);
                        let mx0 = builder.ins().fmax(qx0, zero_vec);
                        let my0 = builder.ins().fmax(qy0, zero_vec);
                        let mz0 = builder.ins().fmax(qz0, zero_vec);
                        let mzz0 = builder.ins().fmul(mz0, mz0);
                        let myy_mzz0 = builder.ins().fma(my0, my0, mzz0);
                        let len_sq0 = builder.ins().fma(mx0, mx0, myy_mzz0);
                        let outside0 = builder.ins().sqrt(len_sq0);
                        let max_yz0 = builder.ins().fmax(qy0, qz0);
                        let max_xyz0 = builder.ins().fmax(qx0, max_yz0);
                        let inside0 = builder.ins().fmin(max_xyz0, zero_vec);
                        let d0 = builder.ins().fadd(outside0, inside0);
                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);

                        let ax1 = builder.ins().fabs(curr_x.1);
                        let ay1 = builder.ins().fabs(curr_y.1);
                        let az1 = builder.ins().fabs(curr_z.1);
                        let qx1 = builder.ins().fsub(ax1, hx);
                        let qy1 = builder.ins().fsub(ay1, hy);
                        let qz1 = builder.ins().fsub(az1, hz);
                        let mx1 = builder.ins().fmax(qx1, zero_vec);
                        let my1 = builder.ins().fmax(qy1, zero_vec);
                        let mz1 = builder.ins().fmax(qz1, zero_vec);
                        let mzz1 = builder.ins().fmul(mz1, mz1);
                        let myy_mzz1 = builder.ins().fma(my1, my1, mzz1);
                        let len_sq1 = builder.ins().fma(mx1, mx1, myy_mzz1);
                        let outside1 = builder.ins().sqrt(len_sq1);
                        let max_yz1 = builder.ins().fmax(qy1, qz1);
                        let max_xyz1 = builder.ins().fmax(qx1, max_yz1);
                        let inside1 = builder.ins().fmin(max_xyz1, zero_vec);
                        let d1 = builder.ins().fadd(outside1, inside1);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);

                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Cylinder => {
                        let r = emitter.emit_splat(&mut builder, inst.params[0]);
                        let h = emitter.emit_splat(&mut builder, inst.params[1]);

                        let zz0 = builder.ins().fmul(curr_z.0, curr_z.0);
                        let xz_sq0 = builder.ins().fma(curr_x.0, curr_x.0, zz0);
                        let xz_len0 = builder.ins().sqrt(xz_sq0);
                        let dx0 = builder.ins().fsub(xz_len0, r);
                        let ay0 = builder.ins().fabs(curr_y.0);
                        let dy0 = builder.ins().fsub(ay0, h);
                        let mx0 = builder.ins().fmax(dx0, zero_vec);
                        let my0 = builder.ins().fmax(dy0, zero_vec);
                        let mzz0 = builder.ins().fmul(my0, my0);
                        let len_sq0 = builder.ins().fma(mx0, mx0, mzz0);
                        let outside0 = builder.ins().sqrt(len_sq0);
                        let im0 = builder.ins().fmax(dx0, dy0);
                        let inside0 = builder.ins().fmin(im0, zero_vec);
                        let d0 = builder.ins().fadd(outside0, inside0);
                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);

                        let zz1 = builder.ins().fmul(curr_z.1, curr_z.1);
                        let xz_sq1 = builder.ins().fma(curr_x.1, curr_x.1, zz1);
                        let xz_len1 = builder.ins().sqrt(xz_sq1);
                        let dx1 = builder.ins().fsub(xz_len1, r);
                        let ay1 = builder.ins().fabs(curr_y.1);
                        let dy1 = builder.ins().fsub(ay1, h);
                        let mx1 = builder.ins().fmax(dx1, zero_vec);
                        let my1 = builder.ins().fmax(dy1, zero_vec);
                        let mzz1 = builder.ins().fmul(my1, my1);
                        let len_sq1 = builder.ins().fma(mx1, mx1, mzz1);
                        let outside1 = builder.ins().sqrt(len_sq1);
                        let im1 = builder.ins().fmax(dx1, dy1);
                        let inside1 = builder.ins().fmin(im1, zero_vec);
                        let d1 = builder.ins().fadd(outside1, inside1);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);

                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Plane => {
                        let nx = emitter.emit_splat(&mut builder, inst.params[0]);
                        let ny = emitter.emit_splat(&mut builder, inst.params[1]);
                        let nz = emitter.emit_splat(&mut builder, inst.params[2]);
                        let dist = emitter.emit_splat(&mut builder, inst.params[3]);

                        let z_nz0 = builder.ins().fmul(curr_z.0, nz);
                        let y_ny0 = builder.ins().fma(curr_y.0, ny, z_nz0);
                        let dot0 = builder.ins().fma(curr_x.0, nx, y_ny0);
                        let d0 = builder.ins().fadd(dot0, dist);
                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);

                        let z_nz1 = builder.ins().fmul(curr_z.1, nz);
                        let y_ny1 = builder.ins().fma(curr_y.1, ny, z_nz1);
                        let dot1 = builder.ins().fma(curr_x.1, nx, y_ny1);
                        let d1 = builder.ins().fadd(dot1, dist);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);

                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Torus => {
                        let maj = emitter.emit_splat(&mut builder, inst.params[0]);
                        let min = emitter.emit_splat(&mut builder, inst.params[1]);

                        let zz0 = builder.ins().fmul(curr_z.0, curr_z.0);
                        let xz_sq0 = builder.ins().fma(curr_x.0, curr_x.0, zz0);
                        let xz0 = builder.ins().sqrt(xz_sq0);
                        let qx0 = builder.ins().fsub(xz0, maj);
                        let yy0 = builder.ins().fmul(curr_y.0, curr_y.0);
                        let q_sq0 = builder.ins().fma(qx0, qx0, yy0);
                        let q0 = builder.ins().sqrt(q_sq0);
                        let d0 = builder.ins().fsub(q0, min);
                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);

                        let zz1 = builder.ins().fmul(curr_z.1, curr_z.1);
                        let xz_sq1 = builder.ins().fma(curr_x.1, curr_x.1, zz1);
                        let xz1 = builder.ins().sqrt(xz_sq1);
                        let qx1 = builder.ins().fsub(xz1, maj);
                        let yy1 = builder.ins().fmul(curr_y.1, curr_y.1);
                        let q_sq1 = builder.ins().fma(qx1, qx1, yy1);
                        let q1 = builder.ins().sqrt(q_sq1);
                        let d1 = builder.ins().fsub(q1, min);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);

                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Capsule => {
                        let ax = emitter.emit_splat(&mut builder, inst.params[0]);
                        let ay = emitter.emit_splat(&mut builder, inst.params[1]);
                        let az = emitter.emit_splat(&mut builder, inst.params[2]);
                        let radius = inst.get_capsule_radius();
                        let r_v = emitter.emit_splat(&mut builder, radius);

                        let ba_x = inst.params[3] - inst.params[0];
                        let ba_y = inst.params[4] - inst.params[1];
                        let ba_z = inst.params[5] - inst.params[2];
                        let ba_dot_val = ba_x * ba_x + ba_y * ba_y + ba_z * ba_z;
                        let inv_ba_dot = if ba_dot_val.abs() < 1e-10 {
                            1.0
                        } else {
                            1.0 / ba_dot_val
                        };

                        let bax = emitter.emit_splat(&mut builder, ba_x);
                        let bay = emitter.emit_splat(&mut builder, ba_y);
                        let baz = emitter.emit_splat(&mut builder, ba_z);
                        let inv_bd = emitter.emit_splat(&mut builder, inv_ba_dot);

                        // Lane 0
                        let pax0 = builder.ins().fsub(curr_x.0, ax);
                        let pay0 = builder.ins().fsub(curr_y.0, ay);
                        let paz0 = builder.ins().fsub(curr_z.0, az);
                        let dot_z0 = builder.ins().fmul(paz0, baz);
                        let dot_yz0 = builder.ins().fma(pay0, bay, dot_z0);
                        let dot0 = builder.ins().fma(pax0, bax, dot_yz0);
                        let h_raw0 = builder.ins().fmul(dot0, inv_bd);
                        let h_min0 = builder.ins().fmin(h_raw0, one_vec);
                        let h0 = builder.ins().fmax(h_min0, zero_vec);
                        let bhx0 = builder.ins().fmul(bax, h0);
                        let bhy0 = builder.ins().fmul(bay, h0);
                        let bhz0 = builder.ins().fmul(baz, h0);
                        let dx0 = builder.ins().fsub(pax0, bhx0);
                        let dy0 = builder.ins().fsub(pay0, bhy0);
                        let dz0 = builder.ins().fsub(paz0, bhz0);
                        let len0 = simd_length3_fma(&mut builder, dx0, dy0, dz0);
                        let d0 = builder.ins().fsub(len0, r_v);
                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);

                        // Lane 1
                        let pax1 = builder.ins().fsub(curr_x.1, ax);
                        let pay1 = builder.ins().fsub(curr_y.1, ay);
                        let paz1 = builder.ins().fsub(curr_z.1, az);
                        let dot_z1 = builder.ins().fmul(paz1, baz);
                        let dot_yz1 = builder.ins().fma(pay1, bay, dot_z1);
                        let dot1 = builder.ins().fma(pax1, bax, dot_yz1);
                        let h_raw1 = builder.ins().fmul(dot1, inv_bd);
                        let h_min1 = builder.ins().fmin(h_raw1, one_vec);
                        let h1 = builder.ins().fmax(h_min1, zero_vec);
                        let bhx1 = builder.ins().fmul(bax, h1);
                        let bhy1 = builder.ins().fmul(bay, h1);
                        let bhz1 = builder.ins().fmul(baz, h1);
                        let dx1 = builder.ins().fsub(pax1, bhx1);
                        let dy1 = builder.ins().fsub(pay1, bhy1);
                        let dz1 = builder.ins().fsub(paz1, bhz1);
                        let len1 = simd_length3_fma(&mut builder, dx1, dy1, dz1);
                        let d1 = builder.ins().fsub(len1, r_v);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);

                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Cone => {
                        let radius_val = inst.params[0];
                        let half_height = inst.params[1];

                        let r = emitter.emit_splat(&mut builder, radius_val);
                        let h = emitter.emit_splat(&mut builder, half_height);

                        let k2x_val = -radius_val;
                        let k2y_val = 2.0 * half_height;
                        let k2_dot_val = k2x_val * k2x_val + k2y_val * k2y_val;
                        let inv_k2d_val = if k2_dot_val.abs() < 1e-10 {
                            1.0
                        } else {
                            1.0 / k2_dot_val
                        };

                        let k2x = emitter.emit_splat(&mut builder, k2x_val);
                        let k2y = emitter.emit_splat(&mut builder, k2y_val);
                        let inv_k2d = emitter.emit_splat(&mut builder, inv_k2d_val);
                        let neg_one_s = builder.ins().f32const(-1.0);
                        let neg_one_v = builder.ins().splat(vec_type, neg_one_s);

                        // Lane 0
                        let q_x0 = simd_length2_fma(&mut builder, curr_x.0, curr_z.0);
                        let q_y0 = curr_y.0;
                        let ca_r0 = simd_select_neg(&mut builder, q_y0, r, zero_vec);
                        let min_q_ca0 = builder.ins().fmin(q_x0, ca_r0);
                        let ca_x0 = builder.ins().fsub(q_x0, min_q_ca0);
                        let abs_qy0 = builder.ins().fabs(q_y0);
                        let ca_y0 = builder.ins().fsub(abs_qy0, h);
                        let neg_qx0 = builder.ins().fneg(q_x0);
                        let diff_y0 = builder.ins().fsub(h, q_y0);
                        let dy_k2y0 = builder.ins().fmul(diff_y0, k2y);
                        let nqx_k2x0 = builder.ins().fma(neg_qx0, k2x, dy_k2y0);
                        let t_raw0 = builder.ins().fmul(nqx_k2x0, inv_k2d);
                        let t_min0 = builder.ins().fmin(t_raw0, one_vec);
                        let t0 = builder.ins().fmax(zero_vec, t_min0);
                        let k2x_t0 = builder.ins().fmul(k2x, t0);
                        let cb_x0 = builder.ins().fadd(q_x0, k2x_t0);
                        let qy_h0 = builder.ins().fsub(q_y0, h);
                        let k2y_t0 = builder.ins().fmul(k2y, t0);
                        let cb_y0 = builder.ins().fadd(qy_h0, k2y_t0);
                        let both_neg_cond0 = builder.ins().fmax(cb_x0, ca_y0);
                        let s0 = simd_select_neg(&mut builder, both_neg_cond0, neg_one_v, one_vec);
                        let ca_sq0 = {
                            let xx = builder.ins().fmul(ca_x0, ca_x0);
                            builder.ins().fma(ca_y0, ca_y0, xx)
                        };
                        let cb_sq0 = {
                            let xx = builder.ins().fmul(cb_x0, cb_x0);
                            builder.ins().fma(cb_y0, cb_y0, xx)
                        };
                        let d2_0 = builder.ins().fmin(ca_sq0, cb_sq0);
                        let dist0 = builder.ins().sqrt(d2_0);
                        let d0 = builder.ins().fmul(s0, dist0);

                        // Lane 1
                        let q_x1 = simd_length2_fma(&mut builder, curr_x.1, curr_z.1);
                        let q_y1 = curr_y.1;
                        let ca_r1 = simd_select_neg(&mut builder, q_y1, r, zero_vec);
                        let min_q_ca1 = builder.ins().fmin(q_x1, ca_r1);
                        let ca_x1 = builder.ins().fsub(q_x1, min_q_ca1);
                        let abs_qy1 = builder.ins().fabs(q_y1);
                        let ca_y1 = builder.ins().fsub(abs_qy1, h);
                        let neg_qx1 = builder.ins().fneg(q_x1);
                        let diff_y1 = builder.ins().fsub(h, q_y1);
                        let dy_k2y1 = builder.ins().fmul(diff_y1, k2y);
                        let nqx_k2x1 = builder.ins().fma(neg_qx1, k2x, dy_k2y1);
                        let t_raw1 = builder.ins().fmul(nqx_k2x1, inv_k2d);
                        let t_min1 = builder.ins().fmin(t_raw1, one_vec);
                        let t1 = builder.ins().fmax(zero_vec, t_min1);
                        let k2x_t1 = builder.ins().fmul(k2x, t1);
                        let cb_x1 = builder.ins().fadd(q_x1, k2x_t1);
                        let qy_h1 = builder.ins().fsub(q_y1, h);
                        let k2y_t1 = builder.ins().fmul(k2y, t1);
                        let cb_y1 = builder.ins().fadd(qy_h1, k2y_t1);
                        let both_neg_cond1 = builder.ins().fmax(cb_x1, ca_y1);
                        let s1 = simd_select_neg(&mut builder, both_neg_cond1, neg_one_v, one_vec);
                        let ca_sq1 = {
                            let xx = builder.ins().fmul(ca_x1, ca_x1);
                            builder.ins().fma(ca_y1, ca_y1, xx)
                        };
                        let cb_sq1 = {
                            let xx = builder.ins().fmul(cb_x1, cb_x1);
                            builder.ins().fma(cb_y1, cb_y1, xx)
                        };
                        let d2_1 = builder.ins().fmin(ca_sq1, cb_sq1);
                        let dist1 = builder.ins().sqrt(d2_1);
                        let d1 = builder.ins().fmul(s1, dist1);

                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Ellipsoid => {
                        let inv_rx = emitter.emit_splat(&mut builder, 1.0 / inst.params[0]);
                        let inv_ry = emitter.emit_splat(&mut builder, 1.0 / inst.params[1]);
                        let inv_rz = emitter.emit_splat(&mut builder, 1.0 / inst.params[2]);
                        let inv_rx2 = emitter
                            .emit_splat(&mut builder, 1.0 / (inst.params[0] * inst.params[0]));
                        let inv_ry2 = emitter
                            .emit_splat(&mut builder, 1.0 / (inst.params[1] * inst.params[1]));
                        let inv_rz2 = emitter
                            .emit_splat(&mut builder, 1.0 / (inst.params[2] * inst.params[2]));
                        let eps_s = builder.ins().f32const(1e-10);
                        let eps = builder.ins().splat(vec_type, eps_s);

                        // Lane 0
                        let px0 = builder.ins().fmul(curr_x.0, inv_rx);
                        let py0 = builder.ins().fmul(curr_y.0, inv_ry);
                        let pz0 = builder.ins().fmul(curr_z.0, inv_rz);
                        let k0_0 = simd_length3_fma(&mut builder, px0, py0, pz0);
                        let qx0 = builder.ins().fmul(curr_x.0, inv_rx2);
                        let qy0 = builder.ins().fmul(curr_y.0, inv_ry2);
                        let qz0 = builder.ins().fmul(curr_z.0, inv_rz2);
                        let k1_0 = simd_length3_fma(&mut builder, qx0, qy0, qz0);
                        let k1_safe0 = builder.ins().fadd(k1_0, eps);
                        let k0_m1_0 = builder.ins().fsub(k0_0, one_vec);
                        let num0 = builder.ins().fmul(k0_0, k0_m1_0);
                        let d0 = builder.ins().fdiv(num0, k1_safe0);

                        // Lane 1
                        let px1 = builder.ins().fmul(curr_x.1, inv_rx);
                        let py1 = builder.ins().fmul(curr_y.1, inv_ry);
                        let pz1 = builder.ins().fmul(curr_z.1, inv_rz);
                        let k0_1 = simd_length3_fma(&mut builder, px1, py1, pz1);
                        let qx1 = builder.ins().fmul(curr_x.1, inv_rx2);
                        let qy1 = builder.ins().fmul(curr_y.1, inv_ry2);
                        let qz1 = builder.ins().fmul(curr_z.1, inv_rz2);
                        let k1_1 = simd_length3_fma(&mut builder, qx1, qy1, qz1);
                        let k1_safe1 = builder.ins().fadd(k1_1, eps);
                        let k0_m1_1 = builder.ins().fsub(k0_1, one_vec);
                        let num1 = builder.ins().fmul(k0_1, k0_m1_1);
                        let d1 = builder.ins().fdiv(num1, k1_safe1);

                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::RoundedCone => {
                        let r1_val = inst.params[0];
                        let r2_val = inst.params[1];
                        let half_height = inst.params[2];
                        let h_val = half_height * 2.0;
                        let inv_h = if h_val.abs() < 1e-10 {
                            1.0
                        } else {
                            1.0 / h_val
                        };
                        let b_val = (r1_val - r2_val) * inv_h;
                        let a_val = (1.0 - b_val * b_val).max(0.0).sqrt();
                        let ah_val = a_val * h_val;

                        let r1_v = emitter.emit_splat(&mut builder, r1_val);
                        let r2_v = emitter.emit_splat(&mut builder, r2_val);
                        let hh_v = emitter.emit_splat(&mut builder, half_height);
                        let h_v = emitter.emit_splat(&mut builder, h_val);
                        let b_v = emitter.emit_splat(&mut builder, b_val);
                        let a_v = emitter.emit_splat(&mut builder, a_val);
                        let ah_v = emitter.emit_splat(&mut builder, ah_val);
                        let neg_b = builder.ins().fneg(b_v);

                        // Lane 0
                        let q_x0 = simd_length2_fma(&mut builder, curr_x.0, curr_z.0);
                        let q_y0 = builder.ins().fadd(curr_y.0, hh_v);
                        let qx_nb0 = builder.ins().fmul(q_x0, neg_b);
                        let k0 = builder.ins().fma(q_y0, a_v, qx_nb0);
                        let len1_0 = simd_length2_fma(&mut builder, q_x0, q_y0);
                        let d1_0 = builder.ins().fsub(len1_0, r1_v);
                        let qy_h0 = builder.ins().fsub(q_y0, h_v);
                        let len2_0 = simd_length2_fma(&mut builder, q_x0, qy_h0);
                        let d2_0 = builder.ins().fsub(len2_0, r2_v);
                        let qyb0 = builder.ins().fmul(q_y0, b_v);
                        let d3_0 = builder.ins().fma(q_x0, a_v, qyb0);
                        let d3_0 = builder.ins().fsub(d3_0, r1_v);
                        let k_ah0 = builder.ins().fsub(ah_v, k0);
                        let inner0 = simd_select_neg(&mut builder, k_ah0, d2_0, d3_0);
                        let d0 = simd_select_neg(&mut builder, k0, d1_0, inner0);

                        // Lane 1
                        let q_x1 = simd_length2_fma(&mut builder, curr_x.1, curr_z.1);
                        let q_y1 = builder.ins().fadd(curr_y.1, hh_v);
                        let qx_nb1 = builder.ins().fmul(q_x1, neg_b);
                        let k1 = builder.ins().fma(q_y1, a_v, qx_nb1);
                        let len1_1 = simd_length2_fma(&mut builder, q_x1, q_y1);
                        let d1_1 = builder.ins().fsub(len1_1, r1_v);
                        let qy_h1 = builder.ins().fsub(q_y1, h_v);
                        let len2_1 = simd_length2_fma(&mut builder, q_x1, qy_h1);
                        let d2_1 = builder.ins().fsub(len2_1, r2_v);
                        let qyb1 = builder.ins().fmul(q_y1, b_v);
                        let d3_1 = builder.ins().fma(q_x1, a_v, qyb1);
                        let d3_1 = builder.ins().fsub(d3_1, r1_v);
                        let k_ah1 = builder.ins().fsub(ah_v, k1);
                        let inner1 = simd_select_neg(&mut builder, k_ah1, d2_1, d3_1);
                        let d1 = simd_select_neg(&mut builder, k1, d1_1, inner1);

                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Pyramid => {
                        let half_height = inst.params[0];
                        let h_val = half_height * 2.0;
                        let m2_val = h_val * h_val + 0.25;
                        let inv_m2_val = 1.0 / m2_val;
                        let inv_m2_025_val = 1.0 / (m2_val + 0.25);

                        let hh_v = emitter.emit_splat(&mut builder, half_height);
                        let h_v = emitter.emit_splat(&mut builder, h_val);
                        let m2_v = emitter.emit_splat(&mut builder, m2_val);
                        let inv_m2_v = emitter.emit_splat(&mut builder, inv_m2_val);
                        let inv_m2_025_v = emitter.emit_splat(&mut builder, inv_m2_025_val);
                        let half_s = builder.ins().f32const(0.5);
                        let neg_half_s = builder.ins().f32const(-0.5);
                        let neg_one_s = builder.ins().f32const(-1.0);
                        let half_v = builder.ins().splat(vec_type, half_s);
                        let neg_half_v = builder.ins().splat(vec_type, neg_half_s);
                        let neg_one_v = builder.ins().splat(vec_type, neg_one_s);

                        // Lane 0
                        let py0 = builder.ins().fadd(curr_y.0, hh_v);
                        let abs_px0 = builder.ins().fabs(curr_x.0);
                        let abs_pz0 = builder.ins().fabs(curr_z.0);
                        let px_s0 = builder.ins().fmax(abs_px0, abs_pz0);
                        let pz_s0 = builder.ins().fmin(abs_px0, abs_pz0);
                        let px_adj0 = builder.ins().fsub(px_s0, half_v);
                        let pz_adj0 = builder.ins().fsub(pz_s0, half_v);
                        let qx0 = pz_adj0;
                        let nhalf_px0 = builder.ins().fmul(neg_half_v, px_adj0);
                        let qy0 = builder.ins().fma(h_v, py0, nhalf_px0);
                        let half_py0 = builder.ins().fmul(half_v, py0);
                        let qz0 = builder.ins().fma(h_v, px_adj0, half_py0);
                        let neg_qx0 = builder.ins().fneg(qx0);
                        let s0 = builder.ins().fmax(neg_qx0, zero_vec);
                        let half_pz0 = builder.ins().fmul(half_v, pz_adj0);
                        let qy_sub0 = builder.ins().fsub(qy0, half_pz0);
                        let t_raw0 = builder.ins().fmul(qy_sub0, inv_m2_025_v);
                        let t_min0 = builder.ins().fmin(t_raw0, one_vec);
                        let t0 = builder.ins().fmax(zero_vec, t_min0);
                        let qx_s0 = builder.ins().fadd(qx0, s0);
                        let qx_s_sq0 = builder.ins().fmul(qx_s0, qx_s0);
                        let m2_qxs0 = builder.ins().fmul(m2_v, qx_s_sq0);
                        let a0 = builder.ins().fma(qy0, qy0, m2_qxs0);
                        let half_t0 = builder.ins().fmul(half_v, t0);
                        let qx_ht0 = builder.ins().fadd(qx0, half_t0);
                        let qx_ht_sq0 = builder.ins().fmul(qx_ht0, qx_ht0);
                        let m2_t0 = builder.ins().fmul(m2_v, t0);
                        let qy_m2t0 = builder.ins().fsub(qy0, m2_t0);
                        let m2_qxht0 = builder.ins().fmul(m2_v, qx_ht_sq0);
                        let b0 = builder.ins().fma(qy_m2t0, qy_m2t0, m2_qxht0);
                        let neg_qx_m2_0 = builder.ins().fmul(neg_qx0, m2_v);
                        let half_qy0 = builder.ins().fmul(half_v, qy0);
                        let cond0 = builder.ins().fsub(neg_qx_m2_0, half_qy0);
                        let min_cond0 = builder.ins().fmin(qy0, cond0);
                        let ab_min0 = builder.ins().fmin(a0, b0);
                        let neg_min_cond0 = builder.ins().fneg(min_cond0);
                        let d2_0 = simd_select_neg(&mut builder, neg_min_cond0, zero_vec, ab_min0);
                        let qz_sq0 = builder.ins().fmul(qz0, qz0);
                        let d2_qz0 = builder.ins().fadd(d2_0, qz_sq0);
                        let d2_sc0 = builder.ins().fmul(d2_qz0, inv_m2_v);
                        let dist0 = builder.ins().sqrt(d2_sc0);
                        let neg_py0 = builder.ins().fneg(py0);
                        let sign_arg0 = builder.ins().fmax(qz0, neg_py0);
                        let signed0 = simd_select_neg(&mut builder, sign_arg0, neg_one_v, one_vec);
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
                        let qy1 = builder.ins().fma(h_v, py1, nhalf_px1);
                        let half_py1 = builder.ins().fmul(half_v, py1);
                        let qz1 = builder.ins().fma(h_v, px_adj1, half_py1);
                        let neg_qx1 = builder.ins().fneg(qx1);
                        let s1 = builder.ins().fmax(neg_qx1, zero_vec);
                        let half_pz1 = builder.ins().fmul(half_v, pz_adj1);
                        let qy_sub1 = builder.ins().fsub(qy1, half_pz1);
                        let t_raw1 = builder.ins().fmul(qy_sub1, inv_m2_025_v);
                        let t_min1 = builder.ins().fmin(t_raw1, one_vec);
                        let t1 = builder.ins().fmax(zero_vec, t_min1);
                        let qx_s1 = builder.ins().fadd(qx1, s1);
                        let qx_s_sq1 = builder.ins().fmul(qx_s1, qx_s1);
                        let m2_qxs1 = builder.ins().fmul(m2_v, qx_s_sq1);
                        let a1 = builder.ins().fma(qy1, qy1, m2_qxs1);
                        let half_t1 = builder.ins().fmul(half_v, t1);
                        let qx_ht1 = builder.ins().fadd(qx1, half_t1);
                        let qx_ht_sq1 = builder.ins().fmul(qx_ht1, qx_ht1);
                        let m2_t1 = builder.ins().fmul(m2_v, t1);
                        let qy_m2t1 = builder.ins().fsub(qy1, m2_t1);
                        let m2_qxht1 = builder.ins().fmul(m2_v, qx_ht_sq1);
                        let b1 = builder.ins().fma(qy_m2t1, qy_m2t1, m2_qxht1);
                        let neg_qx_m2_1 = builder.ins().fmul(neg_qx1, m2_v);
                        let half_qy1 = builder.ins().fmul(half_v, qy1);
                        let cond1 = builder.ins().fsub(neg_qx_m2_1, half_qy1);
                        let min_cond1 = builder.ins().fmin(qy1, cond1);
                        let ab_min1 = builder.ins().fmin(a1, b1);
                        let neg_min_cond1 = builder.ins().fneg(min_cond1);
                        let d2_1 = simd_select_neg(&mut builder, neg_min_cond1, zero_vec, ab_min1);
                        let qz_sq1 = builder.ins().fmul(qz1, qz1);
                        let d2_qz1 = builder.ins().fadd(d2_1, qz_sq1);
                        let d2_sc1 = builder.ins().fmul(d2_qz1, inv_m2_v);
                        let dist1 = builder.ins().sqrt(d2_sc1);
                        let neg_py1 = builder.ins().fneg(py1);
                        let sign_arg1 = builder.ins().fmax(qz1, neg_py1);
                        let signed1 = simd_select_neg(&mut builder, sign_arg1, neg_one_v, one_vec);
                        let d1 = builder.ins().fmul(signed1, dist1);

                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Octahedron => {
                        let size_val = inst.params[0];
                        let s_v = emitter.emit_splat(&mut builder, size_val);
                        let three_s = builder.ins().f32const(3.0);
                        let half_s = builder.ins().f32const(0.5);
                        let inv_sqrt3_s = builder.ins().f32const(0.57735027);
                        let three_v = builder.ins().splat(vec_type, three_s);
                        let half_v = builder.ins().splat(vec_type, half_s);
                        let inv_sqrt3_v = builder.ins().splat(vec_type, inv_sqrt3_s);

                        // Lane 0
                        let apx0 = builder.ins().fabs(curr_x.0);
                        let apy0 = builder.ins().fabs(curr_y.0);
                        let apz0 = builder.ins().fabs(curr_z.0);
                        let sum_xy0 = builder.ins().fadd(apx0, apy0);
                        let sum_xyz0 = builder.ins().fadd(sum_xy0, apz0);
                        let m0 = builder.ins().fsub(sum_xyz0, s_v);
                        let tpx0 = builder.ins().fmul(three_v, apx0);
                        let tpy0 = builder.ins().fmul(three_v, apy0);
                        let tpz0 = builder.ins().fmul(three_v, apz0);
                        let c1_0 = builder.ins().fsub(tpx0, m0);
                        let c2_0 = builder.ins().fsub(tpy0, m0);
                        let c3_0 = builder.ins().fsub(tpz0, m0);
                        let qx_c2_0 = simd_select_neg(&mut builder, c2_0, apy0, apz0);
                        let qy_c2_0 = simd_select_neg(&mut builder, c2_0, apz0, apx0);
                        let qz_c2_0 = simd_select_neg(&mut builder, c2_0, apx0, apy0);
                        let qx0 = simd_select_neg(&mut builder, c1_0, apx0, qx_c2_0);
                        let qy0 = simd_select_neg(&mut builder, c1_0, apy0, qy_c2_0);
                        let qz0 = simd_select_neg(&mut builder, c1_0, apz0, qz_c2_0);
                        let qz_qy0 = builder.ins().fsub(qz0, qy0);
                        let qz_qy_s0 = builder.ins().fadd(qz_qy0, s_v);
                        let hv0 = builder.ins().fmul(half_v, qz_qy_s0);
                        let k_min0 = builder.ins().fmin(hv0, s_v);
                        let k0 = builder.ins().fmax(zero_vec, k_min0);
                        let qy_s0 = builder.ins().fsub(qy0, s_v);
                        let qy_sk0 = builder.ins().fadd(qy_s0, k0);
                        let qz_k0 = builder.ins().fsub(qz0, k0);
                        let detail0 = simd_length3_fma(&mut builder, qx0, qy_sk0, qz_k0);
                        let early0 = builder.ins().fmul(m0, inv_sqrt3_v);
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
                        let k1 = builder.ins().fmax(zero_vec, k_min1);
                        let qy_s1 = builder.ins().fsub(qy1, s_v);
                        let qy_sk1 = builder.ins().fadd(qy_s1, k1);
                        let qz_k1 = builder.ins().fsub(qz1, k1);
                        let detail1 = simd_length3_fma(&mut builder, qx1, qy_sk1, qz_k1);
                        let early1 = builder.ins().fmul(m1, inv_sqrt3_v);
                        let min_c23_1 = builder.ins().fmin(c2_1, c3_1);
                        let any_cond1 = builder.ins().fmin(c1_1, min_c23_1);
                        let d1 = simd_select_neg(&mut builder, any_cond1, detail1, early1);

                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::HexPrism => {
                        let hex_radius = inst.params[0];
                        let half_height = inst.params[1];
                        let kz_val: f32 = 0.57735027;
                        let kz_hr = kz_val * hex_radius;

                        let hr_v = emitter.emit_splat(&mut builder, hex_radius);
                        let hh_v = emitter.emit_splat(&mut builder, half_height);
                        let kx_s = builder.ins().f32const(-0.8660254);
                        let ky_s = builder.ins().f32const(0.5);
                        let two_s = builder.ins().f32const(2.0);
                        let neg_one_s = builder.ins().f32const(-1.0);
                        let neg_kz_hr_s = builder.ins().f32const(-kz_hr);
                        let kz_hr_s = builder.ins().f32const(kz_hr);
                        let kx_v = builder.ins().splat(vec_type, kx_s);
                        let ky_v = builder.ins().splat(vec_type, ky_s);
                        let two_v = builder.ins().splat(vec_type, two_s);
                        let neg_one_v = builder.ins().splat(vec_type, neg_one_s);
                        let neg_kz_hr_v = builder.ins().splat(vec_type, neg_kz_hr_s);
                        let kz_hr_v = builder.ins().splat(vec_type, kz_hr_s);

                        // Lane 0
                        let apx0 = builder.ins().fabs(curr_x.0);
                        let apy0 = builder.ins().fabs(curr_y.0);
                        let apz0 = builder.ins().fabs(curr_z.0);
                        let ky_py0 = builder.ins().fmul(ky_v, apy0);
                        let dot0 = builder.ins().fma(kx_v, apx0, ky_py0);
                        let dot_min0 = builder.ins().fmin(dot0, zero_vec);
                        let reflect0 = builder.ins().fmul(two_v, dot_min0);
                        let rkx0 = builder.ins().fmul(reflect0, kx_v);
                        let rky0 = builder.ins().fmul(reflect0, ky_v);
                        let px_r0 = builder.ins().fsub(apx0, rkx0);
                        let py_r0 = builder.ins().fsub(apy0, rky0);
                        let px_cl0 = builder.ins().fmin(px_r0, kz_hr_v);
                        let clamped0 = builder.ins().fmax(neg_kz_hr_v, px_cl0);
                        let dx0 = builder.ins().fsub(px_r0, clamped0);
                        let dy0 = builder.ins().fsub(py_r0, hr_v);
                        let len_dxy0 = simd_length2_fma(&mut builder, dx0, dy0);
                        let sign_dy0 = simd_select_neg(&mut builder, dy0, neg_one_v, one_vec);
                        let d_xy0 = builder.ins().fmul(len_dxy0, sign_dy0);
                        let d_z0 = builder.ins().fsub(apz0, hh_v);
                        let max_dd0 = builder.ins().fmax(d_xy0, d_z0);
                        let interior0 = builder.ins().fmin(max_dd0, zero_vec);
                        let d_xy_p0 = builder.ins().fmax(d_xy0, zero_vec);
                        let d_z_p0 = builder.ins().fmax(d_z0, zero_vec);
                        let exterior0 = simd_length2_fma(&mut builder, d_xy_p0, d_z_p0);
                        let d0 = builder.ins().fadd(interior0, exterior0);

                        // Lane 1
                        let apx1 = builder.ins().fabs(curr_x.1);
                        let apy1 = builder.ins().fabs(curr_y.1);
                        let apz1 = builder.ins().fabs(curr_z.1);
                        let ky_py1 = builder.ins().fmul(ky_v, apy1);
                        let dot1 = builder.ins().fma(kx_v, apx1, ky_py1);
                        let dot_min1 = builder.ins().fmin(dot1, zero_vec);
                        let reflect1 = builder.ins().fmul(two_v, dot_min1);
                        let rkx1 = builder.ins().fmul(reflect1, kx_v);
                        let rky1 = builder.ins().fmul(reflect1, ky_v);
                        let px_r1 = builder.ins().fsub(apx1, rkx1);
                        let py_r1 = builder.ins().fsub(apy1, rky1);
                        let px_cl1 = builder.ins().fmin(px_r1, kz_hr_v);
                        let clamped1 = builder.ins().fmax(neg_kz_hr_v, px_cl1);
                        let dx1 = builder.ins().fsub(px_r1, clamped1);
                        let dy1 = builder.ins().fsub(py_r1, hr_v);
                        let len_dxy1 = simd_length2_fma(&mut builder, dx1, dy1);
                        let sign_dy1 = simd_select_neg(&mut builder, dy1, neg_one_v, one_vec);
                        let d_xy1 = builder.ins().fmul(len_dxy1, sign_dy1);
                        let d_z1 = builder.ins().fsub(apz1, hh_v);
                        let max_dd1 = builder.ins().fmax(d_xy1, d_z1);
                        let interior1 = builder.ins().fmin(max_dd1, zero_vec);
                        let d_xy_p1 = builder.ins().fmax(d_xy1, zero_vec);
                        let d_z_p1 = builder.ins().fmax(d_z1, zero_vec);
                        let exterior1 = simd_length2_fma(&mut builder, d_xy_p1, d_z_p1);
                        let d1 = builder.ins().fadd(interior1, exterior1);

                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);
                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Link => {
                        let hl = emitter.emit_splat(&mut builder, inst.params[0]);
                        let r1_v = emitter.emit_splat(&mut builder, inst.params[1]);
                        let r2_v = emitter.emit_splat(&mut builder, inst.params[2]);

                        // Lane 0
                        let abs_y0 = builder.ins().fabs(curr_y.0);
                        let y_sub0 = builder.ins().fsub(abs_y0, hl);
                        let qy0 = builder.ins().fmax(y_sub0, zero_vec);
                        let qyy0 = builder.ins().fmul(qy0, qy0);
                        let xy_sq0 = builder.ins().fma(curr_x.0, curr_x.0, qyy0);
                        let xy_len0 = builder.ins().sqrt(xy_sq0);
                        let xy_sub0 = builder.ins().fsub(xy_len0, r1_v);
                        let zz0 = builder.ins().fmul(curr_z.0, curr_z.0);
                        let d_sq0 = builder.ins().fma(xy_sub0, xy_sub0, zz0);
                        let d_len0 = builder.ins().sqrt(d_sq0);
                        let d0 = builder.ins().fsub(d_len0, r2_v);
                        let d0_scaled = builder.ins().fmul(d0, curr_scale.0);

                        // Lane 1
                        let abs_y1 = builder.ins().fabs(curr_y.1);
                        let y_sub1 = builder.ins().fsub(abs_y1, hl);
                        let qy1 = builder.ins().fmax(y_sub1, zero_vec);
                        let qyy1 = builder.ins().fmul(qy1, qy1);
                        let xy_sq1 = builder.ins().fma(curr_x.1, curr_x.1, qyy1);
                        let xy_len1 = builder.ins().sqrt(xy_sq1);
                        let xy_sub1 = builder.ins().fsub(xy_len1, r1_v);
                        let zz1 = builder.ins().fmul(curr_z.1, curr_z.1);
                        let d_sq1 = builder.ins().fma(xy_sub1, xy_sub1, zz1);
                        let d_len1 = builder.ins().sqrt(d_sq1);
                        let d1 = builder.ins().fsub(d_len1, r2_v);
                        let d1_scaled = builder.ins().fmul(d1, curr_scale.1);

                        value_stack.push((d0_scaled, d1_scaled));
                    }

                    OpCode::Union => {
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        value_stack
                            .push((builder.ins().fmin(a.0, b.0), builder.ins().fmin(a.1, b.1)));
                    }

                    OpCode::Intersection => {
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        value_stack
                            .push((builder.ins().fmax(a.0, b.0), builder.ins().fmax(a.1, b.1)));
                    }

                    OpCode::Subtraction => {
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let neg_b0 = builder.ins().fneg(b.0);
                        let neg_b1 = builder.ins().fneg(b.1);
                        value_stack.push((
                            builder.ins().fmax(a.0, neg_b0),
                            builder.ins().fmax(a.1, neg_b1),
                        ));
                    }

                    // Division Exorcism: Dynamic Smooth ops with mul(inv_k)
                    OpCode::SmoothUnion => {
                        let k_raw = inst.params[0];
                        let inv_k_raw = if k_raw.abs() < 1e-10 {
                            1.0
                        } else {
                            1.0 / k_raw
                        };
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        let k_v = emitter.emit_splat(&mut builder, k_raw);
                        let inv_k_v = emitter.emit_splat(&mut builder, inv_k_raw);
                        let quarter_s = builder.ins().f32const(0.25);
                        let quarter = builder.ins().splat(vec_type, quarter_s);

                        let diff0 = builder.ins().fsub(a.0, b.0);
                        let abs_diff0 = builder.ins().fabs(diff0);
                        let h_num0 = builder.ins().fsub(k_v, abs_diff0);
                        let h_num0 = builder.ins().fmax(h_num0, zero_vec);
                        let h0 = builder.ins().fmul(h_num0, inv_k_v);
                        let hh0 = builder.ins().fmul(h0, h0);
                        let hhk0 = builder.ins().fmul(hh0, k_v);
                        let off0 = builder.ins().fmul(hhk0, quarter);
                        let min0 = builder.ins().fmin(a.0, b.0);
                        let res0 = builder.ins().fsub(min0, off0);

                        let diff1 = builder.ins().fsub(a.1, b.1);
                        let abs_diff1 = builder.ins().fabs(diff1);
                        let h_num1 = builder.ins().fsub(k_v, abs_diff1);
                        let h_num1 = builder.ins().fmax(h_num1, zero_vec);
                        let h1 = builder.ins().fmul(h_num1, inv_k_v);
                        let hh1 = builder.ins().fmul(h1, h1);
                        let hhk1 = builder.ins().fmul(hh1, k_v);
                        let off1 = builder.ins().fmul(hhk1, quarter);
                        let min1 = builder.ins().fmin(a.1, b.1);
                        let res1 = builder.ins().fsub(min1, off1);

                        value_stack.push((res0, res1));
                    }

                    OpCode::SmoothIntersection => {
                        let k_raw = inst.params[0];
                        let inv_k_raw = if k_raw.abs() < 1e-10 {
                            1.0
                        } else {
                            1.0 / k_raw
                        };
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        let k_v = emitter.emit_splat(&mut builder, k_raw);
                        let inv_k_v = emitter.emit_splat(&mut builder, inv_k_raw);
                        let quarter_s = builder.ins().f32const(0.25);
                        let quarter = builder.ins().splat(vec_type, quarter_s);

                        let diff0 = builder.ins().fsub(a.0, b.0);
                        let abs_diff0 = builder.ins().fabs(diff0);
                        let h_num0 = builder.ins().fsub(k_v, abs_diff0);
                        let h_num0 = builder.ins().fmax(h_num0, zero_vec);
                        let h0 = builder.ins().fmul(h_num0, inv_k_v);
                        let hh0 = builder.ins().fmul(h0, h0);
                        let hhk0 = builder.ins().fmul(hh0, k_v);
                        let off0 = builder.ins().fmul(hhk0, quarter);
                        let max0 = builder.ins().fmax(a.0, b.0);
                        let res0 = builder.ins().fadd(max0, off0);

                        let diff1 = builder.ins().fsub(a.1, b.1);
                        let abs_diff1 = builder.ins().fabs(diff1);
                        let h_num1 = builder.ins().fsub(k_v, abs_diff1);
                        let h_num1 = builder.ins().fmax(h_num1, zero_vec);
                        let h1 = builder.ins().fmul(h_num1, inv_k_v);
                        let hh1 = builder.ins().fmul(h1, h1);
                        let hhk1 = builder.ins().fmul(hh1, k_v);
                        let off1 = builder.ins().fmul(hhk1, quarter);
                        let max1 = builder.ins().fmax(a.1, b.1);
                        let res1 = builder.ins().fadd(max1, off1);

                        value_stack.push((res0, res1));
                    }

                    OpCode::SmoothSubtraction => {
                        let k_raw = inst.params[0];
                        let inv_k_raw = if k_raw.abs() < 1e-10 {
                            1.0
                        } else {
                            1.0 / k_raw
                        };
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        let k_v = emitter.emit_splat(&mut builder, k_raw);
                        let inv_k_v = emitter.emit_splat(&mut builder, inv_k_raw);
                        let quarter_s = builder.ins().f32const(0.25);
                        let quarter = builder.ins().splat(vec_type, quarter_s);

                        let neg_b0 = builder.ins().fneg(b.0);
                        let diff0 = builder.ins().fsub(a.0, neg_b0);
                        let abs_diff0 = builder.ins().fabs(diff0);
                        let h_num0 = builder.ins().fsub(k_v, abs_diff0);
                        let h_num0 = builder.ins().fmax(h_num0, zero_vec);
                        let h0 = builder.ins().fmul(h_num0, inv_k_v);
                        let hh0 = builder.ins().fmul(h0, h0);
                        let hhk0 = builder.ins().fmul(hh0, k_v);
                        let off0 = builder.ins().fmul(hhk0, quarter);
                        let max0 = builder.ins().fmax(a.0, neg_b0);
                        let res0 = builder.ins().fadd(max0, off0);

                        let neg_b1 = builder.ins().fneg(b.1);
                        let diff1 = builder.ins().fsub(a.1, neg_b1);
                        let abs_diff1 = builder.ins().fabs(diff1);
                        let h_num1 = builder.ins().fsub(k_v, abs_diff1);
                        let h_num1 = builder.ins().fmax(h_num1, zero_vec);
                        let h1 = builder.ins().fmul(h_num1, inv_k_v);
                        let hh1 = builder.ins().fmul(h1, h1);
                        let hhk1 = builder.ins().fmul(hh1, k_v);
                        let off1 = builder.ins().fmul(hhk1, quarter);
                        let max1 = builder.ins().fmax(a.1, neg_b1);
                        let res1 = builder.ins().fadd(max1, off1);

                        value_stack.push((res0, res1));
                    }

                    OpCode::ChamferUnion => {
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        let r_v = emitter.emit_splat(&mut builder, inst.params[0]);
                        let s_v = emitter.emit_splat(&mut builder, std::f32::consts::FRAC_1_SQRT_2);

                        let min0 = builder.ins().fmin(a.0, b.0);
                        let sum0 = builder.ins().fadd(a.0, b.0);
                        let sc0 = builder.ins().fmul(sum0, s_v);
                        let ch0 = builder.ins().fsub(sc0, r_v);
                        let res0 = builder.ins().fmin(min0, ch0);

                        let min1 = builder.ins().fmin(a.1, b.1);
                        let sum1 = builder.ins().fadd(a.1, b.1);
                        let sc1 = builder.ins().fmul(sum1, s_v);
                        let ch1 = builder.ins().fsub(sc1, r_v);
                        let res1 = builder.ins().fmin(min1, ch1);

                        value_stack.push((res0, res1));
                    }

                    OpCode::ChamferIntersection => {
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        let r_v = emitter.emit_splat(&mut builder, inst.params[0]);
                        let s_v = emitter.emit_splat(&mut builder, std::f32::consts::FRAC_1_SQRT_2);

                        let na0 = builder.ins().fneg(a.0);
                        let nb0 = builder.ins().fneg(b.0);
                        let min0 = builder.ins().fmin(na0, nb0);
                        let sum0 = builder.ins().fadd(na0, nb0);
                        let sc0 = builder.ins().fmul(sum0, s_v);
                        let ch0 = builder.ins().fsub(sc0, r_v);
                        let cm0 = builder.ins().fmin(min0, ch0);
                        let res0 = builder.ins().fneg(cm0);

                        let na1 = builder.ins().fneg(a.1);
                        let nb1 = builder.ins().fneg(b.1);
                        let min1 = builder.ins().fmin(na1, nb1);
                        let sum1 = builder.ins().fadd(na1, nb1);
                        let sc1 = builder.ins().fmul(sum1, s_v);
                        let ch1 = builder.ins().fsub(sc1, r_v);
                        let cm1 = builder.ins().fmin(min1, ch1);
                        let res1 = builder.ins().fneg(cm1);

                        value_stack.push((res0, res1));
                    }

                    OpCode::ChamferSubtraction => {
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                        let r_v = emitter.emit_splat(&mut builder, inst.params[0]);
                        let s_v = emitter.emit_splat(&mut builder, std::f32::consts::FRAC_1_SQRT_2);

                        let na0 = builder.ins().fneg(a.0);
                        let min0 = builder.ins().fmin(na0, b.0);
                        let sum0 = builder.ins().fadd(na0, b.0);
                        let sc0 = builder.ins().fmul(sum0, s_v);
                        let ch0 = builder.ins().fsub(sc0, r_v);
                        let cm0 = builder.ins().fmin(min0, ch0);
                        let res0 = builder.ins().fneg(cm0);

                        let na1 = builder.ins().fneg(a.1);
                        let min1 = builder.ins().fmin(na1, b.1);
                        let sum1 = builder.ins().fadd(na1, b.1);
                        let sc1 = builder.ins().fmul(sum1, s_v);
                        let ch1 = builder.ins().fsub(sc1, r_v);
                        let cm1 = builder.ins().fmin(min1, ch1);
                        let res1 = builder.ins().fneg(cm1);

                        value_stack.push((res0, res1));
                    }

                    OpCode::StairsUnion
                    | OpCode::StairsIntersection
                    | OpCode::StairsSubtraction => {
                        let b = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let a = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                        let r = inst.params[0];
                        let n = inst.params[1];

                        // Pre-splat constants once (avoid double-emit in param buffer)
                        let half_s = builder.ins().f32const(0.5);
                        let half_v = builder.ins().splat(vec_type, half_s);
                        let s_v = emitter.emit_splat(&mut builder, std::f32::consts::FRAC_1_SQRT_2);
                        let s2_v = emitter.emit_splat(&mut builder, std::f32::consts::SQRT_2);
                        let rn_f = r / n;
                        let rn_v = emitter.emit_splat(&mut builder, rn_f);
                        let off_f = (r - rn_f) * 0.5 * std::f32::consts::SQRT_2;
                        let off_v = emitter.emit_splat(&mut builder, off_f);
                        let step_f = r * std::f32::consts::SQRT_2 / n;
                        let step_v = emitter.emit_splat(&mut builder, step_f);

                        let (a0, b0, a1, b1) = match inst.opcode {
                            OpCode::StairsUnion => (a.0, b.0, a.1, b.1),
                            OpCode::StairsIntersection => (
                                builder.ins().fneg(a.0),
                                builder.ins().fneg(b.0),
                                builder.ins().fneg(a.1),
                                builder.ins().fneg(b.1),
                            ),
                            OpCode::StairsSubtraction => {
                                (builder.ins().fneg(a.0), b.0, builder.ins().fneg(a.1), b.1)
                            }
                            _ => unreachable!(),
                        };

                        let sm0 = emit_simd_stairs_min_lane(
                            &mut builder,
                            a0,
                            b0,
                            s_v,
                            half_v,
                            s2_v,
                            rn_v,
                            off_v,
                            step_v,
                        );
                        let sm1 = emit_simd_stairs_min_lane(
                            &mut builder,
                            a1,
                            b1,
                            s_v,
                            half_v,
                            s2_v,
                            rn_v,
                            off_v,
                            step_v,
                        );

                        let (res0, res1) = if inst.opcode == OpCode::StairsUnion {
                            (sm0, sm1)
                        } else {
                            (builder.ins().fneg(sm0), builder.ins().fneg(sm1))
                        };

                        value_stack.push((res0, res1));
                    }

                    OpCode::Translate => {
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Translate,
                            param_vec: zero_vec,
                            params: [0.0; 4],
                        });

                        let tx_v = emitter.emit_splat(&mut builder, inst.params[0]);
                        let ty_v = emitter.emit_splat(&mut builder, inst.params[1]);
                        let tz_v = emitter.emit_splat(&mut builder, inst.params[2]);

                        let nx0 = builder.ins().fsub(curr_x.0, tx_v);
                        let ny0 = builder.ins().fsub(curr_y.0, ty_v);
                        let nz0 = builder.ins().fsub(curr_z.0, tz_v);
                        let nx1 = builder.ins().fsub(curr_x.1, tx_v);
                        let ny1 = builder.ins().fsub(curr_y.1, ty_v);
                        let nz1 = builder.ins().fsub(curr_z.1, tz_v);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        curr_z = (nz0, nz1);
                    }

                    // Division Exorcism: Scale uses mul(inv_factor)
                    // params[0] = inv_factor, params[1] = factor
                    OpCode::Scale => {
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Scale,
                            param_vec: zero_vec,
                            params: [0.0; 4],
                        });

                        let inv_factor = inst.params[0];
                        let factor = inst.params[1];

                        let inv_v = emitter.emit_splat(&mut builder, inv_factor);
                        let f_v = emitter.emit_splat(&mut builder, factor);

                        // p *= inv_factor (Division Exorcism)
                        let nx0 = builder.ins().fmul(curr_x.0, inv_v);
                        let ny0 = builder.ins().fmul(curr_y.0, inv_v);
                        let nz0 = builder.ins().fmul(curr_z.0, inv_v);
                        let nx1 = builder.ins().fmul(curr_x.1, inv_v);
                        let ny1 = builder.ins().fmul(curr_y.1, inv_v);
                        let nz1 = builder.ins().fmul(curr_z.1, inv_v);

                        // scale_correction *= factor
                        let ns0 = builder.ins().fmul(curr_scale.0, f_v);
                        let ns1 = builder.ins().fmul(curr_scale.1, f_v);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        curr_z = (nz0, nz1);
                        curr_scale = (ns0, ns1);
                    }

                    OpCode::Rotate => {
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Rotate,
                            param_vec: zero_vec,
                            params: [0.0; 4],
                        });

                        // Quaternion → rotation matrix (compile-time)
                        let qx = -inst.params[0];
                        let qy = -inst.params[1];
                        let qz = -inst.params[2];
                        let qw = inst.params[3];
                        let m00 = 1.0 - 2.0 * (qy * qy + qz * qz);
                        let m01 = 2.0 * (qx * qy - qz * qw);
                        let m02 = 2.0 * (qx * qz + qy * qw);
                        let m10 = 2.0 * (qx * qy + qz * qw);
                        let m11 = 1.0 - 2.0 * (qx * qx + qz * qz);
                        let m12 = 2.0 * (qy * qz - qx * qw);
                        let m20 = 2.0 * (qx * qz - qy * qw);
                        let m21 = 2.0 * (qy * qz + qx * qw);
                        let m22 = 1.0 - 2.0 * (qx * qx + qy * qy);

                        let m00v = emitter.emit_splat(&mut builder, m00);
                        let m01v = emitter.emit_splat(&mut builder, m01);
                        let m02v = emitter.emit_splat(&mut builder, m02);
                        let m10v = emitter.emit_splat(&mut builder, m10);
                        let m11v = emitter.emit_splat(&mut builder, m11);
                        let m12v = emitter.emit_splat(&mut builder, m12);
                        let m20v = emitter.emit_splat(&mut builder, m20);
                        let m21v = emitter.emit_splat(&mut builder, m21);
                        let m22v = emitter.emit_splat(&mut builder, m22);

                        // Lane 0: standard order FMA chain
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
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::ScaleNonUniform,
                            param_vec: zero_vec,
                            params: [0.0; 4],
                        });

                        let isx = emitter.emit_splat(&mut builder, inst.params[0]);
                        let isy = emitter.emit_splat(&mut builder, inst.params[1]);
                        let isz = emitter.emit_splat(&mut builder, inst.params[2]);
                        let mf = emitter.emit_splat(&mut builder, inst.params[3]);

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
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Twist,
                            param_vec: zero_vec,
                            params: [0.0; 4],
                        });

                        let k_v = emitter.emit_splat(&mut builder, inst.params[0]);

                        // Lane 0
                        let angle0 = builder.ins().fmul(curr_y.0, k_v);
                        let (cos0, sin0) = simd_sincos_approx(&mut builder, angle0, vec_type);
                        let cx0 = builder.ins().fmul(cos0, curr_x.0);
                        let sz0 = builder.ins().fmul(sin0, curr_z.0);
                        let nx0 = builder.ins().fsub(cx0, sz0);
                        let sx0 = builder.ins().fmul(sin0, curr_x.0);
                        let cz0 = builder.ins().fmul(cos0, curr_z.0);
                        let nz0 = builder.ins().fadd(sx0, cz0);

                        // Lane 1
                        let angle1 = builder.ins().fmul(curr_y.1, k_v);
                        let (cos1, sin1) = simd_sincos_approx(&mut builder, angle1, vec_type);
                        let cx1 = builder.ins().fmul(cos1, curr_x.1);
                        let sz1 = builder.ins().fmul(sin1, curr_z.1);
                        let nx1 = builder.ins().fsub(cx1, sz1);
                        let sx1 = builder.ins().fmul(sin1, curr_x.1);
                        let cz1 = builder.ins().fmul(cos1, curr_z.1);
                        let nz1 = builder.ins().fadd(sx1, cz1);

                        curr_x = (nx0, nx1);
                        curr_z = (nz0, nz1);
                    }

                    OpCode::Bend => {
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Bend,
                            param_vec: zero_vec,
                            params: [0.0; 4],
                        });

                        let k_v = emitter.emit_splat(&mut builder, inst.params[0]);

                        // Lane 0: x' = cos*x - sin*y, y' = sin*x + cos*y
                        let angle0 = builder.ins().fmul(k_v, curr_x.0);
                        let (cos0, sin0) = simd_sincos_approx(&mut builder, angle0, vec_type);
                        let cx0 = builder.ins().fmul(cos0, curr_x.0);
                        let sy0 = builder.ins().fmul(sin0, curr_y.0);
                        let nx0 = builder.ins().fsub(cx0, sy0);
                        let sx0 = builder.ins().fmul(sin0, curr_x.0);
                        let cy0 = builder.ins().fmul(cos0, curr_y.0);
                        let ny0 = builder.ins().fadd(sx0, cy0);

                        // Lane 1
                        let angle1 = builder.ins().fmul(k_v, curr_x.1);
                        let (cos1, sin1) = simd_sincos_approx(&mut builder, angle1, vec_type);
                        let cx1 = builder.ins().fmul(cos1, curr_x.1);
                        let sy1 = builder.ins().fmul(sin1, curr_y.1);
                        let nx1 = builder.ins().fsub(cx1, sy1);
                        let sx1 = builder.ins().fmul(sin1, curr_x.1);
                        let cy1 = builder.ins().fmul(cos1, curr_y.1);
                        let ny1 = builder.ins().fadd(sx1, cy1);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                    }

                    OpCode::RepeatInfinite => {
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::RepeatInfinite,
                            param_vec: zero_vec,
                            params: [0.0; 4],
                        });

                        let sx = inst.params[0];
                        let sy = inst.params[1];
                        let sz = inst.params[2];
                        let isx = if sx.abs() < 1e-7 { 0.0 } else { 1.0 / sx };
                        let isy = if sy.abs() < 1e-7 { 0.0 } else { 1.0 / sy };
                        let isz = if sz.abs() < 1e-7 { 0.0 } else { 1.0 / sz };

                        let sx_v = emitter.emit_splat(&mut builder, sx);
                        let sy_v = emitter.emit_splat(&mut builder, sy);
                        let sz_v = emitter.emit_splat(&mut builder, sz);
                        let isx_v = emitter.emit_splat(&mut builder, isx);
                        let isy_v = emitter.emit_splat(&mut builder, isy);
                        let isz_v = emitter.emit_splat(&mut builder, isz);

                        // Lane 0
                        let _tm65 = builder.ins().fmul(curr_x.0, isx_v);
                        let rx0 = builder.ins().nearest(_tm65);
                        let _tm66 = builder.ins().fmul(curr_y.0, isy_v);
                        let ry0 = builder.ins().nearest(_tm66);
                        let _tm67 = builder.ins().fmul(curr_z.0, isz_v);
                        let rz0 = builder.ins().nearest(_tm67);
                        let _tm68 = builder.ins().fmul(sx_v, rx0);
                        let nx0 = builder.ins().fsub(curr_x.0, _tm68);
                        let _tm69 = builder.ins().fmul(sy_v, ry0);
                        let ny0 = builder.ins().fsub(curr_y.0, _tm69);
                        let _tm70 = builder.ins().fmul(sz_v, rz0);
                        let nz0 = builder.ins().fsub(curr_z.0, _tm70);

                        // Lane 1
                        let _tm71 = builder.ins().fmul(curr_x.1, isx_v);
                        let rx1 = builder.ins().nearest(_tm71);
                        let _tm72 = builder.ins().fmul(curr_y.1, isy_v);
                        let ry1 = builder.ins().nearest(_tm72);
                        let _tm73 = builder.ins().fmul(curr_z.1, isz_v);
                        let rz1 = builder.ins().nearest(_tm73);
                        let _tm74 = builder.ins().fmul(sx_v, rx1);
                        let nx1 = builder.ins().fsub(curr_x.1, _tm74);
                        let _tm75 = builder.ins().fmul(sy_v, ry1);
                        let ny1 = builder.ins().fsub(curr_y.1, _tm75);
                        let _tm76 = builder.ins().fmul(sz_v, rz1);
                        let nz1 = builder.ins().fsub(curr_z.1, _tm76);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        curr_z = (nz0, nz1);
                    }

                    OpCode::RepeatFinite => {
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::RepeatFinite,
                            param_vec: zero_vec,
                            params: [0.0; 4],
                        });

                        let cx = inst.params[0];
                        let cy = inst.params[1];
                        let cz = inst.params[2];
                        let sx = inst.params[3];
                        let sy = inst.params[4];
                        let sz = inst.params[5];
                        let isx = if sx.abs() < 1e-7 { 0.0 } else { 1.0 / sx };
                        let isy = if sy.abs() < 1e-7 { 0.0 } else { 1.0 / sy };
                        let isz = if sz.abs() < 1e-7 { 0.0 } else { 1.0 / sz };
                        let lx = cx * 0.5;
                        let ly = cy * 0.5;
                        let lz = cz * 0.5;

                        let sx_v = emitter.emit_splat(&mut builder, sx);
                        let sy_v = emitter.emit_splat(&mut builder, sy);
                        let sz_v = emitter.emit_splat(&mut builder, sz);
                        let isx_v = emitter.emit_splat(&mut builder, isx);
                        let isy_v = emitter.emit_splat(&mut builder, isy);
                        let isz_v = emitter.emit_splat(&mut builder, isz);
                        let lx_v = emitter.emit_splat(&mut builder, lx);
                        let ly_v = emitter.emit_splat(&mut builder, ly);
                        let lz_v = emitter.emit_splat(&mut builder, lz);
                        let nlx_v = emitter.emit_splat(&mut builder, -lx);
                        let nly_v = emitter.emit_splat(&mut builder, -ly);
                        let nlz_v = emitter.emit_splat(&mut builder, -lz);

                        // Lane 0
                        let _tm77 = builder.ins().fmul(curr_x.0, isx_v);
                        let rx0 = builder.ins().nearest(_tm77);
                        let _tm78 = builder.ins().fmul(curr_y.0, isy_v);
                        let ry0 = builder.ins().nearest(_tm78);
                        let _tm79 = builder.ins().fmul(curr_z.0, isz_v);
                        let rz0 = builder.ins().nearest(_tm79);
                        let _tn80 = builder.ins().fmin(rx0, lx_v);
                        let rx0 = builder.ins().fmax(nlx_v, _tn80);
                        let _tn81 = builder.ins().fmin(ry0, ly_v);
                        let ry0 = builder.ins().fmax(nly_v, _tn81);
                        let _tn82 = builder.ins().fmin(rz0, lz_v);
                        let rz0 = builder.ins().fmax(nlz_v, _tn82);
                        let _tm83 = builder.ins().fmul(rx0, sx_v);
                        let nx0 = builder.ins().fsub(curr_x.0, _tm83);
                        let _tm84 = builder.ins().fmul(ry0, sy_v);
                        let ny0 = builder.ins().fsub(curr_y.0, _tm84);
                        let _tm85 = builder.ins().fmul(rz0, sz_v);
                        let nz0 = builder.ins().fsub(curr_z.0, _tm85);

                        // Lane 1
                        let _tm86 = builder.ins().fmul(curr_x.1, isx_v);
                        let rx1 = builder.ins().nearest(_tm86);
                        let _tm87 = builder.ins().fmul(curr_y.1, isy_v);
                        let ry1 = builder.ins().nearest(_tm87);
                        let _tm88 = builder.ins().fmul(curr_z.1, isz_v);
                        let rz1 = builder.ins().nearest(_tm88);
                        let _tn89 = builder.ins().fmin(rx1, lx_v);
                        let rx1 = builder.ins().fmax(nlx_v, _tn89);
                        let _tn90 = builder.ins().fmin(ry1, ly_v);
                        let ry1 = builder.ins().fmax(nly_v, _tn90);
                        let _tn91 = builder.ins().fmin(rz1, lz_v);
                        let rz1 = builder.ins().fmax(nlz_v, _tn91);
                        let _tm92 = builder.ins().fmul(rx1, sx_v);
                        let nx1 = builder.ins().fsub(curr_x.1, _tm92);
                        let _tm93 = builder.ins().fmul(ry1, sy_v);
                        let ny1 = builder.ins().fsub(curr_y.1, _tm93);
                        let _tm94 = builder.ins().fmul(rz1, sz_v);
                        let nz1 = builder.ins().fsub(curr_z.1, _tm94);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        curr_z = (nz0, nz1);
                    }

                    OpCode::Elongate => {
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Elongate,
                            param_vec: zero_vec,
                            params: [0.0; 4],
                        });

                        let hx = emitter.emit_splat(&mut builder, inst.params[0]);
                        let hy = emitter.emit_splat(&mut builder, inst.params[1]);
                        let hz = emitter.emit_splat(&mut builder, inst.params[2]);
                        let nhx = builder.ins().fneg(hx);
                        let nhy = builder.ins().fneg(hy);
                        let nhz = builder.ins().fneg(hz);

                        // Lane 0
                        let _tn95 = builder.ins().fmin(curr_x.0, hx);
                        let cx0 = builder.ins().fmax(nhx, _tn95);
                        let _tn96 = builder.ins().fmin(curr_y.0, hy);
                        let cy0 = builder.ins().fmax(nhy, _tn96);
                        let _tn97 = builder.ins().fmin(curr_z.0, hz);
                        let cz0 = builder.ins().fmax(nhz, _tn97);
                        let nx0 = builder.ins().fsub(curr_x.0, cx0);
                        let ny0 = builder.ins().fsub(curr_y.0, cy0);
                        let nz0 = builder.ins().fsub(curr_z.0, cz0);

                        // Lane 1
                        let _tn98 = builder.ins().fmin(curr_x.1, hx);
                        let cx1 = builder.ins().fmax(nhx, _tn98);
                        let _tn99 = builder.ins().fmin(curr_y.1, hy);
                        let cy1 = builder.ins().fmax(nhy, _tn99);
                        let _tn100 = builder.ins().fmin(curr_z.1, hz);
                        let cz1 = builder.ins().fmax(nhz, _tn100);
                        let nx1 = builder.ins().fsub(curr_x.1, cx1);
                        let ny1 = builder.ins().fsub(curr_y.1, cy1);
                        let nz1 = builder.ins().fsub(curr_z.1, cz1);

                        curr_x = (nx0, nx1);
                        curr_y = (ny0, ny1);
                        curr_z = (nz0, nz1);
                    }

                    OpCode::Mirror => {
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Mirror,
                            param_vec: zero_vec,
                            params: [0.0; 4],
                        });

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

                    OpCode::Revolution => {
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Revolution,
                            param_vec: zero_vec,
                            params: [0.0; 4],
                        });

                        let off = emitter.emit_splat(&mut builder, inst.params[0]);

                        let len0 = simd_length2_fma(&mut builder, curr_x.0, curr_z.0);
                        let q0 = builder.ins().fsub(len0, off);
                        let len1 = simd_length2_fma(&mut builder, curr_x.1, curr_z.1);
                        let q1 = builder.ins().fsub(len1, off);

                        curr_x = (q0, q1);
                        curr_z = (zero_vec, zero_vec);
                    }

                    OpCode::SweepBezier => {
                        return Err("SweepBezier not supported in dynamic SIMD JIT".to_string());
                    }

                    OpCode::Extrude => {
                        let hh = inst.params[0];
                        let hh_v = emitter.emit_splat(&mut builder, hh);
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Extrude,
                            param_vec: hh_v,
                            params: [hh, 0.0, 0.0, 0.0],
                        });

                        curr_z = (zero_vec, zero_vec);
                    }

                    OpCode::Noise => {
                        // Noise is nop in JIT (perlin not available)
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Noise,
                            param_vec: zero_vec,
                            params: [0.0; 4],
                        });
                    }

                    OpCode::Round => {
                        let r_v = emitter.emit_splat(&mut builder, inst.params[0]);
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Round,
                            param_vec: r_v,
                            params: [0.0; 4],
                        });
                    }

                    OpCode::Onion => {
                        let t_v = emitter.emit_splat(&mut builder, inst.params[0]);
                        coord_stack.push(DynSimdCoordState {
                            x: curr_x,
                            y: curr_y,
                            z: curr_z,
                            scale: curr_scale,
                            opcode: OpCode::Onion,
                            param_vec: t_v,
                            params: [0.0; 4],
                        });
                    }

                    OpCode::PopTransform => {
                        if let Some(state) = coord_stack.pop() {
                            match state.opcode {
                                OpCode::Round => {
                                    let d = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                                    value_stack.push((
                                        builder.ins().fsub(d.0, state.param_vec),
                                        builder.ins().fsub(d.1, state.param_vec),
                                    ));
                                }
                                OpCode::Onion => {
                                    let d = value_stack.pop().unwrap_or((zero_vec, zero_vec));
                                    let a0 = builder.ins().fabs(d.0);
                                    let a1 = builder.ins().fabs(d.1);
                                    value_stack.push((
                                        builder.ins().fsub(a0, state.param_vec),
                                        builder.ins().fsub(a1, state.param_vec),
                                    ));
                                }
                                OpCode::Extrude => {
                                    let d = value_stack.pop().unwrap_or((zero_vec, zero_vec));

                                    // Lane 0
                                    let az0 = builder.ins().fabs(state.z.0);
                                    let wy0 = builder.ins().fsub(az0, state.param_vec);
                                    let inner0 = builder.ins().fmax(d.0, wy0);
                                    let inside0 = builder.ins().fmin(inner0, zero_vec);
                                    let mx0 = builder.ins().fmax(d.0, zero_vec);
                                    let my0 = builder.ins().fmax(wy0, zero_vec);
                                    let outside0 = simd_length2_fma(&mut builder, mx0, my0);
                                    let r0 = builder.ins().fadd(inside0, outside0);

                                    // Lane 1
                                    let az1 = builder.ins().fabs(state.z.1);
                                    let wy1 = builder.ins().fsub(az1, state.param_vec);
                                    let inner1 = builder.ins().fmax(d.1, wy1);
                                    let inside1 = builder.ins().fmin(inner1, zero_vec);
                                    let mx1 = builder.ins().fmax(d.1, zero_vec);
                                    let my1 = builder.ins().fmax(wy1, zero_vec);
                                    let outside1 = simd_length2_fma(&mut builder, mx1, my1);
                                    let r1 = builder.ins().fadd(inside1, outside1);

                                    value_stack.push((r0, r1));
                                }
                                _ => {}
                            }
                            curr_x = state.x;
                            curr_y = state.y;
                            curr_z = state.z;
                            curr_scale = state.scale;
                        }
                    }

                    OpCode::End => break,

                    _ => {
                        let max_s = builder.ins().f32const(f32::MAX);
                        let max_v = builder.ins().splat(vec_type, max_s);
                        value_stack.push((max_v, max_v));
                    }
                }
            }

            let result = value_stack.pop().unwrap_or((zero_vec, zero_vec));
            builder.ins().store(mem_flags, result.0, ptr_out, 0);
            builder.ins().store(mem_flags, result.1, ptr_out, 16);
            builder.ins().return_(&[]);
            builder.finalize();

            initial_params = emitter.params;
        }

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().map_err(|e| e.to_string())?;

        let code = module.get_finalized_function(func_id);

        Ok(JitSimdSdfDynamic {
            module,
            func_ptr: code,
            params: initial_params,
        })
    }

    /// Update parameters from a modified CompiledSdf (zero-latency, no recompilation)
    pub fn update_params(&mut self, sdf: &CompiledSdf) {
        self.params = extract_simd_params(sdf);
    }

    /// Get current parameter values
    pub fn params(&self) -> &[f32] {
        &self.params
    }

    /// Evaluate 8 points using native SIMD (raw pointer interface)
    #[inline]
    pub unsafe fn eval_8_raw(&self, x: *const f32, y: *const f32, z: *const f32, out: *mut f32) {
        let func: SimdSdfDynamicFn = mem::transmute(self.func_ptr);
        func(x, y, z, out, self.params.as_ptr());
    }

    /// Evaluate 8 points using native SIMD (array interface)
    #[inline]
    pub unsafe fn eval_8(&self, x: &[f32; 8], y: &[f32; 8], z: &[f32; 8]) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        self.eval_8_raw(x.as_ptr(), y.as_ptr(), z.as_ptr(), out.as_mut_ptr());
        out
    }

    /// Evaluate many points (Zero-copy optimization)
    pub fn eval_batch(&self, x: &[f32], y: &[f32], z: &[f32]) -> Vec<f32> {
        let len = x.len();
        assert_eq!(y.len(), len);
        assert_eq!(z.len(), len);

        let mut results = vec![0.0f32; len];
        let chunk_size = 8;
        let loop_len = len - (len % chunk_size);

        if loop_len > 0 {
            let x_ptr = x.as_ptr();
            let y_ptr = y.as_ptr();
            let z_ptr = z.as_ptr();
            let out_ptr = results.as_mut_ptr();

            for i in (0..loop_len).step_by(chunk_size) {
                unsafe {
                    self.eval_8_raw(x_ptr.add(i), y_ptr.add(i), z_ptr.add(i), out_ptr.add(i));
                }
            }
        }

        let remainder = len % chunk_size;
        if remainder > 0 {
            let offset = loop_len;
            let mut x_pad = [0.0f32; 8];
            let mut y_pad = [0.0f32; 8];
            let mut z_pad = [0.0f32; 8];
            let mut out_pad = [0.0f32; 8];

            unsafe {
                std::ptr::copy_nonoverlapping(
                    x.as_ptr().add(offset),
                    x_pad.as_mut_ptr(),
                    remainder,
                );
                std::ptr::copy_nonoverlapping(
                    y.as_ptr().add(offset),
                    y_pad.as_mut_ptr(),
                    remainder,
                );
                std::ptr::copy_nonoverlapping(
                    z.as_ptr().add(offset),
                    z_pad.as_mut_ptr(),
                    remainder,
                );
                self.eval_8_raw(
                    x_pad.as_ptr(),
                    y_pad.as_ptr(),
                    z_pad.as_ptr(),
                    out_pad.as_mut_ptr(),
                );
                std::ptr::copy_nonoverlapping(
                    out_pad.as_ptr(),
                    results.as_mut_ptr().add(offset),
                    remainder,
                );
            }
        }

        results
    }

    /// Evaluate SoAPoints using native SIMD
    pub fn eval_soa(&self, points: &crate::soa::SoAPoints) -> Vec<f32> {
        let (x, y, z) = points.as_slices();
        self.eval_batch(x, y, z)
    }
}

/// Extract SIMD JIT parameters from a CompiledSdf
///
/// Returns parameters in the exact order they are consumed by the dynamic
/// SIMD JIT function. This order must match the emit order in
/// `JitSimdSdfDynamic::compile`.
pub fn extract_simd_params(sdf: &CompiledSdf) -> Vec<f32> {
    let mut params = Vec::new();
    for inst in &sdf.instructions {
        match inst.opcode {
            OpCode::Sphere => {
                params.push(inst.params[0]);
            }
            OpCode::Box3d => {
                params.push(inst.params[0]);
                params.push(inst.params[1]);
                params.push(inst.params[2]);
            }
            OpCode::Cylinder => {
                params.push(inst.params[0]);
                params.push(inst.params[1]);
            }
            OpCode::Plane => {
                params.push(inst.params[0]);
                params.push(inst.params[1]);
                params.push(inst.params[2]);
                params.push(inst.params[3]);
            }
            OpCode::Torus => {
                params.push(inst.params[0]);
                params.push(inst.params[1]);
            }
            OpCode::Capsule => {
                let ax = inst.params[0];
                let ay = inst.params[1];
                let az = inst.params[2];
                let radius = inst.get_capsule_radius();
                let ba_x = inst.params[3] - ax;
                let ba_y = inst.params[4] - ay;
                let ba_z = inst.params[5] - az;
                let ba_dot = ba_x * ba_x + ba_y * ba_y + ba_z * ba_z;
                let inv_ba_dot = if ba_dot.abs() < 1e-10 {
                    1.0
                } else {
                    1.0 / ba_dot
                };
                params.push(ax);
                params.push(ay);
                params.push(az);
                params.push(radius);
                params.push(ba_x);
                params.push(ba_y);
                params.push(ba_z);
                params.push(inv_ba_dot);
            }
            OpCode::Cone => {
                let radius = inst.params[0];
                let half_height = inst.params[1];
                let k2x = -radius;
                let k2y = 2.0 * half_height;
                let k2_dot = k2x * k2x + k2y * k2y;
                let inv_k2d = if k2_dot.abs() < 1e-10 {
                    1.0
                } else {
                    1.0 / k2_dot
                };
                params.push(radius);
                params.push(half_height);
                params.push(k2x);
                params.push(k2y);
                params.push(inv_k2d);
            }
            OpCode::Ellipsoid => {
                let rx = inst.params[0];
                let ry = inst.params[1];
                let rz = inst.params[2];
                params.push(1.0 / rx);
                params.push(1.0 / ry);
                params.push(1.0 / rz);
                params.push(1.0 / (rx * rx));
                params.push(1.0 / (ry * ry));
                params.push(1.0 / (rz * rz));
            }
            OpCode::RoundedCone => {
                let r1 = inst.params[0];
                let r2 = inst.params[1];
                let half_height = inst.params[2];
                let h = half_height * 2.0;
                let inv_h = if h.abs() < 1e-10 { 1.0 } else { 1.0 / h };
                let b = (r1 - r2) * inv_h;
                let a = (1.0 - b * b).max(0.0).sqrt();
                let ah = a * h;
                params.push(r1);
                params.push(r2);
                params.push(half_height);
                params.push(h);
                params.push(b);
                params.push(a);
                params.push(ah);
            }
            OpCode::Pyramid => {
                let half_height = inst.params[0];
                let h = half_height * 2.0;
                let m2 = h * h + 0.25;
                let inv_m2 = 1.0 / m2;
                let inv_m2_025 = 1.0 / (m2 + 0.25);
                params.push(half_height);
                params.push(h);
                params.push(m2);
                params.push(inv_m2);
                params.push(inv_m2_025);
            }
            OpCode::Octahedron => {
                params.push(inst.params[0]);
            }
            OpCode::HexPrism => {
                params.push(inst.params[0]);
                params.push(inst.params[1]);
            }
            OpCode::Link => {
                params.push(inst.params[0]);
                params.push(inst.params[1]);
                params.push(inst.params[2]);
            }
            OpCode::SmoothUnion | OpCode::SmoothIntersection | OpCode::SmoothSubtraction => {
                let k = inst.params[0];
                let inv_k = if k.abs() < 1e-10 { 1.0 } else { 1.0 / k };
                params.push(k);
                params.push(inv_k);
            }
            OpCode::ChamferUnion | OpCode::ChamferIntersection | OpCode::ChamferSubtraction => {
                params.push(inst.params[0]); // r
            }
            OpCode::StairsUnion | OpCode::StairsIntersection | OpCode::StairsSubtraction => {
                params.push(inst.params[0]); // r
                params.push(inst.params[1]); // n
            }
            OpCode::Translate => {
                params.push(inst.params[0]);
                params.push(inst.params[1]);
                params.push(inst.params[2]);
            }
            OpCode::Scale => {
                let s = inst.params[0];
                let inv_s = if s.abs() < 1e-10 { 1.0 } else { 1.0 / s };
                params.push(s);
                params.push(inv_s);
            }
            OpCode::Rotate => {
                // Quaternion → rotation matrix (must match dynamic compile emit order)
                let qx = -inst.params[0];
                let qy = -inst.params[1];
                let qz = -inst.params[2];
                let qw = inst.params[3];
                params.push(1.0 - 2.0 * (qy * qy + qz * qz)); // m00
                params.push(2.0 * (qx * qy - qz * qw)); // m01
                params.push(2.0 * (qx * qz + qy * qw)); // m02
                params.push(2.0 * (qx * qy + qz * qw)); // m10
                params.push(1.0 - 2.0 * (qx * qx + qz * qz)); // m11
                params.push(2.0 * (qy * qz - qx * qw)); // m12
                params.push(2.0 * (qx * qz - qy * qw)); // m20
                params.push(2.0 * (qy * qz + qx * qw)); // m21
                params.push(1.0 - 2.0 * (qx * qx + qy * qy)); // m22
            }
            OpCode::ScaleNonUniform => {
                params.push(inst.params[0]); // inv_sx
                params.push(inst.params[1]); // inv_sy
                params.push(inst.params[2]); // inv_sz
                params.push(inst.params[3]); // min_factor
            }
            OpCode::Twist => {
                params.push(inst.params[0]); // strength
            }
            OpCode::Bend => {
                params.push(inst.params[0]); // curvature
            }
            OpCode::RepeatInfinite => {
                let sx = inst.params[0];
                let sy = inst.params[1];
                let sz = inst.params[2];
                let isx = if sx.abs() < 1e-7 { 0.0 } else { 1.0 / sx };
                let isy = if sy.abs() < 1e-7 { 0.0 } else { 1.0 / sy };
                let isz = if sz.abs() < 1e-7 { 0.0 } else { 1.0 / sz };
                params.push(sx);
                params.push(sy);
                params.push(sz);
                params.push(isx);
                params.push(isy);
                params.push(isz);
            }
            OpCode::RepeatFinite => {
                let cx = inst.params[0];
                let cy = inst.params[1];
                let cz = inst.params[2];
                let sx = inst.params[3];
                let sy = inst.params[4];
                let sz = inst.params[5];
                let isx = if sx.abs() < 1e-7 { 0.0 } else { 1.0 / sx };
                let isy = if sy.abs() < 1e-7 { 0.0 } else { 1.0 / sy };
                let isz = if sz.abs() < 1e-7 { 0.0 } else { 1.0 / sz };
                let lx = cx * 0.5;
                let ly = cy * 0.5;
                let lz = cz * 0.5;
                params.push(sx);
                params.push(sy);
                params.push(sz);
                params.push(isx);
                params.push(isy);
                params.push(isz);
                params.push(lx);
                params.push(ly);
                params.push(lz);
            }
            OpCode::Elongate => {
                params.push(inst.params[0]);
                params.push(inst.params[1]);
                params.push(inst.params[2]);
            }
            OpCode::Revolution => {
                params.push(inst.params[0]); // offset
            }
            OpCode::SweepBezier => {
                for i in 0..6 {
                    params.push(inst.params[i]);
                }
            }
            OpCode::Extrude => {
                params.push(inst.params[0]); // half_height
            }
            OpCode::Round => {
                params.push(inst.params[0]);
            }
            OpCode::Onion => {
                params.push(inst.params[0]);
            }
            // Mirror, Noise: no dynamic params (Mirror is compile-time structural, Noise is nop)
            _ => {}
        }
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiled::CompiledSdf;
    use crate::eval::eval;
    use crate::types::SdfNode;
    use glam::Vec3;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_jit_simd_sphere() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let compiled = CompiledSdf::compile(&sphere);
        let jit = JitSimdSdf::compile(&compiled).unwrap();

        let x = [0.0, 1.0, 2.0, 0.5, -1.0, 0.0, 0.0, 1.5];
        let y = [0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0];
        let z = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0];

        let results = unsafe { jit.eval_8(&x, &y, &z) };

        for i in 0..8 {
            let expected = eval(&sphere, Vec3::new(x[i], y[i], z[i]));
            assert!(
                approx_eq(results[i], expected, 0.001),
                "Mismatch at {}: jit={}, cpu={}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_box() {
        let box3d = SdfNode::Box3d {
            half_extents: Vec3::new(1.0, 0.5, 0.5),
        };
        let compiled = CompiledSdf::compile(&box3d);
        let jit = JitSimdSdf::compile(&compiled).unwrap();

        let x = [0.0, 1.5, 0.0, 0.0, 2.0, 0.5, -1.0, 0.0];
        let y = [0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, -0.5];
        let z = [0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0];

        let results = unsafe { jit.eval_8(&x, &y, &z) };

        for i in 0..8 {
            let expected = eval(&box3d, Vec3::new(x[i], y[i], z[i]));
            assert!(
                approx_eq(results[i], expected, 0.001),
                "Mismatch at {}: jit={}, cpu={}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_union() {
        let shape = SdfNode::Sphere { radius: 1.0 }.union(
            SdfNode::Box3d {
                half_extents: Vec3::splat(0.5),
            }
            .translate(2.0, 0.0, 0.0),
        );
        let compiled = CompiledSdf::compile(&shape);
        let jit = JitSimdSdf::compile(&compiled).unwrap();

        let x = [0.0, 1.0, 2.0, 3.0, -1.0, 0.5, 2.5, 1.5];
        let y = [0.0; 8];
        let z = [0.0; 8];

        let results = unsafe { jit.eval_8(&x, &y, &z) };

        for i in 0..8 {
            let expected = eval(&shape, Vec3::new(x[i], y[i], z[i]));
            assert!(
                approx_eq(results[i], expected, 0.01),
                "Mismatch at {}: jit={}, cpu={}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_smooth_union() {
        let shape = SdfNode::Sphere { radius: 1.0 }.smooth_union(
            SdfNode::Box3d {
                half_extents: Vec3::splat(0.5),
            }
            .translate(1.5, 0.0, 0.0),
            0.3,
        );
        let compiled = CompiledSdf::compile(&shape);
        let jit = JitSimdSdf::compile(&compiled).unwrap();

        let x = [0.0, 0.75, 1.5, 2.0, -0.5, 1.0, 1.25, 0.5];
        let y = [0.0; 8];
        let z = [0.0; 8];

        let results = unsafe { jit.eval_8(&x, &y, &z) };

        for i in 0..8 {
            let expected = eval(&shape, Vec3::new(x[i], y[i], z[i]));
            assert!(
                approx_eq(results[i], expected, 0.01),
                "Mismatch at {}: jit={}, cpu={}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_translate_scale() {
        let shape = SdfNode::Sphere { radius: 1.0 }
            .translate(1.0, 2.0, 3.0)
            .scale(0.5);
        let compiled = CompiledSdf::compile(&shape);
        let jit = JitSimdSdf::compile(&compiled).unwrap();

        let x = [1.0, 2.0, 0.0, 1.5, 0.5, 1.0, 1.0, 1.0];
        let y = [2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 1.0, 2.0];
        let z = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0];

        let results = unsafe { jit.eval_8(&x, &y, &z) };

        for i in 0..8 {
            let expected = eval(&shape, Vec3::new(x[i], y[i], z[i]));
            assert!(
                approx_eq(results[i], expected, 0.01),
                "Mismatch at {}: jit={}, cpu={}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_batch() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let compiled = CompiledSdf::compile(&sphere);
        let jit = JitSimdSdf::compile(&compiled).unwrap();

        let x: Vec<f32> = (0..100).map(|i| i as f32 * 0.05 - 2.5).collect();
        let y: Vec<f32> = vec![0.0; 100];
        let z: Vec<f32> = vec![0.0; 100];

        let results = jit.eval_batch(&x, &y, &z);

        for i in 0..100 {
            let expected = eval(&sphere, Vec3::new(x[i], y[i], z[i]));
            assert!(
                approx_eq(results[i], expected, 0.01),
                "Mismatch at {}: jit={}, cpu={}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_soa() {
        use crate::soa::SoAPoints;

        let shape = SdfNode::Sphere { radius: 1.0 }.smooth_union(
            SdfNode::Cylinder {
                radius: 0.3,
                half_height: 1.5,
            },
            0.2,
        );
        let compiled = CompiledSdf::compile(&shape);
        let jit = JitSimdSdf::compile(&compiled).unwrap();

        let points: Vec<Vec3> = (0..50)
            .map(|i| {
                let t = i as f32 / 50.0 * std::f32::consts::TAU;
                Vec3::new(t.cos() * 2.0, t.sin() * 2.0, 0.0)
            })
            .collect();

        let soa = SoAPoints::from_vec3_slice(&points);
        let results = jit.eval_soa(&soa);

        for (i, p) in points.iter().enumerate() {
            let expected = eval(&shape, *p);
            assert!(
                approx_eq(results[i], expected, 0.01),
                "Mismatch at {}: jit={}, cpu={}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_constant_folding_identity_translate() {
        let shape = SdfNode::Sphere { radius: 1.0 }.translate(0.0, 0.0, 0.0);
        let compiled = CompiledSdf::compile(&shape);
        let jit = JitSimdSdf::compile(&compiled).unwrap();

        let x = [0.0, 1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0];
        let y = [0.0; 8];
        let z = [0.0; 8];

        let results = unsafe { jit.eval_8(&x, &y, &z) };

        let plain_sphere = SdfNode::Sphere { radius: 1.0 };
        for i in 0..4 {
            let expected = eval(&plain_sphere, Vec3::new(x[i], y[i], z[i]));
            assert!(
                approx_eq(results[i], expected, 0.001),
                "Identity translate fold mismatch at {}: jit={}, cpu={}",
                i,
                results[i],
                expected
            );
        }
    }

    // ============ Dynamic SIMD Tests ============

    #[test]
    fn test_jit_simd_dynamic_sphere() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let compiled = CompiledSdf::compile(&sphere);
        let jit = JitSimdSdfDynamic::compile(&compiled).unwrap();

        let x = [0.0, 1.0, 2.0, 0.5, -1.0, 0.0, 0.0, 1.5];
        let y = [0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0];
        let z = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0];

        let results = unsafe { jit.eval_8(&x, &y, &z) };

        for i in 0..8 {
            let expected = eval(&sphere, Vec3::new(x[i], y[i], z[i]));
            assert!(
                approx_eq(results[i], expected, 0.001),
                "Dynamic sphere mismatch at {}: jit={}, cpu={}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_dynamic_update_params() {
        let sphere1 = SdfNode::Sphere { radius: 1.0 };
        let compiled1 = CompiledSdf::compile(&sphere1);
        let mut jit = JitSimdSdfDynamic::compile(&compiled1).unwrap();

        let x = [1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y = [0.0; 8];
        let z = [0.0; 8];

        let results1 = unsafe { jit.eval_8(&x, &y, &z) };
        let expected1 = eval(&sphere1, Vec3::new(1.5, 0.0, 0.0));
        assert!(
            approx_eq(results1[0], expected1, 0.001),
            "Before update: jit={}, cpu={}",
            results1[0],
            expected1
        );

        // Update to radius=2.0 (no recompilation!)
        let sphere2 = SdfNode::Sphere { radius: 2.0 };
        let compiled2 = CompiledSdf::compile(&sphere2);
        jit.update_params(&compiled2);

        let results2 = unsafe { jit.eval_8(&x, &y, &z) };
        let expected2 = eval(&sphere2, Vec3::new(1.5, 0.0, 0.0));
        assert!(
            approx_eq(results2[0], expected2, 0.001),
            "After update: jit={}, cpu={}",
            results2[0],
            expected2
        );
    }

    #[test]
    fn test_jit_simd_dynamic_smooth_union() {
        let shape = SdfNode::Sphere { radius: 1.0 }.smooth_union(
            SdfNode::Box3d {
                half_extents: Vec3::splat(0.5),
            }
            .translate(1.5, 0.0, 0.0),
            0.3,
        );
        let compiled = CompiledSdf::compile(&shape);
        let jit = JitSimdSdfDynamic::compile(&compiled).unwrap();

        let x = [0.0, 0.75, 1.5, 2.0, -0.5, 1.0, 1.25, 0.5];
        let y = [0.0; 8];
        let z = [0.0; 8];

        let results = unsafe { jit.eval_8(&x, &y, &z) };

        for i in 0..8 {
            let expected = eval(&shape, Vec3::new(x[i], y[i], z[i]));
            assert!(
                approx_eq(results[i], expected, 0.01),
                "Dynamic smooth union mismatch at {}: jit={}, cpu={}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_jit_simd_dynamic_extract_params() {
        let shape = SdfNode::Sphere { radius: 1.5 }.translate(2.0, 3.0, 4.0);
        let compiled = CompiledSdf::compile(&shape);
        let params = extract_simd_params(&compiled);

        // Translate: tx, ty, tz (3 params)
        // Sphere: radius (1 param)
        // PopTransform + End: no params
        assert_eq!(params.len(), 4, "Expected 4 params, got {}", params.len());
        assert!((params[0] - 2.0).abs() < 1e-6, "tx");
        assert!((params[1] - 3.0).abs() < 1e-6, "ty");
        assert!((params[2] - 4.0).abs() < 1e-6, "tz");
        assert!((params[3] - 1.5).abs() < 1e-6, "radius");
    }

    #[test]
    fn test_constant_folding_identity_scale() {
        let shape = SdfNode::Sphere { radius: 1.0 }.scale(1.0);
        let compiled = CompiledSdf::compile(&shape);
        let jit = JitSimdSdf::compile(&compiled).unwrap();

        let x = [0.0, 1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0];
        let y = [0.0; 8];
        let z = [0.0; 8];

        let results = unsafe { jit.eval_8(&x, &y, &z) };

        let plain_sphere = SdfNode::Sphere { radius: 1.0 };
        for i in 0..4 {
            let expected = eval(&plain_sphere, Vec3::new(x[i], y[i], z[i]));
            assert!(
                approx_eq(results[i], expected, 0.001),
                "Identity scale fold mismatch at {}: jit={}, cpu={}",
                i,
                results[i],
                expected
            );
        }
    }
}
